"""
build_feature_freq_index.py — Feature Frequency Index for E (Epistemic Grounding)
==================================================================================

Builds a frequency distribution over SAE features by running a reference
corpus through the model's SAE encoders. For each feature, counts how
many tokens activate it above threshold.

Output: feature_freq.npy — shape (n_layers, max_features) where
  feature_freq[layer][feature] = fraction of corpus tokens that
  activate this feature (0.0 = never seen, 1.0 = fires on every token).

High-frequency features = well-learned, stable representations.
Low-frequency features = rare, context-specific, potentially confabulatory.

E (epistemic grounding) = weighted mean feature frequency of the
features that drove a given output, measured via attribution graph.

Patent pending: US 64/020,489 & 64/021,113
"""

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def build_feature_freq_index(
    model_name: str = "google/gemma-2-2b",
    corpus_name: str = "wikipedia",
    n_tokens: int = 100_000,
    batch_size: int = 64,   # tokens per batch (sequence length)
    top_k: int = 50,        # features per layer to count
    output_path: str = "feature_freq_index.npz",
    device: str = "cuda",
):
    """
    Build feature frequency index from reference corpus.

    For each layer's SAE encoder, runs corpus tokens through and counts
    how often each feature activates (above zero after ReLU).

    Saves:
      - feature_freq: (n_layers, n_features) — activation fraction per feature
      - feature_counts: (n_layers, n_features) — raw activation counts
      - total_tokens: int — total tokens processed
      - corpus_name: str
    """
    print(f"[E-Index] Building feature frequency index")
    print(f"[E-Index] Model: {model_name}")
    print(f"[E-Index] Corpus: {corpus_name}")
    print(f"[E-Index] Tokens: {n_tokens:,}")
    print()

    # Load corpus
    print("[E-Index] Loading corpus...")
    corpus_tokens = load_corpus(corpus_name, n_tokens, model_name)
    print(f"[E-Index] Loaded {len(corpus_tokens):,} tokens")

    # Load model
    print("[E-Index] Loading model...")
    from circuit_tracer import ReplacementModel
    model = ReplacementModel.from_pretrained(
        model_name, "gemma", dtype=torch.bfloat16
    )

    # Get inner model for tokenizer
    inner = None
    for attr in ["model", "hooked_model", "_model"]:
        candidate = getattr(model, attr, None)
        if candidate is not None and hasattr(candidate, "to_tokens"):
            inner = candidate
            break
    if inner is None and hasattr(model, "to_tokens"):
        inner = model

    # Get transcoders
    tc_list = model.transcoders.transcoders
    n_layers = len(tc_list)
    print(f"[E-Index] {n_layers} layers")

    # Get feature count per layer (W_enc shape is (d_model, n_features))
    n_features_per_layer = []
    for tc in tc_list:
        W_enc = getattr(tc, "W_enc", None)
        if W_enc is not None:
            # W_enc: (d_model, n_features) — features is dim 1
            # But after transpose in matmul, acts has shape (seq, n_features)
            # So n_features = W_enc.shape[0] if W_enc is (n_features, d_model)
            # or W_enc.shape[1] if W_enc is (d_model, n_features)
            # Check: acts = hidden @ W_enc.T → acts is (seq, W_enc.shape[0])
            n_features_per_layer.append(W_enc.shape[0])
        else:
            n_features_per_layer.append(0)

    max_features = max(n_features_per_layer)
    print(f"[E-Index] Max features per layer: {max_features}")

    # Initialize counters
    feature_counts = np.zeros((n_layers, max_features), dtype=np.int64)
    total_processed = 0

    # Process corpus: ONE forward pass per chunk, capture a FEW layers at a time
    # Then encode each layer's hidden states through SAE on CPU
    n_chunks = (len(corpus_tokens) - 1) // batch_size
    t_start = time.time()
    layers_per_pass = 4  # capture 4 layers at a time to limit VRAM

    for layer_group_start in range(0, n_layers, layers_per_pass):
        layer_group = list(range(layer_group_start,
                                 min(layer_group_start + layers_per_pass, n_layers)))
        group_start = time.time()

        for chunk_idx in range(n_chunks):
            start = chunk_idx * batch_size
            end = start + batch_size + 1
            if end > len(corpus_tokens):
                break

            chunk = corpus_tokens[start:end]
            tokens = torch.tensor([chunk], device=device)

            # Capture hidden states for this layer group
            hidden_states = {}
            hooks = []
            for l in layer_group:
                name = f"blocks.{l}.hook_resid_post"
                def _hook(value, hook, _l=l):
                    hidden_states[_l] = value[0].detach().cpu()  # move to CPU immediately
                    return value
                hooks.append((name, _hook))

            with torch.no_grad():
                inner.run_with_hooks(tokens, fwd_hooks=hooks)

            # Encode each layer's hidden states through SAE (on CPU to save VRAM)
            for l in layer_group:
                hidden = hidden_states.get(l)
                if hidden is None:
                    continue

                tc = tc_list[l]
                W_enc = getattr(tc, "W_enc", None)
                b_enc = getattr(tc, "b_enc", None)
                if W_enc is None:
                    continue

                with torch.no_grad():
                    acts = hidden.float() @ W_enc.cpu().float().T
                    if b_enc is not None:
                        acts = acts + b_enc.cpu().float()
                    acts = F.relu(acts)

                    active_mask = (acts > 0).cpu().numpy()
                    n_feat = active_mask.shape[1]
                    feature_counts[l, :n_feat] += active_mask.sum(axis=0)

            del hidden_states
            torch.cuda.empty_cache()

        elapsed_group = time.time() - group_start
        print(f"  Layers {layer_group[0]:>2}-{layer_group[-1]:>2}: {elapsed_group:.1f}s "
              f"({n_chunks} chunks)")

    total_processed = n_chunks * batch_size

    elapsed = time.time() - t_start
    print(f"\n[E-Index] Processed {total_processed:,} tokens in {elapsed:.1f}s")

    # Normalize to frequencies
    feature_freq = feature_counts.astype(np.float64) / max(total_processed, 1)

    # Save
    np.savez_compressed(
        output_path,
        feature_freq=feature_freq,
        feature_counts=feature_counts,
        total_tokens=total_processed,
        corpus_name=corpus_name,
        model_name=model_name,
        n_layers=n_layers,
        n_features_per_layer=np.array(n_features_per_layer),
    )
    print(f"[E-Index] Saved to {output_path}")

    # Stats
    print(f"\n[E-Index] Feature frequency stats:")
    for l in range(n_layers):
        nf = n_features_per_layer[l]
        if nf == 0:
            continue
        freqs = feature_freq[l, :nf]
        active = (freqs > 0).sum()
        print(f"  Layer {l:>2}: {active}/{nf} features active "
              f"(mean freq={freqs[freqs>0].mean():.4f}, "
              f"max={freqs.max():.4f})")

    return feature_freq


def load_corpus(corpus_name, n_tokens, model_name):
    """Load and tokenize a reference corpus."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if corpus_name == "wikipedia" or corpus_name == "wikitext":
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        text = ""
        for item in ds:
            t = item.get("text", "")
            if t.strip():
                text += t + "\n"
            if len(text) > n_tokens * 5:  # rough char-to-token ratio
                break
        return tokenizer.encode(text)[:n_tokens]

    elif corpus_name == "c4":
        from datasets import load_dataset
        ds = load_dataset("allenai/c4", "en", split="train",
                         streaming=True)
        text = ""
        for item in ds:
            text += item["text"] + "\n"
            tokens = tokenizer.encode(text)
            if len(tokens) >= n_tokens:
                return tokens[:n_tokens]
        return tokenizer.encode(text)[:n_tokens]

    else:
        # Try loading as a text file
        with open(corpus_name, "r", encoding="utf-8") as f:
            text = f.read()
        return tokenizer.encode(text)[:n_tokens]


def compute_E(
    attribution_features: list,
    feature_freq_index: np.ndarray,
    top_k: int = 20,
) -> float:
    """
    Compute E (epistemic grounding) for a single token.

    Args:
        attribution_features: list of (layer, feature_id, weight) tuples
            from circuit_tracer attribution graph
        feature_freq_index: (n_layers, n_features) frequency array
        top_k: number of top features to consider

    Returns:
        E score in [0, 1]. Higher = more grounded in learned knowledge.
    """
    if not attribution_features:
        return 0.0

    # Sort by attribution weight
    sorted_feats = sorted(attribution_features, key=lambda x: -abs(x[2]))[:top_k]

    total_weight = sum(abs(w) for _, _, w in sorted_feats)
    if total_weight < 1e-10:
        return 0.0

    E = 0.0
    for layer, feat_id, weight in sorted_feats:
        if layer < feature_freq_index.shape[0] and feat_id < feature_freq_index.shape[1]:
            freq = feature_freq_index[layer, feat_id]
            E += freq * abs(weight)

    return E / total_weight


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="wikipedia", help="Corpus name or path")
    parser.add_argument("--n-tokens", type=int, default=100_000)
    parser.add_argument("--output", default="feature_freq_index.npz")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    build_feature_freq_index(
        corpus_name=args.corpus,
        n_tokens=args.n_tokens,
        output_path=args.output,
        batch_size=args.batch_size,
    )
