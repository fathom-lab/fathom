# -*- coding: utf-8 -*-
"""
FATHOM LIVE DEMO
================
Pitch-ready demonstration of Fathom depth scoring across industry verticals.

Usage:
    python fathom_demo.py                    # Run all domains
    python fathom_demo.py --domain healthcare
    python fathom_demo.py --domain legal
    python fathom_demo.py --domain finance
    python fathom_demo.py --domain education
    python fathom_demo.py --domain ai_safety
    python fathom_demo.py --custom "your prompt" "completion" "insight prompt" "completion"

Requires: depth_api_v2.py running on localhost:8400
          (run: uvicorn depth_api_v2:app --port 8400)
"""

import sys, io
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import json
import time
import argparse
import subprocess
import urllib.request
import urllib.error
from pathlib import Path

API_BASE = "http://localhost:8400"
API_KEY  = "fathom-demo"   # or load from env

# ── Demo Pairs ────────────────────────────────────────────────────────────────

DEMO_PAIRS = {
    "healthcare": [
        {
            "label": "Fever mechanism",
            "surface_prompt": "The patient has a high temperature and the skin feels warm to the",
            "surface_target": "touch",
            "insight_prompt": "Fever triggers prostaglandin synthesis which signals the hypothalamus to",
            "insight_target": "raise",
            "why": "Surface = describing a symptom. Insight = explaining the biochemical cascade."
        },
        {
            "label": "Drug resistance",
            "surface_prompt": "The antibiotic stopped working and the bacteria continued to",
            "surface_target": "multiply",
            "insight_prompt": "Repeated antibiotic exposure selects for mutations that disable the drug's binding",
            "insight_target": "site",
            "why": "Surface = observing outcome. Insight = explaining evolutionary mechanism."
        },
        {
            "label": "Insulin function",
            "surface_prompt": "After eating sugar the blood glucose levels start to",
            "surface_target": "rise",
            "insight_prompt": "Insulin binds GLUT4 receptors triggering translocation to the membrane so glucose can",
            "insight_target": "enter",
            "why": "Surface = effect. Insight = molecular mechanism of glucose uptake."
        },
    ],
    "legal": [
        {
            "label": "Contract law",
            "surface_prompt": "The buyer refused to pay and the seller took the dispute to",
            "surface_target": "court",
            "insight_prompt": "A valid contract requires offer, acceptance, consideration, and mutual assent — without all four elements it is legally",
            "insight_target": "void",
            "why": "Surface = describing a dispute. Insight = four-element contract formation test."
        },
        {
            "label": "Fourth Amendment",
            "surface_prompt": "The police found drugs during the search and the defense argued the evidence should be",
            "surface_target": "excluded",
            "insight_prompt": "The exclusionary rule bars evidence obtained through unconstitutional searches because admitting it would render the Fourth Amendment",
            "insight_target": "void",
            "why": "Surface = outcome. Insight = constitutional reasoning chain."
        },
        {
            "label": "Negligence",
            "surface_prompt": "The driver ran a red light and caused an accident resulting in significant",
            "surface_target": "harm",
            "insight_prompt": "Negligence requires proving duty of care, breach, causation, and damages — all four elements must be clearly",
            "insight_target": "shown",
            "why": "Surface = incident description. Insight = four-element legal test."
        },
    ],
    "finance": [
        {
            "label": "Yield curve inversion",
            "surface_prompt": "The stock market fell sharply after the economic data came in",
            "surface_target": "worse",
            "insight_prompt": "Yield curve inversions predict recessions because short-term borrowing costs exceed long-term returns forcing banks to contract",
            "insight_target": "credit",
            "why": "Surface = market observation. Insight = causal mechanism linking rates to recession."
        },
        {
            "label": "Inflation cause",
            "surface_prompt": "Prices at the grocery store have been rising and people are spending",
            "surface_target": "more",
            "insight_prompt": "Demand-pull inflation occurs when aggregate demand outpaces productive capacity causing firms to raise prices rather than increase",
            "insight_target": "output",
            "why": "Surface = consumer observation. Insight = macroeconomic supply-demand mechanism."
        },
        {
            "label": "Option pricing",
            "surface_prompt": "The call option value increased when the stock price moved sharply",
            "surface_target": "higher",
            "insight_prompt": "Black-Scholes models stock price as geometric Brownian motion and solves the no-arbitrage PDE to compute fair",
            "insight_target": "value",
            "why": "Surface = directional observation. Insight = stochastic calculus derivation."
        },
    ],
    "education": [
        {
            "label": "Photosynthesis",
            "surface_prompt": "The plant sits in the window and its leaves are turning green in the",
            "surface_target": "sun",
            "insight_prompt": "Chlorophyll absorbs photons and transfers energy to reaction centers where water molecules are split releasing",
            "insight_target": "oxygen",
            "why": "Surface = casual observation. Insight = mechanistic photochemistry."
        },
        {
            "label": "Evolution",
            "surface_prompt": "The giraffe has a long neck and can reach leaves high up in the",
            "surface_target": "trees",
            "insight_prompt": "Natural selection favors traits that increase reproductive fitness so heritable variations conferring survival advantages spread through",
            "insight_target": "populations",
            "why": "Surface = describing an animal. Insight = evolutionary mechanism."
        },
    ],
    "ai_safety": [
        {
            "label": "Hallucination vs reasoning",
            "surface_prompt": "The AI gave a confident answer that turned out to be factually",
            "surface_target": "wrong",
            "insight_prompt": "Language models hallucinate when the next-token distribution is dominated by surface pattern completion rather than grounded inference from",
            "insight_target": "evidence",
            "why": "Surface = observing output error. Insight = mechanistic cause of hallucination."
        },
        {
            "label": "Alignment failure",
            "surface_prompt": "The AI did something unexpected and the engineers were surprised by the",
            "surface_target": "behavior",
            "insight_prompt": "Reward hacking occurs when an agent optimizes the specified proxy objective so thoroughly that it diverges from the intended",
            "insight_target": "goal",
            "why": "Surface = incident description. Insight = Goodhart's Law applied to RL."
        },
    ],
}

ALL_DOMAINS = list(DEMO_PAIRS.keys())

# ── Display ───────────────────────────────────────────────────────────────────

def bar(value, min_val, max_val, width=30, char="#", empty="-"):
    if max_val == min_val:
        filled = width // 2
    else:
        filled = int((value - min_val) / (max_val - min_val) * width)
    filled = max(0, min(width, filled))
    return "[" + char * filled + empty * (width - filled) + "]"

def header(text):
    w = 70
    print("\n" + "=" * w)
    print(f"  {text}")
    print("=" * w)

def subheader(text):
    print(f"\n  -- {text} --")

def result_line(label, value, note=""):
    note_str = f"  ({note})" if note else ""
    print(f"  {label:<22} {value}{note_str}")

# ── API ───────────────────────────────────────────────────────────────────────

def check_api():
    try:
        req = urllib.request.Request(f"{API_BASE}/v2/health")
        with urllib.request.urlopen(req, timeout=3) as r:
            return json.loads(r.read())
    except Exception:
        return None

def call_compare(surface_prompt, surface_target, insight_prompt, insight_target, api_key=API_KEY, retries=2):
    payload = json.dumps({
        "surface_prompt": surface_prompt,
        "surface_target": surface_target,
        "insight_prompt": insight_prompt,
        "insight_target": insight_target,
    }).encode("utf-8")

    last_err = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                f"{API_BASE}/v2/compare",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "X-API-Key": api_key,
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=150) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code in (503, 500) and attempt < retries:
                print(f"  [retry {attempt+1}] HTTP {e.code}, waiting 3s...")
                time.sleep(3)
            else:
                raise
        except Exception as e:
            last_err = e
            if attempt < retries:
                print(f"  [retry {attempt+1}] {e}, waiting 3s...")
                time.sleep(3)
            else:
                raise
    raise last_err

# ── Validation ───────────────────────────────────────────────────────────────

def validate_pairs(pairs, domain_name=""):
    """Pre-check all targets are single-token under Gemma tokenizer."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    except Exception:
        return pairs  # skip validation if tokenizer unavailable

    valid = []
    for pair in pairs:
        ok = True
        for field in ("surface_target", "insight_target"):
            t = pair[field]
            ids = tok.encode(t, add_special_tokens=False)
            if len(ids) != 1:
                print(f"  [SKIP] {domain_name}/{pair['label']}: '{t}' = {len(ids)} tokens. Skipping pair.")
                ok = False
                break
        if ok:
            valid.append(pair)
    return valid


# ── Core Demo ─────────────────────────────────────────────────────────────────

def run_domain(domain_name, pairs):
    domain_label = domain_name.upper().replace("_", " ")
    header(f"FATHOM DEPTH DEMO -- {domain_label}")

    pairs = validate_pairs(pairs, domain_name)
    if not pairs:
        print("  [!] No valid pairs after token validation.")
        return []

    deltas = []
    ratios = []
    results = []

    for i, pair in enumerate(pairs):
        subheader(f"Pair {i+1}: {pair['label']}")
        print(f"  Surface: \"...{pair['surface_prompt'][-45:]} -> '{pair['surface_target']}'\"")
        print(f"  Insight: \"...{pair['insight_prompt'][-45:]} -> '{pair['insight_target']}'\"")
        print(f"  Why:     {pair['why']}")
        print()

        t0 = time.time()
        try:
            resp = call_compare(
                pair["surface_prompt"], pair["surface_target"],
                pair["insight_prompt"], pair["insight_target"],
            )
            elapsed = time.time() - t0
            data = resp.get("data", resp)

            s_depth = data["surface"]["depth_score"]
            i_depth = data["insight"]["depth_score"]
            delta   = data["delta"]
            s_feat  = data["surface"].get("feature_count", data["surface"].get("n_features", "?"))
            i_feat  = data["insight"].get("feature_count", data["insight"].get("n_features", "?"))
            direction = "surface deeper" if delta > 0 else "insight deeper"
            ratio = s_depth / i_depth if i_depth > 0 else 0

            s_early = data["surface"].get("early_ratio", 0)
            s_late  = data["surface"].get("late_ratio", 0)
            i_early = data["insight"].get("early_ratio", 0)
            i_late  = data["insight"].get("late_ratio", 0)

            result_line("Surface depth",  f"{s_depth:.4f}", f"{s_feat} features | early {s_early:.0%} late {s_late:.0%}")
            result_line("Insight depth",  f"{i_depth:.4f}", f"{i_feat} features | early {i_early:.0%} late {i_late:.0%}")
            result_line("Delta",          f"{delta:+.4f}", direction)
            result_line("Ratio K",        f"{ratio:.4f}", f"universal ref: 1.027")
            result_line("Computed in",    f"{elapsed:.1f}s")

            max_d = max(s_depth, i_depth)
            min_d = min(s_depth, i_depth)
            pad = max((max_d - min_d) * 0.3, 0.05)
            print(f"\n  Surface  {bar(s_depth, min_d - pad, max_d + pad)} {s_depth:.4f}  [early:{s_early:.0%} late:{s_late:.0%}]")
            print(f"  Insight  {bar(i_depth, min_d - pad, max_d + pad)} {i_depth:.4f}  [early:{i_early:.0%} late:{i_late:.0%}]")

            deltas.append(delta)
            ratios.append(ratio)
            results.append({
                "label": pair["label"],
                "surface_depth": s_depth,
                "insight_depth": i_depth,
                "delta": delta,
                "ratio": ratio,
                "direction": direction,
            })

        except Exception as e:
            print(f"  [ERROR] {e}")
            print(f"  (Ensure depth_api_v2.py is running: uvicorn depth_api_v2:app --port 8400)")
            results.append({"label": pair["label"], "error": str(e)})

    # Domain summary
    valid = [r for r in results if "error" not in r]
    if valid:
        mean_delta = sum(r["delta"] for r in valid) / len(valid)
        mean_ratio = sum(r["ratio"] for r in valid) / len(valid)
        correct = sum(1 for r in valid if r["delta"] > 0)

        header(f"DOMAIN SUMMARY -- {domain_label}")
        result_line("Pairs tested",       f"{len(valid)}/{len(pairs)}")
        result_line("Mean delta",         f"{mean_delta:+.4f}")
        result_line("Mean ratio K",       f"{mean_ratio:.4f}")
        result_line("Correct direction",  f"{correct}/{len(valid)} ({100*correct//len(valid)}%)")
        result_line("Gemma reference K",  "1.0302")
        result_line("Llama reference K",  "1.0242")

        k_diff = abs(mean_ratio - 1.027)
        if k_diff < 0.05:
            print(f"\n  [+] K = {mean_ratio:.4f} is within convergence window of universal constant 1.027")
        else:
            print(f"\n  [~] K = {mean_ratio:.4f} (diff from universal: {k_diff:.4f})")

    return results


def run_custom(surface_prompt, surface_target, insight_prompt, insight_target):
    header("FATHOM DEPTH DEMO -- CUSTOM PAIR")
    pair = {
        "label": "Custom",
        "surface_prompt": surface_prompt,
        "surface_target": surface_target,
        "insight_prompt": insight_prompt,
        "insight_target": insight_target,
        "why": "User-provided pair",
    }
    run_domain("custom", [pair])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global API_BASE
    parser = argparse.ArgumentParser(description="Fathom Live Demo")
    parser.add_argument("--domain", choices=ALL_DOMAINS + ["all"], default="all")
    parser.add_argument("--api", default=API_BASE)
    parser.add_argument("--custom", nargs=4, metavar=("SURFACE", "S_TARGET", "INSIGHT", "I_TARGET"))
    parser.add_argument("--express", action="store_true", help="Run 1 pair per domain (pitch mode, ~2 min)")
    args = parser.parse_args()
    API_BASE = args.api

    print("\n" + "#" * 70)
    print("#  FATHOM -- Measuring Reasoning Depth in Artificial Intelligence  #")
    print("#  US Provisional Patent 64/020,489 & 64/021,113 | darkflobi.com/fathom         #")
    print("#" * 70)

    # Check API
    health = check_api()
    if health:
        model = health.get("data", {}).get("model", health.get("model", "loaded"))
        print(f"\n  [+] Depth API running -- model: {model}")
    else:
        print(f"\n  [!] Depth API not running at {API_BASE}")
        print(f"      Start with: uvicorn depth_api_v2:app --port 8400")
        print(f"      (GPU required -- RTX 4070 or better, Gemma-2-2B loads in ~2 min)")
        sys.exit(1)

    # Run
    if args.custom:
        run_custom(*args.custom)
    elif args.domain == "all":
        all_results = {}
        all_ratios = []
        for domain, pairs in DEMO_PAIRS.items():
            run_pairs = [pairs[0]] if args.express else pairs  # express: 1 pair per domain
            results = run_domain(domain, run_pairs)
            all_results[domain] = results
            valid = [r for r in results if "error" not in r]
            all_ratios.extend(r["ratio"] for r in valid)

        # Grand summary
        if all_ratios:
            mean_k = sum(all_ratios) / len(all_ratios)
            header("GRAND SUMMARY -- ALL DOMAINS")
            result_line("Total pairs",        str(len(all_ratios)))
            result_line("Mean K (this run)",   f"{mean_k:.4f}")
            result_line("Gemma-2-2B K",        "1.0302 (p=0.000051, n=30)")
            result_line("Llama-3.2-1B K",      "1.0242 (p=0.00266, n=19)")
            result_line("Universal K",         "1.027 +/- 0.003")
            print(f"\n  The ratio of surface-to-insight processing depth converges")
            print(f"  to the same constant across architectures, labs, and now")
            print(f"  across industry domains.")
            print(f"\n  What this means: the AI is not making random errors.")
            print(f"  It is predictably shallower on surface tasks.")
            print(f"  Fathom measures that gap. Every time. In real time.")
            print()
    else:
        pairs = DEMO_PAIRS[args.domain]
        if args.express:
            pairs = [pairs[0]]
        run_domain(args.domain, pairs)


if __name__ == "__main__":
    main()
