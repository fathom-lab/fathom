# -*- coding: utf-8 -*-
"""
upload_atlas_v0_3.py -- Publish the Fathom Cognitive Atlas v0.3
                         as a NEW Zenodo record with its own concept DOI.

This is NOT a new version of the paper concept (10.5281/zenodo.19326174).
It is a separate scientific artifact that will version independently
(v0.1 -> v0.2 -> v0.2.1 -> v0.3 and forward to v0.4, v1.0, ...).
Creates a brand-new deposit and uploads the atlas bundle.

Token: read from ZENODO_TOKEN environment variable only.

Usage:
    python zenodo_v14/upload_atlas_v0_3.py                # dry run
    ZENODO_TOKEN=xxx python zenodo_v14/upload_atlas_v0_3.py --draft
    ZENODO_TOKEN=xxx python zenodo_v14/upload_atlas_v0_3.py --publish
"""

import json
import os
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    print("[FATAL] requests not installed")
    sys.exit(1)

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "zenodo_v14" / "fathom_atlas_v0.3.zip"
BUNDLE_NAME = "fathom_atlas_v0.3.zip"

ZENODO_BASE = "https://zenodo.org/api"

METADATA = {
    "metadata": {
        "upload_type": "dataset",
        "title": (
            "The Fathom Cognitive Atlas v0.3: Pre-registered "
            "Cross-Architecture Replication of the RLHF Attractor"
        ),
        "creators": [
            {"name": "Rodabaugh, Alexander", "affiliation": "Fathom Intelligence"},
        ],
        "description": (
            "<p><strong>The Fathom Cognitive Atlas</strong> is the first "
            "public cross-architecture cognitive state atlas of open-weight "
            "language models. For every model included, we run a fixed probe "
            "set of 90 prompts across 6 categories (retrieval, reasoning, "
            "refusal, creative, adversarial, hallucination) and capture the "
            "full per-token cognitive trajectory (D, entropy, logprob, "
            "top2_margin) for every (model, prompt). The measurement "
            "primitive D = cos(h<sup>(L)</sup>, W<sub>U</sub>[y]) is "
            "architecturally universal: it uses only the final-layer "
            "residual stream and the unembedding matrix, which every "
            "transformer has by construction.</p>"
            "<h3>v0.3 contents</h3>"
            "<ul>"
            "<li><strong>12 open-model captures</strong> (6 base + 6 "
            "instruct) across 3 families (Gemma, Llama, Qwen) and 2 "
            "generations (Gemma-2/Gemma-3, Llama-3.2 3B/1B, "
            "Qwen2.5 3B/1.5B).</li>"
            "<li><strong>1 exploratory closed-model capture</strong> "
            "(OpenAI GPT-4o-mini via logprob API).</li>"
            "<li><strong>Probe set v0.1</strong> - 90 prompts, SHA-pinned "
            "(34a7254920bce654), hand-curated, versioned.</li>"
            "<li><strong>Analysis scripts</strong> - atlas_capture.py, "
            "atlas_analyze.py, atlas_attractor.py, atlas_attractor_n5.py, "
            "atlas_entropy.py, attractor_bootstrap.py, "
            "attractor_bootstrap_entropy.py, validate_entropy_estimator.py. "
            "All reproducible from committed JSON in under a minute on CPU.</li>"
            "</ul>"
            "<h3>Headline result (pre-registered)</h3>"
            "<p>The RLHF Attractor direction prediction test was "
            "pre-registered on 2026-04-10 (sealed in git before any v0.3 "
            "data was captured; see PREREG_v0.3_attractor_replication.md). "
            "The sealed decision rule: H1 SUPPORTED if and only if all "
            "three conditions hold:</p>"
            "<ul>"
            "<li>mean entropy early-window LOO cosine >= 0.40</li>"
            "<li>permutation test p (one-sided, >= 2000 perms) < 0.05</li>"
            "<li>bootstrap 95% CI lower bound > 0</li>"
            "</ul>"
            "<p>At n = 6 model family pairs:</p>"
            "<ul>"
            "<li>mean entropy early-window LOO cosine = <strong>+0.769</strong></li>"
            "<li>permutation p = <strong>0.0315</strong></li>"
            "<li>bootstrap 95% CI = <strong>[+0.571, +0.869]</strong></li>"
            "</ul>"
            "<p><strong>H1 SUPPORTED.</strong> All 6 of 6 families show "
            "positive LOO cosine in the pre-registered early window. "
            "The analysis script was run without modification after the "
            "decision rule was applied. Full per-family numbers, bootstrap "
            "distributions, permutation tests, and limitations are in "
            "FINDINGS_v0.3.md inside the bundle.</p>"
            "<h3>Bulletproof audit</h3>"
            "<p>Prior to v0.3, a rigor audit (FINDINGS_bulletproof_audit.md) "
            "identified real over-framing in the v0.2.1 writeup: the "
            "attractor claim at n=3 was structurally below significance "
            "(permutation floor p_min = 0.125). The audit added bootstrap "
            "CIs, permutation tests, and top-k entropy estimator validation "
            "across 3 model families (mean shape-correlation r = 0.902 "
            "between full-distribution entropy and top-5 API-exposed "
            "entropy). v0.3 addresses the audit's n=3 limitation by "
            "expanding to n=6 under the sealed pre-registration.</p>"
            "<h3>Findings documents included</h3>"
            "<ul>"
            "<li>FINDINGS_v0.1.md - seed release, 3 base models</li>"
            "<li>FINDINGS_v0.2.md - 6-model RLHF convergence observation</li>"
            "<li>FINDINGS_v0.2.1_attractor.md - first attractor formalization "
            "(superseded by v0.3)</li>"
            "<li>FINDINGS_entropy_bridge.md - top-k entropy proxy for "
            "closed-model extension</li>"
            "<li>FINDINGS_bulletproof_audit.md - rigor audit that caught "
            "v0.2.1 over-framing</li>"
            "<li>PREREG_v0.3_attractor_replication.md - sealed "
            "pre-registration for v0.3</li>"
            "<li>FINDINGS_v0.3.md - <strong>H1 SUPPORTED</strong>, the "
            "current headline result</li>"
            "</ul>"
            "<h3>Honest limitations</h3>"
            "<ul>"
            "<li>n = 6 includes scale/generation variants within existing "
            "atlas families; true independent-family replication (Phi, OLMo, "
            "Mistral) is deferred to v0.4.</li>"
            "<li>The D-axis permutation test does not reach conventional "
            "significance even at n = 6 (p = 0.21); the primary claim is "
            "on entropy early-window only.</li>"
            "<li>n = 6 is small for claims about RLHF as a universal "
            "cognitive target; stable cluster-level claims require n >= 15.</li>"
            "</ul>"
            "<h3>Companion paper</h3>"
            "<p>The Fathom paper (Zenodo concept DOI "
            "10.5281/zenodo.19326174) defines the measurement primitives "
            "(K, C, C_delta, S, D) and presents their pilot-scale "
            "validation. This atlas is the companion dataset that applies "
            "them at cross-architecture scale on a fixed probe set.</p>"
            "<h3>License</h3>"
            "<p>CC-BY-4.0 on the dataset. MIT on the analysis code.</p>"
            "<p>Related provisional patents: "
            "US 64/020,489 . 64/021,113 . 64/026,964.</p>"
        ),
        "keywords": [
            "Fathom",
            "cognitive atlas",
            "sparse autoencoder",
            "mechanistic interpretability",
            "cross-architecture measurement",
            "RLHF alignment",
            "pre-registration",
            "replication",
            "leave-one-out validation",
            "bootstrap",
            "permutation test",
            "transformer internals",
            "open-weight language models",
            "cognitive measurement",
            "alignment audit",
        ],
        "license": "cc-by-4.0",
        "publication_date": "2026-04-10",
        "related_identifiers": [
            {
                "relation": "isSupplementTo",
                "identifier": "10.5281/zenodo.19326174",
                "resource_type": "publication-workingpaper",
                "scheme": "doi",
            },
        ],
    }
}


def fail(msg):
    print(f"[FATAL] {msg}")
    sys.exit(1)


def main():
    args = sys.argv[1:]
    dry_run = "--publish" not in args and "--draft" not in args
    do_publish = "--publish" in args

    print("=" * 68)
    print("UPLOAD: Fathom Cognitive Atlas v0.3 to Zenodo (NEW DEPOSIT)")
    print("=" * 68)
    print(f"Bundle:       {BUNDLE.name}")
    print(f"Bundle size:  {BUNDLE.stat().st_size:,} bytes"
          if BUNDLE.exists() else "Bundle:       MISSING")
    print(f"Type:         NEW deposit (not a version bump)")
    print(f"Relates to:   10.5281/zenodo.19326174 (paper concept, isSupplementTo)")
    print(f"Mode:         {'DRY RUN' if dry_run else ('PUBLISH' if do_publish else 'DRAFT ONLY')}")
    print()

    if dry_run:
        print("[DRY RUN] No HTTP calls. Metadata preview:")
        print("-" * 68)
        print(f"  title:    {METADATA['metadata']['title']}")
        print(f"  type:     {METADATA['metadata']['upload_type']}")
        print(f"  license:  {METADATA['metadata']['license']}")
        print(f"  keywords: {len(METADATA['metadata']['keywords'])} tags")
        print(f"  rel_id:   isSupplementTo 10.5281/zenodo.19326174")
        print()
        print("[DRY RUN] To draft only:")
        print("  ZENODO_TOKEN=<token> python zenodo_v14/upload_atlas_v0_3.py --draft")
        print("[DRY RUN] To publish (irreversible):")
        print("  ZENODO_TOKEN=<token> python zenodo_v14/upload_atlas_v0_3.py --publish")
        return

    if not BUNDLE.exists():
        fail("Bundle missing. Run: python zenodo_v14/build_bundles.py")

    token = os.environ.get("ZENODO_TOKEN", "").strip()
    if not token:
        fail("ZENODO_TOKEN env var not set")

    headers = {"Authorization": f"Bearer {token}"}

    # 1. Create new empty deposit
    print("[1/4] Creating NEW deposit ...")
    r = requests.post(
        f"{ZENODO_BASE}/deposit/depositions",
        headers={**headers, "Content-Type": "application/json"},
        json={},
    )
    if r.status_code != 201:
        fail(f"new deposit failed: {r.status_code} {r.text[:300]}")
    dep = r.json()
    new_id = dep["id"]
    concept_id = dep.get("conceptrecid") or "(pending)"
    print(f"      new record ID:   {new_id}")
    print(f"      new concept ID:  {concept_id}")

    # 2. Upload bundle
    print(f"[2/4] Uploading {BUNDLE_NAME} "
          f"({BUNDLE.stat().st_size:,} bytes) ...")
    bucket_url = dep["links"]["bucket"]
    with open(BUNDLE, "rb") as f:
        r = requests.put(
            f"{bucket_url}/{BUNDLE_NAME}",
            data=f,
            headers={"Authorization": f"Bearer {token}"},
        )
    if r.status_code not in (200, 201):
        fail(f"upload failed: {r.status_code} {r.text[:300]}")
    print(f"      uploaded: {r.json().get('size', 0):,} bytes")

    # 3. Set metadata
    print("[3/4] Setting metadata ...")
    r = requests.put(
        f"{ZENODO_BASE}/deposit/depositions/{new_id}",
        json=METADATA,
        headers={**headers, "Content-Type": "application/json"},
    )
    if r.status_code != 200:
        fail(f"metadata failed: {r.status_code} {r.text[:300]}")
    print(f"      metadata set")

    # 4. Publish or leave as draft
    if do_publish:
        print("[4/4] Publishing ...")
        r = requests.post(
            f"{ZENODO_BASE}/deposit/depositions/{new_id}/actions/publish",
            headers=headers,
        )
        if r.status_code != 202:
            fail(f"publish failed: {r.status_code} {r.text[:300]}")
        result = r.json()
        doi = result.get("doi", "unknown")
        url = result.get("links", {}).get("record_html", "unknown")
        concept_doi = result.get("conceptdoi", "unknown")
        print(f"      PUBLISHED")
        print(f"      DOI:         {doi}")
        print(f"      Concept DOI: {concept_doi}")
        print(f"      URL:         {url}")
        out_record = {
            "new_record_id": new_id,
            "doi": doi,
            "concept_doi": concept_doi,
            "url": url,
            "bundle": BUNDLE_NAME,
            "supplement_to": "10.5281/zenodo.19326174",
        }
        with open(ROOT / "zenodo_v14" / "atlas_v0_3_published.json",
                  "w", encoding="utf-8") as f:
            json.dump(out_record, f, indent=2)
    else:
        print("[4/4] Draft created (use --publish to go live)")
        print(f"      Preview: https://zenodo.org/deposit/{new_id}")


if __name__ == "__main__":
    main()
