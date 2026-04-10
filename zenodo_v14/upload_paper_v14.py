# -*- coding: utf-8 -*-
"""
upload_paper_v14.py -- Publish Fathom Paper v14 to Zenodo.

Creates a new version on the existing Fathom concept record
(10.5281/zenodo.19326174), branching from v13 (record 19478434,
published 2026-04-09). Uploads fathom_paper_v14.zip and sets the
v14 metadata, then optionally publishes.

Token: read from ZENODO_TOKEN environment variable only. Never
hardcoded, never committed.

Safety:
  - Default mode is DRY RUN. No HTTP calls.
  - Pass --draft to create the draft but NOT publish.
  - Pass --publish to create the draft and publish it.
  - Any upload requires ZENODO_TOKEN to be set. If missing the
    script aborts with an explicit error.

Usage:
    python zenodo_v14/upload_paper_v14.py                # dry run
    ZENODO_TOKEN=xxx python zenodo_v14/upload_paper_v14.py --draft
    ZENODO_TOKEN=xxx python zenodo_v14/upload_paper_v14.py --publish
"""

import json
import os
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    print("[FATAL] requests package not installed")
    sys.exit(1)

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "zenodo_v14" / "fathom_paper_v14.zip"
BUNDLE_NAME = "fathom_paper_v14.zip"

ZENODO_BASE = "https://zenodo.org/api"
CONCEPT_ID = 19326174  # Fathom concept DOI record
LATEST_V13_ID = 19478434  # v13: "Multi-Layer Coordinated Feature Suppression"

METADATA = {
    "metadata": {
        "upload_type": "publication",
        "publication_type": "workingpaper",
        "title": (
            "Fathom: Cognitive Measurement Instruments for Transformer "
            "Internals via SAE Feature Coherence Geometry"
        ),
        "creators": [
            {"name": "Rodabaugh, Alexander", "affiliation": "Fathom Intelligence"},
        ],
        "description": (
            "<p>Fathom is a suite of cognitive measurement instruments for "
            "transformer internals built on sparse autoencoder (SAE) feature "
            "coherence geometry. v14 extends prior versions with a fifth "
            "measurement primitive: the honesty axis "
            "<em>D = cos(h<sup>(L)</sup>, W<sub>U</sub>[y])</em>, the cosine "
            "similarity between the final-layer residual stream and the "
            "unembedding row of the chosen token. D is computable on any "
            "transformer with a logprob or residual stream interface, with "
            "no SAE required.</p>"
            "<h3>Measurement</h3>"
            "<ul>"
            "<li>Commitment intensity S via inverse participation ratio. "
            "Pooled d = 0.494 across three independent designs "
            "(Fisher p = 0.0008).</li>"
            "<li>Head-to-head beat over logit entropy "
            "(AUC 0.663 vs 0.607, p = 0.013).</li>"
            "<li>Three SAE-free probes across four architectures fail "
            "(Fisher p > 0.6).</li>"
            "</ul>"
            "<h3>Diagnosis</h3>"
            "<ul>"
            "<li>Gini coefficient of per-layer coherence achieves "
            "AUC = 0.685 on n = 200 TruthfulQA pairs, while scalar "
            "depth (K) is at chance.</li>"
            "<li>Cognitive autopsy localizes the fault: layer 11 is the "
            "primary locus in 63% of hallucinations (p = 2.7e-12).</li>"
            "</ul>"
            "<h3>Intervention</h3>"
            "<ul>"
            "<li>Best-of-N sampling scored by Cognitive Health Score "
            "improves on 100% of test prompts vs greedy.</li>"
            "<li>Coordinated suppression of 220 hallucination-associated "
            "features across 22 layers produces 33% first-token changes "
            "on held-out data.</li>"
            "</ul>"
            "<h3>Honesty axis (new in v14)</h3>"
            "<ul>"
            "<li>Cross-architecture D-axis pilot on 30 matched "
            "direct/adversarial prompt pairs across six models from three "
            "families (Gemma-2-2B, Llama-3.2-3B, Qwen2.5-3B, each plus "
            "and minus RLHF). Fisher combined p &lt; 10<sup>-14</sup>.</li>"
            "<li>In the pre-registered early window (tokens 0-6), the "
            "adversarial-framing direction replicates on all six models, "
            "with the strongest single signal at d<sub>z</sub> = -1.86 "
            "(p = 4.6e-11) on Qwen2.5-3B base.</li>"
            "<li>Preliminary orthogonality data indicates D correlates "
            "strongly with C_delta (r ~ 0.99 on paired deltas); D is "
            "therefore reframed as an SAE-free proxy for late-layer "
            "commitment geometry rather than as an independent axis.</li>"
            "</ul>"
            "<p>A separate Zenodo deposit publishes the "
            "<strong>Fathom Cognitive Atlas v0.3</strong>, a companion "
            "dataset that applies these instruments at cross-architecture "
            "scale on a fixed probe set of 90 prompts across 6 models. "
            "The atlas includes a pre-registered replication of the "
            "RLHF Attractor direction test (mean leave-one-out cosine "
            "= +0.769, bootstrap 95% CI [+0.571, +0.869], permutation "
            "p = 0.0315).</p>"
            "<p>Related provisional patents: "
            "US 64/020,489 . 64/021,113 . 64/026,964.</p>"
        ),
        "keywords": [
            "Fathom",
            "sparse autoencoder",
            "mechanistic interpretability",
            "cognitive measurement",
            "hallucination detection",
            "commitment intensity",
            "inverse participation ratio",
            "transformer internals",
            "RLHF",
            "cross-architecture",
            "pre-registration",
            "ICML 2026",
        ],
        "license": "cc-by-4.0",
        "publication_date": "2026-04-10",
        "related_identifiers": [
            {
                "relation": "isNewVersionOf",
                "identifier": f"10.5281/zenodo.{LATEST_V13_ID}",
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
    print("UPLOAD: Fathom Paper v14 to Zenodo")
    print("=" * 68)
    print(f"Bundle:       {BUNDLE.name}")
    print(f"Bundle size:  {BUNDLE.stat().st_size:,} bytes" if BUNDLE.exists()
          else "Bundle:       MISSING -- run build_bundles.py first")
    print(f"Concept DOI:  10.5281/zenodo.{CONCEPT_ID}")
    print(f"Branch from:  10.5281/zenodo.{LATEST_V13_ID} (v13)")
    print(f"Mode:         {'DRY RUN' if dry_run else ('PUBLISH' if do_publish else 'DRAFT ONLY')}")
    print()

    if dry_run:
        print("[DRY RUN] No HTTP calls. Metadata preview:")
        print("-" * 68)
        print(f"  title:    {METADATA['metadata']['title']}")
        print(f"  type:     {METADATA['metadata']['upload_type']} / "
              f"{METADATA['metadata']['publication_type']}")
        print(f"  license:  {METADATA['metadata']['license']}")
        print(f"  keywords: {len(METADATA['metadata']['keywords'])} tags")
        print(f"  rel_id:   isNewVersionOf "
              f"10.5281/zenodo.{LATEST_V13_ID}")
        print()
        print("[DRY RUN] To draft (no publish):")
        print("  ZENODO_TOKEN=<token> python zenodo_v14/upload_paper_v14.py --draft")
        print("[DRY RUN] To publish:")
        print("  ZENODO_TOKEN=<token> python zenodo_v14/upload_paper_v14.py --publish")
        return

    if not BUNDLE.exists():
        fail("Bundle missing. Run: python zenodo_v14/build_bundles.py")

    token = os.environ.get("ZENODO_TOKEN", "").strip()
    if not token:
        fail("ZENODO_TOKEN env var not set. Refusing to run.")

    headers = {"Authorization": f"Bearer {token}"}

    # 1. Create new version draft from v13
    print(f"[1/5] Creating new version from v13 (ID: {LATEST_V13_ID})...")
    r = requests.post(
        f"{ZENODO_BASE}/deposit/depositions/{LATEST_V13_ID}/actions/newversion",
        headers=headers,
    )
    if r.status_code != 201:
        fail(f"newversion failed: {r.status_code} {r.text[:300]}")
    new_dep = r.json()
    draft_url = new_dep["links"].get("latest_draft")
    if not draft_url:
        fail("no latest_draft link in newversion response")
    new_id = int(draft_url.rstrip("/").split("/")[-1])
    print(f"      new draft ID: {new_id}")

    # 2. Get current draft state (for bucket URL + existing files to delete)
    print(f"[2/5] Clearing files inherited from v13 ...")
    r = requests.get(f"{ZENODO_BASE}/deposit/depositions/{new_id}",
                      headers=headers)
    if r.status_code != 200:
        fail(f"GET draft failed: {r.status_code}")
    draft = r.json()
    for f_info in draft.get("files", []):
        fid = f_info["id"]
        fname = f_info.get("filename", fid)
        rd = requests.delete(
            f"{ZENODO_BASE}/deposit/depositions/{new_id}/files/{fid}",
            headers=headers,
        )
        print(f"      deleted: {fname} ({rd.status_code})")

    # 3. Upload the bundle zip
    print(f"[3/5] Uploading {BUNDLE_NAME} "
          f"({BUNDLE.stat().st_size:,} bytes) ...")
    bucket_url = draft["links"]["bucket"]
    with open(BUNDLE, "rb") as f:
        r = requests.put(
            f"{bucket_url}/{BUNDLE_NAME}",
            data=f,
            headers={"Authorization": f"Bearer {token}"},
        )
    if r.status_code not in (200, 201):
        fail(f"upload failed: {r.status_code} {r.text[:300]}")
    print(f"      uploaded: {r.json().get('size', 0):,} bytes")

    # 4. Set metadata
    print(f"[4/5] Setting metadata ...")
    r = requests.put(
        f"{ZENODO_BASE}/deposit/depositions/{new_id}",
        json=METADATA,
        headers={**headers, "Content-Type": "application/json"},
    )
    if r.status_code != 200:
        fail(f"set metadata failed: {r.status_code} {r.text[:300]}")
    print(f"      metadata set for {new_id}")

    # 5. Publish or leave as draft
    if do_publish:
        print(f"[5/5] Publishing ...")
        r = requests.post(
            f"{ZENODO_BASE}/deposit/depositions/{new_id}/actions/publish",
            headers=headers,
        )
        if r.status_code != 202:
            fail(f"publish failed: {r.status_code} {r.text[:300]}")
        result = r.json()
        doi = result.get("doi", "unknown")
        url = result.get("links", {}).get("record_html", "unknown")
        print(f"      PUBLISHED")
        print(f"      DOI: {doi}")
        print(f"      URL: {url}")
        out_record = {
            "new_record_id": new_id,
            "doi": doi,
            "url": url,
            "concept_doi": f"10.5281/zenodo.{CONCEPT_ID}",
            "supersedes": f"10.5281/zenodo.{LATEST_V13_ID}",
            "bundle": BUNDLE_NAME,
        }
        with open(ROOT / "zenodo_v14" / "paper_v14_published.json",
                  "w", encoding="utf-8") as f:
            json.dump(out_record, f, indent=2)
    else:
        print(f"[5/5] Draft created (use --publish to go live)")
        print(f"      Preview: https://zenodo.org/deposit/{new_id}")


if __name__ == "__main__":
    main()
