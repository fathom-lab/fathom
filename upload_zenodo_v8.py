# -*- coding: utf-8 -*-
"""Upload Fathom v8 (Spectrum Analysis) to Zenodo."""

import json
import requests
import sys

TOKEN = "REDACTED_SEE_SECRETS"
BASE = "https://zenodo.org/api"
ZIP_PATH = "zenodo_v8/fathom_spectrum_v8.zip"

HEADERS = {"Authorization": f"Bearer {TOKEN}"}

METADATA = {
    "metadata": {
        "upload_type": "dataset",
        "title": (
            "Fathom v8: Coherence Spectrum Analysis — Hallucination as "
            "Concentration in Per-Layer Feature Coherence"
        ),
        "creators": [{"name": "Rodabaugh, Alexander", "affiliation": "Independent"}],
        "description": (
            "<p><strong>New finding:</strong> The SHAPE of per-layer SAE feature "
            "coherence distributions discriminates hallucination better than any "
            "scalar summary (K, C, C_delta). Hallucination is not a depth shift — "
            "it is a coherence CONCENTRATION.</p>"

            "<p><strong>Key results (n=200, Gemma-2-2B, TruthfulQA):</strong></p>"
            "<ul>"
            "<li>Gini coefficient of per-layer coherence: d=+0.36, p=7.6e-7, AUC=0.685 (best single metric)</li>"
            "<li>13/16 spectral shape metrics significant after Holm-Bonferroni correction</li>"
            "<li>Layer 11 is the fault epicenter in 63% of hallucinations (p=2.7e-12, t=7.45)</li>"
            "<li>23/26 layers show significantly elevated coherence in hallucinated responses (p&lt;0.05)</li>"
            "<li>Scalar depth (K): AUC=0.550, NOT significant — the mean destroys the signal</li>"
            "</ul>"

            "<p><strong>Cognitive Autopsy (first per-layer fault localization):</strong></p>"
            "<ul>"
            "<li>Layers 11+12 jointly account for 41% of all excess coherence in hallucinations</li>"
            "<li>Hallucinations concentrate coherence; correct responses distribute it</li>"
            "</ul>"

            "<p><strong>Practical interventions:</strong></p>"
            "<ul>"
            "<li>Best-of-N spectrum sampler: scores candidates by Cognitive Health Score "
            "(Gini + spread + early mass), improves on 100% of test prompts vs greedy</li>"
            "<li>Cognitive EKG: first real-time visualization of model cognition during generation</li>"
            "</ul>"

            "<p><strong>Contents:</strong> 3 analysis scripts, 3 tool implementations "
            "(spectrum steerer, spectrum sampler, cognitive EKG), 5 result JSONs, "
            "4 figures, findings document, OSF pre-registration.</p>"

            "<p>Related provisional patents: US 64/020,489 and US 64/021,113. "
            "Builds on Zenodo v7 (doi:10.5281/zenodo.19432513).</p>"
        ),
        "keywords": [
            "hallucination detection",
            "sparse autoencoder",
            "mechanistic interpretability",
            "coherence spectrum",
            "Gini coefficient",
            "cognitive autopsy",
            "layer analysis",
            "large language models",
            "TruthfulQA",
            "Fathom",
            "SAE",
            "per-layer coherence",
        ],
        "license": "cc-by-4.0",
        "publication_date": "2026-04-08",
        "related_identifiers": [
            {
                "relation": "isNewVersionOf",
                "identifier": "10.5281/zenodo.19432513",
                "resource_type": "dataset",
                "scheme": "doi",
            },
            {
                "relation": "isSupplementTo",
                "identifier": "10.5281/zenodo.19327451",
                "resource_type": "publication-preprint",
                "scheme": "doi",
            },
        ],
    }
}


def upload():
    # 1. Create deposit
    print("[1/4] Creating deposit...")
    r = requests.post(f"{BASE}/deposit/depositions", json={}, headers=HEADERS)
    if r.status_code != 201:
        print(f"  FAILED: {r.status_code} {r.text[:200]}")
        return
    dep = r.json()
    dep_id = dep["id"]
    bucket_url = dep["links"]["bucket"]
    print(f"  Deposit ID: {dep_id}")

    # 2. Upload file
    print("[2/4] Uploading zip...")
    with open(ZIP_PATH, "rb") as f:
        r = requests.put(
            f"{bucket_url}/fathom_spectrum_v8.zip",
            data=f,
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
    if r.status_code not in (200, 201):
        print(f"  FAILED: {r.status_code} {r.text[:200]}")
        return
    print(f"  Uploaded: {r.json().get('size', 0)} bytes")

    # 3. Set metadata
    print("[3/4] Setting metadata...")
    r = requests.put(
        f"{BASE}/deposit/depositions/{dep_id}",
        json=METADATA,
        headers={**HEADERS, "Content-Type": "application/json"},
    )
    if r.status_code != 200:
        print(f"  FAILED: {r.status_code} {r.text[:200]}")
        return
    print(f"  Metadata set: {r.json()['metadata']['title'][:60]}...")

    # 4. Publish
    if "--publish" in sys.argv:
        print("[4/4] Publishing...")
        r = requests.post(
            f"{BASE}/deposit/depositions/{dep_id}/actions/publish",
            headers=HEADERS,
        )
        if r.status_code != 202:
            print(f"  FAILED: {r.status_code} {r.text[:200]}")
            return
        result = r.json()
        doi = result.get("doi", "unknown")
        url = result.get("links", {}).get("record_html", "unknown")
        print(f"  PUBLISHED!")
        print(f"  DOI: {doi}")
        print(f"  URL: {url}")

        # Save record
        with open("zenodo_v8/zenodo_v8_deposit.json", "w") as f:
            json.dump({"id": dep_id, "doi": doi, "url": url}, f, indent=2)
    else:
        print("[4/4] DRAFT created (run with --publish to publish)")
        print(f"  Preview: https://zenodo.org/deposit/{dep_id}")


if __name__ == "__main__":
    upload()
