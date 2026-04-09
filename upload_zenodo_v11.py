# -*- coding: utf-8 -*-
"""Upload Fathom v11 (Spectrum Analysis) as new version on existing Zenodo record.
Concept: 19326174 | Latest (v10): 19464572
"""

import json
import os
import requests
import sys

TOKEN = "WkwECeWXt1SiW7cmnXyCf2Fe2NXAUn3iRjmqHNR9vXDQ3G8mEqD1Wrq5Fb0u"
BASE = "https://zenodo.org/api"
LATEST_ID = 19464572  # v10 — the last published version on real concept
ZIP_PATH = "zenodo_v8/fathom_spectrum_v8.zip"
ZIP_NAME = "fathom_spectrum_v11.zip"

HEADERS = {"Authorization": f"Bearer {TOKEN}"}

METADATA = {
    "metadata": {
        "upload_type": "dataset",
        "title": (
            "Fathom v11: Coherence Spectrum Analysis — Hallucination as "
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
            "Builds on Zenodo v10 (doi:10.5281/zenodo.19464572).</p>"
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
                "identifier": "10.5281/zenodo.19464572",
                "resource_type": "dataset",
                "scheme": "doi",
            },
        ],
    }
}


def upload():
    # 1. Create new version from latest published
    print(f"[1/5] Creating new version from v10 (ID: {LATEST_ID})...")
    r = requests.post(
        f"{BASE}/deposit/depositions/{LATEST_ID}/actions/newversion",
        headers=HEADERS,
    )
    if r.status_code != 201:
        print(f"  FAILED: {r.status_code} {r.text[:300]}")
        return
    new_dep = r.json()
    # newversion returns the CONCEPT record; latest_draft link has the new draft
    new_draft_url = new_dep["links"].get("latest_draft")
    if not new_draft_url:
        print("  FAILED: no latest_draft link in response")
        print(json.dumps(new_dep.get("links"), indent=2))
        return
    # Extract draft ID from URL
    new_id = int(new_draft_url.rstrip("/").split("/")[-1])
    print(f"  New draft ID: {new_id}")

    # 2. Delete existing files on the new draft
    print("[2/5] Clearing old files from draft...")
    r = requests.get(f"{BASE}/deposit/depositions/{new_id}", headers=HEADERS)
    if r.status_code != 200:
        print(f"  FAILED getting draft: {r.status_code}")
        return
    draft = r.json()
    for f_info in draft.get("files", []):
        fid = f_info["id"]
        fname = f_info.get("filename", fid)
        rd = requests.delete(
            f"{BASE}/deposit/depositions/{new_id}/files/{fid}",
            headers=HEADERS,
        )
        print(f"  Deleted: {fname} ({rd.status_code})")

    # 3. Upload new zip
    print(f"[3/5] Uploading {ZIP_NAME}...")
    bucket_url = draft["links"]["bucket"]
    with open(ZIP_PATH, "rb") as f:
        r = requests.put(
            f"{bucket_url}/{ZIP_NAME}",
            data=f,
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
    if r.status_code not in (200, 201):
        print(f"  FAILED: {r.status_code} {r.text[:200]}")
        return
    print(f"  Uploaded: {r.json().get('size', 0):,} bytes")

    # 4. Set metadata
    print("[4/5] Setting metadata...")
    r = requests.put(
        f"{BASE}/deposit/depositions/{new_id}",
        json=METADATA,
        headers={**HEADERS, "Content-Type": "application/json"},
    )
    if r.status_code != 200:
        print(f"  FAILED: {r.status_code} {r.text[:300]}")
        return
    print(f"  Title: {r.json()['metadata']['title'][:70]}...")

    # 5. Publish
    if "--publish" in sys.argv:
        print("[5/5] Publishing...")
        r = requests.post(
            f"{BASE}/deposit/depositions/{new_id}/actions/publish",
            headers=HEADERS,
        )
        if r.status_code != 202:
            print(f"  FAILED: {r.status_code} {r.text[:300]}")
            return
        result = r.json()
        doi = result.get("doi", "unknown")
        url = result.get("links", {}).get("record_html", "unknown")
        print(f"  PUBLISHED!")
        print(f"  DOI: {doi}")
        print(f"  URL: {url}")
        with open("zenodo_v11_deposit.json", "w") as f:
            json.dump({"id": new_id, "doi": doi, "url": url}, f, indent=2)
    else:
        print("[5/5] DRAFT ready (run with --publish to go live)")
        print(f"  Preview: https://zenodo.org/deposit/{new_id}")


if __name__ == "__main__":
    upload()
