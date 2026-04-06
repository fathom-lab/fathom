# -*- coding: utf-8 -*-
"""Upload fathom_monitor_disclosure.pdf to Zenodo as a new deposit."""

import json
import requests

TOKEN = "WkwECeWXt1SiW7cmnXyCf2Fe2NXAUn3iRjmqHNR9vXDQ3G8mEqD1Wrq5Fb0u"
BASE  = "https://zenodo.org/api"
PDF   = "fathom_monitor_disclosure.pdf"

HEADERS = {"Authorization": f"Bearer {TOKEN}"}

METADATA = {
    "metadata": {
        "upload_type": "publication",
        "publication_type": "technicalnote",
        "title": (
            "Fathom Monitor: Per-Token Hallucination Detection via "
            "Coherence Divergence in Sparse Autoencoder Feature Space"
        ),
        "creators": [{"name": "Rodabaugh, Alexander", "affiliation": "Independent"}],
        "description": (
            "<p>This technical disclosure describes <strong>Fathom Monitor</strong>, "
            "a system and method for detecting hallucination-risk tokens in large "
            "language model (LLM) outputs at the time of generation, using a "
            "mechanistic signal derived from the geometric structure of sparse "
            "autoencoder (SAE) feature activations.</p>"
            "<p>The core innovation is the use of <strong>C_delta</strong> — the "
            "divergence between late-layer and early-layer feature coherence — as a "
            "per-token hallucination indicator. When C_delta exceeds a calibrated "
            "threshold at a given token position, that token is flagged as uncertain "
            "or high-risk and annotated inline.</p>"
            "<p>Empirical validation on TruthfulQA (n=50, Gemma-2-2B): "
            "C_delta discriminates hallucination with p=0.040, Cohen's d=0.407. "
            "Depth (K) is blind to hallucination (p=0.931).</p>"
            "<p>This document constitutes a public technical disclosure establishing "
            "prior art. Related provisional patents: US 64/020,489 (March 29, 2026) "
            "and US 64/021,113 (March 30, 2026). Builds on Zenodo records "
            "doi:10.5281/zenodo.19326175 and doi:10.5281/zenodo.19364702.</p>"
        ),
        "keywords": [
            "hallucination detection",
            "sparse autoencoder",
            "mechanistic interpretability",
            "coherence",
            "large language models",
            "TruthfulQA",
            "per-token uncertainty",
            "C_delta",
            "Fathom",
        ],
        "license": "cc-by-4.0",
        "publication_date": "2026-04-02",
        "related_identifiers": [
            {
                "relation": "isSupplementTo",
                "identifier": "10.5281/zenodo.19364702",
                "resource_type": "publication-preprint",
                "scheme": "doi",
            },
            {
                "relation": "isSupplementTo",
                "identifier": "10.5281/zenodo.19326175",
                "resource_type": "publication-preprint",
                "scheme": "doi",
            },
        ],
        "notes": (
            "Patent pending: US 64/020,489 and US 64/021,113. "
            "This disclosure establishes prior art for the Fathom Monitor system "
            "as of April 2, 2026 under 35 U.S.C. § 102 (AIA). "
            "Provisional patent application to be filed within 12 months."
        ),
    }
}


def main():
    session = requests.Session()
    session.headers.update(HEADERS)

    # Step 1: Create empty deposit
    print("[1/4] Creating Zenodo deposit...")
    r = session.post(f"{BASE}/deposit/depositions", json={})
    r.raise_for_status()
    dep = r.json()
    dep_id   = dep["id"]
    bucket   = dep["links"]["bucket"]
    html_url = dep["links"]["html"]
    print(f"     Deposit ID: {dep_id}")
    print(f"     Draft URL:  {html_url}")

    # Step 2: Upload PDF via bucket API
    print(f"[2/4] Uploading {PDF}...")
    with open(PDF, "rb") as fh:
        r = session.put(f"{bucket}/{PDF}", data=fh,
                        headers={"Content-Type": "application/octet-stream"})
    r.raise_for_status()
    print(f"     Upload OK  ({r.json().get('size', '?')} bytes)")

    # Step 3: Set metadata
    print("[3/4] Setting metadata...")
    r = session.put(f"{BASE}/deposit/depositions/{dep_id}", json=METADATA)
    if r.status_code >= 400:
        print(f"     Metadata error {r.status_code}: {r.text[:500]}")
        r.raise_for_status()
    print(f"     Metadata OK")

    # Step 4: Publish
    print("[4/4] Publishing...")
    r = session.post(f"{BASE}/deposit/depositions/{dep_id}/actions/publish")
    if r.status_code >= 400:
        print(f"     Publish error {r.status_code}: {r.text[:500]}")
        r.raise_for_status()

    pub = r.json()
    doi    = pub.get("doi", "")
    doi_url = pub.get("doi_url", "")
    rec_url = pub.get("links", {}).get("record_html", "")

    print("\n" + "="*60)
    print("ZENODO DEPOSIT PUBLISHED")
    print("="*60)
    print(f"  DOI:    {doi}")
    print(f"  DOI URL: {doi_url}")
    print(f"  Record: {rec_url}")
    print("="*60)

    # Save result
    result = {"deposit_id": dep_id, "doi": doi, "doi_url": doi_url, "record_url": rec_url}
    with open("zenodo_monitor_deposit.json", "w") as f:
        json.dump(result, f, indent=2)
    print("Result saved to zenodo_monitor_deposit.json")


if __name__ == "__main__":
    main()
