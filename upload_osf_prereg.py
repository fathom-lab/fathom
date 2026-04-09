# -*- coding: utf-8 -*-
"""Upload spectrum pre-registration to OSF project."""

import json
import requests

OSF_TOKEN = "ZQXV6Lt3CsfRCFuWLdG6sNBbqLIDZsbSRYeqCLHShMwlNYSB7SjXV66FmunhiHWBQ2Dppo"
PROJECT_ID = "wtkzg"
BASE = "https://api.osf.io/v2"

HEADERS = {
    "Authorization": f"Bearer {OSF_TOKEN}",
    "Content-Type": "application/json",
}

PREREG_FILE = "prereg/osf_prereg_spectrum.md"


def upload_file_to_project():
    """Upload the pre-registration file to the OSF project's file storage."""
    print("[1/3] Checking project access...")

    # Verify project access
    r = requests.get(f"{BASE}/nodes/{PROJECT_ID}/", headers=HEADERS)
    if r.status_code != 200:
        print(f"  FAILED: {r.status_code} {r.text[:200]}")
        return
    project = r.json()["data"]
    print(f"  Project: {project['attributes']['title']}")

    # Get upload URL (OSF uses Waterbutler for file uploads)
    print("[2/3] Uploading pre-registration file...")
    upload_url = f"https://files.osf.io/v1/resources/{PROJECT_ID}/providers/osfstorage/"

    with open(PREREG_FILE, "rb") as f:
        content = f.read()

    r = requests.put(
        upload_url,
        params={"kind": "file", "name": "osf_prereg_spectrum_v8.md"},
        data=content,
        headers={
            "Authorization": f"Bearer {OSF_TOKEN}",
            "Content-Type": "application/octet-stream",
        },
    )

    if r.status_code in (200, 201):
        print(f"  Uploaded: osf_prereg_spectrum_v8.md")
        file_data = r.json()
        print(f"  URL: {file_data.get('data', {}).get('links', {}).get('download', 'check project')}")
    elif r.status_code == 409:
        # File exists, try update
        print("  File exists, updating...")
        r = requests.put(
            upload_url,
            params={"kind": "file", "name": "osf_prereg_spectrum_v8.md", "conflict": "replace"},
            data=content,
            headers={
                "Authorization": f"Bearer {OSF_TOKEN}",
                "Content-Type": "application/octet-stream",
            },
        )
        if r.status_code in (200, 201):
            print(f"  Updated successfully")
        else:
            print(f"  Update failed: {r.status_code} {r.text[:200]}")
    else:
        print(f"  FAILED: {r.status_code} {r.text[:300]}")

    # Also upload the findings doc and key results
    print("[3/3] Uploading findings + results...")
    for local_path, remote_name in [
        ("findings/findings_2026_04_07_SPECTRUM.md", "findings_spectrum_v8.md"),
        ("truthfulqa_results/spectrum_full_analysis.json", "spectrum_full_analysis.json"),
    ]:
        with open(local_path, "rb") as f:
            data = f.read()
        r = requests.put(
            upload_url,
            params={"kind": "file", "name": remote_name, "conflict": "replace"},
            data=data,
            headers={
                "Authorization": f"Bearer {OSF_TOKEN}",
                "Content-Type": "application/octet-stream",
            },
        )
        status = "OK" if r.status_code in (200, 201) else f"FAILED ({r.status_code})"
        print(f"  {remote_name}: {status}")

    print(f"\n  Project URL: https://osf.io/{PROJECT_ID}/")
    print("  Done.")


if __name__ == "__main__":
    upload_file_to_project()
