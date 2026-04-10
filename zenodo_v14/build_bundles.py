# -*- coding: utf-8 -*-
"""
build_bundles.py -- Build the Zenodo v14 upload bundles locally.

Produces two zip files in zenodo_v14/:

  fathom_paper_v14.zip
      Contents: paper/fathom_icml2026_v2.{tex,pdf} + references.bib +
      style files + referenced figures + README.md describing the v14
      delta vs v13. This supersedes Zenodo v13 (record 19478434) as
      a new version on the existing concept (10.5281/zenodo.19326174).

  fathom_atlas_v0.3.zip
      Contents: the entire atlas/ directory (probes, 12 captures + 1
      API capture, all analysis scripts + JSONs, all figures, all
      findings memos, the sealed pre-registration, and the atlas
      README). This is a NEW Zenodo deposit with its own concept DOI,
      NOT a version bump on the paper concept.

Both zips produce a top-level directory prefix matching their name
for clean extraction.

Usage:
    python zenodo_v14/build_bundles.py

No network access. No API calls. Produces local zip files only.
After building, the upload scripts read these zips and push them to
Zenodo via the REST API using ZENODO_TOKEN from the environment.
"""

import hashlib
import json
import os
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "zenodo_v14"
OUT_DIR.mkdir(exist_ok=True)


# ── Paper v14 bundle ─────────────────────────────────────────────

PAPER_FILES = [
    ("paper/fathom_icml2026_v2.tex", "paper/fathom_icml2026_v2.tex"),
    ("paper/fathom_icml2026_v2.pdf", "paper/fathom_icml2026_v2.pdf"),
    ("paper/references.bib",         "paper/references.bib"),
    ("paper/icml2026.sty",           "paper/icml2026.sty"),
    ("paper/icml2026.bst",           "paper/icml2026.bst"),
    ("paper/algorithm.sty",          "paper/algorithm.sty"),
    ("paper/algorithmic.sty",        "paper/algorithmic.sty"),
    ("paper/fancyhdr.sty",           "paper/fancyhdr.sty"),
    ("figures/cognitive_state_map.png",        "paper/figures/cognitive_state_map.png"),
    ("figures/cognitive_autopsy_overlay.png",  "paper/figures/cognitive_autopsy_overlay.png"),
    ("figures/spectrum_n200_full.png",         "paper/figures/spectrum_n200_full.png"),
    ("figures/d_axis_cross_arch.png",          "paper/figures/d_axis_cross_arch.png"),
    ("zenodo_v14/paper/README.md",             "paper/README.md"),
]


# ── Atlas v0.3 bundle ────────────────────────────────────────────

def collect_atlas_files():
    """Yield (src_rel, arc_rel) for every file in atlas/ that should be bundled."""
    atlas_root = ROOT / "atlas"
    for root, dirs, files in os.walk(atlas_root):
        # Skip caches and worktree junk
        dirs[:] = [d for d in dirs if not d.startswith("__pycache__")
                   and not d.startswith(".")]
        for f in sorted(files):
            if f.endswith(".log") or f.endswith(".pyc"):
                continue
            src_abs = Path(root) / f
            src_rel = src_abs.relative_to(ROOT).as_posix()
            arc_rel = src_abs.relative_to(atlas_root).as_posix()
            yield (src_rel, f"atlas/{arc_rel}")


# ── Zip writer ───────────────────────────────────────────────────

def build_zip(zip_path: Path, bundle_name: str, files):
    """
    Write a zip with a top-level directory prefix.
    `files` is an iterable of (src_rel_from_repo_root, arc_rel_inside_prefix).
    """
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    file_count = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for src_rel, arc_rel in files:
            src_abs = ROOT / src_rel
            if not src_abs.exists():
                print(f"  [skip] missing: {src_rel}")
                continue
            arcname = f"{bundle_name}/{arc_rel}"
            z.write(src_abs, arcname=arcname)
            total_bytes += src_abs.stat().st_size
            file_count += 1
    return file_count, total_bytes


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 68)
    print("BUILDING ZENODO v14 + ATLAS v0.3 BUNDLES")
    print("=" * 68)
    print()

    # Paper bundle
    paper_zip = OUT_DIR / "fathom_paper_v14.zip"
    print(f"[1/2] Paper bundle -> {paper_zip.name}")
    n_paper, bytes_paper = build_zip(paper_zip, "fathom_paper_v14", PAPER_FILES)
    paper_hash = sha256(paper_zip)
    print(f"      files:   {n_paper}")
    print(f"      size:    {paper_zip.stat().st_size:>10,} bytes "
          f"(raw: {bytes_paper:,})")
    print(f"      sha256:  {paper_hash}")
    print()

    # Atlas bundle
    atlas_zip = OUT_DIR / "fathom_atlas_v0.3.zip"
    print(f"[2/2] Atlas bundle -> {atlas_zip.name}")
    atlas_files = list(collect_atlas_files())
    n_atlas, bytes_atlas = build_zip(atlas_zip, "fathom_atlas_v0_3", atlas_files)
    atlas_hash = sha256(atlas_zip)
    print(f"      files:   {n_atlas}")
    print(f"      size:    {atlas_zip.stat().st_size:>10,} bytes "
          f"(raw: {bytes_atlas:,})")
    print(f"      sha256:  {atlas_hash}")
    print()

    # Manifest
    manifest = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_commit": _git_head_commit(),
        "paper_bundle": {
            "path": str(paper_zip.relative_to(ROOT)),
            "filename": paper_zip.name,
            "n_files": n_paper,
            "bytes_on_disk": paper_zip.stat().st_size,
            "bytes_raw": bytes_paper,
            "sha256": paper_hash,
        },
        "atlas_bundle": {
            "path": str(atlas_zip.relative_to(ROOT)),
            "filename": atlas_zip.name,
            "n_files": n_atlas,
            "bytes_on_disk": atlas_zip.stat().st_size,
            "bytes_raw": bytes_atlas,
            "sha256": atlas_hash,
        },
    }
    manifest_path = OUT_DIR / "bundle_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[SAVED] {manifest_path.relative_to(ROOT)}")
    print()
    print("Ready for upload. Next steps:")
    print("  1. Inspect the bundle manifest.")
    print("  2. Set ZENODO_TOKEN (+ OSF_TOKEN if OSF is included).")
    print("  3. Run the upload scripts with the --dry-run flag first,")
    print("     then without --dry-run for the actual publish.")


def _git_head_commit() -> str:
    try:
        import subprocess
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        return r.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
