#!/usr/bin/env python3
"""
FATHOM LAUNCH SCRIPT
====================
One command fires the entire launch sequence the moment we have an arXiv ID.

Usage:
    python launch_fathom.py --arxiv-id 2503.XXXXX
    python launch_fathom.py --arxiv-id 2503.XXXXX --dry-run

What it does:
    1. Verifies the arXiv ID exists (HTTP 200)
    2. Replaces all placeholders across the repo
    3. Generates LAUNCH_READY.md with final announcement materials
    4. Prints launch checklist
"""

import argparse
import os
import re
import sys
import json
from pathlib import Path
from datetime import datetime

try:
    import urllib.request
    import urllib.error
    HAS_URLLIB = True
except ImportError:
    HAS_URLLIB = False


# =============================================================================
# CONFIGURATION
# =============================================================================

REPO_ROOT = Path(__file__).parent.resolve()

# Placeholder patterns to replace with the real arXiv URL
PLACEHOLDER_PATTERNS = [
    r'\[arxiv link\]',
    r'\[arXiv link\]',
    r'\[arxiv_link\]',
    r'https://arxiv\.org/abs/XXXX\.XXXXX',
    r'arXiv:XXXX\.XXXXX',
    r'XXXX\.XXXXX',  # bare placeholder in bibtex
]

# Files to scan (relative to repo root). Excludes worktrees and __pycache__
SCAN_EXTENSIONS = {'.md', '.html', '.tex', '.json', '.yaml', '.yml'}

SKIP_DIRS = {
    '.git', '__pycache__', '.claude', 'node_modules',
    'figures/.claude',  # worktrees
}

# Files that CANNOT be edited (just log a warning)
UNEDITABLE = {'paper_fathom_final.pdf', 'paper_claude_v2.pdf'}

# Files to never modify (self, monitor script)
SKIP_FILES = {'launch_fathom.py', 'check_arxiv_status.py'}


def should_skip(path: Path) -> bool:
    """Check if a path should be skipped during scanning."""
    parts = path.relative_to(REPO_ROOT).parts
    for skip in SKIP_DIRS:
        skip_parts = skip.split('/')
        for i in range(len(parts) - len(skip_parts) + 1):
            if parts[i:i+len(skip_parts)] == tuple(skip_parts):
                return True
    return False


def find_target_files() -> list:
    """Find all files that should be scanned for placeholders."""
    targets = []
    for root, dirs, files in os.walk(REPO_ROOT):
        root_path = Path(root)
        if should_skip(root_path):
            dirs.clear()
            continue
        # Prune skip dirs from walk
        dirs[:] = [d for d in dirs if not should_skip(root_path / d)]
        for f in files:
            fp = root_path / f
            if fp.suffix in SCAN_EXTENSIONS and f not in SKIP_FILES:
                targets.append(fp)
    return sorted(targets)


def verify_arxiv_id(arxiv_id: str) -> bool:
    """Check that the arXiv ID resolves to a real paper."""
    url = f"https://arxiv.org/abs/{arxiv_id}"
    print(f"\n  Verifying: {url}")

    if not HAS_URLLIB:
        print("  WARNING: urllib not available. Skipping verification.")
        return True

    try:
        req = urllib.request.Request(url, method='HEAD')
        req.add_header('User-Agent', 'Fathom-Launch/1.0')
        resp = urllib.request.urlopen(req, timeout=15)
        status = resp.getcode()
        if status == 200:
            print(f"  OK: HTTP {status}")
            return True
        else:
            print(f"  WARNING: HTTP {status}")
            return False
    except urllib.error.HTTPError as e:
        print(f"  FAILED: HTTP {e.code}")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def replace_placeholders(arxiv_id: str, dry_run: bool = False) -> dict:
    """
    Scan all repo files and replace arXiv placeholders.
    Returns dict of {filepath: [(old, new, line_num), ...]}
    """
    arxiv_url = f"https://arxiv.org/abs/{arxiv_id}"
    arxiv_ref = f"arXiv:{arxiv_id}"

    # Build replacement map: pattern -> replacement
    replacements_map = {
        r'\[arxiv link\]': arxiv_url,
        r'\[arXiv link\]': arxiv_url,
        r'\[arxiv_link\]': arxiv_url,
        r'https://arxiv\.org/abs/XXXX\.XXXXX': arxiv_url,
        r'arXiv:XXXX\.XXXXX': arxiv_ref,
    }

    # Special: bare XXXX.XXXXX only in specific contexts (bibtex journal lines)
    bare_pattern = re.compile(r'(arXiv preprint arXiv:)XXXX\.XXXXX')
    bare_replacement = f'\\g<1>{arxiv_id}'

    targets = find_target_files()
    all_changes = {}
    uneditable_warnings = []

    for fp in targets:
        # Check if this is an uneditable file
        if fp.name in UNEDITABLE:
            uneditable_warnings.append(fp)
            continue

        try:
            content = fp.read_text(encoding='utf-8')
        except (UnicodeDecodeError, PermissionError):
            continue

        original = content
        changes = []

        for pattern, replacement in replacements_map.items():
            compiled = re.compile(pattern, re.IGNORECASE)
            for match in compiled.finditer(content):
                # Find line number
                line_num = content[:match.start()].count('\n') + 1
                changes.append((match.group(), replacement, line_num))
            content = compiled.sub(replacement, content)

        # Handle bare XXXX.XXXXX in bibtex context
        for match in bare_pattern.finditer(content):
            line_num = content[:match.start()].count('\n') + 1
            changes.append((match.group(), match.group().replace('XXXX.XXXXX', arxiv_id), line_num))
        content = bare_pattern.sub(bare_replacement, content)

        # Also replace "arXiv link pending" / "arXiv link coming" notes
        pending_pattern = re.compile(r'\*?\(arXiv link (?:pending|coming)\)\*?', re.IGNORECASE)
        for match in pending_pattern.finditer(content):
            line_num = content[:match.start()].count('\n') + 1
            changes.append((match.group(), '', line_num))
        content = pending_pattern.sub('', content)

        # Replace "[arXiv link coming]" standalone
        coming_pattern = re.compile(r'\[arXiv link coming\]', re.IGNORECASE)
        for match in coming_pattern.finditer(content):
            line_num = content[:match.start()].count('\n') + 1
            changes.append((match.group(), arxiv_url, line_num))
        content = coming_pattern.sub(arxiv_url, content)

        # Also update ANNOUNCEMENT status line
        status_pattern = re.compile(r'\*\*Status:\*\* READY \(pending arXiv ID\)')
        for match in status_pattern.finditer(content):
            line_num = content[:match.start()].count('\n') + 1
            changes.append((match.group(), f'**Status:** LIVE (arXiv: {arxiv_id})', line_num))
        content = status_pattern.sub(f'**Status:** LIVE (arXiv: {arxiv_id})', content)

        replace_pattern = re.compile(r'\*\*Replace:\*\* \[arxiv link\] with actual arXiv ID when assigned\n?')
        for match in replace_pattern.finditer(content):
            line_num = content[:match.start()].count('\n') + 1
            changes.append((match.group().strip(), '[removed - ID assigned]', line_num))
        content = replace_pattern.sub('', content)

        if changes and content != original:
            rel = fp.relative_to(REPO_ROOT)
            all_changes[str(rel)] = changes
            if not dry_run:
                fp.write_text(content, encoding='utf-8')

    return all_changes, uneditable_warnings


def generate_launch_ready(arxiv_id: str, verified: bool, changes: dict, dry_run: bool = False):
    """Generate LAUNCH_READY.md with final announcement materials."""
    arxiv_url = f"https://arxiv.org/abs/{arxiv_id}"
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

    # Read the announcement thread if it exists
    thread_path = REPO_ROOT / "ANNOUNCEMENT_THREAD.md"
    if thread_path.exists():
        thread_content = thread_path.read_text(encoding='utf-8')
    else:
        thread_content = "[ANNOUNCEMENT_THREAD.md not found]"

    content = f"""# FATHOM LAUNCH READY

**arXiv ID:** {arxiv_id}
**arXiv URL:** {arxiv_url}
**Verified:** {"YES" if verified else "NO - CHECK MANUALLY"}
**Generated:** {timestamp}
**Files Updated:** {len(changes)}

---

## 1. TWEET THREAD (10 tweets)

{thread_content}

---

## 2. DEPLOY CHECKLIST

### fathom.html (Landing Page)
- [ ] Upload fathom.html to Netlify
- [ ] Set custom domain: darkflobi.com/fathom
- [ ] Verify all links work
- [ ] Test on mobile

### GitHub Repository
- [ ] Make sae-reasoning-depth repo PUBLIC
- [ ] Rename to "fathom" if desired
- [ ] Verify README renders correctly
- [ ] Check CITATION.md has correct arXiv ID

### API (Optional for launch)
- [ ] Deploy to RunPod/Railway if ready
- [ ] Or keep as "coming soon" with waitlist

---

## 3. OUTREACH LIST

### Priority 1 - Day of Launch
- **Brian Roemmele** (@BrianRoemmele) - DM with paper link + one-line summary
- **Neel Nanda** - Mechanistic interpretability lead at Anthropic. Tweet @ or DM.
- **Chris Olah** - Interpretability pioneer. Tweet the thread and tag.

### Priority 2 - Within 24 Hours
- **APOLLO Research** - Working on similar SAE problems. Email: info@apolloresearch.ai
- **EleutherAI Discord** - Post in #research channel
- **r/MachineLearning** - Post paper with discussion prompt
- **Alignment Forum / LessWrong** - Post detailed writeup

### Priority 3 - Within 1 Week
- **NeurIPS 2026 submission** - If timing works
- **Hoagy** (endorser) - Thank publicly, share results
- **AI safety community** - MATS, Redwood, ARC

---

## 4. KEY LINKS

| Resource | URL |
|----------|-----|
| Paper | {arxiv_url} |
| Landing Page | https://darkflobi.com/fathom |
| GitHub | https://github.com/darkflobi/fathom |
| Waitlist | https://darkflobi.com/fathom/waitlist |
| API Docs | https://darkflobi.com/fathom/api |
| Leaderboard | https://darkflobi.com/fathom/leaderboard |

---

## 5. FIRST TWEET (copy-paste ready)

```
we built an instrument that measures how deeply AI models think. not what they say. how they compute it. sparse autoencoder circuit tracing across every layer. the first cognitive MRI for language models. paper: {arxiv_url} framework: darkflobi.com/fathom
```

---

*Generated by launch_fathom.py at {timestamp}*
"""

    if not dry_run:
        out_path = REPO_ROOT / "LAUNCH_READY.md"
        out_path.write_text(content, encoding='utf-8')
        print(f"\n  Written: LAUNCH_READY.md")
    else:
        print(f"\n  [DRY RUN] Would write: LAUNCH_READY.md ({len(content)} bytes)")

    return content


def print_checklist(arxiv_id: str, verified: bool, changes: dict, uneditable: list):
    """Print the final launch checklist."""
    v = "YES" if verified else "FAILED"
    c = "YES" if changes else "NO FILES CHANGED"

    print(f"""
======================================================================
  FATHOM LAUNCH CHECKLIST
======================================================================

  [{"x" if verified else " "}] arXiv ID verified: {arxiv_id} -- {v}
  [{"x" if changes else " "}] Placeholders updated: {len(changes)} files -- {c}
  [x] LAUNCH_READY.md generated

  MANUAL STEPS:
  [ ] Push sae-reasoning-depth to GitHub (make public first)
  [ ] Deploy fathom.html to darkflobi.com/fathom via Netlify
  [ ] Post tweet thread from @darkflobi
  [ ] DM Brian Roemmele
  [ ] Email APOLLO researchers
  [ ] Post to r/MachineLearning""")

    if uneditable:
        print(f"\n  WARNINGS:")
        for fp in uneditable:
            print(f"  [!] Cannot auto-edit: {fp.name} -- update arXiv link manually")

    if not verified:
        print(f"\n  *** arXiv ID not verified. Paper may not be live yet. ***")
        print(f"  *** Run check_arxiv_status.py to monitor. ***")

    print(f"""
======================================================================
  READY TO LAUNCH.
======================================================================
""")


def main():
    parser = argparse.ArgumentParser(
        description='Fathom Launch Script - One command fires everything.',
        epilog='Example: python launch_fathom.py --arxiv-id 2503.12345'
    )
    parser.add_argument(
        '--arxiv-id', required=True,
        help='The arXiv paper ID (e.g., 2503.12345)'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Show what would change without making any modifications'
    )
    parser.add_argument(
        '--skip-verify', action='store_true',
        help='Skip arXiv URL verification (useful if paper is not yet live)'
    )
    args = parser.parse_args()

    arxiv_id = args.arxiv_id.strip()
    dry_run = args.dry_run

    # Validate format
    if not re.match(r'^\d{4}\.\d{4,5}(v\d+)?$', arxiv_id):
        print(f"ERROR: '{arxiv_id}' doesn't look like a valid arXiv ID.")
        print(f"Expected format: YYMM.NNNNN (e.g., 2503.12345)")
        sys.exit(1)

    mode = "DRY RUN" if dry_run else "LIVE"
    print(f"""
======================================================================
  FATHOM LAUNCH SEQUENCE -- {mode}
======================================================================
  arXiv ID: {arxiv_id}
  URL:      https://arxiv.org/abs/{arxiv_id}
  Time:     {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
======================================================================
""")

    # Step 1: Verify arXiv ID
    print("STEP 1: Verify arXiv ID")
    if args.skip_verify:
        print("  Skipped (--skip-verify)")
        verified = False
    else:
        verified = verify_arxiv_id(arxiv_id)
        if not verified:
            print("\n  WARNING: arXiv ID not found. Paper may not be live yet.")
            print("  Continuing anyway (placeholders will be updated).")
            print("  Use --skip-verify to suppress this check.")

    # Step 2: Replace placeholders
    print("\nSTEP 2: Replace placeholders")
    changes, uneditable = replace_placeholders(arxiv_id, dry_run=dry_run)
    if changes:
        for filepath, file_changes in sorted(changes.items()):
            action = "Would update" if dry_run else "Updated"
            print(f"  {action}: {filepath} ({len(file_changes)} replacements)")
            for old, new, line in file_changes[:3]:  # Show first 3
                old_short = old[:60] + '...' if len(old) > 60 else old
                new_short = new[:60] + '...' if len(new) > 60 else new
                print(f"    L{line}: '{old_short}' -> '{new_short}'")
            if len(file_changes) > 3:
                print(f"    ... and {len(file_changes) - 3} more")
    else:
        print("  No placeholders found (already updated?)")

    for fp in uneditable:
        print(f"  WARNING: {fp.name} needs manual arXiv link update (binary file)")

    # Step 3: Generate LAUNCH_READY.md
    print("\nSTEP 3: Generate LAUNCH_READY.md")
    generate_launch_ready(arxiv_id, verified, changes, dry_run=dry_run)

    # Step 4: Print checklist
    print_checklist(arxiv_id, verified, changes, uneditable)

    if dry_run:
        print("  This was a DRY RUN. No files were modified.")
        print(f"  Remove --dry-run to execute for real.\n")


if __name__ == '__main__':
    main()
