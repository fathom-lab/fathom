#!/usr/bin/env python3
"""
FATHOM arXiv STATUS MONITOR
============================
Polls arXiv to detect when the paper goes live.

Usage:
    python check_arxiv_status.py                          # Check once
    python check_arxiv_status.py --poll                   # Poll every 5 minutes
    python check_arxiv_status.py --poll --interval 120    # Poll every 2 minutes
    python check_arxiv_status.py --arxiv-id 2503.XXXXX   # Check specific ID

How it works:
    1. Searches arXiv for author "darkflobi"
    2. Also checks the new submissions RSS feed
    3. If --arxiv-id is given, directly checks that URL
    4. When found: prints alert, writes PAPER_LIVE.md, exits

Run this in background after submitting to arXiv.
"""

import argparse
import re
import sys
import time
import json
from datetime import datetime
from pathlib import Path

try:
    import urllib.request
    import urllib.error
    HAS_URLLIB = True
except ImportError:
    HAS_URLLIB = False

REPO_ROOT = Path(__file__).parent.resolve()


def check_by_id(arxiv_id: str) -> dict:
    """Check if a specific arXiv ID is live."""
    url = f"https://arxiv.org/abs/{arxiv_id}"
    try:
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Fathom-Monitor/1.0')
        resp = urllib.request.urlopen(req, timeout=15)
        if resp.getcode() == 200:
            content = resp.read().decode('utf-8', errors='replace')
            # Extract title
            title_match = re.search(r'<title>(.*?)</title>', content, re.DOTALL)
            title = title_match.group(1).strip() if title_match else "Unknown"
            return {
                'found': True,
                'method': 'direct_id',
                'arxiv_id': arxiv_id,
                'url': url,
                'title': title,
            }
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {'found': False, 'method': 'direct_id', 'error': 'Not found (404)'}
        return {'found': False, 'method': 'direct_id', 'error': f'HTTP {e.code}'}
    except Exception as e:
        return {'found': False, 'method': 'direct_id', 'error': str(e)}
    return {'found': False, 'method': 'direct_id', 'error': 'Unknown'}


def check_by_author_search(author: str = "darkflobi") -> dict:
    """Search arXiv for papers by author."""
    url = f"https://arxiv.org/search/?query={author}&searchtype=author"
    try:
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Fathom-Monitor/1.0')
        resp = urllib.request.urlopen(req, timeout=20)
        if resp.getcode() == 200:
            content = resp.read().decode('utf-8', errors='replace')
            # Check for results
            if 'No results' in content or 'Sorry, your query' in content:
                return {'found': False, 'method': 'author_search', 'error': 'No results'}

            # Look for paper links
            paper_ids = re.findall(r'arxiv\.org/abs/(\d{4}\.\d{4,5})', content)
            if paper_ids:
                return {
                    'found': True,
                    'method': 'author_search',
                    'arxiv_ids': list(set(paper_ids)),
                    'url': url,
                }
            return {'found': False, 'method': 'author_search', 'error': 'No paper IDs found in results'}
    except Exception as e:
        return {'found': False, 'method': 'author_search', 'error': str(e)}
    return {'found': False, 'method': 'author_search', 'error': 'Unknown'}


def check_by_title_search() -> dict:
    """Search arXiv for the paper by title keywords."""
    query = "Computational+Graph+Structure+Surface+Recall+Explanatory+Reasoning"
    url = f"https://arxiv.org/search/?query={query}&searchtype=all"
    try:
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Fathom-Monitor/1.0')
        resp = urllib.request.urlopen(req, timeout=20)
        if resp.getcode() == 200:
            content = resp.read().decode('utf-8', errors='replace')
            paper_ids = re.findall(r'arxiv\.org/abs/(\d{4}\.\d{4,5})', content)
            if paper_ids:
                return {
                    'found': True,
                    'method': 'title_search',
                    'arxiv_ids': list(set(paper_ids)),
                    'url': url,
                }
            return {'found': False, 'method': 'title_search', 'error': 'No results'}
    except Exception as e:
        return {'found': False, 'method': 'title_search', 'error': str(e)}
    return {'found': False, 'method': 'title_search', 'error': 'Unknown'}


def on_paper_found(result: dict):
    """Actions when paper is detected live."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    arxiv_id = result.get('arxiv_id') or (result.get('arxiv_ids', [None])[0])
    arxiv_url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "Unknown"

    print(f"""
======================================================================
  PAPER IS LIVE!
======================================================================
  arXiv ID:  {arxiv_id}
  URL:       {arxiv_url}
  Detected:  {now}
  Method:    {result.get('method', 'unknown')}
======================================================================

  NEXT STEP:
    python launch_fathom.py --arxiv-id {arxiv_id}

======================================================================
""")

    # Write marker file
    marker = REPO_ROOT / "PAPER_LIVE.md"
    marker.write_text(
        f"# PAPER IS LIVE\n\n"
        f"**arXiv ID:** {arxiv_id}\n"
        f"**URL:** {arxiv_url}\n"
        f"**Detected:** {now}\n"
        f"**Method:** {result.get('method', 'unknown')}\n\n"
        f"## Next Step\n\n"
        f"```bash\n"
        f"python launch_fathom.py --arxiv-id {arxiv_id}\n"
        f"```\n",
        encoding='utf-8'
    )
    print(f"  Written: PAPER_LIVE.md")

    # Try to make noise (Windows)
    try:
        import winsound
        for _ in range(3):
            winsound.Beep(1000, 300)
            time.sleep(0.2)
    except Exception:
        # Print bell character as fallback
        print("\a\a\a")


def run_check(arxiv_id: str = None, verbose: bool = True) -> bool:
    """Run all checks. Returns True if paper found."""
    now = datetime.now().strftime("%H:%M:%S")

    if not HAS_URLLIB:
        print("ERROR: urllib not available.")
        return False

    results = []

    # Check specific ID if given
    if arxiv_id:
        if verbose:
            print(f"  [{now}] Checking arXiv ID: {arxiv_id}...", end=" ", flush=True)
        result = check_by_id(arxiv_id)
        results.append(result)
        if verbose:
            print("FOUND!" if result['found'] else f"not yet ({result.get('error', '')})")
        if result['found']:
            on_paper_found(result)
            return True

    # Author search
    if verbose:
        print(f"  [{now}] Searching by author: darkflobi...", end=" ", flush=True)
    result = check_by_author_search()
    results.append(result)
    if verbose:
        if result['found']:
            print(f"FOUND! IDs: {result.get('arxiv_ids', [])}")
        else:
            print(f"not yet ({result.get('error', '')})")
    if result['found']:
        on_paper_found(result)
        return True

    # Title search
    if verbose:
        print(f"  [{now}] Searching by title...", end=" ", flush=True)
    result = check_by_title_search()
    results.append(result)
    if verbose:
        if result['found']:
            print(f"FOUND! IDs: {result.get('arxiv_ids', [])}")
        else:
            print(f"not yet ({result.get('error', '')})")
    if result['found']:
        on_paper_found(result)
        return True

    return False


def main():
    parser = argparse.ArgumentParser(
        description='Monitor arXiv for Fathom paper going live.'
    )
    parser.add_argument(
        '--arxiv-id',
        help='Specific arXiv ID to check (e.g., 2503.12345)'
    )
    parser.add_argument(
        '--poll', action='store_true',
        help='Keep polling until paper is found'
    )
    parser.add_argument(
        '--interval', type=int, default=300,
        help='Polling interval in seconds (default: 300 = 5 minutes)'
    )
    args = parser.parse_args()

    print(f"""
======================================================================
  FATHOM arXiv STATUS MONITOR
======================================================================
  Mode:     {"POLLING (every {0}s)".format(args.interval) if args.poll else "SINGLE CHECK"}
  Target:   {args.arxiv_id or "author search: darkflobi"}
  Started:  {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
======================================================================
""")

    if not args.poll:
        # Single check
        found = run_check(arxiv_id=args.arxiv_id)
        if not found:
            print(f"\n  Paper not found yet.")
            print(f"  Run with --poll to keep checking.")
        sys.exit(0 if found else 1)
    else:
        # Polling mode
        check_num = 0
        print(f"  Polling every {args.interval} seconds. Press Ctrl+C to stop.\n")
        try:
            while True:
                check_num += 1
                print(f"  --- Check #{check_num} ---")
                found = run_check(arxiv_id=args.arxiv_id)
                if found:
                    print(f"\n  Paper found after {check_num} checks!")
                    sys.exit(0)
                print(f"  Next check in {args.interval}s...\n")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print(f"\n\n  Stopped after {check_num} checks.")
            sys.exit(1)


if __name__ == '__main__':
    main()
