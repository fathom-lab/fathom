"""
healthcheck.py — verify Fathom API is running correctly
=========================================================
Runs a full integration test against the live API.

Usage:
    python healthcheck.py [--url http://localhost:8500]
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.error


def check(url, name, method="GET", data=None, expect_keys=None, timeout=120):
    """Make a request and verify response."""
    print(f"  [{name}]... ", end="", flush=True)
    try:
        headers = {"User-Agent": "FathomHealthcheck/1.0"}
        if data:
            headers["Content-Type"] = "application/json"
            req = urllib.request.Request(
                url, data=json.dumps(data).encode(),
                headers=headers,
                method=method,
            )
        else:
            req = urllib.request.Request(url, headers=headers)
        resp = urllib.request.urlopen(req, timeout=timeout)
        body = json.loads(resp.read().decode())

        if expect_keys:
            missing = [k for k in expect_keys if k not in body]
            if missing:
                print(f"FAIL (missing keys: {missing})")
                return False

        print(f"OK ({resp.status})")
        return body
    except urllib.error.HTTPError as e:
        print(f"FAIL (HTTP {e.code}: {e.read().decode()[:100]})")
        return False
    except Exception as e:
        print(f"FAIL ({e})")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8500")
    parser.add_argument("--wait", type=int, default=0,
                        help="Wait N seconds for model to load before testing")
    args = parser.parse_args()

    base = args.url.rstrip("/")
    print(f"Fathom API healthcheck — {base}")
    print()

    if args.wait:
        print(f"  Waiting {args.wait}s for model to load...")
        time.sleep(args.wait)

    passed = 0
    total = 0

    # Test 1: Root endpoint
    total += 1
    r = check(f"{base}/", "root", expect_keys=["name", "version"])
    if r:
        passed += 1
        print(f"    version: {r.get('version')}")

    # Test 2: Health
    total += 1
    r = check(f"{base}/health", "health", expect_keys=["status", "model_loaded"])
    if r:
        passed += 1
        print(f"    status: {r['status']}, model: {r.get('model')}")
        if not r.get("model_loaded"):
            print(f"    WARNING: model not loaded yet. Wait and retry.")
            if r.get("model_loading"):
                print(f"    Model is loading... waiting 60s")
                time.sleep(60)
                r = check(f"{base}/health", "health retry",
                          expect_keys=["model_loaded"])
                if r and r.get("model_loaded"):
                    print(f"    Model loaded!")
                else:
                    print(f"    Model still not loaded. Check GPU.")

    # Test 3: Scan — simple question
    total += 1
    r = check(
        f"{base}/scan", "scan (simple)",
        method="POST",
        data={"prompt": "What is the capital of France?", "max_tokens": 10},
        expect_keys=["generated_text", "s_early", "zone", "tokens"],
        timeout=120,
    )
    if r:
        passed += 1
        zone = r.get("zone", {})
        print(f"    text: {r['generated_text'][:60]!r}")
        print(f"    S_early: {r['s_early']:.5f}")
        print(f"    zone: {zone.get('zone')} ({zone.get('halluc_probability',0)*100:.0f}% halluc)")
        print(f"    tokens: {r['total_tokens']}")
        print(f"    time: {r['generation_time_s']:.1f}s")

    # Test 4: Scan — hallucination-prone question
    total += 1
    r = check(
        f"{base}/scan", "scan (adversarial)",
        method="POST",
        data={"prompt": "What did Paul Revere say during his midnight ride?",
              "max_tokens": 20},
        expect_keys=["s_early", "zone"],
        timeout=120,
    )
    if r:
        passed += 1
        zone = r.get("zone", {})
        print(f"    text: {r['generated_text'][:60]!r}")
        print(f"    zone: {zone.get('zone')}")

    # Test 5: Demo page
    total += 1
    try:
        req = urllib.request.Request(f"{base}/demo",
            headers={"User-Agent": "FathomHealthcheck/1.0"})
        resp = urllib.request.urlopen(req, timeout=10)
        content = resp.read().decode()
        if "fathom scan" in content.lower():
            print(f"  [demo page]... OK (HTML served)")
            passed += 1
        else:
            print(f"  [demo page]... FAIL (content doesn't look right)")
    except Exception as e:
        print(f"  [demo page]... FAIL ({e})")

    # Test 6: Validation — empty prompt
    total += 1
    try:
        req = urllib.request.Request(
            f"{base}/scan",
            data=json.dumps({"prompt": "", "max_tokens": 10}).encode(),
            headers={"Content-Type": "application/json",
                     "User-Agent": "FathomHealthcheck/1.0"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
        print(f"  [validation]... FAIL (should reject empty prompt)")
    except urllib.error.HTTPError as e:
        if e.code == 422:
            print(f"  [validation]... OK (rejects empty prompt with 422)")
            passed += 1
        else:
            print(f"  [validation]... FAIL (wrong error code: {e.code})")
    except Exception as e:
        print(f"  [validation]... FAIL ({e})")

    print()
    print(f"Results: {passed}/{total} passed")
    if passed == total:
        print("ALL CHECKS PASSED")
    else:
        print(f"{total - passed} CHECKS FAILED")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
