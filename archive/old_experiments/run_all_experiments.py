"""
Run all Fathom experiments in sequence.
Chains: observer effect → sycophancy → expertise framing
"""
import subprocess, sys
from datetime import datetime

experiments = [
    ("Observer Effect", "fathom_observer_effect.py"),
    ("Sycophancy Detector", "fathom_sycophancy.py"),
    ("Expertise Framing", "fathom_expertise_framing.py"),
]

print(f"\n{'='*60}")
print(f"  FATHOM EXPERIMENT SUITE — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"  Running {len(experiments)} experiments in sequence")
print(f"{'='*60}\n")

for i, (name, script) in enumerate(experiments):
    print(f"\n[{i+1}/{len(experiments)}] Starting: {name}")
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print(f"  [WARN] {name} exited with code {result.returncode}")
    print(f"  [DONE] {name}")

print("\nAll experiments complete.")
try:
    subprocess.run(["clawdbot", "gateway", "wake",
                    "--text", "All 3 Fathom experiments complete — observer effect, sycophancy, expertise framing. Results saved. Check sae-reasoning-depth/ for JSON files.",
                    "--mode", "now"], timeout=10)
except: pass
