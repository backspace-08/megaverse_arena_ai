"""Smoke test: verify run_match works and time a few matches (depth 2)."""
import sys, time
sys.path.insert(0, "src")
from cote_megaverse.benchmark import run_match

times = []
for seed in (0, 1, 2):
    for pol in ("random", "greedy"):
        t0 = time.time()
        r = run_match(seed=seed, policy=pol, depth=2, max_half_turns=60,
                      ai_starts=(seed % 2 == 0))
        dt = time.time() - t0
        times.append((seed, pol, dt, r["winner"]))
        print(f"seed={seed} {pol:12s} ai_starts={seed%2==0} "
              f"{dt:5.2f}s winner={r['winner']} plies={len(r['replay'])}")
avg = sum(x[2] for x in times) / len(times)
print(f"\navg per match: {avg:.2f}s  ({len(times)} matches)")
