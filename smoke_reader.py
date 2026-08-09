"""Smoke test: reader policy works, and bot winrate vs reader (both temps)."""
import sys, time
sys.path.insert(0, "src")
from cote_megaverse.benchmark import run_match

for temp in (0.0, 0.3):
    for seed in (0, 1):
        t0 = time.time()
        r = run_match(seed=seed, policy="reader", depth=2, max_half_turns=60,
                      temperature=temp, ai_starts=(seed % 2 == 0))
        dt = time.time() - t0
        print(f"temp={temp} seed={seed} ai_starts={seed%2==0} "
              f"{dt:6.1f}s winner={r['winner']} plies={len(r['replay'])}")
