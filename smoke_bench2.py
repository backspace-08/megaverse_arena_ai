"""Smoke + time benchmark_policies on a small seed set (both temps)."""
import sys, time
sys.path.insert(0, "src")
from cote_megaverse.benchmark import benchmark_policies

for temp in (0.0, 0.3):
    t0 = time.time()
    r = benchmark_policies(seeds=range(2), depth=2, max_half_turns=60,
                           temperature=temp)
    dt = time.time() - t0
    n = sum(v["games"] for v in r.values())
    print(f"temp={temp}: {n} matches in {dt:.0f}s ({dt/n:.1f}s/match)")
    for pol, v in r.items():
        print(f"   {pol:13s} W{v['wins']} L{v['losses']} D{v['draws']} "
              f"missed_lethal={v['missed_guaranteed_lethal']}")
