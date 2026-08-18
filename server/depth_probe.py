"""Minimal depth probe: ONE seed, few iters, streaming output + per-match wall
clock. Does deeper resolve break the type-disadvantage deadlock?
"""
import os
import sys
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "src"))

from cfr_vs_planner import run_match  # noqa: E402


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1113
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    depths = [int(x) for x in sys.argv[3:]] or [3, 4, 5, 6]
    for depth in depths:
        t0 = time.time()
        r = run_match(seed, new_first=False, cfr_depth=depth, cfr_iters=iters,
                      cfr_cap=6, pl_depth=2, pl_max_nodes=2000)
        wall = time.time() - t0
        print(f"depth={depth} winner={r['winner']:5s} "
              f"half_turns={r['half_turns']:3d} avg_lat={r['lat_ms']:.0f}ms "
              f"max_lat={r['max_lat_ms']:.0f}ms wall={wall:.1f}s",
              flush=True)


if __name__ == "__main__":
    main()
