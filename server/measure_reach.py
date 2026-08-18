"""Diagnostic: measure how often the CFR bot's root belief uses the opponent
reach vs falls back to the uniform infoset prior. Runs matches in-process (no
multiprocessing) so the module-level _BELIEF_DIAG counter aggregates.
"""
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "src"))

from cote_megaverse import cfr_bot  # noqa: E402
from cfr_vs_planner import run_match  # noqa: E402


def main():
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    cfr_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    cfr_iters = int(sys.argv[3]) if len(sys.argv) > 3 else 80
    wins = {"LEFT": 0, "RIGHT": 0, "DRAW": 0}
    first = second = 0
    for i in range(games):
        seed = 1100 + i
        new_first = (i % 2 == 0)
        r = run_match(seed, new_first, cfr_depth, cfr_iters, 6, 2, 2000)
        wins[r["winner"]] += 1
        if r["winner"] == "LEFT":
            if new_first:
                first += 1
            else:
                second += 1
    n = sum(wins.values())
    print(f"games={n}  CFR={wins['LEFT']} Planner={wins['RIGHT']} "
          f"DRAW={wins['DRAW']}  CFR-winrate={100.0*wins['LEFT']/n:.1f}%")
    print(f"  CFR-first={first}/{games//2}  CFR-second={second}/{games//2}")
    print("belief-root diagnostic:")
    total = sum(cfr_bot._BELIEF_DIAG.values())
    d = cfr_bot._BELIEF_DIAG
    for k in ("reach_used", "reach_kept_empty", "reach_total0", "reach_missing",
              "fallback_uniform"):
        v = d.get(k, 0)
        print(f"  {k:18s}: {v:6d}  ({100.0*v/total:5.1f}%)" if total else f"  {k}: {v}")
    print(f"  total choose->_belief_roots calls: {total}")


if __name__ == "__main__":
    main()
