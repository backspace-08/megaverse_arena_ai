"""Re-run the 11 draw seeds through cfr_vs_planner.run_match (now with sampled
rng) and report winners - does sampling let the bot take bank->burst and win?
"""
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "src"))

from cfr_vs_planner import run_match  # noqa: E402

DRAW = [(1108, False), (1113, True), (1113, False), (1121, False),
        (1123, False), (1129, False), (1130, True), (1132, False),
        (1140, True), (1143, True), (1147, True)]


def main():
    w = {"LEFT": 0, "RIGHT": 0, "DRAW": 0}
    for seed, nf in DRAW:
        r = run_match(seed, nf, 3, 80, 6, 2, 2000)
        w[r["winner"]] += 1
        print(f"seed={seed} new_first={nf} -> {r['winner']:5s} "
              f"half_turns={r['half_turns']}", flush=True)
    print("total:", w)


if __name__ == "__main__":
    main()
