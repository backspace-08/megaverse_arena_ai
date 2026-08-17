"""Step: 3v3 telemetry probe - exact CPU breakdown.

Worst case: 3x3 roster, E=8, active healthy. depth=4, iters=40, prune_after=10,
compressed action grid, telemetry=true.

Reports per-depth node visits, leaf evals, and ns in action-gen / leaf-eval /
regret (accumulated over all 40 iterations).
"""
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

import _cote_cfr as c
c.load_1v1_table(os.path.join(REPO, "cote_cfr", "1v1_table.csv"))

TEAM = [(0, 2000, 3000), (1, 2000, 3000), (2, 2000, 3000)]


def encode(hp_units, bank, turn, to_move):
    n = len(hp_units)
    return ([n] + list(range(n)) + hp_units + [bank, 0]
            + [n] + list(range(n)) + hp_units + [bank, 0]
            + [turn, to_move])


def main():
    state = encode([3000, 3000, 3000], 4, 7, 0)
    roots = [(state, 1.0)]
    mt = c.MicroTree(TEAM, TEAM, roots, depth=4, cap=20, start_turn=7,
                     compress=True)
    t0 = time.time()
    mt.solve(40, 0.995, [], 10, telemetry=True)
    total_ms = (time.time() - t0) * 1000
    (d1, d2, d3, d4), n_leaves, ns_ag, ns_le, ns_rg = mt.stats()
    print("[3v3 telemetry] depth=4 iters=40 RBP10 compressed  total=%.0fms" % total_ms,
          flush=True)
    print("  node visits: D1=%d D2=%d D3=%d D4=%d" % (d1, d2, d3, d4), flush=True)
    print("  total nodes (approx) = %d" % (1 + d1 + d2 + d3 + d4), flush=True)
    print("  leaf evals (x40 iters) = %d  (~%d/it)" % (n_leaves, n_leaves // 40), flush=True)
    ag_ms = ns_ag / 1e6
    le_ms = ns_le / 1e6
    rg_ms = ns_rg / 1e6
    other = total_ms - (ag_ms + le_ms + rg_ms)
    tot = ag_ms + le_ms + rg_ms + other
    print("[time breakdown] action-gen=%7.0fms (%4.1f%%)  leaf-eval=%7.0fms (%4.1f%%)  "
          "regret=%7.0fms (%4.1f%%)  other=%7.0fms (%4.1f%%)"
          % (ag_ms, 100 * ag_ms / tot, le_ms, 100 * le_ms / tot,
             rg_ms, 100 * rg_ms / tot, other, 100 * other / tot), flush=True)
    print("[allocations] full_key clones + actions vecs: counts above (see depth "
          "visits ~ allocations per iter)", flush=True)


if __name__ == "__main__":
    main()
