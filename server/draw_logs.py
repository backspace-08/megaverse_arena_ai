"""Rich full logs for the known draw seeds (from the 100-game run), 8 workers.

Writes per-turn logs with rosters, active characters (type/hp/atk), budgets,
multiplier/per-hit for attacks, belief and reach for each draw to a file.
"""
import os
import sys
from multiprocessing import Pool

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "src"))
sys.path.insert(0, os.path.join(BASE, "server"))

from trace_draws import play  # noqa: E402

# (seed, seat) pairs that ended in DRAW in the 100-game benchmark run.
DRAW_TASKS = [
    (1108, False), (1113, True), (1113, False), (1121, False),
    (1123, False), (1129, False), (1130, True), (1132, False),
    (1140, True), (1143, True), (1147, True),
]


def worker(task):
    seed, new_first = task
    winner, log, ra, rb = play(seed, new_first)
    lines = []
    if winner != "DRAW":
        lines.append(f"NOTE: seed={seed} new_first={new_first} -> {winner} "
                     f"(was a draw in the benchmark run; table/env differs)")
    lines.append("A(CFR): " + ra)
    lines.append("B(PL):  " + rb)
    for e in log:
        m = f" mult={e['mult']} ph={e['per_hit']}" if e["mult"] else ""
        bel_s = ""
        if e["bel"]:
            top = sorted(e["bel"], key=lambda x: -x[2])[:3]
            bel_s = " bel:" + ",".join(f"({s},{k})={p:.2f}" for s, k, p in top)
        reach_s = ""
        if e["reach"]:
            top = sorted(e["reach"].items(), key=lambda x: -x[1])[:3]
            reach_s = " reach:" + ",".join(f"({s},{b})={p:.2f}"
                                           for (s, b), p in top)
        lines.append(f"  t{e['turn']:2d} {e['label']} a{e['a']} d{e['d']} "
                     f"b{e['b']} sw={e['sw']} bgt={e['budget']}"
                     f"  A-act:{e['actA']} B-act:{e['actB']}{m}"
                     f"  A={e['hpA']} B={e['hpB']}{bel_s}{reach_s}")
    return {"seed": seed, "new_first": new_first, "winner": winner,
            "lines": lines}


def main():
    with Pool(8) as pool:
        results = list(pool.imap_unordered(worker, DRAW_TASKS))
    results.sort(key=lambda r: (r["seed"], not r["new_first"]))
    out = os.path.join(BASE, "runs", "draw_logs.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for r in results:
            fh.write(f"==== seed={r['seed']} new_first={r['new_first']} "
                     f"-> {r['winner']} ====\n")
            fh.write("\n".join(r["lines"]) + "\n\n")
    print(f"logged {len(results)} draw games -> {out}")
    print("draws:", [(r["seed"], r["new_first"]) for r in results])


if __name__ == "__main__":
    main()
