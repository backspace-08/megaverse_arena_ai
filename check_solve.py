"""Correctness validation of the 1v1 solver on trivial positions."""
import sys
sys.path.insert(0, "src")
from cote_megaverse.solve1v1 import Solver


def solve_from(start, iters=15, cap=14, stall=3):
    s = Solver(cap=cap, stall_cap=stall, start=start)
    s.run(iters, silent=True)
    info = s.info_of[0]
    return s, s.value.get(info, 0.0), len(s.states)


cases = [
    ("A hp=1 vs B hp=1  (A must attack)", (1, 1), +1.0),
    ("A hp=1 vs B hp=60 (A cannot kill)", (1, 60), -1.0),
    ("A hp=1 vs B hp=2  (A kills in one)", (1, 2), +1.0),
    ("A hp=3 vs B hp=3  (A kills in one)", (3, 3), +1.0),
]
for label, (ha, hb), expect in cases:
    start = (ha, hb, 0, 0, 0, 0, 0, 0, 1, 0)
    s, v, states = solve_from(start)
    ok = abs(v - expect) < 0.05
    print(f"{label:40s} V={v:+.3f} expect {expect:+.1f} "
          f"states={states:,}  {'OK' if ok else 'FAIL'}")

print("\nmain position (60,60) — 25 iters, then exploitability:")
s = Solver(cap=30, stall_cap=3)
s.run(25, silent=True)
start_info = s.info_of[0]
v_cur = s.value.get(start_info, 0.0)
brA = s.best_response_value(0)
brB = s.best_response_value(1)
v_avg = (brA - brB) / 2
print(f"  states={len(s.states):,} infosets={len(set(s.info_of)):,}")
print(f"  current-strategy value V(A)={v_cur:+.4f}")
print(f"  best response A={brA:+.4f}  B={brB:+.4f}")
print(f"  exploitability={(brA + brB)/2:+.4f}")
print(f"  average-strategy value (midpoint)={v_avg:+.4f}  win_rate={ (v_avg+1)/2:.4f}")
