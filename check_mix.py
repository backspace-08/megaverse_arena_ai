"""Confirm softmax mixing on a balanced position (opening, 1 action)."""
import sys, random
from collections import Counter
sys.path.insert(0, "src")
from cote_megaverse.agent import Planner
from cote_megaverse.rules import Type, initial

base = initial((Type.A, Type.B, Type.C), (Type.A, Type.C, Type.D), rng=random.Random(5))
print("turn", base.turn, "actions", base.player.actions)

for temp in (0.0, 0.2, 0.3, 0.5):
    c = Counter()
    for i in range(60):
        p = Planner(depth=1, temperature=temp, rng=random.Random(100 + i))
        m = p.choose(base)
        c[m.label] += 1
    print(f"temp={temp:5.2f}: {len(c)} moves in 60 runs  {dict(c)}")

# show raw scores to see the gap
p = Planner(depth=1, temperature=0.0)
p.choose(base)
for label, score, comps in p.last_report["alternatives"]:
    print(f"  {label:10s} score={score:8.1f}")
