"""Verify: bad switches are never sampled by the mixed bot; display fix check."""
import sys, random
from collections import Counter
sys.path.insert(0, "src")
from cote_megaverse.agent import Planner
from cote_megaverse.rules import Character, GameState, Side, Type
from cote_megaverse.strategy import switch_value
from cote_megaverse.interactive import show_resolution

def ch(t, hp, atk):
    return Character(t, hp, atk, hp)

# Bot (player side): Coordinator C#1 active vs user's Defender D#1 (3800).
bot = Side((ch(Type.C, 4700, 1900), ch(Type.D, 5900, 1900),
            ch(Type.D, 6300, 1900)), active=0, stack_order=(0, 1, 2),
           actions=7, bonus=0)
user = Side((ch(Type.D, 3800, 2000), ch(Type.D, 6000, 2000),
             ch(Type.A, 6100, 1900)), active=0, stack_order=(0, 1, 2))
state = GameState(bot, user, turn=7, player_to_move=True)

# Sanity: switch_value for both targets should be negative (dominated).
p0 = Planner(depth=1, temperature=0)
p0.choose(state)
for idx in (1, 2):
    sv = switch_value(state, idx, p0.belief(state))
    print(f"switch to #{idx} (D, {state.player.characters[idx].hp}hp): "
          f"value={sv.value} recommended={sv.recommended}")

# Sample the mixed bot many times; a bad switch must never appear.
c = Counter()
bad_switches = 0
for i in range(120):
    p = Planner(depth=1, temperature=0.3, rng=random.Random(1000 + i))
    m = p.choose(state)
    c[m.label] += 1
    if m.switch:
        sv = switch_value(state, m.switch_to, p.belief(state))
        if sv.value < 0:
            bad_switches += 1
            print("BAD SWITCH SAMPLED:", m.label, "value", sv.value)
print(f"\n120 samples -> {len(c)} distinct moves")
for label, n in c.most_common(8):
    print(f"  {label:14s} {n}")
print(f"bad switches sampled: {bad_switches}  "
      f"{'OK' if bad_switches == 0 else 'FAIL'}")

# Display fix: a switch move reports the switched-to attacker's damage.
from cote_megaverse.rules import Allocation, apply
from contextlib import redirect_stdout
import io
state2 = GameState(Side((ch(Type.C, 4700, 1900), ch(Type.D, 6300, 1900)),
                        active=0, stack_order=(0, 1), actions=3),
                   Side((ch(Type.D, 8000, 2000),), active=0, stack_order=(0,)),
                   turn=7, player_to_move=True)
move = Allocation(2, 0, 0, 1)   # switch to #1 (D#3), 2 attacks = 3 actions
buf = io.StringIO()
with redirect_stdout(buf):
    show_resolution(state2, move, apply(state2, move), "AI")
text = buf.getvalue()
print("\n" + text)
expect = "3800"  # D#3 x1.0 -> 2*1900 = 3800 (not C#1's 2*2500 = 5000)
print(f"display shows {expect} dmg (switched attacker): "
      f"{'OK' if expect in text else 'CHECK'}")

