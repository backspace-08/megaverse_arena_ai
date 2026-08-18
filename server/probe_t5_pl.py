"""Solve the seed-1108 t5 position FROM THE PLANNER'S OWN PERSPECTIVE: does the
opponent (with its public info about the bot's R=2 / banked) prefer to attack
(a3, kill exposed D) or bank (b3)? This isolates whether "banking" is the
solver's genuine (approximate) best response or a depth-1 artifact.
"""
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "src"))

from cote_megaverse.rules import (GameState, Side, Character, Type,  # noqa: E402
                                  base_budget)
from cote_megaverse.cfr_bot import CFRBot  # noqa: E402


def Ch(t, hp, atk):
    return Character(type=t, hp=hp, atk=atk, max_hp=hp)


def main():
    # PL side (acting player at t5): B(3900) active, C+A bench, bank 0.
    pl = Side((Ch(Type.B, 3900, 2100), Ch(Type.C, 5800, 1900),
               Ch(Type.A, 5800, 2100)),
              active=0, stack_order=(0, 1, 2), bonus=0, shields=0,
              actions=base_budget(5))
    # bot side: #0 B dead, #1 C(6100), #2 D(5700) ACTIVE (post-switch), bank 2.
    bot = Side((Ch(Type.B, 0, 2100), Ch(Type.C, 6100, 1900),
                Ch(Type.D, 5700, 1900)),
               active=2, stack_order=(2, 1), bonus=2, shields=0,
               actions=base_budget(5))
    state = GameState(player=pl, opponent=bot, turn=5, player_to_move=True)
    state = state.prepare()

    cfr = CFRBot(depth=3, iters=80, cap=6, compress=True)
    # feed the bot's PUBLIC moves: t2 b2 (R=0), t4 a1 sw (R=2, revealed bank 2)
    cfr.observe(0, 0, False, budget=2, turn=2)
    cfr.observe(1, 0, True, budget=4, turn=4)
    print("PL's belief over bot's split:",
          [(w.shields, w.bank, round(w.probability, 3))
           for w in cfr.model.worlds()])

    move = cfr.choose(state)
    print("PL chosen: a%d d%d b%d sw=%s value=%.4f"
          % (move.attacks, move.defends, move.bonuses, move.switch_to,
             cfr.last_report["value"]))
    print("PL root strategy:")
    for act, p in sorted(cfr.last_report["root_actions"], key=lambda x: -x[1]):
        a, d, b, sw = act
        print(f"  a{a} d{d} b{b} sw={sw if sw >= 0 else '-'}  p={p:.3f}")


if __name__ == "__main__":
    main()
