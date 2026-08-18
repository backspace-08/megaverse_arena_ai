"""Probe the seed-1108 t4 resolve: does the solver's depth-1 opponent exploit
the bot's full-budget public spend (0 shields) by attacking and killing D, and
how does the bot value its own lines?
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
    # Bot A (player): #0 B dead, #1 C 6100/1900 active, #2 D 5700/1900.
    # Bank carried = 2 (banked on t2) -> prepare gives budget base(4)+2 = 4.
    bot = Side((Ch(Type.B, 0, 2100), Ch(Type.C, 6100, 1900),
                Ch(Type.D, 5700, 1900)),
               active=1, stack_order=(1, 2), bonus=2, shields=0,
               actions=base_budget(4) + 2)
    opp = Side((Ch(Type.B, 5800, 2100), Ch(Type.C, 5800, 1900),
                Ch(Type.A, 5800, 2100)),
               active=0, stack_order=(0, 1, 2), bonus=0, shields=0,
               actions=base_budget(4))
    state = GameState(player=bot, opponent=opp, turn=4, player_to_move=True)
    state = state.prepare()

    cfr = CFRBot(depth=3, iters=80, cap=6, compress=True)
    # feed the public opponent history so the belief is (0,0)=1.0:
    cfr.observe(1, 0, False, budget=1, turn=1)   # PL a1, R=0
    cfr.observe(2, 0, False, budget=2, turn=3)   # PL a2, R=0
    print("worlds:", [(w.shields, w.bank, round(w.probability, 3))
                      for w in cfr.model.worlds()])

    move = cfr.choose(state)
    print("chosen: a%d d%d b%d sw=%s value=%.4f"
          % (move.attacks, move.defends, move.bonuses, move.switch_to,
             cfr.last_report["value"]))
    print("root strategy:")
    for act, p in sorted(cfr.last_report["root_actions"], key=lambda x: -x[1]):
        a, d, b, sw = act
        print(f"  a{a} d{d} b{b} sw={sw if sw >= 0 else '-'}  p={p:.3f}")
    print("opponent reach (depth-1 splits -> does the solver's opponent "
          "attack, split (0,0), or defend?):")
    for (s, b), p in sorted(cfr._reach.items(), key=lambda x: -x[1]):
        print(f"  (sh={s},bk={b}) p={p:.3f}")


if __name__ == "__main__":
    main()
