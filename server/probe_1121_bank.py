"""Probe the POST-BANK state of seed-1121: bot has bank 2 -> budget 6, knows the
wall (4,0)=1.0. Does it attack a6 (1+ land -> kill 900 HP) or re-bank/shield?
"""
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "src"))

import _cote_cfr  # noqa: E402
from cote_megaverse.rules import (GameState, Side, Character, Type,  # noqa: E402
                                  base_budget)
from cote_megaverse.cfr_bot import _team_of, _encode_state  # noqa: E402


def Ch(t, hp, atk):
    return Character(type=t, hp=hp, atk=atk, max_hp=hp)


def main():
    # bot: bank 2 carried -> budget base(20)+2 = 6. A:4200 active, C:5700 bench.
    bot = Side((Ch(Type.A, 0, 1900), Ch(Type.A, 4200, 1900),
                Ch(Type.C, 5700, 1900)),
               active=1, stack_order=(1, 2), bonus=2, shields=0,
               actions=base_budget(20) + 2)
    opp = Side((Ch(Type.C, 0, 1900), Ch(Type.C, 0, 2000),
                Ch(Type.B, 900, 2100)),
               active=2, stack_order=(2,), bonus=0, shields=4,
               actions=base_budget(20))
    state = GameState(player=bot, opponent=opp, turn=20, player_to_move=True)
    state = state.prepare()
    enc = _encode_state(state, opp_bank=0, opp_sh=4)
    team_a, team_b = _team_of(bot), _team_of(opp)

    mt = _cote_cfr.MicroTree(team_a, team_b, [(enc, 1.0)], depth=3,
                             cap=6, start_turn=20, compress=True)
    mt.solve(80, 0.995)
    acts, probs, value = mt.strategy()
    print(f"budget=6 value(bot)={value:+.4f}")
    for (a, d, b, s), p in sorted(zip(acts, probs), key=lambda x: -x[1]):
        if p < 0.005:
            continue
        print(f"  a{a} d{d} b{b} sw={s if s >= 0 else '-'}  p={p:.3f}")


if __name__ == "__main__":
    main()
