"""Probe seed-1121 t18 deadlock: bot 2v1 (A:4200/1900 active + C:5700/1900
bench) vs opponent B:900/2100 holding 4 shields. The bot knows (4,0)=1.0.
One landed hit kills 900 HP. Why does the solve prefer a4 (blocked) over
b4 -> a8 (bank then burst, guaranteed kill)?
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


def build():
    bot = Side((Ch(Type.A, 0, 1900), Ch(Type.A, 4200, 1900),
                Ch(Type.C, 5700, 1900)),
               active=1, stack_order=(1, 2), bonus=0, shields=0,
               actions=base_budget(18))
    opp = Side((Ch(Type.C, 0, 1900), Ch(Type.C, 0, 2000),
                Ch(Type.B, 900, 2100)),
               active=2, stack_order=(2,), bonus=0, shields=4,
               actions=base_budget(18))
    state = GameState(player=bot, opponent=opp, turn=18, player_to_move=True)
    state = state.prepare()
    enc = _encode_state(state, opp_bank=0, opp_sh=4)
    return _team_of(bot), _team_of(opp), enc


def main():
    team_a, team_b, enc = build()
    for iters in (80,):
        mt = _cote_cfr.MicroTree(team_a, team_b, [(enc, 1.0)], depth=3,
                                 cap=6, start_turn=18, compress=True)
        mt.solve(iters, 0.995)
        acts, probs, value = mt.strategy()
        print(f"iters={iters} nodes={mt.node_count()} value(bot)={value:+.4f}")
        print("root strategy (all p>=0.005):")
        for (a, d, b, s), p in sorted(zip(acts, probs), key=lambda x: -x[1]):
            if p < 0.005:
                continue
            print(f"  a{a} d{d} b{b} sw={s if s >= 0 else '-'}  p={p:.3f}")
        # specifically find the pure-bank line a0 d0 b4 (burst prep)
        for (a, d, b, s), p in zip(acts, probs):
            if a == 0 and d == 0 and b == 4:
                print(f"  >>> a0 d0 b4 (pure bank) p={p:.3f}")


if __name__ == "__main__":
    main()
