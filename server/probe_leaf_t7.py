"""Compare the static leaf value of the t7 1v2 state (bot C(6100) vs PL
C(5800)+A(5800), PL to move) — which the bot's depth-3 tree uses as the leaf
for the PL's attack line — against deeper dynamic solves. If the static leaf
understates the PL's advantage, the depth-1 opponent in the bot's tree will
undervalue attacking (a3) and prefer banking.
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
    # bot (side 0): C(6100) active, only body; bank 0, shields 0.
    bot = Side((Ch(Type.B, 0, 2100), Ch(Type.C, 6100, 1900),
                Ch(Type.D, 0, 1900)),
               active=1, stack_order=(1,), bonus=0, shields=0,
               actions=base_budget(7))
    # PL (side 1): C(5800) active + A(5800) bench; bank 0, shields 0.
    pl = Side((Ch(Type.B, 0, 2100), Ch(Type.C, 5800, 1900),
               Ch(Type.A, 5800, 2100)),
              active=1, stack_order=(1, 2), bonus=0, shields=0,
              actions=base_budget(7))
    state = GameState(player=bot, opponent=pl, turn=7, player_to_move=False)
    state = state.prepare()
    enc = _encode_state(state, opp_bank=0, opp_sh=0)
    return _team_of(bot), _team_of(pl), enc


def main():
    team_a, team_b, enc = build()
    for depth in (0, 3, 4):
        mt = _cote_cfr.MicroTree(team_a, team_b, [(enc, 1.0)], depth=depth,
                                 cap=6, start_turn=7, compress=True)
        mt.solve(0 if depth == 0 else 80, 0.995)
        acts, probs, value = mt.strategy()
        print(f"depth={depth} nodes={mt.node_count()} "
              f"value(side0/bot)={value:+.4f}  "
              f"top: " + ", ".join(
                  f"a{a}d{d}b{b}sw={s if s >= 0 else '-'}:{p:.2f}"
                  for (a, d, b, s), p in sorted(zip(acts, probs),
                                                key=lambda x: -x[1])[:4]))


if __name__ == "__main__":
    main()
