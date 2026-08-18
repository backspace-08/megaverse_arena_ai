"""Replay seed-1121 to the bot's t18 decision and dump the REAL root belief
(reach x memory blend from _belief_roots) plus the solve's strategy - to see
why the live bot plays a4 instead of the clean-probe passive strategy.
"""
import os
import random
import sys
from dataclasses import replace

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "src"))

from cote_megaverse.rules import (GameState, Type, apply, initial)  # noqa: E402
from cote_megaverse.agent import Planner  # noqa: E402
from cote_megaverse.cfr_bot import CFRBot  # noqa: E402


def main():
    seed, new_first = 1121, False
    rng = random.Random(seed)
    types = list(Type)
    state = initial(tuple(rng.choice(types) for _ in range(3)),
                    tuple(rng.choice(types) for _ in range(3)), rng=rng)
    if not new_first:
        state = replace(state, player_to_move=False).prepare()
    cfr = CFRBot(depth=3, iters=80, cap=6, compress=True)
    pl = Planner(depth=2, max_nodes=2000)
    for _ in range(80):
        if state.player.lost or state.opponent.lost:
            break
        if state.player_to_move:
            before = state
            planning = GameState(state.player, state.opponent, state.turn, True)
            move = cfr.choose(planning)
            state = apply(state, move)
            cfr.observe_shields(before.opponent.shields)
            pl.observe(move.attacks, move.bonuses, move.switch,
                       budget=before.player.actions, turn=before.turn)
            if before.turn == 18:
                print(f"t{18} CFR a{move.attacks} d{move.defends} b{move.bonuses} "
                      f"sw={move.switch_to}")
                print("worlds (memory):",
                      [(w.shields, w.bank, round(w.probability, 3))
                       for w in cfr.model.worlds()])
                print("reach (prev solve):",
                      {k: round(v, 3) for k, v in
                       sorted(cfr._reach.items(), key=lambda x: -x[1])[:6]}
                      if cfr._reach else None)
                w = cfr.model.worlds()
                r_opp = w[0].shields + w[0].bank
                roots = cfr._belief_roots(planning, r_opp, w)
                print("REAL root belief (reach x memory, sum=%d):" % r_opp)
                tot = sum(v for _, v in roots)
                for (enc, v), _ in [(r, r) for r in roots]:
                    pass
                # decode: roots are (encoded_state, weight); we don't have the
                # split directly, so print via the model's memory_prior instead
                mem = cfr.model.memory_prior(r_opp)
                print("  memory_prior:", {k: round(v, 3)
                                          for k, v in mem.items()})
                acts = cfr.last_report["root_actions"]
                print("  value=%.4f  strategy:" % cfr.last_report["value"])
                for act, p in sorted(acts, key=lambda x: -x[1])[:8]:
                    a, d, b, s = act
                    print(f"    a{a} d{d} b{b} sw={s if s >= 0 else '-'} p={p:.3f}")
        else:
            before = state
            planning = GameState(state.opponent, state.player, state.turn, True)
            move = pl.choose(planning)
            state = apply(state, move)
            pl.observe_shields(before.player.shields)
            cfr.observe(move.attacks, move.bonuses, move.switch,
                        budget=before.opponent.actions, turn=before.turn)
    print("final:", "CFR" if state.opponent.lost else
          ("PL" if state.player.lost else "DRAW"))


if __name__ == "__main__":
    main()
