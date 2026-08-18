"""Replay seed-1121 and dump, at every bot turn from t16, the probability of a4
(and the sampled move). Answers: does the strategy converge to ~100% a4, or is
the bot sampling ~0.7 a4 every turn (which would make 20 consecutive a4 ~
impossible)?
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
    bot_attrs = []
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
            if before.turn >= 16:
                pa4 = 0.0
                for act, p in cfr.last_report["root_actions"]:
                    if act[0] == 4 and act[1] == 0 and act[2] == 0:
                        pa4 = p
                mem = cfr.model.memory_prior(4)
                bot_attrs.append((before.turn, move.attacks, move.defends,
                                  move.bonuses, pa4,
                                  round(mem.get((4, 0), 0.0), 3)))
        else:
            before = state
            planning = GameState(state.opponent, state.player, state.turn, True)
            move = pl.choose(planning)
            state = apply(state, move)
            pl.observe_shields(before.player.shields)
            cfr.observe(move.attacks, move.bonuses, move.switch,
                        budget=before.opponent.actions, turn=before.turn)
    print("turn | played | P(a4) | mem(4,0)")
    for t, a, d, b, pa4, mem40 in bot_attrs:
        print(f"  t{t:3d} | a{a} d{d} b{b} | {pa4:.3f} | {mem40:.3f}")


if __name__ == "__main__":
    main()
