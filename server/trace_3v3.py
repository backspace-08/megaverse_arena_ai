"""3v3 trace with type-countering lineups to force switch decisions.

A: active type 0, bench types [2, 2]  (2 beats 1)
B: active type 1, bench types [3, 3]  (1 beats 0)
Matchup 0 vs 1: A's active is at 0.7x disadvantage; A's bench type 2 beats B's
active (1.3x) -> switching is clearly advantageous for A.
"""
import os
import random
import sys
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "src"))

from cote_megaverse.rules import GameState, Type, apply, initial  # noqa: E402
from cote_megaverse.agent import Planner  # noqa: E402
from cote_megaverse.cfr_bot import CFRBot  # noqa: E402
from cote_megaverse.chain_leaf import ChainLeaf  # noqa: E402

TYPE_NAMES = {0: "A", 1: "B", 2: "C", 3: "D"}


def _sw_tag(move):
    st = move.switch_to
    if st is None:
        return "   "
    v = st.value if hasattr(st, "value") else st
    return "SW->%s" % TYPE_NAMES.get(v, "?")


def run(seed, a_types, b_types, label, max_half=36):
    rng = random.Random(seed)
    state = initial(tuple(Type(t) for t in a_types),
                    tuple(Type(t) for t in b_types), rng=rng)
    cfr = CFRBot(depth=3, iters=80, cap=8, value_leaf=ChainLeaf(),
                 compress=True, rng=random.Random(seed))
    pl = Planner(depth=2, max_nodes=2000)
    sw_a = sw_b = 0
    print("=== %s | A types=%s B types=%s ===" % (label, a_types, b_types),
          flush=True)
    for _ in range(max_half):
        if state.player.lost or state.opponent.lost:
            break
        act_t = state.player.characters[state.player.active].type.value
        oact_t = state.opponent.characters[state.opponent.active].type.value
        if state.player_to_move:
            before = state
            planning = GameState(state.player, state.opponent, state.turn, True)
            t0 = time.time()
            move = cfr.choose(planning)
            dt = (time.time() - t0) * 1000
            if move.switch_to is not None:
                sw_a += 1
            tag = _sw_tag(move)
            print("t%2d A[%s]%s a%d d%d b%d  A=%s B=%s  %.0fms" % (
                state.turn, TYPE_NAMES.get(act_t), tag, move.attacks,
                move.defends, move.bonuses,
                [c.hp for c in state.player.characters],
                [c.hp for c in state.opponent.characters], dt), flush=True)
            state = apply(state, move)
            cfr.observe_shields(before.opponent.shields)
            pl.observe(move.attacks, move.bonuses, move.switch,
                       budget=before.player.actions, turn=before.turn)
        else:
            before = state
            planning = GameState(state.opponent, state.player, state.turn, True)
            move = pl.choose(planning)
            if move.switch_to is not None:
                sw_b += 1
            tag = _sw_tag(move)
            print("t%2d B[%s]%s a%d d%d b%d  A=%s B=%s" % (
                state.turn, TYPE_NAMES.get(oact_t), tag, move.attacks,
                move.defends, move.bonuses,
                [c.hp for c in state.player.characters],
                [c.hp for c in state.opponent.characters]), flush=True)
            state = apply(state, move)
            pl.observe_shields(before.player.shields)
            cfr.observe(move.attacks, move.bonuses, move.switch,
                        budget=before.opponent.actions, turn=before.turn)
    winner = "CFR(A)" if state.opponent.lost else "PL(B)" if state.player.lost else "DRAW"
    print("  [end] winner=%s  A-switches=%d B-switches=%d" % (winner, sw_a, sw_b),
          flush=True)


if __name__ == "__main__":
    run(1100, [0, 2, 2], [1, 3, 3], "A weak vs B, counter on bench")
    run(1101, [1, 2, 2], [0, 3, 3], "B weak vs A, counter on bench")
    run(1102, [0, 1, 2], [3, 0, 1], "mixed cycle")

