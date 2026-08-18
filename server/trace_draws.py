"""Trace draw games: play the same seeds as cfr_vs_planner, and for any game
that ends in DRAW print the full move sequence (a/d/b/sw per half-turn) so we
can see whether the bot/planner is turtling (shields/bank, few attacks).
"""
import os
import random
import sys
from dataclasses import replace

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "src"))

from cote_megaverse.rules import GameState, Type, apply, initial  # noqa: E402
from cote_megaverse.agent import Planner  # noqa: E402
from cote_megaverse.cfr_bot import CFRBot  # noqa: E402

MAX_TURNS = 80


def play(seed, new_first, cfr_depth=3, cfr_iters=80, cfr_cap=6,
         pl_depth=2, pl_max_nodes=2000):
    rng = random.Random(seed)
    types = list(Type)
    state = initial(tuple(rng.choice(types) for _ in range(3)),
                    tuple(rng.choice(types) for _ in range(3)), rng=rng)
    if not new_first:
        state = replace(state, player_to_move=False).prepare()
    cfr = CFRBot(depth=cfr_depth, iters=cfr_iters, cap=cfr_cap,
                 compress=True)
    pl = Planner(depth=pl_depth, max_nodes=pl_max_nodes)
    log = []
    for _ in range(MAX_TURNS):
        if state.player.lost or state.opponent.lost:
            break
        turn = state.turn
        if state.player_to_move:  # cfr (player)
            before = state
            planning = GameState(state.player, state.opponent, state.turn, True)
            move = cfr.choose(planning)
            bel = cfr.last_report.get("belief", [])
            state = apply(state, move)
            cfr.observe_shields(before.opponent.shields)
            pl.observe(move.attacks, move.bonuses, move.switch,
                       budget=before.player.actions, turn=before.turn)
            label = "CFR"
            reach = cfr._reach
        else:  # planner
            before = state
            planning = GameState(state.opponent, state.player, state.turn, True)
            move = pl.choose(planning)
            bel = []
            reach = None
            state = apply(state, move)
            pl.observe_shields(before.player.shields)
            cfr.observe(move.attacks, move.bonuses, move.switch,
                        budget=before.opponent.actions, turn=before.turn)
            label = "PL "
        a, d, b = move.attacks, move.defends, move.bonuses
        sw = move.switch_to
        log.append((turn, label, a, d, b, sw,
                    [c.hp for c in state.player.characters],
                    [c.hp for c in state.opponent.characters], bel, reach))
    if state.opponent.lost:
        winner = "CFR"
    elif state.player.lost:
        winner = "PL"
    else:
        winner = "DRAW"
    return winner, log


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    shown = 0
    draws = 0
    for i in range(n):
        seed = 1100 + i
        new_first = (i % 2 == 0)
        winner, log = play(seed, new_first)
        if winner == "DRAW" and shown < 3:
            shown += 1
            print("=" * 70)
            print(f"seed={seed} new_first={new_first} -> DRAW")
            for turn, label, a, d, b, sw, hpA, hpB, bel, reach in log:
                sws = f" sw={sw}" if sw is not None else ""
                bel_s = ""
                if bel:
                    top = sorted(bel, key=lambda x: -x[2])[:2]
                    bel_s = "  bel:" + ",".join(f"({s},{k})={p:.2f}" for s, k, p in top)
                reach_s = ""
                if reach:
                    top = sorted(reach.items(), key=lambda x: -x[1])[:3]
                    reach_s = "  reach:" + ",".join(
                        f"({s},{b})={p:.2f}" for (s, b), p in top)
                print(f"  t{turn:2d} {label} a{a} d{d} b{b}{sws}"
                      f"  A={hpA} B={hpB}{bel_s}{reach_s}")
            draws += 1
        elif winner == "DRAW":
            draws += 1
    print("=" * 70)
    print(f"draws={draws}/{n}")


if __name__ == "__main__":
    main()
