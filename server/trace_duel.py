"""Line-by-line duel trace: CFRBot vs Planner, Profile A, phase-4 start.

Usage (server):
    python server/trace_duel.py [--seed 7000] [--cfr-first] [--start-turn 7]
Prints every move (action, HP both sides) and, on CFR moves, its resolve
value, belief, and root strategy.
"""
import argparse
import os
import random
import sys
from dataclasses import replace

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "src"))

from cote_megaverse.rules import (  # noqa: E402
    GameState, Side, Character, Type, apply)
from cote_megaverse.agent import Planner  # noqa: E402
from cote_megaverse.cfr_bot import CFRBot  # noqa: E402

HP, ATK = 24000, 1500


def make_side(t):
    return Side(characters=(Character(type=t, hp=HP, atk=ATK, max_hp=HP),),
                active=0, stack_order=(0,), bonus=0, shields=0, actions=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--cfr-first", action="store_true",
                    help="CFR moves first (default: Planner first)")
    ap.add_argument("--start-turn", type=int, default=7)
    ap.add_argument("--cfr-iters", type=int, default=120)
    ap.add_argument("--cfr-depth", type=int, default=3)
    ap.add_argument("--max-half-turns", type=int, default=40)
    a = ap.parse_args()

    state = GameState(player=make_side(Type.A), opponent=make_side(Type.A),
                      turn=a.start_turn, player_to_move=True).prepare()
    if not a.cfr_first:
        state = replace(state, player_to_move=False).prepare()
    cfr = CFRBot(depth=a.cfr_depth, iters=a.cfr_iters, cap=6, gamma=0.995,
                 prune_after=20, temperature=1.0, rng=random.Random(a.seed))
    pl = Planner(depth=2, max_nodes=2000)
    print("=== CFR %s (seed %d, start turn %d) ==="
          % ("first" if a.cfr_first else "second", a.seed, a.start_turn),
          flush=True)
    for _ in range(a.max_half_turns):
        if state.player.lost or state.opponent.lost:
            print("[end] %s won at turn %d"
                  % ("CFR" if state.opponent.lost else "PL", state.turn),
                  flush=True)
            return
        hp_p = state.player.characters[0].hp
        hp_o = state.opponent.characters[0].hp
        if state.player_to_move:
            before = state
            planning = GameState(state.player, state.opponent, state.turn, True)
            move = cfr.choose(planning)
            r = cfr.last_report
            strat = " ".join("%s=%.2f" % (str(tuple(x[:3])), p)
                             for x, p in sorted(r["root_actions"],
                                                key=lambda x: -x[1])[:6])
            bel = str([(w.shields, w.bank, round(w.probability, 2))
                       for w in cfr.model.worlds()])
            reach = str({(sh, bk): round(p, 3) for (sh, bk), p in
                         (cfr._reach or {}).items()})
            print("t%2d CFR (a%d d%d b%d)  CFR-HP=%d PL-HP=%d  val=%+.3f"
                  % (state.turn, move.attacks, move.defends, move.bonuses,
                     hp_p, hp_o, r["value"]), flush=True)
            print("     reach: %s" % reach, flush=True)
            print("     bel:   %s" % bel, flush=True)
            print("     strat: %s" % strat, flush=True)
            state = apply(state, move)
            cfr.observe_shields(before.opponent.shields)
            pl.observe(move.attacks, move.bonuses, move.switch,
                       budget=before.player.actions)
        else:
            before = state
            planning = GameState(state.opponent, state.player, state.turn, True)
            move = pl.choose(planning)
            print("t%2d PL  (a%d d%d b%d)  CFR-HP=%d PL-HP=%d"
                  % (state.turn, move.attacks, move.defends, move.bonuses,
                     hp_p, hp_o), flush=True)
            state = apply(state, move)
            pl.observe_shields(before.player.shields)
            cfr.observe(move.attacks, move.bonuses, move.switch,
                        budget=before.opponent.actions)
    print("[cap] DRAW at turn %d (CFR-HP=%d PL-HP=%d)"
          % (state.turn, state.player.characters[0].hp,
             state.opponent.characters[0].hp), flush=True)


if __name__ == "__main__":
    main()

