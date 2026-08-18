"""Focused 1v1-duel depth probe: replay the exact seed-1113 deadlock matchup
(bot B:6100/1900 vs turtler A:6100/2000, turn 13) at CFR depths 3..6. A 1v1
duel is cheap (small tree), so we can see whether a deeper resolve finds the
double-burst (a8x2, since 4x1300=5200 < 6100 needs two bursts) that breaks the
deadlock, and measure the latency.
"""
import os
import sys
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "src"))

import random  # noqa: E402
from cote_megaverse.rules import (GameState, Side, Character, Type,  # noqa: E402
                                  apply)
from cote_megaverse.cfr_bot import CFRBot  # noqa: E402


def Ch(t, hp, atk):
    return Character(type=t, hp=hp, atk=atk, max_hp=hp)


def play_duel(depth, iters=80, start_turn=13):
    # bot = B:6100/1900 (player), turtler = A:6100/2000 (opponent)
    bot = Side((Ch(Type.B, 6100, 1900),), active=0, stack_order=(0,))
    turtler = Side((Ch(Type.A, 6100, 2000),), active=0, stack_order=(0,),
                   shields=4)
    state = GameState(player=bot, opponent=turtler, turn=start_turn,
                      player_to_move=True).prepare()
    cfr = CFRBot(depth=depth, iters=iters, cap=6, compress=True,
                 rng=random.Random(1))
    turns = 0
    lat = []
    while not (state.player.lost or state.opponent.lost) and turns < 80:
        if state.player_to_move:
            before = state
            planning = GameState(state.player, state.opponent, state.turn, True)
            t0 = time.time()
            move = cfr.choose(planning)
            lat.append((time.time() - t0) * 1000)
            state = apply(state, move)
            cfr.observe_shields(before.opponent.shields)
        else:
            before = state
            # turtler: a0 d4 (budget 4 -> 4 shields, 0 attacks, 0 bank)
            move = type("M", (), {"attacks": 0, "defends": 4, "bonuses": 0,
                                  "switch": False})()
            state = apply(state, move)
            cfr.observe(move.attacks, move.bonuses, move.switch,
                        budget=before.opponent.actions, turn=before.turn)
        turns += 1
    if state.opponent.lost:
        winner = "BOT"
    elif state.player.lost:
        winner = "TURT"
    else:
        winner = "DRAW"
    return winner, turns, lat


def main():
    for depth in [3, 4, 5, 6]:
        t0 = time.time()
        w, t, lat = play_duel(depth)
        wall = time.time() - t0
        avg = sum(lat) / len(lat) if lat else 0
        print(f"depth={depth} winner={w:5s} half_turns={t:3d} "
              f"avg_lat={avg:.0f}ms max_lat={max(lat) if lat else 0:.0f}ms "
              f"wall={wall:.1f}s", flush=True)


if __name__ == "__main__":
    main()
