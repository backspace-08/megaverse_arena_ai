"""Deterministic replay of Game 4 (Seed 3) with the EXACT action history.

Human moves (a,d,b) per turn, from the transcript:
  t2 (2,0,0), t4 (0,1,1), t6 (4,0,0), t8 (0,0,4), t10 (4,4,0),
  t12 (2,2,0), t14 (4,0,0)
Dumps CFR-bot diagnostics (belief, root strategy, value) on t9 and t11.
"""
import os
import random
import sys
from dataclasses import replace

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "src"))

from cote_megaverse.rules import (GameState, Type, apply, initial, Allocation)  # noqa: E402
from cote_megaverse.cfr_bot import CFRBot  # noqa: E402

HUMAN = {2: (2, 0, 0), 4: (0, 1, 1), 6: (4, 0, 0), 8: (0, 0, 4),
         10: (4, 4, 0), 12: (2, 2, 0), 14: (4, 0, 0)}


def main():
    rng = random.Random(3)
    types = list(Type)
    state = initial(tuple(rng.choice(types) for _ in range(3)),
                    tuple(rng.choice(types) for _ in range(3)), rng=rng)
    # seed3: bot first -> player_to_move=False
    state = replace(state, player_to_move=False).prepare()
    bot = CFRBot(depth=3, iters=120, cap=8, gamma=0.995, prune_after=20,
                 compress=True, rng=random.Random(3 * 1000003 + 17))
    print("YOU:", [c.type.name for c in state.player.characters], flush=True)
    print("BOT:", [c.type.name for c in state.opponent.characters], flush=True)
    for turn in range(1, 15):
        if state.player.lost or state.opponent.lost:
            break
        if state.player_to_move:  # human
            m = Allocation(*HUMAN[turn])
            before = state
            state = apply(state, m)
            bot.observe(m.attacks, m.bonuses, m.switch,
                        budget=before.player.actions, turn=before.turn)
            bot.observe_shields(before.opponent.shields)
        else:  # bot
            before = state
            planning = GameState(state.opponent, state.player, state.turn, True)
            move = bot.choose(planning)
            r = bot.last_report
            if turn in (7, 9, 11):
                print(f"\n=== TURN {turn} (BOT) ===", flush=True)
                print(f"  bot move: a{move.attacks} d{move.defends} b{move.bonuses}"
                      f" sw={move.switch_to}", flush=True)
                print("  value=%+.4f  belief=%s" % (r["value"], r["belief"]), flush=True)
                print("  root:", [(tuple(a[:3]), round(p, 4))
                                  for a, p in r["root_actions"]], flush=True)
            state = apply(state, move)
            bot.observe_shields(before.player.shields)
    w = "YOU" if state.opponent.lost else "BOT" if state.player.lost else "DRAW"
    print("\nwinner:", w, flush=True)


if __name__ == "__main__":
    main()
