"""Replay a recorded game and dump the CFR bot's belief / root strategy / value
at every bot turn, plus verify each resolution against the recorded log.

The game is described by three tables below (edit to replay any game):

- CHAR_ME / CHAR_BOT: every character of each side as (type, hp, atk) tuples,
  index = character id. Give dead bench characters hp=0 (they are ignored).
- ACTIVE_ME / ACTIVE_BOT: active character id.
- STACK_ME / STACK_BOT: stack order (front = active), living ids only.
- START_BANK_ME: the bank the side carries INTO the start turn (prepare()
  drains it into the action budget). The bot's bank is inferred by apply().
- MOVES: one row per half-turn, in order: (turn, mover, (attacks, defends,
  bonuses, switch_to)). switch_to is 0-based or None. mover in {"me", "bot"}.
- EXPECTED: optional (turn, "me"|"bot", resulting hp of each side's active
  char) to verify the replay against a hand-recorded log.

Usage:
    python server/replay_strange.py
"""

import os
import random
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "src"))

from cote_megaverse.rules import (GameState, Side, Character, Type, apply,  # noqa: E402
                                  Allocation, base_budget)
from cote_megaverse.cfr_bot import CFRBot  # noqa: E402


def Ch(t, hp, atk):
    if isinstance(t, str):
        t = Type[t]
    return Character(type=t, hp=hp, atk=atk, max_hp=hp)


# ---------------- EDIT THIS BLOCK: the game being replayed ----------------
# Replay from turn 1: give FULL teams (3 alive chars), stack (0,1,2),
# banks 0, and the MOVES rows for turns 1..N. Replay from mid-game:
# give the on-board state (dead chars hp=0), live stack, known banks.
START_TURN = 1
BOT_MOVES_FIRST = True       # True = turn 1 is the bot; False = turn 1 is me
CHAR_ME = [("B", 6300, 2000), ("B", 5900, 2100), ("C", 5800, 1900)]
ACTIVE_ME = 0
STACK_ME = (0, 1, 2)
START_BANK_ME = 0

CHAR_BOT = [("D", 6200, 2000), ("A", 6100, 2100), ("A", 6000, 2000)]
ACTIVE_BOT = 0
STACK_BOT = (0, 1, 2)
START_BANK_BOT = 0

# (turn, mover, (attacks, defends, bonuses, switch_to))
MOVES = [
    (1, "bot", (0, 0, 1, None)),     # bank 1
    (2, "me",  (2, 0, 0, None)),     # 4000 to D(6200) -> 2200
    (3, "bot", (1, 0, 1, 1)),        # switch to A(6100), 1 atk (2700), bank 1
    (4, "me",  (0, 1, 1, None)),     # 1 shield, bank 1
    (5, "bot", (4, 0, 0, None)),     # 3 landed vs my 1 shield -> kill B(3600)
    (6, "me",  (4, 0, 0, None)),     # 6000 to A(6100) -> 100
    (7, "bot", (2, 0, 1, 2)),        # switch to idx2 A(6000), 2 atk, 1 bonus
    (8, "me",  (0, 0, 4, None)),
    (9, "bot", (4, 0, 1, None)),     # kills my B(700)
    (10, "me", (4, 4, 0, None)),     # kills bot A(6000)
    (11, "bot", (4, 0, 1, None)),    # BLOCKED by my 4 shields
    (12, "me", (2, 2, 0, None)),     # kills bot A(100)
    (13, "bot", (5, 0, 0, None)),    # 3 landed, 4200 to my C
    (14, "me", (4, 0, 0, None)),     # kills bot D(2200) -> I win
]
EXPECTED_HP = {}
# ---------------------------------------------------------------------------


def build_side(spec_chars, active, stack, bank, turn):
    chars = tuple(Ch(*c) for c in spec_chars)
    return Side(characters=chars, active=active, stack_order=tuple(stack),
                bonus=bank, shields=0,
                actions=min(8, base_budget(turn) + bank))


def dump(bot, move, turn):
    r = bot.last_report
    print("\n" + "=" * 70)
    print(f"BOT turn {turn}")
    print(f"  chosen: a{move.attacks} d{move.defends} b{move.bonuses} "
          f"sw={move.switch_to if move.switch_to is not None else '-'}")
    print(f"  value : {r['value']:+.4f}")
    print(f"  belief: " + ", ".join(
        f"(sh={s},bk={b})={p:.3f}" for s, b, p in r["belief"]))
    full = sorted(r["root_actions"], key=lambda x: -x[1])
    print("  root strategy (top by prob):")
    for act, p in full:
        if p <= 0.005:
            continue
        a, d, b, sw = act
        tag = f"sw={sw}" if sw >= 0 else "    "
        print(f"    a{a} d{d} b{b} {tag}  p={p:.3f}")


def main():
    me = build_side(CHAR_ME, ACTIVE_ME, STACK_ME, START_BANK_ME, START_TURN)
    bot = build_side(CHAR_BOT, ACTIVE_BOT, STACK_BOT, START_BANK_BOT,
                     START_TURN)
    state = GameState(player=me, opponent=bot, turn=START_TURN,
                      player_to_move=(not BOT_MOVES_FIRST)).prepare()
    bot_obj = CFRBot(depth=3, iters=120, cap=8, gamma=0.995, prune_after=20,
                     compress=True, rng=random.Random(1))

    print("YOU:", [f"{i}:{c.type.name}/{c.hp}/{c.atk}"
                   for i, c in enumerate(state.player.characters)])
    print("BOT:", [f"{i}:{c.type.name}/{c.hp}/{c.atk}"
                   for i, c in enumerate(state.opponent.characters)])
    print(f"start turn {START_TURN}, my budget={state.player.actions}")

    for turn, mover, (a, d, b, sw) in MOVES:
        before = state
        if mover == "me":
            move = Allocation(a, d, b, sw)
            state = apply(state, move)
            bot_obj.observe(move.attacks, move.bonuses, move.switch,
                            budget=before.player.actions)
        else:
            # Populate the bot's belief/root strategy for THIS position, but
            # advance the game with the RECORDED move (the bot sampled from the
            # root distribution in the real game; we reproduce the position).
            planning = GameState(state.opponent, state.player, state.turn, True)
            chosen = bot_obj.choose(planning)
            move = Allocation(a, d, b, sw)
            dump(bot_obj, chosen, state.turn)
            state = apply(state, move)
            bot_obj.observe_shields(before.player.shields)

        got = (state.player.characters[state.player.active].hp,
               state.opponent.characters[state.opponent.active].hp)
        exp = EXPECTED_HP.get((turn, mover))
        tag = ""
        if exp is not None and got != exp:
            tag = f"   <-- MISMATCH expected {exp}"
        print(f"  T{turn:2d} {mover:3s}: a{a} d{d} b{b} sw={sw} "
              f"-> active hp (me={got[0]}, bot={got[1]}){tag}")

    print("\nwinner:", "YOU" if state.opponent.lost else "BOT")


if __name__ == "__main__":
    main()
