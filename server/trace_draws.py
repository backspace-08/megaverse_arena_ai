"""Rich per-turn trace of a cfr_vs_planner game.

play() returns (winner, log, rosterA, rosterB). Each log entry is a dict with:
  turn, label (CFR/PL), a/d/b/sw, budget, actA/actB (active idx:Type/hp/atk),
  hpA/hpB (all chars), mult/per_hit of the attack matchup (if any), and the
  bot's belief + reach.
"""
import os
import random
import sys
from dataclasses import replace

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "src"))

from cote_megaverse.rules import (GameState, Type, apply, initial,  # noqa: E402
                                  multiplier, per_hit_damage)
from cote_megaverse.agent import Planner  # noqa: E402
from cote_megaverse.cfr_bot import CFRBot  # noqa: E402

MAX_TURNS = 80


def _roster(side):
    order = list(side.normalized_order())
    chars = []
    for idx in order:
        c = side.characters[idx]
        star = "*" if idx == side.active else " "
        chars.append(f"{star}{idx}:{c.type.name}/{c.hp}/{c.atk}")
    return ", ".join(chars)


def _fmt_active(side):
    c = side.characters[side.active]
    return f"#{side.active}:{c.type.name}/{c.hp}/{c.atk}"


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
        before = state
        pa0 = before.player.active
        oa0 = before.opponent.active
        if state.player_to_move:  # cfr (player)
            planning = GameState(state.player, state.opponent, state.turn, True)
            move = cfr.choose(planning)
            bel = cfr.last_report.get("belief", [])
            state = apply(state, move)
            cfr.observe_shields(before.opponent.shields)
            pl.observe(move.attacks, move.bonuses, move.switch,
                       budget=before.player.actions, turn=before.turn)
            label = "CFR"
            budget = before.player.actions
        else:  # planner
            planning = GameState(state.opponent, state.player, state.turn, True)
            move = pl.choose(planning)
            bel = []
            state = apply(state, move)
            pl.observe_shields(before.player.shields)
            cfr.observe(move.attacks, move.bonuses, move.switch,
                        budget=before.opponent.actions, turn=before.turn)
            label = "PL "
            budget = before.opponent.actions
        # attacker's active AFTER the switch (attacker never dies on its own
        # turn, so its post-switch active is stable); defender's active BEFORE.
        if label.strip() == "CFR":
            atk_c = state.player.characters[state.player.active]
            def_c = before.opponent.characters[oa0]
        else:
            atk_c = state.opponent.characters[state.opponent.active]
            def_c = before.player.characters[pa0]
        mult = per_hit = None
        if move.attacks > 0:
            mult = multiplier(atk_c.type, def_c.type)
            per_hit = per_hit_damage(atk_c, def_c)
        log.append({
            "turn": state.turn - 1, "label": label, "a": move.attacks,
            "d": move.defends, "b": move.bonuses,
            "sw": move.switch_to, "budget": budget,
            "actA": _fmt_active(before.player), "actB": _fmt_active(before.opponent),
            "hpA": [c.hp for c in state.player.characters],
            "hpB": [c.hp for c in state.opponent.characters],
            "mult": mult, "per_hit": per_hit, "bel": bel,
            "reach": cfr._reach,
        })
    if state.opponent.lost:
        winner = "CFR"
    elif state.player.lost:
        winner = "PL"
    else:
        winner = "DRAW"
    return winner, log, _roster(state.player), _roster(state.opponent)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    shown = 0
    for i in range(n):
        seed = 1100 + i
        new_first = (i % 2 == 0)
        winner, log, ra, rb = play(seed, new_first)
        if winner == "DRAW" and shown < 3:
            shown += 1
            print(f"==== seed={seed} new_first={new_first} -> DRAW ====")
            print("A(CFR):", ra)
            print("B(PL):", rb)
            for e in log:
                m = f" mult={e['mult']} ph={e['per_hit']}" if e["mult"] else ""
                print(f"  t{e['turn']:2d} {e['label']} a{e['a']} d{e['d']} "
                      f"b{e['b']} sw={e['sw']} bgt={e['budget']}"
                      f"  A-act:{e['actA']} B-act:{e['actB']}{m}"
                      f"  A={e['hpA']} B={e['hpB']}")


if __name__ == "__main__":
    main()
