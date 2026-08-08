"""Small deterministic benchmark and replay helpers for the new engine."""

import json
import random
from dataclasses import replace

from .agent import Planner
from .observation import observe
from .rules import GameState, Type, apply, exchange_damage, legal_allocations, initial


def _policy_state(state: GameState) -> GameState:
    """Give a human policy its own-side perspective without enemy shields."""
    return GameState(
        replace(state.player, shields=state.player.shields),
        replace(state.opponent, shields=0),
        state.turn,
        True,
    )


def choose_human_policy(state: GameState, policy: str, rng: random.Random):
    """Choose a deterministic human-like move from public own-side state."""
    view = _policy_state(state)
    moves = legal_allocations(view.player)
    if policy == "random":
        return rng.choice(moves)
    target = view.opponent.active_character

    def score(move):
        attacker = view.player.active_character
        if move.switch:
            attacker = view.player.characters[move.switch_to]
        damage = exchange_damage(attacker, target, move.attacks)
        lethal = int(damage >= target.hp)
        survival = move.defends * 700
        bonus = move.bonuses * (500 if policy == "bonus_shield" else 150)
        if policy == "greedy":
            return lethal * 100000 + damage * 10 + survival + bonus
        if policy == "bonus_shield":
            return lethal * 100000 + survival + bonus + damage * 2
        return damage

    return max(moves, key=score)


def run_match(seed=0, policy="greedy", depth=2, max_half_turns=100,
              ai_starts=False):
    """Run one AI-vs-human-policy match with public-information updates."""
    rng = random.Random(seed)
    types = list(Type)
    state = initial(tuple(rng.choice(types) for _ in range(3)),
                    tuple(rng.choice(types) for _ in range(3)), rng=rng)
    if ai_starts:
        state = replace(state, player_to_move=False).prepare()
    planner = Planner(depth=depth)
    metrics = {"missed_guaranteed_lethal": 0, "guaranteed_loss_moves": 0,
               "ai_turns": 0, "human_turns": 0, "hit_turn_limit": False}
    replay = []
    hit_limit = True
    for _ in range(max_half_turns):
        if state.player.lost or state.opponent.lost:
            hit_limit = False
            break
        if state.player_to_move:
            # Human's turn: record what the human did so the planner can
            # update its belief model of the human.  No shield observation
            # here — the human attacking reveals the AI's shields, not the
            # human's; the planner models the human, so we only call
            # observe_shields when the AI attacks and the human's shields
            # are publicly consumed.
            before = state
            move = choose_human_policy(state, policy, rng)
            metrics["human_turns"] += 1
            planner.observe(move.attacks, move.bonuses, move.switch,
                            budget=before.player.actions)
        else:
            # AI's turn: state.player = human, state.opponent = AI.
            # Flip perspective so the planner sees itself as player.
            before = state
            planning_state = state.__class__(
                state.opponent, state.player, state.turn, True)
            move = planner.choose(planning_state)
            metrics["ai_turns"] += 1
            tactical = planner.last_report.get("tactical_outcome", {})
            if any(item.get("guaranteed_lethal")
                   for item in [tactical] if item) and not move.attacks:
                metrics["missed_guaranteed_lethal"] += 1
            if tactical.get("guaranteed_immediate_loss", False):
                metrics["guaranteed_loss_moves"] += 1
            # When the AI attacks it hits the human's shields
            # (before.player is the human).  This is the only moment
            # the human's shield count is publicly revealed.
            if move.attacks:
                planner.observe_shields(before.player.shields)
        replay.append({"turn": state.turn, "player_to_move": state.player_to_move,
                       "move": move.label})
        state = apply(state, move)
    metrics["hit_turn_limit"] = hit_limit
    # ``state.player`` is the human policy and ``state.opponent`` is the AI.
    # A side that ``lost`` has no living characters, so the *other* side won.
    winner = ("human" if state.opponent.lost
              else "ai" if state.player.lost else "draw")
    return {"seed": seed, "policy": policy, "depth": depth,
            "ai_starts": ai_starts,
            "winner": winner, "replay": replay, "metrics": metrics,
            "state": state}


def _seat_stats(matches):
    """Return {games, wins, losses, draws} for a list of match results."""
    return {
        "games": len(matches),
        "wins": sum(m["winner"] == "ai" for m in matches),
        "losses": sum(m["winner"] == "human" for m in matches),
        "draws": sum(m["winner"] == "draw" for m in matches),
    }


def benchmark_policies(seeds=range(20), depth=2, max_half_turns=100):
    """Compare the planner against each bundled human-like policy."""
    result = {}
    for policy in ("random", "greedy", "bonus_shield"):
        matches = [
            run_match(seed, policy, depth, max_half_turns, ai_starts=ai_starts)
            for seed in seeds for ai_starts in (False, True)
        ]
        ai_first_matches = [m for m in matches if m["ai_starts"]]
        human_first_matches = [m for m in matches if not m["ai_starts"]]
        combined = _seat_stats(matches)
        result[policy] = {
            # Combined totals (kept for backward compatibility)
            "games": combined["games"],
            "wins": combined["wins"],
            "losses": combined["losses"],
            "draws": combined["draws"],
            # Per-seat breakdown
            "ai_first": _seat_stats(ai_first_matches),
            "human_first": _seat_stats(human_first_matches),
            "missed_guaranteed_lethal": sum(
                m["metrics"]["missed_guaranteed_lethal"] for m in matches),
            "matches": matches,
        }
    return result


def analyze_position(state: GameState, depth=3) -> dict:
    planner = Planner(depth=depth)
    move = planner.choose(state)
    return {
        "turn": state.turn,
        "player_to_move": state.player_to_move,
        "observation": observe(state),
        "report": planner.last_report,
        "best_move": move.label,
    }


def run_self_play(seed=0, max_half_turns=40, depth=2):
    """Run two independent fair planners and return a replayable report."""
    random.seed(seed)
    types = list(Type)
    rng = random.Random(seed)
    state = initial(tuple(rng.choice(types) for _ in range(3)),
                    tuple(rng.choice(types) for _ in range(3)), rng=rng)
    planners = {True: Planner(depth=depth), False: Planner(depth=depth)}
    replay = []
    for _ in range(max_half_turns):
        if state.player.lost or state.opponent.lost:
            break
        planner = planners[state.player_to_move]
        planning_state = state if state.player_to_move else state.__class__(
            state.opponent, state.player, state.turn, True)
        move = planner.choose(planning_state)
        replay.append({
            "turn": state.turn,
            "player_to_move": state.player_to_move,
            "move": move.label,
            "report": planner.last_report,
            "move_quality": planner.last_report.get("move_quality", {}),
        })
        if move.attacks:
            target = state.opponent if state.player_to_move else state.player
            planner.observe_shields(target.shields)
        planners[not state.player_to_move].observe(
            move.attacks, move.bonuses, move.switch,
            budget=(state.player if state.player_to_move
                    else state.opponent).actions)
        state = apply(state, move)
    winner = "opponent" if state.player.lost else "player" if state.opponent.lost else "draw"
    return {"seed": seed, "winner": winner, "replay": replay, "state": state}


def write_replay(report, path):
    serializable = {key: value for key, value in report.items() if key != "state"}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2, default=str)
