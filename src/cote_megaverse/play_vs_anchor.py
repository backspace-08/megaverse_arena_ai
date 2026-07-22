"""Play human-controlled games against one selected anchor profile.

The actual game loop and keyboard controls remain in ``play_vs_champion``;
this module only adds focused anchor selection for manual strategy practice.
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime

try:
    from .play_vs_champion import (
        ANCHOR_PROFILES, EMOJI, HumanInputAI, PlayerQuit, AdaptiveAI,
        CounterAI, random_team, run_game, save_play_stats)
except ImportError:
    import os
    import sys
    _project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    from src.cote_megaverse.play_vs_champion import (
        ANCHOR_PROFILES, EMOJI, HumanInputAI, PlayerQuit, AdaptiveAI,
        CounterAI, random_team, run_game, save_play_stats)


def anchor_factories():
    """Return fresh anchor constructors so state cannot leak between games."""
    factories = {name: (lambda profile=profile: _profile_agent(profile))
                 for name, profile in ANCHOR_PROFILES}
    factories["Counter"] = CounterAI
    factories["Adaptive"] = AdaptiveAI
    return factories


def _profile_agent(profile):
    # Import locally to keep this wrapper dependent on the existing module API.
    try:
        from .parameterized_ai_v2 import WeightedRandomAIv2
    except ImportError:
        from src.cote_megaverse.parameterized_ai_v2 import WeightedRandomAIv2
    return WeightedRandomAIv2(profile)


def available_anchors():
    return tuple(anchor_factories())


def resolve_anchor(name: str):
    """Resolve an anchor name case-insensitively or raise ValueError."""
    factories = anchor_factories()
    normalized = name.strip().casefold()
    for available, factory in factories.items():
        if available.casefold() == normalized:
            return available, factory
    choices = ", ".join(factories)
    raise ValueError(f"Unknown anchor {name!r}. Available anchors: {choices}")


def _print_game_header(anchor_name: str, game_number: int, games: int,
                       team_you, team_ai, you_first: bool) -> None:
    print(f"\n{'=' * 55}")
    print(f"  Game #{game_number}/{games}: vs {anchor_name}")
    print(f"{'=' * 55}")
    print(f"You: {' '.join(EMOJI.get(t, '?') for t in team_you)}  "
          f"[{','.join(t.value for t in team_you)}]")
    print(f"AI:  {' '.join(EMOJI.get(t, '?') for t in team_ai)}  "
          f"[{','.join(t.value for t in team_ai)}]")
    print(f"\n  {'You go FIRST!' if you_first else 'AI goes FIRST!'}")
    print("  Opponent profile: " + anchor_name)


def play_selected_anchor(anchor_name: str, games: int = 1) -> dict:
    """Play ``games`` manual matches against one anchor and return results."""
    selected_name, factory = resolve_anchor(anchor_name)
    games = max(1, int(games))
    result = {"name": selected_name, "won": 0, "lost": 0, "draw": 0,
              "completed": 0}

    for game_number in range(1, games + 1):
        you_first = random.random() < 0.5
        team_you = random_team()
        team_ai = random_team()
        _print_game_header(selected_name, game_number, games,
                           team_you, team_ai, you_first)
        anchor = factory()
        try:
            winner = run_game(HumanInputAI(), anchor, team_you, team_ai,
                              you_first)
        except PlayerQuit:
            print("\n  [QUIT] Session stopped.")
            break

        result["completed"] += 1
        if winner == 0:
            result["draw"] += 1
            print("\n  [DRAW] Turn limit")
        elif (winner == 1 and you_first) or (winner == 2 and not you_first):
            result["won"] += 1
            print("\n  [WIN] You win!")
        else:
            result["lost"] += 1
            print("\n  [LOSE] You lose")

    print(f"\n--- vs {selected_name}: {result['won']}W / "
          f"{result['lost']}L / {result['draw']}D ---")
    save_play_stats({
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "mode": "single_anchor",
        "games_requested": games,
        "anchor": selected_name,
        "won": result["won"],
        "lost": result["lost"],
        "draw": result["draw"],
        "completed": result["completed"],
        "anchors": [{"name": selected_name, "won": result["won"],
                     "lost": result["lost"]}],
    })
    return result


def choose_anchor() -> str:
    names = available_anchors()
    print("\nAvailable anchors:")
    for index, name in enumerate(names, 1):
        print(f"  {index}. {name}")
    choice = input("\nChoose anchor (name or number): ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(names):
        return names[int(choice) - 1]
    resolve_anchor(choice)  # Produce the useful error for an invalid name.
    return choice


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Play against one selected COTE Megaverse anchor")
    parser.add_argument("--anchor", help="Anchor name, for example Defender")
    parser.add_argument("--games", type=int, default=None,
                        help="Number of games (default: 1)")
    args = parser.parse_args(argv)

    try:
        anchor_name = args.anchor or choose_anchor()
        games = args.games
        if games is None:
            raw_games = input("Games (default=1): ").strip()
            games = int(raw_games) if raw_games.isdigit() else 1
        play_selected_anchor(anchor_name, games)
    except ValueError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
