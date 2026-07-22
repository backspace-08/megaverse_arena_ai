"""
Play against the NeuralAgent champion!
Clean version with proper turn-by-turn display.
"""
import sys, os, random, json, copy
from datetime import datetime
import numpy as np
try:
    from .coevolution import NeuralAgent, SmartNeuralAgent
    from .parameterized_ai_v2 import (BattleEngineV2, BattleAction, Player, Character,
                                     CharType, get_type_multiplier, random_team, make_character,
                                     MAX_BONUS_ACTIONS, MAX_TOTAL_ACTIONS, ACTION_COST_SWITCH, BASE_HP, TURN_ACTIONS,
                                     TurnLog, AIProfile, WeightedRandomAIv2, CounterAI, AdaptiveAI,
                                     calculate_damage, round_damage)
except ImportError:
    # Also support `python src/cote_megaverse/play_vs_champion.py`.
    _project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    from src.cote_megaverse.coevolution import NeuralAgent, SmartNeuralAgent
    from src.cote_megaverse.parameterized_ai_v2 import (
        BattleEngineV2, BattleAction, Player, Character, CharType,
        get_type_multiplier, random_team, make_character, MAX_BONUS_ACTIONS,
        MAX_TOTAL_ACTIONS, ACTION_COST_SWITCH, BASE_HP, TURN_ACTIONS, TurnLog,
        AIProfile, WeightedRandomAIv2, CounterAI, AdaptiveAI, calculate_damage,
        round_damage)

EMOJI = {CharType.A: "[A]", CharType.B: "[B]", CharType.C: "[C]", CharType.D: "[D]"}
TYPE_NAMES = {CharType.A: "Artist", CharType.B: "Brawler", CharType.C: "Coordinator", CharType.D: "Defender"}
_project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
_artifacts_dir = os.path.join(_project_root, "artifacts")
BATTLE_LOG_FILE = os.path.join(_artifacts_dir, "battle_logs.jsonl")
STATS_FILE = os.path.join(_artifacts_dir, "play_stats.json")

class PlayerQuit(Exception):
    pass

def list_versions():
    versions_path = os.path.join(_artifacts_dir, 'versions.json')
    if not os.path.exists(versions_path):
        return []
    try:
        with open(versions_path, encoding="utf-8") as f:
            versions = json.load(f)
        return versions if isinstance(versions, list) else []
    except (OSError, json.JSONDecodeError):
        return []

def _load_agent(genome, name):
    genome = np.asarray(genome, dtype=np.float32)
    if genome.shape[0] == 12:
        return SmartNeuralAgent(genome, name)
    return NeuralAgent(genome, name)

def load_champion():
    ai_dir = _artifacts_dir
    versions = list_versions()
    
    if versions:
        print("\nAvailable champions:")
        print(f"  {'#':2s}  {'Timestamp':16s}  {'Fitness':8s}  {'Type':6s}  {'Params'}")
        print(f"  {'-'*2}  {'-'*16}  {'-'*8}  {'-'*6}  {'-'*30}")
        for i, v in enumerate(versions):
            p = v.get("params", {})
            at = v.get("agent_type", "lstm")
            if isinstance(p, dict):
                pstr = f"pop={p.get('pop_size','?')} gen={p.get('generations','?')}"
            else:
                pstr = str(p)
            print(f"  {i+1:2d}  {v['timestamp']:16s}  {v['best_fitness']:.1%}  {at:6s}  {pstr}")
        print(f"  {'L':2s}  (latest)")
        print()
        choice = input("Select champion (Enter=latest, number=version): ").strip().lower()
        if choice == "q":
            sys.exit(0)
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(versions):
                ts = versions[idx]["timestamp"]
                path = os.path.join(ai_dir, f"best_genome_{ts}.npy")
                if os.path.exists(path):
                    agent = _load_agent(np.load(path), f"v{idx+1}({ts})")
                    print(f"Loaded {agent.name}\n")
                    return agent
        # Fallback to latest
    path = os.path.join(ai_dir, 'best_genome.npy')
    if os.path.exists(path):
        return _load_agent(np.load(path), "Champion")
    try:
        from .coevolution import random_genome, random_smart_genome
    except ImportError:
        from src.cote_megaverse.coevolution import random_genome, random_smart_genome
    print("best_genome.npy NOT FOUND -- using random agent!")
    return _load_agent(random_smart_genome(), "Random")

def bar(val, max_val, width=12):
    filled = int(val / max_val * width) if max_val > 0 else 0
    return "#" * filled + "." * (width - filled)


# ============================================================
# TURN LOG PARSER
# ============================================================

def last_turn_summary(logs, my_player_id, my_prev_hp, opp_prev_hp):
    """Parse recent logs to describe what happened since last my turn."""
    # Filter logs since my last turn (excluding my own turn)
    recent = []
    for t in logs:
        if t.player_id != my_player_id:
            recent.append(t)
    if not recent:
        return None
    return recent[-1]  # Most recent opponent turn

def describe_turn(log, player_label, prev_hp, cur_chars):
    """Human-readable summary of a turn."""
    out = []
    atk = log.attack_actions
    dfn = log.defend_actions
    bon = log.bonus_actions
    dmg = log.total_damage
    blocked = log.unblocked_attacks is not None and log.attack_actions - log.unblocked_attacks
    
    parts = []
    if atk > 0: parts.append(f"{atk} атак ({dmg} ур)")
    if dfn > 0: parts.append(f"{dfn} щитов")
    if bon > 0: parts.append(f"{bon} бонусов")
    if log.switched:
        # Detect which char they switched to by comparing HP changes
        out.append(f"  {player_label} сменил персонажа")
    if parts:
        out.append(f"  {player_label}: {', '.join(parts)}")
    return "\n".join(out)


# ============================================================
# CUSTOM GAME LOOP
# ============================================================

def run_game(human, champion, team_human, team_ai, human_first=True,
             battle_log_path=BATTLE_LOG_FILE):
    """
    Manual game loop with full display control.
    Returns winner: 1=human, 2=AI, 0=draw
    """
    if human_first:
        p1, p2 = Player(1, [make_character(t) for t in team_human]), Player(2, [make_character(t) for t in team_ai])
        p1_ai, p2_ai = human, champion
    else:
        p1, p2 = Player(1, [make_character(t) for t in team_ai]), Player(2, [make_character(t) for t in team_human])
        p1_ai, p2_ai = champion, human
    
    # Track AI active char index changes for switch detection
    ai_active_history = []
    
    turn_num = 1
    current_player = 1
    logs = []
    battle_record = {
        "schema_version": 2,
        "record_type": "champion_user_battle",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "human_first": human_first,
        "players": {
            "human": {"name": getattr(human, "name", "Human"), "player_id": 1 if human_first else 2},
            "champion": {"name": getattr(champion, "name", "Champion"), "player_id": 2 if human_first else 1},
        },
        "teams": {
            "human": [t.value for t in team_human],
            "champion": [t.value for t in team_ai],
        },
        "turns": [],
    }

    def finish_battle(winner):
        """Write the complete record once, including draw/win outcome."""
        battle_record["winner_player_id"] = winner
        battle_record["winner"] = (
            "draw" if winner == 0 else
            "human" if (winner == 1 and human_first) or (winner == 2 and not human_first)
            else "champion")
        battle_record["turn_count"] = len(battle_record["turns"])
        battle_record["finished_at"] = datetime.now().isoformat(timespec="seconds")
        _append_battle_log(battle_log_path, battle_record)
        return winner
    
    def setup_turn(player, tn):
        base = min(TURN_ACTIONS.get(tn, 4), MAX_TOTAL_ACTIONS)
        total = min(base + player.bonus_actions, MAX_TOTAL_ACTIONS)
        used_bonus = min(player.bonus_actions, total - base)
        player.bonus_actions -= used_bonus
        player.base_actions = base
        player.remaining_actions = total
    
    def execute_turn(player, opponent, actions, player_id):
        """Execute actions and return a TurnLog with results."""
        atk_count = sum(1 for a in actions if a.action_type == "attack")
        def_count = sum(1 for a in actions if a.action_type == "defend")
        bon_count = sum(1 for a in actions if a.action_type == "bonus")
        sw_count = 1 if (player.switched_this_round and not player.forced_switch_after_death) else 0
        
        total_actions = atk_count + def_count + bon_count
        player.remaining_actions -= total_actions
        
        player_shields_before = player.shields
        opp_shields = opponent.shields
        unblocked = max(0, atk_count - opp_shields)
        blocked_count = min(atk_count, opp_shields)
        
        total_dmg = 0
        atk_char = player.active_character
        
        # Shields protect only this immediately following turn and then burn
        # completely, even when the opponent made no attacks.
        opponent.shields = 0
        
        if unblocked > 0 and opponent.active_character.is_alive():
            total_dmg = calculate_damage(
                atk_char, opponent.active_character, atk_count, blocked_count)
            total_dmg = min(total_dmg, opponent.active_character.hp)
            opponent.active_character.take_damage(total_dmg)
        
        if not opponent.active_character.is_alive():
            opponent.force_switch_from_death()
        
        actual_bon = min(bon_count, MAX_BONUS_ACTIONS - player.bonus_actions)
        player.bonus_actions += actual_bon
        
        player.shields = def_count
        
        return TurnLog(
            turn_num=0, player_id=player_id,
            attack_actions=atk_count, defend_actions=def_count,
            bonus_actions=bon_count, switched=(sw_count > 0),
            unblocked_attacks=unblocked, total_damage=total_dmg,
            opponent_shields=opp_shields,
            blocked_shields=blocked_count,
            player_shields_before=player_shields_before,
            p1_hp=[c.hp for c in p1.characters],
            p2_hp=[c.hp for c in p2.characters],
        )
    
    def show_full_state(my_p, opp_p, turn_n, after_ai=False):
        """Show the current game state."""
        print()
        print("=" * 55)
        print(f"  TURN #{turn_n} -- {'Your turn' if after_ai else 'Your turn'}")
        print("=" * 55)
        
        # My team
        for i in my_p.stack_order:
            c = my_p.characters[i]
            mark = ">" if i == my_p.active_char_index else " "
            dead = "[X]" if not c.is_alive() else ""
            shield_s = f" [S]x{my_p.shields}" if i == my_p.active_char_index and my_p.shields > 0 else ""
            print(f"  {mark} {EMOJI.get(c.char_type,'?')}#{i+1} {TYPE_NAMES.get(c.char_type,'?'):8s} "
                  f"|{bar(c.hp, c.max_hp)}| {c.hp:4d} ATK:{c.atk}{shield_s} {dead}")
        
        my_used = my_p.remaining_actions - my_p.base_actions
        print(f"  Bonus: {'*' * my_p.bonus_actions}{'.' * (MAX_BONUS_ACTIONS - my_p.bonus_actions)}  (Actions: {my_p.base_actions} + {my_used})")
        
        # Opponent team (shields hidden — fog of war)
        print(f"  ---- AI ----")
        for i in opp_p.stack_order:
            c = opp_p.characters[i]
            mark = ">" if i == opp_p.active_char_index else " "
            dead = "[X]" if not c.is_alive() else ""
            print(f"  {mark} {EMOJI.get(c.char_type,'?')}#{i+1} {TYPE_NAMES.get(c.char_type,'?'):8s} "
                  f"|{bar(c.hp, c.max_hp)}| {c.hp:4d} ATK:{c.atk} {dead}")
        
        print(f"  Bonus: {'*' * opp_p.bonus_actions}{'.' * (MAX_BONUS_ACTIONS - opp_p.bonus_actions)}")
        
        # Matchup
        yt = my_p.active_character.char_type
        at = opp_p.active_character.char_type
        mult = get_type_multiplier(yt, at)
        dmg = int(my_p.active_character.atk * mult)
        adv = "[OK]" if mult > 1.0 else ("[XX]" if mult < 1.0 else "--")
        print(f"\n  {EMOJI.get(yt,'?')} vs {EMOJI.get(at,'?')}: x{mult} {adv}")
        print(f"  Pot dmg: ~{dmg}")
        
        hits = (opp_p.active_character.hp + dmg - 1) // max(dmg, 1)
        max_possible = min(my_p.remaining_actions, opp_p.active_character.hp // max(dmg, 1) + 1)
        if hits <= max_possible and hits > 0:
            print(f"  [KILL possible] {hits} hits needed")
        
        return my_p, opp_p
    
    def show_turn_summary(player, opponent, log, label, base_actions, bonus_actions):
        """Show only resolved actions, damage, blocks, and actual budget."""
        print()
        print("-" * 40)
        print(f"  {label} turn #{turn_num}")
        print("-" * 40)
        
        atk_count = log.attack_actions
        total_dmg = log.total_damage
        
        if log.switched:
            print(f"  {label} switched")
        
        other_label = "Ai" if label == "You" else "You"
        matchup = f"  {label} vs {other_label}"
        if atk_count > 0:
            unblocked = log.unblocked_attacks or 0
            blocked_count = log.blocked_shields
            print(matchup)
            if blocked_count == atk_count:
                print("  BLOCK!")
                print(f"  {atk_count} attacks vs {log.opponent_shields} shields")
            else:
                print(f"  {atk_count} attack{'s' if atk_count != 1 else ''} vs "
                      f"{log.opponent_shields} shields, {total_dmg} dmg")
        elif log.defend_actions > 0 or log.bonus_actions > 0:
            print(f"  {label} did not attack")
        
        if atk_count == 0 and log.defend_actions == 0 and log.bonus_actions == 0 and not log.switched:
            print(f"  {label} did nothing")
        if label == "Ai":
            print(f"  Opponent actions: {base_actions} + {bonus_actions}")
    
    # === GAME LOOP ===
    setup_turn(p1, 1)
    
    while turn_num <= 100:
        if p1.has_lost():
            return finish_battle(2 if human_first else 1)
        if p2.has_lost():
            return finish_battle(1 if human_first else 2)
        
        current = p1 if current_player == 1 else p2
        other = p2 if current_player == 1 else p1
        ai = p1_ai if current_player == 1 else p2_ai
        state_before = {
            "player": _battle_player_state(current, turn_num + 2),
            "opponent": _battle_player_state(other, turn_num + 1),
        }
        
        # Show state before human turn
        if (human_first and current is p1) or (not human_first and current is p2):
            show_full_state(current, other, turn_num, after_ai=(turn_num > 1 or not human_first))
        
        # Track AI active for switch detection
        is_ai_turn = (human_first and current is p2) or (not human_first and current is p1)
        if is_ai_turn:
            ai_active_history.append(current.active_char_index)
        else:
            ai_active_history.append(-1)  # placeholder (human turn)
        
        # Get actions from AI
        action_base = current.base_actions
        action_bonus = max(0, current.remaining_actions - current.base_actions)
        actions = ai.choose_actions(current, other, turn_num, logs, current_player)
        action_records = [_battle_action_record(action) for action in actions]
        decision_diagnostics = copy.deepcopy(getattr(ai, "last_decision_diagnostics", {}))
        switch_from = state_before["player"]["active_index"]
        switch_to = current.active_char_index
        if switch_from != switch_to:
            action_records.insert(0, {
                "type": "switch",
                "from_index": switch_from,
                "to_index": switch_to,
                "cost": ACTION_COST_SWITCH,
            })
        
        # Track AI active after choose_actions (in case it switched)
        if is_ai_turn:
            ai_active = current.active_char_index
            if len(ai_active_history) > 0 and ai_active_history[-1] != ai_active and ai_active_history[-1] != -1:
                print(f"  [AI SW] -> {EMOJI.get(current.characters[ai_active].char_type,'?')}")
            ai_active_history[-1] = ai_active
        
        # Execute turn
        log = execute_turn(current, other, actions, current_player)
        log.turn_num = turn_num
        _resolve_battle_action_records(action_records, log)
        logs.append(log)
        battle_record["turns"].append({
            "turn": turn_num,
            "player_id": current_player,
            "actor": "human" if current_player == (1 if human_first else 2) else "champion",
            "actions": action_records,
            "decision_diagnostics": decision_diagnostics if is_ai_turn else {},
            "state_before": state_before,
            "result": {
                "attack_actions": log.attack_actions,
                "defend_actions": log.defend_actions,
                "bonus_actions": log.bonus_actions,
                "switched": log.switched,
                "unblocked_attacks": log.unblocked_attacks,
                "blocked_shields": log.blocked_shields,
                "shields_before_opponent": log.opponent_shields,
                "damage": log.total_damage,
                "switch_from": switch_from if switch_from != switch_to else None,
                "switch_to": switch_to if switch_from != switch_to else None,
                "forced_switch": bool(current.forced_switch_after_death),
            },
            "state_after": {
                "player": _battle_player_state(current, turn_num + 2),
                "opponent": _battle_player_state(other, turn_num + 1),
            },
        })
        
        # Show turn result before death check
        is_ai = (human_first and current is p2) or (not human_first and current is p1)
        label = "Ai" if is_ai else "You"
        show_turn_summary(current, other, log, label, action_base, action_bonus)
        
        # Check for deaths — return WHICH player won (1=p1, 2=p2)
        if p1.has_lost():
            return finish_battle(2)  # p1 lost -> p2 won
        if p2.has_lost():
            return finish_battle(1)  # p2 lost -> p1 won
        
        # Next player
        current_player = 2 if current_player == 1 else 1
        turn_num += 1
        
        # Reset round state when both have played
        if current_player == 1:
            p1.reset_round_state()
            p2.reset_round_state()
        
        # Setup next player's turn
        next_p = p1 if current_player == 1 else p2
        setup_turn(next_p, turn_num)
    
    return finish_battle(0)  # draw


# ============================================================
# TEST
# ============================================================

def benchmark(agent, n=100):
    """Run benchmark vs standard profiles."""
    print("\nChampion vs reference profiles ({} games each):".format(n))
    opponents = [
        ("AllIn", WeightedRandomAIv2(AIProfile("AllIn", w_attack=18, w_defend=0, w_bonus=0,
                     switch_when_disadvantaged=True, w_switch=3))),
        ("Aggro", WeightedRandomAIv2(AIProfile("Aggro", w_attack=12, w_defend=1, w_bonus=0.5,
                     switch_when_disadvantaged=True, w_switch=3))),
        ("Defender", WeightedRandomAIv2(AIProfile("Defender", w_attack=3, w_defend=12, w_bonus=1,
                        switch_when_disadvantaged=True, w_switch=3))),
        ("Switcher", WeightedRandomAIv2(AIProfile("Switcher", w_attack=10, w_defend=1, w_bonus=2,
                        w_switch=8, switch_when_disadvantaged=True,
                        switch_min_hp_ratio=0.8, aggressive_after_forced_switch=True,
                        save_first_turns=1))),
        ("Gambler", WeightedRandomAIv2(AIProfile("Gambler", w_attack=5, w_defend=5, w_bonus=5,
                       w_switch=5, switch_when_disadvantaged=True,
                       switch_min_hp_ratio=0.5, randomness=1.0))),
        ("BonusBanker", WeightedRandomAIv2(AIProfile("BonusBanker", w_attack=1, w_defend=1, w_bonus=16,
                       w_switch=1, switch_when_disadvantaged=True,
                       bonus_target=4, switch_min_hp_ratio=0.3))),
        ("Counter", CounterAI()),
        ("Adaptive", AdaptiveAI()),
    ]
    for name, opp in opponents:
        w = 0
        for _ in range(n):
            if hasattr(agent, "reset_state"):
                agent.reset_state()
            if hasattr(opp, "reset_state"):
                opp.reset_state()
            t1, t2 = random_team(), random_team()
            if random.random() < 0.5:
                e = BattleEngineV2(agent, opp, t1, t2)
                r = e.run(50)
                if r["winner"] == 1: w += 1
            else:
                e = BattleEngineV2(opp, agent, t1, t2)
                r = e.run(50)
                if r["winner"] == 2: w += 1
        print(f"  vs {name:10s}: {w}%")

# ============================================================
# INTERACTIVE LOOP
# ============================================================

class HumanInputAI:
    """Minimal AI that just wraps choose_actions for keyboard input."""
    def __init__(self):
        self.name = "You"
    
    def choose_actions(self, player, opponent, turn_num, turn_logs, player_id):
        actions = []

        def matchup_preview(character):
            """Describe the real damage and hits needed after selecting a character."""
            multiplier = get_type_multiplier(
                character.char_type, opponent.active_character.char_type)
            damage = int(character.atk * multiplier)
            target_hp = opponent.active_character.hp
            hits = (target_hp + damage - 1) // max(damage, 1)
            tag = "[OK]" if multiplier > 1.0 else ("[XX]" if multiplier < 1.0 else "--")
            return multiplier, damage, hits, tag

        _, dmg, _, _ = matchup_preview(player.active_character)
        
        actions_taken = 0
        while player.remaining_actions - actions_taken > 0:
            remaining = player.remaining_actions - actions_taken
            print(f"\n  Actions: {remaining} | Your bonus: {player.bonus_actions}/{MAX_BONUS_ACTIONS}")
            cmd = input("  a=atk s=shield b=bonus sw=switch > ").strip().lower()
            
            if cmd == "q":
                raise PlayerQuit()
            if cmd == "":
                print("  Choose an action: a, s, b, or sw")
                continue
            
            if cmd == "a":
                actions.append(BattleAction("attack", player.active_char_index, opponent.active_char_index))
                actions_taken += 1
                print(f"  > Attack queued (~{dmg} to {EMOJI.get(opponent.active_character.char_type,'?')})")
            
            elif cmd == "s":
                actions.append(BattleAction("defend", player.active_char_index))
                actions_taken += 1
                print(f"  > Shield queued")
            
            elif cmd == "b":
                if player.bonus_actions >= MAX_BONUS_ACTIONS:
                    print("  [MAX] Bonus full!")
                else:
                    actions.append(BattleAction("bonus", player.active_char_index))
                    actions_taken += 1
                    print(f"  > Bonus queued")
            
            elif cmd == "sw":
                if player.switched_this_round:
                    print("  Already switched!")
                    continue
                if player.remaining_actions - actions_taken < ACTION_COST_SWITCH:
                    print("  Not enough actions!")
                    continue
                print("  Targets:")
                for i in player.stack_order:
                    c = player.characters[i]
                    if c.is_alive() and i != player.active_char_index:
                        multiplier, target_dmg, hits, tag = matchup_preview(c)
                        print(f"    [{i+1}] {EMOJI.get(c.char_type,'?')} {TYPE_NAMES.get(c.char_type,'?')} "
                              f"HP:{c.hp} {tag} x{multiplier} "
                              f"dmg:{target_dmg}/hit hits:{hits}")
                try:
                    ch = int(input("  Number (0=cancel): ").strip())
                    if 1 <= ch <= 3:
                        target = ch - 1
                        if player.switch_character(target):
                            print(f"  > Switched to {EMOJI.get(player.active_character.char_type,'?')}")
                            multiplier, dmg, hits, tag = matchup_preview(player.active_character)
                            print(f"  > Matchup: x{multiplier} {tag}, "
                                  f"Pot dmg: ~{dmg}, hits to defeat active: {hits}")
                        else:
                            print("  Switch failed!")
                except ValueError:
                    print("  Cancel")
            else:
                print("  Choose an action: a, s, b, or sw")
        
        return actions


BATTLE_LOG_FILE = os.path.join(_artifacts_dir, 'battle_logs.jsonl')
def _battle_player_state(player, next_turn_num=None):
    """Return a compact, model-friendly snapshot of one player."""
    state = {
        "active_index": player.active_char_index,
        "stack_order": list(player.stack_order),
        "shields": player.shields,
        "bonus_actions": player.bonus_actions,
        "base_actions": player.base_actions,
        "remaining_actions": player.remaining_actions,
        "characters": [
            {"index": i, "type": character.char_type.value,
             "hp": character.hp, "max_hp": character.max_hp,
             "atk": character.atk, "alive": character.is_alive()}
            for i, character in enumerate(player.characters)
        ],
    }
    if next_turn_num is not None:
        next_base = min(TURN_ACTIONS.get(next_turn_num, MAX_TOTAL_ACTIONS), MAX_TOTAL_ACTIONS)
        state["next_base_actions"] = next_base
        state["next_total_actions"] = min(next_base + player.bonus_actions, MAX_TOTAL_ACTIONS)
    return state


def _battle_action_record(action):
    """Serialize the decision without depending on dataclass internals."""
    return {
        "type": action.action_type,
        "attacker_index": action.attacker_idx,
        "target_index": action.target_idx,
        "damage": action.damage,
        "blocked": action.blocked,
    }


def _resolve_battle_action_records(action_records, log):
    """Attach resolved exchange facts to serialized attack decisions.

    The interactive engine resolves attacks as one aggregate exchange, so the
    old logger left every action at damage=0.  Preserve the aggregate result
    while distributing blocked status and rounded damage deterministically.
    """
    attacks = [item for item in action_records if item.get("type") == "attack"]
    blocked = max(0, min(len(attacks), int(log.blocked_shields)))
    unblocked = len(attacks) - blocked
    for index, item in enumerate(attacks):
        item["blocked"] = index < blocked
        item["damage"] = 0
    if unblocked and log.total_damage:
        base, remainder = divmod(int(log.total_damage), unblocked)
        damage_index = 0
        for item in attacks:
            if not item["blocked"]:
                item["damage"] = base + (1 if damage_index < remainder else 0)
                damage_index += 1


def validate_battle_record(record):
    """Return invariant violations; accept both schema 1 and schema 2 logs."""
    errors = []
    if not isinstance(record, dict):
        return ["record is not an object"]
    if record.get("schema_version") not in (1, 2):
        errors.append("unsupported schema_version")
    turns = record.get("turns")
    if not isinstance(turns, list):
        return errors + ["turns is not a list"]
    for number, turn in enumerate(turns, 1):
        if turn.get("turn") != number and turn.get("turn") != number * 2 - 1:
            errors.append(f"turn {number}: unexpected turn number")
        before = turn.get("state_before", {})
        after = turn.get("state_after", {})
        for label, state in (("before", before), ("after", after)):
            for side in ("player", "opponent"):
                snapshot = state.get(side, {})
                chars = snapshot.get("characters", [])
                active = snapshot.get("active_index")
                if chars and active is not None and 0 <= active < len(chars):
                    if not chars[active].get("alive", chars[active].get("hp", 0) > 0):
                        errors.append(f"turn {number}: {label}.{side} active character is dead")
        result = turn.get("result", {})
        attacks = [a for a in turn.get("actions", []) if a.get("type") == "attack"]
        if result.get("attack_actions") != len(attacks):
            errors.append(f"turn {number}: attack count mismatch")
    return errors


def _append_battle_log(path, record):
    """Append one complete battle as JSONL for replay/training pipelines."""
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
        stream.write("\n")

def play_vs_champion():
    """Play interactive games vs the evolved champion."""
    champion = load_champion()
    benchmark(champion, 100)
    
    input("\nPress Enter for battle...")
    
    game_num = 0
    while True:
        game_num += 1
        team_you = random_team()
        team_ai = random_team()
        
        print(f"\n{'='*55}")
        print(f"  BATTLE #{game_num}")
        print(f"{'='*55}")
        print(f"You: {' '.join(EMOJI.get(t,'?') for t in team_you)}  [{','.join(t.value for t in team_you)}]")
        print(f"AI:  {' '.join(EMOJI.get(t,'?') for t in team_ai)}  [{','.join(t.value for t in team_ai)}]")
        
        human_first = random.random() < 0.5
        print(f"\n  {'You go FIRST!' if human_first else 'AI goes FIRST!'}")
        
        if human_first:
            team_view = team_you
        else:
            team_view = team_ai
        
        print()
        for i, t in enumerate(team_view):
            mark = ">" if i == 0 else " "
            print(f"  {mark} {EMOJI.get(t,'?')} {TYPE_NAMES.get(t,'?'):10s} |{'#'*12}|  HP/ATK ?")
        print(f"  Bonus: ....")
        print()
        print(f"  ---- OPPONENT ----")
        opp_team = team_ai if human_first else team_you
        for i, t in enumerate(opp_team):
            print(f"  {EMOJI.get(t,'?')} {TYPE_NAMES.get(t,'?'):10s} |{'#'*12}|  HP/ATK ?")
        
        try:
            human = HumanInputAI()
            winner = run_game(human, champion, team_you, team_ai, human_first)
        except PlayerQuit:
            print("\nBye!")
            break
        
        if winner == 0:
            print(f"\n  [DRAW] Turn limit")
        elif (winner == 1 and human_first) or (winner == 2 and not human_first):
            print(f"\n  [WIN]  YOU WIN!")
        else:
            print(f"\n  [LOSE] You lose")
        
        again = input("\nAnother? (Enter=yes, q=quit): ").strip().lower()
        if again == "q":
            break
    
    print("\nThanks for playing!")


# ============================================================
# PLAY VS ANCHORS
# ============================================================

ANCHOR_PROFILES = [
    ("AllIn",    AIProfile("AllIn",    w_attack=18, w_defend=0, w_bonus=0,
                    switch_when_disadvantaged=True, w_switch=3)),
    ("Aggro",    AIProfile("Aggro",    w_attack=12, w_defend=1, w_bonus=0.5,
                    switch_when_disadvantaged=True, w_switch=3)),
    ("Defender", AIProfile("Defender", w_attack=3, w_defend=12, w_bonus=1,
                    switch_when_disadvantaged=True, w_switch=3)),
    ("Switcher", AIProfile("Switcher", w_attack=10, w_defend=1, w_bonus=2,
                    w_switch=8, switch_when_disadvantaged=True,
                    switch_min_hp_ratio=0.8, aggressive_after_forced_switch=True,
                    save_first_turns=1)),
    ("Gambler",  AIProfile("Gambler",  w_attack=5, w_defend=5, w_bonus=5,
                    w_switch=5, switch_when_disadvantaged=True,
                    switch_min_hp_ratio=0.5, randomness=1.0)),
    ("BonusBanker", AIProfile("BonusBanker", w_attack=1, w_defend=1, w_bonus=16,
                    w_switch=1, switch_when_disadvantaged=True,
                    bonus_target=4, switch_min_hp_ratio=0.3)),
]

def load_play_stats():
    if not os.path.exists(STATS_FILE):
        return []
    try:
        with open(STATS_FILE) as f:
            return json.load(f)
    except:
        return []

def save_play_stats(session):
    stats = load_play_stats()
    stats.append(session)
    with open(STATS_FILE, 'w') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

def show_play_stats():
    stats = load_play_stats()
    if not stats:
        print("  (no stats yet)")
        return
    print(f"\n  {'Date':16s}  {'Anchor':10s}  {'W':3s}  {'L':3s}  {'Win%':6s}")
    print(f"  {'-'*16}  {'-'*10}  {'-'*3}  {'-'*3}  {'-'*6}")
    for s in stats[-20:]:  # last 20 sessions
        for a in s.get('anchors', []):
            pct = a['won'] / max(1, a['won'] + a['lost']) * 100
            print(f"  {s['timestamp']:16s}  {a['name']:10s}  {a['won']:3d}  {a['lost']:3d}  {pct:5.0f}%")
        tot_w = sum(a['won'] for a in s.get('anchors', []))
        tot_l = sum(a['lost'] for a in s.get('anchors', []))
        tot_pct = tot_w / max(1, tot_w + tot_l) * 100
        print(f"  {'':16s}  {'TOTAL':10s}  {tot_w:3d}  {tot_l:3d}  {tot_pct:5.0f}%")
        print()

def play_vs_anchors():
    """Play interactive games vs each anchor profile."""
    anchors = ANCHOR_PROFILES + [("Counter", CounterAI()), ("Adaptive", AdaptiveAI())]
    
    print(f"\n{'='*55}")
    print(f"  PLAY VS ANCHORS")
    print(f"{'='*55}")
    print(f"  Anchors: {', '.join(n for n, _ in anchors)}")
    
    n_str = input(f"\nGames per anchor (default=5): ").strip()
    n = int(n_str) if n_str.isdigit() else 5
    
    results = {name: {"won": 0, "lost": 0} for name, _ in anchors}
    total_won, total_lost = 0, 0
    session_id = 0
    
    for name, anchor in anchors:
        print(f"\n  --- vs {name} ---")
        for g in range(n):
            session_id += 1
            you_first = random.random() < 0.5
            team_you = random_team()
            team_ai = random_team()
            
            print(f"\n{'='*55}")
            print(f"  Game #{session_id}: vs {name}  ({g+1}/{n})")
            print(f"{'='*55}")
            print(f"You: {' '.join(EMOJI.get(t,'?') for t in team_you)}  [{','.join(t.value for t in team_you)}]")
            print(f"AI:  {' '.join(EMOJI.get(t,'?') for t in team_ai)}  [{','.join(t.value for t in team_ai)}]")
            
            print(f"\n  {'You go FIRST!' if you_first else 'AI goes FIRST!'}")
            
            try:
                human = HumanInputAI()
                winner = run_game(human, anchor, team_you, team_ai, you_first)
            except PlayerQuit:
                print("\n  Skipping rest of this anchor...")
                break
            
            if winner == 0:
                print(f"\n  [DRAW]")
            elif (winner == 1 and you_first) or (winner == 2 and not you_first):
                print(f"\n  [WIN]  YOU WIN!")
                results[name]["won"] += 1
                total_won += 1
            else:
                print(f"\n  [LOSE] You lose")
                results[name]["lost"] += 1
                total_lost += 1
        
        print(f"\n  --- {name}: {results[name]['won']}W / {results[name]['lost']}L ---")
    
    print(f"\n{'='*55}")
    print(f"  FINAL RESULTS")
    print(f"{'='*55}")
    print(f"\n  {'Anchor':12s}  {'W':4s}  {'L':4s}  {'Win%':7s}")
    print(f"  {'-'*12}  {'-'*4}  {'-'*4}  {'-'*7}")
    for name, _ in anchors:
        w = results[name]["won"]
        l = results[name]["lost"]
        pct = w / max(1, w + l) * 100
        print(f"  {name:12s}  {w:4d}  {l:4d}  {pct:6.0f}%")
    print(f"  {'-'*12}  {'-'*4}  {'-'*4}  {'-'*7}")
    total_pct = total_won / max(1, total_won + total_lost) * 100
    print(f"  {'TOTAL':12s}  {total_won:4d}  {total_lost:4d}  {total_pct:6.0f}%")
    
    # Save stats
    session = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "games_per_anchor": n,
        "total_won": total_won,
        "total_lost": total_lost,
        "anchors": [{"name": name, "won": results[name]["won"], "lost": results[name]["lost"]}
                    for name, _ in anchors]
    }
    save_play_stats(session)
    print(f"\nStats saved to play_stats.json")
    
    again = input("\nPlay again? (y/n): ").strip().lower()
    if again.startswith("y"):
        play_vs_anchors()


def main():
    print("\n" + "=" * 55)
    print("  COTE MEGAVERSE")
    print("=" * 55)
    print("  1. Play vs Champion (evolved AI)")
    print("  2. Play vs Anchors (archetype gauntlet)")
    print("  3. View stats history")
    
    choice = input("\nChoose (1/2/3, Enter=1): ").strip()
    
    if choice == "2":
        play_vs_anchors()
    elif choice == "3":
        show_play_stats()
    else:
        play_vs_champion()

if __name__ == "__main__":
    main()
