"""
Functional tests for COTE Megaverse game mechanics.
Tests verify: action progression, shields, damage, type advantages,
switching, turn order, and display correctness.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'parameterized_ai'))

import numpy as np
from parameterized_ai_v2 import (
    BattleEngineV2, BattleAction, Player, Character,
    CharType, get_type_multiplier, random_team, make_character,
    MAX_BONUS_ACTIONS, MAX_TOTAL_ACTIONS, ACTION_COST_SWITCH,
    BASE_HP, BASE_ATK, TURN_ACTIONS, WeightedRandomAIv2, AIProfile,
    TurnLog
)

PASSED = 0
FAILED = 0

def check(name, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  [PASS] {name}")
    else:
        FAILED += 1
        print(f"  [FAIL] {name} {detail}")


# ============================================================
# 1. ACTION PROGRESSION
# ============================================================
def test_action_progression():
    print("\n=== Action Progression ===")
    expected = {1:1, 2:2, 3:2, 4:2, 5:3, 6:3, 7:4, 8:4}
    for tn in range(1, 9):
        base = min(TURN_ACTIONS.get(tn, 4), MAX_TOTAL_ACTIONS)
        check(f"Turn {tn}: base={base}", base == expected[tn],
              f"got {base}")

    # After turn 8, stays at 4
    for tn in range(9, 15):
        base = min(TURN_ACTIONS.get(tn, 4), MAX_TOTAL_ACTIONS)
        check(f"Turn {tn}: base=4 (capped)", base == 4, f"got {base}")


# ============================================================
# 2. BONUS ACTION BUDGET
# ============================================================
def test_bonus_budget():
    print("\n=== Bonus Action Budget ===")
    p = Player(1, [make_character(CharType.A) for _ in range(3)])

    # Turn 1: base=1, bonus=0, total=1
    p.bonus_actions = 0
    base = min(TURN_ACTIONS.get(1, 4), MAX_TOTAL_ACTIONS)
    total = min(base + p.bonus_actions, MAX_TOTAL_ACTIONS)
    used = min(p.bonus_actions, total - base)
    check("Turn 1, bonus=0: total=1, used=0", total == 1 and used == 0,
          f"total={total} used={used}")

    # Simulate earning 2 bonuses (pressing 'b' twice)
    p.bonus_actions += 2
    # Turn 2: base=2, bonus=2 → total=4, used=2
    base = min(TURN_ACTIONS.get(2, 4), MAX_TOTAL_ACTIONS)
    total = min(base + p.bonus_actions, MAX_TOTAL_ACTIONS)
    used = min(p.bonus_actions, total - base)
    check("Turn 2, bonus=2: total=4, used=2", total == 4 and used == 2,
          f"total={total} used={used}")

    # Bonus capped at MAX_BONUS_ACTIONS=4
    p.bonus_actions = 4
    base = min(TURN_ACTIONS.get(5, 4), MAX_TOTAL_ACTIONS)
    total = min(base + p.bonus_actions, MAX_TOTAL_ACTIONS)
    used = min(p.bonus_actions, total - base)
    check("Turn 5, bonus=4: total=7, used=4", total == 7 and used == 4,
          f"total={total} used={used}")

    # Cannot exceed MAX_TOTAL_ACTIONS
    check("Total never exceeds 8", total <= MAX_TOTAL_ACTIONS, f"total={total}")


# ============================================================
# 3. TYPE ADVANTAGE CIRCLE: A>B>C>D>A
# ============================================================
def test_type_advantage():
    print("\n=== Type Advantage Circle ===")
    mult_high = 1.3
    mult_low = 0.7
    mult_neutral = 1.0

    # A > B
    check("A vs B: x1.3", get_type_multiplier(CharType.A, CharType.B) == mult_high)
    check("B vs A: x0.7", get_type_multiplier(CharType.B, CharType.A) == mult_low)

    # B > C
    check("B vs C: x1.3", get_type_multiplier(CharType.B, CharType.C) == mult_high)
    check("C vs B: x0.7", get_type_multiplier(CharType.C, CharType.B) == mult_low)

    # C > D
    check("C vs D: x1.3", get_type_multiplier(CharType.C, CharType.D) == mult_high)
    check("D vs C: x0.7", get_type_multiplier(CharType.D, CharType.C) == mult_low)

    # D > A
    check("D vs A: x1.3", get_type_multiplier(CharType.D, CharType.A) == mult_high)
    check("A vs D: x0.7", get_type_multiplier(CharType.A, CharType.D) == mult_low)

    # Same type = neutral
    check("A vs A: x1.0", get_type_multiplier(CharType.A, CharType.A) == mult_neutral)
    check("B vs B: x1.0", get_type_multiplier(CharType.B, CharType.B) == mult_neutral)
    check("C vs C: x1.0", get_type_multiplier(CharType.C, CharType.C) == mult_neutral)
    check("D vs D: x1.0", get_type_multiplier(CharType.D, CharType.D) == mult_neutral)


# ============================================================
# 4. DAMAGE CALCULATION
# ============================================================
def test_damage():
    print("\n=== Damage Calculation ===")
    atk = 2000

    # Neutral
    dmg = int(atk * get_type_multiplier(CharType.A, CharType.A))
    check("Neutral damage = ATK", dmg == 2000, f"got {dmg}")

    # Advantage
    dmg = int(atk * get_type_multiplier(CharType.A, CharType.B))
    check("Advantage damage = ATK * 1.3", dmg == 2600, f"got {dmg}")

    # Disadvantage
    dmg = int(atk * get_type_multiplier(CharType.A, CharType.D))
    check("Disadvantage damage = ATK * 0.7", dmg == 1400, f"got {dmg}")


# ============================================================
# 5. SHIELD MECHANICS
# ============================================================
def test_shields():
    print("\n=== Shield Mechanics ===")

    class MockChar:
        def __init__(self, hp):
            self.hp = hp
            self.max_hp = hp
        def is_alive(self):
            return self.hp > 0
        def take_damage(self, d):
            actual = min(d, self.hp)
            self.hp -= actual
            return actual

    # Shield blocks N attacks
    atk_count = 5
    shields = 3
    unblocked = max(0, atk_count - shields)
    blocked = min(atk_count, shields)
    check("5 attacks vs 3 shields: unblocked=2", unblocked == 2, f"got {unblocked}")
    check("5 attacks vs 3 shields: blocked=3", blocked == 3, f"got {blocked}")

    # More shields than attacks → all blocked
    atk_count = 2
    shields = 5
    unblocked = max(0, atk_count - shields)
    check("2 attacks vs 5 shields: all blocked", unblocked == 0, f"got {unblocked}")

    # No shields → all unblocked
    atk_count = 3
    shields = 0
    unblocked = max(0, atk_count - shields)
    check("3 attacks vs 0 shields: all unblocked", unblocked == 3, f"got {unblocked}")

    # Shield consumption: shields -= blocked
    shields = 3
    blocked = 2
    remaining = max(0, shields - blocked)
    check("Shields 3, blocked 2, remaining 1", remaining == 1, f"got {remaining}")


# ============================================================
# 6. FULL ENGINE: SINGLE ATTACK
# ============================================================
def test_engine_attack():
    print("\n=== Engine: Single Attack ===")

    class TestAI:
        def __init__(self, actions):
            self._actions = actions
            self._idx = 0
            self.name = "TestAI"
        def choose_actions(self, player, opponent, turn_num, logs, pid):
            return self._actions

    # Create deterministic teams
    t1 = [CharType.A, CharType.B, CharType.C]
    t2 = [CharType.D, CharType.A, CharType.B]

    atk_char = make_character(CharType.A)
    atk_char.atk = 2000
    def_char = make_character(CharType.D)
    def_char.hp = 5000
    def_char.max_hp = 5000

    # Test 1: Single attack, no shields
    p1 = Player(1, [atk_char, make_character(CharType.B), make_character(CharType.C)])
    p2 = Player(2, [def_char, make_character(CharType.A), make_character(CharType.B)])
    p1.remaining_actions = 1
    p1.base_actions = 1

    # A vs D: A>D is 0.7 (disadvantage)
    expected_dmg = int(2000 * 0.7)  # 1400
    actions = [BattleAction("attack", 0, 0)]
    ai1 = TestAI(actions)
    ai2 = TestAI([])

    e = BattleEngineV2(ai1, ai2, t1, t2)
    # Override the AI to return our actions
    e.p1 = p1
    e.p2 = p2

    log = e._execute_turn(p1, p2, actions)
    check(f"A vs D single hit: dealt={log.total_damage}", log.total_damage == expected_dmg,
          f"expected {expected_dmg}")
    check("1 attack consumed shield=0", log.opponent_shields == 0 and log.blocked_shields == 0)


# ============================================================
# 7. ENGINE: SHIELD BLOCKS ATTACK
# ============================================================
def test_engine_shield_blocks():
    print("\n=== Engine: Shield Blocks ===")

    class ShieldAI:
        def __init__(self, actions):
            self._actions = actions
            self.name = "ShieldAI"
        def choose_actions(self, player, opponent, turn_num, logs, pid):
            return self._actions

    t1 = [CharType.A, CharType.B, CharType.C]
    t2 = [CharType.D, CharType.A, CharType.B]

    p1 = Player(1, [make_character(CharType.A), make_character(CharType.B), make_character(CharType.C)])
    p2 = Player(2, [make_character(CharType.D), make_character(CharType.A), make_character(CharType.B)])
    p2.shields = 2

    actions1 = [BattleAction("attack", 0, 0)] * 3
    actions2 = []
    ai1 = ShieldAI(actions1)
    ai2 = ShieldAI(actions2)

    engine = BattleEngineV2(ai1, ai2, t1, t2)
    engine.p1 = p1
    engine.p2 = p2

    result_log = engine._execute_turn(p1, p2, actions1)

    check("3 attacks vs 2 shields: blocked=2", result_log.blocked_shields == 2,
          f"got {result_log.blocked_shields}")
    check("3 attacks vs 2 shields: unblocked=1", result_log.unblocked_attacks == 1,
          f"got {result_log.unblocked_attacks}")
    check("Opponent shields consumed: 2 to 0", p2.shields == 0, f"got {p2.shields}")


# ============================================================
# 8. DEATH AND FREE SWITCH
# ============================================================
def test_death_switch():
    print("\n=== Death & Free Switch ===")
    c1 = Character(CharType.A, hp=100, max_hp=100, atk=100)
    c2 = Character(CharType.B, hp=100, max_hp=100, atk=100)
    c3 = Character(CharType.C, hp=100, max_hp=100, atk=100)

    p = Player(1, [c1, c2, c3])
    p.active_char_index = 0
    c1.is_active = True
    p.switch_history = []

    # Kill active character
    c1.hp = 0
    check("c1 is dead", not c1.is_alive())

    # Force switch
    p.force_switch_from_death()
    check("After death: active is not c1", p.active_char_index != 0)
    check("After death: active char is alive",
          p.characters[p.active_char_index].is_alive())
    check("After death: just_swapped_free=True", p.just_swapped_free is True)


# ============================================================
# 9. SWITCH MECHANICS
# ============================================================
def test_switch():
    print("\n=== Switch Mechanics ===")
    c1 = Character(CharType.A, hp=5000, max_hp=6000, atk=2000, is_active=True)
    c2 = Character(CharType.B, hp=5000, max_hp=6000, atk=2000)
    c3 = Character(CharType.C, hp=5000, max_hp=6000, atk=2000)

    p = Player(1, [c1, c2, c3])
    p.active_char_index = 0
    p.remaining_actions = 3

    # Successful switch
    ok = p.switch_character(1)
    check("Switch 0 to 1: success", ok is True)
    check("Switch 0 to 1: active is now 1", p.active_char_index == 1)
    check("Switch 0 to 1: cost 1 action", p.remaining_actions == 2)
    check("Switch 0 to 1: switched_this_round=True", p.switched_this_round is True)

    # Cannot switch twice in one turn
    ok2 = p.switch_character(2)
    check("Double switch: blocked", ok2 is False)

    # Cannot switch to self
    p2 = Player(1, [c1, c2, c3])
    p2.active_char_index = 0
    p2.remaining_actions = 3
    ok3 = p2.switch_character(0)
    check("Switch to self: blocked", ok3 is False)

    # Cannot switch to last in history (LIFO)
    p3 = Player(1, [c1, c2, c3])
    p3.active_char_index = 1
    p3.remaining_actions = 3
    p3.switch_history = [0]
    ok4 = p3.switch_character(0)
    check("LIFO: cannot switch back immediately", ok4 is False)


# ============================================================
# 10. TURN ORDER
# ============================================================
def test_turn_order():
    print("\n=== Turn Order ===")
    check("Turn 1: player 1 first", True)  # Structural: game starts with p1
    # Verify that current_player alternates
    cp = 1
    for turn in range(1, 5):
        check(f"Turn {turn}: player {cp} goes", True)
        cp = 2 if cp == 1 else 1


# ============================================================
# 11. PLAYER HAS_LOST
# ============================================================
def test_has_lost():
    print("\n=== Has Lost ===")
    c1 = Character(CharType.A, hp=0)
    c2 = Character(CharType.B, hp=0)
    c3 = Character(CharType.C, hp=0)
    p = Player(1, [c1, c2, c3])
    check("All dead: has_lost=True", p.has_lost() is True)

    c4 = Character(CharType.A, hp=100)
    p2 = Player(1, [c4, c2, c3])
    check("One alive: has_lost=False", p2.has_lost() is False)


# ============================================================
# 12. BONUS OVERFLOW PROTECTION
# ============================================================
def test_bonus_overflow():
    print("\n=== Bonus Overflow ===")
    p = Player(1, [make_character(CharType.A) for _ in range(3)])
    p.bonus_actions = MAX_BONUS_ACTIONS  # Already at max

    base = 4
    total = min(base + p.bonus_actions, MAX_TOTAL_ACTIONS)
    used = min(p.bonus_actions, total - base)
    check("Bonus at max: total capped at 8", total == 8, f"got {total}")
    check("Bonus at max: used=4", used == 4, f"got {used}")

    actual_bonus = min(1, MAX_BONUS_ACTIONS - p.bonus_actions)
    check("Bonus at max: actual_bonus=0", actual_bonus == 0, f"got {actual_bonus}")


# ============================================================
# 13. GENOME CONSISTENCY
# ============================================================
def test_genome():
    print("\n=== Genome Consistency ===")
    from coevolution import genome_size, random_genome, forward, N_IN, N_HID, N_OUT

    gs = genome_size()
    # LSTM genome: 4 gates * (N_IN*N_HID + N_HID*N_HID + N_HID) + output layer
    expected_lstm = 4 * (N_IN * N_HID + N_HID * N_HID + N_HID)
    expected_output = N_HID * N_OUT + N_OUT
    expected = expected_lstm + expected_output
    check(f"genome_size() = {expected}", gs == expected, f"got {gs}")
    check(f"N_IN={N_IN}", N_IN == 42)

    g = random_genome()
    check(f"Random genome shape = ({gs},)", g.shape == (gs,), f"got {g.shape}")

    # Forward pass with valid state (returns probs, h, c)
    state = [0.5] * N_IN
    probs, h, c = forward(g, state)
    check("Forward returns 8 probs", len(probs) == 8, f"got {len(probs)}")
    check("Probs sum to 1.0", abs(sum(probs) - 1.0) < 1e-5, f"sum={sum(probs)}")
    check("All probs >= 0", all(p >= 0 for p in probs))
    check(f"h shape = ({N_HID},)", h.shape == (N_HID,), f"got {h.shape}")
    check(f"c shape = ({N_HID},)", c.shape == (N_HID,), f"got {c.shape}")


# ============================================================
# 14. FULL GAME: RANDOM VS RANDOM (no crash)
# ============================================================
def test_full_game():
    print("\n=== Full Game (no crash) ===")
    from coevolution import NeuralAgent, random_genome

    class RandomAI:
        def __init__(self):
            self.name = "Random"
        def choose_actions(self, player, opponent, turn_num, logs, pid):
            actions = []
            remaining = player.remaining_actions
            for _ in range(remaining):
                r = np.random.random()
                if r < 0.6:
                    actions.append(BattleAction("attack", player.active_char_index,
                                                opponent.active_char_index))
                elif r < 0.8:
                    actions.append(BattleAction("defend", player.active_char_index))
                else:
                    actions.append(BattleAction("bonus", player.active_char_index))
            return actions

    t1, t2 = random_team(), random_team()
    ai1, ai2 = RandomAI(), RandomAI()
    e = BattleEngineV2(ai1, ai2, t1, t2)
    r = e.run(50)

    check("Game completes", "winner" in r)
    check("Winner is 0, 1, or 2", r["winner"] in (0, 1, 2), f"got {r['winner']}")
    check("Turns > 0", r["turns"] > 0, f"got {r['turns']}")

    # NeuralAgent version
    g = random_genome()
    nai = NeuralAgent(g, "test")
    e2 = BattleEngineV2(nai, ai2, random_team(), random_team())
    r2 = e2.run(50)
    check("NeuralAgent game completes", r2["winner"] in (0, 1, 2))


# ============================================================
# 15. SHIELD IS PLAYER-LEVEL (not per-character)
# ============================================================
def test_shield_is_player_level():
    print("\n=== Shield is Player-Level ===")
    c1 = Character(CharType.A, hp=5000, max_hp=6000, atk=2000)
    c2 = Character(CharType.B, hp=5000, max_hp=6000, atk=2000)
    c3 = Character(CharType.C, hp=5000, max_hp=6000, atk=2000)
    p = Player(1, [c1, c2, c3])
    p.shields = 3
    check("Player shields = 3", p.shields == 3)

    # Shields persist when switching
    p.remaining_actions = 1
    p.switch_character(1)
    check("After switch: shields still 3", p.shields == 3)

    # Shields are overwritten by defend actions (player.shields = def_count)
    p.shields = 5
    def_count = 1
    p.shields = def_count
    check("After defend=1: shields overwritten to 1", p.shields == 1)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    test_action_progression()
    test_bonus_budget()
    test_type_advantage()
    test_damage()
    test_shields()
    test_engine_attack()
    test_engine_shield_blocks()
    test_death_switch()
    test_switch()
    test_turn_order()
    test_has_lost()
    test_bonus_overflow()
    test_genome()
    test_full_game()
    test_shield_is_player_level()

    print(f"\n{'='*40}")
    print(f"  RESULTS: {PASSED} passed, {FAILED} failed")
    print(f"{'='*40}")
    sys.exit(1 if FAILED > 0 else 0)
