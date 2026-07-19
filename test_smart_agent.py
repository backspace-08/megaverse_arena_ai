"""Regression tests for the adaptive smart agent and tactical planner."""
import sys
import os
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "parameterized_ai"))

from coevolution import OpponentModel, PhaseShiftAI, SmartNeuralAgent, random_smart_genome
from parameterized_ai_v2 import Character, CharType, Player, TurnLog


def _log(turn, player_id, attacks=0, defends=0, bonuses=0):
    return TurnLog(
        turn_num=turn,
        player_id=player_id,
        attack_actions=attacks,
        defend_actions=defends,
        bonus_actions=bonuses,
        switched=False,
        unblocked_attacks=attacks,
        total_damage=0,
        opponent_shields=0,
        blocked_shields=0,
        player_shields_before=0,
        p1_hp=[6000],
        p2_hp=[6000],
    )


class TestOpponentModel(unittest.TestCase):
    def test_same_log_is_processed_once(self):
        model = OpponentModel(adapt_speed=0.5)
        history = [_log(2, 2, attacks=4)]
        model.update(history, player_id=1)
        first = (model.atk_p, model.def_p, model.bon_p)
        model.update(history, player_id=1)
        self.assertEqual(first, (model.atk_p, model.def_p, model.bon_p))

    def test_new_logs_update_and_track_shields(self):
        model = OpponentModel(adapt_speed=0.5)
        model.update([_log(2, 2, defends=3, bonuses=1)], player_id=1)
        self.assertEqual(model.est_current_shields, 3)
        self.assertEqual(model.est_bonus_bank, 1)
        self.assertGreater(model.def_p, model.atk_p)

    def test_bonus_burst_raises_risk(self):
        model = OpponentModel(adapt_speed=0.5)
        model.update([_log(2, 2, bonuses=4)], player_id=1)
        self.assertGreaterEqual(model.burst_risk, 0.8)

    def test_style_shift_is_detected_from_new_turn(self):
        model = OpponentModel(adapt_speed=0.5)
        logs = [_log(2, 2, attacks=4)]
        model.update(logs, player_id=1)
        attack_probability = model.atk_p

        logs.append(_log(4, 2, defends=4))
        model.update(logs, player_id=1)
        self.assertLess(model.atk_p, attack_probability)
        self.assertGreater(model.def_p, model.atk_p)
        self.assertTrue(model.opponent_job_changed)

    def test_phase_shift_opponent_changes_phase(self):
        opponent = PhaseShiftAI()
        player = Player(2, [Character(CharType.A)])
        other = Player(1, [Character(CharType.B)])
        player.remaining_actions = 3
        early = opponent.choose_actions(player, other, 1, [], 2)
        later_logs = [_log(2, 2, bonuses=1), _log(4, 2, defends=2)]
        player.remaining_actions = 3
        late = opponent.choose_actions(player, other, 5, later_logs, 2)
        self.assertNotEqual(
            [a.action_type for a in early], [a.action_type for a in late])


class TestTacticalPlanner(unittest.TestCase):
    def setUp(self):
        self.agent = SmartNeuralAgent(random_smart_genome(seed=7))
        self.player = Player(1, [Character(CharType.A)])
        self.opponent = Player(2, [Character(CharType.B)])
        self.player.remaining_actions = 4

    def test_enumerates_all_legal_allocations(self):
        candidates = self.agent._gen_candidates(4, self.player)
        self.assertEqual(len(candidates), 15)
        self.assertIn((4, 0, 0), candidates)
        self.assertIn((0, 4, 0), candidates)
        self.assertIn((0, 0, 4), candidates)
        self.assertTrue(all(sum(c) == 4 for c in candidates))

    def test_current_shields_reduce_current_damage(self):
        self.opponent.active_character.hp = 12000
        no_shields = self.agent._eval_net(
            4, 0, 0, 0, 0, 2000, 2000, self.player, self.opponent)
        four_shields = self.agent._eval_net(
            4, 0, 0, 4, 0, 2000, 2000, self.player, self.opponent)
        self.assertGreater(no_shields, four_shields)

    def test_own_shields_reduce_future_damage(self):
        self.player.active_character.hp = 6000
        no_defense = self.agent._eval_net(
            0, 0, 0, 0, 6, 2000, 2000, self.player, self.opponent)
        four_defense = self.agent._eval_net(
            0, 4, 0, 0, 6, 2000, 2000, self.player, self.opponent)
        self.assertGreater(four_defense, no_defense)

    def test_next_turn_budget_does_not_use_zero_remaining_actions(self):
        self.opponent.remaining_actions = 0
        scenarios = self.agent._opponent_scenarios(7, self.agent.opp_model)
        self.assertGreater(max(attacks for _, _, attacks in scenarios), 0)

    def test_planner_uses_defense_when_attack_is_already_blocked(self):
        """Do not attack into four known shields when a burst is incoming."""
        self.opponent.active_character.hp = 12000
        self.player.active_character.hp = 6000
        self.agent.opp_model.est_current_shields = 4
        self.agent.opp_model.est_bonus_bank = 4
        self.agent.opp_model.atk_p = 0.8
        self.agent.opp_model.def_p = 0.1
        self.agent.opp_model.bon_p = 0.1
        self.agent.opp_model.burst_risk = 0.9
        weights = self.agent._compute_weights(
            self.player, self.opponent, 7, self.agent.opp_model)
        choice = self.agent._expectimax_best(
            4, self.player, self.opponent, 7, *weights, self.agent.opp_model)
        self.assertGreaterEqual(choice[1], 3)

    def test_planner_banks_against_shield_heavy_opponent(self):
        """Repeated defense is inferior to preparing a burst into a defender."""
        self.opponent.active_character.hp = 12000
        self.agent.opp_model.est_current_shields = 4
        self.agent.opp_model.atk_p = 0.08
        self.agent.opp_model.def_p = 0.82
        self.agent.opp_model.bon_p = 0.10
        self.agent.opp_model.burst_risk = 0.0
        weights = self.agent._compute_weights(
            self.player, self.opponent, 7, self.agent.opp_model)
        choice = self.agent._expectimax_best(
            4, self.player, self.opponent, 7, *weights, self.agent.opp_model)
        self.assertGreaterEqual(choice[2], 2)

    def test_breakthrough_rule_attacks_through_shield_wall(self):
        self.player.remaining_actions = 8
        self.agent.opp_model.est_current_shields = 4
        self.agent.opp_model.atk_p = 0.08
        self.agent.opp_model.def_p = 0.82
        self.agent.opp_model.bon_p = 0.10
        self.agent.opp_model.burst_risk = 0.0
        actions = self.agent.choose_actions(
            self.player, self.opponent, 7, [_log(6, 2, defends=4)], 1)
        self.assertGreaterEqual(sum(a.action_type == "attack" for a in actions), 5)

    def test_banks_when_shield_wall_blocks_entire_turn(self):
        """Prepare instead of repeatedly making attacks that all get blocked."""
        self.player.remaining_actions = 4
        self.agent.opp_model.est_current_shields = 4
        self.agent.opp_model.atk_p = 0.08
        self.agent.opp_model.def_p = 0.82
        self.agent.opp_model.bon_p = 0.10
        self.agent.opp_model.burst_risk = 0.0

        actions = self.agent.choose_actions(
            self.player, self.opponent, 7, [_log(6, 2, defends=4)], 1)

        self.assertEqual(sum(a.action_type == "bonus" for a in actions), 4)
        self.assertEqual(sum(a.action_type == "attack" for a in actions), 0)
        self.assertEqual(sum(a.action_type == "defend" for a in actions), 0)


class TestBattleLogIntegrity(unittest.TestCase):
    def test_result_turn_count_is_not_memory_window(self):
        from parameterized_ai_v2 import BattleEngineV2

        class Passive:
            def choose_actions(self, player, opponent, turn_num, turn_logs, player_id):
                return []

        engine = BattleEngineV2(
            Passive(), Passive(),
            [CharType.A, CharType.B, CharType.C],
            [CharType.A, CharType.B, CharType.C])
        result = engine.run(20)
        self.assertEqual(result["turns"], 20)
        self.assertEqual(len(engine.full_turn_logs), 20)
        self.assertLessEqual(len(engine.turn_logs), 8)

    def test_active_before_and_after_are_separate(self):
        from parameterized_ai_v2 import BattleEngineV2

        class Killer:
            def choose_actions(self, player, opponent, turn_num, turn_logs, player_id):
                from parameterized_ai_v2 import BattleAction
                return [BattleAction("attack", player.active_char_index,
                                     opponent.active_char_index)] * player.remaining_actions

        engine = BattleEngineV2(
            Killer(), Killer(),
            [CharType.A, CharType.B, CharType.C],
            [CharType.A, CharType.B, CharType.C])
        engine.run(20)
        for log in engine.full_turn_logs:
            self.assertGreaterEqual(log.p1_active, 0)
            self.assertGreaterEqual(log.p2_active, 0)
            self.assertGreaterEqual(log.p1_active_after, 0)
            self.assertGreaterEqual(log.p2_active_after, 0)

    def test_switch_back_to_previous_character_is_allowed_next_turn(self):
        from parameterized_ai_v2 import Player
        player = Player(1, [Character(CharType.C), Character(CharType.D)])
        player.remaining_actions = 3
        self.assertTrue(player.switch_character(1))
        player.reset_round_state()
        player.remaining_actions = 3
        self.assertTrue(player.switch_character(0))

    def test_voluntary_switch_moves_character_to_top_of_stack(self):
        from parameterized_ai_v2 import Player
        player = Player(1, [Character(CharType.C), Character(CharType.D), Character(CharType.B)])
        player.remaining_actions = 3
        self.assertTrue(player.switch_character(2))
        self.assertEqual(player.stack_order, [2, 0, 1])

    def test_death_promotes_next_living_and_keeps_dead_second(self):
        from parameterized_ai_v2 import Player
        player = Player(1, [Character(CharType.C), Character(CharType.D), Character(CharType.B)])
        player.characters[0].hp = 0
        player.force_switch_from_death()
        self.assertEqual(player.active_char_index, 1)
        self.assertEqual(player.stack_order, [1, 0, 2])

        player.characters[1].hp = 0
        player.force_switch_from_death()
        self.assertEqual(player.active_char_index, 2)
        self.assertEqual(player.stack_order, [2, 1, 0])

    def test_json_helpers_handle_corrupt_history(self):
        import tempfile
        import coevolution

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "versions.json")
            with open(path, "w") as f:
                f.write("")
            self.assertEqual(coevolution._load_json_or_default(path, []), [])
            coevolution._write_json_atomic(path, [{"ok": True}], indent=2)
            self.assertEqual(coevolution._load_json_or_default(path, [])[0]["ok"], True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
