"""Regression tests for the adaptive smart agent and tactical planner."""
import sys
import os
import unittest
import inspect

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
import cote_megaverse.parameterized_ai_v2 as _mechanics
import cote_megaverse.coevolution as _coevolution
sys.modules["parameterized_ai_v2"] = _mechanics
sys.modules["coevolution"] = _coevolution

from coevolution import (OpponentModel, PhaseShiftAI, SmartNeuralAgent,
                          compare_smart_genomes, random_smart_genome,
                          validate_smart_genome, validate_smart_tasks,
                          pairwise_smart_genome_matrix, HumanShieldBreakerAI,
                          HardDefenderAI, BurstBankerAI, SwitchPunisherAI,
                          TacticalBounds, exact_tactical_bounds,
                          guaranteed_exchange_facts, controlled_fitness_scores)
from cote_megaverse.planner_kernel import rank_macro_candidates, using_numba, resolve_numeric
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


class TestFixedValidation(unittest.TestCase):
    def test_validation_is_deterministic_and_preserves_rng(self):
        genome = random_smart_genome(seed=11)
        np.random.seed(1234)
        numpy_before = np.random.get_state()
        import random
        python_before = random.getstate()

        first = validate_smart_genome(genome, games=1)
        second = validate_smart_genome(genome, games=1)

        self.assertEqual(first, second)
        self.assertEqual(python_before, random.getstate())
        self.assertEqual(numpy_before[0], np.random.get_state()[0])
        np.testing.assert_array_equal(numpy_before[1], np.random.get_state()[1])
        self.assertEqual(numpy_before[2:], np.random.get_state()[2:])

    def test_common_seed_genome_comparison_is_reproducible(self):
        first = compare_smart_genomes(
            random_smart_genome(seed=1), random_smart_genome(seed=2), games=2)
        second = compare_smart_genomes(
            random_smart_genome(seed=1), random_smart_genome(seed=2), games=2)
        self.assertEqual(first, second)
        self.assertEqual(first["games"], 2)
        self.assertGreaterEqual(first["a_winrate"], 0.0)
        self.assertLessEqual(first["a_winrate"], 1.0)

    def test_task_validation_reports_separate_tactical_suites(self):
        report = validate_smart_tasks(random_smart_genome(seed=3), games=1)
        self.assertEqual(set(report["tasks"]), {
            "mechanics_validation", "shield_validation", "burst_validation",
            "switch_validation", "human_strategy_validation",
            "anti_counter_validation",
        })
        self.assertIn("HumanShieldBreaker", report["tasks"]["shield_validation"]["matchups"])
        self.assertIn("confidence_interval_95",
                      report["tasks"]["burst_validation"]["matchups"]["BurstBanker"])
        self.assertGreaterEqual(report["overall"], 0.0)
        self.assertLessEqual(report["overall"], 1.0)
        self.assertIn("draw_rate", report["tasks"]["shield_validation"]["matchups"]["HumanShieldBreaker"])
        self.assertIn("deadline_rate", report["tasks"]["shield_validation"]["matchups"]["HumanShieldBreaker"])
        self.assertIn("draw_and_deadline_penalty", report["controlled_scores"])

    def test_controlled_fitness_penalties_do_not_change_baseline(self):
        scores = controlled_fitness_scores(0.8, 0.6, draw_rate=0.25,
                                           deadline_rate=0.20)
        self.assertAlmostEqual(scores["baseline"], 0.72)
        self.assertLess(scores["draw_penalty"], scores["baseline"])
        self.assertLess(scores["draw_and_deadline_penalty"], scores["draw_penalty"])

    def test_pairwise_matrix_contains_outcomes_and_side_counts(self):
        genomes = [random_smart_genome(seed=4), random_smart_genome(seed=5)]
        report = pairwise_smart_genome_matrix(genomes, games=1)
        pair = report["matrix"][0][1]
        self.assertEqual(pair["games"], 1)
        self.assertEqual(pair["wins"] + pair["losses"] + pair["draws"], 1)
        self.assertEqual(sum(pair["side_counts"].values()), 1)
        self.assertEqual(len(pair["confidence_interval_95"]), 2)

    def test_hard_reference_factories_are_resettable(self):
        for opponent_type in (HumanShieldBreakerAI, HardDefenderAI,
                              BurstBankerAI, SwitchPunisherAI):
            opponent = opponent_type()
            self.assertTrue(hasattr(opponent, "reset_state"))
            opponent.reset_state()

    def test_evolution_does_not_expose_battle_persistence_helpers(self):
        self.assertFalse(hasattr(_coevolution, "_smart_record_battle"))
        self.assertFalse(hasattr(_coevolution, "_record_battle"))
        source = inspect.getsource(_coevolution.run_smart_coevolution)
        self.assertNotIn("open(", source)
        self.assertNotIn("battle_log_dir", source)


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

    def test_optional_kernel_ranks_candidates_without_numba(self):
        candidates = np.asarray([[4, 0, 0, -1], [0, 4, 0, -1],
                                 [2, 0, 2, -1]], dtype=np.int64)
        multipliers = np.ones((4, 4), dtype=np.float64)
        order = rank_macro_candidates(
            candidates, [6000], [2000], [0], 0, 6000, 2000, 1,
            0, 0, 0, multipliers, top_k=2)
        self.assertEqual(len(order), 2)
        self.assertTrue(all(0 <= index < len(candidates) for index in order))
        self.assertIsInstance(using_numba(), bool)

    def test_numba_numeric_resolver_matches_reference_on_random_states(self):
        rng = np.random.RandomState(20260722)
        multipliers = np.asarray([
            [1.0, 1.3, 1.0, 0.7],
            [0.7, 1.0, 1.3, 1.0],
            [1.0, 0.7, 1.0, 1.3],
            [1.3, 1.0, 0.7, 1.0],
        ])
        for _ in range(2000):
            hp = rng.randint(0, 6301, size=3)
            hp[rng.randint(0, 3)] = max(100, hp[rng.randint(0, 3)])
            active = int(rng.randint(0, 3))
            target_active = int(rng.randint(0, 3))
            state = {
                "hp": hp,
                "target_hp": rng.randint(0, 6301, size=3),
                "atk": rng.choice([1900, 1950, 2000, 2050, 2100], size=3),
                "target_atk": rng.choice([1900, 1950, 2000, 2050, 2100], size=3),
                "types": rng.randint(0, 4, size=3),
                "target_types": rng.randint(0, 4, size=3),
                "active": active,
                "target_active": target_active,
                "stack": np.asarray([0, 1, 2], dtype=np.int64),
                "target_stack": np.asarray([0, 1, 2], dtype=np.int64),
                "alive_count": int(np.sum(hp > 0)),
                "target_alive_count": 3,
                "bonus": int(rng.randint(0, 5)),
                "target_bonus": int(rng.randint(0, 5)),
                "shields": int(rng.randint(0, 9)),
                "target_shields": int(rng.randint(0, 9)),
                "remaining": int(rng.randint(0, 9)),
                "target_remaining": int(rng.randint(0, 9)),
                "target_type": int(rng.randint(0, 4)),
                "target_index": int(rng.randint(0, 3)),
            }
            if hp[state["active"]] <= 0:
                state["active"] = int(np.flatnonzero(hp > 0)[0]) if np.any(hp > 0) else 0
            state["alive_count"] = int(np.sum(state["hp"] > 0))
            state["target_active"] = int(np.flatnonzero(state["target_hp"] > 0)[0])
            state["target_alive_count"] = int(np.sum(state["target_hp"] > 0))
            plan = np.asarray([
                int(rng.randint(0, 9)), int(rng.randint(0, 9)),
                int(rng.randint(0, 5)), int(rng.randint(-1, 3))
            ], dtype=np.int64)
            reference = resolve_numeric(state, plan, multipliers, compiled=False)
            compiled = resolve_numeric(state, plan, multipliers, compiled=True)
            for key in ("hp", "active", "stack", "alive_count", "bonus",
                        "shields", "remaining", "damage", "blocked",
                        "unblocked", "switched", "forced_switch", "valid"):
                if isinstance(reference[key], np.ndarray):
                    np.testing.assert_array_equal(reference[key], compiled[key])
                else:
                    self.assertEqual(reference[key], compiled[key], key)

    def test_exact_bounds_prove_minimum_four_shields_at_eight_actions(self):
        self.player.remaining_actions = 8
        self.player.bonus_actions = 0
        bounds = exact_tactical_bounds(self.player)
        self.assertEqual(bounds, TacticalBounds(
            budget=8, min_attack=0, max_attack=8,
            min_defend=4, max_defend=8, min_bonus=0, max_bonus=4))
        self.assertGreaterEqual(bounds.min_defend, 4)

    def test_exact_exchange_facts_prove_four_attacks_cannot_hit_four_shields(self):
        self.player.characters[0].char_type = CharType.B
        self.opponent.characters[0].char_type = CharType.C
        facts = guaranteed_exchange_facts(self.opponent, self.player, 4, 4)
        self.assertTrue(facts["guaranteed_zero_damage"])
        self.assertEqual(facts["unblocked_attacks"], 0)

    def test_exact_exchange_facts_prove_lethal_without_overkill(self):
        self.player.characters[0].char_type = CharType.B
        self.opponent.characters[0].char_type = CharType.C
        self.opponent.active_character.hp = 5200
        facts = guaranteed_exchange_facts(self.player, self.opponent, 2, 0)
        self.assertTrue(facts["guaranteed_lethal"])
        self.assertEqual(facts["overkill_attacks"], 0)

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

    def test_robust_planner_records_mode_and_alternatives(self):
        self.opponent.active_character.hp = 100
        actions = self.agent.choose_actions(self.player, self.opponent, 7, [], 1)
        self.assertEqual(self.agent.last_decision_diagnostics["mode"], "FINISH")
        self.assertIn("selected_plan", self.agent.last_decision_diagnostics)
        self.assertTrue(self.agent.last_decision_diagnostics["alternatives"])
        self.assertGreaterEqual(sum(a.action_type == "attack" for a in actions), 1)

    def test_anti_burst_mode_uses_robust_search(self):
        self.opponent.active_character.hp = 12000
        self.player.characters.append(Character(CharType.A))
        self.player.stack_order.append(1)
        self.opponent.characters.append(Character(CharType.C))
        self.opponent.stack_order.append(1)
        self.agent.opp_model.est_bonus_bank = 2
        self.agent.opp_model.consecutive_bonus = 2
        self.agent.opp_model.burst_risk = 0.9
        self.player.active_character.hp = 6000
        actions = self.agent.choose_actions(self.player, self.opponent, 7, [], 1)
        diagnostics = self.agent.last_decision_diagnostics
        self.assertEqual(diagnostics["mode"], "ANTI_BURST")
        self.assertEqual(diagnostics["search_depth"], 3)
        self.assertLessEqual(len(actions), self.player.base_actions + 4)

    def test_switch_is_represented_as_macro_candidate_with_cost(self):
        from coevolution import MacroAction
        self.player.characters.append(Character(CharType.C))
        self.player.stack_order.append(1)
        self.player.remaining_actions = 2
        candidates = self.agent._macro_candidates(self.player)
        switched = [candidate for candidate in candidates if candidate.switch_to == 1]
        self.assertTrue(switched)
        self.assertTrue(all(candidate.attacks + candidate.defends + candidate.bonuses <= 1
                            for candidate in switched))
        self.assertTrue(any(isinstance(candidate, MacroAction) for candidate in switched))


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
