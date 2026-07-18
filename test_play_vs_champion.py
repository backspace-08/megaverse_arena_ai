"""
Tests for play_vs_champion.py (interactive game + benchmark).
"""
import sys, os, json, io, unittest
from unittest.mock import patch, MagicMock
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "parameterized_ai"))

# ── helpers ──────────────────────────────────────────────────────

def _make_versions(count=3):
    return [
        {"timestamp": f"20260718_17{42+i:02d}00", "best_fitness": 0.75 + i*0.05,
         "params": {"pop_size": 300, "generations": 80}}
        for i in range(count)
    ]

def _write_versions(versions, path):
    with open(path, "w") as f:
        json.dump(versions, f)


# ── Tests ────────────────────────────────────────────────────────

class TestListVersions(unittest.TestCase):
    VERSIONS_PATH = os.path.join("parameterized_ai", "versions.json")

    def setUp(self):
        self._has_real = os.path.exists(self.VERSIONS_PATH)
        if self._has_real:
            with open(self.VERSIONS_PATH) as f:
                self._real_data = json.load(f)
            os.remove(self.VERSIONS_PATH)

    def tearDown(self):
        if self._has_real:
            with open(self.VERSIONS_PATH, "w") as f:
                json.dump(self._real_data, f)

    def test_no_file(self):
        from play_vs_champion import list_versions
        v = list_versions()
        self.assertEqual(v, [])

    def test_with_versions(self):
        from play_vs_champion import list_versions
        data = _make_versions(2)
        _write_versions(data, self.VERSIONS_PATH)
        v = list_versions()
        self.assertEqual(len(v), 2)
        self.assertEqual(v[0]["best_fitness"], 0.75)


class TestLoadChampion(unittest.TestCase):
    VERSIONS_PATH = os.path.join("parameterized_ai", "versions.json")
    NPY_PATH = os.path.join("parameterized_ai", "best_genome.npy")

    def setUp(self):
        # Hide versions.json so load_champion falls through to .npy
        self._vs_backup = None
        if os.path.exists(self.VERSIONS_PATH):
            with open(self.VERSIONS_PATH) as f:
                self._vs_backup = json.load(f)
            os.remove(self.VERSIONS_PATH)

    def tearDown(self):
        if self._vs_backup is not None:
            with open(self.VERSIONS_PATH, "w") as f:
                json.dump(self._vs_backup, f)

    def test_load_latest_npy(self):
        """Fallback to best_genome.npy if no versions.json."""
        from play_vs_champion import load_champion
        if not os.path.exists(self.NPY_PATH):
            self.skipTest("best_genome.npy not found")
        agent = load_champion()
        self.assertIsNotNone(agent)
        self.assertTrue(hasattr(agent, "choose_actions"))
        self.assertTrue(hasattr(agent, "name"))

    def test_random_fallback(self):
        """If no npy file exists, should return random agent."""
        from play_vs_champion import load_champion
        renamed = None
        if os.path.exists(self.NPY_PATH):
            renamed = self.NPY_PATH + ".bak"
            os.rename(self.NPY_PATH, renamed)
        try:
            agent = load_champion()
            self.assertIsNotNone(agent)
            self.assertIn("Random", agent.name)
        finally:
            if renamed:
                os.rename(renamed, self.NPY_PATH)


class TestBenchmark(unittest.TestCase):
    def test_benchmark_runs(self):
        """benchmark completes without error and returns reasonable results."""
        from play_vs_champion import benchmark
        from coevolution import random_smart_genome, SmartNeuralAgent
        agent = SmartNeuralAgent(random_smart_genome(seed=0), "TestAgent")
        # Capture stdout
        captured = io.StringIO()
        old = sys.stdout
        sys.stdout = captured
        try:
            benchmark(agent, n=5)
        finally:
            sys.stdout = old
        output = captured.getvalue()
        self.assertIn("Champion vs reference profiles", output)
        self.assertIn("AllIn", output)
        self.assertIn("BonusBanker", output)
        self.assertIn("Adaptive", output)


class TestTurnSummary(unittest.TestCase):
    def test_last_turn_summary_no_logs(self):
        from play_vs_champion import last_turn_summary
        r = last_turn_summary([], 1, {}, {})
        self.assertIsNone(r)

    def test_last_turn_summary_filters(self):
        from play_vs_champion import last_turn_summary
        from parameterized_ai_v2 import TurnLog
        logs = [
            TurnLog(turn_num=1, player_id=1, attack_actions=3, defend_actions=0, bonus_actions=0,
                    switched=False, unblocked_attacks=3, total_damage=300, opponent_shields=0,
                    blocked_shields=0, player_shields_before=0,
                    p1_hp=[100, 100], p2_hp=[100, 100]),
            TurnLog(turn_num=1, player_id=2, attack_actions=0, defend_actions=2, bonus_actions=1,
                    switched=False, unblocked_attacks=0, total_damage=0, opponent_shields=0,
                    blocked_shields=0, player_shields_before=0,
                    p1_hp=[100, 100], p2_hp=[100, 100]),
        ]
        r = last_turn_summary(logs, 1, [100], [100])
        self.assertIsNotNone(r)
        self.assertEqual(r.player_id, 2)  # opponent's log
        self.assertEqual(r.attack_actions, 0)
        self.assertEqual(r.defend_actions, 2)
        self.assertEqual(r.bonus_actions, 1)


class TestRunGame(unittest.TestCase):
    def setUp(self):
        from coevolution import random_smart_genome, SmartNeuralAgent
        self.agent_a = SmartNeuralAgent(random_smart_genome(seed=1), "A")
        self.agent_b = SmartNeuralAgent(random_smart_genome(seed=2), "B")
        from parameterized_ai_v2 import random_team
        self.team_a = random_team()
        self.team_b = random_team()

    def test_game_ai_vs_ai_completes(self):
        """Two AIs play each other — game completes with a winner."""
        from play_vs_champion import run_game

        class MockHuman:
            """Minimal human with auto-pick AI actions (no keyboard)."""
            name = "Mock"
            def choose_actions(self, player, opponent, turn_num, turn_logs, player_id):
                # Let agent B make the decisions
                return self.agent.choose_actions(player, opponent, turn_num, turn_logs, player_id)

        # Use agent A as 'human' by delegating to a real AI
        human = MockHuman()
        human.agent = self.agent_a

        winner = run_game(human, self.agent_b, self.team_a, self.team_b, human_first=True)
        self.assertIn(winner, [0, 1, 2])

    def test_game_symmetry(self):
        """P2-first game also completes."""
        from play_vs_champion import run_game

        class MockHuman:
            name = "Mock"
            def choose_actions(self, player, opponent, turn_num, turn_logs, player_id):
                return self.agent.choose_actions(player, opponent, turn_num, turn_logs, player_id)

        human = MockHuman()
        human.agent = self.agent_a

        winner = run_game(human, self.agent_b, self.team_a, self.team_b, human_first=False)
        self.assertIn(winner, [0, 1, 2])

    def test_ai_vs_ai_no_crash(self):
        """Both sides use real AIs — no keyboard interaction needed."""
        from play_vs_champion import run_game

        class AIHuman:
            name = "AIHuman"
            def choose_actions(self, p, o, tn, tl, pid):
                return self.ai.choose_actions(p, o, tn, tl, pid)

        h = AIHuman()
        h.ai = self.agent_a
        winner = run_game(h, self.agent_b, self.team_a, self.team_b, human_first=True)
        self.assertIn(winner, [0, 1, 2])


class TestBar(unittest.TestCase):
    def test_bar_full(self):
        from play_vs_champion import bar
        self.assertEqual(bar(100, 100, 5), "#####")
        self.assertEqual(bar(0, 100, 5), ".....")
        self.assertEqual(bar(50, 100, 10), "#####.....")

    def test_bar_zero_max(self):
        from play_vs_champion import bar
        self.assertEqual(bar(50, 0, 5), ".....")


class TestDescribeTurn(unittest.TestCase):
    def test_describe_attack(self):
        from play_vs_champion import describe_turn
        from parameterized_ai_v2 import TurnLog
        log = TurnLog(turn_num=1, player_id=1, attack_actions=3, defend_actions=0,
                      bonus_actions=0, switched=False, unblocked_attacks=3, total_damage=300,
                      opponent_shields=0, blocked_shields=0, player_shields_before=0,
                      p1_hp=[100], p2_hp=[100])
        out = describe_turn(log, "AI", [100], [])
        self.assertIn("3 атак", out)
        self.assertIn("300 ур", out)

    def test_describe_shield(self):
        from play_vs_champion import describe_turn
        from parameterized_ai_v2 import TurnLog
        log = TurnLog(turn_num=1, player_id=2, attack_actions=0, defend_actions=2,
                      bonus_actions=1, switched=False, unblocked_attacks=0, total_damage=0,
                      opponent_shields=0, blocked_shields=0, player_shields_before=0,
                      p1_hp=[100], p2_hp=[100])
        out = describe_turn(log, "You", [100], [])
        self.assertIn("2 щитов", out)
        self.assertIn("1 бонусов", out)

    def test_describe_nothing(self):
        from play_vs_champion import describe_turn
        from parameterized_ai_v2 import TurnLog
        log = TurnLog(turn_num=1, player_id=1, attack_actions=0, defend_actions=0,
                      bonus_actions=0, switched=False, unblocked_attacks=0, total_damage=0,
                      opponent_shields=0, blocked_shields=0, player_shields_before=0,
                      p1_hp=[100], p2_hp=[100])
        out = describe_turn(log, "AI", [100], [])
        self.assertEqual(out, "")


class TestHumanInputAI(unittest.TestCase):
    def test_pass(self):
        from play_vs_champion import HumanInputAI
        from parameterized_ai_v2 import Player, Character, CharType
        c = Character(char_type=CharType.A)
        p = Player(player_id=0, characters=[c])
        p.remaining_actions = 3
        opp = Player(player_id=1, characters=[Character(char_type=CharType.B)])
        opp.remaining_actions = 3
        ai = HumanInputAI()
        with patch("builtins.input", return_value=""):
            actions = ai.choose_actions(p, opp, 1, [], 0)
        self.assertEqual(actions, [])

    def test_attack_command(self):
        from play_vs_champion import HumanInputAI
        from parameterized_ai_v2 import Player, Character, CharType
        c = Character(char_type=CharType.A)
        p = Player(player_id=0, characters=[c])
        p.remaining_actions = 3
        opp = Player(player_id=1, characters=[Character(char_type=CharType.B)])
        opp.remaining_actions = 3
        ai = HumanInputAI()
        with patch("builtins.input", side_effect=["a", ""]):
            actions = ai.choose_actions(p, opp, 1, [], 0)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].action_type, "attack")

    def test_shield_command(self):
        from play_vs_champion import HumanInputAI
        from parameterized_ai_v2 import Player, Character, CharType
        c = Character(char_type=CharType.A)
        p = Player(player_id=0, characters=[c])
        p.remaining_actions = 3
        opp = Player(player_id=1, characters=[Character(char_type=CharType.B)])
        opp.remaining_actions = 3
        ai = HumanInputAI()
        with patch("builtins.input", side_effect=["s", ""]):
            actions = ai.choose_actions(p, opp, 1, [], 0)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].action_type, "defend")

    def test_bonus_command(self):
        from play_vs_champion import HumanInputAI
        from parameterized_ai_v2 import Player, Character, CharType, MAX_BONUS_ACTIONS
        c = Character(char_type=CharType.A)
        p = Player(player_id=0, characters=[c])
        p.remaining_actions = 3
        p.bonus_actions = 0
        opp = Player(player_id=1, characters=[Character(char_type=CharType.B)])
        opp.remaining_actions = 3
        ai = HumanInputAI()
        with patch("builtins.input", side_effect=["b", ""]):
            actions = ai.choose_actions(p, opp, 1, [], 0)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].action_type, "bonus")

    def test_quit_raises(self):
        from play_vs_champion import HumanInputAI, PlayerQuit
        from parameterized_ai_v2 import Player, Character, CharType
        c = Character(char_type=CharType.A)
        p = Player(player_id=0, characters=[c])
        p.remaining_actions = 3
        opp = Player(player_id=1, characters=[Character(char_type=CharType.B)])
        opp.remaining_actions = 3
        ai = HumanInputAI()
        with patch("builtins.input", return_value="q"):
            with self.assertRaises(PlayerQuit):
                ai.choose_actions(p, opp, 1, [], 0)


class TestStats(unittest.TestCase):
    def test_load_stats_nofile(self):
        from play_vs_champion import load_play_stats
        s = load_play_stats()
        self.assertEqual(s, [])

    def test_save_and_load(self):
        from play_vs_champion import save_play_stats, load_play_stats
        session = {"timestamp": "test", "anchors": [{"name": "AllIn", "won": 3, "lost": 2}]}
        save_play_stats(session)
        stats = load_play_stats()
        self.assertGreaterEqual(len(stats), 1)
        self.assertEqual(stats[-1]["timestamp"], "test")
        # Cleanup
        spath = os.path.join("parameterized_ai", "play_stats.json")
        if os.path.exists(spath):
            os.remove(spath)


if __name__ == "__main__":
    unittest.main(verbosity=2)
