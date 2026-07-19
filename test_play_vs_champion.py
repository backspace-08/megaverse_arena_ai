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
            try:
                with open(self.VERSIONS_PATH) as f:
                    self._real_data = json.load(f)
            except (OSError, json.JSONDecodeError):
                self._real_data = None
            os.remove(self.VERSIONS_PATH)

    def tearDown(self):
        if self._has_real and self._real_data is not None:
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

    def test_empty_or_corrupt_file_returns_empty(self):
        from play_vs_champion import list_versions
        with open(self.VERSIONS_PATH, "w") as f:
            f.write("")
        self.assertEqual(list_versions(), [])
        with open(self.VERSIONS_PATH, "w") as f:
            f.write("{broken")
        self.assertEqual(list_versions(), [])


class TestLoadChampion(unittest.TestCase):
    VERSIONS_PATH = os.path.join("parameterized_ai", "versions.json")
    NPY_PATH = os.path.join("parameterized_ai", "best_genome.npy")

    def setUp(self):
        # Tests may create or rename these files, so preserve real training output.
        self._vs_backup = None
        self._npy_backup = None
        if os.path.exists(self.VERSIONS_PATH):
            try:
                with open(self.VERSIONS_PATH) as f:
                    self._vs_backup = json.load(f)
            except (OSError, json.JSONDecodeError):
                self._vs_backup = None
            os.remove(self.VERSIONS_PATH)
        if os.path.exists(self.NPY_PATH):
            with open(self.NPY_PATH, "rb") as f:
                self._npy_backup = f.read()

    def tearDown(self):
        if self._vs_backup is not None:
            with open(self.VERSIONS_PATH, "w") as f:
                json.dump(self._vs_backup, f)
        elif os.path.exists(self.VERSIONS_PATH):
            os.remove(self.VERSIONS_PATH)
        if self._npy_backup is not None:
            with open(self.NPY_PATH, "wb") as f:
                f.write(self._npy_backup)
        elif os.path.exists(self.NPY_PATH):
            os.remove(self.NPY_PATH)

    def test_load_latest_npy(self):
        """Fallback to best_genome.npy if no versions.json."""
        from play_vs_champion import load_champion
        if not os.path.exists(self.NPY_PATH):
            self.skipTest("best_genome.npy not found")
        agent = load_champion()
        self.assertIsNotNone(agent)
        self.assertTrue(hasattr(agent, "choose_actions"))
        self.assertTrue(hasattr(agent, "name"))

    def test_load_version_npy(self):
        from play_vs_champion import load_champion
        data = _make_versions(1)
        _write_versions(data, self.VERSIONS_PATH)
        npy_path = os.path.join("parameterized_ai", f"best_genome_{data[0]['timestamp']}.npy")
        np.save(npy_path, np.arange(12, dtype=np.float32))
        try:
            with patch("builtins.input", return_value="1"):
                agent = load_champion()
            self.assertIsNotNone(agent)
            self.assertTrue(hasattr(agent, "choose_actions"))
            self.assertIn("v1", agent.name)
        finally:
            if os.path.exists(npy_path):
                os.remove(npy_path)
            if os.path.exists(self.VERSIONS_PATH):
                os.remove(self.VERSIONS_PATH)

    def test_load_champion_latest_if_versions_present(self):
        from play_vs_champion import load_champion
        data = _make_versions(1)
        _write_versions(data, self.VERSIONS_PATH)
        np.save(self.NPY_PATH, np.arange(12, dtype=np.float32))
        try:
            with patch("builtins.input", return_value=""):
                agent = load_champion()
            self.assertIsNotNone(agent)
            self.assertEqual(agent.name, "Champion")
            self.assertTrue(hasattr(agent, "choose_actions"))
        finally:
            if os.path.exists(self.NPY_PATH):
                os.remove(self.NPY_PATH)
            if os.path.exists(self.VERSIONS_PATH):
                os.remove(self.VERSIONS_PATH)

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


class TestLoadAgent(unittest.TestCase):
    def test_load_smart_agent(self):
        from play_vs_champion import _load_agent
        agent = _load_agent(np.arange(12, dtype=np.float32), "Smart")
        self.assertIsNotNone(agent)
        self.assertEqual(agent.name, "Smart")
        self.assertTrue(hasattr(agent, "choose_actions"))

    def test_load_neural_agent(self):
        from play_vs_champion import _load_agent
        # NeuralAgent expects the full LSTM genome shape used by coevolution.py
        agent = _load_agent(np.arange(3928, dtype=np.float32), "Neural")
        self.assertIsNotNone(agent)
        self.assertEqual(agent.name, "Neural")
        self.assertTrue(hasattr(agent, "choose_actions"))


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


class _ScriptedAI:
    """Deterministic AI: always plays the same action type with the full budget."""
    def __init__(self, action_type, name="Scripted"):
        self.action_type = action_type
        self.name = name

    def choose_actions(self, player, opponent, turn_num, turn_logs, player_id):
        from parameterized_ai_v2 import BattleAction
        return [BattleAction(self.action_type, player.active_char_index, opponent.active_char_index)
                for _ in range(player.remaining_actions)]


class TestRunGameMechanics(unittest.TestCase):
    """run_game's inline engine must follow the same rules as BattleEngineV2."""

    def setUp(self):
        from parameterized_ai_v2 import CharType
        self.team = [CharType.A, CharType.B, CharType.C]

    def test_attacker_beats_idler(self):
        """Pure attacker beats an AI that does nothing."""
        from play_vs_champion import run_game

        class Idler:
            name = "Idler"
            def choose_actions(self, p, o, tn, tl, pid):
                return []

        attacker = _ScriptedAI("attack", "Attacker")
        # human=attacker goes first as p1 → winner must be 1 (p1)
        winner = run_game(attacker, Idler(), self.team, self.team, human_first=True)
        self.assertEqual(winner, 1)

    def test_pure_defender_never_dies_to_draw(self):
        """Two pure defenders never damage each other → draw (0) at turn cap."""
        from play_vs_champion import run_game
        d1 = _ScriptedAI("defend", "D1")
        d2 = _ScriptedAI("defend", "D2")
        winner = run_game(d1, d2, self.team, self.team, human_first=True)
        self.assertEqual(winner, 0)

    def test_shields_block_attacks(self):
        """Defender's shields absorb attacks in run_game's execute_turn."""
        from play_vs_champion import run_game

        logs_seen = []

        class LoggingAttacker(_ScriptedAI):
            def choose_actions(self, player, opponent, turn_num, turn_logs, player_id):
                logs_seen.extend(t for t in turn_logs if t not in logs_seen)
                return super().choose_actions(player, opponent, turn_num, turn_logs, player_id)

        attacker = LoggingAttacker("attack", "Attacker")
        defender = _ScriptedAI("defend", "Defender")
        run_game(attacker, defender, self.team, self.team, human_first=True)
        # At least one attacker log must show a blocked attack
        atk_logs = [t for t in logs_seen if t.attack_actions > 0]
        self.assertTrue(any(t.blocked_shields > 0 for t in atk_logs),
                        "expected some attacks to be blocked by shields")

    def test_bonus_banking_capped(self):
        """Bonus banker never exceeds MAX_BONUS_ACTIONS in run_game."""
        from play_vs_champion import run_game
        from parameterized_ai_v2 import MAX_BONUS_ACTIONS

        max_seen = [0]

        class Banker(_ScriptedAI):
            def choose_actions(self, player, opponent, turn_num, turn_logs, player_id):
                max_seen[0] = max(max_seen[0], player.bonus_actions)
                return super().choose_actions(player, opponent, turn_num, turn_logs, player_id)

        banker = Banker("bonus", "Banker")
        attacker = _ScriptedAI("attack", "Attacker")
        run_game(banker, attacker, self.team, self.team, human_first=True)
        self.assertLessEqual(max_seen[0], MAX_BONUS_ACTIONS)


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

    def test_switch_command(self):
        from play_vs_champion import HumanInputAI
        from parameterized_ai_v2 import Player, Character, CharType
        c = Character(char_type=CharType.A)
        p = Player(player_id=0, characters=[c, Character(char_type=CharType.B)])
        p.remaining_actions = 3
        opp = Player(player_id=1, characters=[Character(char_type=CharType.C)])
        opp.remaining_actions = 3
        ai = HumanInputAI()
        with patch("builtins.input", side_effect=["sw", "2", ""]):
            actions = ai.choose_actions(p, opp, 1, [], 0)
        self.assertEqual(p.active_char_index, 1)
        self.assertEqual(actions, [])

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

    def test_bonus_full_rejected(self):
        """Bonus command at MAX_BONUS_ACTIONS does not queue an action."""
        from play_vs_champion import HumanInputAI
        from parameterized_ai_v2 import Player, Character, CharType, MAX_BONUS_ACTIONS
        c = Character(char_type=CharType.A)
        p = Player(player_id=0, characters=[c])
        p.remaining_actions = 3
        p.bonus_actions = MAX_BONUS_ACTIONS
        opp = Player(player_id=1, characters=[Character(char_type=CharType.B)])
        opp.remaining_actions = 3
        ai = HumanInputAI()
        with patch("builtins.input", side_effect=["b", ""]):
            actions = ai.choose_actions(p, opp, 1, [], 0)
        self.assertEqual(actions, [])

    def test_unknown_command_ignored(self):
        """Garbage input shows help and doesn't consume actions."""
        from play_vs_champion import HumanInputAI
        from parameterized_ai_v2 import Player, Character, CharType
        c = Character(char_type=CharType.A)
        p = Player(player_id=0, characters=[c])
        p.remaining_actions = 3
        opp = Player(player_id=1, characters=[Character(char_type=CharType.B)])
        opp.remaining_actions = 3
        ai = HumanInputAI()
        with patch("builtins.input", side_effect=["xyz", "a", ""]):
            actions = ai.choose_actions(p, opp, 1, [], 0)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].action_type, "attack")

    def test_switch_already_switched(self):
        """Second switch in the same round is rejected."""
        from play_vs_champion import HumanInputAI
        from parameterized_ai_v2 import Player, Character, CharType
        c = Character(char_type=CharType.A)
        p = Player(player_id=0, characters=[c, Character(char_type=CharType.B)])
        p.remaining_actions = 3
        p.switched_this_round = True
        opp = Player(player_id=1, characters=[Character(char_type=CharType.C)])
        opp.remaining_actions = 3
        ai = HumanInputAI()
        with patch("builtins.input", side_effect=["sw", ""]):
            actions = ai.choose_actions(p, opp, 1, [], 0)
        self.assertEqual(p.active_char_index, 0)  # no switch happened
        self.assertEqual(actions, [])

    def test_switch_cancel(self):
        """Choosing 0 in the switch menu cancels."""
        from play_vs_champion import HumanInputAI
        from parameterized_ai_v2 import Player, Character, CharType
        c = Character(char_type=CharType.A)
        p = Player(player_id=0, characters=[c, Character(char_type=CharType.B)])
        p.remaining_actions = 3
        opp = Player(player_id=1, characters=[Character(char_type=CharType.C)])
        opp.remaining_actions = 3
        ai = HumanInputAI()
        with patch("builtins.input", side_effect=["sw", "0", ""]):
            actions = ai.choose_actions(p, opp, 1, [], 0)
        self.assertEqual(p.active_char_index, 0)
        self.assertEqual(actions, [])

    def test_multiple_actions_queued(self):
        """Actions accumulate until the budget runs out."""
        from play_vs_champion import HumanInputAI
        from parameterized_ai_v2 import Player, Character, CharType
        c = Character(char_type=CharType.A)
        p = Player(player_id=0, characters=[c])
        p.remaining_actions = 3
        opp = Player(player_id=1, characters=[Character(char_type=CharType.B)])
        opp.remaining_actions = 3
        ai = HumanInputAI()
        # 3 actions fill the budget — loop exits without a 4th prompt
        with patch("builtins.input", side_effect=["a", "s", "b"]):
            actions = ai.choose_actions(p, opp, 1, [], 0)
        self.assertEqual([a.action_type for a in actions], ["attack", "defend", "bonus"])


class TestStats(unittest.TestCase):
    STATS_PATH = os.path.join("parameterized_ai", "play_stats.json")

    def setUp(self):
        self._stats_backup = None
        if os.path.exists(self.STATS_PATH):
            with open(self.STATS_PATH, "rb") as f:
                self._stats_backup = f.read()
            os.remove(self.STATS_PATH)

    def tearDown(self):
        if self._stats_backup is not None:
            with open(self.STATS_PATH, "wb") as f:
                f.write(self._stats_backup)
        elif os.path.exists(self.STATS_PATH):
            os.remove(self.STATS_PATH)

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

    def test_load_stats_corrupt_file(self):
        """Corrupt JSON returns [] instead of crashing."""
        from play_vs_champion import load_play_stats
        spath = os.path.join("parameterized_ai", "play_stats.json")
        with open(spath, "w") as f:
            f.write("{not valid json")
        try:
            s = load_play_stats()
            self.assertEqual(s, [])
        finally:
            pass

    def test_show_stats_empty(self):
        from play_vs_champion import show_play_stats
        captured = io.StringIO()
        old = sys.stdout
        sys.stdout = captured
        try:
            show_play_stats()
        finally:
            sys.stdout = old
        self.assertIn("no stats yet", captured.getvalue())

    def test_show_stats_with_sessions(self):
        from play_vs_champion import save_play_stats, show_play_stats
        session = {"timestamp": "20260718_000000",
                   "anchors": [{"name": "AllIn", "won": 3, "lost": 1},
                               {"name": "Defender", "won": 2, "lost": 2}]}
        save_play_stats(session)
        spath = os.path.join("parameterized_ai", "play_stats.json")
        captured = io.StringIO()
        old = sys.stdout
        sys.stdout = captured
        try:
            show_play_stats()
        finally:
            sys.stdout = old
        out = captured.getvalue()
        self.assertIn("AllIn", out)
        self.assertIn("Defender", out)
        self.assertIn("TOTAL", out)
        self.assertIn("75%", out)  # AllIn: 3/4


if __name__ == "__main__":
    unittest.main(verbosity=2)
