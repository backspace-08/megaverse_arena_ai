"""Golden facts from the current human reference battle in example.md."""

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from cote_megaverse.parameterized_ai_v2 import (
    CharType,
    Character,
    calculate_damage,
    round_damage,
)


class TestExampleGoldenReplay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = (ROOT / "docs" / "example.md").read_text(encoding="utf-8")

    def test_rosters_and_result_are_authoritative(self):
        self.assertIn("Fuutarou", self.text)
        self.assertIn("Егор", self.text)
        self.assertIn("Первый ход - противник", self.text)
        self.assertIn("Вы победили Fuutarou!", self.text)

    def test_turn_budget_and_allocations_1_to_16(self):
        expected = [
            (1, 1, "bonus"),
            (2, 2, "attack=2"),
            (3, 3, "attack=3"),
            (4, 2, "attack=1 bonus=1"),
            (5, 3, "bonus=3"),
            (6, 4, "switch=1 shield=1 bonus=2"),
            (7, 7, "switch=1 shield=2 bonus=4"),
            (8, 6, "attack=6"),
            (9, 8, "attack=6 bonus=2"),
            (10, 4, "shield=1 bonus=3"),
            (11, 6, "shield=5 bonus=1"),
            (12, 7, "shield=3 bonus=4"),
            (13, 5, "shield=5"),
            (14, 8, "shield=5 bonus=3"),
            (15, 4, "attack=4"),
            (16, 7, "attack=7"),
        ]
        # These are the agreed replay facts, kept in the test as a compact
        # audit table. The raw UI text remains the evidence source.
        self.assertEqual(len(expected), 16)
        self.assertEqual([row[1] for row in expected], [1, 2, 3, 2, 3, 4, 7, 6, 8, 4, 6, 7, 5, 8, 4, 7])
        self.assertEqual(expected[5][2], "switch=1 shield=1 bonus=2")
        self.assertEqual(expected[8][2], "attack=6 bonus=2")

    def test_golden_attack_exchanges_and_damage(self):
        # Keep each UI message block separate. The final shield block has no
        # damage line, so an unbounded regex must not borrow the next turn's
        # damage value.
        blocks = re.findall(r"> COTE megaverse:\n(.*?)(?=\n> COTE megaverse:|\Z)", self.text, re.S)
        exchanges = []
        for block in blocks:
            match = re.search(r"🔥(\d+) vs 🛡(\d+)", block)
            if not match:
                continue
            damage = re.search(r"Нанесено (\d+) урона", block)
            exchanges.append((int(match.group(1)), int(match.group(2)),
                              int(damage.group(1)) if damage else None))
        self.assertEqual(exchanges, [
            (2, 0, 3400),
            (3, 0, 5100),
            (1, 0, 2200),
            (6, 2, 6800),
            (6, 0, 7100),
            (4, 5, None),
            (7, 0, 8300),
        ])

    def test_attacker_atk_and_rounding_examples(self):
        attacker = Character(CharType.B, hp=4800, max_hp=4800, atk=1700)
        defender = Character(CharType.C, hp=5200, max_hp=5200, atk=1400)
        self.assertEqual(calculate_damage(attacker, defender, 1), 2200)
        self.assertEqual(round_damage(7140), 7100)
        self.assertEqual(round_damage(8330), 8300)


if __name__ == "__main__":
    unittest.main()
