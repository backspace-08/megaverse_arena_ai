import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
import cote_megaverse.play_vs_anchor as play_vs_anchor


class TestAnchorSelection(unittest.TestCase):
    def test_available_anchors_include_defender_and_specialists(self):
        names = play_vs_anchor.available_anchors()
        self.assertIn("Defender", names)
        self.assertIn("Counter", names)
        self.assertIn("Adaptive", names)

    def test_resolve_anchor_is_case_insensitive(self):
        name, factory = play_vs_anchor.resolve_anchor(" defender ")
        self.assertEqual(name, "Defender")
        self.assertEqual(factory().p.name, "Defender")

    def test_unknown_anchor_lists_available_choices(self):
        with self.assertRaisesRegex(ValueError, "Available anchors"):
            play_vs_anchor.resolve_anchor("unknown")


class TestAnchorCli(unittest.TestCase):
    def test_cli_passes_selected_anchor_and_game_count(self):
        captured = {}

        def fake_play(name, games):
            captured.update(name=name, games=games)
            return {"completed": 0}

        with patch.object(play_vs_anchor, "play_selected_anchor", fake_play):
            with redirect_stdout(io.StringIO()):
                self.assertEqual(play_vs_anchor.main(["--anchor", "Defender", "--games", "3"]), 0)
        self.assertEqual(captured, {"name": "Defender", "games": 3})


if __name__ == "__main__":
    unittest.main()
