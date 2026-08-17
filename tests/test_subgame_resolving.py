"""Continual Subgame Resolving regression tests.

1. Reach conservation: the opponent reach profile stored after a resolve sums
   to 1.0 (or is None -> uniform fallback).
2. Turtling punishment: against a 1-HP full-shielder the bot must break the
   a4-into-d4 deadlock (bank -> a8) and win within a few rounds, instead of
   spamming a4 forever.
3. Lethal burst defense: when the opponent holds bank=4 and will burst a8, the
   bot must put >= 4 shields, not leave itself open.
"""
import os
import random
import sys
import unittest
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from cote_megaverse.rules import (  # noqa: E402
    GameState, Side, Character, Type, apply)
from cote_megaverse.cfr_bot import CFRBot  # noqa: E402
from cote_megaverse.infoset import OpponentModel  # noqa: E402

HP, ATK = 24000, 1500


def make_side(t, hp=HP, shields=0):
    return Side(characters=(Character(type=t, hp=hp, atk=ATK, max_hp=hp),),
                active=0, stack_order=(0,), bonus=0, shields=shields, actions=0)


class ReachConservationTests(unittest.TestCase):
    def test_reach_conservation(self):
        bot = CFRBot(depth=3, iters=80, cap=6, gamma=0.995, prune_after=20,
                     temperature=1.0, rng=random.Random(1))
        state = GameState(player=make_side(Type.A), opponent=make_side(Type.A),
                          turn=7, player_to_move=True).prepare()
        planning = GameState(state.player, state.opponent, state.turn, True)
        bot.choose(planning)
        if bot._reach is not None:
            self.assertAlmostEqual(sum(bot._reach.values()), 1.0, places=6)


class TurtlingPunishmentTests(unittest.TestCase):
    def _run(self, seed):
        bot = CFRBot(depth=3, iters=120, cap=6, gamma=0.995, prune_after=20,
                     temperature=1.0, rng=random.Random(seed))
        # CFR 7500 (5 hits) vs PL 1500 (1 hit) holding d4 shields, CFR to move
        state = GameState(player=make_side(Type.A, 7500),
                          opponent=make_side(Type.A, 1500, shields=4),
                          turn=7, player_to_move=True).prepare()
        moves = []
        for _ in range(8):  # 4 full rounds
            if state.player.lost or state.opponent.lost:
                break
            if state.player_to_move:
                before = state
                planning = GameState(state.player, state.opponent, state.turn, True)
                move = bot.choose(planning)
                moves.append((state.turn, move.attacks, move.defends, move.bonuses))
                state = apply(state, move)
                bot.observe_shields(before.opponent.shields)
            else:
                before = state
                move = type("M", (), {"attacks": 0, "defends": 4, "bonuses": 0,
                                      "switch": False})()
                state = apply(state, move)
                bot.observe(move.attacks, move.bonuses, move.switch,
                            budget=before.opponent.actions, turn=before.turn)
        return state, moves

    def test_turtling_1hp_punishment(self):
        state, moves = self._run(seed=1)
        self.assertTrue(state.opponent.lost,
                        "CFR must kill the 1-HP turtler; moves=%s" % moves)
        # the win must NOT be pure a4-into-d4 spam: it needs bank-then-burst
        # (an attack that breaks through d4) or a banked build-up
        punished = [m for m in moves if m[1] >= 5 or m[3] >= 1]
        self.assertTrue(punished,
                        "CFR must bank->burst or attack through d4, "
                        "not a4-spam; moves=%s" % moves)


class LethalBurstDefenseTests(unittest.TestCase):
    def test_lethal_burst_defense(self):
        # PL at 1 HP, holds bank=4 (belief exact), will burst a8. CFR must shield.
        bot = CFRBot(depth=3, iters=120, cap=6, gamma=0.995, prune_after=20,
                     rng=None)  # argmax for determinism
        state = GameState(player=make_side(Type.A, 7500),
                          opponent=make_side(Type.A, 1500),
                          turn=7, player_to_move=True).prepare()
        # pin the belief to (sh=0, bank=4): opponent banked b4 publicly
        bot.observe(attacks=0, bonuses=4, switched=False, budget=8, turn=7)
        bot.model._restrict(lambda shields, bank: bank == 4)
        planning = GameState(state.player, state.opponent, state.turn, True)
        move = bot.choose(planning)
        self.assertGreaterEqual(move.defends, 4,
                                "must shield >=4 vs imminent a8 burst, got (%d,%d,%d)"
                                % (move.attacks, move.defends, move.bonuses))


class ShieldPinLifetimeTests(unittest.TestCase):
    """Shield pins are turn-local: discarded on any new opponent allocation."""

    def test_shield_pin_discarded_on_new_turn_and_switch(self):
        m = OpponentModel()
        m.observe_turn(7, 4, 0)            # opponent leaves remainder 4
        m.observe_our_attack(5, 4)         # we attacked, 4 blocked -> pin
        self.assertEqual(m._shield_pin, (4, 4))

        # a new allocation discards the SHIELD pin (shields are turn-local).
        # The new reveal (bank 0) resolves the PREVIOUS split (shields 4, bank
        # 0), but the fresh split (R=4) is a new independent allocation, so it
        # is unbiased again - the bank is spent and re-placed every turn.
        m.observe_turn(8, 4, 0)
        self.assertIsNone(m._shield_pin)     # shield pin wiped
        self.assertIsNone(m._shield_lb)      # shield lb wiped
        self.assertEqual(m.records[0].confirmed_shields, 4)  # prev split resolved
        self.assertEqual(m.shield_distribution().get(4, 0.0), 0.2)  # fresh = uniform

        # a switch also discards the shield pin (old fighter's shields vanish).
        # Switch budget is E-1: reveal bank 0 on a remainder of 4.
        m2 = OpponentModel()
        m2.observe_turn(7, 4, 0)
        m2.observe_our_attack(5, 4)
        self.assertEqual(m2._shield_pin, (4, 4))
        m2.observe_turn(8, 5, 0, switched=True)   # switch costs 1 -> remainder 4
        self.assertIsNone(m2._shield_pin)
        self.assertIsNone(m2._shield_lb)

    def test_bank_reveal_resolves_previous_split_only(self):
        # reveal bank on turn 8: previous remainder 4, opponent held bank 1.
        # The reveal resolves the PREVIOUS split exactly (bank 1 -> shields 3),
        # but the bank is spent and re-placed every turn, so it must NOT
        # constrain the fresh split (R=5 here).
        m = OpponentModel()
        m.observe_turn(7, 4, 0)            # remainder 4 (candidates (4,0)..(0,4))
        m.observe_turn(8, 5, 0)            # budget 5 = base 4 + bank 1 (revealed)
        # the earlier split is now resolved: shields 3, bank 1 (sum = R=4)
        self.assertEqual(m.records[0].confirmed_shields, 3)
        # the fresh split (R=5) stays unbiased over its partitions (bank capped
        # at 4 -> banks 0..4 -> 5 splits)
        worlds = m.worlds()
        self.assertEqual(len(worlds), 5)
        self.assertTrue(all(w.shields + w.bank == 5 for w in worlds))
        self.assertTrue(all(abs(w.probability - 1.0 / 5) < 1e-9 for w in worlds))


if __name__ == "__main__":
    unittest.main()

