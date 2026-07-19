"""
Coevolution of neural agents for COTE Megaverse.
Each agent = LSTM neural network (42→16 LSTM→8 softmax, 3912 params).
Population evolves through self-play with anti-specialization evaluation.
LSTM provides temporal memory; extra outputs predict opponent actions.
4-turn opponent history enables pattern recognition for bluff detection.
"""

import sys, os, random, math, json, time, warnings, tempfile
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory

import numpy as np
from tqdm import tqdm

# Suppress numpy overflow warnings during sigmoid evaluation (handled via clipping in _p())
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*overflow.*exp.*')

_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_this_dir, "parameterized_ai"))
from parameterized_ai_v2 import (BattleEngineV2, BattleAction, Player, Character,
                                 CharType, get_type_multiplier, random_team,
                                 MAX_BONUS_ACTIONS, MAX_BASE_ACTIONS, MAX_TOTAL_ACTIONS,
                                 ACTION_COST_SWITCH,
                                 BASE_HP, BASE_ATK, TURN_ACTIONS,
                                  WeightedRandomAIv2, AIProfile, CounterAI, AdaptiveAI)


# Fixed anchor profiles — every agent must learn to beat these
ANCHOR_PROFILES = [
    ("AllIn",       AIProfile("AllIn",       w_attack=18, w_defend=0,  w_bonus=0,
                              switch_when_disadvantaged=True, w_switch=3)),
    ("Defender",    AIProfile("Defender",    w_attack=3,  w_defend=12, w_bonus=1,
                              switch_when_disadvantaged=True, w_switch=3)),
    ("Aggro",       AIProfile("Aggro",       w_attack=12, w_defend=1,  w_bonus=0.5,
                              switch_when_disadvantaged=True, w_switch=3)),
    ("Switcher",    AIProfile("Switcher",    w_attack=10, w_defend=1,  w_bonus=2,
                              w_switch=8, switch_when_disadvantaged=True,
                              switch_min_hp_ratio=0.8, aggressive_after_forced_switch=True,
                              save_first_turns=1)),
    ("Gambler",     AIProfile("Gambler",     w_attack=5,  w_defend=5,  w_bonus=5,
                              w_switch=5, switch_when_disadvantaged=True,
                              switch_min_hp_ratio=0.5, randomness=1.0)),
    ("BonusBanker", AIProfile("BonusBanker", w_attack=1,  w_defend=1,  w_bonus=16,
                              w_switch=1, switch_when_disadvantaged=True,
                              bonus_target=4, switch_min_hp_ratio=0.3)),
]


class PhaseShiftAI:
    """Deterministic opponent that changes policy during one game.

    Fixed anchors test counter-play against known styles. This opponent tests
    whether the agent notices a style change instead of committing to one reply.
    """
    def __init__(self):
        self.name = "PhaseShift"
        self._turns = 0

    def reset_state(self):
        self._turns = 0

    def choose_actions(self, player, opponent, turn_num, turn_logs, player_id):
        my_turns = sum(1 for t in turn_logs if t.player_id == player_id)
        if my_turns < 2:
            profile = AIProfile("PhaseSave", w_attack=1, w_defend=1, w_bonus=16,
                                bonus_target=4)
        elif my_turns < 4:
            profile = AIProfile("PhaseGuard", w_attack=2, w_defend=14, w_bonus=1)
        else:
            profile = AIProfile("PhasePress", w_attack=16, w_defend=1, w_bonus=0)
        return WeightedRandomAIv2(profile).choose_actions(
            player, opponent, turn_num, turn_logs, player_id)

# CounterAI and AdaptiveAI are NOT WeightedRandomAIv2 — handled separately in _eval_worker.

# ============================================================
# SHARED MEMORY FOR PARALLEL EVALUATION (avoids pickle overhead)
# ============================================================
# Workers read genomes from shared memory instead of receiving them via pickle.
# On Windows spawn: saves ~9MB pickle per worker per generation.

_shm_genomes_name = None   # shared memory name for population genomes
_shm_genomes_shape = None  # (pop_size, genome_size)
_shm_hof_name = None       # shared memory name for HoF genomes
_shm_hof_shape = None      # (hof_max, genome_size)
_shm_hof_count = None      # mp.Value — how many HoF slots are used

# Cached numpy views (attached once per worker process)
_worker_pop = None
_worker_hof = None

def _init_shared_worker(shm_genomes_name, shm_genomes_shape,
                        shm_hof_name, shm_hof_shape, shm_hof_count):
    """Initializer for worker processes — attaches to shared memory once."""
    global _shm_genomes_name, _shm_genomes_shape
    global _shm_hof_name, _shm_hof_shape, _shm_hof_count
    global _worker_pop, _worker_hof
    _shm_genomes_name = shm_genomes_name
    _shm_genomes_shape = shm_genomes_shape
    _shm_hof_name = shm_hof_name
    _shm_hof_shape = shm_hof_shape
    _shm_hof_count = shm_hof_count
    # Attach once — numpy views stay valid for the lifetime of the worker
    shm_pop = SharedMemory(name=shm_genomes_name)
    _worker_pop = np.ndarray(shm_genomes_shape, dtype=np.float32, buffer=shm_pop.buf)
    shm_hof = SharedMemory(name=shm_hof_name)
    _worker_hof = np.ndarray(shm_hof_shape, dtype=np.float32, buffer=shm_hof.buf)


# ============================================================
# NEURAL NETWORK ARCHITECTURE (LSTM + Action Prediction)
# ============================================================
N_IN = 42
N_HID = 16  # LSTM hidden state size (increased from 8 for capacity)
N_OUT = 8   # 4 action probs + 4 predicted opponent actions

# ============================================================
# SMART AGENT ARCHITECTURE (tiny genome + opponent model)
# ============================================================
# Replaces 3912-param LSTM with 12-param preference genome.
# Opponent model + expectimax are algorithmic (not learned).

SMART_GENOME_SIZE = 12


def _load_json_or_default(path, default):
    """Read a JSON artifact defensively after an interrupted previous run."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, PermissionError, json.JSONDecodeError, OSError):
        return default


def _write_json_atomic(path, value, **dump_kwargs):
    """Write JSON through a sibling temporary file and replace atomically."""
    directory = os.path.dirname(path) or "."
    fd, temp_path = tempfile.mkstemp(prefix=".json-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, **dump_kwargs)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, path)
    except Exception:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
        raise

# Production profile for the smart co-evolution run. Reference games are kept
# separate from self-play games because they are the anti-specialization test.
SMART_PRODUCTION_CONFIG = {
    "pop_size": 240,
    "generations": 120,
    "games_per_eval": 30,
    "elite_frac": 0.07,
    "mut_rate": 0.15,
    "mut_sigma": 0.15,
    "n_jobs": 8,
    "hof_add": 12,
    "hof_ratio": 0.25,
    "hof_max": 200,
    "reference_games": 8,
    "snapshot_interval": 10,
}

# Parameter indices for smart agent
SG_ADAPT      = 0   # opponent model update speed (0=slow, 1=fast)
SG_AGGRO      = 1   # attack vs defend preference (0=shield, 1=attack)
SG_BONUS      = 2   # bonus tendency (0=never, 1=always)
SG_SW_THRESH  = 3   # HP threshold for defensive switch (0=never, 1=always)
SG_SW_AGGRO   = 4   # aggressive switch for type advantage (0=never, 1=always)
SG_PRESS      = 5   # press when ahead in characters
SG_KILL       = 6   # focus fire on low-HP targets
SG_BURST      = 7   # bonus saving tendency
SG_REACT      = 8   # reactivity to opponent model
SG_BLUFF      = 9   # action stochasticity
SG_SHIELD_BON = 10  # shield when opponent hoards bonus
SG_TYPE_SENS  = 11  # sensitivity to type matchup

def smart_genome_size():
    return SMART_GENOME_SIZE

def random_smart_genome(seed=None):
    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
    return rng.randn(SMART_GENOME_SIZE).astype(np.float32) * 0.5


class OpponentModel:
    """Tracks opponent's action tendencies via EMA + N-gram pattern detection.

    - EMA: smooth estimate of atk/def/bon ratios
    - N-gram: tracks last 4 action vectors to detect patterns (burst prep, switch cycling)
    - Burst risk: probability opponent is about to unload banked bonus
    """
    def __init__(self, adapt_speed=0.3):
        self.adapt_speed = adapt_speed
        self.reset()

    def reset(self):
        self.atk_p = 0.33
        self.def_p = 0.33
        self.bon_p = 0.33
        self.est_bonus_bank = 0
        self.est_current_shields = 0

        # N-gram: last 4 opponent turns (atk, def, bon, switched)
        self.history = []

        # Derived flags
        self.burst_risk = 0.0
        self.consecutive_bonus = 0
        self.last_was_attack_wave = False
        self.opponent_job_changed = False  # detected strategy shift
        self._seen_turns = set()

    def update(self, turn_logs, player_id):
        opp_turns = [t for t in turn_logs if t.player_id != player_id]
        new_turns = [t for t in opp_turns if t.turn_num not in self._seen_turns]
        for t in new_turns:
            self._seen_turns.add(t.turn_num)
            total = t.attack_actions + t.defend_actions + t.bonus_actions
            if total == 0:
                continue

            atk_r = t.attack_actions / total
            def_r = t.defend_actions / total
            bon_r = t.bonus_actions / total

            a = self.adapt_speed
            ema_old_atk = self.atk_p
            self.atk_p = self.atk_p * (1.0 - a) + atk_r * a
            self.def_p = self.def_p * (1.0 - a) + def_r * a
            self.bon_p = self.bon_p * (1.0 - a) + bon_r * a
            s = self.atk_p + self.def_p + self.bon_p
            self.atk_p /= s; self.def_p /= s; self.bon_p /= s
            self.opponent_job_changed = abs(self.atk_p - ema_old_atk) > 0.2

            # A bonus action is publicly observable and is spent on the opponent's
            # next turn, so its current bank can be tracked exactly.
            self.est_bonus_bank = min(MAX_BONUS_ACTIONS, t.bonus_actions)
            self.est_current_shields = t.defend_actions

            self.history.append((t.attack_actions, t.defend_actions, t.bonus_actions, 1 if t.switched else 0))
            if len(self.history) > 4:
                self.history.pop(0)

            if t.bonus_actions > 0 and t.attack_actions == 0:
                self.consecutive_bonus = min(3, self.consecutive_bonus + 1)
            elif t.attack_actions > 0:
                self.consecutive_bonus = max(0, self.consecutive_bonus - 1)

            self.last_was_attack_wave = t.attack_actions >= 4

        risk = 0.0
        if self.est_bonus_bank >= 3:
            risk = 0.85
        elif self.est_bonus_bank >= 2:
            risk = 0.65
        elif self.consecutive_bonus >= 2:
            risk = 0.50
        self.burst_risk = risk



class SmartNeuralAgent:
    """Lightweight agent: 12-param genome + OpponentModel (N-gram + EMA) + 1-step expectimax.

    Evolution tunes base preferences and reaction strengths only.
    Pattern detection, anticipation, and counter-play are algorithmic.
    """
    def __init__(self, genome, name="Smart"):
        self.genome = genome.astype(np.float32)
        self.name = name
        self.opp_model = OpponentModel(self._p(SG_ADAPT, 0.3))
        self._last_attack_count = 0
        self._action_counts = [0, 0, 0, 0]
        self.reset_state()

    def _p(self, idx, default=0.5):
        if idx >= len(self.genome):
            return default
        x = float(self.genome[idx])
        if x < -100:
            return 0.0
        if x > 100:
            return 1.0
        return 1.0 / (1.0 + np.exp(-x))

    def reset_state(self):
        self.opp_model.reset()

    def choose_actions(self, player, opponent, turn_num, turn_logs, player_id):
        # 1. Update opponent model (EMA + N-gram detection)
        self.opp_model.update(turn_logs, player_id)
        om = self.opp_model

        # 2. Switch decision with weighted character evaluation
        remaining = player.remaining_actions
        did_switch = self._decide_switch(player, opponent, om)
        if did_switch:
            remaining = player.remaining_actions
            if remaining <= 0:
                return []

        # 3. Compute base action weights
        w_atk, w_def, w_bon = self._compute_weights(player, opponent, turn_num, om)

        # 4. Expectimax: generate candidates, evaluate against opponent model, pick best
        n_attack, n_defend, n_bonus = self._expectimax_best(
            remaining, player, opponent, turn_num, w_atk, w_def, w_bon, om)

        # If the current wall blocks the whole turn, do not spend the turn on
        # guaranteed zero-damage attacks. Bank the available actions instead;
        # the next expanded turn can then use the existing breakthrough rule.
        max_bonus = MAX_BONUS_ACTIONS - player.bonus_actions
        if (om.est_current_shields >= remaining and max_bonus > 0
                and om.def_p > 0.55 and om.atk_p < 0.35
                and om.burst_risk < 0.55):
            n_bonus = min(remaining, max_bonus)
            n_attack = remaining - n_bonus
            n_defend = 0

        # Break shield deadlocks. If the opponent has just shown a large shield
        # wall but is not preparing an attack, spend enough actions to get through
        # it instead of repeatedly defending and banking forever.
        if (om.est_current_shields > 0 and om.est_current_shields < remaining
                and om.def_p > 0.55
                and om.atk_p < 0.35 and om.burst_risk < 0.55):
            breakthrough = min(remaining, om.est_current_shields + 1)
            if breakthrough > n_attack:
                n_attack = breakthrough
                spare = remaining - n_attack
                n_defend = min(n_defend, spare)
                n_bonus = min(n_bonus, spare - n_defend)
                used = n_attack + n_defend + n_bonus
                n_attack += remaining - used

        # 5. Cap bonus, overflow to attack
        if n_bonus > max_bonus:
            n_attack += n_bonus - max_bonus
            n_bonus = max_bonus
        if n_bonus < 0: n_bonus = 0
        if n_attack < 0: n_attack = 0

        # 6. Bluff: swap atk ↔ def to stay unpredictable
        bluff = self._p(SG_BLUFF, 0.05)
        if bluff > 0 and np.random.random() < bluff and n_defend > 0 and n_attack > 0:
            swap = min(n_attack, n_defend, max(1, int(remaining * 0.2)))
            n_attack -= swap; n_defend += swap

        # 7. Build actions
        actions = []
        for _ in range(n_attack):
            actions.append(BattleAction("attack", player.active_char_index, opponent.active_char_index))
        for _ in range(n_defend):
            actions.append(BattleAction("defend", player.active_char_index))
        for _ in range(n_bonus):
            actions.append(BattleAction("bonus", player.active_char_index))
        random.shuffle(actions)
        self._action_counts[0] += n_attack
        self._action_counts[1] += n_defend
        self._action_counts[2] += n_bonus
        self._action_counts[3] += int(did_switch)
        self._last_attack_count = n_attack
        return actions

    # ─── Expectimax 1-step ─────────────────────────────────────────

    def _expectimax_best(self, remaining, player, opponent, turn_num, w_atk, w_def, w_bon, om):
        """Choose a full allocation using the real order of play.

        Current attacks meet shields already set by the opponent. Current shields
        only matter against their next turn, whose action budget is known from the
        turn schedule plus their publicly observed bonus bank.
        """
        candidates = self._gen_candidates(remaining, player)
        scenarios = self._opponent_scenarios(turn_num, om)
        my_char = player.active_character
        opp_char = opponent.active_character
        my_dmg = int(my_char.atk * get_type_multiplier(my_char.char_type, opp_char.char_type))
        opp_dmg = int(opp_char.atk * get_type_multiplier(opp_char.char_type, my_char.char_type))
        total_weight = w_atk + w_def + w_bon

        best_score = -float("inf")
        best_dist = (remaining, 0, 0)
        for atk, df, bon in candidates:
            score = 0.0
            for weight, opp_shields, opp_atk in scenarios:
                score += weight * self._eval_net(
                    atk, df, bon, opp_shields, opp_atk, my_dmg, opp_dmg, player, opponent)

            # Genome preferences only break close tactical ties. They must never
            # remove a legal counter-play from the search space.
            if total_weight > 0:
                prior = (atk * w_atk + df * w_def + bon * w_bon) / total_weight
                score += prior * 35.0

            # A shield-heavy opponent creates a preparation window. Repeating
            # defense into that window is a deadlock; bank actions so the next
            # unshielded turn can break through with tempo.
            if om.def_p > 0.40:
                score += bon * my_dmg * (0.55 + 1.25 * om.def_p)
                score -= df * my_dmg * 0.35 * om.def_p
            if score > best_score:
                best_score = score
                best_dist = (atk, df, bon)
        return best_dist

    def _opponent_scenarios(self, turn_num, om):
        """Return (probability, current_shields, next_turn_attacks) beliefs."""
        next_base = min(TURN_ACTIONS.get(turn_num + 1, MAX_BASE_ACTIONS), MAX_TOTAL_ACTIONS)
        next_actions = min(next_base + int(om.est_bonus_bank), MAX_TOTAL_ACTIONS)
        expected_attacks = int(round(next_actions * om.atk_p))
        current_shields = om.est_current_shields
        calm_weight = 0.55 * (1.0 - 0.65 * om.burst_risk)
        scenarios = [(calm_weight, current_shields, expected_attacks)]

        # A low-shield greedy line and a high-shield control line represent the
        # uncertainty that cannot be observed before committing this allocation.
        scenarios.append((0.20, current_shields,
                          min(next_actions, max(expected_attacks, int(next_actions * 0.70)))))
        scenarios.append((0.15, current_shields,
                          max(0, int(next_actions * 0.25))))
        if om.burst_risk > 0.0:
            scenarios.append((0.15 + 0.60 * om.burst_risk, current_shields,
                              min(next_actions, max(expected_attacks, int(next_actions * 0.80)))))

        total = sum(weight for weight, _, _ in scenarios)
        return [(weight / total, shields, attacks) for weight, shields, attacks in scenarios]

    def _eval_net(self, my_atk, my_def, my_bon, opp_shields, opp_atk,
                  my_dmg, opp_dmg, player, opponent):
        """Two-phase tactical utility for this turn then the opponent's reply."""
        unblocked = max(0, my_atk - opp_shields)
        dealt = min(opponent.active_character.hp, unblocked * my_dmg)
        received = min(player.active_character.hp, max(0, opp_atk - my_def) * opp_dmg)
        score = dealt - received

        # Removing an active character changes matchup and tempo, so a lethal is
        # materially more valuable than its final hit alone. The symmetric penalty
        # makes survival against a likely burst a first-class tactical objective.
        if dealt >= opponent.active_character.hp:
            score += BASE_HP * 0.70
        if received >= player.active_character.hp:
            score -= BASE_HP * 0.85

        bonus_gain = min(my_bon, MAX_BONUS_ACTIONS - player.bonus_actions)
        # A stored action is useful but deliberately valued below guaranteed damage.
        score += bonus_gain * my_dmg * 0.22
        return score

    def _gen_candidates(self, remaining, player):
        """Enumerate every legal attack/defend/bonus allocation (at most 45)."""
        max_bonus = MAX_BONUS_ACTIONS - player.bonus_actions
        return [(atk, defend, remaining - atk - defend)
                for atk in range(remaining + 1)
                for defend in range(remaining - atk + 1)
                if remaining - atk - defend <= max_bonus]

    # ─── Switch logic ──────────────────────────────────────────────

    def _decide_switch(self, player, opponent, om):
        if (player.remaining_actions < ACTION_COST_SWITCH
            or player.switched_this_round
            or len(player.alive_characters) <= 1):
            return False

        my_char = player.active_character
        opp_char = opponent.active_character
        mult = get_type_multiplier(my_char.char_type, opp_char.char_type)
        hp_ratio = my_char.hp / my_char.max_hp

        switch_thresh = self._p(SG_SW_THRESH, 0.3)
        aggro_switch = self._p(SG_SW_AGGRO, 0.3)

        # Defensive: at disadvantage AND low HP
        if mult < 1.0 and hp_ratio < switch_thresh:
            target = self._best_switch_target(player, opponent)
            if target is not None:
                player.switch_character(target)
                return True

        # Aggressive: switch for type advantage when we can press
        if aggro_switch > 0.4 and mult <= 1.0:
            target = self._best_switch_target(player, opponent)
            if target is not None:
                new_mult = get_type_multiplier(player.characters[target].char_type, opp_char.char_type)
                if new_mult >= 1.3:
                    player.switch_character(target)
                    return True

        # Burst pivot: if opponent is about to burst, switch to a high-HP shield
        if om.burst_risk > 0.6 and hp_ratio < 0.5 and mult < 1.0:
            target = self._highest_hp_bench(player)
            if target is not None:
                player.switch_character(target)
                return True

        return False

    def _best_switch_target(self, player, opponent):
        opp_char = opponent.active_character
        best_idx = None
        best_val = -9999
        for i, c in enumerate(player.characters):
            if not c.is_alive() or i == player.active_char_index:
                continue
            mult = get_type_multiplier(c.char_type, opp_char.char_type)
            my_dmg = int(c.atk * mult)
            recv = int(opp_char.atk * get_type_multiplier(opp_char.char_type, c.char_type))

            hp_w = c.hp / c.max_hp
            atk_w = c.atk / 2500
            val = (my_dmg - recv) * hp_w + 300 * atk_w * mult

            if val > best_val:
                best_val = val
                best_idx = i
        return best_idx

    def _highest_hp_bench(self, player):
        """Find bench character with highest HP ratio (for burst defense)."""
        best_idx = None
        best_hp = -1
        for i, c in enumerate(player.characters):
            if not c.is_alive() or i == player.active_char_index:
                continue
            r = c.hp / c.max_hp
            if r > best_hp:
                best_hp = r
                best_idx = i
        return best_idx

    # ─── Weight computation ────────────────────────────────────────

    def _compute_weights(self, player, opponent, turn_num, om):
        my_char = player.active_character
        opp_char = opponent.active_character

        base_aggression = self._p(SG_AGGRO, 0.5)
        base_bonus = self._p(SG_BONUS, 0.3)

        w_atk = max(0.1, base_aggression * 3.0)
        w_def = max(0.1, (1.0 - base_aggression) * 2.0)
        w_bon = max(0.1, base_bonus * 2.0)

        react = self._p(SG_REACT, 0.5)
        type_sens = self._p(SG_TYPE_SENS, 0.5)

        mult = get_type_multiplier(my_char.char_type, opp_char.char_type)
        if mult > 1.0:
            w_atk *= (1.0 + type_sens * 0.6)
            w_def *= (1.0 - type_sens * 0.3)
        elif mult < 1.0:
            w_def *= (1.0 + type_sens * 0.5)
            w_atk *= (1.0 - type_sens * 0.2)

        # Opponent model (EMA)
        if om.atk_p > 0.5:
            w_def *= (1.0 + react * (om.atk_p - 0.5) * 2.0)
        if om.bon_p > 0.25:
            w_atk *= (1.0 + react * om.bon_p)
        if om.def_p > 0.3:
            w_bon *= (1.0 + react * om.def_p)

        # Burst risk: shift to defend when opponent likely to burst
        shield_bon = self._p(SG_SHIELD_BON, 0.4)
        if om.burst_risk > 0.3:
            w_def *= (1.0 + shield_bon * om.burst_risk)
        elif om.est_bonus_bank > 0.3:
            w_def *= (1.0 + shield_bon * om.est_bonus_bank * 0.5)

        # Kill focus
        kill = self._p(SG_KILL, 0.5)
        opp_hp_ratio = opp_char.hp / opp_char.max_hp
        if opp_hp_ratio < 0.3:
            w_atk *= (1.0 + kill * (1.0 - opp_hp_ratio / 0.3))

        # Press when ahead in characters
        my_alive = sum(1 for c in player.characters if c.is_alive())
        opp_alive = sum(1 for c in opponent.characters if c.is_alive())
        if my_alive > opp_alive:
            w_atk *= (1.0 + self._p(SG_PRESS, 0.5) * 0.5)

        # Bonus saving early game
        if turn_num <= 5:
            w_bon *= (1.0 + self._p(SG_BURST, 0.3) * 0.5)

        return w_atk, w_def, w_bon

# LSTM gates:
# Input gate:  W_ii (N_IN*N_HID) + W_hi (N_HID*N_HID) + b_i (N_HID)
# Forget gate: W_if (N_IN*N_HID) + W_hf (N_HID*N_HID) + b_f (N_HID)
# Cell gate:   W_ig (N_IN*N_HID) + W_hg (N_HID*N_HID) + b_g (N_HID)
# Output gate: W_io (N_IN*N_HID) + W_ho (N_HID*N_HID) + b_o (N_HID)
# Total LSTM: 4 * (N_IN*N_HID + N_HID*N_HID + N_HID)
#
# Output layer: W_out (N_HID*N_OUT) + b_out (N_OUT)

def genome_size():
    lstm = 4 * (N_IN * N_HID + N_HID * N_HID + N_HID)
    output = N_HID * N_OUT + N_OUT
    return lstm + output

def random_genome(seed=None):
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    return (rng.randn(genome_size()).astype(np.float32) * 0.5)

def forward_cached(agent, state, h=None, c=None):
    """
    LSTM forward pass using pre-unpacked weights from agent.
    agent: NeuralAgent with cached weight matrices
    state: list/array of 42 floats
    h: hidden state (N_HID,), or None for zeros
    c: cell state (N_HID,), or None for zeros
    Returns: (probs[8], new_h, new_c)
    """
    if h is None:
        h = np.zeros(N_HID, dtype=np.float32)
    if c is None:
        c = np.zeros(N_HID, dtype=np.float32)

    s = np.asarray(state, dtype=np.float32)

    # LSTM gates (using cached weights)
    i_gate = 1.0 / (1.0 + np.exp(-(s @ agent.W_ii + h @ agent.W_hi + agent.b_i)))
    f_gate = 1.0 / (1.0 + np.exp(-(s @ agent.W_if + h @ agent.W_hf + agent.b_f)))
    g_gate = np.tanh(s @ agent.W_ig + h @ agent.W_hg + agent.b_g)
    o_gate = 1.0 / (1.0 + np.exp(-(s @ agent.W_io + h @ agent.W_ho + agent.b_o)))

    new_c = f_gate * c + i_gate * g_gate
    new_h = o_gate * np.tanh(new_c)

    # Output layer (using cached weights)
    logits = new_h @ agent.W_out + agent.b_out
    e = np.exp(logits - np.max(logits))
    probs = e / np.sum(e)

    return probs, new_h, new_c


def forward(genome, state, h=None, c=None):
    """Backward-compatible wrapper for tests. Creates a temporary agent."""
    agent = NeuralAgent(genome)
    return forward_cached(agent, state, h, c)

def _expected_net_per_action(char, opp_char):
    """Expected net damage per action: my dealt - received."""
    my_dmg = int(char.atk * get_type_multiplier(char.char_type, opp_char.char_type))
    recv_dmg = int(opp_char.atk * get_type_multiplier(opp_char.char_type, char.char_type))
    return my_dmg - recv_dmg


# ============================================================
# STATE EXTRACTION
# ============================================================

def get_state(player, opponent, turn_num, turn_logs=None, player_id=None):
    """
    42-feature state vector:
    [0-15]  Original core features (HP, ATK, shields, bench, type, bonus, actions)
    [16-35] Opponent history: last 4 turns × 5 features (atk/def/bon/sw/dmg)
    [36]    Switch EV estimate
    [37-38] Actions next turn: my/opponent projected
    [39]    Game phase (early=0..late=1)
    [40-41] Characters alive: my/opponent
    """
    my_act = player.active_character
    en_act = opponent.active_character
    my_chars = player.characters
    en_chars = opponent.characters
    
    # Bench characters (non-active alive)
    def bench_hp(chars, active_idx):
        vals = []
        for i, c in enumerate(chars):
            if i != active_idx and c.is_alive():
                vals.append(c.hp / c.max_hp)
        while len(vals) < 2:
            vals.append(0.0)
        return vals[:2]
    
    def bench_matchup(chars, active_idx, opp_type):
        vals = []
        for i, c in enumerate(chars):
            if i != active_idx and c.is_alive():
                vals.append(get_type_multiplier(c.char_type, opp_type))
            else:
                vals.append(0.0)
        while len(vals) < 2:
            vals.append(0.0)
        return vals[:2]
    
    my_bench = bench_hp(my_chars, player.active_char_index)
    en_bench = bench_hp(en_chars, opponent.active_char_index)
    my_bench_mult = bench_matchup(my_chars, player.active_char_index, en_act.char_type)
    
    # Type advantage: my active vs enemy active
    mult = get_type_multiplier(my_act.char_type, en_act.char_type)
    if mult > 1.0:
        type_adv = 1.0
    elif mult < 1.0:
        type_adv = 0.0
    else:
        type_adv = 0.5
    
    # Opponent action history: last 4 completed turns (5 features each = 20 total)
    opp_hist = [0.0] * 20
    if turn_logs and player_id is not None:
        opp_turns = [t for t in turn_logs if t.player_id != player_id]
        opp_turns = opp_turns[-4:]  # last 4 turns
        for i, t in enumerate(opp_turns):
            base = i * 5
            opp_hist[base+0] = t.attack_actions / 8
            opp_hist[base+1] = t.defend_actions / 8
            opp_hist[base+2] = t.bonus_actions / 4
            opp_hist[base+3] = 1.0 if t.switched else 0.0
            opp_hist[base+4] = t.total_damage / 6000
    
    # Expected net damage improvement from switching
    current_net = _expected_net_per_action(my_act, en_act)
    best_net = current_net
    for i, c in enumerate(my_chars):
        if not c.is_alive() or i == player.active_char_index:
            continue
        net = _expected_net_per_action(c, en_act)
        if net > best_net:
            best_net = net
    expected_net_diff = max(0.0, best_net - current_net) / 2500
    
    # --- FEATURES [36-41] ---
    
    # [36] Switch EV estimate
    # [37] My projected actions next turn (current base + my bonus banked)
    next_turn = turn_num + 1
    my_next_base = min(TURN_ACTIONS.get(next_turn, 4), MAX_BASE_ACTIONS)
    my_next_total = min(my_next_base + player.bonus_actions, MAX_TOTAL_ACTIONS)
    # [38] Opponent projected actions next turn
    opp_next_base = min(TURN_ACTIONS.get(next_turn, 4), MAX_BASE_ACTIONS)
    opp_next_total = min(opp_next_base + opponent.bonus_actions, MAX_TOTAL_ACTIONS)
    
    # [39] Game phase: early game (turns 1-4) vs late game (turns 7+)
    game_phase = min(turn_num / 20.0, 1.0)
    
    # [40-41] Characters alive
    my_alive = sum(1 for c in my_chars if c.is_alive()) / 3.0
    opp_alive = sum(1 for c in en_chars if c.is_alive()) / 3.0
    
    state = [
        my_act.hp / my_act.max_hp,           # 0
        my_act.atk / 2500,                    # 1
        player.shields / 4,                   # 2
        en_act.hp / en_act.max_hp,            # 3
        my_bench[0],                          # 4
        my_bench[1],                          # 5
        en_bench[0],                          # 6
        en_bench[1],                          # 7
        type_adv,                             # 8
        player.bonus_actions / 4,             # 9
        opponent.bonus_actions / 4,           # 10
        player.remaining_actions / 8,         # 11
        player.base_actions / 8,              # 12
        1.0 if player.forced_switch_after_death else 0.0,  # 13
        my_bench_mult[0],                     # 14
        my_bench_mult[1],                     # 15
        opp_hist[0],                          # 16  opp turn -1: atk
        opp_hist[1],                          # 17  opp turn -1: def
        opp_hist[2],                          # 18  opp turn -1: bon
        opp_hist[3],                          # 19  opp turn -1: sw
        opp_hist[4],                          # 20  opp turn -1: dmg
        opp_hist[5],                          # 21  opp turn -2: atk
        opp_hist[6],                          # 22  opp turn -2: def
        opp_hist[7],                          # 23  opp turn -2: bon
        opp_hist[8],                          # 24  opp turn -2: sw
        opp_hist[9],                          # 25  opp turn -2: dmg
        opp_hist[10],                         # 26  opp turn -3: atk
        opp_hist[11],                         # 27  opp turn -3: def
        opp_hist[12],                         # 28  opp turn -3: bon
        opp_hist[13],                         # 29  opp turn -3: sw
        opp_hist[14],                         # 30  opp turn -3: dmg
        opp_hist[15],                         # 31  opp turn -4: atk
        opp_hist[16],                         # 32  opp turn -4: def
        opp_hist[17],                         # 33  opp turn -4: bon
        opp_hist[18],                         # 34  opp turn -4: sw
        opp_hist[19],                         # 35  opp turn -4: dmg
        expected_net_diff,                    # 36  switch value estimate
        my_next_total / 8,                    # 37  my projected actions next turn
        opp_next_total / 8,                   # 38  opp projected actions next turn
        game_phase,                           # 39  early(0) -> late(1)
        my_alive,                             # 40  my chars alive
        opp_alive,                            # 41  opp chars alive
    ]
    return state


# ============================================================
# NEURAL AGENT
# ============================================================

class NeuralAgent:
    """Agent that uses an LSTM neural network to decide actions."""
    
    def __init__(self, genome, name="Neural"):
        self.genome = genome.astype(np.float32)
        self.name = name
        self._last_attack_count = 0
        self._action_counts = [0, 0, 0, 0]  # atk, def, bon, sw
        self._h = None  # LSTM hidden state
        self._c = None  # LSTM cell state
        self._pred_opp = None  # last predicted opponent actions [atk, def, bon, sw]
        
        # Unpack weights once (avoids 12x reshape per forward() call)
        g = self.genome
        idx = 0
        self.W_ii = g[idx:idx+N_IN*N_HID].reshape(N_IN, N_HID); idx += N_IN*N_HID
        self.W_hi = g[idx:idx+N_HID*N_HID].reshape(N_HID, N_HID); idx += N_HID*N_HID
        self.b_i  = g[idx:idx+N_HID]; idx += N_HID
        self.W_if = g[idx:idx+N_IN*N_HID].reshape(N_IN, N_HID); idx += N_IN*N_HID
        self.W_hf = g[idx:idx+N_HID*N_HID].reshape(N_HID, N_HID); idx += N_HID*N_HID
        self.b_f  = g[idx:idx+N_HID]; idx += N_HID
        self.W_ig = g[idx:idx+N_IN*N_HID].reshape(N_IN, N_HID); idx += N_IN*N_HID
        self.W_hg = g[idx:idx+N_HID*N_HID].reshape(N_HID, N_HID); idx += N_HID*N_HID
        self.b_g  = g[idx:idx+N_HID]; idx += N_HID
        self.W_io = g[idx:idx+N_IN*N_HID].reshape(N_IN, N_HID); idx += N_IN*N_HID
        self.W_ho = g[idx:idx+N_HID*N_HID].reshape(N_HID, N_HID); idx += N_HID*N_HID
        self.b_o  = g[idx:idx+N_HID]; idx += N_HID
        self.W_out = g[idx:idx+N_HID*N_OUT].reshape(N_HID, N_OUT); idx += N_HID*N_OUT
        self.b_out = g[idx:idx+N_OUT]
    
    def reset_state(self):
        """Reset LSTM state at start of a new game."""
        self._h = None
        self._c = None
        self._pred_opp = None
    
    def choose_actions(self, player, opponent, turn_num, turn_logs, player_id):
        actions = []
        
        # Forward pass through LSTM (using cached weights)
        state = get_state(player, opponent, turn_num, turn_logs, player_id)
        all_probs, self._h, self._c = forward_cached(self, state, self._h, self._c)
        
        # Split into action probs and opponent prediction
        action_probs = all_probs[:4]   # [atk, def, bon, switch]
        pred_opp = all_probs[4:]       # predicted opponent [atk, def, bon, switch]
        self._pred_opp = pred_opp
        
        # LSTM learns to use pred_opp internally through backprop (evolution)
        # No hardcoded rules — let the network decide how to react
        
        # Boost switch probability by expected net damage improvement
        boost = 1.0 + state[26] * 2.0
        action_probs[3] *= boost
        action_probs /= action_probs.sum()
        
        remaining = player.remaining_actions
        if remaining <= 0:
            return actions
        
        # Stochastic switch decision via temperature-scaled sampling
        can_switch = (remaining >= ACTION_COST_SWITCH and not player.switched_this_round
                      and len(player.alive_characters) > 1)
        do_switch = False
        if can_switch:
            temp = 0.8
            scaled = action_probs ** temp
            scaled /= scaled.sum()
            do_switch = np.random.random() < scaled[3]
        
        if do_switch:
            target = self._best_switch_target(player, opponent)
            if target is not None and target != player.active_char_index:
                player.switch_character(target)
                remaining = player.remaining_actions
                if remaining <= 0:
                    return actions
                state = get_state(player, opponent, turn_num, turn_logs, player_id)
                all_probs, self._h, self._c = forward_cached(self, state, self._h, self._c)
                action_probs = all_probs[:4]
                action_probs[3] = 0.0  # already switched
        
        # Distribute remaining actions among atk/def/bon using largest-remainder method
        total = action_probs[0] + action_probs[1] + action_probs[2]
        if total > 0:
            frac = [action_probs[0] / total, action_probs[1] / total, action_probs[2] / total]
        else:
            frac = [1.0, 0.0, 0.0]

        exact = [remaining * f for f in frac]
        floored = [int(e) for e in exact]
        remainders = [(exact[i] - floored[i], i) for i in range(3)]
        remainders.sort(reverse=True)
        budget = remaining - sum(floored)
        for rem, idx in remainders:
            if budget <= 0:
                break
            floored[idx] += 1
            budget -= 1
        n_attack, n_defend, n_bonus = floored

        # Cap bonus, overflow to attack
        max_bonus = MAX_BONUS_ACTIONS - player.bonus_actions
        if n_bonus > max_bonus:
            n_attack += (n_bonus - max_bonus)
            n_bonus = max_bonus

        if n_bonus < 0:
            n_attack += n_bonus
            n_bonus = 0

        if n_attack < 0:
            n_attack = 0
        
        for _ in range(n_attack):
            actions.append(BattleAction("attack", player.active_char_index, opponent.active_char_index))
        for _ in range(n_defend):
            actions.append(BattleAction("defend", player.active_char_index))
        for _ in range(n_bonus):
            actions.append(BattleAction("bonus", player.active_char_index))
        
        # Track action distribution
        self._action_counts[0] += n_attack
        self._action_counts[1] += n_defend
        self._action_counts[2] += n_bonus
        self._action_counts[3] += 1 if do_switch else 0
        
        random.shuffle(actions)
        self._last_attack_count = n_attack
        return actions
    
    def record_turn(self, attack_count):
        self._last_attack_count = attack_count
    
    def _best_switch_target(self, player, opponent):
        """Выбрать лучшую цель для свитча по expected net per action."""
        best_idx = None
        best_net = None
        opp_char = opponent.active_character
        for i, c in enumerate(player.characters):
            if not c.is_alive() or i == player.active_char_index:
                continue
            net = _expected_net_per_action(c, opp_char)
            if best_idx is None or net > best_net:
                best_idx = i
                best_net = net
        return best_idx


# ============================================================
# GENETIC OPERATORS
# ============================================================

def crossover(p1, p2):
    """Blend crossover with random blending factor."""
    alpha = np.random.uniform(0.3, 0.7)
    child = alpha * p1 + (1 - alpha) * p2
    return child

def mutate(genome, rate=0.10, sigma=0.12):
    """Gaussian mutation."""
    g = genome.copy()
    mask = np.random.random(g.shape) < rate
    noise = np.random.randn(*g.shape) * sigma
    g[mask] += noise[mask]
    return g

def _cosine_dist(a, b):
    """Cosine distance between two vectors (0 = identical, 2 = opposite)."""
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return 1.0 - dot / (na * nb)

def _hof_is_novel(genome, hof, min_dist=0.05):
    """Check if genome is sufficiently different from all HoF members."""
    for h in hof:
        if _cosine_dist(genome, h) < min_dist:
            return False
    return True

def tournament_select(fitnesses, k=None):
    """Select index via k-way tournament."""
    if k is None:
        k = 4
    best = None
    best_f = -1
    for _ in range(k):
        i = random.randint(0, len(fitnesses) - 1)
        if fitnesses[i] > best_f:
            best_f = fitnesses[i]
            best = i
    return best


# ============================================================
# EVALUATION
# ============================================================

def _eval_worker(args):
    """Worker for parallel evaluation (module-level for Windows spawn).
    
    Reads opponent genomes from shared memory (no pickle overhead).
    Anti-specialization: track per-opponent winrates.
    """
    genome_idx, n_games, seed, hof_ratio = args
    random.seed(seed)
    
    # Read genome from cached shared memory
    my_genome = _worker_pop[genome_idx].copy()
    agent = NeuralAgent(my_genome)
    agent._action_counts = [0, 0, 0, 0]
    agent.reset_state()
    
    # Read HoF count from shared memory
    hof_count = _shm_hof_count.value if _shm_hof_count is not None else 0
    n_hof = int(n_games * hof_ratio) if hof_count > 0 else 0
    n_current = n_games - n_hof
    
    pop_size = _worker_pop.shape[0]
    
    wins = 0
    total = 0
    opponent_wins = defaultdict(lambda: [0, 0])
    
    def _play_one(opp, label="pop"):
        nonlocal wins, total
        agent.reset_state()
        if isinstance(opp, NeuralAgent):
            opp.reset_state()
        team1, team2 = random_team(), random_team()
        if random.random() < 0.5:
            e = BattleEngineV2(agent, opp, team1, team2)
            r = e.run(50)
            w = r["winner"] == 1
        else:
            e = BattleEngineV2(opp, agent, team2, team1)
            r = e.run(50)
            w = r["winner"] == 2
        if w:
            wins += 1
            opponent_wins[label][0] += 1
        opponent_wins[label][1] += 1
        total += 1
    
    # Games against current population (from cached shared memory)
    for _ in range(n_current):
        opp_idx = random.randint(0, pop_size - 1)
        opp = NeuralAgent(_worker_pop[opp_idx])
        _play_one(opp, "pop")
    
    # Games against HoF (from cached shared memory)
    if n_hof > 0:
        for _ in range(n_hof):
            opp_idx = random.randint(0, hof_count - 1)
            opp = NeuralAgent(_worker_hof[opp_idx])
            _play_one(opp, "hof")
    
    # Games against fixed anchor profiles (3 games each for statistical significance)
    for name, profile in ANCHOR_PROFILES:
        opp = WeightedRandomAIv2(profile)
        for _ in range(3):
            _play_one(opp, f"anchor_{name}")
    
    # Games against CounterAI (3 games)
    opp_counter = CounterAI("Counter")
    for _ in range(3):
        _play_one(opp_counter, "CounterAI")
    
    # Games against AdaptiveAI (3 games)
    opp_adaptive = AdaptiveAI()
    for _ in range(3):
        _play_one(opp_adaptive, "AdaptiveAI")
    
    raw_winrate = wins / max(total, 1)
    
    # Anti-specialization penalty: check per-anchor winrates
    anchor_penalties = 0
    for name, profile in ANCHOR_PROFILES:
        key = f"anchor_{name}"
        if opponent_wins[key][1] > 0:
            anchor_wr = opponent_wins[key][0] / opponent_wins[key][1]
            if anchor_wr < 0.20:
                anchor_penalties += 1
    
    # Penalty multiplier: each anchor below 20% costs 10% fitness
    penalty = max(0.4, 1.0 - anchor_penalties * 0.10)
    adjusted_winrate = raw_winrate * penalty
    
    # Balance penalty: if action distribution is too skewed, penalize
    # This prevents convergence to "always attack" monoculture
    # NOTE: bonus excluded from ratio — it's "preparing", not "spamming"
    total_actions = sum(agent._action_counts)
    if total_actions > 0:
        # Only consider atk/def/switch for balance (bonus is strategic, not imbalance)
        atk_def_sw = agent._action_counts[0] + agent._action_counts[1] + agent._action_counts[3]
        if atk_def_sw > 0:
            atk_ratio = agent._action_counts[0] / atk_def_sw
            def_ratio = agent._action_counts[1] / atk_def_sw
            sw_ratio = agent._action_counts[3] / atk_def_sw
            max_ratio = max(atk_ratio, def_ratio, sw_ratio)
            # If any non-bonus action > 75%, apply penalty
            if max_ratio > 0.75:
                balance_penalty = max(0.6, 1.0 - (max_ratio - 0.75) * 4.0)
                adjusted_winrate *= balance_penalty
    
    return adjusted_winrate

def evaluate_all(genomes, n_games=30, pool=None, hof_ratio=0.25):
    """Evaluate all genomes — returns list of winrates.
    Genomes and HoF are read from shared memory (no pickle overhead).
    """
    if pool is not None:
        args = [(i, n_games, i, hof_ratio) for i in range(len(genomes))]
        return pool.map(_eval_worker, args)
    # Sequential fallback: temporarily set shared memory in this process
    global _worker_pop, _worker_hof, _shm_hof_count
    from multiprocessing.shared_memory import SharedMemory as _SM
    gs = genomes.shape[1] if hasattr(genomes, 'shape') else len(genomes[0])
    _shm_pop = _SM(create=True, size=genomes.nbytes)
    _worker_pop = np.ndarray(genomes.shape, dtype=np.float32, buffer=_shm_pop.buf)
    _worker_pop[:] = np.array(genomes, dtype=np.float32)
    _shm_hof_buf = _SM(create=True, size=gs * max(1, int(len(genomes) * hof_ratio)) * 4)
    _worker_hof = np.ndarray((max(1, int(len(genomes) * hof_ratio)), gs), dtype=np.float32, buffer=_shm_hof_buf.buf)
    _shm_hof_count = mp.Value('i', 0)
    try:
        results = [_eval_worker((i, n_games, i, hof_ratio)) for i in range(len(genomes))]
    finally:
        _shm_pop.close(); _shm_pop.unlink()
        _shm_hof_buf.close(); _shm_hof_buf.unlink()
    return results


# ============================================================
# ARCHETYPE CLASSIFICATION
# ============================================================

def classify_agent(genome, n_test_games=15):
    """Play test games and return archetype + detailed metrics."""
    agent = NeuralAgent(genome, "test")
    atk_count = 0
    def_count = 0
    bon_count = 0
    sw_turns = 0
    n_turns = 0
    ds_totals = {}
    n_games = 0
    
    for _ in range(n_test_games):
        agent.reset_state()
        opp = NeuralAgent(random_genome(), "rand")
        t1, t2 = random_team(), random_team()
        if random.random() < 0.5:
            e = BattleEngineV2(agent, opp, t1, t2)
            pid = 1
        else:
            e = BattleEngineV2(opp, agent, t1, t2)
            pid = 2
        result = e.run(50)
        n_games += 1
        for t in e.turn_logs:
            if t.player_id == pid:
                atk_count += t.attack_actions
                def_count += t.defend_actions
                bon_count += t.bonus_actions
                if t.switched:
                    sw_turns += 1
                n_turns += 1
        # Aggregate detailed stats (per-player)
        ds = result.get(f"detailed_stats_p{pid}", {})
        for k, v in ds.items():
            ds_totals[k] = ds_totals.get(k, 0) + v
    
    total = atk_count + def_count + bon_count
    if total == 0:
        return {"archetype": "Unknown", "atk": 0, "def": 0, "bon": 0,
                "sw_per_game": 0, "overkill": 0, "missed_lethal": 0,
                "punished_greed": 0, "swap_value": 0, "swap_ev": 0, "resource_eff": 0, "n_switches": 0}
    
    atk_r = atk_count / total
    def_r = def_count / total
    bon_r = bon_count / total
    sw_per_game = sw_turns / max(n_test_games, 1)
    
    # Detailed metrics
    swap_value_total = ds_totals.get("swap_value", 0)
    n_sw = ds_totals.get("n_switches", 0)
    overkill_avg = ds_totals.get("overkill", 0) / max(1, n_games)
    swap_ev = swap_value_total / max(1, n_sw) if n_sw > 0 else 0
    missed_lethal_avg = ds_totals.get("missed_lethal", 0) / n_games
    punished_greed_avg = ds_totals.get("punished_greed", 0) / n_games
    resource_eff = ds_totals.get("resource_efficiency", 0) / n_games
    
    # Archetype classification
    if bon_r > 0.4:
        arch = "Burst"
    elif def_r > 0.35:
        arch = "Defender"
    elif atk_r > 0.8:
        arch = "AllIn"
    elif atk_r > 0.65:
        arch = "Aggro"
    elif def_r > 0.2 and atk_r < 0.5:
        arch = "Control"
    else:
        arch = "Balanced"
    
    return {"archetype": arch, "atk": atk_r, "def": def_r, "bon": bon_r,
            "sw_per_game": sw_per_game,
            "overkill": overkill_avg, "missed_lethal": missed_lethal_avg,
            "punished_greed": punished_greed_avg, "swap_value": swap_value_total,
            "swap_ev": swap_ev, "resource_eff": resource_eff, "n_switches": n_sw}


# ============================================================
# SMART AGENT EVALUATION (replaces LSTM eval)
# ============================================================

def smart_classify_agent(genome, n_test_games=15):
    """Classify SmartNeuralAgent by playing test games. Returns rich behavioral metrics."""
    agent = SmartNeuralAgent(genome, "test")
    atk_count = 0; def_count = 0; bon_count = 0
    sw_turns = 0; n_turns = 0
    ds_totals = {}; n_games = 0
    game_lengths = []
    wins_as_p1 = wins_as_p2 = games_as_p1 = games_as_p2 = 0
    comebacks = behind_situations = 0  # won from being behind in characters
    burst_attempts = burst_defenses = 0
    action_seq = []  # per-turn (atk, def, bon) across all games

    for _ in range(n_test_games):
        agent.reset_state()
        opp = SmartNeuralAgent(random_smart_genome(), "rand")
        t1, t2 = random_team(), random_team()
        if np.random.random() < 0.5:
            e = BattleEngineV2(agent, opp, t1, t2); pid = 1; games_as_p1 += 1
        else:
            e = BattleEngineV2(opp, agent, t1, t2); pid = 2; games_as_p2 += 1
        result = e.run(50)
        n_games += 1
        game_lengths.append(result["turns"])

        my_turn_logs = []
        for t in e.turn_logs:
            if t.player_id == pid:
                atk_count += t.attack_actions
                def_count += t.defend_actions
                bon_count += t.bonus_actions
                if t.switched: sw_turns += 1
                n_turns += 1
                my_turn_logs.append(t)
                action_seq.append((t.attack_actions, t.defend_actions, t.bonus_actions))

        if result["winner"] == pid:
            if pid == 1: wins_as_p1 += 1
            else: wins_as_p2 += 1

        ds = result.get(f"detailed_stats_p{pid}", {})
        for k, v in ds.items():
            ds_totals[k] = ds_totals.get(k, 0) + v

    total = atk_count + def_count + bon_count
    if total == 0:
        return {"archetype": "Unknown", "atk": 0, "def": 0, "bon": 0,
                "sw_per_game": 0, "overkill": 0, "missed_lethal": 0,
                "punished_greed": 0, "swap_value": 0, "swap_ev": 0,
                "resource_eff": 0, "n_switches": 0, "n_games": 0,
                "games_as_p1": 0, "games_as_p2": 0,
                "wins_as_p1": 0, "wins_as_p2": 0,
                "game_len_avg": 0, "game_len_min": 0, "game_len_max": 0,
                "comeback_rate": 0, "burst_rate": 0, "defense_rate": 0,
                "pattern_bonbon": 0, "pattern_bonatk": 0, "pattern_atkatk": 0, "pattern_atkdef": 0}

    atk_r = atk_count / total; def_r = def_count / total; bon_r = bon_count / total
    sw_per_game = sw_turns / max(n_test_games, 1)
    swap_value_total = ds_totals.get("swap_value", 0)
    n_sw = ds_totals.get("n_switches", 0)
    swap_ev = swap_value_total / max(1, n_sw) if n_sw > 0 else 0

    avg_len = np.mean(game_lengths) if game_lengths else 0
    min_len = min(game_lengths) if game_lengths else 0
    max_len = max(game_lengths) if game_lengths else 0

    # Turn-order advantage
    p1_wr = wins_as_p1 / max(games_as_p1, 1)
    p2_wr = wins_as_p2 / max(games_as_p2, 1)

    # Action N-gram patterns: compute bigrams from action_seq
    bigrams = defaultdict(int)
    for i in range(1, len(action_seq)):
        prev = action_seq[i-1]
        cur = action_seq[i]
        # Classify each turn's dominant action
        def classify_turn(a, d, b):
            if a >= d and a >= b and a > 0: return "A"
            if d >= a and d >= b and d > 0: return "D"
            return "B"
        p = classify_turn(*prev)
        c = classify_turn(*cur)
        bigrams[p + c] += 1

    total_bg = sum(bigrams.values()) or 1
    pat_bonbon = bigrams.get("BB", 0) / total_bg
    pat_bonatk = (bigrams.get("BA", 0) + bigrams.get("AB", 0)) / total_bg
    pat_atkatk = bigrams.get("AA", 0) / total_bg
    pat_atkdef = (bigrams.get("AD", 0) + bigrams.get("DA", 0)) / total_bg

    # Archetype
    if bon_r > 0.4: arch = "Burst"
    elif def_r > 0.35: arch = "Defender"
    elif atk_r > 0.8: arch = "AllIn"
    elif atk_r > 0.65: arch = "Aggro"
    elif def_r > 0.2 and atk_r < 0.5: arch = "Control"
    else: arch = "Balanced"

    return {"archetype": arch, "atk": atk_r, "def": def_r, "bon": bon_r,
            "sw_per_game": sw_per_game,
            "overkill": ds_totals.get("overkill", 0) / max(1, n_games),
            "missed_lethal": ds_totals.get("missed_lethal", 0) / max(1, n_games),
            "punished_greed": ds_totals.get("punished_greed", 0) / max(1, n_games),
            "swap_value": swap_value_total,
            "swap_ev": swap_ev,
            "resource_eff": ds_totals.get("resource_efficiency", 0) / max(1, n_games),
            "n_switches": n_sw, "n_games": n_games,
            "games_as_p1": games_as_p1, "games_as_p2": games_as_p2,
            "wins_as_p1": wins_as_p1, "wins_as_p2": wins_as_p2,
            "p1_wr": p1_wr, "p2_wr": p2_wr,
            "game_len_avg": avg_len, "game_len_min": min_len, "game_len_max": max_len,
            "pattern_bonbon": pat_bonbon, "pattern_bonatk": pat_bonatk,
            "pattern_atkatk": pat_atkatk, "pattern_atkdef": pat_atkdef}


def _smart_eval_worker(args):
    """Worker for parallel smart agent evaluation — receives genome + opponents directly."""
    (genome_idx, genome, n_games, seed, hof_ratio, reference_games,
     population, hall_of_fame) = args
    random.seed(seed)
    np.random.seed(seed)

    agent = SmartNeuralAgent(genome)

    hof_size = len(hall_of_fame) if hall_of_fame else 0
    n_hof = int(n_games * hof_ratio) if hof_size > 0 else 0
    n_current = n_games - n_hof
    pop_size = len(population)

    wins = 0; total = 0
    opponent_wins = defaultdict(lambda: [0, 0])

    def _play_one(opp, label="pop"):
        nonlocal wins, total
        agent.reset_state()
        if hasattr(opp, "reset_state"):
            opp.reset_state()
        team1, team2 = random_team(), random_team()
        if np.random.random() < 0.5:
            e = BattleEngineV2(agent, opp, team1, team2)
            r = e.run(50)
            w = r["winner"] == 1
        else:
            e = BattleEngineV2(opp, agent, team2, team1)
            r = e.run(50)
            w = r["winner"] == 2
        if w: wins += 1; opponent_wins[label][0] += 1
        opponent_wins[label][1] += 1
        total += 1

    for _ in range(n_current):
        opp = SmartNeuralAgent(population[random.randint(0, pop_size - 1)])
        _play_one(opp, "pop")

    if n_hof > 0:
        for _ in range(n_hof):
            opp = SmartNeuralAgent(hall_of_fame[random.randint(0, hof_size - 1)])
            _play_one(opp, "hof")

    for name, profile in ANCHOR_PROFILES:
        opp = WeightedRandomAIv2(profile)
        for _ in range(reference_games):
            _play_one(opp, f"anchor_{name}")

    opp_counter = CounterAI("Counter")
    for _ in range(reference_games): _play_one(opp_counter, "CounterAI")

    opp_adaptive = AdaptiveAI()
    for _ in range(reference_games): _play_one(opp_adaptive, "AdaptiveAI")

    opp_phase = PhaseShiftAI()
    for _ in range(reference_games): _play_one(opp_phase, "PhaseShift")

    raw_winrate = wins / max(total, 1)

    # A champion must be robust, not merely exploit the average self-play agent.
    # The lowest three anchor matchups receive a large weight in fitness.
    reference_keys = [f"anchor_{name}" for name, _ in ANCHOR_PROFILES] + [
        "CounterAI", "AdaptiveAI", "PhaseShift"]
    reference_rates = [opponent_wins[key][0] / opponent_wins[key][1]
                       for key in reference_keys if opponent_wins[key][1] > 0]
    weak_matchups = sorted(reference_rates)[:3]
    robust_rate = sum(weak_matchups) / max(1, len(weak_matchups))
    adjusted_winrate = 0.60 * raw_winrate + 0.40 * robust_rate

    # A one-dimensional action policy has no credible response to hidden intent.
    # This is deliberately a soft penalty: tactical all-in is legal when needed,
    # but a whole evaluation suite of all-in choices is not champion material.
    total_actions = sum(agent._action_counts[:3])
    if total_actions > 0:
        max_action_ratio = max(agent._action_counts[:3]) / total_actions
        if max_action_ratio > 0.90:
            adjusted_winrate *= max(0.65, 1.0 - (max_action_ratio - 0.90) * 3.5)

    return genome_idx, adjusted_winrate


def smart_evaluate_all(genomes, n_games=30, pool=None, hof_ratio=0.25,
                       reference_games=8, hall_of_fame=None, desc=None):
    """Evaluate all smart agent genomes — passes genomes directly (no shared memory)."""
    hall_of_fame = hall_of_fame or []
    if pool is not None:
        n_total = len(genomes)
        results = [0.0] * n_total
        n_workers = getattr(pool, '_processes', 16)
        batch_size = max(1, n_workers * 2)
        batches = [range(i, min(i + batch_size, n_total)) for i in range(0, n_total, batch_size)]
        with tqdm(total=n_total, desc=desc, leave=False) as pbar:
            for batch in batches:
                batch_args = [(i, genomes[i], n_games, i, hof_ratio,
                               reference_games, genomes, hall_of_fame) for i in batch]
                raw = pool.map(_smart_eval_worker, batch_args)
                for idx, fit in raw:
                    results[idx] = fit
                pbar.update(len(batch))
        return results
    # Single-process fallback
    raw = [_smart_eval_worker((i, genomes[i], n_games, i, hof_ratio,
                               reference_games, genomes, hall_of_fame))
           for i in range(len(genomes))]
    results = [0.0] * len(genomes)
    for idx, fit in raw:
        results[idx] = fit
    return results


# ============================================================
# CO-EVOLUTION
# ============================================================

def _record_battle(genome, gen_num, timestamp, save_dir):
    """Record a full battle between the genome and a random opponent for later review.
    Saves detailed turn-by-turn log to a file."""
    agent = NeuralAgent(genome, "champion")
    opp = NeuralAgent(random_genome(), "random")
    t1, t2 = random_team(), random_team()
    e = BattleEngineV2(agent, opp, t1, t2)
    result = e.run(50)
    
    # Build readable battle log
    lines = []
    lines.append(f"Battle: Gen {gen_num}")
    lines.append(f"Teams: P1={[c.char_type.value for c in e.p1.characters]}")
    lines.append(f"       P2={[c.char_type.value for c in e.p2.characters]}")
    lines.append(f"Winner: Player {result['winner']}")
    lines.append(f"Turns: {result['turns']}")
    lines.append("")
    
    for log in e.full_turn_logs:
        p_tag = f"P{log.player_id}"
        atk_str = f"atk={log.attack_actions}" if log.attack_actions else ""
        def_str = f"def={log.defend_actions}" if log.defend_actions else ""
        bon_str = f"bon={log.bonus_actions}" if log.bonus_actions else ""
        sw_str = "SWITCH" if log.switched else ""
        parts = [x for x in [atk_str, def_str, bon_str, sw_str] if x]
        action_str = ", ".join(parts) if parts else "none"
        
        dmg_str = f"dmg={log.total_damage}" if log.total_damage else ""
        block_str = f"blocked={log.blocked_shields}/{log.opponent_shields}" if log.opponent_shields else ""
        shield_str = f"my_shields={log.player_shields_before}" if log.player_shields_before else ""
        
        detail_parts = [x for x in [dmg_str, block_str, shield_str] if x]
        detail_str = " | ".join(detail_parts) if detail_parts else ""
        
        hp_p1 = log.p1_hp
        hp_p2 = log.p2_hp
        
        line = f"  T{log.turn_num:2d} {p_tag}: {action_str}"
        if detail_str:
            line += f" [{detail_str}]"
        line += f"  HP: P1={hp_p1} P2={hp_p2}"
        lines.append(line)
    
    battle_text = "\n".join(lines)
    
    # Save to file
    filename = f"battle_gen{gen_num:04d}_{timestamp}.txt"
    filepath = os.path.join(save_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(battle_text)
    
    # Return data for JSON
    turns = []
    for log in e.turn_logs:
        turns.append({
            "turn": log.turn_num,
            "player": log.player_id,
            "attacks": log.attack_actions,
            "defends": log.defend_actions,
            "bonuses": log.bonus_actions,
            "switched": log.switched,
            "damage": log.total_damage,
            "unblocked": log.unblocked_attacks,
            "blocked": log.blocked_shields,
            "opp_shields": log.opponent_shields,
            "my_shields_before": log.player_shields_before,
        })
    
    return {
        "gen": gen_num,
        "winner": result["winner"],
        "total_turns": result["turns"],
        "p1_team": [c.char_type.value for c in e.p1.characters],
        "p2_team": [c.char_type.value for c in e.p2.characters],
        "p1_final_hp": [c.hp for c in e.p1.characters],
        "p2_final_hp": [c.hp for c in e.p2.characters],
        "turns": turns,
        "file": filename,
    }


def _compute_facts(gen_stats, battle_records):
    """Compute interesting patterns from collected stats."""
    facts = []
    
    if not gen_stats:
        return facts
    
    # Fitness trajectory
    fitnesses = [g["best_fitness"] for g in gen_stats]
    avg_fitnesses = [g["avg_fitness"] for g in gen_stats]
    
    # Best ever fitness
    best_idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
    facts.append(f"Best fitness: {fitnesses[best_idx]:.1%} at generation {gen_stats[best_idx]['gen']}")
    
    # Biggest single-gen improvement
    if len(fitnesses) > 1:
        improvements = [fitnesses[i] - fitnesses[i-1] for i in range(1, len(fitnesses))]
        best_imp_idx = max(range(len(improvements)), key=lambda i: improvements[i])
        facts.append(f"Biggest jump: +{improvements[best_imp_idx]:.1%} at gen {gen_stats[best_imp_idx+1]['gen']}")
    
    # Fitness plateau detection: longest stretch with <0.5% change
    if len(fitnesses) > 10:
        changes = [abs(fitnesses[i] - fitnesses[i-1]) for i in range(1, len(fitnesses))]
        plateaus = []
        cur_plateau = 0
        for c in changes:
            if c < 0.005:
                cur_plateau += 1
            else:
                if cur_plateau > 0:
                    plateaus.append(cur_plateau)
                cur_plateau = 0
        if cur_plateau > 0:
            plateaus.append(cur_plateau)
        if plateaus:
            facts.append(f"Longest plateau: {max(plateaus)} generations with <0.5% change")
    
    # Phase transitions
    phases_seen = []
    for g in gen_stats:
        p = g.get("phase", "")
        if p and (not phases_seen or phases_seen[-1] != p):
            phases_seen.append(p)
    if phases_seen:
        facts.append(f"Adaptive phases: {' -> '.join(phases_seen)}")
    
    # Fitness at phase transitions
    phase_changes = []
    for i in range(1, len(gen_stats)):
        prev_phase = gen_stats[i-1].get("phase", "")
        cur_phase = gen_stats[i].get("phase", "")
        if cur_phase and prev_phase != cur_phase:
            phase_changes.append((gen_stats[i]["gen"], cur_phase, gen_stats[i-1]["best_fitness"], gen_stats[i]["best_fitness"]))
    for gen, phase, before, after in phase_changes:
        facts.append(f"Phase {phase} at gen {gen}: fitness {before:.1%} -> {after:.1%}")
    
    # Archetype diversity (Shannon entropy) over snapshots
    snapshots_with_arch = [g for g in gen_stats if "archetype_dist" in g]
    if snapshots_with_arch:
        entropies = []
        for g in snapshots_with_arch:
            dist = g["archetype_dist"]
            total = sum(dist.values())
            probs = [v/total for v in dist.values() if v > 0]
            entropy = -sum(p * math.log(p) for p in probs if p > 0)
            entropies.append((g["gen"], entropy))
        
        max_ent = max(entropies, key=lambda x: x[1])
        min_ent = min(entropies, key=lambda x: x[1])
        facts.append(f"Highest diversity: gen {max_ent[0]} (entropy={max_ent[1]:.2f})")
        facts.append(f"Lowest diversity: gen {min_ent[0]} (entropy={min_ent[1]:.2f})")
        
        # Dominant archetype trajectory
        dominant = []
        for g in snapshots_with_arch:
            dist = g["archetype_dist"]
            best_arch = max(dist, key=dist.get)
            dominant.append((g["gen"], best_arch, dist[best_arch]))
        
        # Archetype shifts
        for i in range(1, len(dominant)):
            if dominant[i][1] != dominant[i-1][1]:
                facts.append(f"Archetype shift at gen {dominant[i][0]}: {dominant[i-1][1]} ({dominant[i-1][2]:.0%}) -> {dominant[i][1]} ({dominant[i][2]:.0%})")
    
    # Battle statistics
    if battle_records:
        total_turns = sum(b["total_turns"] for b in battle_records)
        avg_turns = total_turns / len(battle_records)
        champion_wins = sum(1 for b in battle_records if b["winner"] == 1)
        facts.append(f"Recorded battles: {len(battle_records)}, champion winrate: {champion_wins}/{len(battle_records)} ({champion_wins/len(battle_records):.0%})")
        facts.append(f"Average battle length: {avg_turns:.1f} turns")
        
        # Shortest and longest battles
        shortest = min(battle_records, key=lambda b: b["total_turns"])
        longest = max(battle_records, key=lambda b: b["total_turns"])
        facts.append(f"Shortest battle: gen {shortest['gen']} ({shortest['total_turns']} turns)")
        facts.append(f"Longest battle: gen {longest['gen']} ({longest['total_turns']} turns)")
    
    # Action ratio trends (from snapshots)
    if snapshots_with_arch:
        first_snap = snapshots_with_arch[0]
        last_snap = snapshots_with_arch[-1]
        if "avg_atk_pct" in first_snap:
            facts.append(f"Atk% trend: {first_snap['avg_atk_pct']:.0%} -> {last_snap['avg_atk_pct']:.0%}")
            facts.append(f"Def% trend: {first_snap['avg_def_pct']:.0%} -> {last_snap['avg_def_pct']:.0%}")
            facts.append(f"Bon% trend: {first_snap['avg_bon_pct']:.0%} -> {last_snap['avg_bon_pct']:.0%}")
            facts.append(f"Sw/game trend: {first_snap['avg_sw_per_game']:.2f} -> {last_snap['avg_sw_per_game']:.2f}")
    
    return facts


def run_coevolution(pop_size=500, generations=100, games_per_eval=30,
                    elite_frac=0.05, mut_rate=0.10, mut_sigma=0.12,
                    n_jobs=0, verbose=True,
                    hof_add=10, hof_ratio=0.25, hof_max=200,
                    snapshot_interval=10, adaptive=False):
    """Run co-evolution of neural agents (parallel with multiprocessing).
    
    hof_add: top N agents per generation added to Hall of Fame
    hof_ratio: fraction of eval games played against Hall of Fame
    hof_max: max size of Hall of Fame
    snapshot_interval: record archetype distribution every N generations
    adaptive: if True, use 3-phase mutation schedule (explore/exploit/polish)
    """
    # Setup timestamp and battle log directory
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    battle_log_dir = os.path.join(_this_dir, "parameterized_ai", f"battles_{run_timestamp}")
    os.makedirs(battle_log_dir, exist_ok=True)
    
    n_jobs = max(1, mp.cpu_count() if n_jobs == 0 else n_jobs)
    n_anchors = len(ANCHOR_PROFILES) + 6  # 6 anchors (with BonusBanker) + 3 CounterAI + 3 AdaptiveAI
    total_games_per = games_per_eval + n_anchors
    total_battles = pop_size * generations * total_games_per
    extra = f"+HoF({hof_ratio:.0%}) +{n_anchors}anchors"
    print(f"Co-evolution: pop={pop_size}, gens={generations}, "
          f"games={total_games_per}{' ('+extra+')' if hof_add>0 else ''}"
          f"{' adaptive' if adaptive else ''}", flush=True)
    print(f"Genome size: {genome_size()} params", flush=True)
    print(f"Workers: {n_jobs}", flush=True)
    print(f"Estimated battles: {total_battles:,}", flush=True)
    est_s = total_battles * 0.0012 / n_jobs  # empirical ~1.2ms per game
    print(f"Estimated time: {est_s:.0f}s ({est_s/60:.1f}m)", flush=True)
    print(flush=True)
    
    # Initialize population and Hall of Fame
    population = [random_genome(seed=i) for i in range(pop_size)]
    hall_of_fame = []
    fitness_history = []
    meta_history = []  # (gen, snapshot)
    gen_stats = []     # per-generation data
    battle_records = [] # recorded battles at snapshot intervals
    
    t0 = time.perf_counter()
    
    # Set up shared memory for population genomes and HoF
    gs = genome_size()
    pop_arr = np.array(population, dtype=np.float32)  # (pop_size, gs)
    shm_pop = SharedMemory(create=True, size=pop_arr.nbytes)
    shm_pop_view = np.ndarray(pop_arr.shape, dtype=np.float32, buffer=shm_pop.buf)
    shm_pop_view[:] = pop_arr
    
    hof_arr = np.zeros((hof_max, gs), dtype=np.float32)
    shm_hof = SharedMemory(create=True, size=hof_arr.nbytes)
    shm_hof_view = np.ndarray(hof_arr.shape, dtype=np.float32, buffer=shm_hof.buf)
    shm_hof_view[:] = hof_arr
    
    hof_count = mp.Value('i', 0)
    
    pool = mp.Pool(n_jobs, initializer=_init_shared_worker,
                   initargs=(shm_pop.name, pop_arr.shape,
                             shm_hof.name, hof_arr.shape, hof_count)) if n_jobs > 1 else None
    
    try:
        for gen in range(generations):
            gen_t0 = time.perf_counter()
            
            # Adaptive parameter schedule (tuned for LSTM 1704 params, 300 gens)
            cur_mut_rate = mut_rate
            cur_mut_sigma = mut_sigma
            cur_elite_frac = elite_frac
            cur_tournament_k = 4
            cur_hof_add = hof_add
            phase_name = ""
            if adaptive:
                if gen < 80:
                    cur_mut_rate = 0.18
                    cur_mut_sigma = 0.28
                    cur_elite_frac = 0.03
                    cur_tournament_k = 3
                    cur_hof_add = 15
                    phase_name = "EXPLORE"
                elif gen < 320:
                    cur_mut_rate = 0.10
                    cur_mut_sigma = 0.14
                    cur_elite_frac = 0.05
                    cur_tournament_k = 4
                    cur_hof_add = 12
                    phase_name = "EXPLOIT"
                else:
                    cur_mut_rate = 0.05
                    cur_mut_sigma = 0.08
                    cur_elite_frac = 0.08
                    cur_tournament_k = 5
                    cur_hof_add = 8
                    phase_name = "POLISH"
            
            # Update shared memory with current population
            shm_pop_view[:] = np.array(population, dtype=np.float32)
            
            # Update shared memory with current HoF
            if hall_of_fame:
                hof_n = min(len(hall_of_fame), hof_max)
                hof_data = np.zeros((hof_max, gs), dtype=np.float32)
                hof_data[:hof_n] = np.array(hall_of_fame[:hof_n], dtype=np.float32)
                shm_hof_view[:] = hof_data
                hof_count.value = hof_n
            else:
                hof_count.value = 0
            
            # Evaluate (genomes/HoF read from shared memory)
            fitnesses = evaluate_all(population, games_per_eval, pool=pool,
                                     hof_ratio=hof_ratio)
            
            # Stats
            best_f = max(fitnesses)
            avg_f = np.mean(fitnesses)
            fitness_history.append((best_f, avg_f))
            
            # Sort by fitness
            indexed = list(zip(range(pop_size), fitnesses, population))
            indexed.sort(key=lambda x: -x[1])
            
            # Record per-generation stats
            gen_stat = {
                "gen": gen + 1,
                "best_fitness": float(best_f),
                "avg_fitness": float(avg_f),
                "phase": phase_name,
                "mut_rate": cur_mut_rate,
                "mut_sigma": cur_mut_sigma,
                "elite_frac": cur_elite_frac,
                "tournament_k": cur_tournament_k,
                "hof_size": len(hall_of_fame) if hall_of_fame else 0,
            }
            gen_stats.append(gen_stat)
            
            # Create next generation
            next_pop = []
            
            # Elitism: keep top elite_frac
            n_elite = max(2, int(pop_size * cur_elite_frac))
            for _, _, genome in indexed[:n_elite]:
                next_pop.append(genome.copy())
            
            # Fill rest by tournament selection + crossover + mutation
            while len(next_pop) < pop_size:
                p1_idx = tournament_select(fitnesses, k=cur_tournament_k)
                p2_idx = tournament_select(fitnesses, k=cur_tournament_k)
                g1, g2 = population[p1_idx], population[p2_idx]
                
                child = crossover(g1, g2)
                child = mutate(child, rate=cur_mut_rate, sigma=cur_mut_sigma)
                next_pop.append(child)
            
            # Add top CURRENT agents to Hall of Fame (with deduplication)
            if hall_of_fame is not None and cur_hof_add > 0:
                top_idx = np.argsort(fitnesses)[-cur_hof_add:]
                for idx in top_idx:
                    g = population[idx].copy()
                    if _hof_is_novel(g, hall_of_fame):
                        hall_of_fame.append(g)
                if len(hall_of_fame) > hof_max:
                    hall_of_fame = hall_of_fame[-hof_max:]
            
            population = next_pop
            
            gen_t = time.perf_counter() - gen_t0
            
            if verbose:
                phase_tag = f" [{phase_name}]" if adaptive else ""
                print(f"  Gen {gen+1:3d}/{generations} | "
                      f"best={best_f:.1%} avg={avg_f:.1%} | "
                      f"{gen_t:.1f}s{phase_tag}", flush=True)
            
            # Meta snapshot every N generations
            if (gen + 1) % snapshot_interval == 0 or gen + 1 == generations:
                sample = random.sample(population, min(100, pop_size))
                arch_counts = defaultdict(int)
                totals = {"atk": 0, "def_": 0, "bon": 0, "sw": 0.0,
                          "overkill": 0, "missed_lethal": 0, "punished_greed": 0,
                          "swap_value": 0, "n_switches": 0, "resource_eff": 0}
                for g in sample:
                    info = classify_agent(g, n_test_games=5)
                    arch_counts[info["archetype"]] += 1
                    totals["atk"] += info["atk"]
                    totals["def_"] += info["def"]
                    totals["bon"] += info["bon"]
                    totals["sw"] += info["sw_per_game"]
                    totals["overkill"] += info["overkill"]
                    totals["missed_lethal"] += info["missed_lethal"]
                    totals["punished_greed"] += info["punished_greed"]
                    totals["swap_value"] += info["swap_value"]
                    totals["resource_eff"] += info["resource_eff"]
                    totals["n_switches"] += info["n_switches"]
                n = len(sample)
                meta_history.append((gen + 1, dict(arch_counts), n))
                
                # Store snapshot in gen_stats
                gen_stat["archetype_dist"] = {a: c/n for a, c in arch_counts.items()}
                gen_stat["avg_atk_pct"] = float(totals["atk"]/n)
                gen_stat["avg_def_pct"] = float(totals["def_"]/n)
                gen_stat["avg_bon_pct"] = float(totals["bon"]/n)
                gen_stat["avg_sw_per_game"] = float(totals["sw"]/n)
                gen_stat["avg_overkill"] = float(totals["overkill"]/n)
                gen_stat["avg_missed_lethal"] = float(totals["missed_lethal"]/n)
                gen_stat["avg_resource_eff"] = float(totals["resource_eff"]/n)
                gen_stat["avg_punished_greed"] = float(totals["punished_greed"]/n)
                
                # Record a battle: top genome vs random genome
                best_genome = indexed[0][2]
                battle_log = _record_battle(best_genome, gen+1, run_timestamp, battle_log_dir)
                battle_records.append(battle_log)
                
                # Mini-summary
                print(f"  --- Pop snapshot (n={n}) ---")
                arch_line = ", ".join(f"{a}={c/n:.0%}" for a, c in sorted(arch_counts.items(), key=lambda x: -x[1]))
                print(f"  Arch: {arch_line}")
                sw_ev_avg = totals["swap_value"] / max(1, totals["n_switches"])
                print(f"  Atk={totals['atk']/n:.0%} Def={totals['def_']/n:.0%} Bon={totals['bon']/n:.0%} "
                      f"Sw={totals['sw']/n:.2f}/g SwEV={sw_ev_avg:+.0f}")
                print(f"  OverAct={totals['overkill']/n:.1f} MissLeth={totals['missed_lethal']/n:.2f}/g "
                      f"ResEff={totals['resource_eff']/n:.0f} Greed={totals['punished_greed']/n:.2f}/g")
                # Top 3 best
                top3 = indexed[:3]
                print(f"  Best3: ", end="")
                for rank, (idx, fit, _) in enumerate(top3):
                    ti = classify_agent(population[idx], n_test_games=5)
                    sw_ev = ti.get("swap_ev", 0)
                    print(f"#{rank+1} fit={fit:.1%} "
                          f"atk={ti['atk']:.0%} def={ti['def']:.0%} bon={ti['bon']:.0%} "
                          f"sw={ti['sw_per_game']:.1f}/g ev={sw_ev:+.0f}", end=" | ")
                print()
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        shm_pop.close()
        shm_hof.close()
    
    total_t = time.perf_counter() - t0
    print(f"\nTotal time: {total_t:.0f}s ({total_t/60:.1f}m)")
    
    # Compute interesting facts
    facts = _compute_facts(gen_stats, battle_records)
    
    # Print meta evolution summary
    if len(meta_history) > 1:
        print(f"\n--- Meta evolution ---")
        all_archs = set()
        for _, snap, _ in meta_history:
            all_archs.update(snap.keys())
        all_archs = sorted(all_archs)
        header = f"  {'Gen':>4s}  " + "  ".join(f"{a:>10s}" for a in all_archs)
        print(header)
        print(f"  {'-'*4}  " + "  ".join(f"{'-'*10}" for _ in all_archs))
        for gen, snap, n in meta_history:
            vals = "  ".join(f"{snap.get(a,0)/n:>10.0%}" for a in all_archs)
            print(f"  {gen:4d}  {vals}")
    
    # Print interesting facts
    if facts:
        print(f"\n--- Interesting Facts ---")
        for f in facts:
            print(f"  {f}")
    
    # Return final population, fitness, meta, stats, battles, facts
    return population, fitness_history, meta_history, gen_stats, battle_records, facts


# ============================================================
# ANALYSIS
# ============================================================

def analyze_population(population, n_sample=30):
    """Classify agents and report archetype distribution + behavioral metrics.
    Auto-detects LSTM (3912 params) vs Smart (12 params) genome."""
    print("\nClassifying agents...")
    
    if len(population) > n_sample:
        sample = random.sample(population, n_sample)
    else:
        sample = population

    use_smart = len(sample[0]) == SMART_GENOME_SIZE if len(sample) > 0 else False
    classify = smart_classify_agent if use_smart else classify_agent
    
    # Aggregate per archetype
    arch_agents = defaultdict(list)
    for genome in sample:
        info = classify(genome)
        arch_agents[info["archetype"]].append(info)
    
    print(f"\nArchetype distribution (n={len(sample)}):")
    print(f"  {'Archetype':12s} {'Count':6s} {'ATK':7s} {'DEF':7s} {'BON':7s} {'SW/g':6s} "
          f"{'SwEV':7s} {'OverAct':8s} {'MissL':7s} {'ResEff':7s}")
    print(f"  {'-'*12} {'-'*6} {'-'*7} {'-'*7} {'-'*7} {'-'*6} "
          f"{'-'*7} {'-'*8} {'-'*7} {'-'*7}")
    for arch in sorted(arch_agents.keys(), key=lambda a: -len(arch_agents[a])):
        agents = arch_agents[arch]
        n = len(agents)
        avg_atk = sum(a["atk"] for a in agents) / n
        avg_def = sum(a["def"] for a in agents) / n
        avg_bon = sum(a["bon"] for a in agents) / n
        avg_sw = sum(a["sw_per_game"] for a in agents) / n
        avg_over = sum(a["overkill"] for a in agents) / n
        avg_miss = sum(a["missed_lethal"] for a in agents) / n
        avg_ev = sum(a["swap_ev"] for a in agents) / n
        avg_eff = sum(a["resource_eff"] for a in agents) / n
        pct = n / len(sample) * 100
        print(f"  {arch:12s} {n:3d} ({pct:2.0f}%)  {avg_atk:.0%}  {avg_def:.0%}  {avg_bon:.0%}  {avg_sw:.1f}  "
              f"{avg_ev:+6.0f}  {avg_over:6.1f}   {avg_miss:.2f}  {avg_eff:5.0f}")
    
    return arch_agents


# ============================================================
# SMART CO-EVOLUTION (tiny genome + opponent model)
# ============================================================

def _smart_record_battle(genome, gen_num, timestamp, save_dir):
    """Record a readable battle report matching the play screen vocabulary."""
    agent = SmartNeuralAgent(genome, "champion")
    opp = SmartNeuralAgent(random_smart_genome(), "random")
    t1, t2 = random_team(), random_team()
    e = BattleEngineV2(agent, opp, t1, t2)
    result = e.run(50)

    def team_lines(chars, hp_values, active, stack_order):
        return [
            f"    {'>' if i == active else ' '} [{chars[i].char_type.value}]#{i + 1} "
            f"HP={hp_values[i]} ATK={chars[i].atk}"
            for i in stack_order]

    def action_line(log):
        parts = []
        if log.attack_actions:
            blocked = log.blocked_shields
            hit = log.unblocked_attacks
            parts.append(f"attack {log.attack_actions}x ({blocked} blocked, {hit} hit)")
        if log.defend_actions:
            parts.append(f"shield {log.defend_actions}")
        if log.bonus_actions:
            parts.append(f"bonus +{log.bonus_actions}")
        if log.switched:
            parts.append("switch")
        return ", ".join(parts) if parts else "pass"

    lines = []
    lines.append("=" * 72)
    lines.append(f"COTE MEGAVERSE | EVOLUTION BATTLE | GENERATION {gen_num}")
    lines.append("=" * 72)
    lines.append(f"P1 champion: {[c.char_type.value for c in e.p1.characters]}")
    lines.append(f"P2 opponent: {[c.char_type.value for c in e.p2.characters]}")
    result_text = "draw" if result["winner"] == 0 else f"Player {result['winner']} wins"
    lines.append(f"Result: {result_text}")
    lines.append(f"Turns: {result['turns']}")
    lines.append("")
    for log in e.turn_logs:
        p1_active = log.p1_active if log.p1_active >= 0 else 0
        p2_active = log.p2_active if log.p2_active >= 0 else 0
        lines.append(f"TURN {log.turn_num:02d} | Player {log.player_id}")
        lines.append(f"  Before P1: {log.p1_hp} | P2: {log.p2_hp}")
        p1_stack = log.p1_stack or list(range(len(e.p1.characters)))
        p2_stack = log.p2_stack or list(range(len(e.p2.characters)))
        lines.append("  State P1:")
        lines.extend(team_lines(e.p1.characters, log.p1_hp, p1_active, p1_stack))
        lines.append("  State P2:")
        lines.extend(team_lines(e.p2.characters, log.p2_hp, p2_active, p2_stack))
        lines.append(f"  Action: {action_line(log)}")
        lines.append(f"  Damage: {log.total_damage} | opponent shields: {log.opponent_shields} | own shields before: {log.player_shields_before}")
        lines.append(f"  After  P1: {log.p1_hp_after} | P2: {log.p2_hp_after} "
                     f"(active P1=#{log.p1_active_after + 1}, P2=#{log.p2_active_after + 1})")
        lines.append("")

    battle_text = "\n".join(lines)
    filename = f"battle_gen{gen_num:04d}_{timestamp}.txt"
    filepath = os.path.join(save_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(battle_text)

    turns = [{"turn": t.turn_num, "player": t.player_id,
              "attacks": t.attack_actions, "defends": t.defend_actions,
              "bonuses": t.bonus_actions, "switched": t.switched,
            "damage": t.total_damage, "unblocked": t.unblocked_attacks,
            "blocked": t.blocked_shields, "opp_shields": t.opponent_shields,
            "p1_hp": t.p1_hp, "p2_hp": t.p2_hp,
            "p1_hp_after": t.p1_hp_after, "p2_hp_after": t.p2_hp_after,
            "p1_active": t.p1_active, "p2_active": t.p2_active,
            "p1_active_after": t.p1_active_after,
            "p2_active_after": t.p2_active_after,
            "p1_stack": t.p1_stack, "p2_stack": t.p2_stack,
            "p1_stack_after": t.p1_stack_after,
            "p2_stack_after": t.p2_stack_after}
             for t in e.full_turn_logs]
    return {"gen": gen_num, "winner": result["winner"], "total_turns": result["turns"],
            "p1_team": [c.char_type.value for c in e.p1.characters],
            "p2_team": [c.char_type.value for c in e.p2.characters],
            "turns": turns, "file": filename}


def run_smart_coevolution(pop_size=240, generations=120, games_per_eval=30,
                           elite_frac=0.07, mut_rate=0.15, mut_sigma=0.15,
                           n_jobs=8, verbose=True,
                           hof_add=12, hof_ratio=0.25, hof_max=200,
                           reference_games=8,
                           snapshot_interval=10):
    """Co-evolution with SmartNeuralAgent (12-param genome + opponent model).
    
    Converges much faster than LSTM (50 gens vs 300) because:
    - 12 params instead of 3912 (300x smaller search space)
    - Opponent model is algorithmic (not learned)
    - Counter-play logic is algorithmic
    Evolution only tunes base preferences and reaction strengths.
    """
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    battle_log_dir = os.path.join(_this_dir, "parameterized_ai", f"battles_{run_timestamp}")
    os.makedirs(battle_log_dir, exist_ok=True)

    n_jobs = max(1, mp.cpu_count() if n_jobs == 0 else n_jobs)
    reference_groups = len(ANCHOR_PROFILES) + 3
    reference_total = reference_groups * reference_games
    total_games_per = games_per_eval + reference_total
    total_battles = pop_size * generations * total_games_per
    extra = f"+HoF({hof_ratio:.0%}) +{reference_groups}refs*{reference_games}"
    print(f"Smart co-evolution: pop={pop_size}, gens={generations}, "
          f"games/agent={total_games_per} ({extra})", flush=True)
    print(f"Genome size: {smart_genome_size()} params", flush=True)
    print(f"Workers: {n_jobs}", flush=True)
    print(f"Estimated battles: {total_battles:,}", flush=True)
    est_s = total_battles * 0.001 / n_jobs  # faster per game (no LSTM)
    print(f"Est. time: {est_s:.0f}s ({est_s/60:.1f}m)", flush=True)
    print(flush=True)

    print("  Starting workers & shared memory...", flush=True)

    population = [random_smart_genome(seed=i) for i in range(pop_size)]
    hall_of_fame = []
    fitness_history = []
    meta_history = []
    gen_stats = []
    battle_records = []

    t0 = time.perf_counter()

    pool = mp.Pool(n_jobs) if n_jobs > 1 else None

    print("  Workers ready, starting evolution...", flush=True)
    gen_pbar = tqdm(total=generations, desc="Training", unit="gen", position=0)
    try:
        for gen in range(generations):
            gen_t0 = time.perf_counter()

            fitnesses = smart_evaluate_all(population, games_per_eval, pool=pool,
                                           hof_ratio=hof_ratio,
                                           reference_games=reference_games,
                                           hall_of_fame=hall_of_fame,
                                           desc=f"Eval gen {gen+1}")

            best_f = max(fitnesses)
            avg_f = np.mean(fitnesses)
            fitness_history.append((best_f, avg_f))

            indexed = list(zip(range(pop_size), fitnesses, population))
            indexed.sort(key=lambda x: -x[1])

            gen_stats.append({
                "gen": gen + 1, "best_fitness": float(best_f), "avg_fitness": float(avg_f),
                "hof_size": len(hall_of_fame) if hall_of_fame else 0,
            })

            # Next generation
            next_pop = []
            n_elite = max(2, int(pop_size * elite_frac))
            for _, _, genome in indexed[:n_elite]:
                next_pop.append(genome.copy())

            while len(next_pop) < pop_size:
                p1_idx = tournament_select(fitnesses)
                p2_idx = tournament_select(fitnesses)
                child = crossover(population[p1_idx], population[p2_idx])
                child = mutate(child, rate=mut_rate, sigma=mut_sigma)
                next_pop.append(child)

            # Hall of Fame
            if hall_of_fame is not None and hof_add > 0:
                top_idx = np.argsort(fitnesses)[-hof_add:]
                for idx in top_idx:
                    g = population[idx].copy()
                    if _hof_is_novel(g, hall_of_fame):
                        hall_of_fame.append(g)
                if len(hall_of_fame) > hof_max:
                    hall_of_fame = hall_of_fame[-hof_max:]

            population = next_pop

            gen_t = time.perf_counter() - gen_t0
            gen_pbar.set_postfix(best=f"{best_f:.1%}", avg=f"{avg_f:.1%}", t=f"{gen_t:.1f}s")
            gen_pbar.update(1)

            # Snapshot
            if (gen + 1) % snapshot_interval == 0 or gen + 1 == generations:
                sample = random.sample(population, min(100, pop_size))
                arch_counts = defaultdict(int)
                totals = {"atk": 0, "def_": 0, "bon": 0, "sw": 0.0,
                          "bb": 0, "ba": 0, "aa": 0, "ad": 0,
                          "len_avg": 0, "len_min": 999, "len_max": 0,
                          "p1_wr": 0, "overkill": 0, "missed_lethal": 0,
                          "punished_greed": 0, "res_eff": 0, "n_sw": 0}
                for g in tqdm(sample, desc="Classify", leave=False):
                    info = smart_classify_agent(g, n_test_games=4)
                    arch_counts[info["archetype"]] += 1
                    totals["atk"] += info["atk"]
                    totals["def_"] += info["def"]
                    totals["bon"] += info["bon"]
                    totals["sw"] += info["sw_per_game"]
                    totals["bb"] += info["pattern_bonbon"]
                    totals["ba"] += info["pattern_bonatk"]
                    totals["aa"] += info["pattern_atkatk"]
                    totals["ad"] += info["pattern_atkdef"]
                    totals["len_avg"] += info["game_len_avg"]
                    totals["len_min"] = min(totals["len_min"], info["game_len_min"])
                    totals["len_max"] = max(totals["len_max"], info["game_len_max"])
                    totals["p1_wr"] += info["p1_wr"]
                    totals["overkill"] += info["overkill"]
                    totals["missed_lethal"] += info["missed_lethal"]
                    totals["punished_greed"] += info["punished_greed"]
                    totals["res_eff"] += info["resource_eff"]
                    totals["n_sw"] += info["n_switches"]
                n = len(sample)
                meta_history.append((gen + 1, dict(arch_counts), n))

                # Store aggregate snapshot in gen_stats
                gen_stats[-1]["archetype_dist"] = {a: c/n for a, c in arch_counts.items()}
                gen_stats[-1]["avg_atk_pct"] = totals["atk"]/n
                gen_stats[-1]["avg_def_pct"] = totals["def_"]/n
                gen_stats[-1]["avg_bon_pct"] = totals["bon"]/n
                gen_stats[-1]["avg_sw_per_game"] = totals["sw"]/n
                gen_stats[-1]["avg_len"] = totals["len_avg"]/n
                gen_stats[-1]["avg_overkill"] = totals["overkill"]/n
                gen_stats[-1]["avg_missed_lethal"] = totals["missed_lethal"]/n
                gen_stats[-1]["avg_punished_greed"] = totals["punished_greed"]/n
                gen_stats[-1]["avg_res_eff"] = totals["res_eff"]/n
                gen_stats[-1]["avg_p1_wr"] = totals["p1_wr"]/n
                gen_stats[-1]["pattern_bb"] = totals["bb"]/n
                gen_stats[-1]["pattern_aa"] = totals["aa"]/n

                best_genome = indexed[0][2]
                battle_log = _smart_record_battle(best_genome, gen+1, run_timestamp, battle_log_dir)
                battle_records.append(battle_log)

                print(f"  --- Pop snapshot (n={n}) ---")
                arch_line = ", ".join(f"{a}={c/n:.0%}" for a, c in sorted(arch_counts.items(), key=lambda x: -x[1]))
                print(f"  Arch: {arch_line}")
                print(f"  Actions: atk={totals['atk']/n:.0%} def={totals['def_']/n:.0%} bon={totals['bon']/n:.0%} "
                      f"sw={totals['sw']/n:.2f}/g")
                print(f"  Patterns: BB={totals['bb']/n:.0%} AA={totals['aa']/n:.0%} "
                      f"BA={totals['ba']/n:.0%} AD={totals['ad']/n:.0%}")
                print(f"  Game: len={totals['len_avg']/n:.1f} ({totals['len_min']:.0f}-{totals['len_max']:.0f}) "
                      f"P1wr={totals['p1_wr']/n:.0%} "
                      f"overkill={totals['overkill']/n:.1f} missL={totals['missed_lethal']/n:.2f}/g")
    finally:
        gen_pbar.close()
        if pool is not None:
            pool.close(); pool.join()

    total_t = time.perf_counter() - t0
    print(f"\nTotal time: {total_t:.0f}s ({total_t/60:.1f}m)")

    facts = _compute_facts(gen_stats, battle_records)

    if len(meta_history) > 1:
        print(f"\n--- Meta evolution ---")
        all_archs = sorted(set(a for _, snap, _ in meta_history for a in snap))
        header = f"  {'Gen':>4s}  " + "  ".join(f"{a:>10s}" for a in all_archs)
        print(header)
        print(f"  {'-'*4}  " + "  ".join(f"{'-'*10}" for _ in all_archs))
        for gen, snap, n in meta_history:
            vals = "  ".join(f"{snap.get(a,0)/n:>10.0%}" for a in all_archs)
            print(f"  {gen:4d}  {vals}")

    if facts:
        print(f"\n--- Interesting Facts ---")
        for f in facts:
            print(f"  {f}")

    return population, fitness_history, meta_history, gen_stats, battle_records, facts


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    run_ts = time.strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(_this_dir, "parameterized_ai")
    log_file = os.path.join(log_dir, f"training_{run_ts}.log")

    class TeeOutput:
        def __init__(self, filepath):
            self.file = open(filepath, "w", encoding="utf-8")
            self.stdout = sys.stdout
        def write(self, data):
            self.stdout.write(data); self.file.write(data); self.file.flush()
        def flush(self):
            self.stdout.flush(); self.file.flush()
        def close(self):
            self.file.close(); sys.stdout = self.stdout

    tee = TeeOutput(log_file)
    sys.stdout = tee

    print("=" * 65)
    print("COTE MEGAVERSE — SMART CO-EVOLUTION (12-param genome + opponent model)")
    print("Evolution only tunes preferences; adaptation is algorithmic")
    print("=" * 65)
    print(f"Log file: {log_file}\n")

    # Run the production smart co-evolution profile.
    pop, hist, _, gen_stats, battle_records, facts = run_smart_coevolution()

    print("\nEvaluating final generation...")
    final_fits = smart_evaluate_all(
        pop,
        n_games=40,
        hof_ratio=SMART_PRODUCTION_CONFIG["hof_ratio"],
        reference_games=SMART_PRODUCTION_CONFIG["reference_games"])

    best_idx = max(range(len(final_fits)), key=lambda i: final_fits[i])
    best_genome = pop[best_idx]
    best_fitness = final_fits[best_idx]

    print(f"\nBest agent fitness: {best_fitness:.1%}")

    champ_info = smart_classify_agent(best_genome, n_test_games=30)
    print(f"Champion: {champ_info['archetype']} "
          f"atk={champ_info['atk']:.0%} def={champ_info['def']:.0%} bon={champ_info['bon']:.0%} "
          f"sw={champ_info['sw_per_game']:.1f}/g SwEV={champ_info['swap_ev']:+3.0f} "
          f"overAct={champ_info['overkill']:.1f} missLeth={champ_info['missed_lethal']:.1f}/g "
          f"resEff={champ_info['resource_eff']:.0f}")

    ai_dir = os.path.join(_this_dir, "parameterized_ai")
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    version_info = {
        "timestamp": timestamp, "best_fitness": best_fitness,
        "generations": len(hist), "population_size": len(pop),
        "params": dict(SMART_PRODUCTION_CONFIG),
        "agent_type": "smart",
        "champion": {k: v for k, v in champ_info.items() if k != "archetype"},
        "archetype": champ_info["archetype"],
    }

    np.save(os.path.join(ai_dir, f"best_genome_{timestamp}.npy"), best_genome)
    _write_json_atomic(
        os.path.join(ai_dir, f"best_genome_{timestamp}.json"),
        version_info, indent=2)
    np.save(os.path.join(ai_dir, "best_genome.npy"), best_genome)

    versions_path = os.path.join(ai_dir, "versions.json")
    versions = _load_json_or_default(versions_path, [])
    if not isinstance(versions, list):
        versions = []
    versions.append(version_info)
    _write_json_atomic(versions_path, versions, indent=2)

    result = {"best_fitness": best_fitness, "generations": len(hist),
              "population_size": len(pop),
              "fitness_history": [(float(b), float(a)) for b, a in hist]}
    _write_json_atomic(os.path.join(ai_dir, "coevolution_result.json"), result, indent=2)

    print(f"Saved version {timestamp} (fitness={best_fitness:.1%})")

    stats = {"config": dict(SMART_PRODUCTION_CONFIG),
             "timestamp": timestamp,
             "genome_size": smart_genome_size(), "agent_type": "smart",
             "total_generations": len(hist),
             "final_best_fitness": float(best_fitness),
             "final_avg_fitness": float(np.mean([f for _, f in hist])),
             "fitness_history": [{"gen": i+1, "best": float(b), "avg": float(a)} for i, (b, a) in enumerate(hist)],
             "gen_stats": gen_stats, "battle_records": battle_records, "facts": facts}
    _write_json_atomic(
        os.path.join(ai_dir, "coevolution_stats.json"),
        stats, indent=2, ensure_ascii=False)
    print(f"Stats saved to {os.path.join(ai_dir, 'coevolution_stats.json')}")

    n_top = max(10, len(pop) // 10)
    top_indices = np.argsort(final_fits)[-n_top:]
    top_genomes = [pop[i] for i in top_indices]
    print(f"\n--- Top {n_top} agents ---")
    analyze_population(top_genomes, n_sample=min(50, n_top))
    print(f"\n--- Full population ---")
    analyze_population(pop, n_sample=100)

    tee.close()
    print(f"\nTraining log saved to: {log_file}")
