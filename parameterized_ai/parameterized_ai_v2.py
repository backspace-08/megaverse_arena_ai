"""
v2 — правильная механика игры (как в COTE megaverse):
- Проактивная защита: щиты ставятся на своём ходу, блокируют атаки на ходу противника
- Смена при смерти = бесплатно
- Все атаки бьют по активному персонажу
- Блеф: выбор между shield/bonus без знания планов противника
"""
import random
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum
from collections import defaultdict

# ========================
# ENUMS & CONSTANTS
# ========================

class CharType(Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"

EMOJI_MAP = {  # для сверки с реальными битвами
    "\U0001f3ad": CharType.A,  # театральная маска (Сакаянаги, Мацушита, Мияби)
    "\U0001f4a2": CharType.B,  # гнев (Чие)
    "\u2b50": CharType.C,      # звезда (Цубаса, Ичиносе, Киджима)
    "\U0001f3af": CharType.D,  # мишень (Казума, Кацураги)
}

TYPE_ADVANTAGE = {
    CharType.A: CharType.B,  # 🎭 → 💢
    CharType.B: CharType.C,  # 💢 → ⭐
    CharType.C: CharType.D,  # ⭐ → 🎯
    CharType.D: CharType.A,  # 🎯 → 🎭
}

BASE_HP = 6000
BASE_ATK = 2000
HP_RANGE = (5700, 6300)
ATK_RANGE = (1900, 2100)
HP_VALUES = [5700, 5800, 5900, 6000, 6100, 6200, 6300]
ATK_VALUES = [1900, 1950, 2000, 2050, 2100]
TYPE_ADVANTAGE_MULT = 1.3
TYPE_DISADVANTAGE_MULT = 0.7

MAX_BASE_ACTIONS = 4
MAX_BONUS_ACTIONS = 4
MAX_TOTAL_ACTIONS = 8

ACTION_COST_SWITCH = 1

TURN_ACTIONS = {
    1: 1, 2: 2, 3: 2, 4: 2,
    5: 3, 6: 3, 7: 4, 8: 4,
}

# Precomputed type advantage table (avoids dict.get + if-chain every call)
_TYPE_MULT_TABLE = {}
for _atk in CharType:
    for _def in CharType:
        if _atk == _def:
            _TYPE_MULT_TABLE[(_atk, _def)] = 1.0
        elif TYPE_ADVANTAGE.get(_atk) == _def:
            _TYPE_MULT_TABLE[(_atk, _def)] = TYPE_ADVANTAGE_MULT
        elif TYPE_ADVANTAGE.get(_def) == _atk:
            _TYPE_MULT_TABLE[(_atk, _def)] = TYPE_DISADVANTAGE_MULT
        else:
            _TYPE_MULT_TABLE[(_atk, _def)] = 1.0

# Cached CharType list (avoids list(CharType) every random_team call)
_CHAR_TYPES = list(CharType)


# ========================
# DATA CLASSES
# ========================

@dataclass
class Character:
    char_type: CharType
    hp: int = BASE_HP
    max_hp: int = BASE_HP
    atk: int = BASE_ATK
    is_active: bool = False

    def is_alive(self) -> bool:
        return self.hp > 0

    def take_damage(self, damage: int) -> int:
        actual = min(damage, self.hp)
        self.hp -= actual
        return actual


@dataclass
class Player:
    player_id: int
    characters: List[Character]
    active_char_index: int = 0
    base_actions: int = 0
    bonus_actions: int = 0
    remaining_actions: int = 0
    shields: int = 0          # щиты, поставленные на этом ходу (защита на ход противника)
    switched_this_round: bool = False
    switch_history: List[int] = field(default_factory=list)  # стек вытесненных персонажей (самый свежий — последний)
    just_swapped_free: bool = False  # флаг для AI: только что зашёл по смерти
    _alive_cache: Optional[List[Character]] = field(default=None, repr=False)
    _alive_count: int = field(default=0, repr=False)

    def __post_init__(self):
        if self.characters:
            self.characters[0].is_active = True
            self._alive_count = sum(1 for c in self.characters if c.is_alive())
            self._alive_cache = None  # lazy init

    @property
    def active_character(self) -> Character:
        return self.characters[self.active_char_index]

    @property
    def alive_characters(self) -> List[Character]:
        if self._alive_cache is None:
            self._alive_cache = [c for c in self.characters if c.is_alive()]
        return self._alive_cache

    def _invalidate_alive_cache(self):
        self._alive_cache = None

    def has_lost(self) -> bool:
        return self._alive_count == 0

    def switch_character(self, new_index: int) -> bool:
        if self.switched_this_round:
            return False
        if new_index == self.active_char_index:
            return False
        if self.switch_history and new_index == self.switch_history[-1]:
            return False  # нельзя вернуться к тому, кого только что вытеснил
        if not self.characters[new_index].is_alive():
            return False
        if self.remaining_actions < ACTION_COST_SWITCH:
            return False
        self.characters[self.active_char_index].is_active = False
        self.switch_history.append(self.active_char_index)
        self.active_char_index = new_index
        self.characters[new_index].is_active = True
        self.switched_this_round = True
        self.remaining_actions -= ACTION_COST_SWITCH
        return True

    def force_switch_from_death(self):
        """Принудительная смена при смерти активного — бесплатно.
        Возвращается последний вытесненный через switch_history (LIFO)."""
        # Деактивируем мёртвого
        self.characters[self.active_char_index].is_active = False
        self._alive_count -= 1
        self._invalidate_alive_cache()
        # Ищем живого в стеке (с конца)
        while self.switch_history:
            idx = self.switch_history.pop()
            if self.characters[idx].is_alive():
                self.active_char_index = idx
                self.characters[idx].is_active = True
                self.switched_this_round = True
                self.just_swapped_free = True
                return
        # Стек пуст или все мертвы — ищем любого живого
        for i, c in enumerate(self.characters):
            if c.is_alive() and i != self.active_char_index:
                self.active_char_index = i
                self.characters[i].is_active = True
                self.switched_this_round = True
                self.just_swapped_free = True
                return

    def reset_round_state(self):
        self.switched_this_round = False
        self.just_swapped_free = False
        # switch_history НЕ сбрасывается — сохраняется до конца боя


@dataclass
class BattleAction:
    action_type: str
    attacker_idx: int
    target_idx: Optional[int] = None
    damage: int = 0
    blocked: bool = False


@dataclass
class TurnLog:
    turn_num: int
    player_id: int
    attack_actions: int
    defend_actions: int
    bonus_actions: int
    switched: bool
    unblocked_attacks: int
    total_damage: int
    opponent_shields: int
    blocked_shields: int
    player_shields_before: int
    p1_hp: List[int]
    p2_hp: List[int]


# ========================
# TYPE SYSTEM
# ========================

def get_type_multiplier(atk_type: CharType, def_type: CharType) -> float:
    """Precomputed lookup — no if-chains."""
    return _TYPE_MULT_TABLE.get((atk_type, def_type), 1.0)


# ========================
# AI v2
# ========================

@dataclass
class AIProfile:
    name: str
    # Веса для аллокации действий
    w_attack: float = 1.0
    w_defend: float = 1.0
    w_bonus: float = 1.0
    w_switch: float = 1.0
    # Стратегия
    bonus_target: int = 0       # сколько бонусов копить перед активной игрой (0 = не копить)
    attack_threshold: float = 0.0
    defend_ratio: float = 0.3
    # Смена
    switch_when_disadvantaged: bool = False
    switch_min_hp_ratio: float = 0.3
    # Спасти первые N своих ходов
    save_first_turns: int = 0   # сколько своих ходов ничего не атаковать
    # После бесплатной смены (смерти) — в атаку
    aggressive_on_free_swap: bool = False
    # После большого залпа — поставить защиту
    defend_after_burst: bool = False
    # Количество щитов после залпа
    shields_after_burst: int = 2
    # Параметр "сколько атак считать залпом"
    burst_threshold: int = 4
    # Случайность
    randomness: float = 0.1


class WeightedRandomAIv2:
    """
    AI для v2 механики.
    Аллокация действий на основе весов + стратегических модификаторов.
    """

    def __init__(self, profile: AIProfile):
        self.p = profile
        self._last_attack_count = 0

    def choose_actions(self, player: Player, opponent: Player,
                       turn_num: int, turn_logs: List[TurnLog],
                       player_id: int) -> List[BattleAction]:
        available = player.remaining_actions
        if available <= 0:
            return []

        # Считаем, сколько своих ходов уже было
        my_turns = sum(1 for t in turn_logs if t.player_id == player_id)

        should_switch = self._decide_switch(player, opponent)
        if should_switch and available <= ACTION_COST_SWITCH:
            should_switch = False

        if should_switch:
            switch_target = self._choose_switch_target(player, opponent)
            if switch_target is not None and switch_target != player.active_char_index:
                player.switch_character(switch_target)

        remaining = player.remaining_actions
        if remaining <= 0:
            return []

        opp_shields = opponent.shields
        max_bonus_can_add = MAX_BONUS_ACTIONS - player.bonus_actions

        # Save-first turns
        if self.p.save_first_turns > 0 and my_turns < self.p.save_first_turns:
            bonus = min(remaining, max_bonus_can_add)
            return self._build_actions(0, 0, bonus, player, opponent)

        # Aggressive on free swap
        if self.p.aggressive_on_free_swap and player.just_swapped_free:
            return self._build_actions(remaining, 0, 0, player, opponent)

        # Базовая аллокация по весам
        raw_attack = self.p.w_attack
        raw_defend = self.p.w_defend
        raw_bonus = self.p.w_bonus

        # Корректировка весов под контекст
        raw_attack, raw_defend, raw_bonus = self._adjust_weights(
            player, opponent, raw_attack, raw_defend, raw_bonus, remaining, opp_shields)

        # Defend after burst
        if self.p.defend_after_burst and self._last_attack_count >= self.p.burst_threshold:
            raw_defend *= 3.0

        total = raw_attack + raw_defend + raw_bonus
        if total == 0:
            raw_attack, raw_defend, raw_bonus = 1.0, 0.0, 0.0
            total = 1.0

        if self.p.randomness > 0 and random.random() < self.p.randomness:
            noise = lambda: random.uniform(0.5, 1.5)
            raw_attack *= noise()
            raw_defend *= noise()
            raw_bonus *= noise()
            total = raw_attack + raw_defend + raw_bonus

        attack = int(remaining * raw_attack / total)
        defend = int(remaining * raw_defend / total)
        bonus = remaining - attack - defend
        bonus = min(bonus, max_bonus_can_add)

        raw_effectiveness = attack - opp_shields
        if raw_effectiveness <= 0 and attack > 0 and self.p.attack_threshold > 0 and bonus < max_bonus_can_add:
            bonus += attack
            attack = 0

        return self._build_actions(attack, defend, bonus, player, opponent)

    def record_turn(self, attack_count: int):
        """Записать ход для статистики."""
        self._last_attack_count = attack_count

    def _build_actions(self, attack: int, defend: int, bonus: int,
                       player: Player, opponent: Player) -> List[BattleAction]:
        actions = []
        for _ in range(attack):
            actions.append(BattleAction("attack", player.active_char_index, opponent.active_char_index))
        for _ in range(defend):
            actions.append(BattleAction("defend", player.active_char_index))
        for _ in range(bonus):
            actions.append(BattleAction("bonus", player.active_char_index))
        random.shuffle(actions)
        self.record_turn(attack)
        return actions

    def _decide_switch(self, player: Player, opponent: Player) -> bool:
        p = self.p
        if not p.switch_when_disadvantaged:
            return False
        if player.switched_this_round:
            return False
        act = player.active_character
        hp_ratio = act.hp / act.max_hp
        if hp_ratio < p.switch_min_hp_ratio and len(player.alive_characters) > 1:
            return True
        alive_opps = opponent.alive_characters
        if not alive_opps:
            return False
        is_disadvantaged = all(
            get_type_multiplier(act.char_type, opp.char_type) < 1.0
            for opp in alive_opps
        )
        return is_disadvantaged

    def _choose_switch_target(self, player: Player, opponent: Player) -> Optional[int]:
        alive_opps = opponent.alive_characters
        best_score = -999
        best_idx = None
        for i, char in enumerate(player.characters):
            if not char.is_alive() or i == player.active_char_index:
                continue
            if player.switch_history and i == player.switch_history[-1]:
                continue
            score = char.hp / char.max_hp * 50
            for opp in alive_opps:
                mult = get_type_multiplier(char.char_type, opp.char_type)
                if mult > 1.0:
                    score += 20
                elif mult < 1.0:
                    score -= 10
            if score > best_score:
                best_score = score
                best_idx = i
        return best_idx

    def _adjust_weights(self, player: Player, opponent: Player,
                        wa: float, wd: float, wb: float,
                        available: int, opp_shields: int) -> Tuple[float, float, float]:
        p = self.p
        # Bonus target — не копить сверх цели
        if p.bonus_target > 0 and player.bonus_actions >= p.bonus_target:
            wa *= 2.0
            wb *= 0.1
        elif p.bonus_target > 0:
            need = p.bonus_target - player.bonus_actions
            urgency = need / max(p.bonus_target, 1)
            wb *= (1.0 + urgency)

        # Kill confirmation
        opp_hp = opponent.active_character.hp
        my_atk = player.active_character.atk
        mult = get_type_multiplier(player.active_character.char_type, opponent.active_character.char_type)
        dmg_per_hit = int(my_atk * mult)
        hits_to_kill = math.ceil(opp_hp / max(dmg_per_hit, 1))
        available_unblocked = max(0, available - opp_shields)
        if hits_to_kill <= available_unblocked and hits_to_kill > 0:
            wa *= 3.0

        # Защита от burst: противник мог накопить или только что потратил
        # (если bonus_actions недавно сброшен, смотрим последние логи)
        opp_bonus_potential = opponent.bonus_actions
        if opp_bonus_potential >= 2:
            wd *= 1.5

        return wa, wd, wb


class CounterAI:
    """
    Counter-attacker: shields at disadvantage, attacks at advantage,
    switches to gain type edge. Demonstrates reactive defense + punish.
    Follows the same switch-as-side-effect pattern as WeightedRandomAIv2/NeuralAgent
    (no BattleAction for switch, just calls player.switch_character directly).
    """
    def __init__(self, name="Counter"):
        self.name = name

    def choose_actions(self, player, opponent, turn_num, turn_logs, player_id):
        actions = []
        my_type = player.active_character.char_type
        opp_type = opponent.active_character.char_type
        mult = get_type_multiplier(my_type, opp_type)
        opp_actions = opponent.remaining_actions

        # Switch if at disadvantage and a better option exists
        if mult < 1.0 and not player.switched_this_round and player.remaining_actions >= ACTION_COST_SWITCH:
            target = self._pick_switch(player, opp_type)
            if target is not None:
                player.switch_character(target)
                my_type = player.active_character.char_type
                mult = get_type_multiplier(my_type, opp_type)

        remaining = player.remaining_actions
        if remaining <= 0:
            return actions

        atk_w = def_w = bon_w = 0

        if mult > 1.0:
            # Advantage: press attack, some shield for safety
            atk_w = int(remaining * 0.65)
            def_w = max(1, int(remaining * 0.2))
        elif opp_actions >= 3:
            # Opponent can attack heavily: shield to block
            def_w = min(remaining, max(1, opp_actions // 2))
        else:
            # Neutral: probe with mild attack
            atk_w = max(1, remaining // 2)
            def_w = remaining // 4

        bon_w = remaining - atk_w - def_w
        if bon_w < 0: bon_w = 0; atk_w = remaining - def_w
        if atk_w < 0: atk_w = 0; def_w = remaining

        for _ in range(atk_w):
            actions.append(BattleAction("attack", player.active_char_index, opponent.active_char_index))
        for _ in range(def_w):
            actions.append(BattleAction("defend", player.active_char_index))
        for _ in range(bon_w):
            if player.bonus_actions < MAX_BONUS_ACTIONS:
                actions.append(BattleAction("bonus", player.active_char_index))

        return actions

    def _pick_switch(self, player, opp_type):
        best_target = None
        best_adv = 0.0
        for i, c in enumerate(player.characters):
            if not c.is_alive() or i == player.active_char_index:
                continue
            if player.switch_history and i == player.switch_history[-1]:
                continue
            mult = get_type_multiplier(c.char_type, opp_type)
            if mult > best_adv:
                best_adv = mult
                best_target = i
        return best_target if best_adv > 1.0 else None


class AdaptiveAI:
    """AI with memory: tracks opponent tendencies and adapts mid-game."""

    def __init__(self, memory: int = 6, adapt_rate: float = 0.3):
        self.memory = memory
        self.adapt_rate = adapt_rate
        self._opp_history: List[Dict] = []

    def choose_actions(self, player: Player, opponent: Player,
                       turn_num: int, turn_logs: List[TurnLog],
                       player_id: int) -> List[BattleAction]:
        # Update opponent history from logs
        for t in turn_logs:
            if t.player_id != player_id and t not in self._opp_history:
                self._opp_history.append({
                    "atk": t.attack_actions,
                    "def": t.defend_actions,
                    "bon": t.bonus_actions,
                    "sw": 1 if t.switched else 0,
                })
        if len(self._opp_history) > self.memory:
            self._opp_history = self._opp_history[-self.memory:]

        # Switch if disadvantaged
        my_type = player.active_character.char_type
        opp_type = opponent.active_character.char_type
        mult = get_type_multiplier(my_type, opp_type)
        if mult < 1.0 and not player.switched_this_round and player.remaining_actions >= ACTION_COST_SWITCH:
            target = self._pick_switch(player, opp_type)
            if target is not None:
                player.switch_character(target)
                my_type = player.active_character.char_type
                mult = get_type_multiplier(my_type, opp_type)

        remaining = player.remaining_actions
        if remaining <= 0:
            return []

        # Compute opponent tendencies from history
        opp_atk_p, opp_def_p, opp_bon_p, opp_sw_p = self._opponent_tendencies()

        # Base weights
        w_atk, w_def, w_bon = 8.0, 2.0, 2.0

        # Adapt: counter opponent tendencies
        if opp_atk_p > 0.4:
            w_def += self.adapt_rate * 10  # shield more vs heavy attacker
        if opp_def_p > 0.25:
            w_bon += self.adapt_rate * 8   # bonus up when opponent shields
        if opp_bon_p > 0.25:
            w_atk += self.adapt_rate * 10  # punish bonus stacking
        if opp_sw_p > 0.3:
            w_atk += self.adapt_rate * 5   # attack more, punish their switch action loss

        # Adjust for type advantage
        if mult > 1.0:
            w_atk *= 1.3
        elif mult < 1.0:
            w_def *= 1.5

        # Small randomness (not perfectly predictable)
        if random.random() < 0.15:
            w_atk *= random.uniform(0.7, 1.3)
            w_def *= random.uniform(0.7, 1.3)
            w_bon *= random.uniform(0.7, 1.3)

        total = w_atk + w_def + w_bon
        attack = int(remaining * w_atk / total)
        defend = int(remaining * w_def / total)
        bonus = remaining - attack - defend
        bonus = min(bonus, MAX_BONUS_ACTIONS - player.bonus_actions)

        actions = []
        for _ in range(attack):
            actions.append(BattleAction("attack", player.active_char_index, opponent.active_char_index))
        for _ in range(defend):
            actions.append(BattleAction("defend", player.active_char_index))
        for _ in range(bonus):
            actions.append(BattleAction("bonus", player.active_char_index))
        random.shuffle(actions)
        return actions

    def _opponent_tendencies(self) -> Tuple[float, float, float, float]:
        if not self._opp_history:
            return 0.33, 0.33, 0.33, 0.0
        total_turns = len(self._opp_history)
        total_acts = sum(t["atk"] + t["def"] + t["bon"] for t in self._opp_history) or 1
        atk_p = sum(t["atk"] for t in self._opp_history) / total_acts
        def_p = sum(t["def"] for t in self._opp_history) / total_acts
        bon_p = sum(t["bon"] for t in self._opp_history) / total_acts
        sw_p = sum(t["sw"] for t in self._opp_history) / total_turns
        return atk_p, def_p, bon_p, sw_p

    def _pick_switch(self, player: Player, opp_type: CharType) -> Optional[int]:
        best_target = None
        best_adv = 0.0
        for i, c in enumerate(player.characters):
            if not c.is_alive() or i == player.active_char_index:
                continue
            if player.switch_history and i == player.switch_history[-1]:
                continue
            mult = get_type_multiplier(c.char_type, opp_type)
            if mult > best_adv:
                best_adv = mult
                best_target = i
        return best_target if best_adv > 1.0 else None


# ========================
# BATTLE ENGINE v2
# ========================

class BattleEngineV2:
    """
    Движок с правильной механикой:
    - Защита проактивная (ставишь на своём ходу)
    - Смена при смерти бесплатная
    - Атака всегда по активному
    - Блеф: shield vs bonus
    """

    def __init__(self, p1_ai, p2_ai,
                 p1_team: List[CharType], p2_team: List[CharType]):
        self.p1 = Player(1, [make_character(t) for t in p1_team])
        self.p2 = Player(2, [make_character(t) for t in p2_team])
        self.p1_ai = p1_ai
        self.p2_ai = p2_ai
        self.round_num = 1
        self.turn_num = 0
        self.current_player = 1
        self.turn_logs: List[TurnLog] = []
        self.stats = defaultdict(int)
        self.detailed_stats_p1 = {
            "overkill": 0, "counter_success": 0, "missed_lethal": 0,
            "punished_greed": 0, "swap_value": 0, "n_switches": 0,
            "total_actions": 0, "total_damage": 0, "total_prevented": 0,
        }
        self.detailed_stats_p2 = {
            "overkill": 0, "counter_success": 0, "missed_lethal": 0,
            "punished_greed": 0, "swap_value": 0, "n_switches": 0,
            "total_actions": 0, "total_damage": 0, "total_prevented": 0,
        }
        self.p1_min_alive = 3
        self.p2_min_alive = 3
        self._last_bonus = {1: 0, 2: 0}  # bonus used per player last turn

    def get_actions_for_turn(self, t: int) -> int:
        return TURN_ACTIONS.get(t, 4)

    def run(self, max_turns: int = 100) -> Dict:
        self.turn_num = 1
        self.current_player = 1
        self._setup_turn(self.p1, 1)

        while self.turn_num <= max_turns:
            if self.p1.has_lost():
                return self._result(2)
            if self.p2.has_lost():
                return self._result(1)

            player = self.p1 if self.current_player == 1 else self.p2
            opponent = self.p2 if self.current_player == 1 else self.p1
            ai = self.p1_ai if self.current_player == 1 else self.p2_ai

            # Invalidate alive caches at start of each turn (safe, cheap)
            self.p1._invalidate_alive_cache()
            self.p2._invalidate_alive_cache()

            self.p1_min_alive = min(self.p1_min_alive, len(self.p1.alive_characters))
            self.p2_min_alive = min(self.p2_min_alive, len(self.p2.alive_characters))
            hp_before_p1 = [c.hp for c in self.p1.characters]
            hp_before_p2 = [c.hp for c in self.p2.characters]

            # AI выбирает действия
            actions = ai.choose_actions(player, opponent, self.turn_num,
                                        self.turn_logs, self.current_player)

            # Исполняем ход
            log = self._execute_turn(player, opponent, actions)
            log.turn_num = self.turn_num
            log.player_id = self.current_player
            log.p1_hp = hp_before_p1
            log.p2_hp = hp_before_p2
            self.turn_logs.append(log)
            # Keep only last 8 turns in memory (enough for 4-turn history + buffer)
            if len(self.turn_logs) > 8:
                self.turn_logs = self.turn_logs[-8:]

            # Punished greed: current player attacked opponent; if opponent hoarded bonus
            # and their character died (force_switch_from_death set just_swapped_free), track it
            ds = self.detailed_stats_p1 if self.current_player == 1 else self.detailed_stats_p2
            if self._last_bonus[opponent.player_id] > 0 and opponent.just_swapped_free:
                ds["punished_greed"] += 1

            # Track bonus hoarding for punished_greed detection next round
            self._last_bonus[self.current_player] = log.bonus_actions

            if self.p1.has_lost():
                return self._result(2)
            if self.p2.has_lost():
                return self._result(1)

            # Смена игрока
            self.current_player = 2 if self.current_player == 1 else 1
            self.turn_num += 1

            # Сброс состояния и подготовка следующего хода
            if self.current_player == 1:
                self.p1.reset_round_state()
                self.p2.reset_round_state()
                self.round_num += 1

            next_player = self.p1 if self.current_player == 1 else self.p2
            self._setup_turn(next_player, self.turn_num)

        return self._result(0)

    def _setup_turn(self, player: Player, turn_num: int):
        base = min(self.get_actions_for_turn(turn_num), MAX_BASE_ACTIONS)
        total = min(base + player.bonus_actions, MAX_TOTAL_ACTIONS)
        # Бонусы расходуются: они превращаются в доступные действия
        used_bonus = min(player.bonus_actions, total - base)
        player.bonus_actions -= used_bonus
        player.base_actions = base
        player.remaining_actions = total

    def _execute_turn(self, player: Player, opponent: Player, actions: List[BattleAction]) -> TurnLog:
        ds = self.detailed_stats_p1 if player.player_id == 1 else self.detailed_stats_p2
        attack_actions = sum(1 for a in actions if a.action_type == "attack")
        defend_actions = sum(1 for a in actions if a.action_type == "defend")
        bonus_actions = sum(1 for a in actions if a.action_type == "bonus")

        remaining_before = player.remaining_actions
        action_total = attack_actions + defend_actions + bonus_actions
        player.remaining_actions -= action_total
        
        # Capture player's own shields before any mutations
        player_shields_before = player.shields

        # Track voluntary switch (detect if AI switched inside choose_actions)
        did_switch = False
        swap_value = 0
        if player.switched_this_round and not player.just_swapped_free:
            did_switch = True
            ds["n_switches"] += 1
            # Compute switch EV: net damage per action after vs before switch
            opp_type = opponent.active_character.char_type
            if player.switch_history:
                prev_idx = player.switch_history[-1]
                prev_char = player.characters[prev_idx]
                cur_char = player.active_character
                prev_net = int(prev_char.atk * get_type_multiplier(prev_char.char_type, opp_type)) \
                           - int(opponent.active_character.atk * get_type_multiplier(opp_type, prev_char.char_type))
                cur_net = int(cur_char.atk * get_type_multiplier(cur_char.char_type, opp_type)) \
                          - int(opponent.active_character.atk * get_type_multiplier(opp_type, cur_char.char_type))
                swap_value = (cur_net - prev_net) * remaining_before
                ds["swap_value"] += swap_value

        opp_shields = opponent.shields
        unblocked = max(0, attack_actions - opp_shields)
        blocked = min(attack_actions, opp_shields)
        opponent.shields -= blocked
        self.stats["attacks_blocked"] += blocked

        total_damage = 0
        overkill_actions = 0
        atk_char = player.active_character

        # Compute dmg_per_hit BEFORE any damage (opponent still unchanged)
        defender_type = opponent.active_character.char_type
        mult = get_type_multiplier(atk_char.char_type, defender_type)
        dmg_per_hit = int(atk_char.atk * mult)

        # Save opponent HP before attacks for overkill calculation
        opp_hp_before = opponent.active_character.hp

        # Counter success: damage prevented by opponent's shields
        counter_success = blocked * dmg_per_hit
        ds["counter_success"] += counter_success
        ds["total_prevented"] += counter_success

        if unblocked > 0 and opponent.active_character.is_alive():
            # Vectorized attack: compute total damage in one operation
            hp = opponent.active_character.hp
            total_damage = min(hp, unblocked * dmg_per_hit)
            opponent.active_character.hp -= total_damage
            self.stats["damage_dealt"] += total_damage
            self.stats["attacks"] += unblocked
            
            if opponent.active_character.hp <= 0:
                opponent.active_character.hp = 0
                opponent.force_switch_from_death()
                # Overkill: attacks past what was needed considering actual shields
                unblocked_needed = (opp_hp_before + dmg_per_hit - 1) // dmg_per_hit
                attacks_needed = opp_shields + unblocked_needed
                overkill_actions = max(0, attack_actions - attacks_needed)
                ds["overkill"] += overkill_actions

        # Missed lethal: opponent alive after our attacks, but we wasted non-attack actions that could have killed
        missed_lethal = False
        if opponent.active_character.is_alive():
            non_attack = remaining_before - attack_actions  # actions spent on defend/bonus
            if non_attack > 0 and attack_actions > 0:
                opp_hp = opponent.active_character.hp
                if dmg_per_hit * non_attack >= opp_hp:
                    missed_lethal = True
                    ds["missed_lethal"] += 1

        # Бонусы: не больше MAX
        actual_bonus = min(bonus_actions, MAX_BONUS_ACTIONS - player.bonus_actions)
        if actual_bonus > 0:
            player.bonus_actions += actual_bonus
            self.stats["bonuses"] += actual_bonus

        # Защита на следующий ход
        player.shields = defend_actions

        ds["total_actions"] += action_total
        ds["total_damage"] += total_damage

        return TurnLog(
            turn_num=0, player_id=0,
            attack_actions=attack_actions,
            defend_actions=defend_actions,
            bonus_actions=bonus_actions,
            switched=did_switch,
            unblocked_attacks=unblocked,
            total_damage=total_damage,
            opponent_shields=opp_shields,
            blocked_shields=blocked,
            player_shields_before=player_shields_before,
            p1_hp=[c.hp for c in self.p1.characters],
            p2_hp=[c.hp for c in self.p2.characters],
        )

    def _result(self, winner: int) -> Dict:
        ds1 = dict(self.detailed_stats_p1)
        ds2 = dict(self.detailed_stats_p2)
        ta1 = max(1, ds1["total_actions"])
        ta2 = max(1, ds2["total_actions"])
        ds1["resource_efficiency"] = (ds1["total_damage"] + ds1["total_prevented"]) / ta1
        ds2["resource_efficiency"] = (ds2["total_damage"] + ds2["total_prevented"]) / ta2
        return {
            "winner": winner,
            "rounds": self.round_num,
            "turns": len(self.turn_logs),
            "p1_final_hp": [c.hp for c in self.p1.characters],
            "p2_final_hp": [c.hp for c in self.p2.characters],
            "p1_min_alive": self.p1_min_alive,
            "p2_min_alive": self.p2_min_alive,
            "stats": dict(self.stats),
            "detailed_stats_p1": ds1,
            "detailed_stats_p2": ds2,
        }


# ========================
# UTILITY
# ========================

def make_character(char_type: CharType) -> Character:
    hp = random.choice(HP_VALUES)
    atk = random.choice(ATK_VALUES)
    return Character(char_type, hp=hp, max_hp=hp, atk=atk)

def random_team() -> List[CharType]:
    """Random team using cached CharType list."""
    return [_CHAR_TYPES[random.randint(0, 3)] for _ in range(3)]


# ========================
# ТРАССИРОВЩИК РЕАЛЬНЫХ БИТВ
# ========================

def trace_battle_v2(ai1, ai2, team1=None, team2=None, seed=42):
    """Одна битва с подробным логом для отладки."""
    if seed is not None:
        random.seed(seed)
    if team1 is None:
        team1 = random_team()
    if team2 is None:
        team2 = random_team()
    # Рандомизируем порядок хода
    if random.random() < 0.5:
        p1_first = True
        engine = BattleEngineV2(ai1, ai2, team1, team2)
    else:
        p1_first = False
        engine = BattleEngineV2(ai2, ai1, team2, team1)

    print(f"{'='*70}")
    print(f"BATTLE: {ai1.p.name if hasattr(ai1, 'p') else 'AI1'} vs "
          f"{ai2.p.name if hasattr(ai2, 'p') else 'AI2'}")
    print(f"P1 {'first' if p1_first else 'second'}: team {[t.value for t in team1]}")
    print(f"P2 {'second' if p1_first else 'first'}: team {[t.value for t in team2]}")
    print(f"{'='*70}")

    result = engine.run(50)

    for log in engine.turn_logs:
        pid = "P1" if log.player_id == 1 else "P2"
        p = engine.p1 if log.player_id == 1 else engine.p2
        opp = engine.p2 if log.player_id == 1 else engine.p1
        print(f"\nTurn {log.turn_num}: {pid}")
        print(f"  Actions: ATK={log.attack_actions} DEF={log.defend_actions} "
              f"BON={log.bonus_actions} SW={log.switched}")
        print(f"  Opponent shields: {log.opponent_shields} | "
              f"Unblocked: {log.unblocked_attacks} | DMG: {log.total_damage}")
        print(f"  After: P1 active={engine.p1.active_character.char_type.value} "
              f"HP={engine.p1.active_character.hp}/{engine.p1.active_character.max_hp}")
        print(f"  P2 active={engine.p2.active_character.char_type.value} "
              f"HP={engine.p2.active_character.hp}/{engine.p2.active_character.max_hp}")
        print(f"  P1 shields={engine.p1.shields} P2 shields={engine.p2.shields}")

    print(f"\n{'='*70}")
    winner = result["winner"]
    print(f"Winner: {'P1' if winner == 1 else 'P2' if winner == 2 else 'Draw'}")
    p1_alive = [c for c in engine.p1.alive_characters]
    p2_alive = [c for c in engine.p2.alive_characters]
    print(f"P1 alive: {len(p1_alive)} | P2 alive: {len(p2_alive)}")
    print(f"Stats: {dict(result['stats'])}")
    return result


# ========================
# ТЕСТОВЫЙ ЗАПУСК
# ========================

def make_profile(name: str, **kwargs) -> AIProfile:
    return AIProfile(name=name, **kwargs)


if __name__ == "__main__":
    random.seed(42)

    # Создаём AI
    human_like = WeightedRandomAIv2(make_profile("HumanLike",
        w_attack=3.0, w_defend=1.5, w_bonus=0.5, w_switch=0.3,
        bonus_target=0, defend_ratio=0.3,
        switch_when_disadvantaged=True, randomness=0.1))

    burst = WeightedRandomAIv2(make_profile("Burst",
        w_attack=0.5, w_defend=0.2, w_bonus=4.0, w_switch=0.1,
        bonus_target=4, defend_ratio=0.0,
        switch_when_disadvantaged=False, randomness=0.1))

    aggro = WeightedRandomAIv2(make_profile("Aggro",
        w_attack=4.0, w_defend=0.5, w_bonus=0.2, w_switch=0.2,
        bonus_target=0, defend_ratio=0.0,
        switch_when_disadvantaged=False, randomness=0.1))

    print("=== TEST 1: HumanLike vs Burst (trace) ===")
    trace_battle_v2(human_like, burst, seed=1)
    print("\n")

    print("=== TEST 2: HumanLike vs Aggro (trace) ===")
    trace_battle_v2(human_like, aggro, seed=1)
    print("\n")

    print("=== TEST 3: 100 battles HumanLike vs Burst ===")
    wins = 0
    for _ in range(100):
        t1 = random_team()
        t2 = random_team()
        # рандомный порядок
        if random.random() < 0.5:
            e = BattleEngineV2(human_like, burst, t1, t2)
            r = e.run(50)
            if r["winner"] == 1:
                wins += 1
        else:
            e = BattleEngineV2(burst, human_like, t1, t2)
            r = e.run(50)
            if r["winner"] == 2:
                wins += 1
    print(f"  HumanLike winrate vs Burst: {wins}%")

    print("\n=== TEST 4: 100 battles HumanLike vs Aggro ===")
    wins = 0
    for _ in range(100):
        t1 = random_team()
        t2 = random_team()
        if random.random() < 0.5:
            e = BattleEngineV2(human_like, aggro, t1, t2)
            r = e.run(50)
            if r["winner"] == 1:
                wins += 1
        else:
            e = BattleEngineV2(aggro, human_like, t1, t2)
            r = e.run(50)
            if r["winner"] == 2:
                wins += 1
    print(f"  HumanLike winrate vs Aggro: {wins}%")
