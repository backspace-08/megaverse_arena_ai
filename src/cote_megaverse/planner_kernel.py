"""Optional Numba kernels with a Python reference implementation."""

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None


def _round100(value):
    return int(round(float(value) / 100.0) * 100)


def _promote(hp, active, stack, alive_count):
    dead = int(active)
    next_index = next((int(index) for index in stack
                       if int(index) != dead and hp[int(index)] > 0), -1)
    alive_count -= 1
    if next_index < 0:
        return active, stack, alive_count
    rest = [int(index) for index in stack
            if int(index) not in (dead, next_index)]
    return next_index, np.asarray([next_index, dead] + rest, dtype=np.int64), alive_count


def _resolve_python(state, plan, multipliers):
    hp = np.asarray(state["hp"], dtype=np.int64).copy()
    target_hp = np.asarray(state["target_hp"], dtype=np.int64).copy()
    stack = np.asarray(state["stack"], dtype=np.int64).copy()
    target_stack = np.asarray(state["target_stack"], dtype=np.int64).copy()
    atk = np.asarray(state["atk"], dtype=np.int64)
    target_atk = np.asarray(state["target_atk"], dtype=np.int64)
    types = np.asarray(state["types"], dtype=np.int64)
    target_types = np.asarray(state["target_types"], dtype=np.int64)
    active = int(state["active"])
    target_active = int(state["target_active"])
    alive_count = int(state["alive_count"])
    target_alive_count = int(state["target_alive_count"])
    bonus = int(state["bonus"])
    target_bonus = int(state["target_bonus"])
    shields = int(state["shields"])
    target_shields = int(state["target_shields"])
    remaining = int(state["remaining"])
    target_remaining = int(state["target_remaining"])
    attacks, defends, bonuses, switch_to = [int(value) for value in plan]
    switched = False
    forced_switch = False
    valid = True
    if switch_to >= 0:
        if switch_to == active or hp[switch_to] <= 0 or remaining < 1:
            valid = False
        else:
            stack = np.asarray([switch_to] + [int(index) for index in stack if int(index) != switch_to], dtype=np.int64)
            active = switch_to
            remaining -= 1
            switched = True
    if attacks + defends + bonuses > remaining:
        valid = False
    blocked = min(attacks, target_shields) if valid else 0
    unblocked = max(0, attacks - blocked) if valid else 0
    damage = 0
    if valid:
        target_shields = 0
        if unblocked > 0 and target_alive_count > 0 and target_hp[target_active] > 0:
            damage = min(target_hp[target_active], _round100(
                atk[active] * multipliers[types[active], target_types[target_active]] * unblocked))
            target_hp[target_active] -= damage
            if target_hp[target_active] <= 0:
                target_active, target_stack, target_alive_count = _promote(
                    target_hp, target_active, target_stack, target_alive_count)
                forced_switch = True
        bonus = min(4, bonus + bonuses)
        shields = defends
        remaining -= attacks + defends + bonuses
    return {
        "hp": hp, "target_hp": target_hp, "active": active,
        "target_active": target_active, "stack": stack,
        "target_stack": target_stack, "alive_count": alive_count,
        "target_alive_count": target_alive_count, "bonus": bonus,
        "target_bonus": target_bonus, "shields": shields,
        "target_shields": target_shields, "remaining": remaining,
        "target_remaining": target_remaining, "damage": damage,
        "blocked": blocked, "unblocked": unblocked,
        "switched": switched, "forced_switch": forced_switch,
        "valid": valid,
    }


if njit is not None:  # pragma: no cover
    @njit(cache=True)
    def _resolve_numba(hp, target_hp, atk, target_atk, types, target_types,
                       active, target_active, stack, target_stack,
                       alive_count, target_alive_count, bonus, target_bonus,
                       shields, target_shields, remaining, target_remaining,
                       plan, multipliers):
        hp = hp.copy()
        target_hp = target_hp.copy()
        stack = stack.copy()
        target_stack = target_stack.copy()
        attacks, defends, bonuses, switch_to = plan[0], plan[1], plan[2], plan[3]
        switched = False
        forced_switch = False
        valid = True
        if switch_to >= 0:
            if switch_to == active or hp[switch_to] <= 0 or remaining < 1:
                valid = False
            else:
                new_stack = np.empty(3, dtype=np.int64)
                new_stack[0] = switch_to
                pos = 1
                for index in range(3):
                    if stack[index] != switch_to:
                        new_stack[pos] = stack[index]
                        pos += 1
                stack = new_stack
                active = switch_to
                remaining -= 1
                switched = True
        if attacks + defends + bonuses > remaining:
            valid = False
        blocked = min(attacks, target_shields) if valid else 0
        unblocked = max(0, attacks - blocked) if valid else 0
        damage = 0
        if valid:
            target_shields = 0
            if unblocked > 0 and target_alive_count > 0 and target_hp[target_active] > 0:
                damage = min(target_hp[target_active], round(atk[active] * multipliers[types[active], target_types[target_active]] * unblocked / 100.0) * 100)
                target_hp[target_active] -= damage
                if target_hp[target_active] <= 0:
                    dead = target_active
                    target_alive_count -= 1
                    next_index = -1
                    for index in range(3):
                        candidate = target_stack[index]
                        if candidate != dead and target_hp[candidate] > 0:
                            next_index = candidate
                            break
                    if next_index >= 0:
                        new_stack = np.empty(3, dtype=np.int64)
                        new_stack[0] = next_index
                        new_stack[1] = dead
                        pos = 2
                        for index in range(3):
                            candidate = target_stack[index]
                            if candidate != dead and candidate != next_index:
                                new_stack[pos] = candidate
                                pos += 1
                        target_stack = new_stack
                        target_active = next_index
                    forced_switch = True
            bonus = min(4, bonus + bonuses)
            shields = defends
            remaining -= attacks + defends + bonuses
        return (hp, target_hp, active, target_active, stack, target_stack,
                alive_count, target_alive_count, bonus, target_bonus, shields,
                target_shields, remaining, target_remaining, damage, blocked,
                unblocked, switched, forced_switch, valid)


def resolve_numeric(state, plan, multipliers, compiled=True):
    """Resolve one exact macro exchange with Python or Numba."""
    plan = np.asarray(plan, dtype=np.int64)
    if not compiled or njit is None:
        return _resolve_python(state, plan, multipliers)
    result = _resolve_numba(
        np.asarray(state["hp"], dtype=np.int64), np.asarray(state["target_hp"], dtype=np.int64),
        np.asarray(state["atk"], dtype=np.int64), np.asarray(state["target_atk"], dtype=np.int64),
        np.asarray(state["types"], dtype=np.int64), np.asarray(state["target_types"], dtype=np.int64),
        int(state["active"]), int(state["target_active"]),
        np.asarray(state["stack"], dtype=np.int64), np.asarray(state["target_stack"], dtype=np.int64),
        int(state["alive_count"]), int(state["target_alive_count"]),
        int(state["bonus"]), int(state["target_bonus"]), int(state["shields"]),
        int(state["target_shields"]), int(state["remaining"]),
        int(state["target_remaining"]), plan, np.asarray(multipliers, dtype=np.float64))
    keys = ("hp", "target_hp", "active", "target_active", "stack", "target_stack",
            "alive_count", "target_alive_count", "bonus", "target_bonus", "shields",
            "target_shields", "remaining", "target_remaining", "damage", "blocked",
            "unblocked", "switched", "forced_switch", "valid")
    return dict(zip(keys, result))


def rank_macro_candidates(candidates, hp, atk, types, active, opponent_hp,
                          opponent_atk, opponent_type, opponent_shields,
                          opponent_bonus, mode, multipliers, top_k=16):
    candidates = np.asarray(candidates, dtype=np.int64)
    scores = np.zeros(len(candidates), dtype=np.float64)
    for index, (attacks, defends, bonuses, switch_to) in enumerate(candidates):
        char_index = active if switch_to < 0 else switch_to
        damage = round(atk[char_index] * multipliers[types[char_index], opponent_type] / 100.0) * 100
        incoming = round(opponent_atk * multipliers[opponent_type, types[char_index]] / 100.0) * 100
        scores[index] = min(opponent_hp, max(0, attacks - opponent_shields) * damage)
        scores[index] -= min(hp[active], max(0, min(8, 4 + opponent_bonus) - defends) * incoming)
        scores[index] += bonuses * damage * (1.0 if mode == 2 else 0.25)
    return np.argsort(scores)[::-1][:max(1, min(int(top_k), len(scores)))].tolist()


def using_numba():
    return njit is not None


def warmup_numba():
    if njit is None:
        return False
    state = {"hp": [6000, 6000, 6000], "target_hp": [6000, 6000, 6000],
             "atk": [2000, 2000, 2000], "target_atk": [2000, 2000, 2000],
             "types": [0, 1, 2], "target_types": [0, 1, 2], "active": 0,
             "target_active": 0, "stack": [0, 1, 2], "target_stack": [0, 1, 2],
             "alive_count": 3, "target_alive_count": 3, "bonus": 0,
             "target_bonus": 0, "shields": 0, "target_shields": 0,
             "remaining": 1, "target_remaining": 1}
    resolve_numeric(state, [1, 0, 0, -1], np.ones((4, 4)), compiled=True)
    return True


def has_exact_search_kernel():
    return njit is not None
