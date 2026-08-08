"""Human-fair belief-aware planning agent."""

from dataclasses import dataclass, field, replace

from .infoset import OpponentModel
from .rules import (Allocation, GameState, MAX_ACTIONS, MAX_BONUS, Side, apply,
                     attacks_to_kill, base_budget, exchange_damage,
                     legal_allocations, next_budget)
from .strategy import Objective, marginal_bonus_value, switch_value


@dataclass
class PublicHistory:
    actions: list[tuple[int, int, int, int]] = field(default_factory=list)
    events: list["PublicEvent"] = field(default_factory=list)
    held_shields: int | None = None
    policy_scores: dict[str, float] = field(default_factory=lambda: {
        "aggressive": 1.0,
        "defensive": 1.0,
        "builder": 1.0,
        "opportunistic": 1.0,
    })

    def observe(self, attacks: int, bonuses: int, switched: bool,
                budget: int | None = None):
        shields = None if budget is None else max(
            0, budget - attacks - bonuses - int(switched))
        self.actions.append((attacks, shields or 0, bonuses, int(switched)))
        self.held_shields = shields
        self.events.append(PublicEvent(attacks, bonuses, int(switched), shields))
        self._update_policy(attacks, shields or 0, bonuses)

    def observe_resolved(self, attacks, defends, bonuses, switched=False):
        self.actions.append((attacks, defends, bonuses, int(switched)))
        self.held_shields = defends
        self.events.append(PublicEvent(attacks, bonuses, int(switched), defends))
        self._update_policy(attacks, defends, bonuses)

    def reveal_latest_defends(self, defends: int):
        if not self.actions:
            return
        attacks, _, bonuses, switched = self.actions[-1]
        self.actions[-1] = (attacks, defends, bonuses, switched)
        self.held_shields = defends
        if self.events:
            event = self.events[-1]
            self.events[-1] = replace(event, resolved_shields=defends)
        self._update_policy(attacks, defends, bonuses)

    def _update_policy(self, attacks: int, defends: int, bonuses: int):
        total = max(1, attacks + defends + bonuses)
        shares = {
            "aggressive": attacks / total,
            "defensive": defends / total,
            "builder": bonuses / total,
            "opportunistic": max(attacks, bonuses) / total,
        }
        for name, share in shares.items():
            self.policy_scores[name] = self.policy_scores[name] * 0.8 + share

    @property
    def policy_belief(self):
        total = sum(self.policy_scores.values()) or 1.0
        return {name: score / total for name, score in self.policy_scores.items()}

    @property
    def attack_rate(self):
        total = sum(a + d + b for a, d, b, _ in self.actions)
        return sum(a for a, _, _, _ in self.actions) / total if total else 1 / 3

    @property
    def defend_rate(self):
        total = sum(a + d + b for a, d, b, _ in self.actions)
        return sum(d for _, d, _, _ in self.actions) / total if total else 1 / 3

    @property
    def bonus_rate(self):
        total = sum(a + d + b for a, d, b, _ in self.actions)
        return sum(b for _, _, b, _ in self.actions) / total if total else 1 / 3

    @property
    def passive_streak(self):
        streak = 0
        for attacks, _, _, _ in reversed(self.actions):
            if attacks:
                break
            streak += 1
        return streak


@dataclass(frozen=True)
class PublicEvent:
    """Facts visible after an allocation, never hidden resolver intent."""

    attacks: int
    bonuses: int
    switched: int
    resolved_shields: int | None = None

    @property
    def attack_rate(self):
        total = sum(a + d + b for a, d, b, _ in self.actions)
        return sum(a for a, _, _, _ in self.actions) / total if total else 1 / 3

    @property
    def defend_rate(self):
        total = sum(a + d + b for a, d, b, _ in self.actions)
        return sum(d for _, d, _, _ in self.actions) / total if total else 1 / 3

    @property
    def bonus_rate(self):
        total = sum(a + d + b for a, d, b, _ in self.actions)
        return sum(b for _, _, b, _ in self.actions) / total if total else 1 / 3


@dataclass(frozen=True)
class ShieldBelief:
    probabilities: dict[int, float]

    @property
    def expected(self):
        return sum(k * v for k, v in self.probabilities.items())

    @property
    def maximum(self):
        return max(self.probabilities)


@dataclass(frozen=True)
class AllocationHypothesis:
    allocation: Allocation
    probability: float


@dataclass(frozen=True)
class TacticalFacts:
    damage_per_hit: int
    hits_to_kill: int
    attacks_for_guaranteed_lethal: int
    lethal_probability: float
    guaranteed_lethal: bool
    expected_damage: float


@dataclass(frozen=True)
class TacticalOutcome:
    guaranteed_lethal: bool
    kill_and_defend: bool
    guaranteed_immediate_loss: bool
    lethal_probability: float
    wins_match: bool = False


class Planner:
    def __init__(self, depth=3, branch_limit=None):
        self.depth = depth
        self.branch_limit = branch_limit
        self.history = PublicHistory()
        self.model = OpponentModel()
        self.objective = Objective()
        self.last_report = {}
        self._search_cache = {}
        self.passive_streak = 0
        self._observed_turns = 0

    def observe(self, attacks, bonuses=None, switched=False, budget=None,
                turn=None):
        """Record a public opponent allocation.

        ``bonuses`` is the opponent's hidden split and is deliberately ignored;
        it is accepted only so existing callers keep working. What is public is
        the budget, the attack count, and a visible switch. Everything else is
        a remainder that stays ambiguous until later evidence resolves it.
        """
        if turn is None:
            turn = self._observed_turns * 2 + 1
        self._observed_turns += 1
        if budget is None:
            budget = attacks + int(switched)
        self.model.observe_turn(turn, budget, attacks, switched)
        self.history.observe(attacks, bonuses or 0, switched, budget)

    def observe_shields(self, shields: int):
        """Our attack met this many shields, which pins the live worlds."""
        self.model.observe_our_attack(shields + 1, shields)
        self.history.reveal_latest_defends(shields)

    def belief(self, opponent: GameState | object) -> ShieldBelief:
        """Distribution over the opponent's currently held shields.

        Derived only from the public remainder and evidence that narrowed it.
        Never from ``state.opponent.shields`` or ``state.opponent.bonus``.
        """
        return ShieldBelief(self.model.shield_distribution())

    def opponent_budgets(self, state: GameState) -> dict[int, float]:
        """Distribution over the opponent's next public budget.

        Their bank is hidden, so their next budget is a distribution, not a
        number. Reading ``state.opponent.bonus`` here would be a leak.
        """
        budget_turn = state.turn + (1 if state.player_to_move else 0)
        distribution = self.model.next_budget_distribution(budget_turn)
        return distribution or {base_budget(budget_turn): 1.0}

    def opponent_allocations(self, state: GameState) -> tuple[AllocationHypothesis, ...]:
        """Estimate legal next allocations using only public behavior history."""
        side = state.opponent
        hypotheses: list[AllocationHypothesis] = []
        for budget, budget_probability in self.opponent_budgets(state).items():
            forecast = replace(side, bonus=0, actions=budget)
            candidates = legal_allocations(forecast)
            weights = []
            for move in candidates:
                total = max(1, move.attacks + move.defends + move.bonuses)
                attack_share = move.attacks / total
                defend_share = move.defends / total
                bonus_share = move.bonuses / total
                weight = (
                    0.15
                    + attack_share * self.history.attack_rate
                    + defend_share * self.history.defend_rate
                    + bonus_share * self.history.bonus_rate
                )
                policy = self.history.policy_belief
                weight += policy["aggressive"] * attack_share * 0.2
                weight += policy["defensive"] * defend_share * 0.2
                weight += policy["builder"] * bonus_share * 0.2
                weights.append(weight)
            total_weight = sum(weights) or 1.0
            hypotheses.extend(
                AllocationHypothesis(move, budget_probability * weight / total_weight)
                for move, weight in zip(candidates, weights)
            )
        return tuple(hypotheses)

    def facts(self, state: GameState, allocation: Allocation, belief: ShieldBelief):
        me, enemy = state.player, state.opponent
        character = me.characters[allocation.switch_to if allocation.switch else me.active]
        target = enemy.active_character
        damage = exchange_damage(character, target, 1)
        needed = attacks_to_kill(character, target, 0) or MAX_ACTIONS + 1
        values = [max(0, allocation.attacks - shields) for shields in belief.probabilities]
        lethal = sum(p for shields, p in belief.probabilities.items()
                     if attacks_to_kill(character, target, shields) is not None
                     and allocation.attacks >= attacks_to_kill(character, target, shields))
        expected = sum(min(target.hp, exchange_damage(
            character, target, max(0, allocation.attacks - shields))) * p
                       for shields, p in belief.probabilities.items())
        return TacticalFacts(
            damage, needed, needed + belief.maximum, lethal,
            all(allocation.attacks >= (attacks_to_kill(character, target, shields) or MAX_ACTIONS + 1)
                 for shields in belief.probabilities), expected)

    def tactical_outcome(self, state: GameState, move: Allocation,
                         belief: ShieldBelief, facts: TacticalFacts):
        """Evaluate hard tactical facts across every live shield world.

        A move loses if *any* world permits a forced loss, and wins only if
        *every* world ends the match. Averaging over worlds would hide exactly
        the disaster branch that matters.
        """
        guaranteed_loss = True
        wins_match = True
        for shields in belief.probabilities:
            resolver_state = replace(
                state, opponent=replace(state.opponent, shields=shields))
            child = apply(resolver_state, move)
            if not child.opponent.lost:
                wins_match = False
            world_loss = False
            if not child.opponent.lost:
                for reply in legal_allocations(child.opponent):
                    if apply(child, reply).player.lost:
                        world_loss = True
                        break
            if not world_loss:
                guaranteed_loss = False
        return TacticalOutcome(
            guaranteed_lethal=facts.guaranteed_lethal,
            kill_and_defend=facts.guaranteed_lethal and move.defends > 0,
            guaranteed_immediate_loss=guaranteed_loss,
            lethal_probability=facts.lethal_probability,
            wins_match=wins_match,
        )

    def choose(self, state: GameState) -> Allocation:
        # Resolver states contain secrets. Enforce the public-information
        # boundary here so callers cannot accidentally expose the enemy's held
        # shields or stored bank. Both are hidden; belief supplies them.
        state = replace(state, opponent=replace(state.opponent, shields=0,
                                                bonus=0))
        bank_belief = self.model.bank_distribution()
        # A burst threat is about the disaster branch, not the mean. Use the
        # largest bank that is still credible, so a possible spike is denied
        # rather than averaged away.
        credible_bank = max((bank for bank, p in bank_belief.items() if p >= 0.15),
                            default=0)
        self._search_cache.clear()
        effective_depth = self.depth + (
            1 if state.player.alive_count + state.opponent.alive_count <= 3 else 0)
        belief = self.belief(state)
        switch_values = [switch_value(state, index, belief)
                         for index, character in enumerate(state.player.characters)
                         if index != state.player.active and character.alive]
        candidates = legal_allocations(state.player)
        if self.branch_limit:
            candidates = tuple(sorted(candidates, key=lambda x: (x.attacks, x.bonuses, x.defends), reverse=True)[:self.branch_limit])
        candidate_facts = [(move, self.facts(state, move, belief)) for move in candidates]
        best_lethal = max((facts.lethal_probability for _, facts in candidate_facts), default=0.0)
        expected_incoming = self._expected_incoming(state, 0)
        self.objective.update(
            state,
            lethal_probability=best_lethal,
            expected_incoming=expected_incoming,
            attack_rate=self.history.attack_rate,
            passive_streak=self.passive_streak,
            turn=state.turn,
            opponent_bank=credible_bank,
        )
        switch_by_target = {value.target: value for value in switch_values}
        scored = []
        for move, facts in candidate_facts:
            continuations = []
            for shields, probability in belief.probabilities.items():
                resolver_state = replace(
                    state, opponent=replace(state.opponent, shields=shields))
                child = apply(resolver_state, move)
                continuations.append(
                    (self._search(child, effective_depth - 1,
                                  state.player_to_move), probability))
            expected_continuation = sum(value * probability
                                        for value, probability in continuations)
            worst_continuation = min(value for value, _ in continuations)
            continuation = expected_continuation * 0.8 + worst_continuation * 0.2
            incoming = self._expected_incoming(state, move.defends)
            survival = max(0.0, state.player.active_character.hp - incoming)
            future_bonus = marginal_bonus_value(state, move.bonuses) * 0.35
            switch = switch_by_target.get(move.switch_to)
            switch_score = switch.value * 0.8 if switch else 0.0
            objective_score = self._objective_score(
                move, facts, survival, state.player.active_character.hp)
            score = continuation + facts.expected_damage * 0.6
            score += facts.lethal_probability * 5000 + future_bonus
            score += survival * 0.25 + switch_score + objective_score
            known_unshielded = belief.probabilities == {0: 1.0}
            if self.history.attack_rate >= 0.65 and known_unshielded:
                score += facts.expected_damage * 0.9 + move.attacks * 250
                if move.attacks == 0:
                    score -= 1800
            if state.turn <= 4 and known_unshielded and move.attacks:
                score += facts.expected_damage * 1.1 + move.attacks * 600
            elif state.turn <= 4 and known_unshielded and move.attacks == 0:
                score -= 2200
            if self.passive_streak >= 2 and move.attacks == 0:
                score -= 4000 + self.passive_streak * 1000
            components = {
                "continuation": continuation,
                "expected_damage": facts.expected_damage * 0.6,
                "lethal_probability": facts.lethal_probability * 5000,
                "future_bonus": future_bonus,
                "survival": survival * 0.25,
                "switch_value": switch_score,
                "objective": objective_score,
                "expected_incoming": incoming,
                "passive_streak": self.passive_streak,
            }
            outcome = self.tactical_outcome(state, move, belief, facts)
            components["guaranteed_lethal"] = outcome.guaranteed_lethal
            components["kill_and_defend"] = outcome.kill_and_defend
            components["guaranteed_immediate_loss"] = outcome.guaranteed_immediate_loss
            scored.append((score, move, facts, components, outcome))
        # Gate precedence matters. A move that ends the match is unconditionally
        # best: nothing can follow it. A kill that merely removes one body is
        # not that, and must never override survival, because the opponent still
        # gets a reply. Safety is therefore checked before ordinary lethality.
        match_winners = [item for item in scored if item[4].wins_match]
        if match_winners:
            scored = match_winners
        else:
            safe = [item for item in scored if not item[4].guaranteed_immediate_loss]
            if safe:
                scored = safe
        scored.sort(key=lambda item: (
            item[4].wins_match,
            not item[4].guaranteed_immediate_loss,
            item[4].kill_and_defend,
            item[4].guaranteed_lethal,
            item[4].lethal_probability,
            item[0],
        ), reverse=True)
        score, move, facts, components, outcome = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else score
        self.last_report = {
            "objective": self.objective.name,
            "objective_reason": self.objective.reason,
            "belief": belief.probabilities,
            "selected": move.label,
            "facts": facts.__dict__,
            "tactical_outcome": outcome.__dict__,
            "score_components": components,
            "move_quality": {
                "rank": 1,
                "score": score,
                "best_score": score,
                "regret_vs_best": 0.0,
                "margin_over_second": score - second_score,
            },
            "alternatives": [(item[1].label, item[0], item[3]) for item in scored[:8]],
            "switch_values": [value.__dict__ for value in switch_values],
            "opponent_allocations": [
                {"move": item.allocation.label, "probability": item.probability}
                for item in self.opponent_allocations(state)[:12]
            ],
            "policy_belief": self.history.policy_belief,
        }
        self.passive_streak = self.passive_streak + 1 if move.attacks == 0 else 0
        return move

    def _expected_incoming(self, state: GameState, shields: int) -> float:
        """Estimate incoming damage from a public-history allocation distribution."""
        enemy = state.opponent.active_character
        return sum(
            exchange_damage(
                enemy, state.player.active_character,
                max(0, hypothesis.allocation.attacks - shields))
            * hypothesis.probability
            for hypothesis in self.opponent_allocations(state)
        )

    def _objective_score(self, move, facts, survival, active_hp):
        if self.objective.name == "finish":
            return facts.lethal_probability * 3500 + (2000 if facts.guaranteed_lethal else 0)
        if self.objective.name == "survive":
            return survival * 1.2 + move.defends * 500 - facts.expected_damage * 0.2
        if self.objective.name == "prepare_burst":
            return move.bonuses * 900 + move.defends * 350 - move.attacks * 80
        if self.objective.name == "deny_burst":
            return move.defends * 1000 + move.bonuses * 450 - move.attacks * 120
        if self.objective.name == "break_stall":
            return move.attacks * 1200 + move.bonuses * 700 - move.defends * 150
        return 0.0

    def _search(self, state, depth, root_turn):
        key = (state, depth, root_turn)
        if key in self._search_cache:
            return self._search_cache[key]
        if state.player.lost or state.opponent.lost:
            value = self.evaluate(state, depth)
            self._search_cache[key] = value
            return value
        if depth <= 0:
            terminal_values = [
                self.evaluate(child, depth)
                for move in legal_allocations(
                    state.player if state.player_to_move else state.opponent)
                if (child := apply(state, move)).player.lost or child.opponent.lost
            ]
            if terminal_values:
                value = (max(terminal_values) if state.player_to_move
                         else min(terminal_values))
            else:
                value = self.evaluate(state, depth)
            self._search_cache[key] = value
            return value
        values = []
        side = state.player if state.player_to_move else state.opponent
        for move in legal_allocations(side):
            values.append(self._search(apply(state, move), depth - 1, root_turn))
        value = ((max(values) if state.player_to_move == root_turn else min(values))
                 if values else self.evaluate(state, depth))
        self._search_cache[key] = value
        return value

    @staticmethod
    def evaluate(state, depth=0):
        if state.opponent.lost:
            return 100000 + max(0, depth) * 100
        if state.player.lost:
            return -100000 - max(0, depth) * 100
        material = (sum(c.hp for c in state.player.characters)
                    - sum(c.hp for c in state.opponent.characters)) * 1.5
        bodies = (state.player.alive_count - state.opponent.alive_count) * 2600
        bonus = (state.player.bonus - state.opponent.bonus) * 450

        def pressure(actor: Side, target: Side, budget: int) -> float:
            best = 0
            for index, character in enumerate(actor.characters):
                if not character.alive:
                    continue
                attacks = max(0, budget - (index != actor.active) - target.shields)
                best = max(best, exchange_damage(
                    character, target.active_character, attacks))
            lethal = 3500 if best >= target.active_character.hp else 0
            return min(best, target.active_character.hp) + lethal

        if state.player_to_move:
            player_budget = state.player.actions
            opponent_budget = next_budget(state.turn + 1, state.opponent)
        else:
            player_budget = next_budget(state.turn + 1, state.player)
            opponent_budget = state.opponent.actions
        tempo = pressure(state.player, state.opponent, player_budget)
        tempo -= pressure(state.opponent, state.player, opponent_budget)
        return material + bodies + bonus + tempo * 0.7
