# COTE Megaverse

COTE Megaverse is a human-fair, hidden-intent, turn-based combat game. This
file is the normative project specification. Code, UI, planner, benchmark,
replay, and tests must follow it.

`GameState` is authoritative resolver state, not a public observation. It
contains secrets needed to resolve combat. A field being present in
`GameState` never implies that the opponent may see or use it.

## 1. Match Model

- A match has two sides: human/player and AI/opponent.
- Each side has exactly three characters. Duplicate types are allowed.
- Each character has one type: `A`, `B`, `C`, or `D`.
- Initial HP is sampled independently from
  `5700, 5800, 5900, 6000, 6100, 6200, 6300`.
- Initial ATK is sampled independently from `1900, 2000, 2100`.
- `max_hp` is the character's sampled initial HP.
- A character is alive while `hp > 0`.
- A side loses when all three of its characters are dead.
- `turn` counts individual side turns, not pairs of turns.
- Either side may act first. Interactive play defaults to the human acting
  first, but the starting side is a match parameter, not a fixed rule.
- Moving first is a measured structural advantage, not a cosmetic detail. With
  identical policies on both seats, the first mover wins about 68 percent of
  games with a random policy and about 58 percent with a greedy policy.
  Therefore every benchmark must play each seed twice, once per starting side,
  and report the seats separately as well as combined. A win rate measured on a
  single seat is not a valid strength measurement.
- Pass a seeded `random.Random` to `initial()` when reproducible teams and
  stats are required.

## 2. Character Types

Type advantage forms this cycle:

```text
A > B > C > D > A
```

- Attacking the next type in the cycle uses multiplier `1.3`.
- Attacking the previous type in the cycle uses multiplier `0.7`.
- Every other matchup, including equal types, uses multiplier `1.0`.
- Type, ATK, current HP, max HP, alive/dead status, and active character are
  public information.

## 3. Turns And Action Budgets

Base actions are determined by the global turn number:

| Turn | Base actions |
| ---: | ---: |
| 1 | 1 |
| 2-4 | 2 |
| 5-6 | 3 |
| 7+ | 4 |

- At the start of the acting side's turn, its available actions are
  `min(8, base actions + stored bonus)`.
- Turn preparation always drains the entire stored bonus into the budget.
  Because base actions never exceed `4` and the bank is capped at `4`, the sum
  never exceeds the action cap, so preparation always leaves the bank at zero.
  A partially spent bank is unreachable, and during its own allocation a side
  always has bank zero and full bank capacity `4`.
- The stored bonus is capped at `4` and is hidden while it is stored.
- The acting side's action budget is public. Because preparation drains the
  whole bank, `bank = budget - base actions` becomes public at the moment the
  bank is spent, which is the owner's next turn. Banking is a delayed reveal,
  not a permanent secret: an action banked on turn `N` is visible on turn
  `N + 2`.
- Available actions are capped at `8`.
- Every available action must be spent. Passing or leaving actions unused is
  illegal.
- Actions form one unordered `Allocation`; they are not resolved as a command
  sequence.

An action can be allocated to:

- `attack`: one attempted attack in the current exchange;
- `defend`: one shield held against the opponent's next turn;
- `bonus`: one stored action for a later turn, up to the bonus cap;
- `switch`: one voluntary active-character change, costing one action.

The allocation invariant is:

```text
attacks + defends + bonuses + (1 if switched else 0) == available actions
```

## 4. Switches, Order, And Death

- A voluntary switch may target any living, non-active character.
- A voluntary switch costs one action.
- At most one voluntary switch may be selected in an allocation.
- A switch happens before that allocation's attack damage is calculated.
- Therefore attacks in the same allocation use the newly active character's
  type and ATK.
- A voluntary switch moves the selected character to the front of
  `stack_order`.
- If an active character dies, the next living character is promoted for free.
- Forced promotion does not consume an action and is not a voluntary switch.
- The promoted character becomes first in `stack_order`; the defeated active
  character is retained immediately behind it in the internal order.
- The UI always renders the active character first. Other living characters
  follow the real `stack_order`; display order is part of game state, not a
  cosmetic sort.
- If no living character remains, the match ends immediately.

## 5. Shields And Resolution

Shields have a one-opponent-turn lifecycle:

1. A side allocates `defends` on its turn.
2. Those `defends` become that side's held shields after its allocation
   resolves.
3. They do not protect against attacks in the allocation where they were
   created.
4. They protect against the opponent's next allocation.
5. Each shield blocks at most one attempted attack.
6. All held shields are cleared after that opponent allocation resolves,
   whether or not the opponent attacked and whether or not every shield
   blocked an attack.

For an attack exchange:

```text
blocked = min(attempted attacks, defender held shields)
landed = attempted attacks - blocked
```

- Shields cannot reduce landed attacks below zero.
- Defends selected by the attacker become future shields only after current
  damage resolution.
- Damage cannot reduce HP below zero.

## 6. Damage And Kill Arithmetic

Damage is computed for the entire exchange and rounded exactly once:

```text
raw damage = attacker ATK * type multiplier * landed attacks
damage = round(raw damage / 100) * 100
```

- Python's `round` behavior used by `rounded_damage()` is authoritative.
- Zero landed attacks deal zero damage.
- Never calculate exchange damage as `rounded single-hit damage * landed`.
- Resolver, planner facts, switch calculations, replay, and UI must use
  `exchange_damage()` rather than duplicate the formula.
- Displayed damage is capped at the target's HP for the completed resolution;
  resolver HP is clamped to zero.

Kill preview has a deliberately narrower meaning:

- It is the minimum number of landed, unblocked attacks whose aggregate damage
  reaches the target's current HP.
- It ignores held, possible, and hypothesized opponent shields.
- It must use the same aggregate rounding as the resolver.
- Example: `2100 ATK * 0.7` against `6000 HP` deals `5900` with four landed
  attacks and `7400` with five. The preview is `5 attacks needed`.
- Internal planner calculations may add hypothesized blocked attacks when
  evaluating a shield belief. Such totals must never replace the UI's landed
  attack preview.

## 7. Information Model

### 7.1 Public Before An Allocation

Both sides may know:

- all character types, ATK, current HP, max HP, and alive/dead status;
- both active characters and stack orders;
- turn number and acting side;
- the acting side's public action budget, and therefore the bank it just
  drained;
- all previously resolved combat results and visible switches.

Each side also knows its own held shields.

### 7.2 Hidden Before Resolution

A side must not know or receive the opponent's:

- currently held shield count;
- currently stored bonus bank;
- current allocation's attacks;
- current allocation's defends;
- current allocation's bonuses;
- current switch choice.

The allocation remains hidden while it is being chosen. Resolver access to
these values does not make them public.

Three channels leak information at three different delays, and the gap between
them is where bluffing lives:

- An attack reveals itself immediately and completely. Attacks are public on
  resolution, so a full-budget attack proves a remainder of zero.
- A bank is revealed when it is spent, not when it is taken. An action banked on
  turn `N` becomes public on turn `N + 2` through the budget.
- A shield is revealed only when an attack hits it or when it expires unused. A
  shield that is never tested is never confirmed.

### 7.3 Public After Resolution

- A visible switch and its target become public.
- If attacks were attempted, total attacks, shields encountered, landed hits,
  and resulting damage become public.
- The defender's held shields are therefore known after an attack exchange.
- If no attack occurred, the exact composition of the opponent's hidden
  allocation is not printed. Public state changes may still reveal bonus or a
  switch normally.
- On every human allocation resolution, the UI explicitly reports the AI's
  expired shield count as `AI spent X shields`, including `X == 0` and turns
  where the human did not attack.

## 8. Human-AI Fairness Contract

- Human and AI must make decisions from equivalent information boundaries.
- AI knows its own state and held shields, just as the human knows theirs.
- `Planner.choose()` must mask exact opponent shields even if a caller passes a
  full resolver `GameState`.
- Planner continuation must preserve chosen shields in resolver state so they
  block the next simulated exchange. Masking may affect decision input, but
  must never erase shields before resolution.
- At the search horizon, planner must still detect an immediately available
  legal match-ending move. A forced win or loss on the next allocation must
  not be replaced by a nonterminal HP evaluation merely because depth reached
  zero.
- Before resolution, AI uses legal-allocation hypotheses, public history, and a
  shield belief. It must not use exact hidden opponent shields.
- Public action accounting is mandatory, but it yields a remainder, not a
  shield count. Given the opponent's known budget, public attacks, and visible
  switch cost, `remainder = budget - attacks - switch cost` is the number of
  actions that went to defends or bonuses in unknown proportion. Deriving that
  remainder is fair inference, not resolver-state leakage.
- The remainder must never be treated as the held-shield count. Every split
  `defends + bonuses == remainder` is a distinct world. A remainder of `r`
  yields `r + 1` candidate shield counts, `0..r`. This ambiguity is the core of
  the game and must not be collapsed.
- A remainder of zero is the one exact case: the opponent holds no shields and
  banked nothing. Attacking with the full budget is therefore self-revealing.
  Spending every action on attacks buys damage at the price of total
  transparency on the following turn.
- Shield worlds are pruned by later public facts, not by assumption. A revealed
  budget of `b` on the owner's next turn proves the bank was `b - base actions`,
  which retroactively fixes the split of the earlier remainder. An attack
  exchange reveals the defender's shields directly. Until such a fact arrives,
  all splits stay live.
- Belief about currently held shields and prediction of the opponent's future
  allocation are separate models and must not be conflated.
- Because held shields are almost never exactly inferable, candidate moves must
  resolve in separate shield-world states. Continuation value combines
  probability-weighted expectation with downside risk; it must not resolve the
  move once against an invented zero-shield state.
- Shield-world priors must use opponent rationality, not a uniform split. A
  split that hands the mover a guaranteed lethal is evidence against itself
  when the opponent had a safe alternative.
- Safety gates are evaluated across all live shield worlds. A move is losing if
  some world permits a guaranteed loss, and lethal only if every world is
  lethal. Prefer a move that loses in no world over a move with a higher score
  that loses in one.
- When no more than three characters remain alive across both sides, planner
  extends search beyond normal depth to reduce endgame horizon errors.
- Evaluation must account for material, living bodies, own bank, the believed
  distribution over the opponent's bank, next-turn action budgets,
  shield-adjusted pressure, immediate lethal, and preservation of a living
  character through switching.
- Tempo is a first-class term. An action banked is an action kept; an action
  attacked into a possible shield may be worth nothing. Evaluation must reward
  keeping future options, not only current HP difference.
- Expected incoming damage must use aggregate `exchange_damage()` for each
  hypothesized landed-attack count, never rounded single-hit damage multiplied
  by attacks.
- Root candidate selection uses hard tactical gates before heuristic ranking:
  guaranteed lethal, guaranteed immediate loss, and `kill + defense` are
  explicit tactical facts. Strategic score is only a tie-break among moves
  with comparable tactical safety.
- A guaranteed lethal plus defense outranks a bare guaranteed lethal when both
  finish the current target. A move that permits guaranteed immediate loss is
  excluded when a non-losing legal alternative exists.
- Bonus preparation and switch value may influence nonterminal choices, but
  cannot override an available tactical safety gate.
- When the opponent's remainder is exactly zero and no tactical safety gate
  applies, concrete damage pressure outranks passive bonus preparation. Do not
  concede free tempo by banking every action against a defenceless opponent.
- Planner and evaluation must never read the opponent's stored `bonus` field.
  Only a revealed budget may be used to reconstruct a past bank. Predicting the
  opponent's next budget requires a distribution over its hidden bank, not the
  resolver value.
- Public history may record attacks, revealed budgets, and visible switches.
- Structured `PublicEvent` history may be used for public facts, but it must
  never contain hidden allocation intent. Policy belief is a soft forecast of
  opponent style, not permission to use private state.
- Exact shields may enter planner history only after an attack exchange has
  publicly revealed them.
- Interactive play, position analysis, benchmark, replay, and self-play must
  enforce the same boundary.
- Self-play must use independent planner histories and alternate sides through
  the same `apply()` transition used by interactive play.
- Never reintroduce perfect-information policies.

## 9. Terminal UI Contract

### 9.1 State Preview

- Render human side, AI side, current turn, acting side, characters, HP bars,
  HP, ATK, types, active marker, and living stack order.
- Human-held shields may appear on the human active character.
- AI-held shields must never appear in normal state preview.
- Before a human allocation, show the human bonus meter and
  `Actions: base + bonus`. This is the human's own bank, which the human owns
  and may see.
- The AI's stored bank must never be rendered, in any turn, in any form. Only
  the AI's action budget on its own turn is public, and that budget reveals the
  bank it just drained. Never render an AI bonus meter, AI held shields, or an
  AI allocation.
- Matchup multiplier and potential one-attack aggregate damage may be shown.
- Show `[KILL possible] N attacks needed` only when the unblocked kill-preview
  count is within the human's currently available actions.
- Never include possible or current AI shields in kill-preview count.

### 9.2 Allocation Input

- Human commands are `a=atk`, `s=shield`, `b=bonus`, and `sw=switch`.
- Prompt until every available action has been allocated.
- Reject bonus allocation above the cap.
- Reject dead, active, nonexistent, duplicate, or otherwise illegal switch
  targets.
- The human naturally sees the allocation they are entering.
- Never print `AI selected: aN/dN/bN`, `move.label`, planner alternatives, or
  any equivalent AI allocation summary in interactive play.

### 9.3 Resolution Output

- Start a separate resolution block labeled `You turn #N` or `AI turn #N`.
- Show a resolved switch when one occurred.
- On every `You turn #N`, print `AI spent X shields` using the AI shields held
  before this resolution. Never use the AI's newly selected future `defends`.
- Never print `You spent X shields`.
- A fully blocked attack prints `BLOCK!` and
  `<side>: N attacks vs X shields`.
- A partially blocked or unblocked attack prints total attempted attacks,
  resolved shields, landed hits, and aggregate damage.
- If no attack was allocated, print `<side> did not attack`.
- `AI spent X shields` still appears when the human did not attack.
- The completed exchange may reveal shields; this is a public result, not an
  information leak.

### 9.4 Match End

- The final move uses the same normal resolution block as every other move.
- After that block, do not render another state table or a `BATTLE OVER`
  preview.
- Do not show actions, matchup, potential damage, or kill preview after game
  over.
- Print only `YOU WIN`, `YOU LOSE`, or `DRAW` as the final result.

## 10. State And Transition Invariants

- `Character`, `Side`, `GameState`, and `Allocation` are immutable value
  objects; transitions return new state.
- `rules.py` is the single authority for legal allocations and resolution.
- `apply()` validates that the complete budget was spent.
- The acting side's current allocation resolves against shields already held
  by the target.
- The actor's newly selected defends and bonuses are stored after resolution.
- The target's old shields are cleared after resolution.
- Turn increments by one, acting side flips, and the next side is prepared.
- `prepare()` drains the whole bank into the budget. Consequently `Side.bonus`
  is zero during its owner's own allocation, and a nonzero `Side.bonus` is
  always the opponent's hidden stored bank. Reading that field from the
  planner is a fairness violation, not an optimisation.
- `Side.shields` is likewise hidden state. Only `observation.py` decides what
  crosses the boundary.
- UI and planner must not independently mutate resolver state.
- Seeded RNG is required for reproducible tests, benchmark, and replay.
- Benchmark must include seeded human-like policies (`random`, `greedy`, and
  `bonus_shield`) and report wins, losses, draws, and missed guaranteed lethal
  counts separately. Self-play alone is not a human-strength benchmark.

## 11. Source Layout

- `src/cote_megaverse/rules.py`: authoritative immutable state, legal moves,
  damage, and transitions.
- `src/cote_megaverse/observation.py`: public observation boundary.
- `src/cote_megaverse/agent.py`: public-history belief model and bounded
  planner.
- `src/cote_megaverse/strategy.py`: objectives and switch valuation.
- `src/cote_megaverse/interactive.py`: human-fair terminal game and display
  contract.
- `src/cote_megaverse/benchmark.py`: seeded fair self-play, analysis, and
  replay reports.
- `src/cote_megaverse/cli.py`: position-analysis entry point.
- `tests/test_new_engine.py`: core rules and planner tests.
- `tests/test_layers.py`: observation, fairness, strategy, and benchmark tests.
- `tests/test_interactive.py`: terminal visibility and rendering tests.

## 12. Verification And Commands

Run the complete suite after changes to rules, state, planner, visibility, or
UI:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
python -m src.cote_megaverse.interactive
python -m src.cote_megaverse.cli --team A,B,C --opponent B,C,C --depth 3
python -m py_compile src/cote_megaverse/rules.py src/cote_megaverse/agent.py src/cote_megaverse/observation.py src/cote_megaverse/interactive.py src/cote_megaverse/benchmark.py src/cote_megaverse/cli.py
git diff --check
```

Run a seeded AI-vs-human-policy baseline:

```powershell
python -c "from src.cote_megaverse.benchmark import benchmark_policies; print(benchmark_policies(seeds=range(100), depth=2, max_half_turns=100))"
```

The benchmark policies are `random`, `greedy`, and `bonus_shield`. Every seed
is played once with the human policy first and once with AI first. Do not
interpret self-play win rate as human win rate. Record wins, losses, draws,
missed guaranteed lethals, and guaranteed-loss moves separately.

Do not reintroduce genome evolution, LSTM/NumPy/Numba legacy planners, legacy
compatibility wrappers, or perfect-information behavior without an explicit
design decision and corresponding specification update.

## 13. Known Defects

These are confirmed by measurement and must be fixed before any win-rate
number is trusted.

- `benchmark.py` calls `planner.observe_shields(before.opponent.shields)` on the
  human's turn. When the human is `player`, `before.opponent` is the AI itself,
  so the AI's own shields are written into its model of the human. Belief
  top-1 was wrong in 9 of 23 audited plies. `interactive.py` passes
  `before.player.shields` and is correct. Because of this, live play and the
  benchmark measured different games, and all historical win-rate numbers from
  the benchmark are void.
- `PublicHistory.observe()` computes `shields = budget - attacks - bonuses -
  switch`, using the opponent's true `bonuses`. That value is hidden at
  decision time. It must derive a remainder and keep every split live.
- `Planner.belief()` returns `ShieldBelief({n: 1.0})` from that derivation, so
  the shield-world machinery required by section 8 never activates. All
  candidate moves resolve in a single invented world.
- `Planner.opponent_allocations()` reads `state.opponent.bonus` through
  `next_budget()`. That is the opponent's hidden bank.
- `PublicEvent.attack_rate`, `defend_rate`, and `bonus_rate` reference
  `self.actions`, which does not exist on that class. They raise
  `AttributeError` if called and are dead code.
- `observation.py` copies `side.bonus` into `PublicSide` for both sides, so the
  public observation leaks the opponent's hidden bank. Only the acting side's
  own bank and the opponent's revealed budget may cross that boundary.
  `PublicObservation.opponent_next_budget` is built from the same leaked field
  and must instead return a distribution over the hidden bank.
- `tests/test_layers.py::test_planner_infers_held_shields_from_public_budget`
  asserts the old, wrong model: budget 5 with 3 attacks and 0 bonuses gives
  `held_shields == 2`. Under section 8 that situation yields a remainder of 2
  and three live shield worlds. This test must be replaced, not preserved.
- Benchmark reporting must alternate seats and report per-seat results, because
  moving first is worth roughly 68 percent for `random` and 58 percent for
  `greedy` in symmetric self-play. A single-seat number is not a strength
  measurement.
