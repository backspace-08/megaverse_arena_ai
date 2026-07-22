# COTE Megaverse: AI Handoff

## Purpose

This document is a compact context handoff for another AI working on the
project. Treat the rules below as the current source of truth unless the code
or tests clearly contradict them. Do not redesign the game from assumptions.

The project is a Python 3.14.5 + NumPy implementation of a turn-based,
hidden-intent combat game inspired by COTE Megaverse. The main engineering
goal is a strong adaptive AI that can play humans and diverse bots, recognize
opponent patterns during a match, plan burst turns, avoid shield deadlocks,
and remain robust instead of overfitting to one deterministic counter-policy.

## Repository Map

- `src/cote_megaverse/parameterized_ai_v2.py`: game rules, data classes, engine,
  baseline AI profiles.
- `src/cote_megaverse/coevolution.py`: neural and Smart agents, opponent model,
  robust macro-action planner, expectimax, co-evolution and fitness.
- `src/cote_megaverse/play_vs_champion.py`: interactive human-vs-champion game,
  champion loading, benchmark and version listing.
- `src/cote_megaverse/play_vs_anchor.py`: focused interactive anchor practice.
- `src/cote_megaverse/anchor_battle.py`: round-robin/reference battles.
- `tests/`: mechanics, Smart, UI and golden replay tests.
- `docs/`: human-readable reference logs and notes.
- `artifacts/`: ignored generated genomes, evolution results and training logs.

Useful commands on Windows:

```powershell
python -m unittest discover -s tests -p 'test_*.py' -q
python tests/test_mechanics.py
python -m py_compile src/cote_megaverse/coevolution.py src/cote_megaverse/parameterized_ai_v2.py src/cote_megaverse/play_vs_champion.py
python -m src.cote_megaverse.coevolution
python -m src.cote_megaverse.play_vs_champion
```

`pytest` is not guaranteed to be installed. Prefer the built-in `unittest`
commands above unless the environment explicitly provides pytest.

Current verification is maintained by the full unittest suite and compile
checks. Do not preserve stale test counts here; report the result from the
latest run.

The Smart line now also includes a deterministic robust tactical planner. The
planner is intentionally not a stochastic MCTS implementation: the game has a
small macro-action space and deterministic mechanics, so exact allocation
search plus strong opponent scenarios gives more reproducible diagnostics.

Do not use production evolution as a validation command. Small explicit
one-generation runs are allowed only for integration checks.

## Game Model

### Teams and characters

Each player has exactly three characters. A character has:

- `char_type`: one of `A`, `B`, `C`, `D`;
- HP, max HP and attack;
- active/alive state.

Default randomized stats:

- HP is selected from `5700, 5800, ..., 6300`;
- attack is selected from `1900, 1950, 2000, 2050, 2100`;
- nominal values are `BASE_HP=6000` and `BASE_ATK=2000`.

Character numbers are logical indices and must never be renumbered. If the
active character changes, its original index remains its identity.

### Type circle

The directed advantage cycle is:

```text
A > B > C > D > A
```

For an attack from type `X` to defender type `Y`:

- advantage: damage multiplier `1.3`;
- disadvantage: damage multiplier `0.7`;
- same or unrelated pair: multiplier `1.0`.

Damage is calculated from the attacking character's ATK only. The defender's
ATK is irrelevant while defending. For an exchange:

```text
final_attack_count = attack_count - blocked_shields
damage_raw = attacker.atk * final_attack_count * get_type_multiplier(attacker_type, defender_type)
damage = round_damage(damage_raw)
round_damage(x) = round(x / 100) * 100
```

All attacks target the opponent's current active character. There is no
manual target selection for attacks in the current engine.

### Action budget

Each own turn receives a base action budget according to the global turn
number:

| Turn | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8+ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base actions | 1 | 2 | 2 | 2 | 3 | 3 | 4 | 4 |

The schedule is stored in `TURN_ACTIONS`; after turn 8 the base budget stays
at 4. The total budget is:

```text
min(base_actions + stored_bonus_actions, 8)
```

At turn setup, stored bonuses are converted into extra available actions and
removed from the persistent bonus bank. The player then allocates the current
budget to attack, defend and bonus actions. The engine does not require the AI
to spend every available action, although all standard agents normally do.

Constants:

- `MAX_BASE_ACTIONS = 4`;
- `MAX_BONUS_ACTIONS = 4`;
- `MAX_TOTAL_ACTIONS = 8`.

### Attack

An `attack` action is one hit on the opponent's active character. The number
of attacks is counted before resolution. Attack damage is calculated using the
active characters at the start of resolution, before damage changes the state.

### Proactive shields

`defend` actions do not block attacks on the same turn. They become the
player's shield count for the opponent's next turn:

```text
player.shields = defend_actions
```

When the opponent attacks:

```text
blocked = min(opponent_attack_count, defender.shields)
unblocked = max(0, opponent_attack_count - defender.shields)
```

Each shield blocks one whole attack, not one damage point. The shield
allocation is a one-turn state: after the opponent's immediately following
turn, the allocation is consumed/reset in full, including shields that did not
block an attack. Extra shields do not carry into another turn.

The resolved shield count is public after every turn. A UI/battle log must
report the shield count used for that resolved exchange even when the attacker
made zero attacks, including `0`. This is not a prediction of a future action
and must not be displayed before the turn resolves. For example:

```text
opponent did not attack
You spent shields: 5

opponent attacked: 4 attacks vs 5 shields: 4 blocked, 0 hit
You spent shields: 5
```

`spent shields: 5` is the public shield allocation for that exchange; it is
not the number of shields that actually blocked attacks. `blocked=4` is the
separate resolved result. No-attack turns must still report the allocation.

This timing is critical:

- existing opponent shields affect attacks made now;
- shields chosen now affect the opponent's next turn;
- the resolved shield allocation is shown only after that turn, never as a
  forecast before the opponent acts;
- a planner must not treat current `defend` actions as immediate protection.

### Bonus actions

Each `bonus` action adds one stored bonus action after the turn. The persistent
bank is capped at 4. Bonus actions are public and observable. On the next own
turn they are automatically spent to expand the action budget, up to the total
cap of 8.

Bonus is therefore a tradeoff:

- it gives no immediate damage or defense;
- it can create a large future turn;
- it reveals preparation and may invite an opponent attack;
- banking too long can be punished by a lethal burst.

An AI must not create more bonus actions than the remaining bank capacity.

### Voluntary switching

A character switch is a side effect performed inside `choose_actions()` by
calling `player.switch_character(new_index)`. It is not represented by a
`BattleAction`.

Rules:

- costs exactly one action (`ACTION_COST_SWITCH=1`);
- target must be alive and different from the active character;
- only one voluntary switch is allowed per round;
- switching is forbidden if fewer than one action remains;
- after switching, the remaining actions are allocated to attack/defend/bonus;
- the active character before the switch is remembered in `switch_history`.

The old behavior that permanently prevented returning to a previously used
character is intentionally removed. A character can be selected again on a
later turn if alive and legal. Only a second voluntary switch in the same
round is forbidden.

### Death and forced promotion

If the active character reaches zero HP:

1. It is marked inactive and its alive count decreases.
2. The next living character in `stack_order` is promoted automatically.
3. The promoted character becomes active without spending an action.
4. The dead character remains second in the visual stack.
5. `forced_switch_after_death=True` is set for the remainder of the round.

This is not a free player action and must not be counted as a voluntary
switch. If no living character remains, the player loses immediately.

At the next round boundary, `forced_switch_after_death` is reset. A profile
may use this flag to play aggressively after forced promotion.

### Stack order and identity

`Player.stack_order` is a visual/order stack, separate from logical character
indices. The active character is always first in the stack.

Example: logical characters `[0, 1, 2]`, active `0`.

- voluntary switch to `2` -> stack `[2, 0, 1]`;
- active `2` dies -> next living character is promoted, dead `2` becomes
  second, for example `[1, 2, 0]`;
- logical indices and character numbers remain `0, 1, 2`.

Any UI or battle log that displays a team must render `stack_order`, while
still showing the original logical number. Do not mutate `characters` merely
to change visual order.

### Turn order and lifecycle

Player 1 starts at global turn 1. Players alternate turns. A full engine run:

1. sets up the current player's budget;
2. calls that AI's `choose_actions(player, opponent, turn_num, turn_logs,
   player_id)`;
3. executes switch side effects and action counts;
4. resolves attacks against existing shields;
5. promotes a replacement if the active character died;
6. adds new bonuses and sets new shields;
7. records the complete turn log;
8. checks loss and advances to the other player.

`turn_logs` is deliberately truncated to the latest 8 turns for AI memory.
`full_turn_logs` retains the complete battle and is used for statistics and
human-readable battle records. `result["turns"]` is the length of the full
battle, not the 8-turn memory window.

## AI Contracts

An AI must expose:

```python
choose_actions(player, opponent, turn_num, turn_logs, player_id) -> list[BattleAction]
```

Stateful AIs should expose `reset_state()` and it must be called before every
independent game. Otherwise observations from one match leak into the next
benchmark or fitness game.

Returned actions use:

- `BattleAction("attack", active_index, opponent_active_index)`;
- `BattleAction("defend", active_index)`;
- `BattleAction("bonus", active_index)`.

The engine counts action types and performs legality through available budget
and bonus caps. New AI code should still generate legal allocations itself:

- do not exceed `player.remaining_actions` after a switch;
- do not exceed `MAX_BONUS_ACTIONS - player.bonus_actions` with bonus actions;
- do not voluntarily switch twice in a round;
- use the current active index after switching.

## Current Smart Agent

The primary line is `SmartNeuralAgent`, not the older LSTM line. It has a
12-value genome and algorithmic tactical logic:

```text
0  adaptation speed
1  aggression: attack versus defense
2  bonus tendency
3  defensive switch HP threshold
4  aggressive type-advantage switching
5  press when ahead in living characters
6  focus low-HP lethal targets
7  early/burst bonus saving
8  reaction strength to opponent model
9  bluff/action stochasticity
10 shield response to opponent bonus hoarding
11 type matchup sensitivity
```

Genome values are mapped through a sigmoid. The genome supplies preferences
and reaction strengths; pattern recognition and tactical search are explicit
algorithms rather than learned neural weights.

### OpponentModel

The model observes opponent `TurnLog` entries and must process each turn only
once. It maintains:

- EMA probabilities of attack, defend and bonus actions;
- estimated opponent bonus bank;
- estimated current shields;
- last four opponent action vectors `(attack, defend, bonus, switched)`;
- consecutive bonus count;
- attack-wave flag;
- strategy-change flag;
- burst risk.

The model uses public information. Bonus actions and shields are observable
through logs/state. Burst risk rises when the opponent has accumulated bonus or
has banked bonus repeatedly. Do not update the same log repeatedly merely
because the engine passes the rolling log window on every decision.

### Smart decision pipeline

The current pipeline is:

1. Update `OpponentModel` with unseen opponent turns.
2. Compute attack/defend/bonus preference weights.
3. Classify the tactical mode: `NORMAL`, `FINISH`, `ANTI_BURST`,
   `INVEST` or `ENDGAME_RACE`.
4. Generate legal `MacroAction` allocations, including voluntary switch
   targets with the one-action switch cost already paid.
5. Search exact copied states over three plies: our allocation, a strong
   opponent response, and our best follow-up. Keep robust worst-case and
   expected values rather than relying on one average opponent action.
6. Use the genome only as a small tie-break prior among tactical candidates.
7. Apply shield-deadlock breakthrough and budget/bonus legality guards.
8. Return shuffled `BattleAction` objects for the selected macro plan.

The robust planner explicitly models two phases:

- current attacks versus shields already placed by the opponent;
- current shields versus a distribution of possible attacks on the opponent's
  next turn, using the real next-turn budget plus the estimated bonus bank.

Planner scenario lines include maximum attack, shield wall, bonus burst,
bonus-plus-shield preparation and an observed-policy allocation. The selected
plan and top rejected alternatives are exposed through
`SmartNeuralAgent.last_decision_diagnostics`, including mode, candidate count,
search depth and robust/worst-case scores. Interactive champion battle records
store these diagnostics for champion turns when available.

The planner is a robust three-ply beam-style search rather than full MCTS.
MCTS may be added later for close bonus-investment decisions, but it must use
macro-actions and only run when exact candidates are close; it must not replace
the hard tactical facts or introduce random rollouts into the normal policy.

Performance notes:

- `src/cote_megaverse/planner_kernel.py` provides an optional Numba JIT kernel
  for cheap macro-candidate pre-ranking;
- without Numba, the deterministic NumPy/Python fallback is used and behavior
  remains identical at the tactical-policy level;
- the exact copied-state search is restricted to the top-ranked candidates;
- planner state copies use a dedicated shallow battle-state clone instead of
  recursive `copy.deepcopy`, preserving characters, stack order and switch
  history while avoiding recursive object traversal;
- the exact resolver is still the Python reference path. A compiled resolver
  is enabled only after differential tests prove identical HP, shields,
  bonuses, active indices and forced-promotion results;
- `planner_kernel.resolve_numeric()` has passed a 2,000-state differential
  corpus against its Python reference resolver. The compiled transition is
  used by the planner when Numba is available; the production battle engine
  remains unchanged and is still the final mechanics reference;
- production evolution sets `classify_snapshots=False` because per-generation
  population classification was slower than the training evaluation itself;
- Numba can be installed with `pip install -e ".[speed]"` when the active
  Python version has a compatible wheel. It is an optimization only, not a
  required dependency.
- On Windows, the parent process warms the Numba specialization before
  creating workers. Keep `n_jobs=4` by default: many independent workers can
  compile their own LLVM kernel and consume about 1 GB of memory each during
  startup.

Because the planner searches copied states over multiple plies, its game cost
is much higher than the former one-step Smart policy. The first production
profile is deliberately smaller than the historical profile:

```text
pop_size=64
generations=80
elite_frac=0.10
mut_rate=0.10
mut_sigma=0.10
n_jobs=4
hof_add=8
hof_ratio=0.20
hof_max=96
reference_games=4
validation_games=12
champion_games=8
final_evaluation_games=30
snapshot_interval=20
```

For development iterations use the fast profile with the same exact planner
and mechanics:

```text
COTE_EVOLUTION_PROFILE=fast
pop_size=16
generations=12
reference_games=1
validation_games=2
classify_snapshots=False
```

On PowerShell:

```powershell
$env:COTE_EVOLUTION_PROFILE="fast"
python -m src.cote_megaverse.coevolution
```

Fast-profile artifacts use the `fast_` prefix and never overwrite the
production `best_genome.npy`. Treat fast runs as directional experiments:
compare tactical metrics and fixed seeds, then use the production profile for
final statistical confirmation.

The old 280x140 profile must not be used with the robust planner without a
new timing and quality study. The current estimate is about 225,280 training
games for the first profile, versus about 5.64 million for the historical
profile. The estimate is hardware-dependent; the code uses
`estimated_game_seconds` for a visible runtime estimate and should be updated
after a representative smoke run.

Do not start production evolution until the following gates pass on the same
workspace:

1. `python -m unittest discover -q`.
2. A fixed-seed validation smoke run with at least 4 games per reference
   matchup and no catastrophic regression against the saved champion.
3. Tactical task smoke validation with at least 2 games per matchup; inspect
   `burst_survival`, `shield_breakthrough`, `switch_quality` and
   `lethal_conversion`.
4. A one-generation parallel smoke run with the intended worker count.

Production selection remains based on the evolution fitness, but the saved
champion must be accepted using fixed validation and task metrics, not the
noisy generation fitness alone.

The utility values damage, survival, lethal kills, bonus preparation and
reduced incoming damage. Genome priors only break close tactical ties; they
must not remove legal counter-play from the search.

Important tactical invariants:

- Do not repeatedly defend into a shield-heavy opponent with no plan to break
  through.
- If the opponent has a known shield wall and low attack/burst risk, allocate
  enough attacks to pierce it.
- A likely lethal or forced character death has much more value than ordinary
  damage.
- `FINISH` must prioritize guaranteed lethal over bonus preparation.
- `ANTI_BURST` must evaluate the opponent's strongest legal next budget and
  cannot assume the opponent follows the average EMA action mix.
- `INVEST` is allowed only when the copied-state search shows a safe, valuable
  future budget; bonus is not intrinsically good.
- `ENDGAME_RACE` values immediate lethal and survival over generic banking.
- A switch costs an action and should be evaluated as lost tempo plus changed
  type matchup.
- A forced promotion after death is not equivalent to a voluntary switch.

## Evaluation and Evolution

Smart fitness combines self-play/population/Hall-of-Fame results with
reference opponents. The current production profile is the `64 x 80` profile
documented above. The historical `280 x 140` profile is not active and must
not be used with the robust planner.

Fast development runs use `COTE_EVOLUTION_PROFILE=fast` and the same exact
planner/mechanics. Fast artifacts have a `fast_` prefix and never overwrite
the production champion.

The reference suite includes:

- `AllIn`;
- `Defender`;
- `Aggro`;
- `Switcher`;
- `Gambler`;
- `BonusBanker`;
- `CounterAI`;
- `AdaptiveAI`;
- `PhaseShift`.

The diagnostic validation suite additionally includes `HumanShieldBreaker`,
`HardDefender`, `BurstBanker` and `SwitchPunisher`; these hard variants are
not silently folded into production fitness.

`PhaseShiftAI` deliberately changes from bonus saving to defense to attack
during one match. It tests adaptation to a changing job, not merely a fixed
counter.

Fitness is deliberately anti-specialization:

```text
adjusted = 0.60 * raw_winrate + 0.40 * average(three weakest reference rates)
```

There is also a soft penalty when more than 90% of non-bonus actions are one
action type. This is not a ban on tactical all-in turns; it discourages an
agent whose entire policy is one-dimensional.

Do not judge a champion by average win rate alone. Always inspect the weakest
reference matchups and action distribution.

### Fitness versus validation

The ordinary fitness score is noisy and moving-target dependent. It is used
for selection only; it is not the final proof that a later generation is
stronger than an earlier one.

The Smart line has an independent fixed validation suite:

- `validate_smart_genome(genome, games, seed_offset)` evaluates the genome
  against the fixed reference opponents using deterministic Python and NumPy
  seeds;
- the validation suite is diagnostic only and must not affect breeding,
  fitness, selection or mutation;
- Python and NumPy RNG states are restored after validation, so validation must
  not perturb the evolution random stream;
- `compare_smart_genomes(a, b, games, seed_offset)` compares two genomes on
  common deterministic games;
- `validate_smart_tasks(genome, games, seed_offset)` reports separate
  mechanics, shield, burst, switch, human-strategy and anti-counter suites;
- `pairwise_smart_genome_matrix(genomes, games, seed_offset)` stores directed
  common-seed wins, losses, draws, side counts and 95% confidence intervals;
- Validation reports include `draw_rate`, `deadline_rate`, average turns and
  games at the practical deadline. `controlled_fitness_scores(...)` reports
  baseline, draw-penalty, deadline-penalty and combined-penalty variants; these
  variants are diagnostic only until a controlled full-suite comparison proves
  that they do not regress other anchors.
- at every Smart snapshot, `gen_stats[*]["fixed_validation"]` stores current
  matchup rates, task metrics and the current champion's winrate against
  previous snapshot champions;
- `gen_stats[*]["champion_pairwise_matrix"]` stores the full snapshot champion
  matrix for RPS/non-transitive matchup analysis;
- The saved champion benchmark is recorded in
  `artifacts/benchmark_saved_champion_20260719.json` (generated and ignored).
  Its primary metric is `wins / all_games`; draws are counted as non-wins,
  because failing to finish within the practical turn horizon is a loss against
  a human who can accumulate a final burst. The recorded benchmark uses 300
  games per opponent and a 50-turn horizon against the full production anchor
  suite (`AllIn`, `Defender`, `Aggro`, `Switcher`, `Gambler`, `BonusBanker`,
  `CounterAI`, `AdaptiveAI`, `PhaseShift`) plus hard diagnostics
  (`HumanShieldBreaker`, `HardDefender`, `BurstBanker`, `SwitchPunisher`).
  New evolution and benchmark games use `PRACTICAL_MAX_TURNS=70`; historical
  50-turn results remain preserved only as comparison baselines.
- the same validation seeds must be reused across generations when comparing
  champions. Do not compare scores from unrelated random suites as if they
  were progress.

Use fixed validation to detect three different cases:

1. Fitness rises and fixed validation rises: likely genuine improvement.
2. Fitness is flat but fixed validation rises: fitness noise or moving-target
   effects are masking improvement.
3. Champion-versus-champion results cycle while fixed validation is flat:
   likely non-transitive rock-paper-scissors co-evolution.

For practical benchmark reporting, never use decisive-only winrate as the main
score. It excludes draws and hides failure to finish control cycles. Report
`wins / all_games`, `draw_rate` and the 95% confidence interval together.

## Persistence and Artifacts

Training writes genomes, version metadata, evolution results, statistics and
training logs under `artifacts/`; fast-profile files use the `fast_` prefix.

`coevolution_result.json` also stores the snapshot fixed-validation history.
`coevolution_stats.json` stores the same generation statistics together with
the validation records, task metrics, pairwise matrix and action/archetype
summaries. Individual battle files are intentionally not persisted.

JSON loading is defensive. Empty or corrupt `versions.json` is treated as an
empty history. JSON writes use a temporary sibling file, `flush`, `fsync` and
`os.replace` so an interrupted run does not leave a zero-byte primary file.

`play_vs_champion.py` detects genome length:

- 12 parameters -> `SmartNeuralAgent`;
- other supported genome length -> legacy `NeuralAgent`.

Do not delete user-generated genome or log artifacts while fixing code.

Generated model, weight, log and metadata artifacts are ignored by Git. Source
code, tests, handoff documentation and configuration remain trackable. Do not
add generated artifacts to commits unless explicitly asked.

## Non-Negotiable Engineering Rules

1. Read the relevant code and tests before changing mechanics.
2. Preserve the actual timing of shields, bonuses, switches and forced death
   promotion.
3. Keep logical character indices separate from visual stack order.
4. Reset stateful AIs between independent games.
5. Use `full_turn_logs` for in-memory complete statistics when needed; use the
   rolling `turn_logs` only as bounded AI memory. Evolution must persist
   aggregate statistics, not individual battle records.
6. Add regression tests for mechanics changes.
7. Prefer the smallest correct change and existing project patterns.
8. Do not run a full production evolution merely to validate a local code
   change; use focused tests or a small deterministic benchmark.
9. Do not overwrite or revert unrelated user changes.
10. If code and this document disagree, inspect tests and current implementation
    and explicitly report the discrepancy.
11. Fixed validation is diagnostic, deterministic and RNG-preserving; never
    fold it into production fitness without an explicit design decision.
12. Do not assume that a newer generation is stronger because it beats the
    previous population. Check fixed validation and champion cross-matchups.
