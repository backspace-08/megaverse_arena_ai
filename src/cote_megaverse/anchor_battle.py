"""Battle of the anchors: WeightedRandomAIv2 profiles + CounterAI."""
import sys, os, itertools, numpy as np
from .parameterized_ai_v2 import BattleEngineV2, WeightedRandomAIv2, CounterAI, AdaptiveAI, random_team, AIProfile
from collections import defaultdict

ANCHORS = [
    ("AllIn",    WeightedRandomAIv2(AIProfile("AllIn",    w_attack=18, w_defend=0,  w_bonus=0,
                                              switch_when_disadvantaged=True, w_switch=3))),
    ("Defender", WeightedRandomAIv2(AIProfile("Defender", w_attack=3,  w_defend=12, w_bonus=1,
                                              switch_when_disadvantaged=True, w_switch=3))),
    ("Aggro",    WeightedRandomAIv2(AIProfile("Aggro",    w_attack=12, w_defend=1,  w_bonus=0.5,
                                              switch_when_disadvantaged=True, w_switch=3))),
    ("Switcher", WeightedRandomAIv2(AIProfile("Switcher", w_attack=10, w_defend=1,  w_bonus=2,
                                              w_switch=8, switch_when_disadvantaged=True,
                                               switch_min_hp_ratio=0.8, aggressive_after_forced_switch=True,
                                              save_first_turns=1))),
    ("Gambler",  WeightedRandomAIv2(AIProfile("Gambler",  w_attack=5,  w_defend=5,  w_bonus=5,
                                              w_switch=5, switch_when_disadvantaged=True,
                                              switch_min_hp_ratio=0.5, randomness=1.0))),
    ("Counter",  CounterAI()),
    ("Adaptive", AdaptiveAI()),
]

N = 1000

results = defaultdict(lambda: defaultdict(float))

for (n1, a1), (n2, a2) in itertools.combinations(ANCHORS, 2):
    w1 = 0
    for _ in range(N):
        t1, t2 = random_team(), random_team()
        if np.random.random() < 0.5:
            e = BattleEngineV2(a1, a2, t1, t2)
            r = e.run(50)
            if r["winner"] == 1: w1 += 1
        else:
            e = BattleEngineV2(a2, a1, t1, t2)
            r = e.run(50)
            if r["winner"] == 2: w1 += 1
    w2 = N - w1
    results[n1][n2] = w1 / N
    results[n2][n1] = w2 / N

# Print matrix
print(f"\n{'':>10s}", end="")
for n, _ in ANCHORS:
    print(f"  {n:>8s}", end="")
print(f"  {'WINS':>6s}")
print(f"{'-'*10}  " + "  ".join(f"{'-'*8}" for _ in ANCHORS) + f"  {'-'*6}")

for n1, _ in ANCHORS:
    wins = sum(results[n1][n2] for n2, _ in ANCHORS if n2 != n1)
    total = len(ANCHORS) - 1
    print(f"{n1:>10s}", end="")
    for n2, _ in ANCHORS:
        if n1 == n2:
            print(f"  {'':>8s}", end="")
        else:
            print(f"  {results[n1][n2]:>7.0%} ", end="")
    print(f"  {wins/total:>5.0%}")
