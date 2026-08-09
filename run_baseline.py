"""Full baseline vs bundled policies. Writes results to baseline_out.txt."""
import sys, time, json
sys.path.insert(0, "src")
from cote_megaverse.benchmark import benchmark_policies

def fmt(seat):
    return f"W{seat['wins']} L{seat['losses']} D{seat['draws']}"

temp = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
seeds = list(range(int(sys.argv[2]) if len(sys.argv) > 2 else 12))
t0 = time.time()
r = benchmark_policies(seeds=seeds, depth=2, max_half_turns=60, temperature=temp)
dt = time.time() - t0
lines = [f"temperature={temp} seeds={len(seeds)} depth=2 elapsed={dt:.0f}s\n"]
lines.append(f"{'policy':14s} {'ai_first':16s} {'human_first':16s} {'combined':14s} missed_lethal\n")
for pol, v in r.items():
    lines.append(f"{pol:14s} {fmt(v['ai_first']):16s} {fmt(v['human_first']):16s} "
                 f"{fmt({'wins':v['wins'],'losses':v['losses'],'draws':v['draws']}):14s} "
                 f"{v['missed_guaranteed_lethal']}\n")
print("".join(lines), flush=True)
with open("baseline_out.txt", "w", encoding="utf-8") as fh:
    fh.writelines(lines)
