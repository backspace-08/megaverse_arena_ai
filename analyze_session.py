"""Analyze the 20-game session log chronologically."""
import json

sess = json.load(open("session_log.json", encoding="utf-8"))
print("game | seat | winner | plies")
for e in sess:
    print(f"{e['game']:4d} | {e['seat']:5s} | {e['winner']:5s} | {e['plies']:3d}")

n = len(sess)
w = sum(1 for e in sess if e["winner"] == "YOU")
print()
print(f"total={n}  human wins={w}  winrate={100*w/n:.1f}%")

for name, blk in (("first 10", sess[:10]), ("last 10", sess[10:])):
    if not blk:
        continue
    w = sum(1 for e in blk if e["winner"] == "YOU")
    print(f"{name}: {w}/{len(blk)} = {100*w/len(blk):.0f}%")

# seat-adjusted: first mover wins more; check per-seat
for seat in ("ai_first", "human_first"):
    blk = [e for e in sess if e["seat"] == seat]
    w = sum(1 for e in blk if e["winner"] == "YOU")
    print(f"{seat}: {w}/{len(blk)} = {100*w/len(blk):.0f}%")
