"""Train the belief-conditioned value network on the ReBeL label dataset.

Input (actor-relative, fixed 3v3): each side's stack-order characters
[type one-hot, atk/2100, hp/600, alive] x3, the acting side's bank/sh, turn,
to_move, r_opp, and the belief vector over the opponent's hidden split (padded
to 9). Output: value in [-1,1] from A's perspective (tanh).

Baseline for comparison: the material heuristic the network replaces.
"""
import argparse
import pickle
import sys
import os

import torch
import torch.nn as nn

BASE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(BASE, ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

MAX_CHARS = 3
BELIEF_DIM = 9


def char_slot(team, order, hp, k):
    if k >= len(order):
        return [0.0] * 4 + [0.0, 0.0, 0.0]  # type onehot 0, atk 0, hp 0, alive 0
    idx = order[k]
    typ, atk, _ = team[idx]
    onehot = [0.0] * 4
    onehot[typ] = 1.0
    return onehot + [atk / 2100.0, hp[k] / 600.0, 1.0]


def encode(row):
    r = row
    a = r["orderA"]
    b = r["orderB"]
    actor, opp = (a, b) if r["to_move"] == 0 else (b, a)
    a_team, b_team = r["teamA"], r["teamB"]
    if r["to_move"] == 0:
        actor_team, opp_team, actor_hp, opp_hp = a_team, b_team, r["hpA"], r["hpB"]
        actor_bank, actor_sh = r["bankA"], r["shA"]
    else:
        actor_team, opp_team, actor_hp, opp_hp = b_team, a_team, r["hpB"], r["hpA"]
        actor_bank, actor_sh = r["bankB"], r["shB"]
    feats = []
    for k in range(MAX_CHARS):
        feats.extend(char_slot(actor_team, actor, actor_hp, k))
    for k in range(MAX_CHARS):
        feats.extend(char_slot(opp_team, opp, opp_hp, k))
    feats += [actor_bank / 4.0, actor_sh / 4.0,
              r["turn"] / 9.0, float(r["to_move"]), r["r_opp"] / 4.0]
    bel = r["belief"] + [0.0] * (BELIEF_DIM - len(r["belief"]))
    feats += bel[:BELIEF_DIM]
    # target: the ACTING side's value (the raw label is A's value; flip when B acts)
    actor_value = r["value"] if r["to_move"] == 0 else -r["value"]
    return feats, actor_value


class ValueNet(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1), nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def material(row):
    ha = sum(row["hpA"])
    hb = sum(row["hpB"])
    bodies = (len(row["hpA"]) - len(row["hpB"])) * 0.35
    return min(1.0, max(-1.0, (ha - hb) / 600.0 + bodies))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(REPO, "table_out", "vdata500.pkl"))
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(REPO, "table_out", "vnet1.pt"))
    a = ap.parse_args()

    with open(a.data, "rb") as fh:
        rows = pickle.load(fh)["rows"]
    torch.manual_seed(a.seed)
    X = []
    y = []
    mat = []
    for r in rows:
        fx, fy = encode(r)
        X.append(fx)
        y.append(fy)
        mat.append(material(r))
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    n = len(rows)
    nv = max(1, n // 5)
    perm = torch.randperm(n)
    val_idx, train_idx = perm[:nv], perm[nv:]
    print(f"{n} rows, in_dim={X.shape[1]}, val={nv}")

    def mae(a, b):
        return float(torch.abs(a - b).mean())

    mat_t = torch.tensor(mat, dtype=torch.float32)
    print(f"baseline material: val MAE={mae(mat_t[val_idx], y[val_idx]):.4f} "
          f"train MAE={mae(mat_t[train_idx], y[train_idx]):.4f}")

    net = ValueNet(X.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    lossf = nn.MSELoss()
    for ep in range(a.epochs):
        net.train()
        perm_t = torch.randperm(train_idx.numel())
        for b in range(0, train_idx.numel(), a.batch):
            bidx = train_idx[perm_t[b:b + a.batch]]
            opt.zero_grad()
            loss = lossf(net(X[bidx]), y[bidx])
            loss.backward()
            opt.step()
        if (ep + 1) % 20 == 0:
            net.eval()
            vp = net(X[val_idx])
            print(f"  ep={ep+1:3d} loss={float(loss):.5f} "
                  f"val MAE={mae(vp, y[val_idx]):.4f}")

    net.eval()
    vp = net(X[val_idx])
    print(f"trained: val MAE={mae(vp, y[val_idx]):.4f} (material {mae(mat_t[val_idx], y[val_idx]):.4f})")
    torch.save(net.state_dict(), a.out)
    print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()
