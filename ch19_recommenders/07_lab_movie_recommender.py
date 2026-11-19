# %% [markdown]
# # Section 19.7 Lab: a movie recommender three ways on MovieLens
#
# Regenerated from Book.md (single source of truth). Real bundled MovieLens
# (data/ratings.csv, data/movies.csv); run from this folder. Seeds fixed so the
# printed numbers reproduce.

# %%
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
rng = np.random.default_rng(0); torch.manual_seed(0)

ratings = pd.read_csv("data/ratings.csv")
movies = pd.read_csv("data/movies.csv")
counts = ratings.groupby("movieId").size()                               # ratings per movie, indexed by movieId
top = counts.sort_values(ascending=False, kind="stable").head(300).index # 300 most-rated; stable sort breaks ties by movieId
sl = ratings[ratings.movieId.isin(top)]
cnt = sl.groupby("userId").size()
keep_users = cnt[cnt >= 20].index                                       # users with >=20 of those movies
sl = sl[sl.userId.isin(keep_users)].copy()
uid = {u: k for k, u in enumerate(sorted(sl.userId.unique()))}          # reindex to 0..nu-1
iid = {m: k for k, m in enumerate(sorted(sl.movieId.unique()))}
title = {iid[m]: t for m, t in zip(movies.movieId, movies.title) if m in iid}
sl["u"] = sl.userId.map(uid); sl["i"] = sl.movieId.map(iid)
nu, ni = len(uid), len(iid)

pairs = sl[["u", "i", "rating"]].to_numpy()
pairs = pairs[rng.permutation(len(pairs))]                              # 80/20 split on OBSERVED ratings
split = int(0.8 * len(pairs)); train, test = pairs[:split], pairs[split:]
train_idx = train[:, :2].astype(int); train_y = train[:, 2]
test_idx = test[:, :2].astype(int); test_y = test[:, 2]
Rtr = np.zeros((nu, ni)); trainmask = np.zeros((nu, ni), bool)
for (u, i), y in zip(train_idx, train_y):
    Rtr[u, i] = y; trainmask[u, i] = True
print(f"real MovieLens slice: {nu} users x {ni} movies, "
      f"{len(train_idx)} train ratings, {len(test_idx)} test ratings")

mu = train_y.mean()

def rmse(pred):
    e = np.array([pred(u, i) for u, i in test_idx]) - test_y
    return float(np.sqrt(np.mean(e ** 2)))

def ndcg10(PRED):
    from collections import defaultdict
    byu = defaultdict(list)
    for (u, i), y in zip(test_idx, test_y): byu[u].append((i, y))
    scores = []
    for u, items in byu.items():
        if len(items) < 3: continue
        order = sorted(items, key=lambda it: -PRED(u, it[0]))[:10]
        gains = np.array([y for _, y in order])
        ideal = np.sort([y for _, y in items])[::-1][:10]
        dcg = np.sum(gains / np.log2(np.arange(2, len(gains) + 2)))
        idcg = np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2)))
        scores.append(dcg / idcg)
    return float(np.mean(scores))

# TIER 1 - item-based collaborative filtering
def item_sim_matrix():
    Rc = np.where(trainmask, Rtr - mu, 0.0)                             # mean-center observed entries
    norms = np.linalg.norm(Rc, axis=0) + 1e-9
    G = Rc.T @ Rc
    return G / np.outer(norms, norms)
SIM = item_sim_matrix()
def cf_pred(u, i):
    rated = np.where(trainmask[u])[0]
    if len(rated) == 0: return mu
    sims = SIM[i, rated]
    sel = sims > 0.05                                                  # only positively similar items
    if sel.sum() == 0: return mu
    w = sims[sel]; r = Rtr[u, rated[sel]]
    return float(np.clip((w @ r) / w.sum(), 0.5, 5))
print(f"\nTIER 1  item-based CF          : RMSE {rmse(cf_pred):.3f}   NDCG@10 {ndcg10(cf_pred):.3f}")

# TIER 2 - biased matrix factorization (SGD)
d = 20; bu = np.zeros(nu); bi = np.zeros(ni)
P = rng.normal(0, .1, (nu, d)); Q = rng.normal(0, .1, (ni, d))
lr, reg = .008, .1
order = np.arange(len(train_idx))
for ep in range(60):
    rng.shuffle(order)
    for j in order:
        u, i = train_idx[j]; y = train_y[j]
        e = y - (mu + bu[u] + bi[i] + P[u] @ Q[i])
        bu[u] += lr * (e - reg * bu[u]); bi[i] += lr * (e - reg * bi[i])
        Pu = P[u].copy()
        P[u] += lr * (e * Q[i] - reg * P[u]); Q[i] += lr * (e * Pu - reg * Q[i])
def mf_pred(u, i): return float(np.clip(mu + bu[u] + bi[i] + P[u] @ Q[i], 0.5, 5))
print(f"TIER 2  matrix factorization   : RMSE {rmse(mf_pred):.3f}   NDCG@10 {ndcg10(mf_pred):.3f}")

# TIER 3 - Neural Collaborative Filtering: embeddings -> concat -> MLP
class NeuralCF(nn.Module):
    def __init__(self, nu, ni, d=32):
        super().__init__()
        self.ue = nn.Embedding(nu, d); self.ie = nn.Embedding(ni, d)
        self.bu = nn.Embedding(nu, 1); self.bi = nn.Embedding(ni, 1)
        nn.init.normal_(self.ue.weight, std=0.05); nn.init.normal_(self.ie.weight, std=0.05)
        nn.init.zeros_(self.bu.weight); nn.init.zeros_(self.bi.weight)
        self.mlp = nn.Sequential(nn.Linear(2 * d, 64), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(64, 1))
    def forward(self, u, i):
        x = torch.cat([self.ue(u), self.ie(i)], dim=-1)
        return self.mlp(x).squeeze(-1) + self.bu(u).squeeze(-1) + self.bi(i).squeeze(-1) + MU_T
MU_T = float(mu)
net = NeuralCF(nu, ni)
opt = torch.optim.Adam(net.parameters(), lr=0.005, weight_decay=1e-5)
us = torch.tensor(train_idx[:, 0]); it = torch.tensor(train_idx[:, 1])
yt = torch.tensor(train_y, dtype=torch.float32)
for ep in range(30):
    net.train(); perm = torch.randperm(len(us))
    for b in range(0, len(us), 256):
        bidx = perm[b:b + 256]
        pred = net(us[bidx], it[bidx]); loss = nn.functional.mse_loss(pred, yt[bidx])
        opt.zero_grad(); loss.backward(); opt.step()
net.eval()
with torch.no_grad():                                                  # score all held-out pairs at once
    allu = torch.tensor(test_idx[:, 0]); alli = torch.tensor(test_idx[:, 1])
    ncf_all = torch.clip(net(allu, alli), 0.5, 5).numpy()
ncf_lookup = {(int(u), int(i)): float(p) for (u, i), p in zip(test_idx, ncf_all)}
def ncf_pred(u, i): return ncf_lookup[(int(u), int(i))]
print(f"TIER 3  neural collaborative CF: RMSE {rmse(ncf_pred):.3f}   NDCG@10 {ndcg10(ncf_pred):.3f}")

print(f"\nbaseline (predict global mean {mu:.2f}): RMSE {rmse(lambda u, i: mu):.3f}"
      f"   NDCG@10 {ndcg10(lambda u, i: mu):.3f}")            # a constant leaves each list shuffled
truth = {(int(u), int(i)): y for (u, i), y in zip(test_idx, test_y)}
print(f"NDCG@10 of the worst possible order: {ndcg10(lambda u, i: -truth[(int(u), int(i))]):.3f}")

# a concrete top-10 list for one user, real titles from movies.csv
u0 = 0
seen = set(np.where(trainmask[u0])[0]) | {i for (u, i) in test_idx if u == u0}
cand = [i for i in range(ni) if i not in seen]
ranked = sorted(cand, key=lambda i: -mf_pred(u0, i))[:10]
print(f"\ntop-10 movies matrix factorization recommends to user index {u0} (unseen by them):")
for i in ranked:
    print(f"  {mf_pred(u0, i):.2f}  {title.get(i, '?')}")

print("\nAll movie-recommender lab checks ran.")

