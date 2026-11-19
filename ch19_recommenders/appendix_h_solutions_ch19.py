"""Appendix H solutions for Chapter 19, as runnable cells.

Generated from Solutions.md (Appendix H). Every code block of this chapter's solutions appears here
verbatim, split into one cell per exercise or drill topic, so each answer can be run and checked against
the expected outputs it states. Keep Solutions.md as the source; regenerate after any change.
"""

# %% [markdown]
# ## 19.8 coding exercises

# %%
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

# the shipped MovieLens ml-latest-small snapshot (GroupLens, generated 26 September 2018), read once
# and reused across the exercises; run from code/ch19_recommenders, where data/ holds the two CSVs
ratings = pd.read_csv("data/ratings.csv")
movies  = pd.read_csv("data/movies.csv")

# %% [markdown]
# 1. Load MovieLens, build the dense slice, see the long tail

# %%
# 1. Load MovieLens, build the dense slice, see the long tail
counts = ratings.movieId.value_counts()
half = (counts.cumsum() <= ratings.shape[0] / 2).sum() + 1
print(f"ratings={len(ratings)}, users={ratings.userId.nunique()}, movies={ratings.movieId.nunique()}")
# expected output: ratings=100836, users=610, movies=9724
print(f"movies holding half of all ratings: {half} of {ratings.movieId.nunique()}")
# expected output: movies holding half of all ratings: 659 of 9724
top = counts.head(150).index
sub = ratings[ratings.movieId.isin(top)]
c = sub.userId.value_counts(); sub = sub[sub.userId.isin(c[c >= 25].index)]
uids = sorted(sub.userId.unique()); iids = sorted(sub.movieId.unique())
uidx = {u: k for k, u in enumerate(uids)}; iidx = {m: k for k, m in enumerate(iids)}
nu, ni = len(uids), len(iids)
print(f"dense slice: {nu} users x {ni} movies, {len(sub)} ratings, "
      f"{len(sub)/(nu*ni)*100:.0f}% dense")
# expected output: dense slice: 326 users x 150 movies, 17896 ratings, 37% dense

# one seeded 80/20 split reused by later exercises
u = sub.userId.map(uidx).to_numpy(); i = sub.movieId.map(iidx).to_numpy(); r = sub.rating.to_numpy()
rs = np.random.RandomState(42)
perm = rs.permutation(len(sub)); ncut = int(0.2 * len(sub))
is_test = np.zeros(len(sub), bool); is_test[perm[:ncut]] = True; tr = ~is_test
tu, ti, trr = u[tr], i[tr], r[tr]
eu, ei, err = u[is_test], i[is_test], r[is_test]
gmean = trr.mean()
Rtr = np.zeros((nu, ni)); Mtr = np.zeros((nu, ni), bool)
for k in np.where(tr)[0]: Rtr[u[k], i[k]] = r[k]; Mtr[u[k], i[k]] = True

# %% [markdown]
# 2. User-based Pearson kNN collaborative filtering vs baseline

# %%
# 2. User-based Pearson kNN collaborative filtering vs baseline
umean_s = np.array([Rtr[x][Mtr[x]].mean() if Mtr[x].any() else gmean for x in range(nu)])
Cc = np.where(Mtr, Rtr - umean_s[:, None], 0.0)
S = np.zeros((nu, nu))
for x in range(nu):
    co = Mtr & Mtr[x]
    num_ = (Cc * Cc[x]).sum(1)
    da = np.sqrt((Cc[x] ** 2 * co).sum(1)); dv = np.sqrt((Cc ** 2 * co).sum(1)); dd = da * dv
    s = np.where(dd > 0, num_ / (dd + 1e-12), 0.0); s[co.sum(1) < 2] = 0.0; s[x] = 0.0; S[x] = s
k_nbrs = 40
def predict_knn(x, j):
    sims = np.where(Mtr[:, j] & (S[x] > 0), S[x], 0.0)
    if not sims.any(): return umean_s[x]
    nb = np.argsort(sims)[::-1][:k_nbrs]; sv = sims[nb]
    if sv.sum() == 0: return umean_s[x]
    return umean_s[x] + (sv @ (Rtr[nb, j] - umean_s[nb])) / sv.sum()
mae_base = np.mean(np.abs(gmean - err))
mae_cf = np.mean([abs(predict_knn(int(x), int(j)) - rt) for x, j, rt in zip(eu, ei, err)])
print(f"MAE global-mean baseline : {mae_base:.3f}")
print(f"MAE user-based kNN (k={k_nbrs}) : {mae_cf:.3f}")
# expected output: MAE global-mean baseline : 0.738
# expected output: MAE user-based kNN (k=40) : 0.602

# %% [markdown]
# 3. Adjusted-cosine item similarity: sign flip + nearest neighbors

# %%
# 3. Adjusted-cosine item similarity: sign flip + nearest neighbors
def adjcos_matrix(R_, M_):
    # adjusted cosine as 19.2.1 defines it: center each user on the mean of their OBSERVED ratings,
    # then sum every item pair over the users who rated BOTH items (co-raters), never over all raters
    um = np.array([R_[x][M_[x]].mean() if M_[x].any() else 0.0 for x in range(R_.shape[0])])
    A_ = np.where(M_, R_ - um[:, None], 0.0)                # centered deviations, 0 where unrated
    Mf = M_.astype(float)
    num_ = A_.T @ A_                                        # products survive only where both rated
    sq = (A_ ** 2).T @ Mf                                   # sq[a, b]: item a's squared deviations over b's raters
    S_ = num_ / (np.sqrt(sq * sq.T) + 1e-12)
    S_[(Mf.T @ Mf) < 2] = 0.0                               # fewer than two co-raters: no evidence
    return S_
Rt = np.array([[5,4,5,1,1,2],[4,5,4,2,1,1],[5,5,4,1,2,1],
               [1,2,1,5,4,5],[2,1,1,4,5,4]], float)
Rt_train = Rt.copy(); Rt_train[0, 5] = 0.0                  # hold out user 0's movie 5, as Example 19-2 does
S_toy = adjcos_matrix(Rt_train, Rt_train != 0)              # means and sums from the training cells only
plain = Rt_train[:, 0] @ Rt_train[:, 3] / (np.linalg.norm(Rt_train[:, 0]) * np.linalg.norm(Rt_train[:, 3]))
print(f"action vs romance: plain cosine {plain:+.2f}, adjusted {S_toy[0, 3]:+.2f}")
print(f"action vs action : adjusted {S_toy[0, 1]:+.2f}")
# expected output: action vs romance: plain cosine +0.54, adjusted -0.99
# expected output: action vs action : adjusted +0.85
Sitem = adjcos_matrix(Rtr, Mtr)                             # the real slice, training ratings only
titles = movies.set_index("movieId").title.to_dict()
mid_by_idx = {v: k for k, v in iidx.items()}
print(f"nearest neighbors of '{titles[mid_by_idx[0]]}':")
for j in [q for q in np.argsort(-Sitem[0], kind="stable") if q != 0][:3]:
    print(f"  {Sitem[0, j]:+.2f}  {titles[mid_by_idx[j]]}")
# expected output: nearest neighbors of 'Toy Story (1995)':
# expected output:   +0.68  Incredibles, The (2004)
# expected output:   +0.62  Toy Story 2 (1999)
# expected output:   +0.43  Aladdin (1992)

# %% [markdown]
# 4. Biased matrix factorization by ALS vs baseline vs bias-only

# %%
# 4. Biased matrix factorization by ALS vs baseline vs bias-only
lam = 10.0; bi = np.zeros(ni); bu = np.zeros(nu)
for y in range(ni):
    m = ti == y; bi[y] = (trr[m] - gmean).sum() / (lam + m.sum()) if m.any() else 0.0
for x in range(nu):
    m = tu == x; bu[x] = (trr[m] - gmean - bi[ti[m]]).sum() / (lam + m.sum()) if m.any() else 0.0
mae_bias = np.mean(np.abs(gmean + bu[eu] + bi[ei] - err))
Res = np.zeros((nu, ni))
for k in range(len(tu)): Res[tu[k], ti[k]] = trr[k] - gmean - bu[tu[k]] - bi[ti[k]]
d, reg, sweeps = 20, 10.0, 15; g = np.random.default_rng(0)
P = g.normal(0, 0.1, (nu, d)); Q = g.normal(0, 0.1, (ni, d)); I_d = np.eye(d)
for _ in range(sweeps):                                     # alternate two least-squares half-steps
    for x in range(nu):
        j = Mtr[x]
        if j.any(): Qj = Q[j]; P[x] = np.linalg.solve(Qj.T @ Qj + reg * I_d, Qj.T @ Res[x, j])
    for y in range(ni):
        xx = Mtr[:, y]
        if xx.any(): Px = P[xx]; Q[y] = np.linalg.solve(Px.T @ Px + reg * I_d, Px.T @ Res[xx, y])
pred = gmean + bu[eu] + bi[ei] + np.einsum('ij,ij->i', P[eu], Q[ei])
print(f"MAE baseline   : {mae_base:.3f}")
print(f"MAE bias-only  : {mae_bias:.3f}")
print(f"MAE ALS+factors: {np.mean(np.abs(pred - err)):.3f}")
# expected output: MAE baseline   : 0.738
# expected output: MAE bias-only  : 0.615
# expected output: MAE ALS+factors: 0.592

# %% [markdown]
# 5. Implicit feedback: confidence weighting + weighted ALS (Hu, Koren, and Volinsky, 2008)

# %%
# 5. Implicit feedback: confidence weighting + weighted ALS (Hu, Koren, and Volinsky, 2008)
plays = np.array([0, 3, 0, 12]); alpha = 40
pref = (plays > 0).astype(int); conf = 1 + alpha * plays
print(f"plays {plays.tolist()} -> preference {pref.tolist()}, confidence {conf.tolist()}")
# expected output: plays [0, 3, 0, 12] -> preference [0, 1, 0, 1], confidence [1, 121, 1, 481]
Pmat = Mtr.astype(float)                                    # rated-at-all is a positive
Cmat = 1.0 + 40.0 * Rtr                                     # confidence grows with the rating
rng5 = np.random.default_rng(1); held = []
Ptrain = Pmat.copy(); Ctrain = Cmat.copy()
for x in range(nu):
    rated = np.where(Mtr[x])[0]; h = rated[rng5.integers(len(rated))]
    Ptrain[x, h] = 0.0; Ctrain[x, h] = 1.0; held.append((x, h))
Ki, regi, sw = 16, 0.1, 8
X = rng5.normal(0, 0.1, (nu, Ki)); Y = rng5.normal(0, 0.1, (ni, Ki)); Ii = np.eye(Ki)
def als_step(Wp, Wc, F):
    FtF = F.T @ F; out = np.zeros((Wp.shape[0], Ki))
    for a in range(Wp.shape[0]):
        cu = Wc[a]; nz = cu != 1.0                          # only items with elevated confidence
        A_ = FtF + (F[nz].T * (cu[nz] - 1.0)) @ F[nz] + regi * Ii
        out[a] = np.linalg.solve(A_, F[nz].T @ (cu[nz] * Wp[a, nz]))
    return out
for _ in range(sw):
    X = als_step(Ptrain, Ctrain, Y); Y = als_step(Ptrain.T, Ctrain.T, X)
Mall = np.zeros((nu, ni), bool); Mall[u, i] = True          # every movie the user rated, train or test
wins = tot = 0
for x, h in held:
    sp = X[x] @ Y[h]
    for _ in range(100):
        j = rng5.integers(ni)
        while Mall[x, j]: j = rng5.integers(ni)             # negatives: movies the user never rated at all
        wins += sp > X[x] @ Y[j]; tot += 1
print(f"implicit weighted-ALS held-out ranking AUC = {wins/tot:.2f}")
# expected output: implicit weighted-ALS held-out ranking AUC = 0.77

# %% [markdown]
# 6. Content-based idf-weighted genre profiles + cold start

# %%
# 6. Content-based idf-weighted genre profiles + cold start
mrow = movies.set_index("movieId").loc[iids]
glists = [gg.split("|") for gg in mrow.genres]
allg = sorted({gg for gl in glists for gg in gl})
gix = {gg: k for k, gg in enumerate(allg)}; ng = len(allg)
G = np.zeros((ni, ng))
for a, gl in enumerate(glists):
    for gg in gl: G[a, gix[gg]] = 1.0
idf = np.log(ni / (1 + G.sum(0))); Gw = G * idf            # rare genres carry more signal
def profile(x):
    liked = np.where(Mtr[x] & (Rtr[x] >= 4))[0]
    if len(liked) == 0: return np.zeros(ng)
    p = Gw[liked].mean(0); n = np.linalg.norm(p); return p / n if n else p
rom = gix["Romance"]; new = np.zeros(ng); new[rom] = idf[rom]; new /= np.linalg.norm(new)
def rfrac(x):
    liked = np.where(Mtr[x] & (Rtr[x] >= 4))[0]
    return G[liked, rom].mean() if len(liked) else -1.0
rf = np.array([rfrac(x) for x in range(nu)])
fans = [x for x in range(nu) if rf[x] >= 0.35]
others = [x for x in range(nu) if 0.0 <= rf[x] <= 0.10]
print(f"idf(Drama)={idf[gix['Drama']]:.2f}  idf(Film-Noir)={idf[gix['Film-Noir']]:.2f}")
print(f"new pure-Romance release, content score for {len(fans)} romance fans : "
      f"{np.mean([profile(x) @ new for x in fans]):.2f}")
print(f"                                       for {len(others)} others       : "
      f"{np.mean([profile(x) @ new for x in others]):.2f}")
# expected output: idf(Drama)=0.92  idf(Film-Noir)=4.32
# expected output: new pure-Romance release, content score for 17 romance fans : 0.49
# expected output:                                        for 72 others       : 0.07

# %% [markdown]
# 7. Ranking metrics from scratch: a new list, MAP over three users, graded gains

# %%
# 7. Ranking metrics from scratch: a new list, MAP over three users, graded gains
rel = np.array([0, 1, 1, 0, 0, 1, 0, 0, 0, 1])           # relevance of the items at ranks 1 to 10
def p_at(x, k): return x[:k].sum() / k
def r_at(x, k): return x[:k].sum() / x.sum()
def dcg(g, k): return float(np.sum(g[:k] / np.log2(np.arange(2, k + 2))))
def ndcg(g, k): return dcg(g, k) / dcg(np.sort(g)[::-1], k)   # ideal order = gains sorted downward
def ap(x):
    hits = 0; s = 0.0
    for idx, v in enumerate(x, 1):
        if v: hits += 1; s += hits / idx                     # precision at each relevant rank
    return s / x.sum()
def rr(x):
    w = np.where(x == 1)[0]; return 1.0 / (w[0] + 1) if len(w) else 0.0
print(f"P@5 {p_at(rel,5):.2f}  R@5 {r_at(rel,5):.2f}  NDCG@5 {ndcg(rel,5):.3f}  AP {ap(rel):.3f}  RR {rr(rel):.2f}")
# by hand: P@5 = 2/5, R@5 = 2/4, DCG@5 = 1/log2(3) + 1/log2(4) = 1.131, IDCG@5 = 1 + 1/log2(3) + 1/log2(4) + 1/log2(5) = 2.562,
# NDCG@5 = 1.131/2.562 = 0.441, AP = (1/2 + 2/3 + 3/6 + 4/10)/4 = 0.517, RR = 1/2
lists = [rel, np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0]), np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])]
print(f"MAP@10 over 3 users {np.mean([ap(x) for x in lists]):.3f}   MRR {np.mean([rr(x) for x in lists]):.3f}")
stars = np.array([5., 3., 4., 2., 5.])                   # five held-out movies in the model's order, stars as gains
swapped = stars[[4, 1, 2, 3, 0]]                         # the two 5-star movies trade places
low_first = stars[[3, 0, 1, 2, 4]]                       # the 2-star movie moved to the top
print(f"NDCG@5, star gains: model order {ndcg(stars,5):.3f}  5-star swap {ndcg(swapped,5):.3f}  2-star first {ndcg(low_first,5):.3f}")
# expected output: P@5 0.40  R@5 0.50  NDCG@5 0.441  AP 0.517  RR 0.50
# expected output: MAP@10 over 3 users 0.489   MRR 0.583
# expected output: NDCG@5, star gains: model order 0.956  5-star swap 0.956  2-star first 0.844

# %% [markdown]
# 8. Pairwise BPR vs pointwise ranking, held-out leave-one-out AUC

# %%
# 8. Pairwise BPR vs pointwise ranking, held-out leave-one-out AUC
pos_df = sub[sub.rating >= 4.0]
cc = pos_df.userId.value_counts(); keep = cc[cc >= 10].index
pos_df = pos_df[pos_df.userId.isin(keep)]
pu = sorted(pos_df.userId.unique()); puix = {uu: k for k, uu in enumerate(pu)}; npu = len(pu)
pos_of = {x: set() for x in range(npu)}
for uu, mm in zip(pos_df.userId, pos_df.movieId): pos_of[puix[uu]].add(iidx[mm])
rng8 = np.random.default_rng(0)
train_pos = {x: set(s) for x, s in pos_of.items()}; heldout = []
for x in range(npu):
    s = list(train_pos[x]); h = s[rng8.integers(len(s))]
    train_pos[x].discard(h); heldout.append((x, h))
train_pairs = [(x, ii) for x in range(npu) for ii in train_pos[x]]
def auc(score):
    ra = np.random.default_rng(7); w = t = 0
    for x, ii in heldout:
        sp = score(x, ii)
        for _ in range(100):
            j = ra.integers(ni)
            while j in pos_of[x]: j = ra.integers(ni)
            w += sp > score(x, j); t += 1
    return w / t
def train_factors(mode, d=32, epochs=30, lr=0.05, reg=0.02):
    rg = np.random.default_rng(3)
    Pp = rg.normal(0, 0.1, (npu, d)); Qq = rg.normal(0, 0.1, (ni, d)); tp = list(train_pairs)
    for _ in range(epochs):
        rg.shuffle(tp)
        for x, ii in tp:
            j = rg.integers(ni)
            while j in train_pos[x]: j = rg.integers(ni)       # training negative: outside the training positives only
            if mode == "pointwise":                            # regress score toward 1 / 0
                for it, lab in [(ii, 1.0), (j, 0.0)]:
                    e = lab - Pp[x] @ Qq[it]; Px = Pp[x].copy()
                    Pp[x] += lr * (e * Qq[it] - reg * Pp[x]); Qq[it] += lr * (e * Px - reg * Qq[it])
            else:                                              # pairwise BPR: widen pos-neg gap
                xg = Pp[x] @ Qq[ii] - Pp[x] @ Qq[j]; gg = 1 / (1 + np.exp(xg)); Px = Pp[x].copy()
                Pp[x] += lr * (gg * (Qq[ii] - Qq[j]) - reg * Pp[x])
                Qq[ii] += lr * (gg * Px - reg * Qq[ii]); Qq[j] += lr * (-gg * Px - reg * Qq[j])
    return Pp, Qq
for mode in ["pointwise", "pairwise"]:
    Pp, Qq = train_factors(mode)
    print(f"{mode:9s} held-out ranking AUC = {auc(lambda x, ii: Pp[x] @ Qq[ii]):.3f}")
# expected output: pointwise held-out ranking AUC = 0.824
# expected output: pairwise  held-out ranking AUC = 0.835

# %% [markdown]
# 9. Two-tower retriever with in-batch negatives (PyTorch), recall@10

# %%
# 9. Two-tower retriever with in-batch negatives (PyTorch), recall@10
import torch, torch.nn as nn
torch.manual_seed(0)
class Tower(nn.Module):
    def __init__(self, n, dd=32):
        super().__init__(); self.emb = nn.Embedding(n, dd); nn.init.normal_(self.emb.weight, std=0.05)
    def forward(self, ids): return nn.functional.normalize(self.emb(ids), dim=-1)
ut, itw = Tower(npu), Tower(ni)
opt = torch.optim.Adam(list(ut.parameters()) + list(itw.parameters()), lr=0.01)
tpa = np.array(train_pairs)
for ep in range(120):
    b = tpa[rng8.choice(len(tpa), 512)]
    ue = ut(torch.tensor(b[:, 0])); ie = itw(torch.tensor(b[:, 1]))
    logits = ue @ ie.T * 10                                   # every user vs every item in the batch
    loss = nn.functional.cross_entropy(logits, torch.arange(len(b)))  # matching item = positive
    opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    UE = ut(torch.arange(npu)); IE = itw(torch.arange(ni)); hit = 0
    for x, ii in heldout:
        order = np.argsort((UE[x] @ IE.T).numpy())[::-1]
        order = [q for q in order if q not in train_pos[x]]
        hit += ii in order[:10]
chance = np.mean([10 / (ni - len(train_pos[x])) for x, ii in heldout])   # 10 over the movies each user did not train on
print(f"two-tower recall@10 = {hit/len(heldout):.2f}  (chance {chance:.3f})")
# expected output: two-tower recall@10 = 0.26  (chance 0.090)

# %% [markdown]
# 10. Miniproject: temporal split vs random split (leakage)

# %%
# 10. Miniproject: temporal split vs random split (leakage)
def als_mf(train_mask, tu_, ti_, trr_, eu_, ei_, err_):       # the fit reads only the training triples
    gm = trr_.mean(); bi_ = np.zeros(ni); bu_ = np.zeros(nu)
    for y in range(ni):
        m = ti_ == y; bi_[y] = (trr_[m] - gm).sum() / (10 + m.sum()) if m.any() else 0.0
    for x in range(nu):
        m = tu_ == x; bu_[x] = (trr_[m] - gm - bi_[ti_[m]]).sum() / (10 + m.sum()) if m.any() else 0.0
    Res_ = np.zeros((nu, ni))
    for k in range(len(tu_)): Res_[tu_[k], ti_[k]] = trr_[k] - gm - bu_[tu_[k]] - bi_[ti_[k]]
    gg = np.random.default_rng(0)
    Pm = gg.normal(0, 0.1, (nu, 20)); Qm = gg.normal(0, 0.1, (ni, 20)); Ik_ = np.eye(20)
    for _ in range(12):
        for x in range(nu):
            j = train_mask[x]
            if j.any(): Qj = Qm[j]; Pm[x] = np.linalg.solve(Qj.T @ Qj + 10 * Ik_, Qj.T @ Res_[x, j])
        for y in range(ni):
            xx = train_mask[:, y]
            if xx.any(): Px = Pm[xx]; Qm[y] = np.linalg.solve(Px.T @ Px + 10 * Ik_, Px.T @ Res_[xx, y])
    pr = gm + bu_[eu_] + bi_[ei_] + np.einsum('ij,ij->i', Pm[eu_], Qm[ei_])
    return float(np.sqrt(np.mean((pr - err_) ** 2)))
rmse_rand = als_mf(Mtr, tu, ti, trr, eu, ei, err)           # random split reused from exercise 1
subt = sub.copy(); subt["ux"] = subt.userId.map(uidx); subt["ix"] = subt.movieId.map(iidx)
subt = subt.sort_values(["ux", "timestamp"])
is_test_t = np.zeros(len(subt), bool); pos = 0
for x, grp in subt.groupby("ux"):                            # each user's latest 20% is the test set
    n = len(grp); cut = int(0.8 * n); is_test_t[pos + cut: pos + n] = True; pos += n
ux = subt.ux.to_numpy(); ix = subt.ix.to_numpy(); ry = subt.rating.to_numpy(); trt = ~is_test_t
Mt = np.zeros((nu, ni), bool); Mt[ux[trt], ix[trt]] = True  # the temporal training mask
rmse_temp = als_mf(Mt, ux[trt], ix[trt], ry[trt], ux[is_test_t], ix[is_test_t], ry[is_test_t])
print(f"MF RMSE, random split   : {rmse_rand:.3f}  (leaks the future -> optimistic)")
print(f"MF RMSE, temporal split : {rmse_temp:.3f}  (each user's later ratings; other users can still leak across)")
# expected output: MF RMSE, random split   : 0.776  (leaks the future -> optimistic)
# expected output: MF RMSE, temporal split : 0.834  (each user's later ratings; other users can still leak across)
base_rand = float(np.sqrt(np.mean((trr.mean() - err) ** 2)))              # a guess that cannot leak anything
base_temp = float(np.sqrt(np.mean((ry[trt].mean() - ry[is_test_t]) ** 2)))
print(f"global-mean RMSE on the same two test sets: random {base_rand:.3f}, temporal {base_temp:.3f}")
span = subt.groupby("ux").timestamp.agg(["min", "max"])                    # each user's first and last slice rating
print(f"users whose slice ratings all fall within one day: {int((span['max'] - span['min'] <= 86400).sum())} of {nu}")
# expected output: global-mean RMSE on the same two test sets: random 0.937, temporal 0.950
# expected output: users whose slice ratings all fall within one day: 163 of 326

# %% [markdown]
# ## 19.8 coding drills

# %%
import numpy as np

# %% [markdown]
# Similarity measures

# %%
# Similarity measures
a = np.array([5., 4., 5.]); b = np.array([4., 5., 4.])
print(round(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))), 3))       # expected output: 0.978
ac, bc = a - a.mean(), b - b.mean()
print(round(float(ac @ bc / (np.linalg.norm(ac) * np.linalg.norm(bc))), 3))   # expected output: -1.0
A = {1, 2, 3}; B = {2, 3, 4}
print(round(len(A & B) / len(A | B), 3))                                      # expected output: 0.5
ci = np.array([5.,4.,1.]) - 3.; cj = np.array([1.,2.,5.]) - 3.                 # subtract the shared user mean
print(round(float(ci @ cj / (np.linalg.norm(ci) * np.linalg.norm(cj))), 3))   # expected output: -1.0

# %% [markdown]
# Matrix factorization

# %%
# Matrix factorization
mu, bu, bi = 3.5, 0.2, -0.1; p = np.array([1., 0.5]); q = np.array([2., -1.])
print(round(mu + bu + bi + float(p @ q), 3))                                  # expected output: 5.1
plays = np.array([0, 2, 10]); alpha = 40
print((1 + alpha * plays).tolist())                                           # expected output: [1, 81, 401]
R = np.array([[5, 0, 3], [0, 4, 0]]); print(int((R == 0).sum()))              # expected output: 3
Q = np.array([[1., 0.], [0., 1.], [1., 1.]]); rr = np.array([5., 4., 3.])
print(np.round(np.linalg.solve(Q.T @ Q + np.eye(2), Q.T @ rr), 3).tolist())   # expected output: [2.125, 1.625]

# %% [markdown]
# Content and hybrid

# %%
# Content and hybrid
genres = ["Action", "Comedy", "Drama"]; movie = ["Action", "Drama"]
print([1.0 if g in movie else 0.0 for g in genres])                           # expected output: [1.0, 0.0, 1.0]
N, dfreq = 100, np.array([50, 10, 80])
print(np.round(np.log(N / (1 + dfreq)), 2).tolist())                          # expected output: [0.67, 2.21, 0.21]
prof = np.array([1., 0., 1.]); item = np.array([1., 1., 0.])
print(round(float(prof @ item / (np.linalg.norm(prof) * np.linalg.norm(item))), 3))  # expected output: 0.5
def z(v): v = np.asarray(v, float); return (v - v.mean()) / (v.std() + 1e-9)
print(np.round(0.8 * z([3., 5., 1.]) + 0.2 * z([0.1, 0.2, 0.05]), 3).tolist()) # expected output: [-0.053, 1.247, -1.194]

# %% [markdown]
# Learning to rank

# %%
# Learning to rank
sp, sn = 2.0, 0.5
print(round(float(1 / (1 + np.exp(-(sp - sn)))), 3))                          # expected output: 0.818
print(round(float(1 / (1 + np.exp(sp - sn))), 3))                             # expected output: 0.182
print(np.arange(4).tolist())                                                  # expected output: [0, 1, 2, 3]
scores = np.array([0.2, 0.9, 0.5, 0.1]); print(np.argsort(scores)[::-1][:2].tolist())  # expected output: [1, 2]

# %% [markdown]
# Evaluation metrics

# %%
# Evaluation metrics
rel = np.array([1, 0, 1, 1, 0])
print(round(float(rel[:3].sum() / 3), 3))                                     # expected output: 0.667
print(round(float(rel[:3].sum() / rel.sum()), 3))                            # expected output: 0.667
print(round(float(np.sum(rel[:5] / np.log2(np.arange(2, 7)))), 3))          # expected output: 1.931
dcg = np.sum(rel / np.log2(np.arange(2, len(rel) + 2)))
idcg = np.sum(np.sort(rel)[::-1] / np.log2(np.arange(2, len(rel) + 2)))
print(round(float(dcg / idcg), 3))                                          # expected output: 0.906
print(round(1.0 / (np.where(rel == 1)[0][0] + 1), 3))                       # expected output: 1.0
print(1 if rel[:3].sum() > 0 else 0)                                        # expected output: 1
