"""Chapter 19: Recommender Systems

Section-aligned walkthrough: every code block from this chapter of the book,
in book order, each in its own cell under its section heading so you can run
and check them one by one. Generated from the chapter text; keep the book as the source.
"""

# %% [markdown]
# ## 19.1 Why recommenders run the internet

# %% [markdown]
# ## 19.1.1 The feedback loop and popularity bias

# %% [markdown]
# Example 19-1. Taste-matching lift and the popularity feedback loop.

# %%
import numpy as np
rng = np.random.default_rng(0)

n_users, n_items, n_genres = 2000, 50, 5
user_taste = rng.integers(0, n_genres, n_users)          # each user prefers one genre
item_genre = rng.integers(0, n_genres, n_items)          # each item belongs to one genre
def clicks(recommend_fn):
    # show each user one recommended item; they click if its genre matches their taste
    total = 0
    for u in range(n_users):
        item = recommend_fn(u)
        total += int(item_genre[item] == user_taste[u])
    return total / n_users                                 # click-through rate

def random_rec(u): return rng.integers(n_items)
def genre_rec(u):                                          # recommends an item from the user's genre
    matches = np.where(item_genre == user_taste[u])[0]
    return rng.choice(matches)
print(f"click-through rate (chance of a click on the one item shown):")
print(f"  random item       : {clicks(random_rec):.2f}")
print(f"  taste-matched item: {clicks(genre_rec):.2f}")
print("  -> both rates follow from five genres and a click exactly when the genres match.")

pop = np.ones(n_items)                                    # click counts so far (start EQUAL)
top_share = []
for round_ in range(60):
    probs = pop ** 2 / np.sum(pop ** 2)                    # exposure grows with popularity SQUARED
    for u in range(n_users):
        choice = rng.choice(n_items, p=probs)             # exposure skewed toward popular items
        if item_genre[choice] == user_taste[u] or rng.random() < 0.1:
            pop[choice] += 1                               # a click reinforces that item's popularity
    top5_now = np.sort(pop)[::-1][:5]                      # the current top-5 items' clicks
    top_share.append(top5_now.sum() / pop.sum())          # share of all clicks held by the top 5
print(f"\npopularity feedback loop (share of clicks held by the top-5 of {n_items} items):")
print(f"  equal start would be 5/{n_items} = {5/n_items:.2f}")
print(f"  round  1: {top_share[0]:.2f}   round 20: {top_share[19]:.2f}   round 60: {top_share[-1]:.2f}")
print(f"  the head's share climbed from ~{5/n_items:.2f} to {top_share[-1]:.2f}: rich-get-richer.")
print("recommenders shape the data they are later trained on, so popularity bias and filter")
print("bubbles are built in; fixing them (exploration, debiasing, diversity) is core to the job.")

print("\nAll why-recommenders checks ran.")

# %% [markdown]
# ## 19.2 Collaborative filtering: user-based methods, item-based methods, and similarity

# %% [markdown]
# ## 19.2.1 Measuring similarity

# %% [markdown]
# Example 19-2. User-based and item-based collaborative filtering.

# %%
import numpy as np, pandas as pd
np.random.seed(0)

# ---- Part A: a tiny constructed 5x6 matrix, checkable by hand ----
# Users 0-2 like the first three "action" films; users 3-4 like the last three "romance".
R = np.array([
    [5, 4, 5, 1, 1, 2],
    [4, 5, 4, 2, 1, 1],
    [5, 5, 4, 1, 2, 1],
    [1, 2, 1, 5, 4, 5],
    [2, 1, 1, 4, 5, 4],
], dtype=float)
target_u, target_i = 0, 5                                 # predict user 0's rating of movie 5 (truly 2)
R_train = R.copy(); true_val = R_train[target_u, target_i]; R_train[target_u, target_i] = 0

def pearson(a, b):
    mask = (a != 0) & (b != 0)                            # only movies both users rated
    if mask.sum() < 2: return 0.0
    x, y = a[mask] - a[mask].mean(), b[mask] - b[mask].mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / denom) if denom > 0 else 0.0

sims = [(v, pearson(R_train[target_u], R_train[v])) for v in range(5) if v != target_u]
print("user-based CF: similarity of user 0 to the others")
for v, s in sims:
    print(f"  user {v}: similarity {s:+.2f}")

def user_mean(u): vals = R_train[u][R_train[u] != 0]; return vals.mean()
num = den = 0.0
for v, s in sims:
    if R_train[v, target_i] != 0 and s > 0:
        num += s * (R_train[v, target_i] - user_mean(v)); den += abs(s)
pred_user = user_mean(target_u) + (num / den if den else 0)
print(f"\npredict user 0's rating of movie 5: user-based = {pred_user:.2f}  (true = {true_val:.0f})")

# adjusted cosine: cosine after subtracting each USER's mean, so a positive scale
# does not fake agreement between opposite items. Every statistic, the user means
# included, comes from R_train only: the held-out cell must not leak into anything.
umean = np.array([R_train[x][R_train[x] != 0].mean() for x in range(5)])  # observed TRAIN ratings only
def adj_cosine(i, j):
    both = (R_train[:, i] != 0) & (R_train[:, j] != 0)    # users who rated BOTH items in train
    if both.sum() < 2: return 0.0
    x = R_train[both, i] - umean[both]; y = R_train[both, j] - umean[both]
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / denom) if denom > 0 else 0.0
plain = R_train[:, 0] @ R_train[:, 3] / (np.linalg.norm(R_train[:, 0]) * np.linalg.norm(R_train[:, 3]))
print(f"\nitem 0 (action) vs item 3 (romance):")
print(f"  plain cosine    = {plain:+.2f}   (misleadingly positive: everyone rates on 1-5)")
print(f"  adjusted cosine = {adj_cosine(0,3):+.2f}   (correctly negative once user means removed)")
print(f"  adjusted cosine item0 vs item1 (both action) = {adj_cosine(0,1):+.2f}")

# item-based prediction using adjusted-cosine item similarities
num = den = 0.0
for j in range(6):
    if j != target_i and R_train[target_u, j] != 0:
        s = adj_cosine(target_i, j)
        if s > 0: num += s * R_train[target_u, j]; den += abs(s)
pred_item = num / den if den else user_mean(target_u)
print(f"predict user 0's rating of movie 5: item-based = {pred_item:.2f}  (true = {true_val:.0f})")
print("both place movie 5 low for user 0, matching the action-vs-romance split.")

# ---- Part B: real MovieLens, a pinned dense slice ----
ratings = pd.read_csv("data/ratings.csv")                 # userId, movieId, rating (0.5-5.0)
top = ratings.movieId.value_counts().head(200).index      # 200 most-rated movies
sub = ratings[ratings.movieId.isin(top)]
c = sub.userId.value_counts()
sub = sub[sub.userId.isin(c[c >= 30].index)]              # users with >=30 of those ratings
uids = sorted(sub.userId.unique()); iids = sorted(sub.movieId.unique())
uidx = {u: k for k, u in enumerate(uids)}; iidx = {m: k for k, m in enumerate(iids)}
nu, ni = len(uids), len(iids)
u = sub.userId.map(uidx).to_numpy(); i = sub.movieId.map(iidx).to_numpy(); r = sub.rating.to_numpy()
print(f"\ndense slice: {nu} users x {ni} movies, {len(sub)} ratings, "
      f"{len(sub)/(nu*ni)*100:.0f}% dense")

rs = np.random.RandomState(42)                            # seed the held-out split
perm = rs.permutation(len(sub)); ncut = int(0.2 * len(sub))
is_test = np.zeros(len(sub), bool); is_test[perm[:ncut]] = True
Rtr = np.zeros((nu, ni)); Mtr = np.zeros((nu, ni), bool)
for k in np.where(~is_test)[0]: Rtr[u[k], i[k]] = r[k]; Mtr[u[k], i[k]] = True
test = list(zip(u[is_test], i[is_test], r[is_test]))

gmean = r[~is_test].mean()                                # global-mean baseline
mae_base = np.mean([abs(gmean - rt) for _, _, rt in test])

umean_s = np.array([Rtr[x][Mtr[x]].mean() if Mtr[x].any() else gmean for x in range(nu)])
Cc = np.where(Mtr, Rtr - umean_s[:, None], 0.0)           # mean-centered ratings
S = np.zeros((nu, nu))                                    # precompute all user-user Pearson sims
for x in range(nu):
    co = Mtr & Mtr[x]                                     # movies both rated
    num_ = (Cc * Cc[x]).sum(1)
    da = np.sqrt((Cc[x] ** 2 * co).sum(1)); dv = np.sqrt((Cc ** 2 * co).sum(1))
    den = da * dv
    s = np.where(den > 0, num_ / (den + 1e-12), 0.0)
    s[co.sum(1) < 2] = 0.0; s[x] = 0.0
    S[x] = s

k_nbrs = 40                                               # keep only the k_nbrs most similar
def predict_knn(x, j):
    sims = np.where(Mtr[:, j] & (S[x] > 0), S[x], 0.0)    # positive neighbors who rated j
    if not sims.any(): return umean_s[x]
    nbrs = np.argsort(sims)[::-1][:k_nbrs]
    s = sims[nbrs]
    if s.sum() == 0: return umean_s[x]
    dev = Rtr[nbrs, j] - umean_s[nbrs]
    return umean_s[x] + (s @ dev) / s.sum()

mae_cf = np.mean([abs(predict_knn(int(x), int(j)) - rt) for x, j, rt in test])
print(f"\nMAE on held-out MovieLens ratings (lower better):")
print(f"  global-mean baseline    : {mae_base:.3f}")
print(f"  user-based Pearson kNN  : {mae_cf:.3f}   (k={k_nbrs} neighbors)")
print("CF beats the baseline using only WHO rated WHAT, with no idea what the movies are.")
print("\nAll collaborative-filtering checks ran.")

# %% [markdown]
# The two refinements promised earlier deserve their own numbers, and Example 19-3, reusing Example 19-2's matrices `Mtr`, `S`, `Rtr`, and the same held-out split, measures both.
#
# Example 19-3. Counts of co-rated movies per user pair, significance weighting, a neighborhood size sweep, negative neighbors, and the Jaccard overlap, on Example 19-2's slice and split.

# %%
# continuing with Example 19-2's matrices: how trustworthy are the similarities?
co_counts = (Mtr.astype(int) @ Mtr.astype(int).T)          # n_uv for every user pair
iu = np.triu_indices(nu, k=1)
print(f"co-rated movies per user pair: median {np.median(co_counts[iu]):.0f}, "
      f"min {co_counts[iu].min()}, share of pairs below gamma=25: {(co_counts[iu] < 25).mean():.0%}")

Sw = S * np.minimum(co_counts, 25) / 25                     # significance-weighted similarities
def predict_knn_w(x, j):
    sims = np.where(Mtr[:, j] & (Sw[x] > 0), Sw[x], 0.0)
    if not sims.any(): return umean_s[x]
    nbrs = np.argsort(sims)[::-1][:k_nbrs]; s = sims[nbrs]
    if s.sum() == 0: return umean_s[x]
    return umean_s[x] + (s @ (Rtr[nbrs, j] - umean_s[nbrs])) / s.sum()
mae_w = np.mean([abs(predict_knn_w(int(x), int(j)) - rt) for x, j, rt in test])
print(f"kNN MAE: plain {mae_cf:.3f} vs significance-weighted {mae_w:.3f}")

def mae_at(k_):
    def predict(x, j):
        sims = np.where(Mtr[:, j] & (S[x] > 0), S[x], 0.0)
        if not sims.any(): return umean_s[x]
        nbrs = np.argsort(sims)[::-1][:k_]
        s = sims[nbrs]
        if s.sum() == 0: return umean_s[x]
        return umean_s[x] + (s @ (Rtr[nbrs, j] - umean_s[nbrs])) / s.sum()
    return np.mean([abs(predict(int(x), int(j)) - rt) for x, j, rt in test])

print("positive-neighbor MAE by neighborhood size:")
for k_ in (5, 40, nu - 1):
    print(f"  k={k_}: {mae_at(k_):.3f}")

def predict_abs(x, j):
    rated = Mtr[:, j].copy(); rated[x] = False
    cand = np.where(rated)[0]
    if len(cand) == 0: return umean_s[x]
    nbrs = cand[np.argsort(np.abs(S[x, cand]))[::-1][:k_nbrs]]
    s = S[x, nbrs]
    den = np.abs(s).sum()
    if den == 0: return umean_s[x]
    return umean_s[x] + (s @ (Rtr[nbrs, j] - umean_s[nbrs])) / den
mae_abs = np.mean([abs(predict_abs(int(x), int(j)) - rt) for x, j, rt in test])
print(f"top 40 by absolute similarity, negatives included: MAE {mae_abs:.3f}")
print(f"share of user pairs with negative similarity: {(S[iu] < 0).mean():.0%}")

# Jaccard on binary liked-item sets (rating >= 4), the similarity for click-style data
liked = Mtr & (Rtr >= 4.0)
inter = int((liked[0] & liked[1]).sum()); union = int((liked[0] | liked[1]).sum())
print(f"Jaccard(user 0, user 1) on liked-item sets: intersection {inter}, union {union}, "
      f"J = {inter / union:.3f}")

# %% [markdown]
# ## 19.3 Matrix factorization: latent factors, ALS, and implicit feedback

# %% [markdown]
# ## 19.3.2 Learning the factors, and implicit feedback

# %% [markdown]
# A few lines check both computations.

# %%
import numpy as np
mu, b_u, b_i, lam, eta = 3.5, 0.2, -0.3, 0.1, 0.1
p_u = np.array([0.3, -0.1]); q_i = np.array([0.5, 0.4]); r = 4.0
e = r - (mu + b_u + b_i + p_u @ q_i)                      # the residual on this rating
p_new = p_u + eta * (e * q_i - lam * p_u)                  # the user step
q_new = q_i + eta * (e * p_u - lam * q_i)                  # the item step, taken with the OLD p_u
b_u_new = b_u + eta * (e - lam * b_u); b_i_new = b_i + eta * (e - lam * b_i)
print(f"residual {e:.2f}; p_u -> ({p_new[0]:.4f}, {p_new[1]:.4f}), q_i -> ({q_new[0]:.4f}, {q_new[1]:.4f})")
print(f"prediction before {r - e:.2f}, after one SGD step {mu + b_u_new + b_i_new + p_new @ q_new:.3f}")
Q_u = np.array([[0.5, 0.4], [0.2, -0.6]]); e_u = np.array([0.6, -0.4])   # two rated items
for lam_ in (0.1, 10.0):
    p_als = np.linalg.solve(Q_u.T @ Q_u + lam_ * np.eye(2), Q_u.T @ e_u)  # (Q^T Q + lam I) p = Q^T e
    fit = Q_u @ p_als
    print(f"ALS solve, lambda {lam_:>4}: p_u = ({p_als[0]:.3f}, {p_als[1]:.3f}), "
          f"fitted residuals ({fit[0]:.3f}, {fit[1]:.3f})")

# %% [markdown]
# Example 19-4. ALS on the bias-removed residual, the implicit-feedback confidence transform, and weighted ALS from scratch.

# %%
import numpy as np, pandas as pd
np.random.seed(0)

ratings = pd.read_csv("data/ratings.csv")
top = ratings.movieId.value_counts().head(200).index
sub = ratings[ratings.movieId.isin(top)]
c = sub.userId.value_counts(); sub = sub[sub.userId.isin(c[c >= 30].index)]
uids = sorted(sub.userId.unique()); iids = sorted(sub.movieId.unique())
uidx = {u: k for k, u in enumerate(uids)}; iidx = {m: k for k, m in enumerate(iids)}
nu, ni = len(uids), len(iids)
u = sub.userId.map(uidx).to_numpy(); i = sub.movieId.map(iidx).to_numpy(); r = sub.rating.to_numpy()
rs = np.random.RandomState(42); perm = rs.permutation(len(sub)); ncut = int(0.2 * len(sub))
is_test = np.zeros(len(sub), bool); is_test[perm[:ncut]] = True; tr = ~is_test
tu, ti, trr = u[tr], i[tr], r[tr]
gmean = trr.mean()

# shrunk global-mean biases, then ALS factorizes the leftover residual
bi = np.zeros(ni); bu = np.zeros(nu); lam = 10.0
for y in range(ni):
    m = ti == y; bi[y] = (trr[m] - gmean).sum() / (lam + m.sum()) if m.any() else 0.0
for x in range(nu):
    m = tu == x; bu[x] = (trr[m] - gmean - bi[ti[m]]).sum() / (lam + m.sum()) if m.any() else 0.0
Rtr = np.zeros((nu, ni)); Mtr = np.zeros((nu, ni), bool)
for k in range(len(tu)):
    Rtr[tu[k], ti[k]] = trr[k] - gmean - bu[tu[k]] - bi[ti[k]]; Mtr[tu[k], ti[k]] = True

d, reg, sweeps = 20, 10.0, 15
g = np.random.default_rng(0)
P = g.normal(0, 0.1, (nu, d)); Q = g.normal(0, 0.1, (ni, d)); I_d = np.eye(d)
for _ in range(sweeps):
    for x in range(nu):                                    # fix Q, least-squares each user vector
        j = Mtr[x]
        if j.any():
            Qj = Q[j]; P[x] = np.linalg.solve(Qj.T @ Qj + reg * I_d, Qj.T @ Rtr[x, j])
    for y in range(ni):                                    # fix P, least-squares each item vector
        xx = Mtr[:, y]
        if xx.any():
            Px = P[xx]; Q[y] = np.linalg.solve(Px.T @ Px + reg * I_d, Px.T @ Rtr[xx, y])
pred = gmean + bu[u[is_test]] + bi[i[is_test]] + np.einsum('ij,ij->i', P[u[is_test]], Q[i[is_test]])
print(f"ALS (biases + {d} factors), held-out MAE = {np.mean(np.abs(pred - r[is_test])):.3f}")
print("ALS alternates two regularized least-squares half-steps. Each half")
print("parallelizes across all users or all items.")

# implicit feedback: no stars, only counts. Turn a play count into a confidence weight.
plays = np.array([0, 3, 0, 12])                            # times a user played four tracks
alpha = 40
pref = (plays > 0).astype(int)                             # 1 if any interaction, else 0
conf = 1 + alpha * plays                                   # c = 1 + alpha * r, confidence in that 0/1
print("\nimplicit feedback (clicks/plays, all positive):")
print("  plays        :", plays.tolist())
print("  preference   :", pref.tolist(), "(the target: did they engage at all)")
print("  confidence c :", conf.tolist(), "(weighted-ALS trusts heavy interactions more;")
print("                 unseen items keep weak confidence 1, a soft negative, not a hard 0)")

# weighted ALS from scratch: the implicit objective, actually optimized.
# Six listeners, five tracks; users 0-2 play the first three tracks, users 3-5 the last two.
plays2 = np.array([[8, 2, 0, 0, 0],
                   [5, 4, 6, 0, 0],
                   [0, 3, 7, 0, 1],
                   [0, 0, 0, 9, 4],
                   [1, 0, 0, 6, 8],
                   [0, 0, 2, 3, 5]], dtype=float)
n_u, n_i = plays2.shape
Pref = (plays2 > 0).astype(float)                       # binary preference targets p_ui
Conf = 1 + 2.0 * plays2                                 # confidence c_ui = 1 + alpha*r, alpha=2
d2, lam2 = 2, 0.1
g2 = np.random.default_rng(0)
Xu = g2.normal(0, 0.1, (n_u, d2)); Yi = g2.normal(0, 0.1, (n_i, d2))
def wals_loss():                                        # the exact objective from the text
    E = Pref - Xu @ Yi.T
    return float((Conf * E * E).sum() + lam2 * ((Xu**2).sum() + (Yi**2).sum()))
print(f"\nweighted ALS from scratch on the 6x5 play matrix, loss before: {wals_loss():.1f}")
for _ in range(20):
    for u2 in range(n_u):                               # solve (Y^T C_u Y + lam I) x_u = Y^T C_u p(u)
        Cu = np.diag(Conf[u2])
        Xu[u2] = np.linalg.solve(Yi.T @ Cu @ Yi + lam2 * np.eye(d2), Yi.T @ Cu @ Pref[u2])
    for i2 in range(n_i):                               # and the mirror solve for each item
        Ci = np.diag(Conf[:, i2])
        Yi[i2] = np.linalg.solve(Xu.T @ Ci @ Xu + lam2 * np.eye(d2), Xu.T @ Ci @ Pref[:, i2])
print(f"loss after 20 sweeps: {wals_loss():.1f}   (every unobserved cell took part, at confidence 1)")
print("user 0 never played track 2 (their group's) or track 3 (the other group's):")
print(f"  predicted preference for track 2 = {Xu[0] @ Yi[2]:.2f}   track 3 = {Xu[0] @ Yi[3]:.2f}")

# %% [markdown]
# The bias formula of this section deserves its own numbers before moving on. A short continuation of Example 19-4 reads them off the fit for two movies, one with few training ratings in the slice and a strong residual, and the one with the most training ratings.

# %%
# continuing with Example 19-4's fit: the shrunk item bias of two movies, by hand
titles = pd.read_csv("data/movies.csv").set_index("movieId").title
print(f"global training mean mu = {gmean:.3f}")
for name in ["Nutty Professor, The (1996)", "Forrest Gump (1994)"]:
    y = iidx[titles[titles == name].index[0]]              # the movie's column in the slice
    m = ti == y                                            # its training ratings
    print(f"{name}: n = {m.sum()}, sum of (r - mu) = {(trr[m] - gmean).sum():.3f}, "
          f"plain mean {(trr[m] - gmean).mean():+.3f}, shrunk b_i = {bi[y]:+.3f}")

# %% [markdown]
# Example 19-5. Matrix factorization by SGD, and the latent factor it discovers.

# %%
import numpy as np, pandas as pd
np.random.seed(0)

ratings = pd.read_csv("data/ratings.csv"); movies = pd.read_csv("data/movies.csv")
top = ratings.movieId.value_counts().head(200).index
sub = ratings[ratings.movieId.isin(top)]
c = sub.userId.value_counts(); sub = sub[sub.userId.isin(c[c >= 30].index)]
uids = sorted(sub.userId.unique()); iids = sorted(sub.movieId.unique())
uidx = {u: k for k, u in enumerate(uids)}; iidx = {m: k for k, m in enumerate(iids)}
nu, ni = len(uids), len(iids)
u = sub.userId.map(uidx).to_numpy(); i = sub.movieId.map(iidx).to_numpy(); r = sub.rating.to_numpy()
rs = np.random.RandomState(42)
perm = rs.permutation(len(sub)); ncut = int(0.2 * len(sub))
is_test = np.zeros(len(sub), bool); is_test[perm[:ncut]] = True
tr = ~is_test
tu, ti, trr = u[tr], i[tr], r[tr]                          # training triples
eu, ei, err = u[is_test], i[is_test], r[is_test]           # held-out triples
gmean = trr.mean()
print(f"{nu} users x {ni} movies; {len(trr)} train / {len(err)} test ratings")

# neighborhood CF (top-k user-based Pearson), recomputed for a fair 3-way comparison
Rtr = np.zeros((nu, ni)); Mtr = np.zeros((nu, ni), bool)
for k in np.where(tr)[0]: Rtr[u[k], i[k]] = r[k]; Mtr[u[k], i[k]] = True
umean_s = np.array([Rtr[x][Mtr[x]].mean() if Mtr[x].any() else gmean for x in range(nu)])
Cc = np.where(Mtr, Rtr - umean_s[:, None], 0.0)
S = np.zeros((nu, nu))
for x in range(nu):
    co = Mtr & Mtr[x]; num_ = (Cc * Cc[x]).sum(1)
    da = np.sqrt((Cc[x] ** 2 * co).sum(1)); dv = np.sqrt((Cc ** 2 * co).sum(1)); den = da * dv
    s = np.where(den > 0, num_ / (den + 1e-12), 0.0); s[co.sum(1) < 2] = 0.0; s[x] = 0.0; S[x] = s
def predict_knn(x, j, k_nbrs=40):
    sims = np.where(Mtr[:, j] & (S[x] > 0), S[x], 0.0)
    if not sims.any(): return umean_s[x]
    nb = np.argsort(sims)[::-1][:k_nbrs]; s = sims[nb]
    if s.sum() == 0: return umean_s[x]
    return umean_s[x] + (s @ (Rtr[nb, j] - umean_s[nb])) / s.sum()
mae_cf = np.mean([abs(predict_knn(int(x), int(j)) - rt) for x, j, rt in zip(eu, ei, err)])

# train MF (and a bias-only ablation) by SGD on observed ratings only
def train_sgd(d, lr, reg, epochs, use_factors, seed=0):
    g = np.random.default_rng(seed)
    bu = np.zeros(nu); bi = np.zeros(ni)
    P = g.normal(0, 0.1, (nu, d)); Q = g.normal(0, 0.1, (ni, d))
    for ep in range(epochs):
        for idx in g.permutation(len(tu)):
            x, j = tu[idx], ti[idx]
            pred = gmean + bu[x] + bi[j] + (P[x] @ Q[j] if use_factors else 0.0)
            e = trr[idx] - pred                            # residual on this rating
            bu[x] += lr * (e - reg * bu[x]); bi[j] += lr * (e - reg * bi[j])
            if use_factors:
                Px = P[x].copy()
                P[x] += lr * (e * Q[j] - reg * P[x])       # step along the error times the item vector
                Q[j] += lr * (e * Px - reg * Q[j])         # step along the error times the user vector
    return bu, bi, P, Q
def mae(bu, bi, P, Q, use_factors):
    pred = gmean + bu[eu] + bi[ei] + (np.einsum('ij,ij->i', P[eu], Q[ei]) if use_factors else 0.0)
    return np.mean(np.abs(pred - err))

bu0, bi0, P0, Q0 = train_sgd(1, 0.008, 0.1, 60, use_factors=False)   # biases only
mae_bias = mae(bu0, bi0, P0, Q0, False)
bu, bi, P, Q = train_sgd(20, 0.008, 0.1, 60, use_factors=True)       # full MF + biases
mae_mf = mae(bu, bi, P, Q, True)

mae_base = np.mean(np.abs(gmean - err))
print(f"\nMAE on held-out MovieLens ratings, same split (lower better):")
print(f"  global-mean baseline   : {mae_base:.3f}")
print(f"  user-based kNN CF       : {mae_cf:.3f}")
print(f"  bias-only (no factors) : {mae_bias:.3f}   <- the offsets alone do most of the work")
print(f"  MF + biases (d=20)     : {mae_mf:.3f}   <- best; factors add a modest extra")

# do the learned item factors line up with any human genre? (answer: only weakly)
mv = movies.set_index('movieId')
gen = [mv.loc[m, 'genres'] for m in iids]
ind = np.array([('Romance' in g) - ('Action' in g) for g in gen], float)  # +1 romance, -1 action
best = max(abs(np.corrcoef(Q[:, kk], ind)[0, 1]) for kk in range(Q.shape[1]))
print(f"\nbest |corr| of one learned item-factor with a Romance-minus-Action indicator = {best:.2f}")
print("no single factor equals a genre; taste axes are real but not human-labeled.")
print("\nAll matrix-factorization checks ran.")

# %% [markdown]
# Example 19-6 puts numbers on that claim with Example 19-5's split and its 20 factors: it fills the holes once with zeros and once with each movie's mean training rating, takes the best rank-20 approximation of each filled matrix, and reads the predictions off the held-out cells.
#
# Example 19-6. A truncated SVD of the rating matrix with the holes filled by zeros and then by movie means, scored on Example 19-5's held-out cells against the SGD fit.

# %%
# continuing with Example 19-5's split: fill the holes, then take a plain truncated SVD
R = np.full((nu, ni), np.nan)
R[tu, ti] = trr                                            # training ratings; NaN marks a missing cell
def truncated_svd(M, d=20):
    U, s, Vt = np.linalg.svd(M, full_matrices=False)       # exact SVD of a complete matrix
    return (U[:, :d] * s[:d]) @ Vt[:d]                     # its best rank-d approximation
zero_fill = truncated_svd(np.nan_to_num(R, nan=0.0))       # a missing rating becomes 0 stars
movie_mean = np.nanmean(R, axis=0)                         # each movie's mean training rating
mean_fill = truncated_svd(np.where(np.isnan(R), movie_mean, R))
print(f"cells holding a training rating: {(~np.isnan(R)).mean():.0%} of {nu * ni}")
print("held-out MAE on Example 19-5's split, 20 factors each (lower better):")
print(f"  SVD, holes filled with 0           : {np.mean(np.abs(zero_fill[eu, ei] - err)):.3f}"
      f"   (mean prediction {zero_fill[eu, ei].mean():.2f}, true mean {err.mean():.2f})")
print(f"  SVD, holes filled with movie means : {np.mean(np.abs(mean_fill[eu, ei] - err)):.3f}")
print(f"  SGD on the observed ratings only   : {mae_mf:.3f}")

# %% [markdown]
# Example 19-7 hands that loss to PyTorch: `nn.Embedding` tables hold the factors and the biases, autograd computes the gradients, and `torch.optim.SGD` applies them with Example 19-5's settings of 20 factors, $\eta = 0.008$, $\lambda = 0.1$, and 60 epochs.
#
# Example 19-7. The biased factorization of Example 19-5 fitted with PyTorch: `nn.Embedding` tables for the factors and biases, autograd for the gradients, and `torch.optim.SGD` for the updates.

# %%
# continuing with Example 19-5's split: the same biased factorization, fitted with PyTorch
import torch, torch.nn as nn
torch.manual_seed(0)
class BiasedMF(nn.Module):
    def __init__(self, n_users, n_items, d):
        super().__init__()
        self.P, self.Q = nn.Embedding(n_users, d), nn.Embedding(n_items, d)   # latent factor tables
        self.bu, self.bi = nn.Embedding(n_users, 1), nn.Embedding(n_items, 1) # user and item biases
        for e in (self.P, self.Q): nn.init.normal_(e.weight, 0.0, 0.1)        # the start Example 19-5 uses
        for e in (self.bu, self.bi): nn.init.zeros_(e.weight)
    def forward(self, x, j):
        p, q = self.P(x), self.Q(j)
        b_u, b_i = self.bu(x).squeeze(1), self.bi(j).squeeze(1)
        pred = gmean + b_u + b_i + (p * q).sum(1)
        penalty = (p ** 2).sum(1) + (q ** 2).sum(1) + b_u ** 2 + b_i ** 2
        return pred, penalty
model = BiasedMF(nu, ni, d=20)
opt = torch.optim.SGD(model.parameters(), lr=0.008)                          # Example 19-5's learning rate
X, J, Y = torch.tensor(tu), torch.tensor(ti), torch.tensor(trr, dtype=torch.float32)
for epoch in range(60):
    order = torch.randperm(len(Y))
    for b in range(0, len(Y), 32):                                           # 32 ratings per step
        k = order[b:b + 32]
        pred, penalty = model(X[k], J[k])
        loss = (0.5 * (Y[k] - pred) ** 2 + 0.5 * 0.1 * penalty).sum()        # per-rating loss, reg 0.1
        opt.zero_grad(); loss.backward(); opt.step()                         # autograd supplies the gradients
with torch.no_grad():
    pred_test, _ = model(torch.tensor(eu), torch.tensor(ei))
print("held-out MAE on Example 19-5's split, 20 factors (lower better):")
print(f"  SGD with the updates derived by hand (Example 19-5) : {mae_mf:.3f}")
print(f"  PyTorch: autograd gradients and torch.optim.SGD     : {np.mean(np.abs(pred_test.numpy() - err)):.3f}")

# %% [markdown]
# ## 19.3.3 SVD++ and time-aware factorization

# %% [markdown]
# A few lines check the arithmetic.

# %%
import numpy as np
mu, bu, bi = 3.5, 0.2, -0.3
p_u = np.array([0.3, -0.1]); q_i = np.array([0.5, 0.4])
y_sum = np.array([0.8, -0.4])                 # sum of y_j over the 4 items the user touched
user_vec = p_u + y_sum / np.sqrt(4)           # p_u + |N(u)|^(-1/2) * sum of y_j
r_hat = mu + bu + bi + q_i @ user_vec
print(f"SVD++ prediction: {r_hat:.2f}")
alpha_u, beta_t, days = 0.02, 0.4, 100
b_u_t = bu + alpha_u * np.sign(days) * abs(days) ** beta_t
print(f"timeSVD++ drifted user bias: {b_u_t:.3f} -> prediction {r_hat - bu + b_u_t:.2f}")

# %% [markdown]
# Example 19-8 fits both ideas on the MovieLens split.
#
# Example 19-8. SVD++, then SVD++ with an item bias per time bin, trained by SGD on Example 19-5's split and scored beside the plain biased factorization.

# %%
# continuing with Example 19-5's split: SVD++, then SVD++ plus an item bias per time bin
ts = sub.timestamp.to_numpy()                              # seconds since 1970, row-aligned with u, i, r
rows_of = [np.where(tu == x)[0] for x in range(nu)]        # each user's training rows
N = [ti[rows] for rows in rows_of]                         # N(u): the movies u rated in training
def train_svdpp(n_bins=0, d=20, lr=0.008, reg=0.1, epochs=60, seed=0):
    g = np.random.default_rng(seed)
    bu = np.zeros(nu); bi = np.zeros(ni); bit = np.zeros((ni, max(n_bins, 1)))
    P = g.normal(0, 0.1, (nu, d)); Q = g.normal(0, 0.1, (ni, d)); Y = g.normal(0, 0.1, (ni, d))
    edges = np.quantile(ts[tr], np.linspace(0, 1, n_bins + 1)[1:-1]) if n_bins else []
    bin_tr = np.searchsorted(edges, ts[tr]); bin_te = np.searchsorted(edges, ts[is_test])
    for ep in range(epochs):
        for x in g.permutation(nu):                        # users in a random order
            w = 1 / np.sqrt(len(N[x]))                     # the |N(u)|^(-1/2) normalization
            z = w * Y[N[x]].sum(0)                         # implicit part, once per user pass
            grad_y = np.zeros(d)
            for idx in g.permutation(rows_of[x]):          # that user's ratings in a random order
                j, b = ti[idx], bin_tr[idx]
                e = trr[idx] - (gmean + bu[x] + bi[j] + bit[j, b] + Q[j] @ (P[x] + z))
                bu[x] += lr * (e - reg * bu[x]); bi[j] += lr * (e - reg * bi[j])
                if n_bins: bit[j, b] += lr * (e - reg * bit[j, b])
                Qj = Q[j].copy()
                Q[j] += lr * (e * (P[x] + z) - reg * Q[j])
                P[x] += lr * (e * Qj - reg * P[x])
                grad_y += e * w * Qj                       # summed over the user's ratings
            Y[N[x]] += lr * (grad_y - len(rows_of[x]) * reg * Y[N[x]])   # one y step per user pass
    Z = np.array([Y[N[x]].sum(0) / np.sqrt(len(N[x])) for x in range(nu)])
    pred = gmean + bu[eu] + bi[ei] + bit[ei, bin_te] + np.einsum('ij,ij->i', Q[ei], P[eu] + Z[eu])
    return np.mean(np.abs(pred - err))
print("held-out MAE on Example 19-5's split, 20 factors (lower better):")
print(f"  MF + biases (Example 19-5)    : {mae_mf:.3f}")
print(f"  SVD++                         : {train_svdpp():.3f}")
print(f"  SVD++ with 10 item time bins  : {train_svdpp(n_bins=10):.3f}")

# %% [markdown]
# Example 19-9. KNNWithMeans, SVD, and SVDpp from scikit-surprise on Example 19-5's split, beside the from-scratch fits.

# %%
# continuing with Example 19-5's split: the same three models from the scikit-surprise library
import surprise
from surprise import Dataset, Reader, KNNWithMeans, SVD, SVDpp, accuracy
train_df = pd.DataFrame({"user": tu, "item": ti, "rating": trr})       # Example 19-5's training triples
reader = Reader(rating_scale=(0.5, 5.0))                                 # MovieLens stars run from 0.5 to 5.0
trainset = Dataset.load_from_df(train_df, reader).build_full_trainset()  # every training triple, no split
testset = list(zip(eu.tolist(), ei.tolist(), err.tolist()))              # the same held-out triples
pearson = {"name": "pearson", "user_based": True}                        # Example 19-2's user similarity
# Example 19-5's settings: d = 20, 60 epochs, eta = 0.008, lambda = 0.1, and a fixed seed
settings = dict(n_factors=20, n_epochs=60, lr_all=0.008, reg_all=0.1, random_state=0)
models = [("KNNWithMeans, Pearson, k=40", KNNWithMeans(k=40, sim_options=pearson, verbose=False)),
          ("SVD, 20 factors, 60 epochs", SVD(**settings)),        # the library's name for MF + biases
          ("SVDpp, 20 factors, 60 epochs", SVDpp(**settings))]    # SVD++, N(u) = the movies u rated
print(f"scikit-surprise {surprise.__version__}: held-out MAE on Example 19-5's split (lower better)")
for name, algo in models:
    preds = algo.fit(trainset).test(testset)                  # fit, then predict every held-out pair
    clipped = accuracy.mae(preds, verbose=False)              # test() clips each prediction into 0.5 to 5.0
    raw = np.mean([abs(algo.predict(x, j, clip=False).est - rt) for x, j, rt in testset])
    print(f"  {name:29s}: {clipped:.3f}   (unclipped {raw:.3f})")
print(f"  from scratch, same split (Example 19-5): kNN {mae_cf:.3f}, MF + biases {mae_mf:.3f}")

# %% [markdown]
# ## 19.4 Content-based filtering and hybrid systems

# %% [markdown]
# ## 19.4.3 Hybrid systems

# %% [markdown]
# Example 19-10. Content-based scoring, cold start, and a hybrid on real MovieLens.

# %%
import numpy as np, pandas as pd
rng = np.random.default_rng(0)

# --- load MovieLens and build the pinned dense slice (top-200 movies, users >=30 ratings) ---
ratings = pd.read_csv("data/ratings.csv")            # userId, movieId, rating, timestamp
movies  = pd.read_csv("data/movies.csv")             # movieId, title, genres (pipe-separated)
top_items = ratings["movieId"].value_counts().head(200).index
sub = ratings[ratings["movieId"].isin(top_items)]
keep_users = sub["userId"].value_counts()
keep_users = keep_users[keep_users >= 30].index      # users with >=30 ratings among these movies
sub = sub[sub["userId"].isin(keep_users)]
users = np.sort(sub["userId"].unique()); items = np.sort(sub["movieId"].unique())
uix = {u: k for k, u in enumerate(users)}; iix = {m: k for k, m in enumerate(items)}
nu, ni = len(users), len(items)
print(f"dense slice: {nu} users x {ni} movies, {len(sub)} ratings, "
      f"{100*len(sub)/(nu*ni):.0f}% dense")

# 80/20 held-out split of the OBSERVED ratings (seeded)
rows = sub.sample(frac=1.0, random_state=0).reset_index(drop=True)
cut = int(0.8 * len(rows)); train_df, test_df = rows.iloc[:cut], rows.iloc[cut:]
R = np.zeros((nu, ni)); mask = np.zeros((nu, ni), bool)
for u, m, r in zip(train_df.userId, train_df.movieId, train_df.rating):
    R[uix[u], iix[m]] = r; mask[uix[u], iix[m]] = True

# item features from real genres: multi-hot, then idf-weight (rare genres carry more signal)
mrow = movies.set_index("movieId").loc[items]
genre_lists = [g.split("|") for g in mrow["genres"]]
all_genres = sorted({g for gl in genre_lists for g in gl})
gix = {g: k for k, g in enumerate(all_genres)}; ng = len(all_genres)
G = np.zeros((ni, ng))                                # multi-hot genre matrix (item x genre)
for i, gl in enumerate(genre_lists):
    for g in gl: G[i, gix[g]] = 1.0
idf = np.log(ni / (1 + G.sum(0)))                     # rare genres carry more signal
Gw = G * idf                                          # idf-weighted item genre vectors
print(f"{ng} genres; idf(Drama)={idf[gix['Drama']]:.2f} (common), "
      f"idf(Film-Noir)={idf[gix['Film-Noir']]:.2f} (rare)")

def content_profile(u):
    liked = np.where(mask[u] & (R[u] >= 4))[0]        # movies this user rated >=4 in TRAIN
    if len(liked) == 0: return np.zeros(ng)
    prof = Gw[liked].mean(0); n = np.linalg.norm(prof)
    return prof / n if n > 0 else prof
def content_score(u, i):
    q = Gw[i]; n = np.linalg.norm(q)
    return content_profile(u) @ (q / n) if n > 0 else 0.0

cs, tr = [], []
for u, m, r in zip(test_df.userId, test_df.movieId, test_df.rating):
    cs.append(content_score(uix[u], iix[m])); tr.append(r)
cs = np.array(cs); tr = np.array(tr)
print(f"\ncontent score vs held-out rating, correlation = {np.corrcoef(cs, tr)[0,1]:.2f}  (WEAK:"
      f" genres are coarse and ratings low-variance)")

# cold start: score a brand-new pure-Romance release with zero ratings
new = np.zeros(ng); new[gix["Romance"]] = idf[gix["Romance"]]; new /= np.linalg.norm(new)
rom = gix["Romance"]
def romance_frac(u):
    liked = np.where(mask[u] & (R[u] >= 4))[0]
    return G[liked, rom].mean() if len(liked) > 0 else -1.0
rf = np.array([romance_frac(u) for u in range(nu)])
fans   = [u for u in range(nu) if rf[u] >= 0.35]      # >=35% of liked movies are Romance
others = [u for u in range(nu) if 0.0 <= rf[u] <= 0.10]  # <=10% Romance among liked
print(f"\nnew pure-Romance release, content score (CF is blind, it has no ratings):")
print(f"  romance-loving users : {np.mean([content_profile(u) @ new for u in fans]):.2f}"
      f"   ({len(fans)} users)")
print(f"  low-romance users    : {np.mean([content_profile(u) @ new for u in others]):.2f}"
      f"   ({len(others)} users)")

# a biased matrix-factorization model (d=20) for the collaborative signal
d = 20; mu = R[mask].mean()
P = rng.normal(0, .1, (nu, d)); Q = rng.normal(0, .1, (ni, d))
bu = np.zeros(nu); bi = np.zeros(ni)
idx = np.argwhere(mask); lr, reg = 0.008, 0.1
for ep in range(60):
    rng.shuffle(idx)
    for u, i in idx:
        e = R[u, i] - (mu + bu[u] + bi[i] + P[u] @ Q[i])
        bu[u] += lr * (e - reg * bu[u]); bi[i] += lr * (e - reg * bi[i])
        Pu = P[u].copy()
        P[u] += lr * (e * Q[i] - reg * P[u]); Q[i] += lr * (e * Pu - reg * Q[i])
def mf_score(u, i): return mu + bu[u] + bi[i] + P[u] @ Q[i]

# hybrid: z-score both signals, then sweep the blend weight
mfv = np.array([mf_score(uix[u], iix[m]) for u, m in zip(test_df.userId, test_df.movieId)])
trv = tr
def z(a): a = np.asarray(a, float); return (a - a.mean()) / (a.std() + 1e-9)
czs, mfz = z(cs), z(mfv)
print(f"\ncorrelation with held-out rating:")
print(f"  content only   : {np.corrcoef(czs, trv)[0,1]:.2f}")
print(f"  MF only        : {np.corrcoef(mfz, trv)[0,1]:.2f}")
print(f"  hybrid (equal) : {np.corrcoef((czs+mfz)/2, trv)[0,1]:.2f}  <- WORSE than MF alone")
print("\nblend sweep  w*MF + (1-w)*content:")
for w in [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]:
    print(f"  w={w:.2f} : corr {np.corrcoef(w*mfz + (1-w)*czs, trv)[0,1]:.3f}")
print("only a strongly MF-heavy blend (w~0.9) edges past pure MF; the hybrid's real payoff is")
print("cold-start coverage: a new item can use content; a brand-new user needs popularity or onboarding.")

print("\nAll content/hybrid checks ran.")

# %% [markdown]
# ## 19.5 Learning to rank and two-tower neural recommenders

# %% [markdown]
# ## 19.5.1 Three ways to learn a ranking

# %% [markdown]
# A short check confirms the arithmetic of one BPR ascent step, with Example 19-13's step size and penalty.

# %%
import numpy as np
sigmoid = lambda x: 1 / (1 + np.exp(-x))
p, qi, qj = np.array([0.5, 0.2]), np.array([0.4, 0.6]), np.array([0.1, -0.3])
lr, reg = 0.05, 0.02                                   # Example 19-13's step size and penalty
x = p @ (qi - qj); g = sigmoid(-x)                     # the margin, and the slope of ln sigmoid at it
p_new  = p  + lr * (g * (qi - qj) - reg * p)           # the three ascent steps of the derivation
qi_new = qi + lr * (g * p - reg * qi)
qj_new = qj + lr * (-g * p - reg * qj)
print(f"margin x = {x:.2f}, slope sigmoid(-x) = {g:.4f}, ln sigmoid(x) = {np.log(sigmoid(x)):.4f}")
print("p_u ->", p_new.round(4), " q_i ->", qi_new.round(4), " q_j ->", qj_new.round(4))
x2 = p_new @ (qi_new - qj_new)
print(f"after one step: margin {x2:.4f}, ln sigmoid {np.log(sigmoid(x2)):.4f}")

# %% [markdown]
# Example 19-11. A factorization machine with a context feature, from scratch on MovieLens.

# %%
import numpy as np, pandas as pd
np.random.seed(0)

ratings = pd.read_csv("data/ratings.csv")
top = ratings.movieId.value_counts().head(200).index
sub = ratings[ratings.movieId.isin(top)]
c = sub.userId.value_counts(); sub = sub[sub.userId.isin(c[c >= 30].index)]
uids = sorted(sub.userId.unique()); iids = sorted(sub.movieId.unique())
uidx = {u: k for k, u in enumerate(uids)}; iidx = {m: k for k, m in enumerate(iids)}
nu, ni = len(uids), len(iids)
u = sub.userId.map(uidx).to_numpy(); i = sub.movieId.map(iidx).to_numpy(); r = sub.rating.to_numpy()
hour = ((sub.timestamp.to_numpy() // 3600) % 24)   # MovieLens timestamps are seconds since 1970-01-01 UTC
ctx = (hour // 6).astype(int)                        # UTC buckets: 0 night, 1 morning, 2 afternoon, 3 evening
rs = np.random.RandomState(42); perm = rs.permutation(len(sub)); ncut = int(0.2 * len(sub))
is_test = np.zeros(len(sub), bool); is_test[perm[:ncut]] = True; tr = ~is_test
print("context bucket counts (night/morning/afternoon/evening):", np.bincount(ctx))

def fit_fm(use_ctx, d=8, lr=0.01, reg=0.02, epochs=25, seed=0):
    g = np.random.default_rng(seed)
    n_feat = nu + ni + (4 if use_ctx else 0)         # one-hot users + items (+ 4 dayparts)
    w0 = r[tr].mean(); w = np.zeros(n_feat); V = g.normal(0, 0.05, (n_feat, d))
    def feats(k):                                    # indices of the active features of sample k
        f = [u[k], nu + i[k]]
        if use_ctx: f.append(nu + ni + ctx[k])
        return f
    idx_tr = np.where(tr)[0]
    for ep in range(epochs):
        for k in g.permutation(idx_tr):
            f = feats(k)
            vs = V[f]
            pair = 0.5 * ((vs.sum(0) ** 2 - (vs ** 2).sum(0)).sum())   # the FM identity
            e = (w0 + w[f].sum() + pair) - r[k]      # signed error on this rating
            w0 -= lr * e
            for a in f:
                w[a] -= lr * (e + reg * w[a])
            S = vs.sum(0)
            for a in f:                              # dL/dv_a = e * (sum of other active v) + reg
                V[a] -= lr * (e * (S - V[a]) + reg * V[a])
    def predict(k):
        f = feats(k)
        vs = V[f]
        return w0 + w[f].sum() + 0.5 * ((vs.sum(0) ** 2 - (vs ** 2).sum(0)).sum())
    mae = np.mean([abs(predict(k) - r[k]) for k in np.where(is_test)[0]])
    return mae, (w0, w, V, feats, predict)

mae_ui, _ = fit_fm(False)
mae_ctx, (w0, w, V, feats, predict) = fit_fm(True)
print(f"FM (user+item)         MAE = {mae_ui:.3f}")
print(f"FM (user+item+daypart) MAE = {mae_ctx:.3f}")

# one prediction taken apart, term by term
k = np.where(is_test)[0][0]
f = feats(k)
names = ["user", "item", "daypart"]
print(f"\nfirst held-out rating: user {u[k]}, item {i[k]}, daypart {ctx[k]}, true {r[k]}")
print(f"w0 = {w0:.3f}; linear terms:", {names[j]: round(float(w[a]), 3) for j, a in enumerate(f)})
pw = {}
for a in range(3):
    for b in range(a + 1, 3):
        pw[f"{names[a]}*{names[b]}"] = round(float(V[f[a]] @ V[f[b]]), 3)
print("pairwise dot products:", pw)
print(f"prediction = {predict(k):.3f}")

# %% [markdown]
# Example 19-12 trains all three families with the same trees and the same four features: XGBoost's `reg:squarederror` objective for the pointwise model, `rank:pairwise` for the pairwise one, and `rank:ndcg` for LambdaMART.
#
# Example 19-12. Pointwise, pairwise (RankNet), and listwise (LambdaMART) XGBoost rankers on four features from the training ratings, scored by NDCG@10 on Example 19-11's split.

# %%
# continuing with Example 19-11's slice and split: one feature set, three ranking objectives
import os; os.environ["OMP_NUM_THREADS"] = "2"   # cap the OpenMP threads the XGBoost fits spawn
import xgboost as xgb
movies = pd.read_csv("data/movies.csv").set_index("movieId")
genre_sets = [set(movies.loc[m, "genres"].split("|")) for m in iids]
all_genres = sorted(set().union(*genre_sets))
G = np.array([[g in gs for g in all_genres] for gs in genre_sets], float)
Gn = G / np.linalg.norm(G, axis=1, keepdims=True)
gsim = Gn @ Gn.T                                           # genre cosine between every two movies
tu, ti, trr = u[tr], i[tr], r[tr]; eu, ei, er = u[is_test], i[is_test], r[is_test]
movie_mean = np.array([trr[ti == y].mean() for y in range(ni)])
log_count = np.log1p(np.bincount(ti, minlength=ni))
user_mean = np.array([trr[tu == x].mean() for x in range(nu)])
Dev = np.zeros((nu, ni)); Rated = np.zeros((nu, ni), bool)
Dev[tu, ti] = trr - user_mean[tu]; Rated[tu, ti] = True    # training ratings, centered per user
def features(us, its):
    W = gsim[its] * Rated[us]                              # genre similarity to each movie the user rated
    W[np.arange(len(us)), its] = 0.0                       # leave the target movie itself out
    taste = (W * Dev[us]).sum(1) / (W.sum(1) + 1e-9)       # genre-weighted deviation from the user's mean
    return np.column_stack([movie_mean[its], log_count[its], user_mean[us], taste])
order = np.argsort(tu, kind="stable")                      # rankers want each user's rows together
Xtr, ytr, qid = features(tu, ti)[order], trr[order], tu[order]
Xte = features(eu, ei)
trees = dict(n_estimators=200, max_depth=4, learning_rate=0.05, n_jobs=2, random_state=0)
pairs = dict(lambdarank_pair_method="topk", lambdarank_num_pair_per_sample=10)
labels = (2 * ytr).astype(int)                             # half stars become integer grades 1 to 10
pointwise = xgb.XGBRegressor(objective="reg:squarederror", **trees).fit(Xtr, ytr)
pairwise = xgb.XGBRanker(objective="rank:pairwise", **pairs, **trees).fit(Xtr, labels, qid=qid)
listwise = xgb.XGBRanker(objective="rank:ndcg", ndcg_exp_gain=False, **pairs, **trees).fit(Xtr, labels, qid=qid)

def ndcg10(score):                                         # rerank each user's held-out movies
    disc = 1 / np.log2(np.arange(2, 12)); out = []
    for x in np.unique(eu):
        m = eu == x
        if m.sum() < 3: continue                           # users with at least 3 held-out ratings
        ranked = er[m][np.argsort(-score[m], kind="stable")[:10]]
        ideal = np.sort(er[m])[::-1][:10]
        out.append((ranked * disc[:len(ranked)]).sum() / (ideal * disc[:len(ideal)]).sum())
    return np.mean(out)
print("NDCG@10 over each user's held-out movies, star ratings as gains (higher better):")
print(f"  random order (the floor)  : {ndcg10(np.random.default_rng(0).random(len(er))):.3f}")
print(f"  movie's mean rating only  : {ndcg10(movie_mean[ei]):.3f}")
print(f"  pointwise, squared error  : {ndcg10(pointwise.predict(Xte)):.3f}")
print(f"  pairwise, RankNet loss    : {ndcg10(pairwise.predict(Xte)):.3f}")
print(f"  listwise, LambdaMART      : {ndcg10(listwise.predict(Xte)):.3f}")

# %% [markdown]
# ## 19.5.2 Two-tower models for retrieval at scale

# %% [markdown]
# Example 19-13. Pointwise versus pairwise (BPR) ranking, and a two-tower retriever, on MovieLens.

# %%
import numpy as np, pandas as pd
import torch, torch.nn as nn
rng = np.random.default_rng(0); torch.manual_seed(0)

# --- implicit-positives slice: rating>=4 is a positive, on the top-200 movies ---
ratings = pd.read_csv("data/ratings.csv")
top_items = ratings["movieId"].value_counts().head(200).index
sub = ratings[ratings["movieId"].isin(top_items)]
pos_df = sub[sub["rating"] >= 4.0]
c = pos_df["userId"].value_counts(); keep = c[c >= 10].index   # >=10 positives per user
pos_df = pos_df[pos_df["userId"].isin(keep)]
users = np.sort(pos_df["userId"].unique()); items = np.sort(sub["movieId"].unique())
uix = {u: k for k, u in enumerate(users)}; iix = {m: k for k, m in enumerate(items)}
nu, ni = len(users), len(items)
pos_of = {u: set() for u in range(nu)}
for u, m in zip(pos_df.userId, pos_df.movieId): pos_of[uix[u]].add(iix[m])
print(f"positives slice: {nu} users x {ni} movies, {len(pos_df)} positive (rating>=4) interactions")

# leave-one-out: hold out one positive per user for evaluation
train_pos = {u: set(s) for u, s in pos_of.items()}; heldout = []
for u in range(nu):
    s = list(train_pos[u]); h = s[rng.integers(len(s))]
    train_pos[u].discard(h); heldout.append((u, h))
train_pairs = [(u, i) for u in range(nu) for i in train_pos[u]]

def auc(score):
    # AUC = P(held-out positive outranks a random unobserved item), 100 negatives per user
    wins = tot = 0
    for u, i in heldout:
        sp = score(u, i)
        for _ in range(100):
            j = rng.integers(ni)
            while j in pos_of[u]: j = rng.integers(ni)
            wins += sp > score(u, j); tot += 1
    return wins / tot

def train_factors(mode, d=32, epochs=30, lr=0.05, reg=0.02):
    P = rng.normal(0, 0.1, (nu, d)); Q = rng.normal(0, 0.1, (ni, d))
    tp = list(train_pairs)
    for _ in range(epochs):
        rng.shuffle(tp)
        for u, i in tp:
            j = rng.integers(ni)
            while j in train_pos[u]: j = rng.integers(ni)        # training negative: outside the training positives only
            if mode == "pointwise":
                for it, label in [(i, 1.0), (j, 0.0)]:
                    e = label - P[u] @ Q[it]; Pu = P[u].copy()
                    P[u] += lr * (e * Q[it] - reg * P[u]); Q[it] += lr * (e * Pu - reg * Q[it])
            else:  # pairwise BPR: maximize log sigmoid(score_i - score_j)
                x = P[u] @ Q[i] - P[u] @ Q[j]
                g = 1 / (1 + np.exp(x))                           # g = sigmoid(-x), the slope of log sigmoid(x) we ascend
                Pu = P[u].copy()
                P[u] += lr * (g * (Q[i] - Q[j]) - reg * P[u])
                Q[i] += lr * (g * Pu - reg * Q[i]); Q[j] += lr * (-g * Pu - reg * Q[j])
    return P, Q

for mode in ["pointwise", "pairwise"]:
    P, Q = train_factors(mode)
    print(f"{mode:9s} held-out ranking AUC = {auc(lambda u, i: P[u] @ Q[i]):.3f}")
print("pairwise (BPR) optimizes ORDER directly, so it ranks held-out likes above random items")
print("a bit better than pointwise.")

# two-tower / dual-encoder retrieval: user & item embeddings, score = dot product,
# trained with IN-BATCH NEGATIVES (other items in the batch act as negatives)
class Tower(nn.Module):
    def __init__(self, n, d=32):
        super().__init__(); self.emb = nn.Embedding(n, d); nn.init.normal_(self.emb.weight, std=0.05)
    def forward(self, ids): return nn.functional.normalize(self.emb(ids), dim=-1)
user_tower, item_tower = Tower(nu), Tower(ni)
opt = torch.optim.Adam(list(user_tower.parameters()) + list(item_tower.parameters()), lr=0.01)

tp = np.array(train_pairs)
for ep in range(120):
    b = tp[rng.choice(len(tp), 512)]
    us = torch.tensor(b[:, 0]); it = torch.tensor(b[:, 1])
    ue = user_tower(us); ie = item_tower(it)
    logits = ue @ ie.T * 10                                       # every user vs every item; x10 = softmax temperature 0.1
    loss = nn.functional.cross_entropy(logits, torch.arange(len(b)))  # matching item = positive
    opt.zero_grad(); loss.backward(); opt.step()

with torch.no_grad():
    UE = user_tower(torch.arange(nu)); IE = item_tower(torch.arange(ni))
    hit10 = 0
    for u, i in heldout:
        order = np.argsort((UE[u] @ IE.T).numpy())[::-1]
        order = [x for x in order if x not in train_pos[u]]      # leave the training items out of the list
        hit10 += i in order[:10]
    chance = float(np.mean([10 / (ni - len(train_pos[u])) for u, i in heldout]))
    print(f"\ntwo-tower recall@10 (held-out positive in top-10 retrieved) = {hit10/len(heldout):.2f}"
          f"  (chance {chance:.3f})")
print("at serving time you embed the user once and fetch nearest item vectors with ANN search;")
print("this is how YouTube-scale systems retrieve a few hundred candidates from millions.")

print("\nAll learning-to-rank checks ran.")

# %% [markdown]
# ## 19.5.3 The same models from a standard library

# %% [markdown]
# Example 19-14. Weighted ALS and BPR from the implicit library on the MovieLens positives slice.

# %%
import numpy as np, pandas as pd
import scipy.sparse as sp
import implicit
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from threadpoolctl import threadpool_limits              # implicit wants a single-threaded BLAS
rng = np.random.default_rng(0)
print(f"implicit version {implicit.__version__}")

# the same implicit-positives slice as the from-scratch ranking models:
# rating >= 4 on the top-200 movies is a positive; keep users with >= 10 positives
ratings = pd.read_csv("data/ratings.csv")
top_items = ratings["movieId"].value_counts().head(200).index
sub = ratings[ratings["movieId"].isin(top_items)]
pos_df = sub[sub["rating"] >= 4.0]
c = pos_df["userId"].value_counts(); keep = c[c >= 10].index
pos_df = pos_df[pos_df["userId"].isin(keep)]
users = np.sort(pos_df["userId"].unique()); items = np.sort(sub["movieId"].unique())
uix = {u: k for k, u in enumerate(users)}; iix = {m: k for k, m in enumerate(items)}
nu, ni = len(users), len(items)
pos_of = {u: set() for u in range(nu)}
for u, m in zip(pos_df.userId, pos_df.movieId): pos_of[uix[u]].add(iix[m])

# identical leave-one-out protocol: hide one positive per user, rank it later
train_pos = {u: set(s) for u, s in pos_of.items()}; heldout = []
for u in range(nu):
    s = list(train_pos[u]); h = s[rng.integers(len(s))]
    train_pos[u].discard(h); heldout.append((u, h))
rows = [u for u in range(nu) for _ in train_pos[u]]
cols = [i for u in range(nu) for i in train_pos[u]]
# confidence c = 1 + alpha*r with alpha = 2 and r = 1 for every kept positive; on this
# small, dense slice a huge alpha would overweight the positives (see the tuning note in Section 19.5.3)
conf = np.full(len(rows), 3.0)
UI = sp.csr_matrix((conf, (rows, cols)), shape=(nu, ni))   # users x movies training matrix
print(f"positives slice: {nu} users x {ni} movies, {UI.nnz} training positives")

def auc(score):
    # AUC = P(held-out positive outranks a random unobserved item), 100 negatives per user
    wins = tot = 0
    for u, i in heldout:
        sp_ = score(u, i)
        for _ in range(100):
            j = rng.integers(ni)
            while j in pos_of[u]: j = rng.integers(ni)
            wins += sp_ > score(u, j); tot += 1
    return wins / tot

with threadpool_limits(limits=1, user_api="blas"):         # one BLAS thread while implicit builds and trains
    als = AlternatingLeastSquares(factors=8, regularization=0.05, iterations=30,
                                  random_state=0, num_threads=2)
    als.fit(UI, show_progress=False)                       # weighted ALS, exactly Section 19.3's model
    bpr = BayesianPersonalizedRanking(factors=32, learning_rate=0.01, regularization=0.01,
                                      iterations=100, random_state=0, num_threads=1)
    bpr.fit(UI, show_progress=False)                       # BPR, exactly Section 19.5's objective

print("\nheld-out ranking AUC, implicit-library models (same protocol as the from-scratch runs):")
print(f"  implicit.als weighted ALS : {auc(lambda u, i: als.user_factors[u] @ als.item_factors[i]):.3f}")
print(f"  implicit.bpr BPR          : {auc(lambda u, i: bpr.user_factors[u] @ bpr.item_factors[i]):.3f}")

# the library's own serving call: rank every movie for one user in a single line
ids, scores = als.recommend(0, UI[0], N=5, filter_already_liked_items=True)
movies = pd.read_csv("data/movies.csv").set_index("movieId")
print("\nimplicit.als top-5 recommendations for user index 0 (training titles filtered out):")
for m, s in zip(ids, scores):
    print(f"  {s:.3f}  {movies.loc[items[m], 'title']}")

# %% [markdown]
# ## 19.5.4 Sequential models and the two-stage architecture

# %% [markdown]
# Example 19-15. A GRU next-item recommender evaluated on a temporal split.

# %%
import numpy as np, pandas as pd
import torch, torch.nn as nn
rng = np.random.default_rng(0); torch.manual_seed(0)

ratings = pd.read_csv("data/ratings.csv")
top_items = ratings["movieId"].value_counts().head(200).index
sub = ratings[ratings["movieId"].isin(top_items)]
pos = sub[sub["rating"] >= 4.0].sort_values("timestamp", kind="stable")   # positives in TIME order;
# kind="stable" keeps tied timestamps in file order, so every platform holds out the same last item
c = pos["userId"].value_counts(); pos = pos[pos["userId"].isin(c[c >= 10].index)]
items = np.sort(sub["movieId"].unique()); iix = {m: k for k, m in enumerate(items)}
ni = len(items)

# temporal, per user: hold out each user's chronologically LAST positive
train_seq, test_next = {}, {}
for uid, grp in pos.groupby("userId"):
    seq = [iix[m] for m in grp["movieId"]]
    train_seq[uid] = seq[:-1]; test_next[uid] = seq[-1]
print(f"{len(train_seq)} users; per-user temporal split: train on all but the last positive")

# aside: how thin a strict global-cutoff split would be on this slice
cutoff = np.quantile(pos["timestamp"], 0.8)
spanning = sum(1 for uid, grp in pos.groupby("userId")
               if (grp["timestamp"] < cutoff).sum() >= 3 and (grp["timestamp"] >= cutoff).sum() >= 1)
print(f"(a strict global cutoff at the 80th-percentile timestamp would leave only {spanning} spanning test users)")
full_cut = np.quantile(ratings["timestamp"], 0.8)
full_span = sum(1 for uid, grp in ratings.groupby("userId")
                if (grp["timestamp"] < full_cut).sum() >= 3 and (grp["timestamp"] >= full_cut).sum() >= 1)
one_day = sum(1 for uid, grp in ratings.groupby("userId")
              if grp["timestamp"].max() - grp["timestamp"].min() <= 86400)
print(f"full file, same cutoff rule: {full_span} spanning users of {ratings.userId.nunique()}; "
      f"{one_day} users place every rating inside one day")

MAXLEN = 20
X, y = [], []
for s in train_seq.values():
    for t in range(1, len(s)):                             # every prefix predicts its next item
        X.append(s[max(0, t - MAXLEN):t]); y.append(s[t])
print(f"next-item training samples: {len(X)}")

def pad(batch):                                            # left-pad with token 0 (item ids + 1)
    L = max(len(b) for b in batch)
    return torch.tensor([[0] * (L - len(b)) + [i + 1 for i in b] for b in batch])

class GRURec(nn.Module):
    def __init__(self, ni, emb=32, hid=64):
        super().__init__()
        self.emb = nn.Embedding(ni + 1, emb, padding_idx=0)
        self.gru = nn.GRU(emb, hid, batch_first=True)
        self.out = nn.Linear(hid, ni)
    def forward(self, x):
        h, _ = self.gru(self.emb(x))
        return self.out(h[:, -1])                          # score every item from the LAST hidden state

model = GRURec(ni)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
order = np.arange(len(X)); B = 128
for epoch in range(10):
    rng.shuffle(order); tot = 0.0
    for k in range(0, len(order), B):
        idx = order[k:k + B]
        logits = model(pad([X[j] for j in idx]))
        loss = nn.functional.cross_entropy(logits, torch.tensor([y[j] for j in idx]))
        opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(idx)
    if epoch % 3 == 2 or epoch == 9: print(f"epoch {epoch + 1}: mean loss {tot / len(order):.3f}")

popularity = np.zeros(ni)
for s in train_seq.values():
    for it in s: popularity[it] += 1

model.eval()
res = {"gru": [0, 0.0], "pop": [0, 0.0]}
with torch.no_grad():
    for uid, target in test_next.items():
        hist = train_seq[uid]
        scores = model(pad([hist[-MAXLEN:]]))[0].numpy().copy()
        popsc = popularity.copy()
        for it in hist:                                    # never recommend what they already saw
            scores[it] = -np.inf; popsc[it] = -np.inf
        for key, sc in [("gru", scores), ("pop", popsc)]:
            rank = int((sc > sc[target]).sum()) + 1
            if rank <= 10:
                res[key][0] += 1; res[key][1] += 1 / rank
n = len(test_next)
print(f"\nnext-item prediction, per-user temporal split ({n} test users, catalog {ni}):")
print(f"  popularity baseline : hit@10 {res['pop'][0] / n:.3f}   MRR@10 {res['pop'][1] / n:.3f}")
print(f"  GRU next-item model : hit@10 {res['gru'][0] / n:.3f}   MRR@10 {res['gru'][1] / n:.3f}")

# %% [markdown]
# ## 19.6 Cold start, popularity bias, and evaluation

# %% [markdown]
# ## 19.6.3 Offline metrics versus online truth

# %% [markdown]
# Example 19-16. Ranking metrics, real cold-start AUC, offline-metric bias, and beyond-accuracy metrics.

# %%
import numpy as np, pandas as pd
rng = np.random.default_rng(0)

# (1) ranking metrics on ONE hand-built ranked list (a constructed illustration)
ranked_rel = np.array([1, 0, 1, 1, 0, 0, 0, 1, 0, 0])    # relevance of items at ranks 1..10
def precision_at_k(rel, k): return rel[:k].sum() / k
def recall_at_k(rel, k):    return rel[:k].sum() / rel.sum()
def dcg(rel, k):            return np.sum(rel[:k] / np.log2(np.arange(2, k + 2)))
def ndcg(rel, k):
    ideal = np.sort(rel)[::-1]
    return dcg(rel, k) / dcg(ideal, k)
for k in [3, 5]:
    print(f"  @{k}: precision {precision_at_k(ranked_rel,k):.2f}  "
          f"recall {recall_at_k(ranked_rel,k):.2f}  NDCG {ndcg(ranked_rel,k):.2f}")
good = np.array([1,1,1,0,0,0,0,0,0,0]); bad = np.array([0,0,0,0,0,0,0,1,1,1])
print(f"\nsame 3 relevant items, NDCG@10: ranked first = {ndcg(good,10):.2f}, "
      f"ranked last = {ndcg(bad,10):.2f}")

# MAP, MRR, hit-rate: also on constructed lists
def average_precision(rel):
    hits = 0; s = 0.0
    for i, r in enumerate(rel, start=1):
        if r: hits += 1; s += hits / i                   # precision at each relevant position
    return s / rel.sum()
def reciprocal_rank(rel):
    idx = np.where(rel == 1)[0]
    return 1.0 / (idx[0] + 1) if len(idx) else 0.0       # 1 / rank of the first relevant item
def hit_rate(rel, k): return 1.0 if rel[:k].sum() > 0 else 0.0
list2 = np.array([0, 0, 1, 0, 1, 0, 0, 0, 0, 0])         # a second query: first relevant at rank 3
print(f"AP (this list)          = {average_precision(ranked_rel):.3f}")
print(f"MRR over 2 queries      = {(reciprocal_rank(ranked_rel)+reciprocal_rank(list2))/2:.3f}"
      f"   (RR = {reciprocal_rank(ranked_rel):.0f} and {reciprocal_rank(list2):.3f})")
print(f"hit-rate@3 (this list)  = {hit_rate(ranked_rel,3):.0f}  (a relevant item is in the top 3)")

# (2) cold start on REAL MovieLens: MF ranking AUC by user history length
ratings = pd.read_csv("data/ratings.csv")
top = ratings.movieId.value_counts().head(200).index
sub = ratings[ratings.movieId.isin(top)]
pos_df = sub[sub.rating >= 4.0]
c = pos_df.userId.value_counts(); keep = c[c >= 2].index  # >=2 so one positive can be held out
pos_df = pos_df[pos_df.userId.isin(keep)]
users = np.sort(pos_df.userId.unique()); items = np.sort(sub.movieId.unique())
uix = {u: k for k, u in enumerate(users)}; iix = {m: k for k, m in enumerate(items)}
nu, ni = len(users), len(items)
pos_of = {u: set() for u in range(nu)}
for u, m in zip(pos_df.userId, pos_df.movieId): pos_of[uix[u]].add(iix[m])
train_pos = {u: set(s) for u, s in pos_of.items()}; heldout = []
for u in range(nu):
    s = list(train_pos[u]); h = s[rng.integers(len(s))]
    train_pos[u].discard(h); heldout.append((u, h))
hist_len = np.array([len(train_pos[u]) for u in range(nu)])
pairs = [(u, i) for u in range(nu) for i in train_pos[u]]

d = 16; P = rng.normal(0, .1, (nu, d)); Q = rng.normal(0, .1, (ni, d)); lr, reg = 0.05, 0.02
for _ in range(25):                                       # BPR pairwise training
    rng.shuffle(pairs)
    for u, i in pairs:
        j = rng.integers(ni)
        while j in train_pos[u]: j = rng.integers(ni)   # training negative, so the held-out item can be sampled
        x = P[u] @ Q[i] - P[u] @ Q[j]; g = 1/(1+np.exp(x)); Pu = P[u].copy()
        P[u] += lr*(g*(Q[i]-Q[j]) - reg*P[u])
        Q[i] += lr*(g*Pu - reg*Q[i]); Q[j] += lr*(-g*Pu - reg*Q[j])

def user_auc(u, i):
    sp = P[u] @ Q[i]; w = 0
    for _ in range(100):
        j = rng.integers(ni)
        while j in pos_of[u]: j = rng.integers(ni)
        w += sp > P[u] @ Q[j]
    return w / 100
aucs = np.array([user_auc(u, i) for u, i in heldout])
cold = aucs[hist_len <= 5]; warm = aucs[hist_len >= 40]
print(f"\nMF ranking AUC by history length (real MovieLens):")
print(f"  COLD users (<=5 positives)  : {cold.mean():.2f}   (n={len(cold)})")
print(f"  WARM users (>=40 positives) : {warm.mean():.2f}   (n={len(warm)})")
print("the model is weakest exactly where it knows the user least; new items are the same story.")

# (3) offline metric vs online truth (a constructed simulation, not from a static file)
nu2, ni2, d2 = 400, 120, 6
Pt = rng.normal(0, 1, (nu2, d2)); Qt = rng.normal(0, 1, (ni2, d2))
true_pop = rng.random(ni2)                                # items' intrinsic appeal
logged = rng.random((nu2, ni2)) < (0.02 + 0.3 * true_pop**3)   # logs biased toward popular items
pop_rank = np.argsort(true_pop)[::-1]
offline = np.mean([logged[u, pop_rank[:10]].mean() for u in range(nu2)])
genuine = Pt @ Qt.T                                        # each user's unbiased true preference
online = np.mean([np.mean([g in np.argsort(genuine[u])[::-1][:10] for g in pop_rank[:10]])
                  for u in range(nu2)])
print(f"\npopularity recommender, precision@10 (constructed simulation):")
print(f"  measured OFFLINE on logged (popularity-biased) data : {offline:.2f}")
print(f"  against users' GENUINE preferences (online proxy)   : {online:.2f}")
in_top10 = np.mean([np.isin(np.arange(ni2), np.argsort(genuine[u])[::-1][:10]) for u in range(nu2)], axis=0)
print(f"what the toy builds in: the logging rate of those ten items averages "
      f"{np.mean(0.02 + 0.3 * true_pop[pop_rank[:10]]**3):.2f};")
print(f"an average item sits in {in_top10.mean():.3f} = 10/{ni2} of users' genuine top 10s, "
      f"these ten popular items in {in_top10[pop_rank[:10]].mean():.2f}")
print("the offline number flatters the popular-items rule because the logs were generated by a")
print("popularity-biased system; an online A/B test measures real user response directly.")

# (4) beyond-accuracy: coverage, intra-list diversity, novelty (MF vs popularity)
counts = np.zeros(ni)
for uu in range(nu):
    for it in train_pos[uu]: counts[it] += 1
p_item = counts / counts.sum()                            # item share of all interactions

def top10(u):
    sc = P[u] @ Q.T
    sc[list(train_pos[u])] = -np.inf                      # never recommend already-seen items
    return np.argsort(sc)[::-1][:10]
pop_order = np.argsort(counts)[::-1]
def top10_pop(u):
    return [i for i in pop_order if i not in train_pos[u]][:10]

Qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)
def ild(lst):                                             # 1 - mean pairwise cosine of the list
    V = Qn[list(lst)]; C = V @ V.T
    return 1 - C[np.triu_indices(len(lst), k=1)].mean()

print("\nbeyond-accuracy metrics on each user's top-10 (200-movie catalog):")
for name, fn in [("MF (BPR)  ", top10), ("popularity", top10_pop)]:
    lists = [fn(u) for u in range(nu)]
    coverage = len(set(i for l in lists for i in l)) / ni
    diversity = np.mean([ild(l) for l in lists])
    novelty = np.mean([-np.log2(p_item[list(l)] + 1e-12).mean() for l in lists])
    print(f"  {name}: coverage {coverage:.2f}   intra-list diversity {diversity:.2f}   novelty {novelty:.2f} bits")

print("\nAll evaluation checks ran.")

# %% [markdown]
# Example 19-17. Inverse propensity scoring on the biased log of Example 19-16.

# %%
# continuing with Example 19-16's simulation: estimate the genuine precision from the biased log itself
rng = np.random.default_rng(1)                            # a fresh seed, so this block's draws are its own
rec = pop_rank[:10]                                        # the ten items the popularity rule shows every user
propensity = 0.02 + 0.3 * true_pop[rec] ** 3               # P(the old system exposed item i): known, the toy built it in
relevant = np.array([np.isin(rec, np.argsort(genuine[u])[::-1][:10]) for u in range(nu2)])  # delta_ui = 1: a shown item in the genuine top 10
def estimates(log_rng):
    exposed = log_rng.random((nu2, 10)) < propensity       # O_ui = 1 when the log holds the pair (user u, item i)
    clicked = exposed & relevant                           # a click is logged only when the item was shown AND is a hit
    return clicked.mean(), (clicked / propensity).mean()   # naive: a missing click is a miss; IPS: divide by propensity
naive, ips = estimates(rng)
print(f"\npopularity recommender, precision@10 estimated from the biased log (same toy):")
print(f"  propensities of the ten shown items     : {propensity.min():.2f} to {propensity.max():.2f}")
print(f"  naive, a missing click counts as a miss : {naive:.3f}")
print(f"  inverse propensity scoring (IPS)        : {ips:.3f}")
print(f"  genuine value, every pair observed      : {relevant.mean():.3f}")
reps = np.array([estimates(rng) for _ in range(200)])      # 200 fresh logs of the same exposure process
print(f"  averaged over 200 fresh logs            : naive {reps[:, 0].mean():.3f}, IPS {reps[:, 1].mean():.3f}")

# %% [markdown]
# ## 19.7 Hands-on lab: a movie recommender, three ways

# %% [markdown]
# Example 19-18. The lab: item-based CF, matrix factorization, and Neural Collaborative Filtering on real MovieLens.

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

# %% [markdown]
# ## 19.8 Interview questions and coding exercises

# %% [markdown]
# ## Coding exercises

# %% [markdown]
# Coding exercise 1: Load MovieLens and build a dense slice.

# %%
import pandas as pd
ratings = pd.read_csv("data/ratings.csv")
def dense_slice(ratings):
    ...
print(ratings.shape)

# %% [markdown]
# Coding exercise 2: User-based Pearson kNN collaborative filtering.

# %%
import pandas as pd
ratings = pd.read_csv("data/ratings.csv")
def knn_mae(ratings):
    ...
print(ratings.userId.nunique())

# %% [markdown]
# Coding exercise 3: Adjusted-cosine item similarity.

# %%
import numpy as np
R = np.array([[5, 4, 5, 1, 1, 2], [4, 5, 4, 2, 1, 1], [5, 5, 4, 1, 2, 1],
              [1, 2, 1, 5, 4, 5], [2, 1, 1, 4, 5, 4]], dtype=float)
def adjusted_cosine(R):
    ...
print(R.shape)

# %% [markdown]
# Coding exercise 4: Biased matrix factorization by ALS.

# %%
import pandas as pd
ratings = pd.read_csv("data/ratings.csv")
def als_mae(ratings):
    ...
print(ratings.rating.min(), ratings.rating.max())

# %% [markdown]
# Coding exercise 5: Implicit-feedback weighted ALS.

# %%
import numpy as np
plays = np.array([0, 3, 0, 12])
def weighted_als_auc(plays):
    ...
print(plays.tolist())

# %% [markdown]
# Coding exercise 6: Content-based profiles and the cold-start win.

# %%
import pandas as pd
movies = pd.read_csv("data/movies.csv")
def content_scores(movies):
    ...
print(movies.shape)

# %% [markdown]
# Coding exercise 7: Ranking metrics from scratch.

# %%
import numpy as np
rel = np.array([0, 1, 1, 0, 0, 1, 0, 0, 0, 1])      # relevance of the items at ranks 1 to 10
def ndcg_at_k(gains, k):
    ...
print(int(rel.sum()))

# %% [markdown]
# Coding exercise 8: Pairwise BPR versus pointwise ranking.

# %%
import pandas as pd
ratings = pd.read_csv("data/ratings.csv")
def ranking_auc(ratings):
    ...
print(int((ratings.rating >= 4).sum()))

# %% [markdown]
# Coding exercise 9: A two-tower retriever with in-batch negatives.

# %%
import pandas as pd
ratings = pd.read_csv("data/ratings.csv")
def recall_at_10(ratings):
    ...
print(ratings.movieId.nunique())

# %% [markdown]
# Coding exercise 10: Miniproject, temporal split versus random split.

# %%
import pandas as pd
ratings = pd.read_csv("data/ratings.csv")
def temporal_rmse(ratings):
    ...
print(int(ratings.timestamp.min()), int(ratings.timestamp.max()))
