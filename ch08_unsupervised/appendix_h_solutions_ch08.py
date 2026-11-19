"""Appendix H solutions for Chapter 8, as runnable cells.

Generated from Solutions.md (Appendix H). Every code block of this chapter's solutions appears here
verbatim, split into one cell per exercise or drill topic, so each answer can be run and checked against
the expected outputs it states. Keep Solutions.md as the source; regenerate after any change.
"""

# %% [markdown]
# ## 8.8 coding exercises

# %%
import warnings
warnings.filterwarnings("ignore")   # hush benign spectral/NMF convergence notices
import numpy as np

# %% [markdown]
# 1. k-means from scratch on wine, stopped by a convergence check

# %%
# 1. k-means from scratch on wine, stopped by a convergence check
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
wine = load_wine()
Xw = StandardScaler().fit_transform(wine.data)                           # standardize the 13 features
yw = wine.target                                                         # cultivars, used only to score
def kmeans_scratch(X, k, tol=1e-6, max_iter=300, seed=0):
    rng = np.random.default_rng(seed)
    centers = X[rng.choice(len(X), k, replace=False)].copy()             # random initial centers
    for it in range(1, max_iter + 1):
        d = np.linalg.norm(X[:, None] - centers[None], axis=2)           # distance to each center
        labels = d.argmin(axis=1)                                        # assignment step
        new = np.array([X[labels == j].mean(0) for j in range(k)])       # update step
        shift = np.linalg.norm(new - centers, axis=1).max()              # farthest move of any center
        centers = new
        if shift < tol:                                                  # converged, so stop early
            break
    inertia = float(((X - centers[labels]) ** 2).sum())
    return labels, inertia, it
runs = [kmeans_scratch(Xw, 3, seed=s) for s in range(5)]
for s, (lab, inertia, it) in enumerate(runs):
    print(s, it, round(inertia, 3), round(adjusted_rand_score(yw, lab), 3))
# expected output: 0 7 1277.928 0.897
# expected output: 1 9 1282.464 0.846
# expected output: 2 7 1278.761 0.915
# expected output: 3 14 1277.928 0.897
# expected output: 4 6 1278.761 0.915
best_lab, best_inertia, _ = min(runs, key=lambda r: r[1])               # keep the lowest-inertia run
sk = KMeans(n_clusters=3, n_init=10, random_state=0).fit(Xw)
print("inertia:", round(best_inertia, 3), round(sk.inertia_, 3))         # expected output: inertia: 1277.928 1277.928
print("ARI:", round(adjusted_rand_score(yw, best_lab), 3),
      round(adjusted_rand_score(yw, sk.labels_), 3))                    # expected output: ARI: 0.897 0.897
# Each run stops at a local minimum of the inertia, and the start decides which one: seeds 1, 2, and 4 stop
# above the 1277.928 that seeds 0 and 3 reach, which is why KMeans restarts (n_init) and keeps the lowest.
# The lowest inertia is not the closest match to the cultivars: seeds 2 and 4 score ARI 0.915 against 0.897,
# because k-means minimizes squared distance and never sees the labels.

# %% [markdown]
# 2. choosing k on wine with three internal metrics and the external ARI

# %%
# 2. choosing k on wine with three internal metrics and the external ARI
from sklearn.datasets import load_wine
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
wine = load_wine()
Xw = StandardScaler().fit_transform(wine.data)
yw = wine.target
for k in range(2, 7):
    l = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(Xw)
    print(k, round(silhouette_score(Xw, l), 3), round(davies_bouldin_score(Xw, l), 3),
          round(calinski_harabasz_score(Xw, l), 0), round(adjusted_rand_score(yw, l), 3))
# expected output: 2 0.268 1.448 69.0 0.389
# expected output: 3 0.285 1.389 71.0 0.897
# silhouette peaks, DB lowest, CH highest, ARI peaks: k=3
# expected output: 4 0.246 1.765 56.0 0.707
# expected output: 5 0.203 1.82 47.0 0.599
# expected output: 6 0.196 1.812 42.0 0.545

# %% [markdown]
# 3. cluster shapes under noise: two moons at three noise levels

# %%
# 3. cluster shapes under noise: two moons at three noise levels
from sklearn.datasets import make_moons
from sklearn.cluster import AgglomerativeClustering, DBSCAN, SpectralClustering
for noise in (0.06, 0.08, 0.10):
    Xm, ym = make_moons(n_samples=400, noise=noise, random_state=0)
    labs = {"kmeans": KMeans(2, n_init=10, random_state=0).fit_predict(Xm),
            "ward": AgglomerativeClustering(n_clusters=2, linkage="ward").fit_predict(Xm),
            "single": AgglomerativeClustering(n_clusters=2, linkage="single").fit_predict(Xm),
            "dbscan": DBSCAN(eps=0.2, min_samples=5).fit_predict(Xm),
            "spectral": SpectralClustering(2, affinity="nearest_neighbors", n_neighbors=10,
                                           random_state=0).fit_predict(Xm)}
    print(noise, " ".join(f"{name} {adjusted_rand_score(ym, lab):.3f}" for name, lab in labs.items()))
    print("   dbscan clusters:", len(set(labs["dbscan"]) - {-1}),
          " single sizes:", np.bincount(labs["single"]).tolist())
# expected output: 0.06 kmeans 0.274 ward 0.239 single 1.000 dbscan 1.000 spectral 1.000
# expected output:    dbscan clusters: 2  single sizes: [200, 200]
# expected output: 0.08 kmeans 0.269 ward 0.482 single 1.000 dbscan 0.000 spectral 0.713
# expected output:    dbscan clusters: 1  single sizes: [200, 200]
# expected output: 0.1 kmeans 0.284 ward 0.428 single 0.000 dbscan 0.000 spectral 0.408
# expected output:    dbscan clusters: 1  single sizes: [399, 1]
# At 0.06 the gap between the crescents is empty, so single linkage, DBSCAN, and the 10-neighbor graph each
# follow a crescent (ARI 1.000), while k-means scores 0.274, as in 8.1.1. As the noise grows, points land in
# the gap and form chains of close neighbors from one crescent to the other. At 0.08 DBSCAN at eps 0.2 joins
# the crescents into one cluster (ARI 0.000) and spectral clustering falls to 0.713. At 0.10 spectral falls to
# 0.408, and single linkage joins the crescents before it absorbs its most isolated point, so the two-cluster
# cut splits off that one point (sizes 399 and 1, ARI 0.000). k-means stays between 0.269 and 0.284 and Ward
# between 0.239 and 0.482, because both cut the plane into two compact groups, and that cut never follows a crescent.

# %% [markdown]
# 4. Gaussian mixtures and BIC on the standardized penguins

# %%
# 4. Gaussian mixtures and BIC on the standardized penguins
import seaborn as sns
from sklearn.mixture import GaussianMixture
pen = sns.load_dataset("penguins").dropna()
feats = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
X = StandardScaler().fit_transform(pen[feats].to_numpy(float))            # standardize the four measurements
ytrue = pen["species"].map({s: i for i, s in enumerate(sorted(pen["species"].unique()))}).to_numpy()
for k in range(1, 6):
    bic = GaussianMixture(n_components=k, covariance_type="full", random_state=0).fit(X).bic(X)
    print("bic", k, round(bic, 0))
# expected output: bic 1 3040.0
# expected output: bic 2 2524.0
# expected output: bic 3 2495.0
# lowest: the three species
# expected output: bic 4 2541.0
# expected output: bic 5 2602.0
for ct in ["spherical", "diag", "tied", "full"]:
    gm = GaussianMixture(n_components=3, covariance_type=ct, random_state=0).fit(X)
    print(ct, round(adjusted_rand_score(ytrue, gm.predict(X)), 3))
# expected output: spherical 0.819
# expected output: diag 0.617
# expected output: tied 0.959
# expected output: full 0.959
print("kmeans", round(adjusted_rand_score(ytrue, KMeans(3, n_init=10, random_state=0).fit_predict(X)), 3))
# expected output: kmeans 0.799
# tied and full (shared or per-cluster full covariance) fit the tilted species clouds best.

# %% [markdown]
# 5. PCA three ways on the digits

# %%
# 5. PCA three ways on the digits
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
Xd, yd = load_digits(return_X_y=True)
Xc = Xd - Xd.mean(0)                                   # center
evals = np.linalg.eigvalsh((Xc.T @ Xc) / len(Xc))[::-1]        # eigenvalue route
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)             # SVD route
sk_evr = PCA().fit(Xd).explained_variance_ratio_             # scikit-learn route
print("eigen ", (evals / evals.sum())[:3].round(4).tolist())  # expected output: eigen  [0.1489, 0.1362, 0.1179]
print("svd   ", (S**2 / (S**2).sum())[:3].round(4).tolist())  # expected output: svd    [0.1489, 0.1362, 0.1179]
print("sklearn", sk_evr[:3].round(4).tolist())                # expected output: sklearn [0.1489, 0.1362, 0.1179]
cum = np.cumsum(sk_evr)
print("comps for 99%:", int(np.argmax(cum >= 0.99)) + 1)      # expected output: comps for 99%: 41
for k in [4, 12, 48]:
    p = PCA(n_components=k).fit(Xd)
    Xrec = p.inverse_transform(p.transform(Xd))               # project down to k, then back up
    print(k, round(np.mean((Xd - Xrec) ** 2), 3),             # reconstruction MSE per pixel
          round(evals[k:].sum() / 64, 3), round(cum[k - 1], 3))   # dropped eigenvalues / 64, variance kept
# expected output: 4 9.628 9.628 0.487
# expected output: 12 4.042 4.042 0.785
# expected output: 48 0.027 0.027 0.999
# At every k the per-pixel error equals the dropped eigenvalues of the 1/n covariance divided by the 64 pixels.

# %% [markdown]
# 6. kernel PCA and the kernel width on concentric circles

# %%
# 6. kernel PCA and the kernel width on concentric circles
from sklearn.datasets import make_circles
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
Xci, yci = make_circles(n_samples=500, factor=0.3, noise=0.05, random_state=0)
for g in [0.1, 0.3, 1, 3, 10, 30, 100]:
    pipe = make_pipeline(KernelPCA(n_components=2, kernel="rbf", gamma=g), LogisticRegression())
    print(g, round(cross_val_score(pipe, Xci, yci, cv=5).mean(), 3))   # each fold fits its own kernel PCA
# expected output: 0.1 0.446
# expected output: 0.3 0.446
# expected output: 1 0.46
# expected output: 3 1.0
# expected output: 10 0.864
# expected output: 30 0.706
# expected output: 100 0.644
lin = PCA(2).fit_transform(Xci)
small = KernelPCA(n_components=2, kernel="rbf", gamma=0.1).fit_transform(Xci)
print([round(float(max(abs(np.corrcoef(small[:, i], lin[:, j])[0, 1]) for j in range(2))), 3) for i in range(2)])
# expected output: [0.999, 0.999]
# Of the seven widths, only gamma = 3 unfolds the rings. For small gamma, exp(-gamma d^2) is close to 1 - gamma d^2, and
# d^2 = |x|^2 + |z|^2 - 2 x.z. Centering the kernel removes the constant and the terms that depend on one point
# only, leaving 2 gamma x.z, the linear kernel: kernel PCA turns into ordinary PCA (the last print shows the
# gamma = 0.1 components match PCA's at correlation 0.999), a rotation, so the score stays at 0.446, the
# raw-feature score that 8.2.2 printed.
# For large gamma every off-diagonal entry exp(-gamma d^2) shrinks toward 0 and the kernel matrix approaches
# the identity, so each point looks unrelated to all but its nearest neighbors, and the two leading components
# no longer carry the contrast between the inner and the outer ring; the score falls to 0.864, 0.706, and 0.644.

# %% [markdown]
# 7. PCA versus LDA on wine (three cultivars)

# %%
# 7. PCA versus LDA on wine (three cultivars)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline
Xwn, ywn = load_wine(return_X_y=True)
for k in [1, 2]:
    pca_s = cross_val_score(
        make_pipeline(StandardScaler(), PCA(n_components=k), LogisticRegression(max_iter=2000)),
        Xwn, ywn, cv=5).mean()
    lda_s = cross_val_score(
        make_pipeline(StandardScaler(), LDA(n_components=k), LogisticRegression(max_iter=2000)),
        Xwn, ywn, cv=5).mean()
    print(k, round(pca_s, 3), round(lda_s, 3))
# expected output: 1 0.849 0.91
# expected output: 2 0.955 0.983
try:
    LDA(n_components=3).fit(StandardScaler().fit_transform(Xwn), ywn)
except ValueError as err:
    print("ValueError:", err)
# expected output: ValueError: n_components cannot be larger than min(n_features, n_classes - 1).
# The scaler and the projection are fit inside each fold, so test-fold labels never shape the LDA axes.
# LDA wins at both k because it uses the labels to choose the directions that maximize the spread between the
# cultivar means relative to the spread within each cultivar; PCA chooses the directions of largest total
# variance, which need not separate the cultivars. The three class means span at most a two-dimensional
# subspace, so LDA can give at most C - 1 = 2 components for 3 classes, and asking for 3 raises the error.

# %% [markdown]
# 8. novelty detection on the digits (train on clean normal, score a held-out mix)

# %%
# 8. novelty detection on the digits (train on clean normal, score a held-out mix)
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score
rng8 = np.random.default_rng(0)
normal_idx = np.where(yd == 1)[0]                             # images of digit 1 are "normal"
anom_idx = rng8.choice(np.where(yd != 1)[0], 60, replace=False)   # a handful of other digits
perm = rng8.permutation(normal_idx)
train, test_norm = perm[:120], perm[120:150]
sc = StandardScaler().fit(Xd[train])                         # scaler learned on clean data only
Xtrain = sc.transform(Xd[train])
Xtest = sc.transform(np.vstack([Xd[test_norm], Xd[anom_idx]]))
ytest = np.r_[np.zeros(len(test_norm)), np.ones(len(anom_idx))].astype(int)   # 1 = novelty (scoring only)
iso = IsolationForest(random_state=0).fit(Xtrain)
ocsvm = OneClassSVM(nu=0.1, gamma="scale").fit(Xtrain)
lof = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(Xtrain)   # novelty=True to score new points
print("iforest", round(roc_auc_score(ytest, -iso.score_samples(Xtest)), 3))     # expected output: iforest 0.968
print("ocsvm  ", round(roc_auc_score(ytest, -ocsvm.decision_function(Xtest)), 3))  # expected output: ocsvm   0.987
print("lof-nov", round(roc_auc_score(ytest, -lof.decision_function(Xtest)), 3))    # expected output: lof-nov 0.972
# For contrast, pour 30 other-digit images (none of them in the test mix) into the training pile and refit.
others = np.setdiff1d(np.where(yd != 1)[0], anom_idx)
mixed = np.r_[train, np.random.default_rng(1).choice(others, 30, replace=False)]
scm = StandardScaler().fit(Xd[mixed])
Xmixed = scm.transform(Xd[mixed])
Xtest_m = scm.transform(np.vstack([Xd[test_norm], Xd[anom_idx]]))
for name, det in [("iforest", IsolationForest(random_state=0)), ("ocsvm  ", OneClassSVM(nu=0.1, gamma="scale")),
                  ("lof-nov", LocalOutlierFactor(n_neighbors=20, novelty=True))]:
    print(name, "mixed", round(roc_auc_score(ytest, -det.fit(Xmixed).decision_function(Xtest_m)), 3))
# expected output: iforest mixed 0.932
# expected output: ocsvm   mixed 0.804
# expected output: lof-nov mixed 0.867
# Every detector loses ground (0.968 to 0.932, 0.987 to 0.804, 0.972 to 0.867). Fit only on images of the digit 1,
# each model's region of normal wraps just those images, so other digits fall outside it. With other digits in the
# training pile the model stretches its region of normal to cover them too, and test anomalies that land in that
# stretched region score as normal, which blurs the boundary.

# %% [markdown]
# 9. association-rule metrics from scratch on ten coffee-shop baskets

# %%
# 9. association-rule metrics from scratch on ten coffee-shop baskets
from scipy.stats import fisher_exact
baskets = [{"coffee", "cake"}, {"coffee", "cake", "milk"}, {"tea", "scone"}, {"tea", "scone", "milk"},
           {"coffee", "milk"}, {"coffee", "cake", "sugar"}, {"tea", "milk", "sugar"},
           {"coffee", "milk", "sugar"}, {"tea", "scone", "cake"}, {"coffee", "cake", "milk"}]
N = len(baskets)
def support(items):   return sum(set(items) <= b for b in baskets) / N   # fraction of baskets with all items
def confidence(A, B): return support(A | B) / support(A)                 # P(B | A)
def lift(A, B):       return confidence(A, B) / support(B)               # vs chance
for A, B in [({"coffee"}, {"cake"}), ({"coffee"}, {"milk"}), ({"tea"}, {"scone"})]:
    print(sorted(A), sorted(B), round(support(A | B), 3), round(confidence(A, B), 3), round(lift(A, B), 3))
# expected output: ['coffee'] ['cake'] 0.4 0.667 1.333
# expected output: ['coffee'] ['milk'] 0.4 0.667 1.111
# expected output: ['tea'] ['scone'] 0.3 0.75 2.5
def table(a, b):                                   # baskets with and without a, by with and without b
    return [[sum(a in x and b in x for x in baskets), sum(a in x and b not in x for x in baskets)],
            [sum(b in x and a not in x for x in baskets), sum(a not in x and b not in x for x in baskets)]]
for a, b in [("coffee", "cake"), ("tea", "scone")]:
    t = table(a, b)
    print(a, b, t, round(fisher_exact(t, alternative="greater").pvalue, 3))
# expected output: coffee cake [[4, 2], [1, 3]] 0.262
# expected output: tea scone [[3, 1], [0, 6]] 0.033
# A lift above 1 on ten baskets describes these ten baskets; it is not a test against chance. The Fisher test
# is one: coffee -> cake (lift 1.333) has p = 0.262, too large to rule out chance, while for tea -> scone
# (lift 2.5) a table at least this lopsided, given its row and column totals, has probability 0.033 if the
# two items are independent. Rank rules by lift, not confidence: both coffee rules have confidence 0.667,
# but milk is in 60 percent of the baskets and cake in 50 percent, so the same confidence is a smaller rise
# over milk's base rate (lift 1.111 against 1.333). Confidence alone rewards a rule whose consequent is common.

# %% [markdown]
# 10. miniproject: topic modeling on four other newsgroups (LDA vs NMF)

# %%
# 10. miniproject: topic modeling on four other newsgroups (LDA vs NMF)
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
cats = ["rec.autos", "sci.med", "comp.sys.mac.hardware", "soc.religion.christian"]
news = fetch_20newsgroups(subset="train", categories=cats, remove=("headers", "footers", "quotes"))
docs = news.data
cv = CountVectorizer(max_df=0.5, min_df=10, stop_words="english")
dtm = cv.fit_transform(docs)
fcv = cv.get_feature_names_out()
print("dtm shape:", dtm.shape)                             # expected output: dtm shape: (2365, 2795)
lda = LatentDirichletAllocation(n_components=4, max_iter=20,
                                learning_method="batch", random_state=0).fit(dtm)
for i, comp in enumerate(lda.components_):
    print("LDA", i, [fcv[j] for j in comp.argsort()[::-1][:6]])
# expected output: LDA 0 ['car', 'like', 'just', 'know', 'don', 'drive']
# expected output: LDA 1 ['don', 'people', 'like', 'know', 'just', 'time']
# expected output: LDA 2 ['god', 'jesus', 'people', 'think', 'church', 'believe']
# expected output: LDA 3 ['edu', 'com', 'mac', 'use', 'health', '1993']
tf = TfidfVectorizer(max_df=0.5, min_df=10, stop_words="english")
Xtf = tf.fit_transform(docs)
ftf = tf.get_feature_names_out()
nmf = NMF(n_components=4, init="nndsvda", max_iter=400, random_state=0).fit(Xtf)
for i, comp in enumerate(nmf.components_):
    print("NMF", i, [ftf[j] for j in comp.argsort()[::-1][:6]])
# expected output: NMF 0 ['mac', 'apple', 'drive', 'thanks', 'does', 'know']
# expected output: NMF 1 ['geb', 'shameful', 'dsl', 'cadre', 'pitt', 'n3jxp']
# expected output: NMF 2 ['god', 'jesus', 'people', 'believe', 'christ', 'faith']
# expected output: NMF 3 ['car', 'like', 'just', 'don', 'good', 'cars']
hits = [i for i, d in enumerate(docs) if "n3jxp" in d.lower()]     # search the posts for a word of NMF 1
print(len(hits), sorted({news.target_names[news.target[i]] for i in hits}))
# expected output: 73 ['sci.med']
# LDA finds cars (0), a filler topic of everyday words (1), and religion (2), and its topic 3 mixes mac and
# health with edu, com, and 1993. NMF finds Mac hardware (0), religion (2), and cars (3), and spends topic 1 on
# one sci.med poster's signature block, which is still in 73 training posts after the filters. The footer
# filter (strip_newsgroup_footer in sklearn/datasets/_twenty_newsgroups.py) cuts a post at its last blank or
# all-hyphen line, and this signature ends with a line of hyphens, so only that line goes; the
# fetch_20newsgroups documentation warns that the footer and quote filters are not always correct
# (scikit-learn developers, 2026d, fetch_20newsgroups).
# Neither method gives sci.med a clean topic of its own, so the match to the four groups is far from one to one.

# %% [markdown]
# 11. ICA versus PCA, then two Gaussian sources

# %%
# 11. ICA versus PCA, then two Gaussian sources
import numpy as np
from sklearn.decomposition import FastICA, PCA
rng = np.random.default_rng(1)
t = np.linspace(0, 8, 2000)
S = np.c_[np.sin(2 * t), np.sign(np.sin(3 * t))]
A = np.array([[1.0, 0.4], [0.3, 1.2]])
def mcorr(R, S):
    return np.mean([max(abs(np.corrcoef(R[:, i], S[:, j])[0, 1]) for i in range(2)) for j in range(2)])
X = S @ A.T
print(round(mcorr(FastICA(n_components=2, random_state=0, whiten="unit-variance").fit_transform(X), S), 3))
# expected output: 0.999
print(round(mcorr(PCA(2).fit_transform(X), S), 3))
# expected output: 0.909
S3 = rng.normal(size=(2000, 2))
X3 = S3 @ A.T
print(round(mcorr(FastICA(n_components=2, random_state=0, whiten="unit-variance").fit_transform(X3), S3), 3))
# expected output: 0.736
print(round(mcorr(PCA(2).fit_transform(X3), S3), 3))
# expected output: 0.819
# On Gaussian sources ICA's edge disappears (0.736 against PCA's 0.819). After whitening, ICA can only choose a
# rotation, and it picks the one that makes the outputs most non-Gaussian; but for independent unit Gaussians s every
# rotation R gives Rs ~ N(0, R R^T) = N(0, I), again independent and Gaussian, so no rotation stands out and the
# sources cannot be identified. This is why ICA needs non-Gaussian sources (8.2.4).

# %% [markdown]
# 12. factor analysis noise on wine

# %%
# 12. factor analysis noise on wine
from sklearn.datasets import load_wine
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
wine = load_wine()
Xw = StandardScaler().fit_transform(wine.data)
fa = FactorAnalysis(n_components=2, random_state=0).fit(Xw)
order = np.argsort(fa.noise_variance_)[::-1][:3]
print([wine.feature_names[i] for i in order])
# expected output: ['ash', 'magnesium', 'alcalinity_of_ash']
print(np.round(fa.noise_variance_[order], 3))
# expected output: [0.894 0.853 0.84 ]

# %% [markdown]
# 13. HDBSCAN versus DBSCAN on the standardized penguins

# %%
# 13. HDBSCAN versus DBSCAN on the standardized penguins
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.metrics import adjusted_rand_score
import seaborn as sns
pen = sns.load_dataset("penguins").dropna()
feats = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
Xp = StandardScaler().fit_transform(pen[feats].to_numpy(float))
yp = pen["species"].to_numpy()                             # species, used only to score
for eps in (0.5, 0.8):
    lab = DBSCAN(eps=eps, min_samples=10).fit_predict(Xp)
    nc = len(set(lab)) - (1 if -1 in lab else 0)
    print(eps, nc, int((lab == -1).sum()), round(adjusted_rand_score(yp, lab), 3))
    for name in sorted(set(yp)):                           # species by DBSCAN label, noise (-1) first
        print(name, [int(((yp == name) & (lab == c)).sum()) for c in sorted(set(lab))])
# expected output: 0.5 4 139 0.379
# expected output: Adelie [73, 73, 0, 0, 0]
# expected output: Chinstrap [39, 3, 10, 16, 0]
# expected output: Gentoo [27, 0, 0, 0, 92]
# expected output: 0.8 2 10 0.632
# expected output: Adelie [3, 143, 0]
# expected output: Chinstrap [4, 64, 0]
# expected output: Gentoo [3, 0, 116]
h = HDBSCAN(min_cluster_size=20, copy=True).fit_predict(Xp)
nc = len(set(h)) - (1 if -1 in h else 0)
print("hdb", nc, int((h == -1).sum()), round(adjusted_rand_score(yp, h), 3))
# expected output: hdb 2 4 0.644
for name in sorted(set(yp)):                               # species by HDBSCAN label: noise, 0, 1
    print(name, [int(((yp == name) & (h == c)).sum()) for c in (-1, 0, 1)])
# expected output: Adelie [1, 145, 0]
# expected output: Chinstrap [2, 66, 0]
# expected output: Gentoo [1, 0, 118]
# A density method splits groups only where a region of low density separates them. Gentoo is set off by one,
# so HDBSCAN and DBSCAN at eps 0.8 both give it a cluster of its own. Adelie and Chinstrap overlap in these four
# measurements (k-means in 8.1.1 put 27 of them in the other species' cluster), and at these settings the points
# between them are dense enough to join them: DBSCAN at eps 0.8, with only 10 birds as noise, puts 143 Adelies
# and 64 Chinstraps in one cluster, and HDBSCAN keeps 145 and 66 together. Each method therefore finds two groups
# where there are three species; the same overlap made the silhouette peak at two clusters in 8.1.1. At eps 0.5
# the radius is too small for the sparser edges: 139 penguins become noise, half the Adelies and 39 of the 68
# Chinstraps among them, and most of the Chinstraps that stay form two small clusters of their own (10 and 16
# birds), so the score falls to 0.379. All three runs score below k-means with k = 3 (0.799 in 8.1.1).

# %% [markdown]
# 14. k-means++ seeding versus random on digits

# %%
# 14. k-means++ seeding versus random on digits
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
Xd = StandardScaler().fit_transform(load_digits().data)
def kpp_inertia(X, k, rng):
    n = len(X)
    centers = [X[int(rng.integers(0, n))]]
    for _ in range(k - 1):
        d2 = np.min(((X[:, None, :] - np.array(centers)[None, :, :]) ** 2).sum(2), axis=1)
        centers.append(X[int(rng.choice(n, p=d2 / d2.sum()))])
    return KMeans(n_clusters=k, init=np.array(centers), n_init=1).fit(X).inertia_
rand_s, pp_s = [], []
for s in range(30):
    rand_s.append(KMeans(n_clusters=10, init="random", n_init=1, random_state=s).fit(Xd).inertia_)
    pp_s.append(kpp_inertia(Xd, 10, np.random.default_rng(s)))
print(round(float(np.mean(rand_s))), round(float(np.max(rand_s))), round(float(np.std(rand_s))))
# expected output: 71053 72884 949
print(round(float(np.mean(pp_s))), round(float(np.max(pp_s))), round(float(np.std(pp_s))))
# expected output: 71491 72901 1000
# No gain here, unlike the twelve blobs of 8.1.1, where k-means++ cut the mean from 1372 to 978 and the worst
# run from 2955 to 1212. The 438 gap between the means (0.6 percent of random seeding's 71053) is less than half
# the run-to-run standard deviation (949 and 1000), and the worst runs differ by 17. No run is stuck far out of
# line: the worst random run is 2.6 percent above the random mean, where on the twelve blobs the worst random run
# (2955) was more than double the mean (1372). Spreading the seeds pays off when a random start can leave two
# seeds in one cluster and none in another, the bad local minimum of 8.1.1; 8.1.1 also notes that k-means++ makes
# no difference on an easy problem, and over these 30 runs on the digits neither seeding falls into such a trap.
# The two comparisons also differ in the seeding rule: 8.1.1 ran scikit-learn's greedy k-means++, which tries
# several candidates at each step and keeps the best (scikit-learn developers, 2026d, KMeans), while this
# exercise implements the plain rule. Careful seeding is no guaranteed gain.

# %% [markdown]
# 15. Ward dendrogram cut on the standardized penguins

# %%
# 15. Ward dendrogram cut on the standardized penguins
from scipy.cluster.hierarchy import fcluster, linkage
Z = linkage(Xp, method="ward")                             # the penguins of exercise 13
print(np.round(Z[-5:, 2], 1).tolist())
# expected output: [6.4, 12.1, 12.1, 18.4, 39.5]
lab_a = fcluster(Z, t=3, criterion="maxclust")
heights = Z[-5:, 2]
gaps = np.diff(heights)
cut = (heights[int(np.argmax(gaps))] + heights[int(np.argmax(gaps)) + 1]) / 2
lab_b = fcluster(Z, t=cut, criterion="distance")
print(round(float(cut), 2), len(np.unique(lab_a)), len(np.unique(lab_b)), int(np.array_equal(lab_a, lab_b)))
# expected output: 28.94 3 2 0
print(round(adjusted_rand_score(yp, lab_a), 3), round(adjusted_rand_score(yp, lab_b), 3))
# expected output: 0.913 0.649
for name in sorted(set(yp)):                               # species by cluster of the height cut
    print(name, [int(((yp == name) & (lab_b == c)).sum()) for c in (1, 2)])
# expected output: Adelie [0, 146]
# expected output: Chinstrap [0, 68]
# expected output: Gentoo [119, 0]
# The largest gap among the last five heights runs from 18.4 to 39.5, just below the final merge, so the
# height cut keeps two clusters: Gentoo, and Adelie with Chinstrap. It scores 0.649 against the 0.913 of
# the three-cluster cut. The tall-gap rule of 8.1.2 picks the strongest split in the features, and here that
# is Gentoo against the rest; the Adelie and Chinstrap split is weaker, though the three-cluster cut that
# makes it matches the species far better. Like the silhouette peak at two in 8.1.1, the gap measures
# separation in the features, not the number of species.

# %% [markdown]
# 16. mean shift on penguins

# %%
# 16. mean shift on penguins
from sklearn.cluster import MeanShift, estimate_bandwidth
pen = sns.load_dataset("penguins").dropna()
feats = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
Xp = StandardScaler().fit_transform(pen[feats].to_numpy(float))
yp = pen["species"].to_numpy()
for q in (0.1, 0.2, 0.3):
    bw = estimate_bandwidth(Xp, quantile=q, random_state=0)
    ms = MeanShift(bandwidth=bw, bin_seeding=True).fit(Xp)
    print(q, round(float(bw), 3), len(np.unique(ms.labels_)), round(adjusted_rand_score(yp, ms.labels_), 3))
# expected output: 0.1 0.991 4 0.817
# expected output: 0.2 1.369 4 0.804
# expected output: 0.3 1.768 2 0.639

# %% [markdown]
# ## 8.8 coding drills

# %%
import numpy as np

# %% [markdown]
# Clustering

# %%
# Clustering
from sklearn.datasets import load_iris, make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score
Xi, yi = load_iris(return_X_y=True)
Xi = StandardScaler().fit_transform(Xi)
km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(Xi)
print(round(km.inertia_, 1))                                   # expected output: 139.8
print(round(silhouette_score(Xi, km.labels_), 3))             # expected output: 0.46
Xb, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=0)
noise = np.random.default_rng(0).uniform(-10, 10, (30, 2))
db = DBSCAN(eps=0.5, min_samples=5).fit_predict(np.vstack([Xb, noise]))
print(int((db == -1).sum()))                                   # expected output: 41
ag = AgglomerativeClustering(n_clusters=3, linkage="ward").fit_predict(Xi)
print(round(adjusted_rand_score(yi, ag), 3))                   # expected output: 0.615
print(round(adjusted_rand_score(yi, km.labels_), 3))          # expected output: 0.62
gm = GaussianMixture(n_components=3, random_state=0).fit(Xi)
print(gm.predict_proba(Xi[:1]).round(3).tolist())            # expected output: [[0.0, 1.0, 0.0]]

# %% [markdown]
# Linear dimensionality reduction

# %%
# Linear dimensionality reduction
from sklearn.datasets import load_digits, load_breast_cancer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
Xd, yd = load_digits(return_X_y=True)
Xbc = StandardScaler().fit_transform(load_breast_cancer().data)   # 569 tumors, 30 measurements
p = PCA().fit(Xbc)
print(p.explained_variance_ratio_[:3].round(3).tolist())      # expected output: [0.443, 0.19, 0.094]
print(int(np.argmax(np.cumsum(p.explained_variance_ratio_) >= 0.90)) + 1)   # expected output: 7
Zw = PCA(n_components=5, whiten=True, random_state=0).fit_transform(Xd)
print(Zw.var(0).round(2).tolist())                             # expected output: [1.0, 1.0, 1.0, 1.0, 1.0]
print(TruncatedSVD(n_components=4, random_state=0).fit_transform(Xd).shape)   # expected output: (1797, 4)
print(LDA().fit(Xd, yd).transform(Xd).shape[1])               # expected output: 9

# %% [markdown]
# Manifold learning

# %%
# Manifold learning
from sklearn.manifold import TSNE, trustworthiness, Isomap, MDS
rng = np.random.default_rng(0)
idx = rng.choice(len(Xd), 400, replace=False)
Xs = Xd[idx]
emb = TSNE(n_components=2, perplexity=30, random_state=0).fit_transform(Xs)
print(emb.shape)                                               # expected output: (400, 2)
print(round(trustworthiness(Xs, emb, n_neighbors=10), 3))      # expected output: 0.985
iso = Isomap(n_neighbors=10, n_components=2).fit_transform(Xs)
print(round(trustworthiness(Xs, iso, n_neighbors=10), 3))      # expected output: 0.837
mds = MDS(n_components=2, init="random", random_state=0, normalized_stress="auto").fit_transform(Xs)
print(round(trustworthiness(Xs, mds, n_neighbors=10), 3))      # expected output: 0.89
print(round(trustworthiness(Xs, PCA(2).fit_transform(Xs), n_neighbors=10), 3))  # expected output: 0.817

# %% [markdown]
# Anomaly detection

# %%
# Anomaly detection
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
Xa, _ = make_blobs(n_samples=200, centers=1, cluster_std=1.0, random_state=0)
Xa = np.vstack([Xa, rng.uniform(-8, 8, (20, 2))])
print(int((IsolationForest(contamination=0.1, random_state=0).fit_predict(Xa) == -1).sum()))  # expected output: 22
print(int((LocalOutlierFactor(n_neighbors=20, contamination=0.1).fit_predict(Xa) == -1).sum()))  # expected output: 22
print(int((EllipticEnvelope(contamination=0.1, random_state=0).fit_predict(Xa) == -1).sum()))    # expected output: 22
x = np.array([2., 4., 3., 5., 100., 4., 3.])
print(int((np.abs((x - x.mean()) / x.std()) > 2).sum()))       # expected output: 1

# %% [markdown]
# Association rules and topic modeling

# %%
# Association rules and topic modeling
baskets = [{"a", "b"}, {"a", "b", "c"}, {"a", "c"}, {"b", "c"}, {"a", "b"}]
N = len(baskets)
sup = lambda s: sum(set(s) <= b for b in baskets) / N
print(round(sup({"a", "b"}), 2))                               # expected output: 0.6
print(round(sup({"a", "b"}) / sup({"a"}), 3))                 # expected output: 0.75
print(round((sup({"a", "b"}) / sup({"a"})) / sup({"b"}), 3))  # expected output: 0.937
# The exact lift is 0.75 / 0.8 = 15/16 = 0.9375, which rounds to 0.938, but Python prints 0.937: 0.6 / 0.8 evaluates
# to 0.7499999999999999 in binary floating point, so the computed lift is 0.9374999999999998.
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import NMF
corpus = ["cat sat mat", "dog ran park", "cat dog play", "park bench sat"]
dtm = CountVectorizer().fit_transform(corpus)
print(dtm.shape)                                               # expected output: (4, 8)
nmf = NMF(n_components=2, init="nndsvda", random_state=0).fit(dtm.toarray().astype(float))
print(bool((nmf.components_ >= 0).all()))                      # expected output: True
