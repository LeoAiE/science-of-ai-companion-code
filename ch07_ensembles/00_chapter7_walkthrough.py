"""Chapter 7: Ensemble Methods: From Random Forests to XGBoost

Section-aligned walkthrough: every code block from this chapter of the book,
in book order, each in its own cell under its section heading so you can run
and check them one by one. Generated from Book.md; keep the book as the source.
"""

# %% [markdown]
# ## 7.1.1 The wisdom of crowds

# %% [markdown]
# If the voters are correlated, all tending to make the same mistakes, the crowd barely improves on one member, as a simulation of 101 voters confirms.

# %%
import numpy as np
from scipy.stats import binom
p = 0.60                                            # each weak voter is 60% accurate
print("voters  majority accuracy")
for M in [1, 5, 15, 51, 101, 501]:
    print(M, round(1 - binom.cdf(M // 2, M, p), 3))   # P(a majority of M is correct)
# 1->0.600  5->0.683  15->0.787  51->0.926  101->0.979  501->1.000
rng = np.random.default_rng(0)
def crowd(M, p, shared, trials=20000):              # `shared`: fraction of fully correlated votes
    c = 0
    for _ in range(trials):
        v = np.repeat(rng.random() < p, M) if rng.random() < shared else rng.random(M) < p
        c += v.sum() > M / 2
    return c / trials
print(round(crowd(101, 0.6, 0.0), 3), round(crowd(101, 0.6, 0.8), 3))
# independent 0.978   correlated 0.677  (correlation destroys the wisdom-of-crowds gain)

# %% [markdown]
# ## 7.2.1 Bootstrap sampling and bagging

# %% [markdown]
# The probability that a particular point is missed in all draws is , which approaches as grows, so each bootstrap sample contains about 63 percent of the distinct points and leaves about 37 percent out.

# %%
import numpy as np
rng = np.random.default_rng(0)
n = 2000
frac = np.mean([len(np.unique(rng.integers(0, n, n))) / n for _ in range(50)])
print(round(frac, 3), round(1 - 1 / np.e, 3))   # 0.631 distinct  vs theory 1 - 1/e = 0.632

# %% [markdown]
# Pasting, random subspaces, and random patches against bagging on the breast-cancer data, 30 deep trees each.

# %%
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
X, y = load_breast_cancer(return_X_y=True)
specs = {
    "bag": dict(max_samples=1.0, bootstrap=True, max_features=1.0, bootstrap_features=False),
    "paste": dict(max_samples=0.7, bootstrap=False, max_features=1.0, bootstrap_features=False),
    "subspace": dict(max_samples=1.0, bootstrap=False, max_features=0.5, bootstrap_features=False),
    "patches": dict(max_samples=0.7, bootstrap=True, max_features=0.5, bootstrap_features=True),
}
for name, kw in specs.items():
    acc = cross_val_score(
        BaggingClassifier(DecisionTreeClassifier(), n_estimators=30, random_state=0, n_jobs=2, **kw),
        X, y, cv=5,
    ).mean()
    print(name, round(float(acc), 3))
# bag 0.956  paste 0.953  subspace 0.963  patches 0.965

# %% [markdown]
# ## 7.2.2 Random forests: decorrelating the trees

# %% [markdown]
# The progression from one tree to bagging to a forest shows each step working.

# %%
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
X, y = load_breast_cancer(return_X_y=True)                  # 569 tumors, 30 features, 2 classes
for name, m in [("single tree", DecisionTreeClassifier(random_state=0)),
                ("bagging", BaggingClassifier(DecisionTreeClassifier(), n_estimators=100,
                                              random_state=0, n_jobs=2)),
                ("forest", RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=2))]:
    print(name, round(cross_val_score(m, X, y, cv=5).mean(), 3))
# single tree 0.917   bagging 0.958 (averaging cuts variance)   forest 0.963 (decorrelation cuts more)

# %% [markdown]
# averaging each ensemble's trees' predictions and computing their pairwise correlation shows the forest's trees are genuinely more diverse.

# %%
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
def avg_tree_corr(ens, Xd):                         # mean correlation between the trees' predictions
    preds = np.array([est.predict(Xd) for est in ens.estimators_])
    C = np.corrcoef(preds)
    return np.nanmean(C[np.triu_indices_from(C, k=1)])
bag = BaggingClassifier(DecisionTreeClassifier(), n_estimators=100, random_state=0, n_jobs=2).fit(X_tr, y_tr)
rf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=2).fit(X_tr, y_tr)
print(round(avg_tree_corr(bag, X_te), 3), round(avg_tree_corr(rf, X_te), 3))
# bagging 0.825   forest 0.794  (the random feature subset makes the forest's trees more diverse)

# %% [markdown]
# It is faster, because the expensive threshold search is gone, and it decorrelates the trees even more, pushing the variance floor of 7.1 lower still, at the price of a little more bias per tree.

# %%
import time
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
X, y = load_breast_cancer(return_X_y=True)          # same 569 tumors, 30 features
def timed(model):                                   # 5-fold CV accuracy and wall-clock seconds
    t = time.perf_counter()
    acc = cross_val_score(model, X, y, cv=5).mean()
    return round(acc, 3), round(time.perf_counter() - t, 2)
rf_acc, rf_t = timed(RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=2))
et_acc, et_t = timed(ExtraTreesClassifier(n_estimators=100, random_state=0, n_jobs=2))
print("random forest", rf_acc, rf_t)                # 0.963 0.39 (searches for the best threshold)
print("extra trees  ", et_acc, et_t)                # 0.97 0.29 (splits at random, so faster)

# %% [markdown]
# Extra Trees fit each tree on the full training set unless bootstrap=True, so an out-of-bag score is not available by default.

# %%
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import ExtraTreesClassifier
X, y = load_breast_cancer(return_X_y=True)
print(ExtraTreesClassifier().get_params()["bootstrap"])
try:
    ExtraTreesClassifier(n_estimators=5, oob_score=True, random_state=0, n_jobs=2).fit(X, y)
except ValueError as err:
    print(type(err).__name__, err)
# False
# ValueError Out of bag estimation only available if bootstrap=True

# %% [markdown]
# ## 7.2.3 Out-of-bag error: validation for free

# %% [markdown]
# It is essentially cross-validation at no extra cost, computed as a side effect of fitting.

# %%
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
rf_oob = RandomForestClassifier(n_estimators=100, oob_score=True,
                                random_state=0, n_jobs=2).fit(X_tr, y_tr)
print(round(rf_oob.oob_score_, 3), round(rf_oob.score(X_te, y_te), 3))
# 0.970 out-of-bag estimate (free, from the ~37% left out)  vs  0.953 held-out test accuracy

# %% [markdown]
# The same estimate works for a regression forest, where oob_score_ reports the out-of-bag R^2: max_features 1.0 against 1/3 on California housing.

# %%
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
X, y = fetch_california_housing(return_X_y=True)      # 20,640 block groups, 8 features
rng = np.random.default_rng(0)
idx = rng.choice(len(y), 6000, replace=False)         # a 6,000-row subsample keeps the fit quick
for name, mf in [("1.0", 1.0), ("1/3", 1 / 3)]:
    rf = RandomForestRegressor(n_estimators=80, max_features=mf, oob_score=True,
                               random_state=0, n_jobs=2).fit(X[idx], y[idx])
    print(name, round(float(rf.oob_score_), 4))       # out-of-bag R^2
# 1.0 0.7717
# 1/3 0.7823

# %% [markdown]
# ## 7.2.4 Feature importance and its pitfalls

# %% [markdown]
# Appending two columns of pure noise, one a high-cardinality integer id and one a binary flag, exposes the flaw.

# %%
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
rng = np.random.default_rng(0)
# two pure-noise columns: a high-cardinality integer id and a binary flag
hi_tr = rng.integers(0, 100000, size=len(X_tr)); hi_te = rng.integers(0, 100000, size=len(X_te))
bn_tr = rng.integers(0, 2, size=len(X_tr));      bn_te = rng.integers(0, 2, size=len(X_te))
Xn_tr = np.column_stack([X_tr, hi_tr, bn_tr])                 # col -2 high-card noise, -1 binary noise
Xn_te = np.column_stack([X_te, hi_te, bn_te])
rf2 = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=2).fit(Xn_tr, y_tr)
perm = permutation_importance(rf2, Xn_te, y_te, n_repeats=5, random_state=0, n_jobs=2)
print(round(rf2.feature_importances_[-2], 4), round(perm.importances_mean[-2], 4))  # high-card noise
print(round(rf2.feature_importances_[-1], 4), round(perm.importances_mean[-1], 4))  # binary noise
# high-card: impurity 0.0020 (falsely nonzero)   permutation 0.0000 (~0, correct)
# binary:    impurity 0.0002                      permutation 0.0000

# %% [markdown]
# ## 7.3.1 AdaBoost: reweighting the mistakes

# %% [markdown]
# To watch weak stumps turn into a strong classifier, take the breast cancer diagnosis data of Chapter 6, 569 tumors with 30 measurements each, each labeled benign or malignant.

# %%
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
X, y = load_breast_cancer(return_X_y=True)   # 569 patients, 30 tumor measurements, benign vs malignant
for n in [1, 5, 50, 200]:
    acc = cross_val_score(AdaBoostClassifier(n_estimators=n, random_state=0), X, y, cv=5).mean()
    print(n, round(acc, 3))
# 1 stump 0.900   5 -> 0.933   50 -> 0.967   200 -> 0.977

# %% [markdown]
# Example 7-1 boosts fifty stumps on a held-out split of the same data, printing the first round's numbers so they can be checked against the reweighting arithmetic.

# %%
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
X, y01 = load_breast_cancer(return_X_y=True)
y = 2 * y01 - 1                                     # AdaBoost's math wants labels in {-1, +1}
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
w = np.full(len(y_tr), 1 / len(y_tr))               # start with equal weights on every point
F_tr, F_te = np.zeros(len(y_tr)), np.zeros(len(y_te))
for m in range(50):
    stump = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X_tr, y_tr, sample_weight=w)
    pred = stump.predict(X_tr)
    eps = w[pred != y_tr].sum() / w.sum()           # weighted error rate of this stump
    alpha = 0.5 * np.log((1 - eps) / eps)           # its vote
    w = w * np.exp(-alpha * y_tr * pred)            # multiply up the mistakes, down the rest
    w = w / w.sum()
    F_tr += alpha * pred                            # the running weighted vote
    F_te += alpha * stump.predict(X_te)
    if m == 0:
        print("round 1: eps", round(eps, 3), "alpha", round(alpha, 3),
              "weight now on the mistakes", round(w[pred != y_tr].sum(), 3))
print("scratch train", round((np.sign(F_tr) == y_tr).mean(), 3),
      "test", round((np.sign(F_te) == y_te).mean(), 3))
sk = AdaBoostClassifier(n_estimators=50, random_state=0).fit(X_tr, y_tr)
print("sklearn test", round(sk.score(X_te, y_te), 3))

# %% [markdown]
# ## 7.3.2 Classification leaves

# %%
import math
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor
X, y = load_breast_cancer(return_X_y=True)
F0 = math.log(float(y.mean()) / (1 - float(y.mean())))
p = np.full(len(y), 1 / (1 + np.exp(-F0)))
resid = y - p
stump = DecisionTreeRegressor(max_depth=1, random_state=0).fit(X, resid)
leaves = stump.apply(X)
for leaf in np.unique(leaves):
    m = leaves == leaf
    newton = float(resid[m].sum() / (p[m] * (1 - p[m])).sum())
    print(int(leaf), round(newton, 4), round(float(resid[m].mean()), 4))
# 1  1.2214  0.2855
# 2 -2.4363 -0.5695
F = np.full(len(y), F0)
for _ in range(20):
    p = 1 / (1 + np.exp(-F))
    resid = y - p
    tree = DecisionTreeRegressor(max_depth=1, random_state=0).fit(X, resid)
    leaves = tree.apply(X)
    step = np.zeros(len(y))
    for leaf in np.unique(leaves):
        m = leaves == leaf
        step[m] = resid[m].sum() / max(float((p[m] * (1 - p[m])).sum()), 1e-12)
    F = F + 0.1 * step
lib = GradientBoostingClassifier(n_estimators=20, max_depth=1, learning_rate=0.1, random_state=0).fit(X, y)
print(round(float(np.max(np.abs(F - lib.decision_function(X)))), 6))  # 0.0

# %%
import math
from sklearn.datasets import load_wine
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
X, y = load_wine(return_X_y=True)
gb = GradientBoostingClassifier(n_estimators=10, random_state=0).fit(X, y)
xgb = XGBClassifier(n_estimators=10, n_jobs=2, random_state=0).fit(X, y)
print(tuple(gb.estimators_.shape), len(xgb.get_booster().get_dump()))
print(round(math.log(2), 3))
# (10, 3) 30
# 0.693

# %% [markdown]
# ## 7.3.3 Building it from scratch

# %% [markdown]
# we want to watch the method recover a function whose true shape we already know, which a real dataset would hide.

# %%
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
rng = np.random.default_rng(0)
X = np.sort(rng.uniform(-4, 4, 400)).reshape(-1, 1)
y = np.sin(X).ravel() + 0.3 * X.ravel() + rng.normal(0, 0.2, 400)
def gradient_boost(X, y, n_trees=60, lr=0.1, max_depth=2):
    F = np.full(len(y), y.mean()); trees, init = [], y.mean()   # start at the mean prediction
    for m in range(n_trees):
        residual = y - F                                        # negative gradient of squared loss
        h = DecisionTreeRegressor(max_depth=max_depth, random_state=0).fit(X, residual)
        F += lr * h.predict(X); trees.append(h)                 # a shrunken step downhill
        if (m + 1) in (1, 5, 20, 60):
            print(m + 1, round(mean_squared_error(y, F), 4))
    return trees, init
trees, init = gradient_boost(X, y)
# train MSE: 1 tree -> 0.9726   5 -> 0.4886   20 -> 0.0821   60 -> 0.0322

# %% [markdown]
# Better still, the trees the loop just grew reproduce scikit-learn exactly; replaying them by hand as an additive model gives the same predictions the library computes.

# %%
from sklearn.ensemble import GradientBoostingRegressor   # continuing with X, y, trees, and the init in Example 7-3
def gb_predict(X, trees, init, lr=0.1):                  # replay the additive model by hand
    return init + lr * sum(t.predict(X) for t in trees)
sk = GradientBoostingRegressor(n_estimators=60, learning_rate=0.1, max_depth=2, random_state=0).fit(X, y)
print(round(mean_squared_error(y, gb_predict(X, trees, init)), 4))   # 0.0322  from scratch
print(round(mean_squared_error(y, sk.predict(X)), 4))                # 0.0322  scikit-learn (identical)

# %% [markdown]
# ## 7.3.4 The learning rate and shrinkage

# %% [markdown]
# The two must be tuned together.

# %%
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
X, y = load_breast_cancer(return_X_y=True)   # 569 patients, 30 features, benign vs malignant
for lr, n in [(1.0, 20), (0.1, 20), (0.1, 200), (0.05, 200)]:
    scores = cross_val_score(GradientBoostingClassifier(learning_rate=lr, n_estimators=n, max_depth=3,
                             random_state=0), X, y, cv=5)
    print(lr, n, round(float(scores.mean()), 3), round(float(scores.std()), 3))
# 1.0 20 0.949 0.020     0.1 20 0.951 0.016
# 0.1 200 0.963 0.021    0.05 200 0.960 0.021

# %% [markdown]
# ## 7.3.5 Stochastic gradient boosting

# %% [markdown]
# On the breast cancer data, sweeping the subsample fraction while holding everything else fixed shows the effect directly.

# %%
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
X, y = load_breast_cancer(return_X_y=True)   # 569 patients, 30 features, benign vs malignant
for sub in [1.0, 0.8, 0.5, 0.3]:
    scores = cross_val_score(GradientBoostingClassifier(subsample=sub, n_estimators=200,
                             learning_rate=0.1, max_depth=3, random_state=0), X, y, cv=5)
    print(sub, round(float(scores.mean()), 3), round(float(scores.std()), 3))
# 1.0 0.963 0.021   0.8 0.965 0.018   0.5 0.968 0.015   0.3 0.972 0.01

# %% [markdown]
# ## 7.3.6 How many trees, and when the curve turns up

# %%
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
X, y = load_diabetes(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
rf = RandomForestRegressor(n_estimators=10, max_features=1/3, random_state=0, n_jobs=2, warm_start=True)
for n in (10, 50, 200):
    rf.set_params(n_estimators=n)
    rf.fit(Xtr, ytr)
    print("forest", n, round(float(mean_squared_error(yte, rf.predict(Xte))), 1))
# forest 10 3611.2    forest 50 3524.2    forest 200 3384.8
gb = GradientBoostingRegressor(n_estimators=30, learning_rate=0.3, max_depth=3, random_state=0).fit(Xtr, ytr)
staged = list(gb.staged_predict(Xte))
for n in (6, 15, 30):
    print("boost", n, round(float(mean_squared_error(yte, staged[n - 1])), 1))
# boost 6 3226.2    boost 15 3507.3    boost 30 3804.3
early = GradientBoostingRegressor(
    n_estimators=200, learning_rate=0.3, max_depth=3,
    n_iter_no_change=10, validation_fraction=0.2, random_state=0,
).fit(Xtr, ytr)
print("early", early.n_estimators_)  # 15

# %% [markdown]
# ## 7.4.2 Fast and regularized in practice

# %% [markdown]
# On a 10,000-row subsample of the adult census income data, its categorical columns ordinal-encoded, XGBoost comes within 0.002 of scikit-learn's gradient boosting on accuracy, beats the random forest, and trains several times faster than scikit-learn's boosting.

# %%
import time, numpy as np, pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier
adult = fetch_openml("adult", version=2, as_frame=True)          # 48,842 census rows, predict income >50K
df = adult.frame.sample(10000, random_state=0)                   # subsample for a quick, repeatable race
y = (df["class"] == ">50K").astype(int).to_numpy()
Xdf = df.drop(columns="class")
cat = Xdf.select_dtypes("category").columns                      # ordinal-encode the categorical columns
Xdf[cat] = OrdinalEncoder(encoded_missing_value=-1).fit_transform(Xdf[cat])
X = Xdf.to_numpy(dtype=float)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
def evaluate(name, model):
    t0 = time.perf_counter(); model.fit(X_tr, y_tr); dt = time.perf_counter() - t0
    auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
    print(name, round(accuracy_score(y_te, model.predict(X_te)), 3), round(auc, 3), round(dt, 2))
evaluate("RandomForest", RandomForestClassifier(n_estimators=300, n_jobs=2, random_state=0))            # 0.85 0.897 0.66
evaluate("sklearn GB", GradientBoostingClassifier(n_estimators=300, max_depth=3, random_state=0))        # 0.868 0.918 1.74
evaluate("XGBoost", XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=4, subsample=0.8,
         colsample_bytree=0.8, reg_lambda=1.0, eval_metric="logloss", n_jobs=2, random_state=0))         # 0.866 0.915 0.15
# seconds are one run on the reference machine; XGBoost fits about 12 times faster than sklearn GB

# %% [markdown]
# The built-in regularization is not decoration; turning up and visibly closes the gap between training and test accuracy that signals overfitting.

# %%
import numpy as np, pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
adult = fetch_openml("adult", version=2, as_frame=True)
df = adult.frame.sample(10000, random_state=0)                   # same subsample as the 7.4.2 race
y = (df["class"] == ">50K").astype(int).to_numpy()
Xdf = df.drop(columns="class"); cat = Xdf.select_dtypes("category").columns
Xdf[cat] = OrdinalEncoder(encoded_missing_value=-1).fit_transform(Xdf[cat])
X = Xdf.to_numpy(dtype=float)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
print("reg_lambda gamma  train  test   gap")
for lam, gam in [(0.0, 0.0), (1.0, 0.0), (10.0, 1.0), (50.0, 5.0)]:
    m = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6, reg_lambda=lam, gamma=gam,
                      eval_metric="logloss", n_jobs=2, random_state=0).fit(X_tr, y_tr)
    tr, te = accuracy_score(y_tr, m.predict(X_tr)), accuracy_score(y_te, m.predict(X_te))
    print(lam, gam, round(tr, 3), round(te, 3), round(tr - te, 3))
# (0,0):  0.940/0.859 gap 0.081          (1,0):  0.925/0.865 gap 0.060 (the default lambda)
# (10,1): 0.882/0.869 gap 0.013 (best)   (50,5): 0.862/0.858 gap 0.004 (starts to underfit)

# %% [markdown]
# ## 7.4.3 Early stopping and missing values

# %% [markdown]
# Early stopping watches the validation log loss, the `eval_metric` set on the model, and halts training when that score stops improving, so you can set a large number of trees and let the algorithm choose how many to actually use. The metric can be changed, for example to `auc`.

# %%
import numpy as np, pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
adult = fetch_openml("adult", version=2, as_frame=True)
df = adult.frame.sample(10000, random_state=0)
y = (df["class"] == ">50K").astype(int).to_numpy()
Xdf = df.drop(columns="class"); cat = Xdf.select_dtypes("category").columns
Xdf[cat] = OrdinalEncoder(encoded_missing_value=-1).fit_transform(Xdf[cat])
X = Xdf.to_numpy(dtype=float)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
X_fit, X_val, y_fit, y_val = train_test_split(X_tr, y_tr, test_size=0.2, random_state=1, stratify=y_tr)
es = XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=4, subsample=0.8,
                   colsample_bytree=0.8, eval_metric="logloss", n_jobs=2,
                   early_stopping_rounds=30, random_state=0)
es.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)   # stop when val log loss stalls
print(es.best_iteration + 1, es.get_booster().num_boosted_rounds(),
      list(es.evals_result()["validation_0"]),
      round(accuracy_score(y_te, es.predict(X_te)), 3))
# 275  305  ['logloss']  0.868

# %% [markdown]
# XGBoost learns, at each split, a default direction to send rows whose split feature is missing, so it trains directly on incomplete data without any imputation.

# %%
import numpy as np, pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
adult = fetch_openml("adult", version=2, as_frame=True)
df = adult.frame.sample(10000, random_state=0)
y = (df["class"] == ">50K").astype(int).to_numpy()
Xdf = df.drop(columns="class"); cat = Xdf.select_dtypes("category").columns
# adult already carries genuine gaps: workclass, occupation, native-country hold '?' -> NaN
Xdf[cat] = OrdinalEncoder(encoded_missing_value=np.nan).fit_transform(Xdf[cat])
X = Xdf.to_numpy(dtype=float)
print(int(np.isnan(X).sum()), "genuinely missing cells")                            # 1361, no imputation
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
xgb = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=4, eval_metric="logloss",
                    n_jobs=2, random_state=0).fit(X_tr, y_tr)                        # learns a default NaN branch
print(round(accuracy_score(y_te, xgb.predict(X_te)), 3))   # 0.865

# %% [markdown]
# ## 7.4.4 Split finding, subsampling, and the knobs that matter

# %% [markdown]
# On the same 10,000-row adult subsample, forcing education-num to be nondecreasing leaves ROC AUC unchanged at three decimals.

# %%
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier
adult = fetch_openml("adult", version=2, as_frame=True)
df = adult.frame.sample(10000, random_state=0)
y = (df["class"] == ">50K").astype(int).to_numpy()
Xdf = df.drop(columns="class")
cat = Xdf.select_dtypes("category").columns
Xdf[cat] = OrdinalEncoder(encoded_missing_value=-1).fit_transform(Xdf[cat])
names = list(Xdf.columns)
edu = names.index("education-num")
X = Xdf.to_numpy(dtype=float)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
mono = [0] * X.shape[1]
mono[edu] = 1
base = dict(n_estimators=120, learning_rate=0.1, max_depth=4, n_jobs=2,
            eval_metric="logloss", random_state=0)
free = XGBClassifier(**base).fit(X_tr, y_tr)
held = XGBClassifier(monotone_constraints=tuple(mono), **base).fit(X_tr, y_tr)
print(round(float(roc_auc_score(y_te, free.predict_proba(X_te)[:, 1])), 3),
      round(float(roc_auc_score(y_te, held.predict_proba(X_te)[:, 1])), 3))
# 0.916 free  0.916 with education-num forced nondecreasing
grid = np.unique(np.round(np.quantile(X_tr[:, edu], np.linspace(0, 1, 8)), 5))
means = []
for v in grid:
    Xv = X_tr.copy(); Xv[:, edu] = v
    means.append(float(held.predict_proba(Xv)[:, 1].mean()))
print([round(m, 4) for m in means])
# [0.1221, 0.1916, 0.237, 0.2548, 0.3161, 0.4044], each larger than the last

# %% [markdown]
# The effect on the adult subsample of 7.4.2 is easy to show.

# %%
import numpy as np, pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, roc_auc_score
from scipy.stats import spearmanr, kendalltau
from xgboost import XGBClassifier
adult = fetch_openml("adult", version=2, as_frame=True)
df = adult.frame.sample(10000, random_state=0)                   # same subsample as 7.4.2
y = (df["class"] == ">50K").astype(int).to_numpy()
Xdf = df.drop(columns="class"); cat = Xdf.select_dtypes("category").columns
Xdf[cat] = OrdinalEncoder(encoded_missing_value=-1).fit_transform(Xdf[cat])
X = Xdf.to_numpy(dtype=float)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
spw = (y_tr == 0).sum() / (y_tr == 1).sum()                      # negatives over positives = 3.15
probs = []
for w in [1.0, spw]:
    m = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=4, scale_pos_weight=w,
                      eval_metric="logloss", n_jobs=2, random_state=0).fit(X_tr, y_tr)
    probs.append(m.predict_proba(X_te)[:, 1])
    print(round(w, 2), round(recall_score(y_te, m.predict(X_te)), 3),
          round(precision_score(y_te, m.predict(X_te)), 3),
          round(roc_auc_score(y_te, probs[-1]), 3))
rho = float(spearmanr(probs[0], probs[1]).statistic)
tau = float(kendalltau(probs[0], probs[1]).statistic)
print(round(rho, 3), round((1 - tau) / 2, 4))
# w=1.0:  recall 0.63   precision 0.788  AUC 0.918
# w=3.15: recall 0.819  precision 0.618  AUC 0.917
# Spearman 0.992, fraction of pairs that change order 0.0368

# %% [markdown]
# This is the default in LightGBM, the tree_method='hist' setting that modern XGBoost uses, and the entire basis of scikit-learn's own HistGradientBoostingClassifier and HistGradientBoostingRegressor, which are the fast, dependency-free boost

# %%
import time
import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
cov = fetch_covtype(as_frame=False)                    # 581,012 cartographic rows, 7 forest cover types
mask = (cov.target == 1) | (cov.target == 2)           # keep the two largest types -> a binary problem
X, y = cov.data[mask], (cov.target[mask] == 2).astype(int)
rng = np.random.default_rng(0)
idx = rng.choice(len(y), 30000, replace=False)         # subsample 30,000 rows for a quick comparison
X, y = X[idx], y[idx]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
exact = GradientBoostingClassifier(n_estimators=200, random_state=0)
hist = HistGradientBoostingClassifier(max_iter=200, random_state=0)
matched = HistGradientBoostingClassifier(
    max_iter=200, learning_rate=0.1, max_depth=3, early_stopping=False,
    min_samples_leaf=1, random_state=0)
print(exact.max_depth, exact.min_samples_leaf)
print(hist.max_leaf_nodes, hist.max_depth, hist.min_samples_leaf, hist.early_stopping)
for name, model in [("exact", exact), ("hist", hist), ("matched", matched)]:
    t0 = time.perf_counter()
    model.fit(Xtr, ytr)
    dt = time.perf_counter() - t0
    trees = getattr(model, "n_iter_", None)
    if trees is None:
        trees = model.n_estimators
    print(name, f"{accuracy_score(yte, model.predict(Xte)):.3f}", f"{dt:.2f}", trees)
# 3 1
# 31 None 20 auto
# exact 0.809 7.41 200
# hist 0.860 1.12 200
# matched 0.807 0.73 200

# %% [markdown]
# Two more regularization knobs: reg_alpha soft-thresholds the leaf weight, checked on a one-leaf model.

# %%
import numpy as np
from xgboost import XGBRegressor
X = np.zeros((2, 1))                                # one constant feature, so the tree is a single leaf
y = np.array([1.5, 1.5])                            # from a start of 0: g = -1.5 each, G = -3, H = 2
for alpha in (0, 1, 3):
    m = XGBRegressor(n_estimators=1, learning_rate=1.0, max_depth=1, reg_lambda=1.0,
                     reg_alpha=alpha, base_score=0.0).fit(X, y)
    print(alpha, round(float(m.predict(X)[0]), 4))
# 0 1.0
# 1 0.6667
# 3 0.0

# %% [markdown]
# Tree depth on a stratified 25 percent test split of the breast-cancer data, 40 trees, learning rate 0.1.

# %%
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
for depth in (1, 2, 3, 6):
    model = XGBClassifier(n_estimators=40, max_depth=depth, learning_rate=0.1,
                          n_jobs=2, eval_metric="logloss", random_state=0).fit(Xtr, ytr)
    print(depth, round(accuracy_score(yte, model.predict(Xte)), 3))
# 1 0.937   2 0.951   3 0.958   6 0.958

# %% [markdown]
# An eight-trial Optuna search of depth and learning rate on the same held-out rows.

# %%
import optuna
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    clf = XGBClassifier(
        n_estimators=40,
        max_depth=trial.suggest_int("max_depth", 2, 4),
        learning_rate=trial.suggest_float("learning_rate", 0.05, 0.2),
        n_jobs=2, eval_metric="logloss", random_state=0,
    )
    clf.fit(Xtr, ytr)
    return accuracy_score(yte, clf.predict(Xte))

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
study.optimize(objective, n_trials=8)
print(round(float(study.best_value), 3), study.best_params["max_depth"],
      round(float(study.best_params["learning_rate"]), 3))
# 0.965 3 0.189

# %% [markdown]
# XGBoost's weight, gain, and cover importances against permutation importance.

# %%
from sklearn.datasets import load_breast_cancer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import numpy as np
X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
model = XGBClassifier(n_estimators=40, max_depth=3, n_jobs=2, eval_metric="logloss",
                      random_state=0).fit(Xtr, ytr)
for kind in ("weight", "gain", "cover"):
    score = model.get_booster().get_score(importance_type=kind)
    top = sorted(score, key=score.get, reverse=True)[:3]
    print(kind, [(k, round(float(score[k]), 3)) for k in top])
perm = permutation_importance(model, Xte, yte, n_repeats=5, random_state=0, n_jobs=2)
order = np.argsort(perm.importances_mean)[::-1][:3]
print("perm", [(int(i), round(float(perm.importances_mean[i]), 4)) for i in order])
# weight [('f21', 25.0), ('f27', 18.0), ('f20', 15.0)]
# gain [('f22', 74.849), ('f27', 8.55), ('f20', 7.615)]
# cover [('f22', 43.275), ('f12', 26.476), ('f10', 23.379)]
# perm [(27, 0.0224), (21, 0.0182), (20, 0.0112)]

# %% [markdown]
# Table 7-3's knobs tuned together: a 40-trial Optuna (TPE) search on the adult subsample of 7.4.2, scored by five-fold cross-validated ROC AUC on the training rows; the hand-set and the tuned model are then each scored once on the test rows.

# %%
import time
import numpy as np
import optuna
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
adult = fetch_openml("adult", version=2, as_frame=True)
df = adult.frame.sample(10000, random_state=0)                   # same subsample as 7.4.2
y = (df["class"] == ">50K").astype(int).to_numpy()
Xdf = df.drop(columns="class"); cat = Xdf.select_dtypes("category").columns
Xdf[cat] = OrdinalEncoder(encoded_missing_value=-1).fit_transform(Xdf[cat])
X = Xdf.to_numpy(dtype=float)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)   # the same five folds for every trial

def fold_aucs(params):                     # five validation AUCs, computed from the training rows only
    model = XGBClassifier(**params, eval_metric="logloss", n_jobs=2, random_state=0)
    return cross_val_score(model, X_tr, y_tr, cv=cv, scoring="roc_auc")

hand = dict(n_estimators=300, learning_rate=0.1, max_depth=4, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=1.0)                 # the hand-set XGBoost of 7.4.2
hand_folds = fold_aucs(hand)

def objective(trial):                      # one trial = one setting of the Table 7-3 knobs
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 1000, step=50),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        max_depth=trial.suggest_int("max_depth", 2, 8),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 0.1, 100.0, log=True),
        gamma=trial.suggest_float("gamma", 0.0, 5.0),
    )
    return fold_aucs(params).mean()        # the score TPE maximizes: mean AUC over the five folds

optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
t0 = time.perf_counter()
study.optimize(objective, n_trials=40)
secs = time.perf_counter() - t0
v = np.array([t.value for t in study.trials])                    # the 40 cross-validated AUCs, in trial order
print(round(hand_folds.mean(), 4), round(hand_folds.std(), 4))         # the hand-set model's CV AUC and fold spread
print(round(v.min(), 4), round(v.max(), 4), int((v > hand_folds.mean()).sum()))
print(study.best_trial.number + 1, round(v[:10].max(), 4),          # TPE's first 10 trials are random draws
      round(v[:10].mean(), 4), round(v[10:].mean(), 4), round(secs, 1))
print({k: (round(x, 3) if isinstance(x, float) else x) for k, x in study.best_params.items()})
gains = fold_aucs(study.best_params) - hand_folds                 # same folds, so the gains are paired
print(np.round(gains, 4).tolist())
for params in (hand, study.best_params):                          # each model is scored ONCE on the test rows
    m = XGBClassifier(**params, eval_metric="logloss", n_jobs=2, random_state=0).fit(X_tr, y_tr)
    print(round(roc_auc_score(y_te, m.predict_proba(X_te)[:, 1]), 4))

# %% [markdown]
# ## 7.5.1 LightGBM: engineered for speed

# %% [markdown]
# Sweeping it on the 10,000-row adult census subsample of 7.4.2, everything else fixed, shows the leaf-wise tree memorizing the training data as the leaf budget grows.

# %%
import warnings; warnings.filterwarnings("ignore")   # hush a benign feature-name notice
import numpy as np, pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
adult = fetch_openml("adult", version=2, as_frame=True)
df = adult.frame.sample(10000, random_state=0)                   # same subsample as 7.4.2
y = (df["class"] == ">50K").astype(int).to_numpy()
Xdf = df.drop(columns="class"); cat = Xdf.select_dtypes("category").columns
Xdf[cat] = OrdinalEncoder(encoded_missing_value=-1).fit_transform(Xdf[cat])
X = Xdf.to_numpy(dtype=float)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
print("num_leaves  train  test   gap  train_errors")
for nl in [7, 31, 127, 511]:
    m = LGBMClassifier(n_estimators=200, learning_rate=0.1, num_leaves=nl,
                       n_jobs=2, verbose=-1, random_state=0).fit(X_tr, y_tr)
    pred_tr = m.predict(X_tr)
    tr, te = accuracy_score(y_tr, pred_tr), accuracy_score(y_te, m.predict(X_te))
    print(f"{nl} {tr:.3f} {te:.3f} {tr - te:.3f} {int(np.sum(pred_tr != y_tr))}")
# 7 0.886 0.867 0.019 854       31 0.955 0.860 0.095 335
# 127 1.000 0.846 0.153 1       511 1.000 0.847 0.153 0

# %% [markdown]
# GOSS is a one-argument switch, so it is worth seeing too; here on 200,000 rows of the two-class forest cover data.

# %%
import warnings; warnings.filterwarnings("ignore")   # hush a benign feature-name notice
import time
import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
cov = fetch_covtype(as_frame=False)                  # the 7.4.4 dataset, at larger scale
mask = (cov.target == 1) | (cov.target == 2)
X, y = cov.data[mask], (cov.target[mask] == 2).astype(int)
rng = np.random.default_rng(0)
idx = rng.choice(len(y), 200000, replace=False)      # 200,000 rows this time
X, y = X[idx], y[idx]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
for name, kw in [("gbdt", {}), ("goss", {"data_sample_strategy": "goss"})]:
    m = LGBMClassifier(n_estimators=300, learning_rate=0.1, num_leaves=63,
                       n_jobs=2, verbose=-1, random_state=0, **kw)
    t0 = time.perf_counter(); m.fit(Xtr, ytr); dt = time.perf_counter() - t0
    print(name, round(accuracy_score(yte, m.predict(Xte)), 3), round(dt, 2))
# gbdt 0.922 1.82      goss 0.93 3.08   (seconds: one run on the reference machine)

# %% [markdown]
# ## 7.5.2 CatBoost: ordered encoding against target leakage

# %% [markdown]
# This one demonstration has to be synthetic, because it needs a categorical column that is guaranteed to be independent of the label, which is exactly what a real dataset never hands you cleanly; only by manufacturing the noise ourselves can

# %%
import numpy as np
from sklearn.model_selection import train_test_split
rng = np.random.default_rng(0)
n = 4000
price = rng.normal(size=n); y_noise = (price > 0).astype(int)   # the label depends only on price
cat = rng.integers(0, 1500, n)                                  # ~1500 categories, INDEPENDENT of the label
c_tr, c_te, y_tr2, y_te2 = train_test_split(cat, y_noise, test_size=0.25, random_state=0)
enc = {}
for c, lab in zip(c_tr, y_tr2): enc.setdefault(c, []).append(lab)
mean_enc = {c: np.mean(v) for c, v in enc.items()}              # naive target encoding (leaks!)
te_tr = np.array([mean_enc.get(c, 0.5) for c in c_tr])
te_te = np.array([mean_enc.get(c, 0.5) for c in c_te])
print(round(float(np.corrcoef(te_tr, y_tr2)[0, 1]), 3))        # 0.653  spuriously high on training
print(round(float(np.corrcoef(te_te, y_te2)[0, 1]), 3))        # 0.025  near zero on test: it was noise all along

# %%
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
rng = np.random.default_rng(0)
n = 4000
price = rng.normal(size=n)
y = (price > 0).astype(int)
cat = rng.integers(0, 1500, n)
c_tr, _, y_tr, _ = train_test_split(cat, y, test_size=0.25, random_state=0)
order = np.random.default_rng(0).permutation(len(y_tr))
prior = float(y_tr.mean())
a = 1.0
enc = np.empty(len(y_tr))
sums, counts = defaultdict(float), defaultdict(int)
for i, (c, lab) in enumerate(zip(c_tr[order], y_tr[order])):
    enc[i] = (sums[c] + a * prior) / (counts[c] + a)
    sums[c] += lab
    counts[c] += 1
back = np.empty(len(y_tr))
back[order] = enc
print(round(float(np.corrcoef(back, y_tr)[0, 1]), 3))  # -0.012  ordered encoding, almost no leak

# %% [markdown]
# The result is that CatBoost consumes raw categorical columns directly and safely.

# %%
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
df = sns.load_dataset("titanic").dropna(subset=["embark_town", "sex", "age", "fare", "survived"])
X = df[["embark_town", "sex", "age", "fare"]]          # two raw string categoricals plus two numbers
y = df["survived"].astype(int)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
cb = CatBoostClassifier(n_estimators=300, depth=4, thread_count=2, verbose=0,
                        cat_features=["embark_town", "sex"], random_state=0).fit(X_tr, y_tr)   # no manual encoding
print(round(accuracy_score(y_te, cb.predict(X_te)), 3), cb.get_all_params()["boosting_type"])
# 0.775 Plain

# %% [markdown]
# Mark the string columns as the pandas category dtype, or encode them as nonnegative integers and pass their names through categorical_feature. Object-dtype strings are rejected.

# %%
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier
from lightgbm import LGBMClassifier
df = sns.load_dataset("titanic").dropna(subset=["embark_town", "sex", "age", "fare", "survived"])
X = df[["embark_town", "sex", "age", "fare"]].copy()
X["embark_town"] = X["embark_town"].astype("category")   # mark as pandas 'category' dtype
X["sex"] = X["sex"].astype("category")                    # LightGBM then splits on it natively
y = df["survived"].astype(int)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
lgb = LGBMClassifier(n_estimators=300, num_leaves=31, n_jobs=2, verbose=-1,
                     random_state=0).fit(X_tr, y_tr)      # no one-hot, no manual encoding
print(round(accuracy_score(y_te, lgb.predict(X_te)), 3))  # 0.798  LightGBM splits on the categories directly
hgb = HistGradientBoostingClassifier(random_state=0).fit(X_tr, y_tr)   # categorical_features="from_dtype"
print(round(accuracy_score(y_te, hgb.predict(X_te)), 3), hgb.is_categorical_.tolist())  # 0.77 [True, True, False, False]
raw = df[["embark_town", "sex", "age", "fare"]]             # object-dtype strings, not category
try:
    LGBMClassifier(n_estimators=5, n_jobs=2, verbose=-1, random_state=0).fit(
        raw, y, categorical_feature=["embark_town", "sex"])
except ValueError as err:
    print(str(err).splitlines()[0])                         # pandas dtypes must be int, float or bool.
codes = raw.copy()
codes["embark_town"] = codes["embark_town"].astype("category").cat.codes
codes["sex"] = codes["sex"].astype("category").cat.codes    # nonnegative integer codes
LGBMClassifier(n_estimators=5, n_jobs=2, verbose=-1, random_state=0).fit(
    codes, y, categorical_feature=["embark_town", "sex"])
print(int(codes["embark_town"].min()), int(codes["sex"].min()))  # 0 0, and the integer fit is accepted

# %% [markdown]
# The copy tested for this chapter is 3.4.1, so the XGBoost Titanic block in this section prints `True` for the default of `enable_categorical`. It still sets the flag explicitly, which keeps the code working on earlier releases such as 3.2.0, where the default was `False`. It scores 0.792.

# %%
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
df = sns.load_dataset("titanic").dropna(subset=["embark_town", "sex", "age", "fare", "survived"])
X = df[["embark_town", "sex", "age", "fare"]].copy()
X["embark_town"] = X["embark_town"].astype("category")
X["sex"] = X["sex"].astype("category")
y = df["survived"].astype(int)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
print(XGBClassifier().get_params()["enable_categorical"])  # True on XGBoost 3.4.1 (False before 3.3.0)
xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                    enable_categorical=True, n_jobs=2, eval_metric="logloss",
                    random_state=0).fit(X_tr, y_tr)
print(round(accuracy_score(y_te, xgb.predict(X_te)), 3))  # 0.792

# %% [markdown]
# ## 7.5.3 The three-way race, and when to pick which

# %% [markdown]
# We race them on the breast-cancer data, 569 tumors with 30 numeric measurements each, benign against malignant.

# %%
import time
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
X, y = load_breast_cancer(return_X_y=True)                 # 569 tumors, 30 numeric features
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
miss = {}
def race(name, model):
    t0 = time.perf_counter(); model.fit(X_tr, y_tr); dt = time.perf_counter() - t0
    pred = model.predict(X_te)
    key = name.strip()
    miss[key] = sorted(int(i) for i in np.flatnonzero(pred != y_te))
    print(name, round(accuracy_score(y_te, pred), 3),
          round(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]), 3),
          round(dt, 2), len(miss[key]))
race("XGBoost ", XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=4, n_jobs=2, eval_metric="logloss", random_state=0))
race("LightGBM", LGBMClassifier(n_estimators=300, learning_rate=0.1, num_leaves=31, n_jobs=2, verbose=-1, random_state=0))
race("CatBoost", CatBoostClassifier(n_estimators=300, learning_rate=0.1, depth=4, thread_count=2, verbose=0, random_state=0))
print(len(y_te), miss["XGBoost"], miss["LightGBM"], miss["CatBoost"])
# XGBoost  0.972 0.999 0.1 4     LightGBM 0.972 0.999 0.15 4     CatBoost 0.972 0.998 0.28 4
# 143 [10, 13, 14, 91] [40, 42, 91, 125] [10, 13, 91, 92]

# %% [markdown]
# ## 7.6 Short PyMC-BART chain

# %%
# Requires: pip install pymc pymc-bart. PYTENSOR_FLAGS=cxx= keeps PyTensor in pure Python.
# pymc_bart.BART starts a helper process, so on Windows and macOS a script needs the __main__ guard.
import os
os.environ["PYTENSOR_FLAGS"] = "cxx="
os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np
import pymc as pm
import pymc_bart as pmb
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
if __name__ == "__main__":
    X, y = load_diabetes(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
    scale = float(y_tr.std())
    with pm.Model():
        mu = pmb.BART("mu", X_tr, y_tr / scale, m=15)
        sigma = pm.HalfNormal("sigma", sigma=1.0)
        pm.Normal("y", mu=mu, sigma=sigma, observed=y_tr / scale)
        idata = pm.sample(draws=20, tune=20, chains=1, cores=1,
                          progressbar=False, random_seed=0)
    mu_draws = idata.posterior["mu"].values.reshape(-1, len(y_tr))
    lo, hi = np.quantile(mu_draws[:, 0] * scale, [0.055, 0.945])
    print(round(float(lo), 1), round(float(hi), 1), round(float(y_tr[0]), 1))
    # 146.8  166.4  85.0   (toy chain: too short for diagnostics)

# %% [markdown]
# ## 7.6.1 A runnable interval, and a self-check

# %% [markdown]
# On the diabetes dataset, 442 patients whose target is a one-year measure of disease progression, three HistGradientBoosting models at the 5.5th, 50th, and 94.5th percentiles give a median prediction wrapped in an 89 percent band.

# %%
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
X, y = load_diabetes(return_X_y=True)              # 442 patients, 10 features, target = disease progression
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
def q_model(q):                                    # one booster fit to a chosen quantile of the target
    return HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=80,
                                         learning_rate=0.05, min_samples_leaf=50,
                                         random_state=0).fit(X_tr, y_tr)
lo, mid, hi = q_model(0.055), q_model(0.5), q_model(0.945)   # low, median, high: an 89% central band
p_lo, p_mid, p_hi = lo.predict(X_te), mid.predict(X_te), hi.predict(X_te)
covered = float(np.mean((y_te >= p_lo) & (y_te <= p_hi)))
print(round(float(y_te[0]), 1), round(float(p_mid[0]), 1), round(float(p_lo[0]), 1), round(float(p_hi[0]), 1))
# 321.0  212.4  97.7  322.8   (truth, median, low, high)
print(round(covered, 3), round(float(np.mean(p_hi - p_lo)), 1))                        # 0.831  157.7  (coverage, mean width)

# %% [markdown]
# A converged PyMC-BART fit: 200 trees, four chains of 2,000 tuning and 4,000 kept iterations, R-hat and effective sample size, then the 89 percent credible and predictive bands on the 89 test patients.

# %%
import os
os.environ["PYTENSOR_FLAGS"] = "cxx="
os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np
import arviz as az
import pymc as pm
import pymc_bart as pmb
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

if __name__ == "__main__":                 # PyMC-BART starts a helper process, so guard the script entry
    X, y = load_diabetes(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)   # the split of the quantile band
    scale = float(y_tr.std())
    with pm.Model() as model:
        Xd = pm.Data("Xd", X_tr)           # a data container, so the test rows can be swapped in later
        mu = pmb.BART("mu", Xd, y_tr / scale, m=200)   # 200 trees, the default of Chipman et al.
        sigma = pm.HalfNormal("sigma", sigma=1.0)
        pm.Normal("y", mu=mu, sigma=sigma, observed=y_tr / scale, shape=mu.shape)
        idata = pm.sample(draws=4000, tune=2000, chains=4, cores=1, progressbar=False, random_seed=0)
    summ = az.summary(idata, var_names=["mu", "sigma"], round_to="none")
    print(round(float(summ["r_hat"].astype(float).max()), 3), int(summ["ess_bulk"].astype(float).min()))
    with model:
        pm.set_data({"Xd": X_te})          # predict for the 89 held-out patients
        pred = pm.sample_posterior_predictive(idata, sample_vars=["mu"], var_names=["mu", "y"],
                                              predictions=True, random_seed=0, progressbar=False)
    f = pred.predictions["mu"].values.reshape(-1, len(y_te)) * scale        # draws of the mean function
    ynew = pred.predictions["y"].values.reshape(-1, len(y_te)) * scale     # draws of a new patient's score
    f_lo, f_hi = np.quantile(f, [0.055, 0.945], axis=0)                    # 89% credible band for f(x)
    y_lo, y_hi = np.quantile(ynew, [0.055, 0.945], axis=0)                 # 89% predictive band for y
    print(f.shape[0], round(float(y_te[0]), 1), round(float(f[:, 0].mean()), 1),
          round(float(f_lo[0]), 1), round(float(f_hi[0]), 1), round(float(y_lo[0]), 1), round(float(y_hi[0]), 1))
    for lo, hi in [(f_lo, f_hi), (y_lo, y_hi)]:
        print(round(float(np.mean((y_te >= lo) & (y_te <= hi))), 3), round(float(np.mean(hi - lo)), 1))

# %% [markdown]
# ## 7.7.1 Voting

# %% [markdown]
# With a set of deliberately diverse base models, soft voting already produces a solid result.

# %%
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
digits = load_digits()                                   # 8-by-8 handwritten digit images, 64 pixel features
mask = np.isin(digits.target, [3, 8])                    # keep one confusable pair of digits: 3 vs 8
X, y = digits.data[mask], (digits.target[mask] == 8).astype(int)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
base = [("logreg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))),      # four
        ("svm",    make_pipeline(StandardScaler(), SVC(probability=True, random_state=0))),  # deliberately
        ("knn",    make_pipeline(StandardScaler(), KNeighborsClassifier(15))),               # diverse
        ("rf",     RandomForestClassifier(n_estimators=200, n_jobs=2, random_state=0))]      # base models
for name, m in base:
    m.fit(X_tr, y_tr)
    print(name, f"{accuracy_score(y_te, m.predict(X_te)):.3f}")       # logreg 0.991  svm 1.000  knn 0.981  rf 0.991
vote = VotingClassifier(base, voting="soft").fit(X_tr, y_tr)
print(round(accuracy_score(y_te, vote.predict(X_te)), 3))                # soft voting test accuracy = 0.991

# %% [markdown]
# ## 7.7.2 Stacking

# %% [markdown]
# On this data the same four base models score on the held-out test set from 0.776 for kNN up to 0.830 for the random forest, and soft voting reaches only 0.812, 1.8 points below the forest, precisely

# %%
import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.metrics import accuracy_score
cov = fetch_covtype(as_frame=True).frame
cov = cov[cov["Cover_Type"].isin([1, 2])].sample(n=6000, random_state=0)   # two majority classes, subsampled
y = (cov["Cover_Type"] == 2).astype(int).to_numpy()
X = cov.drop(columns="Cover_Type").to_numpy(dtype=float)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
base = [("logreg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
        ("svm",    make_pipeline(StandardScaler(), SVC(probability=True, random_state=0))),
        ("knn",    make_pipeline(StandardScaler(), KNeighborsClassifier(15))),
        ("rf",     RandomForestClassifier(n_estimators=200, n_jobs=2, random_state=0))]
for name, m in base:
    m.fit(X_tr, y_tr)
    print(name, round(accuracy_score(y_te, m.predict(X_te)), 3))         # logreg 0.785  svm 0.796  knn 0.776  rf 0.83
hard = VotingClassifier(base, voting="hard").fit(X_tr, y_tr)
vote = VotingClassifier(base, voting="soft").fit(X_tr, y_tr)
print("hard ", round(accuracy_score(y_te, hard.predict(X_te)), 3))       # hard voting  test accuracy = 0.808
print("vote ", round(accuracy_score(y_te, vote.predict(X_te)), 3))       # soft voting  test accuracy = 0.812
# n_jobs=1: the forest already runs 2 threads; parallel stacking jobs on top of it could oversubscribe the cores
stack = StackingClassifier(base, final_estimator=LogisticRegression(), cv=5, n_jobs=1).fit(X_tr, y_tr)
print("stack", round(accuracy_score(y_te, stack.predict(X_te)), 3))      # stacking     test accuracy = 0.831
print(np.round(stack.final_estimator_.coef_, 2).tolist(),
      np.round(stack.final_estimator_.intercept_, 2).tolist())
# [[-0.68, 0.04, 0.82, 6.64]] [-3.48]   (logreg, svm, knn, rf)
print(np.round(np.corrcoef(stack.transform(X_te), rowvar=False)[:, 3], 2).tolist())
# [0.88, 0.9, 0.85, 1.0]   each base model's test probabilities against the forest's

# %% [markdown]
# ## 7.7.3 The leakage trap, and blending

# %% [markdown]
# The numbers make the trap vivid.

# %%
import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.base import clone
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
cov = fetch_covtype(as_frame=True).frame
cov = cov[cov["Cover_Type"].isin([1, 2])].sample(n=6000, random_state=0)   # same 2-class subsample as 7.7.2
y = (cov["Cover_Type"] == 2).astype(int).to_numpy()
X = cov.drop(columns="Cover_Type").to_numpy(dtype=float)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
base = [("logreg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
        ("svm",    make_pipeline(StandardScaler(), SVC(probability=True, random_state=0))),
        ("knn",    make_pipeline(StandardScaler(), KNeighborsClassifier(15))),
        ("rf",     RandomForestClassifier(n_estimators=200, n_jobs=2, random_state=0))]
fitted = [clone(m).fit(X_tr, y_tr) for _, m in base]
meta_leak = np.column_stack([m.predict_proba(X_tr)[:, 1] for m in fitted])              # in-sample (leaks)
meta_oof = np.column_stack([cross_val_predict(clone(m), X_tr, y_tr, cv=5,
                            method="predict_proba")[:, 1] for _, m in base])            # out-of-fold (honest)
meta_te = np.column_stack([m.predict_proba(X_te)[:, 1] for m in fitted])
leak = LogisticRegression().fit(meta_leak, y_tr)
oof = LogisticRegression().fit(meta_oof, y_tr)
print(round(leak.score(meta_leak, y_tr), 3), round(leak.score(meta_te, y_te), 3))   # 1.000 train (fake!), 0.830 test
print(round(oof.score(meta_oof, y_tr), 3), round(oof.score(meta_te, y_te), 3))      # 0.814 train (honest), 0.831 test
print(np.round(leak.coef_[0], 2).tolist())   # [-0.8, -0.32, 0.48, 13.2]
print(np.round(oof.coef_[0], 2).tolist())    # [-0.68, 0.04, 0.82, 6.64]
rf_hat = fitted[-1].predict(X_te)
print(round(float(np.mean(leak.predict(meta_te) == rf_hat)), 3))  # 0.988 agree with the forest

# %% [markdown]
# Blending replaces cross-validation with a single holdout.

# %%
import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
cov = fetch_covtype(as_frame=True).frame
cov = cov[cov["Cover_Type"].isin([1, 2])].sample(n=6000, random_state=0)
y = (cov["Cover_Type"] == 2).astype(int).to_numpy()
X = cov.drop(columns="Cover_Type").to_numpy(dtype=float)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
base = [("logreg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
        ("svm",    make_pipeline(StandardScaler(), SVC(probability=True, random_state=0))),
        ("knn",    make_pipeline(StandardScaler(), KNeighborsClassifier(15))),
        ("rf",     RandomForestClassifier(n_estimators=200, n_jobs=2, random_state=0))]
rng_idx = np.random.default_rng(0).permutation(len(y_tr))
cut = int(0.8 * len(y_tr))
fit_idx, hold_idx = rng_idx[:cut], rng_idx[cut:]          # 3360 fit rows, 840 holdout rows
hold_cols, te_cols = [], []
for _, model in base:
    model.fit(X_tr[fit_idx], y_tr[fit_idx])
    hold_cols.append(model.predict_proba(X_tr[hold_idx])[:, 1])
    te_cols.append(model.predict_proba(X_te)[:, 1])
meta = LogisticRegression().fit(np.column_stack(hold_cols), y_tr[hold_idx])
print(round(accuracy_score(y_te, meta.predict(np.column_stack(te_cols))), 3))
# 0.833  blending test accuracy (same footing as the stack)
print(np.round(meta.coef_[0], 2).tolist())                 # [-0.28, 0.76, 1.02, 4.66]

# %% [markdown]
# A stack also costs more at prediction time, because every base model must score each new row before the meta-model can combine their outputs. A short timing on the breast-cancer data measures that cost: it times `predict_proba` on 1,000 rows for one 40-tree XGBoost and for a two-member stack of that booster with a 40-iteration histogram booster.

# %%
import time
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
single = XGBClassifier(n_estimators=40, max_depth=3, n_jobs=2, eval_metric="logloss", random_state=0)
stack = StackingClassifier(
    estimators=[
        ("xgb", XGBClassifier(n_estimators=40, max_depth=3, n_jobs=2, eval_metric="logloss", random_state=0)),
        ("hgb", HistGradientBoostingClassifier(max_iter=40, random_state=0)),
    ],
    final_estimator=LogisticRegression(max_iter=500),
    cv=3, n_jobs=1,
)
single.fit(Xtr, ytr)
stack.fit(Xtr, ytr)
rows = np.resize(Xte, (1000, Xte.shape[1]))
t0 = time.perf_counter(); single.predict_proba(rows); t_one = time.perf_counter() - t0
t0 = time.perf_counter(); stack.predict_proba(rows); t_stack = time.perf_counter() - t0
print(round(t_one, 4), round(t_stack, 4), round(t_stack / t_one, 2))
print(round(accuracy_score(yte, single.predict(Xte)), 3),
      round(accuracy_score(yte, stack.predict(Xte)), 3))
# 0.0006 0.0019 3.16
# 0.944 0.944

# %% [markdown]
# ## 7.8.1 The benchmark

# %% [markdown]
# A run on genuine data follows shortly to confirm the lesson holds outside the lab.

# %%
import time
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
warnings.filterwarnings("always", category=ConvergenceWarning)
rng = np.random.default_rng(0)
X = rng.normal(size=(6000, 10))
y = (((X[:, 0] > 0.3) & (X[:, 1] < -0.2) & (X[:, 5] > 0.0)) |
     ((X[:, 2] > 0.5) & (X[:, 3] < 0.0)) | ((X[:, 4] > 0.8) & (X[:, 6] < 0.4))).astype(int)
y = y ^ (rng.random(6000) < 0.03)            # axis-aligned rectangles + 3% label noise (constructed on purpose)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
t0 = time.perf_counter()
xgb = XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=6, n_jobs=2, eval_metric="logloss", random_state=0).fit(X_tr, y_tr)
t_xgb = time.perf_counter() - t0
t0 = time.perf_counter()
mlp = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400, random_state=0)).fit(X_tr, y_tr)
t_mlp = time.perf_counter() - t0
print(round(accuracy_score(y_te, xgb.predict(X_te)), 3), round(t_xgb, 1))   # 0.966  0.3
print(round(accuracy_score(y_te, mlp.predict(X_te)), 3), round(t_mlp, 1), int(mlp[-1].n_iter_))
# 0.888  4.0  400    (accuracy, seconds on one run of the reference machine, iterations used)

# %% [markdown]
# ## 7.8.2 Why trees win here

# %% [markdown]
# Appending 100 pure-noise features to the same problem shows the difference starkly.

# %%
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
rng = np.random.default_rng(0)                          # rebuild the same constructed target
X = rng.normal(size=(6000, 10))
y = (((X[:, 0] > 0.3) & (X[:, 1] < -0.2) & (X[:, 5] > 0.0)) |
     ((X[:, 2] > 0.5) & (X[:, 3] < 0.0)) | ((X[:, 4] > 0.8) & (X[:, 6] < 0.4))).astype(int)
y = y ^ (rng.random(6000) < 0.03)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
def acc_with_noise(k):
    Xtr = np.column_stack([X_tr, rng.normal(size=(len(X_tr), k))]) if k else X_tr
    Xte = np.column_stack([X_te, rng.normal(size=(len(X_te), k))]) if k else X_te
    xgb = XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=6, n_jobs=2, eval_metric="logloss", random_state=0).fit(Xtr, y_tr)
    mlp = make_pipeline(StandardScaler(), MLPClassifier((128, 64), max_iter=400, random_state=0)).fit(Xtr, y_tr)
    return accuracy_score(y_te, xgb.predict(Xte)), accuracy_score(y_te, mlp.predict(Xte))
x0, m0 = acc_with_noise(0); x100, m100 = acc_with_noise(100)
print(round(x0, 3), round(x100, 3))   # XGBoost 0.966 -> 0.967  (immune to 100 junk features)
print(round(m0, 3), round(m100, 3))   # MLP     0.888 -> 0.728  (junk features hurt it badly)

# %% [markdown]
# ## 7.8.3 The same verdict on real data

# %% [markdown]
# The forest cover type data from the US Forest Service records 54 cartographic features for map cells across four wilderness areas; we take the two majority cover types and subsample 10,000 rows, give the neural network the standardized inpu

# %%
import time
import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
cov = fetch_covtype(as_frame=True).frame
cov = cov[cov["Cover_Type"].isin([1, 2])].sample(n=10000, random_state=0)   # real numeric tabular, 2-class, subsampled
y = (cov["Cover_Type"] == 2).astype(int).to_numpy()
X = cov.drop(columns="Cover_Type").to_numpy(dtype=float)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
t0 = time.perf_counter()
xgb = XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=6, n_jobs=2, eval_metric="logloss", random_state=0).fit(X_tr, y_tr)
t_xgb = time.perf_counter() - t0
t0 = time.perf_counter()
mlp = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400, random_state=0)).fit(X_tr, y_tr)
t_mlp = time.perf_counter() - t0
print(round(accuracy_score(y_te, xgb.predict(X_te)), 3), round(t_xgb, 1))  # 0.844  0.4s
print(round(accuracy_score(y_te, mlp.predict(X_te)), 3), round(t_mlp, 1))  # 0.843  6.7s

# %% [markdown]
# ## 7.8.4 When deep learning does win
# The TabPFN race of 7.8.4 runs in its own companion script, 08_tabpfn_race.py, not in this notebook: in the reference environment an XGBoost fit in a Python process that had already imported PyTorch crashed, and this notebook fits XGBoost in later cells.

# %% [markdown]
# ## 7.9.1 The data and the problem

# %% [markdown]
# Example 7-13 loads and frames the data.

# %%
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

adult = fetch_openml("adult", version=2, as_frame=True)   # UCI Adult census income
df = adult.frame
target = adult.target_names[0]                              # the column name is "class"
print(f"full data: {len(df)} rows, positive rate (>50K) = {(df[target] == '>50K').mean():.4f}")

# The full set is 48,842 rows; we take a fixed 10,000-row subsample so the whole
# lab (five models, cross-validation, stacking, SHAP) runs in under two minutes.
df = df.sample(n=10000, random_state=0).reset_index(drop=True)
y = (df[target] == ">50K").astype(int).values
X = df.drop(columns=[target])

num = X.select_dtypes("number").columns.tolist()
cat = [c for c in X.columns if c not in num]

print(f"subsample: {len(X)} rows, {len(num)} numeric + {len(cat)} categorical features")
print(f"positive rate (>50K) in subsample = {y.mean():.4f}")
print(f"missing cells in subsample        = {int(X.isna().sum().sum())}")
print(f"columns with missing values       = {X.columns[X.isna().any()].tolist()}")

# %% [markdown]
# ## 7.9.2 A leakage-free preprocessor

# %% [markdown]
# Example 7-14 builds and verifies the transformer.

# %%
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import clone

adult = fetch_openml("adult", version=2, as_frame=True)
df = adult.frame.sample(n=10000, random_state=0).reset_index(drop=True)
y = (df["class"] == ">50K").astype(int).values
X = df.drop(columns=["class"])
num = X.select_dtypes("number").columns.tolist()
cat = [c for c in X.columns if c not in num]

# Hold out 25% as the untouched leaderboard test set, stratified on the label.
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=0)

pre = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
    ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                          OneHotEncoder(handle_unknown="ignore", sparse_output=False)), cat),
])

Xtr = pre.fit_transform(X_tr)   # fit learns medians, means, and categories from TRAIN
Xte = pre.transform(X_te)       # test is transformed with the TRAIN statistics only

print(f"train matrix {Xtr.shape}, test matrix {Xte.shape}")
print(f"numeric train columns have mean ~0: {Xtr[:, :len(num)].mean():+.2e}")
print(f"same numeric columns on test are not recentered: {Xte[:, :len(num)].mean():+.3f}")

# For contrast only, never in practice: a leaky copy fit on ALL 10,000 rows, test rows included.
leaky = clone(pre).fit(X)   # clone copies the settings, not the fitted statistics
print(f"leaky fit on all rows, same test columns: {leaky.transform(X_te)[:, :len(num)].mean():+.3f}")

# %% [markdown]
# ## 7.9.3 A cross-validated leaderboard

# %% [markdown]
# Note that the OpenMP cap takes effect only if it is set before those libraries load, so pasting Example 7-15 into a session that has already imported them may not bind the thread limit.

# %%
import os
os.environ["OMP_NUM_THREADS"] = "2"   # cap the OpenMP threads the HistGB fit spawns
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

adult = fetch_openml("adult", version=2, as_frame=True)
df = adult.frame.sample(n=10000, random_state=0).reset_index(drop=True)
y = (df["class"] == ">50K").astype(int).values
X = df.drop(columns=["class"])
num = X.select_dtypes("number").columns.tolist()
cat = [c for c in X.columns if c not in num]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=0)

pre = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
    ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                          OneHotEncoder(handle_unknown="ignore", sparse_output=False)), cat),
])

# Every candidate is a full pipeline: the preprocessor is refit inside each CV
# fold, so no validation fold ever leaks into the fit statistics.
models = {
    "LogisticRegression":   LogisticRegression(max_iter=2000),
    "RandomForest":         RandomForestClassifier(n_estimators=300, n_jobs=2, random_state=0),
    "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                                           random_state=0),
    "XGBoost":              XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=5,
                                          subsample=0.8, colsample_bytree=0.8, n_jobs=2,
                                          eval_metric="logloss", random_state=0),
    "LightGBM":             LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                                           subsample=0.8, colsample_bytree=0.8, n_jobs=2,
                                           verbose=-1, random_state=0),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
print("5-fold CV ROC AUC on the training set (mean +/- std):")
for name, clf in models.items():
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    s = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=1)
    print(f"  {name:22} {s.mean():.4f} +/- {s.std():.4f}")

# %% [markdown]
# ## 7.9.4 Letting the boosters size themselves

# %% [markdown]
# Example 7-16 runs both.

# %%
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

adult = fetch_openml("adult", version=2, as_frame=True)
df = adult.frame.sample(n=10000, random_state=0).reset_index(drop=True)
y = (df["class"] == ">50K").astype(int).values
X = df.drop(columns=["class"])
num = X.select_dtypes("number").columns.tolist()
cat = [c for c in X.columns if c not in num]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
# carve a validation slice out of TRAIN so early stopping never sees the test set
X_fit, X_val, y_fit, y_val = train_test_split(
    X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=0)

pre = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
    ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                          OneHotEncoder(handle_unknown="ignore", sparse_output=False)), cat),
]).fit(X_fit)
Xf, Xv = pre.transform(X_fit), pre.transform(X_val)

# XGBoost: cap at 2000 trees but stop when validation AUC has not improved for 50 rounds
xgb = XGBClassifier(n_estimators=2000, learning_rate=0.05, max_depth=5,
                    subsample=0.8, colsample_bytree=0.8, n_jobs=2,
                    eval_metric="auc", early_stopping_rounds=50, random_state=0)
xgb.fit(Xf, y_fit, eval_set=[(Xv, y_val)], verbose=False)
print(f"XGBoost stopped at {xgb.best_iteration + 1} trees (cap was 2000), "
      f"val AUC = {roc_auc_score(y_val, xgb.predict_proba(Xv)[:, 1]):.4f}")

# LightGBM: same idea, early stopping via a callback
lgbm = LGBMClassifier(n_estimators=2000, learning_rate=0.05, num_leaves=31,
                      subsample=0.8, colsample_bytree=0.8, n_jobs=2,
                      verbose=-1, random_state=0)
lgbm.fit(Xf, y_fit, eval_set=[(Xv, y_val)], eval_metric="auc",
         callbacks=[early_stopping(50, verbose=False), log_evaluation(0)])
print(f"LightGBM stopped at {lgbm.best_iteration_} trees (cap was 2000), "
      f"val AUC = {roc_auc_score(y_val, lgbm.predict_proba(Xv)[:, 1]):.4f}")

# %% [markdown]
# ## 7.9.5 Stacking the top models

# %% [markdown]
# its base boosters already run with n_jobs=2, and nesting parallel base fits inside a parallel stack is exactly the thread oversubscription that can grind a training run to a crawl.

# %%
import warnings; warnings.filterwarnings("ignore")
import os; os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

adult = fetch_openml("adult", version=2, as_frame=True)
df = adult.frame.sample(n=10000, random_state=0).reset_index(drop=True)
y = (df["class"] == ">50K").astype(int).values
X = df.drop(columns=["class"])
num = X.select_dtypes("number").columns.tolist()
cat = [c for c in X.columns if c not in num]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

pre = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
    ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                          OneHotEncoder(handle_unknown="ignore", sparse_output=False)), cat),
])

xgb  = XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=5, subsample=0.8,
                     colsample_bytree=0.8, n_jobs=2, eval_metric="logloss", random_state=0)
lgbm = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8,
                      colsample_bytree=0.8, n_jobs=2, verbose=-1, random_state=0)
hgb  = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, random_state=0)

# n_jobs=1 on the stack: the base boosters already use n_jobs=2, and nesting
# parallelism inside parallelism oversubscribes the cores and makes fits crawl.
stack = StackingClassifier(
    estimators=[("xgb", xgb), ("lgbm", lgbm), ("hgb", hgb)],
    final_estimator=LogisticRegression(max_iter=2000),
    cv=5, n_jobs=1)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
pipe = Pipeline([("pre", pre), ("clf", stack)])
s = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=1)
print(f"Stacked ensemble  5-fold CV ROC AUC = {s.mean():.4f} +/- {s.std():.4f}")
print("(compare to the best single model, XGBoost at 0.9170 from the leaderboard)")

# %%
probs = {}
for name, clf in [("xgb", xgb), ("lgbm", lgbm), ("hgb", hgb)]:
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    probs[name] = pipe.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
print(round(float(np.corrcoef(probs["xgb"], probs["lgbm"])[0, 1]), 3),
      round(float(np.corrcoef(probs["xgb"], probs["hgb"])[0, 1]), 3),
      round(float(np.corrcoef(probs["lgbm"], probs["hgb"])[0, 1]), 3))
# 0.984  0.98  0.982
print(round(0.98 + (1 - 0.98) / 3, 4))   # 0.9867 of one model's variance remains

# %% [markdown]
# ## 7.9.6 The final held-out report

# %% [markdown]
# Example 7-19 is the submission.

# %%
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, accuracy_score
from xgboost import XGBClassifier

adult = fetch_openml("adult", version=2, as_frame=True)
df = adult.frame.sample(n=10000, random_state=0).reset_index(drop=True)
y = (df["class"] == ">50K").astype(int).values
X = df.drop(columns=["class"])
num = X.select_dtypes("number").columns.tolist()
cat = [c for c in X.columns if c not in num]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

pre = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
    ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                          OneHotEncoder(handle_unknown="ignore", sparse_output=False)), cat),
])

# Cross-validation chose XGBoost; refit it on ALL the training data, score once on test.
final = Pipeline([("pre", pre),
                  ("clf", XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=5,
                                        subsample=0.8, colsample_bytree=0.8, n_jobs=2,
                                        eval_metric="logloss", random_state=0))])
final.fit(X_tr, y_tr)
proba = final.predict_proba(X_te)[:, 1]
pred = (proba >= 0.5).astype(int)
print(f"FINAL held-out test:  ROC AUC = {roc_auc_score(y_te, proba):.4f}")
print(f"                      accuracy = {accuracy_score(y_te, pred):.4f}")
print(f"majority-class accuracy (predict everyone <=50K) = {(y_te == 0).mean():.4f}")

# %% [markdown]
# ## 7.9.7 Interpreting the winner

# %% [markdown]
# If SHAP is not installed, Example 7-20 falls back to permutation importance alone.

# %%
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier

adult = fetch_openml("adult", version=2, as_frame=True)
df = adult.frame.sample(n=10000, random_state=0).reset_index(drop=True)
y = (df["class"] == ">50K").astype(int).values
X = df.drop(columns=["class"])
num = X.select_dtypes("number").columns.tolist()
cat = [c for c in X.columns if c not in num]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

pre = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
    ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                          OneHotEncoder(handle_unknown="ignore", sparse_output=False)), cat),
])
final = Pipeline([("pre", pre),
                  ("clf", XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=5,
                                        subsample=0.8, colsample_bytree=0.8, n_jobs=2,
                                        eval_metric="logloss", random_state=0))]).fit(X_tr, y_tr)

# Permutation importance on the RAW columns: shuffle one column of the test set,
# measure how much held-out ROC AUC drops. Model-agnostic and leakage-safe.
perm = permutation_importance(final, X_te, y_te, scoring="roc_auc",
                              n_repeats=10, random_state=0, n_jobs=2)
order = perm.importances_mean.argsort()[::-1]
print("Top permutation importances (drop in test ROC AUC when shuffled):")
for i in order[:6]:
    print(f"  {X.columns[i]:16} {perm.importances_mean[i]:.4f} +/- {perm.importances_std[i]:.4f}")

# SHAP: exact per-feature contributions from the tree structure, on transformed columns.
try:
    import shap
    Xte_t = final.named_steps["pre"].transform(X_te)
    names = final.named_steps["pre"].get_feature_names_out()
    expl = shap.TreeExplainer(final.named_steps["clf"])
    sv = expl.shap_values(Xte_t)
    mean_abs = np.abs(sv).mean(axis=0)
    top = mean_abs.argsort()[::-1][:6]
    print("\nTop SHAP features (mean |SHAP value| over the test set):")
    for i in top:
        print(f"  {names[i]:28} {mean_abs[i]:.4f}")
except ImportError:
    print("\n(shap not installed; permutation importance above is the interpretation)")

# %% [markdown]
# ## 7.9.8 Is it calibrated?

# %% [markdown]
# Example 7-21 computes both.

# %%
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier

adult = fetch_openml("adult", version=2, as_frame=True)
df = adult.frame.sample(n=10000, random_state=0).reset_index(drop=True)
y = (df["class"] == ">50K").astype(int).values
X = df.drop(columns=["class"])
num = X.select_dtypes("number").columns.tolist()
cat = [c for c in X.columns if c not in num]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

pre = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
    ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                          OneHotEncoder(handle_unknown="ignore", sparse_output=False)), cat),
])
final = Pipeline([("pre", pre),
                  ("clf", XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=5,
                                        subsample=0.8, colsample_bytree=0.8, n_jobs=2,
                                        eval_metric="logloss", random_state=0))]).fit(X_tr, y_tr)
proba = final.predict_proba(X_te)[:, 1]

# Brier score: mean squared error of the probabilities (0 is perfect).
print(f"Brier score on the held-out set = {brier_score_loss(y_te, proba):.4f}")
frac_pos, mean_pred = calibration_curve(y_te, proba, n_bins=5, strategy="quantile")
print("\nReliability (quantile bins): predicted vs. actual positive rate")
for p, a in zip(mean_pred, frac_pos):
    print(f"  predicted ~{p:.2f}  ->  actual {a:.2f}")

# %%
from scipy.stats import binom
def majority_accuracy(m, p):
    ...
print(round(float(1 - binom.cdf(0, 1, 0.55)), 3))

# %%
def correlated_variance(rho, m, sigma2=1.0):
    ...
print([0, 0.1, 0.3, 0.6])

# %%
from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(return_X_y=True)
def bag_predict(X_train, y_train, X_test):
    ...
print(X.shape)

# %%
from sklearn.datasets import load_wine
X, y = load_wine(return_X_y=True)
print(X.shape, len(set(y)))

# %%
from sklearn.datasets import load_diabetes
X, y = load_diabetes(return_X_y=True)
print(X.shape)

# %%
from sklearn.datasets import load_diabetes
X, y = load_diabetes(return_X_y=True)
def boost_mse(X, y):
    ...
print(round(float(y.mean()), 3))

# %%
from sklearn.datasets import load_digits
import numpy as np
digits = load_digits()
mask = np.isin(digits.target, [3, 8])
print(int(mask.sum()))

# %%
def leaf_weight(g, h, lam):
    ...
print(leaf_weight(-3, 2, 1))

# %%
from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(return_X_y=True)
def soft_vote_accuracy(X, y):
    ...
print(X.shape)

# %%
from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(return_X_y=True)
def leaderboard(X, y):
    ...
print(X.shape, int(y.sum()), int(len(y) - y.sum()))

# %%
import math
from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(return_X_y=True)
def newton_leaf(resid, p):
    ...
print(round(float(y.mean()), 4))

# %%
from sklearn.datasets import load_breast_cancer, fetch_openml
X, y = load_breast_cancer(return_X_y=True)
adult = fetch_openml("adult", version=2, as_frame=True)
df = adult.frame.sample(n=10000, random_state=0).reset_index(drop=True)
cat = df.drop(columns=["class"]).select_dtypes(exclude="number").columns.tolist()
print(X.shape, len(cat))

# %%
import numpy as np
import seaborn as sns
df = sns.load_dataset("titanic").dropna(subset=["embark_town", "sex", "age", "fare", "survived"])
noise = np.random.default_rng(0).integers(0, 600, size=len(df))   # a pure-noise id with up to 600 levels
def ordered_encoding(codes, labels, prior, a=1.0):
    ...
print(len(df), len(np.unique(noise)))

# %%
from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(return_X_y=True)
yp = (y == 0).astype(int)          # malignant, the minority label, becomes the positive class
print(int(yp.sum()), int(len(yp) - yp.sum()))

# %%
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
X, y = load_diabetes(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
def blend_r2(base, X_tr, y_tr, X_te, y_te):
    ...
print(len(y_tr), len(y_te))

