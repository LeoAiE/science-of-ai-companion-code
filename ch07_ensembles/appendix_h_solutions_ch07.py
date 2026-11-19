"""Appendix H solutions for Chapter 7, as runnable cells.

Generated from Solutions.md (Appendix H). Every code block of this chapter's solutions appears here
verbatim, split into one cell per exercise or drill topic, so each answer can be run and checked against
the expected outputs it states. Keep Solutions.md as the source; regenerate after any change.
"""

# %% [markdown]
# ## 7.10 coding exercises

# %%
import math
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import seaborn as sns
from collections import defaultdict
from scipy.stats import binom
from sklearn.datasets import fetch_covtype, load_breast_cancer, load_diabetes, load_wine, load_digits
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     cross_val_predict, StratifiedKFold)
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier,
                              RandomForestRegressor, AdaBoostClassifier, GradientBoostingClassifier,
                              GradientBoostingRegressor, HistGradientBoostingClassifier,
                              VotingClassifier, StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.inspection import permutation_importance
from sklearn.metrics import (mean_squared_error, accuracy_score, precision_score, recall_score,
                             roc_auc_score, brier_score_loss)
from sklearn.calibration import calibration_curve
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# %% [markdown]
# 1. Wisdom of a weaker crowd: p = 0.55, the size the theorem then needs

# %%
# 1. Wisdom of a weaker crowd: p = 0.55, the size the theorem then needs
p = 0.55
for M in [1, 11, 101, 1001]:
    print(M, round(1 - binom.cdf(M // 2, M, p), 3))   # P(a majority of M is correct)
# expected output: 1 0.55
# expected output: 11 0.633
# expected output: 101 0.844
# expected output: 1001 0.999
M = 1
while 1 - binom.cdf(M // 2, M, p) <= 0.95:            # smallest odd crowd above 0.95
    M += 2
print("smallest odd M above 0.95:", M)
# expected output: smallest odd M above 0.95: 269
# (the same loop with p = 0.6 stops at 67, so halving the edge roughly quadruples the crowd: 269 / 67 = 4.0)
rng = np.random.default_rng(0)
def crowd(M, p, shared, trials=20000):     # `shared` = fraction of fully correlated votes
    c = 0
    for _ in range(trials):
        v = np.repeat(rng.random() < p, M) if rng.random() < shared else rng.random(M) < p
        c += v.sum() > M / 2
    return c / trials
print(round(crowd(101, 0.55, 0.0), 3), round(crowd(101, 0.55, 0.8), 3))
# expected output: 0.842 0.611
# independent 55% voters reach 0.842; 80%-correlated errors collapse the crowd to 0.611

# %% [markdown]
# 2. Where extra members stop paying: floor sweep, saturation point, Monte Carlo

# %%
# 2. Where extra members stop paying: floor sweep, saturation point, Monte Carlo
def var_avg(M, rho, s2=1.0):
    return rho * s2 + (1 - rho) / M * s2    # rho*sigma^2 + (1-rho)/M * sigma^2
for rho in [0.0, 0.1, 0.3, 0.6]:
    print(rho, round(var_avg(100, rho), 3))
# expected output: 0.0 0.01
# expected output: 0.1 0.109
# expected output: 0.3 0.307
# expected output: 0.6 0.604
# at M=100 the variance is essentially the floor rho*sigma^2 itself
M = 1
while var_avg(M, 0.3) - var_avg(M + 1, 0.3) >= 0.001:   # marginal gain of one more model
    M += 1
print("saturation M:", M)
# expected output: saturation M: 26
# (analytically: the gain is 0.7/(M(M+1)), which drops below 0.001 once M(M+1) > 700)
rng = np.random.default_rng(0)
Mm, rho, trials = 100, 0.6, 40000
common = rng.normal(size=trials)                        # the shared component
idio = rng.normal(size=(trials, Mm))                    # each model's private noise
E = np.sqrt(rho) * common[:, None] + np.sqrt(1 - rho) * idio   # pairwise correlation rho
print(round(E.mean(axis=1).var(), 3), round(var_avg(100, 0.6), 3))
# expected output: 0.606 0.604
# the simulated variance of the 100-model average matches the formula's 0.604

# %% [markdown]
# 3. Bootstrap ~63%, then a from-scratch bagger vs one tree

# %%
# 3. Bootstrap ~63%, then a from-scratch bagger vs one tree
rng = np.random.default_rng(0)
for n in [10, 100, 1000]:
    frac = np.mean([len(np.unique(rng.integers(0, n, n))) / n for _ in range(2000)])
    print(n, round(frac, 3), round(1 - (1 - 1 / n) ** n, 3))   # measured vs exact 1 - (1 - 1/n)^n
print(round(1 - 1 / np.e, 3))                                # the limit 1 - 1/e
# expected output: 10 0.655 0.651
# expected output: 100 0.635 0.634
# expected output: 1000 0.632 0.632
# expected output: 0.632
# the exact fraction is already within 0.002 of the limit at n = 100, and each average over 2000
# bootstrap draws sits within 0.004 of its exact value
X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
def bagging_fit(X, y, B=50):
    r = np.random.default_rng(1); trees = []
    for _ in range(B):
        idx = r.integers(0, len(y), len(y))             # a bootstrap sample of row indices
        trees.append(DecisionTreeClassifier(random_state=0).fit(X[idx], y[idx]))
    return trees
trees = bagging_fit(Xtr, ytr)
maj = (np.array([t.predict(Xte) for t in trees]).mean(0) >= 0.5).astype(int)   # majority vote
single = DecisionTreeClassifier(random_state=0).fit(Xtr, ytr)
print(round((single.predict(Xte) == yte).mean(), 3), round((maj == yte).mean(), 3))
# expected output: 0.906 0.93
# averaging 50 bootstrapped trees beats the single unstable tree

# %% [markdown]
# 4. Decorrelation, out-of-bag, and Extra Trees on the wine data

# %%
# 4. Decorrelation, out-of-bag, and Extra Trees on the wine data
Xw, yw = load_wine(return_X_y=True)                      # 178 wines, 3 cultivars
for name, m in [("tree", DecisionTreeClassifier(random_state=0)),
                ("bag", BaggingClassifier(DecisionTreeClassifier(), n_estimators=100, random_state=0, n_jobs=2)),
                ("forest", RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=2)),
                ("extra", ExtraTreesClassifier(n_estimators=100, random_state=0, n_jobs=2))]:
    print(name, round(cross_val_score(m, Xw, yw, cv=5).mean(), 3))
# expected output: tree 0.888
# expected output: bag 0.955
# expected output: forest 0.983
# expected output: extra 0.972
Xtr, Xte, ytr, yte = train_test_split(Xw, yw, test_size=0.3, random_state=0, stratify=yw)
def tree_diversity(ens, Xd):                             # two probes of how alike the trees' test predictions are
    P = np.array([e.predict_proba(Xd).ravel() for e in ens.estimators_])   # class probabilities, flattened
    L = np.array([e.predict(Xd) for e in ens.estimators_])                 # predicted class labels
    iu = np.triu_indices(len(L), k=1)                                       # every pair of trees once
    corr = np.corrcoef(P)[iu].mean()                     # mean pairwise correlation of the probabilities
    disagree = (L[:, None, :] != L[None, :, :]).mean(axis=2)[iu].mean()     # mean pairwise disagreement rate
    return round(float(corr), 3), round(float(disagree), 3)
bag = BaggingClassifier(DecisionTreeClassifier(), n_estimators=100, random_state=0, n_jobs=2).fit(Xtr, ytr)
rf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=2).fit(Xtr, ytr)
print(tree_diversity(bag, Xte), tree_diversity(rf, Xte))
# expected output: (0.741, 0.173) (0.717, 0.188)
rf_oob = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=0, n_jobs=2).fit(Xtr, ytr)
print(round(rf_oob.oob_score_, 3), round(rf_oob.score(Xte, yte), 3))
# expected output: 0.96 1.0
# The ranking transfers (forest 0.983 > bag 0.955 > tree 0.888), and so does the decorrelation: on
# the 54 test wines the forest's trees agree a little less, with probability correlation 0.717
# against the bagger's 0.741 and a pairwise disagreement rate of 0.188 against 0.173. Correlating
# the raw class codes 0, 1, 2 would not be a valid probe for three classes, because Pearson
# correlation treats nominal codes as points on a line (a 0 against a 2 counts as a bigger miss than
# a 0 against a 1). On the two-class breast cancer data, where the codes 0 and 1 form a valid binary
# variable, the chapter's label probe read 0.825 vs 0.794. The OOB estimate (0.96) again tracks
# held-out accuracy (a perfect 1.0 on this small, easy test split).

# %% [markdown]
# 5. The feature-importance trap in regression (load_diabetes)

# %%
# 5. The feature-importance trap in regression (load_diabetes)
Xd, yd = load_diabetes(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(Xd, yd, test_size=0.3, random_state=0)
rng = np.random.default_rng(0)
hi_tr = rng.integers(0, 100000, size=len(Xtr)); hi_te = rng.integers(0, 100000, size=len(Xte))
bn_tr = rng.integers(0, 2, size=len(Xtr));      bn_te = rng.integers(0, 2, size=len(Xte))
Xn_tr = np.column_stack([Xtr, hi_tr, bn_tr])            # col -2 high-card noise, -1 binary noise
Xn_te = np.column_stack([Xte, hi_te, bn_te])
rf2 = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=2).fit(Xn_tr, ytr)
perm = permutation_importance(rf2, Xn_te, yte, scoring="r2", n_repeats=5, random_state=0, n_jobs=2)
print(round(rf2.feature_importances_[-2], 4), round(perm.importances_mean[-2], 4))   # high-card noise
print(round(rf2.feature_importances_[-1], 4), round(perm.importances_mean[-1], 4))   # binary noise
# expected output: 0.0446 0.003
# expected output: 0.0063 -0.0002
yc_tr = (ytr > np.median(ytr)).astype(int)              # same columns, target split at its median
rfc = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=2).fit(Xn_tr, yc_tr)
print(round(rfc.feature_importances_[-2], 4))            # high-card noise in a classification forest
print(round(np.mean([t.get_n_leaves() for t in rf2.estimators_])),
      round(np.mean([t.get_n_leaves() for t in rfc.estimators_])))   # mean leaves per tree
# expected output: 0.0694
# expected output: 190 53
# The high-cardinality noise column grabs 4.5 percent of the total impurity importance, seven times
# the binary flag's 0.6 percent and more than twenty times the 0.2 percent the same trick earned in
# the chapter's classification demo, while permutation importance scores both at about zero. The
# continuous target is not the cause: a classification forest on the same columns, with the target
# split at its median, gives the id column an even larger 6.9 percent, so the gap against the
# chapter demo comes from the data. On diabetes the default trees, grown until every leaf is pure,
# end up with many leaves (190 per regression tree, 53 per classification tree), and every extra
# split is another chance for the id column, whose many distinct values (about 309 x 0.632 = 195 of
# them reach each bootstrapped tree's root) almost always offer a split that lowers impurity by luck.

# %% [markdown]
# 6. Gradient boosting from scratch == scikit-learn, on load_diabetes

# %%
# 6. Gradient boosting from scratch == scikit-learn, on load_diabetes
X, y = load_diabetes(return_X_y=True)
def gb_scratch(X, y, n_trees=100, lr=0.1, max_depth=2):
    F = np.full(len(y), y.mean()); init = y.mean(); trees = []
    for _ in range(n_trees):
        h = DecisionTreeRegressor(max_depth=max_depth, random_state=0).fit(X, y - F)   # fit the residual
        F += lr * h.predict(X); trees.append(h)                                        # shrunken step
    return trees, init
trees, init = gb_scratch(X, y)
def gb_predict(X, trees, init, lr=0.1):
    return init + lr * sum(t.predict(X) for t in trees)
sk = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=2, random_state=0).fit(X, y)
print(round(mean_squared_error(y, gb_predict(X, trees, init)), 2))
print(round(mean_squared_error(y, sk.predict(X)), 2))
# expected output: 1814.14
# expected output: 1814.14
# the from-scratch residual loop reproduces GradientBoostingRegressor exactly

# %% [markdown]
# 7. Do the boosting recipes transfer? The same three sweeps on the digits' 3-vs-8 pair

# %%
# 7. Do the boosting recipes transfer? The same three sweeps on the digits' 3-vs-8 pair
digits = load_digits()
mask = np.isin(digits.target, [3, 8])
X38, y38 = digits.data[mask], (digits.target[mask] == 8).astype(int)
for n in [1, 5, 50, 200]:                               # each line: mean and spread of the 5 fold scores
    sc = cross_val_score(AdaBoostClassifier(n_estimators=n, random_state=0), X38, y38, cv=5)
    print("ada", n, round(sc.mean(), 3), round(sc.std(), 3))
# expected output: ada 1 0.815 0.119
# expected output: ada 5 0.869 0.118
# expected output: ada 50 0.936 0.074
# expected output: ada 200 0.944 0.063
for lr, n in [(1.0, 20), (0.1, 20), (0.1, 200), (0.05, 200)]:
    sc = cross_val_score(GradientBoostingClassifier(learning_rate=lr, n_estimators=n, max_depth=3,
                         random_state=0), X38, y38, cv=5)
    print("gb", lr, n, round(sc.mean(), 3), round(sc.std(), 3))
# expected output: gb 1.0 20 0.944 0.052
# expected output: gb 0.1 20 0.924 0.033
# expected output: gb 0.1 200 0.938 0.042
# expected output: gb 0.05 200 0.941 0.042
for sub in [1.0, 0.5, 0.3]:
    sc = cross_val_score(GradientBoostingClassifier(subsample=sub, n_estimators=200, learning_rate=0.1,
                         max_depth=3, random_state=0), X38, y38, cv=5)
    print("sub", sub, round(sc.mean(), 3), round(sc.std(), 3))
# expected output: sub 1.0 0.938 0.042
# expected output: sub 0.5 0.947 0.042
# expected output: sub 0.3 0.947 0.048
# Only one pattern clearly transfers. AdaBoost still climbs with more stumps, 0.815 to 0.944, a rise
# of about twice the fold spread at 200 stumps (0.063). The other two come out flat: lr=1.0 with 20
# trees (0.944) and lr=0.1 with 200 trees (0.938) differ by 0.006, and subsample 0.3 (0.947) beats
# 1.0 (0.938) by 0.009, both far inside fold standard deviations of 0.04 to 0.05. On 357 images split
# five ways those gaps sit inside the fold noise, the same verdict 7.3.4 reached on the breast cancer
# data. The recipes are defaults, not laws, which is exactly why every sweep here is worth rerunning
# on your own dataset.

# %% [markdown]
# 8. XGBoost's leaf math, then the histogram booster

# %%
# 8. XGBoost's leaf math, then the histogram booster
def leaf_weight(G, H, lam): return -G / (H + lam)
def split_gain(GL, HL, GR, HR, lam, gamma):
    return 0.5 * (GL**2 / (HL + lam) + GR**2 / (HR + lam)
                  - (GL + GR)**2 / (HL + HR + lam)) - gamma
print(leaf_weight(-3, 2, 1), round(leaf_weight(-3, 2, 10), 3))
# expected output: 1.0 0.25
print(round(split_gain(-3, 2, 5, 3, 1, 0.0), 3))
# expected output: 4.292
X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
hgb = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, early_stopping=True,
                                     validation_fraction=0.2, n_iter_no_change=20, random_state=0).fit(Xtr, ytr)
print(hgb.n_iter_, round(hgb.score(Xte, yte), 3))
# expected output: 118 0.942
Xnan = Xtr.copy(); Xnan[:50, 0] = np.nan
print(int(np.isnan(Xnan).sum()), round(HistGradientBoostingClassifier(random_state=0).fit(Xnan, ytr).score(Xte, yte), 3))
# expected output: 50 0.947
# raising lambda shrinks the leaf weight (1.0 -> 0.25); HistGB stops itself at 118 trees and trains through NaNs

# %% [markdown]
# 9. Voting, stacking, and the leakage trap

# %%
# 9. Voting, stacking, and the leakage trap
X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
base = [("logreg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))),
        ("svm",    make_pipeline(StandardScaler(), SVC(probability=True, random_state=0))),
        ("knn",    make_pipeline(StandardScaler(), KNeighborsClassifier(15))),
        ("rf",     RandomForestClassifier(n_estimators=200, n_jobs=2, random_state=0))]
vote = VotingClassifier(base, voting="soft").fit(Xtr, ytr)
print("vote", round(accuracy_score(yte, vote.predict(Xte)), 3))
# expected output: vote 0.953
stack = StackingClassifier(base, final_estimator=LogisticRegression(max_iter=5000), cv=5, n_jobs=1).fit(Xtr, ytr)
print("stack", round(accuracy_score(yte, stack.predict(Xte)), 3))
# expected output: stack 0.959
fitted = [clone(m).fit(Xtr, ytr) for _, m in base]
meta_leak = np.column_stack([m.predict_proba(Xtr)[:, 1] for m in fitted])          # in-sample (leaks)
meta_oof = np.column_stack([cross_val_predict(clone(m), Xtr, ytr, cv=5,
                            method="predict_proba")[:, 1] for _, m in base])        # out-of-fold (honest)
meta_te = np.column_stack([m.predict_proba(Xte)[:, 1] for m in fitted])
leak = LogisticRegression(max_iter=5000).fit(meta_leak, ytr)
oof = LogisticRegression(max_iter=5000).fit(meta_oof, ytr)
print(round(leak.score(meta_leak, ytr), 3), round(leak.score(meta_te, yte), 3))
print(round(oof.score(meta_oof, ytr), 3), round(oof.score(meta_te, yte), 3))
# expected output: 0.995 0.959
# expected output: 0.985 0.959
# in-sample meta-features inflate the training estimate (0.995) above the honest out-of-fold one
# (0.985), but on this easy dataset the leak adds only one point, and both overstate the 0.959 test
# accuracy (by 3.6 and 2.6 points). The forest cover stack of Example 7-10 shows the dramatic case: a leaky
# training score of 1.000 against an honest 0.814, with test accuracies of 0.830 and 0.831.

# %% [markdown]
# 10. End-to-end miniproject on load_breast_cancer

# %%
# 10. End-to-end miniproject on load_breast_cancer
X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
models = {"LogReg":       make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
          "RandomForest": RandomForestClassifier(n_estimators=300, n_jobs=2, random_state=0),
          "HistGB":       HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, random_state=0),
          "GradBoost":    GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=0)}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
scores = {}
for name, clf in models.items():
    scores[name] = cross_val_score(clf, Xtr, ytr, cv=cv, scoring="roc_auc", n_jobs=1).mean()
    print(name, round(scores[name], 4))
# expected output: LogReg 0.9947
# expected output: RandomForest 0.9959
# expected output: HistGB 0.9936
# expected output: GradBoost 0.9936
best = max(scores, key=scores.get)
final = models[best].fit(Xtr, ytr)
proba = final.predict_proba(Xte)[:, 1]
print(best, round(roc_auc_score(yte, proba), 4), round(accuracy_score(yte, (proba >= 0.5).astype(int)), 4))
# expected output: RandomForest 0.9769 0.9441
perm = permutation_importance(final, Xte, yte, scoring="roc_auc", n_repeats=10, random_state=0, n_jobs=2)
names = load_breast_cancer().feature_names
for i in perm.importances_mean.argsort()[::-1][:3]:
    print(names[i], round(perm.importances_mean[i], 4))
# expected output: worst area 0.0026
# expected output: worst perimeter 0.0023
# expected output: worst radius 0.0018
print("Brier", round(brier_score_loss(yte, proba), 4))
# expected output: Brier 0.0397
fp, mp = calibration_curve(yte, proba, n_bins=5, strategy="quantile")
for p_, a_ in zip(mp, fp):
    print("pred", round(p_, 2), "actual", round(a_, 2))
# expected output: pred 0.0 actual 0.0
# expected output: pred 0.23 actual 0.21
# expected output: pred 0.87 actual 0.97
# expected output: pred 1.0 actual 0.98
print(int((proba == 1.0).sum()), len(proba))             # test probabilities that are exactly 1.0
# expected output: 31 143
# Five bins were asked for but four print: 31 of the 143 test probabilities, more than a fifth, are
# exactly 1.0, so the 80th and 100th percentile edges are both 1.0, the fifth quantile bin is empty,
# and calibration_curve drops empty bins. The random forest wins the CV leaderboard, scores 0.977
# AUC / 0.944 accuracy held out, and keys on the "worst"-region size features. Its Brier score is
# 0.040, but the table is not uniformly good: the bin predicted at 0.87 comes true 97 percent of
# the time, so the forest is underconfident there.

# %% [markdown]
# 11. Classification leaves

# %%
# 11. Classification leaves
# Newton leaves against mean-residual leaves, with depth-2 trees
X, y = load_breast_cancer(return_X_y=True)
F0 = math.log(float(y.mean()) / (1 - float(y.mean())))
print(round(F0, 4))  # expected output: 0.5211
def boost(newton, rounds=20, lr=0.1):
    F = np.full(len(y), F0)                               # every row starts at the base-rate log odds
    for _ in range(rounds):
        p = 1 / (1 + np.exp(-F))
        resid = y - p                                     # negative gradient of the log loss
        tree = DecisionTreeRegressor(max_depth=2, random_state=0).fit(X, resid)
        leaves = tree.apply(X)
        step = np.zeros(len(y))
        for leaf in np.unique(leaves):
            m = leaves == leaf
            if newton:                                    # summed gradient over summed curvature
                step[m] = resid[m].sum() / max(float((p[m] * (1 - p[m])).sum()), 1e-12)
            else:                                         # plain mean residual, a probability-scale step
                step[m] = resid[m].mean()
        F = F + lr * step
    return F
def log_loss_of(F):
    p = 1 / (1 + np.exp(-F))
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
F_newton, F_mean = boost(True), boost(False)
print(round(log_loss_of(np.full(len(y), F0)), 4), round(log_loss_of(F_newton), 4), round(log_loss_of(F_mean), 4))
# expected output: 0.6603 0.1392 0.4014
lib = GradientBoostingClassifier(
    n_estimators=20, max_depth=2, learning_rate=0.1, random_state=0,
).fit(X, y)
print(round(float(np.max(np.abs(F_newton - lib.decision_function(X)))), 6))  # expected output: 0.0
# Both boosters start at log loss 0.6603. After 20 rounds the Newton leaves reach 0.1392 and the
# mean-residual leaves only 0.4014: the mean residual lives on the probability scale, so as a
# log-odds step it is too small by the factor 1 over the leaf's average p(1 - p), which is at least
# 4 because p(1 - p) never exceeds 1/4. GradientBoostingClassifier uses the Newton leaf.

# %% [markdown]
# 12. Three libraries

# %%
# 12. Three libraries
X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
Xfit, Xval, yfit, yval = train_test_split(Xtr, ytr, test_size=0.2, random_state=1, stratify=ytr)
es = XGBClassifier(
    n_estimators=200, learning_rate=0.1, max_depth=3, n_jobs=2,
    eval_metric="logloss", early_stopping_rounds=15, random_state=0,
)
es.fit(Xfit, yfit, eval_set=[(Xval, yval)], verbose=False)
print(int(es.best_iteration + 1))  # expected output: 129
# validation log loss was lowest after 129 rounds, so with patience 15 training stopped after 129 + 15 = 144 of 200
for nl in (7, 31, 127):
    m = LGBMClassifier(
        n_estimators=80, learning_rate=0.1, num_leaves=nl,
        n_jobs=2, verbose=-1, random_state=0,
    ).fit(Xtr, ytr)
    print(nl, round(accuracy_score(yte, m.predict(Xte)), 3))
# expected output: 7 0.951
# expected output: 31 0.944
# expected output: 127 0.944
# on 143 test tumors num_leaves=7 is one tumor ahead of 31 and 127 (136 against 135 correct)
from sklearn.datasets import fetch_openml
adult = fetch_openml("adult", version=2, as_frame=True)
df = adult.frame.sample(n=10000, random_state=0).reset_index(drop=True)
ya = (df["class"] == ">50K").astype(int).to_numpy()
Xa = df.drop(columns=["class"])
cat = Xa.select_dtypes(exclude="number").columns.tolist()   # OpenML delivers these as pandas categories
Xa_tr, Xa_te, ya_tr, ya_te = train_test_split(Xa, ya, test_size=0.25, stratify=ya, random_state=0)
Xs_tr, Xs_te = Xa_tr.astype({c: str for c in cat}), Xa_te.astype({c: str for c in cat})   # NaN becomes "nan"
cb = CatBoostClassifier(
    n_estimators=300, depth=4, thread_count=2, verbose=0,
    cat_features=cat, random_state=0,
).fit(Xs_tr, ya_tr)
print(round(roc_auc_score(ya_te, cb.predict_proba(Xs_te)[:, 1]), 4))  # expected output: 0.9172
lgb = LGBMClassifier(
    n_estimators=300, learning_rate=0.05, num_leaves=31, n_jobs=2, verbose=-1, random_state=0,
).fit(Xa_tr, ya_tr)
print(round(roc_auc_score(ya_te, lgb.predict_proba(Xa_te)[:, 1]), 4))  # expected output: 0.914
# Both native-categorical boosters land within 0.003 of the one-hot XGBoost's 0.9163 from 7.9.6
# (CatBoost 0.0009 above it, LightGBM 0.0023 below): on eight columns of a few dozen levels at most,
# one-hot encoding loses little, and the native handling mainly saves the encoding step.

# %% [markdown]
# 13. Target encoding

# %%
# 13. Target encoding
df = sns.load_dataset("titanic").dropna(subset=["embark_town", "sex", "age", "fare", "survived"])
yt = df["survived"].astype(int)
rng = np.random.default_rng(0)
noise = rng.integers(0, 600, size=len(df))
y_all = yt.to_numpy()
tr_idx, _ = train_test_split(np.arange(len(df)), test_size=0.25, random_state=0)
c_tr, y_tr = noise[tr_idx], y_all[tr_idx]
buckets = defaultdict(list)
for c, lab in zip(c_tr, y_tr):
    buckets[int(c)].append(int(lab))
means = {c: float(np.mean(v)) for c, v in buckets.items()}
naive = np.array([means[int(c)] for c in c_tr])
print(round(float(np.corrcoef(naive, y_tr)[0, 1]), 3))  # expected output: 0.815
order = np.random.default_rng(0).permutation(len(y_tr))
prior = float(y_tr.mean())
a = 1.0
enc = np.empty(len(y_tr))
sums, counts = defaultdict(float), defaultdict(int)
for i, (c, lab) in enumerate(zip(c_tr[order], y_tr[order])):
    enc[i] = (sums[int(c)] + a * prior) / (counts[int(c)] + a)
    sums[int(c)] += float(lab)
    counts[int(c)] += 1
back = np.empty(len(y_tr))
back[order] = enc
print(round(float(np.corrcoef(back, y_tr)[0, 1]), 3))  # expected output: 0.008
sizes = np.bincount(c_tr, minlength=600)
print(int((sizes > 0).sum()), int((sizes == 1).sum()))  # expected output: 356 225
# the 534 training rows fall on 356 distinct ids, 225 of them seen once, so the training-mean encoding leaks
# each row's own label (0.815), and for those 225 rows it is exactly that label; the ordered encoding sees
# only earlier rows and shows essentially no link to the label (0.008)

# %% [markdown]
# 14. Class weights

# %%
# 14. Class weights
X, y = load_breast_cancer(return_X_y=True)
yp = (y == 0).astype(int)
Xtr, Xte, ytr, yte = train_test_split(X, yp, test_size=0.25, random_state=0, stratify=yp)
ratio = float((ytr == 0).sum() / (ytr == 1).sum())
print(round(ratio, 3))  # expected output: 1.679
# 53 of the 143 test tumors are malignant; the recall and precision printed below mean 51 caught with 1 false alarm
# at weight 1.0 and 52 caught with 3 false alarms at 1.679, while the AUC barely moves, as on the adult data in 7.4.4
for w in (1.0, ratio):
    clf = XGBClassifier(
        n_estimators=40, max_depth=3, scale_pos_weight=w, n_jobs=2,
        eval_metric="logloss", random_state=0,
    ).fit(Xtr, ytr)
    pred = clf.predict(Xte)
    proba = clf.predict_proba(Xte)[:, 1]
    print(round(w, 3), round(recall_score(yte, pred), 3),
          round(precision_score(yte, pred), 3), round(roc_auc_score(yte, proba), 3))
# expected output: 1.0 0.962 0.981 0.991
# expected output: 1.679 0.981 0.945 0.993

# %% [markdown]
# 15. Blending

# %%
# 15. Blending
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, StackingRegressor
from sklearn.metrics import r2_score
X, y = load_diabetes(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
base = [("ridge", make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
        ("knn", make_pipeline(StandardScaler(), KNeighborsRegressor(15))),
        ("rf", RandomForestRegressor(n_estimators=200, min_samples_leaf=5, n_jobs=2, random_state=0)),
        ("hgb", HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=0))]
for name, model in base:
    print(name, round(r2_score(y_te, clone(model).fit(X_tr, y_tr).predict(X_te)), 3))
# expected output: ridge 0.358
# expected output: knn 0.348
# expected output: rf 0.266
# expected output: hgb 0.199
rng_idx = np.random.default_rng(0).permutation(len(y_tr))
cut = int(0.8 * len(y_tr))
fit_idx, hold_idx = rng_idx[:cut], rng_idx[cut:]
print(len(fit_idx), len(hold_idx))  # expected output: 264 67
hold_cols, te_cols = [], []
for _, model in base:
    fitted = clone(model).fit(X_tr[fit_idx], y_tr[fit_idx])
    hold_cols.append(fitted.predict(X_tr[hold_idx]))
    te_cols.append(fitted.predict(X_te))
meta = LinearRegression().fit(np.column_stack(hold_cols), y_tr[hold_idx])
print(round(r2_score(y_te, meta.predict(np.column_stack(te_cols))), 3))  # expected output: 0.298
print(np.round(meta.coef_, 2).tolist())  # expected output: [0.33, 0.04, 0.66, 0.05]
stack = StackingRegressor([(n, clone(m)) for n, m in base],
                          final_estimator=LinearRegression(), cv=5, n_jobs=1).fit(X_tr, y_tr)
print(round(r2_score(y_te, stack.predict(X_te)), 3))  # expected output: 0.348
print(np.round(stack.final_estimator_.coef_, 2).tolist())  # expected output: [0.48, 0.39, 0.08, 0.14]
# The blend's meta-model learns its four weights from only the 67 held-out rows, puts the largest
# weight on the forest, and scores 0.298; the stack learns from out-of-fold predictions for all 331
# training rows, weights the ridge and k-nearest neighbors most, and scores 0.348. On a table this
# small, blending's single holdout is too few rows to learn the weights. Neither combination beats
# the ridge's 0.358 on this split.

# %% [markdown]
# ## 7.10 coding drills

# %%
import numpy as np
from scipy.stats import binom
from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              GradientBoostingRegressor, HistGradientBoostingClassifier,
                              VotingClassifier, StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

# %% [markdown]
# Ensemble theory

# %%
# Ensemble theory
print(round(1 - binom.cdf(15 // 2, 15, 0.6), 3))              # expected output: 0.787
print(round(0.3 + (1 - 0.3) / 25, 3))                         # expected output: 0.328
rng = np.random.default_rng(0)
print(round(len(np.unique(rng.integers(0, 1000, 1000))) / 1000, 3))   # expected output: 0.622
print(round(1 / np.e, 3))                                    # expected output: 0.368

# %% [markdown]
# Random forests

# %%
# Random forests
X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
rf = RandomForestClassifier(n_estimators=200, oob_score=True, random_state=0, n_jobs=2).fit(Xtr, ytr)
print(round(rf.oob_score_, 3))                               # expected output: 0.97
print(np.round(np.sort(rf.feature_importances_)[::-1][:3], 3).tolist())   # expected output: [0.133, 0.132, 0.128]
print(int(np.sqrt(X.shape[1])))                              # expected output: 5
print(round(cross_val_score(ExtraTreesClassifier(n_estimators=200, random_state=0, n_jobs=2), X, y, cv=5).mean(), 3))
# expected output: 0.965

# %% [markdown]
# Boosting

# %%
# Boosting
yv = np.array([3.0, 5.0, 2.0]); Fv = np.array([2.5, 5.0, 3.0])
print((yv - Fv).tolist())                                    # expected output: [0.5, 0.0, -1.0]
eps = 0.3
print(round(0.5 * np.log((1 - eps) / eps), 3))               # expected output: 0.424
Xd, yd = load_diabetes(return_X_y=True)
print(round(GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=2, random_state=0).fit(Xd, yd).score(Xd, yd), 3))
# expected output: 0.763

# %% [markdown]
# XGBoost and histograms

# %%
# XGBoost and histograms
def leaf_weight(G, H, lam): return -G / (H + lam)
def split_gain(GL, HL, GR, HR, lam, gamma):
    return 0.5 * (GL**2 / (HL + lam) + GR**2 / (HR + lam) - (GL + GR)**2 / (HL + HR + lam)) - gamma
print(leaf_weight(-4.0, 3.0, 1.0))                           # expected output: 1.0
print(round(split_gain(-3, 2, 5, 3, 1.0, 0.0), 3))          # expected output: 4.292
X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
print(round(HistGradientBoostingClassifier(random_state=0).fit(Xtr, ytr).score(Xte, yte), 3))   # expected output: 0.942
Xn = Xtr.copy(); Xn[:30, 5] = np.nan
print(round(HistGradientBoostingClassifier(random_state=0).fit(Xn, ytr).score(Xte, yte), 3))    # expected output: 0.947

# %% [markdown]
# Combining models

# %%
# Combining models
Xw, yw = load_wine(return_X_y=True)
Xwtr, Xwte, ywtr, ywte = train_test_split(Xw, yw, test_size=0.3, random_state=0, stratify=yw)
b2 = [("lr", make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))),
      ("rf", RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=2))]
vh = VotingClassifier(b2, voting="hard").fit(Xwtr, ywtr)
vs = VotingClassifier(b2, voting="soft").fit(Xwtr, ywtr)
print(round(accuracy_score(ywte, vh.predict(Xwte)), 3), round(accuracy_score(ywte, vs.predict(Xwte)), 3))
# expected output: 1.0 1.0
st = StackingClassifier(b2, final_estimator=LogisticRegression(max_iter=5000), cv=5).fit(Xwtr, ywtr)
print(round(accuracy_score(ywte, st.predict(Xwte)), 3))      # expected output: 1.0
oof = cross_val_predict(RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=2), Xwtr, ywtr, cv=5)
print(round(accuracy_score(ywtr, oof), 3))                   # expected output: 0.96
X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
hg = HistGradientBoostingClassifier(random_state=0).fit(Xtr, ytr)
ml = make_pipeline(StandardScaler(), MLPClassifier((64,), max_iter=500, random_state=0)).fit(Xtr, ytr)
print(round(hg.score(Xte, yte), 3), round(ml.score(Xte, yte), 3))   # expected output: 0.942 0.953
# On this small, clean, all-numeric set the network wins. The 0.011 gap is about two test rows.
