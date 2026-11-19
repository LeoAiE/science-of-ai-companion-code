"""Appendix H solutions for Chapter 4, as runnable cells.

Generated from Solutions.md (Appendix H). Every code block of this chapter's solutions appears here
verbatim, split into one cell per exercise or drill topic, so each answer can be run and checked against
the expected outputs it states. Keep Solutions.md as the source; regenerate after any change.
"""

# %% [markdown]
# ## 4.11 coding exercises

# %%
import numpy as np, pandas as pd, seaborn as sns
from sklearn.model_selection import (train_test_split, cross_val_score, cross_val_predict, validation_curve,
    GridSearchCV, RandomizedSearchCV, StratifiedKFold, permutation_test_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet, LassoCV, RidgeCV
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.datasets import load_wine, load_diabetes, load_breast_cancer, load_digits
from sklearn.metrics import (accuracy_score, adjusted_rand_score, mean_squared_error,
    confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, matthews_corrcoef, cohen_kappa_score,
    precision_recall_curve, log_loss, brier_score_loss)
from scipy.stats import norm, loguniform

# %% [markdown]
# 1. Supervised vs unsupervised on wine, with and without scaling

# %%
# 1. Supervised vs unsupervised on wine, with and without scaling
Xw, yw = load_wine(return_X_y=True)                          # 178 wines, 13 measurements, 3 cultivars
Xtr, Xte, ytr, yte = train_test_split(Xw, yw, test_size=0.3, stratify=yw, random_state=0)
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(Xtr, ytr)
print(round(accuracy_score(yte, clf.predict(Xte)), 3))       # expected output: 1.0
# clustering never sees y; comparing afterward is legitimate because the labels only score, never fit
raw = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(Xw)
scaled = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(StandardScaler().fit_transform(Xw))
print(round(adjusted_rand_score(yw, raw), 3), round(adjusted_rand_score(yw, scaled), 3))  # expected output: 0.371 0.897
# proline spans about 1402 units while most features span under 20, so raw Euclidean distance is
# almost entirely proline; standardizing gives every measurement an equal vote
print(np.round(Xw.max(0) - Xw.min(0), 1).max(), np.sort(np.round(Xw.max(0) - Xw.min(0), 1))[-2])  # expected output: 1402.0 92.0

# %% [markdown]
# 2. Empirical risk vs true risk on a different 45-car sample

# %%
# 2. Empirical risk vs true risk on a different 45-car sample
mpg = sns.load_dataset("mpg").dropna(subset=["horsepower", "mpg"])
sub = mpg.sample(n=45, random_state=2)                       # the chapter used random_state=1
Xh = sub[["horsepower"]].to_numpy(float); yh = sub["mpg"].to_numpy(float)
Xtr, Xte, ytr, yte = train_test_split(Xh, yh, test_size=0.35, random_state=0)
best_d, best_te = None, np.inf
for d in [1, 2, 3, 5, 9, 15]:
    m = make_pipeline(PolynomialFeatures(d), StandardScaler(), LinearRegression()).fit(Xtr, ytr)
    tr = mean_squared_error(ytr, m.predict(Xtr)); te = mean_squared_error(yte, m.predict(Xte))
    if te < best_te: best_te, best_d = te, d
    print(f"deg{d:2d} train {tr:8.3f} test {te:12.3f}")
# expected output: deg 1 train   22.739 test       34.043
# expected output: deg 2 train   17.275 test       29.116
# expected output: deg 3 train   16.893 test       29.399
# expected output: deg 5 train   15.619 test       32.687
# expected output: deg 9 train   14.648 test       37.005
# expected output: deg15 train   14.131 test      444.527
print("best test degree:", best_d)                           # expected output: best test degree: 2
sub1 = mpg.sample(n=45, random_state=1)                      # the chapter's sample, to fit the degree 2 that 4.3.3 skipped
X1 = sub1[["horsepower"]].to_numpy(float); y1 = sub1["mpg"].to_numpy(float)
X1tr, X1te, y1tr, y1te = train_test_split(X1, y1, test_size=0.35, random_state=0)
for d in [2, 3]:
    m = make_pipeline(PolynomialFeatures(d), StandardScaler(), LinearRegression()).fit(X1tr, y1tr)
    print(f"chapter sample deg{d} test {mean_squared_error(y1te, m.predict(X1te)):.3f}")
# expected output: chapter sample deg2 test 18.897
# expected output: chapter sample deg3 test 17.026
# training error still falls at every step; test error is U-shaped again, but its minimum moved from
# degree 3 to degree 2. On the chapter's own sample degree 2 scores 18.897, above degree 3's 17.026, so the
# move is real and not an artifact of the grid 4.3.3 used: the balance point depends on the sample, not on a
# universal degree

# %% [markdown]
# 3. Bias and variance against the amount of data

# %%
# 3. Bias and variance against the amount of data
true_f = lambda x: np.sin(1.5 * x)
x_test = np.linspace(-3, 3, 60).reshape(-1, 1); y_true = true_f(x_test).ravel()
for n in [25, 100, 400]:
    rng = np.random.default_rng(0)
    cells = []
    for d in [1, 5, 9]:
        preds = []
        for _ in range(200):
            Xs = rng.uniform(-3, 3, n).reshape(-1, 1)
            ys = true_f(Xs).ravel() + rng.normal(0, 0.3, n)
            preds.append(make_pipeline(PolynomialFeatures(d), StandardScaler(),
                                       LinearRegression()).fit(Xs, ys).predict(x_test))
        preds = np.array(preds)
        b2 = np.mean((preds.mean(0) - y_true) ** 2); v = np.mean(preds.var(0))
        cells.append(f"deg{d} bias2 {b2:.3f} var {v:.3f}")
    print(f"n={n:3d}: " + " | ".join(cells))
# expected output: n= 25: deg1 bias2 0.487 var 0.053 | deg5 bias2 0.003 var 0.121 | deg9 bias2 0.966 var 669.743
# expected output: n=100: deg1 bias2 0.485 var 0.011 | deg5 bias2 0.002 var 0.007 | deg9 bias2 0.000 var 0.019
# expected output: n=400: deg1 bias2 0.485 var 0.003 | deg5 bias2 0.002 var 0.002 | deg9 bias2 0.000 var 0.003
# bias squared is set by the degree and barely moves with n; variance falls as n grows. At n = 25 the
# degree-nine fit has ten coefficients for 25 noisy points and swings wildly between samples (variance
# 670), which also drags its average model off the truth, so even its measured bias squared is large

# %% [markdown]
# 4. Bayes floor with unequal priors 0.3 and 0.7

# %%
# 4. Bayes floor with unequal priors 0.3 and 0.7
pi0, pi1 = 0.3, 0.7
x_star = np.log(pi0 / pi1) / 2        # by hand: pi0 N(x;-1,1) = pi1 N(x;+1,1)  =>  ln(pi0/pi1) = 2x
bayes = pi0 * (1 - norm.cdf(x_star, -1, 1)) + pi1 * norm.cdf(x_star, 1, 1)
print(round(x_star, 4), round(bayes, 4))                     # expected output: -0.4236 0.1387
rng = np.random.default_rng(4); n = 50000; n0 = int(n * pi0); n1 = n - n0
Xg = np.concatenate([rng.normal(-1, 1, n0), rng.normal(1, 1, n1)]).reshape(-1, 1)
yg = np.concatenate([np.zeros(n0), np.ones(n1)]).astype(int)
Xtr, Xte, ytr, yte = train_test_split(Xg, yg, test_size=0.3, random_state=4, stratify=yg)
clf = LogisticRegression().fit(Xtr, ytr)
print(round(1 - clf.score(Xte, yte), 4))                     # expected output: 0.1349
print(round(-clf.intercept_[0] / clf.coef_[0, 0], 4))        # expected output: -0.4242
# -0.4242 is the learned boundary, beside the hand-derived -0.4236
print(round(np.sqrt(bayes * (1 - bayes) / len(yte)), 4))     # expected output: 0.0028
# the boundary moved toward the rarer class and the floor fell from 0.1587 to 0.1387; the test error
# sits 0.0038 below the floor, about 1.4 standard errors, which a finite sample allows

# %% [markdown]
# 5. Choosing alpha by cross-validation on diabetes

# %%
# 5. Choosing alpha by cross-validation on diabetes
dd = load_diabetes()
Xs = StandardScaler().fit_transform(dd.data); yv = dd.target
alphas = np.logspace(-2, 2, 41)
lcv = LassoCV(alphas=alphas, cv=5).fit(Xs, yv)
rcv = RidgeCV(alphas=alphas, cv=5).fit(Xs, yv)
kept = [f for f, c in zip(dd.feature_names, lcv.coef_) if abs(c) > 1e-6]
print(round(lcv.alpha_, 3), len(kept), [f for f in dd.feature_names if f not in kept])  # expected output: 0.079 9 ['s3']
print(round(rcv.alpha_, 3))                                  # expected output: 0.158
for name, m in [("OLS", LinearRegression()), ("Lasso", Lasso(alpha=lcv.alpha_)), ("Ridge", Ridge(alpha=rcv.alpha_))]:
    mse = -cross_val_score(m, Xs, yv, cv=5, scoring="neg_mean_squared_error").mean()
    print(f"{name:6s} 5-fold MSE {mse:.1f}")
# expected output: OLS    5-fold MSE 2993.1
# expected output: Lasso  5-fold MSE 2991.8
# expected output: Ridge  5-fold MSE 2993.0
# with 442 rows and only 10 features the least-squares fit is already stable, so cross-validation
# picks tiny penalties and all three errors agree to within 0.05 percent; the chapter's alpha=3 was
# chosen to make shrinkage and selection visible in the coefficients, not to minimize error

# %% [markdown]
# 6. Validation curve over C at the chapter's gamma

# %%
# 6. Validation curve over C at the chapter's gamma
Xb, yb = load_breast_cancer(return_X_y=True)
Cs = np.logspace(-3, 3, 7)
pipe = make_pipeline(StandardScaler(), SVC(gamma=0.01))
tr, va = validation_curve(pipe, Xb, yb, param_name="svc__C", param_range=Cs, cv=5, scoring="accuracy")
for c, t, v in zip(Cs, tr.mean(1), va.mean(1)):
    print(f"C {c:9.3f} train {t:.3f} val {v:.3f}")
# expected output: C     0.001 train 0.627 val 0.627
# expected output: C     0.010 train 0.627 val 0.627
# expected output: C     0.100 train 0.952 val 0.951
# expected output: C     1.000 train 0.978 val 0.968
# expected output: C    10.000 train 0.988 val 0.979
# expected output: C   100.000 train 0.994 val 0.968
# expected output: C  1000.000 train 1.000 val 0.965
# underfit at C <= 0.01 (0.627 is the majority-class rate: the heavily regularized SVM predicts one
# class), sweet spot near C = 10, mild overfit by C = 1000 (train 1.000, val 0.965). Raising C only
# loosens the margin penalty, while raising gamma shrinks every kernel bump until the boundary wraps
# individual points, which is why gamma's overfit side collapsed to 0.631 and C's barely dips

# %% [markdown]
# 7. Full metric report on a more imbalanced Titanic set: 60 survivors instead of the chapter's 95

# %%
# 7. Full metric report on a more imbalanced Titanic set: 60 survivors instead of the chapter's 95
df = sns.load_dataset("titanic"); feats = ["pclass", "age", "fare", "sibsp", "parch"]
died, surv = df[df.survived == 0], df[df.survived == 1]
imb = pd.concat([died, surv.sample(n=60, random_state=0)]).reset_index(drop=True)
X, yv = imb[feats], imb["survived"]
print(imb.shape, round(yv.mean(), 3))                        # expected output: (609, 15) 0.099
Xtr, Xte, ytr, yte = train_test_split(X, yv, test_size=0.3, random_state=0, stratify=yv)
imp = SimpleImputer(strategy="median").fit(Xtr)
clf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=2).fit(imp.transform(Xtr), ytr)
pred = clf.predict(imp.transform(Xte)); proba = clf.predict_proba(imp.transform(Xte))[:, 1]
tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
print("tp fp fn tn:", tp, fp, fn, tn)                        # expected output: tp fp fn tn: 1 5 17 160
print(round(accuracy_score(yte, pred), 3), round(precision_score(yte, pred), 3),
      round(recall_score(yte, pred), 3), round(f1_score(yte, pred), 3))  # expected output: 0.88 0.167 0.056 0.083
print(round(roc_auc_score(yte, proba), 3), round(average_precision_score(yte, proba), 3),
      round(yte.mean(), 3))                                  # expected output: 0.681 0.278 0.098
# the third number, the survivors' share of the test set, is average precision's chance level
print(round(matthews_corrcoef(yte, pred), 3), round(cohen_kappa_score(yte, pred), 3))       # expected output: 0.042 0.036
print(round(max(yte.mean(), 1 - yte.mean()), 3))             # expected output: 0.902
pipe7 = make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=2))
oof = cross_val_predict(pipe7, Xtr, ytr, cv=5, method="predict_proba")[:, 1]   # choose the threshold on validation data
prec, rec, thr = precision_recall_curve(ytr, oof)
f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
bi = int(np.argmax(f1s)); yp7 = (proba >= thr[bi]).astype(int)                 # then score it once on the test set
print("max-F1 thr", round(thr[bi], 2), "test prec", round(precision_score(yte, yp7), 3), "rec", round(recall_score(yte, yp7), 3))
# expected output: max-F1 thr 0.07 test prec 0.169 rec 0.556
# The test set holds 18 survivors among 183 passengers, and the forest catches 1 of them, yet accuracy rose from
# the chapter's 0.845 to 0.880: accuracy hid the extra imbalance, and it still sits below the majority baseline,
# now 0.880 against 0.902 where the chapter had 0.845 against 0.851. The metrics that look at the positives
# exposed it: recall fell from 0.241 to 0.056, F1 from 0.318 to 0.083, MCC from 0.258 to 0.042, and kappa from
# 0.241 to 0.036, both now close to chance, while ROC AUC fell from 0.744 to 0.681. Average precision of 0.278 is
# 2.8 times its chance level of 0.098, against about 2.9 times chance in 4.7.4. The max-F1 cutoff rises to 0.07
# and catches 10 of the 18 survivors (recall 0.556) at precision 0.169, against 0.759 and 0.222 at the chapter's 0.04

# %% [markdown]
# 8. Grid vs random vs nested cross-validation on the wine data, which the chapter never tuned on

# %%
# 8. Grid vs random vs nested cross-validation on the wine data, which the chapter never tuned on
Xw, yw = load_wine(return_X_y=True)                          # 178 wines, 13 measurements, 3 cultivars
pipe = make_pipeline(StandardScaler(), SVC())
grid = {"svc__C": [0.1, 1, 10, 100], "svc__gamma": [1e-3, 1e-2, 1e-1, 1]}
gs = GridSearchCV(pipe, grid, cv=5).fit(Xw, yw)
print(gs.best_params_, round(gs.best_score_, 4))
# expected output: {'svc__C': 10, 'svc__gamma': 0.1} 0.9889
dist = {"svc__C": loguniform(1e-1, 1e3), "svc__gamma": loguniform(1e-4, 1e0)}
rs = RandomizedSearchCV(pipe, dist, n_iter=8, cv=5, random_state=0).fit(Xw, yw)
print(round(rs.best_score_, 4))                              # expected output: 0.9889
biggrid = {"svc__C": np.logspace(-2, 3, 8), "svc__gamma": np.logspace(-4, 1, 8)}
inner = StratifiedKFold(5, shuffle=True, random_state=0)
outer = StratifiedKFold(5, shuffle=True, random_state=0)
tuned = GridSearchCV(pipe, biggrid, cv=inner).fit(Xw, yw)
print(round(tuned.best_score_, 4))                           # expected output: 0.9887
# optimistic: the maximum over 64 noisy fold scores
nested = cross_val_score(GridSearchCV(pipe, biggrid, cv=inner), Xw, yw, cv=outer)
print(round(nested.mean(), 4))                               # expected output: 0.9775
# honest: each tuning run is scored on folds the tuning never saw. The gap is 1.1 points here against the half
# point of 4.8.6 on breast cancer. A validation fold holds about 36 wines, so one wine moves a fold score by almost
# 3 points; the 64 cross-validated scores are noisier than on 569 tumors, and the largest of noisier numbers
# carries a larger lucky bump. One dataset shows the size of the gap here, not a general law

# %% [markdown]
# 9. Baseline plus permutation test on the harder digit-8 detection task

# %%
# 9. Baseline plus permutation test on the harder digit-8 detection task
digits = load_digits()
Xc, yc = digits.data, (digits.target == 8).astype(int)
print(round(yc.mean(), 3))                                   # expected output: 0.097
base = cross_val_score(DummyClassifier(strategy="most_frequent"), Xc, yc, cv=5).mean()
lr = LogisticRegression(max_iter=5000)
scores = cross_val_score(lr, Xc, yc, cv=5)
print(round(base, 3), "%.3f +/- %.3f" % (scores.mean(), scores.std()))  # expected output: 0.903 0.942 +/- 0.009
score, perm, pval = permutation_test_score(lr, Xc, yc, cv=5, n_permutations=200,
                                           random_state=0, scoring="accuracy")
print("%.3f  %.3f  %.3f  %.4f" % (score, perm.mean(), perm.max(), pval))  # expected output: 0.942  0.901  0.904  0.0050
# The gap is statistically real: no shuffled run reached 0.942 (the best of the 200 managed 0.904), so the p-value
# sits at 1/201 = 0.0050, the floor that 200 shuffles allow. The margin over the 0.903 baseline is 3.9 points, about
# half the digit-3 detector's 7.5 (0.973 against 0.898 in 4.9), but the folds agree more closely, a standard
# deviation of 0.009 against 0.020 in 4.9.1, and the permutation test reaches the same floor for both tasks

# %% [markdown]
# 10. Miniproject: end-to-end model selection on breast cancer

# %%
# 10. Miniproject: end-to-end model selection on breast cancer
Xb, yb = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(Xb, yb, test_size=0.3, random_state=0, stratify=yb)
cands = {
    "baseline": DummyClassifier(strategy="most_frequent"),
    "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
    "rbf_svc": make_pipeline(StandardScaler(), CalibratedClassifierCV(SVC(random_state=0), ensemble=False)),  # probabilities without the deprecated probability=True
    "rf": RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=2),
}
cvscores = {k: float(round(cross_val_score(v, Xtr, ytr, cv=5, scoring="roc_auc").mean(), 4))
            for k, v in cands.items()}
print(cvscores)  # expected output: {'baseline': 0.5, 'logreg': 0.9921, 'rbf_svc': 0.9923, 'rf': 0.9939}
winner = max((k for k in cands if k != "baseline"), key=lambda k: cvscores[k])
print("winner:", winner)                                     # expected output: winner: rf
best = cands[winner].fit(Xtr, ytr); p = best.predict_proba(Xte)[:, 1]
print(round(roc_auc_score(yte, p), 4), round(log_loss(yte, p), 4), round(brier_score_loss(yte, p), 4))
# expected output: 0.9782 0.3311 0.0404

# %% [markdown]
# 11. Learning curves for three classifiers on breast cancer

# %%
# 11. Learning curves for three classifiers on breast cancer
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
Xb, yb = load_breast_cancer(return_X_y=True)                 # 569 tumors, 30 features; 0.627 are benign
for name, model in [("logreg C=1e-4", make_pipeline(StandardScaler(), LogisticRegression(C=1e-4, max_iter=5000))),
                    ("logreg C=1   ", make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=5000))),
                    ("1-NN         ", make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=1)))]:
    sizes, tr, va = learning_curve(model, Xb, yb, train_sizes=np.linspace(0.2, 1.0, 5), cv=5,
                                   scoring="accuracy", shuffle=True, random_state=0)   # shuffle: the rows are not in random order
    print(name, " ".join(f"n={n}: {t:.3f}/{v:.3f}" for n, t, v in zip(sizes, tr.mean(axis=1), va.mean(axis=1))))
# expected output: logreg C=1e-4 n=91: 0.653/0.627 n=182: 0.643/0.627 n=273: 0.634/0.627 n=364: 0.636/0.638 n=455: 0.644/0.643
# expected output: logreg C=1    n=91: 0.996/0.970 n=182: 0.989/0.979 n=273: 0.989/0.975 n=364: 0.988/0.979 n=455: 0.989/0.981
# expected output: 1-NN          n=91: 1.000/0.931 n=182: 1.000/0.937 n=273: 1.000/0.947 n=364: 1.000/0.951 n=455: 1.000/0.954
# C=1e-4 is underfit: training and validation accuracy sit close together near the 0.627 majority share at
# every size (training 0.634 to 0.653, validation 0.627 to 0.643), the both-low, converged-high-error
# signature, and more data barely moves a model this constrained. C=1 is the good fit: training near 0.99,
# validation climbing from 0.970 to 0.981 as the gap closes, and more data would still help it a little, since
# validation accuracy was still rising at the largest size. 1-NN is overfit: training accuracy is a perfect
# 1.000 at every size because each training point is its own nearest neighbor, while validation trails by 5 to
# 7 points; the gap is narrowing (0.931 to 0.954), so more data still helps it

# %% [markdown]
# 12. Calibration repair for LinearSVC: Platt scaling against isotonic regression

# %%
# 12. Calibration repair for LinearSVC: Platt scaling against isotonic regression
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss
Xtr, Xte, ytr, yte = train_test_split(Xb, yb, test_size=0.3, stratify=yb, random_state=0)   # 398 training rows
for method in ["sigmoid", "isotonic"]:
    cal = make_pipeline(StandardScaler(), CalibratedClassifierCV(LinearSVC(random_state=0), method=method, cv=5)).fit(Xtr, ytr)
    p = cal.predict_proba(Xte)[:, 1]
    frac_pos, mean_pred = calibration_curve(yte, p, n_bins=5, strategy="quantile")
    print(method, "predicted", np.round(mean_pred, 2), "actual", np.round(frac_pos, 2),
          "Brier", round(brier_score_loss(yte, p), 4), "log loss", round(log_loss(yte, p), 3))
# expected output: sigmoid predicted [0.01 0.31 0.85 0.95 0.99] actual [0. 0.21 0.94 1. 1. ] Brier 0.0312 log loss 0.128
# expected output: isotonic predicted [0. 0.36 0.93 0.99 1. ] actual [0. 0.41 0.94 1. 1. ] Brier 0.0269 log loss 0.089
# one split is thin evidence with 171 test rows, so repeat over ten splits before choosing
scores = {"sigmoid": [], "isotonic": []}
for rs in range(10):
    Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(Xb, yb, test_size=0.3, stratify=yb, random_state=rs)
    for method in scores:
        cal = make_pipeline(StandardScaler(), CalibratedClassifierCV(LinearSVC(random_state=0), method=method, cv=5)).fit(Xtr_r, ytr_r)
        p = cal.predict_proba(Xte_r)[:, 1]
        scores[method].append((brier_score_loss(yte_r, p), log_loss(yte_r, p)))
for method, v in scores.items():
    v = np.array(v)
    print(method, "mean Brier over 10 splits", round(v[:, 0].mean(), 4), "mean log loss", round(v[:, 1].mean(), 3))
print("isotonic has the lower Brier score on", int(np.sum(np.array(scores["isotonic"])[:, 0] < np.array(scores["sigmoid"])[:, 0])), "of 10 splits")
print("isotonic has the lower log loss on", int(np.sum(np.array(scores["isotonic"])[:, 1] < np.array(scores["sigmoid"])[:, 1])), "of 10 splits")
# expected output: sigmoid mean Brier over 10 splits 0.0293 mean log loss 0.121
# expected output: isotonic mean Brier over 10 splits 0.0238 mean log loss 0.218
# expected output: isotonic has the lower Brier score on 9 of 10 splits
# expected output: isotonic has the lower log loss on 7 of 10 splits
# on this split isotonic wins both metrics, and over ten splits it wins Brier nine times out of ten and log
# loss seven times out of ten, so by Brier score it is the better choice even at 398 rows. Log loss reverses
# the verdict on average (0.218 against 0.121): the isotonic step function emits exact 0 and 1 probabilities,
# and one such case on the wrong side costs about 36 nats after clipping (4.7.6), which is exactly the
# small-data risk the rule of thumb warns about. Choose by the metric the probabilities will be judged on:
# isotonic if squared error, sigmoid if log loss or if a hard 0 or 1 must never be emitted

# %% [markdown]
# 13. Expected improvement by hand, then Optuna and HalvingRandomSearchCV on the digits data

# %%
# 13. Expected improvement by hand, then Optuna and HalvingRandomSearchCV on the digits data
from scipy.stats import norm, loguniform
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.datasets import load_digits
f_star = 0.978                                               # best score so far
for name, mu, sd in [("A", 0.980, 0.001), ("B", 0.975, 0.008)]:
    z = (mu - f_star) / sd                                   # A: 0.002/0.001 = 2.0 ; B: -0.003/0.008 = -0.375
    ei = (mu - f_star) * norm.cdf(z) + sd * norm.pdf(z)      # exploitation term plus exploration term
    print(name, "z", round(z, 3), "Phi", round(norm.cdf(z), 4), "phi", round(norm.pdf(z), 4), "EI", round(ei, 5))
# expected output: A z 2.0 Phi 0.9772 phi 0.054 EI 0.00201
# expected output: B z -0.375 Phi 0.3538 phi 0.3719 EI 0.00191
# by hand: A = 0.002 x 0.9772 + 0.001 x 0.0540 = 0.001954 + 0.000054 = 0.00201, almost all exploitation;
# B = -0.003 x 0.3538 + 0.008 x 0.3719 = -0.001061 + 0.002975 = 0.00191, all exploration. A is tried first, but
# B's wide uncertainty keeps it within 5 percent of A even though its mean is below the incumbent
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
Xd, yd = load_digits(return_X_y=True)                        # 1797 digits, 64 pixels, 10 classes
def objective(trial):
    C = trial.suggest_float("C", 1e-1, 1e3, log=True)
    gamma = trial.suggest_float("gamma", 1e-4, 1e0, log=True)
    return cross_val_score(make_pipeline(StandardScaler(), SVC(C=C, gamma=gamma)), Xd, yd, cv=5).mean()
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
study.optimize(objective, n_trials=25)
print(round(study.best_value, 4), {k: float(round(v, 4)) for k, v in study.best_params.items()}, "found at trial", study.best_trial.number)
print("best of the 10 random startup trials:", round(max(t.value for t in study.trials[:10]), 4))
# expected output: 0.9571 {'C': 11.369, 'gamma': 0.0055} found at trial 18
# expected output: best of the 10 random startup trials: 0.9544
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingRandomSearchCV
dist = {"svc__C": loguniform(1e-1, 1e3), "svc__gamma": loguniform(1e-4, 1e0)}
hs = HalvingRandomSearchCV(make_pipeline(StandardScaler(), SVC()), dist, factor=3, cv=5, random_state=0).fit(Xd, yd)
print(round(hs.best_score_, 4), hs.n_candidates_, list(hs.n_resources_), {k: float(round(v, 4)) for k, v in hs.best_params_.items()})
print("the halving winner cross-validated on all 1797 digits:", round(cross_val_score(hs.best_estimator_, Xd, yd, cv=5).mean(), 4))
# expected output: 0.9409 [17, 6, 2] [100, 300, 900] {'svc__C': 12.2291, 'svc__gamma': 0.0046}
# expected output: the halving winner cross-validated on all 1797 digits: 0.956
# Optuna's guided phase improved on its random startup (0.9544 to 0.9571, at trial 18, the ninth guided trial;
# this run used Optuna 5.0.0, the version in the reference environment).
# Halving screened 17 candidates on 100 digits, 6 on 300, and 2 on 900; its best_score_ of 0.9409 is measured on
# the 900-row rung, not on the full data, which is why it looks lower; refit and cross-validated on all 1797
# digits the same setting scores 0.956, about a tenth of a point below Optuna's 0.9571

# %% [markdown]
# 14. Early stopping on a different forty tumors; augmentation at three noise scales

# %%
# 14. Early stopping on a different forty tumors; augmentation at three noise scales
from sklearn.linear_model import SGDClassifier, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import seaborn as sns
X_grow, X_hold, y_grow, y_hold = train_test_split(Xb, yb, test_size=0.3, random_state=0, stratify=yb)
def early_stopping_run(fit_seed, sgd_seed):
    X_fit, X_val, y_fit, y_val = train_test_split(X_grow, y_grow, train_size=40, random_state=fit_seed, stratify=y_grow)
    sc = StandardScaler().fit(X_fit)
    Z_fit, Z_val, Z_hold = sc.transform(X_fit), sc.transform(X_val), sc.transform(X_hold)
    sgd = SGDClassifier(loss="log_loss", penalty=None, learning_rate="constant", eta0=0.01,
                        max_iter=1, tol=None, warm_start=True, random_state=sgd_seed)
    hist = []
    for epoch in range(1, 2001):
        sgd.fit(Z_fit, y_fit)
        hist.append((epoch, np.linalg.norm(sgd.coef_), log_loss(y_val, sgd.predict_proba(Z_val)[:, 1], labels=[0, 1]),
                     log_loss(y_hold, sgd.predict_proba(Z_hold)[:, 1], labels=[0, 1])))
    best = min(hist, key=lambda h: h[2])
    print(f"fit seed {fit_seed} sgd seed {sgd_seed}: stop at epoch {best[0]:3d} ||w|| {best[1]:.2f} test loss {best[3]:.3f}"
          f" | epoch 2000 ||w|| {hist[-1][1]:.2f} test loss {hist[-1][3]:.3f}")
for fit_seed in [0, 1, 2]:
    early_stopping_run(fit_seed, 0)
early_stopping_run(0, 1)                                     # the chapter's forty tumors, a different shuffle order
# expected output: fit seed 0 sgd seed 0: stop at epoch 55 ||w|| 2.20 test loss 0.146 | epoch 2000 ||w|| 6.56 test loss 0.286
# expected output: fit seed 1 sgd seed 0: stop at epoch 35 ||w|| 1.82 test loss 0.209 | epoch 2000 ||w|| 6.26 test loss 0.407
# expected output: fit seed 2 sgd seed 0: stop at epoch 116 ||w|| 2.43 test loss 0.179 | epoch 2000 ||w|| 5.02 test loss 0.284
# expected output: fit seed 0 sgd seed 1: stop at epoch 55 ||w|| 2.20 test loss 0.149 | epoch 2000 ||w|| 6.56 test loss 0.286
# the stopping epoch depends on which forty tumors were drawn (35, 55, or 116 epochs) but the weight norm at the
# stop stays small (1.8 to 2.4) and the test loss at the stop beats the 2000-epoch loss every time, by 0.10 to
# 0.20 nats (the four gains are 0.140, 0.198, 0.105, and 0.137); changing only the SGD shuffle order leaves the
# stopping epoch at 55 and the norm at 2.20
mpg = sns.load_dataset("mpg").dropna(subset=["horsepower", "mpg"])
sub = mpg.sample(n=45, random_state=1)                       # the chapter's 45 cars and the 4.6.1 split
Xm = sub[["horsepower"]].to_numpy(float); ym = sub["mpg"].to_numpy(float)
Xm_tr, Xm_te, ym_tr, ym_te = train_test_split(Xm, ym, test_size=0.35, random_state=0)
wild = make_pipeline(PolynomialFeatures(15), StandardScaler(), LinearRegression())
print("no augmentation", round(mean_squared_error(ym_te, wild.fit(Xm_tr, ym_tr).predict(Xm_te)), 1))
for scale in [1.0, 3.0, 10.0]:
    rng = np.random.default_rng(0)
    X_aug = np.vstack([Xm_tr + rng.normal(0, scale, Xm_tr.shape) for _ in range(20)] + [Xm_tr])
    y_aug = np.concatenate([ym_tr] * 21)
    print("jitter scale", scale, "test MSE", round(mean_squared_error(ym_te, wild.fit(X_aug, y_aug).predict(Xm_te)), 1),
          "train MSE", round(mean_squared_error(ym_tr, wild.predict(Xm_tr)), 1))
# expected output: no augmentation 370469.9
# expected output: jitter scale 1.0 test MSE 167233.0 train MSE 6.4
# expected output: jitter scale 3.0 test MSE 21175.5 train MSE 8.2
# expected output: jitter scale 10.0 test MSE 24.4 train MSE 10.5
# the bigger the jitter, the wider the neighborhood over which the curve must behave: 1 hp barely helps,
# 3 hp is the chapter's seventeen-fold gain, and 10 hp brings the degree-fifteen fit down to 24.4, about the
# ridge penalty's 28.4, while training MSE rises from 6.4 to 10.5, the regularization trade of 4.6.5. The limit
# is the label-preserving assumption: at 10 hp the jitter already exceeds the typical gap between neighboring
# training cars (about 5 hp on average, 3.5 hp median), so a larger scale would start to blur real differences
# in fuel economy

# %% [markdown]
# 15. Five percent of the digit labels: supervised, self-training, label spreading; then even-to-odd transfer

# %%
# 15. Five percent of the digit labels: supervised, self-training, label spreading; then even-to-odd transfer
from sklearn.semi_supervised import SelfTrainingClassifier, LabelSpreading
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(Xd, yd, test_size=0.3, stratify=yd, random_state=0)
rng = np.random.default_rng(1)
unlabeled = rng.random(len(yd_tr)) < 0.95                    # hide 95 percent of the training labels
y_semi = yd_tr.copy(); y_semi[unlabeled] = -1
print("labeled", int((y_semi != -1).sum()), "of", len(yd_tr), "classes present", len(np.unique(yd_tr[~unlabeled])))
sup = LogisticRegression(max_iter=2000).fit(Xd_tr[~unlabeled], yd_tr[~unlabeled])
semi = SelfTrainingClassifier(LogisticRegression(max_iter=2000)).fit(Xd_tr, y_semi)
spread = LabelSpreading(kernel="knn", n_neighbors=10).fit(Xd_tr, y_semi)   # a 10-nearest-neighbor graph over all 1257 digits
print("supervised", round(accuracy_score(yd_te, sup.predict(Xd_te)), 3),
      "self-training", round(accuracy_score(yd_te, semi.predict(Xd_te)), 3),
      "label spreading", round(accuracy_score(yd_te, spread.predict(Xd_te)), 3))
# expected output: labeled 73 of 1257 classes present 10
# expected output: supervised 0.824 self-training 0.85 label spreading 0.9
# with 73 labels the supervised model reaches 0.824, self-training adds 2.6 points, and label spreading adds
# 7.6, because digits form tight clusters in pixel space and a graph that follows the clusters carries each
# label through its whole cluster, while self-training can only extend the logistic boundary it already has
digits_src = yd % 2 == 0                                     # source task: the even digits 0, 2, 4, 6, 8
body = MLPClassifier(hidden_layer_sizes=(64,), max_iter=1000, random_state=0).fit(Xd[digits_src], yd[digits_src])
def features(Xq):                                            # the frozen hidden layer
    return np.maximum(0, Xq @ body.coefs_[0] + body.intercepts_[0])
X_odd, y_odd = Xd[~digits_src], yd[~digits_src]              # target task: the odd digits 1, 3, 5, 7, 9
for n_labels in (20, 30, 50):
    X_tr, X_te, y_tr, y_te = train_test_split(X_odd, y_odd, train_size=n_labels, random_state=0, stratify=y_odd)
    scratch = MLPClassifier(hidden_layer_sizes=(64,), max_iter=2000, random_state=0).fit(X_tr, y_tr)
    head = LogisticRegression(max_iter=5000).fit(features(X_tr), y_tr)
    print(f"{n_labels} labels: from scratch {scratch.score(X_te, y_te):.3f}   frozen body + new head {head.score(features(X_te), y_te):.3f}")
# expected output: 20 labels: from scratch 0.658 frozen body + new head 0.865
# expected output: 30 labels: from scratch 0.727 frozen body + new head 0.879
# expected output: 50 labels: from scratch 0.868 frozen body + new head 0.915
# the even-digit body transfers to the odd digits as well as the 0-4 body transferred to 5-9 in 4.2.2: at 20
# labels the frozen features beat training from scratch by 21 points, and the gap narrows to 5 points at 50
# labels as the from-scratch network gets enough data to learn its own features

# %% [markdown]
# 16. Double descent on a different 45-car sample

# %%
# 16. Double descent on a different 45-car sample
sub2 = mpg.sample(n=45, random_state=2)
X2 = sub2[["horsepower"]].to_numpy(float); y2 = sub2["mpg"].to_numpy(float)
X2_tr, X2_te, y2_tr, y2_te = train_test_split(X2, y2, test_size=0.35, random_state=0)
lo, hi = 46.0, 230.0                                         # horsepower's range in the full mpg data
def legendre_features(Xcol, D):
    t = 2 * (Xcol.ravel() - lo) / (hi - lo) - 1
    return np.polynomial.legendre.legvander(t, D)
hp = X2_tr.ravel()
group_mean = np.array([y2_tr[hp == v].mean() for v in hp])
print(len(hp), len(np.unique(hp)), round(np.mean((y2_tr - group_mean) ** 2), 2))
# expected output: 29 21 3.93
y_mean = y2_tr.mean()
for D in [1, 2, 3, 9, 15, 20, 21, 22, 25, 40, 60, 100, 300, 1000]:
    A_tr, A_te = legendre_features(X2_tr, D), legendre_features(X2_te, D)
    coef = np.linalg.pinv(A_tr) @ (y2_tr - y_mean)
    print(f"degree {D:4d}  train MSE {mean_squared_error(y2_tr, A_tr @ coef + y_mean):6.2f}"
          f"  test MSE {mean_squared_error(y2_te, A_te @ coef + y_mean):12.4g}  norm {np.linalg.norm(coef):10.4g}")
# expected output: degree 1 train MSE 22.74 test MSE 34.04 norm 14.2
# expected output: degree 2 train MSE 17.27 test MSE 29.12 norm 14.45
# expected output: degree 3 train MSE 16.89 test MSE 29.4 norm 14.16
# expected output: degree 9 train MSE 14.13 test MSE 315.3 norm 968.7
# expected output: degree 15 train MSE 9.53 test MSE 6.208e+10 norm 3.136e+08
# expected output: degree 20 train MSE 5.26 test MSE 2.59e+19 norm 1.794e+12
# expected output: degree 21 train MSE 3.93 test MSE 1.662e+22 norm 6.168e+13
# expected output: degree 22 train MSE 3.93 test MSE 1.202e+21 norm 2.055e+12
# expected output: degree 25 train MSE 3.93 test MSE 1.157e+17 norm 1.858e+09
# expected output: degree 40 train MSE 3.93 test MSE 5.841e+06 norm 1.546e+04
# expected output: degree 60 train MSE 3.93 test MSE 260.6 norm 63.8
# expected output: degree 100 train MSE 3.93 test MSE 41.71 norm 23.15
# expected output: degree 300 train MSE 3.93 test MSE 36.75 norm 18.48
# expected output: degree 1000 train MSE 3.93 test MSE 38.25 norm 16.56
# this sample's 29 training cars carry 21 distinct horsepower values, so the training floor is 3.93. In exact
# arithmetic degree 20, with 21 coefficients, would already reach it, but its Legendre design matrix has
# numerical rank 20 (matrix_rank first reports 21 at degree 21), so the minimum-norm fit stops at 5.26, the
# effect 4.5.4 describes at degrees 23 and 25. The interpolation threshold is therefore degree 21, the first
# degree whose fit reaches the floor, and the test MSE peaks there at about 1.66e+22 (the last digits of that
# row can differ between machines and NumPy builds: the smallest singular value the minimum-norm fit keeps
# there is only about 1.7e-14 of the largest, so tiny rounding differences are magnified). The second descent
# arrives (260.6 at degree 60, 41.7 at 100, 36.8 at 300) but never drops below the classical minimum of 29.12
# at degree 2, unlike the chapter's sample, where degree 100 beat the sweet spot. Double descent is a shape,
# not a guarantee that the overparameterized side wins

# %% [markdown]
# 17. Macro, weighted, and micro F1 with one rare class on the wine data

# %%
# 17. Macro, weighted, and micro F1 with one rare class on the wine data
from sklearn.datasets import load_wine
from sklearn.metrics import f1_score, top_k_accuracy_score
Xw, yw = load_wine(return_X_y=True)                          # class counts 59, 71, 48
rng = np.random.default_rng(0)
idx2 = np.where(yw == 2)[0]
keep = np.ones(len(yw), bool); keep[rng.choice(idx2, size=len(idx2) - 10, replace=False)] = False   # keep 10 of class 2
Xs, ys = Xw[keep], yw[keep]
Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(Xs, ys, test_size=0.3, stratify=ys, random_state=0)
print("class counts after thinning", np.bincount(ys), "test", np.bincount(ys_te))
knn = KNeighborsClassifier(n_neighbors=5).fit(Xs_tr, ys_tr)  # raw features, so proline dominates the distance (exercise 1)
p = knn.predict(Xs_te)
print("per-class F1", np.round(f1_score(ys_te, p, average=None), 3))
print("macro", round(f1_score(ys_te, p, average="macro"), 3), "weighted", round(f1_score(ys_te, p, average="weighted"), 3),
      "micro", round(f1_score(ys_te, p, average="micro"), 3), "top-2", round(top_k_accuracy_score(ys_te, knn.predict_proba(Xs_te), k=2), 3))
# expected output: class counts after thinning [59 71 10] test [18 21 3]
# expected output: per-class F1 [0.947 0.884 0. ]
# expected output: macro 0.61 weighted 0.848 micro 0.881 top-2 0.952
# the rare cultivar scores F1 of exactly 0: all three of its test wines are called something else. Macro F1
# averages the three per-class scores with equal weight, (0.947 + 0.884 + 0)/3 = 0.61, so the failure counts
# for a full third; weighted F1 weights by support, (18 x 0.947 + 21 x 0.884 + 3 x 0)/42 = 0.848, so the three
# lost wines cost only 3/42 of the score; micro F1 equals accuracy, 37/42 = 0.881. Top-2 accuracy is 0.952
# because the true class is usually the second guess when the first is wrong

# %% [markdown]
# 18. Error analysis for a 3-nearest-neighbor digit classifier, with the worst confusion saved as a figure

# %%
# 18. Error analysis for a 3-nearest-neighbor digit classifier, with the worst confusion saved as a figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
knn3 = KNeighborsClassifier(n_neighbors=3).fit(Xd_tr, yd_tr) # the 4.9.2 split of the digits
pk = knn3.predict(Xd_te)
cm = confusion_matrix(yd_te, pk)
pairs = sorted([(int(cm[i, j]), i, j) for i in range(10) for j in range(10) if i != j and cm[i, j] > 0], reverse=True)
print("accuracy", round(accuracy_score(yd_te, pk), 3), "top confusions (count, true, predicted):", pairs[:3])
print("misclassified per true digit:", [int(cm[i].sum() - cm[i, i]) for i in range(10)])
# expected output: accuracy 0.985 top confusions (count, true, predicted): [(2, 9, 3), (2, 8, 1), (1, 8, 3)]
# expected output: misclassified per true digit: [0, 0, 0, 1, 2, 0, 0, 0, 3, 2]
count, i, j = pairs[0]
wrong = np.where((yd_te == i) & (pk == j))[0]
fig, axes = plt.subplots(1, len(wrong), figsize=(1.8 * len(wrong), 2.0))
for ax, w in zip(np.atleast_1d(axes), wrong):
    ax.imshow(Xd_te[w].reshape(8, 8), cmap="gray_r"); ax.set_title(f"true {i}, predicted {j}", fontsize=8); ax.axis("off")
fig.tight_layout(); fig.savefig("ch04_ex18_knn_confusions.png", dpi=120)   # writes the PNG, prints nothing
# 3-NN makes only 8 errors on 540 digits (0.985 against the logistic regression's 0.961 in 4.9.2). One weakness
# persists from the logistic regression, 8 read as 1, down from 4 cases to 2; the other leading confusion, two
# 9s read as 3, is new (the logistic regression's single error on a 9 was a 5), and 8 is still the hardest
# digit (3 errors). The two leading pairs tie at 2, and the reverse sort on (count, true, predicted) puts
# (2, 9, 3) first only because 9 > 8, so the figure shows the two 9s called 3: inspect them to judge whether
# the closed loop of the 9 is open enough to be a genuine ambiguity (Bayes floor) or a reducible error that a
# stronger model would fix

# %% [markdown]
# 19. Adjusted R-squared on diabetes at 442 rows and 40 rows, then ten noise columns

# %%
# 19. Adjusted R-squared on diabetes at 442 rows and 40 rows, then ten noise columns
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
Xdb, ydb = load_diabetes(return_X_y=True)                    # 442 patients, 10 features
def adj_r2(r2, n, p): return 1 - (1 - r2) * (n - 1) / (n - p - 1)
small = np.random.default_rng(0).choice(len(ydb), size=40, replace=False)
for label, Xq, yq in [("442 rows", Xdb, ydb), (" 40 rows", Xdb[small], ydb[small])]:
    n = len(yq)
    r2 = r2_score(yq, LinearRegression().fit(Xq, yq).predict(Xq))
    Xn = np.hstack([Xq, np.random.default_rng(1).normal(size=(n, 10))])            # ten pure-noise columns
    r2n = r2_score(yq, LinearRegression().fit(Xn, yq).predict(Xn))
    print(f"{label}: R2 {r2:.4f} adjusted {adj_r2(r2, n, 10):.4f} | with 10 noise columns: R2 {r2n:.4f} adjusted {adj_r2(r2n, n, 20):.4f}")
# expected output: 442 rows: R2 0.5177 adjusted 0.5066 | with 10 noise columns: R2 0.5209 adjusted 0.4982
# expected output: 40 rows: R2 0.5143 adjusted 0.3468 | with 10 noise columns: R2 0.8447 adjusted 0.6812
# by hand at 40 rows: (n-1)/(n-p-1) = 39/29 = 1.3448, so adjusted = 1 - 0.4857 x 1.3448 = 0.3468; with 20
# features the factor is 39/19 = 2.0526 and adjusted = 1 - 0.1553 x 2.0526 = 0.6812. At 442 rows the noise
# behaves as the chapter said: R2 creeps up (0.5177 to 0.5209) and adjusted R2 falls (0.5066 to 0.4982). At 40
# rows the ten noise columns soak up about two thirds of the leftover variance by chance (R2 jumps from 0.5143
# to 0.8447), and the adjustment is not enough: adjusted R2 actually rises, to 0.6812. Repeating with 200
# different noise draws shows why:
for label, Xq, yq in [("442 rows", Xdb, ydb), (" 40 rows", Xdb[small], ydb[small])]:
    n = len(yq); base = adj_r2(r2_score(yq, LinearRegression().fit(Xq, yq).predict(Xq)), n, 10)
    adjs = []
    for seed in range(200):
        Xn = np.hstack([Xq, np.random.default_rng(seed).normal(size=(n, 10))])
        adjs.append(adj_r2(r2_score(yq, LinearRegression().fit(Xn, yq).predict(Xn)), n, 20))
    print(f"{label}: adjusted R2 without noise {base:.4f}, mean with noise {np.mean(adjs):.4f}, fell in {int(np.sum(np.array(adjs) < base))} of 200 draws")
# expected output: 442 rows: adjusted R2 without noise 0.5066, mean with noise 0.5065, fell in 119 of 200 draws
# expected output: 40 rows: adjusted R2 without noise 0.3468, mean with noise 0.3380, fell in 101 of 200 draws
# adjusted R2 is built so that pure noise leaves it unchanged on average (0.5065 against 0.5066; 0.3380
# against 0.3468), which removes the systematic upward drift of R2 but not the draw-to-draw swing, and at
# n = 40 that swing is huge. It is a correction of the bias, not a test; a held-out R2 remains the real check

# %% [markdown]
# 20. McNemar's test for a model swap on new breast-cancer splits

# %%
# 20. McNemar's test for a model swap on new breast-cancer splits
from sklearn.ensemble import RandomForestClassifier
from statsmodels.stats.contingency_tables import mcnemar
for rs in [1, 2, 3]:
    Xtr_s, Xte_s, ytr_s, yte_s = train_test_split(Xb, yb, test_size=0.3, stratify=yb, random_state=rs)
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(Xtr_s, ytr_s)
    rf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=2).fit(Xtr_s, ytr_s)
    r1, r2 = lr.predict(Xte_s) == yte_s, rf.predict(Xte_s) == yte_s
    T = np.array([[np.sum(r1 & r2), np.sum(r1 & ~r2)], [np.sum(~r1 & r2), np.sum(~r1 & ~r2)]])
    chi2, exact = mcnemar(T, exact=False, correction=True), mcnemar(T, exact=True)
    print(f"split {rs}: logistic {lr.score(Xte_s, yte_s):.3f} forest {rf.score(Xte_s, yte_s):.3f} b={T[0, 1]} c={T[1, 0]}"
          f" chi2 {chi2.statistic:.3f} p {chi2.pvalue:.4f} exact p {exact.pvalue:.4f}")
# expected output: split 1: logistic 0.959 forest 0.953 b=4 c=3 chi2 0.000 p 1.0000 exact p 1.0000
# expected output: split 2: logistic 0.965 forest 0.959 b=3 c=2 chi2 0.000 p 1.0000 exact p 1.0000
# expected output: split 3: logistic 0.988 forest 0.971 b=4 c=1 chi2 0.800 p 0.3711 exact p 0.3750
# on every split the two models disagree on only five to seven of 171 tumors and the split of disagreements is
# near even, so p is 1.0 twice and 0.37 once: nothing here justifies replacing the logistic regression with the
# forest (or the reverse). With b + c this small the test has almost no power, which is Dietterich's reason to
# prefer the 5x2cv test of 4.9.1 when both models can be retrained

# %% [markdown]
# ## 4.11 coding drills

# %%
import numpy as np, pandas as pd, seaborn as sns
from scipy.spatial.distance import pdist
from scipy.stats import norm
from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes, load_digits
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, permutation_test_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyRegressor
from sklearn.metrics import adjusted_rand_score, mean_squared_error, log_loss

# %% [markdown]
# Paradigms and tasks

# %%
# Paradigms and tasks
X, y = load_iris(return_X_y=True)
print(len(np.unique(y)), X.shape[1])                         # expected output: 3 4
print(round(adjusted_rand_score(y, KMeans(3, n_init=10, random_state=0).fit_predict(X)), 3))  # expected output: 0.73
Xb, yb = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(Xb, yb, test_size=0.3, random_state=0)
Xtr_s = StandardScaler().fit_transform(Xtr)
knn = KNeighborsClassifier(5).fit(Xtr_s, ytr)
log = LogisticRegression(max_iter=5000).fit(Xtr_s, ytr)
print(knn.n_samples_fit_, log.coef_.size + log.intercept_.size)    # expected output: 398 31

# %% [markdown]
# The learning problem and curse of dimensionality

# %%
# The learning problem and curse of dimensionality
rng = np.random.default_rng(0)
for d in [2, 100]:
    D = pdist(rng.random((1000, d)))
    print(d, round((D.max() - D.min()) / D.mean(), 3))
# expected output: 2 2.619
# expected output: 100 0.502
mpg = sns.load_dataset("mpg").dropna(subset=["horsepower", "mpg"])
sub = mpg.sample(n=45, random_state=1)
Xh = sub[["horsepower"]].to_numpy(float); yh = sub["mpg"].to_numpy(float)
for d in [1, 15]:
    m = make_pipeline(PolynomialFeatures(d), StandardScaler(), LinearRegression()).fit(Xh, yh)
    print(f"deg{d} train MSE", round(mean_squared_error(yh, m.predict(Xh)), 3))
# expected output: deg1 train MSE 23.322
# expected output: deg15 train MSE 10.264

# %% [markdown]
# Bias, variance, and the Bayes floor

# %%
# Bias, variance, and the Bayes floor
print(round(0.5 * norm.cdf(0, 1, 1) + 0.5 * (1 - norm.cdf(0, -1, 1)), 4))  # expected output: 0.1587
print(round(norm.cdf(-1), 4))                                # expected output: 0.1587

# %% [markdown]
# Regularization

# %%
# Regularization
dd = load_diabetes(); Xs = StandardScaler().fit_transform(dd.data); yv = dd.target
print(int(np.sum(np.abs(Ridge(alpha=10).fit(Xs, yv).coef_) > 1e-6)))   # expected output: 10
print(int(np.sum(np.abs(Lasso(alpha=3).fit(Xs, yv).coef_) > 1e-6)))    # expected output: 7
for a in [0.1, 3, 30]:
    print(a, int(np.sum(np.abs(Lasso(alpha=a).fit(Xs, yv).coef_) > 1e-6)))
# expected output: 0.1 9
# expected output: 3 7
# expected output: 30 2

# %% [markdown]
# Metrics

# %%
# Metrics
tp, fp, fn = 7, 8, 22
prec = tp / (tp + fp); rec = tp / (tp + fn)
print(round(prec, 3), round(rec, 3), round(2 * prec * rec / (prec + rec), 3))  # expected output: 0.467 0.241 0.318
df = sns.load_dataset("titanic")
imb = pd.concat([df[df.survived == 0], df[df.survived == 1].sample(n=95, random_state=0)])
print(round(max(imb.survived.mean(), 1 - imb.survived.mean()), 3))            # expected output: 0.852
print(round(log_loss([1, 0], [[0.1, 0.9], [0.9, 0.1]]), 3),
      round(log_loss([1, 0], [[0.4, 0.6], [0.6, 0.4]]), 3))                    # expected output: 0.105 0.511

# %% [markdown]
# Tuning and baselines

# %%
# Tuning and baselines
pipe = make_pipeline(StandardScaler(), SVC())
gs = GridSearchCV(pipe, {"svc__C": [1, 10], "svc__gamma": [1e-3, 1e-2]}, cv=5).fit(Xb, yb)
print(gs.best_params_)                                       # expected output: {'svc__C': 10, 'svc__gamma': 0.01}
Xr, yr = load_diabetes(return_X_y=True)
print(round(cross_val_score(DummyRegressor(), Xr, yr, cv=5, scoring="r2").mean(), 3))  # expected output: -0.028
digits = load_digits(); Xc, yc = digits.data, (digits.target == 3).astype(int)
score, perm, pval = permutation_test_score(KNeighborsClassifier(), Xc, yc,
                                           cv=5, n_permutations=100, random_state=0)
print(round(score, 3), round(perm.max(), 3), round(pval, 4))  # expected output: 0.991 0.897 0.0099
# the floor 1/101: no shuffle reaches the real score
