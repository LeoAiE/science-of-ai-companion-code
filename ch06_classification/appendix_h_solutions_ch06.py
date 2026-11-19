"""Appendix H solutions for Chapter 6, as runnable cells.

Generated from Solutions.md (Appendix H). Every code block of this chapter's solutions appears here
verbatim, split into one cell per exercise or drill topic, so each answer can be run and checked against
the expected outputs it states. Keep Solutions.md as the source; regenerate after any change.
"""

# %% [markdown]
# ## 6.7 self-check (1): Gaussian naive Bayes and QDA, raw and scaled

# %%
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

X, y = load_breast_cancer(return_X_y=True)                 # the lab's data and split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
cv = StratifiedKFold(5, shuffle=True, random_state=0)      # the lab's five folds
pairs = {
    "GaussianNB raw":    GaussianNB(),                      # adds var_smoothing times the largest raw variance
    "GaussianNB scaled": make_pipeline(StandardScaler(), GaussianNB()),
    "QDA raw":           QDA(reg_param=0.1),                # mixes in an identity matrix in raw units
    "QDA scaled":        make_pipeline(StandardScaler(), QDA(reg_param=0.1)),
}
for name, m in pairs.items():
    acc = cross_val_score(m, X_tr, y_tr, cv=cv, scoring="accuracy").mean()
    auc = cross_val_score(m, X_tr, y_tr, cv=cv, scoring="roc_auc").mean()
    print(name, round(acc, 3), round(auc, 3))
# expected output: GaussianNB raw 0.939 0.991
# expected output: GaussianNB scaled 0.929 0.988
# expected output: QDA raw 0.944 0.99
# expected output: QDA scaled 0.974 0.991

# %% [markdown]
# ## 6.8 coding exercises

# %%
import numpy as np
import seaborn as sns
from sklearn.datasets import (load_breast_cancer, load_wine, load_digits, load_diabetes,
                              fetch_20newsgroups, fetch_openml)
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis as LDA,
                                           QuadraticDiscriminantAnalysis as QDA)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, SVR
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.multioutput import MultiOutputClassifier, ClassifierChain
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, f1_score

# %% [markdown]
# 1. kNN from scratch on the three wine varieties, and what scaling is worth

# %%
# 1. kNN from scratch on the three wine varieties, and what scaling is worth
def knn_predict(Xtr, ytr, Xq, k=7):
    preds = []
    for x in Xq:
        d = np.linalg.norm(Xtr - x, axis=1)          # distance to every training wine
        nn = np.argsort(d)[:k]                        # the k closest
        preds.append(np.bincount(ytr[nn]).argmax())   # majority vote over three varieties
    return np.array(preds)
Xw, yw = load_wine(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(Xw, yw, test_size=0.3, random_state=0, stratify=yw)
sc = StandardScaler().fit(Xtr)                        # fit scaling on the training split only
mine = knn_predict(sc.transform(Xtr), ytr, sc.transform(Xte))
skl = KNeighborsClassifier(7).fit(sc.transform(Xtr), ytr).predict(sc.transform(Xte))
print(round(accuracy_score(yte, mine), 3), round(accuracy_score(yte, skl), 3), np.array_equal(mine, skl))
# expected output: 0.963 0.963 True
raw = KNeighborsClassifier(7).fit(Xtr, ytr).score(Xte, yte)   # the same model on raw units
print(round(raw, 3), Xw[:, 12].min(), Xw[:, 12].max())        # proline runs into the thousands
# expected output: 0.685 278.0 1680.0
# Scaling is worth 28 points. Proline spans 278 to 1680, magnesium tops out at 162, and the other
# eleven columns stay below 31, so unscaled distances are almost entirely differences in proline.

# %% [markdown]
# 2. Tune k on the penguins, then uniform versus distance weighting

# %%
# 2. Tune k on the penguins, then uniform versus distance weighting
peng = sns.load_dataset("penguins")                   # 344 penguins, seaborn's copy
num = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
peng = peng.dropna(subset=num)                        # 2 birds have none of the four measurements
Xp, yp = peng[num].to_numpy(), peng["species"].to_numpy()
cv = StratifiedKFold(5, shuffle=True, random_state=0) # shuffled: the table is sorted by species
scores = {k: round(cross_val_score(make_pipeline(StandardScaler(), KNeighborsClassifier(k)),
                                   Xp, yp, cv=cv).mean(), 3) for k in [1, 3, 5, 11, 21, 51, 101]}
print(Xp.shape, {k: float(v) for k, v in scores.items()})
# expected output: (342, 4) {1: 0.985, 3: 0.985, 5: 0.982, 11: 0.982, 21: 0.982, 51: 0.971, 101: 0.915}
for k in [5, 151]:
    u = make_pipeline(StandardScaler(), KNeighborsClassifier(k, weights="uniform"))
    d = make_pipeline(StandardScaler(), KNeighborsClassifier(k, weights="distance"))
    print(k, round(cross_val_score(u, Xp, yp, cv=cv).mean(), 3),
          round(cross_val_score(d, Xp, yp, cv=cv).mean(), 3))
# expected output: 5 0.982 0.982
# expected output: 151 0.81 0.927
# The species separate so cleanly that accuracy stays flat from k = 1 to 21 and falls only past 51.
# At k = 151 a uniform vote covers more than half of the 273 or 274 training birds in a fold, so the
# 68 Chinstraps are outvoted by the larger species; distance weighting shrinks those far votes.

# %% [markdown]
# 3. Gaussian naive Bayes from scratch on wine

# %%
# 3. Gaussian naive Bayes from scratch on wine
class GNB:
    def fit(self, X, y):
        self.cl = np.unique(y)
        self.pri = {c: np.mean(y == c) for c in self.cl}
        self.mu  = {c: X[y == c].mean(0) for c in self.cl}
        self.va  = {c: X[y == c].var(0) + 1e-9 for c in self.cl}
        return self
    def predict(self, X):
        out = []
        for x in X:
            best, bl = None, -np.inf
            for c in self.cl:
                lp = np.log(self.pri[c]) + np.sum(
                    -0.5*np.log(2*np.pi*self.va[c]) - (x - self.mu[c])**2/(2*self.va[c]))
                if lp > bl: best, bl = c, lp
            out.append(best)
        return np.array(out)
Xw, yw = load_wine(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(Xw, yw, test_size=0.3, random_state=0)
m = GNB().fit(Xtr, ytr).predict(Xte)
s = GaussianNB().fit(Xtr, ytr).predict(Xte)
print(round(accuracy_score(yte, m), 3), round(accuracy_score(yte, s), 3), np.array_equal(m, s))
# expected output: 0.944 0.944 True

# %% [markdown]
# 4. Laplace smoothing on a new pair of newsgroups: autos versus motorcycles

# %%
# 4. Laplace smoothing on a new pair of newsgroups: autos versus motorcycles
cats, strip = ["rec.autos", "rec.motorcycles"], ("headers", "footers", "quotes")
tr = fetch_20newsgroups(subset="train", categories=cats, remove=strip)
te = fetch_20newsgroups(subset="test",  categories=cats, remove=strip)
print(len(tr.data), len(te.data))
# expected output: 1192 794
for a in [1e-10, 1.0]:
    clf = make_pipeline(CountVectorizer(stop_words="english"), MultinomialNB(alpha=a)).fit(tr.data, tr.target)
    print(a, round(accuracy_score(te.target, clf.predict(te.data)), 3))
# expected output: 1e-10 0.81
# expected output: 1.0 0.864
# Smoothing is worth 0.054 here, three times the 0.018 it bought on baseball versus space (0.918 to 0.936).

# %% [markdown]
# 5. LDA versus QDA on the digits, and LDA as a 9-dimensional projector

# %%
# 5. LDA versus QDA on the digits, and LDA as a 9-dimensional projector
Xd, yd = load_digits(return_X_y=True)
lda = LDA()                                            # affine invariant, so no scaler
qda = make_pipeline(StandardScaler(), QDA(reg_param=0.1))   # reg_param mixes in an identity: scale first
print(round(cross_val_score(lda, Xd, yd, cv=5).mean(), 3),
      round(cross_val_score(qda, Xd, yd, cv=5).mean(), 3))
# expected output: 0.908 0.949
Xtr, Xte, ytr, yte = train_test_split(Xd, yd, test_size=0.3, random_state=0, stratify=yd)
proj = LDA(n_components=9).fit(Xtr, ytr)              # C - 1 = 9 directions for 10 digits
Ztr, Zte = proj.transform(Xtr), proj.transform(Xte)
full = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(Xtr, ytr)
print(Ztr.shape, round(LogisticRegression(max_iter=5000).fit(Ztr, ytr).score(Zte, yte), 3),
      round(full.score(Xte, yte), 3))
# expected output: (1257, 9) 0.95 0.972
try:
    LDA(n_components=10).fit(Xtr, ytr)                 # one more than C - 1
except ValueError as err:
    print(err)
# expected output: n_components cannot be larger than min(n_features, n_classes - 1).
# Nine LDA numbers keep 0.95 of the 0.972 that all 64 pixels reach. A tenth direction does not exist,
# because the between-class scatter of 10 class means has rank at most 9.

# %% [markdown]
# 6. A digits tree that overfits, pruned with an alpha chosen by cross-validation

# %%
# 6. A digits tree that overfits, pruned with an alpha chosen by cross-validation
dig = load_digits()
Xtr, Xte, ytr, yte = train_test_split(dig.data, dig.target, test_size=0.3, random_state=0,
                                      stratify=dig.target)
for d in [1, 2, 4, 6, 8, 10, None]:
    t = DecisionTreeClassifier(max_depth=d, random_state=0).fit(Xtr, ytr)
    print(d, round(t.score(Xtr, ytr), 3), round(t.score(Xte, yte), 3))
# expected output: 1 0.198 0.198
# expected output: 2 0.317 0.324
# expected output: 4 0.59 0.578
# expected output: 6 0.819 0.772
# expected output: 8 0.926 0.83
# expected output: 10 0.976 0.848
# expected output: None 1.0 0.837
full = DecisionTreeClassifier(random_state=0).fit(Xtr, ytr)
alphas = full.cost_complexity_pruning_path(Xtr, ytr).ccp_alphas
cv = StratifiedKFold(5, shuffle=True, random_state=0)
cvs = [cross_val_score(DecisionTreeClassifier(random_state=0, ccp_alpha=a), Xtr, ytr, cv=cv).mean()
       for a in alphas[:-1]]                          # the last alpha collapses the tree to one leaf
a_star = alphas[:-1][int(np.argmax(cvs))]
pruned = DecisionTreeClassifier(random_state=0, ccp_alpha=a_star).fit(Xtr, ytr)
print(len(alphas), round(float(a_star), 5), round(max(cvs), 3))
# expected output: 123 0.00229 0.833
print(full.get_n_leaves(), round(full.score(Xte, yte), 3), pruned.get_n_leaves(), round(pruned.score(Xte, yte), 3))
# expected output: 136 0.837 62 0.839
imp = pruned.feature_importances_
for i in np.argsort(imp)[::-1][:3]:
    print("pixel", i, round(imp[i], 3))
print("sum", round(imp.sum(), 3))
# expected output: pixel 21 0.092
# expected output: pixel 42 0.089
# expected output: pixel 36 0.083
# expected output: sum 1.0
# Unlike the breast-cancer tree of 6.4.5, pruning here buys simplicity rather than accuracy: the
# cross-validated alpha cuts 136 leaves to 62 while test accuracy moves from 0.837 to 0.839 (0.843 to 0.839 on another machine).

# %% [markdown]
# 7. SVM kernels on the wine data, and the scaling trap

# %%
# 7. SVM kernels on the wine data, and the scaling trap
Xw, yw = load_wine(return_X_y=True)
for kern in ["linear", "poly", "rbf"]:
    clf = SVC(kernel="poly", degree=3, coef0=1) if kern == "poly" else SVC(kernel=kern)
    print(kern, round(cross_val_score(make_pipeline(StandardScaler(), clf), Xw, yw, cv=5).mean(), 3))
# expected output: linear 0.961
# expected output: poly 0.967
# expected output: rbf 0.983
Xtr, Xte, ytr, yte = train_test_split(Xw, yw, test_size=0.3, random_state=0, stratify=yw)
raw = SVC(kernel="rbf").fit(Xtr, ytr)                 # proline's large values swamp the RBF distance
scl = make_pipeline(StandardScaler(), SVC(kernel="rbf")).fit(Xtr, ytr)
print(round(raw.score(Xte, yte), 3), round(scl.score(Xte, yte), 3))
# expected output: 0.648 1.0
# Scaling is worth 35 points on this split.

# %% [markdown]
# 8. The hinge primal on a new problem: digits 3 versus 8 at C = 0.1

# %%
# 8. The hinge primal on a new problem: digits 3 versus 8 at C = 0.1
Xd, yd = load_digits(return_X_y=True)
keep = (yd == 3) | (yd == 8)
Xs = StandardScaler().fit_transform(Xd[keep]); y38 = (yd[keep] == 8).astype(int)
m = 2 * y38 - 1                                       # labels as -1 / +1 for the hinge
n, C = len(y38), 0.1
def hinge_primal(w, b):
    return 0.5 * w @ w + C * np.sum(np.maximum(0.0, 1.0 - m * (Xs @ w + b)))
fits = {
    "SVC linear":          SVC(kernel="linear", C=C),
    "LinearSVC hinge":     LinearSVC(loss="hinge", C=C, dual=True, max_iter=100000, random_state=0),
    "LinearSVC hinge x10": LinearSVC(loss="hinge", C=C, dual=True, intercept_scaling=10,
                                     max_iter=100000, random_state=0),    # weakens the intercept penalty
    "SGD matched":         SGDClassifier(loss="hinge", alpha=1 / (n * C), max_iter=20000, tol=None,
                                         random_state=0),
    "LinearSVC default":   LinearSVC(C=C, max_iter=100000, random_state=0),             # squared hinge
    "SGD default":         SGDClassifier(loss="hinge", max_iter=20000, random_state=0),  # alpha 0.0001
}
print(n)
# expected output: 357
for name, clf in fits.items():
    clf.fit(Xs, y38)
    print(name, round(hinge_primal(clf.coef_.ravel(), float(clf.intercept_[0])), 3))
# expected output: SVC linear 0.904
# expected output: LinearSVC hinge 0.917
# expected output: LinearSVC hinge x10 0.904
# expected output: SGD matched 0.904
# expected output: LinearSVC default 1.247
# expected output: SGD default 404.173
mg = np.array([-0.5, 0.25, 0.75, 1.5])
print(np.maximum(0.0, 1 - mg).tolist())               # hinge
print((np.maximum(0.0, 1 - mg) ** 2).tolist())        # squared hinge
print(np.log1p(np.exp(-mg)).round(3).tolist())        # logistic
# expected output: [1.5, 0.75, 0.25, 0.0]
# expected output: [2.25, 0.5625, 0.0625, 0.0]
# expected output: [0.974, 0.576, 0.387, 0.201]
# LinearSVC's liblinear solver penalizes the intercept like any other weight (its intercept_scaling
# documentation says so), so its optimum sits a little above SVC's; intercept_scaling=10 weakens
# that penalty and the three matched fits meet at 0.904. The two defaults solve different problems,
# the squared hinge and alpha = 0.0001, so their primal values are not comparable.

# %% [markdown]
# 9. One-vs-rest versus one-vs-one with an RBF kernel on the digits

# %%
# 9. One-vs-rest versus one-vs-one with an RBF kernel on the digits
Xd, yd = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(Xd, yd, test_size=0.3, random_state=0)
ovr = OneVsRestClassifier(SVC(kernel="rbf")).fit(Xtr, ytr)    # raw pixels, as in 6.6.1
ovo = OneVsOneClassifier(SVC(kernel="rbf")).fit(Xtr, ytr)
native = SVC(kernel="rbf").fit(Xtr, ytr)                      # libsvm's own one-vs-one
print(len(ovr.estimators_), len(ovo.estimators_),
      round(accuracy_score(yte, ovr.predict(Xte)), 3),
      round(accuracy_score(yte, ovo.predict(Xte)), 3),
      round(native.score(Xte, yte), 3))
# expected output: 10 45 0.981 0.991 0.991
# With the flexible RBF kernel the two strategies nearly agree, 0.981 and 0.991, where the linear
# kernel of 6.6.1 gave 0.931 and 0.974; a plain SVC is already one-vs-one inside and matches OvO.

# %% [markdown]
# 10. Miniproject: the classifier showdown on the Pima diabetes data

# %%
# 10. Miniproject: the classifier showdown on the Pima diabetes data
pima = fetch_openml("diabetes", version=1, as_frame=False)    # 768 patients, 8 measurements
X = pima.data.astype(float); y = (pima.target == "tested_positive").astype(int)
print(X.shape, int(y.sum()), (X == 0).sum(axis=0).tolist())   # zeros per column
# expected output: (768, 8) 268 [111, 5, 35, 227, 374, 11, 0, 0]
# Pregnancies can truly be 0, and the pedigree and age columns hold no zeros. A plasma glucose, a
# diastolic blood pressure, a triceps skin fold, or a body mass index of 0 cannot be measured on a
# living patient, and the 374 zeros in the insulin column, nearly half the rows, are best read the
# same way: missing values in disguise (3.3.7), although the dataset's own description lists none.
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
models = {
    "kNN":        make_pipeline(StandardScaler(), KNeighborsClassifier(7)),
    "GaussianNB": make_pipeline(StandardScaler(), GaussianNB()),
    "LDA":        LDA(),
    "Tree":       DecisionTreeClassifier(max_depth=3, random_state=0),
    "LogReg":     make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
    "SVM-RBF":    make_pipeline(StandardScaler(), SVC(kernel="rbf", random_state=0)),
}
cv = StratifiedKFold(5, shuffle=True, random_state=0)         # the same folds for every model
aucs = {name: cross_val_score(mdl, Xtr, ytr, cv=cv, scoring="roc_auc") for name, mdl in models.items()}
for name in models:
    print(name, round(aucs[name].mean(), 3), round(aucs[name].std(), 3))
best = max(aucs, key=lambda name: aucs[name].mean()); print("best", best)
# expected output: kNN 0.742 0.022
# expected output: GaussianNB 0.796 0.018
# expected output: LDA 0.809 0.027
# expected output: Tree 0.732 0.021
# expected output: LogReg 0.808 0.028
# expected output: SVM-RBF 0.804 0.019
# expected output: best LDA
win = models[best].fit(Xtr, ytr)                                # unseal the test set exactly once
print(confusion_matrix(yte, win.predict(Xte)).tolist())
score = win.decision_function(Xte) if hasattr(win, "decision_function") else win.predict_proba(Xte)[:, 1]
print(round(win.score(Xte, yte), 3), round(roc_auc_score(yte, score), 3))
# expected output: [[110, 15], [28, 39]]
# expected output: 0.776 0.87
# Under scikit-learn 1.9.0 this printed [[111, 14], [28, 39]] and 0.781. Version 1.9.1 divides LDA's
# pooled covariance by n instead of n - C (scikit-learn developers, 2026d, release notes for 1.9.1),
# which moves one nondiabetic test patient's score from -0.0011 to 0.0010, just across the boundary.
# LDA, logistic regression, and the RBF SVM sit within 0.005 AUC of one another, well inside their
# fold standard deviations, so the leader is not clear. The false negatives are the 28 of the 67
# diabetic test patients predicted negative; a screen that misses them sends sick patients home
# untreated, so in practice you would lower the threshold and accept more false alarms.

# %% [markdown]
# 11. Joint C and gamma grid on breast-cancer

# %%
# 11. Joint C and gamma grid on breast-cancer
X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
pipe = Pipeline([("scaler", StandardScaler()), ("svc", SVC(kernel="rbf"))])
grid = GridSearchCV(pipe, {"svc__C": [0.1, 1, 10, 100], "svc__gamma": [0.001, 0.01, 0.1, 1]},
                    cv=5, n_jobs=1).fit(Xtr, ytr)
print(grid.best_params_, round(grid.score(Xte, yte), 3))
# expected output: {'svc__C': 100, 'svc__gamma': 0.001} 0.953
# That pair sits on the grid's corner, the largest C and the smallest gamma tried, so a better pair
# could lie outside the grid. Widen both axes past the corner before trusting it.
wide = GridSearchCV(pipe, {"svc__C": [10, 100, 1000, 10000], "svc__gamma": [1e-5, 1e-4, 0.001, 0.01]},
                    cv=5, n_jobs=1).fit(Xtr, ytr)
print(wide.best_params_, round(wide.best_score_, 3), round(wide.score(Xte, yte), 3))
top = wide.cv_results_["rank_test_score"] == 1                # every pair tied for first place
print([(p["svc__C"], p["svc__gamma"]) for p, is_top in zip(wide.cv_results_["params"], top) if is_top])
# expected output: {'svc__C': 100, 'svc__gamma': 0.001} 0.985 0.953
# expected output: [(100, 0.001), (1000, 0.0001), (10000, 1e-05)]
# The wider grid keeps the same pair, now inside the grid, with CV accuracy 0.985 and test accuracy 0.953.
# It ties with two pairs further along the diagonal where C times gamma stays 0.1, and best_params_ is
# the first of them in the grid's order, so widening the grid finds no better setting.

# %% [markdown]
# 12. Best split from scratch matches depth-1 tree root

# %%
# 12. Best split from scratch matches depth-1 tree root
X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)  # chapter split, unstratified
def gini_y(y):
    _, c = np.unique(y, return_counts=True); p = c / c.sum(); return 1.0 - np.sum(p ** 2)
def best_split(X, y):
    best = (-1.0, None, None)
    parent = gini_y(y)
    for j in range(X.shape[1]):
        vals = np.unique(X[:, j])
        for t in (vals[:-1] + vals[1:]) / 2:          # midpoints, matching CART thresholds
            left, right = y[X[:, j] <= t], y[X[:, j] > t]
            if len(left) == 0 or len(right) == 0:
                continue
            gain = parent - (len(left) * gini_y(left) + len(right) * gini_y(right)) / len(y)
            if gain > best[0]:
                best = (gain, j, t)
    return best
gain, j, t = best_split(Xtr, ytr)
sk = DecisionTreeClassifier(max_depth=1, random_state=0).fit(Xtr, ytr)
print(j, round(float(t), 4), sk.tree_.feature[0], round(float(sk.tree_.threshold[0]), 4), round(gain, 3))
# expected output: 27 0.1424 27 0.1423 0.338
vals = np.unique(Xtr[:, 27])
lo, hi = vals[vals <= t][-1], vals[vals > t][0]               # the two training values the cut falls between
print(lo, hi, float(t), float(sk.tree_.threshold[0]), (float(np.float32(lo)) + float(np.float32(hi))) / 2)
# expected output: 0.1423 0.1424 0.14235 0.1423499956727028 0.1423499956727028
# Both pick feature 27, worst concave points, between the same two training values. The tree's fit
# converts X to float32 (scikit-learn developers, 2026d, DecisionTreeClassifier.fit), so its threshold
# is the midpoint of the float32 copies of 0.1423 and 0.1424, just under the float64 midpoint 0.14235
# that best_split returns. That is why they round to 0.1423 and 0.1424: the split is the same.

# %% [markdown]
# 13. SVR epsilon sweep on diabetes

# %%
# 13. SVR epsilon sweep on diabetes
Xd, yd = load_diabetes(return_X_y=True)
Xd = StandardScaler().fit_transform(Xd)
for eps in [5, 20, 50]:
    r = SVR(kernel="rbf", C=100, epsilon=eps)
    print(eps, len(r.fit(Xd, yd).support_), round(cross_val_score(r, Xd, yd, cv=5).mean(), 3))
# expected output: 5 408 0.44
# expected output: 20 295 0.447
# expected output: 50 176 0.446

# %% [markdown]
# 14. Multilabel wrappers on yeast (OpenML version 4)

# %%
# 14. Multilabel wrappers on yeast (OpenML version 4)
yeast = fetch_openml("yeast", version=4, as_frame=True, parser="auto")
X = yeast.data.to_numpy()
Y = (yeast.target == "TRUE").astype(int).to_numpy()
Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.3, random_state=0)
base = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))   # the scaled base model of 6.6.3
br = MultiOutputClassifier(base).fit(Xtr, Ytr)
chain = ClassifierChain(base, order="random", random_state=0).fit(Xtr, Ytr) # 6.6.3's random label order
combos = [tuple(row) for row in Ytr]
classes = {c: i for i, c in enumerate(sorted(set(combos)))}
yp_tr = np.array([classes[tuple(row)] for row in Ytr])
lp = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr, yp_tr)
inv = {i: c for c, i in classes.items()}
pred_lp = np.array([inv[i] for i in lp.predict(Xte)])
for avg in ["micro", "macro"]:
    print(avg, round(f1_score(Yte, br.predict(Xte), average=avg, zero_division=0), 3),
          round(f1_score(Yte, chain.predict(Xte), average=avg, zero_division=0), 3),
          round(f1_score(Yte, pred_lp, average=avg, zero_division=0), 3))
# expected output: micro 0.61 0.605 0.588
# expected output: macro 0.373 0.382 0.41
# With 6.6.3's scaled base model and random chain order the three models reproduce its numbers: binary
# relevance leads on micro-F1, 0.610, and the label powerset on macro-F1, 0.410.

# %% [markdown]
# 15. Fisher eigh on wine + shrinkage LDA on small breast-cancer

# %%
# 15. Fisher eigh on wine + shrinkage LDA on small breast-cancer
from scipy.linalg import eigh
wine = load_wine(); Xs = StandardScaler().fit_transform(wine.data); yw = wine.target
mu = Xs.mean(0); classes = np.unique(yw)
SW = np.zeros((Xs.shape[1], Xs.shape[1])); SB = np.zeros_like(SW)
for c in classes:
    Xc = Xs[yw == c]; nc = len(Xc); mc = Xc.mean(0)
    SW += (Xc - mc).T @ (Xc - mc)
    SB += nc * np.outer(mc - mu, mc - mu)
evals, evecs = eigh(SB, SW)
print(int(np.sum(evals > 1e-8)), round(float(evals[-1]), 3))
# expected output: 2 9.082
Xc, yc = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(Xc, yc, train_size=40, random_state=0, stratify=yc)
print(round(LDA(solver="lsqr").fit(Xtr, ytr).score(Xte, yte), 3),
      round(LDA(solver="lsqr", shrinkage="auto").fit(Xtr, ytr).score(Xte, yte), 3))
# expected output: 0.796 0.964

# %% [markdown]
# ## 6.8 coding drills

# %%
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import load_wine, load_breast_cancer, load_iris, fetch_california_housing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.metrics import f1_score, confusion_matrix

# k-nearest neighbors
q = np.array([0.0, 0.0]); pts = np.array([[1.0, 0.0], [0.0, 3.0], [2.0, 2.0]])
print(np.linalg.norm(pts - q, axis=1).round(3).tolist())   # expected output: [1.0, 3.0, 2.828]
print(np.bincount(np.array([0, 1, 1, 0, 1])).argmax())     # expected output: 1
xs = np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0]])
print(StandardScaler().fit_transform(xs)[:, 0].round(3).tolist())  # expected output: [-1.225, 0.0, 1.225]
print(np.mean([10.0, 12.0, 20.0]))                          # expected output: 14.0
rng = np.random.default_rng(0)
def ratio(dim):
    P = rng.uniform(0, 1, (500, dim)); d = np.linalg.norm(P - P[0], axis=1)[1:]
    return round((d.max() - d.min()) / d.min(), 2)
print(ratio(2), ratio(100))                                 # expected output: 66.13 0.41

# %% [markdown]
# Naive Bayes

# %%
# Naive Bayes
p = np.full(1000, 0.01)
print(np.prod(p), round(np.sum(np.log(p)), 2))              # expected output: 0.0 -4605.17
print(round((3 + 1) / (10 + 14), 3))                        # expected output: 0.167
yl = np.array([0, 0, 0, 1, 1])
print({int(c): float(round(np.mean(yl == c), 2)) for c in np.unique(yl)})  # expected output: {0: 0.6, 1: 0.4}
cv = CountVectorizer()
X = cv.fit_transform(["win money now", "win win"]).toarray()
print(X.tolist(), cv.get_feature_names_out().tolist())     # expected output: [[1, 1, 1], [0, 0, 2]] ['money', 'now', 'win']

# %% [markdown]
# Discriminant analysis

# %%
# Discriminant analysis
Xw, yw = load_wine(return_X_y=True)
print(Xw[yw == 0].mean(0)[:2].round(2).tolist())           # expected output: [13.74, 2.01]
Xs = StandardScaler().fit_transform(Xw)
print(LDA(n_components=2).fit_transform(Xs, yw).shape)     # expected output: (178, 2)
print(150 - 3)                                             # expected output: 147

# %% [markdown]
# Decision trees

# %%
# Decision trees
def gini(y):
    _, c = np.unique(y, return_counts=True); p = c / c.sum(); return round(1 - np.sum(p**2), 3)
def ent(y):
    _, c = np.unique(y, return_counts=True); p = c / c.sum(); return round(abs(-np.sum(p*np.log2(p))), 3)
print(gini([0, 0, 1, 1]), gini([1, 1, 1, 1]))              # expected output: 0.5 0.0
print(ent([0, 0, 1, 1]), ent([0, 0, 0, 1]))               # expected output: 1.0 0.811
par, L, R = np.array([0,0,0,0,1,1,1,1]), np.array([0,0,0,1]), np.array([0,1,1,1])
gp = 1 - sum((np.bincount(par)/len(par))**2)
gc = (len(L)*(1-sum((np.bincount(L)/len(L))**2)) + len(R)*(1-sum((np.bincount(R)/len(R))**2)))/len(par)
print(round(gp - gc, 3))                                   # expected output: 0.125
data = load_breast_cancer()
t = DecisionTreeClassifier(max_depth=3, random_state=0).fit(data.data, data.target)
print(round(t.feature_importances_.sum(), 3))              # expected output: 1.0
ca = fetch_california_housing()
r = DecisionTreeRegressor(max_depth=2).fit(ca.data[:, [0]], ca.target)
print(r.predict([[1.5], [8.0]]).round(2).tolist())         # expected output: [1.36, 4.22]

# %% [markdown]
# Support vector machines

# %%
# Support vector machines
def hinge(m): return max(0.0, 1.0 - m)
print([hinge(m) for m in [-1.0, 0.5, 1.0, 2.0]])           # expected output: [2.0, 0.5, 0.0, 0.0]
iris = load_iris(); mk = iris.target < 2
lin = SVC(kernel="linear", C=1000).fit(iris.data[mk][:, 2:4], iris.target[mk])
print(int(lin.n_support_.sum()), round(2/np.linalg.norm(lin.coef_[0]), 3))  # expected output: 2 1.304
def rbf(a, b, g=0.5): return round(np.exp(-g*np.sum((a - b)**2)), 3)
print(rbf(np.array([0.0, 0.0]), np.array([1.0, 1.0])),
      rbf(np.array([0.0, 0.0]), np.array([0.0, 0.0])))     # expected output: 0.368 1.0

# %% [markdown]
# Multiclass and metrics

# %%
# Multiclass and metrics
K = 10
print(K, K*(K-1)//2)                                       # expected output: 10 45
yt = np.array([[1,0,1],[0,1,0],[1,1,0]]); yp = np.array([[1,0,0],[0,1,0],[1,0,0]])
print(round(f1_score(yt, yp, average="micro"), 3),
      round(f1_score(yt, yp, average="macro"), 3))         # expected output: 0.75 0.556
print(confusion_matrix([0,0,1,1,1], [0,1,1,1,0]).tolist()) # expected output: [[1, 1], [1, 2]]
