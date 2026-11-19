"""Chapter 6: Supervised Learning: Classification Algorithms

Section-aligned walkthrough: every code block from this chapter of the book,
in book order, each in its own cell under its section heading so you can run
and check them one by one. Generated from Book.md; keep the book as the source.
"""

# %% [markdown]
# ## 6.1.2 Distance metrics: what "nearest" means

# %%
import numpy as np

x, z = np.array([3.0, 4.0]), np.array([6.0, 8.0])      # z = 2x: same direction, twice the length
print(round(np.sqrt(np.sum((x - z) ** 2)), 3))          # 5.0    Euclidean
print(round(np.sum(np.abs(x - z)), 3))                  # 7.0    Manhattan
print(round(np.sum(np.abs(x - z) ** 3) ** (1 / 3), 3))  # 4.498  Minkowski, p = 3
print(round(1 - x @ z / (np.linalg.norm(x) * np.linalg.norm(z)), 3))   # 0.0  cosine
a, b = np.array([1, 0, 1, 1]), np.array([1, 1, 0, 1])
print(int(np.sum(a != b)))                              # 2      Hamming: mismatched positions

# %%
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

X, y = load_breast_cancer(return_X_y=True)
for metric in ["euclidean", "manhattan", "cosine"]:
    pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(5, metric=metric))
    print(metric, round(cross_val_score(pipe, X, y, cv=5).mean(), 3))
# euclidean 0.965   manhattan 0.965   cosine 0.961

# %% [markdown]
# ## 6.1.3 From scratch, and the choice of k

# %%
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def knn_predict(X_train, y_train, X_query, k=5):
    preds = []
    for x in X_query:
        d = np.linalg.norm(X_train - x, axis=1)           # distance to every training point
        nn = np.argsort(d)[:k]                             # the k closest
        preds.append(np.bincount(y_train[nn]).argmax())   # majority vote
    return np.array(preds)

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
scaler = StandardScaler().fit(X_tr)                        # fit scaling on the train split only
X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

scratch = knn_predict(X_tr_s, y_tr, X_te_s, k=5)
skl = KNeighborsClassifier(n_neighbors=5).fit(X_tr_s, y_tr).predict(X_te_s)
print('from-scratch accuracy:', round(accuracy_score(y_te, scratch), 3))   # 0.959
print('scikit-learn accuracy:', round(accuracy_score(y_te, skl), 3))       # 0.959
print('predictions identical:', np.array_equal(scratch, skl))              # True

# %%
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

X, y = load_breast_cancer(return_X_y=True)
for k in [1, 3, 5, 11, 21, 51, 101]:
    pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
    print(k, round(cross_val_score(pipe, X, y, cv=5).mean(), 3))
#   1 -> 0.954   (overfit: boundary chases noise)
#   3 -> 0.960
#   5 -> 0.965   (best)
#  11 -> 0.965
#  21 -> 0.956
#  51 -> 0.951
# 101 -> 0.924   (underfit: vote drowned by distant points)

# %%
import numpy as np
from scipy.stats import norm
from sklearn.neighbors import KNeighborsClassifier

def one_nn_ceiling(bayes_err, n_classes=2):
    """Cover and Hart's large-sample upper bound on the error rate of the 1-NN rule."""
    M, R = n_classes, bayes_err
    return R * (2 - M * R / (M - 1))

for R in [0.01, 0.05, 0.10, 0.25]:                        # a few Bayes error rates
    print(R, round(one_nn_ceiling(R), 4), round(one_nn_ceiling(R, n_classes=3), 4))
# 0.01 0.0198 0.0199   0.05 0.095 0.0963   0.1 0.18 0.185   0.25 0.375 0.4062

# The two known Gaussian classes of 4.4.3, N(-1, 1) and N(+1, 1) with equal priors:
# synthetic on purpose, because only known densities give an exact Bayes error.
bayes = norm.cdf(-1.0)                                    # R* = 0.1587, the floor of 4.4.3
grid = np.linspace(-9.0, 9.0, 200_001)
dens = 0.5 * norm.pdf(grid, loc=-1.0) + 0.5 * norm.pdf(grid, loc=1.0)
eta = 0.5 * norm.pdf(grid, loc=1.0) / dens                # P(y = 1 | x) along the line
limit = np.trapezoid(2 * eta * (1 - eta) * dens, grid)    # 1-NN error with unlimited data

rng = np.random.default_rng(0)
def draw(n):
    y = rng.integers(0, 2, n)
    return rng.normal(2.0 * y - 1.0, 1.0).reshape(-1, 1), y
X_tr, y_tr = draw(20_000)                                 # a large training set
X_te, y_te = draw(200_000)                                # and a huge test set
err = 1 - KNeighborsClassifier(1).fit(X_tr, y_tr).score(X_te, y_te)
print(round(bayes, 4), round(err, 4), round(limit, 4), round(one_nn_ceiling(bayes), 4))
# 0.1587 0.2242 0.2248 0.267   Bayes error, 1-NN test error, 1-NN limit, ceiling

# %% [markdown]
# ## 6.1.4 Distance-weighted voting: closer neighbors count more

# %%
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

X, y = load_breast_cancer(return_X_y=True)
print('  k   uniform   distance')
for k in [1, 5, 25, 75, 151, 251]:
    uni = make_pipeline(StandardScaler(), KNeighborsClassifier(k, weights='uniform'))
    dis = make_pipeline(StandardScaler(), KNeighborsClassifier(k, weights='distance'))
    au = cross_val_score(uni, X, y, cv=5).mean()
    ad = cross_val_score(dis, X, y, cv=5).mean()
    print(f'{k:>4}   {au:.3f}     {ad:.3f}')
#    1   0.954     0.954
#    5   0.965     0.965
#   25   0.953     0.954
#   75   0.935     0.942
#  151   0.912     0.919
#  251   0.847     0.881

# %% [markdown]
# ## 6.1.5 Feature scaling is mandatory

# %%
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(0)
n = 600
xa = rng.uniform(0, 1, n)                    # feature A in [0,1] decides the label
y = (xa > 0.5).astype(int)
xb = rng.uniform(0, 1000, n)                 # feature B in [0,1000] is pure noise, 1000x the scale
X = np.column_stack([xa, xb])
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

print(round(KNeighborsClassifier(15).fit(X_tr, y_tr).score(X_te, y_te), 3))  # 0.522
print(round(make_pipeline(StandardScaler(), KNeighborsClassifier(15))
      .fit(X_tr, y_tr).score(X_te, y_te), 3))                                # 0.989

# %% [markdown]
# ## 6.1.6 kNN for regression

# %%
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_squared_error

X, y = load_diabetes(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
for k in [5, 15, 30]:
    reg = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=k)).fit(X_tr, y_tr)
    pred = reg.predict(X_te)
    print(k, round(r2_score(y_te, pred), 3), round(mean_squared_error(y_te, pred) ** 0.5, 1))
#  5 -> R2 0.243, RMSE 62.1
# 15 -> R2 0.378, RMSE 56.3
# 30 -> R2 0.391, RMSE 55.7

# %% [markdown]
# ## 6.1.7 The curse of dimensionality

# %%
import numpy as np
rng = np.random.default_rng(0)
for d in [2, 10, 100, 1000]:
    P = rng.uniform(0, 1, (1000, d))          # 1,000 stored points in the unit cube
    q = rng.uniform(0, 1, d)                  # one query point
    dist = np.linalg.norm(P - q, axis=1)      # distance from the query to every point
    print(d, round((dist.max() - dist.min()) / dist.min(), 3))
#    2 -> 134.618   10 -> 3.484   100 -> 0.389   1000 -> 0.114

# %%
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

X, y = load_digits(return_X_y=True)          # 1,797 handwritten digits, 64 pixel values each
pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(5))      # scale, then a 5-neighbor vote
print(X.shape, round(cross_val_score(pipe, X, y, cv=5).mean(), 3))   # (1797, 64) 0.944

# %% [markdown]
# ## 6.1.8 Making it fast: trees and approximate neighbors

# %%
import time
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import KDTree

rng = np.random.default_rng(0)
n, q = 20000, 1000                                 # 20,000 stored points, 1,000 queries
for d in [3, 100]:                                 # low dimension, then high
    X = rng.uniform(0, 1, (n, d))
    Q = rng.uniform(0, 1, (q, d))

    t0 = time.perf_counter()                       # brute force: every query scans every point
    dist = cdist(Q, X)                             # all 20 million distances at C speed
    near = np.argpartition(dist, 5, axis=1)[:, :5] # the 5 nearest per query
    brute = time.perf_counter() - t0

    tree = KDTree(X)                               # build the tree once ...
    t0 = time.perf_counter()
    tree.query(Q, k=5)                             # ... then let it prune whole regions
    kd = time.perf_counter() - t0
    print(f"d={d:3}   full scan {brute:6.3f} s   KD-tree {kd:6.3f} s")

# %% [markdown]
# ## 6.2.1 Bayes' rule and the naive assumption

# %%
import numpy as np
probs = np.full(1000, 0.01)          # 1000 tiny per-word likelihoods, each 0.01
print(np.prod(probs))                    # 0.0        the raw product underflows to zero
print(round(np.sum(np.log(probs)), 2))   # -4605.17   the log-sum stays finite

# %% [markdown]
# ## 6.2.2 A spam filter by hand

# %%
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# a tiny hand-built corpus, used only to make the counting mechanics visible
docs = ["win money now", "win the prize money", "cheap money offer",
        "team meeting noon", "project deadline today", "lunch meeting noon"]
labels = np.array([1, 1, 1, 0, 0, 0])            # 1 = spam, 0 = ham
nb = make_pipeline(CountVectorizer(), MultinomialNB(alpha=1.0)).fit(docs, labels)
print(nb.predict_proba(["win money"])[0].round(3))   # [0.083 0.917] -> [P(ham), P(spam)]

# %% [markdown]
# ## 6.2.3 Three variants for three kinds of feature

# %%
import numpy as np
from scipy.stats import norm
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
petal = X[:, 2]                                          # petal length in cm, the third column
for c, name in [(0, "setosa"), (1, "versicolor")]:
    mu, var = petal[y == c].mean(), petal[y == c].var()  # .var() divides by the 50 flowers
    dens = norm.pdf([1.5, 3.0, 4.3], loc=mu, scale=np.sqrt(var))   # densities at 1.5, 3.0, 4.3 cm
    print(name, round(mu, 3), round(var, 6), " ".join(f"{d:.4g}" for d in dens))
# setosa 1.462 0.029556 2.265 9.699e-18 1.553e-59
# versicolor 4.26 0.2164 1.947e-08 0.02189 0.8544

# %%
import numpy as np
from sklearn.naive_bayes import CategoricalNB

# one feature, a color coded 0, 1, or 2; class 1's 10 rows hold the codes 4, 3, and 3 times
color = np.array([0] * 4 + [1] * 3 + [2] * 3 + [0] * 6 + [1] * 4).reshape(-1, 1)
label = np.array([1] * 10 + [0] * 10)                 # class 0 never shows code 2
nb = CategoricalNB(alpha=1.0).fit(color, label)
print(np.round(np.exp(nb.feature_log_prob_[0]), 3))   # one row per class: P(code | class)
# [[0.538 0.385 0.077]
#  [0.385 0.308 0.308]]

# %%
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

class GaussianNBScratch:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.prior = {c: np.mean(y == c) for c in self.classes}         # how common each class is
        self.mean  = {c: X[y == c].mean(axis=0) for c in self.classes}  # per-feature mean in the class
        self.var   = {c: X[y == c].var(axis=0) + 1e-9 for c in self.classes}  # per-feature variance
        return self

    def predict(self, X):
        out = []
        for x in X:
            best_c, best_lp = None, -np.inf
            for c in self.classes:
                # log prior + sum of log-Gaussian densities across the independent features
                lp = np.log(self.prior[c]) + np.sum(
                    -0.5 * np.log(2 * np.pi * self.var[c])
                    - (x - self.mean[c]) ** 2 / (2 * self.var[c]))
                if lp > best_lp:
                    best_c, best_lp = c, lp
            out.append(best_c)
        return np.array(out)

X, y = load_iris(return_X_y=True)                                        # 150 flowers, 4 features
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
scratch = GaussianNBScratch().fit(X_tr, y_tr).predict(X_te)
sk = GaussianNB().fit(X_tr, y_tr).predict(X_te)
print(round(accuracy_score(y_te, scratch), 3))   # 1.0
print(round(accuracy_score(y_te, sk), 3))        # 1.0
print(np.array_equal(scratch, sk))               # True

# %% [markdown]
# ## 6.2.5 Real text, and why naive works anyway

# %%
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

cats = ["rec.sport.baseball", "sci.space"]
strip = ("headers", "footers", "quotes")          # drop giveaway metadata, keep the words
train = fetch_20newsgroups(subset="train", categories=cats, remove=strip)
test  = fetch_20newsgroups(subset="test",  categories=cats, remove=strip)
clf = make_pipeline(CountVectorizer(stop_words="english"), MultinomialNB())
clf.fit(train.data, train.target)                 # 1190 posts, default alpha=1.0 smoothing
print(len(train.data), len(test.data))            # 1190 791
print(round(accuracy_score(test.target, clf.predict(test.data)), 3))   # 0.936
raw = make_pipeline(CountVectorizer(stop_words="english"), MultinomialNB(alpha=1e-10))
raw.fit(train.data, train.target)                 # alpha ~ 0: effectively no smoothing
print(round(accuracy_score(test.target, raw.predict(test.data)), 3))   # 0.918

# %%
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.linear_model import LogisticRegression
proba = clf.predict_proba(test.data)
conf = proba.max(axis=1)
mask = conf > 0.99
print(int(mask.sum()), round(float(conf[mask].mean()), 4),
      round(float((proba.argmax(1)[mask] == test.target[mask]).mean()), 3))
# 697  0.9998  0.986
print(round(brier_score_loss(test.target, proba[:, 1]), 4),
      round(log_loss(test.target, proba), 3))
# 0.0447  0.217

# the comparison model: logistic regression on the same word counts,
# scikit-learn's default L2 penalty with C=1.0, max_iter raised for headroom
lr = make_pipeline(CountVectorizer(stop_words="english"), LogisticRegression(max_iter=1000))
lr.fit(train.data, train.target)
lr_proba = lr.predict_proba(test.data)
lr_mask = lr_proba.max(axis=1) > 0.99
print(int(lr_mask.sum()), round(float(lr_mask.mean()), 3),
      round(accuracy_score(test.target, lr.predict(test.data)), 3))
# 288  0.364  0.908
print(round(brier_score_loss(test.target, lr_proba[:, 1]), 4),
      round(log_loss(test.target, lr_proba), 3))
# 0.0641  0.209

# %% [markdown]
# ## 6.2.6 Naive Bayes as a linear classifier

# %%
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

cats = ["rec.sport.baseball", "sci.space"]
strip = ("headers", "footers", "quotes")
train = fetch_20newsgroups(subset="train", categories=cats, remove=strip)
test = fetch_20newsgroups(subset="test", categories=cats, remove=strip)
pipe = make_pipeline(CountVectorizer(stop_words="english"), MultinomialNB()).fit(train.data, train.target)
nb, vec = pipe.named_steps["multinomialnb"], pipe.named_steps["countvectorizer"]
Xte = vec.transform(test.data)
w = nb.feature_log_prob_[1] - nb.feature_log_prob_[0]
b = nb.class_log_prior_[1] - nb.class_log_prior_[0]
log_odds = Xte @ w + b
print(float(np.max(np.abs(log_odds - (nb.predict_log_proba(Xte)[:, 1] - nb.predict_log_proba(Xte)[:, 0])))))
# 9.413270163349807e-11, about 1e-10
print(bool(np.all((log_odds > 0).astype(int) == nb.predict(Xte))))   # True
names = np.array(vec.get_feature_names_out())
print(list(names[np.argsort(w)[-5:]]))   # ['satellite', 'shuttle', 'orbit', 'launch', 'space']
print(list(names[np.argsort(w)[:5]]))    # ['baseball', 'season', 'pitching', 'cubs', 'braves']

# %% [markdown]
# ## 6.2.7 Bernoulli versus multinomial on the same posts

# %%
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

cats = ["rec.sport.baseball", "sci.space"]
strip = ("headers", "footers", "quotes")
train = fetch_20newsgroups(subset="train", categories=cats, remove=strip)
test  = fetch_20newsgroups(subset="test",  categories=cats, remove=strip)

counts = make_pipeline(CountVectorizer(stop_words="english"), MultinomialNB())
counts.fit(train.data, train.target)                          # word-count features
presence = make_pipeline(CountVectorizer(stop_words="english", binary=True), MultinomialNB())
presence.fit(train.data, train.target)                        # same words, presence only
binary = make_pipeline(CountVectorizer(stop_words="english", binary=True), BernoulliNB())
binary.fit(train.data, train.target)                          # presence, plus an absence term per word
capped = make_pipeline(CountVectorizer(stop_words="english", binary=True, max_features=1000), BernoulliNB())
capped.fit(train.data, train.target)
print(round(accuracy_score(test.target, counts.predict(test.data)), 3))    # 0.936  counts
print(round(accuracy_score(test.target, presence.predict(test.data)), 3))  # 0.938  presence, still multinomial
print(round(accuracy_score(test.target, binary.predict(test.data)), 3))    # 0.774  Bernoulli, full vocabulary
print(len(counts.named_steps["countvectorizer"].vocabulary_))              # 16882
print(round(accuracy_score(test.target, capped.predict(test.data)), 3))    # 0.853  Bernoulli, 1000 words (0.852 on another machine)
comp = make_pipeline(CountVectorizer(stop_words="english"), ComplementNB())
comp.fit(train.data, train.target)
print(round(accuracy_score(test.target, comp.predict(test.data)), 3))      # 0.936  ComplementNB

# %% [markdown]
# ## 6.3.2 LDA versus QDA: shared or separate covariance

# %%
import numpy as np
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis as LDA,
                                           QuadraticDiscriminantAnalysis as QDA)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Synthetic on purpose: we need two classes with known, deliberately different
# covariances to expose LDA's one-line weakness, so we build the clouds by hand.
rng = np.random.default_rng(0)
c0 = rng.multivariate_normal([0, 0], [[1.0, 0.0], [0.0, 1.0]], 400)   # round blob
c1 = rng.multivariate_normal([2, 2], [[4.0, 2.5], [2.5, 2.0]], 400)   # elongated, tilted
X = np.vstack([c0, c1]); y = np.array([0]*400 + [1]*400)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

print(round(accuracy_score(y_te, LDA().fit(X_tr, y_tr).predict(X_te)), 3))   # 0.796  (linear, underfits)
print(round(accuracy_score(y_te, QDA().fit(X_tr, y_tr).predict(X_te)), 3))   # 0.846  (curved, fits each shape)

# %% [markdown]
# ## 6.3.3 Discriminant analysis from scratch

# %%
import numpy as np
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

X, y = load_iris(return_X_y=True)          # real data: 150 flowers, 4 features, 3 species
classes = np.unique(y); n, d = X.shape; C = len(classes)

mu = np.array([X[y == c].mean(axis=0) for c in classes])   # class means mu_c
pi = np.array([np.mean(y == c) for c in classes])          # priors pi_c

Sigma = np.zeros((d, d))                                    # pooled covariance
for c in classes:
    Xc = X[y == c] - mu[c]
    Sigma += Xc.T @ Xc
Sigma /= (n - C)
Sinv = np.linalg.inv(Sigma)

# linear score per class: x . Sinv mu_c - 0.5 mu_c . Sinv mu_c + log pi_c
scores = X @ (Sinv @ mu.T) - 0.5 * np.sum((mu @ Sinv) * mu, axis=1) + np.log(pi)
pred = classes[np.argmax(scores, axis=1)]

print((pred == LDA().fit(X, y).predict(X)).mean())         # 1.0   (matches sklearn on all 150)
print((pred == y).mean())                                  # 0.98

# %%
import numpy as np
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
feat = [2, 3]; keep = y < 2                                 # petal length/width, setosa vs versicolor
X2, y2 = X[keep][:, feat], y[keep]
mu2 = np.array([X2[y2 == c].mean(axis=0) for c in (0, 1)])
pi2 = np.array([np.mean(y2 == c) for c in (0, 1)])
S2 = sum((X2[y2 == c] - mu2[c]).T @ (X2[y2 == c] - mu2[c]) for c in (0, 1)) / (len(y2) - 2)
Sinv2 = np.linalg.inv(S2)

x = np.array([3.0, 1.0])                                    # a query petal: 3 cm long, 1 cm wide
for c in (0, 1):
    delta = x @ (Sinv2 @ mu2[c]) - 0.5 * mu2[c] @ Sinv2 @ mu2[c] + np.log(pi2[c])
    print(c, round(delta, 3))                               # 0 -> 22.992,  1 -> 28.753
    diff = x - mu2[c]
    print("pooled", round(diff @ Sinv2 @ diff, 3), round(float(np.linalg.norm(diff)), 3))
    Xc = X2[y2 == c]
    Sc = (Xc - mu2[c]).T @ (Xc - mu2[c]) / (len(Xc) - 1)
    sign, logdet = np.linalg.slogdet(Sc)
    d_own = diff @ np.linalg.inv(Sc) @ diff
    qda = -0.5 * logdet - 0.5 * d_own + np.log(pi2[c])
    dens = np.exp(-0.5 * d_own) / (2 * np.pi * np.exp(0.5 * logdet))   # the 6.3.1 density, p = 2
    print("qda", round(d_own, 3), round(logdet, 3), round(qda, 3), f"{dens:.3g}")
# pooled 24.577  1.713     qda 98.419  -8.118  -45.844  3.92e-21
# pooled 13.056  1.301     qda  7.747  -5.716   -1.708  0.0577

# %% [markdown]
# ## 6.3.4 Generative versus discriminative

# %%
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Synthetic on purpose: both clouds are built with the same covariance so LDA's
# shared-Gaussian assumption holds exactly, a fair ground for the comparison.
rng = np.random.default_rng(0)
g0 = rng.multivariate_normal([0, 0],     [[1.5, 0.5], [0.5, 1.5]], 400)
g1 = rng.multivariate_normal([2.5, 2.5], [[1.5, 0.5], [0.5, 1.5]], 400)
Xg = np.vstack([g0, g1]); yg = np.array([0]*400 + [1]*400)

print(round(cross_val_score(LDA(), Xg, yg, cv=5).mean(), 3))                  # 0.896
print(round(cross_val_score(LogisticRegression(), Xg, yg, cv=5).mean(), 3))   # 0.898

# %%
import numpy as np
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

iris = load_iris()
mask = iris.target < 2
X, y = iris.data[mask][:, 2:4], iris.target[mask]
lda_fit = LDA().fit(X, y)
w, b = lda_fit.coef_[0], lda_fit.intercept_[0]
for t in [0.1, 1.0, 10.0]:
    p1 = 1 / (1 + np.exp(-t * (X @ w + b)))
    print(t, round(log_loss(y, np.column_stack([1 - p1, p1])), 6))
# 0.1 -> 0.049703   1.0 -> 6e-06   10.0 -> 0.0 (float floor)
for tol in [1e-4, 1e-8, 1e-12]:
    lr = LogisticRegression(C=np.inf, max_iter=100000, tol=tol, solver="lbfgs").fit(X, y)
    print(tol, round(float(np.linalg.norm(lr.coef_)), 2))
# 0.0001 -> 12.95   1e-08 -> 27.56   1e-12 -> 44.09
print(round(float(np.linalg.norm(LogisticRegression(C=1, max_iter=10000).fit(X, y).coef_)), 2))  # 2.8

# %%
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

cats = ["rec.sport.baseball", "sci.space"]
strip = ("headers", "footers", "quotes")
train = fetch_20newsgroups(subset="train", categories=cats, remove=strip)
test = fetch_20newsgroups(subset="test", categories=cats, remove=strip)
docs, labels = np.array(train.data, dtype=object), train.target

rng = np.random.default_rng(0)
for n in (10, 100, len(docs)):                                   # 10 posts, 100 posts, all 1,190
    nb_acc, lr_acc = [], []
    for _ in range(10):                                          # 10 random training subsets of size n
        idx = rng.choice(len(docs), n, replace=False)
        nb = make_pipeline(CountVectorizer(stop_words="english"), MultinomialNB())
        lr = make_pipeline(CountVectorizer(stop_words="english"), LogisticRegression(max_iter=1000))
        nb_acc.append(nb.fit(docs[idx], labels[idx]).score(test.data, test.target))
        lr_acc.append(lr.fit(docs[idx], labels[idx]).score(test.data, test.target))
    print(n, round(np.mean(nb_acc), 3), round(np.mean(lr_acc), 3))

# %% [markdown]
# ## 6.3.5 LDA as dimensionality reduction

# %%
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

Xw, yw = load_wine(return_X_y=True)   # real data: 178 wines, 13 features, 3 grape varieties
proj = LDA(n_components=2).fit_transform(StandardScaler().fit_transform(Xw), yw)
print(proj.shape)                     # (178, 2)
print(round(cross_val_score(make_pipeline(StandardScaler(), LDA()), Xw, yw, cv=5).mean(), 3))  # 0.966

# %% [markdown]
# ## 6.3.6 The Fisher objective behind the projection

# %%
import numpy as np
from scipy.linalg import eigh
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

Xw, yw = load_wine(return_X_y=True)          # real data: 178 wines, 13 features, 3 classes
X = StandardScaler().fit_transform(Xw)
overall = X.mean(axis=0); d = X.shape[1]

S_W = np.zeros((d, d)); S_B = np.zeros((d, d))
for c in np.unique(yw):
    Xc = X[yw == c]; mu_c = Xc.mean(axis=0)
    S_W += (Xc - mu_c).T @ (Xc - mu_c)       # within-class scatter
    diff = (mu_c - overall)[:, None]
    S_B += len(Xc) * (diff @ diff.T)         # between-class scatter

eigvals, eigvecs = eigh(S_B, S_W)            # solve S_B w = lambda S_W w
order = np.argsort(eigvals)[::-1]
print(np.round(eigvals[order][:4], 3))       # [9.082 4.128 0. 0.]  -> only C-1 = 2 nonzero
w0 = eigvecs[:, order[0]]
print(round((w0 @ S_B @ w0) / (w0 @ S_W @ w0), 3))  # 9.082 == top eigenvalue

# %% [markdown]
# ## 6.3.7 When the covariance is singular: shrinkage LDA

# %%
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)          # real data: 569 samples, 30 features
X_tr, X_te, y_tr, y_te = train_test_split(X, y, train_size=40, random_state=0, stratify=y)

plain  = make_pipeline(StandardScaler(), LDA(solver="lsqr")).fit(X_tr, y_tr)
shrunk = make_pipeline(StandardScaler(),
                       LDA(solver="lsqr", shrinkage="auto")).fit(X_tr, y_tr)
print(round(accuracy_score(y_te, plain.predict(X_te)), 3))     # 0.796
print(round(accuracy_score(y_te, shrunk.predict(X_te)), 3))    # 0.964

# %% [markdown]
# ## 6.4.1 Impurity: measuring how mixed a node is

# %%
import numpy as np
def gini(y):
    _, c = np.unique(y, return_counts=True); p = c / c.sum()
    return 1.0 - np.sum(p ** 2)
def entropy(y):
    _, c = np.unique(y, return_counts=True); p = c / c.sum()
    return abs(-np.sum(p * np.log2(p)))               # abs tidies -0.0 on a pure node
for name, node in [("50/50", [0, 0, 1, 1]), ("75/25", [0, 0, 0, 1]), ("pure", [1, 1, 1, 1])]:
    print(name, round(gini(node), 3), round(entropy(node), 3))
# 50/50 -> Gini 0.500, entropy 1.000   75/25 -> 0.375, 0.811   pure -> 0.000, 0.000

# %% [markdown]
# ## 6.4.2 Choosing a split: information gain

# %%
import numpy as np
def gini(y):                                           # impurity of a group of labels
    _, c = np.unique(y, return_counts=True); p = c / c.sum()
    return 1.0 - np.sum(p ** 2)
parent = np.array([0, 0, 0, 0, 1, 1, 1, 1])            # a 50/50 node, Gini 0.5
left, right = np.array([0, 0, 0, 1]), np.array([0, 1, 1, 1])   # the two children of a split
weighted = (len(left) * gini(left) + len(right) * gini(right)) / len(parent)
print(round(gini(parent), 3), round(weighted, 3), round(gini(parent) - weighted, 3))
# parent 0.500   weighted children 0.375   information gain = 0.500 - 0.375 = 0.125

# %%
import numpy as np
def entropy(y):
    _, c = np.unique(y, return_counts=True); p = c / c.sum()
    return abs(-np.sum(p * np.log2(p)))                # abs tidies -0.0 on a pure node
def gain_and_ratio(feature, y):
    """ID3's gain and C4.5's gain ratio for a multiway split on every value of one feature."""
    values, counts = np.unique(feature, return_counts=True)
    frac = counts / len(y)                             # n_v / n for each value v
    children = sum(f * entropy(y[feature == v]) for v, f in zip(values, frac))
    gain = entropy(y) - children                       # information gain, in bits
    split_info = -np.sum(frac * np.log2(frac))         # entropy of the partition itself
    return round(float(gain), 3), round(float(split_info), 3), round(float(gain / split_info), 3)

# a 16-row toy, built by hand so every number can be checked on paper
y = np.array([0] * 8 + [1] * 8)                        # a 50/50 parent: entropy 1 bit
A = np.array([0] * 7 + [1] + [0] + [1] * 7)            # value 0: seven 0s and one 1; value 1: one 0, seven 1s
row_id = np.arange(16)                                 # a different value on every row
print("A     ", gain_and_ratio(A, y))                  # (0.456, 1.0, 0.456)  gain, split info, ratio
print("row id", gain_and_ratio(row_id, y))             # (1.0, 4.0, 0.25)

# %% [markdown]
# ## 6.4.3 Building one from scratch

# %%
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def gini(y):                                           # impurity of a group of labels
    _, c = np.unique(y, return_counts=True); p = c / c.sum()
    return 1.0 - np.sum(p ** 2)

class TreeScratch:
    def __init__(self, max_depth=4, min_samples=2):
        self.max_depth, self.min_samples = max_depth, min_samples

    def fit(self, X, y):
        self.root = self._build(X, y, 0)
        return self

    def _best_split(self, X, y):
        best, parent, n = (0.0, None, None), gini(y), len(y)   # best = (gain, feature, threshold)
        for f in range(X.shape[1]):                    # try every feature
            for thr in np.unique(X[:, f]):             # and every candidate threshold
                left = X[:, f] <= thr
                if left.sum() == 0 or left.sum() == n:
                    continue                           # skip splits that separate nothing
                child = (left.sum() * gini(y[left]) + (~left).sum() * gini(y[~left])) / n
                gain = parent - child                  # information gain of this split
                if gain > best[0]:
                    best = (gain, f, thr)
        return best

    def _build(self, X, y, depth):
        if len(np.unique(y)) == 1 or depth >= self.max_depth or len(y) < self.min_samples:
            return {"leaf": np.bincount(y).argmax()}   # stop: leaf predicts the majority class
        gain, f, thr = self._best_split(X, y)          # greedily pick the best question
        if f is None or gain <= 0:
            return {"leaf": np.bincount(y).argmax()}
        left = X[:, f] <= thr
        return {"f": f, "thr": thr,
                "L": self._build(X[left],  y[left],  depth + 1),   # recurse left
                "R": self._build(X[~left], y[~left], depth + 1)}   # recurse right

    def _one(self, x, node):                           # follow one point down to a leaf
        while "leaf" not in node:
            node = node["L"] if x[node["f"]] <= node["thr"] else node["R"]
        return node["leaf"]

    def predict(self, X):
        return np.array([self._one(x, self.root) for x in X])

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
mine = TreeScratch(max_depth=4).fit(X_tr, y_tr)
sk = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X_tr, y_tr)
print("from-scratch", round(accuracy_score(y_te, mine.predict(X_te)), 3),
      " scikit-learn", round(sk.score(X_te, y_te), 3))
# from-scratch 0.936   scikit-learn 0.947  on breast-cancer test data

# %%
# continues from Example 6-5: where the from-scratch tree and scikit-learn part ways
names = load_breast_cancer().feature_names
print((mine.predict(X_tr) != sk.predict(X_tr)).sum(),        # training tumors the trees disagree on
      np.where(mine.predict(X_te) != sk.predict(X_te))[0])   # test tumors they disagree on
t, a, b = sk.tree_, mine.root["L"], mine.root["L"]["L"]      # sklearn's arrays; our depth-1 and depth-2 nodes
print(names[a["f"]], a["thr"], "|", names[t.feature[1]], round(t.threshold[1], 1))  # both depth-1 questions
rows = np.flatnonzero(X_tr[:, t.feature[0]] <= t.threshold[0])  # training tumors sent left at the root
ours = X_tr[rows, a["f"]] <= a["thr"]                         # where our question sends them
theirs = X_tr[rows, t.feature[1]] <= t.threshold[1]           # where scikit-learn's sends them
print(len(rows), ours.sum(), bool((ours == theirs).all()))    # one partition, so one gain: a tie
vals = np.unique(X_tr[rows[ours], b["f"]])                    # sorted training values at depth 2
k = np.searchsorted(vals, b["thr"])                           # our threshold is one of those values
print(names[b["f"]], vals[k], vals[k + 1], round(t.threshold[t.children_left[1]], 2))
print(X_te[57, b["f"]], X_te[116, a["f"]], X_te[116, t.feature[1]])  # the two test tumors' values
# 0 [ 57 116]
# worst radius 17.5 | worst area 952.9
# 263 250 True
# area error 34.78 35.74 35.26
# 35.13 17.71 947.9

# %% [markdown]
# ## 6.4.4 Reading a fitted tree

# %%
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
import numpy as np

data = load_breast_cancer()
X_tr, X_te, y_tr, y_te = train_test_split(data.data, data.target, test_size=0.3, random_state=0)
tree = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X_tr, y_tr)
print(export_text(tree, feature_names=list(data.feature_names)))
# root: worst concave points <= 0.14
path = tree.decision_path(X_te[0:1]).toarray()[0]
print(np.where(path)[0])                    # [0 4 6]
leaf = tree.apply(X_te[0:1])[0]
counts = tree.tree_.value[leaf][0] * tree.tree_.n_node_samples[leaf]
print(leaf, counts, tree.predict_proba(X_te[0:1]))
# 6 [128.   4.] [[0.96969697 0.03030303]]

# %% [markdown]
# ## 6.4.5 Depth, overfitting, and pruning

# %%
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
for d in [1, 2, 3, 4, 5, 6, 7, 8, None]:
    tree = DecisionTreeClassifier(max_depth=d, random_state=0).fit(X_tr, y_tr)
    print(d, round(tree.score(X_tr, y_tr), 3), round(tree.score(X_te, y_te), 3))
# 1 0.930 0.895 | 2 0.960 0.947 | 3 0.967 0.947 | 4 0.977 0.947
# 5 0.987 0.936 | 6 0.997 0.947 | 7 1.000 0.912 | 8 1.000 0.912 | None 1.000 0.912

# %%
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
full = DecisionTreeClassifier(random_state=0).fit(X_tr, y_tr)
alphas = full.cost_complexity_pruning_path(X_tr, y_tr).ccp_alphas
print(len(alphas), round(float(alphas[-1]), 3))          # 12 0.338: the last alpha prunes to one leaf
cv = StratifiedKFold(5, shuffle=True, random_state=0)
scores = [cross_val_score(DecisionTreeClassifier(random_state=0, ccp_alpha=a),
                          X_tr, y_tr, cv=cv).mean() for a in alphas[:-1]]
a_star = alphas[:-1][int(np.argmax(scores))]
pruned = DecisionTreeClassifier(random_state=0, ccp_alpha=a_star).fit(X_tr, y_tr)
print(round(float(a_star), 4), round(max(scores), 3),
      pruned.get_n_leaves(), round(pruned.score(X_te, y_te), 3))
# 0.0139  0.927  3 leaves  test 0.947
for msl in [1, 5, 10, 20]:
    t = DecisionTreeClassifier(min_samples_leaf=msl, random_state=0).fit(X_tr, y_tr)
    print(msl, t.get_n_leaves())
# 1->19  5->14  10->9  20->6 leaves

# %% [markdown]
# ## 6.4.6 Feature importance from a single tree

# %%
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
data = load_breast_cancer()
X_tr, X_te, y_tr, y_te = train_test_split(data.data, data.target, test_size=0.3, random_state=0)
tree = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X_tr, y_tr)
imp = tree.feature_importances_
for i in np.argsort(imp)[::-1][:3]:          # the three most important features
    print(f"{data.feature_names[i]:20s} {imp[i]:.3f}")
print("importances sum to", round(imp.sum(), 3))
# worst concave points 0.817   worst area 0.098   area error 0.054   (sum 1.000)

# %%
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
data = load_breast_cancer()
X_tr, X_te, y_tr, y_te = train_test_split(data.data, data.target, test_size=0.3, random_state=0)
tree = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X_tr, y_tr)
t = tree.tree_                                       # one entry per node in every array
n = t.weighted_n_node_samples[0]                     # training rows that reach the root
raw = np.zeros(X_tr.shape[1])
for node in range(t.node_count):
    L, R = t.children_left[node], t.children_right[node]
    if L == -1:                                      # a leaf splits nothing
        continue
    n_t = t.weighted_n_node_samples[node]
    drop = (n_t / n) * (t.impurity[node]
                        - t.weighted_n_node_samples[L] / n_t * t.impurity[L]
                        - t.weighted_n_node_samples[R] / n_t * t.impurity[R])
    raw[t.feature[node]] += drop                     # credit the feature this node split on
imp = raw / raw.sum()                                # normalize so the importances sum to 1
f0 = t.feature[0]                                    # the feature the root splits on
print(data.feature_names[f0], round(raw[f0], 4), round(raw.sum(), 4), round(imp[f0], 3))
# worst concave points 0.338 0.4139 0.817
print(bool(np.allclose(imp, tree.feature_importances_)))   # True

# %% [markdown]
# ## 6.4.7 Regression trees

# %%
import numpy as np
from sklearn.tree import DecisionTreeRegressor
x = np.array([[1.0], [2.0], [3.0]]); y = np.array([1.0, 2.0, 6.0])   # three points, by hand
print(round(float(np.mean((y - y.mean()) ** 2)), 4))          # 4.6667  the root's variance impurity
reg = DecisionTreeRegressor(max_depth=1).fit(x, y)            # one split, chosen by variance
print(float(reg.tree_.threshold[0]))                          # 2.5     where it cuts
print(np.round(reg.tree_.impurity, 4).tolist())               # [4.6667, 0.25, 0.0]  root, left, right
print(np.round(reg.tree_.value.ravel(), 2).tolist())          # [3.0, 1.5, 6.0]  the node means

# %%
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
data = fetch_california_housing()
medinc = data.data[:, [0]]          # median income of the block group (in $10k units)
price = data.target                 # median house value (in $100k units)
for depth in [2, 5]:
    reg = DecisionTreeRegressor(max_depth=depth).fit(medinc, price)
    preds = reg.predict([[1.5], [4.0], [8.0]])   # low, middle, and high income
    print(depth, np.round(preds, 2))
# depth 2 -> [1.36 2.09 4.22]   depth 5 -> [1.08 2.1  4.29]

# %% [markdown]
# ## 6.4.8 How CART splits, and why one tree is unstable

# %%
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
data = load_breast_cancer()
X, y, names = data.data, data.target, data.feature_names
rng = np.random.default_rng(0)
for b in range(2):
    idx = rng.integers(0, len(y), len(y))    # a bootstrap resample: draw rows with replacement
    stump = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X[idx], y[idx])
    f, thr = stump.tree_.feature[0], stump.tree_.threshold[0]
    print(f"resample {b}: root split = {names[f]} <= {thr:.1f}")
# resample 0: root split = worst area <= 868.2
# resample 1: root split = worst perimeter <= 112.8

# %% [markdown]
# ## 6.4.9 Missing values inside a tree

# %%
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# a six-row toy, built by hand: two passengers have no recorded age
age = np.array([[10.0], [20.0], [30.0], [40.0], [np.nan], [np.nan]])
label = np.array([1, 1, 0, 0, 1, 1])
stump = DecisionTreeClassifier(max_depth=1, random_state=0).fit(age, label)
print(stump.tree_.threshold[0], bool(stump.tree_.missing_go_to_left[0]))   # 25.0 True
print(stump.predict([[np.nan], [35.0]]))                                    # [1 0]

# %%
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

titanic = sns.load_dataset("titanic")                       # 891 passengers, seaborn's copy
titanic["male"] = (titanic["sex"] == "male").astype(int)    # sex as a 0/1 column
cols = ["pclass", "male", "age", "sibsp", "parch", "fare"]
X = titanic[cols].to_numpy(dtype=float)                     # unknown ages arrive as NaN
y = titanic["survived"].to_numpy()
print(int(np.isnan(X).sum()), len(X))                       # 177 891
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

native = DecisionTreeClassifier(max_depth=3, random_state=0)                 # NaN goes in as is
imputed = make_pipeline(SimpleImputer(strategy="median"),                   # fill with the median first
                        DecisionTreeClassifier(max_depth=3, random_state=0))
for name, m in [("native", native), ("imputed", imputed)]:
    cv = cross_val_score(m, X_tr, y_tr, cv=5).mean()                        # training rows only
    print(name, round(cv, 3), round(m.fit(X_tr, y_tr).score(X_te, y_te), 3))
# native 0.806 0.832
# imputed 0.806 0.832
age_nodes = np.where(native.tree_.feature == cols.index("age"))[0]           # the nodes that ask about age
print(native.tree_.threshold[age_nodes], native.tree_.missing_go_to_left[age_nodes].astype(bool))
# [ 2.5 13. ] [False False]
try:
    SVC().fit(X_tr, y_tr)                                    # a distance-based model refuses NaN
except ValueError as err:
    print(type(err).__name__)                                # ValueError

# %% [markdown]
# ## 6.5.1 The maximum margin and the support vectors

# %%
import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
iris = load_iris()
mask = iris.target < 2                         # setosa (0) vs versicolor (1), 100 flowers
X = iris.data[mask][:, 2:4]                     # petal length and petal width, cleanly separable
y = iris.target[mask]
lin = SVC(kernel="linear", C=1000).fit(X, y)    # large C behaves like a hard margin
print(lin.n_support_.sum())                     # 2  support vectors
print(round(np.linalg.norm(lin.coef_[0]), 3))       # 1.534  ||w||
print(round(2 / np.linalg.norm(lin.coef_[0]), 3))   # 1.304  margin width 2/||w||

# %% [markdown]
# ## 6.5.2 The primal, the dual, and why dot products matter

# %%
import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
iris = load_iris()
mask = iris.target < 2                             # the setosa-versicolor petals of 6.5.1
X, y = iris.data[mask][:, 2:4], iris.target[mask]
lin = SVC(kernel="linear", C=1000).fit(X, y)
ay = lin.dual_coef_.ravel()                        # alpha_i * y_i for each support vector
w = ay @ lin.support_vectors_                      # w = sum over i of alpha_i y_i x_i
print(np.round(w, 4), np.round(lin.coef_.ravel(), 4))   # [1.2941 0.8235] [1.2941 0.8235]
print(round(float(ay.sum()), 6))                   # 0.0   the condition on b
alpha = np.abs(ay)                                 # the multipliers themselves
K = lin.support_vectors_ @ lin.support_vectors_.T  # dot products between support vectors
dual = alpha.sum() - 0.5 * ay @ K @ ay             # the dual objective at the solution
print(np.round(alpha, 4), round(float(dual), 4), round(float(0.5 * w @ w), 4))
# [1.1765 1.1765] 1.1765 1.1765   multipliers, dual objective, primal objective

# %% [markdown]
# ## 6.5.3 The kernel trick

# %%
import numpy as np

x, z = np.array([1.0, 2.0]), np.array([2.0, 0.0])
gamma = 0.5
print(round(float(np.exp(-gamma * np.sum((x - z) ** 2))), 3))   # 0.082
print(round(float(np.exp(-gamma * np.sum((x - x) ** 2))), 3))   # 1.0

# %%
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import pairwise_kernels

x, z = np.array([1.0, 2.0]), np.array([3.0, 4.0])
phi = lambda v: np.array([v[0]**2, np.sqrt(2)*v[0]*v[1], v[1]**2])
print(round(float((x @ z)**2), 1), round(float(phi(x) @ phi(z)), 1))   # 121.0 121.0
Xi = StandardScaler().fit_transform(load_iris().data)
for name, kw in [("rbf", dict(metric="rbf", gamma=0.5)),
                 ("poly", dict(metric="poly", degree=2, gamma=1.0, coef0=0.0)),
                 ("sigmoid", dict(metric="sigmoid", gamma=1.0, coef0=1.0))]:
    ev = np.linalg.eigvalsh(pairwise_kernels(Xi, **kw))
    print(name, int(np.sum(ev < -1e-8)), round(float(ev.min()), 2))
# rbf 0 0.0   poly 0 -0.0   sigmoid 75 -19.94

# %%
import numpy as np
from sklearn.datasets import make_moons
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
# make_moons is synthetic on purpose: we need a controlled 2-D curved boundary to picture,
# which no real tabular dataset gives us in exactly two plottable features.
Xm, ym = make_moons(n_samples=400, noise=0.25, random_state=0)
for kern in ["linear", "poly", "rbf"]:
    clf = SVC(kernel="poly", degree=3, coef0=1) if kern == "poly" else SVC(kernel=kern)
    acc = cross_val_score(make_pipeline(StandardScaler(), clf), Xm, ym, cv=5).mean()
    print(kern, round(acc, 3))
# linear 0.865   poly 0.915 (degree=3, coef0=1)   rbf 0.917

# %% [markdown]
# ## 6.5.4 Soft margins, C, and gamma

# %%
import numpy as np
from sklearn.datasets import make_moons
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
Xm, ym = make_moons(n_samples=400, noise=0.25, random_state=0)
for C in [0.1, 1, 100]:
    for g in [0.1, 1, 10]:
        acc = cross_val_score(make_pipeline(StandardScaler(), SVC(kernel="rbf", C=C, gamma=g)),
                              Xm, ym, cv=5).mean()
        print(C, g, f"{acc:.3f}")
# C=0.1,g=0.1 -> 0.863   C=1,g=1 -> 0.923 (best)   C=100,g=10 -> 0.867

# %%
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
import numpy as np

X, y = load_breast_cancer(return_X_y=True)
for C in [0.1, 1.0]:
    svc = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=C)).fit(X, y).named_steps["svc"]
    dc = np.abs(svc.dual_coef_.ravel())
    print(C, len(dc), int(np.isclose(dc, C).sum()), round(float(dc.max()), 1))
# 0.1  230  213  0.1
# 1.0  119   62  1.0

# %% [markdown]
# ## 6.5.5 Why scaling is mandatory, and the cost of kernels

# %%
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
raw = SVC(kernel="rbf").fit(Xtr, ytr)                                    # raw features, mixed scales
scaled = make_pipeline(StandardScaler(), SVC(kernel="rbf")).fit(Xtr, ytr)
print(round(raw.score(Xte, yte), 3))                                    # 0.906  unscaled
print(round(scaled.score(Xte, yte), 3))                                 # 0.953  scaled

# %%
import time
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import Nystroem

# Letter Recognition (Frey and Slate, 1991), OpenML dataset 6: 20,000 capital letters drawn in 20 fonts
# and randomly distorted, each summarized by 16 integer features, to be sorted into 26 classes
X, y = fetch_openml("letter", version=1, as_frame=False, return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

def fit_and_time(model, n):
    """Fit on the first n training rows; return test accuracy and the seconds the fit took."""
    start = time.perf_counter()
    model.fit(X_tr[:n], y_tr[:n])
    return model.score(X_te, y_te), time.perf_counter() - start

exact = make_pipeline(StandardScaler(), SVC(kernel="rbf"))         # gamma="scale" works out to 1/16 after scaling
approx = make_pipeline(StandardScaler(),
                       Nystroem(gamma=1/16, n_components=300, random_state=0),   # 300 landmarks, same kernel
                       SGDClassifier(loss="hinge", random_state=0))              # linear hinge loss on the map
for n in (1875, 3750, 7500, 15000):                                 # the training size doubles each row
    acc_exact, sec_exact = fit_and_time(exact, n)
    acc_approx, sec_approx = fit_and_time(approx, n)
    print(f"n={n:5d}   exact SVC {acc_exact:.3f} in {sec_exact:4.2f} s"
          f"   Nystroem + SGD {acc_approx:.3f} in {sec_approx:4.2f} s")

# %%
from sklearn.svm import LinearSVC
for m in (100, 300, 1000):                                          # more landmarks, closer approximation
    model = make_pipeline(StandardScaler(),
                          Nystroem(gamma=1/16, n_components=m, random_state=0),
                          SGDClassifier(loss="hinge", random_state=0))
    acc, sec = fit_and_time(model, 15000)
    print(f"{m:4d} components + SGD        accuracy {acc:.3f}   fit {sec:5.2f} s")
nys_linsvc = make_pipeline(StandardScaler(),
                           Nystroem(gamma=1/16, n_components=1000, random_state=0),
                           LinearSVC())                             # same 1000 features, the liblinear solver
acc, sec = fit_and_time(nys_linsvc, 15000)
print(f"1000 components + LinearSVC  accuracy {acc:.3f}   fit {sec:5.2f} s")

# %%
import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics.pairwise import rbf_kernel

scaler = StandardScaler().fit(X_tr[:15000])                    # the scaling every pipeline here learns
A, T = scaler.transform(X_tr[:15000]), scaler.transform(X_te[:1000])   # 15,000 training, 1,000 test letters
gamma = 1 / 16
x, z = T[0], T[1]                                              # two test letters
sq = float(np.sum((x - z) ** 2))
print(round(sq, 3), round(float(np.exp(-gamma * sq)), 4))      # squared distance, exact kernel value

rng = np.random.default_rng(0)
for D in (100, 1000, 10000):                                   # the map built by hand at three widths
    W = rng.normal(0.0, np.sqrt(2 * gamma), size=(16, D))      # directions omega from N(0, 2 gamma I)
    b = rng.uniform(0.0, 2 * np.pi, size=D)                    # offsets from U[0, 2 pi]
    zx = np.sqrt(2 / D) * np.cos(x @ W + b)
    zz = np.sqrt(2 / D) * np.cos(z @ W + b)
    print(D, round(float(zx @ zz), 4))                         # Monte Carlo estimate of the kernel value

rff = RBFSampler(gamma=gamma, n_components=300, random_state=0).fit(A)
by_hand = np.sqrt(2 / 300) * np.cos(T @ rff.random_weights_ + rff.random_offset_)
print(np.allclose(rff.transform(T), by_hand), round(float(rff.random_weights_.var()), 3))   # same map; variance 2 gamma

K = rbf_kernel(T, gamma=gamma)                                 # exact kernel among the 1,000 test letters
for D in (100, 300, 1000, 3000):                               # mean absolute error of each map's Gram matrix
    Zr = RBFSampler(gamma=gamma, n_components=D, random_state=0).fit(A).transform(T)
    Zn = Nystroem(gamma=gamma, n_components=D, random_state=0).fit(A).transform(T)
    print(D, round(float(np.abs(K - Zr @ Zr.T).mean()), 4), round(float(np.abs(K - Zn @ Zn.T).mean()), 4))

# %%
for m in (300, 1000, 3000):                                    # random Fourier features, same kernel
    model = make_pipeline(StandardScaler(),
                          RBFSampler(gamma=1/16, n_components=m, random_state=0),
                          SGDClassifier(loss="hinge", random_state=0))
    acc, sec = fit_and_time(model, 15000)
    print(f"{m:4d} random features + SGD  accuracy {acc:.3f}   fit {sec:5.2f} s")
nys3000 = make_pipeline(StandardScaler(),
                        Nystroem(gamma=1/16, n_components=3000, random_state=0),
                        SGDClassifier(loss="hinge", random_state=0))
acc, sec = fit_and_time(nys3000, 15000)
print(f"3000 components + SGD        accuracy {acc:.3f}   fit {sec:5.2f} s")

# %%
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline

strip = ("headers", "footers", "quotes")
cats = ["rec.sport.baseball", "sci.space"]
tr = fetch_20newsgroups(subset="train", categories=cats, remove=strip)
te = fetch_20newsgroups(subset="test", categories=cats, remove=strip)
print(round(make_pipeline(TfidfVectorizer(stop_words="english"), MultinomialNB())
            .fit(tr.data, tr.target).score(te.data, te.target), 3))   # 0.948
print(round(make_pipeline(TfidfVectorizer(stop_words="english"), LinearSVC())
            .fit(tr.data, tr.target).score(te.data, te.target), 3))   # 0.923

# %%
tr_all = fetch_20newsgroups(subset="train", remove=strip)     # no categories filter: all 20 topics
te_all = fetch_20newsgroups(subset="test", remove=strip)
print(len(tr_all.target_names), len(tr_all.data), len(te_all.data))   # 20 11314 7532
print(round(make_pipeline(TfidfVectorizer(stop_words="english"), MultinomialNB())
            .fit(tr_all.data, tr_all.target).score(te_all.data, te_all.target), 3))   # 0.678
print(round(make_pipeline(TfidfVectorizer(stop_words="english"), LinearSVC())
            .fit(tr_all.data, tr_all.target).score(te_all.data, te_all.target), 3))   # 0.693

# %% [markdown]
# ## 6.5.6 Support vector regression

# %%
import numpy as np
from sklearn.svm import SVR
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
# a synthetic noisy sine is the right tool here: one clean 1-D curve lets us picture the
# epsilon-tube and count which points fall outside it.
rng = np.random.default_rng(0)
Xr = np.sort(rng.uniform(0, 5, 120)).reshape(-1, 1)
yr = np.sin(Xr).ravel() + rng.normal(0, 0.12, 120)
svr = SVR(kernel="rbf", C=10, epsilon=0.1).fit(Xr, yr)
print(round(svr.score(Xr, yr), 3), len(svr.support_))   # 0.974  51   training R^2, support vectors of 120
resid = yr - svr.predict(Xr)
print(int(np.sum(np.abs(resid) > 0.1 + 1e-6)))          # 47   points strictly outside the tube (49 on another machine)
folds = KFold(5, shuffle=True, random_state=0)          # shuffled: the points are sorted by x
print(round(cross_val_score(SVR(kernel="rbf", C=10, epsilon=0.1), Xr, yr, cv=folds).mean(), 3))   # 0.969
Xd, yd = load_diabetes(return_X_y=True)
Xd = StandardScaler().fit_transform(Xd)
for eps in [5, 20, 50]:
    print(eps, len(SVR(kernel="rbf", C=100, epsilon=eps).fit(Xd, yd).support_))
# 5 -> 408   20 -> 295   50 -> 176  of 442

# %% [markdown]
# ## 6.5.7 Probabilities from an SVM

# %%
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
raw = make_pipeline(StandardScaler(), SVC(kernel="rbf", random_state=0)).fit(Xtr, ytr)
cal = make_pipeline(StandardScaler(),
                    CalibratedClassifierCV(SVC(kernel="rbf", random_state=0),
                                           method="sigmoid", ensemble=False)).fit(Xtr, ytr)
raw_pred = raw.predict(Xte)
cal_pred = cal.predict(Xte)
cal_hat = cal.predict_proba(Xte).argmax(1)
print(round((raw_pred == yte).mean(), 3))     # 0.953  raw SVM
print(round((cal_pred == yte).mean(), 3))     # 0.947  calibrated
print(int((cal_pred != cal_hat).sum()))       # 0  calibrated predict matches its probabilities
print(int((raw_pred != cal_hat).sum()))       # 1  raw predict vs calibrated class

# %%
import numpy as np
fitted = cal[-1].calibrated_classifiers_[0]       # ensemble=False keeps one SVM and one sigmoid
A, B = fitted.calibrators[0].a_, fitted.calibrators[0].b_
print(round(A, 3), round(B, 3))                   # -3.834 0.408   the learned A and B
f = fitted.estimator.decision_function(cal[0].transform(Xte))    # SVM scores f(x) of the test tumors
print(round(f[0], 3), f"{1 / (1 + np.exp(A * f[0] + B)):.1e}",
      f"{cal.predict_proba(Xte[:1])[0, 1]:.1e}")  # -2.416 6.3e-05 6.3e-05   by hand, then predict_proba
k = np.flatnonzero(raw_pred != cal_hat)           # the test tumors where the two classes disagree
print(k, np.round(f[k], 3), np.round(cal.predict_proba(Xte[k])[:, 1], 3), yte[k])   # [137] [0.053] [0.449] [1]

# %% [markdown]
# ## 6.5.8 The hinge loss: the SVM as loss minimization

# %%
def hinge(m):
    return max(0.0, 1.0 - m)
for m in [-1.0, 0.0, 0.5, 1.0, 2.0]:
    print(m, hinge(m))
# -1.0 2.0   misclassified: large penalty
#  0.0 1.0   on the boundary
#  0.5 0.5   correct but inside the margin: still penalized
#  1.0 0.0   exactly on the margin: zero
#  2.0 0.0   safely correct: contributes nothing

# %%
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
Xs = StandardScaler().fit_transform(X)
n, C = len(X), 1.0
m = y * 2 - 1                                       # map {0,1} to {-1,+1} for the hinge

def hinge_primal(w, b):
    return 0.5 * np.dot(w, w) + C * np.sum(np.maximum(0.0, 1.0 - m * (Xs @ w + b)))

fits = {
    "SVC linear": SVC(kernel="linear", C=C).fit(Xs, y),
    "LinearSVC hinge": LinearSVC(loss="hinge", C=C, dual=True, max_iter=20000, random_state=0).fit(Xs, y),
    "SGD matched": SGDClassifier(loss="hinge", alpha=1 / (n * C), max_iter=20000, tol=None, random_state=0).fit(Xs, y),
    "LinearSVC default": LinearSVC(C=C, max_iter=20000, random_state=0).fit(Xs, y),
    "SGD default": SGDClassifier(loss="hinge", max_iter=20000, random_state=0).fit(Xs, y),
}
for name, clf in fits.items():
    print(name, round(hinge_primal(clf.coef_.ravel(), float(clf.intercept_[0])), 2))
# SVC linear 26.53   LinearSVC hinge 26.53   SGD matched 26.53
# LinearSVC default (squared hinge) 35.42   SGD default 389.57

# %%
import numpy as np
m = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
print(np.maximum(0.0, 1.0 - m).tolist())     # [3.0, 2.0, 1.0, 0.0, 0.0]   hinge
print(np.log1p(np.exp(-m)).round(3).tolist()) # [2.127, 1.313, 0.693, 0.313, 0.127]  logistic

# %% [markdown]
# ## 6.6.1 Multiclass strategies: one-vs-rest, one-vs-one, and native

# %%
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.metrics import accuracy_score

X, y = load_digits(return_X_y=True)                 # 1797 8-by-8 images, 10 classes (digits 0-9)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
ovr = OneVsRestClassifier(SVC(kernel="linear")).fit(Xtr, ytr)   # K models, one per class
ovo = OneVsOneClassifier(SVC(kernel="linear")).fit(Xtr, ytr)    # K(K-1)/2 models, one per pair
print("OvR sub-models:", len(ovr.estimators_))   # 10   one per digit
print("OvO sub-models:", len(ovo.estimators_))   # 45   = 10*9/2 pairs
print("OvR accuracy:", round(accuracy_score(yte, ovr.predict(Xte)), 3))   # 0.931
print("OvO accuracy:", round(accuracy_score(yte, ovo.predict(Xte)), 3))   # 0.974
soft = LogisticRegression(max_iter=5000).fit(Xtr, ytr)          # native softmax: no wrapper at all
print("softmax accuracy:", round(accuracy_score(yte, soft.predict(Xte)), 3))   # 0.954

# %% [markdown]
# ## 6.6.2 Multilabel classification: several labels at once

# %%
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, hamming_loss

yeast = fetch_openml("yeast", version=4, as_frame=False)   # 2417 genes, 103 features, 14 labels
X = yeast.data.astype(float)
Y = (yeast.target == "TRUE").astype(int)                   # the 14 label columns as 0/1
print("feature matrix:", X.shape)                    # (2417, 103)
print("label matrix:", Y.shape)                      # (2417, 14)
print("labels per gene (mean):", round(Y.sum(axis=1).mean(), 2))   # 4.24
print("labels on gene 0:", Y[0].tolist())            # [0,0,0,0,0,0,1,1,0,0,0,1,1,0]

Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.3, random_state=0)
ml = make_pipeline(StandardScaler(), KNeighborsClassifier()).fit(Xtr, Ytr)
pred = ml.predict(Xte)
print("subset accuracy:", round((pred == Yte).all(axis=1).mean(), 3))   # 0.187
print("Hamming loss:", round(hamming_loss(Yte, pred), 3))               # 0.211
print("micro-F1:", round(f1_score(Yte, pred, average="micro"), 3))      # 0.627

# %% [markdown]
# ## 6.6.3 Capturing label correlation: classifier chains and label powerset

# %%
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

yeast = fetch_openml("yeast", version=4, as_frame=False)     # the same 2417 genes as 6.6.2
X = yeast.data.astype(float)
Y = (yeast.target == "TRUE").astype(int)
Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.3, random_state=0)
base = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
br = OneVsRestClassifier(base).fit(Xtr, Ytr)                                   # binary relevance: labels independent
chain = ClassifierChain(base, order="random", random_state=0).fit(Xtr, Ytr)    # each predicted label feeds the next
pb, pc = br.predict(Xte), chain.predict(Xte)
print("binary relevance micro-F1:", round(f1_score(Yte, pb, average="micro"), 3),
      " macro-F1:", round(f1_score(Yte, pb, average="macro", zero_division=0), 3))
print("classifier chain micro-F1:", round(f1_score(Yte, pc, average="micro"), 3),
      " macro-F1:", round(f1_score(Yte, pc, average="macro", zero_division=0), 3))
# binary relevance micro-F1: 0.61   macro-F1: 0.373
# classifier chain micro-F1: 0.605  macro-F1: 0.382

# %%
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

yeast = fetch_openml("yeast", version=4, as_frame=False)     # the same 2417 genes and split
X = yeast.data.astype(float)
Y = (yeast.target == "TRUE").astype(int)
Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.3, random_state=0)

combos = sorted({tuple(row) for row in Ytr})                 # distinct label sets seen in training
combo_id = {c: i for i, c in enumerate(combos)}
y_lp = np.array([combo_id[tuple(row)] for row in Ytr])       # one multiclass label per combination
print("distinct label combinations:", len(combos))           # 181
lp = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr, y_lp)
pred = np.array([combos[i] for i in lp.predict(Xte)])        # map classes back to label rows
print("label powerset micro-F1:", round(f1_score(Yte, pred, average="micro"), 3),
      " macro-F1:", round(f1_score(Yte, pred, average="macro", zero_division=0), 3))
# label powerset micro-F1: 0.588   macro-F1: 0.41

# %%
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.multioutput import ClassifierChain
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

yeast = fetch_openml("yeast", version=4, as_frame=False)     # the same 2417 genes and split as 6.6.3
X = yeast.data.astype(float)
Y = (yeast.target == "TRUE").astype(int)
Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.3, random_state=0)
base = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))

def f1s(pred):                                               # (micro-F1, macro-F1) on the test genes
    return (round(f1_score(Yte, pred, average="micro"), 3),
            round(f1_score(Yte, pred, average="macro", zero_division=0), 3))

rng = np.random.default_rng(0)
m = 50                                                       # ensemble members, as Read et al. (2011) used on yeast
singles, counts, votes = [], [], np.zeros(Yte.shape)
for k in range(m):
    alone = ClassifierChain(base, order="random", random_state=k).fit(Xtr, Ytr).predict(Xte)
    singles.append(f1s(alone))                               # chain k on its own, trained on every gene
    counts.append(alone.sum(axis=0))                         # positives it predicts, label by label
    boot = rng.integers(0, len(Xtr), len(Xtr))               # member k: the same order, a bootstrap sample
    member = ClassifierChain(base, order="random", random_state=k).fit(Xtr[boot], Ytr[boot])
    votes += member.predict(Xte)                             # each member votes 0 or 1 on every label
micro, macro = zip(*singles)
print("single chains  micro-F1", min(micro), "to", max(micro), "  macro-F1", min(macro), "to", max(macro))
ecc = (votes / m >= 0.5).astype(int)                         # a label is on when at least half the members vote for it
mi, ma = f1s(ecc)
print("ensemble       micro-F1", mi, "  macro-F1", ma)
n = len(Yte)
print("labels per gene     true", round(Yte.sum() / n, 2), "  single chains", round(np.sum(counts) / (m * n), 2),
      "  ensemble", round(ecc.sum() / n, 2))
rare = np.argsort(Yte.sum(axis=0))[:4]                       # the four rarest labels among the test genes
print("four rarest labels  true", Yte[:, rare].sum(axis=0).tolist(),
      "  single chains", np.mean(counts, axis=0)[rare].round(1).tolist(),
      "  ensemble", ecc[:, rare].sum(axis=0).tolist())

# %% [markdown]
# ## 6.6.4 Multioutput classification: several targets together

# %%
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from sklearn.utils.multiclass import type_of_target
X, y = load_digits(return_X_y=True)
parity = y % 2                       # target 1: odd (1) or even (0), binary
mod3 = y % 3                         # target 2: residue mod 3, three classes
Y = np.column_stack([parity, mod3])  # two genuinely different targets
print(type_of_target(Y))             # multiclass-multioutput
Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.3, random_state=0)
mo = MultiOutputClassifier(DecisionTreeClassifier(max_depth=8, random_state=0)).fit(Xtr, Ytr)
pred = mo.predict(Xte)
print("prediction shape:", pred.shape)   # (540, 2)   one column per target
print("per-target accuracy:", [round(accuracy_score(Yte[:, j], pred[:, j]), 3) for j in range(2)])
# prediction shape (540, 2); per-target accuracy [0.919, 0.856]

# %% [markdown]
# ## 6.7.2 The leaderboard

# %%
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

X, y = load_breast_cancer(return_X_y=True)        # 569 tumors, 30 features, malignant=0 / benign=1
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                          stratify=y, random_state=0)

# Scale models whose geometry or penalty depends on feature units.
# Plain LDA and the tree do not.
models = {
    "kNN (k=7)":     make_pipeline(StandardScaler(), KNeighborsClassifier(7)),
    "GaussianNB":    make_pipeline(StandardScaler(), GaussianNB()),
    "LDA":           LinearDiscriminantAnalysis(),
    "QDA":           make_pipeline(StandardScaler(), QuadraticDiscriminantAnalysis(reg_param=0.1)),
    "Decision tree": DecisionTreeClassifier(max_depth=3, random_state=0),
    "Logistic reg.": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
    "SVM (linear)":  make_pipeline(StandardScaler(), SVC(kernel="linear", random_state=0)),
    "SVM (RBF)":     make_pipeline(StandardScaler(), SVC(kernel="rbf", random_state=0)),
}

cv = StratifiedKFold(5, shuffle=True, random_state=0)   # the same folds for every model
print(f"{'model':16} CV acc (SD)    CV AUC (SD)")
auc_by_model = {}
for name, m in models.items():
    acc = cross_val_score(m, X_tr, y_tr, cv=cv, scoring="accuracy")   # one score per fold
    auc = cross_val_score(m, X_tr, y_tr, cv=cv, scoring="roc_auc")    # uses decision_function for SVC
    auc_by_model[name] = auc.mean()
    print(f"{name:16} {acc.mean():.3f} ({acc.std():.3f})  {auc.mean():.3f} ({auc.std():.3f})")
print("best by CV AUC:", max(auc_by_model, key=auc_by_model.get))

# %%
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

X, y = load_breast_cancer(return_X_y=True)
X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
leaders = {
    "Logistic reg.": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
    "SVM (linear)":  make_pipeline(StandardScaler(), SVC(kernel="linear", random_state=0)),
    "SVM (RBF)":     make_pipeline(StandardScaler(), SVC(kernel="rbf", random_state=0)),
}
for seed in range(1, 6):                             # deal the same rows into five new folds
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    auc = {name: cross_val_score(m, X_tr, y_tr, cv=cv, scoring="roc_auc").mean()
           for name, m in leaders.items()}
    print(seed, max(auc, key=auc.get), round(max(auc.values()) - min(auc.values()), 4))
# 1 Logistic reg. 0.0012 | 2 SVM (RBF) 0.0023 | 3 SVM (RBF) 0.003 | 4 SVM (RBF) 0.0001 | 5 Logistic reg. 0.0005

# %%
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import brier_score_loss
X, y = load_breast_cancer(return_X_y=True)
X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
cv = StratifiedKFold(5, shuffle=True, random_state=0)
proba = cross_val_predict(make_pipeline(StandardScaler(), GaussianNB()),
                          X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
low, high = proba < 0.1, proba >= 0.9
print(round(brier_score_loss(y_tr, proba), 3))                              # 0.065
print(int(low.sum()), round(proba[low].mean(), 3), round(y_tr[low].mean(), 3))    # 153  0.001  0.078
print(int(high.sum()), round(proba[high].mean(), 3), round(y_tr[high].mean(), 3)) # 264  0.999  0.947

# %% [markdown]
# ## 6.7.3 Judging the winner on held-out data

# %%
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_auc_score
X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", random_state=0)).fit(X_tr, y_tr)
print(confusion_matrix(y_te, svm.predict(X_te)))      # [[50  3]  [ 3 87]]  (class 0 = malignant)
print(round(svm.score(X_te, y_te), 2),                # 0.96 accuracy
      round(roc_auc_score(y_te, svm.decision_function(X_te)), 3))   # 0.993 ROC AUC

from sklearn.calibration import CalibratedClassifierCV
cal = make_pipeline(StandardScaler(),
                    CalibratedClassifierCV(SVC(kernel="rbf", random_state=0),
                                           ensemble=False)).fit(X_tr, y_tr)
print(round(roc_auc_score(y_te, cal.predict_proba(X_te)[:, 1]), 3))  # 0.993 from probabilities

# %%
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
X, y = load_breast_cancer(return_X_y=True)
X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)   # test rows unused
cv = StratifiedKFold(5, shuffle=True, random_state=0)               # the folds of Example 6-13
f = cross_val_predict(make_pipeline(StandardScaler(), SVC(kernel="rbf", random_state=0)),
                      X_tr, y_tr, cv=cv, method="decision_function")   # out-of-fold scores, benign above 0
malignant = y_tr == 0
print("   t  caught  missed  false alarms  cost")
for t in [0.0, 0.25, 0.5, 1.0, 1.5]:
    flag = f < t                                                      # flag as malignant below the threshold
    caught, missed = int((flag & malignant).sum()), int((~flag & malignant).sum())
    alarms = int((flag & ~malignant).sum())
    print(f"{t:4.2f}  {caught:6d}  {missed:6d}  {alarms:12d}  {5 * missed + alarms:4d}")   # 5 per miss, 1 per alarm

# %% [markdown]
# ## 6.8 Coding exercises

# %%
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

def knn_predict(X_train, y_train, X_query, k=7):
    ...

X, y = load_wine(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
print(X_tr.shape, knn_predict(X_tr, y_tr, X_te))

# %%
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

def sweep_k(X, y, ks, cv):
    ...

peng = sns.load_dataset("penguins")
num = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
peng = peng.dropna(subset=num)
X, y = peng[num].to_numpy(), peng["species"].to_numpy()
cv = StratifiedKFold(5, shuffle=True, random_state=0)
print(X.shape, sweep_k(X, y, [1, 3, 5, 11, 21, 51, 101], cv))

# %%
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

class GaussianNBScratch:
    def fit(self, X, y):
        ...

    def predict(self, X):
        ...

X, y = load_wine(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
print(X_tr.shape, GaussianNBScratch().fit(X_tr, y_tr))

# %%
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

def smoothing_gap(train, test, alphas):
    ...

cats, strip = ["rec.autos", "rec.motorcycles"], ("headers", "footers", "quotes")
train = fetch_20newsgroups(subset="train", categories=cats, remove=strip)
test = fetch_20newsgroups(subset="test", categories=cats, remove=strip)
print(len(train.data), len(test.data), smoothing_gap(train, test, [1e-10, 1.0]))

# %%
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis as LDA,
                                           QuadraticDiscriminantAnalysis as QDA)

def compare_lda_qda(X, y):
    ...

def project_and_classify(X_tr, y_tr, X_te, y_te, n_components):
    ...

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
print(X.shape, compare_lda_qda(X, y), project_and_classify(X_tr, y_tr, X_te, y_te, 9))

# %%
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def depth_sweep(X_tr, y_tr, X_te, y_te, depths):
    ...

def prune_by_cv(X_tr, y_tr):
    ...

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
print(X_tr.shape, depth_sweep(X_tr, y_tr, X_te, y_te, [1, 2, 4, 6, 8, 10, None]), prune_by_cv(X_tr, y_tr))

# %%
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

def compare_kernels(X, y):
    ...

def scaling_gap(X_tr, y_tr, X_te, y_te):
    ...

X, y = load_wine(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
print(X.shape, compare_kernels(X, y), scaling_gap(X_tr, y_tr, X_te, y_te))

# %%
import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier

def hinge_primal(w, b, X, m, C):
    ...

X, y = load_digits(return_X_y=True)
keep = (y == 3) | (y == 8)
Xs = StandardScaler().fit_transform(X[keep])
m = np.where(y[keep] == 8, 1, -1)                    # labels as -1 / +1 for the hinge
print(Xs.shape, hinge_primal(np.zeros(Xs.shape[1]), 0.0, Xs, m, 0.1))

# %%
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

def compare_strategies(X_tr, y_tr, X_te, y_te):
    ...

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
print(X_tr.shape, compare_strategies(X_tr, y_tr, X_te, y_te))

# %%
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def showdown(X_tr, y_tr, X_te, y_te):
    ...

pima = fetch_openml("diabetes", version=1, as_frame=False)
X = pima.data.astype(float)
y = (pima.target == "tested_positive").astype(int)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
print(X.shape, int(y.sum()), showdown(X_tr, y_tr, X_te, y_te))

# %%
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

def search_c_gamma(pipe, Xtr, ytr):
    ...

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
pipe = Pipeline([("scaler", StandardScaler()), ("svc", SVC(kernel="rbf"))])
print(Xtr.shape, search_c_gamma(pipe, Xtr, ytr))

# %%
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

def best_split(X, y):
    ...

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
print(Xtr.shape, best_split(Xtr, ytr))

# %%
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

def epsilon_sweep(Xd, yd):
    ...

Xd, yd = load_diabetes(return_X_y=True)
Xd = StandardScaler().fit_transform(Xd)
print(Xd.shape, epsilon_sweep(Xd, yd))

# %%
from sklearn.datasets import fetch_openml

def compare_multilabel(X, Y):
    ...

yeast = fetch_openml("yeast", version=4, as_frame=False)
X = yeast.data.astype(float)
Y = (yeast.target == "TRUE").astype(int)
print(X.shape, Y.shape, compare_multilabel(X, Y))

# %%
from sklearn.datasets import load_wine, load_breast_cancer
from sklearn.preprocessing import StandardScaler

def fisher_and_shrinkage(Xw, yw, X, y):
    ...

Xw, yw = load_wine(return_X_y=True)
Xw = StandardScaler().fit_transform(Xw)
X, y = load_breast_cancer(return_X_y=True)
print(Xw.shape, X.shape, fisher_and_shrinkage(Xw, yw, X, y))
