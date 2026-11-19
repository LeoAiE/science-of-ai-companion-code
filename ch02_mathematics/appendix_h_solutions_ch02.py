"""Appendix H solutions for Chapter 2, as runnable cells.

Generated from Solutions.md (Appendix H). Every code block of this chapter's solutions appears here
verbatim, split into one cell per exercise or drill topic, so each answer can be run and checked against
the expected outputs it states. Keep Solutions.md as the source; regenerate after any change.
"""

# %% [markdown]
# ## 2.10 lab tasks

# %%
import numpy as np
from sklearn.datasets import load_diabetes

# The data of the starter block: the 200 points of Example 2-9 and the 442 patients of Example 2-10.
rng = np.random.default_rng(0)
X_syn = rng.uniform(-3, 3, size=200)
y_syn = 2 * X_syn + 1 + rng.normal(0, 1, size=200)
data = load_diabetes(scaled=False)
bmi, y_db = data.data[:, 2], data.target
X_db = (bmi - bmi.mean()) / bmi.std()

def gradient_descent(X, y, eta, steps=1000):
    w, b, losses = 0.0, 0.0, []
    for step in range(steps):
        error = w * X + b - y                         # the error of the current line at every point
        losses.append(np.mean(error ** 2))            # the loss before this step
        grad_w = (2 / len(X)) * np.sum(error * X)     # the two partial derivatives of the lab
        grad_b = (2 / len(X)) * np.sum(error)
        w -= eta * grad_w
        b -= eta * grad_b
    return w, b, losses

def least_squares_line(X, y):
    A = np.column_stack([X, np.ones(len(X))])         # a slope column and an intercept column
    return np.linalg.solve(A.T @ A, A.T @ y)          # the normal equation of 2.1.11

def minibatch_gradient_descent(X, y, eta, batch_size, steps=1000, seed=0):
    rng_mb = np.random.default_rng(seed)
    w, b = 0.0, 0.0
    for step in range(steps):
        idx = rng_mb.choice(len(X), size=batch_size, replace=False)   # a fresh random batch at every step
        error = w * X[idx] + b - y[idx]
        w -= eta * (2 / batch_size) * np.sum(error * X[idx])
        b -= eta * (2 / batch_size) * np.sum(error)
    return w, b

def hessian_eigenvalues(X):
    # The Hessian of the loss from the preparation step of 2.10: 2 [[mean of x^2, mean of x], [mean of x, 1]]
    return np.linalg.eigvalsh(2 * np.array([[np.mean(X ** 2), X.mean()], [X.mean(), 1.0]]))

# %% [markdown]
# 1. Example 2-9 at eta = 0.5, then at eta = 0.001

# %%
# 1. Example 2-9 at eta = 0.5, then at eta = 0.001
lam_syn = hessian_eigenvalues(X_syn)
print([round(float(v), 2) for v in lam_syn], round(float(2 / lam_syn.max()), 3))   # expected output: [1.95, 6.7] 0.299
w, b, losses = gradient_descent(X_syn, y_syn, eta=0.5)
print(np.round(losses[:5], 1))                        # expected output: [   15.5    78.1   426.4  2349.5 12969.2]
print(np.round(np.array(losses[1:5]) / np.array(losses[:4]), 2))   # expected output: [5.04 5.46 5.51 5.52]
print(round(w, 3), round(b, 3))                       # expected output: nan nan
print(round(losses[0], 3), round(losses[-1], 3))      # expected output: 15.495 nan
# The safe range of 2.6.3 is eta < 2/6.7, about 0.3, so eta = 0.5 overshoots: every step multiplies the error
# along the steep direction by 1 - 0.5 * 6.7, about -2.35, and the loss by about 2.35^2 = 5.52, the ratio the
# second line settles on. The loss grows until the numbers overflow, NumPy prints RuntimeWarning messages about
# an overflow and then an invalid value on the error stream, and w and b end as nan: the divergence regime.
w, b, losses = gradient_descent(X_syn, y_syn, eta=0.001)
print(round(w, 3), round(b, 3))                       # expected output: 1.966 0.819
print(round(losses[0], 3), round(losses[-1], 3))      # expected output: 15.495 1.052
# Stable but slow: after 1,000 steps the intercept is still well short of the 0.921 that eta = 0.05 reaches and
# the loss has not yet come down to 1.042, so the run needs far more iterations: the too-small regime.

# %% [markdown]
# 2. Standardized BMI at eta = 0.5, a single step

# %%
# 2. Standardized BMI at eta = 0.5, a single step
print([round(float(v), 3) for v in hessian_eigenvalues(X_db)])   # expected output: [2.0, 2.0]
w, b, _ = gradient_descent(X_db, y_db, eta=0.5, steps=1)
print(round(w, 3), round(b, 3))                       # expected output: 45.16 152.133
print(least_squares_line(X_db, y_db).round(3))        # expected output: [ 45.16  152.133]
# Standardized x has mean 0 and mean square 1, so the Hessian is 2I and the per-step factor 1 - 2 * eta is
# 1 - 2 * 0.5 = 0 in every direction: the first step removes the whole error and lands on the least-squares
# line, the perfectly conditioned bowl of 2.6.3.

# %% [markdown]
# 3. Raw BMI at eta = 0.05, then at eta = 0.001

# %%
# 3. Raw BMI at eta = 0.05, then at eta = 0.001
lam_raw = hessian_eigenvalues(bmi)
print(round(float(lam_raw[0]), 3), round(float(lam_raw[1]), 1), round(float(2 / lam_raw[1]), 4))
# expected output: 0.054 1432.3 0.0014
w, b, _ = gradient_descent(bmi, y_db, eta=0.05)
print(len(bmi), round(w, 3), round(b, 3))             # expected output: 442 nan nan
w, b, _ = gradient_descent(bmi, y_db, eta=0.001)
print(round(w, 3), round(b, 3))                       # expected output: 6.112 -6.029
print(least_squares_line(bmi, y_db).round(3))         # expected output: [  10.233 -117.773]
print(round(float((1 - 0.001 * lam_raw[0]) ** 1000), 3))   # expected output: 0.947
# On the raw scale the eigenvalues are about 0.054 and 1432.3, a condition number near 26,000, and the safe limit
# 2/1432.3 is about 0.0014. At eta = 0.05, about 36 times that limit, the steep direction is multiplied by
# 1 - 0.05 * 1432.3, about -70.6, at every step, so the run overflows to nan (the companion notebook again shows
# RuntimeWarning messages about an overflow and an invalid value). At eta = 0.001 the run is stable,
# but the flat direction, nearly the intercept, keeps 1 - 0.001 * 0.054 of its error per step, about 0.947 of it
# after 1,000 steps, so the intercept has covered only about 5 percent of its way to -117.773.

# %% [markdown]
# 4. Minibatch stochastic gradient descent: 20 random points per step, three seeds

# %%
# 4. Minibatch stochastic gradient descent: 20 random points per step, three seeds
w_ls, b_ls = least_squares_line(X_syn, y_syn)
full_loss = lambda w, b: float(np.mean((w * X_syn + b - y_syn) ** 2))   # the loss on all 200 points
print(round(w_ls, 3), round(b_ls, 3), round(full_loss(w_ls, b_ls), 3))   # expected output: 1.958 0.921 1.042
for seed in [0, 1, 2]:
    w, b = minibatch_gradient_descent(X_syn, y_syn, eta=0.05, batch_size=20, seed=seed)
    print(seed, round(w, 3), round(b, 3), round(full_loss(w, b), 3))
# expected output: 0 1.994 0.877 1.048
# expected output: 1 1.94 0.91 1.043
# expected output: 2 1.981 0.923 1.044
# Each step now sees 20 random points, so its gradient is a noisy but unbiased estimate of the full-data gradient
# (2.6.4) at a tenth of the cost. The three runs end within 0.05 of the full-batch slope and intercept, each on its
# own noisy path, and their losses on all 200 points sit just above the floor of 1.042.

# %% [markdown]
# ## 2.11 coding exercises

# %%
import numpy as np

# %% [markdown]
# 1. A dense layer by hand: forward pass W x + b with explicit loops

# %%
# 1. A dense layer by hand: forward pass W x + b with explicit loops
def dense_layer(W, x, b):
    out = np.zeros(W.shape[0])
    for i in range(W.shape[0]):            # one output per row (neuron) of W
        s = 0.0
        for j in range(W.shape[1]):        # dot the row with the input vector
            s += W[i, j] * x[j]
        out[i] = s + b[i]                  # add this neuron's bias
    return out

rng = np.random.default_rng(0)
W = rng.standard_normal((4, 3)); x = rng.standard_normal(3); b = rng.standard_normal(4)
print(np.round(dense_layer(W, x, b), 4))   # expected output: [-1.7936 -1.1215 -2.6786  3.4387]
print(np.allclose(dense_layer(W, x, b), W @ x + b))  # expected output: True
# W is (4,3), x is (3,): the inner dimension 3 must match for the multiply to be defined.

# %% [markdown]
# 2. Ill conditioning and the ridge cure

# %%
# 2. Ill conditioning and the ridge cure
def ridge_solve(A, b, lam):
    n = A.shape[1]
    return np.linalg.solve(A.T @ A + lam * np.eye(n), A.T @ b)   # the ridge system (A^T A + lam I) x = A^T b

A = np.array([[1.0, 1.0], [1.0, 1.0001]])
bvec = np.array([2.0, 2.0001])
print("cond:", round(float(np.linalg.cond(A)), 1))          # expected output: cond: 40002.0
print("solve:", np.round(np.linalg.solve(A, bvec), 4))      # expected output: solve: [1. 1.]
b_pert = bvec + np.array([0.0, 0.001])                      # tiny nudge to one entry
print("perturbed:", np.round(np.linalg.solve(A, b_pert), 4))  # expected output: perturbed: [-9. 11.]
lam = 0.01
x_ridge = ridge_solve(A, bvec, lam)
print("ridge:", np.round(x_ridge, 4))                       # expected output: ridge: [0.9975 0.9975]
# The ridge term is added to A^T A, so the like-for-like comparison is A^T A before and after it.
print("cond(A)^2:", f"{np.linalg.cond(A)**2:.4g}")          # expected output: cond(A)^2: 1.6e+09
print("cond(A^T A):", f"{np.linalg.cond(A.T @ A):.4g}")     # expected output: cond(A^T A): 1.6e+09
print("ridge cond:", round(float(np.linalg.cond(A.T @ A + lam*np.eye(2))), 1))  # expected output: ridge cond: 401.0
cond_ridge = np.linalg.cond(A.T @ A + lam*np.eye(2))
print("drop:", round(float(np.linalg.cond(A) / cond_ridge), 1), f"{np.linalg.cond(A.T @ A) / cond_ridge:.3g}")  # expected output: drop: 99.8 3.99e+06
# The singular values of A^T A are the squares of those of A, so forming A^T A squares the condition number,
# 40002 squared, about 1.6e9. Adding 0.01 I cuts it to 401: against A^T A, the matrix the ridge term is added
# to, that is the like-for-like drop the exercise asks for, about four million times (3.99e6); against solving
# A x = b directly it is about 100 times (99.8). The price is the small bias in the ridge solution, 0.9975
# where the exact solution has 1.

# %% [markdown]
# 3. PCA from scratch on the wine dataset

# %%
# 3. PCA from scratch on the wine dataset
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
def explained_variance_ratio(X):
    Xs = StandardScaler().fit_transform(X)        # standardize the columns first
    cov = np.cov(Xs, rowvar=False)                # 13x13 covariance of standardized features
    vals, vecs = np.linalg.eigh(cov)              # symmetric eigendecomposition
    vals = vals[np.argsort(vals)[::-1]]           # largest eigenvalue first
    return vals / vals.sum()                      # explained-variance ratio

wine = load_wine()
Xs = StandardScaler().fit_transform(wine.data)    # the standardized table, for scikit-learn's PCA below
evr = explained_variance_ratio(wine.data)
print("scratch EVR top3:", np.round(evr[:3], 4))  # expected output: scratch EVR top3: [0.362  0.1921 0.1112]
pca = PCA().fit(Xs)
print("sklearn EVR top3:", np.round(pca.explained_variance_ratio_[:3], 4))  # expected output: sklearn EVR top3: [0.362  0.1921 0.1112]
print("first 5 cumulative:", round(float(evr[:5].sum()), 4))  # expected output: first 5 cumulative: 0.8016

# %% [markdown]
# 4. SVD image compression on one digit

# %%
# 4. SVD image compression on one digit
from sklearn.datasets import load_digits
def relative_error(A, k):
    U, s, Vt = np.linalg.svd(A)
    A_k = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]      # rebuilt from the top k singular values
    return np.linalg.norm(A - A_k) / np.linalg.norm(A)   # relative Frobenius error

digits = load_digits()
img = digits.images[0]                            # 8x8 grayscale
U, s, Vt = np.linalg.svd(img)
print("singular values:", np.round(s, 3))
# expected output: singular values: [48.308 24.956  8.021  6.029  3.534  0.616  0.     0.   ]
print("rank:", np.linalg.matrix_rank(img))        # expected output: rank: 6
for k in [1, 2, 4, 8]:
    print(k, round(float(relative_error(img, k)), 4))
# expected output: 1 0.4897
# expected output: 2 0.1923
# expected output: 4 0.0648
# expected output: 8 0.0
# Two singular values are exactly zero, so this image has rank 6, not 8: the rebuilt A_k equals A as soon as k
# reaches 6, and k = 8 is the full size of the matrix rather than its rank.

# %% [markdown]
# 5. Cosine-similarity nearest neighbor on digits

# %%
# 5. Cosine-similarity nearest neighbor on digits
def most_similar(Xn, i):
    sims = Xn @ Xn[i]                                # cosine similarity of row i to every row
    sims[i] = -np.inf                                # exclude the query itself
    return int(np.argmax(sims))

def one_nn_accuracy(Xn, y):
    S = Xn @ Xn.T                                    # every pairwise cosine similarity at once
    np.fill_diagonal(S, -np.inf)                     # forbid matching an image to itself
    return float((y[S.argmax(axis=1)] == y).mean())

X, y = digits.data, digits.target
Xn = X / np.linalg.norm(X, axis=1, keepdims=True)  # unit length: dot == cosine
nn = most_similar(Xn, 0)
print("query:", y[0], "nn:", y[nn], "sim:", round(float(Xn[nn] @ Xn[0]), 4))
# expected output: query: 0 nn: 0 sim: 0.9807
print("1-NN cosine acc:", round(one_nn_accuracy(Xn[:300], y[:300]), 4))  # expected output: 1-NN cosine acc: 0.99

# %% [markdown]
# 6. Gradient descent linear regression, end to end (the chapter's small project)

# %%
# 6. Gradient descent linear regression, end to end (the chapter's small project)
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
def mse_gradient(X, y, w):
    return (2 / len(y)) * X.T @ (X @ w - y)         # closed-form gradient of the MSE

def numerical_gradient(f, w, eps=1e-5):
    grad = np.zeros(len(w))
    for j in range(len(w)):                         # central-difference check, one coordinate at a time
        wp = w.copy(); wp[j] += eps
        wm = w.copy(); wm[j] -= eps
        grad[j] = (f(wp) - f(wm)) / (2 * eps)
    return grad

def gradient_descent(X, y, eta, steps):              # all weights at once; replaces the one-feature lab version
    w = np.zeros(X.shape[1])
    for step in range(steps):
        w -= eta * mse_gradient(X, y, w)
    return w

db = load_diabetes()
Xd = StandardScaler().fit_transform(db.data)
yd = (db.target - db.target.mean()) / db.target.std()
n, d = Xd.shape
mse = lambda w: np.mean((Xd @ w - yd) ** 2)
w0 = rng.standard_normal(d)
print("grad check:", np.allclose(mse_gradient(Xd, yd, w0), numerical_gradient(mse, w0), atol=1e-6))  # expected output: grad check: True
w = gradient_descent(Xd, yd, eta=0.1, steps=20000)
w_normal = np.linalg.solve(Xd.T @ Xd, Xd.T @ yd)   # normal equation
w_sklearn = LinearRegression(fit_intercept=False).fit(Xd, yd).coef_
print("GD == normal eq:", np.allclose(w, w_normal, atol=1e-3))   # expected output: GD == normal eq: True
print("GD == sklearn:", np.allclose(w, w_sklearn, atol=1e-3))    # expected output: GD == sklearn: True
print("final MSE:", round(float(mse(w)), 4))       # expected output: final MSE: 0.4823
L6 = float(np.linalg.eigvalsh((2 / n) * Xd.T @ Xd).max())   # the Hessian of the MSE is (2/n) X^T X at every w
print("safe eta below:", round(2 / L6, 4))         # expected output: safe eta below: 0.2485
for eta in [0.2, 0.3]:                             # the experiment: one rate on each side of 2/L
    w_try = gradient_descent(Xd, yd, eta, steps=200)
    print("eta", eta, "MSE after 200 steps:", f"{mse(w_try):.4g}")
# expected output: eta 0.2 MSE after 200 steps: 0.4832
# expected output: eta 0.3 MSE after 200 steps: 5.415e+59
# Above 2/L the error along the steepest direction is multiplied by 1 - 0.3 * 8.05, about -1.41, at every step,
# so the loss roughly doubles per step (1.41^2 is about 2.0) instead of settling near 0.48: the divergence of 2.6.3.

# %% [markdown]
# 7. Bayes for a rare disease

# %%
# 7. Bayes for a rare disease
def posterior(prior, sensitivity, specificity):
    p_pos = sensitivity * prior + (1 - specificity) * (1 - prior)   # total probability of a positive test
    return sensitivity * prior / p_pos                                # Bayes: P(disease | +)

post = posterior(0.001, 0.99, 0.99)
print("after 1 positive:", round(post, 4))          # expected output: after 1 positive: 0.0902
post2 = posterior(post, 0.99, 0.99)                 # yesterday's posterior is today's prior
print("after 2 positives:", round(post2, 4))        # expected output: after 2 positives: 0.9075

# %% [markdown]
# 8. Bootstrap versus formula standard error

# %%
# 8. Bootstrap versus formula standard error
def formula_se(x):
    return x.std(ddof=1) / np.sqrt(len(x))          # s / sqrt(n), with the sample standard deviation s

def bootstrap_se(x, n_boot=2000, seed=42):
    rng_b = np.random.default_rng(seed)
    boot = np.array([rng_b.choice(x, size=len(x), replace=True).mean() for _ in range(n_boot)])
    return boot.std(ddof=1)                          # the spread of the resampled means

t = db.target.astype(float)
print("mean:", round(float(t.mean()), 3))           # expected output: mean: 152.133
print("formula SE:", round(float(formula_se(t)), 3))   # expected output: formula SE: 3.667
print("bootstrap SE:", round(float(bootstrap_se(t)), 3))  # expected output: bootstrap SE: 3.707

# %% [markdown]
# 9. Softmax and cross-entropy from scratch

# %%
# 9. Softmax and cross-entropy from scratch
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
bc = load_breast_cancer()
Xb = StandardScaler().fit_transform(bc.data); yb = bc.target
proba = LogisticRegression(max_iter=5000).fit(Xb, yb).predict_proba(Xb)
def cross_entropy(y_true, probs, eps=1e-15):
    p = np.clip(probs, eps, 1 - eps)                # keep the log away from zero
    return -np.mean(np.log(p[np.arange(len(y_true)), y_true]))
print("my CE:", round(float(cross_entropy(yb, proba)), 6))   # expected output: my CE: 0.053385
print("log_loss:", round(float(log_loss(yb, proba)), 6))     # expected output: log_loss: 0.053385
def softmax(z):
    z = z - z.max()                                 # subtract the max for numerical stability
    e = np.exp(z); return e / e.sum()
sm = softmax(np.array([2.0, 1.0, 0.1]))
print("softmax:", np.round(sm, 4), "sum:", round(float(sm.sum()), 6))
# expected output: softmax: [0.659  0.2424 0.0986] sum: 1.0

# %% [markdown]
# 10. A two-sample t-test by hand on iris petal length

# %%
# 10. A two-sample t-test by hand on iris petal length
from sklearn.datasets import load_iris
from scipy import stats
iris = load_iris()
petal = iris.data[:, 2]
g0 = petal[iris.target == 0]      # setosa
g1 = petal[iris.target == 1]      # versicolor
def welch_t(a, b):
    return (a.mean() - b.mean()) / np.sqrt(a.var(ddof=1)/len(a) + b.var(ddof=1)/len(b))   # 2.4.5

t_manual = welch_t(g0, g1)
t_scipy, p_scipy = stats.ttest_ind(g0, g1, equal_var=False)
print("manual t:", round(float(t_manual), 3))       # expected output: manual t: -39.493
print("scipy t:", round(float(t_scipy), 3), "p:", f"{p_scipy:.2e}")  # expected output: scipy t: -39.493 p: 9.93e-46
# p far below 0.05: the petal lengths of the two species differ by far more than sampling noise.

# %% [markdown]
# 11. Projection by the hat matrix on the diabetes data

# %%
# 11. Projection by the hat matrix on the diabetes data
X11, y11 = load_diabetes(return_X_y=True)
X11 = np.column_stack([np.ones(len(y11)), X11])         # 442 by 11: the intercept column, then the ten features
def hat_matrix(X):
    return X @ np.linalg.solve(X.T @ X, X.T)            # X (X^T X)^{-1} X^T, solved rather than inverted

def hat_matrix_qr(X):
    Q, _ = np.linalg.qr(X)                              # reduced QR: orthonormal columns spanning the same space
    return Q @ Q.T

H = hat_matrix(X11)
print("H == QQ^T:", np.allclose(H, hat_matrix_qr(X11)))  # expected output: H == QQ^T: True
print("symmetric:", np.allclose(H, H.T), "idempotent:", np.allclose(H @ H, H))
# expected output: symmetric: True idempotent: True
r11 = y11 - H @ y11                                     # the residual: the part of y the columns cannot reach
print("X^T r == 0:", np.allclose(X11.T @ r11, 0))      # expected output: X^T r == 0: True
pred11 = LinearRegression(fit_intercept=False).fit(X11, y11).predict(X11)
print("Hy == sklearn:", np.allclose(H @ y11, pred11))  # expected output: Hy == sklearn: True
# With G = X^T X: H^T = X (G^{-1})^T X^T = X G^{-1} X^T = H, because the inverse of a symmetric matrix is symmetric
# (2.1.4), and H H = X G^{-1} (X^T X) G^{-1} X^T = X G^{-1} X^T = H. Q Q^T is the projection onto the same column
# space written with an orthonormal basis of it, so the two must agree, and X^T r = X^T y - X^T X G^{-1} X^T y = 0
# is the normal equation of 2.1.11 restated.

# %% [markdown]
# 12. The pseudoinverse and the shortest solution

# %%
# 12. The pseudoinverse and the shortest solution
from scipy.linalg import null_space
X12 = np.column_stack([X11, X11[:, 3] / 2.54])         # BMI (column 3 after the ones) again, in new units
print(X12.shape, np.linalg.matrix_rank(X12))           # expected output: (442, 12) 11
def pinv_from_svd(X, rtol=1e-10):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    s_plus = np.zeros_like(s)
    keep = s > rtol * s[0]                              # singular values above the tolerance count as nonzero
    s_plus[keep] = 1 / s[keep]                          # invert those and leave the rest at zero (2.1.10)
    return Vt.T @ np.diag(s_plus) @ U.T
s12 = np.linalg.svd(X12, compute_uv=False)
print("numerically zero singular values:", int(np.sum(s12 <= 1e-10 * s12[0])))  # expected output: numerically zero singular values: 1
X12p = pinv_from_svd(X12)
print("pinv match:", np.allclose(X12p, np.linalg.pinv(X12)))  # expected output: pinv match: True
w_plus = X12p @ y11
# The NumPy documentation of np.linalg.lstsq states that among several minimizers it returns the one of smallest 2-norm.
print("lstsq match:", np.allclose(w_plus, np.linalg.lstsq(X12, y11, rcond=None)[0]))  # expected output: lstsq match: True
n12 = np.zeros(12); n12[3], n12[11] = 1.0, -2.54       # BMI minus 2.54 times its copy is the zero column
print("null_space agrees:", np.isclose(abs(float(null_space(X12).ravel() @ n12)), np.linalg.norm(n12)))
# expected output: null_space agrees: True
print("same predictions:", np.allclose(X12 @ w_plus, X12 @ (w_plus + n12)))  # expected output: same predictions: True
print("norms:", round(float(np.linalg.norm(w_plus)), 4), round(float(np.linalg.norm(w_plus + n12)), 4))
# expected output: norms: 1373.0712 1373.0739
gap = (w_plus + n12) @ (w_plus + n12) - w_plus @ w_plus
print("squared norm gap:", round(float(gap), 4), "n.n:", round(float(n12 @ n12), 4))
# expected output: squared norm gap: 7.4516 n.n: 7.4516
# w+ = V Sigma+ U^T y is a combination of the right singular vectors with nonzero singular values, which span the
# row space; the row space is orthogonal to the nullspace (2.1.7), so w+ . n = 0 and
# ||w+ + t n||^2 = ||w+||^2 + 2t (w+ . n) + t^2 ||n||^2 = ||w+||^2 + t^2 ||n||^2, larger for every t other than 0.
print("BMI weight shared:", round(float(w_plus[3]), 3), round(float(w_plus[11]), 3))
# expected output: BMI weight shared: 450.083 177.198
print("BMI weight of exercise 11:", round(float(np.linalg.lstsq(X11, y11, rcond=None)[0][3]), 3),
      "recombined:", round(float(w_plus[3] + w_plus[11] / 2.54), 3))
# expected output: BMI weight of exercise 11: 519.846 recombined: 519.846
# Any split a + b/2.54 = 519.846 of that weight fits equally well, and the shortest one has b/a = 1/2.54
# (177.198/450.083 is about 0.3937), which is the split the pseudoinverse picked.

# %% [markdown]
# 13. Least squares by Gram-Schmidt QR, and ridge by one Cholesky factorization

# %%
# 13. Least squares by Gram-Schmidt QR, and ridge by one Cholesky factorization
from scipy.linalg import cho_factor, cho_solve
X13 = StandardScaler().fit_transform(db.data)          # z-scores, as in exercise 6
y13 = db.target - db.target.mean()                     # a centered target needs no intercept column
def gram_schmidt_qr(X):
    m, k = X.shape
    Q, R = np.zeros((m, k)), np.zeros((k, k))
    for j in range(k):
        v = X[:, j].copy()
        for i in range(j):
            R[i, j] = Q[:, i] @ X[:, j]                 # classical: the dot product with the original column
            v -= R[i, j] * Q[:, i]                      # remove the projection onto q_i (2.1.11)
        R[j, j] = np.linalg.norm(v)                     # the length that is left
        Q[:, j] = v / R[j, j]
    return Q, R
def back_substitute(R, c):
    w = np.zeros(len(c))
    for i in range(len(c) - 1, -1, -1):                 # the last row first, as in 2.1.4
        w[i] = (c[i] - R[i, i + 1:] @ w[i + 1:]) / R[i, i]
    return w
Q13, R13 = gram_schmidt_qr(X13)
print("Q^T Q == I:", np.allclose(Q13.T @ Q13, np.eye(10)), "QR == X:", np.allclose(Q13 @ R13, X13))
# expected output: Q^T Q == I: True QR == X: True
w_qr = back_substitute(R13, Q13.T @ y13)               # X^T X w = X^T y becomes R^T R w = R^T Q^T y, so R w = Q^T y
print("QR == lstsq:", np.allclose(w_qr, np.linalg.lstsq(X13, y13, rcond=None)[0]))  # expected output: QR == lstsq: True
print(np.round(w_qr, 3).tolist())
# expected output: [-0.476, -11.407, 24.727, 15.429, -37.68, 22.676, 4.806, 8.422, 35.734, 3.217]
lam13 = 10.0
A13 = X13.T @ X13 + lam13 * np.eye(10)                 # symmetric, and positive definite because lam13 > 0
factor13 = cho_factor(A13)                             # the one O(p^3) factorization
w_ridge = cho_solve(factor13, X13.T @ y13)             # each solve is two triangular sweeps, O(p^2)
print("cho == solve:", np.allclose(w_ridge, np.linalg.solve(A13, X13.T @ y13)))  # expected output: cho == solve: True
mse13 = lambda w, target: float(np.mean((X13 @ w - target) ** 2))
rng13 = np.random.default_rng(0)
shuffled = []
for _ in range(100):                                    # the same factor serves 100 new right-hand sides
    y_perm = rng13.permutation(y13)                     # a shuffle breaks any link between the features and the target
    shuffled.append(mse13(cho_solve(factor13, X13.T @ y_perm), y_perm))
real13 = mse13(w_ridge, y13)
print("real MSE:", round(real13, 1), "best shuffled MSE:", round(min(shuffled), 1))
# expected output: real MSE: 2872.2 best shuffled MSE: 5483.6
print("shuffles at least as good:", sum(m <= real13 for m in shuffled))  # expected output: shuffles at least as good: 0
print("variance of y:", round(float(y13.var()), 1))    # expected output: variance of y: 5929.9
# The real fit removes about half of the variance (1 - 2872.2/5929.9 is about 0.52), while even the best of the
# shuffled fits removes under 8 percent of it (1 - 5483.6/5929.9 is about 0.075), so no shuffle comes close.

# %% [markdown]
# ## 2.11 coding drills

# %%
import numpy as np
from scipy import stats

# %% [markdown]
# Linear algebra

# %%
# Linear algebra
a = np.array([1, 2, 3]); b = np.array([4, 5, 6])
print(a @ b)                                   # expected output: 32
print(np.outer(a, b).shape)                    # expected output: (3, 3)
print((a * b).tolist())                        # expected output: [4, 10, 18]
v = np.array([3., -4.])
print(np.linalg.norm(v, 1), np.linalg.norm(v)) # expected output: 7.0 5.0
M = np.array([[1., 0., 2.], [0., 1., 3.], [0., 0., 0.]])
print(np.linalg.matrix_rank(M))                # expected output: 2
print(np.linalg.det(M))                        # expected output: 0.0
# zero, as it must be: M sends all of three-dimensional space into the plane spanned by its columns (2.1.7),
# and a solid flattened onto a plane has volume 0 (2.1.4)
A = np.array([[2., 1.], [1., 2.]])
# A is symmetric, so eigvalsh, NumPy's symmetric-matrix routine of 2.1.8, returns its real eigenvalues [1. 3.]
print(round(float(np.linalg.eigvalsh(A).sum()), 4), round(float(np.trace(A)), 4))  # expected output: 4.0 4.0
print(round(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))), 4))  # expected output: 0.9746

# %% [markdown]
# Calculus

# %%
# Calculus
h = 1e-6
f = lambda x: x**2
print(round((f(3 + h) - f(3 - h)) / (2 * h), 4))         # expected output: 6.0
g = lambda x, y: x**2 + 3*y**2
print(round((g(1+h, 2) - g(1-h, 2)) / (2*h), 4),
      round((g(1, 2+h) - g(1, 2-h)) / (2*h), 4))         # expected output: 2.0 12.0
print(max(0.0, -3), max(0.0, 2.5))                       # expected output: 0.0 2.5
sig = lambda z: 1 / (1 + np.exp(-z))
print(round(float(sig(0)), 3), round(float(sig(2)), 3))  # expected output: 0.5 0.881

# %% [markdown]
# Probability

# %%
# Probability
faces = np.array([1, 2, 3, 4, 5, 6])
print(faces.mean(), round(float(faces.var()), 4))        # expected output: 3.5 2.9167
x1 = np.array([1., 2., 3., 4., 5.]); y1 = np.array([2., 4., 5., 4., 5.])
print(round(float(np.corrcoef(x1, y1)[0, 1]), 4))        # expected output: 0.7746
print(round((0.9*0.01) / (0.9*0.01 + 0.05*0.99), 4))     # expected output: 0.1538
s = np.random.default_rng(0).normal(5, 2, 100000)
print(round(float(s.mean()), 2), round(float(s.std()), 2))  # expected output: 5.0 2.0

# %% [markdown]
# Statistics

# %%
# Statistics
data = np.array([4., 8., 6., 5., 3., 9., 7.])
print(np.median(data), round(float(data.std(ddof=1)), 4))   # expected output: 6.0 2.1602
z = (data - data.mean()) / data.std(ddof=1)     # divisor n - 1, the sample standard deviation of the line above
print(np.round(z, 3).tolist())   # expected output: [-0.926, 0.926, 0.0, -0.463, -1.389, 1.389, 0.463]
z_n = (data - data.mean()) / data.std()         # divisor n, NumPy's default, as in 2.4.1 and StandardScaler
print(np.round(z_n, 3).tolist())  # expected output: [-1.0, 1.0, 0.0, -0.5, -1.5, 1.5, 0.5]
print(round(float(stats.norm.ppf(0.975)), 4))               # expected output: 1.96
tval, pval = stats.ttest_1samp(data, 5.0)
print(round(float(tval), 4), round(float(pval), 4))         # expected output: 1.2247 0.2666

# %% [markdown]
# Information theory

# %%
# Information theory
def entropy(p):
    p = np.array(p); p = p[p > 0]
    return -np.sum(p * np.log2(p))
print(round(entropy([0.5, 0.5]), 4))            # expected output: 1.0
print(round(entropy([0.9, 0.1]), 4))            # expected output: 0.469
print(round(entropy([0.25]*4), 4))              # expected output: 2.0
def kl(p, q):
    p = np.array(p); q = np.array(q)
    return float(np.sum(p * np.log2(p / q)))
print(round(kl([0.5, 0.5], [0.9, 0.1]), 4))     # expected output: 0.737
print(round(-np.sum(np.array([1, 0]) * np.log2([0.8, 0.2])), 4))  # expected output: 0.3219

# %% [markdown]
# Optimization and logic

# %%
# Optimization and logic
def softmax(z):
    z = np.array(z, dtype=float); z -= z.max()
    e = np.exp(z); return e / e.sum()
print(np.round(softmax([1, 2, 3]), 4).tolist())  # expected output: [0.09, 0.2447, 0.6652]
x = 5.0; x -= 0.1 * 2 * x; print(x)              # expected output: 4.0
print([((p, q), (not p) or q) for p in (False, True) for q in (False, True)])
# expected output: [((False, False), True), ((False, True), True), ((True, False), False), ((True, True), True)]
sat = [(A, B, C) for A in (0, 1) for B in (0, 1) for C in (0, 1)
       if (A or B) and ((not A) or C)]
print(len(sat))                                  # expected output: 4
