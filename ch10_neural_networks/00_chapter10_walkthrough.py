"""Chapter 10: Neural Networks from Scratch

Section-aligned walkthrough: every code block from this chapter of the book,
in book order, each in its own cell under its section heading so you can run
and check them one by one. Generated from Book.md; keep the book as the source.
"""

# %% [markdown]
# ## 10.1.1 Hebbian updates, and the perceptron as a gated form

# %% [markdown]
# The plain Hebbian rule keeps growing on one repeated input, while the perceptron's error gate stops it after one correction.

# %%
import numpy as np
x = np.array([1.0, 0.0]); eta = 0.1
w = np.array([1.0, 0.0])                              # plain Hebbian, y = w . x
for _ in range(10):
    w = w + eta * float(w @ x) * x
print("Hebbian norm after 10 steps:", round(float(np.linalg.norm(w)), 4))
w = np.zeros(2); b = 0.0                              # perceptron on the same point, label 1
for _ in range(10):
    pred = int(w @ x + b > 0)
    w = w + eta * (1 - pred) * x; b = b + eta * (1 - pred)
print("perceptron norm, bias:", round(float(np.linalg.norm(w)), 4), b)

# %% [markdown]
# ## 10.2 The perceptron and why XOR breaks it

# %% [markdown]
# The two-input logic gates show it working in Example 10-1.

# %% [markdown]
# Example 10-1. A from-scratch perceptron trained on the AND, OR, and XOR gates.

# %%
import numpy as np
class Perceptron:                                         # one neuron: step(w . x + b)
    def __init__(self, lr=0.1, epochs=50): self.lr, self.epochs = lr, epochs
    def predict(self, X): return (X @ self.w + self.b > 0).astype(int)
    def fit(self, X, y):
        self.w = np.zeros(X.shape[1]); self.b = 0.0; self.converged = None
        for ep in range(self.epochs):
            errors = 0
            for xi, yi in zip(X, y):
                update = self.lr * (yi - int(xi @ self.w + self.b > 0))   # 0 when already correct
                self.w += update * xi; self.b += update; errors += int(update != 0)
            if errors == 0: self.converged = ep; break    # a full clean pass means it converged
        return self
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
gates = {"AND": np.array([0, 0, 0, 1]), "OR": np.array([0, 1, 1, 1]), "XOR": np.array([0, 1, 1, 0])}
for name, y in gates.items():
    p = Perceptron().fit(X, y)
    conv = f"converged at epoch {p.converged}" if p.converged is not None else "NEVER converged"
    print(f"{name:4}: accuracy = {np.mean(p.predict(X) == y):.2f}   ({conv})")
# AND : 1.00 (converged at epoch 5)   OR : 1.00 (converged at epoch 3)   XOR : 0.50 (NEVER converged)

# %% [markdown]
# A short check runs Example 10-1's rule on AND with the bias folded in, counts the corrections, and computes the same three numbers.

# %%
import numpy as np
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]); y_and = np.array([0, 0, 0, 1])
Xa = np.hstack([X, np.ones((4, 1))])                      # append a constant 1: the bias becomes a weight
t = 2 * y_and - 1                                         # the same labels written as -1 and +1
R = np.linalg.norm(Xa, axis=1).max()                      # length of the longest augmented input
u = np.array([2.0, 2.0, -3.0]); u = u / np.linalg.norm(u) # the separator with the widest margin, length 1
gamma = (t * (Xa @ u)).min()                              # its margin: the smallest signed score t * (u . x)
w = np.zeros(3); corrections = 0
for sweep in range(50):                                   # Example 10-1's rule with the bias folded into w
    changed = 0
    for xi, yi in zip(Xa, y_and):
        step = 0.1 * (yi - int(w @ xi > 0))               # +0.1 for a missed positive, -0.1 for a false alarm
        w = w + step * xi; changed += int(step != 0)
    corrections += changed
    if changed == 0: break
print(f"R = {R:.4f}   gamma = {gamma:.4f}   bound (R/gamma)^2 = {(R / gamma) ** 2:.1f}")
print(f"corrections made = {corrections}, final weights and bias = {w.round(2).tolist()}")

# %% [markdown]
# A brute-force scan over a grid of 21 values for each of the three parameters, 9,261 boundaries in all, illustrates the same fact numerically, though it cannot prove it, since it never looks between the grid points.

# %%
best = 0.0                                                # scan every linear boundary on a grid
for w1 in np.linspace(-2, 2, 21):
    for w2 in np.linspace(-2, 2, 21):
        for b in np.linspace(-2, 2, 21):
            pred = ((X @ np.array([w1, w2]) + b) > 0).astype(int)
            best = max(best, np.mean(pred == gates["XOR"]))
print(round(best, 2))   # 0.75  no single straight line ever classifies all four XOR points correctly

# %% [markdown]
# ## 10.2.1 The perceptron on real data

# %% [markdown]
# Example 10-1's perceptron on iris petal length and width: setosa against versicolor is linearly separable, versicolor against virginica is not.

# %%
import numpy as np
from sklearn.datasets import load_iris
class Perceptron:
    def __init__(self, lr=0.1, epochs=50):
        self.lr, self.epochs = lr, epochs
    def fit(self, X, y):
        self.w = np.zeros(X.shape[1]); self.b = 0.0; self.updates = 0; self.converged = None
        for ep in range(self.epochs):
            errors = 0
            for xi, yi in zip(X, y):
                update = self.lr * (yi - int(xi @ self.w + self.b > 0))
                self.w += update * xi; self.b += update
                errors += int(update != 0); self.updates += int(update != 0)
            if errors == 0:
                self.converged = ep; break
        return self
    def score(self, X, y):
        return float(np.mean((X @ self.w + self.b > 0).astype(int) == y))
iris = load_iris(); petal = iris.data[:, 2:4]
sep = iris.target < 2
p = Perceptron().fit(petal[sep], iris.target[sep])
hard = iris.target > 0
p2 = Perceptron(epochs=200).fit(petal[hard], (iris.target[hard] == 2).astype(int))
print("separable", p.score(petal[sep], iris.target[sep]), "epoch", p.converged, "updates", p.updates)
print("not separable", round(p2.score(petal[hard], (iris.target[hard] == 2).astype(int)), 2),
      "epoch", p2.converged, "updates", p2.updates)

# %% [markdown]
# Scikit-learn's `Perceptron` is the same update inside a library estimator, and `SGDClassifier` with the perceptron loss prints the same coefficients.

# %%
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron, SGDClassifier
iris = load_iris(); X = iris.data[iris.target < 2, 2:4]; y = iris.target[iris.target < 2]
sk = Perceptron(eta0=1.0, random_state=0).fit(X, y)
sgd = SGDClassifier(loss="perceptron", learning_rate="constant", eta0=1.0,
                    penalty=None, random_state=0).fit(X, y)
print(np.round(np.r_[sk.coef_.ravel(), sk.intercept_], 6).tolist())
print(np.allclose(sk.coef_, sgd.coef_) and np.allclose(sk.intercept_, sgd.intercept_))

# %% [markdown]
# ## 10.2.2 Adaline: learning before the threshold

# %% [markdown]
# Adaline's delta rule on standardized iris petal features, versicolor against virginica: a learning rate of 1/lambda_max converges and 4/lambda_max diverges.

# %%
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
iris = load_iris()
m = iris.target > 0
X = StandardScaler().fit_transform(iris.data[m, 2:4])
y = (iris.target[m] == 2).astype(float)
Xa = np.hstack([X, np.ones((len(X), 1))])             # bias becomes a weight
lam = np.linalg.eigvalsh(Xa.T @ Xa / len(X)).max()
def run(lr, epochs=20):
    w = np.zeros(Xa.shape[1]); costs = []
    for _ in range(epochs):
        err = y - Xa @ w
        w = w + lr * (Xa.T @ err) / len(X)
        costs.append(float(np.mean(err ** 2)))           # mean squared error, twice the loss L
        if costs[-1] > 1e3:
            break
    return costs
print("lambda_max", round(float(lam), 4))
print("stable", [round(c, 4) for c in run(1 / lam)[::5]])
print("too big", [round(c, 3) for c in run(4 / lam)[:5]])

# %% [markdown]
# ## 10.3.1 Layers and forward propagation

# %% [markdown]
# NumPy confirms the algebra on random inputs.

# %%
import numpy as np
W1 = np.array([[1.0, 2.0], [0.0, -1.0]]); b1 = np.array([1.0, 0.0])   # layer 1: 2 inputs -> 2 units
W2 = np.array([[3.0, 1.0]]);              b2 = np.array([-2.0])       # layer 2: 2 units -> 1 output
W, b = W2 @ W1, W2 @ b1 + b2                                          # the single layer they collapse into
print("collapsed W =", W.tolist(), "  b =", b.tolist())
X = np.random.default_rng(0).normal(size=(5, 2))                       # five random inputs, one per row
two_layers = (X @ W1.T + b1) @ W2.T + b2                               # linear layer, then linear layer
print("two linear layers equal one:", np.allclose(two_layers, X @ W.T + b))
x = np.array([1.0, 1.0]); relu = lambda z: np.maximum(0, z)
print("at x = (1, 1):", W2 @ (W1 @ x + b1) + b2, "  with a ReLU between:", W2 @ relu(W1 @ x + b1) + b2)

# %% [markdown]
# Building a tiny two-hidden-unit network for XOR by hand in Example 10-2 shows forward propagation with a nonlinear activation producing the right answer where one neuron could not.

# %% [markdown]
# Example 10-2. A hand-weighted two-hidden-unit network computing XOR by forward propagation.

# %%
import numpy as np
step = lambda z: (z > 0).astype(float)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
W1 = np.array([[1.0, 1.0], [1.0, 1.0]]); b1 = np.array([-0.5, -1.5])   # unit1 = OR(>0.5), unit2 = AND(>1.5)
W2 = np.array([1.0, -1.0]); b2 = -0.5                                  # output = h_OR - h_AND
H = step(X @ W1.T + b1)                                                # hidden layer activations
out = step(H @ W2 + b2)                                                # output layer
for xi, hi, oi in zip(X, H, out):
    print(f"input {xi.tolist()} -> hidden {hi.tolist()} -> output {int(oi)}")
print("matches XOR [0,1,1,0]:", out.tolist() == [0, 1, 1, 0])
# [0,0]->[0,0]->0   [0,1]->[1,0]->1   [1,0]->[1,0]->1   [1,1]->[1,1]->0   matches XOR: True

# %% [markdown]
# ## 10.3.2 Learning XOR from data

# %% [markdown]
# Setting weights by hand proves an MLP can represent XOR; training shows it can learn the weights on its own, which the perceptron could never do.

# %%
import numpy as np
from sklearn.neural_network import MLPClassifier
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]); y_xor = np.array([0, 1, 1, 0])
clf = MLPClassifier(hidden_layer_sizes=(4,), activation="tanh", max_iter=5000,
                    random_state=1).fit(X, y_xor)
print(round(clf.score(X, y_xor), 2))   # 1.0  a single hidden layer makes XOR learnable from data

# %% [markdown]
# ## 10.3.3 The universal approximation theorem

# %% [markdown]
# Fitting a wiggly target with MLPs of growing width shows the approximating power growing with the number of hidden units.

# %%
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
rng = np.random.default_rng(0)
xt = rng.uniform(-3, 3, 400); yt = np.sin(2 * xt) + 0.3 * xt           # a wiggly continuous target
Xr = xt.reshape(-1, 1)
fits = {}                                                               # keep every fit for a second look
print("hidden units   R^2")
for h in [1, 4, 16, 64]:
    fits[h] = [MLPRegressor(hidden_layer_sizes=(h,), activation="tanh", max_iter=12000,
                            random_state=s).fit(Xr, yt) for s in range(4)]
    best = max(r2_score(yt, m.predict(Xr)) for m in fits[h])
    print(f"{h:>10}   {best:.3f}")
# 1 -> 0.222 (severe underfit)   4 -> 0.819   16 -> 0.803   64 -> 0.875   (see text on the 4-vs-16 dip)

# %% [markdown]
# Two checks separate capacity from optimization: reading how many epochs each stored fit actually used, and refitting the same widths with L-BFGS, run to a far tighter tolerance.

# %%
print("units   Adam epochs used   Adam R^2   L-BFGS R^2")
for h in [1, 4, 16, 64]:
    lbfgs = [MLPRegressor(hidden_layer_sizes=(h,), activation="tanh", solver="lbfgs",
                          max_iter=20000, tol=1e-9, random_state=s).fit(Xr, yt) for s in range(4)]
    used = [m.n_iter_ for m in fits[h]]                   # epochs before Adam's stopping rule fired
    adam_best = max(r2_score(yt, m.predict(Xr)) for m in fits[h])
    lbfgs_best = max(r2_score(yt, m.predict(Xr)) for m in lbfgs)
    print(f"{h:>5}   {min(used):>6} to {max(used):>4}   {adam_best:>8.3f}   {lbfgs_best:>10.3f}")

# %% [markdown]
# ## 10.3.4 Why one hidden layer can approximate: tents of ReLUs

# %% [markdown]
# The piecewise-linear interpolant through the knots of sin(2x) + 0.3x, built as a 1 to n to 1 ReLU network with hand-set weights, and the two-ReLU triangle whose compositions double the number of linear pieces.

# %% [markdown]
# Example 10-3. A one-hidden-layer ReLU network whose hand-set weights interpolate $\sin(2x) + 0.3x$ with 8, 16, and 64 pieces, checked against `np.interp`, and the two-ReLU triangle composed to count its linear pieces.

# %%
import numpy as np
xs = np.linspace(-3, 3, 20001)
target = np.sin(2 * xs) + 0.3 * xs
def relu_net(n_int):                                      # the interpolant as a 1 -> n_int -> 1 ReLU network
    knots = np.linspace(-3, 3, n_int + 1)                 # n_int + 1 knots on the target curve
    vals = np.sin(2 * knots) + 0.3 * knots
    slopes = np.diff(vals) / np.diff(knots)               # slope of each straight piece
    b1 = -knots[:-1]                                      # hidden unit j computes ReLU(x - knot_j): weight 1, bias -knot_j
    w2 = np.r_[slopes[0], np.diff(slopes)]                # output weights: the first slope, then each change of slope
    hidden = np.maximum(0, xs[:, None] + b1)              # the hidden layer on every grid point, one column per unit
    return hidden @ w2 + vals[0]                          # output bias: the curve's value at the left end
def max_err(n_int):
    return float(np.max(np.abs(relu_net(n_int) - target)))
print([round(max_err(n), 4) for n in (8, 16, 64)])
knots = np.linspace(-3, 3, 9)                             # check the 8-unit network against NumPy's interpolant
print(bool(np.max(np.abs(relu_net(8) - np.interp(xs, knots, np.sin(2 * knots) + 0.3 * knots))) < 1e-12))
def tri(x):
    return np.maximum(0, 2 * x) - 2 * np.maximum(0, 2 * x - 1)
def n_pieces(k):
    xs = np.linspace(0, 1, 200001); y = xs.copy()
    for _ in range(k):
        y = tri(y)
    s = np.sign(np.diff(y)); s[s == 0] = 1
    return int(np.sum(s[1:] * s[:-1] < 0) + 1)
print([n_pieces(k) for k in (1, 2, 3, 4)])

# %% [markdown]
# ## 10.3.5 How wide, how deep, and how many parameters

# %% [markdown]
# Four networks, 64 to 32 to k to 10, trained for 20 epochs on the digits split that Example 10-8 will use. Example 10-4 previews the softmax, cross-entropy gradient, backward pass, He initialization, and minibatches of 10.5 to 10.7; its comments name the section that explains each line.

# %% [markdown]
# Example 10-4. A 64 to 32 to $k$ to 10 ReLU network trained from scratch on the digits with a second hidden layer of width 2, 4, 8, or 32, a preview of the training loop that 10.5 to 10.7 derive.

# %%
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
Xd, yd = load_digits(return_X_y=True)
Xd = StandardScaler().fit_transform(Xd)
Y = np.eye(10)[yd]
Xtr, Xte, Ytr, Yte, ytr, yte = train_test_split(Xd, Y, yd, test_size=0.25, random_state=0)
def n_params(k):
    return 64 * 32 + 32 + 32 * k + k + k * 10 + 10
def train(k, epochs=20, lr=0.3):
    rng = np.random.default_rng(0)
    W1 = rng.normal(0, np.sqrt(2 / 64), (64, 32)); b1 = np.zeros(32)   # He initialization, sqrt(2 / n_in) (Example 10-8, 10.7)
    W2 = rng.normal(0, np.sqrt(2 / 32), (32, k)); b2 = np.zeros(k)
    W3 = rng.normal(0, np.sqrt(2 / k), (k, 10)); b3 = np.zeros(10)
    for _ in range(epochs):
        perm = rng.permutation(len(Xtr))                     # shuffled minibatches of 64 rows (Example 10-11, 10.7)
        for s in range(0, len(Xtr), 64):
            idx = perm[s:s + 64]
            xb, yb = Xtr[idx], Ytr[idx]
            z1 = xb @ W1 + b1; a1 = np.maximum(0, z1)       # forward: first hidden layer, ReLU (10.3.1)
            z2 = a1 @ W2 + b2; a2 = np.maximum(0, z2)       # forward: second hidden layer, ReLU
            z3 = a2 @ W3 + b3                               # forward: one raw score per class
            p = np.exp(z3 - z3.max(1, keepdims=True)); p = p / p.sum(1, keepdims=True)   # softmax (10.4.3)
            d3 = (p - yb) / len(xb)                         # error at the output: (p - y) / batch, the cross-entropy gradient (10.5.5, 10.6)
            d2 = (d3 @ W3.T) * (z2 > 0)                     # backward through W3, then the ReLU gate (10.6)
            d1 = (d2 @ W2.T) * (z1 > 0)                     # backward through W2, then the ReLU gate
            W3 -= lr * (a2.T @ d3); b3 -= lr * d3.sum(0)    # gradient descent step on every layer (10.6, 10.7)
            W2 -= lr * (a1.T @ d2); b2 -= lr * d2.sum(0)
            W1 -= lr * (xb.T @ d1); b1 -= lr * d1.sum(0)
    a1 = np.maximum(0, Xte @ W1 + b1)
    a2 = np.maximum(0, a1 @ W2 + b2)
    return float(np.mean((a2 @ W3 + b3).argmax(1) == yte))
for k in (2, 4, 8, 32):
    print(k, n_params(k), round(train(k), 3))

# %% [markdown]
# ## 10.4.1 Sigmoid and tanh, and the vanishing gradient

# %% [markdown]
# Both share a fatal flaw for deep networks, visible in their derivatives.

# %%
import numpy as np
sigmoid = lambda z: 1 / (1 + np.exp(-z))
print(f"sigmoid'(0) = {sigmoid(0) * (1 - sigmoid(0)):.3f}")    # 0.250   the largest a sigmoid gradient ever gets
print(f"sigmoid'(6) = {sigmoid(6) * (1 - sigmoid(6)):.4f}")    # 0.0025  saturated: the gradient nearly vanishes
print(f"tanh'(0) = {1 - np.tanh(0) ** 2:.3f}   tanh'(3) = {1 - np.tanh(3) ** 2:.4f}")   # 1.000 ; 0.0099 (also saturates)
print(f"tanh(2) = {(np.exp(2) - np.exp(-2)) / (np.exp(2) + np.exp(-2)):.3f}   2*sigmoid(4) - 1 = {2 * sigmoid(4) - 1:.3f}")   # 0.964 both ways

# %% [markdown]
# To see the vanishing gradient as a number rather than a claim, take the best case for a sigmoid, where every layer's derivative sits at its maximum of 0.25, and multiply that factor once per layer of depth.

# %%
for d in range(1, 8):
    print(f"depth {d}: 0.25**{d} = {0.25**d:.3g}")

# %% [markdown]
# ## 10.4.2 ReLU and its variants

# %% [markdown]
# Their outputs at a few inputs show the family's shapes, plotted in Figure 10-3.

# %%
import numpy as np
from scipy.stats import norm
sigmoid = lambda z: 1 / (1 + np.exp(-z)); relu = lambda z: np.maximum(0, z)
leaky = lambda z, a=0.01: np.where(z > 0, z, a * z)
elu = lambda z, a=1.0: np.where(z > 0, z, a * (np.exp(z) - 1))   # smooth negative curve
gelu = lambda z: z * norm.cdf(z)              # smooth, used in transformers
swish = lambda z: z * sigmoid(z)              # self-gated, also called SiLU
print(f"{'z':>4} {'sigmoid':>9}{'tanh':>8}{'relu':>7}{'leaky':>8}{'elu':>8}{'gelu':>8}{'swish':>8}")
for z in [-2.0, 0.0, 2.0]:
    print(f"{z:>4} {sigmoid(z):>9.3f}{np.tanh(z):>8.3f}{relu(z):>7.1f}"
          f"{leaky(z):>8.3f}{elu(z):>8.3f}{gelu(z):>8.3f}{swish(z):>8.3f}")
#  z=-2 -> 0.119 -0.964 0.0 -0.020 -0.865 -0.046 -0.238   z=0 -> all 0 except sigmoid 0.5
#  z=2  -> 0.881 0.964 2.0 2.000 2.000 1.954 1.762

# %% [markdown]
# Neither equation has a closed-form root, so a one-dimensional root finder, SciPy's `brentq`, solves each on the interval [-3, 0], where the slope changes sign once.

# %%
from scipy.optimize import brentq
dgelu = lambda z: norm.cdf(z) + z * norm.pdf(z)              # product rule: Phi(z) + z Phi'(z)
dswish = lambda z: sigmoid(z) * (1 + z * (1 - sigmoid(z)))   # product rule, factored: sigma(z)(1 + z(1 - sigma(z)))
for name, f, df in [("GELU", gelu, dgelu), ("Swish", swish, dswish)]:
    z_star = brentq(df, -3.0, 0.0)                           # the slope changes sign once on [-3, 0]
    print(f"{name}: slope is zero at z = {z_star:.5f}, minimum value {f(z_star):.5f}")
print(f"slopes at z = 2: GELU {dgelu(2.0):.3f}, Swish {dswish(2.0):.3f}")

# %% [markdown]
# A brute-force scan over a fine grid is an independent check that needs no calculus.

# %%
zg = np.linspace(-6, 6, 1200001)              # a fine grid over the negative dip
for name, f in [("GELU", gelu), ("Swish", swish)]:
    vals = f(zg); i = int(np.argmin(vals))
    print(name, "minimum", round(float(vals[i]), 4), "at z =", round(float(zg[i]), 4))
# GELU minimum -0.17 at z = -0.7518
# Swish minimum -0.2785 at z = -1.2785

# %% [markdown]
# Drive one unit's preactivation permanently negative and compare the gradient ReLU gives it against the gradient leaky ReLU gives the same unit.

# %%
z = -6.0                                   # a unit whose input is stuck negative
relu_grad  = 1.0 if z > 0 else 0.0
leaky_grad = 1.0 if z > 0 else 0.01
print(f"z = {z}")
print(f"ReLU:  output {max(0.0, z):.2f}, gradient {relu_grad}")
print(f"Leaky: output {(z if z > 0 else 0.01 * z):.2f}, gradient {leaky_grad}")

# %% [markdown]
# ## 10.4.4 Softplus, SELU, Mish, and a cheap GELU

# %% [markdown]
# Softplus, exact and approximate GELU, Mish, and SELU at z = -2, 0, 2; a central-difference check that the derivative of softplus is the sigmoid; the SELU floor; and the grid scan that finds the Mish minimum behind Table 10-1.

# %%
import numpy as np
from scipy.stats import norm
def softplus(z):
    return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0)
def mish(z):
    return z * np.tanh(softplus(z))
def gelu_tanh(z):
    return 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z ** 3)))
def gelu_sig(z):
    return z / (1 + np.exp(-1.702 * z))
alpha, scale = 1.6732632423543773, 1.0507009873554805
def selu(z):
    return scale * np.where(z > 0, z, alpha * (np.exp(z) - 1))
for z in (-2.0, 0.0, 2.0):
    print(z, round(float(softplus(z)), 4), round(float(z * norm.cdf(z)), 4),
          round(float(gelu_tanh(z)), 4), round(float(gelu_sig(z)), 4),
          round(float(mish(z)), 4), round(float(selu(z)), 4))
zs = np.array([-1.5, 0.0, 1.2])
fd = (softplus(zs + 1e-5) - softplus(zs - 1e-5)) / 2e-5
print("softplus' vs sigmoid below 1e-9:", bool(np.max(np.abs(fd - 1 / (1 + np.exp(-zs)))) < 1e-9))
print("SELU floor", round(-scale * alpha, 4))
zg = np.linspace(-6, 6, 1200001)                          # the grid 10.4.2 used for GELU and Swish
vm = mish(zg); i = int(np.argmin(vm))
print("Mish minimum", round(float(vm[i]), 4), "at z =", round(float(zg[i]), 4))

# %% [markdown]
# PReLU on the 64 to 32 to 10 digits network: the backward pass also learns the negative slope alpha, started at 0.25.

# %% [markdown]
# Example 10-5. PReLU on the 64 to 32 to 10 digits network: 15 epochs in which the backward pass also learns the negative slope $\alpha$, started at 0.25.

# %%
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
Xd, yd = load_digits(return_X_y=True)
Xd = StandardScaler().fit_transform(Xd)
Y = np.eye(10)[yd]
Xtr, Xte, Ytr, Yte, ytr, yte = train_test_split(Xd, Y, yd, test_size=0.25, random_state=0)
rng = np.random.default_rng(0)
W1 = rng.normal(0, np.sqrt(2 / 64), (64, 32)); b1 = np.zeros(32)   # He initialization (Example 10-8, 10.7)
W2 = rng.normal(0, np.sqrt(2 / 32), (32, 10)); b2 = np.zeros(10)
alpha = 0.25                                                      # the learned negative slope, started at 0.25
for _ in range(15):
    perm = rng.permutation(len(Xtr))                              # shuffled minibatches of 64 rows (Example 10-11, 10.7)
    for s in range(0, len(Xtr), 64):
        idx = perm[s:s + 64]
        xb, yb = Xtr[idx], Ytr[idx]
        z1 = xb @ W1 + b1                                         # forward: hidden preactivations
        a1 = np.where(z1 > 0, z1, alpha * z1)                     # PReLU: identity above 0, slope alpha below
        z2 = a1 @ W2 + b2                                         # forward: one raw score per class
        p = np.exp(z2 - z2.max(1, keepdims=True)); p = p / p.sum(1, keepdims=True)   # softmax (10.4.3)
        n = len(xb)
        d2 = (p - yb) / n                                         # error at the output: (p - y) / batch (10.5.5, 10.6)
        gate = np.where(z1 > 0, 1.0, alpha)                       # PReLU's slope, the gate the backward pass multiplies by
        da = d2 @ W2.T                                            # backward through W2: the error at the hidden activations (10.6)
        alpha -= 0.05 * float(np.sum(da * z1 * (z1 <= 0))) / n    # the new line: a gradient step on alpha, mean of (upstream) * z over negative z
        W1 -= 0.3 * (xb.T @ (da * gate)); b1 -= 0.3 * (da * gate).sum(0)   # gradient steps on the weights (10.6, 10.7)
        W2 -= 0.3 * (a1.T @ d2); b2 -= 0.3 * d2.sum(0)
print(round(alpha, 4))

# %% [markdown]
# ## 10.4.5 Subgradients at a kink

# %% [markdown]
# A central difference that straddles the ReLU kink, and the subgradient PyTorch 2.15 returns at 0 for `relu` and `abs`.

# %%
import torch
z = 1e-7; h = 1e-5
fd = (max(0.0, z + h) - max(0.0, z - h)) / (2 * h)
r = torch.tensor(0.0, requires_grad=True)
torch.nn.functional.relu(r).backward()
a = torch.tensor(0.0, requires_grad=True)
torch.abs(a).backward()
print(round(fd, 3), float(r.grad), float(a.grad))

# %% [markdown]
# ## 10.4.6 The experiment, and a self-check

# %% [markdown]
# We standardize the pixels, then compare logistic, tanh, and ReLU by 5-fold cross-validation on an identical 20-by-20 hidden architecture. One random initialization can flatter an activation by luck, so each comparison is repeated over five seeds.

# %%
import numpy as np
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
X, y = load_digits(return_X_y=True)                       # 1797 real 8-by-8 digit images, 10 classes
for act in ["logistic", "tanh", "relu"]:
    scores = []
    for seed in range(5):                                 # five initializations, not one lucky draw
        clf = make_pipeline(StandardScaler(),
                            MLPClassifier(hidden_layer_sizes=(20, 20), activation=act,
                                          max_iter=2000, random_state=seed))
        scores.append(cross_val_score(clf, X, y, cv=5).mean())
    per_seed = " ".join(f"{s:.3f}" for s in scores)
    print(f"{act:>8}: per seed {per_seed}   mean {np.mean(scores):.3f}")

# %% [markdown]
# `MLPClassifier` offers only identity, logistic, tanh, and relu; the from-scratch network can use any activation whose slope is known. Example 10-6 trains the 64 to 32 to 10 digit network of Example 10-5 in 10.4.4 seven times, once per hidden activation, with the gate replaced by the activation's own derivative, after checking each derivative formula against a central difference away from the ReLU kink.

# %% [markdown]
# Example 10-6. The 64 to 32 to 10 digits network trained seven times from the same start, once per hidden activation, after each activation's slope formula is checked against a central difference.

# %%
import numpy as np
from scipy.stats import norm
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
sig = lambda z: 1 / (1 + np.exp(-z))
acts = {                                               # (phi, phi'): the slopes of 10.4.1 and 10.4.2
    "sigmoid": (sig, lambda z: sig(z) * (1 - sig(z))),
    "tanh":    (np.tanh, lambda z: 1 - np.tanh(z) ** 2),
    "ReLU":    (lambda z: np.maximum(0, z), lambda z: (z > 0).astype(float)),
    "leaky":   (lambda z: np.where(z > 0, z, 0.01 * z), lambda z: np.where(z > 0, 1.0, 0.01)),
    "ELU":     (lambda z: np.where(z > 0, z, np.exp(np.minimum(z, 0)) - 1),
                lambda z: np.where(z > 0, 1.0, np.exp(np.minimum(z, 0)))),
    "GELU":    (lambda z: z * norm.cdf(z), lambda z: norm.cdf(z) + z * norm.pdf(z)),
    "Swish":   (lambda z: z * sig(z), lambda z: sig(z) * (1 + z * (1 - sig(z)))),
}
Xd, yd = load_digits(return_X_y=True); Xd = StandardScaler().fit_transform(Xd)
Y = np.eye(10)[yd]
Xtr, Xte, Ytr, Yte, ytr, yte = train_test_split(Xd, Y, yd, test_size=0.25, random_state=0)
zs = np.array([-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0])   # probes off the ReLU kink at 0 (10.4.5)
print(f"{'activation':>10} {'slope < 1e-9':>12} {'test acc':>9}")
for name, (phi, dphi) in acts.items():
    fd = (phi(zs + 1e-5) - phi(zs - 1e-5)) / 2e-5          # central difference of the activation itself
    err = float(np.max(np.abs(fd - dphi(zs))))
    rng = np.random.default_rng(0)                          # same He start and same batches for every activation
    W1 = rng.normal(0, np.sqrt(2 / 64), (64, 32)); b1 = np.zeros(32)
    W2 = rng.normal(0, np.sqrt(2 / 32), (32, 10)); b2 = np.zeros(10)
    for _ in range(15):
        perm = rng.permutation(len(Xtr))
        for s in range(0, len(Xtr), 64):
            idx = perm[s:s + 64]; xb, yb = Xtr[idx], Ytr[idx]
            z1 = xb @ W1 + b1; a1 = phi(z1); z2 = a1 @ W2 + b2
            p = np.exp(z2 - z2.max(1, keepdims=True)); p = p / p.sum(1, keepdims=True)
            d2 = (p - yb) / len(xb)                         # the (p - y) seed, mean loss
            d1 = (d2 @ W2.T) * dphi(z1)                     # phi'(z) in place of the ReLU gate
            W1 -= 0.3 * (xb.T @ d1); b1 -= 0.3 * d1.sum(0)
            W2 -= 0.3 * (a1.T @ d2); b2 -= 0.3 * d2.sum(0)
    acc = float(np.mean((phi(Xte @ W1 + b1) @ W2 + b2).argmax(1) == yte))
    print(f"{name:>10} {str(err < 1e-9):>12} {acc:>9.3f}")

# %% [markdown]
# ## 10.5.1 Regression losses

# %% [markdown]
# Injecting one outlier into otherwise small residuals shows the difference.

# %%
import numpy as np
def mse(r): return np.mean(r ** 2)
def mae(r): return np.mean(np.abs(r))
def huber(r, d=1.0):
    a = np.abs(r); return np.mean(np.where(a <= d, 0.5 * r ** 2, d * (a - 0.5 * d)))
resid = np.array([0.5, -0.3, 0.2, -0.4, 0.1])             # well-behaved residuals
resid_out = resid.copy(); resid_out[0] = 8.0              # inject one big outlier
print("        clean   outlier")
print(f"MSE   {mse(resid):>7.3f} {mse(resid_out):>8.3f}")      # 0.110 -> 12.860  (x117: the squared 8 dominates)
print(f"MAE   {mae(resid):>7.3f} {mae(resid_out):>8.3f}")      # 0.300 ->  1.800  (x6: the outlier counts as 8)
print(f"Huber {huber(resid):>7.3f} {huber(resid_out):>8.3f}")  # 0.055 ->  1.530  (x28: the outlier counts as 7.5)

# %% [markdown]
# ## 10.5.2 Cross-entropy for classification

# %% [markdown]
# Its defining behavior is that it punishes confident mistakes severely.

# %%
import numpy as np
def bce(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)            # guard log(0)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))
print(f"p=0.99 -> BCE = {bce(1, 0.99):.3f}")    # 0.010  confident and correct: almost no loss
print(f"p=0.50 -> BCE = {bce(1, 0.50):.3f}")    # 0.693  unsure: this is -log(0.5)
print(f"p=0.01 -> BCE = {bce(1, 0.01):.3f}")    # 4.605  confident and WRONG: severe penalty

# %% [markdown]
# A short check plugs in p* = 0.7.

# %%
import numpy as np
p_star = 0.7                                   # the true probability: the label is 1 on seven cases in ten
for p in [0.5, 0.7, 0.9]:                      # three candidate predicted probabilities
    expected = -(p_star * np.log(p) + (1 - p_star) * np.log(1 - p))
    print(f"predict p = {p}: expected BCE = {expected:.3f}")

# %% [markdown]
# For multiclass problems, categorical cross-entropy generalizes this to -log p_true, the negative log of the probability the model assigned to the correct class.

# %%
import numpy as np
def softmax(z):
    e = np.exp(z - z.max()); return e / e.sum()
probs = softmax(np.array([2.0, 1.0, 0.1]))      # [0.659, 0.242, 0.099]
print(round(float(-np.log(probs[0])), 3))   # 0.417  true class 0 has decent probability -> low loss
print(round(float(-np.log(probs[2])), 3))   # 2.317  true class 2 has low probability -> high loss

# %% [markdown]
# ## 10.5.4 Pairing loss with output activation

# %% [markdown]
# The block evaluates both gradients through a sigmoid at p = 0.01 and y = 1, and their ratio.

# %%
import numpy as np
p, y = 0.01, 1.0
mse_g = 2 * (p - y) * p * (1 - p)          # d/dz of (p - y)^2 through a sigmoid
ce_g = p - y                               # d/dz of binary cross-entropy through a sigmoid
print(round(mse_g, 6), round(ce_g, 2), round(ce_g / mse_g, 2))

# %% [markdown]
# ## 10.5.5 The loss directly from the logits

# %% [markdown]
# Both forms on the scores (2.0, 1.0, 0.1), the overflow case (1000, 0, 0), and a binary logit of 30, against the PyTorch functions.

# %%
import numpy as np
import torch
import torch.nn.functional as F
def cross_entropy_from_logits(Z, y_int):
    Z = np.atleast_2d(np.asarray(Z, dtype=float))
    y_int = np.atleast_1d(np.asarray(y_int, dtype=int))
    m = Z.max(axis=1, keepdims=True)
    lse = (m + np.log(np.exp(Z - m).sum(axis=1, keepdims=True))).ravel()
    return lse - Z[np.arange(len(y_int)), y_int]
def bce_with_logits(z, y):
    z = np.asarray(z, dtype=float); y = np.asarray(y, dtype=float)
    return np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))
z = np.array([2.0, 1.0, 0.1])
print(round(float(cross_entropy_from_logits(z, [0])[0]), 3),
      round(float(F.cross_entropy(torch.as_tensor(z, dtype=torch.float32).reshape(1, -1), torch.tensor([0]))), 3))
print(round(float(cross_entropy_from_logits([1000.0, 0.0, 0.0], [1])[0]), 3))
p = np.exp(np.array([1000.0, 0.0, 0.0]) - 1000.0); p = np.clip(p / p.sum(), 1e-12, 1)
print("clipped -log p", round(float(-np.log(p[1])), 3))
print(round(float(bce_with_logits(30.0, 0.0)), 3),
      round(float(F.binary_cross_entropy_with_logits(torch.tensor(30.0), torch.tensor(0.0))), 3))
pc = np.clip(1 / (1 + np.exp(-30.0)), 1e-12, 1 - 1e-12)
print("clipped BCE", round(float(-np.log(1 - pc)), 3))

# %% [markdown]
# ## 10.5.6 Multilabel heads

# %% [markdown]
# A 103 to 32 ReLU to 14 sigmoid network on the OpenML yeast split of 6.6.2, trained 40 epochs.

# %% [markdown]
# Example 10-7. A 103 to 32 ReLU to 14 sigmoid multilabel network trained from scratch for 40 epochs on the OpenML yeast split of 6.6.2, scored by subset accuracy, Hamming loss, and micro F1.

# %%
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, hamming_loss
yeast = fetch_openml("yeast", version=4, as_frame=False)
X = yeast.data.astype(float)
Y = (yeast.target == "TRUE").astype(float)
Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.3, random_state=0)
sc = StandardScaler().fit(Xtr)
Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
rng = np.random.default_rng(0)
W1 = rng.normal(0, np.sqrt(2 / 103), (103, 32)); b1 = np.zeros(32)
W2 = rng.normal(0, np.sqrt(2 / 32), (32, 14)); b2 = np.zeros(14)
def sig(z):
    return 1 / (1 + np.exp(-np.clip(z, -40, 40)))
for _ in range(40):
    perm = rng.permutation(len(Xtr))
    for k in range(0, len(Xtr), 64):
        idx = perm[k:k + 64]
        xb, yb = Xtr[idx], Ytr[idx]
        z1 = xb @ W1 + b1; a1 = np.maximum(0, z1); z2 = a1 @ W2 + b2
        p = sig(z2); n = len(xb)
        dz2 = (p - yb) / n
        dz1 = (dz2 @ W2.T) * (z1 > 0)
        W1 -= 0.2 * (xb.T @ dz1); b1 -= 0.2 * dz1.sum(0)
        W2 -= 0.2 * (a1.T @ dz2); b2 -= 0.2 * dz2.sum(0)
pred = (sig(np.maximum(0, Xte @ W1 + b1) @ W2 + b2) >= 0.5).astype(int)
print("subset", round(float(np.mean(np.all(pred == Yte, axis=1))), 3))
print("Hamming", round(float(hamming_loss(Yte, pred)), 3))
print("micro-F1", round(float(f1_score(Yte, pred, average="micro")), 3))

# %% [markdown]
# The initial weights of Example 10-7, before any epoch, pass a central-difference check on one entry of W1.

# %%
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
yeast = fetch_openml("yeast", version=4, as_frame=False)
X = yeast.data.astype(float); Y = (yeast.target == "TRUE").astype(float)
Xtr, _, Ytr, _ = train_test_split(X, Y, test_size=0.3, random_state=0)
Xtr = StandardScaler().fit(Xtr).transform(Xtr)
rng = np.random.default_rng(0)
W1 = rng.normal(0, np.sqrt(2 / 103), (103, 32)); b1 = np.zeros(32)
W2 = rng.normal(0, np.sqrt(2 / 32), (32, 14)); b2 = np.zeros(14)
xb, yb = Xtr[:16], Ytr[:16]
z1 = xb @ W1 + b1; a1 = np.maximum(0, z1)
def sig(z):
    return 1 / (1 + np.exp(-np.clip(z, -40, 40)))
p = sig(a1 @ W2 + b2); n = len(xb)
ana = (xb.T @ (((p - yb) / n) @ W2.T * (z1 > 0)))[3, 4]
o = W1[3, 4]
def loss(v):
    W = W1.copy(); W[3, 4] = v
    aa = np.maximum(0, xb @ W + b1); logits = aa @ W2 + b2
    per = np.maximum(logits, 0) - logits * yb + np.log1p(np.exp(-np.abs(logits)))
    return float(per.sum(1).mean())
num = (loss(o + 1e-5) - loss(o - 1e-5)) / 2e-5
print("relative error below 1e-6:", bool(abs(num - ana) / (abs(num) + abs(ana) + 1e-12) < 1e-6))

# %% [markdown]
# ## 10.6.1 A forward and backward pass by hand

# %% [markdown]
# A short block confirms the whole computation and checks one entry with a central finite difference.

# %%
import numpy as np
x = np.array([1.0, 0.5]); y = 1.0                     # one example, positive label
W1 = np.array([[0.5, -0.2], [0.3, 0.4]]); b1 = np.array([0.1, -0.6])
w2 = np.array([0.8, -0.5]); b2 = -0.2
def fwd(W1, y):                                       # loss is valid for either label
    z1 = W1 @ x + b1; a1 = np.maximum(0, z1)          # ReLU hidden layer
    p = 1 / (1 + np.exp(-(w2 @ a1 + b2)))             # sigmoid output
    loss = -(y * np.log(p) + (1 - y) * np.log(1 - p)) # binary cross-entropy
    return z1, a1, p, loss
z1, a1, p, L = fwd(W1, y)
d2 = p - y                                            # output error (p - y)
d1 = (w2 * d2) * (z1 > 0)                             # backprop through the ReLU gate
print(round(float(p), 4), round(float(L), 4), round(float(d2), 4))   # 0.5498 0.5981 -0.4502
print((d2 * a1).round(4), d1.round(4))                # [-0.2251 -0.] [-0.3601  0.]
print(np.outer(d1, x).round(4))                       # [[-0.3601 -0.1801] [ 0.  0.]]
Wp = W1.copy(); Wp[0, 0] += 1e-6; Wm = W1.copy(); Wm[0, 0] -= 1e-6
print(round(float((fwd(Wp, y)[3] - fwd(Wm, y)[3]) / 2e-6), 4))        # -0.3601

# %% [markdown]
# ## 10.6.2 Gradient checking on a real network

# %% [markdown]
# This is the very class that 10.7 will train on the full digits data; here, only its gradients are on trial.

# %% [markdown]
# Example 10-8. The from-scratch MLP class (64 to 32 ReLU to 10 softmax), with sampled entries of every gradient checked by central differences.

# %%
import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
rng = np.random.default_rng(0)
relu = lambda z: np.maximum(0, z)
def softmax(Z):
    e = np.exp(Z - Z.max(axis=1, keepdims=True)); return e / e.sum(axis=1, keepdims=True)
class MLP:                                                # 64 -> 32 (ReLU) -> 10 (softmax)
    def __init__(self, n_in, n_hidden, n_out, weights=None):
        if weights is None:                               # He init when no weights are supplied
            self.W1 = rng.normal(0, np.sqrt(2 / n_in), (n_in, n_hidden))
            self.W2 = rng.normal(0, np.sqrt(2 / n_hidden), (n_hidden, n_out))
        else:
            self.W1, self.W2 = weights
        self.b1 = np.zeros(n_hidden); self.b2 = np.zeros(n_out)
    @classmethod
    def from_weights(cls, W1, W2):                        # alternative constructor, as in 1.7.4
        return cls(W1.shape[0], W1.shape[1], W2.shape[1], (W1.copy(), W2.copy()))
    def forward(self, X):
        self.X = X; self.z1 = X @ self.W1 + self.b1; self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2; self.p = softmax(self.z2); return self.p
    def loss(self, X, Y): return -np.mean(np.sum(Y * np.log(self.forward(X) + 1e-12), axis=1))
    def backward(self, Y):
        n = len(self.X); dz2 = (self.p - Y) / n           # mean loss: (p - y) / n
        dW2 = self.a1.T @ dz2; db2 = dz2.sum(0)
        dz1 = (dz2 @ self.W2.T) * (self.z1 > 0)            # backprop through the ReLU
        return self.X.T @ dz1, dz1.sum(0), dW2, db2
    def step(self, grads, lr):
        dW1, db1, dW2, db2 = grads
        self.W1 = self.W1 - lr * dW1; self.b1 = self.b1 - lr * db1
        self.W2 = self.W2 - lr * dW2; self.b2 = self.b2 - lr * db2
Xd, yd = load_digits(return_X_y=True); Xd = StandardScaler().fit_transform(Xd)
Y = np.eye(10)[yd]
Xtr, Xte, Ytr, Yte, ytr, yte = train_test_split(Xd, Y, yd, test_size=0.25, random_state=0)
net = MLP(64, 32, 10); Xb, Yb = Xtr[:32], Ytr[:32]
net.forward(Xb); analytic = net.backward(Yb)              # all four gradients
pairs = (
    (net.W1, analytic[0], [(5, 10), (20, 3), (40, 25), (60, 7)]),
    (net.b1, analytic[1], [(0,), (7,), (15,), (31,)]),
    (net.W2, analytic[2], [(0, 0), (8, 3), (16, 9), (31, 5)]),
    (net.b2, analytic[3], [(0,), (3,), (9,)]))
errs = []
for arr, ana, idxs in pairs:                              # sampled entries of every array
    for idx in idxs:
        o = arr[idx]
        arr[idx] = o + 1e-5; lp = net.loss(Xb, Yb)
        arr[idx] = o - 1e-5; lm = net.loss(Xb, Yb); arr[idx] = o
        num = (lp - lm) / 2e-5
        errs.append(abs(num - ana[idx]) / (abs(num) + abs(ana[idx]) + 1e-12))
print("max relative error below 1e-6:", bool(max(errs) < 1e-6))   # over all 15 sampled entries

# %% [markdown]
# ## 10.6.3 Automatic differentiation in a few lines

# %% [markdown]
# A reverse-mode scalar autograd class recomputes the 2-2-1 loss and the partial with respect to W1[0, 0].

# %% [markdown]
# Example 10-9. A scalar reverse-mode automatic differentiation class, `Value`, that records the 2-2-1 loss of 10.6.1 as a graph and reads off its partial derivative with respect to $W^{(1)}_{11}$.

# %%
import math
class Value:
    def __init__(self, data, prev=()):
        self.data = float(data); self.grad = 0.0; self._prev = prev; self._backward = lambda: None
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))
        def back():
            self.grad += out.grad; other.grad += out.grad
        out._backward = back; return out
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))
        def back():
            self.grad += other.data * out.grad; other.grad += self.data * out.grad
        out._backward = back; return out
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,))
        def back():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = back; return out
    def sigmoid(self):
        s = 1 / (1 + math.exp(-self.data)); out = Value(s, (self,))
        def back():
            self.grad += s * (1 - s) * out.grad
        out._backward = back; return out
    def log(self):
        out = Value(math.log(self.data), (self,))
        def back():
            self.grad += (1 / self.data) * out.grad
        out._backward = back; return out
    def backward(self):
        topo, seen = [], set()
        def build(v):
            if id(v) not in seen:
                seen.add(id(v))
                for c in v._prev: build(c)
                topo.append(v)
        build(self); self.grad = 1.0
        for v in reversed(topo): v._backward()
W00 = Value(0.5)
z1 = (W00 * Value(1.0) + Value(-0.2) * Value(0.5) + Value(0.1)).relu()
z2 = Value(0.8) * z1 + Value(-0.2)          # dead unit contributes 0
loss = z2.sigmoid().log() * Value(-1.0)
loss.backward()
print(round(loss.data, 4), round(W00.grad, 4))

# %% [markdown]
# Dual numbers recover the same partial in forward mode, one extra sweep.

# %% [markdown]
# Example 10-10. A dual-number class, `Dual`, that recovers the same partial derivative in forward mode with one extra sweep.

# %%
import math
class Dual:
    def __init__(self, r, e=0.0):
        self.r = float(r); self.e = float(e)
    def __add__(self, o):
        o = o if isinstance(o, Dual) else Dual(o); return Dual(self.r + o.r, self.e + o.e)
    def __radd__(self, o): return self + o
    def __mul__(self, o):
        o = o if isinstance(o, Dual) else Dual(o)
        return Dual(self.r * o.r, self.r * o.e + self.e * o.r)
    def __rmul__(self, o): return self * o
    def relu(self):
        return Dual(self.r if self.r > 0 else 0.0, self.e if self.r > 0 else 0.0)
    def sigmoid(self):
        s = 1 / (1 + math.exp(-self.r)); return Dual(s, s * (1 - s) * self.e)
    def log(self):
        return Dual(math.log(self.r), self.e / self.r)
w = Dual(0.5, 1.0)
z1 = (w * Dual(1.0) + Dual(-0.2) * Dual(0.5) + Dual(0.1)).relu()
loss = (Dual(0.8) * z1 + Dual(-0.2)).sigmoid().log() * Dual(-1.0)
print(round(loss.e, 4))

# %% [markdown]
# ## 10.7 Build a neural network in pure NumPy

# %% [markdown]
# Training for sixty epochs shows the network learning.

# %% [markdown]
# Example 10-11. Sixty epochs of minibatch training for the from-scratch digit MLP.

# %%
net = MLP(64, 32, 10)                                     # a fresh network, same architecture
lr, epochs, bs = 0.5, 60, 64
for ep in range(epochs):
    perm = rng.permutation(len(Xtr))                      # shuffle each epoch
    for k in range(0, len(Xtr), bs):
        idx = perm[k:k + bs]
        net.forward(Xtr[idx]); net.step(net.backward(Ytr[idx]), lr)   # one minibatch SGD step
    if ep in (0, 9, 29, 59):
        tr = (net.forward(Xtr).argmax(1) == ytr).mean()
        te = (net.forward(Xte).argmax(1) == yte).mean()
        print(f"epoch {ep+1:>2}: train acc {tr:.3f}, test acc {te:.3f}")
# epoch 1: 0.921/0.909   epoch 10: 1.000/0.973   epoch 30: 1.000/0.978   epoch 60: 1.000/0.980

# %% [markdown]
# Recording the cross-entropy on the full training set and on the held-out test set after every epoch traces the learning curve, which, in the author's view, is the clearest single picture of training health and the first thing to plot for any model. Example 10-12 records both after each of the same sixty epochs.

# %% [markdown]
# Example 10-12. The learning curve of the digit MLP: full-set training and test cross-entropy after each of sixty epochs, and the epoch of the lowest test loss.

# %%
# a fresh network; continues the session of Examples 10-8 and 10-11 (rng, lr, epochs, bs)
net = MLP(64, 32, 10)
hist = {"train": [], "test": []}
for ep in range(epochs):
    perm = rng.permutation(len(Xtr))
    for k in range(0, len(Xtr), bs):
        idx = perm[k:k + bs]
        net.forward(Xtr[idx]); net.step(net.backward(Ytr[idx]), lr)
    hist["train"].append(net.loss(Xtr, Ytr))        # full-set cross-entropy after this epoch
    hist["test"].append(net.loss(Xte, Yte))
for ep in (0, 1, 2, 4, 9, 29, 59):
    print(f"epoch {ep+1:>2}: train loss {hist['train'][ep]:.3f}   test loss {hist['test'][ep]:.3f}")
best = int(np.argmin(hist["test"]))
print(f"lowest test loss {hist['test'][best]:.3f} at epoch {best+1}")

# %% [markdown]
# Training the same digit network from four starting points, zero, small random, Xavier, and He, shows what matters at this shallow depth.

# %% [markdown]
# Example 10-13. Four initializations of the digit MLP.

# %%
def train_init(W1, W2, perm_seed, n_epochs=30):
    m = MLP.from_weights(W1, W2)                      # set weights; do not draw a new He init
    r = np.random.default_rng(perm_seed)             # its own shuffling stream, fixed across schemes
    for ep in range(n_epochs):
        perm = r.permutation(len(Xtr))
        for k in range(0, len(Xtr), bs):
            idx = perm[k:k + bs]
            m.forward(Xtr[idx]); m.step(m.backward(Ytr[idx]), lr)
    return m, (m.forward(Xte).argmax(1) == yte).mean()

n_in, n_h, n_out = 64, 32, 10
r0 = np.random.default_rng(1)                         # draws each scheme's starting weights
schemes = {
    "zeros":  (np.zeros((n_in, n_h)),                             np.zeros((n_h, n_out))),
    "small":  (r0.normal(0, 0.01, (n_in, n_h)),                   r0.normal(0, 0.01, (n_h, n_out))),
    "Xavier": (r0.normal(0, np.sqrt(2/(n_in+n_h)), (n_in, n_h)),  r0.normal(0, np.sqrt(2/(n_h+n_out)), (n_h, n_out))),
    "He":     (r0.normal(0, np.sqrt(2/n_in), (n_in, n_h)),        r0.normal(0, np.sqrt(2/n_h), (n_h, n_out))),
}
for name, (W1, W2) in schemes.items():
    m, acc = train_init(W1, W2, perm_seed=2)
    note = f"   hidden units identical: {np.allclose(m.W1, m.W1[:, :1])}" if name == "zeros" else ""
    print(f"{name:>6}: test acc {acc:.3f}{note}")

# %%
print("weight_seed shuffle_seed   small Xavier He")
for ws, ps in [(3, 4), (5, 6), (7, 8), (9, 10), (11, 12)]:
    rr = np.random.default_rng(ws)
    accs = []
    for kind in ("small", "Xavier", "He"):
        if kind == "small":
            W1 = rr.normal(0, 0.01, (n_in, n_h)); W2 = rr.normal(0, 0.01, (n_h, n_out))
        elif kind == "Xavier":
            W1 = rr.normal(0, np.sqrt(2 / (n_in + n_h)), (n_in, n_h))
            W2 = rr.normal(0, np.sqrt(2 / (n_h + n_out)), (n_h, n_out))
        else:
            W1 = rr.normal(0, np.sqrt(2 / n_in), (n_in, n_h))
            W2 = rr.normal(0, np.sqrt(2 / n_h), (n_h, n_out))
        _, acc = train_init(W1, W2, ps); accs.append(acc)
    print(f"{ws:>11} {ps:>12}   " + " ".join(f"{a:.3f}" for a in accs))

# %% [markdown]
# ## 10.7.1 A regression network on California housing

# %% [markdown]
# The clipped California housing split of the 5.7 lab: an 8 to 32 ReLU to 1 linear network, then MLPRegressor on the same arrays.

# %% [markdown]
# Example 10-14. An 8 to 32 ReLU to 1 linear regression network trained from scratch on the clipped California housing split of the 5.7 lab, beside scikit-learn's `MLPRegressor` on the same arrays.

# %%
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
df = fetch_california_housing(as_frame=True).frame
feats = fetch_california_housing().feature_names
Xtr, Xte, ytr, yte = train_test_split(df[feats], df["MedHouseVal"], test_size=0.2, random_state=0)
caps = {"AveRooms": 12.0, "AveBedrms": 3.0, "Population": 6000.0, "AveOccup": 6.0}
for c, h in caps.items():
    Xtr[c] = Xtr[c].clip(upper=h); Xte[c] = Xte[c].clip(upper=h)
mu, sd = Xtr.mean(), Xtr.std()
Xtr_s = ((Xtr - mu) / sd).to_numpy(); Xte_s = ((Xte - mu) / sd).to_numpy()
ym, ys = float(ytr.mean()), float(ytr.std())
ytr_s = ((ytr.to_numpy() - ym) / ys).reshape(-1, 1)
rng = np.random.default_rng(0)
W1 = rng.normal(0, np.sqrt(2 / 8), (8, 32)); b1 = np.zeros(32)
W2 = rng.normal(0, np.sqrt(2 / 32), (32, 1)); b2 = np.zeros(1)
for _ in range(40):
    perm = rng.permutation(len(Xtr_s))
    for k in range(0, len(Xtr_s), 64):
        idx = perm[k:k + 64]
        xb, yb = Xtr_s[idx], ytr_s[idx]
        z1 = xb @ W1 + b1; a1 = np.maximum(0, z1); pred = a1 @ W2 + b2
        dpred = (pred - yb) / len(xb)
        dz1 = (dpred @ W2.T) * (z1 > 0)
        W2 -= 0.05 * (a1.T @ dpred); b2 -= 0.05 * dpred.sum(0)
        W1 -= 0.05 * (xb.T @ dz1); b1 -= 0.05 * dz1.sum(0)
def rmse(X, y):
    pred = (np.maximum(0, X @ W1 + b1) @ W2 + b2).ravel() * ys + ym
    return float(np.sqrt(np.mean((pred - y) ** 2)))
print(round(rmse(Xtr_s, ytr.to_numpy()), 3), round(rmse(Xte_s, yte.to_numpy()), 3))
reg = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(32,), max_iter=200, random_state=0))
reg.fit(Xtr.to_numpy(), ytr.to_numpy())
pred = reg.predict(Xte.to_numpy())
print(round(float(np.sqrt(np.mean((pred - yte.to_numpy()) ** 2))), 3))

# %% [markdown]
# The heads named at the start of 10.7.1, and the Huber loss, trained on the same session: the softplus, exponential, and scaled sigmoid heads, each with its slope in the backward pass, and the linear head with the Huber loss at delta = 1.

# %% [markdown]
# Example 10-15. The California housing network trained with linear, softplus, exponential, and scaled sigmoid output heads, and once more with the Huber loss, reporting RMSE, MAE, and the predictions outside the target's range.

# %%
# continues the session of the standardized run: Xtr_s, Xte_s, ytr, yte, ym, ys, and its trained W1, b1, W2, b2
lin_pred = (np.maximum(0, Xte_s @ W1 + b1) @ W2 + b2).ravel() * ys + ym
print("training districts", len(ytr), " target range", round(float(ytr.min()), 5), "to",
      round(float(ytr.max()), 5), " at 5 or more:", int((ytr >= 5).sum()))
print("linear head, test predictions below 0:", int((lin_pred < 0).sum()), " above 5:", int((lin_pred > 5).sum()))
sig = lambda z: 1 / (1 + np.exp(-z))
softplus = lambda z: np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0)
heads = {                                   # output unit g(z) and its slope g'(z)
    "linear":   (lambda z: z, lambda z: np.ones_like(z)),
    "softplus": (softplus, sig),
    "exp":      (np.exp, np.exp),
    "sigmoid":  (lambda z: 5 * sig(z), lambda z: 5 * sig(z) * (1 - sig(z))),   # 5 is the cap of the target
}
ytr_raw = ytr.to_numpy().reshape(-1, 1)     # the target in its own units, not centered
def train_head(head, loss="mse", delta=1.0, epochs=40, lr=0.05):
    g, dg = heads[head]
    r = np.random.default_rng(0)
    W1 = r.normal(0, np.sqrt(2 / 8), (8, 32)); b1 = np.zeros(32)
    W2 = r.normal(0, np.sqrt(2 / 32), (32, 1)); b2 = np.zeros(1)
    for _ in range(epochs):
        perm = r.permutation(len(Xtr_s))
        for k in range(0, len(Xtr_s), 64):
            idx = perm[k:k + 64]; xb, yb = Xtr_s[idx], ytr_raw[idx]
            z1 = xb @ W1 + b1; a1 = np.maximum(0, z1); z2 = a1 @ W2 + b2; pred = g(z2)
            e = pred - yb                                   # residual, prediction minus target
            if loss == "huber":
                e = np.clip(e, -delta, delta)               # Huber's gradient saturates at +-delta
            dz2 = e * dg(z2) / len(xb)                      # (pred - y) times the head's slope
            dz1 = (dz2 @ W2.T) * (z1 > 0)
            W2 -= lr * (a1.T @ dz2); b2 -= lr * dz2.sum(0)
            W1 -= lr * (xb.T @ dz1); b1 -= lr * dz1.sum(0)
    out = lambda X: g(np.maximum(0, X @ W1 + b1) @ W2 + b2).ravel()
    p = out(Xte_s); err = p - yte.to_numpy()
    tr = float(np.sqrt(np.mean((out(Xtr_s) - ytr.to_numpy()) ** 2)))
    return tr, float(np.sqrt(np.mean(err ** 2))), float(np.mean(np.abs(err))), int((p < 0).sum()), int((p > 5).sum())
print(f"{'head':>8} {'loss':>6} {'train RMSE':>10} {'test RMSE':>9} {'test MAE':>8}  below 0  above 5")
for head in heads:
    tr, te, mae, lo, hi = train_head(head)
    print(f"{head:>8} {'mse':>6} {tr:>10.3f} {te:>9.3f} {mae:>8.3f} {lo:>8} {hi:>8}")
tr, te, mae, lo, hi = train_head("linear", loss="huber")
print(f"{'linear':>8} {'huber':>6} {tr:>10.3f} {te:>9.3f} {mae:>8.3f} {lo:>8} {hi:>8}")

# %% [markdown]
# ## 10.7.2 Why the loss is not convex

# %% [markdown]
# A trained digits network, its hidden-unit permutation, and their midpoint: two equal losses and a far worse midpoint.

# %%
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
Xd, yd = load_digits(return_X_y=True)
Xd = StandardScaler().fit_transform(Xd)
Y = np.eye(10)[yd]
Xtr, Xte, Ytr, Yte, ytr, yte = train_test_split(Xd, Y, yd, test_size=0.25, random_state=0)
rng = np.random.default_rng(0)
W1 = rng.normal(0, np.sqrt(2 / 64), (64, 32)); b1 = np.zeros(32)
W2 = rng.normal(0, np.sqrt(2 / 32), (32, 10)); b2 = np.zeros(10)
for _ in range(30):
    perm = rng.permutation(len(Xtr))
    for k in range(0, len(Xtr), 64):
        idx = perm[k:k + 64]
        xb, yb = Xtr[idx], Ytr[idx]
        z1 = xb @ W1 + b1; a1 = np.maximum(0, z1); z2 = a1 @ W2 + b2
        p = np.exp(z2 - z2.max(1, keepdims=True)); p = p / p.sum(1, keepdims=True)
        dz2 = (p - yb) / len(xb); dz1 = (dz2 @ W2.T) * (z1 > 0)
        W1 -= 0.5 * (xb.T @ dz1); b1 -= 0.5 * dz1.sum(0)
        W2 -= 0.5 * (a1.T @ dz2); b2 -= 0.5 * dz2.sum(0)
def ce(W1, b1, W2, b2):
    a1 = np.maximum(0, Xtr @ W1 + b1); z2 = a1 @ W2 + b2
    p = np.exp(z2 - z2.max(1, keepdims=True)); p = p / p.sum(1, keepdims=True)
    return float(-np.mean(np.sum(Ytr * np.log(p + 1e-12), axis=1)))
order = rng.permutation(32)
print(round(ce(W1, b1, W2, b2), 4),
      round(ce(W1[:, order], b1[order], W2[order, :], b2), 4),
      round(ce(0.5 * (W1 + W1[:, order]), 0.5 * (b1 + b1[order]),
               0.5 * (W2 + W2[order, :]), b2), 4))

# %% [markdown]
# ## 10.7.3 Layers with a forward and a backward

# %% [markdown]
# Dense, ReLU, and Seq layers with forward and backward methods, checked on one entry of the first weight matrix.

# %% [markdown]
# Example 10-16. `Dense`, `ReLU`, and `Seq` layer classes with forward and backward methods, assembled into a 64 to 32 to 10 network and checked on one entry of the first weight matrix by a central difference.

# %%
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
Xd, yd = load_digits(return_X_y=True)
Xd = StandardScaler().fit_transform(Xd)
Y = np.eye(10)[yd]
Xtr, Xte, Ytr, Yte, ytr, yte = train_test_split(Xd, Y, yd, test_size=0.25, random_state=0)
rng = np.random.default_rng(0)
class Dense:
    def __init__(self, n_in, n_out):
        self.W = rng.normal(0, np.sqrt(2 / n_in), (n_in, n_out)); self.b = np.zeros(n_out)
    def forward(self, X):
        self.X = X; return X @ self.W + self.b
    def backward(self, dY):
        self.dW = self.X.T @ dY; self.db = dY.sum(0); return dY @ self.W.T
class ReLU:
    def forward(self, Z):
        self.Z = Z; return np.maximum(0, Z)
    def backward(self, dA):
        return dA * (self.Z > 0)
class Seq:
    def __init__(self, layers):
        self.layers = layers
    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X
    def backward(self, d):
        for layer in reversed(self.layers):
            d = layer.backward(d)
h1, h2 = Dense(64, 32), Dense(32, 10)
net = Seq([h1, ReLU(), h2])
xb, yb = Xtr[:32], Ytr[:32]
logits = net.forward(xb)
p = np.exp(logits - logits.max(1, keepdims=True)); p = p / p.sum(1, keepdims=True)
net.backward((p - yb) / len(xb))
o = h1.W[5, 10]
def loss(val):
    h1.W[5, 10] = val
    z = net.forward(xb)
    pp = np.exp(z - z.max(1, keepdims=True)); pp = pp / pp.sum(1, keepdims=True)
    return float(-np.mean(np.sum(yb * np.log(pp + 1e-12), 1)))
num = (loss(o + 1e-5) - loss(o - 1e-5)) / 2e-5
h1.W[5, 10] = o
print("relative error below 1e-6:", bool(abs(num - h1.dW[5, 10]) / (abs(num) + abs(h1.dW[5, 10]) + 1e-12) < 1e-6))

# %% [markdown]
# A 64 to 32 to 32 to 10 network trained 15 epochs at learning rate 0.3, with its own seed.

# %% [markdown]
# Example 10-17. The layer classes with a step method, training a 64 to 32 to 32 to 10 network for 15 epochs at learning rate 0.3 from its own seed.

# %%
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
Xd, yd = load_digits(return_X_y=True)
Xd = StandardScaler().fit_transform(Xd)
Y = np.eye(10)[yd]
Xtr, Xte, Ytr, Yte, ytr, yte = train_test_split(Xd, Y, yd, test_size=0.25, random_state=0)
rng = np.random.default_rng(1)
class Dense:
    def __init__(self, n_in, n_out):
        self.W = rng.normal(0, np.sqrt(2 / n_in), (n_in, n_out)); self.b = np.zeros(n_out)
    def forward(self, X):
        self.X = X; return X @ self.W + self.b
    def backward(self, dY):
        self.dW = self.X.T @ dY; self.db = dY.sum(0); return dY @ self.W.T
    def step(self, lr):
        self.W -= lr * self.dW; self.b -= lr * self.db
class ReLU:
    def forward(self, Z):
        self.Z = Z; return np.maximum(0, Z)
    def backward(self, dA):
        return dA * (self.Z > 0)
class Seq:
    def __init__(self, layers):
        self.layers = layers
    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X
    def backward(self, d):
        for layer in reversed(self.layers):
            d = layer.backward(d)
layers = [Dense(64, 32), ReLU(), Dense(32, 32), ReLU(), Dense(32, 10)]
net = Seq(layers)
for _ in range(15):
    perm = rng.permutation(len(Xtr))
    for k in range(0, len(Xtr), 64):
        idx = perm[k:k + 64]
        xb, yb = Xtr[idx], Ytr[idx]
        z = net.forward(xb)
        p = np.exp(z - z.max(1, keepdims=True)); p = p / p.sum(1, keepdims=True)
        net.backward((p - yb) / len(xb))
        for layer in layers:
            if hasattr(layer, "step"):
                layer.step(0.3)
print(round(float(np.mean(net.forward(Xte).argmax(1) == yte)), 3))

# %% [markdown]
# Ten sigmoid layers of width 16: the error-signal norms shrink about 4,150 times from the last layer to the first.

# %%
import numpy as np
rng = np.random.default_rng(0)
Ws = [rng.normal(0, 0.5, (16, 16)) for _ in range(10)]
a = rng.normal(size=(32, 16)); acts = []
for W in Ws:
    z = a @ W
    a = 1 / (1 + np.exp(-z)); acts.append(a)
delta = np.ones_like(a); norms = []
for i in reversed(range(10)):
    delta = delta * acts[i] * (1 - acts[i])
    norms.append(float(np.linalg.norm(delta)))
    delta = delta @ Ws[i].T
print([round(v, 6) for v in reversed(norms)])

# %%
import numpy as np
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
def fit(X, y):
    ...
print(X.shape, fit(X, None))

# %%
import numpy as np
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
def hidden(X):
    ...
print(X.shape, hidden(X))

# %%
def dsigmoid(z):
    ...
print(dsigmoid(0))

# %%
import numpy as np
def softmax_T(z, T=1.0):
    ...
z = np.array([2.0, 1.0, 0.1])
print(z.shape, softmax_T(z))

# %%
import numpy as np
def huber(r, d):
    ...
r = np.array([8.0, -0.3, 0.2, -0.4, 0.1])
print(r.shape, huber(r, 1.0))

# %%
from sklearn.datasets import load_wine
X, y = load_wine(return_X_y=True)
def check_grad():
    ...
print(X.shape, y.shape, check_grad())

# %%
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
X, y = load_wine(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
def train():
    ...
print(Xtr.shape, Xte.shape, train())

# %%
from sklearn.datasets import load_wine
X, y = load_wine(return_X_y=True)
def train_init(W1, W2):
    ...
print(X.shape, train_init(None, None))

# %%
from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(return_X_y=True)
def compare():
    ...
print(X.shape, y.shape, compare())

# %%
import numpy as np
rng = np.random.default_rng(0)
x = rng.uniform(-2, 2, 400)
def best_r2(width):
    ...
print(x.shape, best_r2(1))

# %%
import numpy as np
x = np.array([1.0, -1.0])
W1 = np.array([[0.2, -0.1], [0.4, 0.3]])
def backward():
    ...
print(x.shape, W1.shape, backward())

# %%
from sklearn.datasets import load_digits
X, y = load_digits(return_X_y=True)
def dead_run(lr):
    ...
print(X.shape, y.shape, dead_run(0.5))

