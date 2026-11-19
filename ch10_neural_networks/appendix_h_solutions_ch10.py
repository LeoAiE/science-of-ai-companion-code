"""Appendix H solutions for Chapter 10, as runnable cells.

Generated from Solutions.md (Appendix H). Every code block of this chapter's solutions appears here
verbatim, split into one cell per exercise or drill topic, so each answer can be run and checked against
the expected outputs it states. Keep Solutions.md as the source; regenerate after any change.
"""

# %% [markdown]
# ## 10.8 coding exercises

# %%
import numpy as np
from sklearn.datasets import load_wine, load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import r2_score

# %% [markdown]
# 1. Perceptron from scratch on NAND, NOR, and XOR, counting updates

# %%
# 1. Perceptron from scratch on NAND, NOR, and XOR, counting updates
class Perceptron:                                      # one neuron: step(w . x + b)
    def __init__(self, lr=0.1, epochs=50): self.lr, self.epochs = lr, epochs
    def predict(self, X): return (X @ self.w + self.b > 0).astype(int)
    def fit(self, X, y):
        self.w = np.zeros(X.shape[1]); self.b = 0.0; self.converged = None; self.updates = 0
        for ep in range(self.epochs):
            errors = 0
            for xi, yi in zip(X, y):
                upd = self.lr * (yi - int(xi @ self.w + self.b > 0))
                self.w += upd * xi; self.b += upd
                errors += int(upd != 0); self.updates += int(upd != 0)
            if errors == 0: self.converged = ep; break     # a clean pass = converged
        return self
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
gates = {"NAND": np.array([1, 1, 1, 0]), "NOR": np.array([1, 0, 0, 0]),
         "XOR": np.array([0, 1, 1, 0])}                    # only XOR is not linearly separable
for name, y in gates.items():
    p = Perceptron().fit(X, y)
    conv = f"epoch {p.converged}" if p.converged is not None else "never"
    print(f"{name:4}: acc {np.mean(p.predict(X) == y):.2f}  updates {p.updates}  converged {conv}")
# expected output: NAND: acc 1.00  updates 8  converged epoch 3
# expected output: NOR : acc 1.00  updates 5  converged epoch 3
# expected output: XOR : acc 0.50  updates 197  converged never
# XOR makes 197 of the 200 updates its budget allows (50 passes over 4 rows): no pass is ever clean, so it never
# converges; the three rows it skips are ones the current line happens to get right on that pass.

# %% [markdown]
# 2. Hand-built XNOR network, forward propagation

# %%
# 2. Hand-built XNOR network, forward propagation
step = lambda z: (z > 0).astype(float)
W1 = np.array([[1.0, 1.0], [-1.0, -1.0]]); b1 = np.array([-1.5, 0.5])  # unit1=AND, unit2=NOR
W2 = np.array([1.0, 1.0]); b2 = -0.5                                   # output fires if either does
H = step(X @ W1.T + b1); out = step(H @ W2 + b2)
print("hidden:", H.astype(int).tolist())
print("output:", out.astype(int).tolist(), "matches XNOR:", out.tolist() == [1, 0, 0, 1])
# expected output: hidden: [[0, 1], [0, 0], [0, 0], [1, 0]]
# expected output: output: [1, 0, 0, 1] matches XNOR: True
# unit 1 (AND) captures the corner (1,1) alone and unit 2 (NOR) the corner (0,0) alone, the two corners where XNOR is 1;
# the hidden pairs are (0,1), (0,0), (0,0), (1,0), and the output's OR of the two units, h1 + h2 - 0.5 > 0, is one line.

# %% [markdown]
# 3. Saturation: the realistic vanishing factor, not just the best case

# %%
# 3. Saturation: the realistic vanishing factor, not just the best case
sigmoid = lambda z: 1 / (1 + np.exp(-z))
dsig = lambda z: sigmoid(z) * (1 - sigmoid(z))
print(round(float(dsig(0.0)), 3), round(float(dsig(4.0)), 5))
print(f"best case 0.25**5 = {0.25 ** 5:.3e}   saturated {float(dsig(4.0)) ** 5:.3e}")
import math
print("depth where 0.25**d < 1e-6:", math.ceil(math.log(1e-6) / math.log(0.25)))
# expected output: 0.25 0.01766
# expected output: best case 0.25**5 = 9.766e-04   saturated 1.719e-09
# expected output: depth where 0.25**d < 1e-6: 10

# %% [markdown]
# 4. Stable softmax with a temperature

# %%
# 4. Stable softmax with a temperature
def softmax_T(z, T=1.0):
    zt = z / T
    e = np.exp(zt - zt.max())                  # subtract the max for stability
    return e / e.sum()
z = np.array([2.0, 1.0, 0.1])
for T in [0.5, 1.0, 2.0]:
    print(T, softmax_T(z, T).round(3).tolist())
big = softmax_T(np.array([1000.0, 1001.0, 1002.0]))
print("large logits:", big.round(3).tolist(), "sum", round(float(big.sum()), 3))
# expected output: 0.5 [0.864, 0.117, 0.019]
# expected output: 1.0 [0.659, 0.242, 0.099]
# expected output: 2.0 [0.502, 0.304, 0.194]
# expected output: large logits: [0.09, 0.245, 0.665] sum 1.0

# %% [markdown]
# 5. Huber's threshold, and BCE's asymmetric growth

# %%
# 5. Huber's threshold, and BCE's asymmetric growth
def huber(r, d):
    a = np.abs(r); return float(np.mean(np.where(a <= d, 0.5 * r ** 2, d * (a - 0.5 * d))))
def bce(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12); return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)))
ro = np.array([8.0, -0.3, 0.2, -0.4, 0.1])     # one outlier among small residuals
for d in [0.5, 1.0, 2.0]:
    print(d, round(huber(ro, d), 3))
print([round(bce(1, p), 3) for p in [0.9, 0.6, 0.1]])
# expected output: 0.5 0.805
# expected output: 1.0 1.53
# expected output: 2.0 2.83
# expected output: [0.105, 0.511, 2.303]
# the four small residuals stay inside |r| <= delta at every delta, so their quadratic terms are fixed (they sum to 0.15),
# while the outlier's linear term delta * (8 - delta / 2) grows with delta, 3.875, 7.5, and 14, hence the means 0.805, 1.53, 2.83;
# BCE at y = 1 is -log p, so it roughly quintuples from 0.105 to 0.511 between p = 0.9 and 0.6 and reaches 2.303, about 22 times, at p = 0.1.

# %% [markdown]
# 6. Pure-NumPy tanh MLP with gradient checking on the wine data

# %%
# 6. Pure-NumPy tanh MLP with gradient checking on the wine data
rng = np.random.default_rng(0)
def softmax2(Z):
    e = np.exp(Z - Z.max(axis=1, keepdims=True)); return e / e.sum(axis=1, keepdims=True)
class MLP:                                            # 13 -> 16 (tanh) -> 3 (softmax)
    def __init__(self, n_in, n_h, n_out):
        self.W1 = rng.normal(0, np.sqrt(2 / (n_in + n_h)), (n_in, n_h)); self.b1 = np.zeros(n_h)  # Xavier
        self.W2 = rng.normal(0, np.sqrt(2 / (n_h + n_out)), (n_h, n_out)); self.b2 = np.zeros(n_out)
    @classmethod
    def from_weights(cls, W1, W2):
        m = cls.__new__(cls)
        m.W1, m.W2 = W1.copy(), W2.copy()
        m.b1 = np.zeros(W1.shape[1]); m.b2 = np.zeros(W2.shape[1])
        return m
    def forward(self, Xb):
        self.X = Xb; self.z1 = Xb @ self.W1 + self.b1; self.a1 = np.tanh(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2; self.p = softmax2(self.z2); return self.p
    def loss(self, Xb, Yb):
        return float(-np.mean(np.sum(Yb * np.log(self.forward(Xb) + 1e-12), axis=1)))
    def backward(self, Yb):
        n = len(self.X); dz2 = (self.p - Yb) / n          # mean loss: (p - y) / n
        dW2 = self.a1.T @ dz2; db2 = dz2.sum(0)
        dz1 = (dz2 @ self.W2.T) * (1 - self.a1 ** 2)      # tanh derivative, not a ReLU gate
        return self.X.T @ dz1, dz1.sum(0), dW2, db2
    def step(self, grads, lr):
        dW1, db1, dW2, db2 = grads
        self.W1 = self.W1 - lr * dW1; self.b1 = self.b1 - lr * db1
        self.W2 = self.W2 - lr * dW2; self.b2 = self.b2 - lr * db2
Xw, yw = load_wine(return_X_y=True)                   # 178 bottles, 13 features, 3 cultivars
Yw = np.eye(3)[yw]
Xtr, Xte, Ytr, Yte, ytr, yte = train_test_split(Xw, Yw, yw, test_size=0.3,
                                                random_state=0, stratify=yw)
sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)   # scaler fit on the training fold only (3.3.5)
net = MLP(13, 16, 3); Xb, Yb = Xtr[:32], Ytr[:32]
net.forward(Xb); analytic = net.backward(Yb)[0]           # dW1 from backprop
errs = []
for i, j in [(2, 5), (7, 11), (12, 0), (5, 15)]:
    o = net.W1[i, j]
    net.W1[i, j] = o + 1e-5; lp = net.loss(Xb, Yb)
    net.W1[i, j] = o - 1e-5; lm = net.loss(Xb, Yb); net.W1[i, j] = o
    num = (lp - lm) / 2e-5
    errs.append(abs(num - analytic[i, j]) / (abs(num) + abs(analytic[i, j]) + 1e-12))
print("max relative error below 1e-6:", bool(max(errs) < 1e-6))
# expected output: max relative error below 1e-6: True

# %% [markdown]
# 7. Train the tanh wine network end to end (miniproject)

# %%
# 7. Train the tanh wine network end to end (miniproject)
net = MLP(13, 16, 3); lr, epochs, bs = 0.5, 200, 16
for ep in range(epochs):
    perm = rng.permutation(len(Xtr))                      # shuffle each epoch
    for k in range(0, len(Xtr), bs):
        idx = perm[k:k + bs]
        net.forward(Xtr[idx]); net.step(net.backward(Ytr[idx]), lr)   # one minibatch step
tr = (net.forward(Xtr).argmax(1) == ytr).mean()
te = (net.forward(Xte).argmax(1) == yte).mean()
print(f"train acc {tr:.3f}  test acc {te:.3f}")
# expected output: train acc 1.000  test acc 0.981

# %% [markdown]
# 8. Zero init is not the only symmetric trap

# %%
# 8. Zero init is not the only symmetric trap
def train_init(W1, W2, n_epochs=100):
    m = MLP.from_weights(W1, W2)                          # set weights; do not draw a new init
    r = np.random.default_rng(2)                          # fixed shuffling stream
    for ep in range(n_epochs):
        perm = r.permutation(len(Xtr))
        for k in range(0, len(Xtr), bs):
            idx = perm[k:k + bs]; m.forward(Xtr[idx]); m.step(m.backward(Ytr[idx]), lr)
    return m, (m.forward(Xte).argmax(1) == yte).mean()
r0 = np.random.default_rng(1)
schemes = {"zeros": (np.zeros((13, 16)), np.zeros((16, 3))),
           "const 0.5": (np.full((13, 16), 0.5), np.full((16, 3), 0.5)),
           "Xavier": (r0.normal(0, np.sqrt(2 / (13 + 16)), (13, 16)), r0.normal(0, np.sqrt(2 / (16 + 3)), (16, 3)))}
for name, (W1, W2) in schemes.items():
    m, acc = train_init(W1, W2)
    ident = f"  units identical: {np.allclose(m.W1, m.W1[:, :1])}" if name != "Xavier" else ""
    print(f"{name:>9}: test acc {acc:.3f}{ident}")
# expected output:     zeros: test acc 0.389  units identical: True
# expected output: const 0.5: test acc 0.944  units identical: True
# expected output:    Xavier: test acc 0.981
# zeros: with W1 = 0 and W2 = 0 the hidden layer outputs tanh(0) = 0, so dW2 = a1.T @ dz2 = 0 and dz1 = (dz2 @ W2.T) * (1 - a1**2) = 0;
# only b2 ever moves, toward the class priors, and the net predicts the majority class, 21 of the 54 test bottles, 0.389.
# const 0.5: nonzero weights give nonzero gradients, so it trains, but all 16 hidden units share the same weights and
# the same gradient at every step, so they stay one shared feature, and 0.944 is what one feature buys.
# Xavier: 16 distinct random units give 16 distinct features, 0.981.

# %% [markdown]
# 9. Activation comparison on breast cancer: the digits ordering need not replicate

# %%
# 9. Activation comparison on breast cancer: the digits ordering need not replicate
Xb2, yb2 = load_breast_cancer(return_X_y=True)        # 569 tumors, 30 features, binary
for act in ["logistic", "tanh", "relu"]:
    cv_means = []
    for seed in range(5):                              # five initializations, as 10.4.6 does
        clf = make_pipeline(StandardScaler(),
                            MLPClassifier(hidden_layer_sizes=(16, 16), activation=act,
                                          max_iter=3000, random_state=seed))
        cv_means.append(cross_val_score(clf, Xb2, yb2, cv=5).mean())
    print(f"{act:>8}: seeds {np.round(cv_means, 3).tolist()}  mean {np.mean(cv_means):.3f}")
# expected output: logistic: seeds [0.974, 0.974, 0.97, 0.972, 0.974]  mean 0.973
# expected output:     tanh: seeds [0.972, 0.97, 0.975, 0.974, 0.975]  mean 0.973
# expected output:     relu: seeds [0.968, 0.97, 0.977, 0.972, 0.97]  mean 0.972
# Seed 0 alone would rank logistic > tanh > relu, but seed 2 ranks relu > tanh > logistic and seeds 3 and 4 put tanh
# first: on this dataset the three activations sit within seed noise, which is why 10.4.6 repeats its comparison over
# five seeds before ranking anything. The digits ranking (logistic last on every seed) need not replicate here, and no
# vanishing-gradient ranking has to: with two hidden layers there is no deep stack for the 0.25^d shrinkage of the
# sigmoid derivative to act on, and a cross-validated accuracy never measures a gradient.

# %% [markdown]
# 10. Universal approximation on a kinked target |sin(3x)|

# %%
# 10. Universal approximation on a kinked target |sin(3x)|
rng2 = np.random.default_rng(0)
xt = rng2.uniform(-2, 2, 400); yt = np.abs(np.sin(3 * xt))             # kinks at multiples of pi/3
Xk = xt.reshape(-1, 1)
for h in [1, 4, 16, 64]:
    best = max(r2_score(yt, MLPRegressor(hidden_layer_sizes=(h,), activation="tanh",
               solver="lbfgs", max_iter=20000, tol=1e-9,
               random_state=s).fit(Xk, yt).predict(Xk)) for s in range(5))
    print(f"h={h:>2} best R2", "below 0.1" if best < 0.1 else "above 0.99" if best > 0.99 else "between 0.1 and 0.99")
# expected output: h= 1 best R2 below 0.1
# expected output: h= 4 best R2 between 0.1 and 0.99
# expected output: h=16 best R2 above 0.99
# expected output: h=64 best R2 above 0.99
# The listing prints the band of each width's best R2, not its third decimal. A fit ends in whichever basin of
# the loss its path reaches, and the last bits of the rounding can change the path: printed to three decimals,
# the same fits gave a best R2 of 0.693 at four units on the author's Windows PC and 0.750 on the Mac. The
# staircase does not move: near zero for one unit, partial for four, essentially perfect from sixteen up.

# %% [markdown]
# 11. Hand backward pass, tanh hidden units, sigmoid output, label 0

# %%
# 11. Hand backward pass, tanh hidden units, sigmoid output, label 0
x = np.array([1.0, -1.0]); y = 0.0
W1 = np.array([[0.2, -0.1], [0.4, 0.3]]); b1 = np.array([0.1, -0.2])
w2 = np.array([0.5, -0.4]); b2 = 0.1
z1 = W1 @ x + b1; a1 = np.tanh(z1); z2 = float(w2 @ a1 + b2)
p = 1 / (1 + np.exp(-z2))
L = -(y * np.log(p) + (1 - y) * np.log(1 - p))
d2 = p - y; d1 = w2 * d2 * (1 - a1 ** 2)
print([round(float(v), 4) for v in z1], [round(float(v), 4) for v in a1], round(z2, 4), round(p, 4), round(L, 4))
print([round(float(v), 4) for v in d1], round(d2, 4))
print(np.outer(d1, x).round(4), (d2 * a1).round(4), round(d2, 4))
def loss_of(W):
    a = np.tanh(W @ x + b1)
    pp = 1 / (1 + np.exp(-(float(w2 @ a) + b2)))
    return -(y * np.log(pp) + (1 - y) * np.log(1 - pp))
fd = (loss_of(W1 + np.array([[1e-5, 0],[0, 0]])) - loss_of(W1 - np.array([[1e-5, 0],[0, 0]]))) / 2e-5
print(round(float(fd), 4))
# expected output: [0.4, -0.1] [0.3799, -0.0997] 0.3298 0.5817 0.8716
# expected output: [0.2489, -0.2304] 0.5817
# expected output: [[ 0.2489 -0.2489]
# expected output:  [-0.2304  0.2304]] [ 0.221 -0.058] 0.5817
# expected output: 0.2489

# %% [markdown]
# 12. Dead-unit count at two learning rates, 20 epochs, He init

# %%
# 12. Dead-unit count at two learning rates, 20 epochs, He init
from sklearn.datasets import load_digits
Xd, yd = load_digits(return_X_y=True)
Xd = StandardScaler().fit_transform(Xd)
Y = np.eye(10)[yd]
Xtr, Xte, Ytr, Yte, ytr, yte = train_test_split(Xd, Y, yd, test_size=0.25, random_state=0)
def dead_run(lr):
    rng = np.random.default_rng(0)
    W1 = rng.normal(0, np.sqrt(2 / 64), (64, 32)); b1 = np.zeros(32)
    W2 = rng.normal(0, np.sqrt(2 / 32), (32, 10)); b2 = np.zeros(10)
    blowup = None                                      # first epoch after which the largest |W1| entry exceeds 1e6
    for ep in range(20):
        perm = rng.permutation(len(Xtr))
        for k in range(0, len(Xtr), 64):
            idx = perm[k:k + 64]
            xb, yb = Xtr[idx], Ytr[idx]
            z1 = xb @ W1 + b1; a1 = np.maximum(0, z1); z2 = a1 @ W2 + b2
            z2s = z2 - z2.max(1, keepdims=True)
            pr = np.exp(z2s); pr = pr / pr.sum(1, keepdims=True)
            dz2 = (pr - yb) / len(xb); dz1 = (dz2 @ W2.T) * (z1 > 0)
            W1 -= lr * (xb.T @ dz1); b1 -= lr * dz1.sum(0)
            W2 -= lr * (a1.T @ dz2); b2 -= lr * dz2.sum(0)
        if np.abs(W1).max() > 1e6: blowup = ep + 1; break  # a diverged run stops, and is read, where it diverges
    active = np.mean(Xtr @ W1 + b1 > 0, axis=0)        # share of training rows on which each hidden unit fires
    dead = int(np.sum(active == 0))                    # dead: never positive on any training row
    near_dead = int(np.sum(active < 0.01))             # near dead: fires on fewer than 1 percent of the rows
    pred = (np.maximum(0, Xte @ W1 + b1) @ W2 + b2).argmax(1)
    print(f"lr {lr}: dead {dead}, near dead {near_dead}, test acc {np.mean(pred == yte):.3f}, "
          f"smallest active share {active.min():.2f}, largest |W1| {np.abs(W1).max():.0e}, "
          f"first epoch past 1e6: {blowup}, top predicted class share {np.bincount(pred).max() / len(pred):.2f}")
print("largest class share of the test set:", round(float(np.bincount(yte).max() / len(yte)), 3))
dead_run(0.5)
dead_run(3.0)
# expected output: largest class share of the test set: 0.116
# expected output: lr 0.5: dead 0, near dead 0, test acc 0.969, smallest active share 0.38, largest |W1| 9e-01, first epoch past 1e6: None, top predicted class share 0.11
# expected output: lr 3.0: dead 0, near dead 0, test acc 0.142, smallest active share 0.02, largest |W1| 4e+07, first epoch past 1e6: 2, top predicted class share 0.51
# Why the learning rate 3 run collapses without a single dead unit, and why it is read at its second epoch. Once
# the weights explode, every logit is a sum of products of enormous numbers and the last bits of the rounding
# decide which units fire and which class wins: run on to 20 epochs, the same training printed test accuracies of
# 0.118 on the author's Windows PC and 0.280 on the Mac. So the run stops at the end of the first epoch after which
# the largest |W1| entry exceeds 1e6 and is read there, before that growth reaches the printed digits.
# 1. Divergence, not death. At learning rate 3 the steps are far too large for the loss surface: the largest entry
#    of W1 passes 1e6 in the second epoch and stands at about 4e7 at its end, against 9e-01 after 20 epochs at
#    0.5. The two layers feed each other's growth, because the gradient of W2 is a1.T @ dz2, which scales with the
#    hidden activations and so with W1, while the gradient of W1 carries W2.T, so every step multiplies both.
# 2. No unit dies, because a dead unit needs its hyperplane to miss the whole training cloud: z = x . w + b must
#    be negative or zero on all 1,347 standardized training rows. Here each unit's weights and bias grow together,
#    driven by the same error signal dz1 (the bias gradient is its column sum, the weight gradient its product
#    with the inputs), and the hyperplane keeps cutting through the cloud: the least active unit still fires on 2
#    percent of the rows at learning rate 3, against 38 percent at 0.5. Dying ReLU (10.4.2) needs a unit pushed
#    off the data for good, which this run has not done, so the dead count and the near-dead count are both zero.
# 3. Accuracy still collapses because every logit is a sum of products of large numbers, so the units with the
#    largest weights decide the answer on most inputs where they fire: 51 percent of the test digits receive the
#    same class, and the test accuracy is 0.142, only a little above the 0.116 that always predicting the largest
#    class would score and far below the 0.969 of learning rate 0.5. The dead-unit count flags only one way a ReLU
#    network can fail, so a loss curve and a check on the size of the weights belong beside it.

# %% [markdown]
# ## 10.8 coding drills

# %%
import numpy as np
from scipy.stats import norm

# %% [markdown]
# Perceptrons and MLPs

# %%
# Perceptrons and MLPs
w = np.zeros(2); b = 0.0; xi = np.array([1, 0]); yi = 1; lr = 0.1
pred = int(xi @ w + b > 0); w = w + lr * (yi - pred) * xi; b = b + lr * (yi - pred)
print("after 1 update:", w.tolist(), b)                  # expected output: after 1 update: [0.1, 0.0] 0.1
step = lambda z: (z > 0).astype(float)
print("step(-0.5), step(0.5):", step(np.array([-0.5, 0.5])))  # expected output: step(-0.5), step(0.5): [0. 1.]
Xg = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Hrep = step(Xg @ np.array([[1., 1.], [1., 1.]]).T + np.array([-0.5, -1.5]))
print("XOR hidden rep:", Hrep.astype(int).tolist())      # expected output: XOR hidden rep: [[0, 0], [1, 0], [1, 0], [1, 1]]

# %% [markdown]
# Activations

# %%
# Activations
sigmoid = lambda z: 1 / (1 + np.exp(-z))
print("sigmoid(0,2,-2):", [round(float(sigmoid(z)), 3) for z in [0, 2, -2]])  # expected output: sigmoid(0,2,-2): [0.5, 0.881, 0.119]
print("sig'(0):", round(float(sigmoid(0) * (1 - sigmoid(0))), 3))             # expected output: sig'(0): 0.25
print("leaky(-2):", round(float(np.where(-2 > 0, -2, 0.01 * -2)), 3))         # expected output: leaky(-2): -0.02
print("gelu(1):", round(float(1 * norm.cdf(1)), 3))                           # expected output: gelu(1): 0.841
sm = np.exp(np.array([3., 1., 0.2]) - 3.); sm = sm / sm.sum()
print("softmax sum:", round(float(sm.sum()), 3))                              # expected output: softmax sum: 1.0
print("0.25**5:", f"{0.25 ** 5:.3e}")                                         # expected output: 0.25**5: 9.766e-04

# %% [markdown]
# Losses

# %%
# Losses
r = np.array([8., 0.1, 0.1])
print("mse:", round(float(np.mean(r ** 2)), 3), "mae:", round(float(np.mean(np.abs(r))), 3))  # expected output: mse: 21.34 mae: 2.733
p = np.clip(0.02, 1e-12, 1 - 1e-12)
print("bce confident wrong:", round(float(-np.log(p)), 3))                    # expected output: bce confident wrong: 3.912
probs = np.exp(np.array([2., 1., 0.1]) - 2.); probs = probs / probs.sum()
print("cce true class2:", round(float(-np.log(probs[2])), 3))                 # expected output: cce true class2: 2.317
hard = np.array([0., 1., 0.]); eps = 0.1
print("label smooth:", (hard * (1 - eps) + eps / 3).round(3).tolist())        # expected output: label smooth: [0.033, 0.933, 0.033]

# %% [markdown]
# Backpropagation

# %%
# Backpropagation
pv = np.array([0.7, 0.2, 0.1]); yv = np.array([1., 0., 0.])
print("output error p-y:", (pv - yv).round(3).tolist())                       # expected output: output error p-y: [-0.3, 0.2, 0.1]
f = lambda w: (w - 3) ** 2; w0 = 5.0; e = 1e-5
print("finite diff:", round((f(w0 + e) - f(w0 - e)) / (2 * e), 3), "analytic:", 2 * (w0 - 3))  # expected output: finite diff: 4.0 analytic: 4.0
print("He scale sqrt(2/64):", round(float(np.sqrt(2 / 64)), 4))               # expected output: He scale sqrt(2/64): 0.1768
