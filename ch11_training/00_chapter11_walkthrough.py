"""Chapter 11: Training Deep Networks

Section-aligned walkthrough regenerated from the chapter slice.
"""

# %%
import os as _os
__file__ = _os.path.join(_os.getcwd(), "00_chapter11_walkthrough.py")

# %% [markdown]
# ## 11.1 Optimizers

# %% [markdown]
# ## 11.1.1 SGD and momentum

# %% [markdown]
# Example 11-1. The from-scratch optimizer race on an ill-conditioned quadratic: SGD, momentum, and Nesterov.

# %%
import numpy as np
def f(p): return 0.5 * (10 * p[0] ** 2 + p[1] ** 2)   # an ill-conditioned bowl (10:1 curvature)
def grad(p): return np.array([10 * p[0], p[1]])       # steep across, shallow along: a ravine
start = np.array([-4.0, -4.0])
def run(update, steps=300, tol=1e-4):                  # (first step below tol, step it stays below from)
    p, state, losses = start.copy(), {}, [f(start)]
    for tt in range(1, steps + 1):
        p = update(p, grad(p), tt, state); losses.append(f(p))
    below = np.array(losses) < tol
    first = int(np.argmax(below)) if below.any() else None
    last_above = int(np.flatnonzero(~below)[-1])       # the last step whose loss was not yet below tol
    return first, (last_above + 1 if last_above < steps else None)   # None: still above tol at the end
def sgd(p, g, t, s, lr=0.05): return p - lr * g
def momentum(p, g, t, s, lr=0.05, beta=0.9):
    s["v"] = beta * s.get("v", 0) + g; return p - lr * s["v"]
def nesterov(p, g, t, s, lr=0.05, beta=0.9):
    s["v"] = beta * s.get("v", 0) + g; return p - lr * (g + beta * s["v"])   # look-ahead correction
for name, upd in [("SGD", sgd), ("Momentum", momentum), ("Nesterov", nesterov)]:
    print(name, run(upd))
# SGD (111, 111)   Momentum (78, 128)   Nesterov (36, 73)   (first below, then settled for good)

# %%
from functools import partial                          # freeze settings into an update rule (1.7.11)
L, mu = 10.0, 1.0                                      # the bowl's largest and smallest curvatures
root = (np.sqrt(L / mu) - 1) / (np.sqrt(L / mu) + 1)   # momentum's best swing factor, sqrt(beta)
lr_sgd = 2 / (L + mu)                                  # 2.6.3's best fixed step for plain SGD
lr_mom, beta_mom = (1 + root) ** 2 / L, root ** 2      # the tuned momentum rate and beta
print(f"SGD      lr={lr_sgd:.3f}:", run(partial(sgd, lr=lr_sgd)))
print(f"Momentum lr={lr_mom:.3f}, beta={beta_mom:.3f}:", run(partial(momentum, lr=lr_mom, beta=beta_mom)))

# %% [markdown]
# ## 11.1.2 Adaptive methods: AdaGrad and RMSProp

# %% [markdown]
# Example 11-2. AdaGrad and RMSProp from scratch, raced on the ravine with the `run` harness of Example 11-1, and where RMSProp sits after 300 steps.

# %%
def adagrad(p, g, t, s, lr=0.5, eps=1e-8):
    s["G"] = s.get("G", 0) + g ** 2                    # accumulate squared gradients forever
    return p - lr * g / (np.sqrt(s["G"]) + eps)
def rmsprop(p, g, t, s, lr=0.1, rho=0.9, eps=1e-8):
    s["E"] = rho * s.get("E", 0) + (1 - rho) * g ** 2  # a moving average instead, so it never dies
    return p - lr * g / (np.sqrt(s["E"]) + eps)
for name, upd in [("AdaGrad", adagrad), ("RMSProp", rmsprop)]:
    print(name, run(upd))
p, s = start.copy(), {}
for tt in range(1, 301): p = rmsprop(p, grad(p), tt, s)   # where RMSProp is after the 300 steps
print("RMSProp at step 300:", p.round(3), " loss", round(f(p), 4))
# AdaGrad (135, 135): slow but steady   RMSProp (63, None): fast at first, then it hops at +-0.05 for good

# %% [markdown]
# ## 11.1.3 Adam and AdamW

# %% [markdown]
# Example 11-3. Adam and AdamW from scratch on the ravine, then both on the same bowl moved so that its minimum sits at (3, 3).

# %%
def adam(p, g, t, s, lr=0.3, b1=0.9, b2=0.999, eps=1e-8):
    s["m"] = b1 * s.get("m", 0) + (1 - b1) * g         # 1st moment (momentum)
    s["v"] = b2 * s.get("v", 0) + (1 - b2) * g ** 2    # 2nd moment (RMSProp)
    mhat, vhat = s["m"] / (1 - b1 ** t), s["v"] / (1 - b2 ** t)   # bias correction
    return p - lr * mhat / (np.sqrt(vhat) + eps)
def adamw(p, g, t, s, lr=0.3, b1=0.9, b2=0.999, eps=1e-8, wd=0.01):
    s["m"] = b1 * s.get("m", 0) + (1 - b1) * g
    s["v"] = b2 * s.get("v", 0) + (1 - b2) * g ** 2
    mhat, vhat = s["m"] / (1 - b1 ** t), s["v"] / (1 - b2 ** t)
    return p - lr * (mhat / (np.sqrt(vhat) + eps) + wd * p)       # decoupled weight decay
for name, upd in [("Adam", adam), ("AdamW", adamw)]:
    print(name, run(upd))
center = np.array([3.0, 3.0])                          # the same bowl, moved so its minimum sits at (3, 3)
for name, upd in [("Adam", adam), ("AdamW", adamw)]:
    p, s = start.copy(), {}
    for tt in range(1, 2001): p = upd(p, grad(p - center), tt, s)   # the gradient of f(p - center)
    print(f"{name:5} on the shifted bowl after 2000 steps:", p.round(3), f" loss {f(p - center):.1e}")
# Adam (80, 127)   AdamW (60, 125)   shifted bowl: Adam lands on (3, 3), AdamW is held short of it

# %%
import torch
nadam_start = torch.tensor([-4.0, -4.0])
def nadam_loss(p): return 0.5 * (10 * p[0] ** 2 + p[1] ** 2)
def race(ctor, steps=300, tol=1e-4):
    p = nadam_start.clone().requires_grad_(True); opt = ctor([p])
    losses = [float(nadam_loss(p).detach())]
    for _ in range(steps):
        opt.zero_grad(); loss = nadam_loss(p); loss.backward(); opt.step()
        losses.append(float(nadam_loss(p).detach()))
    below = torch.tensor(losses) < tol
    first = int(below.float().argmax()) if bool(below.any()) else None
    last = int((~below).nonzero()[-1])
    return first, (int(last) + 1 if int(last) < steps else None)
print("NAdam", race(lambda ps: torch.optim.NAdam(ps, lr=0.3)))
print("Adamax", race(lambda ps: torch.optim.Adamax(ps, lr=0.3)))

# %% [markdown]
# Example 11-4. Adamax, a one-line Nesterov correction on Adam, and the NAdam rule printed in the PyTorch documentation, written from scratch for the `run` harness of Example 11-1.

# %%
def adamax_scratch(p, g, t, s, lr=0.3, b1=0.9, b2=0.999, eps=1e-8):
    s["m"] = b1 * s.get("m", 0) + (1 - b1) * g
    s["u"] = np.maximum(b2 * s.get("u", 0), np.abs(g))          # running maximum, no bias correction
    return p - lr * (s["m"] / (1 - b1 ** t)) / (s["u"] + eps)
def nadam_one_line(p, g, t, s, lr=0.3, b1=0.9, b2=0.999, eps=1e-8):
    s["m"] = b1 * s.get("m", 0) + (1 - b1) * g
    s["v"] = b2 * s.get("v", 0) + (1 - b2) * g ** 2
    mhat, vhat = s["m"] / (1 - b1 ** t), s["v"] / (1 - b2 ** t)
    return p - lr * (b1 * mhat + (1 - b1) * g / (1 - b1 ** t)) / (np.sqrt(vhat) + eps)   # the one line
def nadam_documented(p, g, t, s, lr=0.3, b1=0.9, b2=0.999, eps=1e-8, psi=0.004):
    mu = b1 * (1 - 0.5 * 0.96 ** (t * psi))                     # momentum coefficient at step t
    mu_next = b1 * (1 - 0.5 * 0.96 ** ((t + 1) * psi))          # and at step t + 1
    s["prod"] = s.get("prod", 1.0) * mu                         # running product of the mu's
    s["m"] = b1 * s.get("m", 0) + (1 - b1) * g
    s["v"] = b2 * s.get("v", 0) + (1 - b2) * g ** 2
    mbar = mu_next * s["m"] / (1 - s["prod"] * mu_next) + (1 - mu) * g / (1 - s["prod"])
    return p - lr * mbar / (np.sqrt(s["v"] / (1 - b2 ** t)) + eps)
for name, upd in [("Adamax from scratch", adamax_scratch), ("one-line Nesterov Adam", nadam_one_line),
                  ("documented NAdam", nadam_documented)]:
    print(name, run(upd))

# %% [markdown]
# Example 11-5. The four-step PyTorch training loop with `torch.optim` SGD with momentum, Adam, and AdamW on the ravine, checked against the from-scratch rules, and two param groups that exempt a bias from weight decay.

# %%
import torch
torch.manual_seed(0)
def make_p(): return torch.tensor([-4.0, -4.0], requires_grad=True)   # same start as the by-hand runs
def loss_of(p): return 0.5 * (10 * p[0]**2 + p[1]**2)               # the ill-conditioned bowl
def run(opt_ctor, steps=60):
    p = make_p(); opt = opt_ctor([p])
    for _ in range(steps):
        opt.zero_grad()      # clear gradients from the previous step (backward accumulates)
        loss = loss_of(p)
        loss.backward()      # autodiff fills p.grad
        opt.step()           # the optimizer applies its own update rule
    return loss.item()       # computed before the last update, so this is the loss after 59 updates
print("SGD+momentum:", round(run(lambda ps: torch.optim.SGD(ps, lr=0.05, momentum=0.9)), 6))
print("Adam        :", round(run(lambda ps: torch.optim.Adam(ps, lr=0.3)), 6))
print("AdamW       :", round(run(lambda ps: torch.optim.AdamW(ps, lr=0.3, weight_decay=0.01)), 6))
def scratch(update, steps=59):                       # the from-scratch rule for the same 59 updates
    p, s = start.copy(), {}
    for tt in range(1, steps + 1): p = update(p, grad(p), tt, s)
    return round(f(p), 6)
print("from scratch:", scratch(momentum), scratch(adam), scratch(adamw))
net = torch.nn.Linear(4, 1)                                          # a real layer to show param groups
opt = torch.optim.AdamW([
    {"params": net.weight, "weight_decay": 0.01},    # decay the weights
    {"params": net.bias,   "weight_decay": 0.0},     # but not the bias
], lr=1e-3)
print("weight decays:", [g["weight_decay"] for g in opt.param_groups])

# %% [markdown]
# ## 11.1.4 What we built, and a self-check

# %% [markdown]
# ## 11.2 Learning-rate strategies

# %% [markdown]
# Example 11-6. Five learning-rate schedules computed by hand over 1,000 steps: constant, step decay, exponential, cosine, and a 100-step warmup followed by cosine.

# %%
import numpy as np
T, base = 1000, 0.1
t = np.arange(T)
schedules = {
    "constant":    np.full(T, base),
    "step decay":  base * 0.5 ** (t // 300),                    # halve every 300 steps
    "exponential": base * np.exp(-3 * t / T),
    "cosine":      base * 0.5 * (1 + np.cos(np.pi * t / T)),     # smoothly anneal to 0
}
warm = 100
schedules["warmup+cosine"] = np.where(t < warm, base * t / warm,
                                      base * 0.5 * (1 + np.cos(np.pi * (t - warm) / (T - warm))))
print("schedule        step0   step100  step500  step999")
for name, lr in schedules.items():
    print(f"{name:14} {lr[0]:.4f}  {lr[100]:.4f}  {lr[500]:.4f}  {lr[999]:.4f}")
# constant 0.1 flat; step decay 0.1/0.1/0.05/0.0125; cosine 0.1->0; warmup+cosine ramps 0->0.1 then anneals to 0

# %%
def minimize(lr_sched, steps=1000, seed=0):
    r = np.random.default_rng(seed); x = 5.0; losses = []
    for i in range(steps):
        g = 2 * x + r.normal(0, 2.0)            # noisy gradient of x^2, as minibatches give
        x -= lr_sched[i] * g; losses.append(x ** 2)
    return np.mean(losses[-50:])                # average loss over the last 50 steps
print(round(minimize(np.full(1000, 0.1)), 4))      # 0.2053  a constant rate keeps bouncing on the noise
print(round(minimize(schedules["cosine"]), 4))     # 0.0028  cosine anneals down and settles far lower

# %%
rates = np.logspace(-4, 0.7, 40)                # sweep the rate up exponentially: 1e-4 -> ~5
x = 5.0; r = np.random.default_rng(0); losses = []
for lr in rates:                                # one step per rate, recording the loss after each
    g = 2 * x + r.normal(0, 2.0)                # the same noisy gradient of x^2
    x -= lr * g; losses.append(x ** 2)
best = int(np.argmin(losses))
print(f"loss at lr=1e-4: {losses[0]:.1f}   lowest {losses[best]:.4f} at lr={rates[best]:.2f}   at lr={rates[-1]:.1f}: {losses[-1]:.1e}")

# %%
import torch
p = torch.nn.Parameter(torch.zeros(1))
opt = torch.optim.SGD([p], lr=0.1)                        # base learning rate 0.1
cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1000)   # half-cosine to ~0 over 1000 steps
lrs = []
for step in range(1000):
    opt.step()                                            # a (no-op) update so the step counts
    lrs.append(opt.param_groups[0]["lr"])                 # the rate this step used
    cos.step()                                            # advance the schedule one step
print("cosine   step 0/250/500/999:", [round(lrs[i], 4) for i in (0, 250, 500, 999)])

opt2 = torch.optim.SGD([p], lr=0.1)                       # a fresh optimizer for a warmup schedule
warm = torch.optim.lr_scheduler.LambdaLR(opt2, lr_lambda=lambda s: min(1.0, s / 100))  # ramp over 100
lrs2 = []
for step in range(300):
    opt2.step(); lrs2.append(opt2.param_groups[0]["lr"]); warm.step()
print("warmup   step 0/50/100/200:", [round(lrs2[i], 4) for i in (0, 50, 100, 200)])

# %% [markdown]
# ## 11.2.1 Plateau cuts, the one-cycle policy, and warm restarts

# %%
import torch
opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
plat = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.1, patience=2)
lrs = [round(opt.param_groups[0]["lr"], 4)]
for _ in range(6):
    plat.step(1.0); lrs.append(round(opt.param_groups[0]["lr"], 4))
print("plateau", lrs)
opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
oc = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.1, total_steps=1000)
marks = []
for i in range(1000):
    if i in (0, 299, 999):
        marks.append(round(opt.param_groups[0]["lr"], 6))
    opt.step(); oc.step()
print("onecycle", marks)
opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
wr = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=200)
marks = []
for i in range(401):
    if i in (0, 100, 200, 400):
        marks.append((i, round(opt.param_groups[0]["lr"], 6)))
    opt.step(); wr.step()
print("restarts", marks)

# %% [markdown]
# ## 11.2.2 Batch size, and retuning the learning rate

# %% [markdown]
# Example 11-7. A 64-32-10 digit network trained for ten epochs with Adam at batch sizes 32 and 256, with the rate held at 0.001 and then multiplied by 8, scored on a validation split.

# %%
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
torch.manual_seed(0)
Xd, yd = load_digits(return_X_y=True)
Xd = StandardScaler().fit_transform(Xd)
Xtr, Xte, ytr, yte = train_test_split(Xd, yd, test_size=0.25, random_state=0)
Xtr, Xva, ytr, yva = train_test_split(Xtr, ytr, test_size=0.25, random_state=0)
Xva_t = torch.tensor(Xva, dtype=torch.float32); yva_t = torch.tensor(yva, dtype=torch.long)
def run(bs, lr):
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
    opt = torch.optim.Adam(net.parameters(), lr=lr); lf = nn.CrossEntropyLoss()
    dl = DataLoader(TensorDataset(torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.long)),
                    batch_size=bs, shuffle=True)
    updates = 0
    for _ in range(10):
        net.train()
        for xb, yb in dl:
            opt.zero_grad(); lf(net(xb), yb).backward(); opt.step(); updates += 1
    net.eval()
    with torch.no_grad():
        acc = (net(Xva_t).argmax(1) == yva_t).float().mean().item()
    return round(acc, 3), updates
for bs, lr in ((32, 0.001), (256, 0.001), (256, 0.008)):
    print(bs, lr, run(bs, lr))

# %% [markdown]
# ## 11.2.3 What we built, and a self-check

# %% [markdown]
# ## 11.3 Weight initialization

# %% [markdown]
# Example 11-8. A random signal propagated through thirty 256-wide tanh layers at three weight scales, with its standard deviation at layers 1, 10, 20, and 30.

# %%
import numpy as np
rng = np.random.default_rng(0)
N, L, B = 256, 30, 512
def propagate(scale, act):                         # signal std through L layers at a given weight scale
    a = rng.normal(size=(B, N)); stds = []
    for _ in range(L):
        a = act(a @ rng.normal(0, scale, (N, N))); stds.append(a.std())
    return np.array(stds)
xavier = 1 / np.sqrt(N)
print(f"tanh std at layers [1,10,20,30]  (Xavier scale = {xavier:.4f})")
for name, scale in [("too small (0.01)", 0.01), ("Xavier", xavier), ("too large (0.15)", 0.15)]:
    s = propagate(scale, np.tanh)
    print(f"{name:16} {s[0]:.3f} {s[9]:.3f} {s[19]:.3f} {s[29]:.3f}")
# too small: 0.156 0.000 0.000 0.000 (vanishes)   Xavier: 0.628 0.228 0.160 0.128 (stable)   too large: 0.830 0.785 0.788 0.784 (saturates)

# %%
he = np.sqrt(2 / N)
print(f"ReLU std at layers [1,10,20,30]  (He scale = {he:.4f})")
for name, scale in [("Xavier (too small)", xavier), ("He", he)]:
    s = propagate(scale, lambda z: np.maximum(0, z))
    print(f"{name:18} {s[0]:.3f} {s[9]:.3f} {s[19]:.3f} {s[29]:.3f}")
# Xavier: 0.585 0.016 0.001 0.000 (vanishes for ReLU)   He: 0.830 0.798 0.844 0.749 (stable: He doubles the variance)

# %%
import torch, torch.nn as nn
torch.manual_seed(0)
lin_relu = nn.Linear(256, 256)
nn.init.kaiming_normal_(lin_relu.weight, nonlinearity="relu")   # He: variance 2/n_in for ReLU
lin_tanh = nn.Linear(256, 256)
nn.init.xavier_uniform_(lin_tanh.weight)                       # Xavier/Glorot for tanh or sigmoid
print(f"He (ReLU)   std {lin_relu.weight.std().item():.4f}  target {(2/256)**0.5:.4f}")
print(f"Xavier tanh std {lin_tanh.weight.std().item():.4f}  target {(2/(256+256))**0.5:.4f}")
default = nn.Linear(256, 256)                                  # what PyTorch does if you say nothing
print(f"default     std {default.weight.std().item():.4f}")

# %% [markdown]
# ## 11.3.1 What we built, and a self-check

# %% [markdown]
# ## 11.4 Batch normalization

# %% [markdown]
# ## 11.4.1 What it does

# %%
import numpy as np
rng = np.random.default_rng(0)
def batchnorm(x, gamma=1.0, beta=0.0, eps=1e-5):
    return gamma * (x - x.mean(0)) / np.sqrt(x.var(0) + eps) + beta   # standardize, then rescale
x = rng.normal(5.0, 3.0, (256, 4))                 # a shifted, scaled batch
xn = batchnorm(x)
print(f"before BN: mean={x.mean():.2f}, std={x.std():.2f}")    # mean=4.85, std=2.92
print(f"after  BN: mean={xn.mean():.2f}, std={xn.std():.2f}")  # mean=-0.00, std=1.00 (before the learnable gamma/beta)

# %%
def propagate(use_bn, scale=0.01):                     # weights too small, as in 11.3: the raw signal shrinks
    a = rng.normal(size=(256, 128)); stds = []
    for _ in range(20):
        a = np.tanh(a @ rng.normal(0, scale, (128, 128)))
        stds.append(a.std())                           # measured BEFORE normalization, so BN cannot force it
        if use_bn: a = batchnorm(a)
    return np.array(stds)
no_bn, with_bn = propagate(False), propagate(True)
print(f"without BN: {no_bn[0]:.3f}  {no_bn[9]:.1e}  {no_bn[19]:.1e}")
print(f"with    BN: {with_bn[0]:.3f}  {with_bn[9]:.3f}    {with_bn[19]:.3f}")

# %%
import torch
torch.manual_seed(0)
m, d, eps = 6, 3, 1e-5
x = torch.randn(m, d, requires_grad=True)
gamma = torch.tensor([1.5, 0.8, 1.2], requires_grad=True); beta = torch.zeros(d, requires_grad=True)
mu, var = x.mean(0), x.var(0, unbiased=False)
xhat = (x - mu) / torch.sqrt(var + eps)
y = gamma * xhat + beta
w = torch.randn(m, d)                              # random weights: a loss with no special structure
loss = (w * y).sum() + (y ** 3).sum()              # so dL/dy = w + 3y^2, a generic upstream gradient
loss.backward()
gy = (w + 3 * y ** 2).detach()                     # the upstream gradient g_i, by hand
xh = xhat.detach(); scale = gamma.detach() / torch.sqrt(var.detach() + eps)
dx = scale * (gy - gy.mean(0) - xh * (gy * xh).mean(0))   # the compact formula
no_center = scale * (gy - xh * (gy * xh).mean(0))         # the same formula minus its centering term
print("dL/dx matches autograd below 1e-5:", bool(torch.allclose(dx, x.grad, atol=1e-5)))
print("without the centering term:", bool(torch.allclose(no_center, x.grad, atol=1e-5)))
print("dL/dgamma matches below 1e-5:", bool(torch.allclose((gy * xh).sum(0), gamma.grad, atol=1e-5)),
      " dL/dbeta matches below 1e-5:", bool(torch.allclose(gy.sum(0), beta.grad, atol=1e-5)))

# %% [markdown]
# ## 11.4.2 Why it helps

# %% [markdown]
# Example 11-9. An MLP with five hidden layers trained on the breast-cancer data by full-batch SGD with and without batch normalization, and the loss of predicting only the class shares.

# %%
import math, torch, torch.nn as nn
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
torch.manual_seed(0)
data = load_breast_cancer()                        # 569 samples, 30 features, a real tumor dataset
Xb = StandardScaler().fit_transform(data.data)     # scale each feature to zero mean, unit variance
Xt = torch.tensor(Xb, dtype=torch.float32); yt = torch.tensor(data.target, dtype=torch.long)
def make_net(use_bn):                              # a deepish MLP, optionally with BatchNorm
    layers, d = [], 30
    for _ in range(5):
        layers += [nn.Linear(d, 64)] + ([nn.BatchNorm1d(64)] if use_bn else []) + [nn.ReLU()]; d = 64
    return nn.Sequential(*layers, nn.Linear(64, 2))
def train(net, lr, epochs=60):
    opt = torch.optim.SGD(net.parameters(), lr=lr); lossf = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad(); loss = lossf(net(Xt), yt); loss.backward(); opt.step()
    return loss.item()
print(f"without BatchNorm: {train(make_net(False), 0.5):.4f}")
print(f"with    BatchNorm: {train(make_net(True), 0.5):.4f}")
print(f"without BatchNorm at lr 0.05: {train(make_net(False), 0.05):.4f}")   # a gentler rate does not rescue it
frac = yt.float().mean().item()                    # the share of benign tumors (class 1)
print(f"loss of predicting only the class shares: {-(frac * math.log(frac) + (1 - frac) * math.log(1 - frac)):.4f}")

# %% [markdown]
# ## 11.4.3 Inference, placement, and the normalization family

# %%
import numpy as np, torch, torch.nn as nn
x = np.array([[1.0, 2.0, 6.0], [3.0, 8.0, 4.0]])       # a batch of 2 samples, 3 features
bn = (x - x.mean(0)) / np.sqrt(x.var(0) + 1e-5)         # per FEATURE, across the batch (columns)
ln = (x - x.mean(1, keepdims=True)) / np.sqrt(x.var(1, keepdims=True) + 1e-5)   # per SAMPLE (rows)
rms = x / np.sqrt((x ** 2).mean(1, keepdims=True) + 1e-5)                       # rescale rows only, no centering
print("batch norm:", bn.round(3).tolist())
print("layer norm:", ln.round(3).tolist())
print("RMS norm  :", rms.round(3).tolist())
bn1 = nn.BatchNorm1d(3, momentum=None)                  # momentum=None: running stats are plain averages
xt = torch.tensor(x, dtype=torch.float32)
bn1.train(); out_tr = bn1(xt).detach()                  # TRAIN mode: normalize by THIS batch's statistics
bn1.eval();  out_ev = bn1(xt).detach()                  # EVAL mode: normalize by the stored running statistics
print("BatchNorm1d train:", out_tr.numpy().astype(float).round(3).tolist())
print("BatchNorm1d eval :", out_ev.numpy().astype(float).round(3).tolist())

# %% [markdown]
# ## 11.4.4 Group norm and instance norm

# %% [markdown]
# Example 11-10. `GroupNorm(1, 4)` checked against `LayerNorm` on eight digit images, then a small convolutional net trained at batch size 2 with `BatchNorm2d` and with `GroupNorm(4, 8)`.

# %%
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
torch.manual_seed(0)
Xd, yd = load_digits(return_X_y=True)
imgs = torch.tensor(Xd[:8].reshape(8, 1, 8, 8) / 16.0, dtype=torch.float32)
conv = nn.Conv2d(1, 4, 3, padding=1)
with torch.no_grad():                              # no graph: the norm layers have learnable weights
    h = conv(imgs)
    gvl = float((nn.GroupNorm(1, 4)(h) - nn.LayerNorm([4, 8, 8])(h)).abs().max())
    print("group-vs-layer below 1e-5:", gvl < 1e-5)
def conv_acc(norm):
    torch.manual_seed(0)
    X = Xd.reshape(-1, 1, 8, 8) / 16.0
    Xtr, Xte, ytr, yte = train_test_split(X, yd, test_size=0.25, random_state=0)
    Xva = torch.tensor(Xte, dtype=torch.float32); yva = torch.tensor(yte, dtype=torch.long)
    dl = DataLoader(TensorDataset(torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.long)),
                    batch_size=2, shuffle=True)
    net = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), norm, nn.ReLU(), nn.Flatten(), nn.Linear(8 * 8 * 8, 10))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3); lf = nn.CrossEntropyLoss()
    for _ in range(8):
        net.train()
        for xb, yb in dl:
            opt.zero_grad(); lf(net(xb), yb).backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return round(float((net(Xva).argmax(1) == yva).float().mean()), 3)
print("batch norm", conv_acc(nn.BatchNorm2d(8)), "group norm", conv_acc(nn.GroupNorm(4, 8)))

# %% [markdown]
# ## 11.4.5 What we built, and a self-check

# %% [markdown]
# ## 11.5 Regularization in deep learning

# %% [markdown]
# ## 11.5.1 Dropout

# %% [markdown]
# Example 11-11. A 64-128-128-10 network trained on 100 digits with no regularization, with dropout, and with Adam's weight decay, with train accuracy, test accuracy, and the gap for each.

# %%
import numpy as np, torch, torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
torch.manual_seed(0); rng = np.random.default_rng(0)
Xd, yd = load_digits(return_X_y=True)                            # 1797 8-by-8 handwritten digits, 10 classes
Xd = StandardScaler().fit_transform(Xd)
Xtr_np, Xte_np, ytr_np, yte_np = train_test_split(               # keep only 100 digits to train on
    Xd, yd, train_size=100, stratify=yd, random_state=0)         #   -> a big net memorizes them easily
Xtr = torch.tensor(Xtr_np, dtype=torch.float32); ytr = torch.tensor(ytr_np, dtype=torch.long)
Xte = torch.tensor(Xte_np, dtype=torch.float32); yte = torch.tensor(yte_np, dtype=torch.long)
def make_net(p_drop=0.0):
    return nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Dropout(p_drop),
                         nn.Linear(128, 128), nn.ReLU(), nn.Dropout(p_drop), nn.Linear(128, 10))
def train(net, weight_decay=0.0, epochs=400, seed=0, X=None, y=None):
    torch.manual_seed(seed); X = Xtr if X is None else X; y = ytr if y is None else y
    opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=weight_decay)
    lossf = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad(); lossf(net(X), y).backward(); opt.step()
    return net
def acc(net, X, y):
    net.eval()
    with torch.no_grad(): return (net(X).argmax(1) == y).float().mean().item()
print(f"{'model':18} train  test   gap")
for name, net, wd in [("no regularization", make_net(0.0), 0.0), ("dropout p=0.5", make_net(0.5), 0.0),
                      ("weight decay 1e-2", make_net(0.0), 1e-2)]:
    train(net, weight_decay=wd); tr, te = acc(net, Xtr, ytr), acc(net, Xte, yte)
    print(f"{name:18} {tr:.2f}  {te:.2f}  {tr - te:.2f}")   # two decimals: last-bit paths move the third
# no reg 1.00/0.86/0.14   dropout 1.00/0.89/0.11   weight decay 1.00/0.89/0.11

# %% [markdown]
# ## 11.5.2 Weight decay, early stopping, and data augmentation

# %% [markdown]
# Example 11-12. Early stopping with a patience of 30 epochs on the network of Example 11-11, restoring the best validation weights and comparing them with the full 400-epoch run.

# %%
Xva, Xho = Xte[:600], Xte[600:]; yva, yho = yte[:600], yte[600:]   # validation drives stopping; holdout reports
torch.manual_seed(0); net = make_net(0.0)
opt = torch.optim.Adam(net.parameters(), lr=0.01); lossf = nn.CrossEntropyLoss()
best_val, best_state, patience, since = float("inf"), None, 30, 0
for epoch in range(1, 401):                                        # same 400-epoch budget as 11.5.1
    net.train(); opt.zero_grad(); lossf(net(Xtr), ytr).backward(); opt.step()
    net.eval()
    with torch.no_grad(): val = lossf(net(Xva), yva).item()
    if val < best_val - 1e-4:                                      # improved: remember these weights
        best_val, best_state, since = val, {k: v.clone() for k, v in net.state_dict().items()}, 0
    else:
        since += 1
        if since >= patience:                                      # stuck for `patience` epochs: stop
            print(f"stopped at epoch {epoch}, best val loss {best_val:.3f}"); break
net.load_state_dict(best_state)                                    # restore the BEST weights, not the last
full = train(make_net(0.0))                                        # the same net trained all 400 epochs
print(f"early stopped: holdout acc = {acc(net, Xho, yho):.3f}   trained full 400: {acc(full, Xho, yho):.3f}")
full.eval()
with torch.no_grad(): vfull = lossf(full(Xva), yva).item()
print(f"val loss: early-stopped {best_val:.3f} vs full-400 {vfull:.3f}")

# %% [markdown]
# ## 11.5.3 Label smoothing

# %%
import torch, torch.nn as nn
torch.manual_seed(0)
logits = torch.randn(4, 10)                       # a batch of 4 examples, 10 classes
y = torch.tensor([3, 7, 0, 3])
print(f"plain CE    = {nn.CrossEntropyLoss()(logits, y).item():.4f}")
print(f"smoothed CE = {nn.CrossEntropyLoss(label_smoothing=0.1)(logits, y).item():.4f}")
soft = torch.full((4, 10), 0.1 / 10); soft[torch.arange(4), y] += 0.9   # the (1-eps)+eps/K soft targets
print(f"by hand     = {-(soft * logits.log_softmax(1)).sum(1).mean().item():.4f}")

# %% [markdown]
# ## 11.5.4 Cheap uncertainty: MC dropout and deep ensembles

# %% [markdown]
# Example 11-13. Monte Carlo dropout: 100 stochastic passes of a trained dropout network over the test digits, with the spread of the predicted-class probability on correct and incorrect predictions.

# %%
torch.manual_seed(0)                               # fix the starting weights: train seeds only after make_net
net = train(make_net(0.5))                         # train a dropout net
net.train()                                        # keep dropout ACTIVE at inference
with torch.no_grad():
    passes = torch.stack([net(Xte).softmax(1) for _ in range(100)])   # (100, N, 10): 100 stochastic passes
mean_p = passes.mean(0); pred = mean_p.argmax(1)   # class chosen by the averaged pass
rows = torch.arange(len(pred))
std_p = passes[:, rows, pred].std(0)               # spread of the predicted-class probability = uncertainty
correct = pred == yte
sc, si = float(std_p[correct].mean()), float(std_p[~correct].mean())
print(f"std on CORRECT smaller than on INCORRECT: {si > sc}")
print(f"incorrect std at least 2x correct: {si >= 2 * sc}")   # the teaching point; absolute stds move across machines

# %% [markdown]
# Example 11-14. A five-member deep ensemble trained on bootstrap resamples of the 100 digits, with its accuracy against its members' and its disagreement on correct and incorrect predictions.

# %%
nets = []
for s in range(5):
    idx = rng.integers(0, len(Xtr), len(Xtr))      # bootstrap resample for diversity
    nets.append(train(make_net(0.0), seed=s, X=Xtr[idx], y=ytr[idx]))
with torch.no_grad():
    probs = torch.stack([n(Xte).softmax(1) for n in nets])       # (5, N, 10)
ens_pred = probs.mean(0).argmax(1); correct = ens_pred == yte
rows = torch.arange(len(ens_pred))
disagree = probs[:, rows, ens_pred].std(0)                       # member spread on the chosen class
print(f"avg single-net acc = {np.mean([acc(n, Xte, yte) for n in nets]):.3f}")   # 0.811
print(f"ensemble acc       = {(ens_pred == yte).float().mean().item():.3f}")     # 0.863 (averaging helps)
print(f"disagree CORRECT   = {disagree[correct].mean():.3f}")    # 0.145
print(f"disagree INCORRECT = {disagree[~correct].mean():.3f}")   # 0.425  members disagree on the hard cases

# %% [markdown]
# ## 11.5.5 Temperature scaling

# %% [markdown]
# Example 11-15. Temperature scaling from scratch: an unregularized network trained on 100 digits, $T$ chosen on a validation split by negative log likelihood, and the holdout negative log likelihood, ECE, and accuracy before and after scaling.

# %%
import numpy as np, torch, torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
torch.manual_seed(0)
Xd, yd = load_digits(return_X_y=True)
Xs = StandardScaler().fit_transform(Xd)
Xtr, Xrest, ytr, yrest = train_test_split(Xs, yd, train_size=100, stratify=yd, random_state=0)
Xva, Xho, yva, yho = train_test_split(Xrest, yrest, test_size=0.5, stratify=yrest, random_state=0)
Xt = torch.tensor(Xtr, dtype=torch.float32); yt = torch.tensor(ytr, dtype=torch.long)
Xv = torch.tensor(Xva, dtype=torch.float32); yv = torch.tensor(yva, dtype=torch.long)
Xh = torch.tensor(Xho, dtype=torch.float32); yh = torch.tensor(yho, dtype=torch.long)
net = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10))
opt = torch.optim.Adam(net.parameters(), lr=0.01); lf = nn.CrossEntropyLoss()
for _ in range(200):
    opt.zero_grad(); lf(net(Xt), yt).backward(); opt.step()
net.eval()
def nll_ece(logits, y, T):
    prob = torch.softmax(logits / T, 1)
    nll = float(nn.functional.nll_loss(prob.log().clamp(min=-30), y))
    conf, pred = prob.max(1); correct = pred == y; ece = 0.0
    for b in range(10):
        lo, hi = b / 10, (b + 1) / 10
        m = (conf > lo) & (conf <= hi) if b else (conf >= 0) & (conf <= hi)
        if int(m.sum()) == 0: continue
        ece += float(m.sum()) / len(y) * abs(float(correct[m].float().mean()) - float(conf[m].mean()))
    return nll, ece, float(correct.float().mean())
with torch.no_grad():
    best_T, best_nll = 1.0, 1e9
    for T in np.linspace(0.5, 5.0, 46):
        nll, _, _ = nll_ece(net(Xv), yv, float(T))
        if nll < best_nll: best_T, best_nll = round(float(T), 2), nll
    before = nll_ece(net(Xh), yh, 1.0); after = nll_ece(net(Xh), yh, best_T)
print("T", best_T)
print("holdout T=1", [round(v, 4) for v in before])
print("holdout scaled", [round(v, 4) for v in after])

# %%
with torch.no_grad():
    conf, pred = torch.softmax(net(Xh), 1).max(1)      # T = 1: each holdout digit's top probability
hit = (pred == yh).float()                             # 1 where the prediction is right
for b in range(10):                                    # the ten bins (b/10, (b+1)/10]
    m = (conf > b / 10) & (conf <= (b + 1) / 10)
    if int(m.sum()) == 0: continue                     # an empty bin adds nothing
    n_b, acc_b, conf_b = int(m.sum()), float(hit[m].mean()), float(conf[m].mean())
    print(f"({b / 10:.1f}, {(b + 1) / 10:.1f}]  n={n_b:3d}  acc={acc_b:.4f}  conf={conf_b:.4f}"
          f"  term={n_b / len(yh) * abs(acc_b - conf_b):.4f}")

# %% [markdown]
# ## 11.5.6 Max-norm regularization

# %% [markdown]
# Example 11-16. Max-norm projection with `torch.renorm` after every Adam step, at radii 1, 2, 3, and 4, on the 100-digit split of Example 11-15.

# %%
import torch, torch.nn as nn
torch.manual_seed(0)
def maxnorm_run(r):
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 10))
    opt = torch.optim.Adam(net.parameters(), lr=0.01); lf = nn.CrossEntropyLoss()
    for _ in range(80):
        opt.zero_grad(); lf(net(Xt), yt).backward(); opt.step()
        with torch.no_grad():
            for mod in net:
                if isinstance(mod, nn.Linear):
                    mod.weight.copy_(torch.renorm(mod.weight, p=2, dim=0, maxnorm=r))
    net.eval()
    with torch.no_grad():
        tr = float((net(Xt).argmax(1) == yt).float().mean())
        te = float((net(Xh).argmax(1) == yh).float().mean())
    return round(tr, 3), round(te, 3)
for r in (1, 2, 3, 4):
    print(r, maxnorm_run(r))

# %% [markdown]
# ## 11.5.7 What we built, and a self-check

# %% [markdown]
# ## 11.6 Vanishing and exploding gradients

# %% [markdown]
# Example 11-17. Weight-gradient norms at the first and last hidden layers of an 18-layer stack under sigmoid, ReLU, ReLU with weights scaled by 3, and ReLU with He initialization.

# %%
import torch, torch.nn as nn
torch.manual_seed(0)
DEPTH, W = 18, 48
X = torch.randn(128, W); y = torch.randn(128, 1)
def build(activation, weight_gain=1.0):
    layers = []
    for _ in range(DEPTH): layers += [nn.Linear(W, W), activation()]
    net = nn.Sequential(*layers, nn.Linear(W, 1))
    if weight_gain != 1.0:                         # scale weights up to force an explosion
        for m in net.modules():
            if isinstance(m, nn.Linear):
                with torch.no_grad(): m.weight *= weight_gain
    return net
def layer_grad_norms(net):
    net.zero_grad(); nn.MSELoss()(net(X), y).backward()
    return [m.weight.grad.norm().item() for m in net.modules() if isinstance(m, nn.Linear)]
configs = {"sigmoid (vanishes)": build(nn.Sigmoid), "ReLU (far healthier)": build(nn.ReLU),
           "large weights (explodes)": build(nn.ReLU, weight_gain=3.0)}
he_net = build(nn.ReLU)                            # the ReLU stack again, reinitialized with He's 2/n_in
for m in he_net.modules():
    if isinstance(m, nn.Linear): nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
configs["ReLU + He init"] = he_net
print(f"{'config':26} layer 1     layer 18")
for name, net in configs.items():
    g = layer_grad_norms(net); print(f"{name:26} {g[0]:.2e}  {g[-2]:.2e}")

# %%
def act_grads(net):                                # walk the stack by hand, keeping each activation
    acts, a = [], X
    for layer in net:
        a = layer(a)
        if not isinstance(layer, nn.Linear): a.retain_grad(); acts.append(a)   # keep dL/da after backward
    net.zero_grad(); nn.MSELoss()(a, y).backward()
    return [t.grad.norm().item() for t in acts], [t.norm().item() for t in acts]
print(f"{'config':26} dL/da layer 1  layer 18   per layer")
for name, net in configs.items():
    gn, an = act_grads(net)
    print(f"{name:26} {gn[0]:.2e}      {gn[-1]:.2e}   {(gn[0] / gn[-1]) ** (1 / 17):.3f}")
gn, an = act_grads(configs["large weights (explodes)"])
print(f"exploding stack, activation norm: layer 1 {an[0]:.1f}, layer 18 {an[-1]:.1f}")

# %%
net = configs["large weights (explodes)"]
net.zero_grad(); nn.MSELoss()(net(X), y).backward()
before = torch.sqrt(sum(p.grad.pow(2).sum() for p in net.parameters())).item()
torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)   # rescale so the global norm <= 5
after = torch.sqrt(sum(p.grad.pow(2).sum() for p in net.parameters())).item()
print(f"grad norm before clipping = {before:.1f}")    # 5446.7
print(f"grad norm after  clipping = {after:.2f}")     # 5.00  capped at max_norm, the standard fix for explosions

# %% [markdown]
# ## 11.6.1 Clipping by value and by norm

# %%
import torch
g = torch.tensor([0.9, 100.0])
p = torch.nn.Parameter(torch.zeros(2))
p.grad = g.clone(); torch.nn.utils.clip_grad_value_(p, 1.0); by_value = p.grad.detach().clone()
p.grad = g.clone(); seen = torch.nn.utils.clip_grad_norm_(p, 1.0); by_norm = p.grad.detach().clone()
def cos(a, b):
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))
print("value", [round(float(v), 4) for v in by_value], "cos", round(cos(by_value, g), 4))
print("norm", [round(float(v), 4) for v in by_norm], "cos", round(cos(by_norm, g), 4), "seen", round(float(seen), 4))

# %% [markdown]
# ## 11.6.2 What we built, and a self-check

# %% [markdown]
# ## 11.7 Debugging training

# %% [markdown]
# Example 11-18. The learning-rate signatures on the digits: a 64-64-10 network trained by full-batch SGD for 200 epochs at rates of $10^{-5}$, 0.1, and $10^{4}$.

# %%
import numpy as np, torch, torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
torch.manual_seed(0)
Xd, yd = load_digits(return_X_y=True); Xd = StandardScaler().fit_transform(Xd)
Xtr, _, ytr, _ = train_test_split(Xd, yd, test_size=0.25, random_state=0)
Xtr = torch.tensor(Xtr, dtype=torch.float32); ytr = torch.tensor(ytr, dtype=torch.long)
def net():
    torch.manual_seed(0); return nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))
def run(lr, epochs=200):
    m = net(); opt = torch.optim.SGD(m.parameters(), lr=lr); lf = nn.CrossEntropyLoss()
    first = lf(m(Xtr), ytr).item()
    for _ in range(epochs):
        opt.zero_grad(); lf(m(Xtr), ytr).backward(); opt.step()
    return first, lf(m(Xtr), ytr).item()
print(f"{'lr':8} initial  final")
for lr in [1e-5, 0.1, 1e4]:
    f0, f1 = run(lr); print(f"{lr:<8} {f0:.3f}    {f1:.3f}")
# 1e-5: 2.331->2.330 STUCK (lr too low)   0.1: 2.331->0.140 healthy   1e4: 2.331->nan (lr too high, diverges)

# %% [markdown]
# Example 11-19. The overfitting signature: a 64-256-256-10 network trained with Adam on 40 digits, with the training and validation loss tracked over 600 epochs.

# %%
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
torch.manual_seed(1)
Xd, yd = load_digits(return_X_y=True); Xd = StandardScaler().fit_transform(Xd)
Xs_np, Xv_np, ys_np, yv_np = train_test_split(            # 40 digits to train, the rest to validate
    Xd, yd, train_size=40, stratify=yd, random_state=2)
Xs = torch.tensor(Xs_np, dtype=torch.float32); ys = torch.tensor(ys_np, dtype=torch.long)
Xv = torch.tensor(Xv_np, dtype=torch.float32); yv = torch.tensor(yv_np, dtype=torch.long)
m = nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 10))
opt = torch.optim.Adam(m.parameters(), lr=0.01); lf = nn.CrossEntropyLoss()
print("epoch  train   val")
best_val, best_ep = float("inf"), 0
for ep in range(1, 601):
    opt.zero_grad(); lf(m(Xs), ys).backward(); opt.step()
    with torch.no_grad():
        val = lf(m(Xv), yv).item()
        if val < best_val: best_val, best_ep = val, ep    # track where the validation loss bottoms out
        if ep in (1, 50, 200, 600):
            print(f"{ep:>4}   {lf(m(Xs), ys).item():.3f}   {val:.3f}")
print(f"lowest validation loss {best_val:.3f} at epoch {best_ep}")
# epoch 1: 1.591/1.867   50: 0.000/6.262   200: 0.000/6.346   600: 0.000/6.302   lowest val 0.702 at epoch 4

# %% [markdown]
# ## 11.7.1 Two sanity checks before you trust a run

# %%
import torch, torch.nn as nn, math
torch.manual_seed(0)
clf = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))   # a fresh 10-class head
X = torch.randn(200, 64); y = torch.randint(0, 10, (200,))
with torch.no_grad():                                                  # loss at init should be about ln(C)
    print(f"init loss = {nn.CrossEntropyLoss()(clf(X), y).item():.3f}   expected ln(10) = {math.log(10):.3f}")
drop = nn.Dropout(0.5); ones = torch.ones(1, 6)
drop.train(); print("train mode:", drop(ones).tolist())   # some entries zeroed, survivors scaled by 1/(1-p)=2
drop.eval();  print("eval  mode:", drop(ones).tolist())   # identity: every entry passes through unchanged

# %% [markdown]
# ## 11.7.2 Reproducible runs, and seed to seed variance

# %% [markdown]
# Example 11-20. Seeded runs of the 64-32-10 digit network: one seed run twice, then five seeds without and with batch normalization, with the mean and standard deviation of test accuracy.

# %%
import statistics
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
Xd, yd = load_digits(return_X_y=True); Xd = StandardScaler().fit_transform(Xd)
Xtr, Xte, ytr, yte = train_test_split(Xd, yd, test_size=0.25, random_state=0)   # the split of 10.7
Xtr_t, ytr_t = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.long)
Xte_t, yte_t = torch.tensor(Xte, dtype=torch.float32), torch.tensor(yte, dtype=torch.long)
def seeded_run(seed, batch_norm, epochs=8):
    torch.manual_seed(seed)                                    # the seed fixes the initial weights
    model = nn.Sequential(nn.Linear(64, 32), *([nn.BatchNorm1d(32)] if batch_norm else []),
                          nn.ReLU(), nn.Linear(32, 10))
    gen = torch.Generator().manual_seed(seed)                  # and the loader's generator fixes the shuffle
    loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=64, shuffle=True, generator=gen)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3); lossf = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(); lossf(model(xb), yb).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        return round((model(Xte_t).argmax(1) == yte_t).float().mean().item(), 4)
print("seed 0, run twice:", seeded_run(0, False), seeded_run(0, False))
plain = [seeded_run(s, False) for s in range(5)]
bn = [seeded_run(s, True) for s in range(5)]
for name, accs in [("plain", plain), ("batch norm", bn)]:
    print(f"{name}: {accs}, mean {statistics.mean(accs):.4f}, sd {statistics.stdev(accs):.4f}")
print(f"gap between the means {statistics.mean(bn) - statistics.mean(plain):.4f}")

# %% [markdown]
# ## 11.7.3 Reading a run back from TensorBoard

# %%
import tempfile
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
plain = [0.9133, 0.92, 0.8689, 0.9, 0.9156]
bn = [0.9444, 0.9311, 0.9178, 0.9356, 0.9333]
logdir = tempfile.mkdtemp()                       # a fresh folder: a rerun cannot add to old event files
writer = SummaryWriter(logdir)
for i, (a, b) in enumerate(zip(plain, bn)):
    writer.add_scalar("val_acc/plain", a, i); writer.add_scalar("val_acc/bn", b, i)
writer.flush(); writer.close()
acc = EventAccumulator(logdir); acc.Reload()
print([round(e.value, 4) for e in acc.Scalars("val_acc/plain")])
print([round(e.value, 4) for e in acc.Scalars("val_acc/bn")])

# %% [markdown]
# ## 11.7.4 What we built, and a self-check

# %% [markdown]
# ## 11.8 PyTorch and the framework landscape

# %% [markdown]
# ## 11.8.1 The framework landscape

# %% [markdown]
# ## 11.8.2 The core pieces

# %%
import torch
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2 + 2 * x
y.backward()
print(x.grad.item())   # autograd dy/dx = 8.0   (analytic 2*3 + 2 = 8)

# %% [markdown]
# Example 11-21. The canonical PyTorch workflow: a `DataLoader`, a `DigitNet` module, and the four-step loop, trained for 20 epochs with Adam on the digits and scored on the test split.

# %%
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
torch.manual_seed(0)
Xd, yd = load_digits(return_X_y=True)
Xd = StandardScaler().fit_transform(Xd)
Xtr, Xte, ytr, yte = train_test_split(Xd, yd, test_size=0.25, random_state=0)
to_t = lambda a, dt: torch.tensor(a, dtype=dt)
train_dl = DataLoader(TensorDataset(to_t(Xtr, torch.float32), to_t(ytr, torch.long)),
                      batch_size=64, shuffle=True)             # yields shuffled minibatches
Xte_t, yte_t = to_t(Xte, torch.float32), to_t(yte, torch.long)
class DigitNet(nn.Module):                                    # a model is an nn.Module with forward()
    def __init__(self):
        super().__init__(); self.net = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
    def forward(self, x): return self.net(x)
model = DigitNet()
opt = torch.optim.Adam(model.parameters(), lr=1e-3); lossf = nn.CrossEntropyLoss()
for epoch in range(20):
    model.train()
    for xb, yb in train_dl:                                   # the canonical four-step loop:
        opt.zero_grad()                                       # 1. clear old gradients
        loss = lossf(model(xb), yb)                           # 2. forward -> loss
        loss.backward()                                       # 3. autograd backprop
        opt.step()                                            # 4. optimizer update
model.eval()
with torch.no_grad():
    print(f"test accuracy = {(model(Xte_t).argmax(1) == yte_t).float().mean().item():.3f}")   # 0.956

# %%
import torch
x = torch.tensor(3.0, requires_grad=True)
tracked = x ** 2                       # graph recorded: this tensor can be backpropagated
with torch.no_grad():
    untracked = x ** 2                 # no graph built: cheaper, for inference only
print(tracked.requires_grad, untracked.requires_grad)   # True False

# %% [markdown]
# ## 11.8.3 GPUs, saving, and the ecosystem

# %%
torch.save(model.state_dict(), "digitnet.pt")                 # state_dict = the learned weights
reloaded = DigitNet(); reloaded.load_state_dict(torch.load("digitnet.pt")); reloaded.eval()
with torch.no_grad():
    same = torch.equal(model(Xte_t).argmax(1), reloaded(Xte_t).argmax(1))
print("reloaded model gives identical predictions:", same)    # True

# %% [markdown]
# Example 11-22. `torch.profiler` over ten full-batch training steps of a 64-1024-1024-10 digit network, printing the two entries with the largest self CPU time and how many times each was called.

# %%
import torch, torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
torch.manual_seed(0)
Xd, yd = load_digits(return_X_y=True)
Xd = StandardScaler().fit_transform(Xd)
Xtr, _, ytr, _ = train_test_split(Xd, yd, test_size=0.25, random_state=0)
net = nn.Sequential(nn.Linear(64, 1024), nn.ReLU(), nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, 10))
opt = torch.optim.Adam(net.parameters(), lr=1e-3); lf = nn.CrossEntropyLoss()
xb = torch.tensor(Xtr, dtype=torch.float32); yb = torch.tensor(ytr, dtype=torch.long)   # all 1,347 training rows
import os
_err_fd = os.open(os.devnull, os.O_WRONLY); _saved_err = os.dup(2); os.dup2(_err_fd, 2)  # hush Mac USDT profiler lines on fd 2
try:
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        for _ in range(10):
            with record_function("step"):
                opt.zero_grad(); lf(net(xb), yb).backward(); opt.step()
finally:
    os.dup2(_saved_err, 2); os.close(_saved_err); os.close(_err_fd)
top = sorted(prof.key_averages(), key=lambda e: e.self_cpu_time_total, reverse=True)[:2]
print([(e.key, e.count) for e in top])        # [('aten::mm', 50), ('aten::addmm', 30)]

# %% [markdown]
# ## 11.8.4 What an nn.Module actually tracks

# %%
import torch, torch.nn as nn
from torch.nn import functional as F
print("Linear(64, 128) parameters:", sum(p.numel() for p in nn.Linear(64, 128).parameters()))
class Plain(nn.Module):
    def __init__(self):
        super().__init__(); self.w = torch.randn(4, 3)
class Listed(nn.Module):
    def __init__(self):
        super().__init__(); self.layers = nn.ModuleList([nn.Linear(3, 3), nn.Linear(3, 3)])
print("plain tensor parameters:", len(list(Plain().parameters())))
print("ModuleList parameters:", sum(p.numel() for p in Listed().parameters()))
torch.manual_seed(0)
y = F.dropout(torch.ones(20), 0.5)
print("zeros", int((y == 0).sum()), "survivor", float(y[y != 0][0]))

# %% [markdown]
# ## 11.8.5 Autograd: accumulation, detach, and a second derivative

# %%
import torch
x = torch.tensor(3.0, requires_grad=True)
(x * x.detach()).backward(); print("detach grad", float(x.grad))
x2 = torch.tensor(3.0, requires_grad=True)
g1 = torch.autograd.grad(x2 ** 3, x2, create_graph=True)[0]
g2 = torch.autograd.grad(g1, x2)[0]
print("d1", float(g1.detach()), "d2", float(g2))
x3 = torch.tensor(2.0, requires_grad=True)
(x3 ** 2).backward(); (x3 ** 2).backward(); print("accumulated", float(x3.grad))

# %% [markdown]
# ## 11.8.6 Resuming a run from a checkpoint

# %% [markdown]
# Example 11-23. A checkpoint of the model, optimizer, scheduler, and generator state written after 5 of 10 epochs and resumed three ways, each checked against the uninterrupted run.

# %%
import io
train_ds = TensorDataset(to_t(Xtr, torch.float32), to_t(ytr, torch.long))
lossf = nn.CrossEntropyLoss()
def fresh_run():                                    # what a new process builds before it loads anything
    torch.manual_seed(0); model = DigitNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)   # cosine over the 10 epochs
    gen = torch.Generator().manual_seed(0)          # the loader's own shuffle generator
    loader = DataLoader(train_ds, batch_size=64, shuffle=True, generator=gen)
    return model, opt, sched, gen, loader
def train_epochs(model, opt, sched, loader, epochs):
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(); lossf(model(xb), yb).backward(); opt.step()
        sched.step()
def flat(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])
model, opt, sched, gen, loader = fresh_run()
train_epochs(model, opt, sched, loader, 10); straight = flat(model)   # ten epochs in one go
model, opt, sched, gen, loader = fresh_run()
train_epochs(model, opt, sched, loader, 5)                            # five epochs, then a checkpoint
print("parameters", sum(p.numel() for p in model.parameters()),
      "Adam moments", sum(v.numel() for st in opt.state.values() for k, v in st.items() if k != "step"))
buf = io.BytesIO()                                  # an in-memory file
torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
            "sched": sched.state_dict(), "gen": gen.get_state()}, buf)
def resume(parts):                                  # rebuild everything, then load only the named parts
    buf.seek(0); ckpt = torch.load(buf)
    model, opt, sched, gen, loader = fresh_run()
    model.load_state_dict(ckpt["model"])
    if "opt" in parts: opt.load_state_dict(ckpt["opt"])
    if "sched" in parts: sched.load_state_dict(ckpt["sched"])
    if "gen" in parts: gen.set_state(ckpt["gen"])
    train_epochs(model, opt, sched, loader, 5)      # the last five epochs
    return flat(model)
for label, parts in [("model, optimizer, scheduler, generator", {"opt", "sched", "gen"}),
                     ("model, optimizer, scheduler", {"opt", "sched"}),
                     ("model only", set())]:
    w = resume(parts)
    print(f"{label}: allclose {torch.allclose(w, straight)}, max|diff| {(w - straight).abs().max().item():.4g}")

# %% [markdown]
# ## 11.8.7 Freezing a registered parameter

# %% [markdown]
# Example 11-24. Five epochs of Adam on `DigitNet` with its first `nn.Linear` frozen and with nothing frozen, counting trainable parameters and checking that the frozen weight did not move.

# %%
def five_epochs(freeze_first):
    torch.manual_seed(0); model = DigitNet()
    first = model.net[0]                            # the first nn.Linear(64, 32)
    if freeze_first:
        first.requires_grad_(False)                 # its weight and bias stop requiring grad
    w0 = first.weight.detach().clone()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    loader = DataLoader(train_ds, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(0))
    for _ in range(5):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(); lossf(model(xb), yb).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        acc = (model(Xte_t).argmax(1) == yte_t).float().mean().item()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, torch.equal(first.weight, w0), round(acc, 3)
for freeze in (True, False):
    trainable, unchanged, acc = five_epochs(freeze)
    print(f"frozen={freeze}: trainable {trainable}, weight unchanged {unchanged}, test accuracy {acc}")

# %% [markdown]
# ## 11.8.8 A map-style Dataset

# %% [markdown]
# Example 11-25. Sixty-four digit rows written to `.npy` files and read back through a map-style `Dataset` and a `DataLoader` with `num_workers=0`.

# %%
import tempfile
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader, Dataset
class DigitFiles(Dataset):
    def __init__(self, folder):
        self.folder = Path(folder)
        self.y = [int(s) for s in (self.folder / "labels.csv").read_text(encoding="utf-8").split()]
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return torch.tensor(np.load(self.folder / f"{i}.npy")), self.y[i]
folder = Path(tempfile.mkdtemp())
for i in range(64):
    np.save(folder / f"{i}.npy", Xtr[i].astype(np.float32))
(folder / "labels.csv").write_text("\n".join(str(int(v)) for v in ytr[:64]), encoding="utf-8")
ds = DigitFiles(folder)
loader = DataLoader(ds, batch_size=16, num_workers=0)
print(len(ds), len(loader), sum(len(yb) for _, yb in loader))

# %% [markdown]
# ## 11.8.9 What we built, and a self-check

# %% [markdown]
# ## 11.9 Scaling up training

# %% [markdown]
# ## 11.9.1 Gradient accumulation

# %% [markdown]
# Example 11-26. Gradient accumulation over four microbatches of 16 on random inputs and labels, checked against the gradient of one full batch of 64.

# %%
import torch, torch.nn as nn
torch.manual_seed(0)
X = torch.randn(64, 10); y = torch.randint(0, 3, (64,))   # random stand-in inputs and labels
def fresh():
    torch.manual_seed(1); return nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 3))
lossf = nn.CrossEntropyLoss()
m1 = fresh(); m1.zero_grad(); lossf(m1(X), y).backward()          # one full batch of 64
g_full = torch.cat([p.grad.flatten() for p in m1.parameters()])
m2 = fresh(); m2.zero_grad()
for i in range(4):                                                # 4 microbatches of 16, accumulated
    xb, yb = X[i*16:(i+1)*16], y[i*16:(i+1)*16]
    (lossf(m2(xb), yb) / 4).backward()                            # /4 is the 1/K of the derivation
g_accum = torch.cat([p.grad.flatten() for p in m2.parameters()])
print("max |full - accumulated| below 1e-6:", bool((g_full - g_accum).abs().max() < 1e-6))

# %% [markdown]
# ## 11.9.2 Mixed precision

# %% [markdown]
# Example 11-27. `torch.finfo` for fp32, fp16, and bfloat16, the rounding and underflow checks, and a CPU `GradScaler` built with `enabled=True`.

# %%
import torch
for name, dt, layout in [
    ("fp32", torch.float32, "1 sign, 8 exponent, 23 mantissa"),
    ("fp16", torch.float16, "1 sign, 5 exponent, 10 mantissa"),
    ("bf16", torch.bfloat16, "1 sign, 8 exponent, 7 mantissa")]:
    fi = torch.finfo(dt)
    print(f"{name:4} {layout:36} eps {fi.eps:.3e}  smallest normal {fi.smallest_normal:.3e}  max {fi.max:.3e}")
a = torch.tensor(1.0, dtype=torch.float16)
print("fp16  1+0.0003", float(a + torch.tensor(0.0003, dtype=torch.float16)),
      " 1-0.0003", float(a - torch.tensor(0.0003, dtype=torch.float16)))
b = torch.tensor(1.0, dtype=torch.bfloat16)
print("bf16  1+0.0003", float(b + torch.tensor(0.0003, dtype=torch.bfloat16)))
print("fp16  1e-8", float(torch.tensor(1e-8, dtype=torch.float16)),
      " scaled", float(torch.tensor(1e-8 * 65536, dtype=torch.float32).to(torch.float16)))
print("bf16  1e-8", float(torch.tensor(1e-8, dtype=torch.bfloat16)))
sc = torch.amp.GradScaler("cpu", enabled=True)
print("cpu GradScaler enabled", bool(sc.is_enabled()), "scale", sc.get_scale())

# %% [markdown]
# Example 11-28. A three-step mixed-precision loop with `torch.autocast` and `torch.amp.GradScaler`, the scaler enabled only on CUDA.

# %%
import torch, torch.nn as nn
torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 4)).to(device)
opt = torch.optim.Adam(net.parameters(), lr=1e-3); lossf = nn.CrossEntropyLoss()
X = torch.randn(128, 20, device=device); y = torch.randint(0, 4, (128,), device=device)  # random stand-ins
use_amp = (device == "cuda")                          # this block turns the scaler on only for CUDA
scaler = torch.amp.GradScaler(device, enabled=use_amp)
for step in range(3):
    opt.zero_grad()
    with torch.autocast(device_type=device, dtype=torch.float16):
        loss = lossf(net(X), y)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    print(f"amp step {step}: loss = {loss.item():.4f}, grad-scale = {scaler.get_scale()}")
print(f"device = {device}, GradScaler enabled = {use_amp}")

# %% [markdown]
# ## 11.9.3 Many GPUs, and GPUs versus TPUs

# %% [markdown]
# Example 11-29. Distributed Data Parallel, illustrative: one process per GPU, identical updates.

# %%
# Illustrative only. Not run here: it needs several GPUs. Launch with
#   torchrun --nproc_per_node=4 train_ddp.py
import torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
dist.init_process_group("nccl")                        # one process per GPU
rank = dist.get_rank()
torch.cuda.set_device(rank)
model = DDP(build_model().to(rank), device_ids=[rank]) # build_model stands for DigitNet (11.8)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)    # after the wrap, not before
sampler = DistributedSampler(train_dataset)            # train_dataset is 11.8's training set
loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)
for epoch in range(num_epochs):                        # num_epochs is the epoch count
    sampler.set_epoch(epoch)
    for xb, yb in loader:
        opt.zero_grad()
        loss = loss_fn(model(xb.to(rank)), yb.to(rank))  # loss_fn is CrossEntropyLoss
        loss.backward()                                # backward also all-reduces the gradients
        opt.step()
dist.destroy_process_group()

# %%
from torch.distributed.fsdp import fully_shard
for block in model.blocks:     # shard the blocks first
    fully_shard(block)
fully_shard(model)             # then the root; illustrative, like Example 11-29

# %% [markdown]
# Example 11-30. Distributed Data Parallel and fully_shard run on the CPU: two processes over gloo, checked against one process.

# %%
# Example 11-30 is the companion script 12_ddp_gloo_cpu.py. It runs as a plain script, not in this
# notebook, because torch.multiprocessing.spawn starts fresh processes that import their worker function
# from a file. This cell runs it and prints what it printed.
import subprocess, sys  # run the companion script as its own process
run = subprocess.run([sys.executable, "-X", "utf8", "12_ddp_gloo_cpu.py"],
                     capture_output=True, text=True, encoding="utf-8")
print(run.stdout, end="")
assert run.returncode == 0, run.stderr[-2000:]

# %%
import torch
def bubble(p, m):
    return (p - 1) / (m + p - 1)
print([round(bubble(4, m), 4) for m in (1, 4, 16)])
torch.manual_seed(0)
A = torch.randn(6, 8); B = torch.randn(8, 10)
parts = torch.cat([A @ B[:, :5], A @ B[:, 5:]], dim=1)
print("split below 1e-6:", float((parts - A @ B).abs().max()) < 1e-6)
def delayed(delay, steps=80, lr=0.05):
    import numpy as np
    p = np.array([-4.0, -4.0]); hist = []
    for t in range(steps):
        g = np.array([10 * p[0], p[1]]); hist.append(g)
        use = hist[t - delay] if t >= delay else hist[0]
        p = p - lr * use
    return round(float(0.5 * (10 * p[0] ** 2 + p[1] ** 2)), 4)
print("stale", [delayed(d) for d in (0, 2, 8)])

# %% [markdown]
# ## 11.9.4 Gradient checkpointing, torch.compile, and large batch training

# %% [markdown]
# Example 11-31. Gradient checkpointing on one small block: outputs and input gradients checked against the plain block, and the bytes the autograd graph saves counted with a hook.

# %%
import torch, torch.nn as nn
from torch.utils.checkpoint import checkpoint
torch.manual_seed(0)
block = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 16))
x = torch.randn(8, 16, requires_grad=True)          # random stand-in input
y_plain = block(x); g_plain = torch.autograd.grad(y_plain.sum(), x, retain_graph=True)[0]
y_ckpt = checkpoint(block, x, use_reentrant=False)
g_ckpt = torch.autograd.grad(y_ckpt.sum(), x)[0]
print("outputs identical:", torch.allclose(y_plain, y_ckpt))
print("gradients identical:", torch.allclose(g_plain, g_ckpt), "max|diff| =", (g_plain - g_ckpt).abs().max().item())

def saved_bytes(fn):
    total = []
    def pack(t):
        total.append(t.numel() * t.element_size()); return t
    def unpack(t):
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        fn()
    return sum(total)

def plain():
    y = block(x); y.sum().backward()
def ckpt():
    y = checkpoint(block, x, use_reentrant=False); y.sum().backward()
for p in list(block.parameters()) + [x]:
    p.grad = None
print("bytes saved, plain:", saved_bytes(plain))
for p in list(block.parameters()) + [x]:
    p.grad = None
print("bytes saved, checkpoint:", saved_bytes(ckpt))

# %%
torch.manual_seed(1)
calls = {"n": 0}
class CountBlock(nn.Module):
    def __init__(self):
        super().__init__(); self.lin1 = nn.Linear(16, 16); self.lin2 = nn.Linear(16, 16)
    def forward(self, z):
        calls["n"] += 1; return self.lin2(torch.relu(self.lin1(z)))
cb, z = CountBlock(), torch.randn(4, 16, requires_grad=True)
y1 = cb(z); g1 = torch.autograd.grad(y1.sum(), z)[0]; plain_calls = calls["n"]
calls["n"] = 0
y2 = checkpoint(cb, z, use_reentrant=False)
g2 = torch.autograd.grad(y2.sum(), z)[0]
print("forward calls, plain then checkpoint:", plain_calls, calls["n"],
      "grads match:", bool(torch.allclose(g1, g2)))

# %%
import time
torch.manual_seed(0)
model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))
xx = torch.randn(8, 16)
with torch.no_grad():
    y_ref = model(xx).clone()
compiled = torch.compile(model, backend="aot_eager")   # no C++ toolchain; Inductor is the one that needs one
t0 = time.perf_counter()
with torch.no_grad():
    y1 = compiled(xx)
first = time.perf_counter() - t0
t0 = time.perf_counter()
with torch.no_grad():
    y2 = compiled(xx)
second = time.perf_counter() - t0
print("matches", bool(torch.allclose(y_ref, y1)), "max|diff| below 1e-6:", float((y_ref - y1).abs().max()) < 1e-6)
print("first call slower than second:", first > second)   # absolute times are one run; the gap is the point

# %% [markdown]
# ## 11.9.5 What we built, and a self-check

# %% [markdown]
# ## 11.10 AutoML and neural architecture search

# %% [markdown]
# Example 11-32. A 25-trial Optuna search with `TPESampler` over the depth, widths, and learning rate of an `MLPClassifier`, scored by three-fold cross-validation against a default network.

# %%
import warnings
warnings.filterwarnings("ignore", message="IProgress not found")   # tqdm, which optuna imports, warns in Jupyter without ipywidgets
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
X, y = load_digits(return_X_y=True)
default = MLPClassifier(hidden_layer_sizes=(50,), learning_rate_init=0.001, max_iter=300, random_state=0)
default_score = cross_val_score(default, X, y, cv=3).mean()
print(f"default (one 50-unit layer): CV accuracy = {default_score:.4f}")   # 0.9321
def objective(trial):                              # each trial proposes a depth, widths, and lr
    n_layers = trial.suggest_int("n_layers", 1, 3)
    widths = tuple(trial.suggest_int(f"w{i}", 16, 128) for i in range(n_layers))
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    return cross_val_score(MLPClassifier(hidden_layer_sizes=widths, learning_rate_init=lr,
                           max_iter=300, random_state=0), X, y, cv=3).mean()
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
study.optimize(objective, n_trials=25)
print("best beats default:", study.best_value > default_score)
print("improvement over default at least 0.01:", (study.best_value - default_score) >= 0.01)
# exact best trial, config, and trial list are one run on the reference machine (PLAN 13 (d)); see the prose below

# %%
from sklearn.model_selection import train_test_split
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
p = study.best_params
widths = tuple(p[f"w{i}"] for i in range(p["n_layers"]))
hold = {}
for name, sizes, lr in [("default", (50,), 0.001), ("chosen", widths, p["lr"])]:
    clf = MLPClassifier(hidden_layer_sizes=sizes, learning_rate_init=lr, max_iter=300, random_state=0)
    hold[name] = clf.fit(Xtr, ytr).score(Xte, yte)
    print(f"{name}: holdout accuracy {hold[name]:.2f}")   # two decimals; error counts move with the chosen trial
print("chosen holdout at least default:", hold["chosen"] >= hold["default"])

# %% [markdown]
# ## 11.10.1 What we built, and a self-check

# %% [markdown]
# ## 11.11 Hands-on lab: training MLPs with full experiments

# %% [markdown]
# ## 11.11.1 Part A: tabular data

# %% [markdown]
# Example 11-33. The lab's Part A: a tabular MLP on the breast-cancer data, scaler fit on the training split only.

# %%
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
torch.manual_seed(0)
Xc, yc = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(Xc, yc, test_size=0.25, random_state=0, stratify=yc)
Xtr, Xva, ytr, yva = train_test_split(Xtr, ytr, test_size=0.25, random_state=0, stratify=ytr)
scaler = StandardScaler().fit(Xtr)                 # training rows only; then transform the other splits
Xtr, Xva, Xte = scaler.transform(Xtr), scaler.transform(Xva), scaler.transform(Xte)
f = lambda a, dt: torch.tensor(a, dtype=dt)
dl = DataLoader(TensorDataset(f(Xtr, torch.float32), f(ytr, torch.long)), batch_size=64, shuffle=True)
Xtr_t, ytr_t = f(Xtr, torch.float32), f(ytr, torch.long)
Xva_t, yva_t = f(Xva, torch.float32), f(yva, torch.long)
Xte_t, yte_t = f(Xte, torch.float32), f(yte, torch.long)
net = nn.Sequential(nn.Linear(30, 32), nn.ReLU(), nn.Linear(32, 2))
opt = torch.optim.Adam(net.parameters(), lr=1e-3); lossf = nn.CrossEntropyLoss()
for ep in range(40):
    net.train()
    for xb, yb in dl:
        opt.zero_grad(); lossf(net(xb), yb).backward(); opt.step()
    net.eval()
    with torch.no_grad():
        if (ep + 1) in (1, 10, 20, 40):
            tr = lossf(net(Xtr_t), ytr_t).item(); va = lossf(net(Xva_t), yva_t).item()
            print(f"epoch {ep+1:2d}: train loss = {tr:.3f}, val loss = {va:.3f}")
net.eval()
with torch.no_grad():
    acc = (net(Xte_t).argmax(1) == yte_t).float().mean().item()
    auc = roc_auc_score(yte, net(Xte_t).softmax(1)[:, 1].numpy())
print(f"Part A tabular MLP: test accuracy = {acc:.3f}, ROC AUC = {auc:.3f}")

# %% [markdown]
# ## 11.11.2 Part B: an ablation on image data

# %% [markdown]
# Example 11-34. The lab's Part B: an ablation on the digit images, scaler fit on the training split, each optimizer at a rate chosen for it.

# %%
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
torch.manual_seed(0); lossf = nn.CrossEntropyLoss()
f = lambda a, dt: torch.tensor(a, dtype=dt)
Xd, yd = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(Xd, yd, test_size=0.25, random_state=0, stratify=yd)
Xtr, Xva, ytr, yva = train_test_split(Xtr, ytr, test_size=0.25, random_state=0, stratify=ytr)
scaler = StandardScaler().fit(Xtr)
Xtr, Xva, Xte = scaler.transform(Xtr), scaler.transform(Xva), scaler.transform(Xte)
dltr = DataLoader(TensorDataset(f(Xtr, torch.float32), f(ytr, torch.long)), batch_size=64, shuffle=True)
Xva_t, yva_t = f(Xva, torch.float32), f(yva, torch.long)
Xte_t, yte_t = f(Xte, torch.float32), f(yte, torch.long)
def build(use_bn, use_drop):
    L = [nn.Linear(64, 64), *([nn.BatchNorm1d(64)] if use_bn else []), nn.ReLU(),
         *([nn.Dropout(0.3)] if use_drop else []),
         nn.Linear(64, 64), *([nn.BatchNorm1d(64)] if use_bn else []), nn.ReLU(),
         *([nn.Dropout(0.3)] if use_drop else []),
         nn.Linear(64, 10)]
    return nn.Sequential(*L)
def train_model(model, opt):
    for _ in range(40):
        model.train()
        for xb, yb in dltr:
            opt.zero_grad(); lossf(model(xb), yb).backward(); opt.step()
    return model.eval()
def val_of(model):
    with torch.no_grad():
        hit = int((model(Xva_t).argmax(1) == yva_t).sum())
    return hit / len(yva_t), hit
for lr in (0.01, 0.03, 0.1):
    torch.manual_seed(0); m = build(False, False)
    train_model(m, torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9))
    acc_lr, hit = val_of(m); print(f"SGD momentum lr={lr:<4} val acc = {acc_lr:.3f}")
torch.manual_seed(0); m = build(False, False)
train_model(m, torch.optim.SGD(m.parameters(), lr=1e-3, momentum=0.9))
acc_bad, _ = val_of(m); print(f"SGD momentum lr=0.001 val acc = {acc_bad:.3f}")
configs = [("SGD with momentum, rate 0.1", False, False, "sgd"),
           ("Adam", False, False, "adam"),
           ("Adam with batch norm", True, False, "adam"),
           ("Adam with dropout", False, True, "adam"),
           ("Adam with batch norm and dropout", True, True, "adam")]
results = []
for name, bn, drop, opt_name in configs:
    torch.manual_seed(0); m = build(bn, drop)
    opt = (torch.optim.Adam(m.parameters(), lr=1e-3) if opt_name == "adam"
           else torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9))
    train_model(m, opt); acc_v, hit = val_of(m)
    results.append((name, acc_v, m)); print(f"{name:36} val acc = {acc_v:.3f}  ({hit}/{len(yva_t)})")
best_i = max(range(len(results)), key=lambda i: (results[i][1], -i))
best_name, best_va, best_model = results[best_i]
with torch.no_grad():
    test_acc = (best_model(Xte_t).argmax(1) == yte_t).float().mean().item()
print(f"selected '{best_name}' (val {best_va:.3f}) -> test accuracy = {test_acc:.3f}")

# %% [markdown]
# ## 11.11.3 What the lab showed

# %% [markdown]
# ## 11.11.4 What we built, and a self-check

# %% [markdown]
# ## 11.12 Interview questions and coding exercises

# %% [markdown]
# ## Optimization and learning rate

# %% [markdown]
# ## Initialization and normalization

# %% [markdown]
# ## Regularization and gradients

# %% [markdown]
# ## Practice and debugging

# %% [markdown]
# ## Coding exercises

# %%
import numpy as np
start = np.array([-4.0, -4.0])
def race():
    ...
print(start)

# %%
import numpy as np
T = 1000
def schedules(base=0.1):
    ...
print(T)

# %%
import numpy as np
N = 256
def propagate(scale, act):
    ...
print(N)

# %%
import numpy as np
x = np.array([[1.0, 2.0], [3.0, 4.0]])
def batch_norm(x):
    ...
print(x.shape)

# %%
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
Xd, yd = load_digits(return_X_y=True)
Xd = StandardScaler().fit_transform(Xd)          # a whole-table fit, as in 11.5.1 (11.11.1 explains the leak)
Xtr, Xte, ytr, yte = train_test_split(Xd, yd, train_size=100, stratify=yd, random_state=0)
def train_configs():
    ...
print(Xtr.shape, Xte.shape)

# %%
import torch
DEPTH = 18
def stack():
    ...
print(DEPTH)

# %%
import math
from sklearn.datasets import load_digits
Xd, yd = load_digits(return_X_y=True)
def checks():
    ...
print(round(math.log(10), 3), Xd.shape)

# %%
import torch
X = torch.randn(58, 10)
def accumulate():
    ...
print(tuple(X.shape))

# %%
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
X, y = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
def search():
    ...
print(len(ytr), len(yte))

# %%
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
def fit_scaler_on_train():
    ...
print(X.shape, len(ytr), len(yte))

# %%
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
Xd, yd = load_digits(return_X_y=True)
rows = StandardScaler().fit_transform(Xd[:32])
def batchnorm_backward(g, x, gamma, eps=1e-5):
    ...
print(rows.shape)

# %%
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
Xd, yd = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(Xd, yd, train_size=100, stratify=yd, random_state=0)
def uncertainty():
    ...
print(len(ytr), len(yte))

# %%
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
Xd, yd = load_digits(return_X_y=True)
Xd = StandardScaler().fit_transform(Xd)
def overfit_one_batch():
    ...
print(Xd[:32].shape)

# %%
def checkpoint_demo():
    ...
print(6)

# %%
from sklearn.datasets import load_digits
X, y = load_digits(return_X_y=True)
def batch_sweep():
    ...
print(X.shape)

# %% [markdown]
# ## Coding drills
