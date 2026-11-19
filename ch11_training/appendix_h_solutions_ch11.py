"""Appendix H solutions for Chapter 11, as runnable cells.

Generated from Solutions.md (Appendix H). Every code block of this chapter's solutions appears here
verbatim, split into one cell per exercise or drill topic, so each answer can be run and checked against
the expected outputs it states. Keep Solutions.md as the source; regenerate after any change.
"""

# %% [markdown]
# ## 11.12 coding exercises

# %%
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn, math

# %% [markdown]
# 1. Optimizer race from scratch on the ill-conditioned quadratic

# %%
# 1. Optimizer race from scratch on the ill-conditioned quadratic
def f(p): return 0.5 * (10 * p[0] ** 2 + p[1] ** 2)   # 10:1 curvature ravine
def grad(p): return np.array([10 * p[0], p[1]])
start = np.array([-4.0, -4.0])
def run(update, steps=300, tol=1e-4):                  # (first step below tol, step it stays below from)
    p, state, losses = start.copy(), {}, [f(start)]
    for tt in range(1, steps + 1):
        p = update(p, grad(p), tt, state); losses.append(f(p))
    below = np.array(losses) < tol
    first = int(np.argmax(below)) if below.any() else None
    last_above = int(np.flatnonzero(~below)[-1])       # the last step not yet below tol
    return first, (last_above + 1 if last_above < steps else None)   # None: never settles
def sgd(p, g, t, s, lr=0.05): return p - lr * g
def momentum(p, g, t, s, lr=0.05, beta=0.9):
    s["v"] = beta * s.get("v", 0) + g; return p - lr * s["v"]
def nesterov(p, g, t, s, lr=0.05, beta=0.9):
    s["v"] = beta * s.get("v", 0) + g; return p - lr * (g + beta * s["v"])
def adagrad(p, g, t, s, lr=0.5, eps=1e-8):
    s["G"] = s.get("G", 0) + g ** 2
    return p - lr * g / (np.sqrt(s["G"]) + eps)
def rmsprop(p, g, t, s, lr=0.1, rho=0.9, eps=1e-8):
    s["E"] = rho * s.get("E", 0) + (1 - rho) * g ** 2   # decaying avg of squared grads
    return p - lr * g / (np.sqrt(s["E"]) + eps)
def adam(p, g, t, s, lr=0.3, b1=0.9, b2=0.999, eps=1e-8):
    s["m"] = b1 * s.get("m", 0) + (1 - b1) * g
    s["v"] = b2 * s.get("v", 0) + (1 - b2) * g ** 2
    mhat, vhat = s["m"] / (1 - b1 ** t), s["v"] / (1 - b2 ** t)   # bias correction
    return p - lr * mhat / (np.sqrt(vhat) + eps)
for name, upd in [("SGD", sgd), ("Momentum", momentum), ("Nesterov", nesterov),
                  ("AdaGrad", adagrad), ("RMSProp", rmsprop), ("Adam", adam)]:
    print(name, run(upd))
# expected output: SGD (111, 111)
# expected output: Momentum (78, 128)
# expected output: Nesterov (36, 73)
# expected output: AdaGrad (135, 135)
# expected output: RMSProp (63, None)
# expected output: Adam (80, 127)
# momentum at beta = 0.9 reaches 1e-4 first (78 vs 111) but overshoots and settles after SGD (128):
# its roots have size sqrt(0.9) = 0.949, no faster than SGD's 0.95 crawl; RMSProp's short memory makes it
# fast at first, then its effective rate outgrows the stability limit and it ends hopping at +-0.05
# the extension: the same race on a ravine ten times steeper, f = (100 theta0^2 + theta1^2) / 2
from functools import partial
L, mu = 100.0, 1.0                                     # the steeper bowl's largest and smallest curvatures
def f(p): return 0.5 * (L * p[0] ** 2 + mu * p[1] ** 2)   # run() looks f and grad up by name, so it uses these
def grad(p): return np.array([L * p[0], mu * p[1]])
print("SGD at 0.05", run(sgd), "momentum at 0.05", run(momentum))
root = (np.sqrt(L / mu) - 1) / (np.sqrt(L / mu) + 1)   # the tuned sqrt(beta) of 11.1.1, 9/11 here
lr_sgd, lr_mom, beta_mom = 2 / (L + mu), (1 + root) ** 2 / L, root ** 2
print(f"tuned: SGD lr {lr_sgd:.4f}, momentum lr {lr_mom:.4f} with beta {beta_mom:.3f}")
for name, upd in [("SGD tuned", partial(sgd, lr=lr_sgd)),
                  ("Momentum tuned", partial(momentum, lr=lr_mom, beta=beta_mom)),
                  ("AdaGrad", adagrad), ("RMSProp", rmsprop), ("Adam", adam)]:
    print(name, run(upd))
print("SGD tuned, 600 steps", run(partial(sgd, lr=lr_sgd), steps=600))
# expected output: SGD at 0.05 (None, None) momentum at 0.05 (None, None)
# expected output: tuned: SGD lr 0.0198, momentum lr 0.0331 with beta 0.669
# expected output: SGD tuned (None, None)
# expected output: Momentum tuned (64, 64)
# expected output: AdaGrad (158, 158)
# expected output: RMSProp (66, None)
# expected output: Adam (80, 145)
# expected output: SGD tuned, 600 steps (398, 398)
# the shared rate of 0.05 now multiplies the steep coordinate by 1 - 100 * 0.05 = -4 at every SGD step, so
# SGD diverges, and momentum at beta 0.9 fares no better; SGD's best fixed step, 2/101, shrinks the error by
# only 99/101 = 0.980 per step and needs 398 steps; tuned momentum shrinks it by 9/11 = 0.818, the factor
# tuned SGD had on the 10:1 bowl, yet settles at 64 rather than 35, because it starts from a loss of 808
# instead of 88 and, at the tuned setting, both curvatures sit at the ends of the complex range of 11.1.1,
# where the repeated roots multiply r**t by a factor that grows linearly with t; AdaGrad, RMSProp, and Adam
# divide each coordinate by its own gradient scale, so the 100 cancels and their paths are the 10:1 paths,
# still in lockstep on the diagonal, where the loss is now 101/11, about 9.2, times larger at every point;
# that alone moves AdaGrad from 135 to 158, RMSProp's first crossing from 63 to 66, and Adam's settling
# step from 127 to 145, while Adam's first crossing stays at 80, a dip already deep enough to absorb it

# %% [markdown]
# 2. Learning-rate schedules and why decay beats a constant rate

# %%
# 2. Learning-rate schedules and why decay beats a constant rate
T, base = 1000, 0.1; t = np.arange(T)
constant = np.full(T, base)                             # the flat baseline
step = base * 0.5 ** (t // 300)                         # step decay: halve every 300 steps
cosine = base * 0.5 * (1 + np.cos(np.pi * t / T))       # half-cosine to ~0
warm = 100
warmcos = np.where(t < warm, base * t / warm,           # ramp up, then anneal
                   base * 0.5 * (1 + np.cos(np.pi * (t - warm) / (T - warm))))
print("constant 0/500/999:", round(constant[0],4), round(constant[500],4), round(constant[999],4))
print("step 0/500/999:", round(step[0],4), round(step[500],4), round(step[999],4))
print("cosine 0/500/999:", round(cosine[0],4), round(cosine[500],4), round(cosine[999],4))
print("warmcos 0/100/999:", round(warmcos[0],4), round(warmcos[100],4), round(warmcos[999],4))
# expected output: constant 0/500/999: 0.1 0.1 0.1
# expected output: step 0/500/999: 0.1 0.05 0.0125
# expected output: cosine 0/500/999: 0.1 0.05 0.0
# expected output: warmcos 0/100/999: 0.0 0.1 0.0
def minimize(lr_sched, steps=1000, seed=0, noise=2.0):
    r = np.random.default_rng(seed); x = 5.0; losses = []
    for i in range(steps):
        g = 2 * x + r.normal(0, noise); x -= lr_sched[i] * g; losses.append(x ** 2)
    return np.mean(losses[-50:])
print("constant final loss:", round(minimize(constant), 4))
print("cosine   final loss:", round(minimize(cosine), 4))
print("noise4 constant:", round(minimize(constant, noise=4.0), 4))
print("noise4 cosine:", round(minimize(cosine, noise=4.0), 4))
# expected output: constant final loss: 0.2053
# expected output: cosine   final loss: 0.0028
# expected output: noise4 constant: 0.8212
# expected output: noise4 cosine: 0.0113
p = torch.nn.Parameter(torch.zeros(1)); opt = torch.optim.SGD([p], lr=0.1)
cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1000); lrs = []
for step in range(1000):
    opt.step(); lrs.append(opt.param_groups[0]["lr"]); cos.step()
print("torch cosine 0/500/999:", [round(lrs[i], 4) for i in (0, 500, 999)])
# expected output: torch cosine 0/500/999: [0.1, 0.05, 0.0]

# %% [markdown]
# 3. Signal propagation and initialization

# %%
# 3. Signal propagation and initialization
rng = np.random.default_rng(0); N = 256; L, B = 30, 512
def propagate(scale, act):                              # activation std through L layers
    a = rng.normal(size=(B, N)); stds = []
    for _ in range(L):
        a = act(a @ rng.normal(0, scale, (N, N))); stds.append(a.std())
    return np.array(stds)
xavier = 1 / np.sqrt(N); he = np.sqrt(2 / N)
for name, scale, act in [("too small tanh", 0.01, np.tanh), ("Xavier tanh", xavier, np.tanh),
                         ("too large tanh", 0.15, np.tanh),
                         ("Xavier ReLU", xavier, lambda z: np.maximum(0, z)),
                         ("He ReLU", he, lambda z: np.maximum(0, z))]:
    s = propagate(scale, act); print(f"{name:16} L1={s[0]:.3f} L30={s[29]:.3f}")
gain = float(nn.init.calculate_gain("leaky_relu", 0.01))
s = propagate(gain / np.sqrt(N), lambda z: np.where(z > 0, z, 0.01 * z))
print(f"{'leaky ReLU':16} L1={s[0]:.3f} L30={s[29]:.3f} gain={gain:.4f}")
# expected output: too small tanh   L1=0.156 L30=0.000
# expected output: Xavier tanh      L1=0.628 L30=0.128
# expected output: too large tanh   L1=0.830 L30=0.784
# expected output: Xavier ReLU      L1=0.585 L30=0.000
# expected output: He ReLU          L1=0.830 L30=0.749
# expected output: leaky ReLU       L1=0.830 L30=0.754 gain=1.4141
# leaky ReLU holds the signal, 0.830 at layer 1 and 0.754 at layer 30, as He does for ReLU: its gain
# sqrt(2 / (1 + 0.01**2)) = 1.4141 is He's sqrt(2) adjusted for the small negative slope
torch.manual_seed(0)
lin = nn.Linear(256, 256); nn.init.kaiming_normal_(lin.weight, nonlinearity="relu")
print(f"He std {lin.weight.std().item():.4f} target {(2/256)**0.5:.4f}")
# expected output: He std 0.0888 target 0.0884

# %% [markdown]
# 4. Normalization layers from scratch, checked against PyTorch

# %%
# 4. Normalization layers from scratch, checked against PyTorch
rng = np.random.default_rng(0)
x = rng.normal(5.0, 3.0, (8, 4)).astype(np.float32)
def bn(x, eps=1e-5): return (x - x.mean(0)) / np.sqrt(x.var(0) + eps)          # over the batch, per feature
def ln(x, eps=1e-5): return (x - x.mean(1, keepdims=True)) / np.sqrt(x.var(1, keepdims=True) + eps)  # over features
def rmsn(x, eps=1e-5): return x / np.sqrt((x**2).mean(1, keepdims=True) + eps)  # RMS scale only
xb = bn(x); print("BN largest |feature mean|:", f"{np.abs(xb.mean(0)).max():.1e}", "feature stds:", xb.std(0).round(3).tolist())
xt = torch.tensor(x)
tln = nn.LayerNorm(4, elementwise_affine=False)(xt).detach().numpy()
print("LN matches torch:", np.allclose(ln(x), tln, atol=1e-4))
trms = nn.RMSNorm(4, elementwise_affine=False)(xt).detach().numpy()
print("RMSNorm matches torch:", np.allclose(rmsn(x), trms, atol=1e-4))
# expected output: BN largest |feature mean|: 2.3e-07 feature stds: [1.0, 1.0, 1.0, 1.0]
# expected output: LN matches torch: True
# expected output: RMSNorm matches torch: True

# %% [markdown]
# 5. Regularization closes the train/test gap on 100 memorizable digits

# %%
# 5. Regularization closes the train/test gap on 100 memorizable digits
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
torch.manual_seed(0)
Xd, yd = load_digits(return_X_y=True); Xd = StandardScaler().fit_transform(Xd)
Xtr_np, Xte_np, ytr_np, yte_np = train_test_split(Xd, yd, train_size=100, stratify=yd, random_state=0)
Xtr = torch.tensor(Xtr_np, dtype=torch.float32); ytr = torch.tensor(ytr_np, dtype=torch.long)
Xte = torch.tensor(Xte_np, dtype=torch.float32); yte = torch.tensor(yte_np, dtype=torch.long)
def make_net(p_drop=0.0):
    return nn.Sequential(nn.Linear(64,128), nn.ReLU(), nn.Dropout(p_drop),
                         nn.Linear(128,128), nn.ReLU(), nn.Dropout(p_drop), nn.Linear(128,10))
def train(net, wd=0.0, smooth=0.0, epochs=400, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=wd)
    lf = nn.CrossEntropyLoss(label_smoothing=smooth)
    for _ in range(epochs):
        opt.zero_grad(); lf(net(Xtr), ytr).backward(); opt.step()
    return net
def acc(net, X, y):
    net.eval()
    with torch.no_grad(): return (net(X).argmax(1) == y).float().mean().item()
for name, net, wd, smooth in [("none", make_net(0.0), 0.0, 0.0), ("dropout", make_net(0.5), 0.0, 0.0),
                             ("wdecay", make_net(0.0), 1e-2, 0.0), ("smooth", make_net(0.0), 0.0, 0.1)]:
    train(net, wd=wd, smooth=smooth); tr, te = acc(net, Xtr, ytr), acc(net, Xte, yte)
    print(f"{name:8} train {tr:.2f} test {te:.2f} gap {tr-te:.2f}")
# expected output: none     train 1.00 test 0.86 gap 0.14
# expected output: dropout  train 1.00 test 0.89 gap 0.11
# expected output: wdecay   train 1.00 test 0.89 gap 0.11
# expected output: smooth   train 1.00 test 0.88 gap 0.12
# all three regularizers keep training at 1.0 yet lift test accuracy and shrink the gap

# %% [markdown]
# 6. Vanishing, exploding, and gradient clipping

# %%
# 6. Vanishing, exploding, and gradient clipping
torch.manual_seed(0)
DEPTH, Wd = 18, 48
Xg = torch.randn(128, Wd); yg = torch.randn(128, 1)
def build(activation, weight_gain=1.0, ln=False):
    layers = []
    for _ in range(DEPTH):
        layers.append(nn.Linear(Wd, Wd))
        if ln:
            layers.append(nn.LayerNorm(Wd))
        layers.append(activation())
    net = nn.Sequential(*layers, nn.Linear(Wd, 1))
    if weight_gain != 1.0:
        for m in net.modules():
            if isinstance(m, nn.Linear):
                with torch.no_grad(): m.weight *= weight_gain
    return net
def layer_grad_norms(net):
    net.zero_grad(); nn.MSELoss()(net(Xg), yg).backward()
    return [m.weight.grad.norm().item() for m in net.modules() if isinstance(m, nn.Linear)]
nets6 = [("sigmoid", build(nn.Sigmoid)), ("ReLU", build(nn.ReLU)), ("large", build(nn.ReLU, 3.0))]
for name, net in nets6:
    g = layer_grad_norms(net); print(f"{name:8} L1={g[0]:.2e} L18={g[-2]:.2e}")
# expected output: sigmoid  L1=3.98e-16 L18=3.33e-01
# expected output: ReLU     L1=1.58e-07 L18=4.62e-03
# expected output: large    L1=3.36e+02 L18=2.15e+03
net = nets6[2][1]; net.zero_grad(); nn.MSELoss()(net(Xg), yg).backward()   # clip the 'large' network itself
before = torch.sqrt(sum(p.grad.pow(2).sum() for p in net.parameters())).item()
torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
after = torch.sqrt(sum(p.grad.pow(2).sum() for p in net.parameters())).item()
print(f"clip before {before:.1f} after {after:.2f}")
g = layer_grad_norms(build(nn.ReLU, ln=True)); print(f"{'relu+ln':8} L1={g[0]:.2e} L18={g[-2]:.2e}")
# expected output: clip before 5446.7 after 5.00
# expected output: relu+ln  L1=3.49e-01 L18=7.59e-01
# the 'large' network is the gain-3 stack of 11.6 under the same seed, so its global norm before clipping
# is the chapter's 5446.7; with a LayerNorm after each hidden linear layer the ReLU stack's layer 1 and
# layer 18 weight-gradient norms stay of order one (0.349 and 0.759), against 1.58e-07 and 4.62e-03 without it

# %% [markdown]
# 7. Sanity checks: loss at init, train/eval dropout, and the three lr signatures

# %%
# 7. Sanity checks: loss at init, train/eval dropout, and the three lr signatures
torch.manual_seed(0)
clf = nn.Sequential(nn.Linear(64,64), nn.ReLU(), nn.Linear(64,10))
Xc = torch.randn(200, 64); yc = torch.randint(0, 10, (200,))
with torch.no_grad():
    print(f"init loss {nn.CrossEntropyLoss()(clf(Xc), yc).item():.3f} expected lnC {math.log(10):.3f}")
perm = torch.tensor(np.random.default_rng(0).permutation(len(yc)))
print(f"shuffled labels {nn.CrossEntropyLoss()(clf(Xc), yc[perm]).item():.3f}")
# expected output: init loss 2.327 expected lnC 2.303
# expected output: shuffled labels 2.345
drop = nn.Dropout(0.5); ones = torch.ones(1, 6)
drop.train(); print("train mode:", drop(ones).tolist())
drop.eval();  print("eval mode:", drop(ones).tolist())
# expected output: train mode: [[0.0, 2.0, 0.0, 0.0, 2.0, 0.0]]
# expected output: eval mode: [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
Xtr2, _, ytr2, _ = train_test_split(Xd, yd, test_size=0.25, random_state=0)
Xtr2 = torch.tensor(Xtr2, dtype=torch.float32); ytr2 = torch.tensor(ytr2, dtype=torch.long)
def net7():
    torch.manual_seed(0); return nn.Sequential(nn.Linear(64,64), nn.ReLU(), nn.Linear(64,10))
def runlr(lr, epochs=200):
    m = net7(); opt = torch.optim.SGD(m.parameters(), lr=lr); lf = nn.CrossEntropyLoss()
    first = lf(m(Xtr2), ytr2).item()
    for _ in range(epochs):
        opt.zero_grad(); lf(m(Xtr2), ytr2).backward(); opt.step()
    return first, lf(m(Xtr2), ytr2).item()
for lr in [1e-5, 0.1, 1e4]:
    f0, f1 = runlr(lr); print(f"lr {lr:<8} {f0:.3f} -> {f1:.3f}")
# expected output: lr 1e-05    2.331 -> 2.330
# expected output: lr 0.1      2.331 -> 0.140
# expected output: lr 10000.0  2.331 -> nan

# %% [markdown]
# 8. Unequal microbatches, and fp16 versus bfloat16

# %%
# 8. Unequal microbatches, and fp16 versus bfloat16
torch.manual_seed(0)
Xu = torch.randn(58, 10); yu = torch.randint(0, 3, (58,))
def fresh():
    torch.manual_seed(1); return nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 3))
loss_sum = nn.CrossEntropyLoss(reduction="sum")
m1 = fresh(); m1.zero_grad(); (loss_sum(m1(Xu), yu) / len(yu)).backward()
g_full = torch.cat([p.grad.flatten() for p in m1.parameters()])
m2 = fresh(); m2.zero_grad()
for a, b in zip([0, 16, 32, 48], [16, 32, 48, 58]):
    (loss_sum(m2(Xu[a:b]), yu[a:b]) / len(yu)).backward()
g_acc = torch.cat([p.grad.flatten() for p in m2.parameters()])
print(f"unequal max diff {float((g_full - g_acc).abs().max()):.3e}")
for dt, name in [(torch.float16, "fp16"), (torch.bfloat16, "bf16")]:
    one = torch.tensor(1.0, dtype=dt)
    print(name, "1+0.0003", float(one + torch.tensor(0.0003, dtype=dt)),
          "1e-8", float(torch.tensor(1e-8, dtype=dt)))
# expected output: unequal max diff 2.235e-08
# expected output: fp16 1+0.0003 1.0 1e-8 0.0
# expected output: bf16 1+0.0003 1.0 1e-8 1.0011717677116394e-08

# %% [markdown]
# 9. TPE and random search, scored once on a holdout the search did not see

# %%
# 9. TPE and random search, scored once on a holdout the search did not see
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
Xo, yo = load_digits(return_X_y=True)
Xtr_o, Xte_o, ytr_o, yte_o = train_test_split(Xo, yo, test_size=0.25, random_state=0, stratify=yo)
def run_study(sampler):
    def objective(trial):
        n_layers = trial.suggest_int("n_layers", 1, 3)
        widths = tuple(trial.suggest_int(f"w{i}", 16, 128) for i in range(n_layers))
        lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
        clf = MLPClassifier(hidden_layer_sizes=widths, learning_rate_init=lr, max_iter=300, random_state=0)
        return cross_val_score(clf, Xtr_o, ytr_o, cv=3).mean()
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=15)
    return study
def holdout(widths, lr):
    clf = MLPClassifier(hidden_layer_sizes=widths, learning_rate_init=lr, max_iter=300, random_state=0)
    clf.fit(Xtr_o, ytr_o)
    return accuracy_score(yte_o, clf.predict(Xte_o))
default_h = holdout((50,), 0.001)
print(f"default hold {default_h:.2f}")
studies9, holds9 = {}, {}
for name, sampler in [("tpe", optuna.samplers.TPESampler(seed=0)),
                      ("random", optuna.samplers.RandomSampler(seed=0))]:
    study = run_study(sampler); bp = study.best_params; studies9[name] = study
    widths = tuple(bp[f"w{i}"] for i in range(bp["n_layers"]))
    holds9[name] = holdout(widths, bp["lr"])
    print(f"{name} hold at least default: {holds9[name] >= default_h}")
first9 = next(i for i, (a, b) in enumerate(zip(studies9["tpe"].trials, studies9["random"].trials)) if a.params != b.params)
print("samplers first differ at trial", first9)
# expected output: default hold 0.97
# expected output: tpe hold at least default: True
# expected output: random hold at least default: True
# expected output: samplers first differ at trial 10
# TPESampler samples at random until n_startup_trials, 10 by default, have finished (Optuna Contributors, 2026),
# and with seed 0 those are the same ten draws RandomSampler(seed=0) makes, so trials 0 to 9 coincide.
# Exact best-trial scores move across machines; on the reference machine TPE's trial 13 beat the shared
# trial 0 on the three-fold score (0.9807 against 0.9792), and on the untouched holdout both winners
# met or beat the default (0.9711 against 0.9689). The predicate printed here is the cross-machine check.

# %% [markdown]
# 10. Miniproject: full training with cosine schedule and early stopping

# %%
# 10. Miniproject: full training with cosine schedule and early stopping
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from torch.utils.data import TensorDataset, DataLoader
torch.manual_seed(0)
Xbc, ybc = load_breast_cancer(return_X_y=True)
Xtr3, Xte3, ytr3, yte3 = train_test_split(Xbc, ybc, test_size=0.25, random_state=0, stratify=ybc)
Xtr3, Xva3, ytr3, yva3 = train_test_split(Xtr3, ytr3, test_size=0.25, random_state=0, stratify=ytr3)
sc10 = StandardScaler().fit(Xtr3)
Xtr3, Xva3, Xte3 = sc10.transform(Xtr3), sc10.transform(Xva3), sc10.transform(Xte3)
ff = lambda a, dt: torch.tensor(a, dtype=dt)
dl = DataLoader(TensorDataset(ff(Xtr3, torch.float32), ff(ytr3, torch.long)), batch_size=32, shuffle=True)
Xva_t, yva_t = ff(Xva3, torch.float32), ff(yva3, torch.long)
Xte_t, yte_t = ff(Xte3, torch.float32), ff(yte3, torch.long)
net = nn.Sequential(nn.Linear(30,32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32,2))
opt = torch.optim.AdamW(net.parameters(), lr=1e-2, weight_decay=1e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60)
lf = nn.CrossEntropyLoss()
best_val, best_state, patience, since = 1e9, None, 10, 0
for ep in range(60):
    net.train()
    for xb, yb in dl:
        opt.zero_grad(); lf(net(xb), yb).backward(); opt.step()
    sched.step()
    net.eval()
    with torch.no_grad(): vloss = lf(net(Xva_t), yva_t).item()
    if vloss < best_val - 1e-4:
        best_val, best_state, since = vloss, {k: v.clone() for k, v in net.state_dict().items()}, 0
    else:
        since += 1
        if since >= patience: stopped = ep + 1; break
else:
    stopped = 60
net.load_state_dict(best_state); net.eval()
with torch.no_grad():
    te_acc = (net(Xte_t).argmax(1) == yte_t).float().mean().item()
    te_auc = roc_auc_score(yte3, net(Xte_t).softmax(1)[:, 1].numpy())
print(f"stopped at epoch {stopped}, best val loss {best_val:.3f}")
print(f"test acc {te_acc:.3f} AUC {te_auc:.3f}")
# expected output: stopped at epoch 15, best val loss 0.100
# expected output: test acc 0.965 AUC 0.994

# %% [markdown]
# 11. Batch-norm backward in NumPy on 32 standardized digit rows, checked against autograd

# %%
# 11. Batch-norm backward in NumPy on 32 standardized digit rows, checked against autograd
rows11 = StandardScaler().fit_transform(load_digits().data[:32])       # (32, 64), float64
def batchnorm_backward(g, x, gamma, eps=1e-5):
    xhat = (x - x.mean(0)) / np.sqrt(x.var(0) + eps)                   # batch statistics, per feature
    dx = gamma / np.sqrt(x.var(0) + eps) * (g - g.mean(0) - xhat * (g * xhat).mean(0))   # compact formula of 11.4.1
    return dx, (g * xhat).sum(0), g.sum(0)                             # dL/dx, dL/dgamma, dL/dbeta
rng11 = np.random.default_rng(0)
gamma11 = rng11.uniform(0.5, 1.5, 64); beta11 = rng11.normal(0, 0.1, 64); w11 = rng11.normal(size=(32, 64))
xt11 = torch.tensor(rows11, requires_grad=True)                        # float64 tensors: autograd is the reference
gt11 = torch.tensor(gamma11, requires_grad=True); bt11 = torch.tensor(beta11, requires_grad=True)
yt11 = gt11 * (xt11 - xt11.mean(0)) / torch.sqrt(xt11.var(0, unbiased=False) + 1e-5) + bt11
((torch.tensor(w11) * yt11).sum() + (yt11 ** 3).sum()).backward()      # generic upstream gradient w + 3y^2
dx11, dgamma11, dbeta11 = batchnorm_backward(w11 + 3 * yt11.detach().numpy() ** 2, rows11, gamma11)
print(f"max diff dx {np.abs(dx11 - xt11.grad.numpy()).max():.1e} dgamma {np.abs(dgamma11 - gt11.grad.numpy()).max():.1e} "
      f"dbeta {np.abs(dbeta11 - bt11.grad.numpy()).max():.1e}")
flat11 = rows11.var(0) == 0                                            # pixel columns constant on these 32 rows
print(f"constant columns {int(flat11.sum())} largest |dx| there {np.abs(dx11[:, flat11]).max():.1f} "
      f"elsewhere {np.abs(dx11[:, ~flat11]).max():.1f}")
print(f"largest |column sum of dx| {np.abs(dx11.sum(0)).max():.1e}")
# expected output: max diff dx 1.3e-13 dgamma 4.5e-13 dbeta 8.5e-14
# expected output: constant columns 13 largest |dx| there 1248.8 elsewhere 40.8
# expected output: largest |column sum of dx| 9.0e-13
# all three gradients match autograd to float64 round-off. Each column of dx sums to zero: summing the bracket
# over the batch gives sum(g) - sum(g) - sum(xhat) * mean(g * xhat), and sum(xhat) = 0 because xhat is centered,
# so the printed column sums are round-off. The gradients are largest on the constant columns, whose zero
# variance makes the scale gamma / sqrt(eps), about 316 times gamma

# %% [markdown]
# 12. MC dropout, a five-member ensemble, and selective prediction

# %%
# 12. MC dropout, a five-member ensemble, and selective prediction
torch.manual_seed(0); rng12 = np.random.default_rng(0)
Xd12, yd12 = load_digits(return_X_y=True); Xd12 = StandardScaler().fit_transform(Xd12)
Xtr12, Xte12, ytr12, yte12 = train_test_split(Xd12, yd12, train_size=100, stratify=yd12, random_state=0)
Xtr12 = torch.tensor(Xtr12, dtype=torch.float32); ytr12 = torch.tensor(ytr12, dtype=torch.long)
Xte12 = torch.tensor(Xte12, dtype=torch.float32); yte12 = torch.tensor(yte12, dtype=torch.long)
def make12(p):
    return nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Dropout(p),
                         nn.Linear(128, 128), nn.ReLU(), nn.Dropout(p), nn.Linear(128, 10))
def train12(net, seed, X, y):
    torch.manual_seed(seed); opt = torch.optim.Adam(net.parameters(), lr=0.01); lf = nn.CrossEntropyLoss()
    for _ in range(400):
        opt.zero_grad(); lf(net(X), y).backward(); opt.step()
    return net
net12 = train12(make12(0.5), 0, Xtr12, ytr12); net12.train()
with torch.no_grad():
    passes12 = torch.stack([net12(Xte12).softmax(1) for _ in range(100)])
pred12 = passes12.mean(0).argmax(1); rows12 = torch.arange(len(pred12))
std12 = passes12[:, rows12, pred12].std(0); ok12 = pred12 == yte12
sc12, si12 = float(std12[ok12].mean()), float(std12[~ok12].mean())
print(f"mc wrong std larger: {si12 > sc12}; at least 2x: {si12 >= 2 * sc12}")
nets12 = []
for s in range(5):
    idx = rng12.integers(0, len(Xtr12), len(Xtr12))
    nets12.append(train12(make12(0.0), s, Xtr12[idx], ytr12[idx]))
for n in nets12: n.eval()
with torch.no_grad():
    probs12 = torch.stack([n(Xte12).softmax(1) for n in nets12])
ens12 = probs12.mean(0).argmax(1)
member12 = np.mean([(n(Xte12).argmax(1) == yte12).float().mean().item() for n in nets12])
print(f"member {member12:.3f} ensemble {(ens12 == yte12).float().mean().item():.3f}")
dis12 = probs12[:, torch.arange(len(ens12)), ens12].std(0)
keep12 = dis12 <= torch.quantile(dis12, 0.9)
print(f"selective {(ens12 == yte12)[keep12].float().mean().item():.3f} kept {int(keep12.sum())}")
# expected output: mc wrong std larger: True; at least 2x: True
# expected output: member 0.811 ensemble 0.863
# expected output: selective 0.900 kept 1527

# %% [markdown]
# 13. Overfit one batch, then plant three bugs

# %%
# 13. Overfit one batch, then plant three bugs
torch.manual_seed(0)
Xd13, yd13 = load_digits(return_X_y=True); Xd13 = StandardScaler().fit_transform(Xd13)
Xb13 = torch.tensor(Xd13[:32], dtype=torch.float32); yb13 = torch.tensor(yd13[:32], dtype=torch.long)
def steps_until(bug):
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
    opt = torch.optim.Adam(net.parameters(), lr=1e-2); lf = nn.CrossEntropyLoss()
    labels = yb13 if bug != "shuffle" else yb13[torch.randperm(len(yb13))]
    for step in range(1, 401):
        if bug != "no_zero": opt.zero_grad()
        logits = net(Xb13)
        if bug == "softmax": logits = logits.softmax(1)
        loss = lf(logits, labels); loss.backward(); opt.step()
        if loss.item() < 0.01: return step, round(loss.item(), 4)
    return None, round(loss.item(), 4)
for bug in ("none", "softmax", "no_zero", "shuffle"):
    print(bug, *steps_until(bug))
# expected output: none 24 0.0099
# expected output: softmax None 1.491
# expected output: no_zero 37 0.0022
# expected output: shuffle 42 0.009

# %% [markdown]
# 14. Six checkpointed blocks run their forward twice

# %%
# 14. Six checkpointed blocks run their forward twice
from torch.utils.checkpoint import checkpoint
calls14 = {"n": 0}
class Block14(nn.Module):
    def __init__(self):
        super().__init__(); self.lin1 = nn.Linear(16, 16); self.lin2 = nn.Linear(16, 16)
    def forward(self, z):
        calls14["n"] += 1; return self.lin2(torch.relu(self.lin1(z)))
class Plain14(nn.Module):
    def __init__(self):
        super().__init__(); self.blocks = nn.ModuleList(Block14() for _ in range(6))
    def forward(self, z, use_ckpt):
        for b in self.blocks:
            z = checkpoint(b, z, use_reentrant=False) if use_ckpt else b(z)
        return z
torch.manual_seed(0); z14 = torch.randn(4, 16, requires_grad=True); m14 = Plain14()
calls14["n"] = 0; y14 = m14(z14, False); g14 = torch.autograd.grad(y14.sum(), z14)[0]; c_plain = calls14["n"]
calls14["n"] = 0; y14b = m14(z14, True); g14b = torch.autograd.grad(y14b.sum(), z14)[0]
print("calls", c_plain, calls14["n"], "match", bool(torch.allclose(g14, g14b)), "max", float((g14 - g14b).abs().max()))
# expected output: calls 6 12 match True max 0.0

# %% [markdown]
# 15. Batch size 32 versus 256, and an 8x learning rate

# %%
# 15. Batch size 32 versus 256, and an 8x learning rate
torch.manual_seed(0)
Xd15, yd15 = load_digits(return_X_y=True); Xd15 = StandardScaler().fit_transform(Xd15)
Xtr15, Xte15, ytr15, yte15 = train_test_split(Xd15, yd15, test_size=0.25, random_state=0)
Xtr15, Xva15, ytr15, yva15 = train_test_split(Xtr15, ytr15, test_size=0.25, random_state=0)
Xva15_t, yva15_t = torch.tensor(Xva15, dtype=torch.float32), torch.tensor(yva15, dtype=torch.long)
def run15(bs, lr):
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
    opt = torch.optim.Adam(net.parameters(), lr=lr); lf = nn.CrossEntropyLoss()
    dl = DataLoader(TensorDataset(torch.tensor(Xtr15, dtype=torch.float32), torch.tensor(ytr15, dtype=torch.long)),
                    batch_size=bs, shuffle=True)
    updates = 0
    for _ in range(10):
        net.train()
        for xb, yb in dl:
            opt.zero_grad(); lf(net(xb), yb).backward(); opt.step(); updates += 1
    net.eval()
    with torch.no_grad():
        acc = (net(Xva15_t).argmax(1) == yva15_t).float().mean().item()
    return acc, updates
for bs, lr in ((32, 0.001), (256, 0.001), (256, 0.008)):
    acc, updates = run15(bs, lr)
    print(f"bs {bs} lr {lr} val {acc:.3f} updates {updates}")
# expected output: bs 32 lr 0.001 val 0.944 updates 320
# expected output: bs 256 lr 0.001 val 0.819 updates 40
# expected output: bs 256 lr 0.008 val 0.950 updates 40
# the extension: Adam's rate at batch 256, around the two scaling rules of 11.9.4
for lr in (0.001 * math.sqrt(8), 0.016, 0.032, 0.064):  # the square-root rule's rate, then 2 to 8 times the linear rule's
    acc, updates = run15(256, lr)
    print(f"bs 256 lr {lr:.4f} val {acc:.3f} ({round(acc * len(yva15))}/{len(yva15)} correct)")
# expected output: bs 256 lr 0.0028 val 0.896 (302/337 correct)
# expected output: bs 256 lr 0.0160 val 0.973 (328/337 correct)
# expected output: bs 256 lr 0.0320 val 0.955 (322/337 correct)
# expected output: bs 256 lr 0.0640 val 0.970 (327/337 correct)
# the square-root rate alone, 0.0028, recovers only part of the drop, 0.896 against 0.944 at batch 32;
# Malladi et al. (2022) pair that rate with beta1 = 1 - 8 * (1 - 0.9) = 0.2, beta2 = 1 - 8 * (1 - 0.999) = 0.992,
# and eps divided by sqrt(8), which this sweep leaves at PyTorch's defaults; from the linear rule's 0.008
# (0.950, 320 correct) up to 0.064, the four runs land within 8 of the 337 validation images, 320 to 328
# correct, a spread one seed cannot rank; on this ten-epoch budget the run favors the linear rule's rate or
# a larger one, and neither formula replaces the short sweep that 11.9.4 recommends

# %% [markdown]
# ## 11.12 coding drills

# %%
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn, math

# %% [markdown]
# Optimizers

# %%
# Optimizers
g = np.array([10.0, 1.0])                                  # a gradient in a ravine
v = 0.9 * 0.0 + g; print(np.round(-0.05 * v, 3))           # one momentum step from rest
# expected output: [-0.5  -0.05]
v = g.copy(); print(np.round(-0.05 * (g + 0.9 * v), 3))  # one Nesterov step from rest
# expected output: [-0.95  -0.095]
E = 0.9 * 0.0 + 0.1 * g**2; print(np.round(-0.1 * g / (np.sqrt(E) + 1e-8), 3))  # RMSProp equalizes the axes
# expected output: [-0.316 -0.316]
m = 0.1 * g; vv = 0.001 * g**2                             # Adam moments after t=1
mhat, vhat = m / (1 - 0.9), vv / (1 - 0.999)
print(np.round(-0.3 * mhat / (np.sqrt(vhat) + 1e-8), 3))   # each coordinate moves by about -lr
# expected output: [-0.3 -0.3]

# %% [markdown]
# Learning-rate schedules

# %%
# Learning-rate schedules
base = 0.1
print(round(base * 0.5 * (1 + math.cos(math.pi * 0.5)), 4))  # cosine at the midpoint
# expected output: 0.05
print(round(base * 0.5 ** (600 // 300), 4))                  # step decay, halved twice
# expected output: 0.025
print(round(base * min(1.0, 50 / 100), 4))                   # warmup halfway up the ramp
# expected output: 0.05
print(round(base * math.exp(-3 * 500 / 1000), 4))            # exponential at the midpoint
# expected output: 0.0223

# %% [markdown]
# Initialization and normalization

# %%
# Initialization and normalization
print(round((2 / 256) ** 0.5, 4))                            # He std for n_in=256
# expected output: 0.0884
print(round((2 / (256 + 256)) ** 0.5, 4))                    # Xavier std, Glorot form
# expected output: 0.0625
print(round((1 / 256) ** 0.5, 4))                            # the 1/n_in simplification when n_in = n_out
# expected output: 0.0625
x = np.array([[5., 1., 9.], [3., 7., 2.]])
xb = (x - x.mean(0)) / (x.std(0) + 1e-8); print(np.round(xb.mean(0), 3))  # per-feature mean 0
# expected output: [0. 0. 0.]
row = np.array([2., 4., 6., 8.])
print(np.round((row - row.mean()) / row.std(), 3))           # layer norm
# expected output: [-1.342 -0.447  0.447  1.342]
print(np.round(row / np.sqrt((row**2).mean()), 3))           # RMS norm
# expected output: [0.365 0.73  1.095 1.461]

# %% [markdown]
# Regularization

# %%
# Regularization
p = 0.5; print(round(1 / (1 - p), 1))                        # dropout survivor scaling
# expected output: 2.0
net = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10))
print(sum(pp.numel() for pp in net.parameters()))           # trainable params
# expected output: 9610
w = torch.tensor([3.0, -4.0]); print(round(0.5 * (w**2).sum().item(), 1))  # L2 penalty
# expected output: 12.5

# %% [markdown]
# Gradients and PyTorch

# %%
# Gradients and PyTorch
xg = torch.tensor(3.0, requires_grad=True); yv = xg**2 + 2*xg; yv.backward()
print(xg.grad.item())                                        # autograd dy/dx = 2x+2 at x=3
# expected output: 8.0
torch.manual_seed(0)
m2 = nn.Linear(4, 1)
for pp in m2.parameters(): pp.grad = torch.ones_like(pp) * 10.0
before = torch.sqrt(sum(pp.grad.pow(2).sum() for pp in m2.parameters())).item()
torch.nn.utils.clip_grad_norm_(m2.parameters(), 1.0)
after = torch.sqrt(sum(pp.grad.pow(2).sum() for pp in m2.parameters())).item()
print(round(before, 2), round(after, 2))                     # norm capped at max_norm
# expected output: 22.36 1.0
print(round(math.log(10), 3))                                # loss at init for 10 balanced classes
# expected output: 2.303
x3 = torch.tensor(2.0, requires_grad=True)
with torch.no_grad(): z = x3 * x3
print(z.requires_grad)                                       # no_grad detaches the result
# expected output: False

# %% [markdown]
# Memory

# %%
# Memory
bytes_pp = 4 * 4                                             # fp32 weight, gradient, and two Adam moments, 4 bytes each
gb_1b, gb_7b = 1e9 * bytes_pp / 1e9, 7e9 * bytes_pp / 1e9    # decimal gigabytes for 1 billion and 7 billion parameters
print(round(gb_1b), round(gb_7b), round(gb_1b / 8), round(gb_7b / 8))  # total GB, then GB per GPU sharded over 8
# expected output: 16 112 2 14
