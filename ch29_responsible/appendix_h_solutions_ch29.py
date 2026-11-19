"""Appendix H solutions for Chapter 29, as runnable cells.

Generated from Solutions.md (Appendix H). Every code block of this chapter's solutions appears here
verbatim, split into one cell per exercise or drill topic, so each answer can be run and checked against
the expected outputs it states. Keep Solutions.md as the source; regenerate after any change.
"""

# %% [markdown]
# ## 29.10 lab tasks

# %% [markdown]
# 1. Same seed and one Laplace draw as the lab, with epsilon 0.1 so the scale is 10.

# %%
# 1. Same seed and one Laplace draw as the lab, with epsilon 0.1 so the scale is 10.
from sklearn.datasets import load_breast_cancer
import numpy as np
bc = load_breast_cancer()
print(int((bc.target == 0).sum()))  # expected output: 212

def released_at_epsilon(epsilon):   # the lab's Laplace release of the malignant-case count, at any budget
    rng = np.random.default_rng(3)  # the lab's seed; its Laplace release is the first draw from it
    return round(float(int((bc.target == 0).sum()) + rng.laplace(0, 1 / epsilon)), 1)
print(released_at_epsilon(0.1))     # expected output: 194.4
# The release misses the true 212 by 17.6, more than the expected average error of 10, the scale; for Laplace
# noise of scale 10 an error at least that large has probability exp(-17.6 / 10), about 0.17.

# %% [markdown]
# ## 29.11 coding exercises

# %%
import warnings; warnings.filterwarnings("ignore")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
import numpy as np
from sklearn.model_selection import train_test_split
# Each solution completes its starter's function and calls it as the starter does; the function returns its
# report as text, so the starter's print(...) shows the report where the starter printed None.

# %% [markdown]
# 1. A race audit on the Adult census data

# %%
# 1. A race audit on the Adult census data
from sklearn.datasets import fetch_openml
from sklearn.ensemble import HistGradientBoostingClassifier
adult = fetch_openml("adult", version=2, as_frame=True)      # 48842 real census rows
print(adult.data.shape)                                      # expected output: (48842, 14)

def fairness_audit():                                        # Example 29-1's model and split, audited by race
    X = adult.data.copy()
    y = (adult.target == ">50K").astype(int).to_numpy()      # 1 if the person earns > $50K
    race = X["race"].astype(str).to_numpy()
    for c in X.select_dtypes(include=["object", "category"]).columns:
        X[c] = X[c].astype("category")
    Xtr, Xte, ytr, yte, _, rte = train_test_split(X, y, race, test_size=0.3, random_state=0, stratify=y)
    clf = HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=0).fit(Xtr, ytr)
    score = clf.predict_proba(Xte)[:, 1]
    def rates(pred, g):                                      # test rows, selection, TPR, FPR, base rate of group g
        m = rte == g
        return int(m.sum()), pred[m].mean(), pred[m & (yte == 1)].mean(), pred[m & (yte == 0)].mean(), yte[m].mean()
    pred = (score > 0.5).astype(int)
    r = {g: rates(pred, g) for g in ("White", "Black")}
    out = [f"{g:5s} n {n}  sel {s:.3f}  tpr {t:.3f}  fpr {f:.3f}  base {b:.3f}" for g, (n, s, t, f, b) in r.items()]
    (_, sW, tW, fW, _), (_, sB, tB, fB, _) = r["White"], r["Black"]
    out.append(f"DP gap {abs(sW - sB):.3f}  EO gap {abs(tW - tB):.3f}  "
               f"EqOdds gap {max(abs(tW - tB), abs(fW - fB)):.3f}  ratio {min(sW, sB) / max(sW, sB):.3f}")
    low, high = ("Black", "White") if tB < tW else ("White", "Black")
    pos = (rte == low) & (yte == 1)                          # the lower-TPR group's truly positive test rows
    thr = max(t for t in np.linspace(0.01, 0.99, 197) if (score[pos] > t).mean() >= r[high][2])
    pred2 = np.where(rte == low, score > thr, score > 0.5).astype(int)
    _, sl, tl, fl, _ = rates(pred2, low); _, sh, th, _, _ = rates(pred2, high)
    out.append(f"lower the {low} threshold to {thr:.3f}: {low} tpr {tl:.3f} fpr {fl:.3f} sel {sl:.3f}, {high} sel {sh:.3f}")
    out.append(f"EO gap now {abs(th - tl):.3f}, DP gap still {abs(sh - sl):.3f}")
    return "\n".join(out)
print(fairness_audit())
# expected output: White n 12510  sel 0.212  tpr 0.654  fpr 0.060  base 0.255
# expected output: Black n 1436  sel 0.082  tpr 0.494  fpr 0.031  base 0.110
# expected output: DP gap 0.130  EO gap 0.161  EqOdds gap 0.161  ratio 0.388
# expected output: lower the Black threshold to 0.330: Black tpr 0.658 fpr 0.052 sel 0.119, White sel 0.212
# expected output: EO gap now 0.004, DP gap still 0.093
# The selection-rate identity, selection = FPR + (TPR - FPR) x base rate, explains the gap that remains. From the
# printed rates the Black group gets 0.052 + (0.658 - 0.052) x 0.110 = 0.119, and the White group gets
# 0.060 + (0.654 - 0.060) x 0.255 = 0.211, the printed 0.212 up to rounding. With the true positive rates matched, the
# selection rates stay apart because the base rates differ, 0.110 against 0.255: a direct consequence of unequal base rates.

# %% [markdown]
# 2. Two global rankings of one diagnostic model

# %%
# 2. Two global rankings of one diagnostic model
import shap
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
bc = load_breast_cancer()                                    # 569 real tumors, 30 features
print(bc.data.shape)                                         # expected output: (569, 30)

def top_drivers():                                           # the model of Example 29-5, ranked two ways
    names = [str(n) for n in bc.feature_names]
    yb = (bc.target == 0).astype(int)                        # 1 = malignant
    Xtr, Xte, ytr, yte = train_test_split(bc.data, yb, test_size=0.25, random_state=0, stratify=yb)
    gb = GradientBoostingClassifier(random_state=0).fit(Xtr, ytr)
    pi = permutation_importance(gb, Xte, yte, n_repeats=10, random_state=0)
    imp = pi.importances_mean
    sv = np.abs(shap.TreeExplainer(gb).shap_values(Xte)).mean(0)      # mean |SHAP| over the test tumors
    po, so = np.argsort(imp, kind="stable")[::-1][:5], np.argsort(sv, kind="stable")[::-1][:5]
    both = sorted(set(names[i] for i in po[:3]) & set(names[i] for i in so[:3]))
    zero = imp == 0
    r = np.corrcoef(bc.data[:, names.index("mean concave points")],
                    bc.data[:, names.index("worst concave points")])[0, 1]
    return "\n".join(["permutation: " + ", ".join(f"{names[i]} {imp[i]:.4f}" for i in po),
                      "mean |SHAP|: " + ", ".join(f"{names[i]} {sv[i]:.3f}" for i in so),
                      f"in both top threes: {both}",
                      f"zero permutation importance: {int(zero.sum())} features, "
                      f"{int((pi.importances[zero] == 0).all(axis=1).sum())} of them zero in all 10 shuffles",
                      f"test tumors {len(yte)}; correlation of mean and worst concave points {r:.3f}"])
print(top_drivers())
# expected output: permutation: worst area 0.0420, worst texture 0.0210, area error 0.0210, mean concave points 0.0154, worst concave points 0.0147
# expected output: mean |SHAP|: worst area 1.606, worst concave points 1.443, mean concave points 1.437, area error 0.902, worst perimeter 0.522
# expected output: in both top threes: ['worst area']
# expected output: zero permutation importance: 7 features, 7 of them zero in all 10 shuffles
# expected output: test tumors 143; correlation of mean and worst concave points 0.910
# The two rankings measure different things. Permutation importance is a drop in accuracy on 143 test tumors, so it
# moves in steps of 1/143, about 0.007 (0.0420 is six tumors on average, 0.0210 three), and 9 features never move it;
# mean |SHAP| averages how far each feature pushes every tumor's log odds. Section 29.2's caveat about correlated
# features applies too: mean and worst concave points correlate at 0.910, so shuffling one alone pairs the two in
# combinations real tumors rarely show, and the pair ranks 4th and 5th by permutation but 2nd and 3rd by SHAP. Of the
# permutation ranking's top three, worst area and area error are size measurements and worst texture is a texture one.

# %% [markdown]
# 3. Exact Shapley values and the efficiency axiom (top 8 features)

# %%
# 3. Exact Shapley values and the efficiency axiom (top 8 features)
from itertools import combinations
from math import factorial
bc = load_breast_cancer()
print(bc.data.shape)                                         # expected output: (569, 30)
yb3 = (bc.target == 0).astype(int)
Xtr3, Xte3, ytr3, yte3 = train_test_split(bc.data, yb3, test_size=0.25, random_state=0, stratify=yb3)
full = GradientBoostingClassifier(random_state=0).fit(Xtr3, ytr3)
imp3 = permutation_importance(full, Xte3, yte3, n_repeats=10, random_state=0).importances_mean
sel = list(np.argsort(imp3, kind="stable")[::-1][:8])       # the eight features exercise 2's permutation ranks highest
m3 = GradientBoostingClassifier(random_state=0).fit(Xtr3[:, sel], ytr3)
baseline = Xtr3[:, sel].mean(0)                              # "feature absent" = its training mean
def v(coalition, x):                                         # model margin with only `coalition` known
    z = baseline.copy()
    for i in coalition: z[i] = x[i]
    return m3.decision_function(z.reshape(1, -1))[0]

def exact_shapley(row):                                      # row: one tumor's 30 features
    x = row[sel]; n = len(x); phi = np.zeros(n)
    for i in range(n):
        rest = [j for j in range(n) if j != i]
        for k in range(len(rest) + 1):
            for S in combinations(rest, k):                  # every coalition S not containing i
                w = factorial(k) * factorial(n - k - 1) / factorial(n)
                phi[i] += w * (v(set(S) | {i}, x) - v(set(S), x))
    out = [f"P(malignant) {m3.predict_proba(x.reshape(1, -1))[0, 1]:.3f}"]
    out += [f"  {bc.feature_names[sel[i]]}: {phi[i]:+.3f}" for i in np.argsort(-np.abs(phi), kind="stable")]
    out.append(f"  sum {phi.sum():+.3f} == margin - baseline {v(set(range(n)), x) - v(set(), x):+.3f}")
    return "\n".join(out)
print(exact_shapley(bc.data[0]))
# expected output: P(malignant) 0.999
# expected output: mean concave points: +4.264
# expected output: worst concave points: +2.242
# expected output: worst texture: -1.769
# expected output: area error: +1.661
# expected output: worst area: +1.109
# expected output: worst radius: +0.751
# expected output: mean texture: -0.478
# expected output: worst perimeter: +0.216
# expected output: sum +7.997 == margin - baseline +7.997
# The eight values add up to the margin minus the baseline, +7.997, exactly as the efficiency axiom requires; the
# concave points and the size features push this tumor toward malignant, while the two texture features push it
# toward benign.

# %% [markdown]
# 4. LIME-style local surrogate vs partial dependence on the diabetes data

# %%
# 4. LIME-style local surrogate vs partial dependence on the diabetes data
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import partial_dependence
dia = load_diabetes()
print(dia.data.shape)                                        # expected output: (442, 10)

def local_and_global():                                      # explain patient 0 locally, then sweep its top feature
    rng = np.random.default_rng(0)
    Xd, yd, dnames = dia.data, dia.target, dia.feature_names
    reg = GradientBoostingRegressor(random_state=0).fit(Xd, yd)
    x0 = Xd[0]
    Z = x0 + rng.normal(0, 0.02, size=(1000, Xd.shape[1]))   # sample points NEAR the patient
    pz = reg.predict(Z)
    wt = np.exp(-np.linalg.norm(Z - x0, axis=1) ** 2 / (0.02 ** 2))   # closer points weigh more
    Zc = Z - Z.mean(0)
    coef = np.linalg.lstsq(Zc.T @ np.diag(wt) @ Zc, Zc.T @ np.diag(wt) @ (pz - pz.mean()), rcond=None)[0]
    top = np.argsort(np.abs(coef))[::-1][:3]
    f = int(top[0])
    # method="brute" averages the model's predictions over all 442 patients, as in Example 29-4; the default
    # for gradient boosting walks the trees instead and leaves out the model's starting constant
    pdp = partial_dependence(reg, Xd, [f], grid_resolution=5, method="brute")
    grid, avg = pdp["grid_values"][0], pdp["average"][0]
    return "\n".join(["local linear top features: " + str([dnames[i] for i in top]),
                      f"local slope of {dnames[f]} at patient 0 ({dnames[f]} {x0[f]:+.3f}): {coef[f]:+.1f}",
                      f"PDP of {dnames[f]}: " + str([f"{g:+.2f}->{a:.1f}" for g, a in zip(grid, avg)]),
                      "PDP slopes between grid points: " + str([round(float(s), 1) for s in np.diff(avg) / np.diff(grid)])])
print(local_and_global())
# expected output: local linear top features: ['bmi', 'bp', 's5']
# expected output: local slope of bmi at patient 0 (bmi +0.062): +1377.6
# expected output: PDP of bmi: ['-0.07->130.9', '-0.03->132.6', '+0.01->170.3', '+0.05->175.4', '+0.09->209.7']
# expected output: PDP slopes between grid points: [43.9, 979.4, 130.9, 891.6]
# Near patient 0 the surrogate's steepest slope is bmi's, +1377.6 per unit of the scaled bmi. The partial dependence
# curve, averaged over all 442 patients, also rises with bmi, from 130.9 to 209.7 across the grid, but unevenly, with
# slopes from 43.9 to 979.4 between grid points, so one local slope describes the model only near its patient.

# %% [markdown]
# 5. The Laplace mechanism on a bounded mean

# %%
# 5. The Laplace mechanism on a bounded mean
bc = load_breast_cancer()
radius = bc.data[:, list(bc.feature_names).index("mean radius")]
print(radius.shape)                                          # expected output: (569,)

def laplace_errors():                                        # release the clipped mean of radius at three budgets
    lo, hi = 0.0, 30.0                                       # the public bounds
    clipped = np.clip(radius, lo, hi)
    true_mean = clipped.mean()
    sens = (hi - lo) / len(clipped)                          # one tumor's record moves the clipped mean by at most this
    rng = np.random.default_rng(0)
    out = [f"true clipped mean {true_mean:.4f}, sensitivity {sens:.5f}"]
    for eps in (0.1, 1.0, 10.0):
        noisy = true_mean + rng.laplace(0, sens / eps, 20000)
        out.append(f"  eps {eps}: scale {sens / eps:.5f}, avg error {np.abs(noisy - true_mean).mean():.5f}")
    return "\n".join(out)
print(laplace_errors())
# expected output: true clipped mean 14.1273, sensitivity 0.05272
# expected output: eps 0.1: scale 0.52724, avg error 0.52488
# expected output: eps 1.0: scale 0.05272, avg error 0.05246
# expected output: eps 10.0: scale 0.00527, avg error 0.00531
# Each average error sits close to its noise scale, the expected absolute error E|X| = b that Section 29.3 derives:
# 0.52488 against 0.52724, 0.05246 against 0.05272, and 0.00531 against 0.00527. Because the public bounds hold the
# sensitivity at 30/569, the mean needs 0.0527 of the noise that a count, with sensitivity 1, needs at the same epsilon.

# %% [markdown]
# 6. Federated averaging when clients hold different labels

# %%
# 6. Federated averaging when clients hold different labels
from sklearn.preprocessing import StandardScaler
bc = load_breast_cancer()
print(bc.data.shape)                                         # expected output: (569, 30)

def fedavg_versus_pooled():                                  # Example 29-9's FedAvg on random and label-sorted shards
    Xs = StandardScaler().fit_transform(bc.data); yy = bc.target
    Xtr, Xte, ytr, yte = train_test_split(Xs, yy, test_size=0.3, random_state=0, stratify=yy)
    d = Xtr.shape[1]
    sig = lambda z: 1 / (1 + np.exp(-z))
    def local_update(w, b, Xc, yc, steps, lr=0.1):           # a client trains on ITS OWN data only
        for _ in range(steps):
            g = sig(Xc @ w + b) - yc
            w = w - lr * (Xc.T @ g) / len(yc); b = b - lr * g.mean()
        return w, b
    acc = lambda w, b: ((sig(Xte @ w + b) > 0.5).astype(int) == yte).mean()
    wc, b_c = local_update(np.zeros(d), 0.0, Xtr, ytr, steps=300)     # centralized baseline on the pooled rows
    out = [f"centralized {acc(wc, b_c):.3f}"]
    for name, order in (("random", np.random.default_rng(0).permutation(len(Xtr))),
                        ("sorted by label", np.argsort(ytr, kind="stable"))):
        shards = np.array_split(order, 5)                    # 5 clients
        wf, bf = np.zeros(d), 0.0
        for _ in range(60):                                  # each FedAvg round
            ups = [local_update(wf.copy(), bf, Xtr[s], ytr[s], steps=5) for s in shards]
            wf = np.mean([u[0] for u in ups], 0); bf = np.mean([u[1] for u in ups])
        out.append(f"{name} shards {[len(s) for s in shards]}, label 1 share "
                   f"{[round(float(ytr[s].mean()), 2) for s in shards]}: federated {acc(wf, bf):.3f}")
    return "\n".join(out)
print(fedavg_versus_pooled())
# expected output: centralized 0.959
# expected output: random shards [80, 80, 80, 79, 79], label 1 share [0.6, 0.7, 0.64, 0.63, 0.57]: federated 0.959
# expected output: sorted by label shards [80, 80, 80, 79, 79], label 1 share [0.0, 0.15, 1.0, 1.0, 1.0]: federated 0.959
# Both splits reach the centralized 0.959. Averaged over shards of almost equal size, one local step from the shared
# weights is nearly one gradient step on the pooled loss, because the pooled gradient is the mean of the shard
# gradients; 60 rounds of 5 small steps then track the 300 centralized steps closely, and the logistic loss is convex.
# That explains this model's result; it shows nothing about a nonconvex model or many local steps.

# %% [markdown]
# 7. Regularization against membership inference

# %%
# 7. Regularization against membership inference
from sklearn.ensemble import RandomForestClassifier
bc = load_breast_cancer()
print(bc.data.shape)                                         # expected output: (569, 30)

def membership_attack():                                     # Example 29-9's attack on three forests
    Xm, Xn, ym, yn = train_test_split(bc.data, bc.target, test_size=0.5, random_state=0, stratify=bc.target)
    out = []
    for leaf in (1, 10, 25):
        rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=leaf, random_state=0, n_jobs=2).fit(Xm, ym)
        cm = rf.predict_proba(Xm)[np.arange(len(ym)), ym]    # confidence in the true label, members
        cn = rf.predict_proba(Xn)[np.arange(len(yn)), yn]    # and nonmembers
        allc = np.concatenate([cm, cn]); lab = np.concatenate([np.ones_like(cm), np.zeros_like(cn)])
        best = max(((allc > tau).astype(int) == lab).mean() for tau in np.linspace(allc.min(), allc.max(), 400))
        out.append(f"leaf {leaf:2d}: train {rf.score(Xm, ym):.3f} test {rf.score(Xn, yn):.3f}, "
                   f"confidence members {cm.mean():.3f} nonmembers {cn.mean():.3f}, attack {best:.3f}")
    return "\n".join(out)
print(membership_attack())
# expected output: leaf  1: train 1.000 test 0.944, confidence members 0.969 nonmembers 0.912, attack 0.583
# expected output: leaf 10: train 0.972 test 0.930, confidence members 0.911 nonmembers 0.880, attack 0.543
# expected output: leaf 25: train 0.965 test 0.916, confidence members 0.882 nonmembers 0.853, attack 0.541
# Regularizing the forest narrows the confidence gap between members and nonmembers, from 0.969 against 0.912 at
# leaf 1 to 0.882 against 0.853 at leaf 25, and the attack falls from 0.583 to 0.541, still above a coin flip, while
# test accuracy falls from 0.944 to 0.916. Each attack figure takes the best threshold on the scored records
# themselves, so each is an optimistic estimate of what an attacker would reach.

# %% [markdown]
# 8. Attacking the hardened network

# %%
# 8. Attacking the hardened network
import torch, torch.nn as nn
from sklearn.datasets import load_digits
torch.set_num_threads(2)      # the trained network and every accuracy it prints change with the thread count
digits = load_digits()
print(digits.data.shape)                                     # expected output: (1797, 64)

def fgsm_and_pgd():                                          # Example 29-12's hardened network against FGSM and PGD
    X = digits.images.reshape(len(digits.images), -1) / 16.0 # pixels scaled into [0, 1]
    Xtr, Xte, ytr, yte = train_test_split(X, digits.target, test_size=0.3, random_state=0, stratify=digits.target)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32); ytr_t = torch.tensor(ytr)
    Xte_t = torch.tensor(Xte, dtype=torch.float32); yte_t = torch.tensor(yte)
    torch.manual_seed(0)                                     # Example 29-12's make_net()
    net = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    for _ in range(400):                                     # adversarial training on clean + FGSM(0.2) copies
        xb = Xtr_t.clone().requires_grad_(True)
        opt.zero_grad(); nn.functional.cross_entropy(net(xb), ytr_t).backward()
        xadv = (xb + 0.2 * xb.grad.sign()).clamp(0, 1).detach()
        both = torch.cat([Xtr_t, xadv]); lab = torch.cat([ytr_t, ytr_t])
        opt.zero_grad(); nn.functional.cross_entropy(net(both), lab).backward(); opt.step()
    def fgsm(Xb, eps):                                       # one signed-gradient step, clamped to a valid image
        xb = Xb.clone().requires_grad_(True)
        nn.functional.cross_entropy(net(xb), yte_t).backward()
        return (xb + eps * xb.grad.sign()).clamp(0, 1).detach()
    def pgd(Xb, eps, alpha, steps):                          # random start, then projected signed steps
        xadv = (Xb + torch.empty_like(Xb).uniform_(-eps, eps)).clamp(0, 1)
        for _ in range(steps):
            xadv = xadv.detach().requires_grad_(True)
            nn.functional.cross_entropy(net(xadv), yte_t).backward()
            xadv = xadv + alpha * xadv.grad.sign()
            xadv = torch.min(torch.max(xadv, Xb - eps), Xb + eps).clamp(0, 1).detach()
        return xadv
    acc = lambda Xb: (net(Xb).argmax(1) == yte_t).float().mean().item()
    out = [f"hardened clean {acc(Xte_t):.3f}"]
    for eps in (0.1, 0.2, 0.3):
        torch.manual_seed(0)                                 # the same random start at every budget
        out.append(f"  eps {eps}: FGSM {acc(fgsm(Xte_t, eps)):.3f}, PGD {acc(pgd(Xte_t, eps, eps / 5, 10)):.3f}")
    return "\n".join(out)
print(fgsm_and_pgd())
# expected output: hardened clean 0.963
# expected output: eps 0.1: FGSM 0.793, PGD 0.789
# expected output: eps 0.2: FGSM 0.502, PGD 0.428
# expected output: eps 0.3: FGSM 0.111, PGD 0.035
# At epsilon 0.2, the budget the network was trained against, FGSM leaves 0.504 accuracy and PGD only 0.430; at 0.3
# the two give 0.128 and 0.046. A defense tested only against the attack it was trained with looks stronger than it is.

# %% [markdown]
# 9. How much poison a backdoor needs

# %%
# 9. How much poison a backdoor needs
digits = load_digits()
print(digits.data.shape)                                     # expected output: (1797, 64)

def backdoor():                                              # Example 29-12's backdoor at four poisoning budgets
    X = digits.images.reshape(len(digits.images), -1) / 16.0
    Xtr, Xte, ytr, yte = train_test_split(X, digits.target, test_size=0.3, random_state=0, stratify=digits.target)
    TARGET = 0
    def stamp(Xflat):                                        # trigger = bright top-left 2x2 corner
        Zi = Xflat.copy().reshape(-1, 8, 8)
        Zi[:, 0, 0] = Zi[:, 0, 1] = Zi[:, 1, 0] = Zi[:, 1, 1] = 1.0
        return Zi.reshape(len(Zi), -1)
    mask = yte != TARGET
    out = [f"training digits {len(Xtr)}, test digits other than 0: {int(mask.sum())}"]
    for k in (10, 30, 75, 150):
        pidx = np.random.default_rng(0).choice(len(Xtr), k, replace=False)   # a fresh generator for every budget
        Xp = np.vstack([Xtr, stamp(Xtr[pidx])]); yp = np.concatenate([ytr, np.full(k, TARGET)])
        bd = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=2).fit(Xp, yp)
        out.append(f"{k:3d} stamped copies: clean accuracy {bd.score(Xte, yte):.3f}, "
                   f"stamped digits other than 0 sent to 0: {(bd.predict(stamp(Xte[mask])) == TARGET).mean():.3f}")
    return "\n".join(out)
print(backdoor())
# expected output: training digits 1257, test digits other than 0: 486
# expected output: 10 stamped copies: clean accuracy 0.978, stamped digits other than 0 sent to 0: 0.002
# expected output: 30 stamped copies: clean accuracy 0.969, stamped digits other than 0 sent to 0: 0.193
# expected output: 75 stamped copies: clean accuracy 0.974, stamped digits other than 0 sent to 0: 0.998
# expected output: 150 stamped copies: clean accuracy 0.970, stamped digits other than 0 sent to 0: 1.000
# Clean accuracy stays between 0.969 and 0.972 at every budget, so a clean test cannot reveal the poison, while the
# share of stamped digits sent to 0 climbs from 0.002 with 10 copies to 0.189 with 30, 0.998 with 75, and 1.000 with
# 150; the 75 copies are 6.0 percent of the 1,257 training digits.

# %% [markdown]
# 10. Miniproject: the datasheet that accompanies the lab's model card

# %%
# 10. Miniproject: the datasheet that accompanies the lab's model card
import json
bc = load_breast_cancer()
print(bc.data.shape)                                         # expected output: (569, 30)

def datasheet():                                             # returns the datasheet as JSON text
    ds = {"dataset": "Breast Cancer Wisconsin (Diagnostic)",
          "source": "UCI Machine Learning Repository, Wolberg et al. (1993), https://doi.org/10.24432/C5DW2B",
          "collection": ("each record describes the cell nuclei in a digitized image of a sample drawn from "
                         "a breast mass with a fine needle")}
    ds["records"], ds["features"] = bc.data.shape            # composition, computed not asserted
    n_mal, n_ben = int((bc.target == 0).sum()), int((bc.target == 1).sum())
    ds["labeling"] = f"diagnosis label (malignant or benign): {n_mal} malignant, {n_ben} benign"
    ds["missing_values"] = int(np.isnan(bc.data).sum())
    ds["who_is_not_in_it"] = ("no demographic attributes recorded, so coverage across sex, age, or "
                              "ethnicity cannot be verified")
    ds["license_and_access"] = ("Creative Commons Attribution 4.0 International (CC BY 4.0) on the UCI page as "
                                "of November 2026; bundled with scikit-learn as load_breast_cancer")
    return json.dumps(ds, indent=2)                          # printed the way the lab printed its card
print(datasheet())
# expected output: {
# expected output: "dataset": "Breast Cancer Wisconsin (Diagnostic)",
# expected output: "source": "UCI Machine Learning Repository, Wolberg et al. (1993), https://doi.org/10.24432/C5DW2B",
# expected output: "collection": "each record describes the cell nuclei in a digitized image of a sample drawn from a breast mass with a fine needle",
# expected output: "records": 569,
# expected output: "features": 30,
# expected output: "labeling": "diagnosis label (malignant or benign): 212 malignant, 357 benign",
# expected output: "missing_values": 0,
# expected output: "who_is_not_in_it": "no demographic attributes recorded, so coverage across sex, age, or ethnicity cannot be verified",
# expected output: "license_and_access": "Creative Commons Attribution 4.0 International (CC BY 4.0) on the UCI page as of November 2026; bundled with scikit-learn as load_breast_cancer"
# expected output: }
# The datasheet records where the data came from, computes its composition from the data itself, and states plainly
# that no demographic attribute is recorded, the gap the lab's card had to list among its limitations.

# %% [markdown]
# ## 29.11 coding drills

# %%
import numpy as np
from collections import Counter

# %% [markdown]
# Fairness metrics

# %%
# Fairness metrics
y = np.array([1, 0, 1, 1, 0, 1, 1, 0]); pred = np.array([1, 0, 1, 1, 0, 0, 1, 0])
g = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
selA, selB = pred[g == "A"].mean(), pred[g == "B"].mean()
print(round(abs(selA - selB), 3))                                 # expected output: 0.5
tprA = pred[(g == "A") & (y == 1)].mean(); tprB = pred[(g == "B") & (y == 1)].mean()
print(round(abs(tprA - tprB), 3))                                 # expected output: 0.5
print(round(selB / selA, 3), selB / selA >= 0.8)                  # expected output: 0.333 False
# The ratio 0.333 is well below 0.800, so the four-fifths rule would flag this selection gap.
fprA = pred[(g == "A") & (y == 0)].mean(); fprB = pred[(g == "B") & (y == 0)].mean()
print(round(max(abs(tprA - tprB), abs(fprA - fprB)), 3))          # expected output: 0.5

# %% [markdown]
# Explainability

# %%
# Explainability
w = np.array([3.0, 1.5, 0.0]); base = np.array([0.0, 0.0, 0.0]); x = np.array([2.0, 1.0, 1.0])
phi = w * (x - base)                                              # exact Shapley for a linear model
print(phi.tolist())                                              # expected output: [6.0, 1.5, 0.0]
print(round(phi.sum(), 3), round(float(w @ x - w @ base), 3))    # expected output: 7.5 7.5

# %% [markdown]
# Privacy

# %%
# Privacy
print(round(1 / 0.5, 3))                                         # expected output: 2.0
quasi = [("1990", "021"), ("1990", "021"), ("1985", "100"), ("1990", "021")]   # (birth year, ZIP prefix)
print(dict(Counter(quasi)))                                     # expected output: {('1990', '021'): 3, ('1985', '100'): 1}
mem, non = np.array([0.98, 0.99, 0.97]), np.array([0.90, 0.88, 0.91])
print(mem.mean() > non.mean())                                  # expected output: True

# %% [markdown]
# Security

# %%
# Security
img = np.array([0.2, 0.9, 0.5]); grad = np.array([0.4, -0.3, 0.1]); eps = 0.1
print(np.clip(img + eps * np.sign(grad), 0, 1).round(3).tolist())  # expected output: [0.3, 0.8, 0.6]
orig = np.array([0.5, 0.5]); pert = np.array([0.9, 0.1]); e = 0.2
print(np.clip(pert, orig - e, orig + e).round(3).tolist())      # expected output: [0.7, 0.3]
tgt = np.array([1, 0, 2, 1]); sur = np.array([1, 0, 2, 2])
print(round((tgt == sur).mean(), 3))                            # expected output: 0.75
out = "Sure, PWNED"; print("pwned" in out.lower())              # expected output: True

# %% [markdown]
# Regulation and sustainability

# %%
# Regulation and sustainability
def tier(u):
    banned = {"social scoring"}; high = {"credit scoring", "medical diagnosis"}; limited = {"chatbot"}
    return "unacceptable" if u in banned else "high" if u in high else "limited" if u in limited else "minimal"
print(tier("credit scoring"), tier("spam filter"))             # expected output: high minimal
kwh = 1000
print(round(kwh * 0.40, 1), round(kwh * 1.8, 1))              # expected output: 400.0 1800.0
print(round(4380 / 130, 1))                                   # expected output: 33.7
print(round((0.3 - 0.15) / 1000 * 100_000_000 * 365 * 0.40 / 1000, 1))   # expected output: 2190.0

# %% [markdown]
# Frontier and staying current

# %%
# Frontier and staying current
f = lambda x: np.sin(1.5 * x)
xs = np.linspace(4, 8, 50); flat = np.full_like(xs, -1.0)      # a net that flattens past its training range
print(round(float(np.mean((flat - f(xs)) ** 2)), 3) > 1.0)    # expected output: True
interests = {"agents", "efficiency"}
papers = [("Tool use", {"agents", "tools"}), ("KV-Cache", {"efficiency"}), ("Protein", {"biology"})]
print(sorted(papers, key=lambda p: len(interests & p[1]), reverse=True)[0][0])   # expected output: Tool use
