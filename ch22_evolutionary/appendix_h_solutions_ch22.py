"""Appendix H solutions for Chapter 22, as runnable cells.

Generated from Solutions.md (Appendix H). Every code block of this chapter's solutions appears here
verbatim, split into one cell per exercise or drill topic, so each answer can be run and checked against
the expected outputs it states. Keep Solutions.md as the source; regenerate after any change.
"""

# %% [markdown]
# ## 22.11 coding exercises

# %%
import warnings; warnings.filterwarnings("ignore")
import numpy as np

def rastrigin(x):                                   # rugged test surface, global min 0 at the origin
    return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))

# %% [markdown]
# 1. When gradients fail, as the dimension grows: Example 22-1's comparison in 2, 5, 10, and 20 dimensions

# %%
# 1. When gradients fail, as the dimension grows: Example 22-1's comparison in 2, 5, 10, and 20 dimensions
def num_grad(f, x, eps=1e-5):
    g = np.zeros_like(x)
    for i in range(len(x)):
        d = np.zeros_like(x); d[i] = eps
        g[i] = (f(x + d) - f(x - d)) / (2 * eps)    # central finite difference
    return g
def compare(dim, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-5, 5, dim)                     # the start is drawn first, as in Example 22-1
    for _ in range(500): x -= 0.002 * num_grad(rastrigin, x)     # small lr: rolls into the nearest dip
    best = min((rng.uniform(-5, 5, dim) for _ in range(5000)), key=rastrigin)   # best of 5,000 guesses
    return rastrigin(x), rastrigin(best)
gd0, rs0 = compare(2, 0)
print(f"1. seed 0, 2-D: gradient descent {gd0:.2f}   random search {rs0:.2f}   (Example 22-1)")
for dim in (2, 5, 10, 20):
    runs = [compare(dim, s) for s in range(10)]
    gd, rs = np.mean([r[0] for r in runs]), np.mean([r[1] for r in runs])
    wins = sum(r[1] < r[0] for r in runs)
    print(f"1. {dim:2d}-D, 10 seeds: gradient descent mean {gd:6.2f}   random search mean {rs:6.2f}   "
          f"random wins {wins}/10")
# expected output: 1. seed 0, 2-D: gradient descent 4.97   random search 0.23   (Example 22-1)
# expected output: 1.  2-D, 10 seeds: gradient descent mean  15.72   random search mean   1.02   random wins 10/10
# expected output: 1.  5-D, 10 seeds: gradient descent mean  45.67   random search mean  18.98   random wins 10/10
# expected output: 1. 10-D, 10 seeds: gradient descent mean  87.85   random search mean  72.54   random wins 8/10
# expected output: 1. 20-D, 10 seeds: gradient descent mean 167.85   random search mean 206.43   random wins 1/10
# In 2 and 5 dimensions random search wins on every seed, but in 20 dimensions it wins on only 1 of 10.
# Gradient descent rolls each coordinate into its nearest dip, so its mean grows in step with d, about 8 to 9
# per coordinate in every dimension here. A low value from random sampling needs every coordinate near 0 at
# once, and the share of the box where that holds shrinks exponentially with d, so a fixed 5,000 samples fall
# behind. The chapter's 2-D win, 0.23 against 4.97, is a low-dimensional result, not a general one.

# %% [markdown]
# 2. Local search at equal budgets: Example 22-2's thirty starts, both methods at 1,000 and at 4,000 evaluations

# %%
# 2. Local search at equal budgets: Example 22-2's thirty starts, both methods at 1,000 and at 4,000 evaluations
def fresh():                                        # Example 22-2's stream: seed 0, the thirty starts drawn first
    rng = np.random.default_rng(0)
    return rng, [rng.uniform(-5, 5, 2) for _ in range(30)]
def neighbor(rng, x): return np.clip(x + rng.normal(0, 0.3, len(x)), -5.12, 5.12)
def hill_mean(n_evals):
    rng, starts = fresh(); finals = []
    for s in starts:
        x = s.copy(); fx = rastrigin(x)
        for _ in range(n_evals):
            c = neighbor(rng, x); fc = rastrigin(c)
            if fc < fx: x, fx = c, fc               # accept ONLY improvements
        finals.append(fx)
    return np.mean(finals)
def anneal_mean(n_evals, T0=3.0):
    rng, starts = fresh(); finals = []
    cool = 0.999 ** (4000 / n_evals)                # both budgets end at T = 3 * 0.999**4000, about 0.055
    for s in starts:
        x = s.copy(); fx = rastrigin(x); best = fx; T = T0
        for _ in range(n_evals):
            c = neighbor(rng, x); fc = rastrigin(c); d = fc - fx
            if d < 0 or rng.random() < np.exp(-d / T):  # accept a worse move with prob exp(-d/T)
                x, fx = c, fc; best = min(best, fx)
            T *= cool
        finals.append(best)
    return np.mean(finals)
for n in (1000, 4000):
    print(f"2. {n} evaluations: hill climbing mean f = {hill_mean(n):.2f}   annealing mean best f = {anneal_mean(n):.2f}")
# expected output: 2. 1000 evaluations: hill climbing mean f = 7.91   annealing mean best f = 6.73
# expected output: 2. 4000 evaluations: hill climbing mean f = 2.91   annealing mean best f = 1.47
# Annealing returns the best value each run visited, as Example 22-2 does, so its line says mean best f; hill
# climbing accepts only improvements, so its final value is also its best.
# At equal budgets annealing still wins: 6.73 against 7.91 at 1,000 evaluations, and 1.47 against 2.91 at
# 4,000. But hill climbing alone falls from 7.91 to 2.91 when it gets 4,000 evaluations, so most of the
# chapter's gap of 7.91 against 1.93 came from the fourfold budget. Annealing's 1.47 differs from the chapter's
# 1.93 because its moves here come from a fresh stream seeded with 0, while in Example 22-2 earlier runs had
# already used part of that stream.

# %% [markdown]
# 3. Does crossover pull its weight? Example 22-6's GA with and without crossover, in 5 and 10 dimensions

# %%
# 3. Does crossover pull its weight? Example 22-6's GA with and without crossover, in 5 and 10 dimensions
DIM = 5                                             # exercises 7 and 9 below reuse DIM and rast_batch
def rast_batch(X): return 10 * X.shape[-1] + np.sum(X ** 2 - 10 * np.cos(2 * np.pi * X), axis=-1)
def run_ga(dim, seed, crossover=True, POP=120, GENS=80, ELITE=4):
    rng = np.random.default_rng(seed)
    pop = rng.uniform(-5.12, 5.12, (POP, dim))
    def tournament(fit, k=3):
        idx = rng.integers(0, POP, k); return idx[np.argmin(fit[idx])]
    for gen in range(GENS):
        fit = rast_batch(pop); order = np.argsort(fit)
        new = [pop[order[i]].copy() for i in range(ELITE)]           # elitism
        while len(new) < POP:
            a, b = pop[tournament(fit)], pop[tournament(fit)]
            if crossover:
                al = rng.uniform(-0.25, 1.25, dim)
                child = al * a + (1 - al) * b                        # BLX blend crossover
            else:
                child = a.copy()                                     # crossover removed: copy one winner
            child += rng.normal(0, 0.3, dim) * (rng.random(dim) < 0.3)  # Gaussian mutation, 30% of genes
            new.append(np.clip(child, -5.12, 5.12))
        pop = np.array(new)
    return rast_batch(pop).min()
print(f"3. seed 0, 5-D, with crossover: best f = {run_ga(5, 0):.3f}   (Example 22-6)")
for dim in (5, 10):
    with_cx = [run_ga(dim, s) for s in range(5)]
    without = [run_ga(dim, s, crossover=False) for s in range(5)]
    better = sum(a < b for a, b in zip(with_cx, without))        # same seed, with against without
    print(f"3. {dim:2d}-D, 5 seeds: with crossover mean {np.mean(with_cx):.3f}   "
          f"without crossover mean {np.mean(without):.3f}   crossover better on {better}/5 seeds")
# expected output: 3. seed 0, 5-D, with crossover: best f = 0.016   (Example 22-6)
# expected output: 3.  5-D, 5 seeds: with crossover mean 0.223   without crossover mean 3.893   crossover better on 5/5 seeds
# expected output: 3. 10-D, 5 seeds: with crossover mean 5.193   without crossover mean 24.922   crossover better on 5/5 seeds
# Removing crossover makes this GA about 17 times worse in 5 dimensions (3.893 against 0.223) and about 5
# times worse in 10 (24.922 against 5.193), and crossover wins on every seed in both. Blend crossover can give
# a child good coordinates from two different parents, and Rastrigin is a sum of one-coordinate terms, so good
# coordinates found separately can be combined: the building-block account of Section 22.3 at work. The test
# shows that crossover helps this GA on this surface; it does not prove the hypothesis in general, and a
# stronger mutation was not tried as a substitute.

# %% [markdown]
# 4. Shape adaptation on a turned valley: Example 22-8's two valley strategies, then pycma

# %%
# 4. Shape adaptation on a turned valley: Example 22-8's two valley strategies, then pycma
def valley(X): return X[..., 0] ** 2 + 100 * X[..., 1] ** 2       # Example 22-8's valley
def turned(X):                                      # the same valley turned 45 degrees: its floor is x0 = x1
    s = (X[..., 0] + X[..., 1]) / np.sqrt(2); t = (X[..., 0] - X[..., 1]) / np.sqrt(2)
    return s ** 2 + 100 * t ** 2
def es(f, start, per_axis, seed, gens=60, lam=40, mu=10):
    rng = np.random.default_rng(seed)
    m = np.array(start, float); sig = np.array([2.0, 2.0])
    for g in range(gens):
        kids = m + sig * rng.normal(0, 1, (lam, 2))
        elite = kids[np.argsort(f(kids))[:mu]]; m = elite.mean(0)
        if per_axis: sig = 0.5 * sig + 0.5 * (elite.std(0) + 1e-9)   # one step size per coordinate axis
        else:
            step = 0.5 * sig.mean() + 0.5 * elite.std(0).mean(); sig = np.array([step, step])   # isotropic
    return f(m[None])[0]
start_turned = (8 * np.sqrt(2), 0.0)                # (8, 8) turned with the valley: g there is 6464 = f(8, 8)
for name, f, start in (("original", valley, (8.0, 8.0)), ("turned", turned, start_turned)):
    iso = [es(f, start, False, s) for s in range(10)]
    per = [es(f, start, True, s) for s in range(10)]
    print(f"4. {name:8s} valley, 10 seeds: isotropic median {np.median(iso):.2f}   "
          f"per-axis median {np.median(per):.4f} ({sum(v < 1e-3 for v in per)} of 10 below 0.001)")
import cma                                          # pycma, version 4.5.0 in the reference environment
es_c = cma.CMAEvolutionStrategy(list(start_turned), 2.0, {"seed": 1, "verbose": -9})
es_c.optimize(lambda z: float(turned(np.asarray(z))))
C = es_c.C                                          # pycma's learned covariance shape
print(f"4. pycma on the turned valley: f = {es_c.result.fbest:.2e} after {es_c.result.evaluations} evaluations, "
      f"correlation of x0 and x1 in its covariance {C[0, 1] / np.sqrt(C[0, 0] * C[1, 1]):.3f}")
# expected output: 4. original valley, 10 seeds: isotropic median 19.19   per-axis median 0.0000 (8 of 10 below 0.001)
# expected output: 4. turned   valley, 10 seeds: isotropic median 16.12   per-axis median 17.8585 (0 of 10 below 0.001)
# expected output: 4. pycma on the turned valley: f = 4.63e-17 after 588 evaluations, correlation of x0 and x1 in its covariance 0.989
# On the original valley the long and short directions are the coordinate axes, so one step size per axis
# fits both, and the per-axis strategy wins (median 0.0000, 8 of 10 seeds below 0.001). Turned by 45 degrees,
# the long direction is the diagonal x0 = x1, which an axis-aligned sampling ellipse cannot follow, so the
# per-axis strategy is no better than the isotropic one (medians 17.8585 and 16.12, no run below 0.001). The
# isotropic strategy samples a circle, which looks the same after a turn, so its medians are similar on both
# valleys (19.19 and 16.12). pycma adapts a full covariance matrix: the correlation of 0.989 between x0 and x1
# in it shows its ellipse tilted onto the diagonal, and it reaches 4.63e-17 after 588 evaluations (pycma 4.5.0).

# %% [markdown]
# 5. Genetic programming and the function set: fit x + sin(2x) with and without a sine operator

# %%
# 5. Genetic programming and the function set: fit x + sin(2x) with and without a sine operator
xs = np.linspace(-3, 3, 40); ys = xs + np.sin(2 * xs)
ARITY = {"+": 2, "-": 2, "*": 2, "sin": 1}
def gp(ops, seed=1, POP=300, GENS=60, LAM=0.001):   # Example 22-11's parsimony run, any function set
    rng = np.random.default_rng(seed)
    def rtree(depth):
        if depth == 0 or (depth < 3 and rng.random() < 0.3):
            return ("x",) if rng.random() < 0.6 else ("c", round(rng.uniform(-3, 3), 1))
        op = str(rng.choice(ops))
        return (op,) + tuple(rtree(depth - 1) for _ in range(ARITY[op]))   # sin gets one child
    def ev(n, x):
        if n[0] == "x": return x
        if n[0] == "c": return np.full_like(x, n[1])
        if n[0] == "sin": return np.sin(ev(n[1], x))
        a, b = ev(n[1], x), ev(n[2], x)
        return a + b if n[0] == "+" else a - b if n[0] == "-" else a * b
    def to_str(n):
        if n[0] == "x": return "x"
        if n[0] == "c": return str(n[1])
        if n[0] == "sin": return f"sin({to_str(n[1])})"
        return f"({to_str(n[1])} {n[0]} {to_str(n[2])})"
    def size(n): return 1 if n[0] in ("x", "c") else 1 + sum(size(k) for k in n[1:])
    def mse(n):
        with np.errstate(all="ignore"):
            e = np.mean((ev(n, xs) - ys) ** 2)
        return e if np.isfinite(e) else 1e9
    def cost(n): return mse(n) + LAM * size(n)                   # parsimony pressure, as in Example 22-11
    def paths(n, p=()):                                          # the address of every node in the tree
        yield p
        if n[0] not in ("x", "c"):
            for i in range(1, len(n)): yield from paths(n[i], p + (i,))
    def get(n, p):
        for i in p: n = n[i]
        return n
    def put(n, p, new):
        if not p: return new
        kids = list(n); kids[p[0]] = put(n[p[0]], p[1:], new); return tuple(kids)
    def crossover(a, b):                                         # graft a random subtree of b into a
        pa, pb = list(paths(a)), list(paths(b))
        return put(a, pa[rng.integers(len(pa))], get(b, pb[rng.integers(len(pb))]))
    def mutate(a):
        pa = list(paths(a)); return put(a, pa[rng.integers(len(pa))], rtree(2))
    pop = [rtree(3) for _ in range(POP)]; first_exact = None
    for gen in range(GENS):
        pop.sort(key=cost)
        if first_exact is None and mse(pop[0]) < 1e-12: first_exact = gen
        new = pop[:6]                                            # elitism
        while len(new) < POP:
            a = min(rng.choice(len(pop), 4), key=lambda i: cost(pop[i]))
            b = min(rng.choice(len(pop), 4), key=lambda i: cost(pop[i]))
            child = crossover(pop[a], pop[b])
            if rng.random() < 0.3: child = mutate(child)
            new.append(child)
        pop = new
    pop.sort(key=cost)
    return to_str(pop[0]), mse(pop[0]), size(pop[0]), first_exact
for label, ops in (("{+, -, *}", ["+", "-", "*"]), ("{+, -, *, sin}", ["+", "-", "*", "sin"])):
    formula, err, nodes, first = gp(ops)
    found = f"{formula}, exact from generation {first}" if first is not None else "never exact"
    print(f"5. {label}: MSE = {err:.5f}, size = {nodes} nodes, {found}")
# expected output: 5. {+, -, *}: MSE = 0.30543, size = 41 nodes, never exact
# expected output: 5. {+, -, *, sin}: MSE = 0.00000, size = 6 nodes, (x + sin((x + x))), exact from generation 3
# With {+, -, *} every tree is a polynomial in x, so no tree equals x + sin(2x) as a function; the run settles
# on a 41-node polynomial with MSE 0.30543. With sin in the set, the exact answer is the six-node tree
# x + sin(x + x), and the run finds it at generation 3. Example 22-11's target was itself a polynomial, which is
# why {+, -, *} sufficed there. The function set decides which formulas exist in the search space at all; the
# search only chooses among them.

# %% [markdown]
# 6. Neuroevolution: evolve a classifier's weights on the real Iris dataset (no backprop)

# %%
# 6. Neuroevolution: evolve a classifier's weights on the real Iris dataset (no backprop)
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
rng = np.random.default_rng(0)
Xi, yi = load_iris(return_X_y=True)                             # 150 flowers, 4 features, 3 classes
Xtr, Xte, ytr, yte = train_test_split(Xi, yi, test_size=0.3, random_state=0, stratify=yi)
sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
NF, NCLS = 4, 3; NWT = NF * NCLS + NCLS
def predict(w, X):
    W, b = w[:NF * NCLS].reshape(NF, NCLS), w[NF * NCLS:]
    return np.argmax(X @ W + b, axis=1)                         # linear softmax scores
def acc(w, X, y): return np.mean(predict(w, X) == y)
POP, GENS, MU, SIG = 40, 60, 8, 0.4
popw = rng.normal(0, 1, (POP, NWT))
for gen in range(GENS):
    fits = np.array([acc(w, Xtr, ytr) for w in popw])           # fitness = training accuracy
    elite = popw[np.argsort(-fits, kind="stable")[:MU]]    # best first; tied accuracies keep population order
    popw = np.vstack([elite, [elite[rng.integers(MU)] + rng.normal(0, SIG, NWT)
                              for _ in range(POP - MU)]])
best = popw[np.argmax([acc(w, Xtr, ytr) for w in popw])]
grad = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
ev_ok, lr_ok = int((predict(best, Xte) == yte).sum()), int((grad.predict(Xte) == yte).sum())   # correct flowers
print(f"6. evolved test acc = {ev_ok / len(yte):.3f} ({ev_ok} of {len(yte)})   "
      f"gradient (LogReg) test acc = {lr_ok / len(yte):.3f} ({lr_ok} of {len(yte)})")
# expected output: 6. evolved test acc = 0.956 (43 of 45)   gradient (LogReg) test acc = 0.978 (44 of 45)
# The test split holds 45 flowers, so the two accuracies are 43 and 44 correct, a one-flower difference that
# favors the gradient method. Both are linear models and both nearly solve this problem; the run does not show
# that one way of training wins. Training accuracy ties often, so the elite is ranked with a stable sort, which
# keeps tied candidates in population order and picks the same parents on every machine (Section 22.3.2).

# %% [markdown]
# 7. Particle swarm inertia: Example 22-14's swarm with w = 0.4, 0.7, and 0.9

# %%
# 7. Particle swarm inertia: Example 22-14's swarm with w = 0.4, 0.7, and 0.9
def pso(w, seed, NPS=40, ITERS=200, c1=1.5, c2=1.5):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-5, 5, (NPS, DIM)); v = rng.normal(0, 1, (NPS, DIM))
    pbest = x.copy(); pbest_f = rast_batch(x)
    gbest = pbest[pbest_f.argmin()].copy(); gbest_f = pbest_f.min()
    for it in range(ITERS):
        r1, r2 = rng.random((NPS, DIM)), rng.random((NPS, DIM))
        v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)   # inertia + cognitive + social
        x = np.clip(x + v, -5.12, 5.12); f = rast_batch(x)
        imp = f < pbest_f; pbest[imp] = x[imp]; pbest_f[imp] = f[imp]
        if pbest_f.min() < gbest_f:
            gbest = pbest[pbest_f.argmin()].copy(); gbest_f = pbest_f.min()
    spread = np.linalg.norm(x - x.mean(0), axis=1).mean()          # mean distance from the swarm's centroid
    return gbest_f, spread
print(f"7. seed 0, w = 0.7: best f = {pso(0.7, 0)[0]:.3f}   (Example 22-14)")
for w in (0.4, 0.7, 0.9):
    runs = [pso(w, s) for s in range(10)]
    print(f"7. w = {w}: mean best f = {np.mean([r[0] for r in runs]):.3f}   "
          f"mean final spread = {np.mean([r[1] for r in runs]):.3f}")
# expected output: 7. seed 0, w = 0.7: best f = 0.995   (Example 22-14)
# expected output: 7. w = 0.4: mean best f = 3.084   mean final spread = 0.031
# expected output: 7. w = 0.7: mean best f = 1.095   mean final spread = 0.169
# expected output: 7. w = 0.9: mean best f = 7.409   mean final spread = 4.681
# Low inertia (w = 0.4) lets the swarm contract onto whatever basin it finds: final spread 0.031, mean best
# 3.084. High inertia (w = 0.9) keeps the particles moving, so the swarm never settles: spread 4.681, mean best
# 7.409. w = 0.7 sits between, with spread 0.169 and the best mean, 1.095. Inertia is the explore-exploit dial
# of PSO: too little exploits too soon, and too much never exploits at all.

# %% [markdown]
# 8. Ant colony optimization on a TSP over real US state capitals (lat/lon, planar)

# %%
# 8. Ant colony optimization on a TSP over real US state capitals (lat/lon, planar)
rng = np.random.default_rng(0)
# twelve state capitals, GeoNames cities15000 coordinates rounded to two decimals (the source of Section 22.7)
CAP = {"Sacramento": (38.58, -121.49), "Phoenix": (33.45, -112.07), "Denver": (39.74, -104.98),
       "Austin": (30.27, -97.74), "Des Moines": (41.60, -93.61), "Springfield": (39.80, -89.64),
       "Nashville": (36.17, -86.78), "Atlanta": (33.75, -84.39), "Columbus": (39.96, -83.00),
       "Raleigh": (35.77, -78.64), "Harrisburg": (40.27, -76.88), "Boston": (42.36, -71.06)}
coords = np.array(list(CAP.values())); NCC = len(CAP)
Dm = np.sqrt(((coords[:, None] - coords[None, :]) ** 2).sum(2)) + np.eye(NCC)
def tlen(t): return sum(Dm[t[i], t[(i + 1) % NCC]] for i in range(NCC))
def nn_tour():
    unv = set(range(1, NCC)); t = [0]
    while unv:
        nx = min(unv, key=lambda c: Dm[t[-1], c]); t.append(nx); unv.discard(nx)
    return tlen(t)
eta = 1 / Dm; tau = np.ones((NCC, NCC)) * 0.1
alpha, beta, rho, ANTS = 1.0, 3.0, 0.5, 20
best_len = np.inf
for it in range(60):
    tours = []
    for _ in range(ANTS):
        s = rng.integers(NCC); t = [s]; unv = set(range(NCC)) - {s}
        while unv:
            i = t[-1]; js = list(unv)
            wt = (tau[i, js] ** alpha) * (eta[i, js] ** beta)   # prob ~ pheromone^a * heuristic^b
            nx = js[rng.choice(len(js), p=wt / wt.sum())]; t.append(nx); unv.discard(nx)
        tours.append(t); best_len = min(best_len, tlen(t))
    tau *= (1 - rho)                                            # evaporation
    for t in tours:
        L = tlen(t)
        for i in range(NCC):
            a, b = t[i], t[(i + 1) % NCC]; tau[a, b] += 1.0 / L; tau[b, a] += 1.0 / L
print(f"8. nearest-neighbor = {nn_tour():.2f}   ACO best = {best_len:.2f}   ACO wins: {best_len <= nn_tour()}")
# expected output: 8. nearest-neighbor = 127.79   ACO best = 109.62   ACO wins: True
# The colony's best tour, 109.62, is (127.79 - 109.62) / 127.79 = 14.2 percent shorter than the greedy tour.

# %% [markdown]
# 9. Differential evolution's two knobs: Example 22-16's DE for nine settings of F and CR, against random search

# %%
# 9. Differential evolution's two knobs: Example 22-16's DE for nine settings of F and CR, against random search
BUDGET = 15000                                      # Example 22-16's budget and population, on 5-D Rastrigin
def de(f, rng, pop=40, F=0.5, CR=0.9):              # Example 22-16's loop, with the generator passed in
    P = rng.uniform(-5, 5, (pop, DIM)); fit = np.array([f(p) for p in P]); evals = pop
    while evals < BUDGET:
        for i in range(pop):
            idxs = rng.choice([j for j in range(pop) if j != i], 3, replace=False)
            a, b, c = P[idxs]; m = np.clip(a + F * (b - c), -5.12, 5.12)   # v = a + F(b - c)
            mask = rng.random(DIM) < CR; trial = np.where(mask, m, P[i])  # binomial crossover
            ft = f(trial); evals += 1
            if ft < fit[i]: P[i], fit[i] = trial, ft               # greedy selection
        if evals >= BUDGET: break
    return fit.min()
for F in (0.3, 0.5, 0.8):
    cells = []
    for CR in (0.1, 0.5, 0.9):
        bests = [de(rastrigin, np.random.default_rng(s), F=F, CR=CR) for s in range(3)]   # seeds 0, 1, 2
        cells.append(f"CR {CR}: {np.mean(bests):.3f} ({sum(v < 1e-3 for v in bests)}/3 below 0.001)")
    print(f"9. F = {F}:  " + "   ".join(cells))
def random_search(f, rng): return min(f(rng.uniform(-5, 5, DIM)) for _ in range(BUDGET))
rs = [random_search(rastrigin, np.random.default_rng(s)) for s in range(3)]
print(f"9. random search, same budget and seeds: mean best f = {np.mean(rs):.3f}")
# expected output: 9. F = 0.3:  CR 0.1: 0.000 (3/3 below 0.001)   CR 0.5: 0.000 (3/3 below 0.001)   CR 0.9: 0.332 (2/3 below 0.001)
# expected output: 9. F = 0.5:  CR 0.1: 0.000 (3/3 below 0.001)   CR 0.5: 0.000 (3/3 below 0.001)   CR 0.9: 0.000 (3/3 below 0.001)
# expected output: 9. F = 0.8:  CR 0.1: 0.000 (3/3 below 0.001)   CR 0.5: 0.000 (3/3 below 0.001)   CR 0.9: 2.722 (0/3 below 0.001)
# expected output: 9. random search, same budget and seeds: mean best f = 13.279
# Seven of the nine settings end below 0.001 on all three seeds, Example 22-16's F = 0.5 and CR = 0.9 among them.
# The two failures both have CR = 0.9: F = 0.3 misses on one seed (mean 0.332) and F = 0.8 on all three
# (mean 2.722). Random search under the same 15,000 evaluations ends at 13.279. CR is the chance that each
# coordinate of the trial comes from the mutant. At CR = 0.1 a trial takes no coordinate from it with probability
# 0.9^5 = 0.590, an evaluation spent on a copy of the target, and exactly one with probability
# 5 x 0.1 x 0.9^4 = 0.328, so a trial that changes anything changes one coordinate 0.328 / 0.410 = 80 percent of
# the time. Rastrigin is a sum of one-coordinate terms, so a change of one coordinate is kept exactly when that
# coordinate's own term falls, and the other four are left alone. At CR = 0.9 a trial takes all five coordinates
# from the mutant with probability 0.9^5 = 0.590, so it is kept only when the whole new vector scores better.

# %% [markdown]
# 10. Miniproject: GA feature selection on real breast-cancer data + accuracy/size tradeoff

# %%
# 10. Miniproject: GA feature selection on real breast-cancer data + accuracy/size tradeoff
from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
rng = np.random.default_rng(0)
data = load_breast_cancer(); Xb, yb = data.data, data.target; Dd = Xb.shape[1]   # 569 x 30
def model(): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
def cvacc(mask): return cross_val_score(model(), Xb[:, mask], yb, cv=3).mean()
def fitness(mask):
    if mask.sum() == 0: return 0.0
    return cvacc(mask) - 0.002 * mask.sum()                      # accuracy minus a size penalty
baseline = cross_val_score(model(), Xb, yb, cv=3).mean()
POP, GENS = 20, 12
pop = [rng.random(Dd) < 0.5 for _ in range(POP)]
archive = {}                                                    # best accuracy seen at each subset size
for gen in range(GENS):
    fits = np.array([fitness(m) for m in pop])
    for m in pop:
        k = int(m.sum()); a = cvacc(m) if m.sum() else 0.0
        if k not in archive or a > archive[k]: archive[k] = a
    pop = [pop[i] for i in np.argsort(-fits, kind="stable")]   # best first; equal scores keep population order
    new = pop[:4]                                               # elitism
    while len(new) < POP:
        a, b = pop[rng.integers(8)], pop[rng.integers(8)]
        child = np.where(rng.random(Dd) < 0.5, a, b)            # uniform crossover
        child = np.where(rng.random(Dd) < 0.05, ~child, child)  # ~5% mutation
        new.append(child)
    pop = new
best = pop[int(np.argmax([fitness(m) for m in pop]))]
print(f"10. baseline (all {Dd}) = {baseline:.3f}   GA kept {int(best.sum())} features -> acc {cvacc(best):.3f}")
top = max(archive.values()); top_k = min(k for k, a in archive.items() if a == top)   # best accuracy and its size
cheapest = min(k for k, a in archive.items() if a >= top - 0.005)   # smallest subset within 0.005 of the best
print(f"    best archived acc {top:.5f} at {top_k} features; threshold {top - 0.005:.5f}")
print("    archive (size: best acc): " + ", ".join(f"{k}: {archive[k]:.5f}" for k in sorted(archive)))
print(f"    tradeoff: {cheapest} features already reach acc {archive[cheapest]:.3f}")
# expected output: 10. baseline (all 30) = 0.975   GA kept 7 features -> acc 0.975
# expected output:     best archived acc 0.98067 at 14 features; threshold 0.97567
# expected output:     archive (size: best acc): 7: 0.97538, 8: 0.97538, 9: 0.97892, 10: 0.97891, 11: 0.97891, 12: 0.97891, 13: 0.98066, 14: 0.98067, 15: 0.97892, 16: 0.98067, 17: 0.97891
# expected output:     tradeoff: 9 features already reach acc 0.979
# The GA keeps 7 features at 0.975, level with the all-30 baseline of 0.975. The archive is the tradeoff sketch:
# the best accuracy seen at each size is 0.97538 at 7 and 8 features, about 0.9789 from 9 to 12, 0.98066 at 13,
# and 0.98067, the best, at 14 and again at 16, while 15 and 17 fall back to about 0.9789. The threshold is
# 0.98067 - 0.005 = 0.97567, so 7 and 8 features fall short by 0.0003 and 9 is the smallest size within it.
# The GA still prefers its 7, because the penalty charges 0.002 per feature:
# 0.97538 - 7 x 0.002 = 0.96138 beats 0.97892 - 9 x 0.002 = 0.96092. All these scores are cross-validated on the
# same tumors the GA selected on, so they flatter the chosen subsets; Section 22.9 holds out test tumors for the
# honest comparison.

# %% [markdown]
# 11. Schema bounds

# %%
# 11. Schema bounds
def schema_bound(delta, order, m=30, ratio=1.2, pc=0.7, pm=0.02, ell=8):
    bracket = 1 - pc * delta / (ell - 1) - order * pm
    return bracket, m * ratio * bracket
for name, delta in (("1******0", 7), ("**10****", 1)):
    bracket, bound = schema_bound(delta, 2)
    print(f"11. {name} bracket {bracket:.2f} bound {bound:.2f}")
# expected output: 11. 1******0 bracket 0.26 bound 9.36
# expected output: 11. **10**** bracket 0.86 bound 30.96
# The bound favors **10****. Its two fixed genes are adjacent (defining length 1), so only one of the seven cut
# points separates them, the bracket is 1 - 0.7 x 1/7 - 2 x 0.02 = 0.86, and 30 x 1.2 x 0.86 = 30.96 exceeds the
# 30 copies it started with, a growing share. The fixed genes of 1******0 sit at the two ends (defining length
# 7), so every cut separates them, the bracket falls to 1 - 0.7 x 7/7 - 0.04 = 0.26, and the bound shrinks to
# 30 x 1.2 x 0.26 = 9.36. Both patterns have order 2 and the same fitness ratio; only the defining length differs.

# %% [markdown]
# 12. Protected division

# %%
# 12. Protected division
def protected_div(a, b):
    out = np.ones_like(a, dtype=float)
    ok = np.abs(b) >= 1e-8
    out[ok] = a[ok] / b[ok]
    return out
OPS = ["+", "-", "*", "/"]
def random_tree(rng, depth):
    if depth == 0 or (depth < 3 and rng.random() < 0.3):
        return ("x",) if rng.random() < 0.6 else ("c", round(float(rng.uniform(-3, 3)), 1))
    return (str(rng.choice(OPS)), random_tree(rng, depth - 1), random_tree(rng, depth - 1))
def ev(node, x, protect):
    if node[0] == "x":
        return x
    if node[0] == "c":
        return np.full_like(x, node[1], dtype=float)
    a, b = ev(node[1], x, protect), ev(node[2], x, protect)
    if node[0] == "+":
        return a + b
    if node[0] == "-":
        return a - b
    if node[0] == "*":
        return a * b
    return protected_div(a, b) if protect else a / b
xs = np.linspace(-3, 3, 40)
def count(protect):
    rng = np.random.default_rng(0)
    bad = 0
    for _ in range(300):
        with np.errstate(divide="ignore", invalid="ignore"):
            val = ev(random_tree(rng, 3), xs, protect)
        if not np.all(np.isfinite(val)):
            bad += 1
    return bad
print(f"12. unprotected nonfinite {count(False)} of 300; protected {count(True)} of 300")
# expected output: 12. unprotected nonfinite 16 of 300; protected 0 of 300

# %% [markdown]
# 13. CMA step with new draws

# %%
# 13. CMA step with new draws
m = np.array([4.0, 0.0])
ys = [np.array(v, float) for v in [(-1, 0.2), (2, 0), (0, -1), (0.5, 0.5)]]
fs = [(m + y)[0] ** 2 + 100 * (m + y)[1] ** 2 for y in ys]
winners = [ys[i] for i in np.argsort(fs)[:2]]
mp = m + 0.5 * winners[0] + 0.5 * winners[1]
rank = 0.5 * np.outer(winners[0], winners[0]) + 0.5 * np.outer(winners[1], winners[1])
C = 0.6 * np.eye(2) + 0.4 * rank
ps = np.sqrt(0.5 * 1.5 * 2.0) * (mp - m)
sigp = float(np.exp(0.5 * (np.linalg.norm(ps) / np.sqrt(np.pi / 2) - 1)))
print(f"13. m' {np.round(mp, 4).tolist()} C' {np.round(C, 4).tolist()} sigma' {sigp:.4f}")
# expected output: 13. m' [4.5, 0.1] C' [[1.6, -0.04], [-0.04, 0.608]] sigma' 0.7781

# %% [markdown]
# ## 22.11 coding drills

# %%
import numpy as np

# %% [markdown]
# Test functions and landscapes

# %%
# Test functions and landscapes
rastrigin = lambda x: 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))
print(round(float(rastrigin(np.array([0.0, 0.0]))), 1))     # expected output: 0.0
print(round(float(rastrigin(np.array([1.0, 1.0]))), 1))     # expected output: 2.0
print(float(np.sum(np.array([3.0, 4.0]) ** 2)))             # expected output: 25.0
print(float(np.sum(np.floor(np.abs(np.array([2.6, -1.2]))) ** 2)))  # expected output: 5.0
def num_grad(f, x, eps=1e-5):
    g = np.zeros_like(x)
    for i in range(len(x)):
        d = np.zeros_like(x); d[i] = eps; g[i] = (f(x + d) - f(x - d)) / (2 * eps)
    return g
stair = lambda x: np.sum(np.floor(np.abs(x)) ** 2)
print(num_grad(stair, np.array([2.6, -1.2])).tolist())     # expected output: [0.0, 0.0]

# %% [markdown]
# Local search

# %%
# Local search
for T in [10.0, 2.0, 0.5]:
    print(T, round(float(np.exp(-3 / T)), 3))
# expected output: 10.0 0.741
# expected output: 2.0 0.223
# expected output: 0.5 0.002
rng = np.random.default_rng(0)
print(np.round(np.clip(np.array([2.0, -1.0]) + rng.normal(0, 0.3, 2), -5.12, 5.12), 3).tolist())
# expected output: [2.038, -1.04]

# %% [markdown]
# Genetic operators

# %%
# Genetic operators
p1 = [1, 0, 1, 0, 1, 0]; p2 = [0, 1, 0, 1, 0, 1]
print(p1[:2] + p2[2:])                                     # expected output: [1, 0, 0, 1, 0, 1]
rng = np.random.default_rng(3); mask = rng.random(6) < 0.5   # True takes the gene from the first parent
print(mask.tolist(), [p1[i] if mask[i] else p2[i] for i in range(6)])
# expected output: [True, True, False, False, True, True] [1, 0, 0, 1, 1, 0]
a, b, al = np.array([0.0, 0.0]), np.array([4.0, 4.0]), 0.3
print((al * a + (1 - al) * b).tolist())                    # expected output: [2.8, 2.8]
def order_crossover(p1, p2, lo, hi, n):
    child = [-1] * n; child[lo:hi] = p1[lo:hi]
    fill = [c for c in p2 if c not in child[lo:hi]]; j = 0
    for i in list(range(hi, n)) + list(range(lo)):
        child[i] = fill[j]; j += 1
    return child
print(order_crossover([0, 1, 2, 3, 4], [4, 3, 2, 1, 0], 1, 3, 5))  # expected output: [0, 1, 2, 4, 3]

# %% [markdown]
# Swarm methods

# %%
# Swarm methods
xp, vp = np.array([1.0, 1.0]), np.array([0.5, -0.5])
pb, gb = np.array([0.8, 0.9]), np.array([0.0, 0.0])
print(np.round(0.7 * vp + 1.5 * 0.5 * (pb - xp) + 1.5 * 0.5 * (gb - xp), 3).tolist())
# expected output: [-0.55, -1.175]
tau, eta = np.array([0.2, 0.5, 0.1]), np.array([1.0, 2.0, 0.5])
w = (tau ** 1) * (eta ** 3); print(np.round(w / w.sum(), 3).tolist())
# expected output: [0.047, 0.95, 0.003]
print(np.round(np.array([0.2, 0.5, 0.1]) * (1 - 0.5), 3).tolist())
# expected output: [0.1, 0.25, 0.05]

# %% [markdown]
# Multiobjective and differential evolution

# %%
# Multiobjective and differential evolution
dominates = lambda p, q: bool(np.all(p <= q) and np.any(p < q))
print(dominates(np.array([3, 4]), np.array([3, 5])), dominates(np.array([3, 4]), np.array([2, 6])),
      dominates(np.array([2, 6]), np.array([3, 4])))       # expected output: True False False
a, b, c = np.array([0.0, 2.0]), np.array([3.0, 1.0]), np.array([1.0, -1.0])
print((a + 0.5 * (b - c)).tolist())                        # expected output: [1.0, 3.0]
one_fifth = lambda sr, sigma, c=1.2: sigma * c if sr > 0.2 else sigma / c
print(round(one_fifth(0.25, 2.0), 3), round(one_fifth(0.15, 2.0), 3))
# expected output: 2.4 1.667

# %% [markdown]
# Schema

# %%
# Schema
schema = ['*', '1', '*', '0', '1']
fixed = [i for i, s in enumerate(schema) if s != '*']
print(len(fixed), fixed[-1] - fixed[0])                    # expected output: 3 3
