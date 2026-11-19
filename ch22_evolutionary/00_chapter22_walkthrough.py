"""Chapter 22: Evolutionary & Swarm Intelligence: Optimization Without Gradients

Section-aligned walkthrough: every code block from this chapter of the book,
in book order, each in its own cell under its section heading so you can run
and check them one by one. Generated from Book.md; keep the book as the source.
"""

# %%
import os as _os
__file__ = _os.path.join(_os.getcwd(), "00_chapter22_walkthrough.py")

# %% [markdown]
# ## 22.1 When gradients fail

# %% [markdown]
# Example 22-1. Where gradient descent fails.

# %%
import numpy as np
rng = np.random.default_rng(0)

def rastrigin(x): return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))

def numeric_grad(f, x, eps=1e-5):
    g = np.zeros_like(x)
    for i in range(len(x)):
        d = np.zeros_like(x); d[i] = eps
        g[i] = (f(x + d) - f(x - d)) / (2 * eps)
    return g

start = rng.uniform(-5, 5, 2)
x = start.copy()
for _ in range(500):                                    # gradient descent (lr below the ~0.005 stability limit)
    x -= 0.002 * numeric_grad(rastrigin, x)
print(f"RUGGED (Rastrigin, global min = 0):")
print(f"  gradient descent from {np.round(start,2)} -> f = {rastrigin(x):.2f} (stuck in a local min)")

best = min((rng.uniform(-5, 5, 2) for _ in range(5000)), key=rastrigin)   # random search (gradient-free)
print(f"  random search (5000 samples)            -> f = {rastrigin(best):.2f} (found a far better basin)")
print("  gradient descent rolls into the NEAREST dip; random search explores many basins.")

def staircase(x): return np.sum(np.floor(np.abs(x)) ** 2)   # flat within each integer step
x = np.array([4.3, -3.7])
g = numeric_grad(staircase, x)
print(f"\nNON-DIFFERENTIABLE (staircase):")
print(f"  numeric gradient at {x} = {np.round(g,3)} (zero: the surface is flat here, no descent direction)")
x_gd = x.copy()
for _ in range(200): x_gd -= 0.1 * numeric_grad(staircase, x_gd)
print(f"  gradient descent stays put: f {staircase(x):.0f} -> {staircase(x_gd):.0f} (cannot move)")
best2 = min((rng.uniform(-5, 5, 2) for _ in range(2000)), key=staircase)
print(f"  random search                -> f = {staircase(best2):.0f} (improves with no gradient at all)")

print("\nthe explore-vs-exploit dial (the same tradeoff as bandits and RL):")
print("  too much EXPLOIT -> stuck in a local optimum (like plain gradient descent / hill climbing)")
print("  too much EXPLORE -> pure random search, slow to refine")
print("  the methods ahead (annealing, GAs, ES, PSO) are different ways to balance the two,")
print("  needing only the ability to EVALUATE candidates: never to differentiate them.")

print("\nAll when-gradients-fail checks ran.")

# %% [markdown]
# ## 22.2.1 Simulated annealing

# %% [markdown]
# Example 22-2. Hill climbing, simulated annealing, and tabu search.

# %%
import numpy as np
rng = np.random.default_rng(0)

def rastrigin(x): return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))
def neighbor(x, step=0.3): return np.clip(x + rng.normal(0, step, len(x)), -5.12, 5.12)

def hill_climb(start, iters=1000):
    x = start.copy(); fx = rastrigin(x)
    for _ in range(iters):
        cand = neighbor(x); fc = rastrigin(cand)
        if fc < fx: x, fx = cand, fc                     # accept ONLY if strictly better
    return fx

starts = [rng.uniform(-5, 5, 2) for _ in range(30)]
hc = np.mean([hill_climb(s) for s in starts])
print(f"hill climbing            : mean final f = {hc:.2f}  (stuck in local minima)")

rr = np.mean([min(hill_climb(rng.uniform(-5, 5, 2)) for _ in range(10)) for _ in range(30)])
print(f"random-restart (x10)     : mean best f  = {rr:.2f}  (more starts -> better odds)")

def simulated_annealing(start, iters=4000, T0=3.0, cool=0.999):
    x = start.copy(); fx = rastrigin(x); best = fx; T = T0
    for _ in range(iters):
        cand = neighbor(x); fc = rastrigin(cand); delta = fc - fx
        if delta < 0 or rng.random() < np.exp(-delta / T):   # accept better, OR worse with prob exp(-d/T)
            x, fx = cand, fc; best = min(best, fx)
        T *= cool                                            # cool down
    return best                                              # the best value ever visited
sa = np.mean([simulated_annealing(s) for s in starts])
print(f"simulated annealing      : mean best f  = {sa:.2f}  (escapes local minima -> near global 0)")

def tabu_search(start, iters=1000, n_cand=20, tenure=25, radius=0.2):
    x = start.copy(); fx = rastrigin(x); best = fx
    tabu = [x.copy()]                                        # short-term memory of recent points
    for _ in range(iters):
        cands = [neighbor(x) for _ in range(n_cand)]
        scores = [rastrigin(c) for c in cands]               # score each candidate ONCE: 20 evaluations a step
        T = np.array(tabu)
        for k in sorted(range(n_cand), key=lambda k: scores[k]):   # move to the BEST candidate that is not
            c, fc = cands[k], scores[k]                      # within `radius` of a remembered point,
            if fc < best or ((T - c) ** 2).sum(1).min() >= radius ** 2:   # unless it beats the best ever
                x, fx = c, fc; break                         # (the move may be WORSE than x: that is the point)
        tabu.append(x.copy()); tabu = tabu[-tenure:]         # entries expire after `tenure` iterations
        best = min(best, fx)
    return best
tb = np.mean([tabu_search(s) for s in starts])
print(f"tabu search              : mean best f  = {tb:.2f}  (memory bans revisits -> forced onto new ground)")

print("\naccepting a worse move that costs delta=5, at different temperatures T:")
for T in [10.0, 3.0, 1.0, 0.3]:
    print(f"  T={T:4}: P(accept) = exp(-5/{T}) = {np.exp(-5/T):.3f}")
print("high T accepts almost anything (explore); low T rejects nearly all worsening moves (exploit).")
print("cooling slowly from high to low T is what lets annealing climb OUT of a local minimum early,")
print("then settle into a good one late. Tabu search instead BANS recent points to avoid revisiting.")

print("\nAll local-search checks ran.")

# %% [markdown]
# ## 22.2.2 Cooling schedules

# %% [markdown]
# Example 22-3. Three cooling schedules on the twenty-capital map.

# %%
import math
import numpy as np

# The twenty state capitals as (latitude, longitude) in degrees, rounded to two decimals from the
# GeoNames gazetteer (Section 22.7 names the file); Example 22-14 and the lab of Section 22.10 use the same map.
capitals = {
    "Sacramento": (38.58, -121.49), "Phoenix": (33.45, -112.07), "Denver": (39.74, -104.98),
    "Austin": (30.27, -97.74), "Salem": (44.94, -123.04), "Boise": (43.61, -116.20),
    "Santa Fe": (35.69, -105.94), "Oklahoma City": (35.47, -97.52), "Topeka": (39.05, -95.68),
    "Des Moines": (41.60, -93.61), "Saint Paul": (44.94, -93.09), "Jefferson City": (38.58, -92.17),
    "Springfield": (39.80, -89.64), "Nashville": (36.17, -86.78), "Atlanta": (33.75, -84.39),
    "Columbus": (39.96, -83.00), "Montgomery": (32.37, -86.30), "Tallahassee": (30.44, -84.28),
    "Raleigh": (35.77, -78.64), "Richmond": (37.55, -77.46),
}
cities = np.array(list(capitals.values()))
n = len(cities)
D = np.sqrt(((cities[:, None] - cities[None, :]) ** 2).sum(2))   # straight-line distance in degrees

def length(tour):
    return float(sum(D[tour[i], tour[(i + 1) % n]] for i in range(n)))

def reversal_change(tour, i, j):
    # Reversing tour[i..j] swaps the edges (a, b) and (c, d) for (a, c) and (b, d), the 2-opt
    # edge change of Section 22.10.1, so a move costs four lookups, not a full tour length.
    if i == 0 and j == n - 1:
        return 0.0                                       # reversing the whole tour changes nothing
    a, b, c, d = tour[i - 1], tour[i], tour[j], tour[(j + 1) % n]
    return D[a, c] + D[b, d] - D[a, b] - D[c, d]

# Starting temperature: average 500 uphill changes on fresh random tours, then accept that
# average uphill move with probability p0 = 0.8.
rng = np.random.default_rng(0)
ups = []
while len(ups) < 500:
    tour = list(rng.permutation(n))
    i, j = sorted(rng.choice(n, 2, replace=False))
    change = reversal_change(tour, i, j)
    if change > 0:
        ups.append(change)
mean_up, p0 = float(np.mean(ups)), 0.8
T0 = -mean_up / math.log(p0)
print(f"mean uphill change over 500 reversals: {mean_up:.4f}; T0 = {T0:.2f}")

K = 20000                                                # moves per run
schedules = {
    "geometric":   lambda k: T0 * 0.9995 ** k,
    "linear":      lambda k: T0 * (1 - k / K),
    "logarithmic": lambda k: T0 * math.log(2) / math.log(k + 2),
}

def anneal(T_of, seed):
    r = np.random.default_rng(seed)                      # one generator per run
    tour = list(r.permutation(n)); L = length(tour)
    best, best_tour = L, tour.copy()
    for k in range(K):
        i, j = sorted(r.choice(n, 2, replace=False))
        change = reversal_change(tour, i, j); T = T_of(k)
        if change < 0 or (T > 0 and r.random() < math.exp(-change / T)):   # the Metropolis rule
            tour[i:j + 1] = tour[i:j + 1][::-1]; L += change
            if L < best:
                best, best_tour = L, tour.copy()
    return length(best_tour)                             # measure the best tour itself

for name, T_of in schedules.items():
    bests = [anneal(T_of, seed) for seed in range(5)]
    print(f"{name:11s} best tours {[round(b, 2) for b in bests]}  mean {np.mean(bests):.2f}  "
          f"T at the last move {T_of(K - 1):.4f}")

# %% [markdown]
# The block prints the two half-lives, $692.8$ and $1385.9$, the starting temperature $50.22$ with acceptance $0.800$, and the three basin probabilities $0.1810$, $0.5626$, and $0.9999$.

# %%
import math
import numpy as np

def rastrigin_1d(x):
    return 10 + x ** 2 - 10 * np.cos(2 * np.pi * x)

print(f"steps to halve a=0.999:   {math.log(2) / -math.log(0.999):.1f}")
print(f"steps to halve a=0.9995:  {math.log(2) / -math.log(0.9995):.1f}")
mean_up, p0 = 11.2056, 0.8          # the mean uphill change Example 22-3 printed
T0 = -mean_up / math.log(p0)
print(f"T0 = {T0:.2f}; P(accept mean uphill) = {math.exp(-mean_up / T0):.3f}")
grid = np.linspace(-5.12, 5.12, 401)
f = rastrigin_1d(grid)
for T in (10.0, 1.0, 0.1):
    w = np.exp(-(f - f.min()) / T)
    p = w / w.sum()
    print(f"T={T:4}: P(|x|<=0.5) = {p[np.abs(grid) <= 0.5].sum():.4f}")

# %% [markdown]
# ## 22.2.3 Nelder-Mead, a derivative-free local search

# %% [markdown]
# Example 22-4. A from-scratch Nelder-Mead simplex for least absolute deviation.

# %%
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import QuantileRegressor

dia = load_diabetes()
bmi, y = dia.data[:, 2], dia.target

def mae(p):
    return float(np.mean(np.abs(y - (p[0] + p[1] * bmi))))

def nelder_mead(f, x0, step=10.0, max_eval=2000):
    n = len(x0)
    simplex = [np.array(x0, float)]
    for i in range(n):
        e = np.array(x0, float); e[i] += step; simplex.append(e)
    vals = [f(s) for s in simplex]
    evals = n + 1
    while evals < max_eval and max(vals) - min(vals) > 1e-8:
        order = np.argsort(vals)
        simplex = [simplex[i] for i in order]; vals = [vals[i] for i in order]
        best, worst = simplex[0], simplex[-1]
        centroid = np.mean(simplex[:-1], axis=0)
        xr = centroid + (centroid - worst); fr = f(xr); evals += 1
        if vals[0] <= fr < vals[-2]:
            simplex[-1], vals[-1] = xr, fr
        elif fr < vals[0]:
            xe = centroid + 2 * (xr - centroid); fe = f(xe); evals += 1
            simplex[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        else:
            xc = centroid + 0.5 * (worst - centroid); fc = f(xc); evals += 1
            if fc < vals[-1]:
                simplex[-1], vals[-1] = xc, fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = best + 0.5 * (simplex[i] - best)
                    vals[i] = f(simplex[i]); evals += 1
    i = int(np.argmin(vals))
    return simplex[i], vals[i], evals

p, m, ev = nelder_mead(mae, np.array([y.mean(), 0.0]))
print(f"Nelder-Mead: intercept {p[0]:.3f}, slope {p[1]:.3f}, MAE {m:.2f}, evals {ev}")
qr = QuantileRegressor(quantile=0.5, alpha=0, solver="highs").fit(bmi.reshape(-1, 1), y)
pred = qr.predict(bmi.reshape(-1, 1))
print(f"QuantileRegressor: intercept {qr.intercept_:.3f}, slope {qr.coef_[0]:.3f}, MAE {np.mean(np.abs(y - pred)):.2f}")

# %% [markdown]
# ## 22.2.4 Min-conflicts for a constraint problem

# %% [markdown]
# Example 22-5. Min-conflicts on the 8-queens and 100-queens problems.

# %%
import numpy as np

def conflicts(q, col, val):
    c = 0
    for i in range(len(q)):
        if i != col and (q[i] == val or abs(int(q[i]) - val) == abs(i - col)):
            c += 1
    return c

def min_conflicts(n, seed, limit=2000):
    rng = np.random.default_rng(seed)
    q = rng.integers(0, n, n)
    for step in range(limit):
        bad = [i for i in range(n) if conflicts(q, i, int(q[i])) > 0]
        if not bad:
            return step
        col = int(rng.choice(bad))
        scores = [conflicts(q, col, v) for v in range(n)]
        best = min(scores)
        q[col] = int(rng.choice([v for v, s in enumerate(scores) if s == best]))
    return None

for n in (8, 100):
    steps = [min_conflicts(n, s) for s in range(20)]
    ok = [s for s in steps if s is not None]
    print(f"n={n}: solved {len(ok)}/20, median steps {int(np.median(ok))}")

# %% [markdown]
# ## 22.3 Genetic algorithms

# %% [markdown]
# Example 22-6. A genetic algorithm on a rugged function.

# %%
import numpy as np
rng = np.random.default_rng(0)

p1 = np.array([1, 1, 1, 1, 0, 0]); p2 = np.array([0, 0, 0, 0, 1, 1])
pt = 3
child = np.concatenate([p1[:pt], p2[pt:]])               # one-point crossover at position 3
print(f"parent 1: {p1.tolist()}\nparent 2: {p2.tolist()}")
print(f"one-point crossover at {pt} -> child {child.tolist()} (first part of p1 + second part of p2)")
m = child.copy(); flip = 4; m[flip] ^= 1
print(f"mutation flips gene {flip}              -> {m.tolist()} (a random tweak adds diversity)")

DIM = 5
def rastrigin(x): return 10 * DIM + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x), axis=-1)

POP, GENS, ELITE = 120, 80, 4
pop = rng.uniform(-5.12, 5.12, (POP, DIM))

def tournament(fit, k=3):
    idx = rng.integers(0, POP, k)
    return idx[np.argmin(fit[idx])]                      # lowest Rastrigin among k random = winner

history = []
for gen in range(GENS):
    fit = rastrigin(pop)
    order = np.argsort(fit)
    new = [pop[order[i]].copy() for i in range(ELITE)]   # ELITISM: keep the best few
    while len(new) < POP:
        a, b = pop[tournament(fit)], pop[tournament(fit)]
        alpha = rng.uniform(-0.25, 1.25, DIM)            # BLX blend crossover
        child = alpha * a + (1 - alpha) * b
        child += rng.normal(0, 0.3, DIM) * (rng.random(DIM) < 0.3)   # Gaussian mutation, 30% of genes
        new.append(np.clip(child, -5.12, 5.12))
    pop = np.array(new)
    history.append(rastrigin(pop).min())

print(f"\nGA on 5D Rastrigin (pop {POP}, {GENS} generations):")
for g in [0, 10, 30, 79]:
    print(f"  generation {g:2d}: best f = {history[g]:.3f}")
print(f"final best f = {history[-1]:.3f} (global min is 0); fitness climbs as good genes spread.")
print("random search in 5D rarely gets close; the population + crossover + selection do.")

print("\nGA design choices:")
print("  selection pressure (tournament size): high -> fast convergence but risks premature collapse")
print("  mutation rate: high -> more exploration, low -> more exploitation (the recurring dial)")
print("  crossover: one-point/uniform for bit-strings, blend (BLX) for real vectors")
print("  elitism: protects the best solution but too much elitism reduces diversity")

print("\nAll genetic-algorithm checks ran.")

# %% [markdown]
# ## 22.3.1 How a parent is chosen

# %% [markdown]
# The block prints roulette probabilities `[0.4, 0.3, 0.2, 0.1]`, rank probabilities `[0.125, 0.208, 0.292, 0.375]`, and tournament probability `0.024792`.

# %%
import numpy as np
f = np.array([4, 3, 2, 1], float)
print("roulette", (f / f.sum()).tolist())
N, s = 4, 1.5
r = np.array([1, 2, 3, 4])  # worst to best
p = ((2 - s) + 2 * (s - 1) * (r - 1) / (N - 1)) / N
print("rank", np.round(p, 3).tolist(), "best", s / N, "worst", (2 - s) / N)
print(f"tournament P(best) {1 - (1 - 1/120) ** 3:.6f}")

# %% [markdown]
# ## 22.3.2 A deceptive trap

# %% [markdown]
# Example 22-7. A truncation genetic algorithm on concatenated deceptive traps.

# %%
import numpy as np

def trap4(bits):
    u = int(np.sum(bits))
    return 4 if u == 4 else 3 - u

vals0 = [trap4([0, (i >> 0) & 1, (i >> 1) & 1, (i >> 2) & 1]) for i in range(8)]
vals1 = [trap4([1, (i >> 0) & 1, (i >> 1) & 1, (i >> 2) & 1]) for i in range(8)]
print(f"0*** mean {np.mean(vals0):.3f}   1*** mean {np.mean(vals1):.3f}")

def fitness(bits):
    return sum(trap4(bits[i:i + 4]) for i in range(0, 40, 4))

ones = zeros = 0
for seed in range(8):
    rng = np.random.default_rng(seed)
    pop = rng.integers(0, 2, (40, 40))
    for _ in range(60):
        fit = np.array([fitness(b) for b in pop])
        order = np.argsort(-fit, kind="stable")      # best first; equal scores keep population order
        new = [pop[order[i]].copy() for i in range(4)]
        while len(new) < 40:
            a, b = pop[order[rng.integers(8)]], pop[order[rng.integers(8)]]
            cut = int(rng.integers(1, 40))
            child = np.concatenate([a[:cut], b[cut:]])
            flip = rng.random(40) < 0.02
            child = child.copy(); child[flip] = 1 - child[flip]
            new.append(child)
        pop = np.array(new)
    best = pop[int(np.argmax([fitness(b) for b in pop]))]
    blocks = best.reshape(10, 4)
    ones += int(np.all(blocks == 1, axis=1).sum())
    zeros += int(np.all(blocks == 0, axis=1).sum())
print(f"linked blocks: {ones} at all ones, {zeros} at all zeros, out of 80")

def fitness_scattered(bits):
    total = 0
    for b in range(10):
        total += trap4([bits[b + 10 * k] for k in range(4)])
    return total

ones = zeros = 0
for seed in range(8):
    rng = np.random.default_rng(seed)
    pop = rng.integers(0, 2, (40, 40))
    for _ in range(60):
        fit = np.array([fitness_scattered(b) for b in pop])
        order = np.argsort(-fit, kind="stable")      # best first; equal scores keep population order
        new = [pop[order[i]].copy() for i in range(4)]
        while len(new) < 40:
            a, b = pop[order[rng.integers(8)]], pop[order[rng.integers(8)]]
            cut = int(rng.integers(1, 40))
            child = np.concatenate([a[:cut], b[cut:]])
            flip = rng.random(40) < 0.02
            child = child.copy(); child[flip] = 1 - child[flip]
            new.append(child)
        pop = np.array(new)
    best = pop[int(np.argmax([fitness_scattered(b) for b in pop]))]
    blocks = np.array([[best[b + 10 * k] for k in range(4)] for b in range(10)])
    ones += int(np.all(blocks == 1, axis=1).sum())
    zeros += int(np.all(blocks == 0, axis=1).sum())
print(f"scattered bits: {ones} at all ones, {zeros} at all zeros, out of 80")

# %% [markdown]
# ## 22.4.1 CMA-ES: adapting the search distribution

# %% [markdown]
# Started at the same corner of the valley as the toy comparison of Example 22-8, pycma's full CMA-ES drives the objective to machine precision in a few hundred evaluations.

# %%
import cma                                               # pycma: the maintained CMA-ES
es_cma = cma.CMAEvolutionStrategy([8.0, 8.0], 2.0, {"seed": 1, "verbose": -9})
es_cma.optimize(lambda z: z[0] ** 2 + 100 * z[1] ** 2)
print(f"pycma CMA-ES on the valley: f = {es_cma.result.fbest:.2e} after {es_cma.result.evaluations} evaluations")

# %% [markdown]
# Example 22-8. Evolution strategies and shape adaptation.

# %%
import numpy as np
rng = np.random.default_rng(0)

def rastrigin(X): return 10 * X.shape[-1] + np.sum(X ** 2 - 10 * np.cos(2 * np.pi * X), axis=-1)
def es_rastrigin(dim=5, lam=40, mu=10, gens=120, sigma=2.0):
    m = rng.uniform(-5, 5, dim)
    for g in range(gens):
        kids = m + sigma * rng.normal(0, 1, (lam, dim))      # sample lambda offspring ~ N(m, sigma^2 I)
        fit = rastrigin(kids)
        elite = kids[np.argsort(fit)[:mu]]                   # keep the mu best (mu, lambda)
        m = elite.mean(0)                                    # recenter on the elite
        sigma *= 0.98                                        # cool the step size
    return rastrigin(m[None])[0]
print(f"(mu,lambda)-ES on 5D Rastrigin: final f = {es_rastrigin():.3f}  (global min 0)")

def ill(X): return X[..., 0] ** 2 + 100 * X[..., 1] ** 2     # optimum at (0,0); 100:1 conditioning

def es(adapt_shape, gens=60, lam=40, mu=10):
    m = np.array([8.0, 8.0]); sig = np.array([2.0, 2.0])
    for g in range(gens):
        kids = m + sig * rng.normal(0, 1, (lam, 2))      # per-axis step sizes sig
        elite = kids[np.argsort(ill(kids))[:mu]]
        m = elite.mean(0)
        if adapt_shape:
            sig = 0.5 * sig + 0.5 * (elite.std(0) + 1e-9)    # CMA idea: each axis' step <- elite spread on that axis
        else:
            s = 0.5 * sig.mean() + 0.5 * elite.std(0).mean()  # isotropic: ONE step size for both axes
            sig = np.array([s, s])
    return ill(m[None])[0]

iso = es(adapt_shape=False)
adapted = es(adapt_shape=True)                           # not `cma`, so the pycma module name stays free
print(f"\nill-conditioned valley (x0^2 + 100 x1^2), optimum 0, after 60 generations:")
print(f"  isotropic ES (one step size) : f = {iso:.4f}  (crawls: one sigma cannot fit both axes)")
print(f"  covariance-adapting (CMA idea): f = {adapted:.4f}  (learns the valley's shape -> far faster)")

print("\nES notation and use:")
print("  (mu, lambda) : select from offspring only (non-elitist; better at escaping)")
print("  (mu + lambda): select from parents + offspring (elitist; never loses the best)")
print("  CMA-ES       : adapts a full covariance matrix AND step size; a strong black-box")
print("                 optimizer for problems of low to moderate dimension (Section 22.4.1 names its uses).")
print("  ES parallelizes trivially (evaluate all offspring at once): a plus for expensive simulators.")

print("\nAll evolution-strategy checks ran.")

# %% [markdown]
# ## 22.4.2 The search as a gradient estimate

# %% [markdown]
# Example 22-9. Plain and antithetic gradient estimates on the breast cancer logistic loss.

# %%
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(load_breast_cancer().data)
y = load_breast_cancer().target.astype(float)
Xb = np.column_stack([np.ones(len(y)), X])

def nll(theta):
    z = Xb @ theta
    return float(np.mean(np.logaddexp(0, z) - y * z))

def grad_nll(theta):
    z = np.clip(Xb @ theta, -40, 40)
    p = 1 / (1 + np.exp(-z))
    return Xb.T @ (p - y) / len(y)

theta = np.zeros(Xb.shape[1])
g_true = grad_nll(theta)

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

for sigma in (0.05, 0.5):
    rng = np.random.default_rng(0)                       # the same draws at both noise scales
    for n_eval in (10, 100, 1000):
        cs = []
        for _ in range(20):
            eps = rng.normal(size=(n_eval, len(theta)))
            F = np.array([nll(theta + sigma * e) for e in eps])
            g = (F[:, None] * eps).sum(0) / (n_eval * sigma)
            cs.append(cosine(g, g_true))
        print(f"sigma {sigma}: plain      {n_eval:4d} evals  mean cosine {np.mean(cs):.3f}")
    for n_pair, n_eval in ((5, 10), (50, 100), (500, 1000)):
        cs = []
        for _ in range(20):
            eps = rng.normal(size=(n_pair, len(theta)))
            diff = np.array([nll(theta + sigma * e) - nll(theta - sigma * e) for e in eps])
            g = (diff[:, None] * eps).sum(0) / (2 * n_pair * sigma)
            cs.append(cosine(g, g_true))
        print(f"sigma {sigma}: antithetic {n_eval:4d} evals  mean cosine {np.mean(cs):.3f}")

# %% [markdown]
# ## 22.4.3 The cross-entropy method

# %% [markdown]
# Example 22-10. The cross-entropy method on five-dimensional Rastrigin, with and without a floor on sigma.

# %%
import numpy as np

def rastrigin(X):
    X = np.atleast_2d(X)
    return 10 * X.shape[-1] + np.sum(X ** 2 - 10 * np.cos(2 * np.pi * X), axis=-1)

def cem(seed, floor):
    rng = np.random.default_rng(seed)
    mu, sig = np.zeros(5), np.ones(5) * 2
    best = 1e9
    for _ in range(40):
        X = np.clip(mu + sig * rng.normal(size=(40, 5)), -5.12, 5.12)
        f = rastrigin(X)
        best = min(best, float(f.min()))
        elite = X[np.argsort(f)[:10]]                    # the best quarter of 40
        mu = 0.7 * elite.mean(0) + 0.3 * mu              # smoothed refit of the mean ...
        sig = np.maximum(0.7 * elite.std(0) + 0.3 * sig, floor)   # ... and of sigma, held above the floor
    return best, mu, sig

best, mu, sig = cem(0, 0.05)
print(f"CEM on 5D Rastrigin: best f = {best:.4f}")
for seed, floor in ((0, 0.0), (2, 0.0), (2, 0.05)):
    best, mu, sig = cem(seed, floor)
    print(f"seed {seed}, floor {floor}: best f = {best:.4f}, final mean {(np.round(mu, 3) + 0.0).tolist()}, "
          f"largest final sigma {sig.max():.1e}")

# %% [markdown]
# ## 22.5 Genetic programming

# %% [markdown]
# Example 22-11. Genetic programming for symbolic regression.

# %%
import numpy as np
rng = np.random.default_rng(1)

xs = np.linspace(-3, 3, 40)
ys = xs ** 2 + 2 * xs + 1
OPS = ["+", "-", "*"]

def random_tree(depth):
    if depth == 0 or (depth < 3 and rng.random() < 0.3):
        return ("x",) if rng.random() < 0.6 else ("c", round(rng.uniform(-3, 3), 1))
    return (rng.choice(OPS), random_tree(depth - 1), random_tree(depth - 1))

def evaluate(node, x):
    if node[0] == "x": return x
    if node[0] == "c": return np.full_like(x, node[1])
    a, b = evaluate(node[1], x), evaluate(node[2], x)
    return a + b if node[0] == "+" else a - b if node[0] == "-" else a * b

def to_str(node):
    if node[0] == "x": return "x"
    if node[0] == "c": return str(node[1])
    return f"({to_str(node[1])} {node[0]} {to_str(node[2])})"

def mse(node):
    try:
        e = np.mean((evaluate(node, xs) - ys) ** 2)
        return e if np.isfinite(e) else 1e9
    except Exception:
        return 1e9

def size(node):                                          # number of nodes in the tree
    return 1 if len(node) < 3 else 1 + size(node[1]) + size(node[2])

def nodes(node, path=()):
    yield path, node
    if len(node) == 3:
        yield from nodes(node[1], path + (1,)); yield from nodes(node[2], path + (2,))

def replace_at(node, path, new):
    if not path: return new
    i, rest = path[0], path[1:]
    children = [node[1], node[2]]
    children[i - 1] = replace_at(children[i - 1], rest, new)
    return (node[0], children[0], children[1])

def crossover(a, b):
    pa = [p for p, _ in nodes(a)]; pb = [(p, n) for p, n in nodes(b)]
    path_a = pa[rng.integers(len(pa))]
    _, sub_b = pb[rng.integers(len(pb))]
    return replace_at(a, path_a, sub_b)

def mutate(a):
    paths = [p for p, _ in nodes(a)]
    return replace_at(a, paths[rng.integers(len(paths))], random_tree(2))

def evolve(fit_key, label):                              # one GP run, selecting on fit_key
    pop = [random_tree(3) for _ in range(POP)]
    for gen in range(GENS):
        pop.sort(key=fit_key)
        if gen in (0, 10, 30, 59):
            print(f"  generation {gen:2d}: best MSE = {mse(pop[0]):.4f}, mean tree size = {np.mean([size(t) for t in pop]):5.1f} nodes")
        new = pop[:6]                                    # elitism
        while len(new) < POP:
            a = min(rng.choice(len(pop), 4), key=lambda i: fit_key(pop[i]))   # tournament parent
            b = min(rng.choice(len(pop), 4), key=lambda i: fit_key(pop[i]))
            child = crossover(pop[a], pop[b])
            if rng.random() < 0.3: child = mutate(child)
            new.append(child)
        pop = new
    pop.sort(key=fit_key)
    best = pop[0]
    print(f"  {label} best: {to_str(best)}")
    print(f"    MSE = {mse(best):.5f}, size = {size(best)} nodes")
    return best

POP, GENS = 300, 60
print("fitness = MSE alone (no size pressure):")
best = evolve(mse, "plain-MSE")

LAM = 0.001                                              # parsimony pressure: each node costs a little fitness
def score(node): return mse(node) + LAM * size(node)     # the error plus a charge per node
print("\nfitness = MSE + 0.001 * size (parsimony pressure):")
best_p = evolve(score, "parsimony")

print("\nboth runs fit the data exactly (target was x^2 + 2x + 1), but the parsimony run's")
print("population stays lean while the plain-MSE population bloats with useless subtrees.")
print("GP rediscovered the underlying equation from data alone: a readable formula, not a black box.")
print("that interpretability is the reason to want the formula itself, not only a fitted score.")

print("\nAll genetic-programming checks ran.")

# %% [markdown]
# ## 22.6 Neuroevolution

# %% [markdown]
# Example 22-12. Neuroevolution: learning XOR without backpropagation.

# %%
import numpy as np
rng = np.random.default_rng(0)

# XOR: not linearly separable, needs a hidden layer
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
y = np.array([0, 1, 1, 0], float)

def unpack(w):
    W1 = w[:8].reshape(2, 4); b1 = w[8:12]; W2 = w[12:16].reshape(4, 1); b2 = w[16]
    return W1, b1, W2, b2
def forward(w, X):
    W1, b1, W2, b2 = unpack(w)
    h = np.tanh(X @ W1 + b1)
    return 1 / (1 + np.exp(-(h @ W2 + b2))).ravel()
def fitness(w):                                          # higher is better: negative loss
    p = forward(w, X)
    return -np.mean((p - y) ** 2)
def accuracy(w):
    return np.mean((forward(w, X) > 0.5) == (y > 0.5))

N_W = 17

POP, GENS, MU = 60, 200, 10
pop = rng.normal(0, 1, (POP, N_W))
print("evolving 17 weights of a 2-4-1 MLP to solve XOR (no backprop):")
for gen in range(GENS):
    fits = np.array([fitness(w) for w in pop])
    elite = pop[np.argsort(-fits, kind="stable")[:MU]]   # keep the MU fittest (stable sort, Section 22.3.2)
    if gen in (0, 20, 60, 199):
        best = pop[np.argmax(fits)]
        print(f"  generation {gen:3d}: best loss {-fits.max():.4f}, accuracy {accuracy(best):.2f}")
    children = [elite[rng.integers(MU)] + rng.normal(0, 0.5, N_W) for _ in range(POP - MU)]
    pop = np.vstack([elite, children])                   # elite + mutated offspring

best = pop[np.argmax([fitness(w) for w in pop])]
print(f"\nfinal: XOR accuracy = {accuracy(best):.2f}; predictions = {np.round(forward(best, X), 2).tolist()}")
print("the network learned XOR purely by selection and mutation, never computing a gradient.")

print("\nthe neuroevolution landscape:")
print("  fixed-topology: evolve only the weights (what we did): simple, good for small nets")
print("  NEAT          : the name for evolving TOPOLOGY along with the weights; not developed here")
print("  HyperNEAT     : the name for evolving a function that GENERATES the weights; not developed here")
print("  ES vs policy gradients (RL): Evolution Strategies optimize the reward DIRECTLY (no")
print("                  backprop through time) and can evaluate candidates independently.")
print("                  The lab (22.10) compares that idea with Chapter 18 on CartPole.")

print("\nAll neuroevolution checks ran.")

# %% [markdown]
# ## 22.6.1 Novelty search and quality-diversity

# %% [markdown]
# Example 22-13. Novelty search and MAP-Elites against fitness on a deceptive maze.

# %%
import numpy as np

# A constructed deceptive maze, a mechanics demo rather than a dataset. The box is the unit
# square. Three walls form an upside-down cup over the start: a lid along y = 0.5 from
# x = 0.1 to 0.9, and two sides that hang from its ends down to y = 0.05, so the only way
# out is through the gap under a side. The goal sits above the lid.
WALLS = [((0.1, 0.5), (0.9, 0.5)), ((0.1, 0.05), (0.1, 0.5)), ((0.9, 0.05), (0.9, 0.5))]
START, GOAL = np.array([0.5, 0.25]), np.array([0.5, 0.9])
N_MOVES, STEP = 12, 0.15          # a genome is 12 moves (dx, dy), each coordinate in [-0.15, 0.15]
N_GENES = 2 * N_MOVES

def side(o, a, b):
    # sign of the cross product (a - o) x (b - o): which side of the line o -> a the point b is on
    return np.sign((a[..., 0] - o[..., 0]) * (b[..., 1] - o[..., 1])
                   - (a[..., 1] - o[..., 1]) * (b[..., 0] - o[..., 0]))

def blocked(p, q):
    # True for each move p -> q (one per row) that crosses a wall: two segments cross
    # when each one has the other's two end points on opposite sides of it
    hit = np.zeros(len(p), bool)
    for a, b in WALLS:
        a, b = np.array(a), np.array(b)
        hit |= (side(p, q, a) != side(p, q, b)) & (side(a, b, p) != side(a, b, q))
    return hit

def final_positions(G):
    # run every genome (one row of G) through the maze at once; return where each one ends
    moves = G.reshape(len(G), N_MOVES, 2)
    p = np.tile(START, (len(G), 1))
    for t in range(N_MOVES):
        q = np.clip(p + moves[:, t], 0, 1)       # the edges of the box stop the point
        ok = ~blocked(p, q)
        p[ok] = q[ok]                            # a move into a wall is simply not taken
    return p

def novelty(B, ref, k):
    # n(x) for each behavior (row of B): the mean distance to its k nearest rows of ref.
    # B is part of ref, so the smallest distance of each row, 0 to itself, is skipped.
    D = np.linalg.norm(B[:, None, :] - ref[None, :, :], axis=2)
    D.sort(axis=1)
    return D[:, 1:k + 1].mean(axis=1)

# the hand calculation: A and B are jammed under the lid, C and D head for the gap
hand = np.array([[0.5, 0.45], [0.6, 0.45], [0.2, 0.05], [0.5, 0.05]])
print("hand check, k = 2: novelty", np.round(novelty(hand, hand, 2), 3).tolist(),
      "| distance to goal", np.round(np.linalg.norm(hand - GOAL, axis=1), 3).tolist())

POP, MU, SIGMA, K_NN, ADD = 50, 10, 0.05, 10, 2

def evolve(rule, seed, evals):
    # truncation selection as in Example 22-12, ranked by one of three scores
    rng = np.random.default_rng(seed)
    pop = rng.uniform(-STEP, STEP, (POP, N_GENES))
    archive = np.empty((0, 2))                   # behaviors novelty search has stored
    best = np.inf                                # closest any genome has come to the goal
    for gen in range(evals // POP):              # POP evaluations per generation
        B = final_positions(pop)
        dist = np.linalg.norm(B - GOAL, axis=1)
        best = min(best, dist.min())
        if rule == "fitness":
            score = -dist                        # the objective: end near the goal
        elif rule == "random":
            score = rng.random(POP)              # no pressure at all, the control
        else:
            score = novelty(B, np.vstack([B, archive]), K_NN)
            archive = np.vstack([archive, B[np.argsort(-score, kind="stable")[:ADD]]])
        parents = pop[np.argsort(-score, kind="stable")[:MU]]   # stable order, as in Section 22.3.2
        kids = parents[rng.integers(MU, size=POP - MU)] + rng.normal(0, SIGMA, (POP - MU, N_GENES))
        pop = np.vstack([parents, np.clip(kids, -STEP, STEP)])
    return best

def map_elites(seed, evals, cells=10):
    # a cells x cells grid over the final position; each cell keeps its fittest genome
    rng = np.random.default_rng(seed)
    cell_fit = np.full((cells, cells), -np.inf)  # -inf marks an empty cell
    cell_genome = np.zeros((cells, cells, N_GENES))
    best = np.inf
    for done in range(0, evals, POP):            # the same POP evaluations per round
        if done < 2 * POP:                       # the first 100 genomes are random
            G = rng.uniform(-STEP, STEP, (POP, N_GENES))
        else:                                    # then mutate the elites of uniformly drawn filled cells
            filled = np.argwhere(np.isfinite(cell_fit))
            pick = filled[rng.integers(len(filled), size=POP)]
            G = cell_genome[pick[:, 0], pick[:, 1]] + rng.normal(0, SIGMA, (POP, N_GENES))
            G = np.clip(G, -STEP, STEP)
        B = final_positions(G)
        f = -np.linalg.norm(B - GOAL, axis=1)
        best = min(best, -f.max())
        ij = np.minimum((B * cells).astype(int), cells - 1)
        for (i, j), fi, g in zip(ij, f, G):      # a child takes its cell only if it is fitter
            if fi > cell_fit[i, j]:
                cell_fit[i, j], cell_genome[i, j] = fi, g
    return best, int(np.isfinite(cell_fit).sum())

for evals in (5000, 20000):
    print(f"\nbudget {evals} evaluations per run, seeds 0 to 9 (reached = ends within 0.05 of the goal):")
    for rule, label in (("fitness", "fitness only"), ("random", "random selection"), ("novelty", "novelty search")):
        b = [evolve(rule, s, evals) for s in range(10)]
        print(f"  {label:16s}: reached {sum(v < 0.05 for v in b):2d} of 10; best distance per seed {np.round(b, 2).tolist()}")
    me = [map_elites(s, evals) for s in range(10)]
    b = [v for v, _ in me]
    print(f"  {'MAP-Elites':16s}: reached {sum(v < 0.05 for v in b):2d} of 10; best distance per seed {np.round(b, 2).tolist()}")
    print(f"  {'':16s}  cells filled of 100: {[c for _, c in me]}")

# %% [markdown]
# ## 22.7 Swarm intelligence: particle swarms and ant colonies

# %% [markdown]
# Example 22-14. Particle swarm optimization and ant colony optimization.

# %%
import numpy as np
rng = np.random.default_rng(0)

def rastrigin(X): return 10 * X.shape[-1] + np.sum(X ** 2 - 10 * np.cos(2 * np.pi * X), axis=-1)
DIM, NP, ITERS = 5, 40, 200
x = rng.uniform(-5, 5, (NP, DIM)); v = rng.normal(0, 1, (NP, DIM))
pbest = x.copy(); pbest_f = rastrigin(x)
gbest = pbest[pbest_f.argmin()].copy(); gbest_f = pbest_f.min()
w, c1, c2 = 0.7, 1.5, 1.5
for it in range(ITERS):
    r1, r2 = rng.random((NP, DIM)), rng.random((NP, DIM))
    v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)    # the PSO velocity update
    x = np.clip(x + v, -5.12, 5.12)
    f = rastrigin(x)
    improved = f < pbest_f
    pbest[improved] = x[improved]; pbest_f[improved] = f[improved]   # each particle's own best
    if pbest_f.min() < gbest_f:
        gbest = pbest[pbest_f.argmin()].copy(); gbest_f = pbest_f.min()   # swarm's best
print(f"PSO on 5D Rastrigin: best f = {gbest_f:.3f} at x = {(np.round(gbest, 3) + 0.0).tolist()}")   # + 0.0 turns -0.0 into 0.0
print(f"  distance from the origin = {np.linalg.norm(gbest):.3f}  (the global minimum is f = 0 at the origin)")

# 20 real US state capitals: (latitude, longitude) in degrees, rounded to two decimals from the
# GeoNames cities15000 file (feature code PPLA), downloaded 8 November 2026. Euclidean distance
# on (lat, lon) is a planar approximation that ignores Earth's curvature; every tour of this map
# in the chapter is measured the same way. The lab in 22.10 evolves a tour of this SAME map with a GA.
capitals = {
    "Sacramento": (38.58, -121.49), "Phoenix": (33.45, -112.07), "Denver": (39.74, -104.98),
    "Austin": (30.27, -97.74), "Salem": (44.94, -123.04), "Boise": (43.61, -116.20),
    "Santa Fe": (35.69, -105.94), "Oklahoma City": (35.47, -97.52), "Topeka": (39.05, -95.68),
    "Des Moines": (41.60, -93.61), "Saint Paul": (44.94, -93.09), "Jefferson City": (38.58, -92.17),
    "Springfield": (39.80, -89.64), "Nashville": (36.17, -86.78), "Atlanta": (33.75, -84.39),
    "Columbus": (39.96, -83.00), "Montgomery": (32.37, -86.30), "Tallahassee": (30.44, -84.28),
    "Raleigh": (35.77, -78.64), "Richmond": (37.55, -77.46),
}
names = list(capitals); cities = np.array([capitals[n] for n in names])
NC = len(names)
D = np.sqrt(((cities[:, None] - cities[None, :]) ** 2).sum(2)) + np.eye(NC)   # distances (avoid /0)
def length(t): return sum(D[t[i], t[(i + 1) % NC]] for i in range(NC))
def nearest_neighbor():                                  # greedy baseline: hop to the closest unvisited city
    unv = set(range(1, NC)); tour = [0]
    while unv:
        nxt = min(unv, key=lambda c: D[tour[-1], c]); tour.append(nxt); unv.discard(nxt)
    return length(tour)

eta = 1 / D                                              # heuristic: prefer near cities
tau = np.ones((NC, NC)) * 0.1                            # pheromone, starts uniform
alpha, beta, rho, ANTS = 1.0, 3.0, 0.5, 20
best_tour, best_len = None, np.inf
for it in range(60):
    all_tours = []
    for _ in range(ANTS):
        start = rng.integers(NC); tour = [start]; unvisited = set(range(NC)) - {start}
        while unvisited:
            i = tour[-1]; js = list(unvisited)
            w_ = (tau[i, js] ** alpha) * (eta[i, js] ** beta)    # prob ~ pheromone^a * heuristic^b
            nxt = js[rng.choice(len(js), p=w_ / w_.sum())]
            tour.append(nxt); unvisited.discard(nxt)
        all_tours.append(tour)
        L = length(tour)
        if L < best_len: best_len, best_tour = L, tour
    tau *= (1 - rho)                                     # evaporation
    for tour in all_tours:                               # deposit: shorter tours leave more pheromone
        L = length(tour)
        for i in range(NC):
            a, b = tour[i], tour[(i + 1) % NC]; tau[a, b] += 1.0 / L; tau[b, a] += 1.0 / L

print(f"\nACO on a {NC}-city TSP over real US state capitals (lat/lon, planar):")
print(f"  nearest-neighbor tour = {nearest_neighbor():.2f}")
print(f"  best ACO tour length  = {best_len:.2f}  ({(1-best_len/nearest_neighbor())*100:.0f}% shorter than greedy)")
print("ants reinforce short edges with pheromone; the colony beats the greedy baseline with no global view.")
print("the lab in 22.10 evolves a tour of this same 20-capital map with a genetic algorithm.")

print("\nthe swarm idea: many simple agents + a shared signal (best position, or pheromone) =")
print("emergent global search. PSO suits continuous spaces; ACO suits routing/ordering problems.")
print("both balance explore (inertia / evaporation + randomness) against exploit (attraction to good spots).")

print("\nAll swarm-intelligence checks ran.")

# %% [markdown]
# ## 22.8.1 Multiobjective optimization and Pareto fronts

# %% [markdown]
# Example 22-15. NSGA-II from scratch on a biobjective benchmark.

# %%
import numpy as np
rng = np.random.default_rng(0)

# One-variable bi-objective check with a KNOWN answer: minimize both
f1 = lambda x: x ** 2                 # smallest at x = 0
f2 = lambda x: (x - 2) ** 2           # smallest at x = 2
# No single x minimizes both; the true Pareto set is exactly the interval x in [0, 2].
objs = lambda pop: np.stack([f1(pop), f2(pop)], axis=1)      # (N, 2) objective vectors

def dominates(p, q):                  # p dominates q: no worse on all, strictly better on one
    return np.all(p <= q) and np.any(p < q)

def non_dominated_sort(F):            # rank the population into successive Pareto fronts
    N = len(F); S = [[] for _ in range(N)]; n = np.zeros(N, int); fronts = [[]]
    for p in range(N):
        for q in range(N):
            if dominates(F[p], F[q]): S[p].append(q)
            elif dominates(F[q], F[p]): n[p] += 1
        if n[p] == 0: fronts[0].append(p)                    # front 1 (stored at index 0): dominated by nobody
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0: nxt.append(q)
        i += 1; fronts.append(nxt)
    fronts.pop(); return fronts

def crowding(F, front):               # spread measure: prefer solutions in sparse regions
    d = np.zeros(len(front)); pts = F[front]
    for m in range(F.shape[1]):
        o = np.argsort(pts[:, m], kind="stable"); d[o[0]] = d[o[-1]] = np.inf   # keep the extremes
        span = pts[o[-1], m] - pts[o[0], m]
        if span == 0: continue
        for k in range(1, len(front) - 1):
            d[o[k]] += (pts[o[k + 1], m] - pts[o[k - 1], m]) / span
    return d

POP, GENS = 60, 40
pop = rng.uniform(-3, 5, POP)                                 # the decision variable x
for gen in range(GENS):
    par = pop[rng.integers(0, POP, (POP, 2))]; a = rng.random(POP)
    child = a * par[:, 0] + (1 - a) * par[:, 1]               # blend crossover
    child = np.clip(child + rng.normal(0, 0.3, POP) * (rng.random(POP) < 0.3), -3, 5)  # mutate
    union = np.concatenate([pop, child]); F = objs(union)
    fronts = non_dominated_sort(F)
    new = []                                                  # environmental selection
    for fr in fronts:
        if len(new) + len(fr) <= POP: new.extend(fr)          # whole front fits: take it
        else:                                                 # last front: take the least crowded
            order = np.array(fr)[np.argsort(-crowding(F, fr), kind="stable")]
            new.extend(order[:POP - len(new)].tolist()); break
    pop = union[new]

F = objs(pop); pf = np.array(non_dominated_sort(F)[0])
xs = np.sort(pop[pf])
print("NSGA-II on a bi-objective problem (minimize f1=x^2 AND f2=(x-2)^2):")
print(f"  non-dominated (Pareto-front) solutions: {len(pf)} of {POP}")
print(f"  their x-values span [{xs.min():.2f}, {xs.max():.2f}]   (true Pareto set is x in [0, 2])")
Fpf = F[pf][np.argsort(F[pf][:, 0], kind="stable")]
print("  sample of the front (f1, f2):", [f"({u:.2f},{w:.2f})" for u, w in Fpf[::12]])
print("NSGA-II returns the whole tradeoff frontier, not one winner.")

# %% [markdown]
# ## 22.8.2 No free lunch

# %% [markdown]
# Example 22-16. Differential evolution and the no-free-lunch reminder.

# %%
import numpy as np
rng = np.random.default_rng(0)

def sphere(x): return np.sum(x ** 2)                               # smooth, single optimum
def rastrigin(x): return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))   # rugged, many optima
DIM, BUDGET = 5, 15000                                            # equal evaluation budget for fairness

def differential_evolution(f, pop=40, F=0.5, CR=0.9):
    P = rng.uniform(-5, 5, (pop, DIM)); fit = np.array([f(p) for p in P]); evals = pop
    while evals < BUDGET:
        for i in range(pop):
            idxs = rng.choice([j for j in range(pop) if j != i], 3, replace=False)
            a, b, c = P[idxs]
            mutant = np.clip(a + F * (b - c), -5.12, 5.12)        # difference-vector mutation
            mask = rng.random(DIM) < CR; trial = np.where(mask, mutant, P[i])
            ft = f(trial); evals += 1
            if ft < fit[i]: P[i], fit[i] = trial, ft              # greedy selection
        if evals >= BUDGET: break
    return fit.min()

de_rastrigin = np.mean([differential_evolution(rastrigin) for _ in range(3)])
print(f"differential evolution on {DIM}D Rastrigin: mean best f over 3 runs = {de_rastrigin:.3f}  (min 0)")

def staircase(x): return np.sum(np.floor(np.abs(x)) ** 2)         # flat steps: gradient is useless
def grad_descent(f, lr=0.1, eps=1e-4):
    x = rng.uniform(-5, 5, DIM)
    for _ in range(BUDGET // (2 * DIM)):                          # each step costs 2*DIM evals (numeric grad)
        g = np.array([(f(x + eps * (np.arange(DIM) == i)) - f(x - eps * (np.arange(DIM) == i))) / (2 * eps)
                      for i in range(DIM)])
        x = np.clip(x - lr * g, -5.12, 5.12)
    return f(x)
def random_search(f):
    return min(f(rng.uniform(-5, 5, DIM)) for _ in range(BUDGET))

results = {}
for name, f in [("sphere (smooth)", sphere), ("staircase (non-diff.)", staircase)]:
    results[name] = {"gradient descent": np.mean([grad_descent(f) for _ in range(5)]),
                     "random search": np.mean([random_search(f) for _ in range(5)])}
print("\nmean final f under an EQUAL evaluation budget (lower = better):")
print(f"  {'function':24s} {'gradient descent':>17s} {'random search':>15s}   winner")
for name, r in results.items():
    win = min(r, key=r.get)
    print(f"  {name:24s} {r['gradient descent']:17.3f} {r['random search']:15.3f}   {win}")
print("\ngradient descent wins on the smooth function; random search wins on the flat-stepped one")
print("(where the gradient is useless). NO single method is best everywhere: the No Free Lunch")
print("theorem (Chapter 4): a method only wins by matching the problem's structure.")

print("\nAll DE / no-free-lunch checks ran.")

# %% [markdown]
# ## 22.8.3 The libraries you would actually call

# %% [markdown]
# One call shows the flavor, running SciPy's DE with its default strategy on the same five-dimensional Rastrigin function the chapter has used throughout.

# %%
import numpy as np
from scipy.optimize import differential_evolution

def rastrigin(x): return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))

res = differential_evolution(rastrigin, bounds=[(-5.12, 5.12)] * 5,
                             seed=0, maxiter=300, polish=False)  # polish=False keeps it pure DE
print(f"scipy differential_evolution: f = {res.fun:.4f} at x = {np.round(res.x, 4)}")

# %% [markdown]
# ## 22.9 Where evolutionary methods are used in practice

# %% [markdown]
# Example 22-17. Feature selection by genetic algorithm.

# %%
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split

data = load_breast_cancer()
X, y = data.data, data.target                            # 569 tumors, 30 features, malignant vs benign
D = X.shape[1]

# How redundant are the thirty features? Count the strongly correlated pairs.
r = np.abs(np.corrcoef(X, rowvar=False))[np.triu_indices(D, 1)]
print(f"feature pairs: {r.size}; |r| above 0.8: {(r > 0.8).sum()}; above 0.9: {(r > 0.9).sum()}")

# Hold out a stratified quarter of the tumors BEFORE the search: the GA never sees them.
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
print(f"training tumors: {len(ytr)}; held-out test tumors: {len(yte)}")

def model():                                             # standardize, then logistic regression
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))

def fitness(mask):                                       # scored on the training tumors only
    if mask.sum() == 0: return 0.0
    acc = cross_val_score(model(), Xtr[:, mask], ytr, cv=5).mean()
    return acc - 0.002 * mask.sum()                      # reward accuracy, gently penalize using many features

baseline = cross_val_score(model(), Xtr, ytr, cv=5).mean()
print(f"\nall {D} features: training CV accuracy = {baseline:.3f}")

rng = np.random.default_rng(0)
POP, GENS = 30, 25
pop = [rng.random(D) < 0.5 for _ in range(POP)]
for gen in range(GENS):
    fits = np.array([fitness(m) for m in pop])
    order = np.argsort(-fits, kind="stable")             # best first; tied scores keep population order (Section 22.3.2)
    pop = [pop[i] for i in order]
    new = pop[:4]                                        # elitism
    while len(new) < POP:
        a, b = pop[rng.integers(8)], pop[rng.integers(8)]    # parents from the top 8
        child = np.where(rng.random(D) < 0.5, a, b)          # uniform crossover
        flip = rng.random(D) < 0.05                           # mutation: flip ~5% of bits
        child = np.where(flip, ~child, child)
        new.append(child)
    pop = new

best = pop[int(np.argmax([fitness(m) for m in pop]))]
sel_cv = cross_val_score(model(), Xtr[:, best], ytr, cv=5).mean()
print(f"GA-selected {int(best.sum())} of {D} features: training CV accuracy = {sel_cv:.3f}  (the score the GA maximized)")
print("chosen features:", [str(n) for n in data.feature_names[best]])

# The honest comparison: fit each model on the training tumors, score it once on the held-out ones.
full_acc = model().fit(Xtr, ytr).score(Xte, yte)
sub_acc = model().fit(Xtr[:, best], ytr).score(Xte[:, best], yte)
print(f"\nheld-out accuracy on {len(yte)} tumors: all {D} features {full_acc:.3f} "
      f"({round(full_acc * len(yte))} correct), GA subset {sub_acc:.3f} ({round(sub_acc * len(yte))} correct)")
print("the GA searched a 2^30 space of subsets using only model-accuracy evaluations, no gradient.")

print("\nwhere evolutionary / swarm methods are used in practice:")
print("  feature selection (Example 22-17): pick a feature subset to maximize a model's score")
print("  hyperparameter tuning & NAS: search learning rates, architectures (non-diff., expensive)")
print("  combinatorial optimization: TSP, vehicle routing, scheduling, bin packing, circuit layout")
print("  black-box RL: evolve policies from the reward when it is non-differentiable (ES, 22.6)")
print("the common thread: a non-differentiable, often discrete objective you can only EVALUATE.")

print("\nAll feature-selection checks ran.")

# %% [markdown]
# ## 22.10 Hands-on lab: TSP by genetic algorithm and a neuroevolved controller

# %% [markdown]
# Example 22-18. The lab: TSP by genetic algorithm and a neuroevolved controller.

# %%
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import gymnasium as gym
rng = np.random.default_rng(0)

# Same rounded capital pairs as Example 22-14, repeated so this block runs on its own.
capitals = {
    "Sacramento": (38.58, -121.49), "Phoenix": (33.45, -112.07), "Denver": (39.74, -104.98),
    "Austin": (30.27, -97.74), "Salem": (44.94, -123.04), "Boise": (43.61, -116.20),
    "Santa Fe": (35.69, -105.94), "Oklahoma City": (35.47, -97.52), "Topeka": (39.05, -95.68),
    "Des Moines": (41.60, -93.61), "Saint Paul": (44.94, -93.09), "Jefferson City": (38.58, -92.17),
    "Springfield": (39.80, -89.64), "Nashville": (36.17, -86.78), "Atlanta": (33.75, -84.39),
    "Columbus": (39.96, -83.00), "Montgomery": (32.37, -86.30), "Tallahassee": (30.44, -84.28),
    "Raleigh": (35.77, -78.64), "Richmond": (37.55, -77.46),
}
names = list(capitals); cities = np.array([capitals[n] for n in names])
NC = len(names)
D = np.sqrt(((cities[:, None] - cities[None, :]) ** 2).sum(2)) + np.eye(NC)
def length(t): return sum(D[t[i], t[(i + 1) % NC]] for i in range(NC))
def nearest_neighbor():
    unv = set(range(1, NC)); tour = [0]
    while unv:
        nxt = min(unv, key=lambda c: D[tour[-1], c]); tour.append(nxt); unv.discard(nxt)
    return length(tour)

def order_crossover(p1, p2):
    a, b = sorted(rng.choice(NC, 2, replace=False))
    child = [-1] * NC; child[a:b] = p1[a:b]              # keep a slice of parent 1
    fill = [c for c in p2 if c not in child[a:b]]        # fill the rest in parent-2 order
    j = 0
    for i in list(range(b, NC)) + list(range(a)):
        child[i] = fill[j]; j += 1
    return child

def mutate(tour):                                        # reverse a random segment (2-opt-like)
    a, b = sorted(rng.choice(NC, 2, replace=False))
    return tour[:a] + tour[a:b][::-1] + tour[b:]

POP, GENS = 200, 300
pop = [list(rng.permutation(NC)) for _ in range(POP)]
for gen in range(GENS):
    pop.sort(key=length)
    new = pop[:10]                                       # elitism
    while len(new) < POP:
        p1 = min((pop[i] for i in rng.integers(0, 40, 3)), key=length)   # tournament among the top 40
        p2 = min((pop[i] for i in rng.integers(0, 40, 3)), key=length)
        child = order_crossover(p1, p2)
        if rng.random() < 0.4: child = mutate(child)
        new.append(child)
    pop = new
pop.sort(key=length)
ga_len = length(pop[0])

# baseline: the average random tour (nearest_neighbor is the greedy tour of Example 22-14)
rand_len = np.mean([length(list(rng.permutation(NC))) for _ in range(3000)])
print("PART 1 - Traveling Salesman Problem (20 real US state capitals), tour length (lower is better):")
print(f"  random tour (avg)      : {rand_len:.2f}")
print(f"  nearest-neighbor       : {nearest_neighbor():.2f}")
print(f"  genetic algorithm      : {ga_len:.2f}  (shorter than nearest neighbor; 2-opt can be shorter)")

env = gym.make("CartPole-v1")
NO, NA = 4, 2
def policy_action(w, obs):
    W = w[:NO * NA].reshape(NO, NA); b = w[NO * NA:]
    return int(np.argmax(obs @ W + b))                   # linear policy: 4 obs -> 2 action scores
def reward_of(w, episodes=3):
    total = 0.0
    for ep in range(episodes):
        obs, _ = env.reset(seed=1000 + ep); done = False
        while not done:
            obs, r, term, trunc, _ = env.step(policy_action(w, obs)); total += r; done = term or trunc
    return total / episodes

NWT = NO * NA + NA                                       # 10 weights
POP2, GENS2, MU = 24, 30, 6
popw = rng.normal(0, 1, (POP2, NWT))
steps = 0                                                # environment steps the search spends
print("\nPART 2 - neuroevolution on CartPole (max reward 500):")
for gen in range(GENS2):
    fits = np.array([reward_of(w) for w in popw])
    steps += round(3 * fits.sum())                       # CartPole pays +1 per step: 3 episodes x mean reward = steps
    elite = popw[np.argsort(-fits, kind="stable")[:MU]]  # best first; tied rewards keep population order (Section 22.3.2)
    if gen in (0, 10, 29):
        print(f"  generation {gen:2d}: best mean reward = {fits.max():.0f}")
    popw = np.vstack([elite, [elite[rng.integers(MU)] + rng.normal(0, 0.3, NWT) for _ in range(POP2 - MU)]])
best_reward = max(reward_of(w, episodes=5) for w in popw)
print(f"  final best policy: mean reward = {best_reward:.0f}")
print(f"  environment steps spent by the 30 generations of search: {steps:,}")
print("\nneuroevolution solved CartPole by evolving 10 weights from REWARD alone: no backprop,")
print("no value function, no policy gradient. Chapter 18's PPO also reached 500, with gradients and a")
print("30,000-timestep training budget; ES needed no gradient code but spent many more environment steps.")

print("\nAll evolutionary-lab checks ran.")

# %% [markdown]
# ## 22.10.1 The 2-opt move, and the exact tour

# %% [markdown]
# Example 22-19. The exact shortest tour of the twenty capitals by the Held-Karp dynamic program, with 2-opt local search for comparison.
#
# Example 22-19 prints $60822550204416000$ distinct tours, the exact optimum $122.71$, and three 2-opt lines: nearest neighbor $138.14$ (12.58 percent over), nearest neighbor then 2-opt $123.02$ (0.25 percent over), and the best of 30 random restarts $122.71$ (0.000 percent over), reached by 2 of the 30.

# %%
import math
import numpy as np

capitals = {
    "Sacramento": (38.58, -121.49), "Phoenix": (33.45, -112.07), "Denver": (39.74, -104.98),
    "Austin": (30.27, -97.74), "Salem": (44.94, -123.04), "Boise": (43.61, -116.20),
    "Santa Fe": (35.69, -105.94), "Oklahoma City": (35.47, -97.52), "Topeka": (39.05, -95.68),
    "Des Moines": (41.60, -93.61), "Saint Paul": (44.94, -93.09), "Jefferson City": (38.58, -92.17),
    "Springfield": (39.80, -89.64), "Nashville": (36.17, -86.78), "Atlanta": (33.75, -84.39),
    "Columbus": (39.96, -83.00), "Montgomery": (32.37, -86.30), "Tallahassee": (30.44, -84.28),
    "Raleigh": (35.77, -78.64), "Richmond": (37.55, -77.46),
}
cities = np.array(list(capitals.values()))
n = len(cities)
D = np.sqrt(((cities[:, None] - cities[None, :]) ** 2).sum(2))

def length(t):
    return float(sum(D[t[i], t[(i + 1) % n]] for i in range(n)))

def two_opt(tour):
    tour = list(tour)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n if i else n - 1):
                a, b, c, d = tour[i], tour[i + 1], tour[j], tour[(j + 1) % n]
                delta = D[a, c] + D[b, d] - D[a, b] - D[c, d]
                if delta < -1e-12:
                    tour[i + 1:j + 1] = reversed(tour[i + 1:j + 1])
                    improved = True
    return tour

def held_karp(dist):
    size = 1 << n
    inf = 1e100
    dp = np.full((size, n), inf)
    dp[1, 0] = 0.0
    for mask in range(size):
        for u in range(n):
            if (mask & (1 << u)) == 0 or dp[mask, u] >= inf:
                continue
            for v in range(n):
                if mask & (1 << v):
                    continue
                cand = dp[mask, u] + dist[u, v]
                nm = mask | (1 << v)
                if cand < dp[nm, v]:
                    dp[nm, v] = cand
    full = size - 1
    return float(min(dp[full, u] + dist[u, 0] for u in range(1, n)))

opt = held_karp(D)
print(f"distinct tours (n-1)!/2 = {math.factorial(n - 1) // 2}")
print(f"exact optimum {opt:.2f}")

def nearest_neighbor_tour():                             # the greedy tour of Example 22-14, from city 0
    tour, left = [0], set(range(1, n))
    while left:
        nxt = min(left, key=lambda c: D[tour[-1], c]); tour.append(nxt); left.discard(nxt)
    return tour

def gap(x):                                              # percent over the exact optimum
    return 100 * (x - opt) / opt

nn = length(nearest_neighbor_tour())
nn_2opt = length(two_opt(nearest_neighbor_tour()))
rng = np.random.default_rng(0)
restarts = [length(two_opt(rng.permutation(n))) for _ in range(30)]
best = min(restarts)
print(f"nearest neighbor {nn:.2f} ({gap(nn):.2f} percent over)")
print(f"nearest neighbor then 2-opt {nn_2opt:.2f} ({gap(nn_2opt):.2f} percent over)")
print(f"best of 30 random 2-opt restarts {best:.2f} ({gap(best):.3f} percent over); "
      f"{sum(abs(r - opt) < 1e-9 for r in restarts)} of the 30 end at the optimum")

# %% [markdown]
# ## 22.11 Coding exercises

# %% [markdown]
# 1. When gradients fail, as the dimension grows.

# %%
import numpy as np
def rastrigin(x):
    return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))
def compare(dim, seed):
    ...
print('starter', [float(rastrigin(np.zeros(d))) for d in (2, 5, 10, 20)])

# %% [markdown]
# 2. Local search at equal budgets.

# %%
import numpy as np
def rastrigin(x):
    return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))
def hill_mean(n_evals):
    ...
def anneal_mean(n_evals, T0=3.0):
    ...
print('final temperature', round(3.0 * 0.999 ** 4000, 3))

# %% [markdown]
# 3. Does crossover pull its weight?

# %%
import numpy as np
def rastrigin(X):
    return 10 * X.shape[-1] + np.sum(X ** 2 - 10 * np.cos(2 * np.pi * X), axis=-1)
def run_ga(dim, seed, crossover=True):
    ...
print('starter', rastrigin(np.zeros((3, 10))).tolist())

# %% [markdown]
# 4. Shape adaptation on a turned valley.

# %%
import numpy as np
def valley(X):
    return X[..., 0] ** 2 + 100 * X[..., 1] ** 2
def turned(X):
    ...
def es(f, start, per_axis, seed):
    ...
print('starter', valley(np.array([8.0, 8.0])), round(8 * np.sqrt(2), 4))

# %% [markdown]
# 5. Genetic programming and the function set.

# %%
import numpy as np
xs = np.linspace(-3, 3, 40)
ys = xs + np.sin(2 * xs)
def gp(ops, seed=1):
    ...
print(xs.shape, round(float(ys[0]), 3))

# %% [markdown]
# 6. Neuroevolution without backprop.

# %%
from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)
print(X.shape, len(set(y)))

# %% [markdown]
# 7. Particle swarm inertia.

# %%
import numpy as np
def rastrigin(X):
    return 10 * X.shape[-1] + np.sum(X ** 2 - 10 * np.cos(2 * np.pi * X), axis=-1)
def pso(w, seed, n_particles=40, iters=200):
    ...
print('starter', rastrigin(np.zeros((40, 5))).shape)

# %% [markdown]
# 8. Ant colony optimization on a real map.

# %%
import numpy as np
CAP = {"Sacramento": (38.58, -121.49), "Phoenix": (33.45, -112.07), "Denver": (39.74, -104.98),
       "Austin": (30.27, -97.74), "Des Moines": (41.60, -93.61), "Springfield": (39.80, -89.64),
       "Nashville": (36.17, -86.78), "Atlanta": (33.75, -84.39), "Columbus": (39.96, -83.00),
       "Raleigh": (35.77, -78.64), "Harrisburg": (40.27, -76.88), "Boston": (42.36, -71.06)}
cities = np.array(list(CAP.values()))   # (latitude, longitude) in degrees, one row per capital
def nearest_neighbor_length(cities):
    ...
def aco(cities, seed):
    ...
print(len(CAP), CAP["Boston"])

# %% [markdown]
# 9. Differential evolution's two knobs.

# %%
import numpy as np
def rastrigin(x):
    return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))
def differential_evolution(f, F, CR, seed, pop=40, budget=15000):
    ...
def random_search(f, seed, budget=15000):
    ...
print('starter', int(rastrigin(np.zeros(5))))

# %% [markdown]
# 10. Miniproject: feature selection by genetic algorithm.

# %%
from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(return_X_y=True)
print(X.shape)

# %% [markdown]
# 11. Schema bound.

# %%
import numpy as np
def schema_bound(delta, order, m=30, ratio=1.2, pc=0.7, pm=0.02, ell=8):
    ...
print("schema starter", 8)

# %% [markdown]
# 12. Protected division.

# %%
import numpy as np
def protected_div(a, b):
    ...
xs = np.linspace(-3, 3, 40)
print(xs.shape)

# %% [markdown]
# 13. CMA-ES arithmetic.

# %%
import numpy as np
def cma_step(draws):
    ...
print("cma starter", len([(-1, 0.2), (2, 0), (0, -1), (0.5, 0.5)]))
