"""Appendix H solutions for Chapter 9, as runnable cells.

Generated from Solutions.md (Appendix H). Every code block of this chapter's solutions appears here
verbatim, split into one cell per exercise or drill topic, so each answer can be run and checked against
the expected outputs it states. Keep Solutions.md as the source; regenerate after any change.
"""

# %% [markdown]
# ## 9.14 coding exercises

# %%
import warnings
import numpy as np

# %% [markdown]
# 1. Bayesian estimate of a proportion, from scratch (breast cancer base rate)

# %%
# 1. Bayesian estimate of a proportion, from scratch (breast cancer base rate)
from sklearn.datasets import load_breast_cancer
from scipy.stats import beta as beta_dist
y = load_breast_cancer().target                 # 1 = benign, 0 = malignant; 569 real tumors
mal = int((y == 0).sum()); ben = int((y == 1).sum())
a0, b0 = 2.0, 2.0                               # weak Beta(2,2) prior on the malignant rate
a_post, b_post = a0 + mal, b0 + ben             # conjugate update: add successes and failures
mle = mal / (mal + ben)
post_mean = a_post / (a_post + b_post)          # squared-error loss  -> posterior mean
post_median = float(beta_dist(a_post, b_post).median())  # absolute-error loss -> median
post_mode = (a_post - 1) / (a_post + b_post - 2)         # 0-1 loss -> mode (MAP)
ci = beta_dist.ppf([0.025, 0.975], a_post, b_post)       # 95% credible interval
print(mal, ben, round(mle, 4))
# expected output: 212 357 0.3726
print(round(post_mean, 4), round(post_median, 4), round(post_mode, 4))
# expected output: 0.3735 0.3733 0.373
print(ci.round(4).tolist())
# expected output: [0.3343, 0.4134]
# the prior barely moves the estimate here because 569 tumors overwhelm 4 pseudo-counts

# %% [markdown]
# 2. L2 regularization is a Gaussian prior (diabetes)

# %%
# 2. L2 regularization is a Gaussian prior (diabetes)
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
X, yy = load_diabetes(return_X_y=True)
X = StandardScaler().fit_transform(X); yy = yy - yy.mean()
sigma2, tau2 = 4.0, 1.0                          # noise variance, prior variance
A = X.T @ X / sigma2 + np.eye(X.shape[1]) / tau2 # posterior precision
mu_map = np.linalg.solve(A, X.T @ yy / sigma2)   # MAP = posterior mean
ridge = Ridge(alpha=sigma2 / tau2, fit_intercept=False).fit(X, yy)
print("MAP equals ridge within 1e-10:", bool(np.max(np.abs(mu_map - ridge.coef_)) < 1e-10))
# expected output: MAP equals ridge within 1e-10: True
# the gap itself is float64 rounding error, whose digits depend on the machine (9.2.1)
for a in [1.0, 10.0, 100.0]:                     # tighter prior (bigger alpha) shrinks the weights
    c = Ridge(alpha=a, fit_intercept=False).fit(X, yy).coef_
    print(a, round(float(np.sqrt(np.sum(c ** 2))), 3))
# expected output: 1.0 57.527
# expected output: 10.0 42.569
# expected output: 100.0 34.752

# %% [markdown]
# 3. Conjugate priors: Normal-Inverse-Gamma + Dirichlet-Multinomial (penguins)

# %%
# 3. Conjugate priors: Normal-Inverse-Gamma + Dirichlet-Multinomial (penguins)
import seaborn as sns
from scipy.stats import t as student_t
pen = sns.load_dataset("penguins").dropna()      # the starter's filter: 333 complete rows
x = pen.query("species == 'Adelie'")["bill_length_mm"].to_numpy()   # real bill lengths (mm)
n, xbar = len(x), x.mean()
mu0, k0, al0, be0 = 40.0, 1.0, 2.0, 1.0          # Normal-Inverse-Gamma prior hyperparameters
kn = k0 + n
mun = (k0 * mu0 + n * xbar) / kn                 # accumulate the sufficient statistics
an = al0 + n / 2
bn = be0 + 0.5 * np.sum((x - xbar) ** 2) + 0.5 * k0 * n * (xbar - mu0) ** 2 / kn
var_mean = bn / (an - 1)                         # posterior mean of the variance
scale = np.sqrt(bn * (kn + 1) / (an * kn))
lo, hi = student_t.ppf([0.025, 0.975], 2 * an, loc=mun, scale=scale)   # Student-t predictive
print(n, round(xbar, 3), round(mun, 3), round(var_mean, 3))
# expected output: 146 38.824 38.832 6.969
print(round(lo, 2), round(hi, 2))
# expected output: 33.63 44.03
counts = pen["species"].value_counts().sort_index()
post = np.ones(len(counts)) + counts.to_numpy()  # flat Dirichlet prior + observed counts
print(counts.to_dict())
# expected output: {'Adelie': 146, 'Chinstrap': 68, 'Gentoo': 119}
print((post / post.sum()).round(3).tolist())     # smoothed probabilities, none ever zero
# expected output: [0.438, 0.205, 0.357]

# %% [markdown]
# 4. Approximate inference toolkit on a Beta(40,10) posterior

# %%
# 4. Approximate inference toolkit on a Beta(40,10) posterior
from scipy import optimize, stats
rng = np.random.default_rng(0)
A_, B_ = 40, 10
def target(th):                                  # unnormalized Beta(40,10) density
    return th ** (A_ - 1) * (1 - th) ** (B_ - 1)
true_mean = A_ / (A_ + B_)
grid = np.linspace(1e-6, 1 - 1e-6, 200000)
c = target(grid).max() * 1.01                    # envelope constant c that bounds the density
prop = rng.uniform(0, 1, 300000)
keep = rng.uniform(0, c, 300000) < target(prop)  # rejection sampling
print(round(keep.mean(), 3), round(prop[keep].mean(), 3))
# expected output: 0.138 0.799
xs = rng.uniform(0, 1, 300000); w = target(xs)   # self-normalized importance sampling
print(round(float(np.sum(w * xs) / np.sum(w)), 3), int(np.sum(w) ** 2 / np.sum(w ** 2)))
# expected output: 0.8 58953
def log_post(t):                                 # only the unnormalized log-posterior is needed
    if t <= 0 or t >= 1: return -np.inf
    return (A_ - 1) * np.log(t) + (B_ - 1) * np.log(1 - t)
def chain(start, n=20000, step=0.15):            # from-scratch random-walk Metropolis-Hastings
    t = start; lp = log_post(t); out = np.empty(n); acc = 0
    for i in range(n):
        pr = t + rng.normal(0, step); lpp = log_post(pr)
        if np.log(rng.uniform()) < lpp - lp:     # accept with prob min(1, posterior ratio)
            t, lp, acc = pr, lpp, acc + 1
        out[i] = t
    return out, acc / n
draws, rate = chain(0.8); s = draws[2000:]       # discard burn-in
print(round(rate, 2), round(s.mean(), 3), round(true_mean, 3))
# expected output: 0.41 0.8 0.8
chains = np.array([chain(st)[0][2000:] for st in (0.6, 0.7, 0.8, 0.9)])
mm, nn = chains.shape
Bv = nn * chains.mean(1).var(ddof=1); Wv = chains.var(1, ddof=1).mean()
print(round(np.sqrt(((nn - 1) / nn * Wv + Bv / nn) / Wv), 3))          # R-hat across 4 chains
# expected output: 1.0
def neg_lp(t): return -log_post(t)               # Laplace approximation: Gaussian at the mode
mode = optimize.minimize_scalar(neg_lp, bounds=(1e-6, 1 - 1e-6), method="bounded").x
curv = (neg_lp(mode + 1e-5) - 2 * neg_lp(mode) + neg_lp(mode - 1e-5)) / 1e-5 ** 2
print(round(mode, 3), round(1 / curv ** 0.5, 3), round(float(stats.beta(A_, B_).std()), 3))
# expected output: 0.813 0.056 0.056
# all four agree on ~0.8; Bishop (2006, Section 11.2) notes that rejection and importance
# sampling are severely limited in high dimensions, which is why he turns to MCMC next

# %% [markdown]
# 5. Gaussian process on the annual sunspot record + Bayesian optimization

# %%
# 5. Gaussian process on the annual sunspot record + Bayesian optimization
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, WhiteKernel, ConstantKernel as C
import statsmodels.api as sm
sun = sm.datasets.sunspots.load_pandas().data    # 309 years of annual sunspot numbers, 1700-2008
Xg = (sun["YEAR"].to_numpy() - 1700.0).reshape(-1, 1)
yg = sun["SUNACTIVITY"].to_numpy()
def sun_kernel(p0):                              # the composite kernel, period started at p0 years
    return (C(1.0) * RBF(40.0)                   # slow envelope
            + C(1.0) * ExpSineSquared(length_scale=5.0, periodicity=p0,
                                      periodicity_bounds=(8.0, 15.0))   # the solar cycle
            + WhiteKernel(0.2))                  # observation noise
from scipy.optimize import minimize
def tight(obj_func, theta0, bounds):             # L-BFGS-B run until the LML stops changing
    r = minimize(obj_func, theta0, jac=True, method="L-BFGS-B", bounds=bounds,
                 options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 20000})
    return r.x, r.fun
gp = GaussianProcessRegressor(sun_kernel(11.0), normalize_y=True, random_state=0,
                              optimizer=tight).fit(Xg, yg)       # no random restarts (see below)
print(round(float(gp.kernel_.k1.k2.k2.periodicity), 2), round(float(gp.log_marginal_likelihood_value_), 2))
# expected output: 11.0 -367.82
# the fitted period and the log marginal likelihood (LML) it reached; now refit from other starts
for p0 in [9.0, 10.0, 12.0, 13.0]:               # no restarts, so each fit keeps its own start
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")          # count the warnings instead of hiding them
        g5 = GaussianProcessRegressor(sun_kernel(p0), normalize_y=True, random_state=0,
                                      optimizer=tight).fit(Xg, yg)
    print(p0, round(float(g5.kernel_.k1.k2.k2.periodicity), 2),
          round(float(g5.log_marginal_likelihood_value_), 2),
          f"{float(g5.kernel_.k1.k2.k1.constant_value):.4g}", len(caught))   # four significant digits
# expected output: 9.0 15.0 -175.78 1e-05 2
# expected output: 10.0 11.0 -367.82 0.7156 0
# expected output: 12.0 15.0 -175.78 1e-05 3
# expected output: 13.0 15.0 -175.78 1e-05 3
print(round(float(g5.kernel_.k1.k1.k2.length_scale), 2),     # start-13 fit: RBF length scale,
      round(float(g5.kernel_.k2.noise_level), 4),            # its noise level,
      round(float(gp.kernel_.k2.noise_level), 3))            # and the 11-year fit's noise level
# expected output: 2.0 0.0277 0.542
# The LML is multimodal in the period. From a start of 10 or 11 the optimizer stops at 11.0
# with LML -367.82. From 9, 12, or 13 it reaches -175.78, 192 log units higher, but there the
# periodic term's constant (its variance, the fourth column) has fallen to its 1e-05 floor and
# the period has run to its 15-year bound, which the ConvergenceWarnings (last column) report.
# That optimum switches the cycle off: an RBF with a 2-year length scale follows each cycle and
# the noise level falls from 0.542 to 0.0277 of the normalized variance, so its period means
# nothing. The 11.0 is the local optimum that the 11-year start selects, not a period the data
# chose over all others, so the printed period is a property of the initialization as much as
# of the record. Two choices keep these digits the same when the data move in their last bit.
# Every fit runs L-BFGS-B until the LML stops changing (ftol 1e-15, gtol 1e-10): under
# scikit-learn's default optimizer, SciPy's L-BFGS-B with ftol 2.2e-9 and gtol 1e-5, the
# start-10 fit stopped on a flat ridge of the LML, where its fourth column ranged from 0.71
# to 0.72 and once stopped elsewhere, at 1.28. And the main fit uses no random restarts: with
# n_restarts_optimizer=3, one restart begins at extreme values drawn from the kernel's bounds,
# and from there the optimizer reached a period of 13.2 (LML -402.55) or the cycle-off optimum
# near -175.3, depending on the last bit of the data, and the best of the four fits changed
# with it
grid = np.linspace(0, float(Xg.max()) + 20, 500).reshape(-1, 1)
_, std = gp.predict(grid, return_std=True)
i_in = np.argmin(abs(grid.ravel() - 150))
print(round(float(std[i_in]), 2), round(float(std[-1]), 2))
# expected output: 30.64 35.24
# the band is wide everywhere because annual sunspot counts are genuinely noisy, so the
# noise floor dominates; the forecast widens on top of it, less dramatically than clean CO2
_, std5 = g5.predict(grid, return_std=True)
print(round(float(std5[i_in]), 2), round(float(std5[-1]), 2))
# expected output: 8.25 41.58
# the cycle-off fit from the start at 13 is tighter inside the record, where its short RBF
# tracks each cycle, and wider 20 years out, where that RBF has forgotten the data
def black_box(v): return np.sin(2 * v) + 0.3 * v # the optimizer never sees this formula
dom = np.linspace(-3, 3, 400).reshape(-1, 1)
Xo = np.array([[-2.5], [2.5]]); yo = black_box(Xo).ravel()       # 2 initial probes
for _ in range(10):
    g = GaussianProcessRegressor(C(1.0) * RBF(1.0), alpha=1e-6, normalize_y=True,
                                 random_state=0).fit(Xo, yo)
    mu, sd = g.predict(dom, return_std=True)
    nxt = dom[np.argmax(mu + 2.0 * sd)]          # upper-confidence-bound acquisition
    Xo = np.vstack([Xo, nxt]); yo = np.append(yo, black_box(nxt))
print(round(float(Xo[np.argmax(yo)][0]), 3), round(float(dom[np.argmax(black_box(dom))][0]), 3))
# expected output: 0.865 0.865
# (12 evaluations land exactly on the grid's interior optimum ~0.86)

# %% [markdown]
# 6. Explaining away in the sprinkler network of Russell and Norvig (2010, Figure 14.12(a))

# %%
# 6. Explaining away in the sprinkler network of Russell and Norvig (2010, Figure 14.12(a))
from itertools import product
pC = {True: 0.5, False: 0.5}
pS = {True: 0.1, False: 0.5}                     # P(Sprinkler=T | Cloudy)
pR = {True: 0.8, False: 0.2}                     # P(Rain=T | Cloudy)
pW = {(True, True): 0.99, (True, False): 0.9, (False, True): 0.9, (False, False): 0.0}
def joint(c, s, r, w):
    p = pC[c] * (pS[c] if s else 1 - pS[c]) * (pR[c] if r else 1 - pR[c])
    return p * (pW[(s, r)] if w else 1 - pW[(s, r)])
def prob(query, evidence):                       # P(query | evidence) by enumerating 16 rows
    num = den = 0.0
    for c, s, r, w in product([True, False], repeat=4):
        asg = {"C": c, "S": s, "R": r, "W": w}
        if all(asg[k] == v for k, v in evidence.items()):
            p = joint(c, s, r, w); den += p
            if all(asg[k] == v for k, v in query.items()): num += p
    return num / den
print(round(prob({"S": True}, {}), 3))                        # prior
# expected output: 0.3
print(round(prob({"S": True}, {"W": True}), 3))               # wet grass raises it
# expected output: 0.43
print(round(prob({"S": True}, {"W": True, "R": True}), 3))    # rain explains it away
# expected output: 0.194
print(round(prob({"S": True}, {"W": True, "R": False}), 3))   # no rain: must be the sprinkler
# expected output: 1.0
# the last value is exactly 1 because P(W | not S, not R) = 0: wet grass with no rain
# leaves the sprinkler as the only possible cause

# %% [markdown]
# 7. HMM: the three algorithms on a machine-health model

# %%
# 7. HMM: the three algorithms on a machine-health model
from scipy.special import logsumexp
states = ["OK", "Faulty"]
pi = np.array([0.9, 0.1]); Amat = np.array([[0.95, 0.05], [0.2, 0.8]])
Bmat = np.array([[0.7, 0.25, 0.05], [0.1, 0.4, 0.5]])   # emissions: normal, warning, error
obs = [0, 1, 2, 2]                               # observed: normal, warning, error, error
def forward(o):
    al = pi * Bmat[:, o[0]]
    for k in o[1:]: al = (al @ Amat) * Bmat[:, k]
    return al.sum()
brute = 0.0
for p in product(range(2), repeat=4):            # all 2^4 = 16 hidden paths
    pr = pi[p[0]] * Bmat[p[0], obs[0]]
    for t in range(1, 4): pr *= Amat[p[t - 1], p[t]] * Bmat[p[t], obs[t]]
    brute += pr
print(round(forward(obs), 6), round(brute, 6), np.isclose(forward(obs), brute))
# expected output: 0.004658 0.004658 True
def viterbi(o):
    delta = pi * Bmat[:, o[0]]; back = []
    for k in o[1:]:
        sc = delta[:, None] * Amat; back.append(sc.argmax(0)); delta = sc.max(0) * Bmat[:, k]
    path = [int(delta.argmax())]
    for b in reversed(back): path.append(int(b[path[-1]]))
    return path[::-1]
print([states[s] for s in viterbi(obs)])
# expected output: ['OK', 'Faulty', 'Faulty', 'Faulty']
T = len(obs)
al = np.zeros((T, 2)); al[0] = pi * Bmat[:, obs[0]]
for t in range(1, T): al[t] = (al[t - 1] @ Amat) * Bmat[:, obs[t]]
be = np.zeros((T, 2)); be[-1] = 1.0                        # backward pass
for t in range(T - 2, -1, -1): be[t] = Amat @ (Bmat[:, obs[t + 1]] * be[t + 1])
gam = al * be; gam /= gam.sum(1, keepdims=True)            # smoothing posteriors P(z_t | all)
print(gam[:, 1].round(3).tolist())                         # P(Faulty) each day
# expected output: [0.115, 0.559, 0.887, 0.904]
# day two is already 0.559 despite only a warning, because the two later errors flow
# backward through beta: the machine was probably already faulty when it warned
la = np.log(pi) + np.log(Bmat[:, obs[0]])                  # same likelihood in log space
for t in range(1, T): la = logsumexp(la[:, None] + np.log(Amat), axis=0) + np.log(Bmat[:, obs[t]])
print(round(float(logsumexp(la)), 6), round(float(np.exp(logsumexp(la))), 6))
# expected output: -5.369077 0.004658

# %% [markdown]
# 8. Kalman filter: sensor-quality sweep + RTS smoother

# %%
# 8. Kalman filter: sensor-quality sweep + RTS smoother
rng = np.random.default_rng(0)
F = np.array([[1, 1.0], [0, 1]]); H = np.array([[1.0, 0.0]])
Q = np.eye(2) * 0.05; Tn = 60                    # state = [position, velocity]
true_x = np.zeros((Tn, 2)); true_x[0] = [0.0, 1.0]
for k in range(1, Tn): true_x[k] = F @ true_x[k - 1] + rng.multivariate_normal([0, 0], Q)
noise = rng.normal(0, 1, Tn)                     # one fixed unit-noise draw, scaled per R
for Rv in [1.0, 9.0, 25.0]:
    meas = true_x[:, 0] + np.sqrt(Rv) * noise
    R = np.array([[Rv]])
    x, P = np.array([0.0, 0.0]), np.eye(2) * 10.0
    est = np.zeros(Tn)
    for k in range(Tn):
        x = F @ x; P = F @ P @ F.T + Q                            # predict
        S = H @ P @ H.T + R; K = P @ H.T @ np.linalg.inv(S)       # Kalman gain
        x = x + (K @ (meas[k] - H @ x).reshape(1,)).ravel()
        P = (np.eye(2) - K @ H) @ P
        est[k] = x[0]
    print(Rv, round(float(K[0, 0]), 3), round(float(np.sqrt(np.mean((est - true_x[:, 0]) ** 2))), 3),
          round(float(np.sqrt(np.mean((meas - true_x[:, 0]) ** 2))), 3))
# expected output: 1.0 0.508 0.706 0.969
# expected output: 9.0 0.326 1.829 2.907
# expected output: 25.0 0.261 2.818 4.845
# noisier sensor -> smaller gain (trust the model more) and worse RMSE, but always beats raw
Rv = 9.0; meas = true_x[:, 0] + np.sqrt(Rv) * noise; R = np.array([[Rv]])
x, P = np.array([0.0, 0.0]), np.eye(2) * 10.0
xf = np.zeros((Tn, 2)); Pf = np.zeros((Tn, 2, 2)); xp = np.zeros((Tn, 2)); Pp = np.zeros((Tn, 2, 2))
for k in range(Tn):
    x = F @ x; P = F @ P @ F.T + Q; xp[k], Pp[k] = x, P
    S = H @ P @ H.T + R; K = P @ H.T @ np.linalg.inv(S)
    x = x + (K @ (meas[k] - H @ x).reshape(1,)).ravel()
    P = (np.eye(2) - K @ H) @ P; xf[k], Pf[k] = x, P
xs = xf.copy()
for k in range(Tn - 2, -1, -1):                  # RTS backward smoothing sweep
    Ck = Pf[k] @ F.T @ np.linalg.inv(Pp[k + 1])
    xs[k] = xf[k] + Ck @ (xs[k + 1] - xp[k + 1])
rmse = lambda e: round(float(np.sqrt(np.mean((e - true_x[:, 0]) ** 2))), 3)
print(rmse(xf[:, 0]), rmse(xs[:, 0]))
# expected output: 1.829 1.201
# the backward pass lets later data fix earlier states

# %% [markdown]
# 9. Survival analysis: the wexp covariate misbehaves

# %%
# 9. Survival analysis: the wexp covariate misbehaves
from lifelines.datasets import load_rossi
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test, proportional_hazard_test
rossi = load_rossi()                             # 432 released prisoners, followed one year
def km(dur, evt):                                # Kaplan-Meier product-limit estimator
    S = 1.0
    for t in np.unique(dur[evt == 1]):
        nr = np.sum(dur >= t); dd = np.sum((dur == t) & (evt == 1))   # at risk, events at t
        S *= 1 - dd / nr
    return S
for name, sub in [("no work exp", rossi[rossi.wexp == 0]), ("work exp", rossi[rossi.wexp == 1])]:
    print(name, len(sub), int(sub["arrest"].sum()),
          round(float(km(sub["week"].values, sub["arrest"].values)), 3))
# expected output: no work exp 185 62 0.665
# expected output: work exp 247 52 0.789
a = rossi[rossi.wexp == 0]; b = rossi[rossi.wexp == 1]
res = logrank_test(a["week"], b["week"], a["arrest"], b["arrest"])
print(round(float(res.test_statistic), 3), round(float(res.p_value), 4))
# expected output: 9.91 0.0016
# the marginal curves separate decisively
cph = CoxPHFitter().fit(rossi, duration_col="week", event_col="arrest")
print(round(float(cph.summary.loc["wexp", "exp(coef)"]), 3),
      round(float(cph.summary.loc["wexp", "p"]), 3))
# expected output: 0.861 0.48
# adjusted hazard ratio is weak: age and priors soak up the effect
ph = proportional_hazard_test(cph, rossi, time_transform="rank")
print(round(float(ph.summary.loc["wexp", "p"]), 4))
# expected output: 0.0068
# and proportional hazards fails for wexp: its effect drifts over the year
# presentation: show the KM curves (a real marginal difference), but do not quote 0.861 as
# "the" effect; wexp is confounded with age/priors and violates PH, so stratify or interact with time

# %% [markdown]
# 10. Miniproject: conformal prediction on diabetes and digits

# %%
# 10. Miniproject: conformal prediction on diabetes and digits
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes, load_digits
Xc, yc = load_diabetes(return_X_y=True)          # 442 patients, disease-progression target
X_tr, X_tmp, y_tr, y_tmp = train_test_split(Xc, yc, test_size=0.5, random_state=0)
X_cal, X_te, y_cal, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=0)
def split_conformal(model, alpha=0.1):
    model.fit(X_tr, y_tr)
    scores = np.abs(y_cal - model.predict(X_cal))     # nonconformity = |residual| on calibration set
    q = np.quantile(scores, np.ceil((len(scores) + 1) * (1 - alpha)) / len(scores))
    pred = model.predict(X_te)
    cov = np.mean((y_te >= pred - q) & (y_te <= pred + q))
    return cov, 2 * q
cov_gb, w_gb = split_conformal(GradientBoostingRegressor(random_state=0))
cov_lin, w_lin = split_conformal(LinearRegression())
print(round(cov_gb, 3), round(w_gb, 1), round(cov_lin, 3), round(w_lin, 1))
# expected output: 0.928 210.6 0.919 183.3
# both hit ~90% coverage, but here the LINEAR model earns the narrower band: on 442 nearly
# linear rows the booster overfits, so its calibration residuals are larger, the reverse
# of the California-housing result
for al in [0.2, 0.1, 0.05]:                      # both models at three coverage targets
    cov_g, _ = split_conformal(GradientBoostingRegressor(random_state=0), alpha=al)
    cov_l, _ = split_conformal(LinearRegression(), alpha=al)
    print(al, round(1 - al, 2), round(cov_g, 3), round(cov_l, 3))
# expected output: 0.2 0.8 0.82 0.892
# expected output: 0.1 0.9 0.928 0.919
# expected output: 0.05 0.95 0.955 0.937
# the test set has only 111 rows, so each coverage carries a binomial standard error of about
# 0.04 at the 80 percent level and 0.02 at 95 percent: 0.937 is within one standard error of
# 0.95, and the linear model's 0.892 at the 80 percent level is over coverage on this one split
Xd, yd = load_digits(return_X_y=True)            # 1797 digit images, 10 classes
Xd_tr, Xd_tmp, yd_tr, yd_tmp = train_test_split(Xd, yd, test_size=0.5, random_state=0)
Xd_cal, Xd_te, yd_cal, yd_te = train_test_split(Xd_tmp, yd_tmp, test_size=0.5, random_state=0)
clf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=2).fit(Xd_tr, yd_tr)
col = {c: i for i, c in enumerate(clf.classes_)}
cal_i = np.array([col[v] for v in yd_cal])
cal = 1 - clf.predict_proba(Xd_cal)[np.arange(len(yd_cal)), cal_i]   # nonconformity = 1 - P(true)
qh = np.quantile(cal, np.ceil((len(cal) + 1) * 0.9) / len(cal))
in_set = (1 - clf.predict_proba(Xd_te)) <= qh    # keep every confident-enough class
te_i = np.array([col[v] for v in yd_te])
cov = float(in_set[np.arange(len(yd_te)), te_i].mean()); size = float(in_set.sum(1).mean())
se = np.sqrt(0.9 * 0.1 / len(yd_te))             # binomial standard error of a 90% coverage on 450 digits
print(abs(cov - 0.9) <= 2 * se, round(size, 1), size < 1)
# expected output: True 0.9 True
# the coverage lies within two binomial standard errors (0.028) of the 90% target, as
# promised, and the average set size, 0.9, is below 1: the forest is so accurate that the
# calibrated threshold is strict, and ambiguous test digits whose top probability misses
# the threshold receive an EMPTY set, an abstention that counts as a miss; one fix is to
# always include the model's top class. The check stands in for the coverage's third
# decimal, which moved by one test digit of 450 between this book's two builds (0.893
# and 0.896), because the two builds' forests gave different probabilities

# %% [markdown]
# ## 9.14 coding drills

# %%
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from scipy.stats import beta, gamma

# %% [markdown]
# Bayes and conjugacy

# %%
# Bayes and conjugacy
print(round((2 + 8) / ((2 + 8) + (2 + 2)), 3))                # Beta(10,4) posterior mean
# expected output: 0.714
print(round((10 - 1) / (10 + 4 - 2), 3))                      # Beta(10,4) mode (MAP)
# expected output: 0.75
print(beta.ppf([0.025, 0.975], 10, 4).round(3).tolist())      # credible interval
# expected output: [0.462, 0.909]
a_post, b_post = 2.0 + 73, 1.0 + 20                           # Gamma(2,1) prior + 73 events / 20 periods
print(round(a_post / b_post, 3))                              # posterior mean rate
# expected output: 3.571
post = np.ones(6) + np.array([14, 14, 13, 12, 18, 49])   # flat Dirichlet + the 9.3 die counts
print(round((post / post.sum())[5], 3))                       # smoothed P(face 6)
# expected output: 0.397
prev, sens, spec = 0.01, 0.99, 0.99
print(round(sens * prev / (sens * prev + (1 - spec) * (1 - prev)), 3))
# expected output: 0.5
# P(disease | positive): the 1% base rate dominates

# %% [markdown]
# Approximate inference

# %%
# Approximate inference
lp1, lp0 = np.log(0.3 ** 9 * 0.7 ** 3), np.log(0.5 ** 9 * 0.5 ** 3)
print(round(min(1.0, np.exp(lp1 - lp0)), 3))                  # MH accept prob (ratio)
# expected output: 0.028
w = np.array([0.1, 0.4, 0.3, 0.2]); print((w / w.sum()).round(3).tolist())
# expected output: [0.1, 0.4, 0.3, 0.2]
# self-normalized importance weights
print(round(float(w.sum() ** 2 / (w ** 2).sum()), 3))         # Kish effective sample size
# expected output: 3.333
n = 20
print(round(-2 * (18 * np.log(0.9) + 2 * np.log(0.1)) + 1 * np.log(n), 2))
# expected output: 16.0
# BIC of the free-bias coin model
from scipy.special import beta as Bfun
print(round(Bfun(19, 3) / 0.5 ** 20, 1))                      # Bayes factor vs a fair coin
# expected output: 262.8

# %% [markdown]
# Sequential and graphical models

# %%
# Sequential and graphical models
pi = np.array([0.6, 0.4]); A = np.array([[0.7, 0.3], [0.4, 0.6]])
B = np.array([[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]])
alpha = pi * B[:, 0]; alpha = (alpha @ A) * B[:, 1]           # forward two steps: walk, shop
print(round(float(alpha.sum()), 5))                          # P(walk, shop)
# expected output: 0.1038
print(2 ** 3 - 1, 1 + 1 + 4)                                  # full joint vs alarm factors
# expected output: 7 6
rng = np.random.default_rng(1)
X = rng.normal(0, 1, 4000); Y = rng.normal(0, 1, 4000); Cc = X + Y + rng.normal(0, 0.5, 4000)
print(round(float(np.corrcoef(X, Y)[0, 1]), 3))              # independent as built
# expected output: -0.009
from sklearn.linear_model import LinearRegression
print(round(float(LinearRegression().fit(Y.reshape(-1, 1), X).coef_[0]), 3))   # X on Y alone
# expected output: -0.009
# regressing one on the other alone finds nothing
print(round(float(LinearRegression().fit(np.c_[Y, Cc], X).coef_[0]), 3))       # X on Y and C
# expected output: -0.789
# conditioning on the collider C couples X and Y

# %% [markdown]
# Causal inference

# %%
# Causal inference
rng = np.random.default_rng(0)
Xc = rng.normal(0, 1, 5000); Tc = Xc + rng.normal(0, 1, 5000)   # confounder X pushes the treatment T
Yc = 2.0 * Tc + 3.0 * Xc + rng.normal(0, 1, 5000)               # true effect of T on Y is 2.0
print(round(float(LinearRegression().fit(Tc.reshape(-1, 1), Yc).coef_[0]), 3))
# expected output: 3.486
# naive, confounded by X
print(round(float(LinearRegression().fit(np.c_[Tc, Xc], Yc).coef_[0]), 3))
# expected output: 2.013
# adjusting for X recovers the truth
Tr = rng.integers(0, 2, 5000); Yr = 2.0 * Tr + 3.0 * Xc + rng.normal(0, 1, 5000)
print(round(float(Yr[Tr == 1].mean() - Yr[Tr == 0].mean()), 3))
# expected output: 1.983
# randomization makes the plain difference unbiased
from sklearn.linear_model import LogisticRegression
rng = np.random.default_rng(0)                                  # the 9.10.5 simulation
Xp = rng.normal(0, 1, 5000)
Tp = rng.binomial(1, 1 / (1 + np.exp(-1.2 * Xp)))               # high Xp raises the chance of treatment
Yp = 2.0 * Tp + 3.0 * Xp + rng.normal(0, 1, 5000)               # true ATE is 2.0
e = LogisticRegression().fit(Xp.reshape(-1, 1), Tp).predict_proba(Xp.reshape(-1, 1))[:, 1]
wt = np.where(Tp == 1, 1 / e, 1 / (1 - e))                      # inverse-probability weights
ate = (np.sum(wt * Tp * Yp) / np.sum(wt * Tp)
       - np.sum(wt * (1 - Tp) * Yp) / np.sum(wt * (1 - Tp)))
print(round(float(Yp[Tp == 1].mean() - Yp[Tp == 0].mean()), 3), round(float(ate), 3))
# expected output: 4.872 2.083
# inverse-probability weighting pulls the confounded 4.872 back to about the true 2.0
sA = (81 + 192) / (87 + 263); sB = (234 + 55) / (270 + 80)
print(round(sA, 3), round(sB, 3), 81 / 87 > 234 / 270, 192 / 263 > 55 / 80)
# expected output: 0.78 0.826 True True
# Simpson: B wins overall, yet A wins on small stones (0.931 vs 0.867) and on large stones (0.730 vs 0.688)

# %% [markdown]
# Gaussian processes and conformal prediction

# %%
# Gaussian processes and conformal prediction
def rbf(x1, x2, ell=1.0): return float(np.exp(-(x1 - x2) ** 2 / (2 * ell ** 2)))
print(round(rbf(0.0, 0.0), 3), round(rbf(0.0, 2.0), 3))      # decays with distance
# expected output: 1.0 0.135
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
Xg = np.linspace(0, 5, 6).reshape(-1, 1); yg = np.sin(Xg).ravel()
g = GaussianProcessRegressor(RBF(1.0), alpha=1e-6).fit(Xg, yg)
_, s_in = g.predict([[2.5]], return_std=True); _, s_out = g.predict([[9.0]], return_std=True)
print(round(float(s_in[0]), 3), round(float(s_out[0]), 3))   # tiny near data, large far
# expected output: 0.003 0.952
scores = np.array([0.2, 0.5, 0.1, 0.9, 0.4, 0.3, 0.7, 0.6, 0.15, 0.25])
print(round(float(np.quantile(scores, np.ceil((len(scores) + 1) * 0.9) / len(scores))), 3))
# expected output: 0.9
# finite-sample conformal quantile for 90% coverage

# %% [markdown]
# Kalman and survival

# %%
# Kalman and survival
P, H, R = 4.0, 1.0, 9.0
print(round(P * H / (H * P * H + R), 3))                     # scalar Kalman gain
# expected output: 0.308
dur = np.array([5, 5, 8, 12, 12, 20]); evt = np.array([1, 1, 0, 1, 1, 0])   # 8 and 20 censored
S = 1.0
for t in np.unique(dur[evt == 1]):
    nr = np.sum(dur >= t); dd = np.sum((dur == t) & (evt == 1)); S *= 1 - dd / nr
print(round(float(S), 3))                                    # Kaplan-Meier survival
# expected output: 0.222
print(round(float(np.exp(-0.379)), 3))                       # Cox coef -> hazard ratio
# expected output: 0.685
# 9.9's table shows 0.684 for fin because it exponentiates the unrounded coefficient,
# about -0.3794; rounding the coefficient to -0.379 first moves the third decimal
