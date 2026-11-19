"""Chapter 9: reruns of the Experiment paragraphs and of the unbounded Bayesian optimization loop.

This is a verification harness, not part of the walkthrough. Each part repeats a listing from the
chapter with the single change its Experiment paragraph describes, and prints the numbers the
paragraph quotes, so every one of them can be checked:

  9.5   the Bayesian optimization loop with scikit-learn's default lengthscale bounds
  9.7   Baum-Welch with 500 observations instead of 50,000
  9.8   the Example 9-16 Kalman filter told R = 90 while the sensor's true variance stays 9
  9.9   the Cox model refit on fin alone, and covariate balance between the aid groups
  9.10  the regression discontinuity estimate at three bandwidths, with local lines and with plain means
  9.11  split conformal calibrated on 50 random calibration points, five draws
  9.12  Example 9-28 at target_accept 0.99, then on the six small counties alone (RAMSEY and STEARNS dropped)

Run it with:  python experiments_ch09.py   (the 9.12 part needs PyMC and network access for radon.csv)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ["PYTENSOR_FLAGS"] = "cxx=,numba__fastmath=False"   # the flags Example 9-26 sets (9.12)
import math, sys, warnings
import numpy as np

if hasattr(sys.stdout, "reconfigure"):          # a plain terminal; a notebook kernel's stream has no reconfigure
    sys.stdout.reconfigure(encoding="utf-8")

# %% 9.5: the loop of 9.5 with the RBF lengthscale left at scikit-learn's default bounds (1e-05, 1e5)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.exceptions import ConvergenceWarning


def black_box(x):
    return np.sin(3 * x) - 0.3 * (x - 1) ** 2


dom = np.linspace(-2, 4, 400).reshape(-1, 1)
Xo = np.array([[-1.5], [3.5]]); yo = black_box(Xo).ravel()
floor_hits, n_warn, floor_probes = 0, 0, []
for _ in range(10):
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        g = GaussianProcessRegressor(kernel=C(1.0) * RBF(1.0), alpha=1e-6,
                                     normalize_y=True, random_state=0).fit(Xo, yo)
    n_warn += sum(1 for w in wl if issubclass(w.category, ConvergenceWarning))
    at_floor = g.kernel_.k2.length_scale <= 1e-5 * 1.0001
    mu, sd = g.predict(dom, return_std=True)
    nxt = dom[np.argmax(mu + 2.0 * sd)]
    if at_floor:
        floor_hits += 1; floor_probes.append(round(float(nxt[0]), 3))
    Xo = np.vstack([Xo, nxt]); yo = np.append(yo, black_box(nxt))
print("9.5  fits at the lengthscale floor:", floor_hits, " probes they chose:", floor_probes,
      " ConvergenceWarnings:", n_warn, " best x:", round(float(Xo[np.argmax(yo)][0]), 3))

# %% 9.7: Baum-Welch on 500 observations instead of 50,000
from hmmlearn.hmm import CategoricalHMM
A_true = np.array([[0.7, 0.3], [0.4, 0.6]])
B_true = np.array([[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]])
gen = CategoricalHMM(n_components=2, random_state=0)
gen.startprob_ = np.array([0.6, 0.4]); gen.transmat_ = A_true; gen.emissionprob_ = B_true
X, _ = gen.sample(500, random_state=0)
fits = [CategoricalHMM(n_components=2, n_iter=500, tol=1e-6, random_state=s).fit(X) for s in range(5)]
print("9.7  true parameters score", round(float(gen.score(X)), 1))
for s in sorted(range(5), key=lambda s: -fits[s].score(X)):
    perm = np.argsort(fits[s].emissionprob_[:, 0])
    print("     seed", s, round(float(fits[s].score(X)), 1), " aligned emissions",
          np.round(fits[s].emissionprob_[perm], 3).tolist())


# %% 9.8: Example 9-16 with the filter's R raised from 9 to 90; the simulated sensor keeps std 3
def track(R_filter):
    rng = np.random.default_rng(0)
    F = np.array([[1, 1.0], [0, 1]]); H = np.array([[1.0, 0.0]])
    Q = np.eye(2) * 0.05; R = np.array([[R_filter]]); T = 60
    true_x = np.zeros((T, 2)); true_x[0] = [0.0, 1.0]
    for k in range(1, T):
        true_x[k] = F @ true_x[k - 1] + rng.multivariate_normal([0, 0], Q)
    meas = true_x[:, 0] + rng.normal(0, 3, T)
    x, P, est = np.array([0.0, 0.0]), np.eye(2) * 10.0, np.zeros(T)
    for k in range(T):
        x = F @ x; P = F @ P @ F.T + Q
        S = H @ P @ H.T + R; K = P @ H.T @ np.linalg.inv(S)
        x = x + (K @ (meas[k] - H @ x).reshape(1,)).ravel()
        P = (np.eye(2) - K @ H) @ P
        est[k] = x[0]
    return (round(float(K[0, 0]), 3), round(float(np.sqrt(np.mean((est - true_x[:, 0]) ** 2))), 3),
            round(float(np.sqrt(P[0, 0])), 3), round(float(np.sqrt(np.mean((meas - true_x[:, 0]) ** 2))), 3))


for r_f in (9.0, 90.0):
    print("9.8  filter R", r_f, " final gain, filtered RMSE, final sd, raw RMSE:", track(r_f))

# %% 9.9: the Cox model on fin alone, and covariate balance between the randomized groups
from lifelines import CoxPHFitter
from lifelines.datasets import load_rossi
rossi = load_rossi()
alone = CoxPHFitter().fit(rossi[["week", "arrest", "fin"]], duration_col="week", event_col="arrest")
full = CoxPHFitter().fit(rossi, duration_col="week", event_col="arrest")
print("9.9  fin alone: HR", round(float(alone.summary.loc["fin", "exp(coef)"]), 3), "p", round(float(alone.summary.loc["fin", "p"]), 3),
      "  full model: HR", round(float(full.summary.loc["fin", "exp(coef)"]), 3), "p", round(float(full.summary.loc["fin", "p"]), 3))
print("     mean prior convictions by fin:", rossi.groupby("fin")["prio"].mean().round(2).tolist(),
      " mean age by fin:", rossi.groupby("fin")["age"].mean().round(1).tolist())

# %% 9.10: the 9.10.4 simulations in their book order, then the bandwidth comparison
rng = np.random.default_rng(0)
n = 5000
U = rng.normal(0, 1, n); Z = rng.normal(0, 1, n)                      # instrument block
T = 0.5 * Z + U + rng.normal(0, 1, n); Y = 2.0 * T + 3.0 * U + rng.normal(0, 1, n)
U2 = rng.normal(0, 1, n); T2 = U2 + rng.normal(0, 1, n)                # front-door block
M = 1.0 * T2 + rng.normal(0, 1, n); Y2 = 1.5 * M + 2.0 * U2 + rng.normal(0, 1, n)
ng = 2000                                                              # difference-in-differences block
pre_c = 5 + rng.normal(0, 1, ng); post_c = 5 + 1.0 + rng.normal(0, 1, ng)
pre_t = 10 + rng.normal(0, 1, ng); post_t = 10 + 1.0 + 2.0 + rng.normal(0, 1, ng)
r = rng.uniform(-1, 1, n)                                              # regression discontinuity simulation
Trd = (r >= 0).astype(float)
Yrd = 3 + 1.5 * r + 2.0 * Trd + rng.normal(0, 0.5, n)
for h in (0.2, 0.5, 1.0):
    lf = np.polyfit(r[(r < 0) & (r > -h)], Yrd[(r < 0) & (r > -h)], 1)
    rf = np.polyfit(r[(r >= 0) & (r < h)], Yrd[(r >= 0) & (r < h)], 1)
    means = Yrd[(r >= 0) & (r < h)].mean() - Yrd[(r < 0) & (r > -h)].mean()
    print("9.10 h", h, " local lines:", round(float(np.polyval(rf, 0) - np.polyval(lf, 0)), 3),
          " plain means:", round(float(means), 3))

# %% 9.11: split conformal calibrated on 50 random calibration points, five draws
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
Xh, yh = fetch_california_housing(return_X_y=True)
X_tr, X_tmp, y_tr, y_tmp = train_test_split(Xh, yh, test_size=0.5, random_state=0)
X_cal, X_te, y_cal, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=0)
gb = GradientBoostingRegressor(random_state=0).fit(X_tr, y_tr)
cal_scores = np.abs(y_cal - gb.predict(X_cal)); pred_te = gb.predict(X_te)
k50 = math.ceil(51 * 0.9)
covs = []
for seed in range(5):
    sc = cal_scores[np.random.default_rng(seed).choice(len(cal_scores), 50, replace=False)]
    q = np.quantile(sc, k50 / 50)
    covs.append(round(float(np.mean(np.abs(y_te - pred_te) <= q)), 3))
print("9.11 k for n = 50:", k50, " coverages:", covs, " mean:", round(float(np.mean(covs)), 3))

# %% 9.12: the two tweaks of the 9.12.2 Experiment, each over five seeds (the model of Example 9-28)
import logging
import pandas as pd, pymc as pm
logging.getLogger("pymc").setLevel(logging.ERROR)     # the counts are printed below, so the sampler log is not needed
radon = pd.read_csv(pm.get_data("radon.csv"))
radon["county"] = radon["county"].str.strip()
keep = ["AITKIN", "CLAY", "DOUGLAS", "KOOCHICHING", "LAC QUI PARLE", "MURRAY", "RAMSEY", "STEARNS"]
sub = radon[radon["county"].isin(keep)]
cc, names = pd.factorize(sub["county"], sort=True)
y = sub["log_radon"].values; J = len(names)


def radon_fit(centered, target_accept, seed):            # the partial-pooling model of Example 9-27, in either form
    with pm.Model():
        mu_a = pm.Normal("mu_a", 1, 2); tau = pm.HalfNormal("tau", 2)
        if centered:
            a = pm.Normal("a", mu_a, tau, shape=J)       # centered: each a_j drawn around mu_a with spread tau
        else:
            a = pm.Deterministic("a", mu_a + pm.Normal("z", 0, 1, shape=J) * tau)   # noncentered
        sig = pm.HalfNormal("sigma", 2)
        pm.Normal("y", mu=a[cc], sigma=sig, observed=y)
        return pm.sample(draws=1000, tune=1000, chains=4, cores=1, random_seed=seed,
                         target_accept=target_accept, progressbar=False,
                         compute_convergence_checks=False)   # silences the end-of-fit warning; the count is read off below


def divergence_row(label, fits):                         # the columns of Example 9-28, one row per tweak
    div = [int(f.sample_stats["diverging"].values.sum()) for f in fits]
    flag = np.concatenate([f.sample_stats["diverging"].values.ravel() for f in fits])
    tau = np.concatenate([f.posterior["tau"].values.ravel() for f in fits])
    step = np.mean([float(f.sample_stats["step_size"].mean()) for f in fits])
    leap = np.mean([float(f.sample_stats["n_steps"].mean()) for f in fits])
    print("9.12", label, " divergences by seed:", div, " total:", sum(div),
          " step size", round(float(step), 2), " leapfrog steps per draw", round(float(leap), 1),
          " median tau", round(float(np.median(tau)), 3),
          " median tau of divergent draws", round(float(np.median(tau[flag])), 3) if flag.any() else "none")
    return sum(div), float(np.median(tau)), tau[flag]


for centered in (True, False):                                         # tweak 1: target_accept 0.99 on the full data
    divergence_row(("centered" if centered else "noncentered") + " target_accept 0.99",
                   [radon_fit(centered, 0.99, seed) for seed in range(5)])
small = ~sub["county"].isin(["RAMSEY", "STEARNS"]).values              # tweak 2: drop the two data-rich counties
y_s = y[small]; cc_s, names_s = pd.factorize(sub["county"][small], sort=True); J_s = len(names_s)
print("9.12 six small counties:", dict(zip(names_s, np.bincount(cc_s).tolist())), "houses", len(y_s))


def small_fit(centered, target_accept, seed):                          # radon_fit on the six-county subset
    with pm.Model():
        mu_a = pm.Normal("mu_a", 1, 2); tau = pm.HalfNormal("tau", 2)
        if centered:
            a = pm.Normal("a", mu_a, tau, shape=J_s)
        else:
            a = pm.Deterministic("a", mu_a + pm.Normal("z", 0, 1, shape=J_s) * tau)
        sig = pm.HalfNormal("sigma", 2)
        pm.Normal("y", mu=a[cc_s], sigma=sig, observed=y_s)
        return pm.sample(draws=1000, tune=1000, chains=4, cores=1, random_seed=seed,
                         target_accept=target_accept, progressbar=False, compute_convergence_checks=False)


six = {}                                                               # form: total, median tau, divergent taus
for centered in (True, False):
    six[centered] = divergence_row(("centered" if centered else "noncentered") + " six small counties, target_accept 0.8",
                                   [small_fit(centered, 0.8, seed) for seed in range(5)])
print("9.12 six small counties: median tau of divergent draws, centered below the overall median",
      bool(np.median(six[True][2]) < six[True][1]), " noncentered above it", bool(np.median(six[False][2]) > six[False][1]))
