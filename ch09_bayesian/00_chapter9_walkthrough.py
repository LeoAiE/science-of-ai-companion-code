"""Chapter 9: Probabilistic & Bayesian Machine Learning

Section-aligned walkthrough: every code block from this chapter of the book,
in book order, each in its own cell under its section heading so you can run
and check them one by one. Generated from Book.md; keep the book as the source.
"""

# %% [markdown]
# ## 9.1.1 Bayes' rule for beliefs

# %% [markdown]
# With a disease prevalence of 1 percent, a sensitivity of 99 percent, and a specificity of 99 percent, what is the chance you actually have the disease given a positive test?

# %%
import numpy as np
prev, sens, spec = 0.01, 0.99, 0.99                 # 1% have the disease; test is 99% accurate
p_pos = sens * prev + (1 - spec) * (1 - prev)        # total probability of a positive test
print(round(sens * prev / p_pos, 3))                 # 0.5     P(disease | positive), not 0.99!

# %% [markdown]
# ## 9.1.2 Estimating a coin's bias

# %% [markdown]
# Here it means the posterior is simply $\text{Beta}(2+8, 2+2) = \text{Beta}(10, 4)$, and from it we can read every summary we might want.

# %%
from scipy.stats import beta
a, b, heads, tails = 2, 2, 8, 2                       # prior Beta(2,2), then 8 heads, 2 tails
pa, pb = a + heads, b + tails                         # posterior = Beta(10, 4)
print(round(heads / (heads + tails), 3))             # 0.8     MLE (data only)
print(round(pa / (pa + pb), 3))                      # 0.714   posterior mean (pulled toward 0.5)
print(round((pa - 1) / (pa + pb - 2), 3))            # 0.75    MAP (posterior mode)
print(beta.ppf([0.025, 0.975], pa, pb).round(3))     # [0.462 0.909]   95% credible interval

# %% [markdown]
# Computing all three on the same Beta(10, 4) posterior shows them pulling apart.

# %%
from scipy.stats import beta
post = beta(10, 4)                                   # the Beta(10, 4) coin posterior of 9.1.2
print(round(post.mean(), 3))                         # 0.714  squared-error loss -> posterior mean
print(round(post.median(), 3))                       # 0.725  absolute-error loss -> posterior median
print(round((10 - 1) / (10 + 4 - 2), 3))             # 0.75   0-1 loss -> posterior mode (MAP)

# %% [markdown]
# ## 9.1.3 Priors regularize

# %% [markdown]
# Three priors against the same 8-of-10 data make this concrete.

# %%
for (pa, pb), label in [((1, 1), "uniform Beta(1,1)"), ((2, 2), "weak Beta(2,2)"), ((20, 20), "strong Beta(20,20)")]:
    print(label, round((pa + 8) / (pa + pb + 10), 3))   # 8 heads, 2 tails added to each prior
# uniform -> 0.75    weak -> 0.714    strong -> 0.56   (a strong prior resists the data)

# %% [markdown]
# ## 9.1.4 Updating as data arrives

# %% [markdown]
# Feeding the ten flips in one at a time shows the belief sharpening.

# %%
flips = [1, 1, 0, 1, 1, 1, 0, 1, 1, 1]               # 8 heads, 2 tails, seen one at a time
ca, cb = 2, 2
lo, hi = beta.ppf([0.025, 0.975], ca, cb)
print(0, round(ca / (ca + cb), 3), round(hi - lo, 3))  # prior Beta(2, 2), before any flip
for i, f in enumerate(flips, 1):
    ca += f; cb += 1 - f                             # conjugate update: just add to the counts
    if i in (1, 6, 7, 10):
        lo, hi = beta.ppf([0.025, 0.975], ca, cb)
        print(i, round(ca / (ca + cb), 3), round(hi - lo, 3))
# 0 0.5 0.811
# 1 0.6 0.738
# 6 0.7 0.525
# 7 0.636 0.531
# 10 0.714 0.447

# %% [markdown]
# ## 9.2.1 L2 regularization is a Gaussian prior

# %% [markdown]
# Computing both ways confirms they are the same, apart from rounding error.

# %%
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
X, y = load_diabetes(return_X_y=True)                # 442 patients, 10 real health features
X = StandardScaler().fit_transform(X)                # put every feature on the same scale
y = y - y.mean()                                     # center the target so no intercept is needed
sigma2, tau2 = 4.0, 1.0                              # noise variance, prior variance
A = X.T @ X / sigma2 + np.eye(X.shape[1]) / tau2     # posterior precision
mu_map = np.linalg.solve(A, X.T @ y / sigma2)        # MAP estimate = posterior mean
ridge = Ridge(alpha=sigma2 / tau2, fit_intercept=False).fit(X, y)   # alpha = sigma^2/tau^2 = 4.0
print("MAP equals ridge within 1e-10:", bool(np.max(np.abs(mu_map - ridge.coef_)) < 1e-10))  # True
print(ridge.coef_.round(3))
# [ -0.351 -11.175  24.77   15.259 -19.009   7.874  -3.352   6.29   28.537
#    3.371]

# %% [markdown]
# ## 9.2.2 Predictions with error bars

# %% [markdown]
# At $x_* = 10$, far beyond the data, it is $4 + 10^2 \times 0.2222 = 26.222$, a standard deviation of 5.121 against 2.055, because the uncertain slope is multiplied by a large input.

# %%
import numpy as np
x = np.array([1.0, 2.0, 3.0]); sigma2, tau2 = 4.0, 1.0      # one feature, three training inputs
Sigma_post = 1 / (np.sum(x ** 2) / sigma2 + 1 / tau2)       # posterior variance of the single weight
for x_star in (1.0, 10.0):                                   # near the data, then far beyond it
    print(x_star, round(Sigma_post, 4), round(sigma2 + x_star ** 2 * Sigma_post, 3))
# 1.0 0.2222 4.222
# 10.0 0.2222 26.222

# %% [markdown]
# Real data shows the same effect.

# %%
import numpy as np
import seaborn as sns
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import PolynomialFeatures
g = sns.load_dataset("geyser")                       # Old Faithful: 272 eruptions
dur, wait = g["duration"].to_numpy(), g["waiting"].to_numpy()
band = (dur >= 2.5) & (dur <= 4.2)                    # train on a central band of durations only
poly = PolynomialFeatures(4, include_bias=True)
br = BayesianRidge().fit(poly.fit_transform(dur[band].reshape(-1, 1)), wait[band])
grid = np.linspace(0.5, 7.5, 300)
mean, std = br.predict(poly.transform(grid.reshape(-1, 1)), return_std=True)
print(round(std[np.argmin(abs(grid - 3.5))], 2))     # 6.32    predictive std at the band center
print(round(std[np.argmin(abs(grid - 7.2))], 2))     # 148.84  far outside the data (23.6x larger)
print(round(1 / br.alpha_, 2), round(1 / br.lambda_, 2))   # 38.98 2.95   learned noise and weight variances

# %% [markdown]
# ## 9.3 Conjugate priors

# %% [markdown]
# With the prior $\text{Gamma}(2, 1)$ and 73 events in 20 periods, the posterior is $\text{Gamma}(2 + 73,\; 1 + 20) = \text{Gamma}(75, 21)$, with mean $75/21 = 3.571$.

# %%
import numpy as np
from scipy.stats import gamma
rng = np.random.default_rng(0)
counts = rng.poisson(3.5, size=20)                   # 20 periods, true rate 3.5
a0, b0 = 2.0, 1.0                                    # Gamma(2,1) prior
a_post, b_post = a0 + counts.sum(), b0 + len(counts) # posterior Gamma(75, 21), shape and rate
print(int(a_post), int(b_post), int(counts.sum()))   # 75 21 73
print(round(counts.mean(), 3), round(a_post / b_post, 3))   # 3.65 3.571   MLE, then posterior mean
print(gamma.ppf([0.025, 0.975], a_post, scale=1 / b_post).round(3))   # [2.809 4.424] credible interval

# %% [markdown]
# Estimating the face probabilities of a possibly loaded die, a Dirichlet prior plus the observed roll counts gives a Dirichlet posterior, and the posterior mean of each face is its smoothed frequency: with a flat prior of all ones and 49 sixes in 120 rolls, face six gets $(1 + 49)/(6 + 120) = 50/126 = 0.397$.

# %%
true_p = np.array([1, 1, 1, 1, 1, 3]) / 8.0          # a die loaded toward face 6
rolls = rng.choice(6, size=120, p=true_p)             # keep the individual rolls
counts6 = np.bincount(rolls, minlength=6)
post = np.ones(6) + counts6                           # flat Dirichlet prior + observed counts
print(counts6.tolist())                               # [14, 14, 13, 12, 18, 49]
print((post / post.sum()).round(3).tolist())          # [0.119, 0.119, 0.111, 0.103, 0.151, 0.397]
# P(face 6): posterior 0.397 vs true 0.375  (loaded die detected)

# %% [markdown]
# A short check on the same 120 rolls verifies it.

# %%
seq = np.ones(6)                                      # start over from the flat prior
for r in rolls:
    seq[r] += 1                                       # each roll's posterior is the next roll's prior
print(bool(np.all(seq == post)))                      # True   one-at-a-time equals one batch update

# %% [markdown]
# SciPy's `dirichlet.pdf` takes the point first and the parameters second.

# %%
from scipy.special import gamma as gamma_fn          # the Gamma function, not the Gamma distribution
from scipy.stats import dirichlet, beta
conc, p3 = np.array([2.0, 2.0, 2.0]), np.array([0.2, 0.3, 0.5])
by_hand = gamma_fn(conc.sum()) / np.prod(gamma_fn(conc)) * np.prod(p3 ** (conc - 1))
print(round(by_hand, 4), round(dirichlet.pdf(p3, conc), 4))        # 3.6 3.6   the formula, then SciPy
print(round(dirichlet.pdf(p3, np.ones(3)), 4))                      # 2.0   flat Dirichlet(1, 1, 1): Gamma(3) = 2 everywhere
print(round(dirichlet.pdf([0.7, 0.3], [10, 4]), 4), round(beta.pdf(0.7, 10, 4), 4))   # 3.1161 3.1161   K = 2 is the Beta of 9.1.2

# %% [markdown]
# Fitting it to the 50 setosa sepal-length measurements in the iris dataset makes this concrete.

# %% [markdown]
# Example 9-1. The Normal-Inverse-Gamma update on fifty setosa sepal lengths from the iris data, with the posterior mean, the variance estimate, and a Student-t predictive interval.

# %%
import numpy as np, seaborn as sns
from scipy.stats import t as student_t
x = sns.load_dataset("iris").query("species == 'setosa'")["sepal_length"].to_numpy()  # 50 real cm measurements
n, xbar = len(x), x.mean()
mu0, kappa0, a0, b0 = 5.0, 1.0, 2.0, 0.1             # Normal-Inverse-Gamma prior hyperparameters
kappa_n = kappa0 + n                                 # update all four with the data
mu_n = (kappa0 * mu0 + n * xbar) / kappa_n
a_n = a0 + n / 2
b_n = b0 + 0.5 * np.sum((x - xbar) ** 2) + 0.5 * kappa0 * n * (xbar - mu0) ** 2 / kappa_n
print(round(mu_n, 3), round(b_n / (a_n - 1), 4))    # 5.006  posterior mean of the variance 0.1209
scale = np.sqrt(b_n * (kappa_n + 1) / (a_n * kappa_n))
lo, hi = student_t.ppf([0.025, 0.975], 2 * a_n, loc=mu_n, scale=scale)
print(round(lo, 3), round(hi, 3))                   # 4.315 5.697  95% predictive interval (Student-t, 54 dof)

# %% [markdown]
# SciPy's `invgamma` takes the shape $\alpha$ and the scale $\beta$, and here the scale is $\beta$ itself, not its reciprocal.

# %%
from scipy.special import gamma as gamma_fn
from scipy.stats import invgamma
dens = b0 ** a0 / gamma_fn(a0) * 0.1 ** (-a0 - 1) * np.exp(-b0 / 0.1)   # the iris prior's density at sigma^2 = 0.1
print(round(dens, 4), round(invgamma.pdf(0.1, a0, scale=b0), 4))        # 3.6788 3.6788   the formula, then SciPy (scale = beta)
v = 1 / np.random.default_rng(0).gamma(a_n, 1 / b_n, size=200_000)      # reciprocals of Gamma(a_n, rate b_n) draws
print(round(b_n / (a_n - 1), 4), round(v.mean(), 4), round(invgamma.mean(a_n, scale=b_n), 4))   # 0.1209 0.1209 0.1209
print(np.sqrt(invgamma.ppf([0.025, 0.975], a_n, scale=b_n)).round(3))  # [0.287 0.42 ]   95% credible interval for sigma, in cm

# %% [markdown]
# ## 9.4.1 Markov chain Monte Carlo

# %% [markdown]
# Both validate against the analytic Beta(10, 4) mean of 0.714 in Example 9-2.

# %% [markdown]
# Example 9-2. Rejection sampling and self-normalized importance sampling on the coin posterior, both checked against the analytic mean.

# %%
import numpy as np
rng = np.random.default_rng(0)
def target(th):                                       # unnormalized Beta(10,4) posterior
    return th ** 9 * (1 - th) ** 3

# Rejection sampling: propose from Uniform(0,1) under an envelope c, keep draws below the curve
grid = np.linspace(1e-6, 1 - 1e-6, 100000)
c = target(grid).max() * 1.01                         # c bounds the unnormalized density
prop = rng.uniform(0, 1, 200000)
keep = rng.uniform(0, c, 200000) < target(prop)       # accept if under the density
print(round(keep.mean(), 3), round(prop[keep].mean(), 3))   # 0.295  accept rate; mean 0.714

# Self-normalized importance sampling: weight uniform draws by the target, constants cancel
x = rng.uniform(0, 1, 200000)
w = target(x)                                         # weight = target / proposal density (=1)
print(round(np.sum(w * x) / np.sum(w), 3))            # 0.714  weighted mean matches analytic
print(int(np.sum(w) ** 2 / np.sum(w ** 2)))          # 82264  Kish effective sample size of 200000

# %% [markdown]
# Example 9-3. A from-scratch random-walk Metropolis-Hastings sampler with burn-in, acceptance rate, effective sample size, and R-hat diagnostics.

# %%
import numpy as np
from scipy.stats import beta
rng = np.random.default_rng(0)
a, b, h, t = 2, 2, 8, 2                                # prior Beta(2,2), data 8 heads and 2 tails
def log_post(theta):                                  # ONLY the unnormalized log posterior: Beta(10,4)
    if theta <= 0 or theta >= 1:
        return -np.inf
    return (h + a - 1) * np.log(theta) + (t + b - 1) * np.log(1 - theta)

def run_chain(start, n=20000, step=0.3):
    theta = start; lp = log_post(theta); draws = np.empty(n); accepted = 0
    for i in range(n):
        prop = theta + rng.normal(0, step)            # symmetric random-walk proposal
        lp_prop = log_post(prop)
        if np.log(rng.uniform()) < lp_prop - lp:      # accept with prob min(1, exp(change in log posterior))
            theta, lp, accepted = prop, lp_prop, accepted + 1
        draws[i] = theta
    return draws, accepted / n

draws, rate = run_chain(0.5)
s = draws[2000:]                                      # discard 2000 samples of burn-in
print(round(rate, 2))                                 # 0.42   acceptance rate (about 0.2 to 0.5 is a practical band)
print(round(s.mean(), 3), round((a + h) / (a + h + b + t), 3))       # 0.713 vs analytic 0.714
print(np.quantile(s, [0.025, 0.975]).round(3))        # [0.459 0.908] vs analytic [0.462 0.909]

centered = s - s.mean()
ac1 = np.sum(centered[1:] * centered[:-1]) / np.sum(centered ** 2)   # lag-1 autocorrelation
def ess(x):                                           # effective sample size
    x = x - x.mean(); n = len(x); v = np.sum(x * x) / n
    acf = np.correlate(x, x, mode="full")[n - 1:] / (v * n)
    tau = 1.0
    for k in range(1, len(acf)):
        if acf[k] < 0:
            break
        tau += 2 * acf[k]
    return n / tau
print(round(ac1, 3), int(ess(s)))                     # 0.633  ESS 3905 of 18000 draws

chains = np.array([run_chain(st)[0][2000:] for st in (0.2, 0.4, 0.6, 0.8)])
m, n = chains.shape
B = n * chains.mean(1).var(ddof=1); W = chains.var(1, ddof=1).mean()
print(round(np.sqrt(((n - 1) / n * W + B / n) / W), 3))   # 1.0   R-hat near 1 across 4 chains

# %% [markdown]
# ## 9.4.2 Gibbs sampling

# %% [markdown]
# The sampler of Example 9-4 is nothing but these two draws, alternated.

# %% [markdown]
# Example 9-4. A Gibbs sampler for a bivariate Gaussian with correlation 0.8, drawing each coordinate from its full conditional.

# %%
import numpy as np
rng = np.random.default_rng(0)
rho = 0.8                                            # target: bivariate standard normal, correlation 0.8
sd = np.sqrt(1 - rho ** 2)                           # each conditional is N(rho * other, 1 - rho^2)
x1, x2 = 0.0, 0.0
draws = np.empty((20000, 2))
for i in range(20000):
    x1 = rng.normal(rho * x2, sd)                    # draw x1 from P(x1 | x2)
    x2 = rng.normal(rho * x1, sd)                    # draw x2 from P(x2 | x1), using the new x1
    draws[i] = x1, x2
s = draws[1000:]                                     # discard burn-in
print(s.mean(0).round(3).tolist(), round(float(np.corrcoef(s.T)[0, 1]), 3))
# [0.013, 0.01] 0.798   means near 0, sample correlation 0.798 vs the true 0.8

# %% [markdown]
# ## 9.4.3 Hamiltonian Monte Carlo and NUTS

# %% [markdown]
# Example 9-5. Hamiltonian Monte Carlo from scratch on the coin posterior, tuned and then mistuned.

# %%
import numpy as np

def U(th):                                            # potential energy: minus the log posterior
    if th <= 0 or th >= 1:
        return np.inf                                 # outside (0,1): an infinitely high wall
    return -(9 * np.log(th) + 3 * np.log(1 - th))     # same unnormalized Beta(10,4) as Example 9-3

def gradU(th):                                        # its slope: -(9/theta - 3/(1-theta))
    return -(9 / th - 3 / (1 - th))

def hmc(n=5000, eps=0.05, L=4, seed=0):
    rng = np.random.default_rng(seed)
    th = 0.5; draws = np.empty(n); accepted = 0
    for i in range(n):
        r = rng.normal()                              # fresh random momentum, r ~ N(0, 1)
        H0 = U(th) + 0.5 * r * r                      # energy at the start of the flight
        t, rr, ok = th, r - 0.5 * eps * gradU(th), True   # leapfrog: half momentum step
        for step in range(L):
            t = t + eps * rr                          # full position step
            if t <= 0 or t >= 1:
                ok = False; break                     # flew out of (0,1): certain rejection
            if step < L - 1:
                rr = rr - eps * gradU(t)              # full momentum step
        if ok:
            rr = rr - 0.5 * eps * gradU(t)            # closing half momentum step
            if np.log(rng.uniform()) < H0 - (U(t) + 0.5 * rr * rr):
                th = t; accepted += 1                 # accept unless the energy drifted up
        draws[i] = th
    return draws, accepted / n

draws, rate = hmc()                                   # tuned: eps = 0.05, L = 4
s = draws[500:]                                       # discard 500 burn-in draws
c = s - s.mean()
ac1 = np.sum(c[1:] * c[:-1]) / np.sum(c ** 2)
print(round(rate, 2), round(s.mean(), 3), np.quantile(s, [0.025, 0.975]).round(3))
print(round(ac1, 3), int(ess(s)))                     # reuse the ess helper from Example 9-3
for eps, L in [(0.01, 5), (0.03, 20), (0.2, 3)]:      # mistuned: too short, too long, too coarse
    d, r_ = hmc(eps=eps, L=L)
    print(eps, L, round(r_, 2), int(ess(d[500:])))
# 0.97 0.716 [0.455 0.907]
# -0.032 4500
# 0.01 5 1.0 227
# 0.03 20 0.98 537
# 0.2 3 0.35 114

# %% [markdown]
# ## 9.4.4 Variational inference

# %% [markdown]
# One practical guard: a Gaussian has tails outside $(0, 1)$ where the Beta log density is undefined, so the samples are clipped into the open interval, which leaves the log finite and steeply penalizes any mass that strays out of range. Example 9-6 runs the optimization.

# %% [markdown]
# Example 9-6. Variational inference from scratch: a Gaussian fitted to the coin posterior by maximizing the ELBO over reparameterized samples.

# %%
import numpy as np
from scipy.optimize import minimize
from scipy.stats import beta
rng = np.random.default_rng(0)
eps = rng.normal(size=20000)                         # fixed base noise: theta = mu + sd * eps
def neg_elbo(params):                                # ELBO = E_q[log p(D, theta)] + entropy of q
    mu, log_sd = params
    sd = np.exp(log_sd)
    th = np.clip(mu + sd * eps, 1e-6, 1 - 1e-6)      # q's samples, kept inside (0, 1)
    logp = 9 * np.log(th) + 3 * np.log(1 - th)       # unnormalized Beta(10, 4) log posterior
    entropy = 0.5 * np.log(2 * np.pi * np.e) + log_sd
    return -(logp.mean() + entropy)                  # minimize the negative ELBO
res = minimize(neg_elbo, x0=[0.5, np.log(0.2)], method="Nelder-Mead")
mu_q, sd_q = res.x[0], float(np.exp(res.x[1]))
print(round(mu_q, 3), round(sd_q, 3))                              # 0.705 0.093   the fitted q
print(round(beta(10, 4).mean(), 3), round(beta(10, 4).std(), 3))   # 0.714 0.117   the exact posterior

# %% [markdown]
# ## 9.4.5 The Laplace approximation

# %% [markdown]
# The flat prior leaves the posterior more skewed, and a skewed posterior is exactly the case that exposes what the Laplace approximation gets wrong, which is the point of Example 9-7.

# %% [markdown]
# Example 9-7. The Laplace approximation to the skewed Beta(9, 3) posterior, a Gaussian at the mode with the curvature there as its precision.

# %%
import numpy as np
from scipy import optimize, stats
# posterior from 8 heads, 2 tails with a flat prior is exactly Beta(9, 3)
def neg_log_post(th):
    return -(8 * np.log(th) + 2 * np.log(1 - th))        # -log of the unnormalized posterior
mode = optimize.minimize_scalar(neg_log_post, bounds=(1e-6, 1 - 1e-6), method="bounded").x
# curvature (second derivative of neg log posterior) at the mode -> precision
curv = (neg_log_post(mode + 1e-5) - 2 * neg_log_post(mode) + neg_log_post(mode - 1e-5)) / 1e-5**2
print(round(mode, 3), round(1 / curv ** 0.5, 3))     # 0.8 0.126   Laplace mode and std
true = stats.beta(9, 3)
print(round(true.mean(), 3), round(true.std(), 3))   # 0.75 0.12   true posterior mean and std

# %% [markdown]
# A short scan over left endpoints finds both on the same Beta(9, 3).

# %%
import numpy as np
from scipy.stats import beta
d = beta(9, 3)                                        # skewed posterior: 8 heads, 2 tails, flat prior
et = d.ppf([0.025, 0.975])                            # equal-tailed 95% interval
lo = np.linspace(d.ppf(1e-5), d.ppf(0.05), 20001)     # every left endpoint below the 5% quantile
hi = d.ppf(d.cdf(lo) + 0.95)                          # right endpoint that encloses 95%
i = np.argmin(hi - lo)                                # shortest such interval is the HPD
print(et.round(3), round(et[1] - et[0], 3))           # [0.482 0.94 ] 0.458  equal-tailed
print(np.array([lo[i], hi[i]]).round(3), round(hi[i] - lo[i], 3))  # [0.516 0.959] 0.443  HPD (shorter)

# %% [markdown]
# ## 9.4.6 Bayesian model selection: evidence, Bayes factors, and BIC

# %% [markdown]
# We compare a fair-coin model against a free-bias model on data of 18 heads in 20 flips.

# %%
from scipy.special import beta as Beta
ml_m0 = 0.5 ** 20                       # M0: fair coin, theta fixed at 0.5
ml_m1 = Beta(18 + 1, 2 + 1)             # M1: theta ~ Uniform, integrated out
print(f"{ml_m0:.2e}", f"{ml_m1:.2e}", round(ml_m1 / ml_m0, 1))   # 9.54e-07 2.51e-04 262.8

# %% [markdown]
# On this coin, $-2\ln(2.51\times 10^{-4}) = 16.58$ against a BIC of 16.0 for the free model, and $-2\ln(9.54\times 10^{-7}) = 27.73$, equal to the fair coin's BIC because that model has no free parameter.

# %%
import numpy as np
ll_m0 = 20 * np.log(0.5)                                 # fair coin, 0 free parameters
ll_m1 = 18 * np.log(0.9) + 2 * np.log(0.1)               # MLE theta=0.9, 1 free parameter
print(round(-2 * ll_m0, 2), round(-2 * ll_m1 + 1 * np.log(20), 2))   # 27.73 16.0

# %% [markdown]
# ## 9.5 Gaussian processes

# %% [markdown]
# The code performs the inversion and the two products.

# %%
import numpy as np
X = np.array([0.0, 1.0, 2.0]); y = np.array([1.0, 2.0, 0.0])
k = lambda a, b: np.exp(-(a - b) ** 2 / 2)           # RBF kernel, lengthscale 1
A = k(X[:, None], X[None, :]) + 0.1 * np.eye(3)      # K_XX plus noise on the diagonal
Ainv = np.linalg.inv(A)
alpha = Ainv @ y                                     # the weight vector [K_XX + s2 I]^{-1} y
ks = k(1.5, X)                                       # K_*X = [0.3247, 0.8825, 0.8825]
print(alpha.round(4))                                # [-0.5458  2.9917 -1.5824]
print(round(ks @ alpha, 4))                          # 1.0665   mu_* at x = 1.5
print(round(1.0 - ks @ Ainv @ ks, 4))                # 0.0824   sigma_*^2  (sd 0.287)
far = k(9.0, X)                                      # x = 9: kernel to all data is ~0
print(round(far @ alpha, 4), round(1.0 - far @ Ainv @ far, 4))   # -0.0 1.0

# %% [markdown]
# The canonical demonstration, from Rasmussen and Williams (2006, Section 5.4.3), is the record of atmospheric carbon dioxide measured at the Mauna Loa Observatory in Hawaii, which statsmodels ships and loads offline.

# %% [markdown]
# Example 9-8. A composite-kernel Gaussian process fit to the Mauna Loa CO2 record with a five-year forecast.

# %%
import numpy as np
from statsmodels.datasets import co2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, WhiteKernel, ConstantKernel as C
d = co2.load_pandas().data["co2"].resample("MS").mean().dropna()   # monthly Mauna Loa CO2 (ppm), 1958-2001
t0 = d.index[0]
X = ((d.index - t0).days / 365.25).to_numpy().reshape(-1, 1)       # years since March 1958
y = d.to_numpy()
kernel = (C(50.0) * RBF(length_scale=40.0)                        # slow rising trend
          + C(2.0) * ExpSineSquared(length_scale=1.0, periodicity=1.0,
                                    periodicity_bounds="fixed")     # locked 1-year seasonal cycle
          + WhiteKernel(noise_level=0.5))                         # measurement noise
gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True).fit(X, y)   # optimizer ON: learns hyperparameters
print(gp.kernel_)                                                  # the learned kernel
print("log-marginal-likelihood", round(gp.log_marginal_likelihood(gp.kernel_.theta), 1))
ystd = float(gp._y_train_std)                                     # normalize_y puts kernel units on this scale
noise = float(gp.kernel_.k2.noise_level)                          # WhiteKernel, in units of ystd
print("noise sd ppm", round(ystd * noise ** 0.5, 3))
sig = gp.kernel_.k1                                               # trend plus seasonal, without the white noise
K = sig(X) + noise * np.eye(len(y))
L = np.linalg.cholesky(K)
xs20 = np.array([[20.0]])                                         # year 1978, about 20 years after March 1958
ks = sig(xs20, X)
resid = sig(xs20) - ks @ np.linalg.solve(L.T, np.linalg.solve(L, ks.T))
print("latent sd ppm at 1978", round(float(np.sqrt(resid).ravel()[0]) * ystd, 3))
future = float(X.max()) + 5                                       # forecast 5 years past the data
grid = np.linspace(0, future, 700).reshape(-1, 1)
mean, std = gp.predict(grid, return_std=True)                     # predicted CO2 AND its uncertainty
for label, yr in [("inside data 1978", 20.0), ("forecast +5yr 2006", future)]:
    i = np.argmin(abs(grid.ravel() - yr))
    print(label, round(mean[i], 1), round(std[i], 3))
for dy in (0, 3, 5):
    m, s = gp.predict(np.array([[float(X.max()) + dy]]), return_std=True)
    print("years past end", dy, round(float(m[0]), 1), round(float(s[0]), 3))
# 0.899**2 * RBF(length_scale=5.54) + 0.251**2 * ExpSineSquared(length_scale=2.03, periodicity=1) + WhiteKernel(noise_level=0.000583)
# log-marginal-likelihood 1123.9
# noise sd ppm 0.412
# latent sd ppm at 1978 0.078
# inside data 1978 336.5 0.419
# forecast +5yr 2006 371.0 5.177
# years past end 0 370.5 0.449
# years past end 3 372.4 2.211
# years past end 5 371.0 5.177

# %% [markdown]
# Swapping each kernel in as the trend term of the CO2 model, keeping the seasonal and noise terms, and forecasting five years past the data shows the difference.

# %%
from sklearn.gaussian_process.kernels import Matern, RationalQuadratic, DotProduct
seasonal = C(2.0) * ExpSineSquared(length_scale=1.0, periodicity=1.0, periodicity_bounds="fixed")
trends = {"Matern 3/2": C(50.0) * Matern(length_scale=40.0, nu=1.5),             # stationary, rougher
          "rational quadratic": C(50.0) * RationalQuadratic(length_scale=40.0),  # stationary, mixed scales
          "linear + RBF": C(1.0) * DotProduct(sigma_0=1.0) + C(50.0) * RBF(length_scale=40.0)}
x_end = np.array([[float(X.max())], [float(X.max()) + 5]])        # last month of data, then 5 years on
for name, trend in trends.items():
    g = GaussianProcessRegressor(kernel=trend + seasonal + WhiteKernel(noise_level=0.5),
                                 normalize_y=True).fit(X, y)
    m, s = g.predict(x_end, return_std=True)
    print(name, round(g.log_marginal_likelihood(g.kernel_.theta), 1), m.round(1), s.round(3))
# Matern 3/2 1266.7 [370.8 365.6] [ 0.283 10.963]
# rational quadratic 1228.0 [370.6 365.2] [0.348 6.45 ]
# linear + RBF 1237.8 [370.5 374.8] [0.343 1.833]

# %% [markdown]
# Three draws from the prior, before any data is seen, and three from the posterior, after conditioning on the CO2 record, show the collapse that learning produces, reusing the composite-kernel `gp` fitted to the CO2 record.

# %%
Xs = np.linspace(0, 6, 200).reshape(-1, 1)                        # a 6-year window
prior = GaussianProcessRegressor(kernel=gp.kernel_, optimizer=None)  # the fitted kernel, before conditioning
u = np.random.default_rng(0).standard_normal((len(Xs), 3))        # 3 columns of independent N(0, 1) numbers
def draw(model):                                                  # x = m + L u, where K = L L^T (Cholesky)
    m, K = model.predict(Xs, return_cov=True)
    return m[:, None] + np.linalg.cholesky(K) @ u
prior_draws = draw(prior) * y.std() + y.mean()                    # kernel units back to ppm
post_draws = draw(gp)                                             # posterior draws are already in ppm
_, prior_sd = prior.predict(Xs, return_std=True)
prior_sd = prior_sd * y.std()                                    # kernel units back to ppm
_, post_sd = gp.predict(Xs, return_std=True)
i = 100                                                          # Xs[100] is 3.015 years
print("prior draws", np.round(prior_draws[i], 1))
print("posterior draws", np.round(post_draws[i], 3))
print("predictive std", f"{float(prior_sd[i]):.2f}", f"{float(post_sd[i]):.3f}")
# prior draws [332.6 343.1 364.9]
# posterior draws [319.558 319.348 319.293]
# predictive std 15.93 0.420

# %% [markdown]
# The kernel's lengthscale controls how smooth the function is assumed to be, and it directly governs the wiggliness of the fit, which we can see by fixing the lengthscale of a plain RBF kernel by hand and measuring how rough the resulting fit of the CO2 curve becomes.

# %%
grid = np.linspace(0, float(X.max()), 600).reshape(-1, 1)         # dense grid across the data span
for ls in [0.3, 1.5, 8.0]:                                        # RBF lengthscale in years, held fixed
    g = GaussianProcessRegressor(kernel=C(1.0) * RBF(ls), alpha=0.5, normalize_y=True,
                                 optimizer=None).fit(X, y)
    print(ls, round(float(np.sum(np.abs(np.diff(g.predict(grid), 2)))), 3))   # total curvature of the fit
# 0.3 -> 164.31 (follows every seasonal wiggle)   1.5 -> 2.403   8.0 -> 0.266 (smooth trend only)

# %% [markdown]
# Running this loop on a black-box function, in Example 9-9, finds the best of 400 grid points in twelve evaluations.

# %% [markdown]
# Example 9-9. Bayesian optimization of a black-box function with a Gaussian process and an upper-confidence-bound acquisition function.

# %%
def black_box(x):                                    # the optimizer never sees this formula
    return np.sin(3 * x) - 0.3 * (x - 1) ** 2
dom = np.linspace(-2, 4, 400).reshape(-1, 1)
Xo = np.array([[-1.5], [3.5]]); yo = black_box(Xo).ravel()   # 2 initial probes
for _ in range(10):
    g = GaussianProcessRegressor(kernel=C(1.0) * RBF(1.0, length_scale_bounds=(0.1, 10.0)),
                                 alpha=1e-6, normalize_y=True, random_state=0).fit(Xo, yo)
    mu, sd = g.predict(dom, return_std=True)
    nxt = dom[np.argmax(mu + 2.0 * sd)]              # UCB acquisition: balance mean and uncertainty
    Xo = np.vstack([Xo, nxt]); yo = np.append(yo, black_box(nxt))
print(round(Xo[np.argmax(yo)][0], 3), round(dom[np.argmax(black_box(dom))][0], 3), int(np.argmax(yo)) + 1)
# 0.556 0.556 11   best x found, the true optimum on the grid, and the evaluation that found it

# %% [markdown]
# ## 9.6.1 The graph and the factorization

# %% [markdown]
# This section's example, the burglar alarm network that Russell and Norvig (2010, Section 14.1) credit to Judea Pearl, keeps three of their five binary variables: a burglary and an earthquake, independent causes, both of which can set off a house alarm.

# %%
from itertools import product
pB = {True: 0.01, False: 0.99}                       # P(Burglary)
pE = {True: 0.02, False: 0.98}                       # P(Earthquake)
pA = {(True, True): 0.95, (True, False): 0.94,       # P(Alarm=True | Burglary, Earthquake)
      (False, True): 0.40, (False, False): 0.01}
def joint(b, e, a):                                  # the network factorizes the full joint
    return pB[b] * pE[e] * (pA[(b, e)] if a else 1 - pA[(b, e)])
full = 2 ** 3 - 1                                    # free numbers in the full joint table
net = (len(pB) - 1) + (len(pE) - 1) + len(pA)        # P(B): 1, P(E): 1, P(A|B,E): 4
total = sum(joint(b, e, a) for b, e, a in product([True, False], repeat=3))
print(full, net, round(total, 12))                   # 7 6 1.0

# %% [markdown]
# ## 9.6.3 Inference and explaining away

# %% [markdown]
# The collider produces the phenomenon of explaining away, which Example 9-10 shows numerically on the alarm network.

# %% [markdown]
# Example 9-10. Inference by enumeration over the alarm network's joint distribution, showing explaining away.

# %%
def prob(query, evidence):                           # P(query | evidence) by enumerating the joint
    num = den = 0.0
    for b, e, a in product([True, False], repeat=3):
        assign = {"B": b, "E": e, "A": a}
        if all(assign[k] == v for k, v in evidence.items()):
            p = joint(b, e, a); den += p
            if all(assign[k] == v for k, v in query.items()): num += p
    return num / den
print(round(prob({"B": True}, {}), 3))                       # 0.01   prior
print(round(prob({"B": True}, {"E": True}), 3))              # 0.01   a quake alone says nothing about B
print(round(prob({"B": True}, {"A": True}), 3))              # 0.348  the alarm raises suspicion
print(round(prob({"B": True}, {"A": True, "E": True}), 3))   # 0.023  a quake "explains away" the alarm
print(round(prob({"B": True}, {"A": True, "E": False}), 3))  # 0.487  no quake -> burglary likely

# %% [markdown]
# Feeding it the same alarm network reproduces the by-hand posteriors exactly.

# %%
import warnings
warnings.filterwarnings("ignore", message="`pgmpy.estimators.StructureScore` is deprecated",
                        category=FutureWarning)               # raised by pgmpy.inference itself
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
net = DiscreteBayesianNetwork([("B", "A"), ("E", "A")])           # Burglary -> Alarm <- Earthquake
net.add_cpds(                                                     # state 0 = True, state 1 = False
    TabularCPD("B", 2, [[0.01], [0.99]]),
    TabularCPD("E", 2, [[0.02], [0.98]]),
    TabularCPD("A", 2, [[0.95, 0.94, 0.40, 0.01],                 # P(Alarm=True | B, E)
                        [0.05, 0.06, 0.60, 0.99]],
               evidence=["B", "E"], evidence_card=[2, 2]))
ve = VariableElimination(net)
query_B = lambda ev: round(float(ve.query(["B"], evidence=ev, show_progress=False).values[0]), 3)
print("P(B)        ", query_B({}))
print("P(B|A)      ", query_B({"A": 0}))
print("P(B|A,E)    ", query_B({"A": 0, "E": 0}))
print("P(B|A,notE) ", query_B({"A": 0, "E": 1}))
# P(B)         0.01
# P(B|A)       0.348
# P(B|A,E)     0.023
# P(B|A,notE)  0.487

# %% [markdown]
# Simulating fifty thousand nights from the true network and then relearning the alarm table recovers it from the counts alone.

# %%
import numpy as np, pandas as pd
from pgmpy.parameter_estimator import DiscreteMLE
rng = np.random.default_rng(0)
N = 50000
bt = rng.random(N) < 0.01; et = rng.random(N) < 0.02             # sample true Burglary / Earthquake
pa = np.select([bt & et, bt & ~et, ~bt & et], [0.95, 0.94, 0.40], default=0.01)
at = rng.random(N) < pa                                          # sample the Alarm from its true CPT
data = pd.DataFrame({"B": (~bt).astype(int), "E": (~et).astype(int), "A": (~at).astype(int)})
mle = DiscreteMLE().fit(DiscreteBayesianNetwork([("B", "A"), ("E", "A")]), data)
v = next(c for c in mle.parameters_ if c.variable == "A").values  # the learned P(Alarm | B, E)
print("learned P(A=True|no burglary,no quake):", round(float(v[0, 1, 1]), 3), "(true 0.01)")
print("learned P(A=True|no burglary,quake)   :", round(float(v[0, 1, 0]), 3), "(true 0.40)")
print("learned P(A=True|burglary,quake)      :", round(float(v[0, 0, 0]), 3), "(true 0.95)")
print("nights with a burglary and a quake    :", int(((data.B == 0) & (data.E == 0)).sum()))
# learned P(A=True|no burglary,no quake): 0.011 (true 0.01)
# learned P(A=True|no burglary,quake)   : 0.405 (true 0.40)
# learned P(A=True|burglary,quake)      : 1.0 (true 0.95)
# nights with a burglary and a quake    : 9

# %% [markdown]
# Evaluating it gives 25 DAGs on three nodes and $4{,}175{,}098{,}976{,}430{,}598{,}143$ on ten, about $4.175 \times 10^{18}$, so no method can enumerate them.

# %%
import math
def n_dags(n, memo={0: 1}):
    if n not in memo:                                # a(0) = 1, then the recurrence
        memo[n] = sum((-1)**(k + 1) * math.comb(n, k) * 2**(k * (n - k)) * n_dags(n - k)
                      for k in range(1, n + 1))
    return memo[n]
print(n_dags(3), n_dags(10))
# 25 4175098976430598143

# %% [markdown]
# Scoring three candidate graphs on the fifty thousand simulated nights shows the score doing its job, and then the two search algorithms recover the structure with no hints.

# %%
from pgmpy.structure_score import BIC
from pgmpy.causal_discovery import HillClimbSearch, PC
bic = BIC(data)
empty = DiscreteBayesianNetwork(); empty.add_nodes_from(["B", "E", "A"])
for label, g in [("collider B->A<-E", DiscreteBayesianNetwork([("B", "A"), ("E", "A")])),
                 ("chain B->A->E   ", DiscreteBayesianNetwork([("B", "A"), ("A", "E")])),
                 ("no edges        ", empty)]:
    print(label, round(bic.score(g), 1))                          # higher BIC score is better
hc = HillClimbSearch(scoring_method="bic-d", show_progress=False).fit(data)   # greedy score search
print("hill-climb:", sorted(hc.causal_graph_.edges()))
pc = PC(show_progress=False).fit(data)                            # independence-test search
print("PC        :", sorted(pc.causal_graph_.edges()))
# collider B->A<-E -11272.7
# chain B->A->E    -11428.4
# no edges         -13912.1
# hill-climb: [('B', 'A'), ('E', 'A')]
# PC        : [('B', 'A'), ('E', 'A')]

# %% [markdown]
# ## 9.7.2 The forward algorithm

# %% [markdown]
# Example 9-11. The forward algorithm on the weather HMM, checked against brute-force enumeration of all eight hidden paths.

# %%
import numpy as np
from itertools import product
states = ["Rainy", "Sunny"]
pi = np.array([0.6, 0.4])                            # initial state distribution
A = np.array([[0.7, 0.3], [0.4, 0.6]])               # transition P(next | current)
B = np.array([[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]])     # emission P(obs | state)
obs = [0, 1, 2]                                      # observed: walk, shop, clean
def forward(obs):                                    # P(observations), summed over every state path
    alpha = pi * B[:, obs[0]]
    for o in obs[1:]:
        alpha = (alpha @ A) * B[:, o]                # propagate through A, then emit
    return alpha.sum()
brute = sum(pi[p[0]] * B[p[0], obs[0]] * A[p[0], p[1]] * B[p[1], obs[1]] * A[p[1], p[2]] * B[p[2], obs[2]]
            for p in product(range(2), repeat=3))    # enumerate all 8 hidden paths directly
print(round(forward(obs), 6), round(float(brute), 6), bool(np.isclose(forward(obs), brute)))
# 0.033612 0.033612 True

# %% [markdown]
# ## 9.7.3 The Viterbi algorithm

# %% [markdown]
# Example 9-12 reuses the model parameters $\pi$, $A$, and $B$ that Example 9-11 defined in 9.7.2.

# %% [markdown]
# Example 9-12. The Viterbi algorithm decoding the weather HMM, checked against a brute-force search over all eight hidden paths.

# %%
def viterbi(obs):                                    # the single most likely path of hidden states
    delta = pi * B[:, obs[0]]; back = []
    for o in obs[1:]:
        sc = delta[:, None] * A; back.append(sc.argmax(0)); delta = sc.max(0) * B[:, o]
    path = [int(delta.argmax())]
    for b in reversed(back):
        path.append(int(b[path[-1]]))
    return path[::-1]
print([states[s] for s in viterbi(obs)])             # ['Sunny', 'Rainy', 'Rainy']
def path_prob(p):                                    # joint probability of one hidden path and obs
    return (pi[p[0]] * B[p[0], obs[0]] * A[p[0], p[1]] * B[p[1], obs[1]]
            * A[p[1], p[2]] * B[p[2], obs[2]])
best = max(product(range(2), repeat=3), key=path_prob)   # brute force over all 8 paths
print([states[s] for s in best], round(float(path_prob(best)), 5))   # ['Sunny', 'Rainy', 'Rainy'] 0.01344

# %% [markdown]
# ## 9.7.4 The backward pass and smoothing

# %% [markdown]
# Example 9-13. The forward-backward pass computing per-day smoothing posteriors, checked against brute-force enumeration.

# %%
import numpy as np
from itertools import product
pi = np.array([0.6, 0.4]); A = np.array([[0.7, 0.3], [0.4, 0.6]])
B = np.array([[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]]); obs = [0, 1, 2]     # walk, shop, clean
T = len(obs)
alpha = np.zeros((T, 2)); alpha[0] = pi * B[:, obs[0]]               # forward: P(x_1:t, z_t=s)
for t in range(1, T):
    alpha[t] = (alpha[t - 1] @ A) * B[:, obs[t]]
beta = np.zeros((T, 2)); beta[-1] = 1.0                              # backward: P(x_{t+1:T} | z_t=s)
for t in range(T - 2, -1, -1):
    beta[t] = A @ (B[:, obs[t + 1]] * beta[t + 1])
gamma = alpha * beta; gamma /= gamma.sum(axis=1, keepdims=True)      # smoothing: P(z_t=s | x_1:T)
for t in range(T):
    print("day", t + 1, "P(Rainy|all)=", round(float(gamma[t, 0]), 3),
          " P(Sunny|all)=", round(float(gamma[t, 1]), 3))
day1 = np.zeros(2)                                                   # brute force: all 8 hidden paths
for p in product(range(2), repeat=3):
    pr = (pi[p[0]] * B[p[0], obs[0]] * A[p[0], p[1]] * B[p[1], obs[1]]
          * A[p[1], p[2]] * B[p[2], obs[2]])
    day1[p[0]] += pr                                                 # bucket by the day-one state
print("brute-force day 1:", (day1 / day1.sum()).round(3))
# day 1 P(Rainy|all)= 0.232  P(Sunny|all)= 0.768
# day 2 P(Rainy|all)= 0.624  P(Sunny|all)= 0.376
# day 3 P(Rainy|all)= 0.864  P(Sunny|all)= 0.136
# brute-force day 1: [0.232 0.768]

# %% [markdown]
# The log-space forward pass computes the same total, only stably.

# %%
from scipy.special import logsumexp
la = np.log(pi) + np.log(B[:, obs[0]])                              # forward in log-space
for t in range(1, len(obs)):
    la = logsumexp(la[:, None] + np.log(A), axis=0) + np.log(B[:, obs[t]])
print("log P(obs) =", round(float(logsumexp(la)), 6), " P(obs) =", round(float(np.exp(logsumexp(la))), 6))
# log P(obs) = -3.392872  P(obs) = 0.033612

# %% [markdown]
# ## 9.7.5 Learning, and where HMMs live

# %% [markdown]
# Example 9-14 runs one iteration on the three-day weather sequence, reusing `alpha`, `beta`, and `gamma` from Example 9-13, to show the update at work and check it against a single step of hmmlearn started from the same parameters.

# %% [markdown]
# Example 9-14. One Baum-Welch iteration on the three-day weather sequence, checked against a single hmmlearn step from the same parameters.

# %%
P_obs = alpha[-1].sum()                                          # P(x_1:T), 0.033612 as in 9.7.2
xi = np.array([alpha[t][:, None] * A * (B[:, obs[t + 1]] * beta[t + 1])[None, :] / P_obs
               for t in range(T - 1)])                           # xi_t(i, j) for t = 1, ..., T - 1
print(np.allclose(xi.sum(axis=2), gamma[:-1]))                   # summing out j gives gamma_t back
pi_new = gamma[0]                                                # M-step: expected frequencies
A_new = xi.sum(axis=0) / gamma[:-1].sum(axis=0)[:, None]
B_new = np.stack([gamma[np.array(obs) == k].sum(axis=0) for k in range(3)], axis=1)
B_new /= gamma.sum(axis=0)[:, None]
print(pi_new.round(3), A_new.round(3).tolist(), B_new.round(3).tolist())
a_new = pi_new * B_new[:, obs[0]]                                # forward pass under the new parameters
for o in obs[1:]:
    a_new = (a_new @ A_new) * B_new[:, o]
print(round(float(P_obs), 6), "->", round(float(a_new.sum()), 6))
from hmmlearn.hmm import CategoricalHMM
hm = CategoricalHMM(n_components=2, n_iter=1, init_params="")   # start exactly at pi, A, B
hm.startprob_, hm.transmat_, hm.emissionprob_ = pi, A, B
hm.fit(np.array(obs).reshape(-1, 1))                             # one Baum-Welch iteration
print(np.allclose(hm.startprob_, pi_new), np.allclose(hm.transmat_, A_new), np.allclose(hm.emissionprob_, B_new))
# True
# [0.232 0.768] [[0.894, 0.106], [0.632, 0.368]] [[0.135, 0.363, 0.502], [0.6, 0.294, 0.106]]
# 0.033612 -> 0.071447
# True True True

# %% [markdown]
# And the state labels are arbitrary, since nothing in the data says which hidden state we chose to call Rainy, so the learned matrices come back in whichever order the initialization happened to fall into, and must be aligned by their emission signatures before comparing (here, Rainy is the state that rarely emits a walk). Example 9-15 runs the five fits and prints them ranked by likelihood.

# %% [markdown]
# Example 9-15. Baum-Welch recovery of the weather model from fifty thousand sampled activities, with five hmmlearn restarts aligned and ranked by likelihood.

# %%
import numpy as np
from hmmlearn.hmm import CategoricalHMM  # tested with hmmlearn 0.3.3
A_true = np.array([[0.7, 0.3], [0.4, 0.6]])            # the weather model of 9.7.1
B_true = np.array([[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]])
gen = CategoricalHMM(n_components=2, random_state=0)   # the true model, used only to sample
gen.startprob_ = np.array([0.6, 0.4]); gen.transmat_ = A_true; gen.emissionprob_ = B_true
X, _ = gen.sample(50000, random_state=0)               # 50000 activities; the states stay hidden
fits = [CategoricalHMM(n_components=2, n_iter=500, tol=1e-6, random_state=s).fit(X)
        for s in range(5)]                             # Baum-Welch (EM) from 5 random starts
def aligned(m):
    perm = np.argsort(m.emissionprob_[:, 0])            # Rainy = the state with low P(walk)
    return (np.round(m.transmat_[perm][:, perm], 3),
            np.round(m.emissionprob_[perm], 3))
print("true", round(float(gen.score(X)), 1))           # score of the parameters that generated X
for s in sorted(range(5), key=lambda s: -fits[s].score(X)):
    A_hat, B_hat = aligned(fits[s])
    print("seed", s, round(float(fits[s].score(X)), 1))
    print(A_hat)
    print(B_hat)
# true -54586.7
# seed 4 -54585.4
# [[0.721 0.279]
#  [0.404 0.596]]
# [[0.106 0.402 0.492]
#  [0.613 0.292 0.095]]
# seed 3 -54585.6
# [[0.581 0.419]
#  [0.29  0.71 ]]
# [[0.003 0.421 0.576]
#  [0.528 0.312 0.159]]
# seed 1 -54589.4
# [[0.734 0.266]
#  [0.512 0.488]]
# [[0.098 0.402 0.5  ]
#  [0.729 0.269 0.002]]
# seed 2 -54858.3
# [[0.116 0.884]
#  [0.673 0.327]]
# [[0.301 0.397 0.302]
#  [0.323 0.326 0.351]]
# seed 0 -54858.4
# [[0.488 0.512]
#  [0.73  0.27 ]]
# [[0.299 0.401 0.299]
#  [0.333 0.294 0.373]]

# %% [markdown]
# ## 9.8 Kalman filters

# %% [markdown]
# ## 9.8.1 The linear Kalman filter

# %% [markdown]
# Multiplying the two densities numerically on a fine grid confirms the algebra.

# %%
import numpy as np
x_hat, P, z, R = 10.0, 4.0, 13.0, 9.0                # prediction, its variance, reading, sensor variance
grid = np.linspace(-10.0, 30.0, 400001)              # fine grid over the state
post = np.exp(-(grid - x_hat) ** 2 / (2 * P) - (z - grid) ** 2 / (2 * R))   # prior times likelihood
post /= post.sum()                                   # normalize on the grid
m = np.sum(grid * post); v = np.sum((grid - m) ** 2 * post)
K = P / (P + R)                                      # the scalar Kalman gain
print(round(m, 3), round(x_hat + K * (z - x_hat), 3))   # 10.923 10.923  numeric mean vs the update
print(round(v, 3), round((1 - K) * P, 3))               # 2.769 2.769   numeric variance vs (1 - K) P

# %% [markdown]
# Example 9-16. A Kalman filter tracking a constant-velocity object from noisy position readings.

# %%
import numpy as np
rng = np.random.default_rng(0)
F = np.array([[1, 1.0], [0, 1]])        # state = [position, velocity]; position += velocity
H = np.array([[1.0, 0.0]])              # we measure position only
Q = np.eye(2) * 0.05                     # process noise (the model is not perfect)
R = np.array([[9.0]])                    # measurement noise variance (std = 3)
T = 60
true_x = np.zeros((T, 2)); true_x[0] = [0.0, 1.0]
for k in range(1, T):
    true_x[k] = F @ true_x[k - 1] + rng.multivariate_normal([0, 0], Q)
meas = true_x[:, 0] + rng.normal(0, 3, T)            # noisy position readings
x, P, est = np.array([0.0, 0.0]), np.eye(2) * 10.0, np.zeros(T)
for k in range(T):
    x = F @ x; P = F @ P @ F.T + Q                                       # predict
    S = H @ P @ H.T + R; K = P @ H.T @ np.linalg.inv(S)                  # Kalman gain
    x = x + (K @ (meas[k] - H @ x).reshape(1,)).ravel()                  # correct with measurement
    P = (np.eye(2) - K @ H) @ P
    est[k] = x[0]
print(round(np.sqrt(np.mean((meas - true_x[:, 0]) ** 2)), 3))    # 2.907  raw measurements
print(round(np.sqrt(np.mean((est - true_x[:, 0]) ** 2)), 3))     # 1.829  Kalman estimate (1.6x better)
print(round(float(np.sqrt(P[0, 0])), 3))                         # 1.713  final uncertainty (from 3.16)

# %% [markdown]
# ## 9.8.2 Smoothing with the Rauch-Tung-Striebel pass

# %% [markdown]
# Example 9-17. The Rauch-Tung-Striebel smoother added to the tracking filter, returning smoothed means and covariances.

# %%
import numpy as np
rng = np.random.default_rng(0)
F = np.array([[1, 1.0], [0, 1]]); H = np.array([[1.0, 0.0]])
Q = np.eye(2) * 0.05; R = np.array([[9.0]]); T = 60
true_x = np.zeros((T, 2)); true_x[0] = [0.0, 1.0]
for k in range(1, T):
    true_x[k] = F @ true_x[k - 1] + rng.multivariate_normal([0, 0], Q)
meas = true_x[:, 0] + rng.normal(0, 3, T)
x, P = np.array([0.0, 0.0]), np.eye(2) * 10.0
xf = np.zeros((T, 2)); Pf = np.zeros((T, 2, 2))          # filtered mean and covariance
xp = np.zeros((T, 2)); Pp = np.zeros((T, 2, 2))          # one-step predicted mean and covariance
for k in range(T):
    x = F @ x; P = F @ P @ F.T + Q                        # predict
    xp[k], Pp[k] = x, P
    S = H @ P @ H.T + R; K = P @ H.T @ np.linalg.inv(S)
    x = x + (K @ (meas[k] - H @ x).reshape(1,)).ravel()   # update with the measurement
    P = (np.eye(2) - K @ H) @ P
    xf[k], Pf[k] = x, P
xs, Ps = xf.copy(), Pf.copy()                             # RTS backward smoothing pass
for k in range(T - 2, -1, -1):
    Ck = Pf[k] @ F.T @ np.linalg.inv(Pp[k + 1])           # smoother gain C_k
    xs[k] = xf[k] + Ck @ (xs[k + 1] - xp[k + 1])          # smoothed mean
    Ps[k] = Pf[k] + Ck @ (Ps[k + 1] - Pp[k + 1]) @ Ck.T   # smoothed covariance
rmse = lambda e: round(float(np.sqrt(np.mean((e - true_x[:, 0]) ** 2))), 3)
print("raw measurements", rmse(meas))
print("filtered        ", rmse(xf[:, 0]))
print("RTS-smoothed    ", rmse(xs[:, 0]))
for k in (0, 29, T - 1):                                  # position sd: filtered, then smoothed
    print("step", k + 1, round(float(np.sqrt(Pf[k, 0, 0])), 3), round(float(np.sqrt(Ps[k, 0, 0])), 3))
# raw measurements 2.907
# filtered         1.829
# RTS-smoothed     1.201
# step 1 2.492 1.441
# step 30 1.713 0.961
# step 60 1.713 1.713

# %% [markdown]
# ## 9.8.4 Particle filters, and the three nonlinear filters compared

# %% [markdown]
# Example 9-18. A bootstrap particle filter tracking a nonlinear state through a squared, sign-blind sensor.

# %%
import numpy as np
from scipy import stats
rng = np.random.default_rng(0)
T = 50
x = np.zeros(T); y = np.zeros(T); x[0] = 0.1
for k in range(1, T):                                   # the hidden nonlinear state and its sensor
    x[k] = 0.5 * x[k - 1] + np.cos(0.4 * k) + rng.normal(0, 0.5)
    y[k] = 0.05 * x[k] ** 2 + rng.normal(0, 0.2)        # squared, sign-blind observation
particles = rng.normal(0, 1, 2000); est = np.zeros(T)
for k in range(1, T):
    particles = 0.5 * particles + np.cos(0.4 * k) + rng.normal(0, 0.5, 2000)   # propagate
    w = stats.norm.pdf(y[k], loc=0.05 * particles ** 2, scale=0.2)             # weight by likelihood
    w = w / w.sum()
    est[k] = np.sum(w * particles)                                            # posterior mean
    particles = particles[rng.choice(2000, 2000, p=w)]                        # resample
print(round(float(np.sqrt(np.mean((est[1:] - x[1:]) ** 2))), 3))   # 0.558  it tracks the state

# %% [markdown]
# Example 9-19. The extended, unscented, and particle filters compared on the univariate nonstationary growth model.

# %%
import numpy as np
from scipy import stats
rng = np.random.default_rng(0)
T = 100
def f(xv, k): return 0.5 * xv + 25 * xv / (1 + xv ** 2) + 8 * np.cos(1.2 * k)
x = np.zeros(T); y = np.zeros(T)
for k in range(1, T):                                   # simulate the benchmark
    x[k] = f(x[k - 1], k) + rng.normal(0, np.sqrt(10))
    y[k] = x[k] ** 2 / 20 + rng.normal(0, 1)
Q, R = 10.0, 1.0                                        # process and sensor noise variances

def df(xv): return 0.5 + 25 * (1 - xv ** 2) / (1 + xv ** 2) ** 2   # Jacobian of f
xe, Pe = 0.0, 5.0; ekf = np.zeros(T)                    # EKF: linearize f and h at the estimate
for k in range(1, T):
    F = df(xe); xp = f(xe, k); Pp = F * Pe * F + Q      # predict with the local slope
    H = xp / 10                                         # Jacobian of h(x) = x^2/20
    S = H * Pp * H + R; K = Pp * H / S                  # gain from the linearization
    xe = xp + K * (y[k] - xp ** 2 / 20); Pe = (1 - K * H) * Pp
    ekf[k] = xe

W = np.array([2/3, 1/6, 1/6])                           # UKF: n=1, kappa=2 sigma-point weights
xu, Pu = 0.0, 5.0; ukf = np.zeros(T)
for k in range(1, T):
    sig = np.array([xu, xu + np.sqrt(3 * Pu), xu - np.sqrt(3 * Pu)])   # x, x +/- sqrt(3P)
    fs = f(sig, k)                                      # through the real dynamics, no derivative
    xp = W @ fs; Pp = W @ (fs - xp) ** 2 + Q            # unscented mean and variance
    sig = np.array([xp, xp + np.sqrt(3 * Pp), xp - np.sqrt(3 * Pp)])   # fresh points to update
    hs = sig ** 2 / 20                                  # through the real sensor model
    yh = W @ hs
    S = W @ (hs - yh) ** 2 + R                          # innovation variance
    C = W @ ((sig - xp) * (hs - yh))                    # state-measurement covariance
    K = C / S
    xu = xp + K * (y[k] - yh); Pu = Pp - K * S * K
    ukf[k] = xu

particles = rng.normal(0, np.sqrt(5), 2000); pf = np.zeros(T)   # particle filter, as in Example 9-18
for k in range(1, T):
    particles = f(particles, k) + rng.normal(0, np.sqrt(10), 2000)
    w = stats.norm.pdf(y[k], loc=particles ** 2 / 20, scale=1.0)
    w = w / w.sum()
    pf[k] = np.sum(w * particles)
    particles = particles[rng.choice(2000, 2000, p=w)]

rmse = lambda e: round(float(np.sqrt(np.mean((e[1:] - x[1:]) ** 2))), 3)
print("EKF", rmse(ekf), " UKF", rmse(ukf), " PF", rmse(pf))
# EKF 21.409  UKF 13.888  PF 3.955

# %% [markdown]
# ## 9.9 Survival analysis

# %% [markdown]
# ## 9.9.1 The survival function and the Kaplan-Meier estimator

# %% [markdown]
# Example 9-20. The Kaplan-Meier product-limit estimator from scratch on the Rossi recidivism data.

# %%
import numpy as np
from lifelines.datasets import load_rossi  # tested with lifelines 0.30.3
rossi = load_rossi()                                 # 432 released prisoners, 9 columns
def kaplan_meier(dur, evt):                          # the product-limit estimator
    S, ts, Ss = 1.0, [0.0], [1.0]
    for t in np.unique(dur[evt == 1]):
        n_risk = np.sum(dur >= t); d = np.sum((dur == t) & (evt == 1))   # at risk, events at t
        S *= 1 - d / n_risk; ts.append(t); Ss.append(S)
    return np.array(ts), np.array(Ss)
for name, sub in [("no aid", rossi[rossi.fin == 0]), ("aid", rossi[rossi.fin == 1])]:
    dur, evt = sub["week"].values, sub["arrest"].values
    ts, Ss = kaplan_meier(dur, evt)                  # survival here means not yet rearrested
    S52 = float(Ss[-1]); S26 = float(Ss[ts <= 26][-1])   # survival past weeks 52 and 26
    print(name, "n", len(sub), "arrests", int(evt.sum()),
          "S(26)=", round(S26, 3), "S(52)=", round(S52, 3))
# no aid n 216 arrests 66 S(26)= 0.852 S(52)= 0.694
# aid n 216 arrests 48 S(26)= 0.898 S(52)= 0.778

# %% [markdown]
# The `kaplan_meier` function of Example 9-20 reproduces the five-subject example, and one more line shows a fact about Rossi that makes its curves checkable by hand.

# %%
toy_t = np.array([2, 3, 5, 7, 9]); toy_e = np.array([1, 0, 1, 1, 0])   # weeks; 0 marks a censored subject
ts, Ss = kaplan_meier(toy_t, toy_e)
print(ts.tolist(), Ss.round(4).tolist())             # steps only at the event weeks 2, 5, 7
print(rossi.loc[rossi.arrest == 0, "week"].unique().tolist(), int((rossi.arrest == 0).sum()))
# [0.0, 2.0, 5.0, 7.0] [1.0, 0.8, 0.5333, 0.2667]
# [52] 318

# %% [markdown]
# ## 9.9.2 The log-rank test

# %% [markdown]
# Whether the gap between the two curves is more than sampling noise is what the log-rank test decides. Example 9-21, a short loop over the Rossi event weeks, computes it from scratch.

# %% [markdown]
# Example 9-21. The log-rank test from scratch on the Rossi recidivism data, comparing the aid and no-aid groups.

# %%
from scipy import stats
dur, evt, grp = rossi["week"].values, rossi["arrest"].values, rossi["fin"].values
O1 = E1 = V = 0.0
for t in np.unique(dur[evt == 1]):                        # every week with at least one arrest
    at_risk = dur >= t
    n, n1 = at_risk.sum(), (at_risk & (grp == 0)).sum()   # at risk in all, and without aid
    d = ((dur == t) & (evt == 1)).sum()                   # arrests that week, both groups
    O1 += ((dur == t) & (evt == 1) & (grp == 0)).sum()    # observed arrests without aid
    E1 += d * n1 / n                                      # expected if the curves were equal
    V += d * (n1 / n) * (1 - n1 / n) * (n - d) / (n - 1)  # hypergeometric variance
chi2 = (O1 - E1) ** 2 / V
print(int(O1), round(E1, 3), round(V, 3), round(chi2, 3), round(float(stats.chi2.sf(chi2, 1)), 4))
# 66 55.574 28.323 3.838 0.0501

# %% [markdown]
# lifelines' `logrank_test` computes the same test.

# %%
from lifelines.statistics import logrank_test
a = rossi[rossi.fin == 0]; b = rossi[rossi.fin == 1]
res = logrank_test(a["week"], b["week"], a["arrest"], b["arrest"])
print(round(float(res.test_statistic), 3), round(float(res.p_value), 4))   # 3.838 0.0501

# %% [markdown]
# ## 9.9.3 Cox proportional hazards regression

# %% [markdown]
# The quantity that matters is $\exp(\beta_j)$, the hazard ratio: how much covariate $j$ multiplies the instantaneous risk.

# %%
from lifelines import CoxPHFitter
cph = CoxPHFitter().fit(rossi, duration_col="week", event_col="arrest")   # all 7 covariates at once
print(cph.summary[["coef", "exp(coef)", "p"]].round(3).to_string())
print("concordance:", round(float(cph.concordance_index_), 3))

# %% [markdown]
# ## 9.9.4 Accelerated failure time models and the proportional-hazards check

# %% [markdown]
# Raising $x_j$ by one adds $\beta_j$ to $\log T$, which multiplies every survival time, and so the expected time to event, by the time ratio $\exp(\beta_j)$; the fitted shape $\rho$ turns that time ratio into a hazard ratio, $\exp(\beta_j)^{-\rho}$.

# %%
from lifelines import WeibullAFTFitter
aft = WeibullAFTFitter().fit(rossi, duration_col="week", event_col="arrest")
tr = float(aft.summary.loc[("lambda_", "fin"), "exp(coef)"])         # time ratio for financial aid
rho = float(aft.summary.loc[("rho_", "Intercept"), "exp(coef)"])     # the fitted Weibull shape
print(round(tr, 3), round(rho, 3), round(tr ** -rho, 3))   # 1.313 1.404 0.682

# %% [markdown]
# The check lifelines runs, from Grambsch and Therneau (1994), regresses the scaled Schoenfeld residuals against the rank of the event times (the listing's `time_transform="rank"`), and a significant slope means the ratio drifts and the assumption is violated.

# %%
from lifelines.statistics import proportional_hazard_test
ph = proportional_hazard_test(cph, rossi, time_transform="rank")
print(ph.summary[["test_statistic", "p"]].round(4).to_string())

# %% [markdown]
# ## 9.10.1 Confounding and the backdoor criterion

# %% [markdown]
# Simulating data where the true causal effect of $T$ on $Y$ is exactly 2, with a confounder $X$ affecting both, shows the damage and the fix.

# %%
import numpy as np
from sklearn.linear_model import LinearRegression
rng = np.random.default_rng(0)
X = rng.normal(0, 1, 5000)                           # measured confounder X
T = X + rng.normal(0, 1, 5000)                       # X pushes the treatment T
Y = 2.0 * T + 3.0 * X + rng.normal(0, 1, 5000)       # the true effect of T on Y is 2.0
naive = LinearRegression().fit(T.reshape(-1, 1), Y).coef_[0]
adjusted = LinearRegression().fit(np.c_[T, X], Y).coef_[0]
print(round(naive, 3), round(adjusted, 3))           # 3.486 (biased: X leaks in)   2.013 (X blocked)

# %% [markdown]
# ## 9.10.2 Simpson's paradox

# %% [markdown]
# The adjustment formula of 9.10.1 turns the strata into a single adjusted comparison, provided stone size is the only confounder.

# %%
p_small, p_large = 357 / 700, 343 / 700                         # stone-size mix of all 700 patients
do_A = (81 / 87) * p_small + (192 / 263) * p_large              # backdoor adjustment, treatment A
do_B = (234 / 270) * p_small + (55 / 80) * p_large              # and treatment B
print(round(p_small, 2), round(p_large, 2), round(do_A, 3), round(do_B, 3))   # 0.51 0.49 0.833 0.779

# %% [markdown]
# ## 9.10.3 Randomization and uplift

# %% [markdown]
# Randomizing the treatment in the same simulation makes the naive estimate correct.

# %%
T = rng.integers(0, 2, 5000)                         # treatment assigned at RANDOM, independent of X
Y2 = 2.0 * T + 3.0 * X + rng.normal(0, 1, 5000)
print(round(float(Y2[T == 1].mean() - Y2[T == 0].mean()), 3))   # 1.983: a plain difference in means
# randomization breaks the T<-X link, so the naive estimate is unbiased: why we run experiments

# %% [markdown]
# Because the outcome is linear in $X$ within each arm, a linear regression per arm is the right outcome model.

# %%
rng = np.random.default_rng(0)
n = 4000
X = rng.uniform(-1, 1, n)                            # one covariate, uniform on [-1, 1]
T = rng.integers(0, 2, n)                            # randomized treatment
Y = 0.5 + X + (1 + 2 * X) * T + rng.normal(0, 1, n)  # true effect tau(x) = 1 + 2x
m1 = LinearRegression().fit(X[T == 1].reshape(-1, 1), Y[T == 1])   # outcome model, treated units
m0 = LinearRegression().fit(X[T == 0].reshape(-1, 1), Y[T == 0])   # outcome model, untreated units
x_new = np.array([[-0.5], [0.0], [0.5]])
tau_hat = m1.predict(x_new) - m0.predict(x_new)      # T-learner: difference of the two predictions
print(tau_hat.round(3).tolist(), (1 + 2 * x_new.ravel()).tolist())   # [0.086, 1.034, 1.981] [0.0, 1.0, 2.0]
print(round(float(Y[T == 1].mean() - Y[T == 0].mean()), 3))           # 0.985   one average for everyone

# %% [markdown]
# ## 9.10.4 The observational toolkit: instruments, front doors, differences, and discontinuities

# %% [markdown]
# In the simulation below $\mathrm{Cov}(Z, T) = 0.5$ and $\mathrm{Cov}(Z, Y) = 2 \times 0.5 = 1$ by construction, so the ratio is 2.

# %%
import numpy as np
from sklearn.linear_model import LinearRegression
rng = np.random.default_rng(0)
n = 5000
U = rng.normal(0, 1, n)                              # unmeasured confounder: in no dataset column
Z = rng.normal(0, 1, n)                              # instrument: random, pushes only the treatment
T = 0.5 * Z + U + rng.normal(0, 1, n)                # treatment, confounded by U
Y = 2.0 * T + 3.0 * U + rng.normal(0, 1, n)          # true causal effect of T on Y is 2.0
naive = LinearRegression().fit(T.reshape(-1, 1), Y).coef_[0]
t_hat = LinearRegression().fit(Z.reshape(-1, 1), T).predict(Z.reshape(-1, 1))   # stage 1
tsls = LinearRegression().fit(t_hat.reshape(-1, 1), Y).coef_[0]                 # stage 2
ratio = np.cov(Z, Y)[0, 1] / np.cov(Z, T)[0, 1]                                 # Cov(Z, Y) / Cov(Z, T)
print(round(naive, 3), round(tsls, 3), round(ratio, 3))   # 3.296 (confounded)   2.072 (2SLS)   2.072 (ratio)

# %% [markdown]
# In the linear case the two stages are regression slopes, and the front-door estimate is simply the product of the two coefficients.

# %%
U2 = rng.normal(0, 1, n)                             # unmeasured confounder again
T2 = U2 + rng.normal(0, 1, n)
M = 1.0 * T2 + rng.normal(0, 1, n)                   # mediator: caused by T2 alone, not by U2
Y2 = 1.5 * M + 2.0 * U2 + rng.normal(0, 1, n)        # true T2 effect = 1.0 * 1.5 = 1.5, all via M
naive2 = LinearRegression().fit(T2.reshape(-1, 1), Y2).coef_[0]
b1 = LinearRegression().fit(T2.reshape(-1, 1), M).coef_[0]      # T -> M leg
b2 = LinearRegression().fit(np.c_[M, T2], Y2).coef_[0]          # M -> Y leg, controlling T
print(round(naive2, 3), round(b1, 3), round(b2, 3), round(b1 * b2, 3))
# 2.453 (confounded)   0.978   1.536   1.502 (front-door recovers 1.5)

# %% [markdown]
# Difference-in-differences compares the before-to-after change in a treated group against the same change in an untreated group, so any fixed difference between the groups and any trend common to both cancel out.

# %%
ng = 2000                                            # a policy hits the treated group between periods
pre_c = 5 + rng.normal(0, 1, ng);  post_c = 5 + 1.0 + rng.normal(0, 1, ng)          # control: trend 1.0
pre_t = 10 + rng.normal(0, 1, ng); post_t = 10 + 1.0 + 2.0 + rng.normal(0, 1, ng)   # + effect 2.0
naive3 = post_t.mean() - post_c.mean()               # compares groups with a baseline gap of 5
did = (post_t.mean() - pre_t.mean()) - (post_c.mean() - pre_c.mean())
print(round(naive3, 3), round(did, 3))               # 7.017 (gap + effect)   2.07 (DiD recovers 2.0)

# %% [markdown]
# Regression discontinuity exploits a sharp eligibility threshold: units just above and just below the cutoff score are effectively randomized by their tiny distance from the line, so the causal effect is the jump in the outcome at the cutoff.

# %%
r = rng.uniform(-1, 1, n)                            # running variable, cutoff at 0
Trd = (r >= 0).astype(float)                         # sharp rule: treated iff r >= 0
Yrd = 3 + 1.5 * r + 2.0 * Trd + rng.normal(0, 0.5, n)   # outcome trends in r; true jump 2.0
naive4 = Yrd[Trd == 1].mean() - Yrd[Trd == 0].mean() # all treated vs all untreated
h = 0.2                                              # local-linear fit within a narrow bandwidth
lf = np.polyfit(r[(r < 0) & (r > -h)], Yrd[(r < 0) & (r > -h)], 1)
rf = np.polyfit(r[(r >= 0) & (r < h)], Yrd[(r >= 0) & (r < h)], 1)
print(round(naive4, 3), round(np.polyval(rf, 0) - np.polyval(lf, 0), 3))
# 3.521 (trend leaks in)   2.051 (the jump at the cutoff recovers 2.0)

# %% [markdown]
# ## 9.10.5 Propensity scores and inverse-probability weighting

# %% [markdown]
# Simulate an observational study where a confounder X raises both the chance of treatment and the outcome, with a true ATE of 2.0.

# %% [markdown]
# Example 9-22. Inverse-probability weighting with a logistic propensity model on a simulated observational study, in its self-normalized and Horvitz-Thompson forms.

# %%
import numpy as np
from sklearn.linear_model import LogisticRegression
rng = np.random.default_rng(0)
n = 5000
X = rng.normal(0, 1, n)                              # measured confounder
p = 1 / (1 + np.exp(-1.2 * X))                       # X pushes up the treatment probability
T = rng.binomial(1, p)                               # observational (nonrandom) assignment
Y = 2.0 * T + 3.0 * X + rng.normal(0, 1, n)          # the true effect of T on Y is 2.0
naive = Y[T == 1].mean() - Y[T == 0].mean()          # ignores the confounder
ps = LogisticRegression().fit(X.reshape(-1, 1), T).predict_proba(X.reshape(-1, 1))[:, 1]
w = np.where(T == 1, 1 / ps, 1 / (1 - ps))           # inverse-probability weights
ipw = (np.sum(w * T * Y) / np.sum(w * T)             # self-normalized form
       - np.sum(w * (1 - T) * Y) / np.sum(w * (1 - T)))
ht = np.mean(T * Y / ps) - np.mean((1 - T) * Y / (1 - ps))   # Horvitz-Thompson form
print(round(float(naive), 3), round(float(ipw), 3), round(float(ht), 3))   # 4.872 (biased)   2.083   2.096

# %% [markdown]
# ## 9.10.6 Colliders: controlling for the wrong variable

# %% [markdown]
# A simulation makes the spurious association appear on demand.

# %%
rng = np.random.default_rng(1)
A = rng.normal(0, 1, 5000)                           # two independent causes
B = rng.normal(0, 1, 5000)
C = A + B + rng.normal(0, 0.5, 5000)                 # collider: A -> C <- B
print(round(float(np.corrcoef(A, B)[0, 1]), 3))      # -0.034  independent, as built
from sklearn.linear_model import LinearRegression
coef_plain = LinearRegression().fit(B.reshape(-1, 1), A).coef_[0]
coef_ctrl = LinearRegression().fit(np.c_[B, C], A).coef_[0]   # controlling for the collider C
print(round(float(coef_plain), 3), round(float(coef_ctrl), 3))   # -0.034   -0.798

# %% [markdown]
# ## 9.11 Conformal prediction

# %% [markdown]
# The split conformal code of Examples 9-23 to 9-25 hands `np.quantile` the level $k/n$; with NumPy's default linear interpolation that returns the $k$-th smallest score or a value slightly above it, which can only add coverage.

# %%
import math
for n_cal in (9, 19, 8):                             # calibration sizes, alpha = 0.1
    k = math.ceil((n_cal + 1) * 0.9)                 # rank of the calibration score used as q
    print(n_cal, k, round(k / (n_cal + 1), 3) if k <= n_cal else "infinite interval")
# 9 9 0.9
# 19 18 0.9
# 8 9 infinite interval

# %% [markdown]
# One consequence of the guarantee is that coverage holds even for a badly wrong model, which Example 9-23 shows on real data.

# %% [markdown]
# Example 9-23. Split conformal regression on California housing, wrapping a gradient booster and a linear model with the same coverage guarantee.

# %%
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
X, y = fetch_california_housing(return_X_y=True)      # 20640 census block groups; median house value
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.5, random_state=0)
X_cal, X_te, y_cal, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=0)
def split_conformal(model, alpha=0.1):
    model.fit(X_tr, y_tr)
    scores = np.abs(y_cal - model.predict(X_cal))    # nonconformity = |residual| on a held-out calibration set
    q = np.quantile(scores, np.ceil((len(scores) + 1) * (1 - alpha)) / len(scores))   # finite-sample quantile
    pred = model.predict(X_te)
    cov = np.mean((y_te >= pred - q) & (y_te <= pred + q))
    return cov, 2 * q                                # interval is pred +/- q, so width = 2q
cov_gb, w_gb = split_conformal(GradientBoostingRegressor(random_state=0))
cov_lin, w_lin = split_conformal(LinearRegression())          # weaker model on this data
print(round(cov_gb, 3), round(w_gb, 2))              # 0.895 coverage, width 1.62  (tighter)
print(round(cov_lin, 3), round(w_lin, 2))            # 0.903 coverage, width 2.18  (wider, still valid)

# %% [markdown]
# And the promise holds at every confidence level you request.

# %%
for alpha in [0.2, 0.1, 0.05]:
    cov, _ = split_conformal(GradientBoostingRegressor(random_state=0), alpha=alpha)
    print(alpha, round(1 - alpha, 2), round(cov, 3))
# 0.2 0.8 0.801
# 0.1 0.9 0.895
# 0.05 0.95 0.95

# %% [markdown]
# The method extends naturally to classification, where the nonconformity score is one minus the predicted probability of the true class, and the output is a prediction set containing every class plausible enough to clear the calibrated threshold; Example 9-24 builds those sets for forest cover types.

# %% [markdown]
# Example 9-24. Conformal prediction sets from a random forest on the forest cover-type data, with their coverage and average set size.

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_covtype
X_all, y_all = fetch_covtype(return_X_y=True)         # 581,012 forest plots, 7 cover types
sub = np.random.default_rng(0).choice(len(X_all), 6000, replace=False)   # a 6000-row subset
Xc, yc = X_all[sub], y_all[sub]
Xc_tr, Xc_tmp, yc_tr, yc_tmp = train_test_split(Xc, yc, test_size=0.5, random_state=0)
Xc_cal, Xc_te, yc_cal, yc_te = train_test_split(Xc_tmp, yc_tmp, test_size=0.5, random_state=0)
clf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=2).fit(Xc_tr, yc_tr)
col = {c: i for i, c in enumerate(clf.classes_)}      # map labels 1..7 to probability columns
cal_i = np.array([col[v] for v in yc_cal])
cal = 1 - clf.predict_proba(Xc_cal)[np.arange(len(yc_cal)), cal_i]   # nonconformity = 1 - P(true class)
qhat = np.quantile(cal, np.ceil((len(cal) + 1) * 0.9) / len(cal))
in_set = (1 - clf.predict_proba(Xc_te)) <= qhat       # each prediction set keeps the confident-enough classes
te_i = np.array([col[v] for v in yc_te])
cov = float(in_set[np.arange(len(yc_te)), te_i].mean())   # coverage of the true label
size = float(in_set.sum(axis=1).mean())                   # average set size (of 7 classes)
se = np.sqrt(0.9 * 0.1 / len(yc_te))                      # binomial standard error of a 90% coverage on 1500 rows
print(abs(cov - 0.9) <= 2 * se, round(size, 1))           # True 1.4   within two standard errors of 0.9
votes = [t.predict_proba(X) for t in clf.estimators_ for X in (Xc_cal, Xc_te)]   # each tree on both sets
print(all(np.isin(v, [0.0, 1.0]).all() for v in votes))   # True   every tree votes 0 or 1 for each class

# %% [markdown]
# ## 9.11.1 Adaptive intervals with conformalized quantile regression

# %% [markdown]
# Example 9-25 reuses the California housing split of Example 9-23.

# %% [markdown]
# Example 9-25. Conformalized quantile regression on California housing, with intervals whose width follows the input and whose coverage is calibrated to 90 percent.

# %%
lo = GradientBoostingRegressor(loss="quantile", alpha=0.05, random_state=0).fit(X_tr, y_tr)
hi = GradientBoostingRegressor(loss="quantile", alpha=0.95, random_state=0).fit(X_tr, y_tr)
E = np.maximum(lo.predict(X_cal) - y_cal, y_cal - hi.predict(X_cal))   # distance outside the band
q = np.quantile(E, np.ceil((len(E) + 1) * 0.9) / len(E))
L = lo.predict(X_te) - q; U = hi.predict(X_te) + q    # conformalized 90% band
widths = U - L
print(round(float(np.mean((y_te >= L) & (y_te <= U))), 3))            # 0.899 coverage
print(round(float(widths.min()), 2), round(float(np.median(widths)), 2), round(float(widths.max()), 2))
# 0.35 1.64 4.67   narrow where the model is confident, wide where it is not

# %% [markdown]
# ## 9.12 Probabilistic programming with PyMC

# %% [markdown]
# To trust the tool, it helps to point it first at a problem whose answer we already know, the coin from 9.1, in Example 9-26.

# %% [markdown]
# Example 9-26. The coin posterior in PyMC, sampled with NUTS and checked against the analytic Beta(10, 4) mean.

# %%
import os
os.environ["PYTENSOR_FLAGS"] = "cxx=,numba__fastmath=False"  # numba backend, no C compiler; strict
                                                             # float math makes a fresh and a cached compile agree
import pymc as pm, arviz as az
with pm.Model():
    theta = pm.Beta("theta", alpha=2, beta=2)        # prior
    pm.Binomial("y", n=10, p=theta, observed=8)      # likelihood: 8 heads in 10 flips
    idata = pm.sample(1500, tune=1000, chains=4, cores=1, random_seed=0, progressbar=False)   # NUTS
s = az.summary(idata, var_names=["theta"], round_to="none")
theta_mean, theta_mcse = float(s["mean"].iloc[0]), float(s["mcse_mean"].iloc[0])   # the mean and its Monte Carlo error
print(f"{theta_mcse:.1g}", abs(theta_mean - 10 / 14) < 3 * theta_mcse, bool(s["r_hat"].iloc[0] < 1.01))
# 0.002 True True   within 3 standard errors of the analytic mean 10/14 = 0.714; R-hat under 1.01

# %% [markdown]
# The same handful of lines scales to models with no closed form, like a Bayesian linear regression.

# %%
import numpy as np
rng = np.random.default_rng(0)
x = rng.uniform(-3, 3, 120); y = 1.5 + 2.0 * x + rng.normal(0, 1.0, 120)   # true: intercept 1.5, slope 2.0
with pm.Model():
    a = pm.Normal("intercept", 0, 5)
    b = pm.Normal("slope", 0, 5)
    sigma = pm.HalfNormal("sigma", 2)
    pm.Normal("y", mu=a + b * x, sigma=sigma, observed=y)
    idata2 = pm.sample(1500, tune=1000, chains=4, cores=1, random_seed=0, progressbar=False)
reg = az.summary(idata2, var_names=["intercept", "slope", "sigma"], round_to="none")
print(reg["mean"].astype(float).round(1).tolist())
# [1.4, 2.0, 1.0]   posterior means to one decimal (each comes with a full credible interval)

# %% [markdown]
# The convergence check in PyMC 6.4.0, the version these listings ran on, says the same thing in its own words: with two or three chains it warns "We recommend running at least 4 chains for robust computation of convergence diagnostics," and with a single chain it says instead that some convergence checks cannot be run at all (PyMC developers, 2026e, `pymc/stats/convergence.py`).

# %%
print(bool(az.summary(idata2, round_to="none")["r_hat"].astype(float).max() < 1.01))
# True   every R-hat is below the 1.01 screen

# %% [markdown]
# ## 9.12.1 Hierarchical models and partial pooling

# %% [markdown]
# Example 9-27. Complete pooling, no pooling, and partial pooling fit to the Minnesota radon data.

# %%
import numpy as np, pandas as pd, pymc as pm, arviz as az
radon = pd.read_csv(pm.get_data("radon.csv"))          # Minnesota houses of the EPA radon survey (Gelman and Hill, 2006)
radon["county"] = radon["county"].str.strip()
keep = ["AITKIN", "CLAY", "DOUGLAS", "KOOCHICHING", "LAC QUI PARLE", "MURRAY", "RAMSEY", "STEARNS"]
sub = radon[radon["county"].isin(keep)]
cc, names = pd.factorize(sub["county"], sort=True)     # a county index 0..7 for each of the 94 houses
y = sub["log_radon"].values; J = len(names)
S = dict(draws=1000, tune=1000, chains=4, cores=1, random_seed=0, progressbar=False)
with pm.Model() as pooled_m:                           # complete pooling: one radon level for every county
    mu = pm.Normal("mu", 1, 2); sig = pm.HalfNormal("sigma", 2)
    pm.Normal("y", mu=mu, sigma=sig, observed=y)
    pooled = pm.sample(**S)
    pm.compute_log_likelihood(pooled)                  # per-house log likelihoods, for LOO in 9.12.2
with pm.Model() as nopool_m:                           # no pooling: each county fit on its own, wide prior
    a = pm.Normal("a", 1, 5, shape=J); sig = pm.HalfNormal("sigma", 2)
    pm.Normal("y", mu=a[cc], sigma=sig, observed=y)
    nopool = pm.sample(**S)
    pm.compute_log_likelihood(nopool)
with pm.Model() as hier:                               # partial pooling: counties share a common Normal
    mu_a = pm.Normal("mu_a", 1, 2); tau = pm.HalfNormal("tau", 2)
    a = pm.Deterministic("a", mu_a + pm.Normal("z", 0, 1, shape=J) * tau)   # noncentered form
    sig = pm.HalfNormal("sigma", 2)
    pm.Normal("y", mu=a[cc], sigma=sig, observed=y)
    partial = pm.sample(target_accept=0.95, **S)
    pm.compute_log_likelihood(partial)
print(round(float(y.mean()), 3), round(float(pooled.posterior["mu"].mean()), 3))  # 1.337 house mean, 1.336 pooled
mp = partial.posterior
print([round(float(mp[v].mean()), 3) for v in ["mu_a", "tau", "sigma"]])        # [1.439, 0.74, 0.684]

# %% [markdown]
# Placing each county's independent estimate next to its partially pooled one shows the shrinkage directly.

# %%
nop = nopool.posterior["a"].mean(("chain", "draw")).values
par = partial.posterior["a"].mean(("chain", "draw")).values
print(f"{'county':13} {'n':>2} {'no-pool':>8} {'partial':>8}")
for j, nm in enumerate(names):
    n = int((cc == j).sum())
    print(f"{nm:13} {n:2d} {nop[j]:8.3f} {par[j]:8.3f}")
print([nm for nm, d in zip(names, np.abs(par - nop)) if d > 0.1])   # the counties moved by more than 0.1

# %% [markdown]
# The code uses the unrounded means.

# %%
j = list(names).index("MURRAY"); n_j = int((cc == j).sum()); ybar = float(y[cc == j].mean())
s2, t2 = float(mp["sigma"].mean()) ** 2, float(mp["tau"].mean()) ** 2
mu_hat = float(mp["mu_a"].mean())
w = (n_j / s2) / (n_j / s2 + 1 / t2)                   # weight on the county's own mean
print(n_j, round(ybar, 3), round(w, 3), round(w * ybar + (1 - w) * mu_hat, 3), round(float(par[j]), 3))
# 1 2.501 0.539 2.012 1.975

# %% [markdown]
# ## 9.12.2 Criticizing and comparing models

# %% [markdown]
# PyMC produces the replicas with a single call.

# %%
with hier:                                             # reuse the fitted partial-pooling model
    rep = pm.sample_posterior_predictive(partial, random_seed=0, progressbar=False)
yrep = rep.posterior_predictive["y"].values.reshape(-1, len(y))   # 4000 replicated 94-house datasets
obs_sd = float(y.std()); rep_sd = yrep.std(axis=1)
print(f"{obs_sd:.3f}", f"{float(rep_sd.mean()):.2f}")             # 0.770 0.78  real vs simulated spread
print(f"{float(np.mean(rep_sd >= obs_sd)):.1f}")                  # 0.5  Bayesian p-value (0.5 is ideal)

# %% [markdown]
# The spread of the eight county means is such a quantity for complete pooling, which has no county effects at all.

# %%
with pooled_m:                                         # the same check on the complete-pooling fit
    rep_p = pm.sample_posterior_predictive(pooled, random_seed=0, progressbar=False)
yrep_p = rep_p.posterior_predictive["y"].values.reshape(-1, len(y))
def county_sd(v):                                      # spread of the eight county means
    return np.std([v[cc == k].mean() for k in range(J)])
obs_c = county_sd(y)
for label, reps in [("partial ", yrep), ("complete", yrep_p)]:
    sd_all = reps.std(axis=1); sd_c = np.array([county_sd(r) for r in reps])
    print(label, f"{float(np.mean(sd_all >= obs_sd)):.1f}", f"{obs_c:.3f}",
          f"{float(sd_c.mean()) / obs_c:.1f}", f"{float(np.mean(sd_c >= obs_c)):.1f}")
# partial  0.5 0.720 0.9 0.3
# complete 0.5 0.720 0.5 0.0

# %% [markdown]
# The remaining question is which model to prefer, and this subsection asks it out of sample: which model best predicts data it has not seen.

# %%
diag = az.summary(partial, round_to="none")
print(int(partial.sample_stats["diverging"].values.sum()),
      bool(diag["r_hat"].astype(float).max() < 1.01),
      bool(diag["ess_bulk"].astype(float).min() > 400))           # 0  True  True
cmp = az.compare({"complete": pooled, "no_pool": nopool, "partial": partial}, round_to="none")
show = cmp[["elpd", "elpd_diff", "dse", "p"]].round(2)
show["diag_diff"] = cmp["diag_diff"].astype(str)
show["diag_elpd"] = cmp["diag_elpd"].astype(str)
print(show.to_string())
gap = -cmp["elpd_diff"] / cmp["dse"]                              # how far below the best, in standard errors
print(list(cmp.index), round(float(gap["complete"])), bool(gap["partial"] < 1))

# %% [markdown]
# The next block runs az.loo on the no-pooling model alone and prints the county of the house with the largest Pareto-k, that value, and whether every other house stays below 0.7.

# %%
loo_np = az.loo(nopool)                                # LOO for the no-pooling model on its own
k = np.asarray(loo_np.pareto_k)                        # one Pareto-k per house
i = int(np.argmax(k))                                  # the house with the largest Pareto-k
print(names[cc[i]], round(float(k[i]), 2), bool(np.delete(k, i).max() < 0.7))
# MURRAY 0.84 True

# %% [markdown]
# Running the same check on the partial-pooling model shows shrinkage toward $\mu_a$ damping that instability.

# %%
loo_pp = az.loo(partial)                               # the same check on the partial-pooling model
kp = np.asarray(loo_pp.pareto_k)
murray_house = int(np.flatnonzero(cc == list(names).index("MURRAY"))[0])   # its single house
print("MURRAY", round(float(kp[murray_house]), 2), bool(kp[murray_house] < k[murray_house]),
      bool(np.delete(kp, murray_house).max() < 0.7))
# MURRAY 0.7 True True

# %% [markdown]
# The snippet integrates the unit Gaussian bowl $U(q) = q^2/2$ from $(q, r) = (1, 0)$ for twenty leapfrog steps at three step sizes and prints the energy at the end, which a perfect integrator would hold at 0.5.

# %%
def leapfrog_energy(eps, steps=20):                      # U = q^2/2 (a unit Gaussian), K = r^2/2, start at (q, r) = (1, 0)
    q, r = 1.0, 0.0
    for _ in range(steps):
        r -= 0.5 * eps * q; q += eps * r; r -= 0.5 * eps * q   # half kick, drift, half kick
    return 0.5 * q ** 2 + 0.5 * r ** 2                   # the energy, exactly 0.5 if the integrator were perfect
for eps in (0.5, 1.9, 2.1):
    print(eps, f"{leapfrog_energy(eps):.4g}")

# %% [markdown]
# Example 9-28. The centered and noncentered radon models fitted five times each at two target acceptance rates, with their divergence counts, adapted step sizes, leapfrog costs, and the $\tau$ values at which they diverge.

# %%
import time, logging
logging.getLogger("pymc").setLevel(logging.ERROR)        # the printed table reports what the sampler log would say

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
                         compute_convergence_checks=False)   # silences the end-of-fit warning; the count is read off the printed table

rows = {}                                                # (form, target): total, step, leapfrogs, median tau, divergent taus
t0 = time.time()
print("form          target  divergences by seed   total  step  leapfrog  tau all  tau div")
for centered in (True, False):
    for ta in (0.95, 0.8):
        fits = [radon_fit(centered, ta, seed) for seed in range(5)]
        div = [int(f.sample_stats["diverging"].values.sum()) for f in fits]
        flag = np.concatenate([f.sample_stats["diverging"].values.ravel() for f in fits])
        tau = np.concatenate([f.posterior["tau"].values.ravel() for f in fits])
        step = np.mean([float(f.sample_stats["step_size"].mean()) for f in fits])   # adapted step size
        leap = np.mean([float(f.sample_stats["n_steps"].mean()) for f in fits])     # leapfrog steps per draw
        tau_div = f"{np.median(tau[flag]):.3f}" if flag.any() else "none"
        print(f"{'centered' if centered else 'noncentered':13} {ta:5}  {str(div):21} {sum(div):4d}  {step:4.2f}  {leap:8.1f}  {np.median(tau):7.3f}  {tau_div:>7}")
        rows[centered, ta] = (sum(div), step, leap, np.median(tau), tau[flag])
neck = all(np.median(rows[True, ta][4]) < rows[True, ta][3] for ta in (0.95, 0.8))   # centered: in the neck
wide = bool(np.median(rows[False, 0.8][4]) > rows[False, 0.8][3])                  # noncentered at 0.8: large tau
print("median tau of divergent draws: centered below the overall median", neck, "| noncentered at 0.8 above it", wide)
fewer = all(rows[c, 0.95][0] < rows[c, 0.8][0] for c in (True, False))
costs = all(rows[c, 0.95][1] < rows[c, 0.8][1] and rows[c, 0.95][2] > rows[c, 0.8][2] for c in (True, False))
print("target_accept 0.95 against 0.8: fewer divergences", fewer, "| smaller steps, more leapfrog steps", costs)
print(round(time.time() - t0), "seconds for the twenty fits")

# %% [markdown]
# ## 9.13.1 Part A: Bayesian A/B testing

# %% [markdown]
# Example 9-29. Bayesian A/B testing against the frequentist p-value on the same conversion data.

# %%
import numpy as np
from scipy.stats import norm
rng = np.random.default_rng(0)
nA, cA, nB, cB = 1000, 100, 1000, 125               # A converted 100/1000; B converted 125/1000
postA = rng.beta(1 + cA, 1 + nA - cA, 200_000)      # Beta posterior per variant (flat prior)
postB = rng.beta(1 + cB, 1 + nB - cB, 200_000)
diff = postB - postA
print(round(float(np.mean(postB > postA)), 3))      # 0.961   P(B is better at all)
print(round(float(np.mean(diff >= 0.025)), 3))      # 0.497   P(lift reaches 2.5 points)
print(round(float(diff.mean() * 100), 2))           # 2.49    expected uplift (percentage points)
print((np.percentile(diff, [2.5, 97.5]) * 100).round(2).tolist())   # [-0.28, 5.28]  95% credible interval
se = np.sqrt((cA + cB) / (nA + nB) * (1 - (cA + cB) / (nA + nB)) * (1 / nA + 1 / nB))
z = (cB / nB - cA / nA) / se
print(round(float(2 * (1 - norm.cdf(abs(z)))), 3))  # 0.077  two-sided pooled z test
print(round(float(1 - norm.cdf(z)), 3))             # 0.038  one-sided p-value
print(round(float(norm.cdf(z)), 3))                 # 0.962  one minus that one-sided p-value
se_w = np.sqrt(cA / nA * (1 - cA / nA) / nA + cB / nB * (1 - cB / nB) / nB)
print(((cB / nB - cA / nA) + np.array([-1.96, 1.96]) * se_w).round(4).tolist())  # [-0.0027, 0.0527]

# %% [markdown]
# ## 9.13.2 Part B: Gaussian process regression with calibrated uncertainty

# %% [markdown]
# Example 9-30. A Gaussian process on Mauna Loa CO2 with its predictive band validated for coverage on held-out months.

# %%
import numpy as np, statsmodels.api as sm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, ExpSineSquared, WhiteKernel
rng = np.random.default_rng(0)
co2 = sm.datasets.co2.load_pandas().data.dropna()       # Mauna Loa weekly CO2, 1958-2001
m = co2["co2"].resample("MS").mean().dropna()           # monthly means: 521 points, 313-374 ppm
year = (m.index.year + (m.index.month - 1) / 12.0).values
X, y = year.reshape(-1, 1), m.values
idx = rng.permutation(len(y)); ntr = int(0.65 * len(y))
tr, te = np.sort(idx[:ntr]), np.sort(idx[ntr:])         # random 65/35 interpolation split
x0, ymean = X[tr].mean(), y[tr].mean()
Xtr, ytr, Xte, yte = X[tr] - x0, y[tr] - ymean, X[te] - x0, y[te]
k_signal = (C(50.0) * RBF(20.0)                         # the composite kernel of 9.5: trend
            + C(4.0) * ExpSineSquared(1.0, 1.0, periodicity_bounds="fixed"))   # plus 1-year season
g1 = GaussianProcessRegressor(k_signal + WhiteKernel(0.5),  # learn signal and noise together
                              normalize_y=False, random_state=0).fit(Xtr, ytr)
noise_var = g1.kernel_.k2.noise_level                   # WhiteKernel variance, in ppm squared
gp = GaussianProcessRegressor(g1.kernel_.k1, alpha=noise_var,  # reuse that signal kernel; do not refit it
                              optimizer=None, normalize_y=False, random_state=0).fit(Xtr, ytr)
mean_te, std_te = gp.predict(Xte, return_std=True)
pred = mean_te + ymean
total_std = np.sqrt(std_te ** 2 + noise_var)            # latent uncertainty plus observation noise
print(round(float(noise_var ** 0.5), 3))                                 # 0.398  learned noise std (ppm)
print(round(float(np.sqrt(np.mean((yte - pred) ** 2))), 3))              # 0.454  test RMSE (ppm)
print(round(float(np.mean(np.abs(yte - pred) <= 1.96 * std_te)), 3))     # 0.404  latent band only
print(round(float(np.mean(np.abs(yte - pred) <= 1.96 * total_std)), 3))  # 0.907  noise added: 166 of 183
print(round(float(g1.log_marginal_likelihood_value_), 3))   # -239.492  log marginal likelihood of the first fit
print(round(float(gp.log_marginal_likelihood_value_), 3))   # -239.492  the same optimum, reused by the second call
print(gp.kernel_)
# 15.8**2 * RBF(length_scale=5.75) + 5.15**2 * ExpSineSquared(length_scale=2.41, periodicity=1)

# %% [markdown]
# ## 9.14 Coding exercises

# %% [markdown]
# 1. Bayesian estimate of a proportion, from scratch.

# %%
from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(return_X_y=True)
def posterior_summary(y):
    ...
print(X.shape, int((y == 0).sum()), posterior_summary(y))

# %% [markdown]
# 2. L2 regularization is a Gaussian prior.

# %%
from sklearn.datasets import load_diabetes
X, y = load_diabetes(return_X_y=True)
def map_weights(X, y):
    ...
print(X.shape, map_weights(X, y))

# %% [markdown]
# 3. Conjugate priors beyond the coin.

# %%
import seaborn as sns
p = sns.load_dataset("penguins").dropna()
def nig_summary(bill):
    ...
print(p.shape, nig_summary(p.loc[p.species == "Adelie", "bill_length_mm"]))

# %% [markdown]
# 4. Approximate-inference toolkit on one posterior.

# %%
def four_ways():
    ...
print(four_ways())

# %% [markdown]
# 5. Gaussian process regression and Bayesian optimization on a new series.

# %%
import statsmodels.api as sm
sun = sm.datasets.sunspots.load_pandas().data
def gp_period(sun):
    ...
print(sun.shape, gp_period(sun))

# %% [markdown]
# 6. Explaining away in a network you build yourself.

# %%
def sprinkler():
    ...
print(16, sprinkler())

# %% [markdown]
# 7. The three HMM algorithms on a machine-health model.

# %%
def hmm_day2():
    ...
print(16, hmm_day2())

# %% [markdown]
# 8. Kalman filter, sensor quality, and the RTS smoother.

# %%
def kalman_three():
    ...
print([1, 9, 25], kalman_three())

# %% [markdown]
# 9. Survival analysis: a covariate that misbehaves.

# %%
from lifelines.datasets import load_rossi
r = load_rossi()
def wexp_report(r):
    ...
print(r.shape, wexp_report(r))

# %% [markdown]
# 10. Miniproject: conformal prediction on new datasets.

# %%
from sklearn.datasets import load_diabetes, load_digits
Xd, yd = load_diabetes(return_X_y=True)
digits = load_digits()
def conformal_report(X, y):
    ...
print(Xd.shape, digits.data.shape, conformal_report(Xd, yd))
