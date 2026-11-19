# %% [markdown]
# # Section 9.13 lab: Bayesian A/B testing and GP regression on Mauna Loa CO2
# Regenerated from Book.md (single source of truth); run top to bottom.

# %%
import os
os.environ["PYTENSOR_FLAGS"] = "cxx="   # a PyTensor flag; this lab imports no PyMC, so it acts on nothing

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
