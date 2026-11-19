"""Chapter 5: Supervised Learning: Regression

Section-aligned walkthrough: every code block from this chapter of the book,
in book order, each in its own cell under its section heading so you can run
and check them one by one. Generated from Book.md; keep the book as the source.
"""

# %% [markdown]
# ## 5.1.2 Solving it two ways

# %% [markdown]
# The same formulas on a real sample, seaborn's penguins (Gorman, Williams, and Fraser, 2014), the 342 birds that record both flipper length and body mass, the same complete cases 3.4.2 used, recover a slope of 49.686 grams per millimeter of flipper and an intercept of $-5780.831$, matching `np.polyfit` and statsmodels, with $r = 0.871$ and $R^2 = r^2 = 0.759$.

# %%
import numpy as np
import seaborn as sns
import statsmodels.api as sm

x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])            # synthetic five-point check
y = np.array([2.0, 3.0, 5.0, 4.0, 6.0])
Sxx = np.sum((x - x.mean()) ** 2)
Sxy = np.sum((x - x.mean()) * (y - y.mean()))
b1, b0 = Sxy / Sxx, y.mean() - (Sxy / Sxx) * x.mean()
print(round(b0, 2), round(b1, 2), round(float(np.corrcoef(x, y)[0, 1] ** 2), 2))
# 1.3  0.9  0.81

peng = sns.load_dataset("penguins").dropna(subset=["flipper_length_mm", "body_mass_g"])
xf = peng["flipper_length_mm"].to_numpy(float)     # 342 Palmer penguins
yf = peng["body_mass_g"].to_numpy(float)
Sxxf = np.sum((xf - xf.mean()) ** 2)
Sxyf = np.sum((xf - xf.mean()) * (yf - yf.mean()))
b1f = Sxyf / Sxxf
b0f = yf.mean() - b1f * xf.mean()
print(round(b0f, 3), round(b1f, 3), round(float(np.corrcoef(xf, yf)[0, 1] ** 2), 3))
print(np.round(np.polyfit(xf, yf, 1)[::-1], 3).tolist())
print(np.round(sm.OLS(yf, sm.add_constant(xf)).fit().params, 3).tolist())
# -5780.831  49.686  0.759
# [-5780.831, 49.686]
# [-5780.831, 49.686]


# %% [markdown]
# Example 5-1. Linear regression three ways: the normal equation, gradient descent, and scikit-learn.

# %%
import numpy as np
rng = np.random.default_rng(0)
n = 200
X = rng.normal(0, 1, (n, 3))                       # 200 rows, 3 features
true_w, true_b = np.array([2.0, -1.5, 0.5]), 4.0   # a known truth, so we can verify the fit recovers it
y = X @ true_w + true_b + rng.normal(0, 0.5, n)    # target with a little Gaussian noise

# the normal equation: exact, closed form
X_aug = np.column_stack([np.ones(n), X])           # a column of 1s for the intercept
theta = np.linalg.inv(X_aug.T @ X_aug) @ X_aug.T @ y
print(theta.round(3))                              # [ 3.982  1.955 -1.548  0.527]

# gradient descent: iterative, scalable
w = np.zeros(4)
for _ in range(2000):
    w -= 0.1 * (2 / n) * X_aug.T @ (X_aug @ w - y)  # step downhill on the MSE
print(w.round(3))                                  # [ 3.982  1.955 -1.548  0.527]

from sklearn.linear_model import LinearRegression
lr = LinearRegression().fit(X, y)
print(np.round([lr.intercept_, *lr.coef_], 3))     # [ 3.982  1.955 -1.548  0.527]
print(np.linalg.lstsq(X_aug, y, rcond=None)[0].round(3))
# [ 3.982  1.955 -1.548  0.527]


# %% [markdown]
# ## 5.1.3 Assumptions and interpreting the coefficients

# %% [markdown]
# Seaborn's penguins (Gorman, Williams, and Fraser, 2014), the same 342 complete cases as 5.1.2, show the reading: body mass on flipper length plus species.

# %%
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

peng = sns.load_dataset("penguins").dropna(subset=["flipper_length_mm", "body_mass_g", "species"])
dummies = pd.get_dummies(peng["species"], drop_first=True).astype(float)
flip = peng["flipper_length_mm"].to_numpy(float)
y = peng["body_mass_g"].to_numpy(float)
X = sm.add_constant(np.column_stack([flip, dummies.to_numpy()]))
m = sm.OLS(y, X).fit()
print(list(dummies.columns), np.round(m.params, 3).tolist(), np.round(m.pvalues, 4).tolist())
d2 = pd.get_dummies(peng["species"], drop_first=False).astype(float).drop(columns=["Gentoo"])
m2 = sm.OLS(y, sm.add_constant(np.column_stack([flip, d2.to_numpy()]))).fit()
print(list(d2.columns), np.round(m2.params, 3).tolist())
print(float(np.max(np.abs(m.fittedvalues - m2.fittedvalues))))
Xint = sm.add_constant(np.column_stack([flip, dummies.to_numpy(), flip[:, None] * dummies.to_numpy()]))
mi = sm.OLS(y, Xint).fit()
m0 = sm.OLS(y, sm.add_constant(flip)).fit()
fstat, pval, df = mi.compare_f_test(m0)
print(np.round(mi.params, 3).tolist(), round(float(fstat), 2), float(pval), df)
# ['Chinstrap', 'Gentoo'] [-4031.477, 40.705, -206.51, 266.81] [0.0, 0.0, 0.0004, 0.0054]
# ['Adelie', 'Chinstrap'] [-3764.667, 40.705, -266.81, -473.32]
# 4.210960469208658e-10
# [-2535.837, 32.832, -501.359, -4251.444, 1.742, 21.791] 12.21 2.8007817339129075e-09 4.0


# %% [markdown]
# ## 5.1.4 Inference: which coefficients actually matter

# %% [markdown]
# We fit a model with two real features, a third irrelevant one, and an intercept, so the inference has a known right answer.

# %%
import numpy as np
import statsmodels.api as sm
rng = np.random.default_rng(0)
n = 200
x1, x2, x3 = rng.normal(0, 1, n), rng.normal(0, 1, n), rng.normal(0, 1, n)
# synthetic on purpose: x3's true coefficient is exactly 0, so "not significant" has a known right answer
y = 2.0 + 3.0 * x1 - 1.5 * x2 + 0.0 * x3 + rng.normal(0, 1.0, n)
X = np.column_stack([x1, x2, x3])

model = sm.OLS(y, sm.add_constant(X)).fit()
print(np.round(model.params, 3))      # [ 1.973  3.07  -1.386 -0.062]  (const, x1, x2, x3)
print(np.round(model.bse, 3))         # [0.072 0.075 0.07  0.072]   standard errors
print(np.round(model.pvalues, 4))     # [0.     0.     0.     0.3919] x3 is NOT significant
print(np.round(model.conf_int()[3], 3))   # [-0.203  0.08 ]  x3's 95% CI straddles zero
print(round(float(model.fvalue), 1), round(model.f_pvalue, 6),
      round(model.rsquared, 3), round(model.rsquared_adj, 3))   # 734.5 0.0 0.918 0.917
sigma2 = model.mse_resid                  # residual variance: SSR / (n - p)
Xa = sm.add_constant(X)
se_x1 = np.sqrt(sigma2 * np.linalg.inv(Xa.T @ Xa)[1, 1])   # rebuild x1's SE from the formula
print(round(float(sigma2), 3), round(float(se_x1), 3))     # 1.022 0.075
F = ((model.centered_tss - model.ssr) / 3) / (model.ssr / (n - 4))  # rebuild F from its formula
print(round(float(F), 1))                                  # 734.5
print(round(float(model.centered_tss), 1), round(float(model.ssr), 1))  # 2452.9 200.4
print(round(float(model.tvalues[3]), 3))                   # -0.858
m2 = sm.OLS(y, sm.add_constant(np.column_stack([x1, x2]))).fit()
print(round(m2.rsquared_adj, 5), round(model.rsquared_adj, 5))  # 0.91718 0.91706
print(round(m2.rsquared, 5), round(model.rsquared, 5))      # 0.91801 0.91831


# %% [markdown]
# Ask the fitted model to predict at a fresh input, here $x_1 = 1$, $x_2 = 0$, $x_3 = 0$, and read off both intervals from `get_prediction(...).summary_frame()`.

# %%
import numpy as np
import statsmodels.api as sm
rng = np.random.default_rng(0)
n = 200
x1, x2, x3 = rng.normal(0, 1, n), rng.normal(0, 1, n), rng.normal(0, 1, n)
y = 2.0 + 3.0 * x1 - 1.5 * x2 + 0.0 * x3 + rng.normal(0, 1.0, n)   # x3 truly zero, a known ground truth
model = sm.OLS(y, sm.add_constant(np.column_stack([x1, x2, x3]))).fit()

newx = sm.add_constant(np.array([[1.0, 0.0, 0.0]]), has_constant="add")
pred = model.get_prediction(newx).summary_frame(alpha=0.05)
print(pred[["mean", "mean_ci_lower", "mean_ci_upper",
            "obs_ci_lower", "obs_ci_upper"]].round(3).to_string(index=False))
#  mean  mean_ci_lower  mean_ci_upper  obs_ci_lower  obs_ci_upper
# 5.042          4.839          5.246         3.038         7.047

from scipy.stats import t as t_dist
Xa = sm.add_constant(np.column_stack([x1, x2, x3]))
x0 = newx[0]                                     # (1, 1, 0, 0): constant, x1, x2, x3
q = float(x0 @ np.linalg.inv(Xa.T @ Xa) @ x0)    # x0^T (X^T X)^{-1} x0
tstar = float(t_dist.ppf(0.975, n - 4))          # t quantile on n - p = 196 degrees of freedom
s2 = float(model.mse_resid)                      # residual variance, sigma hat squared
print(round(q, 4), round(tstar, 4), round(s2, 4))
print(round(tstar * np.sqrt(s2 * q), 3), round(tstar * np.sqrt(s2 * (1 + q)), 3))
# 0.0104 1.9721 1.0223
# 0.203 2.004


# %% [markdown]
# ## 5.1.5 Diagnostics: checking the assumptions before you trust the model

# %% [markdown]
# The residual plot itself takes a few lines of Matplotlib. The block draws it twice: for the synthetic fit of 5.1.4, built to obey every assumption, and for a real fit that breaks two of them, price on carat in seaborn's diamonds table of 3.4, 53,940 stones. To put numbers on what the eye sees, it also cuts the fitted values into thirds and prints the mean and the standard deviation of the residuals in each third: a correct model has means near zero in all three, and constant variance makes the three standard deviations similar.

# %%
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).parent if "__file__" in globals() else Path(".")  # save beside this script

rng = np.random.default_rng(0)                      # the synthetic fit of 5.1.4, built to obey the assumptions
n = 200
x1, x2, x3 = rng.normal(0, 1, n), rng.normal(0, 1, n), rng.normal(0, 1, n)
y = 2.0 + 3.0 * x1 - 1.5 * x2 + 0.0 * x3 + rng.normal(0, 1.0, n)
good = sm.OLS(y, sm.add_constant(np.column_stack([x1, x2, x3]))).fit()
dia = sns.load_dataset("diamonds")                  # real data: 53,940 stones, price on carat
bad = sm.OLS(dia["price"].to_numpy(float), sm.add_constant(dia["carat"].to_numpy(float))).fit()

fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
for ax, m, title in [(axes[0], good, "synthetic: a shapeless band"),
                     (axes[1], bad, "diamonds: a curve that fans out")]:
    ax.scatter(m.fittedvalues, m.resid, s=4, alpha=0.3)
    ax.axhline(0, color="black", lw=0.8)              # residuals should straddle this line evenly
    ax.set_xlabel("fitted value"); ax.set_ylabel("residual"); ax.set_title(title)
fig.tight_layout()
fig.savefig(HERE / "residual_vs_fitted.png", dpi=80)

for name, m in [("synthetic", good), ("diamonds", bad)]:
    f, e = np.asarray(m.fittedvalues), np.asarray(m.resid)
    lo, hi = np.quantile(f, [1 / 3, 2 / 3])           # cut the fitted values into thirds
    parts = [f <= lo, (f > lo) & (f <= hi), f > hi]
    print(name, [round(float(e[s].mean()), 2) for s in parts],
          [round(float(e[s].std()), 2) for s in parts])
# synthetic [0.1, -0.13, 0.03] [1.0, 1.09, 0.88]
# diamonds [343.9, -532.71, 160.83] [354.13, 794.9, 2486.97]


# %% [markdown]
# The quantile-quantile plot named in the second diagnostic is the construction of 2.4.1 applied to the residuals: sort them, pair them with the matching Gaussian quantiles, and look for a straight line, with ends that peel away from the line signaling the heavy tails or skew that make the p-values of 5.1.4 untrustworthy.

# %%
import numpy as np
import statsmodels.api as sm
rng = np.random.default_rng(0)
n = 200
x1, x2, x3 = rng.normal(0, 1, n), rng.normal(0, 1, n), rng.normal(0, 1, n)
y = 2.0 + 3.0 * x1 - 1.5 * x2 + 0.0 * x3 + rng.normal(0, 1.0, n)   # generated to obey the assumptions, so the checks should pass
X = np.column_stack([x1, x2, x3])
model = sm.OLS(y, sm.add_constant(X)).fit()

resid, fitted = model.resid, model.fittedvalues
print(round(float(np.corrcoef(fitted, resid**2)[0, 1]), 3))   # -0.013  homoscedastic (flat spread)
from scipy.stats import jarque_bera
print(round(float(jarque_bera(resid)[1]), 3))                  # 0.762   residuals look normal
from statsmodels.stats.stattools import durbin_watson
print(round(float(durbin_watson(resid)), 3))                   # 1.924   no autocorrelation
print(round(float(model.get_influence().cooks_distance[0].max()), 4))   # 0.0477  no dominant point
e = np.asarray(resid)
r_lag = float(np.corrcoef(e[1:], e[:-1])[0, 1])
print(round(r_lag, 3), round(2 * (1 - r_lag), 3))                      # 0.035  1.93
from scipy.stats import skew, kurtosis
S, K = float(skew(e, bias=True)), float(kurtosis(e, fisher=False, bias=True))
print(round(S, 4), round(K, 4), round(float(jarque_bera(e)[0]), 4))    # -0.0194  2.7475  0.5437
from statsmodels.stats.outliers_influence import variance_inflation_factor
Xc = sm.add_constant(X)
print([round(float(variance_inflation_factor(Xc, j)), 2) for j in range(1, Xc.shape[1])])  # [1.01, 1.01, 1.01]


# %% [markdown]
# The studentized residuals are $-0.397, -0.150, 1.405, -1.352, 0.397$, and Cook's distances are $0.118, 0.005, 0.247, 0.392, 0.118$, all far below 1, matching `get_influence()`.

# %%
import numpy as np
import statsmodels.api as sm

x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y = np.array([2.0, 3.0, 5.0, 4.0, 6.0])
X = sm.add_constant(x)
H = X @ np.linalg.inv(X.T @ X) @ X.T
h = np.diag(H)
e = y - (1.3 + 0.9 * x)
p, sigma = 2, np.sqrt(np.sum(e ** 2) / (5 - 2))
r_int = e / (sigma * np.sqrt(1 - h))
D = (r_int ** 2 / p) * (h / (1 - h))
print(np.round(h, 2).tolist(), round(float(h.sum()), 1))
print(np.round(r_int, 3).tolist())
print(np.round(D, 3).tolist())
inf = sm.OLS(y, X).fit().get_influence()
print(np.round(inf.hat_matrix_diag, 2).tolist(),
      np.round(inf.resid_studentized_internal, 3).tolist(),
      np.round(inf.cooks_distance[0], 3).tolist())
# [0.6, 0.3, 0.2, 0.3, 0.6]  2.0
# [-0.397, -0.15, 1.405, -1.352, 0.397]
# [0.118, 0.005, 0.247, 0.392, 0.118]
# same three lists from get_influence()


# %% [markdown]
# Dataset III instead shows a large residual at moderate leverage: the point at $x = 13$, $y = 12.74$ has residual $3.241$, studentized residual $3.000$, and Cook's distance $1.393$, the only value above 1 in that set.

# %%
import numpy as np
import seaborn as sns
import statsmodels.api as sm

ans = sns.load_dataset("anscombe")
for ds in ("III", "IV"):
    d = ans[ans.dataset == ds]
    xd, yd = d.x.to_numpy(float), d.y.to_numpy(float)
    inf = sm.OLS(yd, sm.add_constant(xd)).fit().get_influence()
    print(ds, np.round(inf.hat_matrix_diag, 3).tolist())
    print(ds, np.round(inf.resid_studentized_internal, 3).tolist())
    print(ds, np.round(inf.cooks_distance[0], 3).tolist())
# III [0.1, 0.1, 0.236, 0.091, 0.127, 0.318, 0.173, 0.318, 0.173, 0.127, 0.236]
# III [-0.46, -0.196, 3.0, -0.331, -0.597, -1.135, 0.07, 0.381, -0.755, -0.07, 0.212]
# III [0.012, 0.002, 1.393, 0.005, 0.026, 0.301, 0.001, 0.034, 0.06, 0.0, 0.007]
# IV [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1]
# IV [-0.359, -1.059, 0.605, 1.569, 1.253, 0.033, -1.494, nan, -1.229, 0.775, -0.095]
# IV [0.007, 0.062, 0.02, 0.137, 0.087, 0.0, 0.124, nan, 0.084, 0.033, 0.0]
# then statsmodels' RuntimeWarning: invalid value encountered in sqrt


# %% [markdown]
# A short synthetic example, noise standard deviation growing with $x$, shows the repair at work.

# %%
import numpy as np
import statsmodels.api as sm

rng = np.random.default_rng(1)                         # synthetic: noise std grows with x
xw = np.linspace(0.5, 5.0, 80)
sigma = 0.4 * xw                                      # known spread, used as the weights
yw = 1.5 + 2.0 * xw + rng.normal(0, 1.0, 80) * sigma
Xw = sm.add_constant(xw)
ols = sm.OLS(yw, Xw).fit()
wls = sm.WLS(yw, Xw, weights=1.0 / sigma ** 2).fit()
print(np.round(ols.params, 3).tolist(), np.round(ols.bse, 3).tolist())
# [1.579, 1.929]  [0.258, 0.085]
print(np.round(wls.params, 3).tolist(), np.round(wls.bse, 3).tolist())
# [1.603, 1.918]  [0.097, 0.063]
v = 1.0 / sigma[:3] ** 2
e = yw[:3] - (wls.params[0] + wls.params[1] * xw[:3])
print([round(float(a), 3) for a in v], [round(float(a), 3) for a in v * e ** 2])
# [25.0, 20.148, 16.583]  [0.001, 0.317, 0.013]


# %% [markdown]
# The data is the UCI Bike Sharing daily file of Chapter 17 (Fanaee-T and Gama, 2013; UCI Machine Learning Repository): 731 days in 2011 and 2012, daily totals of the hourly rental records, bundled with the companion code as `code/ch17_time_series/data/bike_sharing_daily.csv`, a path relative to the folder that holds the companion `code` directory; when the block runs from `code/ch05_regression` instead, as the companion notebook does, it finds the same file from the folder two levels up. Example 5-2 runs the Breusch-Pagan test and HC3 on the diamonds fit and Newey-West on the bike counts, and it rebuilds both sets of sandwich standard errors from their formulas to check statsmodels.
#
# Example 5-2. The Breusch-Pagan statistic and HC3 standard errors on the diamonds fit, then the Durbin-Watson statistic and Newey-West standard errors on the daily bike counts, with both sets of sandwich standard errors rebuilt from their formulas and checked against statsmodels.

# %%
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from pathlib import Path
from scipy.stats import chi2

dia = sns.load_dataset("diamonds")
y = dia["price"].to_numpy(float)
X = sm.add_constant(dia["carat"].to_numpy(float))
ols = sm.OLS(y, X).fit()
print(np.round(ols.params, 2).tolist(), np.round(ols.bse, 2).tolist())
print(round(float(np.corrcoef(ols.fittedvalues, ols.resid ** 2)[0, 1]), 3))
bp = sm.OLS(ols.resid ** 2, X).fit()
lm = len(y) * bp.rsquared
print(round(float(lm), 1), round(float(chi2.sf(lm, 1)), 6))
hc3 = sm.OLS(y, X).fit(cov_type="HC3")
print(np.round(hc3.bse, 2).tolist())
XtX_inv = np.linalg.inv(X.T @ X)
h = np.sum((X @ XtX_inv) * X, axis=1)          # hat diagonal only; n is 53,940
meat = X.T @ (((ols.resid ** 2) / (1 - h) ** 2)[:, None] * X)
V = XtX_inv @ meat @ XtX_inv
print(np.round(np.sqrt(np.diag(V)), 2).tolist())
# [-2256.36, 7756.43] [13.06, 14.07]
# 0.411
# 9131.2 0.0
# [16.13, 25.41]
# [16.13, 25.41]

DATA = Path("code/ch17_time_series/data")                # relative to the book folder
if not DATA.exists():                                    # run from code/ch05_regression instead
    DATA = Path.cwd().resolve().parents[1] / DATA
bike = pd.read_csv(DATA / "bike_sharing_daily.csv")
yb = bike["cnt"].to_numpy(float)
Xb = sm.add_constant(bike[["temp", "hum", "windspeed", "workingday"]].to_numpy(float))
bols = sm.OLS(yb, Xb).fit()
bhac = sm.OLS(yb, Xb).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
print(round(float(sm.stats.durbin_watson(bols.resid)), 3))
print(np.round(bols.bse, 3).tolist())
print(np.round(bhac.bse, 3).tolist())
u = Xb * bols.resid[:, None]                   # row t is e_t x_t
S = u.T @ u                                    # lag 0
for l in range(1, 8):                          # lags 1 to L = 7
    G = u[l:].T @ u[:-l]                       # sum over t of e_t e_(t-l) x_t^T x_(t-l)
    S += (1 - l / 8) * (G + G.T)               # weight w_l = 1 - l/(L + 1)
Xb_inv = np.linalg.inv(Xb.T @ Xb)
print(np.round(np.sqrt(np.diag(Xb_inv @ S @ Xb_inv)), 3).tolist())
# 0.404
# [344.524, 293.385, 383.979, 708.801, 113.55]
# [503.057, 696.609, 584.897, 755.622, 84.659]
# [503.057, 696.609, 584.897, 755.622, 84.659]


# %% [markdown]
# On the clipped California housing linear fit of 5.7, using the four constant caps of that lab, the quadratic extra sum of squares in the partial residual is 54.8 for median income and 268.6 for average occupancy: occupancy is the feature whose term is more clearly not a straight line.

# %%
import numpy as np
import statsmodels.api as sm
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

df = fetch_california_housing(as_frame=True).frame
for c, h in {"AveRooms": 12.0, "AveBedrms": 3.0, "Population": 6000.0, "AveOccup": 6.0}.items():
    df[c] = df[c].clip(upper=h)
feats = fetch_california_housing().feature_names
Xtr, _, ytr, _ = train_test_split(df[feats], df["MedHouseVal"], test_size=0.2, random_state=0)
X = sm.add_constant(Xtr.to_numpy(float))
m = sm.OLS(ytr.to_numpy(float), X).fit()
e = m.resid
for j, name in [(1, "MedInc"), (6, "AveOccup")]:
    pr = e + m.params[j] * X[:, j]
    z = sm.add_constant(X[:, j])
    lin = sm.OLS(pr, z).fit().ssr
    quad = sm.OLS(pr, np.column_stack([z, X[:, j] ** 2])).fit().ssr
    print(name, round(lin, 1), round(quad, 1), round(lin - quad, 1))
# MedInc 7468.2 7413.4 54.8
# AveOccup 7468.2 7199.6 268.6


# %% [markdown]
# ## 5.2.1 Curves from a linear model

# %% [markdown]
# Scikit-learn's `PolynomialFeatures` builds the transformed features for you.

# %%
from sklearn.preprocessing import PolynomialFeatures

print(PolynomialFeatures(3).fit_transform([[2.0]])[0])   # [1. 2. 4. 8.]


# %% [markdown]
# ## 5.2.2 Choosing the degree, and overfitting

# %% [markdown]
# On data generated from a true quadratic, this recovers the truth.

# %%
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# Synthetic on purpose: we build the data from a known quadratic so we can
# check that cross-validation recovers the true degree (2) on its own.
rng = np.random.default_rng(0)
true_f = lambda x: 0.5 * x**2 - x + 2
X = rng.uniform(-3, 3, 80).reshape(-1, 1)
y = true_f(X).ravel() + rng.normal(0, 1.0, 80)

scores = {}
for d in range(1, 11):
    model = make_pipeline(PolynomialFeatures(d), StandardScaler(), LinearRegression())
    scores[d] = -cross_val_score(model, X, y, cv=5,
                                 scoring="neg_mean_squared_error").mean()
    print(d, round(scores[d], 3))
print("best degree:", min(scores, key=scores.get))
# one degree per line: 1 3.484, 2 0.937, 3 0.953, 4 1.039, 5 1.108,
# 6 1.135, 7 1.013, 8 1.064, 9 1.312, 10 2.099, then best degree: 2


# %% [markdown]
# ## 5.3.1 Choosing the penalty strength, and Ridge's closed form

# %% [markdown]
# Small $d_j$, the directions $X$ barely sees, shrink most.

# %%
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

X = StandardScaler().fit_transform(load_diabetes().data)
y = load_diabetes().target - load_diabetes().target.mean()
lam = 10.0
w = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)
print(np.round(w, 3).tolist())
print(np.round(Ridge(alpha=lam, fit_intercept=False).fit(X, y).coef_, 3).tolist())
U, s, Vt = np.linalg.svd(X, full_matrices=False)
print(np.round(s ** 2 / (s ** 2 + lam), 3).tolist())
print(round(float(np.linalg.cond(X.T @ X)), 1),
      round(float(np.linalg.cond(X.T @ X + lam * np.eye(X.shape[1]))), 1))
print(round(float(np.corrcoef(X[:, 4], X[:, 5])[0, 1]), 3))
# [-0.258, -10.936, 24.6, 15.094, -11.296, 1.809, -6.562, 5.6, 25.332, 3.523]
# [-0.258, -10.936, 24.6, 15.094, -11.296, 1.809, -6.562, 5.6, 25.332, 3.523]
# [0.994, 0.985, 0.982, 0.977, 0.967, 0.964, 0.96, 0.95, 0.776, 0.275]
# 470.1 129.8
# 0.897


# %% [markdown]
# ## 5.3.2 Standardize before you penalize

# %% [markdown]
# Fitting the same Lasso at the same penalty on the raw features and on standardized copies shows the penalty selecting different features purely because of their units.

# %%
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LassoCV

data = load_diabetes(scaled=False)               # real data in raw units: years, mm Hg, serum levels
X, y = data.data, data.target
print(np.round(X.std(axis=0), 2).tolist())                 # [13.09, 0.5, 4.41, 13.82, 34.57, 30.38, 12.92, 1.29, 0.52, 11.48]
names = np.array(data.feature_names)
Xs = StandardScaler().fit_transform(X)           # the standardized copy of the same features

raw = Lasso(alpha=5.0, max_iter=100_000).fit(X, y)    # one penalty strength, two scalings
std = Lasso(alpha=5.0, max_iter=100_000).fit(Xs, y)
print("raw kept:", names[np.abs(raw.coef_) > 1e-6].tolist())
print("std kept:", names[np.abs(std.coef_) > 1e-6].tolist())
# raw kept: ['age', 'bmi', 'bp', 's1', 's2', 's3', 's6']
# std kept: ['sex', 'bmi', 'bp', 's3', 's5']

cv = LassoCV(cv=5, max_iter=100_000).fit(Xs, y)  # CV picks the strength on the standardized copy
print("LassoCV chose alpha_ =", round(cv.alpha_, 3))   # LassoCV chose alpha_ = 0.079


# %% [markdown]
# ## 5.3.3 When each one wins

# %% [markdown]
# Example 5-3 builds exactly that: 80 samples but 100 features, only 10 of which actually drive the target, and compares ordinary least squares against the three regularized models.
#
# Example 5-3. Ordinary least squares against cross-validated Ridge, Lasso, and Elastic Net on 80 synthetic samples with 100 features, 10 of them informative, scored by held-out $R^2$ and by how many features each keeps.

# %%
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import r2_score

# 80 samples, 100 features, but only 10 truly drive y: a known-sparse ground
# truth, so we can count how many of the 100 features Lasso keeps.
X, y, coef = make_regression(n_samples=80, n_features=100, n_informative=10,
                             noise=10.0, random_state=0, coef=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

# Standardize first: the penalty is scale dependent (fit the scaler on train only).
scaler = StandardScaler().fit(X_tr)
X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)

lasso = None
for name, m in [("OLS", LinearRegression()),
                ("Ridge (CV)", RidgeCV(alphas=np.logspace(-2, 3, 30))),
                ("Lasso (CV)", LassoCV(cv=5, max_iter=5000)),
                ("ElasticNet (CV)", ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 0.99],
                                                 cv=5, max_iter=5000))]:
    m.fit(X_tr, y_tr)
    if name.startswith("Lasso"):
        lasso = m
    r2 = r2_score(y_te, m.predict(X_te))
    k = int(np.sum(np.abs(m.coef_) > 1e-6))
    print(f"{name:16s} {r2:.3f} {k:3d}/100")
# OLS              0.131 100/100
# Ridge (CV)       0.146 100/100
# Lasso (CV)       0.983  32/100
# ElasticNet (CV)  0.950  51/100
print(round(LinearRegression().fit(X_tr, y_tr).score(X_tr, y_tr), 6))
# 1.0
info = np.where(np.abs(coef) > 0)[0]
kept = np.where(np.abs(lasso.coef_) > 1e-6)[0]
dropped = [int(i) for i in info if i not in set(kept.tolist())]
print(int(np.isin(kept, info).sum()), int((~np.isin(kept, info)).sum()),
      dropped, [round(float(coef[i]), 3) for i in dropped])
# 9 23 [8] [1.214]


# %% [markdown]
# ## 5.3.4 The regularization path

# %% [markdown]
# We can reproduce the path directly with scikit-learn's `lasso_path`, on the same standardized data as Example 5-3.

# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import lasso_path

HERE = Path(__file__).parent if "__file__" in globals() else Path(".")  # save beside this script

# the same known-sparse data as 5.3.3, standardized on the training split
X, y = make_regression(n_samples=80, n_features=100, n_informative=10,
                       noise=10.0, random_state=0)
X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.3, random_state=0)
X_tr = StandardScaler().fit_transform(X_tr)

alphas, coefs, _ = lasso_path(X_tr, y_tr)      # coefs has shape (n_features, n_alphas)
plt.figure()
for c in coefs:                                # one line per feature's coefficient
    plt.plot(alphas, c, lw=0.8)
plt.xscale("log"); plt.gca().invert_xaxis()    # strongest penalty on the left
plt.xlabel("alpha (penalty strength)"); plt.ylabel("coefficient value")
plt.savefig(HERE / "lasso_path.png", dpi=80)
print(len(alphas),
      int(np.sum(np.abs(coefs[:, 0]) > 1e-6)),      # nonzero at the strongest penalty
      int(np.sum(np.abs(coefs[:, -1]) > 1e-6)))     # nonzero at the weakest penalty
# 100 0 55


# %% [markdown]
# ## 5.3.5 Choosing among models: AIC, BIC, Cp, and stepwise search

# %% [markdown]
# On the ten standardized diabetes features of 5.3.2, $n = 442$ so $\ln n = 6.091$, and in Example 5-4 the AIC and BIC winners disagree in a way the formulas predict.
#
# Example 5-4. Forward stepwise search by AIC and exhaustive best-subset search by AIC and BIC on the ten standardized diabetes features, set beside cross-validated sequential selection, LassoCV, and Mallows' $C_p$.

# %%
import numpy as np
from itertools import combinations
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.feature_selection import SequentialFeatureSelector
import statsmodels.api as sm

data = load_diabetes(scaled=False)               # same raw clinical units as 5.3.2
X, y = data.data, data.target
names = list(data.feature_names)
Xs = StandardScaler().fit_transform(X)
n, p = Xs.shape

def aic_of(cols):
    return float(sm.OLS(y, sm.add_constant(Xs[:, cols])).fit().aic)

remaining, chosen, best = list(range(p)), [], aic_of([])
path = []
while remaining:
    a, j = min((aic_of(chosen + [j]), j) for j in remaining)
    if a >= best:
        break
    chosen.append(j)
    remaining.remove(j)
    best = a
    path.append(([names[k] for k in chosen], round(a, 2)))
print(path)
# [(['bmi'], 4912.04), (['bmi', 's5'], 4828.4), (['bmi', 's5', 'bp'], 4813.23),
#  (['bmi', 's5', 'bp', 's1'], 4804.96), (['bmi', 's5', 'bp', 's1', 'sex'], 4800.08),
#  (['bmi', 's5', 'bp', 's1', 'sex', 's2'], 4788.6)]

best_a, best_a_set = 1e300, None
best_b, best_b_set = 1e300, None
for k in range(p + 1):
    for cols in combinations(range(p), k):
        m = sm.OLS(y, sm.add_constant(Xs[:, list(cols)])).fit()
        if m.aic < best_a:
            best_a, best_a_set = float(m.aic), cols
        if m.bic < best_b:
            best_b, best_b_set = float(m.bic), cols
print(round(best_a, 2), [names[i] for i in best_a_set])
print(round(best_b, 2), [names[i] for i in best_b_set])
# 4788.6 ['sex', 'bmi', 'bp', 's1', 's2', 's5']
# 4816.81 ['sex', 'bmi', 'bp', 's3', 's5']

sfs = SequentialFeatureSelector(
    LinearRegression(), n_features_to_select="auto",
    scoring="neg_mean_squared_error", cv=5).fit(Xs, y)
print(sfs.n_features_to_select_, [names[i] for i in np.where(sfs.get_support())[0]])
# 5 ['sex', 'bmi', 'bp', 's3', 's5']

cv = LassoCV(cv=5, max_iter=100_000).fit(Xs, y)
print([names[i] for i in np.where(np.abs(cv.coef_) > 1e-6)[0]], round(float(cv.alpha_), 3))
# ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's4', 's5', 's6'] 0.079

full = sm.OLS(y, sm.add_constant(Xs)).fit()
sig2 = float(full.mse_resid)
print(round(sig2, 1), round(2 * sig2 / n, 2))     # sigma hat squared and the gap 2 sigma hat squared / n
# 2932.7 13.27
for label, cols in [("AIC", list(best_a_set)), ("BIC", list(best_b_set)), ("full", list(range(p)))]:
    m = sm.OLS(y, sm.add_constant(Xs[:, cols])).fit()
    cp = (float(m.ssr) + 2 * (int(m.df_model) + 1) * sig2) / n
    print(label, round(float(m.aic), 2), round(float(m.bic), 2), round(cp, 1))
# AIC 4788.6 4817.24 2969.6
# BIC 4792.26 4816.81 2993.4
# full 4793.99 4838.99 3005.7


# %% [markdown]
# ## 5.4.1 From a score to a probability: the sigmoid

# %% [markdown]
# We met its values before, and they are worth seeing again as the model's actual outputs.

# %%
import numpy as np

sigmoid = lambda z: 1 / (1 + np.exp(-z))
print(round(sigmoid(0), 3), round(sigmoid(2), 3), round(sigmoid(-2), 3))   # 0.5 0.881 0.119


# %% [markdown]
# ## 5.4.3 From scratch, and with scikit-learn

# %% [markdown]
# Example 5-5. Logistic regression from scratch and with scikit-learn on the breast-cancer data.

# %%
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

sigmoid = lambda z: 1 / (1 + np.exp(-z))

data = load_breast_cancer()                           # 569 real tumors: benign vs malignant
X = data.data[:, [0, 1]]                              # two features: mean radius, mean texture
y = data.target                                       # 1 = benign, 0 = malignant
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
scaler = StandardScaler().fit(X_tr)                   # fit the scaler on training rows only
X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)

def fit_logistic(X, y, eta=0.5, iters=5000):
    Xa = np.column_stack([np.ones(len(X)), X])        # add an intercept column
    w = np.zeros(Xa.shape[1])
    for _ in range(iters):
        p = sigmoid(Xa @ w)
        w -= eta * (Xa.T @ (p - y)) / len(y)          # the X^T (p - y) / n gradient
    return w

w = fit_logistic(X_tr, y_tr)
print(w.round(3))                                     # [ 0.585 -3.579 -0.883]
pred = (sigmoid(np.column_stack([np.ones(len(X_te)), X_te]) @ w) >= 0.5).astype(int)
print(round(accuracy_score(y_te, pred), 3))           # 0.912

clf = LogisticRegression(C=1e6).fit(X_tr, y_tr)       # C=1e6, nearly unregularized, to match the hand fit
print(np.round([clf.intercept_[0], *clf.coef_[0]], 3))   # [ 0.585 -3.578 -0.883]
print(round(accuracy_score(y_te, clf.predict(X_te)), 3)) # 0.912
hand_tr = (sigmoid(np.column_stack([np.ones(len(X_tr)), X_tr]) @ w) >= 0.5).astype(int)
print(round(accuracy_score(y_tr, hand_tr), 3),
      round(accuracy_score(y_tr, clf.predict(X_tr)), 3))  # 0.892 0.892


# %% [markdown]
# Statsmodels reaches those estimates by iteratively reweighted least squares, Newton's method of 2.6.5 written as the weighted least squares of 5.1.5: at the current $\mathbf{w}$ one forms $p_i = \sigma(x_i^\top\mathbf{w})$, $W$, and the working response $z = X\mathbf{w} + W^{-1}(\mathbf{y} - \mathbf{p})$, then solves $X^\top W X\,\mathbf{w}_{\mathrm{new}} = X^\top W z$. Example 5-6 runs six of those steps from scratch on the two standardized features of Example 5-5 and checks them against statsmodels.
#
# Example 5-6. Iteratively reweighted least squares from scratch on the two breast-cancer features, checked against statsmodels' Logit, with the standard errors, Wald $z$ values, and a likelihood-ratio test.

# %%
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from scipy.stats import chi2

Xb = load_breast_cancer().data[:, [0, 1]]
yb = load_breast_cancer().target
Xtr, _, ytr, _ = train_test_split(Xb, yb, test_size=0.3, random_state=0)
Xtr = StandardScaler().fit_transform(Xtr)
Xa = np.column_stack([np.ones(len(Xtr)), Xtr])
sig = lambda z: 1 / (1 + np.exp(-np.clip(z, -30, 30)))
w = np.zeros(3)
for it in range(6):
    p = sig(Xa @ w)
    W = np.clip(p * (1 - p), 1e-12, None)
    z = Xa @ w + (ytr - p) / W
    w = np.linalg.solve((Xa.T * W) @ Xa, (Xa.T * W) @ z)
    print(it, np.round(w, 4).tolist())
# 0 [0.5025, -1.2818, -0.3732]
# 1 [0.5942, -2.2308, -0.5944]
# 2 [0.5966, -3.043, -0.7754]
# 3 [0.5871, -3.4869, -0.8658]
# 4 [0.5852, -3.5766, -0.8828]
# 5 [0.5852, -3.5795, -0.8834]
m = sm.Logit(ytr, Xa).fit(disp=0)
print(np.round(m.params, 4).tolist(), np.round(m.bse, 4).tolist())
p = sig(Xa @ m.params)
se = np.sqrt(np.diag(np.linalg.inv((Xa.T * (p * (1 - p))) @ Xa)))
print(np.round(se, 4).tolist(), np.round(m.params / m.bse, 3).tolist())
m0 = sm.Logit(ytr, Xa[:, [0]]).fit(disp=0)
lrt = 2 * (m.llf - m0.llf)
print(round(float(lrt), 2), round(float(chi2.sf(lrt, 2)), 6))
# [0.5852, -3.5795, -0.8834] [0.1754, 0.4035, 0.179]
# [0.1754, 0.4035, 0.179] [3.336, -8.872, -4.936]
# 310.14 0.0


# %% [markdown]
# ## 5.4.4 When the classes separate perfectly

# %% [markdown]
# Fitting `LogisticRegression` on the two standardized petal columns, then statsmodels' unpenalized `Logit`, shows what the theory predicts.

# %%
import numpy as np
import warnings
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

iris = load_iris()
mask = iris.target < 2                               # setosa vs versicolor, 100 flowers
X = StandardScaler().fit_transform(iris.data[mask][:, 2:4])
y = iris.target[mask]
print(iris.data[iris.target == 0][:, 2:4].max(0), iris.data[iris.target == 1][:, 2:4].min(0))
# [1.9 0.6] [3. 1.]

for C in [0.01, 0.1, 1, 10, 100, 1e4, 1e6]:
    clf = LogisticRegression(C=C, solver="lbfgs", max_iter=5000).fit(X, y)
    print(C, round(float(np.linalg.norm(clf.coef_)), 3), round(clf.score(X, y), 3))
# 0.01 0.463 1.0
# 0.1 1.503 1.0
# 1 2.942 1.0
# 10 4.722 1.0
# 100 6.774 1.0
# 10000.0 9.544 1.0
# 1000000.0 9.627 1.0

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")                 # record every warning it issues
    res = sm.Logit(y, sm.add_constant(X)).fit(disp=0)   # unpenalized maximum likelihood
print(sorted({rec.category.__name__ for rec in caught}), res.mle_retvals["converged"])
print(res.mle_retvals["iterations"], round(float(np.linalg.norm(res.params[1:])), 3))
# ['ConvergenceWarning', 'PerfectSeparationWarning'] False
# 35 32.061


# %% [markdown]
# ## 5.4.5 Interpreting coefficients and going multiclass

# %% [markdown]
# We fit it on the wine dataset from scikit-learn, 178 real wines from the same region of Italy, derived from three different cultivars (grape varieties) and described by 13 chemical measurements (the UCI Wine data, bundled in scikit-learn).

# %%
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

wine = load_wine()                                    # 178 wines, 3 cultivars, 13 features
X_tr, X_te, y_tr, y_te = train_test_split(wine.data, wine.target, test_size=0.3, random_state=0)
scaler = StandardScaler().fit(X_tr)                   # fit the scaler on training rows only
X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)

clf = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
print(clf.coef_.shape)                                    # (3, 13)
print(round(accuracy_score(y_te, clf.predict(X_te)), 3)) # 1.0


# %% [markdown]
# The next block refits softmax and fits one-versus-rest on the same wine split and scaling.

# %%
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, log_loss

wine = load_wine()                                  # same split and scaling as the softmax block
X_tr, X_te, y_tr, y_te = train_test_split(wine.data, wine.target, test_size=0.3, random_state=0)
scaler = StandardScaler().fit(X_tr)
X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)

soft = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)                      # one joint softmax
ovr = OneVsRestClassifier(LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)  # three binary fits
print(len(ovr.estimators_), ovr.estimators_[0].coef_.shape)
# 3 (1, 13)
raw = np.column_stack([m.predict_proba(X_te)[:, 1] for m in ovr.estimators_])  # model k: P(class k)
sums = raw.sum(axis=1)
print(raw[:3].round(3).tolist(), sums[:3].round(3).tolist())
# [[0.977, 0.008, 0.006], [0.002, 0.001, 0.997], [0.051, 0.987, 0.0]] [0.99, 1.0, 1.039]
print(raw[2].round(4).tolist(), ovr.predict_proba(X_te)[2].round(3).tolist())  # third wine
# [0.0513, 0.9875, 0.0001] [0.049, 0.951, 0.0]
print(len(y_te), round(float(sums.min()), 3), round(float(sums.max()), 3),
      bool(np.allclose(ovr.predict_proba(X_te), raw / sums[:, None])))
# 54 0.814 1.189 True
print(accuracy_score(y_te, soft.predict(X_te)), accuracy_score(y_te, ovr.predict(X_te)))
# 1.0 1.0
print(round(log_loss(y_te, soft.predict_proba(X_te)), 4),
      round(log_loss(y_te, ovr.predict_proba(X_te)), 4))
# 0.0359 0.0597


# %% [markdown]
# ## 5.4.6 Sparse logistic regression

# %% [markdown]
# On all 30 standardized breast-cancer features, a $C$ grid and then `LogisticRegressionCV` show the sparsity-accuracy trade.

# %%
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score

Xtr, Xte, ytr, yte = train_test_split(
    load_breast_cancer().data, load_breast_cancer().target, test_size=0.3, random_state=0)
sc = StandardScaler().fit(Xtr)
Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
for C in [0.01, 0.1, 1.0, 10.0]:
    clf = LogisticRegression(l1_ratio=1.0, solver="saga", C=C, max_iter=8000, random_state=0).fit(Xtr, ytr)
    print(C, int(np.sum(np.abs(clf.coef_) > 1e-6)), round(accuracy_score(yte, clf.predict(Xte)), 3))
cv = LogisticRegressionCV(solver="saga", l1_ratios=(1.0,), Cs=10, cv=5, max_iter=8000,
                          random_state=0, scoring="accuracy",
                          use_legacy_attributes=False).fit(Xtr, ytr)   # the 1.10 attribute form, no warning
print(round(float(np.asarray(cv.C_).ravel()[0]), 3), int(np.sum(np.abs(cv.coef_) > 1e-6)),
      round(accuracy_score(yte, cv.predict(Xte)), 3))
en = LogisticRegressionCV(solver="saga", l1_ratios=(0.2, 0.5, 0.8), Cs=8, cv=5, max_iter=8000,
                          random_state=0, scoring="accuracy",
                          use_legacy_attributes=False).fit(Xtr, ytr)
print(round(float(np.asarray(en.C_).ravel()[0]), 3), round(float(np.asarray(en.l1_ratio_).ravel()[0]), 2),
      int(np.sum(np.abs(en.coef_) > 1e-6)), round(accuracy_score(yte, en.predict(Xte)), 3))
# 0.01 3 0.83
# 0.1 7 0.965
# 1.0 14 0.965
# 10.0 26 0.936
# 2.783 23 0.936
# 0.268 0.2 26 0.971


# %% [markdown]
# ## 5.5.1 Robust regression with the Huber loss

# %% [markdown]
# The effect on data with a few outliers is clear.

# %%
import numpy as np
from sklearn.linear_model import LinearRegression, HuberRegressor

rng = np.random.default_rng(0)                         # synthetic so the TRUE slope is known
X = rng.uniform(0, 10, 120).reshape(-1, 1)
y = 2.0 * X.ravel() + 1.0 + rng.normal(0, 1.0, 120)    # true slope 2, true intercept 1
y[::12] += 40.0                                        # inject 10 big outliers into the line

print(round(LinearRegression().fit(X, y).coef_[0], 2))  # 1.53  pulled down by the outliers
print(round(HuberRegressor().fit(X, y).coef_[0], 2))    # 1.94  close to the true slope of 2


# %% [markdown]
# On this finite sample the robust slope is about 87 percent as efficient as least squares, the price of the robustness that let Huber recover a slope of 1.94 from the outlier data.

# %%
import numpy as np
from sklearn.linear_model import LinearRegression, HuberRegressor

rng = np.random.default_rng(0)
x = np.linspace(0, 10, 80)
ols_s, hub_s = [], []
for _ in range(400):
    y = 2.0 * x + 1.0 + rng.normal(0, 1.0, 80)
    X = x.reshape(-1, 1)
    ols_s.append(LinearRegression().fit(X, y).coef_[0])
    hub_s.append(HuberRegressor().fit(X, y).coef_[0])
print(round(float(np.var(ols_s, ddof=1)), 6), round(float(np.var(hub_s, ddof=1)), 6),
      round(float(np.var(ols_s, ddof=1) / np.var(hub_s, ddof=1)), 3))
# 0.001505 0.001734 0.868


# %% [markdown]
# RANSAC is the standard tool when a sizeable fraction of the data is plainly wrong rather than noisy, as in computer vision where it fits geometry to point matches riddled with mismatches.

# %%
import numpy as np
from sklearn.linear_model import LinearRegression, RANSACRegressor

rng = np.random.default_rng(0)                         # synthetic: true slope 2 is the ground truth
X = rng.uniform(0, 10, 120).reshape(-1, 1)
y = 2.0 * X.ravel() + 1.0 + rng.normal(0, 1.0, 120)
y[::12] += 40.0                                        # 10 gross outliers on a slope-2 line

print(round(LinearRegression().fit(X, y).coef_[0], 2))  # 1.53  least squares is dragged off
ransac = RANSACRegressor(random_state=0).fit(X, y)
print(round(ransac.estimator_.coef_[0], 2))             # 1.96  recovers the true slope
print(int(ransac.inlier_mask_.sum()))                   # 109   the inliers it chose to trust (of 120)
print(round(np.median(np.abs(y - np.median(y))), 1))    # 5.2   the default residual_threshold (MAD of y)


# %% [markdown]
# On the same slope-2 data with its ten outliers it recovers the slope in one line.

# %%
import numpy as np
from sklearn.linear_model import TheilSenRegressor

rng = np.random.default_rng(0)                         # same synthetic slope-2 data, known truth
X = rng.uniform(0, 10, 120).reshape(-1, 1)
y = 2.0 * X.ravel() + 1.0 + rng.normal(0, 1.0, 120)
y[::12] += 40.0                                        # the same 10 gross outliers

print(round(TheilSenRegressor(random_state=0).fit(X, y).coef_[0], 2))   # 1.97


# %% [markdown]
# ## 5.5.2 Quantile regression

# %% [markdown]
# Fitting the 10th, 50th, and 90th percentiles gives three lines that bracket the data.

# %%
import numpy as np
from sklearn.linear_model import QuantileRegressor

rng = np.random.default_rng(0)                         # synthetic so the true spread is known
Xq = rng.uniform(0, 10, 300).reshape(-1, 1)
yq = 2.0 * Xq.ravel() + 1.0 + rng.normal(0, 2.0, 300)
for q in [0.1, 0.5, 0.9]:
    qr = QuantileRegressor(quantile=q, alpha=0.0, solver="highs").fit(Xq, yq)
    print(q, round(qr.coef_[0], 2), round(qr.intercept_, 2))
# 0.1  2.04  -1.53
# 0.5  2.01   0.95
# 0.9  1.92   3.98


# %% [markdown]
# ## 5.5.3 Generalized linear models

# %% [markdown]
# For Gaussian noise that gap is $\mathrm{RSS}/(2\sigma^2)$, so the deviance is $\mathrm{RSS}/\sigma^2$, and once $\sigma^2$ is factored out it is exactly the residual sum of squares.

# %%
import numpy as np
from sklearn.linear_model import LinearRegression, PoissonRegressor

rng = np.random.default_rng(0)                         # synthetic: the true log-linear rate is known
Xc = rng.uniform(0, 3, 400).reshape(-1, 1)
yc = rng.poisson(np.exp(0.5 + 0.8 * Xc.ravel()))      # counts from a log-linear rate
pois = PoissonRegressor(alpha=0.0).fit(Xc, yc)
print(round(pois.intercept_, 2), round(pois.coef_[0], 2))            # 0.4 0.84  (true 0.5, 0.8)
print(round(LinearRegression().fit(Xc, yc).predict([[0.0]])[0], 2))  # -1.11  a negative count!


# %% [markdown]
# A negative binomial GLM keeps the log link and lets the variance grow faster than the mean, which is what the next two blocks check (`sm.GLM` with `family=sm.families.NegativeBinomial()`, or `sm.NegativeBinomial`).

# %%
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import PoissonRegressor

rng = np.random.default_rng(0)
Xc = rng.uniform(0, 3, 400).reshape(-1, 1)
yc = rng.poisson(np.exp(0.5 + 0.8 * Xc.ravel()))      # the well-specified counts of the Poisson demo
mu = PoissonRegressor(alpha=0.0).fit(Xc, yc).predict(Xc)
print(round(float(np.sum((yc - mu) ** 2 / mu) / (400 - 2)), 3))   # 1.062  near 1: Poisson is fair

rng = np.random.default_rng(2)                         # synthetic extra-Poisson spread
lam = np.exp(0.5 + 0.8 * Xc.ravel())
yo = rng.poisson(rng.gamma(shape=1.0, scale=lam))      # gamma-Poisson mixture, variance > mean
print(round(float(yo.mean()), 1), round(float(yo.var(ddof=1)), 1))  # 7.1 97.4  variance far above the mean
Xsm = sm.add_constant(Xc.ravel())
poi = sm.GLM(yo, Xsm, family=sm.families.Poisson()).fit()
nb = sm.GLM(yo, Xsm, family=sm.families.NegativeBinomial(alpha=1.0)).fit()
print(round(float(poi.pearson_chi2 / poi.df_resid), 3))            # 7.461  overdispersed
print(round(float(nb.pearson_chi2 / nb.df_resid), 3))              # 0.927  negative binomial absorbs it


# %% [markdown]
# The OpenML table `freMTPL2freq` is 678,013 French motor policies with a ClaimNb count and an Exposure column; a 5,000-row subsample drawn with seed 0 is enough to see the offset move the coefficients.

# %%
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path
from sklearn.datasets import fetch_openml

DATA = Path("code/ch17_time_series/data")                # relative to the book folder
if not DATA.exists():                                    # run from code/ch05_regression instead
    DATA = Path.cwd().resolve().parents[1] / DATA
bike = pd.read_csv(DATA / "bike_sharing_daily.csv")
y = bike["cnt"].to_numpy(float)
X = sm.add_constant(bike[["temp", "hum", "windspeed", "workingday"]].to_numpy(float))
print(round(float(y.mean()), 1), round(float(y.var(ddof=1)), 1))
poi = sm.GLM(y, X, family=sm.families.Poisson()).fit()
nb = sm.NegativeBinomial(y, X).fit(disp=0, maxiter=200)
print(round(float(poi.pearson_chi2 / poi.df_resid), 3), np.round(poi.params, 4).tolist())
print(np.round(poi.bse, 4).tolist())
print(np.round(nb.params, 4).tolist(), np.round(nb.bse, 4).tolist())
# 4504.3 3752788.2
# 516.034 [8.2469, 1.4612, -0.6561, -1.1011, 0.0309]
# [0.0038, 0.0031, 0.0041, 0.0079, 0.0012]
# [8.2241, 1.7762, -0.8433, -1.2801, 0.0446, 0.1508] [0.0991, 0.0872, 0.1116, 0.1957, 0.0309, 0.0077]

fr = fetch_openml("freMTPL2freq", version=1, as_frame=True, parser="auto").frame
rng = np.random.default_rng(0)
sub = fr.iloc[rng.choice(len(fr), size=5000, replace=False)]
Xoff = sm.add_constant(np.column_stack([
    sub["VehPower"].to_numpy(float), sub["VehAge"].to_numpy(float),
    sub["DrivAge"].to_numpy(float), np.log(sub["Density"].to_numpy(float) + 1),
]))
yoff = sub["ClaimNb"].to_numpy(float)
off = np.log(sub["Exposure"].to_numpy(float))
m_off = sm.GLM(yoff, Xoff, family=sm.families.Poisson(), offset=off).fit()
m_no = sm.GLM(yoff, Xoff, family=sm.families.Poisson()).fit()
print(round(float(m_off.params[0]), 4), round(float(m_no.params[0]), 4),
      round(float(sub["Exposure"].mean()), 3))
print(round(float(sub["Exposure"].min()), 4), round(float(sub["Exposure"].max()), 4))
print(np.round(m_off.params, 4).tolist())
print(np.round(m_no.params, 4).tolist())
# -2.4092 -3.2356 0.53
# 0.0027 1.27
# [-2.4092, 0.0461, -0.0485, -0.0016, 0.0503]
# [-3.2356, 0.0228, -0.0224, 0.0072, 0.0073]


# %% [markdown]
# Scikit-learn's `mean_absolute_percentage_error` returns the fraction, not the percent, which is why the Gamma comparison multiplies it by $100$.

# %%
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, GammaRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

X, y = fetch_california_housing(return_X_y=True)         # MedHouseVal: a positive, right-skewed target
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)

gamma = make_pipeline(StandardScaler(), GammaRegressor(alpha=0.0)).fit(Xtr, ytr)
ols   = make_pipeline(StandardScaler(), LinearRegression()).fit(Xtr, ytr)
print(round(gamma.predict(Xte).min(), 3))   # 0.332   a Gamma log-link is never negative
print(round(ols.predict(Xte).min(), 3))     # -0.974  OLS predicts an impossible negative price
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
print(round(mean_absolute_percentage_error(yte, gamma.predict(Xte)) * 100, 1),   # 30.9  Gamma
      round(mean_absolute_percentage_error(yte, ols.predict(Xte)) * 100, 1))      # 32.3  OLS
print(round(mean_squared_error(yte, gamma.predict(Xte)) ** 0.5, 3),               # 0.984 Gamma RMSE
      round(mean_squared_error(yte, ols.predict(Xte)) ** 0.5, 3))                 # 0.727 OLS RMSE


# %% [markdown]
# ## 5.5.4 Transforming the target

# %% [markdown]
# The California median house values are skewed to the right, with a skewness of 0.98, which makes them a natural candidate.

# %%
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import root_mean_squared_error

X, y = fetch_california_housing(return_X_y=True)        # MedHouseVal is positive and skewed to the right
from scipy.stats import skew
print(round(float(skew(y)), 2))                         # 0.98
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)

plain  = LinearRegression().fit(Xtr, ytr)
logged = TransformedTargetRegressor(                   # fit on log(y), predict back through exp
    regressor=LinearRegression(), func=np.log, inverse_func=np.exp).fit(Xtr, ytr)

def mape(p): return np.mean(np.abs((yte - p) / yte)) * 100   # mean absolute percentage error
for name, m in [("plain", plain), ("log", logged)]:
    p = m.predict(Xte)
    print(name, round(root_mean_squared_error(yte, p), 3), round(mape(p), 1))
# plain 0.727 32.3
# log   0.946 28.3


# %% [markdown]
# Seaborn's diamonds give a real elasticity; the California fit of this subsection gives a real smearing factor.

# %%
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dia = sns.load_dataset("diamonds")
dia = dia[(dia["price"] > 0) & (dia["carat"] > 0)]
ml = sm.OLS(np.log(dia["price"].to_numpy(float)),
            sm.add_constant(np.log(dia["carat"].to_numpy(float)))).fit()
print(np.round(ml.params, 4).tolist())
# [8.4487, 1.6758]

Xh, yh = fetch_california_housing(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(Xh, yh, test_size=0.2, random_state=0)
ols_log = LinearRegression().fit(Xtr, np.log(ytr))
resid = np.log(ytr) - ols_log.predict(Xtr)
smear = float(np.mean(np.exp(resid)))
naive = np.exp(ols_log.predict(Xte))
print(round(smear, 4), round(float(yte.mean()), 4),
      round(float(naive.mean()), 4), round(float((naive * smear).mean()), 4))
# 1.0651 2.0528 1.9639 2.0917


# %% [markdown]
# ## 5.6.1 Regression splines

# %% [markdown]
# Asking scikit-learn's `SplineTransformer` to expand a single value shows this directly.

# %%
import numpy as np
from sklearn.preprocessing import SplineTransformer

rng = np.random.default_rng(0)
X = np.sort(rng.uniform(0, 6, 120)).reshape(-1, 1)   # x-values sampled from the known curve f(x)=sin(2x)+0.3x

st = SplineTransformer(n_knots=6, degree=3).fit(X)   # learn knot positions from the data
print(st.transform([[2.0]]).round(2))                # one x -> 8 basis features
# [[0.   0.01 0.37 0.57 0.05 0.   0.   0.  ]]


# %% [markdown]
# Running linear regression on these features recovers the true curve far better than a global polynomial of comparable flexibility.

# %%
import numpy as np
from sklearn.preprocessing import SplineTransformer, PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

rng = np.random.default_rng(0)
true_f = lambda x: np.sin(2.0 * x) + 0.3 * x         # the known truth we are trying to recover
X = np.sort(rng.uniform(0, 6, 120)).reshape(-1, 1)
y = true_f(X).ravel() + rng.normal(0, 0.25, 120)     # noisy observations of it
grid = np.linspace(0, 6, 400).reshape(-1, 1)         # dense grid to score recovery against the truth
truth = true_f(grid).ravel()

poly_raw = make_pipeline(PolynomialFeatures(10), LinearRegression()).fit(X, y)
poly_sc  = make_pipeline(PolynomialFeatures(10), StandardScaler(), LinearRegression()).fit(X, y)
spline   = make_pipeline(SplineTransformer(n_knots=8, degree=3), LinearRegression()).fit(X, y)
print(f"degree-10 raw powers        : {mean_squared_error(truth, poly_raw.predict(grid)):.3f}")
print(f"degree-10 scaled powers     : {mean_squared_error(truth, poly_sc.predict(grid)):.3f}")
print(f"cubic spline (8 knots)      : {mean_squared_error(truth, spline.predict(grid)):.3f}")
sq = (truth - poly_raw.predict(grid)) ** 2            # raw fit's squared error at each grid point
edge = (grid.ravel() <= 0.5) | (grid.ravel() >= 5.5)  # the two edge strips [0, 0.5] and [5.5, 6]
print(f"raw powers training MSE     : {mean_squared_error(y, poly_raw.predict(X)):.3f}")
print(f"raw powers edge error share : {100 * sq[edge].sum() / sq.sum():.1f}%")
# degree-10 raw powers        : 0.228
# degree-10 scaled powers     : 0.009
# cubic spline (8 knots)      : 0.009
# raw powers training MSE     : 0.268
# raw powers edge error share : 5.4%


# %% [markdown]
# On seaborn's `mpg` table, after dropping the rows that miss horsepower, 392 cars remain; in Example 5-7, horsepower predicting mpg makes the two bases comparable.
#
# Example 5-7. The truncated power basis built by hand for horsepower on the mpg cars, matched against SplineTransformer at the same knots, with the quantile knot count chosen by five-fold cross-validation.

# %%
import numpy as np
import seaborn as sns
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

mpg = sns.load_dataset("mpg").dropna(subset=["horsepower", "mpg"])
x = mpg["horsepower"].to_numpy(float)
y = mpg["mpg"].to_numpy(float)
knots = np.quantile(x, [0.2, 0.4, 0.6, 0.8])
print(np.round(knots, 2).tolist())

def truncated_power(x, knots):
    x = np.asarray(x, float).ravel()
    cols = [np.ones(len(x)), x, x ** 2, x ** 3]
    for k in knots:
        cols.append(np.maximum(x - k, 0.0) ** 3)
    return np.column_stack(cols)

T = truncated_power(x, knots)
yp = T @ np.linalg.lstsq(T, y, rcond=None)[0]
st = SplineTransformer(degree=3, knots=np.concatenate([[x.min()], knots, [x.max()]]).reshape(-1, 1),
                       include_bias=True)
yp2 = LinearRegression(fit_intercept=False).fit(st.fit_transform(x.reshape(-1, 1)), y).predict(
    st.transform(x.reshape(-1, 1)))
print(round(float(np.mean((y - yp) ** 2)), 3), round(float(np.mean((y - yp2) ** 2)), 3),
      round(float(np.max(np.abs(yp - yp2))), 4))
scores = []
for nk in range(3, 9):
    pipe = make_pipeline(SplineTransformer(n_knots=nk, degree=3, knots="quantile"), LinearRegression())
    scores.append((nk, round(float(-cross_val_score(pipe, x.reshape(-1, 1), y, cv=5,
                     scoring="neg_mean_squared_error").mean()), 3)))
print(scores)
# [72.0, 88.0, 100.0, 140.0]
# 18.136 18.136 0.0
# [(3, 23.76), (4, 23.078), (5, 23.203), (6, 23.389), (7, 23.36), (8, 23.248)]


# %% [markdown]
# Example 5-8. A natural cubic spline basis built from scratch, checked against patsy, and compared with an ordinary cubic spline on the mpg cars.

# %%
import numpy as np
import seaborn as sns
from patsy import dmatrix
from sklearn.preprocessing import SplineTransformer
from sklearn.model_selection import KFold

def natural_cubic_basis(x, xi):
    """Columns 1, x, B_1, ..., B_(K-2): cubic between the knots xi, straight outside them."""
    x = np.asarray(x, float).ravel()
    cube = lambda t: np.maximum(x - t, 0.0) ** 3          # truncated cube (x - t)^3_+
    gap = xi[-1] - xi[-2]
    cols = [np.ones_like(x), x]
    for k in range(len(xi) - 2):                         # one column per knot except the last two
        cols.append(cube(xi[k]) - (xi[-1] - xi[k]) / gap * cube(xi[-2])
                    + (xi[-2] - xi[k]) / gap * cube(xi[-1]))
    return np.column_stack(cols)

# by hand: knots 0, 1, 2, 3; past x = 3 each column should rise by a constant step
knots4 = np.array([0.0, 1.0, 2.0, 3.0])
print(natural_cubic_basis([-1.0, 2.5, 3.0, 4.0, 5.0], knots4).round(3).tolist())
# [[1.0, -1.0, 0.0, 0.0], [1.0, 2.5, 15.25, 3.125], [1.0, 3.0, 24.0, 6.0],
#  [1.0, 4.0, 42.0, 12.0], [1.0, 5.0, 60.0, 18.0]]

mpg = sns.load_dataset("mpg").dropna(subset=["horsepower", "mpg"])     # the same 392 cars
x, y = mpg["horsepower"].to_numpy(float), mpg["mpg"].to_numpy(float)
xi = np.concatenate([[x.min()], np.quantile(x, [0.2, 0.4, 0.6, 0.8]), [x.max()]])
N = natural_cubic_basis(x, xi)                                         # natural cubic spline
st = SplineTransformer(degree=3, knots=xi.reshape(-1, 1), include_bias=True,
                       extrapolation="continue").fit(x.reshape(-1, 1))  # ordinary cubic spline
B = st.transform(x.reshape(-1, 1))
cr_formula = "cr(x, knots=k, lower_bound=lo, upper_bound=hi) - 1"      # patsy's natural basis
P = np.asarray(dmatrix(cr_formula, {"x": x, "k": xi[1:-1], "lo": xi[0], "hi": xi[-1]}))
fit = lambda M: np.linalg.lstsq(M, y, rcond=None)[0]                   # least squares, as in 5.1
bN, bB, bP = fit(N), fit(B), fit(P)
print(xi.tolist(), N.shape[1], B.shape[1], P.shape[1])
# [46.0, 72.0, 88.0, 100.0, 140.0, 230.0] 6 8 6
print(round(float(np.mean((y - N @ bN) ** 2)), 3), round(float(np.mean((y - B @ bB) ** 2)), 3),
      bool(np.max(np.abs(N @ bN - P @ bP)) < 1e-8))
# 18.297 18.136 True
print(f"{np.linalg.cond(N):.1e}", round(float(np.linalg.cond(P)), 1))
# 3.0e+07 5.1

def curve_se(M, b, M0):                    # sigma^2 x0^T (M^T M)^-1 x0, the variance of 5.1.4
    s2 = np.sum((y - M @ b) ** 2) / (len(y) - M.shape[1])
    return np.sqrt(s2 * np.sum((M0 @ np.linalg.inv(M.T @ M)) * M0, axis=1))

x0 = np.array([46.0, 100.0, 230.0])                                    # both ends and the middle
P0 = np.asarray(dmatrix(cr_formula, {"x": x0, "k": xi[1:-1], "lo": xi[0], "hi": xi[-1]}))
print(curve_se(P, bP, P0).round(3).tolist(),
      curve_se(B, bB, st.transform(x0.reshape(-1, 1))).round(3).tolist())
# [1.442, 0.394, 1.524] [2.069, 0.415, 2.241]
xe = np.array([30.0, 250.0, 300.0])                                    # horsepower outside the data
print((natural_cubic_basis(xe, xi) @ bN).round(2).tolist(),
      (st.transform(xe.reshape(-1, 1)) @ bB).round(2).tolist())
# [35.66, 12.9, 12.99] [7.77, 18.15, 39.97]

cv = {"natural": [], "cubic": []}
for tr, te in KFold(10, shuffle=True, random_state=0).split(x):
    for name, basis in [("natural", lambda v: natural_cubic_basis(v, xi)),
                        ("cubic", lambda v: st.transform(v.reshape(-1, 1)))]:
        b = np.linalg.lstsq(basis(x[tr]), y[tr], rcond=None)[0]
        cv[name].append(np.mean((y[te] - basis(x[te]) @ b) ** 2))
print({k: round(float(np.mean(v)), 3) for k, v in cv.items()})
# {'natural': 18.929, 'cubic': 19.145}


# %% [markdown]
# ## 5.6.2 Smoothing splines

# %% [markdown]
# SciPy's `make_smoothing_spline` minimizes exactly this objective, choosing $\lambda$ by generalized cross-validation when none is given.

# %%
import numpy as np
from scipy.interpolate import make_smoothing_spline
from sklearn.metrics import mean_squared_error

rng = np.random.default_rng(0)
true_f = lambda x: np.sin(2.0 * x) + 0.3 * x         # the same known truth as 5.6.1
X = np.sort(rng.uniform(0, 6, 120)).reshape(-1, 1)
y = true_f(X).ravel() + rng.normal(0, 0.25, 120)
grid = np.linspace(0, 6, 400)
truth = true_f(grid)

ss = make_smoothing_spline(X.ravel(), y)             # lambda picked by generalized cross-validation
print(f"smoothing spline truth-recovery MSE = {mean_squared_error(truth, ss(grid)):.3f}")
# smoothing spline truth-recovery MSE = 0.006


# %% [markdown]
# The returned object is a `BSpline` and does not store the $\lambda$ it chose, so Example 5-9 builds $S_\lambda$ column by column by smoothing the unit vectors at a fixed `lam`, confirms the leave-one-out shortcut against a brute-force loop, compares a five-point GCV grid with the library's default fit, and ends with ridge's degrees of freedom, condition numbers, and cross-validated penalty on the standardized diabetes data.
#
# Example 5-9. The smoothing spline's smoother matrix built column by column, with its effective degrees of freedom, the leave-one-out shortcut checked against brute force, and a five-point GCV grid, then ridge's degrees of freedom, condition numbers, and cross-validated penalty on the standardized diabetes data.

# %%
import numpy as np
from scipy.interpolate import make_smoothing_spline
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV

rng = np.random.default_rng(0)
true_f = lambda x: np.sin(2.0 * x) + 0.3 * x
x = np.sort(rng.uniform(0, 6, 120))
y = true_f(x) + rng.normal(0, 0.25, 120)
n = len(x)
eye = np.eye(n)

def smoother_matrix(lam):
    S = np.empty((n, n))
    for j in range(n):
        S[:, j] = make_smoothing_spline(x, eye[:, j], lam=lam)(x)
    return S

for lam in [1e-4, 1e-2, 0.1, 1.0, 100.0]:
    S = smoother_matrix(lam)
    yhat = make_smoothing_spline(x, y, lam=lam)(x)
    tr = float(np.trace(S))
    loo = float(np.mean(((y - yhat) / (1.0 - np.diag(S))) ** 2))
    gcv = float(np.mean((y - yhat) ** 2) / (1.0 - tr / n) ** 2)
    print(lam, round(tr, 3), round(loo, 5), round(gcv, 5))
# 0.0001 41.711 0.09409 0.08382
# 0.01 14.918 0.07104 0.07157
# 0.1 8.893 0.0729 0.07336
# 1.0 5.453 0.15762 0.15778
# 100.0 2.471 0.48172 0.48561

S = smoother_matrix(0.1)
yhat = make_smoothing_spline(x, y, lam=0.1)(x)
loo = float(np.mean(((y - yhat) / (1.0 - np.diag(S))) ** 2))
brute = [ (y[i] - make_smoothing_spline(np.delete(x, i), np.delete(y, i), lam=0.1)(x[i])) ** 2
          for i in range(n) ]
print(round(loo, 6), round(float(np.mean(brute)), 6))
# 0.072901 0.072901

grid = np.linspace(0, 6, 400)
print(round(mean_squared_error(true_f(grid), make_smoothing_spline(x, y)(grid)), 3),
      round(mean_squared_error(true_f(grid), make_smoothing_spline(x, y, lam=0.01)(grid)), 3))
# 0.006 0.007

Xd = StandardScaler().fit_transform(load_diabetes().data)
yd = load_diabetes().target
_, s, _ = np.linalg.svd(Xd, full_matrices=False)
yc = yd - yd.mean()
for lam in [0.0, 1.0, 10.0, 100.0, 1000.0]:
    df = float(np.sum(s ** 2 / (s ** 2 + lam)))
    print(lam, round(df, 3), round(float(np.linalg.cond(Xd.T @ Xd + lam * np.eye(Xd.shape[1]))), 1))
# 0.0 10.0 470.1
# 1.0 9.74 372.0
# 10.0 8.829 129.8
# 100.0 6.592 18.1
# 1000.0 2.509 2.8
print(round(float(RidgeCV(alphas=np.logspace(-2, 3, 30)).fit(Xd, yd).alpha_), 4))
# 1.7433


# %% [markdown]
# ## 5.6.3 Local regression

# %% [markdown]
# Example 5-10 implements the whole method in a handful of lines.
#
# Example 5-10. LOESS from scratch: a tricube-weighted local linear fit at each grid point, scored against the true curve.

# %%
import numpy as np
from sklearn.metrics import mean_squared_error

rng = np.random.default_rng(0)
true_f = lambda x: np.sin(2.0 * x) + 0.3 * x         # known truth, so recovery can be scored
X = np.sort(rng.uniform(0, 6, 120)).reshape(-1, 1)
y = true_f(X).ravel() + rng.normal(0, 0.25, 120)
grid = np.linspace(0, 6, 400).reshape(-1, 1)
truth = true_f(grid).ravel()

def loess_point(x0, X, y, span=0.3):
    d = np.abs(X.ravel() - x0)
    h = np.quantile(d, span)                       # window = span fraction of points
    w = np.clip(1 - (d / h)**3, 0, 1)**3           # tricube weights, zero past the window
    Xa = np.column_stack([np.ones(len(X)), X.ravel()])
    beta = np.linalg.solve(Xa.T @ (w[:, None] * Xa), Xa.T @ (w * y))   # weighted least squares
    return beta[0] + beta[1] * x0

loess_curve = np.array([loess_point(x0, X, y) for x0 in grid.ravel()])   # slide the window across the grid
print(f"LOESS (span 0.3) truth-recovery MSE = {mean_squared_error(truth, loess_curve):.3f}")
# LOESS (span 0.3) truth-recovery MSE = 0.040


# %% [markdown]
# ## 5.6.4 Generalized additive models

# %% [markdown]
# Building one is just a per-feature spline expansion followed by linear regression.

# %%
import numpy as np
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

rng = np.random.default_rng(0)
# two features with known shapes, so we can confirm each curve is recovered
x1 = rng.uniform(-3, 3, 500)
x2 = rng.uniform(0, 6, 500)
y2 = 0.5 * x1**2 + np.sin(x2) + rng.normal(0, 0.3, 500)   # a parabola in x1 plus a sine in x2
X2 = np.column_stack([x1, x2])
X2_tr, X2_te, y2_tr, y2_te = train_test_split(X2, y2, test_size=0.3, random_state=0)

gam = make_pipeline(
    ColumnTransformer([("s1", SplineTransformer(n_knots=8, degree=3), [0]),
                       ("s2", SplineTransformer(n_knots=8, degree=3), [1])]),
    LinearRegression()).fit(X2_tr, y2_tr)
lin = LinearRegression().fit(X2_tr, y2_tr)
print(f"plain linear model test R2 = {r2_score(y2_te, lin.predict(X2_te)):.3f}")   # 0.088
print(f"additive spline GAM test R2 = {r2_score(y2_te, gam.predict(X2_te)):.3f}")  # 0.956
g1 = np.linspace(-3, 3, 80)
g2 = np.linspace(0, 6, 80)
x2m, x1m = float(X2_tr[:, 1].mean()), float(X2_tr[:, 0].mean())
p1 = gam.predict(np.column_stack([g1, np.full_like(g1, x2m)]))
p2 = gam.predict(np.column_stack([np.full_like(g2, x1m), g2]))
p1c = p1 - p1.mean(); t1c = 0.5 * g1**2 - (0.5 * g1**2).mean()
p2c = p2 - p2.mean(); t2c = np.sin(g2) - np.sin(g2).mean()
print(round(float(np.max(np.abs(p1c - t1c))), 3),
      round(float(np.max(np.abs(p2c - t2c))), 3))
# 0.131  0.197   largest gap of each centered partial from the true shape


# %% [markdown]
# Example 5-11. A Poisson GAM of claim frequency on the French motor policies, against a Poisson GLM.

# %%
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance

fr = fetch_openml("freMTPL2freq", version=1, as_frame=True, parser="auto").frame  # 678,013 rows
fr["LogDensity"] = np.log(fr["Density"])               # inhabitants per square kilometer, logged
feats = ["DrivAge", "VehAge", "LogDensity"]
band = pd.cut(fr["DrivAge"], [17, 20, 25, 35, 50, 65, 80, 100])   # by hand: claims per policy-year
rates = fr.groupby(band, observed=True)[["ClaimNb", "Exposure"]].sum()
print((rates["ClaimNb"] / rates["Exposure"]).round(3).tolist(),
      round(float((fr["ClaimNb"] == 0).mean()), 3))       # share of policies with no claim
# [0.249, 0.161, 0.096, 0.099, 0.096, 0.095, 0.092] 0.95

tr, te = train_test_split(fr, test_size=0.25, random_state=0)
y_tr, y_te = tr["ClaimNb"] / tr["Exposure"], te["ClaimNb"] / te["Exposure"]   # claim frequency
w_tr, w_te = tr["Exposure"], te["Exposure"]                                   # exposure as weight
glm = make_pipeline(StandardScaler(), PoissonRegressor(alpha=0.0, max_iter=1000))
gam = make_pipeline(                          # one cubic spline per feature, then the log link
    ColumnTransformer([(f, SplineTransformer(n_knots=8, degree=3, include_bias=False), [f])
                       for f in feats]),
    PoissonRegressor(alpha=0.0, max_iter=1000))
for m in (glm, gam):
    m.fit(tr[feats], y_tr, poissonregressor__sample_weight=w_tr)
null = np.full(len(te), np.average(y_tr, weights=w_tr))   # one rate for every policy
dev = [mean_poisson_deviance(y_te, p, sample_weight=w_te)
       for p in (null, glm.predict(te[feats]), gam.predict(te[feats]))]
print([round(float(d), 4) for d in dev], [round(float(1 - d / dev[0]), 4) for d in dev[1:]])
# [0.6254, 0.6186, 0.613] [0.0108, 0.0198]

ages = np.array([18, 21, 25, 35, 45, 55, 65, 75, 85])
med = tr[feats].median()                               # hold the other two features at the median
grid = pd.DataFrame({"DrivAge": ages, "VehAge": med["VehAge"], "LogDensity": med["LogDensity"]})
print(glm.predict(grid).round(3).tolist())
# [0.121, 0.119, 0.117, 0.111, 0.106, 0.101, 0.096, 0.091, 0.087]
print(gam.predict(grid).round(3).tolist())
# [0.231, 0.179, 0.116, 0.07, 0.086, 0.079, 0.071, 0.084, 0.082]
print(int(te["ClaimNb"].sum()), round(float((glm.predict(te[feats]) * w_te).sum()), 1),
      round(float((gam.predict(te[feats]) * w_te).sum()), 1))
# 9033 9055.9 9038.0


# %% [markdown]
# ## 5.7.1 Load and explore

# %% [markdown]
# Example 5-12. Load the California housing frame and scan its extremes.

# %%
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing(as_frame=True)
df = data.frame
print(df.shape)                                          # (20640, 9)
print("target: mean %.2f, min %.2f, max %.2f"
      % (df.MedHouseVal.mean(), df.MedHouseVal.min(), df.MedHouseVal.max()))
print(df.corr(numeric_only=True)["MedHouseVal"].drop("MedHouseVal").sort_values().round(2))
for c in ["AveOccup", "AveRooms", "Population", "AveBedrms"]:
    print(f"  {c:11} p99 = {df[c].quantile(0.99):7.1f}   max = {df[c].max():7.1f}")
print("districts at the MedHouseVal cap:", int((df.MedHouseVal == df.MedHouseVal.max()).sum()))
print("unrounded max", float(df.MedHouseVal.max()))
for col in ("AveRooms", "AveBedrms", "AveOccup", "Population"):
    i = df[col].idxmax()
    pop, occ = float(df.loc[i, "Population"]), float(df.loc[i, "AveOccup"])
    print(col, round(pop, 1), round(pop / occ, 1))
# target: mean 2.07, min 0.15, max 5.00
# correlations, sorted: Latitude -0.14, AveBedrms -0.05, Longitude -0.05,
#   Population -0.02, AveOccup -0.02, HouseAge 0.11, AveRooms 0.15, MedInc 0.69
# AveOccup p99 = 5.4, max = 1243.3; AveRooms p99 = 10.4, max = 141.9
# Population p99 = 5805.8, max = 35682.0; AveBedrms p99 = 2.1, max = 34.1
# districts at the MedHouseVal cap: 965
# unrounded max 5.00001
# AveRooms 30.0 11.0
# AveBedrms 36.0 15.0
# AveOccup 7460.0 6.0
# Population 35682.0 4769.0


# %% [markdown]
# ## 5.7.2 Why cleaning matters

# %% [markdown]
# Example 5-13. The clipping experiment: the same spline GAM on raw and on clipped features.

# %%
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import (SplineTransformer, FunctionTransformer,
                                   PolynomialFeatures, StandardScaler)
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

df = fetch_california_housing(as_frame=True).frame
feats = fetch_california_housing().feature_names
X_tr, X_te, y_tr, y_te = train_test_split(df[feats], df["MedHouseVal"],
                                          test_size=0.2, random_state=0)

caps = {"AveRooms": 12.0, "AveBedrms": 3.0, "Population": 6000.0, "AveOccup": 6.0}
clipper = FunctionTransformer(
    lambda Xd: Xd.assign(**{c: Xd[c].clip(upper=h) for c, h in caps.items()}))

cv = KFold(n_splits=5, shuffle=True, random_state=0)
raw_gam  = make_pipeline(SplineTransformer(n_knots=5, degree=3), LinearRegression())
clip_gam = make_pipeline(clipper, SplineTransformer(n_knots=5, degree=3), LinearRegression())
rmse = lambda m: -cross_val_score(m, X_tr, y_tr, cv=cv,
                                  scoring="neg_root_mean_squared_error").mean()
print(f"Spline GAM CV RMSE on RAW features     = {rmse(raw_gam):.3f}")   # 13.902 (blows up)
print(f"Spline GAM CV RMSE on CLIPPED features = {rmse(clip_gam):.3f}")  #  0.611 (fixed)
print("Linear",
      f"{rmse(make_pipeline(StandardScaler(), LinearRegression())):.3f}",
      f"{rmse(make_pipeline(clipper, StandardScaler(), LinearRegression())):.3f}")
poly_raw = make_pipeline(StandardScaler(), PolynomialFeatures(2), LinearRegression())
poly_clip = make_pipeline(clipper, StandardScaler(), PolynomialFeatures(2),
                          LinearRegression())
print("Poly2", f"{rmse(poly_raw):.3f}", f"{rmse(poly_clip):.3f}")
# Linear 0.728 0.673
# Poly2 2.266 0.609
st = SplineTransformer(n_knots=5, degree=3).fit(X_tr[["AveOccup"]])
data_knots = st.bsplines_[0].t[3:-3]
print(np.round(data_knots, 1).tolist())
# [0.8, 150.5, 300.2, 450.0, 599.7]
occ = X_tr["AveOccup"].to_numpy()
print(round(float((occ <= data_knots[1]).mean()), 4),
      round(float(occ.max()), 1), round(float(X_te["AveOccup"].max()), 1))
# 0.9998 599.7 1243.3
Bte = st.transform(X_te[["AveOccup"]])
print(round(float(Bte.min()), 3), round(float(Bte.max()), 3))
# 0.0 0.667

# the 13.902 is a mean of five folds; look at each fold, then inside the bad one
folds = -cross_val_score(raw_gam, X_tr, y_tr, cv=cv,
                         scoring="neg_root_mean_squared_error")
print(np.round(folds, 3).tolist())
# [66.543, 1.076, 0.634, 0.629, 0.627]
tr, va = next(cv.split(X_tr))                      # fold 0, the one that blows up
occ_tr, occ_va = X_tr["AveOccup"].iloc[tr], X_tr["AveOccup"].iloc[va]
knot2 = np.linspace(occ_tr.min(), occ_tr.max(), 5)[1]   # this fold's second knot
miss = (y_tr.iloc[va] - raw_gam.fit(X_tr.iloc[tr], y_tr.iloc[tr])
        .predict(X_tr.iloc[va])).abs()
far = occ_va > knot2                               # validation rows past that knot
print(round(float(knot2), 1), int((occ_tr > knot2).sum()),
      occ_va[far].round(1).tolist(), miss[far].round(1).tolist())
# 150.7 1 [502.5, 230.2] [666.1, 3760.8]


# %% [markdown]
# ## 5.7.3 Compare models with cross-validation

# %% [markdown]
# Example 5-14. The lab's cross-validated leaderboard: six pipelines on the same five folds.

# %%
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import (StandardScaler, PolynomialFeatures,
                                   SplineTransformer, FunctionTransformer)
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.pipeline import make_pipeline

df = fetch_california_housing(as_frame=True).frame
feats = fetch_california_housing().feature_names
X_tr, X_te, y_tr, y_te = train_test_split(df[feats], df["MedHouseVal"],
                                          test_size=0.2, random_state=0)
caps = {"AveRooms": 12.0, "AveBedrms": 3.0, "Population": 6000.0, "AveOccup": 6.0}
clipper = FunctionTransformer(
    lambda Xd: Xd.assign(**{c: Xd[c].clip(upper=h) for c, h in caps.items()}))

models = {
    "Baseline (mean)":     DummyRegressor(),
    "Linear regression":   make_pipeline(clipper, StandardScaler(), LinearRegression()),
    "Ridge (CV)":          make_pipeline(clipper, StandardScaler(),
                                         RidgeCV(alphas=np.logspace(-2, 3, 30))),
    "Lasso (CV)":          make_pipeline(clipper, StandardScaler(),
                                         LassoCV(cv=5, random_state=0)),
    "Polynomial degree 2": make_pipeline(clipper, StandardScaler(),
                                         PolynomialFeatures(2), LinearRegression()),
    "Spline GAM":          make_pipeline(clipper, SplineTransformer(n_knots=5, degree=3),
                                         LinearRegression()),
}
cv = KFold(n_splits=5, shuffle=True, random_state=0)
for name, m in models.items():
    rmse = -cross_val_score(m, X_tr, y_tr, cv=cv,
                            scoring="neg_root_mean_squared_error").mean()
    print(f"{name:20} {rmse:.3f}")
# Baseline (mean)      1.157
# Linear regression    0.673
# Ridge (CV)           0.673
# Lasso (CV)           0.673
# Polynomial degree 2  0.609
# Spline GAM           0.611


# %% [markdown]
# Example 5-15. Spatial GroupKFold over one-degree cells.

# %%
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, GroupKFold
from sklearn.preprocessing import (StandardScaler, PolynomialFeatures,
                                   SplineTransformer, FunctionTransformer)
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

df = fetch_california_housing(as_frame=True).frame
feats = fetch_california_housing().feature_names
X_tr, X_te, y_tr, y_te = train_test_split(df[feats], df["MedHouseVal"],
                                          test_size=0.2, random_state=0)
caps = {"AveRooms": 12.0, "AveBedrms": 3.0, "Population": 6000.0, "AveOccup": 6.0}
clipper = FunctionTransformer(
    lambda Xd: Xd.assign(**{c: Xd[c].clip(upper=h) for c, h in caps.items()}))
groups = (np.floor(X_tr["Latitude"]).astype(int).astype(str) + "_" +
          np.floor(X_tr["Longitude"]).astype(int).astype(str))
print(int(groups.nunique()))   # 54 one-degree cells on the training split
gkf = GroupKFold(n_splits=5)
spatial = {
    "Linear regression":   make_pipeline(clipper, StandardScaler(), LinearRegression()),
    "Polynomial degree 2": make_pipeline(clipper, StandardScaler(),
                                         PolynomialFeatures(2), LinearRegression()),
    "Spline GAM":          make_pipeline(clipper, SplineTransformer(n_knots=5, degree=3),
                                         LinearRegression()),
}
for name, m in spatial.items():
    rmse = -cross_val_score(m, X_tr, y_tr, cv=gkf, groups=groups,
                            scoring="neg_root_mean_squared_error").mean()
    print(f"{name:20} {rmse:.3f}")
# Linear regression    0.694
# Polynomial degree 2  0.649
# Spline GAM           0.659


# %% [markdown]
# ## 5.7.4 Evaluate the winner

# %% [markdown]
# Example 5-16. The winner refit on all training data and scored once on the held-out test set.

# %%
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, FunctionTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = fetch_california_housing(as_frame=True).frame
feats = fetch_california_housing().feature_names
X_tr, X_te, y_tr, y_te = train_test_split(df[feats], df["MedHouseVal"],
                                          test_size=0.2, random_state=0)
caps = {"AveRooms": 12.0, "AveBedrms": 3.0, "Population": 6000.0, "AveOccup": 6.0}
clipper = FunctionTransformer(
    lambda Xd: Xd.assign(**{c: Xd[c].clip(upper=h) for c, h in caps.items()}))

winner = make_pipeline(clipper, StandardScaler(),
                       PolynomialFeatures(2), LinearRegression()).fit(X_tr, y_tr)
pred = winner.predict(X_te)
print(f"RMSE = {np.sqrt(mean_squared_error(y_te, pred)):.3f}")  # 0.606 (~$61k typical error)
print(f"MAE  = {mean_absolute_error(y_te, pred):.3f}")          # 0.431
print(f"R2   = {r2_score(y_te, pred):.3f}")                     # 0.719
resid_te = y_te.to_numpy() - pred
outside = (pred < 0) | (pred > 5.2) | (resid_te < -3) | (resid_te > 3)
cap = y_te.to_numpy() >= 4.999
vis = cap & ~outside
print(len(pred), int(outside.sum()),
      round(float(pred[vis].min()), 2), round(float(pred[vis].max()), 2))
# 4128 30 2.12 5.17


# %% [markdown]
# Example 5-17. Split conformal prediction: a 90 percent interval around the winner's predictions.

# %%
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, FunctionTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

df = fetch_california_housing(as_frame=True).frame
feats = fetch_california_housing().feature_names
X_tr, X_te, y_tr, y_te = train_test_split(df[feats], df["MedHouseVal"],
                                          test_size=0.2, random_state=0)
caps = {"AveRooms": 12.0, "AveBedrms": 3.0, "Population": 6000.0, "AveOccup": 6.0}
clipper = FunctionTransformer(
    lambda Xd: Xd.assign(**{c: Xd[c].clip(upper=h) for c, h in caps.items()}))

# 90% prediction interval by split conformal: calibrate the band on unseen residuals
X_fit, X_cal, y_fit, y_cal = train_test_split(X_tr, y_tr, test_size=0.25, random_state=1)
band = make_pipeline(clipper, StandardScaler(),
                     PolynomialFeatures(2), LinearRegression()).fit(X_fit, y_fit)
resid = np.abs(y_cal - band.predict(X_cal))     # calibration errors, target unseen in the fit
n_cal = len(resid)
level = np.ceil((n_cal + 1) * 0.90) / n_cal      # finite-sample level: 0.9004 for 4,128 rows
# method="higher" takes one order statistic above the k-th smallest (see text)
q = np.quantile(resid, level, method="higher")
center = band.predict(X_te)
lo, hi = center - q, center + q
coverage = ((y_te >= lo) & (y_te <= hi)).mean()
print(n_cal, round(float(level), 4))
# 4128 0.9004
k = int(np.ceil((n_cal + 1) * 0.90))             # the recipe's order statistic
print(k, round(float(np.sort(resid)[k - 1]), 4), round(float(q), 4))
# 3717 0.9096 0.9106
print(f"90% band half-width q = {q:.3f}  (+/- ${q*100:.0f}k)")   # 0.911 (+/- $91k)
print(f"empirical coverage on test = {coverage:.3f}")            # 0.891
from sklearn.metrics import mean_squared_error
y_cal_a = np.asarray(y_cal)
resid_a = np.abs(y_cal_a - band.predict(X_cal))
print(round(mean_squared_error(y_cal_a, band.predict(X_cal)) ** 0.5, 3))  # 0.597  calibration RMSE

def q_on(mask):
    r = resid_a[mask]
    n = len(r)
    return float(np.quantile(r, np.ceil((n + 1) * 0.90) / n, method="higher"))

print(int((y_cal_a >= 4.99).sum()),
      round(q_on(y_cal_a < 4.99), 3),
      round(q_on(y_cal_a <= np.quantile(y_cal_a, 0.80)), 3))
# 192 0.849 0.747
inside = (np.asarray(y_te) >= lo) & (np.asarray(y_te) <= hi)
pred_q = np.quantile(center, [0.2, 0.4, 0.6, 0.8])
bins = np.digitize(center, pred_q)
print([round(float(inside[bins == b].mean()), 3) for b in range(5)])
# [0.969, 0.964, 0.914, 0.84, 0.766]
capped = np.asarray(y_te) >= 4.99
print(int(capped.sum()), round(float(inside[capped].mean()), 2),
      round(float(inside[~capped].mean()), 2))
# 185 0.47 0.91


# %% [markdown]
# ## 5.7.5 Read the model

# %% [markdown]
# Example 5-18. Standardized coefficients and the VIF check of 5.1.5.

# %%
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LinearRegression
df = fetch_california_housing(as_frame=True).frame
feats = fetch_california_housing().feature_names
X_tr, X_te, y_tr, y_te = train_test_split(df[feats], df["MedHouseVal"],
                                          test_size=0.2, random_state=0)
# clip the four artifact columns at the fixed caps of 5.7.2
caps = {"AveRooms": 12.0, "AveBedrms": 3.0, "Population": 6000.0, "AveOccup": 6.0}
clipper = FunctionTransformer(
    lambda Xd: Xd.assign(**{c: Xd[c].clip(upper=h) for c, h in caps.items()}))
lin = make_pipeline(clipper, StandardScaler(), LinearRegression()).fit(X_tr, y_tr)
coefs = lin.named_steps["linearregression"].coef_           # standardized -> directly comparable
for i in np.argsort(np.abs(coefs))[::-1]:
    print(f"{feats[i]:12} {coefs[i]:+.3f}")
# Latitude -0.928   Longitude -0.857   MedInc +0.806   AveOccup -0.264
# AveBedrms +0.153   HouseAge +0.141   AveRooms -0.129   Population +0.046
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import r2_score
print(round(float(X_tr["Latitude"].corr(X_tr["Longitude"])), 3))  # -0.925
Xc = sm.add_constant(clipper.transform(X_tr))
print(round(float(variance_inflation_factor(Xc, list(feats).index("Latitude") + 1)), 2),
      round(float(variance_inflation_factor(Xc, list(feats).index("Longitude") + 1)), 2))
# 9.89 9.45
print(f"{r2_score(y_te, lin.predict(X_te)):.3f}")  # 0.647
for drop in (["Latitude", "Longitude"], ["MedInc"]):
    keep = [c for c in feats if c not in drop]
    m = make_pipeline(clipper, StandardScaler(), LinearRegression()).fit(X_tr[keep], y_tr)
    print(drop, f"{r2_score(y_te, m.predict(X_te[keep])):.3f}")
# ['Latitude', 'Longitude'] 0.580
# ['MedInc'] 0.466


# %% [markdown]
# ## 5.8 Coding exercises

# %% [markdown]
# Example 5-19, which follows the fourteen exercises, is one starter for all of them: it loads each exercise's data and leaves the unfinished step as `_not_yet()`, which returns None. It runs as given and prints only a small check.
#
# Example 5-19. One starter for the fourteen coding exercises, with each exercise's data loaded and its unfinished step left as `_not_yet()`.

# %%
import numpy as np
from pathlib import Path
from sklearn.datasets import (fetch_california_housing, load_diabetes,
                              load_breast_cancer, load_wine)
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns
import pandas as pd

def _not_yet():
    return None

# 1. California housing, 2,000-row subsample
X, y = fetch_california_housing(return_X_y=True)
idx = np.random.default_rng(0).choice(len(X), 2000, replace=False)
print(X[idx].shape, _not_yet())
# (2000, 8) None

# 2. diabetes, ten features
d = load_diabetes()
print(d.data.shape, _not_yet())
# (442, 10) None

# 3. same 2,000-row income column
print(X[idx, 0].shape, _not_yet())
# (2000,) None

# 4. diabetes degree-2 expansion
print(PolynomialFeatures(2, include_bias=False).fit_transform(d.data).shape, _not_yet())
# (442, 65) None

# 5. two breast-cancer features
bc = load_breast_cancer()
print(bc.data[:, [0, 1]].shape, _not_yet())
# (569, 2) None

# 6. wine
wine = load_wine()
print(wine.data.shape, _not_yet())
# (178, 13) None

# 7. diabetes BMI
print(d.data[:, 2].shape, _not_yet())
# (442,) None

# 8. same 2,000 prices
print(y[idx].min(), _not_yet())
# 0.25 None

# 9-10. full California housing
print(X.shape, _not_yet())
# (20640, 8) None

# 11. diamonds
print(len(sns.load_dataset("diamonds")), _not_yet())
# 53940 None

# 12-13. daily bike counts
HERE = Path("code/ch17_time_series/data")         # run from the book folder
if not HERE.exists():                             # or from code/ch05_regression
    HERE = Path("../ch17_time_series/data")
print(len(pd.read_csv(HERE / "bike_sharing_daily.csv")), _not_yet())
# 731 None

# 14. diabetes BMI again
print(d.data[:, [2]].shape, _not_yet())
# (442, 1) None
