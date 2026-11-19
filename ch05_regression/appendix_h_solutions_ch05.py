"""Appendix H solutions for Chapter 5, as runnable cells.

Generated from Solutions.md (Appendix H). Every code block of this chapter's solutions appears here
verbatim, split into one cell per exercise or drill topic, so each answer can be run and checked against
the expected outputs it states. Keep Solutions.md as the source; regenerate after any change.
"""

# %% [markdown]
# ## 5.8 coding exercises

# %%
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from sklearn.datasets import (fetch_california_housing, load_diabetes,
                              load_breast_cancer, load_wine)
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GroupKFold
from sklearn.preprocessing import (StandardScaler, PolynomialFeatures,
                                   SplineTransformer, FunctionTransformer)
from sklearn.linear_model import (LinearRegression, RidgeCV, LassoCV, ElasticNetCV,
                                  LogisticRegression, HuberRegressor, TheilSenRegressor,
                                  GammaRegressor, QuantileRegressor)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
import statsmodels.api as sm

rng = np.random.default_rng(0)

# %% [markdown]
# 1. Linear regression three ways on real California housing (they must agree)

# %%
# 1. Linear regression three ways on real California housing (they must agree)
X, y = fetch_california_housing(return_X_y=True)
idx = rng.choice(len(X), 2000, replace=False)            # subsample for speed
Xs, ys = StandardScaler().fit_transform(X[idx]), y[idx]
n = len(ys); Xa = np.column_stack([np.ones(n), Xs])
theta = np.linalg.inv(Xa.T @ Xa) @ Xa.T @ ys            # normal equation, closed form
w = np.zeros(Xa.shape[1])
for _ in range(3000):
    w -= 0.1 * (2 / n) * Xa.T @ (Xa @ w - ys)           # gradient descent
lr = LinearRegression().fit(Xs, ys)
print(np.allclose(theta, w, atol=1e-3),
      np.allclose(theta[1:], lr.coef_, atol=1e-3),
      np.allclose(theta[0], lr.intercept_))
# expected output: True True True
assert np.allclose(theta, w, atol=1e-3) and np.allclose(theta[1:], lr.coef_, atol=1e-3)
assert np.allclose(theta[0], lr.intercept_)      # all three agree, so both asserts pass silently

# %% [markdown]
# 2. Inference with statsmodels on the real diabetes dataset

# %%
# 2. Inference with statsmodels on the real diabetes dataset
d = load_diabetes()
Xd = sm.add_constant(StandardScaler().fit_transform(d.data))
m = sm.OLS(d.target, Xd).fit()
names = ["const"] + list(d.feature_names)
sig = [names[i] for i in range(1, len(names)) if m.pvalues[i] < 0.05]
print(sorted(sig), round(m.rsquared, 3), round(m.f_pvalue, 6))
# expected output: ['bmi', 'bp', 's5', 'sex'] 0.518 0.0
abs_order = np.argsort(np.abs(m.params[1:]))[::-1]
print([names[i + 1] for i in abs_order[:4]],
      [round(float(m.params[i + 1]), 1) for i in abs_order[:4]],
      [round(float(m.pvalues[i + 1]), 3) for i in abs_order[:4]])
# expected output: ['s1', 's5', 'bmi', 's2'] [-37.7, 35.7, 24.7, 22.7] [0.058, 0.0, 0.0, 0.16]
from statsmodels.stats.outliers_influence import variance_inflation_factor
print(round(float(variance_inflation_factor(Xd, names.index('s1'))), 1),
      round(float(variance_inflation_factor(Xd, names.index('s2'))), 1))
# expected output: 59.2 39.2
print({c: round(float(m.bse[names.index(c)]), 1) for c in ("s1", "s2", "bmi", "bp", "sex")})
# expected output: {'s1': 19.8, 's2': 16.1, 'bmi': 3.2, 'bp': 3.1, 'sex': 2.9}
# s1 and s2 hold two of the four largest coefficients (-37.7 and 22.7) yet miss
# the 0.05 cut (p = 0.058 and 0.16). Their VIFs of 59.2 and 39.2 multiply their
# standard errors by about sqrt(59.2) = 7.7 and sqrt(39.2) = 6.3 relative to an
# uncorrelated feature, to 19.8 and 16.1 against 3.2 for bmi. The two serum
# measures overlap (correlation 0.897, 5.3.1), so the data cannot say which one
# carries the effect. bp and sex join the significant set with smaller
# coefficients because their standard errors, 3.1 and 2.9, stay small.

# %% [markdown]
# 3. Choose polynomial degree by cross-validation (median income -> price)

# %%
# 3. Choose polynomial degree by cross-validation (median income -> price)
Xm = X[idx, 0:1]
cv_mse = {}
for deg in range(1, 7):
    mp = make_pipeline(PolynomialFeatures(deg), StandardScaler(), LinearRegression())
    cv_mse[deg] = -cross_val_score(mp, Xm, ys, cv=5,
                                   scoring="neg_mean_squared_error").mean()
print(min(cv_mse, key=cv_mse.get), round(cv_mse[1], 3), round(cv_mse[3], 3))
# expected output: 3 0.675 0.65
# degree 3 wins but only barely: income vs price is nearly linear

# %% [markdown]
# 4. Ridge/Lasso/ElasticNet on degree-2 expanded diabetes (65 features)

# %%
# 4. Ridge/Lasso/ElasticNet on degree-2 expanded diabetes (65 features)
Xp = PolynomialFeatures(2, include_bias=False).fit_transform(d.data)
Xtr, Xte, ytr, yte = train_test_split(Xp, d.target, test_size=0.3, random_state=0)
sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
for name, mm in [("OLS  ", LinearRegression()),
                 ("Ridge", RidgeCV(alphas=np.logspace(-2, 3, 30))),
                 ("Lasso", LassoCV(cv=5, max_iter=10000)),
                 ("ENet ", ElasticNetCV(l1_ratio=[.2, .5, .9], cv=5, max_iter=10000))]:
    mm.fit(Xtr, ytr)
    nz = int(np.sum(np.abs(np.ravel(mm.coef_)) > 1e-6))
    print(name, round(r2_score(yte, mm.predict(Xte)), 3), nz)
# expected output: OLS   0.225 65
# expected output: Ridge 0.375 65
# expected output: Lasso 0.373 14
# expected output: ENet  0.387 30
# OLS spends 65 free coefficients on 309 training rows (442 minus the 133 held
# out), so it fits noise and reaches only 0.225 on the held-out rows. Shrinkage
# trades a little bias for a large cut in variance, which lifts Ridge to 0.375
# with all 65 terms kept. Lasso's selection adds a far smaller model at almost
# the same score, 0.373 with 14 nonzero terms, a model you can read term by
# term. Elastic Net scores best, 0.387 with 30 terms.

# %% [markdown]
# 5. Logistic regression from scratch vs scikit-learn on breast cancer

# %%
# 5. Logistic regression from scratch vs scikit-learn on breast cancer
bc = load_breast_cancer()
Xbtr, Xbte, ybtr, ybte = train_test_split(bc.data[:, [0, 1]], bc.target,
                                          test_size=0.3, random_state=0)
sb = StandardScaler().fit(Xbtr)                          # fit the scaler on training rows only
Xbtr, Xbte = sb.transform(Xbtr), sb.transform(Xbte)
sigmoid = lambda z: 1 / (1 + np.exp(-z))
Xaug = np.column_stack([np.ones(len(Xbtr)), Xbtr]); wl = np.zeros(3)
for _ in range(5000):
    p = sigmoid(Xaug @ wl); wl -= 0.5 * (Xaug.T @ (p - ybtr)) / len(ybtr)
pred = (sigmoid(np.column_stack([np.ones(len(Xbte)), Xbte]) @ wl) >= 0.5).astype(int)
clf = LogisticRegression(C=1e6).fit(Xbtr, ybtr)
print(wl.round(3).tolist(), round(accuracy_score(ybte, pred), 3),
      np.allclose(wl, [clf.intercept_[0], *clf.coef_[0]], atol=1e-2))
# expected output: [0.585, -3.579, -0.883] 0.912 True
# the same weights as Example 5-5, which also fits its scaler on training rows only

# %% [markdown]
# 6. Multiclass softmax logistic regression on wine + odds ratios

# %%
# 6. Multiclass softmax logistic regression on wine + odds ratios
wine = load_wine()
Xwtr, Xwte, ywtr, ywte = train_test_split(wine.data, wine.target, test_size=0.3, random_state=0)
sw = StandardScaler().fit(Xwtr)                          # training rows only, as in exercise 4
Xwtr, Xwte = sw.transform(Xwtr), sw.transform(Xwte)
cw = LogisticRegression(max_iter=2000).fit(Xwtr, ywtr)
j = list(wine.feature_names).index("proline")
print(cw.coef_.shape, round(accuracy_score(ywte, cw.predict(Xwte)), 3),
      round(float(np.exp(cw.coef_[0, j] - cw.coef_[1, j])), 3),
      round(float(np.exp(cw.coef_[0, j] - cw.coef_[2, j])), 3))
# expected output: (3, 13) 1.0 7.567 2.757

# %% [markdown]
# 7. Robust regression on real data with high-leverage outliers

# %%
# 7. Robust regression on real data with high-leverage outliers
Xr = d.data[:, 2:3]; yr = d.target.astype(float)         # BMI feature
clean = LinearRegression().fit(Xr, yr).coef_[0]
order = np.argsort(Xr.ravel())
yc = yr.copy(); yc[order[:15]] += 600.0                  # push 15 lowest-BMI targets up
print(round(clean, 1),
      round(LinearRegression().fit(Xr, yc).coef_[0], 1),
      round(HuberRegressor().fit(Xr, yc).coef_[0], 1),
      round(TheilSenRegressor(random_state=0).fit(Xr, yc).coef_[0], 1))
# expected output: 949.4 230.7 926.6 817.0
# OLS collapses to 231; Huber (927) and Theil-Sen (817) hold near the clean 949

# %% [markdown]
# 8. Gamma GLM vs OLS on positive skewed prices + a quantile interval

# %%
# 8. Gamma GLM vs OLS on positive skewed prices + a quantile interval
Xgtr, Xgte, ygtr, ygte = train_test_split(X[idx], ys, test_size=0.3, random_state=0)
gam = make_pipeline(StandardScaler(), GammaRegressor(alpha=0.0)).fit(Xgtr, ygtr)
ols = make_pipeline(StandardScaler(), LinearRegression()).fit(Xgtr, ygtr)
lo = QuantileRegressor(quantile=0.1, alpha=0.0, solver="highs").fit(Xgtr, ygtr).predict(Xgte)
hi = QuantileRegressor(quantile=0.9, alpha=0.0, solver="highs").fit(Xgtr, ygtr).predict(Xgte)
print(round(ols.predict(Xgte).min(), 3), round(gam.predict(Xgte).min(), 3),
      round(float(((ygte >= lo) & (ygte <= hi)).mean()), 3))
# expected output: -1.018 0.377 0.78
# OLS predicts an impossible negative price; Gamma's log link stays positive;
# the 10th-90th band covers about 78% of held-out districts (target 80%)

# %% [markdown]
# 9. Additive spline GAM vs plain linear on all California districts, clipped

# %%
# 9. Additive spline GAM vs plain linear on all California districts, clipped
df9 = fetch_california_housing(as_frame=True).frame
feats9 = list(fetch_california_housing().feature_names)
caps9 = {"AveRooms": 12.0, "AveBedrms": 3.0, "Population": 6000.0, "AveOccup": 6.0}
Xh = df9[feats9].assign(**{c: df9[c].clip(upper=h) for c, h in caps9.items()})
yh = df9["MedHouseVal"]
Xhtr, Xhte, yhtr, yhte = train_test_split(Xh, yh, test_size=0.3, random_state=0)
cols = [(f"s{j}", SplineTransformer(n_knots=5, degree=3), [j]) for j in range(Xh.shape[1])]
gam9 = make_pipeline(ColumnTransformer(cols), LinearRegression()).fit(Xhtr, yhtr)
lin9 = make_pipeline(StandardScaler(), LinearRegression()).fit(Xhtr, yhtr)
print(round(r2_score(yhte, lin9.predict(Xhte)), 3),
      round(r2_score(yhte, gam9.predict(Xhte)), 3))
# expected output: 0.648 0.709
# after clipping, the additive splines lift R^2 from 0.648 to 0.709

# %% [markdown]
# 10. Transfer: spatial GroupKFold plus a conformalized quantile band

# %%
# 10. Transfer: spatial GroupKFold plus a conformalized quantile band
df = fetch_california_housing(as_frame=True).frame
feats = list(fetch_california_housing().feature_names)
Xtr2, Xte2, ytr2, yte2 = train_test_split(df[feats], df["MedHouseVal"],
                                          test_size=0.2, random_state=0)
caps = {"AveRooms": 12.0, "AveBedrms": 3.0, "Population": 6000.0, "AveOccup": 6.0}
clip = FunctionTransformer(lambda Xd: Xd.assign(**{c: Xd[c].clip(upper=h)
                                                   for c, h in caps.items()}))
groups = (np.floor(Xtr2["Latitude"]).astype(int).astype(str) + "_" +
          np.floor(Xtr2["Longitude"]).astype(int).astype(str))
gkf = GroupKFold(n_splits=5)
poly2 = make_pipeline(clip, StandardScaler(), PolynomialFeatures(2), LinearRegression())
print(round(-cross_val_score(poly2, Xtr2, ytr2, cv=gkf, groups=groups,
                             scoring="neg_root_mean_squared_error").mean(), 3))
# expected output: 0.649
Xf, Xc, yf, yc2 = train_test_split(Xtr2, ytr2, test_size=0.25, random_state=1)
cols = ["MedInc", "AveOccup"]
sc = StandardScaler().fit(clip.transform(Xf)[cols])
lo_m = QuantileRegressor(quantile=0.05, alpha=0.0, solver="highs").fit(
    sc.transform(clip.transform(Xf)[cols]), yf)
hi_m = QuantileRegressor(quantile=0.95, alpha=0.0, solver="highs").fit(
    sc.transform(clip.transform(Xf)[cols]), yf)
lo_c = lo_m.predict(sc.transform(clip.transform(Xc)[cols]))
hi_c = hi_m.predict(sc.transform(clip.transform(Xc)[cols]))
scores = np.maximum(lo_c - np.asarray(yc2), np.asarray(yc2) - hi_c)
n_cal = len(scores)
q = float(np.quantile(scores, np.ceil((n_cal + 1) * 0.90) / n_cal))
lo_te = lo_m.predict(sc.transform(clip.transform(Xte2)[cols]))
hi_te = hi_m.predict(sc.transform(clip.transform(Xte2)[cols]))
inside = (np.asarray(yte2) >= lo_te - q) & (np.asarray(yte2) <= hi_te + q)
width = float(np.mean((hi_te + q) - (lo_te - q)))
print(round(q, 3), round(float(inside.mean()), 3), round(width, 3))
# expected output: -0.027 0.886 2.453
# grouped CV RMSE 0.649; CQR q is slightly negative because the raw 5th-95th
# band already overcovers the calibration slice; mean width 2.453 vs the lab's
# fixed width 1.822 (= 2 * 0.911); coverage 88.6 percent

# %% [markdown]
# 11. WLS on diamonds

# %%
# 11. WLS on diamonds
import seaborn as sns
dia = sns.load_dataset("diamonds")
y = dia["price"].to_numpy(float)
x = dia["carat"].to_numpy(float)
X = sm.add_constant(x)
ols = sm.OLS(y, X).fit()
print(np.round(ols.params, 2).tolist(), np.round(ols.bse, 2).tolist())
print(round(float(np.corrcoef(x, ols.resid ** 2)[0, 1]), 3))
aux = sm.OLS(np.log(ols.resid ** 2 + 1e-12), sm.add_constant(x)).fit()
wls = sm.WLS(y, X, weights=1.0 / np.exp(aux.fittedvalues)).fit()
print(np.round(wls.params, 2).tolist(), np.round(wls.bse, 2).tolist())
wr = wls.wresid                                  # weighted residuals, sqrt(weight) * residual
lo, hi = x <= np.quantile(x, 0.1), x >= np.quantile(x, 0.9)   # lightest, heaviest tenth
print(round(float(np.corrcoef(x, wr ** 2)[0, 1]), 3),
      round(float(ols.resid[hi].std() / ols.resid[lo].std()), 1),
      round(float(wr[hi].std() / wr[lo].std()), 2))
# expected output: [-2256.36, 7756.43] [13.06, 14.07]
# expected output: 0.411
# expected output: [-1536.18, 6548.57] [7.03, 12.9]
# expected output: 0.174 15.3 2.63
# The OLS standard errors, 13.06 and 14.07, assume one constant variance,
# which the 0.411 correlation contradicts (HC3 in 5.1.5 widened the slope's
# to 25.41). WLS reports 7.03 and 12.9 because the quiet light stones now
# pull harder. Its weighted residuals spread far more evenly: their squares
# correlate 0.174 with carat instead of 0.411, and on the heaviest tenth of
# stones they spread 2.63 times as wide as on the lightest tenth, against
# 15.3 times for the OLS residuals. The fan shrinks but does not vanish, so
# the estimated variance function is only partly right.
# WLS is preferable when you want to stay on the original price scale and the
# variance function can be estimated; a log transform changes what the
# coefficients mean.

# %% [markdown]
# 12. DW, JB, Cook, VIF on daily bike counts

# %%
# 12. DW, JB, Cook, VIF on daily bike counts
import pandas as pd
from pathlib import Path
from statsmodels.stats.outliers_influence import variance_inflation_factor
DATA = Path("code/ch17_time_series/data")                # relative to the book folder
if not DATA.exists():                                    # run from code/ch05_regression instead
    DATA = Path.cwd().resolve().parents[1] / DATA
bike = pd.read_csv(DATA / "bike_sharing_daily.csv")
yb = bike["cnt"].to_numpy(float)
Xb = sm.add_constant(bike[["temp", "hum", "windspeed", "workingday"]].to_numpy(float))
m = sm.OLS(yb, Xb).fit()
print(round(float(sm.stats.durbin_watson(m.resid)), 3))
print(round(float(sm.stats.jarque_bera(m.resid)[1]), 4))
cook = m.get_influence().cooks_distance[0]
top = np.argsort(cook)[-3:][::-1]
print(bike.loc[top, "dteday"].tolist(), np.round(cook[top], 3).tolist())
print([round(float(variance_inflation_factor(Xb, i)), 2) for i in range(1, Xb.shape[1])])
# expected output: 0.404
# expected output: 0.0196
# expected output: ['2011-03-10', '2011-07-23', '2011-08-27'] [0.07, 0.022, 0.019]
# expected output: [1.04, 1.08, 1.08, 1.0]
# DW 0.404 points at Newey-West standard errors or a time-series model
# (Chapter 17). JB p = 0.0196 is below 0.05, so normality of the residuals is
# rejected at the 5 percent level. Once the autocorrelation is handled, 731
# days make a large sample, in which the t and F tests hold approximately
# without Gaussian errors (5.1.3); a transform of cnt (5.5.4) or a count GLM
# like exercise 13's addresses the residual shape itself. Cook values are all
# far below the flag of 1 (5.1.5), so no single day dominates the fit, and
# feature VIFs near 1 show no collinearity, so no feature needs dropping.

# %% [markdown]
# 13. PoissonRegressor vs OLS on the last 20 percent of days

# %%
# 13. PoissonRegressor vs OLS on the last 20 percent of days
from sklearn.linear_model import PoissonRegressor, LinearRegression
from sklearn.metrics import mean_poisson_deviance
n = len(bike); cut = int(0.8 * n)
X = bike[["temp", "hum", "windspeed", "workingday"]].to_numpy(float)
y = bike["cnt"].to_numpy(float)
Xtr, Xte, ytr, yte = X[:cut], X[cut:], y[:cut], y[cut:]
poi = PoissonRegressor(alpha=0.0, max_iter=1000).fit(Xtr, ytr)
ols = LinearRegression().fit(Xtr, ytr)
pp, po = poi.predict(Xte), ols.predict(Xte)
print(round(float(mean_poisson_deviance(yte, np.clip(pp, 1e-12, None))), 1),
      round(float(mean_poisson_deviance(yte, np.clip(po, 1e-12, None))), 1))
print(round(float(pp.min()), 1), round(float(po.min()), 1))
print(round(float(poi.coef_[0]), 4), round(float(np.exp(poi.coef_[0])), 4),
      round(float(np.exp(0.1 * poi.coef_[0])), 3))       # per unit and per 0.1 of temp
print(round(float(X[:, 0].min()), 3), round(float(X[:, 0].max()), 3))   # range of temp
# expected output: 1156.0 1048.6
# expected output: 2098.5 1472.4
# expected output: 1.5799 4.8547 1.171
# expected output: 0.059 0.862
# OLS had the lower Poisson deviance on this overdispersed holdout; Poisson
# still cannot go negative. temp is the normalized temperature of the UCI
# variables table (Fanaee-T, 2013), a 0 to 1 rescaling of degrees Celsius, and
# it spans only 0.059 to 0.862 here. A unit rise, which multiplies expected
# rentals by 4.85, is wider than that whole range; a 0.1 rise multiplies them
# by 1.171, about 17 percent more rentals.

# %% [markdown]
# 14. Mean interval vs prediction interval on diabetes BMI

# %%
# 14. Mean interval vs prediction interval on diabetes BMI
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
d = load_diabetes()
Xtr, Xte, ytr, yte = train_test_split(d.data[:, 2], d.target, test_size=0.3, random_state=0)
mod = sm.OLS(ytr, sm.add_constant(Xtr)).fit()
qs = np.quantile(Xtr, [0.1, 0.5, 0.9])
sf = mod.get_prediction(sm.add_constant(qs)).summary_frame()
print(np.round(qs, 4).tolist())
print(np.round(sf[["mean_ci_lower", "mean_ci_upper"]].to_numpy(), 2).tolist())
print(np.round(sf[["obs_ci_lower", "obs_ci_upper"]].to_numpy(), 2).tolist())
pte = mod.get_prediction(sm.add_constant(Xte)).summary_frame()
print(round(float(np.mean((yte >= pte["mean_ci_lower"]) & (yte <= pte["mean_ci_upper"]))), 3),
      round(float(np.mean((yte >= pte["obs_ci_lower"]) & (yte <= pte["obs_ci_upper"]))), 3))
# expected output: [-0.0601, -0.0084, 0.0682]
# expected output: [[81.51, 103.59], [137.88, 152.04], [210.22, 234.77]]
# expected output: [[-31.12, 216.21], [21.59, 268.34], [98.72, 346.27]]
# expected output: 0.068 0.955
# The prediction interval is wider because it covers one new draw of the noise.
# On the holdout, 6.8 percent of targets fall in the mean interval and 95.5
# percent in the prediction interval, as a 95 percent prediction interval should.

# %% [markdown]
# ## 5.8 coding drills

# %%
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, SplineTransformer
from sklearn.linear_model import Lasso, RidgeCV

# %% [markdown]
# Linear regression

# %%
# Linear regression
X, y = load_diabetes(return_X_y=True)
Xone = np.column_stack([np.ones(len(X)), X[:, 2]])                # intercept + BMI
b = np.linalg.solve(Xone.T @ Xone, Xone.T @ y)
print(b.round(2).tolist())                                       # normal-equation fit
# expected output: [152.13, 949.44]
pred = Xone @ b
r2 = 1 - ((y - pred)**2).sum() / ((y - y.mean())**2).sum()       # R^2 by hand
print(round(r2, 3))
# expected output: 0.344
Xg = np.column_stack([np.ones(5), np.arange(5)]); yg = np.array([1., 3., 2., 5., 4.])
print(((2/5) * Xg.T @ (Xg @ np.zeros(2) - yg)).round(2).tolist())  # MSE gradient at w=0
# expected output: [-6.0, -15.2]

# %% [markdown]
# Regularization

# %%
# Regularization
Xz = StandardScaler().fit_transform(X)
print(int(np.sum(np.abs(Lasso(alpha=1.0).fit(Xz, y).coef_) > 1e-6)))   # nonzero Lasso coefs
# expected output: 7
print(round(float(RidgeCV(alphas=np.logspace(-3, 3, 50)).fit(Xz, y).alpha_), 3))  # CV-picked alpha
# expected output: 2.024
raw = load_diabetes().data[:, 2] * 1000
print(round(float(raw.std()), 2), round(float(StandardScaler().fit_transform(raw.reshape(-1, 1)).std()), 2))
# expected output: 47.57 1.0

# %% [markdown]
# Logistic regression

# %%
# Logistic regression
sigmoid = lambda z: 1 / (1 + np.exp(-z))
print([round(float(sigmoid(v)), 3) for v in (-2, 0, 2)])       # squashes score to (0,1)
# expected output: [0.119, 0.5, 0.881]
yt, pp = np.array([1, 0, 1]), np.array([0.9, 0.2, 0.7])
print(round(float(-np.mean(yt*np.log(pp) + (1-yt)*np.log(1-pp))), 3))   # cross-entropy
# expected output: 0.228
print(round(float(np.exp(0.7)), 3))                            # odds ratio for coef 0.7
# expected output: 2.014
z = np.array([2.0, 1.0, 0.1]); print((np.exp(z)/np.exp(z).sum()).round(3).tolist())  # softmax
# expected output: [0.659, 0.242, 0.099]
print(int(sigmoid(0.3) >= 0.5), int(sigmoid(-0.3) >= 0.5))     # threshold at 0.5
# expected output: 1 0

# %% [markdown]
# Robust regression and GLMs

# %%
# Robust regression and GLMs
xs, ys2 = np.array([0., 1., 2., 3., 4.]), np.array([0., 2., 4., 6., 100.])
slopes = [(ys2[j]-ys2[i])/(xs[j]-xs[i]) for i in range(5) for j in range(i+1, 5)]
print(round(float(np.median(slopes)), 2))                      # Theil-Sen: median pairwise slope
# expected output: 2.0
print(round(float(np.exp(0.5 + 0.8*2)), 3))                    # Poisson log-link rate, always > 0
# expected output: 8.166
pinball = lambda yt, yp, q: float(np.mean(np.maximum(q*(yt-yp), (q-1)*(yt-yp))))
print(round(pinball(np.array([3.]), np.array([2.]), 0.9), 3),
      round(pinball(np.array([1.]), np.array([2.]), 0.9), 3))  # underprediction vs overprediction at q=0.9
# expected output: 0.9 0.1

# %% [markdown]
# Splines and nonlinearity

# %%
# Splines and nonlinearity
print(PolynomialFeatures(3).fit_transform([[2.0]])[0].tolist())   # x -> 1, x, x^2, x^3
# expected output: [1.0, 2.0, 4.0, 8.0]
st = SplineTransformer(n_knots=6, degree=3).fit(np.linspace(0, 6, 50).reshape(-1, 1))
basis = st.transform([[2.0]])[0]
print(int(np.sum(basis > 1e-3)), basis.shape[0])               # local support: few active bases
# expected output: 4 8
print(PolynomialFeatures(2, interaction_only=True, include_bias=False).fit_transform([[3.0, 5.0]])[0].tolist())
# expected output: [3.0, 5.0, 15.0]
