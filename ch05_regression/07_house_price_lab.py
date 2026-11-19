"""Chapter 5.7 house-price lab, regenerated from the working book slice.

Every python block in section 5.7 appears here in book order.
"""

# %% [markdown]
# ## 5.7 Hands-on lab: house price prediction end to end

# %% [markdown]
# ## 5.7.1 Load and explore

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

