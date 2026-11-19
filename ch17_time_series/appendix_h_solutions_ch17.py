"""Appendix H solutions for Chapter 17, as runnable cells.

Generated from Solutions.md (Appendix H). Every code block of this chapter's solutions appears here
verbatim, split into one cell per exercise or drill topic, so each answer can be run and checked against
the expected outputs it states. Keep Solutions.md as the source; regenerate after any change.
"""

# %% [markdown]
# ## 17.9 lab tasks

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
import warnings; warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
# 1.
import pandas as pd
bike = pd.read_csv("data/bike_sharing_daily.csv", parse_dates=["dteday"])
print(len(bike), bike.loc[bike["cnt"].idxmin(), "dteday"].date(), int(bike["cnt"].min()))
  # expected output: 731 2012-10-29 22

# %% [markdown]
# ## 17.10 coding exercises

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
import warnings; warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, seaborn as sns

def rmse(a, b): return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))
def mae(a, b):  return float(np.mean(np.abs(np.asarray(a, float) - np.asarray(b, float))))

flights = sns.load_dataset("flights")
y = pd.Series(flights["passengers"].to_numpy(float),
              index=pd.date_range("1949-01-01", periods=len(flights), freq="MS"), name="passengers")
yv = y.to_numpy(); n = len(yv); t = np.arange(n)
bike = pd.read_csv("data/bike_sharing_daily.csv", parse_dates=["dteday"]).set_index("dteday")
b = bike["cnt"].astype(float); bv = b.to_numpy(); nb = len(bv); tb = np.arange(nb)

# %% [markdown]
# 1. Autocorrelation and the time-ordered split, on daily data

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
import warnings; warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
# 1. Autocorrelation and the time-ordered split, on daily data
def acf(x, lag):
    x = x - x.mean()
    return float(np.sum(x[lag:] * x[:len(x) - lag]) / np.sum(x * x))
sl, ic = np.polyfit(tb, bv, 1); bdet = bv - (sl * tb + ic)
print([round(acf(bv, k), 2) for k in (1, 7, 14)])  # expected output: [0.85, 0.74, 0.71]
print([round(acf(bdet, k), 2) for k in (1, 7, 14)])  # expected output: [0.75, 0.55, 0.5]
def interp_pred(tt, yy, q):
    o = np.argsort(tt); return np.interp(q, tt[o], yy[o])
rng = np.random.default_rng(0); idx = rng.permutation(nb)
rte, rtr = np.sort(idx[:int(0.3 * nb)]), np.sort(idx[int(0.3 * nb):])
cut = int(0.7 * nb)
print(round(rmse(interp_pred(rtr, bv[rtr], rte), bv[rte]), 2),
      round(rmse(interp_pred(tb[:cut], bv[:cut], tb[cut:]), bv[cut:]), 2))  # expected output: 1032.96 1750.33
# Lag 7 sits below lag 1: the yearly swing and the trend dominate the daily series, and the weekly rhythm is
# secondary. The time split's RMSE is 1.7 times the random split's, a smaller ratio than the airline series'
# 4.5 (105.75 against 23.68), but the random split still flatters the interpolator.

# %% [markdown]
# 2. Stationarity with a weekly season (ADF + KPSS)

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
import warnings; warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
# 2. Stationarity with a weekly season (ADF + KPSS)
from statsmodels.tsa.stattools import adfuller, kpss
print(round(bv[:365].mean(), 1), round(bv[365:].mean(), 1))  # expected output: 3405.8 5599.9
def adfp(x): return round(adfuller(np.asarray(x), autolag="AIC")[1], 3)
d1, w7 = np.diff(bv), bv[7:] - bv[:-7]; both = np.diff(w7)
print(adfp(bv), adfp(d1), adfp(w7), adfp(both))  # expected output: 0.343 0.0 0.0 0.0
from statsmodels.tools.sm_exceptions import InterpolationWarning
warnings.simplefilter("ignore", InterpolationWarning)   # the KPSS caps are explained below
def kpssp(x): return round(kpss(np.asarray(x), regression="c", nlags="auto")[1], 3)
print(kpssp(bv), kpssp(both))  # expected output: 0.01 0.1
# Unlike the airline series, one first difference already satisfies ADF here (p = 0.000), and the weekly
# difference alone does too; KPSS rejects stationarity on the raw counts and not on the double difference.
# Both KPSS values are caps, as in Example 17-2: statsmodels interpolates the p-value from the table of
# Kwiatkowski et al. (1992) and returns the table's edge, 0.01 or 0.1, when the statistic falls outside it
# (statsmodels developers, 2026, kpss). So the true p is below 0.01 on the raw counts and above 0.10 on the
# double difference; the filter above keeps the InterpolationWarning that reports this out of the output.

# %% [markdown]
# 3. STL with a weekly period, and a contextual anomaly

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
import warnings; warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
# 3. STL with a weekly period, and a contextual anomaly
from statsmodels.tsa.seasonal import STL
res = STL(b, period=7, robust=True).fit()
print(f"{np.abs(b - (res.trend + res.seasonal + res.resid)).max():.2e}")  # expected output: 9.09e-13
print(round(res.resid.iloc[:365].std(), 1), round(res.resid.iloc[365:].std(), 1))  # expected output: 665.0 985.1
b2 = b.copy(); jday = b2.index.get_loc(pd.Timestamp("2012-01-18"))
b2.iloc[jday] = float(b.loc["2012-07"].median())                 # a July level on a January day
r2 = STL(b2, period=7, robust=True).fit().resid.values
print(round(b2.iloc[jday], 1), round(float((b2.iloc[jday] - b2.mean()) / b2.std()), 2),
      round(float((r2[jday] - r2.mean()) / r2.std()), 2))  # expected output: 6685.0 1.12 4.29
# The residual spread grows with the level from the first year to the second, a multiplicative sign. The planted
# day is unremarkable in raw terms (z = 1.12) but 4.29 residual standard deviations out once the trend and the
# weekly season are removed.

# %% [markdown]
# 4. The exponential smoothing family on a 36-month holdout

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
import warnings; warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
# 4. The exponential smoothing family on a 36-month holdout
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing
tr4, te4 = y.iloc[:-36], y.iloc[-36:]
# statsmodels 0.15.0 raises "TypeError: initialization_method must be a string" for SimpleExpSmoothing and
# Holt without an initialization method, so all four fits name "estimated", the default the documentation
# of these classes states (statsmodels developers, 2026, ExponentialSmoothing, Holt, and SimpleExpSmoothing).
init = "estimated"
fits4 = {"SES": SimpleExpSmoothing(tr4, initialization_method=init).fit(),
         "Holt": Holt(tr4, initialization_method=init).fit(),
         "HW add": ExponentialSmoothing(tr4, trend="add", seasonal="add", seasonal_periods=12,
                                        initialization_method=init).fit(),
         "HW mul": ExponentialSmoothing(tr4, trend="add", seasonal="mul", seasonal_periods=12,
                                        initialization_method=init).fit()}
print({k: round(rmse(te4, m4.forecast(36)), 2) for k, m4 in fits4.items()})
  # expected output: {'SES': 121.14, 'Holt': 87.73, 'HW add': 26.38, 'HW mul': 27.37}
# Both Holt-Winters forms beat Holt and SES by far, as in Example 17-5, but here the additive form edges the
# multiplicative one, 26.38 against 27.37, the reverse of the 24-month split. One split cannot settle a gap
# that small; a backtest over several origins gives the fairer comparison.

# %% [markdown]
# 5. The airline model on a 12-month holdout, and a Ljung-Box check

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
import warnings; warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
# 5. The airline model on a 12-month holdout, and a Ljung-Box check
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
tr5, te5 = y.iloc[:-12], y.iloc[-12:]
ar5 = SARIMAX(tr5, order=(1, 1, 1)).fit(disp=False)
air = SARIMAX(tr5, order=(0, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False)
airlog = SARIMAX(np.log(tr5), order=(0, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False)
print(round(rmse(te5, tr5.iloc[-12:].to_numpy()), 2), round(rmse(te5, ar5.forecast(12)), 2),
      round(rmse(te5, air.forecast(12)), 2), round(rmse(te5, np.exp(airlog.forecast(12))), 2))
  # expected output: 50.71 91.23 21.09 18.6
print(ar5.mle_retvals["converged"], air.mle_retvals["converged"], airlog.mle_retvals["converged"])
  # expected output: True True True
p5 = acorr_ljungbox(airlog.resid[13:], lags=[24], model_df=2, return_df=True)["lb_pvalue"].iloc[0]
print(f"{p5:.3f}")  # expected output: 0.670
# All three fits report convergence. The log airline model wins, and with p = 0.670 on 24 - 2 = 22 degrees
# of freedom its residuals show no evidence of leftover autocorrelation.

# %% [markdown]
# 6. Error metrics, MASE, and a rolling-window backtest

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
import warnings; warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
# 6. Error metrics, MASE, and a rolling-window backtest
a6, f6 = yv[-12:], yv[-24:-12]                          # 1960 actuals against 1959 repeated
e6 = f6 - a6
scale6 = np.mean(np.abs(yv[12:-12] - yv[:-24]))          # in-sample seasonal-naive MAE over 1949-1959
print(round(np.mean(np.abs(e6)), 2), round(np.sqrt(np.mean(e6 ** 2)), 2),
      round(np.mean(np.abs(e6) / np.abs(a6)) * 100, 1),
      round(np.mean(2 * np.abs(e6) / (np.abs(a6) + np.abs(f6))) * 100, 1), round(np.mean(np.abs(e6)) / scale6, 2))
  # expected output: 47.83 50.71 10.0 10.6 1.57
rs = []
for c6 in [96, 108, 120, 132]:
    m6 = ExponentialSmoothing(y.iloc[c6 - 72:c6], trend="add", seasonal="mul", seasonal_periods=12).fit()
    rs.append(rmse(m6.forecast(12), y.iloc[c6:c6 + 12].values))
rs = np.array(rs)
print(np.round(rs, 2), round(rs.mean(), 2), round(rs.std(), 2))  # expected output: [16.5  20.11 35.42 14.31] 21.58 8.25
# MASE 1.57 says this forecast's error is larger than the average in-sample seasonal-naive error: the changes
# from 1959 to 1960, at the series' highest level, are larger in passengers than the average year-over-year
# change. The 72-month rolling window averages 21.58 with a spread of 8.25, worse and less stable than the
# expanding window's 18.80 and 2.35, so on this series keeping the older data helped.

# %% [markdown]
# 7. Machine-learning forecasting with lag features, and a leakage demo

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
import warnings; warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
# 7. Machine-learning forecasting with lag features, and a leakage demo
from sklearn.ensemble import GradientBoostingRegressor
df = pd.DataFrame({"y": y})
df["lag1"] = df["y"].shift(1); df["lag12"] = df["y"].shift(12)
df["roll3"] = df["y"].shift(1).rolling(3).mean()        # shift BEFORE roll => no peeking
df["month"] = df.index.month - 1
df = df.dropna(); feats = ["lag1", "lag12", "roll3", "month"]; c = len(df) - 24
gb = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=0)
gb.fit(df[feats].iloc[:c], df["y"].iloc[:c])
snaive = [df["y"].iloc[c - 12 + (i % 12)] for i in range(24)]
print(round(rmse(df["y"].iloc[c:], gb.predict(df[feats].iloc[c:])), 2), round(rmse(df["y"].iloc[c:], snaive), 2))
  # expected output: 51.31 76.99
print({k: round(float(v), 2) for k, v in zip(feats, gb.feature_importances_)})
  # expected output: {'lag1': 0.1, 'lag12': 0.87, 'roll3': 0.02, 'month': 0.01}
df2 = df.copy(); df2["leaky"] = y.rolling(3, center=True).mean().reindex(df2.index); df2 = df2.dropna(); c2 = len(df2) - 24
hon = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=0).fit(df2[feats].iloc[:c2], df2["y"].iloc[:c2])
lk = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=0).fit(df2[feats + ["leaky"]].iloc[:c2], df2["y"].iloc[:c2])
print(round(rmse(df2["y"].iloc[:c2], hon.predict(df2[feats].iloc[:c2])), 2),
      round(rmse(df2["y"].iloc[:c2], lk.predict(df2[feats + ["leaky"]].iloc[:c2])), 2))
  # expected output: 2.06 1.45
# Gradient boosting's 51.31 clears the seasonal-naive bar of 76.99, and lag12 carries 0.87 of the importance.
# The centered mean lowers the in-sample RMSE from 2.06 to 1.45, a gain that cannot survive in production,
# because the centered window needs the current month and the next one, which a live forecast does not have.

# %% [markdown]
# 8. A Prophet-style additive model on daily data: weekly and yearly Fourier seasons

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
import warnings; warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
# 8. A Prophet-style additive model on daily data: weekly and yearly Fourier seasons
def design(tt, Kw, Ky=4):
    cols = [np.ones_like(tt, float), tt.astype(float)]
    for period, K in ((7, Kw), (365.25, Ky)):
        for k in range(1, K + 1):
            cols += [np.sin(2 * np.pi * k * tt / period), np.cos(2 * np.pi * k * tt / period)]
    return np.column_stack(cols)
tr8, te8, ttr8, tte8 = bv[:-28], bv[-28:], tb[:-28], tb[-28:]
for Kw in (1, 2, 3, 4):
    X8 = design(ttr8, Kw); c8, *_ = np.linalg.lstsq(X8, tr8, rcond=None)
    print(Kw, X8.shape[1], np.linalg.matrix_rank(X8), round(rmse(tr8, X8 @ c8), 2))
# expected output: 1 12 12 947.96
# expected output: 2 14 14 946.76
# expected output: 3 16 16 946.34
# expected output: 4 18 16 946.34
c3, *_ = np.linalg.lstsq(design(ttr8, 3), tr8, rcond=None)
clog8, *_ = np.linalg.lstsq(design(ttr8, 3), np.log(tr8), rcond=None)
snaive8 = np.array([tr8[-7 + (i % 7)] for i in range(28)])
print(round(rmse(te8, design(tte8, 3) @ c3), 2), round(rmse(te8, np.exp(design(tte8, 3) @ clog8)), 2),
      round(rmse(te8, snaive8), 2))  # expected output: 2208.47 1735.84 2370.88
# For period 7, harmonics k and 7 - k coincide on whole days, so K_w = 4 repeats K_w = 3 and adds no rank. The
# weekly terms barely move the in-sample fit, because the yearly cycle and the large day-to-day swings dominate
# this series. Both fits beat the seasonal-naive forecast on the final four weeks, and the log fit does best.

# %% [markdown]
# 9. Quantiles at other levels

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
import warnings; warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
# 9. Quantiles at other levels
def pinball(yy, q, tau):
    ee = np.asarray(yy) - np.asarray(q); return float(np.mean(np.maximum(tau * ee, (tau - 1) * ee)))
sample = np.random.default_rng(0).exponential(10, 5000); grid = np.linspace(0, 40, 4001)
print(round(grid[np.argmin([pinball(sample, q, 0.75) for q in grid])], 2), round(np.quantile(sample, 0.75), 2))
  # expected output: 13.87 13.87
from sklearn.ensemble import GradientBoostingRegressor
def supervised(vals):
    d = pd.DataFrame({"y": np.asarray(vals, float)}); tt = np.arange(len(d))
    d["lag1"], d["lag12"], d["month"] = d["y"].shift(1), d["y"].shift(12), tt % 12
    return d.dropna().reset_index(drop=True)
featq = ["lag1", "lag12", "month"]
def gbq(alpha, X, yy):
    return GradientBoostingRegressor(loss="quantile", alpha=alpha, n_estimators=300, max_depth=2,
                                     learning_rate=0.05, random_state=0).fit(X, yy)
def coverage(d, lo_tau=0.1, hi_tau=0.9):
    cc = len(d) - 24; Xtr, ytr = d[featq].iloc[:cc], d["y"].iloc[:cc]
    Xte, yte = d[featq].iloc[cc:], d["y"].iloc[cc:].values
    lo, hi = gbq(lo_tau, Xtr, ytr).predict(Xte), gbq(hi_tau, Xtr, ytr).predict(Xte)
    return int(round(np.mean((yte >= lo) & (yte <= hi)) * 100))
print(coverage(supervised(yv)), coverage(supervised(np.r_[np.nan, np.diff(np.log(yv))])))  # expected output: 33 62
# Against a nominal 80 percent, the raw interval covers 25 percent because the trees cannot extrapolate the
# trend; the log differences lift it to 62 percent, still below nominal on these 24 months.

# %% [markdown]
# 10. Miniproject: three-way forecast + expanding-window backtest (RMSE and MASE)

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
import warnings; warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
# 10. Miniproject: three-way forecast + expanding-window backtest (RMSE and MASE)
def f_naive(tr, hh): return np.array([tr.iloc[-12 + (i % 12)] for i in range(hh)])
conv10 = []                                          # did each log SARIMA fit converge?
def f_sarima(tr, hh):
    m = SARIMAX(np.log(np.asarray(tr, float)), order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
    conv10.append(bool(m.mle_retvals["converged"]))
    return np.exp(np.asarray(m.forecast(hh)))
def fourier(tt, period, K):                         # intercept, trend, K harmonic pairs
    cols = [np.ones_like(tt, float), tt.astype(float)]
    for k in range(1, K + 1):
        cols += [np.sin(2 * np.pi * k * tt / period), np.cos(2 * np.pi * k * tt / period)]
    return np.column_stack(cols)
def f_prophet(tr, hh):
    arr = np.log(np.asarray(tr, float)); tt = np.arange(len(arr))
    cf = np.linalg.lstsq(fourier(tt, 12, 4), arr, rcond=None)[0]
    return np.exp(fourier(np.arange(len(arr), len(arr) + hh), 12, 4) @ cf)
def f_gb(tr, hh):
    d = supervised(np.asarray(tr, float))
    mdl = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0).fit(d[featq], d["y"])
    hist, out = list(np.asarray(tr, float)), []
    for _ in range(hh):
        p = len(hist); pred = float(mdl.predict(np.array([[hist[-1], hist[-12], p % 12]]))[0])
        out.append(pred); hist.append(pred)
    return np.array(out)
def mase_scale(arr, m=12): arr = np.asarray(arr, float); return float(np.mean(np.abs(arr[m:] - arr[:-m])))
names = ["seasonal-naive", "SARIMA(log)", "Prophet-style", "GBoost"]; H = 12
R = {k: [] for k in names}; M = {k: [] for k in names}
for cut in [96, 108, 120, 132]:
    trn, act = y.iloc[:cut], y.iloc[cut:cut + H].values; sc = mase_scale(trn.values, 12)
    preds = {"seasonal-naive": f_naive(trn, H), "SARIMA(log)": f_sarima(trn, H),
             "Prophet-style": f_prophet(trn, H), "GBoost": f_gb(trn, H)}
    for k, p in preds.items():
        R[k].append(rmse(act, p)); M[k].append(mae(act, p) / sc)
for k in names:
    print(f"{k:14s} RMSE {np.mean(R[k]):6.2f}  MASE {np.mean(M[k]):.2f}")
# expected output: seasonal-naive RMSE  39.61  MASE 1.25
# expected output: SARIMA(log)    RMSE  22.05  MASE 0.65
# expected output: Prophet-style  RMSE  35.36  MASE 1.06
# expected output: GBoost         RMSE  36.19  MASE 0.98
print("all four log SARIMA fits converged:", all(conv10))  # expected output: all four log SARIMA fits converged: False
# All three contenders beat seasonal-naive on both metrics. Prophet-style and GBoost swap places: Prophet-style
# wins on RMSE (35.36 against 36.19), GBoost on MASE (0.98 against 1.06). RMSE squares each error, so a few
# large misses dominate it, while MASE averages absolute errors and divides each fold by its own in-sample
# scale. A MASE above 1 can still beat naive: Prophet-style's 1.06 is below naive's 1.25 on the same folds,
# because the scale is the in-sample seasonal-naive error, not the naive forecast's error on these folds.

# %% [markdown]
# 11. Zero-shot forecasting on the lab's backtest

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
import warnings; warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
# 11. Zero-shot forecasting on the lab's backtest
import torch
from chronos import BaseChronosPipeline
pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-tiny", device_map="cpu", torch_dtype=torch.float32)
zs, sn = [], []
for cut11 in [nb - 4 * 14, nb - 3 * 14, nb - 2 * 14, nb - 14]:
    ctx, act = bv[:cut11], bv[cut11:cut11 + 14]
    scale11 = float(np.mean(np.abs(ctx[7:] - ctx[:-7])))    # the lab's MASE scale
    q11, _ = pipe.predict_quantiles(torch.tensor(ctx, dtype=torch.float32), prediction_length=14,
                                    quantile_levels=[0.5])
    zs.append(mae(act, q11[0, :, 0].numpy()) / scale11)
    sn.append(mae(act, np.array([ctx[-7 + (i % 7)] for i in range(14)])) / scale11)
print(np.round(zs, 2), round(float(np.mean(zs)), 2), np.round(sn, 2), round(float(np.mean(sn)), 2))
  # expected output: [0.74 1.34 0.53 2.81] 1.36 [1.2  1.15 0.87 2.76] 1.49
# With no fitting, the zero-shot median averages a MASE of 1.36: better than seasonal-naive (1.49) and the lab's
# LSTM (1.52), level with SARIMA (1.34), and a little behind XGBoost (1.29). Its worst fold is the hard
# year-end fold, 2.81.

# %% [markdown]
# ## 17.10 coding drills

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
import warnings; warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL

flights = sns.load_dataset("flights")
y = pd.Series(flights["passengers"].to_numpy(float),
              index=pd.date_range("1949-01-01", periods=len(flights), freq="MS"))
yv = y.to_numpy(); n = len(yv)

def rmse(a, b): return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))

# %% [markdown]
# Autocorrelation and time-ordered splits

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
import warnings; warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
# Autocorrelation and time-ordered splits
def acf(x, lag):
    x = x - x.mean(); return float(np.sum(x[lag:] * x[:len(x) - lag]) / np.sum(x * x))
print(round(acf(yv, 1), 2))  # expected output: 0.95
print(int(0.7 * n), n - int(0.7 * n))  # expected output: 100 44
# np.interp holds flat past the last training point: it clamps to the final known value, so a time split must extrapolate.

# %% [markdown]
# Stationarity and differencing

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
import warnings; warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
# Stationarity and differencing
print(len(np.diff(yv)))  # expected output: 143
print(round((yv[12] - yv[0]), 1))  # expected output: 3.0
print(round(adfuller(np.diff(yv), autolag="AIC")[1], 3))  # expected output: 0.054
# 0.054 is just above 0.05, so one first difference does not clear the line: it removes the trend but leaves
# the strong yearly season (Section 17.2.1).

# %% [markdown]
# Decomposition and smoothing

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
import warnings; warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
# Decomposition and smoothing
res = STL(y, period=12, robust=True).fit()
print(round(res.seasonal.max() - res.seasonal.min(), 1))  # expected output: 250.9
a = 0.3; print([round(a * (1 - a) ** k, 3) for k in range(4)])  # expected output: [0.3, 0.21, 0.147, 0.103]
# SES tracks only a level, so with nothing to extend it its multistep forecast is a flat line.

# %% [markdown]
# Evaluation metrics

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
import warnings; warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
# Evaluation metrics
snaive = np.array([yv[-24 - 12 + (i % 12)] for i in range(24)])   # last observed same month: 1958 repeated
print(round(rmse(yv[-24:], snaive), 2))  # expected output: 76.99
mae = np.mean(np.abs(yv[-24:] - snaive))
naive_mae = np.mean(np.abs(np.diff(yv[:120])))                    # in-sample one-step naive MAE, 1949-1958
print(round(mae, 2), round(naive_mae, 2), round(mae / naive_mae, 2))  # expected output: 71.25 22.16 3.22
# MASE 3.22 is far above 1: errors made up to 24 months ahead on a trending series are more than three times
# the in-sample month-to-month change, which says nothing about beating a naive forecast on these 24 months.

# %% [markdown]
# Feature engineering and windowing

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
import warnings; warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
# Feature engineering and windowing
df = pd.DataFrame({"y": y})
df["lag12"] = df["y"].shift(12)             # same month last year
df["roll3"] = df["y"].shift(1).rolling(3).mean()   # shift BEFORE roll: never includes the current target
print(df[["lag12", "roll3"]].dropna().iloc[0].round(2).tolist())  # expected output: [112.0, 113.67]
# 112.33 is the unshifted rolling(3), which includes the current target and leaks.
def windows(arr, L, H):
    X, Y = [], []
    for i in range(len(arr) - L - H + 1):
        X.append(arr[i:i + L]); Y.append(arr[i + L:i + L + H])
    return np.array(X)[:, :, None], np.array(Y)
Xw, Yw = windows(yv, 24, 24)
print(Xw.shape, Yw.shape)  # expected output: (97, 24, 1) (97, 24)

# %% [markdown]
# Quantiles and intervals

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
import warnings; warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
# Quantiles and intervals
def pinball(yy, q, tau):
    e = np.asarray(yy) - np.asarray(q); return float(np.mean(np.maximum(tau * e, (tau - 1) * e)))
print(round(2 * pinball(yv[-24:], snaive, 0.5), 2), round(mae, 2))  # expected output: 71.25 71.25
# A regression tree predicts by averaging training targets, so it can never output a value above the highest level it saw.
