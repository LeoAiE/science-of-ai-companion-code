"""Chapter 17: Time Series & Forecasting

Section-aligned walkthrough: every code block from this chapter of the book,
in book order, each in its own cell under its section heading so you can run
and check them one by one. Generated from Book.md; keep the book as the source.
One setup cell before the Prophet listing is not in the book: it puts the TBB folder
bundled inside Prophet's cmdstan first on PATH on Windows, as Section 17.6 describes.
Example 17-18 of Section 17.5.6 runs in its own script, 14_tft_bike.py, because its twenty
trainings would about double this notebook's run time.
"""

# %% [markdown]
# ## 17.1.2 Why a random split is a crime

# %% [markdown]
# Example 17-1. Time series structure and the time-ordered train/test split.

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
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")  # torch + xgboost OpenMP clash on this Mac (PLAN section 4)
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")                       # headless backend: save figures, never pop a window
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).parent if "__file__" in globals() else Path(".")  # save beside this script

flights = sns.load_dataset("flights")        # downloaded on first use, then cached: monthly passengers, 1949-1960
y = pd.Series(flights["passengers"].to_numpy(float),
              index=pd.date_range("1949-01-01", periods=len(flights), freq="MS"),
              name="passengers")
yv = y.to_numpy()
n = len(y); t = np.arange(n)
print(f"series: {n} monthly points, range {yv.min():.0f}..{yv.max():.0f} thousand passengers")
print(f"first year: {np.round(yv[:12], 1)}")

def acf(x, lag):
    x = x - x.mean()
    return float(np.sum(x[lag:] * x[:len(x) - lag]) / np.sum(x * x))

slope, intercept = np.polyfit(t, yv, 1)      # estimate a straight-line trend
detrended = yv - (slope * t + intercept)     # remove it; what's left is season + noise

print("\nautocorrelation at several lags (1 = perfect, 0 = none):")
print("  lag :     1      3      6     12")
print("  raw :  " + " ".join(f"{acf(yv, k):+.2f}" for k in [1, 3, 6, 12]) +
      "   (high and slowly decaying = a TREND dominates)")
print("  detr:  " + " ".join(f"{acf(detrended, k):+.2f}" for k in [1, 3, 6, 12]) +
      "   (after detrending the SEASON appears)")

def interp_predict(train_t, train_y, query_t):
    order = np.argsort(train_t)
    return np.interp(query_t, train_t[order], train_y[order])

rng = np.random.default_rng(0)
idx = rng.permutation(n)                      # random split: test points scattered in time
rtest = np.sort(idx[:int(0.3 * n)]); rtrain = np.sort(idx[int(0.3 * n):])
rand_pred = interp_predict(rtrain, yv[rtrain], rtest)
rand_rmse = np.sqrt(np.mean((rand_pred - yv[rtest]) ** 2))

cut = int(0.7 * n)                            # time split: train on the first 70%, test the last 30%
time_pred = interp_predict(t[:cut], yv[:cut], t[cut:])   # np.interp holds flat past the last point
time_rmse = np.sqrt(np.mean((time_pred - yv[cut:]) ** 2))

print(f"\nsame model, same data, two evaluation protocols:")
print(f"  RANDOM split RMSE : {rand_rmse:6.2f}  (test points sit between train points -> easy)")
print(f"  TIME   split RMSE : {time_rmse:6.2f}  (must extrapolate into the future -> honest)")

fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
ax[0].plot(t, yv, "-", lw=1.1, label="observed")
ax[0].plot(t, slope * t + intercept, "--", label="fitted straight-line trend")
ax[0].set_title("Airline passengers = trend + seasonality"); ax[0].legend()
ax[1].axvspan(cut, n, color="orange", alpha=0.2, label="time-split test (the future)")
ax[1].plot(t, yv, "-", lw=1.1); ax[1].axvline(cut, color="orange")
ax[1].set_title("Honest split: test set is the future"); ax[1].set_xlabel("month index"); ax[1].legend()
fig.tight_layout(); fig.savefig(HERE / "flights_structure.png", dpi=110)

# %% [markdown]
# ## 17.2.1 Stationarity and differencing

# %% [markdown]
# Example 17-2. Stationarity and differencing, tested with ADF and KPSS.

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
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tools.sm_exceptions import InterpolationWarning
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore", InterpolationWarning)

flights = sns.load_dataset("flights")
y = pd.Series(flights["passengers"].to_numpy(float),
              index=pd.date_range("1949-01-01", periods=len(flights), freq="MS"),
              name="passengers")
yv = y.to_numpy()

h = len(y) // 2
print(f"mean of first half  = {yv[:h].mean():.1f}")
print(f"mean of second half = {yv[h:].mean():.1f}  (the level moved: NOT stationary)")

def adf_p(x, label):
    p = adfuller(np.asarray(x), autolag="AIC")[1]
    verdict = "stationary" if p < 0.05 else "nonstationary"
    print(f"  {label:28s} ADF p = {p:.3f}  -> {verdict}")

print("\nADF (null = nonstationary; reject when p < 0.05):")
adf_p(yv, "raw series")
diff1 = np.diff(yv)                               # first difference removes the trend
adf_p(diff1, "after 1st difference")
seas = yv[12:] - yv[:-12]                         # seasonal difference at lag 12 removes the season
adf_p(seas, "after seasonal difference")
both = np.diff(seas)                              # both operations together
adf_p(both, "seasonal + 1st difference")

def kpss_p(x, label):
    stat, p, *_ = kpss(np.asarray(x), regression="c", nlags="auto")
    verdict = "nonstationary" if p < 0.05 else "stationary"
    print(f"  {label:28s} KPSS p = {p:.3f}  -> {verdict}")

print("\nKPSS (null = STATIONARY, the opposite null; p is capped at 0.01 and 0.10):")
kpss_p(yv, "raw series")
kpss_p(diff1, "after 1st difference")
kpss_p(both, "seasonal + 1st difference")

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
stat, p, lags, nobs, crit, _ = adfuller(yv, autolag="AIC")
dy = np.diff(yv)                                  # one-step changes, dy_t = y_t - y_{t-1}
X = np.column_stack([np.ones(len(dy) - lags), yv[lags:-1]] +              # constant, lagged level
                    [dy[lags - i:len(dy) - i] for i in range(1, lags + 1)])  # lagged changes
beta, *_ = np.linalg.lstsq(X, dy[lags:], rcond=None)
resid = dy[lags:] - X @ beta
s2 = resid @ resid / (len(resid) - X.shape[1])    # residual variance of the regression
se_gamma = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
print(f"ADF regression on raw passengers: {lags} lagged changes, gamma = {beta[1]:+.4f}, "
      f"t = {beta[1] / se_gamma:+.3f}")
print(f"adfuller: statistic {stat:+.3f}, 5% critical value {crit['5%']:+.3f}, p = {p:.3f}")

e = yv - yv.mean()                                # KPSS with a constant: the demeaned series
S = np.cumsum(e)                                  # partial sums S_t
kstat, kp, klags, kcrit = kpss(yv, regression="c", nlags="auto")
lrv = (e @ e + 2 * sum((1 - j / (klags + 1)) * (e[j:] @ e[:-j]) for j in range(1, klags + 1))) / len(e)
print(f"KPSS by hand: {np.sum(S ** 2) / (len(e) ** 2 * lrv):.3f} with {klags} lags; "
      f"kpss(): {kstat:.3f}, 5% critical value {kcrit['5%']}")

# %% [markdown]
# ## 17.2.2 Reading ACF and PACF

# %% [markdown]
# Example 17-3. ACF and PACF signatures of AR and MA processes.

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
import numpy as np
import seaborn as sns
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_process import ArmaProcess

rng = np.random.default_rng(0)
N = 4000                                          # long synthetic series so the estimates are clean

ar1 = ArmaProcess(ar=[1, -0.7], ma=[1]).generate_sample(N, distrvs=rng.standard_normal)
print("AR(1), coefficient 0.7:")
print(f"  ACF  lags 1-4: {np.round(acf(ar1, nlags=4)[1:], 2)}  (geometric decay 0.7, 0.49, ...)")
print(f"  PACF lags 1-4: {np.round(pacf(ar1, nlags=4)[1:], 2)}  (one spike at lag 1, then ~0)")

ma1 = ArmaProcess(ar=[1], ma=[1, 0.7]).generate_sample(N, distrvs=rng.standard_normal)
print("\nMA(1), coefficient 0.7:")
print(f"  ACF  lags 1-4: {np.round(acf(ma1, nlags=4)[1:], 2)}  (one spike at lag 1, then ~0)")
print(f"  PACF lags 1-4: {np.round(pacf(ma1, nlags=4)[1:], 2)}  (gradual decay)")

# now the REAL series, differenced to stationarity: (1-B)(1-B^12) on log passengers
ly = np.log(sns.load_dataset("flights")["passengers"].to_numpy(float))
D = np.diff(ly)                                   # first difference (remove trend)
D = D[12:] - D[:-12]                              # seasonal difference of it (remove season)
print("\nlog airline passengers after (1-B)(1-B^12):")
print(f"  ACF  lags 1-13: {np.round(acf(D, nlags=13)[1:], 2)}")
print(f"  PACF lags 1-13: {np.round(pacf(D, nlags=13)[1:], 2)}")
print("  -> dominant negative spikes at lag 1 and lag 12 => SARIMA(0,1,1)(0,1,1)_12")

band = 1.96 / np.sqrt(len(D))                     # approximate 95% band for the estimates of white noise
def outside(r):                                   # lags 1 to 24 whose estimate leaves the band
    return ", ".join(f"{k}: {r[k]:+.2f}" for k in range(1, 25) if abs(r[k]) > band)
print(f"  95% band = +/- 1.96/sqrt({len(D)}) = +/- {band:.3f}")
print(f"  ACF  lags 1-24 outside the band (lag: value): {outside(acf(D, nlags=24))}")
print(f"  PACF lags 1-24 outside the band (lag: value): {outside(pacf(D, nlags=24))}")

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
def last_ar_coef(x, k):                           # PACF at lag k: last coefficient of an AR(k) regression
    X = np.column_stack([np.ones(len(x) - k)] + [x[k - i:len(x) - i] for i in range(1, k + 1)])
    return np.linalg.lstsq(X, x[k:], rcond=None)[0][-1]
print("AR(k) regression, last coefficient, k = 1, 2, 3:", [round(float(last_ar_coef(ar1, k)), 3) for k in (1, 2, 3)])
print("statsmodels pacf at lags 1, 2, 3:              ", [round(float(v), 3) for v in pacf(ar1, nlags=3)[1:]])

# %% [markdown]
# ## 17.2.3 Decomposition with STL

# %% [markdown]
# Example 17-4. STL decomposition into trend, season, and residual.

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
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.seasonal import STL

HERE = Path(__file__).parent if "__file__" in globals() else Path(".")  # save beside this script

flights = sns.load_dataset("flights")
y = pd.Series(flights["passengers"].to_numpy(float),
              index=pd.date_range("1949-01-01", periods=len(flights), freq="MS"),
              name="passengers")

res = STL(y, period=12, robust=True).fit()
trend, seasonal, resid = res.trend, res.seasonal, res.resid

reconstructed = trend + seasonal + resid
print(f"max |original - (trend+seasonal+resid)| = {np.abs(y - reconstructed).max():.2e}  (=> they sum back)")
print(f"seasonal peak-to-peak = {seasonal.max() - seasonal.min():.1f} thousand passengers")
ptp1949 = float(seasonal.loc["1949"].max() - seasonal.loc["1949"].min())
ptp1960 = float(seasonal.loc["1960"].max() - seasonal.loc["1960"].min())
print(f"seasonal peak-to-peak 1949 = {ptp1949:.1f}, 1960 = {ptp1960:.1f} thousand passengers")
print(f"residual std          = {resid.std():.1f}")
print(f"residual std first 6 yrs = {resid.iloc[:72].std():.1f}, last 6 yrs = {resid.iloc[72:].std():.1f}"
      "  (it GROWS: multiplicative season)")

fig, ax = plt.subplots(4, 1, figsize=(8, 8), sharex=True)
for a, series, name in zip(ax, [y, trend, seasonal, resid],
                           ["observed", "trend", "seasonal", "residual"]):
    a.plot(y.index, np.asarray(series)); a.set_ylabel(name)
ax[0].set_title("STL decomposition of monthly airline passengers"); ax[-1].set_xlabel("year")
fig.tight_layout(); fig.savefig(HERE / "flights_stl.png", dpi=110)

# %% [markdown]
# ## 17.2.4 Box-Cox: the family the log transform belongs to

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
from scipy.stats import boxcox

yv = y.to_numpy()
transformed, lam = boxcox(yv)                     # scipy picks lambda by maximum likelihood
print(f"Box-Cox lambda for the airline series: {lam:.3f}  (0 = log, 1 = no reshaping)")
print(f"January 1949, y = {yv[0]:.0f}: ({yv[0]:.0f}^lambda - 1)/lambda = {(yv[0] ** lam - 1) / lam:.2f} "
      f"(boxcox: {transformed[0]:.2f}; ln y = {np.log(yv[0]):.2f})")
resid_log = STL(pd.Series(np.log(yv), index=y.index), period=12, robust=True).fit().resid
print(f"STL residual std on the LOG scale: first 6 yrs {resid_log.iloc[:72].std():.4f}, "
      f"last 6 yrs {resid_log.iloc[72:].std():.4f}")

# %% [markdown]
# ## 17.2.5 Exponential smoothing: SES, Holt, Holt-Winters

# %% [markdown]
# Example 17-5. SES, Holt, and Holt-Winters compared.

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
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing

flights = sns.load_dataset("flights")
y = pd.Series(flights["passengers"].to_numpy(float),
              index=pd.date_range("1949-01-01", periods=len(flights), freq="MS"),
              name="passengers")
train, test = y.iloc[:-24], y.iloc[-24:]          # forecast the last 24 months (time-ordered)

alpha = 0.3
weights = [alpha * (1 - alpha) ** k for k in range(6)]
print(f"SES weight on the last 6 points (alpha={alpha}): {np.round(weights, 3)}")
print(f"  they fade by (1-alpha)={1-alpha} each step back and sum to ~1; recent points dominate.")

def rmse(a, b): return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))

# statsmodels 0.15.0 rejects the old default None of SES and Holt, so all four fits name one method:
# "estimated" fits the starting level, trend, and season together with the smoothing parameters
ses = SimpleExpSmoothing(train, initialization_method="estimated").fit()
holt = Holt(train, initialization_method="estimated").fit()
hw_add = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12,
                              initialization_method="estimated").fit()
hw_mul = ExponentialSmoothing(train, trend="add", seasonal="mul", seasonal_periods=12,
                              initialization_method="estimated").fit()

print(f"\nforecast RMSE on the 24-month holdout (lower is better):")
print(f"  SES              (level only)        : {rmse(test, ses.forecast(24)):6.2f}  flat line, misses trend and season")
print(f"  Holt             (level + trend)     : {rmse(test, holt.forecast(24)):6.2f}  adds the trend, still no season")
print(f"  Holt-Winters add (level+trend+season): {rmse(test, hw_add.forecast(24)):6.2f}  follows the yearly cycle")
print(f"  Holt-Winters mul (season x level)    : {rmse(test, hw_mul.forecast(24)):6.2f}  best: the season grows with the level")

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
a, b_star, g = (hw_mul.params[k] for k in ("smoothing_level", "smoothing_trend", "smoothing_seasonal"))
lvl, slope, s = hw_mul.level, hw_mul.trend, hw_mul.season       # one value per training month
y_t, prev = train.iloc[-1], lvl.iloc[-2] + slope.iloc[-2]        # December 1958; November level + slope
level = a * y_t / s.iloc[-13] + (1 - a) * prev                   # s.iloc[-13] is December 1957
season = g * y_t / prev + (1 - g) * s.iloc[-13]
print(f"alpha = {a:.3f}, beta* = {b_star:.3f}, gamma = {g:.3f}, slope = {slope.iloc[-1]:.2f}")
print(f"December 1957 factor {s.iloc[-13]:.4f}; November 1958 level + slope = "
      f"{lvl.iloc[-2]:.2f} + {slope.iloc[-2]:.2f} = {prev:.2f}")
print(f"December 1958, y = {y_t:.0f}: level {level:.2f} (fitted {lvl.iloc[-1]:.2f}), "
      f"factor {season:.4f} (fitted {s.iloc[-1]:.4f})")
print(f"January 1959: ({lvl.iloc[-1]:.2f} + {slope.iloc[-1]:.2f}) x {s.iloc[-12]:.4f} = "
      f"{(lvl.iloc[-1] + slope.iloc[-1]) * s.iloc[-12]:.2f}; forecast() gives {hw_mul.forecast(1).iloc[0]:.2f}")

# %% [markdown]
# ## 17.2.6 The ARIMA family

# %% [markdown]
# Example 17-6. The ARIMA family: ARIMA, SARIMA, and SARIMAX.

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
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox

flights = sns.load_dataset("flights")
y = pd.Series(flights["passengers"].to_numpy(float),
              index=pd.date_range("1949-01-01", periods=len(flights), freq="MS"),
              name="passengers")
train, test = y.iloc[:-24], y.iloc[-24:]

def rmse(a, b): return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))

naive = [train.iloc[-12 + (i % 12)] for i in range(24)]    # last observed same month; 24 months back in year 2
print(f"seasonal-naive baseline            : {rmse(test, naive):6.2f}  (repeat last year)")

arima = SARIMAX(train, order=(1, 1, 1)).fit(disp=False)
print(f"ARIMA(1,1,1) (no season)           : {rmse(test, arima.forecast(24)):6.2f}  WORSE than naive: no season")

sarima = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
print(f"SARIMA(1,1,1)(1,1,1)_12 on raw y   : {rmse(test, sarima.forecast(24)):6.2f}  barely beats naive")

logtrain = np.log(train)
slog = SARIMAX(logtrain, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
print(f"SARIMA on LOG(y)                   : {rmse(test, np.exp(slog.forecast(24))):6.2f}  best: log tames the multiplicative season")

# AIC order search, written out as an explicit grid so the mechanics auto-ARIMA tools automate stay visible.
# Some candidates warn about their start values or stop before converging; those warnings are silenced
# here only, and a candidate that did not converge is counted and left out of the ranking.
best, stopped, tried = None, 0, 0
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for p in range(3):
        for q in range(3):
            for P in range(2):
                for Q in range(2):
                    try:
                        m = SARIMAX(logtrain, order=(p, 1, q), seasonal_order=(P, 1, Q, 12)).fit(disp=False)
                        tried += 1
                        if not m.mle_retvals["converged"]:
                            stopped += 1          # its likelihood may not be at the maximum, so its AIC is not used
                        elif best is None or m.aic < best[0]:
                            best = (m.aic, (p, 1, q), (P, 1, Q, 12))
                    except Exception:
                        pass
print(f"\nmin-AIC orders on log(y)           : SARIMA{best[1]}{best[2][:3]}_{best[2][3]}  (AIC {best[0]:.1f})")
print(f"grid fits that did not converge    : {stopped} of {tried} (left out of the ranking)")

lb = acorr_ljungbox(slog.resid[13:], lags=[12], return_df=True)
lb4 = acorr_ljungbox(slog.resid[13:], lags=[12], model_df=4, return_df=True)
print(f"Ljung-Box(12) on (1,1,1)(1,1,1)_12 log resid: p = {lb['lb_pvalue'].iloc[0]:.3f} (12 df), {lb4['lb_pvalue'].iloc[0]:.3f} (model_df=4)")
sel = SARIMAX(logtrain, order=best[1], seasonal_order=best[2]).fit(disp=False)
lbs = acorr_ljungbox(sel.resid[13:], lags=[12], return_df=True)
lbs2 = acorr_ljungbox(sel.resid[13:], lags=[12], model_df=2, return_df=True)
print(f"selected model Ljung-Box(12): p = {lbs['lb_pvalue'].iloc[0]:.3f} (12 df), {lbs2['lb_pvalue'].iloc[0]:.3f} (model_df=2)")
print(f"selected model 24-month RMSE: {rmse(test, np.exp(sel.forecast(24))):.2f}")
print(f"selected model k={len(sel.params)} llf={float(sel.llf):.3f} AIC={float(sel.aic):.2f} AICc={float(sel.aicc):.2f} BIC={float(sel.bic):.2f}")
print(f"selected model coefficients: ma.L1 = {sel.params['ma.L1']:+.3f}, ma.S.L12 = {sel.params['ma.S.L12']:+.3f}, "
      f"sigma2 = {sel.params['sigma2']:.5f}")

# Illustrative step flag. The forecast needs exog at every horizon month.
step = (y.index.year >= 1958).astype(float)
mx = SARIMAX(logtrain, exog=step[:-24], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
print(f"January 1958 step coefficient on the log scale: {float(mx.params['x1']):+.3f}")
print(f"log-SARIMA   (no exog)             : {rmse(test, np.exp(slog.forecast(24))):6.2f}")
print(f"log-SARIMAX  (January 1958 step)   : {rmse(test, np.exp(mx.forecast(24, exog=step[-24:]))):6.2f}")
step_oct = ((y.index.year > 1958) | ((y.index.year == 1958) & (y.index.month >= 10))).astype(float)
mx_oct = SARIMAX(logtrain, exog=step_oct[:-24], order=(1, 1, 1),
                 seasonal_order=(1, 1, 1, 12)).fit(disp=False, maxiter=500)   # the default 50 iterations stop short
print(f"October 1958 step coefficient: {float(mx_oct.params['x1']):+.3f}; RMSE {rmse(test, np.exp(mx_oct.forecast(24, exog=step_oct[-24:]))):.2f}")
fits = {"ARIMA": arima, "SARIMA raw": sarima, "SARIMA log": slog, "selected": sel,
        "January step": mx, "October step": mx_oct}
print("optimizer converged:", ", ".join(f"{k} {r.mle_retvals['converged']}" for k, r in fits.items()))

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
from scipy.stats import chi2

r = np.asarray(sel.resid[13:]); nr = len(r); rc = r - r.mean()
rk = np.array([rc[k:] @ rc[:-k] / (rc @ rc) for k in range(1, 13)])   # r_1 .. r_12, the 17.1.1 formula
Q = nr * (nr + 2) * np.sum(rk ** 2 / (nr - np.arange(1, 13)))
print(f"{nr} residuals; largest |r_k| over lags 1 to 12 = {np.abs(rk).max():.3f}")
print(f"Q = {Q:.3f} (acorr_ljungbox: {lbs['lb_stat'].iloc[0]:.3f}); "
      f"p with 12 df = {chi2.sf(Q, 12):.3f}, with 12 - 2 = 10 df = {chi2.sf(Q, 10):.3f}")

# %% [markdown]
# ## 17.2.7 The state-space view: the machinery inside SARIMAX

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
from statsmodels.tsa.statespace.structural import UnobservedComponents

uc = UnobservedComponents(np.log(train.to_numpy()), level="local linear trend", seasonal=12)
ucr = uc.fit(disp=False)
uc_fc = np.exp(ucr.forecast(24))
print(f"structural model (level+slope+season on log): 24-month RMSE {rmse(test, uc_fc):.2f}")
print(f"log likelihood, computed by the Kalman filter: {ucr.llf:.1f}")

# %% [markdown]
# ## 17.2.8 Theta, Croston, and forecast combination

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
from statsmodels.tsa.forecasting.theta import ThetaModel

theta_fc = ThetaModel(train, period=12).fit().forecast(24)
print(f"Theta method: 24-month RMSE {rmse(test, theta_fc):.2f}")

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
th = ThetaModel(train, period=12).fit()
b0, alpha_th = th.params["b0"], th.params["alpha"]
comp = th.forecast_components(24)                  # columns: trend, ses, seasonal
drift = 0.5 * b0 * (1 / alpha_th - (1 - alpha_th) ** len(train) / alpha_th)   # the h = 1 drift term
print(f"b0 = {b0:.3f}, alpha = {alpha_th:.3f}, SES level = {comp['ses'].iloc[0]:.2f}, "
      f"January index = {comp['seasonal'].iloc[0]:.4f}, "
      f"(1 - alpha)^{len(train)} = {(1 - alpha_th) ** len(train):.1e}")
print(f"January 1959 by hand: ({comp['ses'].iloc[0]:.2f} + {drift:.2f}) x {comp['seasonal'].iloc[0]:.4f} = "
      f"{(comp['ses'].iloc[0] + drift) * comp['seasonal'].iloc[0]:.2f}; ThetaModel: {theta_fc.iloc[0]:.2f}")

# %% [markdown]
# Example 17-7. Croston's method on an intermittent demand series.

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
import numpy as np

demand = np.array([0, 0, 3, 0, 0, 0, 2, 0, 4, 0, 0, 5], float)   # intermittent: mostly zeros
alpha, z, p, gap = 0.2, None, None, 0
for d in demand:
    gap += 1
    if d > 0:                                     # update ONLY when a demand occurs
        z = d if z is None else z + alpha * (d - z)       # smoothed demand size
        p = gap if p is None else p + alpha * (gap - p)   # smoothed interval between demands
        gap = 0
print(f"Croston: size z = {z:.3f}, interval p = {p:.3f}, forecast {z/p:.3f} per period")
print(f"mean demand of the series = {demand.mean():.3f}")

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
hw_fc = hw_mul.forecast(24).to_numpy()            # multiplicative Holt-Winters fitted in Example 17-5
sar_fc = np.exp(slog.forecast(24).to_numpy())     # log SARIMA fitted, and checked for convergence, in Example 17-6
comb = (hw_fc + sar_fc) / 2
print(f"HW-mul {rmse(test, hw_fc):.2f} | log-SARIMA {rmse(test, sar_fc):.2f} | "
      f"average of the two {rmse(test, comb):.2f}")

# %% [markdown]
# ## 17.2.9 Beyond one series: VAR, volatility models, and hierarchies

# %% [markdown]
# Example 17-8. A vector autoregression on US GDP and consumption growth.

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
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
import warnings; warnings.filterwarnings("ignore")

macro = sm.datasets.macrodata.load_pandas().data          # real US quarterly macro data, 1959-2009
growth = 100 * np.log(macro[["realgdp", "realcons"]]).diff().dropna()  # quarterly growth in percent
growth.index = pd.period_range("1959Q2", periods=len(growth), freq="Q")

model = VAR(growth)
p = model.select_order(8).aic                              # lag order chosen by AIC
res = model.fit(p)
print(f"{len(growth)} quarters of GDP and consumption growth; VAR({p}) chosen by AIC")
print(f"L1.realcons -> realgdp coefficient: {res.params.loc['L1.realcons', 'realgdp']:+.3f}")
gc = res.test_causality("realgdp", ["realcons"])
print(f"Granger test, consumption growth helps predict GDP growth: p = {gc.pvalue:.4f}")
fc = res.forecast(growth.values[-p:], steps=4)
print("next 4 quarters, forecast growth in percent:")
print(pd.DataFrame(fc, columns=["realgdp", "realcons"]).round(2).to_string(index=False))

# %% [markdown]
# Example 17-9. A GARCH(1,1) model of daily S&P 500 return volatility.

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
import arch.data.sp500
from arch import arch_model

sp500 = arch.data.sp500.load()                              # real daily S&P 500 prices, 1999-2018
returns = 100 * sp500["Adj Close"].pct_change().dropna()    # daily returns in percent
res = arch_model(returns, vol="GARCH", p=1, q=1).fit(disp="off")
mu, om, al, be = res.params["mu"], res.params["omega"], res.params["alpha[1]"], res.params["beta[1]"]
print(f"GARCH(1,1) on {len(returns)} daily returns: mu={mu:.4f}, omega={om:.4f}, alpha={al:.4f}, beta={be:.4f}")
print(f"persistence alpha+beta = {al + be:.4f}; long-run daily variance = {om / (1 - al - be):.2f}")
print(f"to six decimals: omega = {om:.6f}, 1 - alpha - beta = {1 - al - be:.6f}, ratio = {om / (1 - al - be):.3f}")
print(f"sample std of returns = {float(returns.std()):.2f}")
print(f"last return r = {returns.iloc[-1]:+.2f}, shock eps = r - mu = {res.resid.iloc[-1]:+.2f}, "
      f"last variance = {res.conditional_volatility.iloc[-1] ** 2:.2f}")
print(f"next-day variance forecast = {res.forecast(horizon=1).variance.iloc[-1, 0]:.2f}")

# %% [markdown]
# Example 17-10. Bottom-up and MinT reconciliation of a two-child hierarchy.

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
import numpy as np

rng = np.random.default_rng(0)
S = np.array([[1, 1], [1, 0], [0, 1]], float)      # rows: total, A, B
sig = np.array([2.0, 4.0, 3.0])                    # base-error std per level: total, A, B
truth = np.array([105.0, 60.0, 45.0])              # true total, A, B (coherent by construction)
base = truth + rng.normal(0, sig, size=(10000, 3)) # 10,000 independent sets of base forecasts
G_bu = np.array([[0, 1, 0], [0, 0, 1]], float)     # bottom-up: keep the children, drop the total
W = np.diag(sig ** 2); Winv = np.linalg.inv(W)
G_mint = np.linalg.inv(S.T @ Winv @ S) @ S.T @ Winv
def per_series_rmse(f): return np.sqrt(np.mean((f - truth) ** 2, axis=0))
print("RMSE per series   [total,  A,     B]")
print("base (incoherent)", np.round(per_series_rmse(base), 2))
print("bottom-up        ", np.round(per_series_rmse(base @ (S @ G_bu).T), 2))
print("MinT (WLS)       ", np.round(per_series_rmse(base @ (S @ G_mint).T), 2))
print("G_MinT rows A and B, columns total, A, B:", np.round(G_mint, 4).tolist())
one = np.array([105.0, 62.0, 47.0])                # one incoherent day: 62 + 47 = 109, not 105
rec = S @ (G_mint @ one)
print("one day: base", one, "-> MinT", np.round(rec, 2), " (children sum to the total)")

# %% [markdown]
# ## 17.3.4 Backtesting: many origins, not one lucky split

# %% [markdown]
# Example 17-11. Forecast error metrics and an expanding-window backtest.

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
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings; warnings.filterwarnings("ignore")

# --- illustrative arithmetic on a tiny synthetic error vector (checkable by hand) ---
actual = np.array([100., 110., 90., 105.])
fcst = np.array([95., 108., 100., 100.])
err = fcst - actual

mae = np.mean(np.abs(err))                                  # mean absolute error (same units)
rmse = np.sqrt(np.mean(err ** 2))                           # root mean squared error (punishes big misses)
mape = np.mean(np.abs(err) / np.abs(actual)) * 100          # mean absolute PERCENTAGE error
smape = np.mean(2 * np.abs(err) / (np.abs(actual) + np.abs(fcst))) * 100   # symmetric MAPE
print(f"MAE   = {mae:.2f}   (average miss, in the data's own units)")
print(f"RMSE  = {rmse:.2f}   (>= MAE; the gap grows when a few misses are large)")
print(f"MAPE  = {mape:.1f}%  (average miss as a percent of actual)")
print(f"sMAPE = {smape:.1f}%  (percent error, symmetric in actual vs forecast)")

under0 = abs(0 - 100) / 100 * 100                          # most extreme underforecast: predict 0
over3x = abs(300 - 100) / 100 * 100                        # a 3x overforecast
print(f"  underforecast to 0   : actual=100, forecast=0   -> MAPE {under0:.0f}% (cannot exceed 100%)")
print(f"  overforecast x3      : actual=100, forecast=300 -> MAPE {over3x:.0f}% (no upper bound)")

train = np.array([100., 102., 101., 103., 104., 103., 105., 106.])
naive_mae = np.mean(np.abs(np.diff(train)))                 # MAE of "tomorrow = today" in-sample
mase = mae / naive_mae
print(f"\nMASE = MAE/naive = {mae:.2f}/{naive_mae:.2f} = {mase:.2f}  "
      f"({'below' if mase < 1 else 'above'} the in-sample naive scale)")

# --- honest backtest on REAL data: the monthly airline-passenger series ---
flights = sns.load_dataset("flights")
y = pd.Series(flights["passengers"].to_numpy(float),
              index=pd.date_range("1949-01-01", periods=len(flights), freq="MS"), name="passengers")

H = 12                                                      # forecast horizon per fold
origins = [96, 108, 120, 132]                              # expanding train end-points
rmses = []
for cut in origins:
    model = ExponentialSmoothing(y.iloc[:cut], trend="add", seasonal="mul",
                                 seasonal_periods=12).fit()
    f = model.forecast(H)
    rmses.append(np.sqrt(np.mean((f.values - y.iloc[cut:cut + H].values) ** 2)))
    print(f"  train[:{cut}] -> next {H}: RMSE {rmses[-1]:.2f}")
rmses = np.array(rmses)
print(f"expanding-window backtest ({len(origins)} folds): mean RMSE {rmses.mean():.2f} +/- {rmses.std():.2f}")

# %% [markdown]
# ## 17.4.2 Gradient boosting, the quiet champion

# %% [markdown]
# Example 17-12. Recursive and direct multistep forecasts from one origin, against the one-step score with true lags.

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
import numpy as np
import seaborn as sns
from xgboost import XGBRegressor              # n_jobs=2 below, as in every XGBoost listing of the chapter

flights = sns.load_dataset("flights")
y = flights["passengers"].to_numpy(float)    # 144 months, January 1949 to December 1960
month = np.arange(len(y)) % 12               # calendar month of each index, 0 = January
origin = len(y) - 25                         # index 119, December 1958: the last month the forecaster knows
H = 24                                       # forecast January 1959 to December 1960

def feats(v, t, h):
    """Example 17-13's five features for target month t, built only from values known h months earlier."""
    o = t - h                                          # the forecast origin of this row
    k = int(np.ceil(h / 12))                           # whole years back to the latest known same-month value
    last3 = v[o - 2:o + 1]                             # the three months that end at the origin
    return [v[o], v[t - 12 * k], last3.mean(), last3.std(ddof=1), month[t]]

def table(h):                                          # training rows for horizon h: every target up to the origin
    rows = list(range(max(12 * int(np.ceil(h / 12)), h + 2), origin + 1))
    return np.array([feats(y, t, h) for t in rows]), y[rows]

def fit(X, target):
    return XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, n_jobs=2).fit(X, target)

def rmse(a, b): return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))

test = y[origin + 1:]
one_step = fit(*table(1))                              # h = 1 is Example 17-13's model on the same 108 rows
true_lags = one_step.predict(np.array([feats(y, t, 1) for t in range(origin + 1, len(y))]))

hist = list(y[:origin + 1])                            # recursive: each forecast becomes the next month's lag
for t in range(origin + 1, len(y)):
    hist.append(float(one_step.predict(np.array([feats(np.array(hist), t, 1)]))[0]))
recursive = np.array(hist[origin + 1:])

direct = np.array([fit(*table(h)).predict(np.array([feats(y, origin + h, h)]))[0]   # one model per horizon,
                   for h in range(1, H + 1)])                                         # all fed the origin's data

print("direct features for March 1959 (h = 3):", np.round(feats(y, origin + 3, 3), 2).tolist())
print(f"training rows: {len(table(1)[1])} for h = 1, {len(table(13)[1])} for h = 13, {len(table(24)[1])} for h = 24")
print("24-month test RMSE, January 1959 to December 1960:")
print(f"  one-step, true lags (Example 17-13): {rmse(test, true_lags):6.2f}")
print(f"  recursive, one model               : {rmse(test, recursive):6.2f}")
print(f"  direct, 24 models                  : {rmse(test, direct):6.2f}")
print("RMSE by year of the horizon (months 1 to 12 | 13 to 24):")
for name, f in [("true lags", true_lags), ("recursive", recursive), ("direct", direct)]:
    print(f"  {name:9s}: {rmse(test[:12], f[:12]):6.2f} | {rmse(test[12:], f[12:]):6.2f}   highest forecast {f.max():.1f}")
print(f"training maximum {y[:origin + 1].max():.0f}, test maximum {test.max():.0f}")

# %% [markdown]
# ## 17.4.3 Avoiding leakage

# %% [markdown]
# Example 17-13. Machine-learning forecasting with lag features, and a leakage demonstration.

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
import numpy as np
import pandas as pd
import seaborn as sns
from xgboost import XGBRegressor              # n_jobs=2 below: a small thread pool is fastest on a table this tiny
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings; warnings.filterwarnings("ignore")

flights = sns.load_dataset("flights")
y = pd.Series(flights["passengers"].to_numpy(float),
              index=pd.date_range("1949-01-01", periods=len(flights), freq="MS"), name="passengers")

df = pd.DataFrame({"y": y})
df["lag1"] = df["y"].shift(1)                  # last month
df["lag12"] = df["y"].shift(12)               # same month last year (season)
df["roll3_mean"] = df["y"].shift(1).rolling(3).mean()   # trailing 3-month mean (shift first = no peeking)
df["roll3_std"] = df["y"].shift(1).rolling(3).std()
df["month"] = df.index.month - 1               # calendar feature, 0..11
df = df.dropna()
print("supervised table (first 3 rows): target y plus past-only features")
print(df.head(3).round(2))

feats = ["lag1", "lag12", "roll3_mean", "roll3_std", "month"]
cut = len(df) - 24
Xtr, ytr = df[feats].iloc[:cut], df["y"].iloc[:cut]
Xte, yte = df[feats].iloc[cut:], df["y"].iloc[cut:]

def rmse(a, b): return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))

model = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, n_jobs=2)
model.fit(Xtr, ytr)
xgb_fc = model.predict(Xte)
xgb_rmse = rmse(yte, xgb_fc)

snaive = [df["y"].iloc[cut - 12 + (i % 12)] for i in range(24)]
fit_raw = SARIMAX(ytr, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
fit_log = SARIMAX(np.log(ytr), order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(
    disp=False, method="nm", maxiter=10000)    # Nelder-Mead: the default L-BFGS stops early on this window
sarima, sarima_log = np.asarray(fit_raw.forecast(24)), np.exp(np.asarray(fit_log.forecast(24)))
print("24-month test RMSE:")
print(f"  seasonal-naive : {rmse(yte, snaive):5.2f}  (the bar to beat)")
print(f"  SARIMA (raw)   : {rmse(yte, sarima):5.2f}")
print(f"  XGBoost on lags: {xgb_rmse:5.2f}  (beats naive AND raw SARIMA)")
print(f"  SARIMA (log)   : {rmse(yte, sarima_log):5.2f}  (multiplicative-season fix still wins)")
print(f"SARIMA fits converged: raw {fit_raw.mle_retvals['converged']}, log {fit_log.mle_retvals['converged']}")
print("importances:", {f: round(float(v), 2) for f, v in zip(feats, model.feature_importances_)})
summer = yte.index.month.isin([7, 8])          # the four July and August test months
print(f"XGBoost forecast max {xgb_fc.max():.1f}, training max {ytr.max():.0f}, test max {yte.max():.0f}")
for name, fc in [("XGBoost", xgb_fc), ("SARIMA (raw)", sarima)]:
    print(f"  {name:12s} RMSE: July and August {rmse(yte[summer], fc[summer]):6.2f}, "
          f"other 20 months {rmse(yte[~summer], fc[~summer]):6.2f}")

df2 = df.copy()
df2["leaky"] = y.rolling(3, center=True).mean().reindex(df2.index)   # centered = uses future points
df2 = df2.dropna()
cut2 = len(df2) - 24
honest = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, n_jobs=2).fit(
    df2[feats].iloc[:cut2], df2["y"].iloc[:cut2])
leaky = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, n_jobs=2).fit(
    df2[feats + ["leaky"]].iloc[:cut2], df2["y"].iloc[:cut2])
honest_train = rmse(df2["y"].iloc[:cut2], honest.predict(df2[feats].iloc[:cut2]))
leaky_train = rmse(df2["y"].iloc[:cut2], leaky.predict(df2[feats + ["leaky"]].iloc[:cut2]))
print(f"in-sample RMSE: honest {honest_train:.2f} vs + leaky centered mean {leaky_train:.2f} (unearned gain)")
print(f"holdout RMSE ({df2.index[cut2]:%b %Y} to {df2.index[-1]:%b %Y}): "
      f"honest {rmse(df2['y'].iloc[cut2:], honest.predict(df2[feats].iloc[cut2:])):.2f} "
      f"vs leaky {rmse(df2['y'].iloc[cut2:], leaky.predict(df2[feats + ['leaky']].iloc[cut2:])):.2f}")
rng = np.random.default_rng(0)
df2["noise"] = rng.normal(size=len(df2))
noisy = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, n_jobs=2).fit(
    df2[feats + ["noise"]].iloc[:cut2], df2["y"].iloc[:cut2])
print(f"in-sample RMSE with a noise column: {rmse(df2['y'].iloc[:cut2], noisy.predict(df2[feats + ['noise']].iloc[:cut2])):.2f}")

# %% [markdown]
# ## 17.5.1 From LSTM to multi-output forecasting

# %% [markdown]
# Example 17-14. A multi-output LSTM forecaster.

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
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings; warnings.filterwarnings("ignore")

torch.manual_seed(0)
torch.set_num_threads(1)
flights = sns.load_dataset("flights")
series = flights["passengers"].to_numpy(float)    # 144 months, 1949-1960
train_raw, test_raw = series[:-24], series[-24:]

mu, sd = train_raw.mean(), train_raw.std()        # statistics from train only
scaled = (series - mu) / sd
L, H = 24, 24                                       # look back 2 seasons, forecast 2 seasons
def windows(arr, L, H):
    X, Y = [], []
    for i in range(len(arr) - L - H + 1):
        X.append(arr[i:i + L]); Y.append(arr[i + L:i + L + H])
    return np.array(X)[:, :, None], np.array(Y)     # (N, L, 1) inputs, (N, H) targets
cut = len(train_raw)
Xtr, Ytr = windows(scaled[:cut], L, H)
Xtr = torch.tensor(Xtr, dtype=torch.float32); Ytr = torch.tensor(Ytr, dtype=torch.float32)

class LSTMForecaster(nn.Module):
    def __init__(self, hidden=48, horizon=H):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, batch_first=True)
        self.head = nn.Linear(hidden, horizon)        # one shot: hidden state -> H predictions
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])               # (batch, H)

model = LSTMForecaster()
opt = torch.optim.Adam(model.parameters(), lr=0.01)
lossfn = nn.MSELoss()
for epoch in range(400):
    opt.zero_grad()
    loss = lossfn(model(Xtr), Ytr)
    loss.backward(); opt.step()

model.eval()
with torch.no_grad():
    x = torch.tensor(scaled[cut - L:cut], dtype=torch.float32).view(1, L, 1)
    lstm_fc = model(x).numpy().ravel() * sd + mu      # undo the scaling

def rmse(a, b): return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))
snaive = [train_raw[-12 + (i % 12)] for i in range(24)]
fit_raw = SARIMAX(train_raw, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
fit_log = SARIMAX(np.log(train_raw), order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
sarima, sarima_log = fit_raw.forecast(24), np.exp(fit_log.forecast(24))
print(f"{Xtr.shape[0]} training windows; 24-month test RMSE:")
print(f"  seasonal-naive : {rmse(test_raw, snaive):5.2f}")
print(f"  SARIMA (raw)   : {rmse(test_raw, sarima):5.2f}")
print(f"  SARIMA (log)   : {rmse(test_raw, sarima_log):5.2f}")
print(f"  LSTM (multi-output) : {rmse(test_raw, lstm_fc):5.2f}")
print(f"SARIMA fits converged: raw {fit_raw.mle_retvals['converged']}, log {fit_log.mle_retvals['converged']}")

# %% [markdown]
# ## 17.5.3 Zero-shot forecasting, demonstrated

# %% [markdown]
# Example 17-15. Zero-shot forecasting with the Chronos-Bolt tiny checkpoint.

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
import time
import torch
from chronos import BaseChronosPipeline          # pip install chronos-forecasting (version 2.3.2 here)

start = time.perf_counter()
pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-tiny", device_map="cpu",
                                           torch_dtype=torch.float32)
loaded = time.perf_counter()
quantiles, mean = pipe.predict_quantiles(torch.tensor(train_raw, dtype=torch.float32),
                                         prediction_length=24, quantile_levels=[0.1, 0.5, 0.9])
done = time.perf_counter()
zero_shot = quantiles[0, :, 1].numpy()           # the median forecast
print(f"Chronos-Bolt tiny, zero-shot: 24-month RMSE {rmse(test_raw, zero_shot):.2f}")
print(f"loading the checkpoint took {loaded - start:.1f} s, the forecast call {done - loaded:.2f} s")

# %% [markdown]
# ## 17.5.4 N-BEATS: a deep forecaster built from plain blocks

# %% [markdown]
# Example 17-16. N-BEATS from scratch on the airline series.

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
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings; warnings.filterwarnings("ignore")

torch.manual_seed(0)
torch.set_num_threads(1)
flights = sns.load_dataset("flights")
series = flights["passengers"].to_numpy(float)     # 144 months, 1949-1960
train_raw, test_raw = series[:-24], series[-24:]
mu, sd = train_raw.mean(), train_raw.std()         # scaling statistics from the training months only
scaled = (series - mu) / sd
L, H = 24, 24                                       # the LSTM's setup: read 24 months, emit the next 24
starts = range(len(train_raw) - L - H + 1)
X = torch.tensor(np.array([scaled[i:i + L] for i in starts]), dtype=torch.float32)
Y = torch.tensor(np.array([scaled[i + L:i + L + H] for i in starts]), dtype=torch.float32)

class NBeatsBlock(nn.Module):                       # one generic block: four ReLU layers, two linear heads
    def __init__(self, width=64, n_theta=8):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(L, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU(),
                                nn.Linear(width, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU())
        self.theta_b = nn.Linear(width, n_theta, bias=False)   # backcast expansion coefficients
        self.theta_f = nn.Linear(width, n_theta, bias=False)   # forecast expansion coefficients
        self.basis_b = nn.Linear(n_theta, L)                    # generic basis: V^b theta^b + b^b
        self.basis_f = nn.Linear(n_theta, H)                    # generic basis: V^f theta^f + b^f
    def forward(self, x):
        h = self.fc(x)
        return self.basis_b(self.theta_b(h)), self.basis_f(self.theta_f(h))

class NBeats(nn.Module):
    def __init__(self, n_blocks=3):
        super().__init__()
        self.blocks = nn.ModuleList([NBeatsBlock() for _ in range(n_blocks)])
    def forward(self, x):
        forecast = torch.zeros(x.shape[0], H)
        for block in self.blocks:
            backcast, partial = block(x)
            x = x - backcast                        # pass on only what this block did not explain
            forecast = forecast + partial           # and add this block's partial forecast
        return forecast

model = NBeats()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(500):
    opt.zero_grad()
    loss = nn.functional.mse_loss(model(X), Y)
    loss.backward(); opt.step()

model.eval()
with torch.no_grad():
    window = torch.tensor(scaled[len(train_raw) - L:len(train_raw)], dtype=torch.float32).view(1, L)
    nbeats_fc = model(window).numpy().ravel() * sd + mu    # undo the scaling

def rmse(a, b): return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))
snaive = [train_raw[-12 + (i % 12)] for i in range(24)]
fit_log = SARIMAX(np.log(train_raw), order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
sarima_log = np.exp(fit_log.forecast(24))
print(f"{X.shape[0]} training windows, {sum(p.numel() for p in model.parameters())} weights, "
      f"final training MSE {loss.item():.4f}")
print(f"  seasonal-naive     : {rmse(test_raw, snaive):5.2f}")
print(f"  SARIMA (log)       : {rmse(test_raw, sarima_log):5.2f}")
print(f"  N-BEATS (3 blocks) : {rmse(test_raw, nbeats_fc):5.2f}")
print(f"SARIMA (log) fit converged: {fit_log.mle_retvals['converged']}")

# %% [markdown]
# ## 17.5.5 DeepAR: a recurrent network that forecasts a distribution

# %% [markdown]
# Example 17-17. A compact DeepAR on the airline series, trained by likelihood and forecast by sampling paths.

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
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn

torch.manual_seed(0)
torch.set_num_threads(1)
flights = sns.load_dataset("flights")
series = flights["passengers"].to_numpy(float)      # 144 months, 1949-1960
train_raw, test_raw = series[:-24], series[-24:]
T, C, H = len(train_raw), 24, 24                     # 120 training months; 24 to condition on, 24 to predict
month = np.arange(len(series)) % 12                   # the covariate, known for every month in advance
month = (month - month[:T].mean()) / month[:T].std() # standardized with training statistics only

def windows(starts):                                  # training windows of C + H months from train_raw
    z = torch.tensor(np.array([train_raw[s:s + C + H] for s in starts]), dtype=torch.float32)
    x = torch.tensor(np.array([month[s:s + C + H] for s in starts]), dtype=torch.float32).unsqueeze(-1)
    nu = 1 + z[:, :C].mean(dim=1, keepdim=True)       # scale: 1 + mean of the conditioning range
    z_prev = torch.cat([torch.zeros(len(z), 1), z[:, :-1]], dim=1) / nu   # input at t is z(t-1), and z(0) = 0
    return z, x, nu, z_prev

class DeepAR(nn.Module):
    def __init__(self, hidden=40, layers=3):         # the paper's size for its smaller datasets
        super().__init__()
        self.lstm = nn.LSTM(2, hidden, num_layers=layers, batch_first=True)
        for name, p in self.lstm.named_parameters():  # forget-gate bias 1.0, as the paper sets it
            if name.startswith("bias_ih"):
                p.data[hidden:2 * hidden] = 1.0
            elif name.startswith("bias_hh"):
                p.data[hidden:2 * hidden] = 0.0
        self.mu = nn.Linear(hidden, 1)                # mean: an affine map of the hidden state
        self.sigma = nn.Linear(hidden, 1)             # standard deviation: affine map, then softplus
    def forward(self, z_prev, x, state=None):         # inputs: the scaled previous value and the covariate
        h, state = self.lstm(torch.cat([z_prev.unsqueeze(-1), x], dim=-1), state)
        return self.mu(h).squeeze(-1), nn.functional.softplus(self.sigma(h)).squeeze(-1), state

def nll(model, z, x, nu, z_prev, last=None):         # Gaussian negative log likelihood on the passenger scale
    mu, sigma, _ = model(z_prev, x)
    logp = torch.distributions.Normal(mu * nu, sigma * nu).log_prob(z)   # scale the parameters back up by nu
    return -(logp if last is None else logp[:, -last:]).mean()

def train(starts, epochs, seed, val=None, patience=200):
    torch.manual_seed(seed)                           # both fits of one seed start from the same weights
    model = DeepAR()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    data, best = windows(starts), (np.inf, 0)
    for epoch in range(1, epochs + 1):
        opt.zero_grad()
        loss = nll(model, *data)                      # every step of every window, as in the paper
        loss.backward(); opt.step()
        if val is not None and epoch % 10 == 0:
            with torch.no_grad():
                best = min(best, (nll(model, *val, last=H).item(), epoch))
            if epoch - best[1] >= patience:
                break
    return model, best

def forecast(seed, S=200):                            # S sample paths, as in the paper
    # early stopping inside the training months: fit the 49 windows that end by December 1956 and track
    # how likely 1957-1958 is given 1955-1956; then refit all 73 windows for the best number of epochs
    _, (val_nll, epochs) = train(range(T - 24 - C - H + 1), 1000, seed, val=windows([T - C - H]))
    model, _ = train(range(T - C - H + 1), epochs, seed)
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        ctx = torch.tensor(train_raw[-C:], dtype=torch.float32).view(1, C)   # 1957-1958, the conditioning range
        nu0 = 1 + ctx.mean()
        x_ctx = torch.tensor(month[T - C:T], dtype=torch.float32).view(1, C, 1)
        _, _, state = model(torch.cat([torch.zeros(1, 1), ctx[:, :-1]], dim=1) / nu0, x_ctx)
        state = tuple(s.repeat(1, S, 1) for s in state)                      # one copy of the state per path
        z_last, paths = ctx[:, -1].repeat(S) / nu0, []
        for t in range(T, T + H):                                            # ancestral sampling, month by month
            mu, sigma, state = model(z_last.view(S, 1), torch.full((S, 1, 1), float(month[t])), state)
            draw = torch.normal(mu.view(S) * nu0, sigma.view(S) * nu0, generator=gen)
            paths.append(draw)
            z_last = draw / nu0                                              # each draw is the next input
    return torch.stack(paths, dim=1).numpy(), epochs, val_nll               # S paths by 24 months

def rmse(a, b): return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))
def score(paths):                                     # median, and how many months each band contains
    q05, q10, q50, q90, q95 = np.quantile(paths, [0.05, 0.1, 0.5, 0.9, 0.95], axis=0)
    in80 = int(((test_raw >= q10) & (test_raw <= q90)).sum())
    in90 = int(((test_raw >= q05) & (test_raw <= q95)).sum())
    return rmse(test_raw, q50), in80, in90, q90 - q10, q50

print(f"{len(windows(range(T - C - H + 1))[0])} training windows, "
      f"{sum(p.numel() for p in DeepAR().parameters())} weights per network")
pooled, rmses, n80, n90 = [], [], 0, 0
for seed in range(5):
    paths, epochs, val_nll = forecast(seed)
    r, in80, in90, w80, _ = score(paths)
    pooled.append(paths); rmses.append(r); n80 += in80; n90 += in90
    print(f"seed {seed}: {epochs} epochs (validation NLL {val_nll:.3f}); median RMSE {r:.2f}; "
          f"inside 80% band {in80}/24, 90% band {in90}/24; 80% width {w80[0]:.1f} at 1 month, {w80[-1]:.1f} at 24")
print(f"5 seeds: median RMSE {np.mean(rmses):.2f} +/- {np.std(rmses):.2f}; "
      f"inside 80% band {n80} of 120, 90% band {n90} of 120")
r, in80, in90, w80, q50 = score(np.concatenate(pooled))
print(f"{5 * 200} pooled paths: median RMSE {r:.2f}; inside 80% band {in80}/24, 90% band {in90}/24; "
      f"mean 80% width {w80.mean():.1f}")
print("pooled median, July and August 1960:", np.round(q50[18:20], 1).tolist(), "actual:", test_raw[18:20].tolist())

# %% [markdown]
# ## 17.5.6 The Temporal Fusion Transformer: covariates, gates, and attention
#
# Example 17-18 runs in its own companion script, 14_tft_bike.py, because its twenty
# trainings would about double this notebook's run time.

# %% [markdown]
# ## 17.6 Prophet and friends: when "good enough, fast" wins

# %% [markdown]
# Example 17-19. A Prophet-style additive model from scratch.

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
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings; warnings.filterwarnings("ignore")

flights = sns.load_dataset("flights")                          # monthly airline passengers, 1949-1960
y = pd.Series(flights["passengers"].to_numpy(float),
              index=pd.date_range("1949-01-01", periods=len(flights), freq="MS"))
yv = y.to_numpy(); n = len(yv); t = np.arange(n)
train, test = yv[:-24], yv[-24:]                               # hold out the last 24 months
ttr = t[:-24]

def fourier(t, period, K):
    cols = [np.ones_like(t, dtype=float), t.astype(float)]      # intercept + linear trend
    for k in range(1, K + 1):
        cols.append(np.sin(2 * np.pi * k * t / period))
        cols.append(np.cos(2 * np.pi * k * t / period))
    return np.column_stack(cols)

K = 4
Xtr = fourier(ttr, 12, K)
coef, *_ = np.linalg.lstsq(Xtr, train, rcond=None)             # least squares: the whole fit
print(f"fitted trend slope = {coef[1]:.3f} thousand passengers/month")

def rmse(a, b): return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))
prophet_fc = fourier(t[-24:], 12, K) @ coef
snaive = [train[-12 + (i % 12)] for i in range(24)]
fit_raw = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
sarima = fit_raw.forecast(24)
print(f"24-month RMSE: seasonal-naive {rmse(test, snaive):.2f} | "
      f"SARIMA {rmse(test, sarima):.2f} | Prophet-style additive {rmse(test, prophet_fc):.2f}")
print(f"SARIMA fit converged: {fit_raw.mle_retvals['converged']}")

print("\nFourier harmonics K vs in-sample fit RMSE of the full additive model:")
for k in [1, 2, 3, 4, 5, 6, 8]:                                # six is the most a 12-month period allows
    X = fourier(ttr, 12, k)
    c, *_ = np.linalg.lstsq(X, train, rcond=None)
    print(f"  K={k}: {X.shape[1]:2d} columns, rank {np.linalg.matrix_rank(X):2d}, "
          f"in-sample RMSE {rmse(train, X @ c):.2f}")

# fitting the same model on log(y) approximates Prophet's multiplicative-seasonality option
clog = np.linalg.lstsq(fourier(ttr, 12, K), np.log(train), rcond=None)[0]
log_fc = np.exp(fourier(t[-24:], 12, K) @ clog)
print(f"\nProphet-style on log(y): 24-month RMSE {rmse(test, log_fc):.2f}")

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
import os as _os, glob as _glob, prophet as _prophet      # environment guard, companion script only
_tbb = _glob.glob(_os.path.join(_os.path.dirname(_prophet.__file__), "stan_model", "cmdstan-*",
                                "stan", "lib", "stan_math", "lib", "tbb"))
if _tbb:                                                  # Windows: put Prophet's bundled TBB folder first on PATH
    _os.environ["PATH"] = _tbb[0] + _os.pathsep + _os.environ.get("PATH", "")

# %% [markdown]
# Example 17-20. The Prophet library on the airline split, with the coverage of its default interval.

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
from prophet import Prophet

dfp = pd.DataFrame({"ds": y.index[:-24], "y": train})
m = Prophet(seasonality_mode="multiplicative", weekly_seasonality=False, daily_seasonality=False)
m.fit(dfp)
np.random.seed(0)                       # predict() draws its interval simulations from NumPy's global generator
fc_frame = m.predict(pd.DataFrame({"ds": y.index[-24:]}))
fc_real = fc_frame["yhat"].to_numpy()
print(f"real Prophet (multiplicative mode): 24-month RMSE {rmse(test, fc_real):.2f}")
lo, hi = fc_frame["yhat_lower"].to_numpy(), fc_frame["yhat_upper"].to_numpy()
inside = int(((test >= lo) & (test <= hi)).sum())
print(f"holdout months inside the default 80% interval: {inside} of 24, mean width {np.mean(hi - lo):.1f}")

# %% [markdown]
# ## 17.7.1 The pinball loss

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
import numpy as np

def pinball(y, q, tau):
    e = np.asarray(y) - np.asarray(q)
    return float(np.mean(np.maximum(tau * e, (tau - 1) * e)))

rng = np.random.default_rng(0)                              # illustrative arithmetic on a plain sample
sample = rng.normal(100, 15, 5000); tau = 0.9
grid = np.linspace(90, 140, 501)
best_q = grid[np.argmin([pinball(sample, q, tau) for q in grid])]
print(f"tau=0.9: pinball minimizer {best_q:.1f} = empirical 90th percentile {np.quantile(sample, tau):.1f}")
ten = np.arange(1.0, 11.0)                                  # the values 1, 2, ..., 10, so tau * n = 9
print("ten values, average loss at q = 8, 9, 9.5, 10, 11:",
      [round(pinball(ten, q, tau), 3) for q in (8, 9, 9.5, 10, 11)])

# %% [markdown]
# ## 17.7.2 Prediction intervals and calibration

# %% [markdown]
# Example 17-21. Quantile forecasting and prediction-interval coverage.

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
import numpy as np, pandas as pd, seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings; warnings.filterwarnings("ignore")

def pinball(y, q, tau):
    e = np.asarray(y) - np.asarray(q)
    return float(np.mean(np.maximum(tau * e, (tau - 1) * e)))

flights = sns.load_dataset("flights")
yv = flights["passengers"].to_numpy(float)
feats = ["lag1", "lag12", "month"]

def supervised(values):
    d = pd.DataFrame({"y": np.asarray(values)}); t = np.arange(len(d))
    d["lag1"] = d["y"].shift(1); d["lag12"] = d["y"].shift(12); d["month"] = t % 12
    return d.dropna().reset_index(drop=True)

def gbq(alpha, X, yy):
    return GradientBoostingRegressor(loss="quantile", alpha=alpha, n_estimators=300,
                                     max_depth=2, learning_rate=0.05, random_state=0).fit(X, yy)

def gb_quantiles(d):                                        # fit tau = 0.05, 0.5, 0.95 and score all three
    c = len(d) - 24
    Xtr, ytr = d[feats].iloc[:c], d["y"].iloc[:c]
    Xte, yte = d[feats].iloc[c:], d["y"].iloc[c:].values
    lo = gbq(0.05, Xtr, ytr).predict(Xte)
    med = gbq(0.50, Xtr, ytr).predict(Xte)
    hi = gbq(0.95, Xtr, ytr).predict(Xte)
    cov = float(np.mean((yte >= lo) & (yte <= hi)))         # how often [q05, q95] contains the actual
    width = float(np.mean(hi - lo))                          # average interval width
    med_mae = float(np.mean(np.abs(yte - med)))              # MAE of the tau=0.5 point forecast
    avg_pin = float(np.mean([pinball(yte, lo, 0.05), pinball(yte, med, 0.50),
                             pinball(yte, hi, 0.95)]))       # mean pinball over the three quantiles
    return cov, width, med_mae, avg_pin, lo, hi

cov_raw, w_raw, mae_raw, pin_raw, lo_raw, hi_raw = gb_quantiles(supervised(yv))   # trees on RAW flights
print("tree 90% interval [q05, q95] on the 24-month flights holdout:")
print(f"  RAW flights     : coverage {cov_raw*100:3.0f}%   width {w_raw:.1f}   "
      f"median MAE {mae_raw:.2f}   avg pinball {pin_raw:.2f}")
train_max, test_raw = yv[:-24].max(), yv[-24:]              # highest training month, and the 24 test months
print(f"    training max {train_max:.0f}, upper edge peaks at {hi_raw.max():.1f}; test months above "
      f"{train_max:.0f}: {int((test_raw > train_max).sum())}, above the upper edge: "
      f"{int((test_raw > hi_raw).sum())}, below the lower edge: {int((test_raw < lo_raw).sum())}")
logdiff = np.r_[np.nan, np.diff(np.log(yv))]
cov_sd, w_sd, mae_sd, pin_sd, _, _ = gb_quantiles(supervised(logdiff))    # trees on the stationary log-diffs
print(f"  log-differenced : coverage {cov_sd*100:3.0f}%   (log-return units: width {w_sd:.3f}, "
      f"median MAE {mae_sd:.3f})")

train, test = yv[:-24], yv[-24:]
res = SARIMAX(np.log(train), order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
ci = res.get_forecast(24).conf_int(alpha=0.10)              # native interval that extrapolates
lo_s, hi_s = np.exp(ci[:,0]), np.exp(ci[:,1])
cov_s = float(np.mean((test >= lo_s) & (test <= hi_s)))
w_s = hi_s - lo_s                                           # band width at each of the 24 steps
months = pd.date_range("1949-01-01", periods=len(yv), freq="MS")[-24:]   # January 1959 to December 1960
print("native and conformal 90% intervals on the same holdout:")
print(f"  SARIMAX(log) get_forecast : coverage {cov_s*100:3.0f}%   width {w_s.mean():.1f}")
print(f"    converged {res.mle_retvals['converged']}; width {w_s[0]:.1f} one month ahead, {w_s[-1]:.1f} at 24, "
      f"widest {w_s.max():.1f} in {months[w_s.argmax()]:%b %Y}")

d = supervised(logdiff)                                     # split-conformal on log-diff residuals
te, cal, tr = d.iloc[-24:], d.iloc[-48:-24], d.iloc[:-48]
pm = GradientBoostingRegressor(n_estimators=300, max_depth=2, learning_rate=0.05, random_state=0).fit(tr[feats], tr["y"])
scores = np.abs(cal["y"].values - pm.predict(cal[feats]))
qhat = np.quantile(scores, np.ceil(25 * 0.9) / 24)          # conformal quantile level
pred = pm.predict(te[feats])
cov_c = np.mean((te["y"].values >= pred - qhat) & (te["y"].values <= pred + qhat))
print(f"  split-conformal (log-diff): coverage {cov_c*100:3.0f}%   half-width {qhat:.3f}")

# %% [markdown]
# ## 17.8 Anomaly detection in time series

# %% [markdown]
# Example 17-22. Anomaly detection on raw values versus on residuals.

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
import numpy as np, pandas as pd, seaborn as sns
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings; warnings.filterwarnings("ignore")

flights = sns.load_dataset("flights")
idx = pd.date_range("1949-01-01", periods=len(flights), freq="MS")
yv = flights["passengers"].to_numpy(float).copy()
n = len(yv); truth = np.zeros(n, dtype=bool)
yv[54] += 300; truth[54] = True                      # point spike
yv[128] -= 230; truth[128] = True                    # point dip
yv[82] = 470; truth[82] = True                       # CONTEXTUAL: summer-peak level in a November trough
s = pd.Series(yv, index=idx)
print(f"injected anomalies at indices {np.where(truth)[0].tolist()}  (index 82 is the contextual one)")

z_global = (yv - yv.mean()) / yv.std()
flag_global = np.abs(z_global) > 3
res = STL(s, period=12, robust=True).fit().resid.values
z_res = (res - res.mean()) / res.std()
flag_res = np.abs(z_res) > 3

def pr(flag):
    tp = int((flag & truth).sum()); fp = int((flag & ~truth).sum()); fn = int((~flag & truth).sum())
    prec = tp/(tp+fp) if tp+fp else 0.0; rec = tp/(tp+fn) if tp+fn else 0.0
    return f"TP={tp} FP={fp} FN={fn}  precision={prec:.2f} recall={rec:.2f}"

print("global 3-sigma :", pr(flag_global))
print("STL-residual   :", pr(flag_res))
print(f"overall std {yv.std():.1f}, so the +300 spike at index 54 has a raw z of only {z_global[54]:+.2f}")
print(f"contextual (Nov 1955): raw z={z_global[82]:+.2f} (MISSED), residual z={z_res[82]:+.2f} (caught)")

w = 12                                                # model-free rolling z-score
zr = (s - s.shift(1).rolling(w).mean()) / s.shift(1).rolling(w).std()
flag_roll = (zr.abs() > 3).fillna(False).values
print("rolling z (w=12):", pr(flag_roll))
for i in np.where(flag_roll)[0]:
    print(f"  flags {s.index[i]:%b %Y}: value {yv[i]:.0f}, rolling z {zr.iloc[i]:+.2f}")
print(f"  misses the dip, {s.index[128]:%b %Y}: rolling z {zr.iloc[128]:+.2f}")

m = SARIMAX(np.log(yv), order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
ci = m.get_prediction().conf_int(alpha=0.01)          # prediction-interval detector (99% band)
band_lo, band_hi = np.exp(ci[:,0]), np.exp(ci[:,1])
flag_pi = (yv < band_lo) | (yv > band_hi); flag_pi[:13] = False
print("SARIMAX 99% PI :", pr(flag_pi), f"  (optimizer converged: {m.mle_retvals['converged']})")

# %% [markdown]
# ## 17.8.1 Collective anomalies over sliding windows

# %% [markdown]
# Example 17-23. A collective anomaly found by window statistics and by an Isolation Forest over windows.

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
import numpy as np, pandas as pd, seaborn as sns
from statsmodels.tsa.seasonal import STL
from sklearn.ensemble import IsolationForest

flights = sns.load_dataset("flights")
idx = pd.date_range("1949-01-01", periods=len(flights), freq="MS")
clean = flights["passengers"].to_numpy(float)
yv = clean.copy()
yv[96:108] = yv[96:108].mean()                      # COLLECTIVE: 1957 keeps its level but loses its season
z_raw = (yv - yv.mean()) / yv.std()
print(f"1957 ran from {clean[96:108].min():.0f} to {clean[96:108].max():.0f}; flattened to {yv[96]:.1f}, "
      f"raw z of every 1957 month = {z_raw[96]:+.2f}")

W = 12
def resid(x):                                       # STL residual: what the trend and season do not explain
    return STL(pd.Series(x, index=idx), period=12, robust=True).fit().resid.values
def windows(x):                                     # one row per start month: W consecutive values
    return np.array([x[i:i + W] for i in range(len(x) - W + 1)])
def span(i):
    return f"{idx[i]:%b %Y} to {idx[i + W - 1]:%b %Y}"

res = resid(yv)
z_res = (res - res.mean()) / res.std()
print("pointwise STL-residual rule flags:", [f"{d:%b %Y}" for d in idx[np.abs(z_res) > 3]])

rms = np.sqrt((windows(res) ** 2).mean(axis=1))      # window statistic: RMS residual over twelve months
rms_clean = np.sqrt((windows(resid(clean)) ** 2).mean(axis=1))
top, hot = int(np.argmax(rms)), np.where(rms > 3 * np.median(rms))[0]
print(f"window RMS: top window {span(top)} at {rms[top]:.2f}, {rms[top] / np.median(rms):.1f} x the median; "
      f"{len(hot)} windows above 3 x median, starting {idx[hot[0]]:%b %Y} to {idx[hot[-1]]:%b %Y}")
print(f"clean series: top window {rms_clean.max() / np.median(rms_clean):.1f} x its median, "
      f"{int((rms_clean > 3 * np.median(rms_clean)).sum())} windows above 3 x median")

shape = windows(np.log(yv))
shape = shape - shape.mean(axis=1, keepdims=True)   # raw windows with each window's own level removed
for name, rows in [("raw shape windows", shape), ("residual windows", windows(res))]:
    score = -IsolationForest(n_estimators=500, random_state=0).fit(rows).score_samples(rows)
    print(f"Isolation Forest on {name}: top window {span(int(np.argmax(score)))}; "
          f"the Jan 1957 to Dec 1957 window ranks {int((score > score[96]).sum()) + 1} of {len(score)}")

# %% [markdown]
# ## 17.9.1 The data and the three contenders

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
import pandas as pd
bike = pd.read_csv("data/bike_sharing_daily.csv", parse_dates=["dteday"]).set_index("dteday")
y = bike["cnt"].astype(float)
print(f"highest day {y.idxmax():%d %B %Y}: {y.max():.0f} rentals")
trend = y.rolling(30, center=True, min_periods=15).mean()
rem = (y - trend).dropna()
dow = rem.groupby(rem.index.dayofweek).mean()
print("day-of-week mean remainder, Monday to Sunday:", [round(float(v), 1) for v in dow])
print(f"day-of-week span: {float(dow.max() - dow.min()):.1f} rentals")

# %% [markdown]
# ## 17.9.3 Reading the results like a practitioner

# %% [markdown]
# Example 17-24. Forecasting daily bike demand three ways with an expanding-window backtest.

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
import numpy as np
import pandas as pd
from xgboost import XGBRegressor              # import before torch: env libomp must load first on this Mac
import torch
import torch.nn as nn
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings; warnings.filterwarnings("ignore")

torch.manual_seed(0)
torch.set_num_threads(1)

bike = pd.read_csv("data/bike_sharing_daily.csv", parse_dates=["dteday"]).set_index("dteday")
y = bike["cnt"].astype(float)                            # 731 real daily rental counts
m = 7                                                    # weekly seasonal period
H = 14                                                   # forecast horizon: two weeks

def mae(a, b):  return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))
def rmse(a, b): return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))
def mase_scale(train, m=7):                              # in-sample seasonal-naive MAE
    a = np.asarray(train, dtype=float)
    return float(np.mean(np.abs(a[m:] - a[:-m])))

sarima_converged = []                                    # optimizer flag of each SARIMA fit, one per origin
def f_sarima(train, h):
    mdl = SARIMAX(np.asarray(train, float), order=(1, 1, 1),
                  seasonal_order=(1, 1, 1, 7)).fit(disp=False)
    sarima_converged.append(bool(mdl.mle_retvals["converged"]))
    return np.asarray(mdl.forecast(h))

def make_feats(hist, p):
    return [hist[p - 1], hist[p - 7], hist[p - 14], np.mean(hist[p - 7:p]), p % 7]

def f_xgb(train, h):
    arr = np.asarray(train, dtype=float)
    X = [make_feats(arr, p) for p in range(14, len(arr))]
    Y = arr[14:]
    model = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, n_jobs=2)
    model.fit(np.array(X), Y)
    hist = list(arr); preds = []
    for _ in range(h):                                   # recursive: append each prediction
        p = len(hist)
        pred = float(model.predict(np.array([make_feats(np.array(hist), p)]))[0])
        preds.append(pred); hist.append(pred)
    return np.array(preds)

def f_lstm(train, h, L=28):
    arr = np.asarray(train, dtype=float)
    mu, sd = arr.mean(), arr.std()
    sc = (arr - mu) / sd
    X, Yt = [], []
    for i in range(len(sc) - L - h + 1):
        X.append(sc[i:i + L]); Yt.append(sc[i + L:i + L + h])
    X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1)
    Yt = torch.tensor(np.array(Yt), dtype=torch.float32)
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(1, 32, batch_first=True); self.head = nn.Linear(32, h)
        def forward(self, x):
            o, _ = self.lstm(x); return self.head(o[:, -1, :])
    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=0.01); lf = nn.MSELoss()
    for _ in range(250):
        opt.zero_grad(); loss = lf(net(X), Yt); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        x = torch.tensor(sc[-L:], dtype=torch.float32).view(1, L, 1)
        return net(x).numpy().ravel() * sd + mu

N = len(y)
origins = [N - 4 * H, N - 3 * H, N - 2 * H, N - 1 * H]
models = ["seasonal-naive", "SARIMA", "XGBoost", "LSTM"]
rmse_res = {k: [] for k in models}
mase_res = {k: [] for k in models}
for cut in origins:
    train, actual = y.iloc[:cut], y.iloc[cut:cut + H].values
    scale = mase_scale(train, m)
    snaive = np.array([y.iloc[cut - 7 + (i % 7)] for i in range(H)])
    preds = {"seasonal-naive": snaive, "SARIMA": f_sarima(train, H),
             "XGBoost": f_xgb(train, H), "LSTM": f_lstm(train, H)}
    for name, p in preds.items():
        rmse_res[name].append(rmse(actual, p))
        mase_res[name].append(mae(actual, p) / scale)

print(f"expanding-window backtest, {len(origins)} folds, horizon {H} days")
print("RMSE per fold:")
naive_rmse = np.mean(rmse_res["seasonal-naive"])
for name in models:
    e = np.array(rmse_res[name])
    v = "" if name == "seasonal-naive" else \
        ("  beats naive" if e.mean() < naive_rmse else "  loses to naive")
    print(f"  {name:14s}: {np.round(e,1)}  mean {e.mean():6.1f} +/- {e.std():5.1f}{v}")
print("MASE per fold (scale = in-sample one-step seasonal-naive MAE; below 1 beats that scale):")
for name in models:
    e = np.array(mase_res[name])
    print(f"  {name:14s}: {np.round(e,2)}  mean {e.mean():5.2f} +/- {e.std():4.2f}")
print("SARIMA optimizer converged, origin by origin:", sarima_converged)

# %% [markdown]
# Example 17-25. Rerunning the lab's LSTM at five seeds and scoring each by MASE.

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
torch.set_num_threads(1)                                 # the lab's thread pin, kept for this check
lstm_means = []
for seed in range(5):                                    # five draws of the LSTM's starting weights
    torch.manual_seed(seed)
    folds = [mae(y.iloc[cut:cut + H].values, f_lstm(y.iloc[:cut], H)) / mase_scale(y.iloc[:cut], m)
             for cut in origins]
    lstm_means.append(np.mean(folds))
    print(f"seed {seed}: LSTM MASE per fold {np.round(folds, 2)}  mean {np.mean(folds):.2f}")
runs = np.array(lstm_means)
print(f"seeds 0 to 4: mean {runs.mean():.2f}, lowest {runs.min():.2f}, highest {runs.max():.2f}; "
      f"seasonal-naive {np.mean(mase_res['seasonal-naive']):.2f}")

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
import pandas as pd
bike = pd.read_csv("data/bike_sharing_daily.csv", parse_dates=["dteday"])  # lab task starter
print(bike.columns[:2].tolist())
def min_day(df):
    ...

# %% [markdown]
# ## Coding exercises

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
import numpy as np
import pandas as pd
bike = pd.read_csv("data/bike_sharing_daily.csv", parse_dates=["dteday"])  # exercise 1 starter
print(bike.shape)
def acf(x, lag):
    ...

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
import pandas as pd
bike = pd.read_csv("data/bike_sharing_daily.csv", parse_dates=["dteday"])  # exercise 2 starter
print(bike["cnt"].head(3).tolist())
def adf_report(x):
    ...

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
import pandas as pd
bike = pd.read_csv("data/bike_sharing_daily.csv", parse_dates=["dteday"]).set_index("dteday")  # exercise 3 starter
print(bike.index.min().date(), bike.index.max().date())
def stl_check(y):
    ...

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
import seaborn as sns
flights = sns.load_dataset("flights")  # exercise 4 starter
print(len(flights))
def smoothing_rmse(y, holdout=36):
    ...

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
import seaborn as sns
flights = sns.load_dataset("flights")  # exercise 5 starter
print(flights["passengers"].iloc[0])
def arima_board(y, holdout=12):
    ...

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
import numpy as np
import seaborn as sns
y = sns.load_dataset("flights")["passengers"].to_numpy(float)  # exercise 6 starter
print(y[-12:].shape)
def metrics(actual, fcst, scale):
    ...

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
import seaborn as sns
flights = sns.load_dataset("flights")  # exercise 7 starter
print(flights.columns.tolist())
def lag_table(y):
    ...

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
import numpy as np
import pandas as pd
bike = pd.read_csv("data/bike_sharing_daily.csv", parse_dates=["dteday"])  # exercise 8 starter
print(int(bike["cnt"].sum()))
def fourier_design(t, Kw, Ky=4):
    ...

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
import seaborn as sns
flights = sns.load_dataset("flights")  # exercise 9 starter
print(int(flights["passengers"].max()))
def coverage(y, lo=0.1, hi=0.9):
    ...

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
import seaborn as sns
flights = sns.load_dataset("flights")  # exercise 10 starter
print(int(flights["passengers"].min()), int(flights["passengers"].max()))
def backtest(y):
    ...

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
import numpy as np
import pandas as pd
bike = pd.read_csv("data/bike_sharing_daily.csv", parse_dates=["dteday"])  # exercise 11 starter
y = bike["cnt"].to_numpy(float)
print([len(y) - k * 14 for k in (4, 3, 2, 1)])
def zero_shot_mase(y, origins, h=14):
    ...
