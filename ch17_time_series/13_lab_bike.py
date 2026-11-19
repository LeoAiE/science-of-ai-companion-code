# %% [markdown]
# # Section 17.9 Lab: forecast bike demand three ways (Example 17-24)
#
# The code cell below is Example 17-24 of the book, verbatim. It reads the real UCI Bike Sharing
# daily counts bundled as data/bike_sharing_daily.csv and prints the RMSE and MASE tables that
# Section 17.9 reports.

# %%
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")  # torch + xgboost OpenMP clash on this Mac (PLAN section 4)
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
