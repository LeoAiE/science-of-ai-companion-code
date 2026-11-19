# %% [markdown]
# # Section 17.5.6: a compact Temporal Fusion Transformer on the bike data (Example 17-18)
#
# The code cell below is Example 17-18 of the book, verbatim. It reads the real UCI Bike Sharing
# daily counts bundled as data/bike_sharing_daily.csv, trains the network at the four origins of the
# lab in Section 17.9 with seeds 0 to 4, and prints the backtest, the variable selection weights, and
# the attention summary that Section 17.5.6 reports. It runs in its own script, not in the walkthrough,
# because its twenty trainings would about double the walkthrough notebook's run time.

# %%
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

torch.manual_seed(0)
torch.set_num_threads(1)
bike = pd.read_csv("data/bike_sharing_daily.csv", parse_dates=["dteday"])
y = bike["cnt"].to_numpy(float)                           # 731 daily rental counts, 2011 and 2012
known = {"weekday": 7, "workingday": 2, "holiday": 2, "mnth": 12, "season": 4}   # calendar: known in advance
observed = ["temp", "hum", "windspeed"]                   # weather: observed, so only past days are used
codes = np.column_stack([bike["weekday"], bike["workingday"], bike["holiday"], bike["mnth"] - 1, bike["season"] - 1])
weather = bike[observed].to_numpy(float)
K, H, d = 28, 14, 20                                      # 28 past days, 14 future days, state size 20
Q = torch.tensor([0.1, 0.5, 0.9])                         # the three quantiles the network outputs

class GLU(nn.Module):                                     # a sigmoid gate times a linear value
    def __init__(self, n):
        super().__init__()
        self.gate, self.value = nn.Linear(n, n), nn.Linear(n, n)
    def forward(self, g):
        return torch.sigmoid(self.gate(g)) * self.value(g)

class GRN(nn.Module):                                     # gated residual network: LayerNorm(a + GLU(eta1))
    def __init__(self, n_in, n_out=d):
        super().__init__()
        self.skip = nn.Identity() if n_in == n_out else nn.Linear(n_in, n_out)
        self.w2, self.w1 = nn.Linear(n_in, d), nn.Linear(d, n_out)
        self.drop, self.glu, self.norm = nn.Dropout(0.1), GLU(n_out), nn.LayerNorm(n_out)
    def forward(self, a):
        eta1 = self.w1(nn.functional.elu(self.w2(a)))
        return self.norm(self.skip(a) + self.glu(self.drop(eta1)))

class VariableSelection(nn.Module):                       # softmax weights over m inputs, then a weighted sum
    def __init__(self, m):
        super().__init__()
        self.weights = GRN(m * d, m)
        self.each = nn.ModuleList([GRN(d) for _ in range(m)])
    def forward(self, xi):                                # xi: (batch, days, m inputs, d)
        v = torch.softmax(self.weights(xi.flatten(-2)), dim=-1)
        processed = torch.stack([grn(xi[..., j, :]) for j, grn in enumerate(self.each)], dim=-2)
        return (v.unsqueeze(-1) * processed).sum(-2), v

class InterpretableAttention(nn.Module):                  # every head shares one value projection
    def __init__(self, heads=4):
        super().__init__()
        a = d // heads
        self.q = nn.ModuleList([nn.Linear(d, a) for _ in range(heads)])
        self.k = nn.ModuleList([nn.Linear(d, a) for _ in range(heads)])
        self.v, self.out, self.scale = nn.Linear(d, a), nn.Linear(a, d), a ** 0.5
    def forward(self, query, key, mask):
        A = torch.stack([torch.softmax(q(query) @ k(key).transpose(-1, -2) / self.scale + mask, dim=-1)
                         for q, k in zip(self.q, self.k)]).mean(0)          # average the heads' weights
        return self.out(A @ self.v(key)), A

class TFT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.ModuleList([nn.Embedding(n, d) for n in known.values()])       # categorical inputs
        self.lin = nn.ModuleList([nn.Linear(1, d) for _ in range(1 + len(observed))])  # count and weather
        self.select_past = VariableSelection(1 + len(observed) + len(known))
        self.select_future = VariableSelection(len(known))
        self.encoder, self.decoder = nn.LSTM(d, d, batch_first=True), nn.LSTM(d, d, batch_first=True)
        self.gate1, self.norm1 = GLU(d), nn.LayerNorm(d)
        self.attention = InterpretableAttention()
        self.gate2, self.norm2 = GLU(d), nn.LayerNorm(d)
        self.feed_forward = GRN(d)
        self.gate3, self.norm3 = GLU(d), nn.LayerNorm(d)
        self.head = nn.Linear(d, len(Q))                                    # one output per quantile
        self.register_buffer("mask", torch.full((H, K + H), float("-inf")).triu(K + 1))  # no later days
    def forward(self, past_cont, cat):                    # past_cont: (batch, K, 4), cat: (batch, K + H, 5)
        e_cat = torch.stack([emb(cat[..., j]) for j, emb in enumerate(self.embed)], dim=-2)
        e_cont = torch.stack([lin(past_cont[..., j:j + 1]) for j, lin in enumerate(self.lin)], dim=-2)
        past, v_past = self.select_past(torch.cat([e_cont, e_cat[:, :K]], dim=-2))
        future, v_future = self.select_future(e_cat[:, K:])
        enc, state = self.encoder(past)                   # local processing of the past days...
        dec, _ = self.decoder(future, state)              # ...and of the known future days
        phi = self.norm1(torch.cat([past, future], 1) + self.gate1(torch.cat([enc, dec], 1)))
        beta, A = self.attention(phi[:, K:], phi, self.mask)                # each future day looks back
        delta = self.norm2(phi[:, K:] + self.gate2(beta))
        psi = self.norm3(phi[:, K:] + self.gate3(self.feed_forward(delta)))
        return self.head(psi), v_past, v_future, A

def quantile_loss(pred, target):                          # the pinball loss, summed over the three quantiles
    e = target.unsqueeze(-1) - pred
    return torch.maximum(Q * e, (Q - 1) * e).sum(-1).mean()

def tft_forecast(cut, seed, max_epochs=60, patience=10):
    torch.manual_seed(seed)                               # starting weights, dropout, and batch order
    mu, sd = y[:cut].mean(), y[:cut].std()                # scaling statistics from the training days only
    wmu, wsd = weather[:cut].mean(0), weather[:cut].std(0)
    cont = torch.tensor(np.column_stack([(y - mu) / sd, (weather - wmu) / wsd]), dtype=torch.float32)
    cat = torch.tensor(codes)
    def batch(starts):                                    # windows of K past days and H future days
        win = torch.as_tensor(starts)[:, None] + torch.arange(K + H)
        return cont[win[:, :K]], cat[win], cont[win[:, K:], 0]
    starts = list(range(cut - 2 * H - K + 1))             # training targets end before the last fortnight,
    val = batch([cut - H - K])                            # which serves as the validation window
    model = TFT()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best, best_state, wait = np.inf, None, 0
    for epoch in range(max_epochs):
        model.train()
        for idx in torch.randperm(len(starts)).split(64):
            past, k, target = batch([starts[j] for j in idx.tolist()])
            opt.zero_grad()
            loss = quantile_loss(model(past, k)[0], target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            v = quantile_loss(model(*val[:2])[0], val[2]).item()
        if v < best:
            best, best_state, wait = v, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait == patience:                          # early stopping
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        past, k, _ = batch([cut - K])                     # the last 28 days and the calendar of the next 14
        pred, v_past, v_future, A = model(past, k)
    return pred[0].numpy() * sd + mu, v_past[0].mean(0).numpy(), v_future[0].mean(0).numpy(), A[0, 0].numpy()

def mase_scale(train, m=7):                               # the lab's scale: in-sample seasonal-naive MAE
    a = np.asarray(train, dtype=float)
    return float(np.mean(np.abs(a[m:] - a[:-m])))
def rmse(a, b): return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))

N = len(y)
origins = [N - 4 * H, N - 3 * H, N - 2 * H, N - H]        # the four folds of the lab in Section 17.9
seeds = range(5)
naive_r, naive_m = [], []
for cut in origins:
    snaive = np.array([y[cut - 7 + (i % 7)] for i in range(H)])
    naive_r.append(rmse(y[cut:cut + H], snaive))
    naive_m.append(np.mean(np.abs(y[cut:cut + H] - snaive)) / mase_scale(y[:cut]))
R, M = np.zeros((len(seeds), len(origins))), np.zeros((len(seeds), len(origins)))
inside, widths, vp, vf, attn = 0, [], [], [], []
for s in seeds:
    for f, cut in enumerate(origins):
        actual = y[cut:cut + H]
        q, v_past, v_future, a = tft_forecast(cut, s)
        R[s, f] = rmse(actual, q[:, 1])
        M[s, f] = np.mean(np.abs(actual - q[:, 1])) / mase_scale(y[:cut])
        inside += int(((actual >= q[:, 0]) & (actual <= q[:, 2])).sum())
        widths.append(np.mean(q[:, 2] - q[:, 0]))
        vp.append(v_past); vf.append(v_future); attn.append(a)

print(f"backtest, 4 folds, horizon {H} days, seeds 0 to 4; {sum(p.numel() for p in TFT().parameters())} weights per fit")
print(f"  {'seasonal-naive':15s}: RMSE {np.round(naive_r, 1)} mean {np.mean(naive_r):6.1f} | "
      f"MASE {np.round(naive_m, 2)} mean {np.mean(naive_m):.2f}")
for s in seeds:
    print(f"  {'TFT, seed ' + str(s):15s}: RMSE {np.round(R[s], 1)} mean {R[s].mean():6.1f} | "
          f"MASE {np.round(M[s], 2)} mean {M[s].mean():.2f}")
print(f"  {'TFT, 5 seeds':15s}: RMSE {np.round(R.mean(0), 1)} mean {R.mean():6.1f} | "
      f"MASE {np.round(M.mean(0), 2)} mean {M.mean():.2f}")
print(f"  spread of the five seed means: RMSE +/- {R.mean(1).std():.1f}, MASE +/- {M.mean(1).std():.2f}")
print(f"  10% to 90% band: {inside} of {len(seeds) * len(origins) * H} days inside, "
      f"mean width {np.mean(widths):.0f} rentals")
names = ["cnt"] + observed + list(known)
print("past-input weights, mean of 20 fits:  ", {n: round(float(w), 3) for n, w in zip(names, np.mean(vp, 0))})
print("future-input weights, mean of 20 fits:", {n: round(float(w), 3) for n, w in zip(known, np.mean(vf, 0))})
a = np.mean(attn, 0)[:K]                                  # one-day-ahead weights; position j is K - j days back
weekly = [K - 7, K - 14, K - 21, K - 28]
print(f"one-day-ahead attention over the 28 past days, mean of 20 fits: lags 7, 14, 21, 28 average "
      f"{a[weekly].mean():.3f}, other lags {np.delete(a, weekly).mean():.3f}; largest {a.max():.3f} at lag "
      f"{K - int(a.argmax())}")
