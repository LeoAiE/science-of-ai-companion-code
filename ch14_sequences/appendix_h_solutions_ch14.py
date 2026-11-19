"""Appendix H solutions for Chapter 14, as runnable cells.

Generated from Solutions.md (Appendix H). Every code block of this chapter's solutions appears here
verbatim, split into one cell per exercise or drill topic, so each answer can be run and checked against
the expected outputs it states. Keep Solutions.md as the source; regenerate after any change.
"""

# %% [markdown]
# ## 14.11 coding exercises

# %%
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

# %% [markdown]
# 1. Vanilla RNN forward from scratch, checked against nn.RNN

# %%
# 1. Vanilla RNN forward from scratch, checked against nn.RNN
def rnn_forward(xs, Wxh, Whh, b):
    h = np.zeros(Whh.shape[0]); states = []
    for x in xs:
        h = np.tanh(Wxh @ x + Whh @ h + b)      # same weights every step
        states.append(h)
    return np.array(states)
rng = np.random.default_rng(0); H, D = 5, 3
Wxh = rng.normal(0, 0.5, (H, D)); Whh = rng.normal(0, 0.5, (H, H)); b = np.zeros(H)
for length in [4, 9]:
    st = rnn_forward(rng.normal(size=(length, D)), Wxh, Whh, b)
    print(f"length {length}: {st.shape[0]} states of dim {st.shape[1]}")
# expected output: length 4: 4 states of dim 5
# expected output: length 9: 9 states of dim 5
torch_rnn = nn.RNN(D, H, batch_first=True)
with torch.no_grad():
    torch_rnn.weight_ih_l0.copy_(torch.tensor(Wxh, dtype=torch.float32))
    torch_rnn.weight_hh_l0.copy_(torch.tensor(Whh, dtype=torch.float32))
    torch_rnn.bias_ih_l0.zero_(); torch_rnn.bias_hh_l0.zero_()   # zero BOTH bias vectors
xs = rng.normal(size=(6, D)); mine = rnn_forward(xs, Wxh, Whh, b)
with torch.no_grad():
    theirs, _ = torch_rnn(torch.tensor(xs, dtype=torch.float32).unsqueeze(0))
print("matches nn.RNN:", np.allclose(mine, theirs[0].numpy(), atol=1e-5))
# expected output: matches nn.RNN: True

# %% [markdown]
# 2. Vanishing vs controllable gradient: product of Jacobians vs forget-gate^T

# %%
# 2. Vanishing vs controllable gradient: product of Jacobians vs forget-gate^T
Hg = 4
def grad_norm(spectral, T, seed=0):
    rng = np.random.default_rng(seed)                  # self-seeded as in 14.3: one fixed matrix per spectral norm
    M = rng.normal(size=(Hg, Hg)); M = spectral * M / np.linalg.norm(M, 2)   # fix spectral norm
    h = rng.normal(size=Hg) * 0.1; J = np.eye(Hg)
    for _ in range(T):
        h = np.tanh(M @ h); J = (np.diag(1 - h ** 2) @ M) @ J
    return np.linalg.norm(J, 2)
for T in [10, 30, 60]:
    print(f"T={T:>3}  rnn 0.5={grad_norm(0.5,T):.2e}  "
          f"rnn 0.9={grad_norm(0.9,T):.2e}  lstm f=0.99 -> {0.99**T:.3e}")
# expected output: T= 10  rnn 0.5=6.22e-06  rnn 0.9=2.22e-03  lstm f=0.99 -> 9.044e-01
# expected output: T= 30  rnn 0.5=1.42e-16  rnn 0.9=6.45e-09  lstm f=0.99 -> 7.397e-01
# expected output: T= 60  rnn 0.5=1.61e-32  rnn 0.9=3.33e-17  lstm f=0.99 -> 5.472e-01

# %% [markdown]
# 3. LSTM cell from scratch, checked against nn.LSTM

# %%
# 3. LSTM cell from scratch, checked against nn.LSTM
def lstm_step(x, h, c, P):
    z = P["Wx"] @ x + P["Wh"] @ h + P["b"]
    i, f, g, o = np.split(z, 4)               # PyTorch gate order: input, forget, cell, output
    sig = lambda v: 1 / (1 + np.exp(-v))
    i, f, g, o = sig(i), sig(f), np.tanh(g), sig(o)
    c = f * c + i * g                          # additive cell update
    h = o * np.tanh(c)
    return h, c
rng = np.random.default_rng(1); Hd, Dd = 4, 3
lstm = nn.LSTM(Dd, Hd, batch_first=True)
P = {"Wx": lstm.weight_ih_l0.detach().numpy(),
     "Wh": lstm.weight_hh_l0.detach().numpy(),
     "b": (lstm.bias_ih_l0 + lstm.bias_hh_l0).detach().numpy()}   # PyTorch keeps two biases; sum them
xs = rng.normal(size=(5, Dd)); h, c = np.zeros(Hd), np.zeros(Hd)
for x in xs:
    h, c = lstm_step(x, h, c, P)
with torch.no_grad():
    _, (hn, cn) = lstm(torch.tensor(xs, dtype=torch.float32).unsqueeze(0))
print("scratch h matches nn.LSTM:", np.allclose(h, hn[0, 0].numpy(), atol=1e-5))
print("scratch c matches nn.LSTM:", np.allclose(c, cn[0, 0].numpy(), atol=1e-5))
# expected output: scratch h matches nn.LSTM: True
# expected output: scratch c matches nn.LSTM: True

# %% [markdown]
# 4. GRU from scratch and the parameter ladder

# %%
# 4. GRU from scratch and the parameter ladder
def gru_step(x, h, P):
    sig = lambda v: 1 / (1 + np.exp(-v))
    z = sig(P["Wz"] @ x + P["Uz"] @ h + P["bz"])        # update gate
    r = sig(P["Wr"] @ x + P["Ur"] @ h + P["br"])        # reset gate
    ht = np.tanh(P["Wh"] @ x + P["Uh"] @ (r * h) + P["bh"])
    return (1 - z) * h + z * ht                          # interpolate old vs candidate
rng = np.random.default_rng(0); Hd, Dd = 4, 3
def W(a, b): return rng.normal(0, 0.3, (a, b))
Pg = {"Wz": W(Hd, Dd), "Uz": W(Hd, Hd), "bz": np.zeros(Hd),
      "Wr": W(Hd, Dd), "Ur": W(Hd, Hd), "br": np.zeros(Hd),
      "Wh": W(Hd, Dd), "Uh": W(Hd, Hd), "bh": np.zeros(Hd)}
h = np.zeros(Hd)
for x in rng.normal(size=(5, Dd)):
    h = gru_step(x, h, Pg)
print(f"scratch GRU final hidden norm = {np.linalg.norm(h):.3f}")
def n_params(cell): return sum(p.numel() for p in cell(1, 32, batch_first=True).parameters())
print(f"RNN {n_params(nn.RNN)}  GRU {n_params(nn.GRU)}  LSTM {n_params(nn.LSTM)}")
# expected output: scratch GRU final hidden norm = 0.201
# expected output: RNN 1120  GRU 3360  LSTM 4480
# nn.RNN has one block of H*D + H*H + 2H = 32 + 1,024 + 64 = 1,120 parameters at H = 32, D = 1 (two bias vectors).
# nn.GRU stacks three such blocks (reset, update, candidate) and nn.LSTM four (input, forget, cell, output),
# so 1,120 : 3,360 : 4,480 is exactly 1 : 3 : 4 here.

# %% [markdown]
# 5. The memory-gap experiment

# %%
# 5. The memory-gap experiment
def memory_data(n, T, seed):
    r = np.random.default_rng(seed)
    X = r.normal(0, 0.3, (n, T, 1)).astype("float32"); bit = r.integers(0, 2, n)
    X[:, 0, 0] = bit                                     # answer planted at step 0
    return torch.tensor(X), torch.tensor(bit)
class RNNClf(nn.Module):
    def __init__(self):
        super().__init__(); self.rnn = nn.RNN(1, 16, batch_first=True); self.fc = nn.Linear(16, 2)
    def forward(self, x):
        out, _ = self.rnn(x); return self.fc(out[:, -1])    # classify from the LAST state
for T in [10, 50, 150]:
    torch.manual_seed(0)
    Xtr, ytr = memory_data(600, T, 1); Xte, yte = memory_data(300, T, 2)
    m = RNNClf(); opt = torch.optim.Adam(m.parameters(), lr=0.01); lf = nn.CrossEntropyLoss()
    for _ in range(150):
        opt.zero_grad(); lf(m(Xtr), ytr).backward(); opt.step()
    with torch.no_grad():
        print(f"gap T={T:>3}: test acc = {(m(Xte).argmax(1)==yte).float().mean().item():.3f}")
# expected output: gap T= 10: test acc = 1.000
# expected output: gap T= 50: test acc = 1.000
# expected output: gap T=150: test acc = 0.507

# %% [markdown]
# 6. Causal dilated convolution, receptive field, and a copy task

# %%
# 6. Causal dilated convolution, receptive field, and a copy task
torch.set_num_threads(2)      # the plain stack's accuracy changes in its last digit with the thread count
class CausalConv1d(nn.Module):
    def __init__(self, cin, cout, k, dilation):
        super().__init__()
        self.pad = (k - 1) * dilation                   # left pad only, so no output sees the future
        self.conv = nn.Conv1d(cin, cout, k, dilation=dilation)
    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))
torch.manual_seed(0)
layer = CausalConv1d(1, 1, 3, 1); x = torch.randn(1, 1, 6); y1 = layer(x)
x2 = x.clone(); x2[0, 0, -1] += 5.0; y2 = layer(x2)
print("causal (earlier outputs unchanged):", torch.allclose(y1[0, 0, :5], y2[0, 0, :5]))
stack = nn.Sequential(*[CausalConv1d(1, 1, 3, d) for d in [1, 2, 4, 8, 16, 32, 64, 128]])
base = torch.zeros(1, 1, 700, requires_grad=True); stack(base)[0, 0, -1].backward()
print("empirical receptive field:", int((base.grad[0, 0].abs() > 1e-12).sum()))
def make_copy(n, T=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    xx = torch.randint(0, 2, (n, 1, T), generator=g).float() * 2 - 1
    bb = torch.randint(0, 2, (n,), generator=g); xx[:, 0, 0] = bb.float() * 2 - 1   # bit at position 0
    return xx, bb
class TCN(nn.Module):
    def __init__(self, dils, ch=16):
        super().__init__(); layers = []; cin = 1
        for d in dils:
            layers += [CausalConv1d(cin, ch, 3, d), nn.ReLU()]; cin = ch
        self.net = nn.Sequential(*layers); self.head = nn.Linear(ch, 2)
    def forward(self, x): return self.head(self.net(x)[:, :, -1])
def train_tcn(dils):
    torch.manual_seed(0)
    m = TCN(dils); opt = torch.optim.Adam(m.parameters(), lr=0.01); lf = nn.CrossEntropyLoss()
    Xtr, ytr = make_copy(1500, 40, 1)
    for _ in range(60):
        opt.zero_grad(); lf(m(Xtr), ytr).backward(); opt.step()
    Xte, yte = make_copy(500, 40, 2)
    with torch.no_grad():
        return (m(Xte).argmax(1) == yte).float().mean().item()
print(f"plain 5-layer (RF 11): {train_tcn([1,1,1,1,1]):.3f}")
print(f"dilated 5-layer (RF 63): {train_tcn([1,2,4,8,16]):.3f}")
# expected output: causal (earlier outputs unchanged): True
# expected output: empirical receptive field: 511
# expected output: plain 5-layer (RF 11): 0.480
# expected output: dilated 5-layer (RF 63): 1.000

# %% [markdown]
# 7. Scaled dot-product attention vs the bottleneck mean

# %%
# 7. Scaled dot-product attention vs the bottleneck mean
rng = np.random.default_rng(0)
def softmax(z): e = np.exp(z - z.max()); return e / e.sum()
def attention(q, K, V):
    w = softmax(K @ q / np.sqrt(K.shape[1])); return w @ V, w    # context = weighted sum of values
n, d = 8, 16
K = rng.normal(size=(n, d)); Vv = rng.normal(size=(n, d)); target = 5
q = K[target] + rng.normal(0, 0.1, d)                           # a query that matches position 5
ctx, w = attention(q, K, Vv)
def cos(a, b): return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
print(f"weight on pos {target} = {w[target]:.3f}")
print(f"attention cosine to V[{target}] = {cos(ctx, Vv[target]):.3f}")
print(f"bottleneck (mean) cosine to V[{target}] = {cos(Vv.mean(0), Vv[target]):.3f}")
# expected output: weight on pos 5 = 0.644
# expected output: attention cosine to V[5] = 0.965
# expected output: bottleneck (mean) cosine to V[5] = 0.573

# %% [markdown]
# 8. Bidirectional vs unidirectional sentiment on real reviews

# %%
# 8. Bidirectional vs unidirectional sentiment on real reviews
import re
from collections import Counter
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
from datasets import load_dataset
torch.manual_seed(0)
ds = load_dataset("cornell-movie-review-data/rotten_tomatoes")
train = ds["train"].shuffle(seed=0).select(range(3000)); test = ds["test"].shuffle(seed=0).select(range(1000))
def toks(s): return re.findall(r"[a-z']+", s.lower())
counts = Counter(w for ex in train for w in toks(ex["text"]))
vocab = {w: i + 2 for i, (w, _) in enumerate(counts.most_common(4000))}   # 0 = pad, 1 = unknown
Vsize = len(vocab) + 2; T = 30
def encode(s):
    ids = [vocab.get(w, 1) for w in toks(s)][:T]; return ids + [0] * (T - len(ids))
def tens(split):
    return (torch.tensor([encode(ex["text"]) for ex in split]),
            torch.tensor([ex["label"] for ex in split]))
Xtr, ytr = tens(train); Xte, yte = tens(test)
class Clf(nn.Module):
    def __init__(self, bidir):
        super().__init__()
        self.emb = nn.Embedding(Vsize, 64, padding_idx=0)
        self.lstm = nn.LSTM(64, 64, batch_first=True, bidirectional=bidir)
        self.fc = nn.Linear(128 if bidir else 64, 2)
    def forward(self, x):
        lengths = (x != 0).sum(1).clamp(min=1)          # real length, ignoring pad
        out, _ = self.lstm(self.emb(x))
        if self.lstm.bidirectional:
            H = out.size(-1) // 2
            fwd = out[torch.arange(len(x)), lengths - 1, :H]  # forward half after the real words
            bwd = out[:, 0, H:]                                # backward half ends at position 0
            last = torch.cat([fwd, bwd], dim=1)
        else:
            last = out[torch.arange(len(x)), lengths - 1]
        return self.fc(last)
def run(bidir):
    torch.manual_seed(0)
    m = Clf(bidir); opt = torch.optim.Adam(m.parameters(), lr=0.003); lf = nn.CrossEntropyLoss()
    for _ in range(6):
        perm = torch.randperm(len(Xtr))
        for k in range(0, len(Xtr), 64):
            idx = perm[k:k+64]
            opt.zero_grad(); lf(m(Xtr[idx]), ytr[idx]).backward(); opt.step()
    with torch.no_grad():
        return (m(Xte).argmax(1) == yte).float().mean().item()
print(f"vocab {Vsize}")
print(f"unidirectional test acc = {run(False):.3f}")
print(f"bidirectional  test acc = {run(True):.3f}")
# expected output: vocab 4002
# expected output: unidirectional test acc = 0.647
# expected output: bidirectional  test acc = 0.646
# The backward half is read at position 0. At the last real word that half has seen only the last word and the pads.
# Printed accuracies 0.647 and 0.646. The gap on this whole-sentence label stays small: the forward state at the last
# real word has already read every word of the review, so one label for the whole sentence has its full context
# before the backward half adds anything. The peak tagger of 14.6 labels every position, and at position i the
# forward state has not yet seen the right neighbor that the label depends on, which the backward half supplies.
# On 1,000 test reviews the standard error of an accuracy near 0.65 is sqrt(0.647 * 0.353 / 1000) = 0.015, so a
# 0.001 gap from one seed is far inside the noise.

# %% [markdown]
# 9. Character-level text generation on a longer slice (miniproject) on tiny-shakespeare

# %%
# 9. Character-level text generation on a longer slice (miniproject) on tiny-shakespeare
torch.manual_seed(0)
raw = load_dataset("Trelis/tiny-shakespeare")
text = "".join(raw["train"]["Text"])[:80000]             # ten times the slice of Example 14-15
chars = sorted(set(text)); Vc = len(chars)
stoi = {c: i for i, c in enumerate(chars)}; itos = {i: c for c, i in stoi.items()}
data = torch.tensor([stoi[c] for c in text]); SEQ = 40
X = torch.stack([data[i:i+SEQ] for i in range(len(data)-SEQ)])
Yc = torch.stack([data[i+1:i+SEQ+1] for i in range(len(data)-SEQ)])
val_text = "".join(c for c in "".join(raw["test"]["Text"])[:2400] if c in stoi)[:2000]   # held-out test split
vdata = torch.tensor([stoi[c] for c in val_text])
Xv = torch.stack([vdata[i:i+SEQ] for i in range(len(vdata)-SEQ)])
Yv = torch.stack([vdata[i+1:i+SEQ+1] for i in range(len(vdata)-SEQ)])
class CharLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(Vc, 32); self.lstm = nn.LSTM(32, 128, batch_first=True); self.fc = nn.Linear(128, Vc)
    def forward(self, x, h=None):
        out, h = self.lstm(self.emb(x), h); return self.fc(out), h
model = CharLSTM(); opt = torch.optim.Adam(model.parameters(), lr=0.004); lf = nn.CrossEntropyLoss()
updates = 0
for epoch in range(2):                                    # 2 epochs of 625 batches: 1,250 updates
    perm = torch.randperm(len(X))
    for k in range(0, len(X), 128):
        idx = perm[k:k+128]; logits, _ = model(X[idx])          # teacher forcing
        opt.zero_grad(); lf(logits.reshape(-1, Vc), Yc[idx].reshape(-1)).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)       # clip the BPTT blow-up
        opt.step(); updates += 1
model.eval()
def score(Xs, Ys):                                        # accuracy and loss, 4,096 windows at a time
    hits = loss = 0.0
    with torch.no_grad():
        for k in range(0, len(Xs), 4096):
            lg, _ = model(Xs[k:k+4096]); yy = Ys[k:k+4096]
            hits += (lg.argmax(-1) == yy).sum().item()
            loss += F.cross_entropy(lg.reshape(-1, Vc), yy.reshape(-1), reduction="sum").item()
    return hits / Ys.numel(), loss / Ys.numel()
tr_acc, _ = score(X, Yc); val_acc, val_loss = score(Xv, Yv)
print(f"vocab {Vc}, updates {updates}")
print(f"train next-char accuracy = {tr_acc:.3f}, held-out accuracy = {val_acc:.3f}, held-out loss = {val_loss:.3f}")
def generate(seed, n=200, temp=0.6):
    torch.manual_seed(0)
    idx = torch.tensor([[stoi[c] for c in seed]]); h = None; out = seed
    with torch.no_grad():
        logits, h = model(idx, h)
        for _ in range(n):
            nxt = int(torch.multinomial(torch.softmax(logits[0, -1] / temp, -1), 1))
            out += itos[nxt]; logits, h = model(torch.tensor([[nxt]]), h)
    return out
sample = generate("First Citizen:\n")
print(repr(sample))
cont = sample[len("First Citizen:\n"):]
k = 0
while k < len(cont) and cont[:k+1] in text: k += 1
print(f"longest verbatim prefix in the training slice: {k} of {len(cont)} characters")
# expected output: vocab 60, updates 1250
# expected output: train next-char accuracy = 0.646, held-out accuracy = 0.401, held-out loss = 2.435
# expected output: 'First Citizen:\nYou have stood and to beseed,\nYour motiness and grave from the common powers\nAnd my coulterad following of the heart, that stay,\nThat I shall fellow speak for my common wholesow the past his man: make'
# expected output: longest verbatim prefix in the training slice: 15 of 200 characters
# Against Example 14-15 on 8,000 characters (train 0.942, held-out 0.272, loss 5.159, prefix 200 of 200), nearly the
# same number of updates on ten times the text lowers the training accuracy, raises the held-out accuracy, cuts the
# held-out loss, and the continuation stops matching the training slice after its first 15 characters.
# It has the shape of a speech, with line breaks and capitals, and it invents words such as motiness and coulterad,
# which appear nowhere in the training slice.

# %% [markdown]
# 10. Seq2seq with teacher forcing, evaluated by free-running decode

# %%
# 10. Seq2seq with teacher forcing, evaluated by free-running decode
torch.manual_seed(0)
Vt = 10; BOS = 10; VOCAB = 11; Lt = 8
def make_batch(n, seed):
    g = torch.Generator().manual_seed(seed)
    src = torch.randint(0, Vt, (n, Lt), generator=g); return src, torch.flip(src, dims=[1])
class Encoder(nn.Module):
    def __init__(self, hh=64):
        super().__init__(); self.emb = nn.Embedding(VOCAB, 32); self.lstm = nn.LSTM(32, hh, batch_first=True)
    def forward(self, x):
        _, (h, c) = self.lstm(self.emb(x)); return (h, c)       # context = final hidden and cell
class Decoder(nn.Module):
    def __init__(self, hh=64):
        super().__init__(); self.emb = nn.Embedding(VOCAB, 32); self.lstm = nn.LSTM(32, hh, batch_first=True); self.fc = nn.Linear(hh, VOCAB)
    def forward(self, tokens, state):
        out, state = self.lstm(self.emb(tokens), state); return self.fc(out), state
enc, dec = Encoder(), Decoder()
opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=0.01); lf = nn.CrossEntropyLoss()
for step in range(300):
    src, tgt = make_batch(128, seed=step); state = enc(src)
    dec_in = torch.cat([torch.full((128, 1), BOS), tgt[:, :-1]], dim=1)   # teacher forcing
    logits, _ = dec(dec_in, state)
    opt.zero_grad(); lf(logits.reshape(-1, VOCAB), tgt.reshape(-1)).backward(); opt.step()
src, tgt = make_batch(500, seed=9999)
with torch.no_grad():
    state = enc(src); tok = torch.full((500, 1), BOS); preds = []
    for _ in range(Lt):
        logits, state = dec(tok, state)
        tok = logits[:, -1].argmax(-1, keepdim=True); preds.append(tok)   # feed its OWN prediction back
    preds = torch.cat(preds, dim=1)
print(f"token acc = {(preds==tgt).float().mean().item():.3f}, seq acc = {(preds==tgt).all(1).float().mean().item():.3f}")
print("src :", src[0].tolist()); print("pred:", preds[0].tolist())
# expected output: token acc = 0.974, seq acc = 0.906
# expected output: src : [2, 5, 6, 1, 5, 7, 0, 3]
# expected output: pred: [3, 0, 7, 5, 1, 6, 5, 2]

# %% [markdown]
# ## 14.10 lab tasks

# %% [markdown]
# 1. The two slices, and the five characters the longer one adds

# %%
# 1. The two slices, and the five characters the longer one adds
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
from datasets import load_dataset
raw = load_dataset("Trelis/tiny-shakespeare")
text = "".join(raw["train"]["Text"])
a, b = text[:8000], text[:80000]
print(sorted(set(b) - set(a)))                      # characters that first appear after position 8000
# expected output: ['G', 'J', 'K', 'Q', 'x']
# The longer slice adds the capitals G, J, K, and Q and the lowercase x, so a model trained on it has 60 symbols
# instead of 55, the counts the 14.10 starter prints.
print(len(a), len(set(a)), len(b), len(set(b)))
# expected output: 8000 55 80000 60

# 2.
# Prediction: on the longer slice the training accuracy falls, the held-out accuracy rises, the held-out loss
# falls, and the verbatim prefix shrinks. The 1,250 updates now spread over 79,960 windows, each seen twice, where
# Example 14-15 showed each of its 7,960 windows twenty times, so the network has far less chance to memorize the
# slice. Coding exercise 9 measures it: training accuracy 0.646 (from 0.942), held-out accuracy 0.401 (from 0.272),
# held-out loss 2.435 (from 5.159), and a verbatim prefix of 15 of 200 characters (from 200 of 200).

# %% [markdown]
# ## 14.11 coding exercises

# %% [markdown]
# 11. Truncated BPTT on the synthetic series

# %%
# 11. Truncated BPTT on the synthetic series
import torch, torch.nn as nn
def last_loss(chunk, detach=True):
    torch.manual_seed(0)
    rnn = nn.RNN(1, 8, batch_first=True)
    head = nn.Linear(8, 1)
    opt = torch.optim.Adam(list(rnn.parameters()) + list(head.parameters()), lr=0.01)
    series = torch.randn(1, 200, 1)
    target = torch.randn(1, 200, 1)
    h = torch.zeros(1, 1, 8)
    loss = None
    for i, start in enumerate(range(0, 200, chunk), start=1):
        if detach:
            h = h.detach()
        xc = series[:, start:start + chunk]
        out, h = rnn(xc, h)
        loss = ((head(out) - target[:, start:start + chunk]) ** 2).mean()
        opt.zero_grad()
        try:
            loss.backward()
        except RuntimeError as err:                     # reached only when the detach is left out
            return f"chunk {i}: RuntimeError: {str(err).split(' (')[0]}"
        opt.step()
    return loss.item()
print(f"chunk 25 last loss {last_loss(25):.4f}")
print(f"chunk 100 last loss {last_loss(100):.4f}")
print("without detach,", last_loss(25, detach=False))
# expected output: chunk 25 last loss 0.8129
# expected output: chunk 100 last loss 0.6855
# expected output: without detach, chunk 2: RuntimeError: Trying to backward through the graph a second time
# The second chunk's graph runs back through the first chunk's hidden state, whose saved values the first
# backward() already freed, so autograd refuses; h.detach() starts each chunk's graph at the boundary instead.

# %% [markdown]
# 12. Additive attention in NumPy

# %%
# 12. Additive attention in NumPy
import numpy as np
def additive_weights(s, enc, W1, W2, v):
    scores = np.tanh(W1 @ s + enc @ W2.T) @ v   # row i of enc @ W2.T is W2 h_i, so score_i = v^T tanh(W1 s + W2 h_i)
    e = np.exp(scores - scores.max())
    return e / e.sum()
rng = np.random.default_rng(0)
H = 3
s = rng.normal(size=H)
enc = rng.normal(size=(5, H))
W1 = rng.normal(size=(H, H)) * 0.1
W2 = rng.normal(size=(H, H)) * 0.1
v = rng.normal(size=H) * 0.1
w = additive_weights(s, enc, W1, W2, v)
print("weights", np.round(w, 4))
print("weights sum", round(float(w.sum()), 6))
print("argmax", int(w.argmax()))
# expected output: weights [0.2014 0.2054 0.1981 0.1958 0.1993]
# expected output: weights sum 1.0
# expected output: argmax 1

# %% [markdown]
# 13. Packing versus a solo sequence

# %%
# 13. Packing versus a solo sequence
from torch.nn.utils.rnn import pack_padded_sequence
torch.manual_seed(0)
seq = torch.randint(1, 20, (17,))
solo_net = nn.LSTM(8, 4, batch_first=True, bidirectional=True)
emb = nn.Embedding(21, 8, padding_idx=0)
with torch.no_grad():
    e = emb(seq.view(1, -1))
    _, (h_solo, _) = solo_net(e)
    padded = torch.cat([seq, torch.zeros(16, dtype=torch.long)]).view(1, -1)
    ep = emb(padded)
    _, (h_naive, _) = solo_net(ep)
    packed = pack_padded_sequence(ep, torch.tensor([17]), batch_first=True, enforce_sorted=False)
    _, (h_pack, _) = solo_net(packed)
print(f"naive forward max abs {(h_naive[-2] - h_solo[-2]).abs().max().item():.6f}")
print("naive backward max abs below 1e-5:", bool((h_naive[-1] - h_solo[-1]).abs().max().item() < 1e-5))
print(f"packed forward max abs {(h_pack[-2] - h_solo[-2]).abs().max().item():.1f}")
print(f"packed backward max abs {(h_pack[-1] - h_solo[-1]).abs().max().item():.1f}")
# expected output: naive forward max abs 0.256843
# expected output: naive backward max abs below 1e-5: True
# expected output: packed forward max abs 0.0
# expected output: packed backward max abs 0.0

# %% [markdown]
# 14. A positive forget-gate bias

# %%
# 14. A positive forget-gate bias
torch.manual_seed(0)
layer = nn.LSTM(1, 16)
H = 16
before = float(torch.sigmoid(layer.bias_ih_l0[H:2 * H]).mean())
print(f"{before:.3f}")
with torch.no_grad():
    gate0 = torch.sigmoid(layer.bias_ih_l0[H:2 * H] + layer.bias_hh_l0[H:2 * H])   # forget gate at zero input and state
    layer.bias_ih_l0[H:2 * H].fill_(1.0)
    layer.bias_hh_l0[H:2 * H].zero_()
    gate1 = torch.sigmoid(layer.bias_ih_l0[H:2 * H] + layer.bias_hh_l0[H:2 * H])
after = float(torch.sigmoid(layer.bias_ih_l0[H:2 * H]).mean())
print(f"{after:.3f}")
print(f"forget gate at zero input: mean {float(gate0.mean()):.3f} before, {float(gate1.mean()):.3f} after")
print(f"largest 100-step product of one unit's gate: {float((gate0 ** 100).max()):.1e} before, {float((gate1 ** 100).max()):.1e} after")
# expected output: 0.474
# expected output: 0.731
# expected output: forget gate at zero input: mean 0.460 before, 0.731 after
# expected output: largest 100-step product of one unit's gate: 6.7e-29 before, 2.5e-14 after
# With zero input and zero state a unit's forget gate is sigmoid(b_ih + b_hh) over its forget rows. The default
# initialization leaves it near 0.5 (mean 0.460 here), so even the most open unit multiplies down to 6.7e-29 over
# 100 steps. Filling the input-hidden forget bias with 1 and zeroing the hidden-hidden one starts every gate at
# sigmoid(1) = 0.731, and 0.731**100 = 2.5e-14, more than fourteen orders of magnitude larger. Along the cell path
# the direct step-to-step factor is the forget gate, so at the start of training the gradient there survives far
# more steps, and the network can still learn to close a gate where forgetting helps.

# %% [markdown]
# ## 14.11 coding drills

# %%
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

# %% [markdown]
# RNN mechanics

# %%
# RNN mechanics
rng = np.random.default_rng(0)
Wxh = rng.normal(size=(4, 3)); Whh = rng.normal(size=(4, 4)); b = np.zeros(4); h = np.zeros(4)
for x in rng.normal(size=(5, 3)):
    h = np.tanh(Wxh @ x + Whh @ h + b)
print(round(float(np.linalg.norm(h)), 3))                          # expected output: 1.536
print(8*1 + 8*8 + 8 + 8, sum(p.numel() for p in nn.RNN(1, 8, batch_first=True).parameters()))   # expected output: 88 88
o, hn = nn.RNN(3, 6, batch_first=True)(torch.randn(1, 5, 3))
print(tuple(o.shape), tuple(hn.shape))                             # expected output: (1, 5, 6) (1, 1, 6)
print(f"{0.9**25:.3e}")                                            # expected output: 7.179e-02

# %% [markdown]
# Gates

# %%
# Gates
sig = lambda v: 1 / (1 + np.exp(-v))
print(round(float(sig(0)), 3), round(float(sig(2)), 3))           # expected output: 0.5 0.881
print(tuple(nn.LSTM(10, 20).weight_ih_l0.shape))                  # expected output: (80, 10)
print(round(0.9 * 1.0 + 0.5 * 0.4, 3))                            # expected output: 1.1
print(f"{0.95**50:.3e}")                                          # expected output: 7.694e-02

# %% [markdown]
# Conv1d and TCN

# %%
# Conv1d and TCN
K = 3
print(1 + sum((K - 1) for _ in range(6)))                         # expected output: 13
print(1 + sum((K - 1) * d for d in [1, 2, 4, 8]))                 # expected output: 31
print(nn.Conv1d(1, 4, 3)(torch.randn(1, 1, 10)).shape[-1])        # expected output: 8
xp = F.pad(torch.randn(1, 1, 10), (2, 0))
print(nn.Conv1d(1, 1, 3)(xp).shape[-1])                           # expected output: 10

# %% [markdown]
# Attention

# %%
# Attention
def softmax(z): e = np.exp(z - z.max()); return e / e.sum()
w = softmax(np.array([1.0, 3.0, 0.0]))
print(np.round(w, 3), round(float(w.sum()), 3))                   # expected output: [0.114 0.844 0.042] 1.0
Kmat = np.eye(4); q = Kmat[2]
print(int(softmax(Kmat @ q).argmax()))                           # expected output: 2
print(round(1 / np.sqrt(64), 4))                                  # expected output: 0.125
print(512 * 512)                                                  # expected output: 262144

# %% [markdown]
# Seq2seq and shapes

# %%
# Seq2seq and shapes
emb = nn.Embedding(11, 8, padding_idx=0)
print(bool((emb.weight[0].abs().sum() == 0).item()))              # expected output: True
a = [torch.tensor([1, 2, 3]), torch.tensor([4, 5]), torch.tensor([6])]
print(tuple(pad_sequence(a, batch_first=True, padding_value=0).shape))   # expected output: (3, 3)
x = torch.randint(1, 5, (2, 6)); x[0, 4:] = 0
print((x != 0).sum(1).tolist())                                   # expected output: [4, 6]
print(torch.flip(torch.tensor([1, 2, 3, 4]), dims=[0]).tolist())  # expected output: [4, 3, 2, 1]
