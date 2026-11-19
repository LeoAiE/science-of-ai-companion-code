# %% [markdown]
# # Lab: char-level tiny-shakespeare generation + rotten_tomatoes sentiment
# Regenerated from Book.md; run top to bottom (HF text data downloads once).

# %%
import warnings; warnings.filterwarnings("ignore")
import torch, torch.nn as nn
torch.set_num_threads(2)      # the held-out loss changes in its last digit with the thread count
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
from datasets import load_dataset
torch.manual_seed(0)
raw = load_dataset("Trelis/tiny-shakespeare")         # real Shakespeare, character level
text = "".join(raw["train"]["Text"])[:8000]           # a small slice, about 8000 characters
chars = sorted(set(text)); V = len(chars)             # vocabulary of distinct characters
stoi = {c: i for i, c in enumerate(chars)}; itos = {i: c for c, i in stoi.items()}
data = torch.tensor([stoi[c] for c in text]); SEQ = 40
X = torch.stack([data[i:i+SEQ] for i in range(len(data)-SEQ)])       # each row: 40 characters
Y = torch.stack([data[i+1:i+SEQ+1] for i in range(len(data)-SEQ)])   # the next character at every step
val_text = "".join(c for c in "".join(raw["test"]["Text"])[:2400] if c in stoi)[:2000]   # the dataset's own
vdata = torch.tensor([stoi[c] for c in val_text])                    # held-out test split: text never trained on
Xv = torch.stack([vdata[i:i+SEQ] for i in range(len(vdata)-SEQ)])
Yv = torch.stack([vdata[i+1:i+SEQ+1] for i in range(len(vdata)-SEQ)])
class CharLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, 32); self.lstm = nn.LSTM(32, 128, batch_first=True); self.fc = nn.Linear(128, V)
    def forward(self, x, h=None):
        out, h = self.lstm(self.emb(x), h); return self.fc(out), h
model = CharLSTM(); opt = torch.optim.Adam(model.parameters(), lr=0.004); lf = nn.CrossEntropyLoss()
for epoch in range(20):
    perm = torch.randperm(len(X))
    for k in range(0, len(X), 128):
        idx = perm[k:k+128]; logits, _ = model(X[idx])          # teacher forcing: the true chars are the inputs
        opt.zero_grad(); lf(logits.reshape(-1, V), Y[idx].reshape(-1)).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)       # gradient clipping guards the BPTT blow-up
        opt.step()
model.eval()
with torch.no_grad():
    tr_logits, _ = model(X); val_logits, _ = model(Xv)
    tr_acc  = (tr_logits.argmax(-1) == Y).float().mean().item()
    val_acc = (val_logits.argmax(-1) == Yv).float().mean().item()
    val_loss = lf(val_logits.reshape(-1, V), Yv.reshape(-1)).item()
print(f"train next-char accuracy = {tr_acc:.3f}, held-out accuracy = {val_acc:.3f}, held-out loss = {val_loss:.3f}")
def generate(seed, n=200, temp=0.6):                  # sample one character at a time, feeding it back in
    idx = torch.tensor([[stoi[c] for c in seed]]); h = None; out = seed
    with torch.no_grad():
        logits, h = model(idx, h)
        for _ in range(n):
            nxt = int(torch.multinomial(torch.softmax(logits[0, -1] / temp, -1), 1))
            out += itos[nxt]; logits, h = model(torch.tensor([[nxt]]), h)
    return out
sample = generate("First Citizen:\n")
print(sample)
cont = sample[len("First Citizen:\n"):]               # how much of the sample is copied training text?
k = 0
while k < len(cont) and cont[:k+1] in text: k += 1
print(f"longest verbatim prefix of the continuation found in the training slice: {k} of {len(cont)} characters")

# %%
import torch, torch.nn as nn
threads = torch.get_num_threads(); torch.set_num_threads(1)   # one intra-op thread, so the printed numbers reproduce
class VarCharLSTM(nn.Module):                         # Example 14-15's model, stepped one character at a time
    def __init__(self, p):
        super().__init__(); self.p = p
        self.emb = nn.Embedding(V, 32); self.cell = nn.LSTMCell(32, 128); self.fc = nn.Linear(128, V)
    def mask(self, n, size, like):                    # one mask per sequence, kept units scaled by 1/(1-p)
        if not self.training or self.p == 0:
            return 1.0
        return torch.bernoulli(like.new_full((n, size), 1 - self.p)) / (1 - self.p)
    def forward(self, x, state=None):
        e = self.emb(x); n = x.size(0)
        h, c = state if state is not None else (e.new_zeros(n, 128), e.new_zeros(n, 128))
        m_in, m_rec, m_out = self.mask(n, 32, e), self.mask(n, 128, e), self.mask(n, 128, e)   # drawn once
        outs = []
        for t in range(x.size(1)):                    # the same three masks at every step
            h, c = self.cell(e[:, t] * m_in, (h * m_rec, c))
            outs.append(h * m_out)
        return self.fc(torch.stack(outs, 1)), (h, c)
def train_and_score(p):
    torch.manual_seed(0)
    model = VarCharLSTM(p); opt = torch.optim.Adam(model.parameters(), lr=0.004); lf = nn.CrossEntropyLoss()
    for epoch in range(20):                           # Example 14-15's schedule, unchanged
        model.train(); perm = torch.randperm(len(X))
        for k in range(0, len(X), 128):
            idx = perm[k:k+128]; logits, _ = model(X[idx])
            opt.zero_grad(); lf(logits.reshape(-1, V), Y[idx].reshape(-1)).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
    model.eval()                                      # no masks at evaluation: every unit is used
    with torch.no_grad():
        tr_logits, _ = model(X); val_logits, _ = model(Xv)
        tr_acc = (tr_logits.argmax(-1) == Y).float().mean().item()
        val_acc = (val_logits.argmax(-1) == Yv).float().mean().item()
        val_loss = lf(val_logits.reshape(-1, V), Yv.reshape(-1)).item()
        logits, state = model(torch.tensor([[stoi[ch] for ch in "First Citizen:\n"]])); cont = ""
        for _ in range(200):                          # 200 characters at temperature 0.6, as Example 14-15 samples
            nxt = int(torch.multinomial(torch.softmax(logits[0, -1] / 0.6, -1), 1))
            cont += itos[nxt]; logits, state = model(torch.tensor([[nxt]]), state)
    k = 0
    while k < len(cont) and cont[:k+1] in text: k += 1
    print(f"p = {p}: train accuracy = {tr_acc:.3f}, held-out accuracy = {val_acc:.3f}, "
          f"held-out loss = {val_loss:.3f}, verbatim prefix = {k} of 200")
    return cont
samples = {p: train_and_score(p) for p in (0.0, 0.3)}
print(samples[0.3])
torch.set_num_threads(threads)                        # restore the thread count for the blocks that follow

# %%
import warnings; warnings.filterwarnings("ignore")
import torch, torch.nn as nn
from collections import Counter
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
from datasets import load_dataset
torch.manual_seed(0)
ds = load_dataset("cornell-movie-review-data/rotten_tomatoes")     # real movie-review sentiment
def toks(s): return s.split()                                      # the same whitespace tokenizer as 14.6.1
counts = Counter(w for s in ds["train"]["text"] for w in toks(s))
vocab = [w for w, _ in counts.most_common(4000)]                   # keep the 4000 most frequent words
stoi = {w: i + 2 for i, w in enumerate(vocab)}                     # index 0 = padding, index 1 = unknown
T = 40                                                             # the same truncation length as 14.6.1
def encode(s):
    ids = [stoi.get(w, 1) for w in toks(s)][:T]
    return ids + [0] * (T - len(ids))                             # right-pad short reviews with 0
def tensors(split):
    return (torch.tensor([encode(s) for s in ds[split]["text"]]),
            torch.tensor(ds[split]["label"]))
Xtr, ytr = tensors("train"); Xte, yte = tensors("test")
class SentLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(len(stoi) + 2, 64, padding_idx=0)  # the pad row stays zero and is ignored
        self.lstm = nn.LSTM(64, 96, batch_first=True); self.fc = nn.Linear(96, 2)
    def forward(self, x):
        lengths = (x != 0).sum(1).clamp(min=1)                    # real length of each review, ignoring pad
        out, _ = self.lstm(self.emb(x))
        last = out[torch.arange(len(x)), lengths - 1]             # take the state at the last real word
        return self.fc(last)
clf = SentLSTM(); opt = torch.optim.Adam(clf.parameters(), lr=0.003); lf = nn.CrossEntropyLoss()
for epoch in range(8):
    perm = torch.randperm(len(Xtr))
    for k in range(0, len(Xtr), 64):
        idx = perm[k:k+64]
        opt.zero_grad(); lf(clf(Xtr[idx]), ytr[idx]).backward(); opt.step()
with torch.no_grad():
    print(f"vocab {len(stoi)+2}, test reviews {len(Xte)}")
    print(f"test accuracy = {(clf(Xte).argmax(1) == yte).float().mean().item():.3f}")

# %%
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
def first_words(s): return toks(s)[:T]                         # the same first 40 tokens that encode() keeps
bag = CountVectorizer(analyzer=first_words, vocabulary=vocab)   # one column per word of the 4000-word vocabulary
C_tr = bag.transform(ds["train"]["text"]); C_te = bag.transform(ds["test"]["text"])   # a word outside it adds nothing
bow = LogisticRegression(max_iter=2000).fit(C_tr, ds["train"]["label"])   # one weight per word count, order ignored
print(f"count matrix {C_tr.shape}, test accuracy = {bow.score(C_te, ds['test']['label']):.3f}")
