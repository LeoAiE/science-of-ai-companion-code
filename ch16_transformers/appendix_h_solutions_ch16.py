"""Appendix H solutions for Chapter 16, as runnable cells.

Generated from Solutions.md (Appendix H). Every code block of this chapter's solutions appears here
verbatim, split into one cell per exercise or drill topic, so each answer can be run and checked against
the expected outputs it states. Keep Solutions.md as the source; regenerate after any change.
"""

# %% [markdown]
# ## 16.13 coding exercises

# %%
import warnings; warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

def softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True)); return e / e.sum(axis=-1, keepdims=True)

# %% [markdown]
# 1. Scaled dot-product self-attention from scratch, vs PyTorch

# %%
# 1. Scaled dot-product self-attention from scratch, vs PyTorch
rng = np.random.default_rng(0)
n, d = 4, 8
X = rng.normal(size=(n, d))
WQ, WK, WV = (rng.normal(0, 0.5, (d, d)) for _ in range(3))
Q, K, V = X @ WQ, X @ WK, X @ WV
def attention(Q, K, V):
    w = softmax(Q @ K.T / np.sqrt(Q.shape[-1]))          # softmax(QK^T / sqrt(d_k))
    return w @ V, w
out, w = attention(Q, K, V)
print(bool(np.allclose(w.sum(1), 1)))                     # expected output: True
tw = F.scaled_dot_product_attention(torch.tensor(Q)[None], torch.tensor(K)[None],
                                    torch.tensor(V)[None]).numpy()[0]
print(bool(np.allclose(out, tw)))                         # expected output: True
perm = [2, 0, 3, 1]
op, _ = attention(X[perm] @ WQ, X[perm] @ WK, X[perm] @ WV)
print(bool(np.allclose(op, out[perm])))                   # expected output: True

# %% [markdown]
# 2. The sqrt(d_k) scaling, measured

# %%
# 2. The sqrt(d_k) scaling, measured
rng = np.random.default_rng(0)
for dk in [16, 64, 256]:
    q = rng.normal(0, 1, (20000, dk)); k = rng.normal(0, 1, (20000, dk))
    dots = np.sum(q * k, axis=1)
    print(dk, round(float(dots.var()), 1), round(float((dots / np.sqrt(dk)).var()), 3))
    # expected output: 16 16.0 0.998
    # expected output: 64 63.3 0.989
    # expected output: 256 257.8 1.007
def sm(z):
    e = np.exp(z - z.max()); return e / e.sum()
q1 = rng.normal(0, 1, 64); Kt = rng.normal(0, 1, (8, 64)); logits = Kt @ q1
print(round(float(sm(logits).max()), 3), round(float(sm(logits / 8).max()), 3))
# expected output: 0.986 0.471

# %% [markdown]
# 3. Multi-head attention from scratch: twelve dimensions split into three heads of four

# %%
# 3. Multi-head attention from scratch: twelve dimensions split into three heads of four
rng = np.random.default_rng(1)
n, d, h = 6, 12, 3
dk = d // h
X = rng.normal(size=(n, d))
WQ, WK, WV, WO = (rng.normal(0, 0.5, (d, d)) for _ in range(4))
Q, K, V = X @ WQ, X @ WK, X @ WV
def to_heads(M): return M.reshape(n, h, dk).transpose(1, 0, 2)   # (n, d) -> (h, n, dk)
Qh, Kh, Vh = to_heads(Q), to_heads(K), to_heads(V)
heads, head_w = [], []
for i in range(h):
    wi = softmax(Qh[i] @ Kh[i].T / np.sqrt(dk))
    head_w.append(wi); heads.append(wi @ Vh[i])
mh_out = np.concatenate(heads, axis=1) @ WO                # concat heads, project
print(mh_out.shape)                                       # expected output: (6, 12)
print(round(float(np.abs(head_w[0] - head_w[1]).max()), 3), round(float(np.abs(head_w[1] - head_w[2]).max()), 3))
# expected output: 0.76 0.924
print(d * d, h * (d * dk), (d * d) == (h * (d * dk)))     # expected output: 144 144 True

# %% [markdown]
# 4. Sinusoidal positional encoding: fourteen positions in twenty-four dimensions

# %%
# 4. Sinusoidal positional encoding: fourteen positions in twenty-four dimensions
def positional_encoding(n, d):
    pos = np.arange(n)[:, None]; idx = np.arange(d)[None, :]
    angle = pos / (10000 ** (2 * (idx // 2) / d))
    return np.where(idx % 2 == 0, np.sin(angle), np.cos(angle))
PE = positional_encoding(14, 24)
print(PE.shape, len(np.unique(PE.round(6), axis=0)) == 14)   # expected output: (14, 24) True
rows = []
for p in (0, 4, 9):
    row = [round(float(PE[p] @ PE[p + k]), 2) for k in range(14 - p)]
    rows.append(row)
    print(p, row)
# expected output: 0 [12.0, 11.41, 10.07, 8.93, 8.61, 8.92, 9.07, 8.52, 7.47, 6.73, 6.94, 7.96, 8.92, 8.98]
# expected output: 4 [12.0, 11.41, 10.07, 8.93, 8.61, 8.92, 9.07, 8.52, 7.47, 6.73]
# expected output: 9 [12.0, 11.41, 10.07, 8.93, 8.61]
print(all(rows[i][k] == rows[0][k] for i in range(3) for k in range(len(rows[i]))))
# expected output: True
sims = rows[0]                         # offset 0 gives d/2 = 12, one cos(0) = 1 per frequency pair
print(sims[0] == max(sims), float(np.mean(sims[:7])) > float(np.mean(sims[7:])))
# expected output: True True
# the row dips to 8.61 at offset 4, climbs to 9.07 at offset 6, dips to 6.73 at 9, and climbs to 8.98:
# a downward drift on average, not a monotone fall

# %% [markdown]
# 5. Rotary embeddings (RoPE): six dimensions, the dot product depends only on the offset

# %%
# 5. Rotary embeddings (RoPE): six dimensions, the dot product depends only on the offset
def rope(vec, pos, base=10000.0):
    d = vec.shape[0]; out = np.empty_like(vec)
    for i in range(d // 2):                    # i indexes the pair, as in theta_i
        ang = pos * base ** (-2.0 * i / d)
        cos_a, sin_a = np.cos(ang), np.sin(ang)
        x0, x1 = vec[2 * i], vec[2 * i + 1]
        out[2 * i], out[2 * i + 1] = x0 * cos_a - x1 * sin_a, x0 * sin_a + x1 * cos_a   # 2x2 rotation of the pair
    return out
rng = np.random.default_rng(1)
q, k = rng.normal(size=6), rng.normal(size=6)
off3 = [round(float(rope(q, t) @ rope(k, s)), 5) for t, s in [(4, 1), (8, 5), (3, 0)]]
print(off3, len(set(off3)) == 1)          # expected output: [-0.10635, -0.10635, -0.10635] True
print(round(float(rope(q, 6) @ rope(k, 1)), 5))   # expected output: -0.38203

# %% [markdown]
# 6. LayerNorm, RMSNorm, GELU on a six-dimensional vector

# %%
# 6. LayerNorm, RMSNorm, GELU on a six-dimensional vector
rng = np.random.default_rng(2)
v = rng.normal(-1, 3, 6)
def layernorm(x, eps=1e-5): return (x - x.mean()) / np.sqrt(x.var() + eps)   # center AND scale
def rmsnorm(x, eps=1e-5): return x / np.sqrt((x ** 2).mean() + eps)          # scale only
print(round(float(layernorm(v).mean()), 3), round(float(layernorm(v).std()), 3))   # expected output: 0.0 1.0
print(round(float(rmsnorm(v).mean()), 3))                 # expected output: -0.267
ln = nn.LayerNorm(6)                                      # gamma=1, beta=0 by default
print(bool(np.allclose(layernorm(v), ln(torch.tensor(v, dtype=torch.float32)).detach().numpy(), atol=1e-4)))
# expected output: True
def gelu(x): return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
for xv in [-1.0, -0.3, 0.6, 1.8]:
    print(xv, round(max(0.0, xv), 3), round(float(gelu(np.array(xv))), 3))
    # expected output: -1.0 0.0 -0.159
    # expected output: -0.3 0.0 -0.115
    # expected output: 0.6 0.6 0.435
    # expected output: 1.8 1.8 1.735

# %% [markdown]
# 7. Causal and padding masks on a four-token sequence

# %%
# 7. Causal and padding masks on a four-token sequence
rng = np.random.default_rng(3)
n = 4
scores = rng.normal(0, 1, (n, n))
causal = softmax(scores + np.triu(np.full((n, n), -np.inf), k=1))   # future -> -inf
print(bool(np.allclose(causal[np.triu_indices(n, k=1)], 0)))        # expected output: True
print(bool(np.allclose(causal.sum(1), 1)))                          # expected output: True
print(int((causal[0] > 0).sum()), int((causal[3] > 0).sum()))       # expected output: 1 4
pad = np.zeros((n, n)); pad[:, [2, 3]] = -np.inf                    # ignore filler columns 2 and 3
print(round(float(softmax(scores + pad)[:, [2, 3]].sum()), 3))      # expected output: 0.0

# %% [markdown]
# 8. Quadratic wall, KV cache, and the cache of an 80-layer model at 32,768 tokens

# %%
# 8. Quadratic wall, KV cache, and the cache of an 80-layer model at 32,768 tokens
for nn_ in [1024, 2048, 4096, 8192, 16384]:
    print(nn_, nn_ * nn_)
    # expected output: 1024 1048576
    # expected output: 2048 4194304
    # expected output: 4096 16777216
    # expected output: 8192 67108864
    # expected output: 16384 268435456
T = 2000
no_cache = sum(t * t for t in range(1, T + 1))     # recompute the whole prefix every step (~T^3/3)
with_cache = sum(t for t in range(1, T + 1))        # only the new token each step (~T^2/2)
print(no_cache, with_cache, round(no_cache / with_cache))
# expected output: 2668667000 2001000 1334
n_layers, d_head, seq_len, batch, bpe = 80, 128, 32768, 1, 2   # Llama 3 70B shape: 64 query heads, fp16
def kv_gb(nkv): return 2 * n_layers * nkv * d_head * seq_len * batch * bpe / 1e9
for name, kv in [("MHA", 64), ("GQA8", 8), ("MQA", 1)]:
    print(name, kv, round(kv_gb(kv), 3), 64 // kv)
    # expected output: MHA 64 85.899 1
    # expected output: GQA8 8 10.737 8
    # expected output: MQA 1 1.342 64

# %% [markdown]
# 9. Bidirectional (BERT) vs causal (distilGPT2), and perplexity, on real models

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
import os; os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import warnings; warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
from transformers import logging as hf_logging
hf_logging.set_verbosity_error(); hf_logging.disable_progress_bar()
# 9. Bidirectional (BERT) vs causal (distilGPT2), and perplexity, on real models
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
fill = pipeline("fill-mask", model="google-bert/bert-base-uncased", device="cpu")
for top in fill("The largest planet in the solar system is [MASK].", top_k=2):
    print(top["token_str"], round(top["score"], 3))
    # expected output: pluto 0.339
    # expected output: jupiter 0.276
tok = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
gpt = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2").eval()
ids = tok("She opened the door and", return_tensors="pt").input_ids
with torch.no_grad():
    probs = torch.softmax(gpt(ids).logits[0, -1], dim=-1)
p, i = torch.topk(probs, 1)
print(repr(tok.decode(i[0])), round(p[0].item(), 3))   # expected output: ' saw' 0.099
ids = tok("It was the best of times, it was the worst of times.", return_tensors="pt").input_ids
with torch.no_grad():
    loss = gpt(ids, labels=ids).loss                # mean next-token cross-entropy over the sentence
print(ids.shape[1], round(loss.item(), 3), round(torch.exp(loss).item(), 1))
# expected output: 14 2.389 10.9

# %% [markdown]
# 10. Miniproject: a char-level mini-GPT trained on Trelis/tiny-shakespeare

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
# 10. Miniproject: a char-level mini-GPT trained on Trelis/tiny-shakespeare
from datasets import load_dataset
torch.manual_seed(0)
ds = load_dataset("Trelis/tiny-shakespeare")["train"]
text = "".join(ds[i]["Text"] for i in range(4))      # a small slice, trains in seconds on CPU
chars = sorted(set(text)); V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}; itos = {i: c for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
block_size, n_embd, n_head, n_layer = 32, 64, 4, 2
def get_batch(bs=32):
    ix = torch.randint(0, len(data) - block_size - 1, (bs,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])   # targets = inputs shifted by one
    return x, y
class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(n_embd, 3 * n_embd); self.proj = nn.Linear(n_embd, n_embd)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))
    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(n_embd, dim=2); hs = C // n_head
        q = q.view(B, T, n_head, hs).transpose(1, 2)
        k = k.view(B, T, n_head, hs).transpose(1, 2)
        v = v.view(B, T, n_head, hs).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / (hs ** 0.5)
        att = att.masked_fill(self.mask[:T, :T] == 0, float('-inf'))   # mask the future
        att = F.softmax(att, dim=-1)
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd); self.attn = CausalSelfAttention()
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.GELU(), nn.Linear(4 * n_embd, n_embd))
    def forward(self, x):
        x = x + self.attn(self.ln1(x)); x = x + self.ffn(self.ln2(x)); return x   # pre-LN residuals
class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(V, n_embd); self.pos = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(n_embd); self.head = nn.Linear(n_embd, V)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T))
        logits = self.head(self.lnf(self.blocks(x)))
        loss = None if targets is None else F.cross_entropy(logits.view(-1, V), targets.view(-1))
        return logits, loss
    @torch.no_grad()
    def generate(self, idx, n_new):
        for _ in range(n_new):
            logits, _ = self(idx[:, -block_size:])
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx
model = MiniGPT()
print(sum(p.numel() for p in model.parameters()), round(float(np.log(V)), 3))
# expected output: 109368 4.025
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
_, l0 = model(*get_batch())
for step in range(1, 801):
    x, y = get_batch(); _, loss = model(x, y)
    opt.zero_grad(); loss.backward(); opt.step()
print(round(l0.item(), 2), round(loss.item(), 2))
# expected output: 4.15 0.55
torch.manual_seed(1)
out = model.generate(torch.tensor([[stoi['T']]]), 120)[0].tolist()
print("".join(itos[i] for i in out)[:70])
# expected output: The disconacke spor come from you most:
# expected output: That once you the cranks, and

# %% [markdown]
# ## 16.13 coding drills

# %%
import numpy as np

def softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True)); return e / e.sum(axis=-1, keepdims=True)

# %% [markdown]
# Attention math

# %%
# Attention math
print(softmax(np.array([2.0, 1.0, 0.0])).round(3))   # expected output: [0.665 0.245 0.09 ]
print(np.sqrt(64))                                   # expected output: 8.0
qk = np.array([1.0, 2.0, 3.0]) @ np.array([0.5, 0.5, 0.5])
print(round(qk / np.sqrt(3), 3))                     # expected output: 1.732
print(bool(np.isclose(softmax(np.array([1.0, 2.0, 3.0])).sum(), 1.0)))   # expected output: True

# %% [markdown]
# Masks

# %%
# Masks
print(np.triu(np.full((3, 3), -np.inf), k=1))
# expected output: [[  0. -inf -inf]
# expected output:  [  0.   0. -inf]
# expected output:  [  0.   0.   0.]]
m = softmax(np.zeros((3, 3)) + np.triu(np.full((3, 3), -np.inf), k=1))
print(m.round(3))
# expected output: [[1.    0.    0.   ]
# expected output:  [0.5   0.5   0.   ]
# expected output:  [0.333 0.333 0.333]]
print(int((m[0] > 0).sum()))                         # expected output: 1

# %% [markdown]
# Positional encodings

# %%
# Positional encodings
def positional_encoding(n, d):
    pos = np.arange(n)[:, None]; idx = np.arange(d)[None, :]
    angle = pos / (10000 ** (2 * (idx // 2) / d))
    return np.where(idx % 2 == 0, np.sin(angle), np.cos(angle))
pe = positional_encoding(4, 4)
print(pe.shape)                                      # expected output: (4, 4)
print(bool(len(np.unique(pe.round(6), axis=0)) == 4))   # expected output: True
print(round(float(pe[0] @ pe[0]), 2))                # expected output: 2.0

# %% [markdown]
# Norms and activations

# %%
# Norms and activations
def layernorm(x, eps=1e-5): return (x - x.mean()) / np.sqrt(x.var() + eps)
def rmsnorm(x, eps=1e-5): return x / np.sqrt((x ** 2).mean() + eps)
def gelu(x): return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
x = np.array([1.0, 2.0, 3.0, 4.0])
print(round(float(layernorm(x).mean()), 3), round(float(layernorm(x).std()), 3))   # expected output: 0.0 1.0
print(round(float(rmsnorm(x).mean()), 3))            # expected output: 0.913
print(round(float(gelu(np.array(1.0))), 3), max(0.0, 1.0))   # expected output: 0.841 1.0

# %% [markdown]
# Cost and caching

# %%
# Cost and caching
print(1000 ** 2, (2 * 1000) ** 2, (2 * 1000) ** 2 // 1000 ** 2)   # expected output: 1000000 4000000 4
print(sum(range(1, 101)), sum(t * t for t in range(1, 101)))       # expected output: 5050 338350
print(2 * 40 * 8 * 128 * 8192 * 4 * 2 / 1e9)         # expected output: 5.36870912

# %% [markdown]
# Mixture of Experts and state-space models

# %%
# Mixture of Experts and state-space models
ne, tk, dd = 256, 8, 64                              # 256 routed experts, top-8, as in DeepSeek-V3's config
each = 2 * dd * 4 * dd
print(ne * each, tk * each, (ne * each) // (tk * each))   # expected output: 8388608 262144 32
def ssm(x, a=0.8):
    h, ys = 0.0, []
    for xt in x: h = a * h + xt; ys.append(h)
    return np.array(ys)
xx = np.zeros(20); xx[1] = 1.0                       # a single spike at position 1
yy = ssm(xx)
print(round(float(yy[1]), 3), round(float(yy[4]), 3), round(float(yy[8]), 3))   # expected output: 1.0 0.512 0.21
