"""Chapter 16: Transformers: The Architecture That Ate the World

Section-aligned walkthrough: every code block from this chapter of the book,
in book order, each in its own cell under its section heading so you can run
and check them one by one. Generated from Book.md; keep the book as the source.
"""

# %% [markdown]
# ## 16.1.1 Why self-attention wins on cost when the sequence is not too long

# %%
def layer_macs(n, d):
    qk = n * n * d          # one query-key product
    recurrent = n * d * d   # one recurrent layer
    return qk, recurrent
for n, d in [(512, 768), (4096, 768)]:
    qk, recurrent = layer_macs(n, d)
    print(f"n={n}, d={d}: attention {qk:.6e}, recurrent {recurrent:.6e}")

# %% [markdown]
# ## 16.2 Self-attention from scratch

# %% [markdown]
# Example 16-1 applies the attention formula to a 3-token sequence and shows the attention matrix.

# %%
import warnings; warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
import numpy as np, torch, torch.nn.functional as F
rng = np.random.default_rng(0)
def softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True)); return e / e.sum(axis=-1, keepdims=True)
n, d = 3, 4
X = np.array([[1.0, 0, 1, 0], [0, 2, 0, 2], [1, 1, 1, 1]])     # 3 token embeddings
WQ = rng.normal(0, 0.5, (d, d)); WK = rng.normal(0, 0.5, (d, d)); WV = rng.normal(0, 0.5, (d, d))
Q, K, V = X @ WQ, X @ WK, X @ WV                                # each token's query, key, value
scores = Q @ K.T / np.sqrt(d)                                  # how much each token matches each
weights = softmax(scores)                                     # attention weights (rows sum to 1)
out = weights @ V                                             # each output = weighted sum of values
print(np.round(weights, 3))
print("each row sums to 1:", np.allclose(weights.sum(1), 1))
tw = F.scaled_dot_product_attention(torch.tensor(Q)[None], torch.tensor(K)[None], torch.tensor(V)[None]).numpy()[0]
print("matches torch.scaled_dot_product_attention:", np.allclose(out, tw))
# weights [[0.416,0.256,0.328],[0.171,0.558,0.271],[0.322,0.358,0.320]]; rows sum to 1: True; matches PyTorch: True

# %% [markdown]
# One property is crucial and easy to miss: self-attention is permutation equivariant.

# %%
perm = [2, 0, 1]
Xp = X[perm]                                                  # shuffle the input rows
Qp, Kp, Vp = Xp @ WQ, Xp @ WK, Xp @ WV
outp = softmax(Qp @ Kp.T / np.sqrt(d)) @ Vp
print("permuted input gives permuted output?", np.allclose(outp, out[perm]))
# True: attention has NO notion of order ('dog bites man' == 'man bites dog'); positional encodings (16.4) inject it

# %% [markdown]
# ## 16.2.1 Why divide by the square root of d_k

# %% [markdown]
# Measuring it directly confirms the algebra.

# %%
import numpy as np
rng = np.random.default_rng(0)
d_k = 64
q = rng.normal(0, 1, (20000, d_k))                # 20000 random query vectors
k = rng.normal(0, 1, (20000, d_k))                # 20000 random key vectors
dots = np.sum(q * k, axis=1)                       # the dot product q.k for each pair
print(f"unscaled q.k : mean = {dots.mean():+.3f}, var = {dots.var():.2f}, std = {dots.std():.2f}")
scaled = dots / np.sqrt(d_k)                        # divide by sqrt(d_k)
print(f"q.k/sqrt(d_k): mean = {scaled.mean():+.3f}, var = {scaled.var():.3f}, std = {scaled.std():.3f}")
def softmax(z):
    e = np.exp(z - z.max()); return e / e.sum()
Q1 = rng.normal(0, 1, d_k); Kt = rng.normal(0, 1, (8, d_k))   # one query, 8 keys
logits = Kt @ Q1
print(f"softmax max weight  unscaled = {softmax(logits).max():.3f}, "
      f"scaled = {softmax(logits/np.sqrt(d_k)).max():.3f}")

# %% [markdown]
# ## 16.3 Multi-head attention

# %% [markdown]
# Example 16-2 is a constructed toy on five tokens with random weights, kept small on purpose so that each head's short attention row can be read off and checked by hand.

# %%
import numpy as np
rng = np.random.default_rng(0)
def softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True)); return e / e.sum(axis=-1, keepdims=True)
n, d, h = 5, 8, 2
dk = d // h                                                   # each head works in d/h dimensions
X = rng.normal(size=(n, d))
WQ, WK, WV, WO = (rng.normal(0, 0.5, (d, d)) for _ in range(4))
Q, K, V = X @ WQ, X @ WK, X @ WV
def to_heads(M): return M.reshape(n, h, dk).transpose(1, 0, 2)   # (n, d) -> (h, n, dk)
Qh, Kh, Vh = to_heads(Q), to_heads(K), to_heads(V)
head_w = [softmax(Qh[i] @ Kh[i].T / np.sqrt(dk)) for i in range(h)]   # one attention matrix per head
heads = [head_w[i] @ Vh[i] for i in range(h)]
mixed = np.concatenate(heads, axis=1) @ WO
print("head 0 attention (token 0's row):", np.round(head_w[0][0], 3))
print("head 1 attention (token 0's row):", np.round(head_w[1][0], 3))
print(f"the two heads differ (max abs diff = {np.abs(head_w[0]-head_w[1]).max():.3f})")
print("concatenated and projected shape:", mixed.shape)

# %% [markdown]
# Splitting one dimension-$d$ attention into $h$ heads of dimension $d/h$ keeps the projection arithmetic the same, which a count of the query projection's parameters shows.

# %%
d, h = 8, 2; dk = d // h
single = d * d                                               # Q-projection parameters, one head of size d
multi = h * (d * dk)                                         # h heads, each projecting d -> d/h
print(f"Q projection, one head of dim {d} = {single}; {h} heads of dim {dk} = {multi}")
# 64 == 64: query weights match. Each head still stores its own n by n weight matrix.

# %% [markdown]
# ## 16.4 Positional encodings

# %% [markdown]
# This gives every position a unique signature.

# %%
import numpy as np
def positional_encoding(n, d):                                # PE[pos,2i]=sin(pos/10000^(2i/d)), odd: cos
    pos = np.arange(n)[:, None]; idx = np.arange(d)[None, :]
    angle = pos / (10000 ** (2 * (idx // 2) / d))
    return np.where(idx % 2 == 0, np.sin(angle), np.cos(angle))
n, d = 12, 16
PE = positional_encoding(n, d)
print("shape =", PE.shape, " every position unique:", len(np.unique(PE.round(6), axis=0)) == n)
for p in (0, 3, 7):
    row = [PE[p] @ PE[p + k] for k in range(n - p)]
    print(f"start {p}: " + " ".join(f"{v:.2f}" for v in row))

# %% [markdown]
# The `swap01` experiment of this section is a constructed toy with random weights on a short sequence, arranged so the effect of position can be confirmed by moving a single token and comparing two rows by hand.

# %%
rng = np.random.default_rng(0)
X = rng.normal(size=(n, d))
WQ, WK, WV = (rng.normal(0, 0.3, (d, d)) for _ in range(3))
def softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)
def attn(Z):
    Q, K, V = Z @ WQ, Z @ WK, Z @ WV
    return softmax(Q @ K.T / np.sqrt(d)) @ V
def swap01(M):
    M2 = M.copy(); M2[[0, 1]] = M2[[1, 0]]; return M2          # move the token at row 1 to row 0
print("same representation regardless of position?")
print("  without positional encoding:", np.allclose(attn(X)[1], attn(swap01(X))[0]))            # True (order blind)
print("  with positional encoding   :", np.allclose(attn(X + PE)[1], attn(swap01(X) + PE)[0]))  # False (order encoded)

# %% [markdown]
# ## 16.4.1 Rotary embeddings (RoPE), ALiBi, and relative position

# %% [markdown]
# Example 16-4 confirms it in NumPy: it rotates the same $\mathbf{q}$ and $\mathbf{k}$ for several position pairs that share an offset and reads off the dot product.

# %%
import numpy as np
def rope(vec, pos, base=10000.0):                  # rotate each 2D pair i by pos * theta_i
    d = vec.shape[0]; out = np.empty_like(vec)
    for i in range(d // 2):                        # i indexes the pair, as in theta_i
        ang = pos * base ** (-2.0 * i / d)
        cos_a, sin_a = np.cos(ang), np.sin(ang)
        x0, x1 = vec[2*i], vec[2*i+1]
        out[2*i], out[2*i+1] = x0*cos_a - x1*sin_a, x0*sin_a + x1*cos_a   # 2x2 rotation of the pair
    return out
rng = np.random.default_rng(0)
q, k = rng.normal(size=8), rng.normal(size=8)
for t, s in [(5, 3), (9, 7), (2, 0)]:              # same offset t-s = 2, different absolute positions
    print(f"t={t}, s={s}, offset {t-s}: q_t . k_s = {rope(q, t) @ rope(k, s):+.5f}")
for t, s in [(5, 2), (9, 5)]:                      # different offsets, for contrast
    print(f"t={t}, s={s}, offset {t-s}: q_t . k_s = {rope(q, t) @ rope(k, s):+.5f}")

# %% [markdown]
# Example 16-5 tabulates the bias and makes the ladder concrete.

# %%
import numpy as np
def alibi_slopes(n_heads):
    # Press, Smith, and Lewis (2022): start at 2^(-8/h), then the same ratio
    start = 2.0 ** (-8.0 / n_heads)
    return start ** np.arange(1, n_heads + 1)
m = alibi_slopes(8)
m16 = alibi_slopes(16)
print("8-head slopes:", " ".join(f"{slope:.6g}" for slope in m))
print(f"16-head slopes start {m16[0]:.6f}, then {m16[1]:.6f}, end {m16[-1]:.8f}")
for dist in [0, 1, 4, 16]:
    print(f"|t-s| = {dist:2d}: head-1 bias {-m[0] * dist:+6.2f},  head-8 bias {-m[7] * dist:+.4f}")

# %% [markdown]
# ## 16.5 The rest of the block: LayerNorm, residuals, and feed-forward layers

# %% [markdown]
# Unlike the batch normalization of 11.4, which normalizes each feature across the batch, layer normalization normalizes each token's vector across its own features: for a token $\mathbf{x}$ with $d$ features, mean $\mu$, and variance $\sigma^2$ over those features, feature $j$ becomes $\gamma_j\,(x_j - \mu)/\sqrt{\sigma^2 + \epsilon} + \beta_j$, which is zero mean and unit variance rescaled by a learned gain $\gamma_j$ and shift $\beta_j$, with $\epsilon$ a small constant that keeps the division safe.

# %%
import numpy as np
rng = np.random.default_rng(0)
def layernorm(x, gamma=1.0, beta=0.0, eps=1e-5):
    mu = x.mean(-1, keepdims=True); var = x.var(-1, keepdims=True)
    return gamma * (x - mu) / np.sqrt(var + eps) + beta       # normalize each TOKEN across its features
n, d = 4, 16
x = rng.normal(3, 2, (n, d))                                  # tokens with arbitrary mean and scale
xn = layernorm(x)
print("before LayerNorm: per-token mean ~", x.mean(-1).round(2))    # [2.47 2.92 4.02 3.12]
print("after  LayerNorm: per-token mean =", xn.mean(-1).round(3), "std =", xn.std(-1).round(3))   # ~0 and 1
# LayerNorm uses ONLY each token's own features, so it is batch-size independent (unlike BatchNorm)

# %% [markdown]
# The third is the position-wise feed-forward network, a small two-layer MLP applied to each token independently, that expands the dimension by a factor of four, applies a nonlinearity, and projects back.

# %%
W1, b1 = rng.normal(0, 0.1, (d, 4 * d)), np.zeros(4 * d)
W2, b2 = rng.normal(0, 0.1, (4 * d, d)), np.zeros(d)
def ffn(z): return np.maximum(0, z @ W1 + b1) @ W2 + b2       # a ReLU MLP applied to each token: d -> 4d -> d
ffn_params = d * 4 * d + 4 * d * d                            # the two FFN matrices: 8 d^2
attn_params = 4 * d * d                                       # W_Q, W_K, W_V, W_O: 4 d^2
print(f"feed-forward {d} -> {4*d} -> {d}: {ffn_params:,} params  vs  attention (4 matrices of {d}x{d}): {attn_params:,}")
print(f"FFN share of the block: {ffn_params / (ffn_params + attn_params):.2f}")
# FFN 2,048 vs attention 1,024: the position-wise FFN holds two thirds of the block's parameters

# %% [markdown]
# Example 16-8 combines an attention function with the `layernorm` of Example 16-6 and the `ffn` of Example 16-7. It is a constructed toy with random weights on four tokens of sixteen dimensions, sized so that the shape going in equals the shape coming out and every step can be traced by hand.

# %%
WQ, WK, WV = (rng.normal(0, 0.3, (d, d)) for _ in range(3))
def softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)
def attention(z):
    Q, K, V = z @ WQ, z @ WK, z @ WV
    return softmax(Q @ K.T / np.sqrt(d)) @ V
def block(x):                                                 # a pre-LN transformer block
    x = x + attention(layernorm(x))                           # residual around attention
    x = x + ffn(layernorm(x))                                 # residual around the feed-forward
    return x
out = block(x)
print(f"transformer block: input {x.shape} -> output {out.shape}")
# (4, 16) -> (4, 16): same shape in and out, which is exactly why blocks can be stacked into a deep transformer

# %% [markdown]
# ## 16.5.1 Modern variants: GELU, SwiGLU, and RMSNorm

# %% [markdown]
# Example 16-9 runs both norms on one vector, beside the GELU and SwiGLU numbers, so the transformer-specific point, what RMSNorm leaves alone, is visible.

# %%
import numpy as np
rng = np.random.default_rng(1)
v = rng.normal(3, 2, 8)                             # one token vector, arbitrary mean and scale
def layernorm(x, eps=1e-5):                         # center AND scale
    return (x - x.mean()) / np.sqrt(x.var() + eps)
def rmsnorm(x, eps=1e-5):                           # scale only, by the root mean square
    return x / np.sqrt((x**2).mean() + eps)
print("LayerNorm mean =", round(layernorm(v).mean(), 3), " std =", round(layernorm(v).std(), 3))
print("RMSNorm   mean =", round(rmsnorm(v).mean(), 3), " RMS scale =", round(np.sqrt((v**2).mean()), 3))
def gelu(x):                                        # tanh approximation (GPT-2 / BERT)
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
for xv in [-2.0, -0.5, 0.5, 2.0]:
    print(f"x={xv:+.1f}: ReLU={max(0.0, xv):+.3f}, GELU={gelu(np.array(xv)):+.3f}")
def swish(z): return z / (1 + np.exp(-z))           # Swish(z) = z * sigmoid(z)
rng2 = np.random.default_rng(0)
d2, h2 = 6, 16                                      # SwiGLU hidden width 8/3 * d: 16 for d = 6
W1s, W3s, W2s = (rng2.normal(0, 0.4, s) for s in [(d2, h2), (d2, h2), (h2, d2)])
xs = rng2.normal(0, 1, d2)                          # one token vector
sw_out = (swish(xs @ W1s) * (xs @ W3s)) @ W2s       # gate one branch with the other, project back
print("SwiGLU output:", sw_out.round(3))
print(f"params: plain FFN 2*(d*4d) = {2 * d2 * 4 * d2},  SwiGLU 3*(d*8d/3) = {3 * d2 * h2}  (equal)")

# %% [markdown]
# ## 16.6 Encoder, decoder, and encoder-decoder

# %% [markdown]
# Example 16-10 is a constructed toy at a small dimension: seven random vectors stand in for the encoder's output, three for the decoder's tokens so far, and the one change from the self-attention of 16.2 is where the queries come from.

# %%
import numpy as np
rng = np.random.default_rng(0)
def softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)
n_enc, n_dec, d = 7, 3, 8                          # 7 source tokens, 3 generated so far
enc_out = rng.normal(size=(n_enc, d))              # encoder output: keys and values come from here
dec_x = rng.normal(size=(n_dec, d))                # decoder representations: queries come from here
WQ, WK, WV = (rng.normal(0, 0.4, (d, d)) for _ in range(3))
Q, K, V = dec_x @ WQ, enc_out @ WK, enc_out @ WV   # the one change from self-attention
weights = softmax(Q @ K.T / np.sqrt(d))            # scores are n_dec x n_enc
out = weights @ V
print(f"score matrix {weights.shape}, rows sum to 1: {np.allclose(weights.sum(1), 1)}, output {out.shape}")
print("output token 0 draws on the 7 inputs as:", weights[0].round(2))

# %% [markdown]
# T5 frames every task as text to text, so an instruction prefix on the input selects the task, and the Hugging Face classes met in 15.5 load and run it.

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
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("google-t5/t5-small")
t5 = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small").eval()
ids = tok("translate English to French: The cat sat on the mat.", return_tensors="pt").input_ids
out_ids = t5.generate(ids, max_new_tokens=20)
print(tok.decode(out_ids[0], skip_special_tokens=True))
pieces = [p for p in tok.convert_ids_to_tokens(out_ids[0].tolist()) if p not in ("<pad>", "</s>")]
print(len(pieces), "sentencepiece pieces:", pieces)
print(f"parameters: {sum(p.numel() for p in t5.parameters()):,}")

# %% [markdown]
# ## 16.7 Masked and causal attention

# %% [markdown]
# Bidirectional attention uses no mask, letting every token attend everywhere.

# %%
import numpy as np
rng = np.random.default_rng(0)
def softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)
n = 5
scores = rng.normal(0, 1, (n, n))                             # raw attention scores Q@K.T/sqrt(d)
bi = softmax(scores)                                          # no mask: every token attends to all
print("bidirectional (token 0's row):", np.round(bi[0], 3))
# [0.202 0.156 0.339 0.198 0.104]: nonzero everywhere, the BERT encoder's full view of the sequence

# %% [markdown]
# Causal attention, used by decoders, applies a triangular mask that sets every future position to negative infinity, so each token can attend only to itself and the tokens before it, which is what makes autoregressive generation valid: when predicting the next token, the model must not peek at it.

# %%
causal_mask = np.triu(np.full((n, n), -np.inf), k=1)          # everything above the diagonal -> -inf
causal = softmax(scores + causal_mask)
print(np.round(causal, 2))
print("future weights all 0:", np.allclose(causal[np.triu_indices(n, k=1)], 0),
      " rows still sum to 1:", np.allclose(causal.sum(1), 1))
# lower-triangular: each token sees only itself and earlier tokens, the GPT decoder's mask for generation

# %% [markdown]
# A third kind, the padding mask, handles the practical problem that batched sequences have different lengths and are padded to a common length with filler tokens; the padding mask sets those positions to negative infinity so no real token wastes attention on them.

# %%
pad_mask = np.zeros((n, n)); pad_mask[:, [3, 4]] = -np.inf     # mask the padded COLUMNS so nobody attends to them
padded = softmax(scores + pad_mask)
print(f"attention weight on the padded positions 3,4 = {padded[:, [3, 4]].sum():.3f}")
# 0.000: padding tokens (filler that makes a batch rectangular) are completely ignored

# %% [markdown]
# ## 16.8 Training paradigms: masked language modeling vs next-token prediction

# %% [markdown]
# So from one sentence we get a full set of training pairs for free, by sliding along.

# %%
text = "the cat sat on the mat".split()
inp, tgt = text[:-1], text[1:]              # target is the input shifted left by one
print("tokens:", text)
print("\nnext-token training pairs (predict each token from all previous):")
for i in range(len(inp)):
    print(f"  given {text[:i+1]} -> predict '{text[i+1]}'")
print("input = sequence[:-1], target = sequence[1:]")

# %% [markdown]
# DistilGPT-2 is a next-token model; given a prompt it returns a probability distribution over only the next token, computed from the left context alone.

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
import warnings; warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2").eval()
ids = tok("The cat sat on the", return_tensors="pt").input_ids
with torch.no_grad():
    probs = torch.softmax(model(ids).logits[0, -1], dim=-1)   # only the last position's row
for p, i in zip(*torch.topk(probs, 5)):
    print(f"  {tok.decode(i)!r:>10}  {p.item():.3f}")

# %% [markdown]
# One nuance from the original BERT recipe (Devlin, Chang, Lee, and Toutanova, 2019): of the fifteen percent of positions selected for prediction, only 80 percent are actually replaced by [MASK], while 10 percent are swapped for a random token and 10 percent are left unchanged, so the model cannot rely on the [MASK] symbol, which never appears in downstream text, always marking the positions it must get right.

# %%
text = "the cat sat on the mat".split()
mask_positions = [1, 4]                     # 2 of 6 tokens, fixed here for the demo; BERT selects about 15% at random during training
targets = {p: text[p] for p in mask_positions}
shown = [w if i not in mask_positions else "[MASK]" for i, w in enumerate(text)]
print("masked input :", shown)
print("targets      : " + ", ".join(f"pos {p} -> '{targets[p]}'" for p in mask_positions))

# %% [markdown]
# BERT, loaded through a fill-mask pipeline, recovers a [MASK] token using the words on both sides of it.

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
import warnings; warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
from transformers import pipeline

fill = pipeline("fill-mask", model="google-bert/bert-base-uncased", device="cpu")
for sent in ["The capital of France is [MASK].", "The cat sat on the [MASK]."]:
    print(sent)
    for p in fill(sent, top_k=3):
        print(f"  {p['token_str']:>8}  {p['score']:.3f}")

# %% [markdown]
# ## 16.8.1 Span corruption

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
from datasets import load_dataset
ds = load_dataset("Trelis/tiny-shakespeare")["train"]
line = ""
for row in ds:
    for candidate in row["Text"].splitlines():
        if len(candidate.split()) >= 8:
            line = candidate.strip()
            break
    if line:
        break
tokens = line.split()
start, span = 2, 3
corrupted = tokens[:start] + ["<extra_id_0>"] + tokens[start + span:]
target = ["<extra_id_0>"] + tokens[start:start + span] + ["<extra_id_1>"]
print(line)
print(" ".join(corrupted))
print(" ".join(target))

# %% [markdown]
# The real T5 tokenizer and model take exactly this format.

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
import math, torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("google-t5/t5-small")
t5 = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small").eval()
enc_ids = tok(" ".join(corrupted), return_tensors="pt").input_ids    # the damaged line, for the encoder
label_ids = tok(" ".join(target), return_tensors="pt").input_ids     # the sentinel target, for the decoder
print("sentinel ids  :", tok.convert_tokens_to_ids(["<extra_id_0>", "<extra_id_1>"]))
print("encoder input :", tok.convert_ids_to_tokens(enc_ids[0]))
print("decoder target:", tok.convert_ids_to_tokens(label_ids[0]))
with torch.no_grad():
    loss = t5(input_ids=enc_ids, labels=label_ids).loss               # next-token cross-entropy on the target
print(f"loss {loss.item():.3f} nats per target token, perplexity {math.exp(loss.item()):.1f}, "
      f"uniform guess {math.log(t5.config.vocab_size):.3f} nats")

# %% [markdown]
# ## 16.9 Implement a mini-transformer (GPT-style) from scratch in PyTorch

# %% [markdown]
# Example 16-12 loads the data, builds the character vocabulary, fixes the hyperparameters that the rest of the listings refer to, and pins PyTorch to two CPU threads, because the trained weights change slightly with the number of threads, and the printed losses and the sampled text change with them.

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
import warnings; warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset

torch.manual_seed(0)
torch.set_num_threads(2)   # pin two CPU threads: the trained weights change slightly with the thread count

# Real public data: Trelis/tiny-shakespeare, ~30k characters is plenty for a tiny model.
ds = load_dataset("Trelis/tiny-shakespeare")["train"]
text = "".join(ds[i]["Text"] for i in range(12))    # concatenate the first 12 rows

chars = sorted(set(text))
V = len(chars)                                       # the vocabulary is the distinct characters
stoi = {c: i for i, c in enumerate(chars)}           # char -> integer id
itos = {i: c for i, c in enumerate(chars)}           # integer id -> char
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
print(f"corpus: {len(text)} chars, vocab size V = {V}")

block_size = 32        # context length: how many previous chars the model sees at once
n_embd = 64            # embedding / model dimension
n_head = 4             # attention heads (each of size n_embd / n_head = 16)
n_layer = 3            # stacked transformer blocks
batch_size = 32
steps = 2000
lr = 3e-3

def get_batch():                                     # a batch of (context, next-char) pairs
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])   # targets = inputs shifted by one
    return x, y

# %% [markdown]
# Example 16-13 packs the queries, keys, and values into one linear projection, splits them across heads, computes scaled dot-product scores, masks the future with the lower-triangular trick from Section 16.7, softmaxes, and mixes the heads back together.

# %%
class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(n_embd, 3 * n_embd)        # queries, keys, values at once
        self.proj = nn.Linear(n_embd, n_embd)           # mix heads back together
        self.n_head = n_head
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(n_embd, dim=2)
        hs = C // self.n_head
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)   # (B, heads, T, head_size)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / (hs ** 0.5)        # scaled dot-product scores
        att = att.masked_fill(self.mask[:T, :T] == 0, float('-inf'))   # mask the future
        att = F.softmax(att, dim=-1)
        out = att @ v                                         # weighted sum of values
        out = out.transpose(1, 2).contiguous().view(B, T, C) # concatenate heads
        return self.proj(out)

# %% [markdown]
# The two residual paths are the gradient highways that let many blocks stack without the signal washing out.

# %%
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention()
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(), nn.Linear(4 * n_embd, n_embd))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))      # residual around attention
        x = x + self.ffn(self.ln2(x))       # residual around feed-forward
        return x

# %% [markdown]
# Example 16-15 assembles the full model. Its generate method is autoregressive: predict the distribution for the next character, sample one, append it, crop the context to the last block_size characters, and repeat.

# %%
class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(V, n_embd)              # token -> vector
        self.pos_emb = nn.Embedding(block_size, n_embd)     # position -> vector
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, V)                    # vector -> next-token logits

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T)
        x = self.tok_emb(idx) + self.pos_emb(pos)           # token meaning + position
        x = self.ln_f(self.blocks(x))
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, V), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, n_new):
        for _ in range(n_new):
            logits, _ = self(idx[:, -block_size:])          # crop to the context window
            probs = F.softmax(logits[:, -1, :], dim=-1)     # next-token distribution
            nxt = torch.multinomial(probs, 1)               # sample one token
            idx = torch.cat([idx, nxt], dim=1)
        return idx

# %% [markdown]
# Example 16-16 trains the model by next-token prediction with the AdamW optimizer, and the loss falls steadily from the random-guessing baseline.

# %%
model = MiniGPT()
n_params = sum(p.numel() for p in model.parameters())
print(f"mini-GPT: {n_layer} layers, {n_head} heads, d={n_embd}, {n_params:,} parameters")
print(f"loss of a random model should be about ln(V) = {torch.log(torch.tensor(float(V))):.3f}")

# Rows 12-23 of the same dataset are never in the training string (rows 0-11).
held_text = "".join(ds[i]["Text"] for i in range(12, 24))
held = torch.tensor([stoi[c] for c in held_text if c in stoi], dtype=torch.long)
print(f"held-out: {len(held_text)} chars, {len(held)} in the vocabulary, {len(held_text) - len(held)} dropped")
g = torch.Generator().manual_seed(1)
val_batches = []
for _ in range(8):
    ix = torch.randint(0, len(held) - block_size - 1, (batch_size,), generator=g)
    xb = torch.stack([held[i:i + block_size] for i in ix])
    yb = torch.stack([held[i + 1:i + block_size + 1] for i in ix])
    val_batches.append((xb, yb))

def val_loss():
    total = 0.0
    with torch.no_grad():
        for xb, yb in val_batches:
            _, vloss = model(xb, yb)
            total += vloss.item()
    return total / len(val_batches)

opt = torch.optim.AdamW(model.parameters(), lr=lr)
_, loss0 = model(*get_batch())
print(f"\nstep    0: loss {loss0.item():.3f}  (perplexity {torch.exp(loss0).item():.1f})  val {val_loss():.3f}")
for step in range(1, steps + 1):
    x, y = get_batch()
    _, loss = model(x, y)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 500 == 0:
        print(f"step {step:4d}: loss {loss.item():.3f}  (perplexity {torch.exp(loss).item():.2f})  val {val_loss():.3f}")

# %% [markdown]
# Sampling from the seed character "T" produces Shakespeare-shaped text, not a clean scene.

# %%
torch.manual_seed(1)
start = torch.tensor([[stoi['T']]], dtype=torch.long)   # seed with a capital T
out = model.generate(start, n_new=300)[0].tolist()
print("".join(itos[i] for i in out))

# %% [markdown]
# ## 16.10 The Hugging Face ecosystem

# %% [markdown]
# The `AutoTokenizer` loads the exact vocabulary the model was trained with.

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
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
sentence = "Transformers are powerful."
ids = tok(sentence)["input_ids"]            # text -> subword ids
pieces = tok.convert_ids_to_tokens(ids)     # the human-readable subwords
print(f"text   : {sentence!r}")
print(f"ids    : {ids}")
print(f"pieces : {pieces}")
print(f"decode : {tok.decode(ids)!r}")      # ids -> the original string

# %% [markdown]
# A short comparison block reads both settings side by side: the DistilGPT-2 column comes from its configuration, and the mini-GPT column from the hyperparameters of Example 16-12 and a fresh, untrained copy of Example 16-15. It reuses those names, so run Example 16-12, which also imports torch.nn as nn, and Examples 16-13 to 16-15 first, in the same Python session.

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
from transformers import AutoConfig

cfg = AutoConfig.from_pretrained("distilbert/distilgpt2")
toy = MiniGPT()                                       # a fresh, untrained copy of Example 16-15, only inspected
toy_act = toy.blocks[0].ffn[1]                        # the nn.GELU() inside the first block's feed-forward
toy_drop = [mod.p for mod in toy.modules() if isinstance(mod, nn.Dropout)]
rows = [("layers", n_layer, cfg.n_layer), ("heads", n_head, cfg.n_head),
        ("model dim d", n_embd, cfg.n_embd), ("vocabulary", V, cfg.vocab_size),
        ("context", block_size, cfg.n_positions),
        ("activation", "gelu, exact" if toy_act.approximate == "none" else "gelu, tanh",
         cfg.activation_function),
        ("dropout rates", toy_drop or "none", (cfg.attn_pdrop, cfg.resid_pdrop, cfg.embd_pdrop)),
        ("tied head", toy.head.weight is toy.tok_emb.weight, cfg.tie_word_embeddings)]
print(f"{'setting':14s} {'mini-GPT':>12s} {'distilbert/distilgpt2':>16s}")
for name, ours, theirs in rows:
    print(f"{name:14s} {str(ours):>12s} {str(theirs):>16s}")

# %% [markdown]
# Loading its weights and generating from a prompt with greedy decoding takes a few lines.

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
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
n_params = sum(p.numel() for p in model.parameters())
prompt = "The future of artificial intelligence is"
in_ids = tok(prompt, return_tensors="pt").input_ids
out_ids = model.generate(in_ids, max_new_tokens=12, do_sample=False,   # greedy: reproducible
                         pad_token_id=tok.eos_token_id)
generated = tok.decode(out_ids[0], skip_special_tokens=True).strip()
print(f"distilbert/distilgpt2 has {n_params:,} parameters ({n_params/1e6:.0f}M)")
print(f"prompt    : {prompt!r}")
print(f"generated : {generated!r}")

# %% [markdown]
# For most applications you go one level higher still, to the `pipeline`, which bundles tokenizer, model, and pre- and post-processing into a single callable.

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
from transformers import pipeline

clf = pipeline("sentiment-analysis",
               model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", device="cpu")
for txt in ["This book is wonderful.", "That was a complete waste of time."]:
    r = clf(txt)[0]
    print(f"  {txt!r} -> {r['label']} ({r['score']:.2f})")

# %% [markdown]
# A small public split downloads in seconds.

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
from datasets import load_dataset

shakespeare = load_dataset("Trelis/tiny-shakespeare", split="train")
print("rows        :", shakespeare.num_rows)
print("columns     :", shakespeare.column_names)
print("row 0 length:", len(shakespeare[0]["Text"]), "chars")

# %% [markdown]
# ## 16.10.1 Fine-tuning DistilBERT on Rotten Tomatoes with the Trainer

# %% [markdown]
# Examples 16-17 to 16-19 fine-tune distilbert-base-uncased on all 8,530 Rotten Tomatoes training sentences. Two epochs take tens of minutes on the reference CPU, so the three listings run in the companion script `08_trainer_finetune_rotten_tomatoes.py` rather than in this walkthrough.

# %% [markdown]
# ## 16.10.2 The same loop on any hardware with Accelerate

# %% [markdown]
# Example 16-20 runs one epoch of the mini-GPT twice, from the same starting weights and in the same batch order, once with a plain loop and once through Accelerate. It reuses Examples 16-12 to 16-15.

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
from accelerate import Accelerator
from torch.utils.data import DataLoader, TensorDataset

windows = data.unfold(0, block_size + 1, block_size)        # 33-character windows, a new one every 32
train_set = TensorDataset(windows[:, :-1], windows[:, 1:])  # (context, next-character targets) pairs

def one_epoch(accelerator=None):
    torch.manual_seed(0)                                    # the same starting weights on every call
    model = MiniGPT()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(0))   # the same batch order on every call
    if accelerator is not None:
        model, opt, loader = accelerator.prepare(model, opt, loader)  # wrapped for the setup it found
    losses = []
    for x, y in loader:                                     # with Accelerate, on accelerator.device
        _, loss = model(x, y)
        opt.zero_grad()
        if accelerator is None:
            loss.backward()
        else:
            accelerator.backward(loss)                      # in place of loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses

plain = one_epoch()                                         # a plain PyTorch loop
accelerator = Accelerator(cpu=True)                                 # reads how the script was launched
accel = one_epoch(accelerator)                              # the same loop through Accelerate
print(f"device {accelerator.device}, processes {accelerator.num_processes}, "
      f"mixed precision {accelerator.mixed_precision}")
print(f"{len(train_set)} windows, {len(plain)} steps; loss {plain[0]:.3f} at the first step, "
      f"{plain[-1]:.3f} at the last")
print("identical losses with and without Accelerate:", plain == accel)

# %% [markdown]
# ## 16.11 Computational cost: why attention is quadratic, and the tricks that help

# %% [markdown]
# The score-matrix sizes make the wall concrete.

# %%
print("seq_len n   score-matrix entries   memory (MB, float32)")
prev = None
for n in [512, 1024, 2048, 4096]:
    entries = n * n                 # one score per ordered pair of tokens
    mem_mb = entries * 4 / 1e6      # float32 = 4 bytes each
    ratio = "" if prev is None else f"   ({entries // prev}x)"
    print(f"  {n:5d}      {entries:14,}        {mem_mb:10.1f}{ratio}")
    prev = entries

# %% [markdown]
# Measuring wall-clock time tests that growth directly: a short timing loop runs a full NumPy attention at two lengths, keeps the fastest of 20 calls at each, and prints in parentheses each time divided by the time at the previous length.

# %%
import time
import numpy as np
rng = np.random.default_rng(0)
def softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)
d = 64
prev_ms = None
for n in [1024, 2048]:
    Q, K, V = (rng.normal(size=(n, d)).astype(np.float32) for _ in range(3))
    # keep the fastest of 20 calls, the best of several repeats habit of 1.3.4
    best = float("inf")
    for _ in range(20):
        t0 = time.perf_counter()
        out = softmax(Q @ K.T / np.sqrt(d)) @ V     # one full attention
        best = min(best, time.perf_counter() - t0)
    ms = best * 1000
    # the measured time divided by the time at the previous length
    ratio = "" if prev_ms is None else f"   ({ms / prev_ms:.1f}x)"
    print(f"n = {n}: one attention takes {ms:5.1f} ms{ratio}")
    prev_ms = ms

# %% [markdown]
# ## 16.11.1 The KV cache: the trick that makes chat fast

# %% [markdown]
# The keys and values for past tokens never change, so caching them and computing only the new token's scores drops each step to $O(t)$ and the whole reply to about $T^2/2$.

# %%
T = 1000
no_cache = sum(t * t for t in range(1, T + 1))   # score pairs if the prefix is rerun
with_cache = sum(t for t in range(1, T + 1))     # only the new token's scores
print(f"generating {T} tokens (score-pair units, not projections):")
print(f"  no KV cache  : {no_cache:,.0f}  (~T^3/3, rerun the prefix)")
print(f"  with KV cache: {with_cache:11,.0f}  (~T^2/2, reuse stored keys and values)")
print(f"  KV cache is ~{no_cache / with_cache:.0f}x less score work here")

# %% [markdown]
# ## 16.11.3 Grouped-query and multi-query attention: shrinking the KV cache

# %% [markdown]
# Full multi-head attention keeps all 32 key/value heads, GQA with eight groups keeps eight, and MQA keeps one.

# %%
n_heads = 32   # h, the number of query heads (MHA keeps one K/V head for each)
n_layers, d_head, seq_len, batch, bytes_per = 32, 128, 4096, 1, 2  # ~7B shape, fp16

def kv_cache_bytes(n_kv_heads):
    # keys and values (the leading 2), for every layer, head, and stored token, in bytes
    return 2 * n_layers * n_kv_heads * d_head * seq_len * batch * bytes_per

for name, kv in [("full multi-head (MHA)", n_heads), ("GQA, 8 groups", 8), ("MQA, 1 group", 1)]:
    gb = kv_cache_bytes(kv) / 1e9
    saving = "(baseline)" if kv == n_heads else f"({n_heads // kv}x smaller than MHA)"
    print(f"  {name:22s}: {kv:2d} K/V heads   cache = {gb:6.3f} GB   {saving}")

# %% [markdown]
# ## 16.11.4 Exact attention without storing the score matrix

# %%
import numpy as np
def softmax_rows(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)
def full_attn(Q, K, V):
    d = Q.shape[-1]
    return softmax_rows(Q @ K.T / np.sqrt(d)) @ V
def online_attn(Q, K, V, block=64):
    n, d = Q.shape
    out = np.zeros_like(Q)
    m = np.full(n, -np.inf)
    ell = np.zeros(n)
    scale = np.sqrt(d)
    for s in range(0, n, block):
        scores = Q @ K[s:s + block].T / scale
        m_new = np.maximum(m, scores.max(axis=1))
        exp_old = np.exp(m - m_new)
        exp_s = np.exp(scores - m_new[:, None])
        ell = ell * exp_old + exp_s.sum(axis=1)
        out = out * exp_old[:, None] + exp_s @ V[s:s + block]
        m = m_new
    return out / ell[:, None]
rng = np.random.default_rng(0)
n, d, block = 128, 32, 64
Q, K, V = (rng.normal(size=(n, d)) for _ in range(3))
gap = np.max(np.abs(full_attn(Q, K, V) - online_attn(Q, K, V, block)))
print(f"max abs difference {gap:.3e}")
print(f"full score entries {n * n}, one tile {n * block}")
print(f"at n=4096, float32 matrix {4096 * 4096 * 4 / 1e6:.1f} MB, one tile of 64 {4096 * 64 * 4 / 1e6:.1f} MB")

# %% [markdown]
# ## 16.12.1 Mixture of Experts: more parameters, same compute per token

# %% [markdown]
# Example 16-23 is a minimal illustrative toy: a random, untrained router over untrained experts, built only to expose this arithmetic, not a trained model.

# %%
import numpy as np
rng = np.random.default_rng(0)
def softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True)); return e / e.sum(axis=-1, keepdims=True)

n_experts, top_k, d = 8, 2, 64
W_r = rng.normal(0, 1, (d, n_experts))       # the router, stored d x N: one score per expert
x = rng.normal(0, 1, d)                      # a single token's vector
p = softmax(x @ W_r / np.sqrt(d))            # router probabilities p_i(x); the sqrt(d) is a toy device
chosen = np.argsort(p)[::-1][:top_k]         # the top-k experts win
g = p[chosen] / p[chosen].sum()              # gates renormalized over the winners (Mixtral)
W1 = rng.normal(0, 0.1, (n_experts, d, 4 * d))   # every expert is its own d -> 4d -> d ReLU network
W2 = rng.normal(0, 0.1, (n_experts, 4 * d, d))
def expert(i, v): return np.maximum(0, v @ W1[i]) @ W2[i]
y = sum(g[j] * expert(chosen[j], x) for j in range(top_k))   # y = g_4 E_4(x) + g_1 E_1(x)
print(f"router probabilities: {np.round(p, 3)}")
print(f"winners {chosen.tolist()} renormalized to {np.round(g, 2).tolist()}")
print(f"layer output y: shape {y.shape}, norm {np.linalg.norm(y):.3f}")

batch = rng.normal(0, 1, (512, d))                          # a batch of 512 token vectors
P_tok = softmax(batch @ W_r / np.sqrt(d))                   # p_i(x) for every token in the batch
f = np.bincount(P_tok.argmax(1), minlength=n_experts) / len(batch)   # share whose top choice is expert i
P = P_tok.mean(0)                                           # mean router probability of expert i
alpha = 0.01
print("f =", np.round(f, 3))
print("P =", np.round(P, 3))
print(f"aux loss = {alpha * n_experts * np.sum(f * P):.4f}   "
      f"(perfectly uniform: {alpha:.4f}, all tokens on one expert: {alpha * n_experts:.4f})")

# %% [markdown]
# The payoff is the gap between capacity and compute.

# %%
ffn_params_each = 2 * d * 4 * d              # one expert feed-forward net: d -> 4d -> d
total_params = n_experts * ffn_params_each   # every expert exists in memory (capacity)
active_params = top_k * ffn_params_each      # only the chosen experts run (compute)
print(f"total expert params : {total_params:,}  (the model's CAPACITY)")
print(f"active params/token : {active_params:,}  (the COMPUTE actually done)")
print(f"MoE gives {total_params / active_params:.0f}x the capacity for the same per-token compute.")

# %% [markdown]
# ## 16.12.2 State-space models: an O(n) alternative to attention

# %% [markdown]
# Example 16-24 codes that scalar form, a deliberate simplification rather than a trained Mamba layer: a fixed decay applied to one input spike, so the memory mechanic can be checked by hand.

# %%
def ssm(x, a=0.9, b=1.0, c=1.0):
    h, ys = 0.0, []
    for xt in x:              # one left-to-right pass, no n-by-n matrix is ever formed
        h = a * h + b * xt    # the state carries a fading summary of all of the past
        ys.append(c * h)      # read the state out
    return np.array(ys)

x = np.zeros(20); x[2] = 1.0  # a single spike early in the sequence
y = ssm(x)
print("input spike at position 2; SSM output decays forward:")
print(f"  y[2]={y[2]:.3f}  y[5]={y[5]:.3f}  y[10]={y[10]:.3f}  y[15]={y[15]:.3f}")
kernel = 1.0 * 0.9 ** np.arange(len(x)) * 1.0          # K_m = c a^m b, the unrolled recurrence
print("convolution with K matches the loop:", np.allclose(np.convolve(x, kernel)[:len(x)], y))

# %% [markdown]
# Example 16-25 gives the step size one job: read a marker feature and decide which of two identical spikes to keep.

# %%
def selective_ssm(values, delta, A=-1.0, B=1.0, C=1.0):   # one channel, state size N = 1
    h, ys = 0.0, []
    for x_t, d_t in zip(values, delta):
        a_bar = np.exp(d_t * A)               # A_bar = exp(Delta A), the zero-order hold
        b_bar = (a_bar - 1.0) / A * B         # B_bar = (Delta A)^-1 (exp(Delta A) - 1) Delta B
        h = a_bar * h + b_bar * x_t
        ys.append(C * h)
    return np.array(ys)

tokens = np.zeros((20, 2))                    # each token is (value, marker)
tokens[2] = [1.0, 1.0]                        # a spike whose marker says: keep this
tokens[10] = [1.0, 0.0]                       # the same spike with no marker
w, bias = np.array([0.0, 4.0]), -3.0          # the step size reads only the marker
delta = np.log1p(np.exp(tokens @ w + bias))   # Delta_t = softplus(w . x_t + bias)
print(f"step size: marked token {delta[2]:.3f}, every other token {delta[0]:.3f}")
y_fix = ssm(tokens[:, 0])                     # the fixed decay of Example 16-24, both spikes alike
y_sel = selective_ssm(tokens[:, 0], delta)
for name, yy in [("fixed decay", y_fix), ("selective  ", y_sel)]:
    print(f"{name}: " + "  ".join(f"y[{p}]={yy[p]:.3f}" for p in (2, 5, 10, 13, 18)))

# %% [markdown]
# Because the model is one left-to-right pass, its cost grows linearly with length rather than quadratically.

# %%
print("cost to process a sequence of length n:")
for n in [1000, 100000]:
    print(f"  n={n:6d}: attention ~ n^2 = {n * n:,}   vs   SSM ~ n = {n:,}")
d, N = 4096, 16                                # a model width and a state size per channel
for n in [1000, 100000]:
    print(f"  n={n:6d}, d={d}, N={N}: scores n^2 d = {n * n * d:.2e}, scan n d N = {n * d * N:.2e}, "
          f"one projection n d^2 = {n * d * d:.2e}")

# %% [markdown]
# ## 16.13 Coding exercises

# %% [markdown]
# Exercise 1 starter.

# %%
import numpy as np
def attention(Q, K, V):
    ...
rng = np.random.default_rng(0)
n, d = 4, 8
X = rng.normal(size=(n, d))
WQ, WK, WV = (rng.normal(0, 0.5, (d, d)) for _ in range(3))
print(X.shape)

# %% [markdown]
# Exercise 2 starter.

# %%
import numpy as np
rng = np.random.default_rng(0)
q = rng.normal(size=(20000, 16))
print(q.shape)

# %% [markdown]
# Exercise 3 starter.

# %%
d, h = 12, 3
print(d, h, d // h)

# %% [markdown]
# Exercise 4 starter.

# %%
n, d = 14, 24
print(n, d)

# %% [markdown]
# Exercise 5 starter.

# %%
def rope(vec, pos):
    ...
print("offset check", 3)

# %% [markdown]
# Exercise 6 starter.

# %%
import numpy as np
rng = np.random.default_rng(2)
v = rng.normal(-1, 3, 6)
print(v.shape)

# %% [markdown]
# Exercise 7 starter.

# %%
import numpy as np
rng = np.random.default_rng(3)
scores = rng.normal(0, 1, (4, 4))
print(scores.shape)

# %% [markdown]
# Exercise 8 starter.

# %%
for n in (1024, 2048, 4096, 8192, 16384):
    print(n, n * n)

# %% [markdown]
# Exercise 9 starter.

# %%
print("google-bert/bert-base-uncased")
print("distilbert/distilgpt2")

# %% [markdown]
# Exercise 10 starter.

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
from datasets import load_dataset
ds = load_dataset("Trelis/tiny-shakespeare")["train"]
print(ds[0]["Text"][:12])
