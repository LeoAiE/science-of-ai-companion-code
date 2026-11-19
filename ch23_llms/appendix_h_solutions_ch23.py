"""Appendix H solutions for Chapter 23, as runnable cells.

Generated from Solutions.md (Appendix H). Every code block of this chapter's solutions appears here
verbatim, split into one cell per exercise or drill topic, so each answer can be run and checked against
the expected outputs it states. Keep Solutions.md as the source; regenerate after any change.
"""

# %% [markdown]
# ## 23.11 coding exercises

# %%
import warnings; warnings.filterwarnings("ignore")
import os; os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import numpy as np, torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# %% [markdown]
# 1. Kaplan's parameter law, then Hoffmann et al. equation (4)

# %%
# 1. Kaplan's parameter law, then Hoffmann et al. equation (4)
N_c, alpha_N = 8.8e13, 0.076
N = np.array([1e6, 1e7, 1e8, 1e9, 1e10, 1e11])
loss = (N_c / N) ** alpha_N
slope = np.polyfit(np.log(N), np.log(loss), 1)[0]
print(f"loss {loss[0]:.3f} -> {loss[-1]:.3f}; recovered alpha = {-slope:.3f}")
E, A, B, a, b = 1.69, 406.4, 410.7, 0.34, 0.28
C = 6 * 70e9 * 1.4e12
G = ((a * A) / (b * B)) ** (1 / (a + b))
a_exp, b_exp = b / (a + b), a / (a + b)
N_star = G * (C / 6) ** a_exp
D_star = (C / 6) ** b_exp / G
trained = E + A / (70e9)**a + B / (1.4e12)**b
print(f"N* = {N_star/1e9:.1f}B, D* = {D_star/1e12:.2f}T, D*/N* = {D_star/N_star:.1f}")
print(f"L(70B, 1.4T) = {trained:.2f}")
# expected output: loss 4.016 -> 1.674; recovered alpha = 0.076
# expected output: N* = 32.5B, D* = 3.02T, D*/N* = 92.8
# expected output: L(70B, 1.4T) = 1.94

# %% [markdown]
# 2. Count LoRA parameters and QLoRA memory

# %%
# 2. Count LoRA parameters and QLoRA memory
d = k = 4096; full = d * k
for r in [8, 16, 64]:
    lora = r * (d + k)
    print(f"r={r:<3}: {lora:,} params, {full//lora}x fewer, {100*lora/full:.2f}% of full")
B = np.random.default_rng(0).standard_normal((d, 8))
Amat = np.random.default_rng(1).standard_normal((8, d))
print(f"rank(B@A) = {np.linalg.matrix_rank((B@Amat)[:40, :40])} (capped at r=8)")
for bits, nm in [(16, 'fp16'), (8, 'int8'), (4, 'int4')]:
    print(f"  7B frozen base in {nm}: {7e9*bits/8/1e9:.1f} GB")
# expected output: r=8  : 65,536 params, 256x fewer, 0.39% of full
# expected output: r=16 : 131,072 params, 128x fewer, 0.78% of full
# expected output: r=64 : 524,288 params, 32x fewer, 3.12% of full
# expected output: rank(B@A) = 8 (capped at r=8)
# expected output:   7B frozen base in fp16: 14.0 GB
# expected output:   7B frozen base in int8: 7.0 GB
# expected output:   7B frozen base in int4: 3.5 GB

# %% [markdown]
# 3. Implement DPO from preference pairs

# %%
# 3. Implement DPO from preference pairs
torch.manual_seed(0); rng = np.random.default_rng(0)
quality = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9, 0.2])   # hidden goodness of each response
ref_logits = torch.randn(6); ref_logp = F.log_softmax(ref_logits, 0)
pairs = []
for _ in range(400):
    i, j = rng.choice(6, 2, replace=False)
    pr = torch.sigmoid((quality[i] - quality[j]) * 5).item()   # prefer the better one, noisily
    pairs.append((i, j) if rng.random() < pr else (j, i))
pol = ref_logits.clone().detach().requires_grad_(True)   # start FROM the reference
opt = torch.optim.Adam([pol], lr=0.05); beta = 1.0
for _ in range(300):
    lp = F.log_softmax(pol, 0)
    loss = sum(-F.logsigmoid(beta*((lp[w]-ref_logp[w]) - (lp[l]-ref_logp[l]))) for w, l in pairs)/len(pairs)
    opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    imp = beta * (F.log_softmax(pol, 0) - ref_logp)      # DPO's implicit reward
print(f"DPO loss 0.693 -> {loss.item():.3f}")
print(f"corr(implicit reward, quality) = {np.corrcoef(imp.numpy(), quality.numpy())[0,1]:.2f}")
print(f"ref best #{int(ref_logits.argmax())} (q={quality[ref_logits.argmax()]:.1f}); "
      f"DPO best #{int(imp.argmax())} (q={quality[imp.argmax()]:.1f})")
# expected output: DPO loss 0.693 -> 0.380
# expected output: corr(implicit reward, quality) = 0.99
# expected output: ref best #0 (q=0.1); DPO best #4 (q=0.9)

# %% [markdown]
# 4. Audit tokenization cost on a real dataset

# %%
# 4. Audit tokenization cost on a real dataset
tok = AutoTokenizer.from_pretrained("gpt2")
ds = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="test")
texts = ds["text"][:500]
tc = [len(tok(t)["input_ids"]) for t in texts]
wc = [len(t.split()) for t in texts]; cc = [len(t) for t in texts]
print(f"{len(texts)} reviews: {np.mean(tc):.1f} tokens vs {np.mean(wc):.1f} words")
print(f"chars/token={sum(cc)/sum(tc):.2f}; tokens/word={sum(tc)/sum(wc):.2f}")
worst = int(np.argmax([t/max(w, 1) for t, w in zip(tc, wc)]))
print(f"most fragmented: {tc[worst]} tokens / {wc[worst]} words; fit in 512: {sum(c<=512 for c in tc)}/{len(texts)}")
# expected output: 500 reviews: 25.2 tokens vs 21.3 words
# expected output: chars/token=4.63; tokens/word=1.19
# expected output: most fragmented: 12 tokens / 6 words; fit in 512: 500/500

# %% [markdown]
# 5. Reconstruct the decoding knobs from scratch

# %%
# 5. Reconstruct the decoding knobs from scratch
lm_tok = AutoTokenizer.from_pretrained("distilgpt2")
lm = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
ids = lm_tok("The capital of France is", return_tensors="pt").input_ids
def greedy(n=10):
    cur = ids.clone()
    for _ in range(n):
        with torch.no_grad(): nxt = lm(cur).logits[0, -1].argmax()
        cur = torch.cat([cur, nxt.view(1, 1)], 1)
    return lm_tok.decode(cur[0, ids.shape[1]:])
print(f"greedy run1 == run2: {greedy() == greedy()}")     # deterministic by construction
with torch.no_grad():
    logits = lm(ids).logits[0, -1].numpy()
def softmax_T(z, T): e = np.exp((z - z.max())/T); return e/e.sum()
print(f"top-1 prob at T=0.5: {softmax_T(logits, 0.5).max():.3f}, at T=2.0: {softmax_T(logits, 2.0).max():.3f}")
p = softmax_T(logits, 1.0); order = np.argsort(p)[::-1]
nucleus = np.searchsorted(np.cumsum(p[order]), 0.9) + 1
print(f"top-p=0.9 nucleus keeps {nucleus} of {len(p)} tokens")
# expected output: greedy run1 == run2: True
# expected output: top-1 prob at T=0.5: 0.459, at T=2.0: 0.007
# expected output: top-p=0.9 nucleus keeps 844 of 50257 tokens

# %% [markdown]
# 6. Build a MoE router and a KV cache calculator

# %%
# 6. Build a MoE router and a KV cache calculator
rng = np.random.default_rng(0)        # a fresh generator, so this answer does not depend on exercise 3's draws
Ee, topk, dm, dff = 8, 2, 4096, 14336
x = rng.standard_normal(dm); Wg = rng.standard_normal((Ee, dm))/np.sqrt(dm)
def sm(z): e = np.exp(z - z.max()); return e/e.sum()
gate = sm(Wg @ x); top = np.argsort(gate)[::-1][:topk]
w = gate[top]/gate[top].sum(); per = 3*dm*dff              # SwiGLU expert: gate+up+down
print(f"routed to {top.tolist()} w={np.round(w,3).tolist()}; "
      f"capacity {Ee*per/1e9:.2f}B active {topk*per/1e9:.2f}B = {Ee//topk}x")
L, dh, ctx = 32, 128, 8192
def kv_cache_gb(kv_heads): return 2*L*kv_heads*dh*ctx*2/1e9   # 2(K,V) * ... * 2 bytes fp16
for nm, h in [("MHA(32)", 32), ("GQA(8)", 8), ("MQA(1)", 1)]:
    print(f"  {nm}: {kv_cache_gb(h):.2f} GB")
# expected output: routed to [1, 2] w=[0.612, 0.388]; capacity 1.41B active 0.35B = 4x
# expected output:   MHA(32): 4.29 GB
# expected output:   GQA(8): 1.07 GB
# expected output:   MQA(1): 0.13 GB

# %% [markdown]
# 7. Quantify chain of thought and self-consistency

# %%
# 7. Quantify chain of thought and self-consistency
for S in [2, 8]:
    print(f"S={S}: direct {0.75**S:.2f}, chain-of-thought {0.97**S:.2f}")
def sc_acc(kk, p=0.55, nd=4, trials=20000, seed=0):
    r = np.random.default_rng(seed); c = 0
    for _ in range(trials):
        v = np.where(r.random(kk) < p, 0, r.integers(1, 1+nd, kk))   # 0 = correct answer
        b = np.bincount(v, minlength=1+nd); win = np.flatnonzero(b == b.max())
        c += (win[r.integers(len(win))] == 0)                        # majority vote, random tie-break
    return c/trials
for kk in [1, 5, 11, 21]:
    print(f"  k={kk:2d}: {sc_acc(kk):.2f}")
# expected output: S=2: direct 0.56, chain-of-thought 0.94
# expected output: S=8: direct 0.10, chain-of-thought 0.78
# expected output:   k= 1: 0.55
# expected output:   k= 5: 0.78
# expected output:   k=11: 0.93
# expected output:   k=21: 0.99

# %% [markdown]
# 8. Use perplexity as a fluency detector on real sentences

# %%
# 8. Use perplexity as a fluency detector on real sentences
def ppl(text):
    tid = lm_tok(text, return_tensors="pt").input_ids
    with torch.no_grad(): return float(torch.exp(lm(tid, labels=tid).loss))
sents = [ds["text"][i] for i in range(3)]
rsh = np.random.default_rng(0); c_ppl, s_ppl = [], []
for s in sents:
    c_ppl.append(ppl(s)); ww = s.split(); rsh.shuffle(ww); s_ppl.append(ppl(" ".join(ww)))
print(f"coherent mean ppl {np.mean(c_ppl):.1f}; shuffled mean ppl {np.mean(s_ppl):.1f}")
print(f"shuffling raised perplexity on {sum(a<b for a, b in zip(c_ppl, s_ppl))}/{len(sents)}")
# expected output: coherent mean ppl 653.2; shuffled mean ppl 17627.8
# expected output: shuffling raised perplexity on 3/3

# %% [markdown]
# 9. Zero-shot sentiment by likelihood scoring (miniproject)

# %%
# 9. Zero-shot sentiment by likelihood scoring (miniproject)
zs_tok = AutoTokenizer.from_pretrained("gpt2")
zs_lm = AutoModelForCausalLM.from_pretrained("gpt2").eval()
def completion_logprob(prompt, completion):
    pid = zs_tok(prompt, return_tensors="pt").input_ids
    cid = zs_tok(completion, return_tensors="pt").input_ids
    full = torch.cat([pid, cid], 1)
    with torch.no_grad(): lp = F.log_softmax(zs_lm(full).logits[0], -1)
    return sum(lp[pos-1, full[0, pos]].item() for pos in range(pid.shape[1], full.shape[1]))
def margin(text):                                     # log p(" positive") minus log p(" negative")
    prompt = f"Review: {text}\nSentiment:"
    return completion_logprob(prompt, " positive") - completion_logprob(prompt, " negative")
# the test split lists its 533 positive reviews first, so draw 75 of each label
pos = [i for i, y in enumerate(ds["label"]) if y == 1][:75]
neg = [i for i, y in enumerate(ds["label"]) if y == 0][:75]
idx = pos + neg
m = np.array([margin(ds[i]["text"]) for i in idx])
y = np.array([ds[i]["label"] for i in idx])
cf = margin("N/A")                                    # content-free margin: the prior toward " positive"
for name, thr in [("raw", 0.0), ("calibrated", cf)]:
    pred = (m > thr).astype(int)
    print(f"{name:10s}: threshold {thr:.3f}, accuracy {(pred == y).mean():.3f} ({(pred == y).sum()}/{len(y)}), "
          f"predicted positive {pred.sum()}, negative {len(y) - pred.sum()} (chance 0.500)")
# expected output: raw       : threshold 0.000, accuracy 0.573 (86/150), predicted positive 125, negative 25 (chance 0.500)
# expected output: calibrated: threshold 0.619, accuracy 0.693 (104/150), predicted positive 43, negative 107 (chance 0.500)

# %% [markdown]
# 10. Measure gender bias in next-token probabilities

# %%
# 10. Measure gender bias in next-token probabilities
def next_token_prob(ctx, word):
    i = lm_tok(ctx, return_tensors="pt").input_ids
    with torch.no_grad(): pr = torch.softmax(lm(i).logits[0, -1], -1)
    return float(pr[lm_tok(word, add_special_tokens=False).input_ids[0]])
for prof in ["The doctor said that", "The nurse said that", "The engineer said that"]:
    he, she = next_token_prob(prof, " he"), next_token_prob(prof, " she")
    print(f"  {prof:26s} he/she = {he/she:5.1f}x")
# expected output:   The doctor said that       he/she =   3.9x
# expected output:   The nurse said that        he/she =   0.1x
# expected output:   The engineer said that     he/she =   9.9x

# %% [markdown]
# 11. Mask a next-token distribution down to a legal set

# %%
# 11. Mask a next-token distribution down to a legal set
logits = np.array([2.0, 1.0, 0.1, -1.0, 0.4])
legal = {0, 2, 4}
masked = logits.copy()
for i in range(len(masked)):
    if i not in legal:
        masked[i] = -1e9
z = masked - masked.max()
p = np.exp(z) / np.exp(z).sum()
print(np.round(p, 6).tolist())
print(round(float(p[sorted(legal)].sum()), 6))
# expected output: [0.739938, 0.0, 0.110671, 0.0, 0.149391]
# expected output: 1.0

# %% [markdown]
# ## 23.11 coding drills

# %%
import warnings; warnings.filterwarnings("ignore")
import os; os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained("gpt2")
lm_tok = AutoTokenizer.from_pretrained("distilgpt2")
lm = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
def softmax(z, T=1.0): e = np.exp((z - z.max())/T); return e/e.sum()
def ppl(text):
    tid = lm_tok(text, return_tensors="pt").input_ids
    with torch.no_grad(): return float(torch.exp(lm(tid, labels=tid).loss))
def sc_acc(k, p=0.55, nd=4, trials=20000, seed=0):
    r = np.random.default_rng(seed); c = 0
    for _ in range(trials):
        v = np.where(r.random(k) < p, 0, r.integers(1, 1+nd, k))
        b = np.bincount(v, minlength=1+nd); w = np.flatnonzero(b == b.max())
        c += (w[r.integers(len(w))] == 0)
    return c/trials

# %% [markdown]
# Scaling and Chinchilla

# %%
# Scaling and Chinchilla
print(round((8.8e13 / 1e9)**0.076, 3))                                        # expected output: 2.376
print(f"{6*7e9*1.4e12:.1e}")                                                  # expected output: 5.9e+22
print(92.8)                                                                   # expected output: 92.8
print(round(-np.polyfit(np.log([1e8, 1e10]), np.log([0.5, 0.35]), 1)[0], 3))  # expected output: 0.077

# %% [markdown]
# LoRA and memory

# %%
# LoRA and memory
print(8*(4096 + 4096))                                                        # expected output: 65536
print(round((512*512)/(8*(512 + 512)), 1))                                    # expected output: 32.0
print(70e9*0.5/1e9)                                                           # expected output: 35.0
print(70e9*2/1e9)                                                             # expected output: 140.0

# %% [markdown]
# Tokenization

# %%
# Tokenization
print(len(tok("strawberry")["input_ids"]))                                    # expected output: 3
print([t.replace("Ġ", "_") for t in tok.convert_ids_to_tokens(tok("strawberry")["input_ids"])])
# expected output: ['st', 'raw', 'berry']
s = "The cat sat on the mat."
print(round(len(s)/len(tok(s)["input_ids"]), 2))                              # expected output: 3.29
print(len(tok("The year was 1234567890.")["input_ids"]) <= 512)              # expected output: True

# %% [markdown]
# Decoding

# %%
# Decoding
zz = np.array([2.0, 1.0, 0.1, -1.0])
print(np.round(softmax(zz, 0.5), 3).tolist())                                 # expected output: [0.862, 0.117, 0.019, 0.002]
print(np.argsort(zz)[::-1][:2].tolist())                                      # expected output: [0, 1]
pp = softmax(zz); od = np.argsort(pp)[::-1]
print(int(np.searchsorted(np.cumsum(pp[od]), 0.9) + 1))                       # expected output: 3
print(int(zz.argmax()))                                                       # expected output: 0

# %% [markdown]
# MoE and KV cache

# %%
# MoE and KV cache
g4 = softmax(np.array([1.0, 3.0, 2.0, 0.5])); t2 = np.argsort(g4)[::-1][:2]
print(t2.tolist(), np.round(g4[t2]/g4[t2].sum(), 3).tolist())                 # expected output: [1, 2] [0.731, 0.269]
print(8//2)                                                                   # expected output: 4
print(round(2*32*8*128*8192*2/1e9, 2))                                        # expected output: 1.07
print(32//8)                                                                  # expected output: 4

# %% [markdown]
# Reasoning and perplexity

# %%
# Reasoning and perplexity
print(round(0.9**5, 3))                                                       # expected output: 0.59
print(round(sc_acc(3), 2))                                                    # expected output: 0.66
print(round(ppl("The cat sat on the mat and looked out the window."), 1))     # expected output: 45.0
print(ppl("The cat sat on the mat.") < ppl("mat the cat on sat the"))         # expected output: True
