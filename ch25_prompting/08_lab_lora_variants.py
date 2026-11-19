# %% [markdown]
# # Section 25.7 Lab variants (the Appendix H 25.7 answers)
#
# Companion script for the five experiments of the Section 25.7.1 self-check. Every run repeats Example 25-20 (LoRA on
# DistilGPT-2, Hub id distilbert/distilgpt2, seed 0, lora_alpha 16, six epochs of Adam at learning rate 2e-3 in
# batches of 16, completion-only loss masking) with exactly one setting changed: (1) the rank r, (2) the target
# modules, (3) the dataset, SST-2 (stanfordnlp/sst2) in place of Rotten Tomatoes, (4) the number of training reviews,
# and (5) merging the trained adapter into the base with merge_and_unload. The Rotten Tomatoes runs score the lab's
# 200 held-out reviews (Pang and Lee, 2005), so the rank 8 run of question (1) is Example 25-20 itself. Nine LoRA
# training runs on a CPU at two threads take about an hour, which is why they live here and not in the chapter
# walkthrough; the last cell prints the total. Tested with peft 0.21.2, transformers 5.22.1, datasets 5.1.0 and
# torch 2.15.0 (CPU).

# %%
import warnings; warnings.filterwarnings("ignore")
import random, time
import numpy as np
import torch
torch.set_num_threads(2)                                    # two threads, the thread count of the book's builds
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

SEED = 0
T_START = time.time()
tok = AutoTokenizer.from_pretrained("distilbert/distilgpt2"); tok.pad_token = tok.eos_token
LABEL = {0: " negative", 1: " positive"}
def prompt_of(text): return f"Review: {text}\n Sentiment:"
pos_ids, neg_ids = tok(LABEL[1]).input_ids, tok(LABEL[0]).input_ids
rt = load_dataset("cornell-movie-review-data/rotten_tomatoes")

def splits(dataset, n_train):
    """Training rows, the 200 scored rows, and the text field, each shuffled with seed 0 as in Example 25-20."""
    if dataset == "rotten_tomatoes":
        return rt["train"].shuffle(seed=SEED).select(range(n_train)), rt["test"].shuffle(seed=SEED).select(range(200)), "text"
    sst = load_dataset("stanfordnlp/sst2")                  # its test labels are all -1, so the validation split is scored
    return (sst["train"].shuffle(seed=SEED).select(range(n_train)),
            sst["validation"].shuffle(seed=SEED).select(range(200)), "sentence")

@torch.no_grad()
def label_logprobs(model, test, key):
    """log p(" positive") and log p(" negative") after each prompt, the scores Example 25-20 compares."""
    model.eval(); pairs = []
    for r in test:
        p = tok(prompt_of(r[key])).input_ids
        scores = []
        for cont in (pos_ids, neg_ids):
            lp = torch.log_softmax(model(torch.tensor([p + cont])).logits[0], dim=-1)
            scores.append(sum(lp[len(p) - 1 + k, t].item() for k, t in enumerate(cont)))
        pairs.append(scores)
    return pairs

def accuracy_of(pairs, test):                               # a tie goes to negative, as in Example 25-20
    return sum(int((1 if s_pos > s_neg else 0) == r["label"]) for (s_pos, s_neg), r in zip(pairs, test)) / len(test)

def run(r=8, targets=("c_attn",), n_train=400, dataset="rotten_tomatoes", before=False):
    """Example 25-20 with one setting changed. Returns the trained model and a dict of its numbers."""
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    base = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
    cfg = LoraConfig(r=r, lora_alpha=16, target_modules=list(targets), lora_dropout=0.0, task_type="CAUSAL_LM")
    model = get_peft_model(base, cfg)
    out = {"trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
           "total": sum(p.numel() for p in model.parameters()),
           "adapted": sum(1 for name, _ in model.named_modules() if name.endswith("lora_A"))}
    train, test, key = splits(dataset, n_train)
    def encode(text, label_id):                             # completion-only masking: -100 on the prompt tokens
        p_ids = tok(prompt_of(text)).input_ids; c_ids = tok(LABEL[label_id]).input_ids
        return p_ids + c_ids, [-100] * len(p_ids) + c_ids
    enc = [encode(x[key], x["label"]) for x in train]
    def pad_batch(rows):
        m = max(len(i) for i, _ in rows)
        ids = [i + [tok.pad_token_id] * (m - len(i)) for i, _ in rows]
        lab = [l + [-100] * (m - len(l)) for _, l in rows]
        att = [[1] * len(i) + [0] * (m - len(i)) for i, _ in rows]
        return torch.tensor(ids), torch.tensor(lab), torch.tensor(att)
    if before:
        with model.disable_adapter():
            out["before"] = accuracy_of(label_logprobs(model, test, key), test)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=2e-3)
    order = list(range(len(enc))); model.train(); t0 = time.time()
    for ep in range(6):
        random.Random(SEED + ep).shuffle(order)
        losses = []
        for s in range(0, len(order), 16):
            ids, lab, att = pad_batch([enc[i] for i in order[s:s + 16]])
            o = model(input_ids=ids, attention_mask=att, labels=lab)
            opt.zero_grad(); o.loss.backward(); opt.step(); losses.append(o.loss.item())
    out["train_s"] = time.time() - t0
    out["loss"] = sum(losses) / len(losses)
    out["after"] = accuracy_of(label_logprobs(model, test, key), test)
    return model, out

def report(name, out):
    before = f"before {out['before']:.3f}, " if "before" in out else ""
    print(f"{name:26s} {out['adapted']:2d} matrices, trainable {out['trainable']:,} of {out['total']:,} "
          f"= {100 * out['trainable'] / out['total']:.4f}% | {before}after {out['after']:.3f} | "
          f"final epoch loss {out['loss']:.3f} | training {out['train_s']:.0f} s")

# %% [markdown]
# ## (1) The rank r: 1, 4, 8 (Example 25-20) and 32

# %%
results = {}
for r in (1, 4, 8, 32):
    model_r, results[f"r={r}"] = run(r=r, before=(r == 8))
    report(f"r = {r}, c_attn, 400 reviews", results[f"r={r}"])
    if r == 8:
        lab_model = model_r                                 # kept for question (5)
per_rank = {r: results[f"r={r}"]["trainable"] / r for r in (1, 4, 8, 32)}
print("trainable parameters per unit of rank:", sorted(set(per_rank.values())))

# %% [markdown]
# ## (2) The target modules: c_attn (Example 25-20), all of c_attn, c_fc and c_proj, and c_fc with c_proj
#
# In a GPT-2 block the name c_proj matches two layers, the attention output projection and the MLP output projection.

# %%
for targets in (("c_attn", "c_fc", "c_proj"), ("c_fc", "c_proj")):
    _, results["+".join(targets)] = run(targets=targets)
    report("+".join(targets), results["+".join(targets)])
report("c_attn (Example 25-20)", results["r=8"])

# %% [markdown]
# ## (3) The dataset: SST-2 (stanfordnlp/sst2), 400 training sentences, 200 validation sentences

# %%
_, results["sst2"] = run(dataset="sst2", before=True)
report("SST-2, r = 8, c_attn", results["sst2"])

# %% [markdown]
# ## (4) The training-set size: 100, 400 (Example 25-20) and 800 reviews

# %%
import matplotlib
matplotlib.use("Agg")                                       # headless: write the figure to a file
import matplotlib.pyplot as plt
for n in (100, 800):
    _, results[f"n={n}"] = run(n_train=n)
    report(f"{n} training reviews", results[f"n={n}"])
sizes = [100, 400, 800]
accs = [results["n=100"]["after"], results["r=8"]["after"], results["n=800"]["after"]]
print("held-out accuracy by training size:", dict(zip(sizes, accs)))

fig, ax = plt.subplots(figsize=(5.2, 3.2), facecolor="#fcfcfb")
ax.set_facecolor("#fcfcfb")
ax.axhline(results["r=8"]["before"], color="#8a8985", linewidth=1.2, linestyle="--")
ax.text(820, results["r=8"]["before"] + 0.006, f"before LoRA {results['r=8']['before']:.3f}", color="#52514e",
        fontsize=8, ha="right", va="bottom")
ax.plot(sizes, accs, color="#2a78d6", linewidth=2, solid_capstyle="round", marker="o", markersize=8,
        markeredgecolor="#fcfcfb", markeredgewidth=2)
for x, y in zip(sizes, accs):
    ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color="#0b0b0b")
ax.set_xticks(sizes)
ax.set_xlabel("training reviews", color="#52514e")
ax.set_ylabel("held-out accuracy (200 reviews)", color="#52514e")
ax.set_ylim(0.6, 0.86)
ax.set_title("LoRA on DistilGPT-2: accuracy by training-set size", fontsize=10, color="#0b0b0b")
ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.tight_layout()
fig.savefig("08_lab_lora_variants_train_size.png", dpi=150)
print("saved 08_lab_lora_variants_train_size.png")

# %% [markdown]
# ## (5) Merging the adapter with merge_and_unload, and the scoring time before and after

# %%
_, test, key = splits("rotten_tomatoes", 400)
t0 = time.time(); unmerged = label_logprobs(lab_model, test, key); t_unmerged = time.time() - t0
merged_model = lab_model.merge_and_unload()                 # W + (alpha / r) B A folded into each c_attn weight
t0 = time.time(); merged = label_logprobs(merged_model, test, key); t_merged = time.time() - t0
plain = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")      # the base with no adapter at all
t0 = time.time(); plain_scores = label_logprobs(plain, test, key); t_plain = time.time() - t0
gap = max(abs(a - b) for u, m in zip(unmerged, merged) for a, b in zip(u, m))
print(f"unmerged adapter: accuracy {accuracy_of(unmerged, test):.3f}, scoring 200 reviews took {t_unmerged:.1f} s")
print(f"merged model    : accuracy {accuracy_of(merged, test):.3f}, scoring 200 reviews took {t_merged:.1f} s")
print(f"plain base      : accuracy {accuracy_of(plain_scores, test):.3f}, scoring 200 reviews took {t_plain:.1f} s")
print(f"parameters: merged {sum(p.numel() for p in merged_model.parameters()):,}, "
      f"plain base {sum(p.numel() for p in plain.parameters()):,}; "
      f"largest gap between merged and unmerged label log probabilities {gap:.1e}")
print(f"total time of this script: {time.time() - T_START:.0f} s")
