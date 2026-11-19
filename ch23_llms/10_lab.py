# %% [markdown]
# # Section 23.10 Lab: an end-to-end LLM pipeline on real reviews
#
# Regenerated from Book.md (Example 23-19); chains the chapter's pieces on the Rotten Tomatoes test split:
# token accounting (Section 23.4), the two decoding regimes (Section 23.3), zero-shot benchmark scoring
# by likelihood on a balanced subset, raw and with content-free calibration (Section 23.8), and the
# perplexity fluency check (Sections 15.5.2 and 23.8).

# %%
import warnings; warnings.filterwarnings("ignore")
import os; os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed
from datasets import load_dataset

ds = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="test")
print(f"STEP 0 - corpus: {len(ds)} Rotten Tomatoes test reviews, e.g.")
print(f"  {ds['text'][0][:66]!r}...")
labels = list(ds["label"])                               # 1 = positive, 0 = negative
changes = sum(a != b for a, b in zip(labels, labels[1:]))
print(f"  labels: {sum(labels)} positive, {labels.count(0)} negative; label changes along the split: "
      f"{changes}, the first after row {labels.index(0) - 1}")

# STEP 1 - token accounting (Section 23.4): what does this corpus cost?
tok = AutoTokenizer.from_pretrained("gpt2")
counts = [len(tok(t)["input_ids"]) for t in ds["text"]]
chars = sum(len(t) for t in ds["text"])
total = sum(counts)
# $0.15 per million input tokens is an example rate for the arithmetic, not a price list
price_per_m = 0.15
print(f"\nSTEP 1 - tokens: {total:,} tokens for {chars:,} characters "
      f"-> {chars/total:.2f} chars/token, mean {np.mean(counts):.1f} tokens/review")
print(f"  at ${price_per_m:.2f} per million input tokens, feeding the whole split through a model")
print(f"  once costs about ${total/1e6*price_per_m:.4f} in input tokens")

# STEP 2 - the two decoding regimes (Section 23.3) on a review-style prefix
# device="cpu" keeps the pipeline on the processor the book's outputs come from (Section 23.3)
gen = pipeline("text-generation", model="distilbert/distilgpt2", device="cpu")
prefix = "The movie was"
greedy = gen(prefix, max_new_tokens=10, do_sample=False,
             pad_token_id=50256, truncation=True)[0]["generated_text"]
set_seed(0)
sampled = gen(prefix, max_new_tokens=10, do_sample=True, temperature=0.7, top_p=0.9,
              pad_token_id=50256, truncation=True)[0]["generated_text"]
print(f"\nSTEP 2 - greedy (T=0)          : {greedy.strip()!r}")
print(f"         sampled (T=0.7, p=0.9) : {sampled.strip()!r}")

# STEP 3 - a real benchmark scored by likelihood (Section 23.8): zero-shot sentiment.
# Score each label as the log probability GPT-2 assigns to the completion tokens.
zs_lm = AutoModelForCausalLM.from_pretrained("gpt2").eval()
def completion_logprob(prompt, completion):
    pid = tok(prompt, return_tensors="pt").input_ids
    cid = tok(completion, return_tensors="pt").input_ids
    full = torch.cat([pid, cid], 1)
    with torch.no_grad():
        lp = torch.log_softmax(zs_lm(full).logits[0], -1)
    return sum(lp[pos - 1, full[0, pos]].item() for pos in range(pid.shape[1], full.shape[1]))

# the margin m(x) = log P(" positive" | prompt) - log P(" negative" | prompt)
def margin(text):
    prompt = f"Review: {text}\nSentiment:"
    return completion_logprob(prompt, " positive") - completion_logprob(prompt, " negative")

# STEP 0 showed the split stores all its positive reviews first, so "the first n rows" would hold
# one class. Draw a balanced subset by label instead: the first per_class reviews of each label.
per_class = 100
pos_idx = [i for i, label in enumerate(labels) if label == 1][:per_class]
neg_idx = [i for i, label in enumerate(labels) if label == 0][:per_class]
y = np.array([1] * per_class + [0] * per_class)
m = np.array([margin(ds[i]["text"]) for i in pos_idx + neg_idx])

# Content-free calibration (Zhao et al., 2021): the margin of the prompt with the review replaced
# by "N/A" measures how far the model leans toward one label before it reads any review.
lp_pos_cf = completion_logprob("Review: N/A\nSentiment:", " positive")
lp_neg_cf = completion_logprob("Review: N/A\nSentiment:", " negative")
m_cf = lp_pos_cf - lp_neg_cf
print(f"\nSTEP 3 - zero-shot sentiment by likelihood on {2 * per_class} reviews "
      f"({per_class} positive, {per_class} negative; chance 0.500)")
print(f"  content-free 'N/A': log P(' positive') {lp_pos_cf:.3f}, "
      f"log P(' negative') {lp_neg_cf:.3f}, margin {m_cf:.3f}")
for name, threshold in [("raw (m > 0)", 0.0), (f"calibrated (m > {m_cf:.3f})", m_cf)]:
    pred = (m > threshold).astype(int)     # a margin equal to the threshold counts as negative
    print(f"  {name:22s}: accuracy {np.mean(pred == y):.3f}, "
          f"{pred.sum()} of {len(y)} predicted positive")
    print(f"  {'':22s}  accuracy on positives {np.mean(pred[y == 1] == 1):.3f}, "
          f"on negatives {np.mean(pred[y == 0] == 0):.3f}")

# STEP 4 - the intrinsic check (Sections 15.5.2 and 23.8): perplexity flags fluency.
lm_tok = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
lm = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2").eval()
def perplexity(text):
    ids = lm_tok(text, return_tensors="pt").input_ids
    with torch.no_grad():
        return float(torch.exp(lm(ids, labels=ids).loss))

rng = np.random.default_rng(0)
ppl_real, ppl_shuf = [], []
for t in ds["text"][:5]:
    ppl_real.append(perplexity(t))
    w = t.split(); rng.shuffle(w)
    ppl_shuf.append(perplexity(" ".join(w)))
print(f"\nSTEP 4 - perplexity on the first 5 reviews (distilbert/distilgpt2):")
print(f"  as written : mean {np.mean(ppl_real):7.1f}")
print(f"  shuffled   : mean {np.mean(ppl_shuf):7.1f}")
print(f"  shuffling raised perplexity on {sum(a < b for a, b in zip(ppl_real, ppl_shuf))}/5 reviews")

print("\nLab pipeline complete: tokens counted, both decodings run, benchmark scored, fluency checked.")
