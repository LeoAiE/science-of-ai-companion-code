# %% [markdown]
# # Section 25.7 Lab: LoRA fine-tuning on real Rotten Tomatoes reviews
#
# The code cell is Example 25-20 of Section 25.7: LoRA on DistilGPT-2 (Hub id distilbert/distilgpt2) with the
# Rotten Tomatoes reviews of Pang and Lee (2005), scored on 200 held-out reviews before and after training.

# %%
import warnings; warnings.filterwarnings("ignore")
import random
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

tok = AutoTokenizer.from_pretrained("distilbert/distilgpt2"); tok.pad_token = tok.eos_token
base = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")

# Rank-8 LoRA on the attention projection; the base weights stay frozen.
cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["c_attn"], lora_dropout=0.0, task_type="CAUSAL_LM")
model = get_peft_model(base, cfg)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"LoRA: training {trainable:,} of {total:,} params = {100*trainable/total:.2f}%")

# Real data: Rotten Tomatoes sentiment (Pang and Lee, 2005). In datasets 5.1.0 the bare
# id "rotten_tomatoes" fails with HfUriError: a repository id must be namespace/name.
ds = load_dataset("cornell-movie-review-data/rotten_tomatoes")
train = ds["train"].shuffle(seed=SEED).select(range(400))
test = ds["test"].shuffle(seed=SEED).select(range(200))
LABEL = {0: " negative", 1: " positive"}
def prompt_of(text): return f"Review: {text}\n Sentiment:"

# Completion-only masking: -100 on prompt tokens so only the label word trains.
def encode(text, label_id):
    p_ids = tok(prompt_of(text)).input_ids
    c_ids = tok(LABEL[label_id]).input_ids
    return p_ids + c_ids, [-100] * len(p_ids) + c_ids
enc = [encode(r["text"], r["label"]) for r in train]

def pad_batch(rows):
    m = max(len(i) for i, _ in rows)
    ids = [i + [tok.pad_token_id] * (m - len(i)) for i, _ in rows]
    lab = [l + [-100] * (m - len(l)) for _, l in rows]
    att = [[1] * len(i) + [0] * (m - len(i)) for i, _ in rows]
    return torch.tensor(ids), torch.tensor(lab), torch.tensor(att)

# Classify by comparing the log-prob of " positive" vs " negative" after the prompt.
pos_ids, neg_ids = tok(LABEL[1]).input_ids, tok(LABEL[0]).input_ids
@torch.no_grad()
def completion_logprob(text, cont):
    p = tok(prompt_of(text)).input_ids
    lp = torch.log_softmax(model(torch.tensor([p + cont])).logits[0], dim=-1)
    return sum(lp[len(p) - 1 + k, t].item() for k, t in enumerate(cont))
@torch.no_grad()
def accuracy():
    model.eval()
    return sum(int((1 if completion_logprob(r["text"], pos_ids) > completion_logprob(r["text"], neg_ids)
                    else 0) == r["label"]) for r in test) / len(test)

print("\nscoring 200 held-out reviews BEFORE LoRA...")
with model.disable_adapter(): before = accuracy()
print(f"BEFORE LoRA held-out accuracy: {before:.3f}")

opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=2e-3)
order = list(range(len(enc))); model.train()
for ep in range(6):
    random.Random(SEED + ep).shuffle(order)
    losses = []
    for s in range(0, len(order), 16):
        ids, lab, att = pad_batch([enc[i] for i in order[s:s + 16]])
        out = model(input_ids=ids, attention_mask=att, labels=lab)
        opt.zero_grad(); out.loss.backward(); opt.step(); losses.append(out.loss.item())
    last = sum(losses) / len(losses)
    print(f"  epoch {ep+1}: mean loss {last:.3f}")

after = accuracy()
print(f"\nLoRA training: final epoch mean loss {last:.3f}")
print(f"AFTER LoRA held-out accuracy: {after:.3f}  (before {before:.3f}, delta {after-before:+.3f})")

print("\nAll LoRA-lab checks ran.")

