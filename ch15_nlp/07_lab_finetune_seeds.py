# %% [markdown]
# # 15.6 lab: how far does the fine-tuned rung move with the seed?
#
# Rung 3 of Example 15-24 fine-tunes distilbert-base-uncased on 800 Rotten Tomatoes sentences and scores
# 400 test sentences. Its seed fixes two random choices: the initial weights of the new classification
# head, drawn inside from_pretrained, and the dropout masks drawn during training. The batch order is
# fixed by np.random.default_rng(epoch), exactly as in the lab, so it is the same for every seed.
# This script repeats the same fine-tune with seeds 0 to 4. Seed 0 reproduces the lab's printed rung 3.

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
"""Seed spread of the fine-tuned rung of the Section 15.6 lab (Example 15-24, rung 3).
Run from this folder. The thread count is pinned to 2, the notebook builder's setting.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import warnings; warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import numpy as np, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, logging as hf_logging
hf_logging.set_verbosity_error(); hf_logging.disable_progress_bar()
torch.set_num_threads(2)                         # the last digits of a CPU fine-tune can depend on it

# The lab's data: the same 800 training and 400 test sentences, shuffle seed 0.
ds = load_dataset("cornell-movie-review-data/rotten_tomatoes")
tr = ds["train"].shuffle(seed=0).select(range(800))
te = ds["test"].shuffle(seed=0).select(range(400))
Xtr, ytr = tr["text"], np.array(tr["label"])
Xte, yte = te["text"], np.array(te["label"])
tok = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

# %%
def fine_tune(seed):
    """Rung 3 of Example 15-24 with its seed as a parameter; returns the number of correct test sentences."""
    torch.manual_seed(seed)                      # before from_pretrained: fixes the new head's weights
    ft = AutoModelForSequenceClassification.from_pretrained(
        "distilbert/distilbert-base-uncased", num_labels=2)
    torch.manual_seed(seed)                      # before training: fixes the dropout masks
    opt = torch.optim.AdamW(ft.parameters(), lr=2e-5)
    ft.train()
    Xtr_l = list(Xtr)
    for epoch in range(2):
        order = np.random.default_rng(epoch).permutation(len(Xtr_l))   # same batch order for every seed
        for i in range(0, len(Xtr_l), 16):
            idx = order[i:i + 16]
            enc = tok([Xtr_l[j] for j in idx], return_tensors="pt", padding=True,
                      truncation=True, max_length=64)
            enc["labels"] = torch.tensor(ytr[idx])
            opt.zero_grad()
            ft(**enc).loss.backward()
            opt.step()
    ft.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(Xte), 16):
            enc = tok(list(Xte)[i:i + 16], return_tensors="pt", padding=True,
                      truncation=True, max_length=64)
            pred = ft(**enc).logits.argmax(-1).tolist()
            correct += sum(int(p == int(y)) for p, y in zip(pred, yte[i:i + 16]))
    return correct

# %%
counts = []
for seed in range(5):
    c = fine_tune(seed)
    counts.append(c)
    print(f"seed {seed}: {c} of {len(yte)} correct, accuracy {c / len(yte):.4f}")
accs = np.array(counts) / len(yte)
print(f"min {accs.min():.4f}  max {accs.max():.4f}  mean {accs.mean():.4f}  "
      f"spread {accs.max() - accs.min():.4f}")
