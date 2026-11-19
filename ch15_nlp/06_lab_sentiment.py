# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
"""Full solution of the explanatory lab in Section 15.6.
Run from this folder. OMP_NUM_THREADS=2.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import warnings; warnings.filterwarnings("ignore")
import os; os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import numpy as np, torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
                          pipeline)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# %%
# Real data: rotten_tomatoes movie reviews, label 1 = positive, 0 = negative.
ds = load_dataset("cornell-movie-review-data/rotten_tomatoes")
tr = ds["train"].shuffle(seed=0).select(range(800))   # small slices keep the lab fast
te = ds["test"].shuffle(seed=0).select(range(400))
Xtr, ytr = tr["text"], np.array(tr["label"])
Xte, yte = te["text"], np.array(te["label"])

# %%
# Rung 1: TF-IDF counts, no English stop list. That list deletes not, no, never, nothing, but, very, too.
m = make_pipeline(TfidfVectorizer(max_features=5000),
                  LogisticRegression(max_iter=1000)).fit(Xtr, ytr)
acc_tfidf = (m.predict(Xte) == yte).mean()

# %%
# Rung 2: frozen DistilBERT contextual embeddings, mean-pooled -> logistic regression.
tok = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
enc_model = AutoModel.from_pretrained("distilbert/distilbert-base-uncased").eval()

# %%
def embed(texts, bs=32):
    vecs = []
    for k in range(0, len(texts), bs):
        enc = tok(list(texts[k:k+bs]), return_tensors="pt",
                  padding=True, truncation=True, max_length=64)
        with torch.no_grad():
            h = enc_model(**enc).last_hidden_state         # (batch, tokens, 768)
        mask = enc["attention_mask"].unsqueeze(-1)
        vecs.append(((h * mask).sum(1) / mask.sum(1)).numpy())   # average the real tokens
    return np.vstack(vecs)

# %%
lr = LogisticRegression(max_iter=1000).fit(embed(Xtr), ytr)
acc_emb = (lr.predict(embed(Xte)) == yte).mean()

# %%
# Rung 3: fine-tune distilbert-base-uncased on these 800 reviews. Two epochs, AdamW, lr 2e-5.
torch.manual_seed(0)                                  # seed first: from_pretrained draws the new head's weights
ft = AutoModelForSequenceClassification.from_pretrained(
    "distilbert/distilbert-base-uncased", num_labels=2)
torch.manual_seed(0)
opt = torch.optim.AdamW(ft.parameters(), lr=2e-5)
ft.train()
Xtr_l = list(Xtr)
for epoch in range(2):
    order = np.random.default_rng(epoch).permutation(len(Xtr_l))
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
acc_ft = correct / len(yte)

# %%
# The off-the-shelf SST-2 checkpoint is a warning, not a rung: it saw much of this test set.
def _norm(s):
    return "".join(ch for ch in s.lower() if ch.isalnum())
sst_rows = load_dataset("stanfordnlp/sst2", split="train")
seen = {_norm(s) for s in sst_rows["sentence"]}
mask = np.array([_norm(s) in seen for s in Xte])
sst = pipeline("sentiment-analysis",
               model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
               truncation=True, device="cpu")
sst_pred = np.array([1 if r["label"] == "POSITIVE" else 0
                     for r in sst(list(Xte), batch_size=32)])

# %%
print(f"1. TF-IDF + logistic regression         : {acc_tfidf:.3f}")
print(f"2. DistilBERT embeddings + logistic reg : {acc_emb:.3f}")
print(f"3. fine-tuned on these 800 reviews      : {acc_ft:.3f}")
print(f"SST-2 checkpoint, all / overlap / clean : "
      f"{(sst_pred == yte).mean():.3f} / {(sst_pred[mask] == yte[mask]).mean():.3f} / "
      f"{(sst_pred[~mask] == yte[~mask]).mean():.3f}  "
      f"({int(mask.sum())} overlapping of {len(yte)})")
