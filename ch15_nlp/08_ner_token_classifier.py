"""Section 15.5.4: fine-tune distilbert-base-cased as a token classifier on CoNLL-2003.
Run from this folder. One run takes tens of minutes on the reference CPU, so it lives here
rather than in the walkthrough. torch.set_num_threads(2) pins the thread count.
"""
# %% [markdown]
# ## The entity-level score of 15.5.4, repeated so this script runs on its own

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
def spans(tags):                                      # (start, end, type) for each entity in IOB2 tags
    out, start, kind = [], None, None
    for i, t in enumerate(list(tags) + ["O"]):        # a final O closes an entity that runs to the end
        if start is not None and not (t.startswith("I-") and t[2:] == kind):
            out.append((start, i, kind)); start = None
        if t.startswith("B-") or (t.startswith("I-") and start is None):   # an I- after O also opens one
            start, kind = i, t[2:]
    return out
def entity_f1(gold_seqs, pred_seqs):
    g = {(k,) + s for k, seq in enumerate(gold_seqs) for s in spans(seq)}
    p = {(k,) + s for k, seq in enumerate(pred_seqs) for s in spans(seq)}
    tp = len(g & p); prec = tp / max(len(p), 1); rec = tp / max(len(g), 1)
    return prec, rec, 2 * prec * rec / max(prec + rec, 1e-12)

# %% [markdown]
# ## Example 15-19. Fine-tuning distilbert-base-cased as a token classifier on CoNLL-2003

# %%
import os, time, warnings; warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import numpy as np, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, logging as hf_logging
hf_logging.set_verbosity_error(); hf_logging.disable_progress_bar()
torch.set_num_threads(2)                              # the last digits depend on the thread count
conll = load_dataset("eriktks/conll2003", revision="refs/convert/parquet")
names = conll["train"].features["ner_tags"].feature.names
tok = AutoTokenizer.from_pretrained("distilbert/distilbert-base-cased")   # cased: capitals mark names
def encode(rows):                                     # word labels -> one label per subword piece
    enc = tok(rows["tokens"], is_split_into_words=True, truncation=True, max_length=128,
              padding=True, return_tensors="pt")
    labels = []
    for b, tags in enumerate(rows["ner_tags"]):
        prev, lab = None, []
        for w in enc.word_ids(b):                     # None for [CLS], [SEP], and padding
            lab.append(-100 if w is None or w == prev else tags[w]); prev = w   # -100: the loss skips it
        labels.append(lab)
    enc["labels"] = torch.tensor(labels)
    return enc
demo = encode(conll["train"][:1])
print(list(zip(tok.convert_ids_to_tokens(demo["input_ids"][0]),
               [names[l] if l >= 0 else -100 for l in demo["labels"][0].tolist()])))
torch.manual_seed(0)                                  # before from_pretrained: it draws the new head
model = AutoModelForTokenClassification.from_pretrained("distilbert/distilbert-base-cased",
                                                        num_labels=len(names))
print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters; head {model.classifier}")
train = conll["train"]; lengths = np.array([len(t) for t in train["tokens"]])
rng = np.random.default_rng(0)
def batches():                                        # shuffle, sort each block of 1024 by length, cut 16s
    order = rng.permutation(len(lengths))
    blocks = [sorted(order[i:i + 1024], key=lambda j: lengths[j]) for i in range(0, len(order), 1024)]
    out = [b[i:i + 16] for b in blocks for i in range(0, len(b), 16)]
    return [out[i] for i in rng.permutation(len(out))]   # batches of similar length cut the padding
epochs = 2; steps = epochs * -(-len(lengths) // 16)
opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1 - s / steps)   # linear decay to zero
model.train(); t0 = time.time()
for epoch in range(epochs):
    total = 0.0; bs = batches()
    for idx in bs:
        enc = encode(train[[int(i) for i in idx]])
        opt.zero_grad(); loss = model(**enc).loss; loss.backward(); opt.step(); sched.step()
        total += loss.item()
    print(f"epoch {epoch + 1}: mean loss {total / len(bs):.4f}")
train_secs = time.time() - t0
model.eval(); test = conll["test"]; pred = [None] * len(test)
order = np.argsort([len(t) for t in test["tokens"]])
with torch.no_grad():
    for i in range(0, len(order), 64):
        idx = [int(j) for j in order[i:i + 64]]; rows = test[idx]
        enc = tok(rows["tokens"], is_split_into_words=True, truncation=True, max_length=128,
                  padding=True, return_tensors="pt")
        best = model(**enc).logits.argmax(-1)
        for b, k in enumerate(idx):                   # each word takes its first piece's label
            tags, prev = ["O"] * len(rows["tokens"][b]), None
            for pos, w in enumerate(enc.word_ids(b)):
                if w is not None and w != prev: tags[w] = names[int(best[b, pos])]
                prev = w
            pred[k] = tags
gold = [[names[t] for t in s] for s in test["ner_tags"]]
acc = sum(a == b for g, q in zip(gold, pred) for a, b in zip(g, q)) / sum(len(g) for g in gold)
p, r, f1 = entity_f1(gold, pred)
print(f"trained {epochs} epochs on {len(lengths)} sentences in {train_secs / 60:.1f} min")
print(f"test token accuracy {acc:.4f}; entity precision {p:.4f} recall {r:.4f} F1 {f1:.4f}")
