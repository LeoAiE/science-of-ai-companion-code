"""Section 15.5.6: natural language inference and zero-shot classification by entailment.
Run from this folder. facebook/bart-large-mnli has about 400 million parameters, so one run takes
15 to 20 minutes on the reference CPU and lives here rather than in the walkthrough.
"""
# %% [markdown]
# ## NLI on three English XNLI test pairs, then accuracy on 500

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
import os, warnings; warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import numpy as np, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, logging as hf_logging
hf_logging.set_verbosity_error(); hf_logging.disable_progress_bar()
torch.set_num_threads(2)                              # the last digits depend on the thread count
nli_id = "facebook/bart-large-mnli"                   # BART large fine-tuned on MultiNLI only
ntok = AutoTokenizer.from_pretrained(nli_id)
nli = AutoModelForSequenceClassification.from_pretrained(nli_id).eval()
print(nli.config.id2label, f"{sum(p.numel() for p in nli.parameters()) / 1e6:.0f}M parameters")
xnli = load_dataset("facebook/xnli", "en", split="test")   # its labels: 0 entailment, 1 neutral, 2 contradiction
names = xnli.features["label"].names
to_xnli = {nli.config.label2id["entailment"]: 0, nli.config.label2id["neutral"]: 1,
           nli.config.label2id["contradiction"]: 2}       # the two label orders differ
def nli_probs(premises, hypotheses):
    enc = ntok(premises, hypotheses, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        return torch.softmax(nli(**enc).logits, -1).numpy()
for lab in range(3):
    r = xnli[[i for i, y in enumerate(xnli["label"][:30]) if y == lab][0]]
    pr = nli_probs([r["premise"]], [r["hypothesis"]])[0]
    print(f"gold {names[lab]:13} predicted {names[to_xnli[int(pr.argmax())]]:13} p = {pr.max():.3f}")
    print(f"   P: {r['premise']}\n   H: {r['hypothesis']}")
sub = xnli.select(range(500)); pred = []
for i in range(0, 500, 16):
    b = sub[i:i + 16]; pred += [to_xnli[int(k)] for k in nli_probs(b["premise"], b["hypothesis"]).argmax(1)]
print(f"accuracy on the first 500 English XNLI test pairs: {np.mean(np.array(pred) == np.array(sub['label'])):.3f}")

# %% [markdown]
# ## Zero-shot sentiment by entailment, from scratch and through the pipeline

# %%
rt = load_dataset("cornell-movie-review-data/rotten_tomatoes")["test"].shuffle(seed=0).select(range(400))
reviews = list(rt["text"])                            # a plain list: the pipeline then returns one result per text
labels, template = ["negative", "positive"], "This example is {}."   # label 0 negative, 1 positive
ent = nli.config.label2id["entailment"]
def zero_shot(texts, labels, template, bs=8):
    out = []
    for i in range(0, len(texts), bs):
        chunk = texts[i:i + bs]
        prem = [t for t in chunk for _ in labels]     # each text is the premise once per label
        hyp = [template.format(l) for _ in chunk for l in labels]
        enc = ntok(prem, hyp, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            logit = nli(**enc).logits[:, ent].reshape(len(chunk), len(labels))   # entailment logits
        out.append(torch.softmax(logit, -1).numpy())  # softmax across the candidate labels
    return np.vstack(out)
P = zero_shot(reviews, labels, template)
acc = np.mean(P.argmax(1) == np.array(rt["label"]))
print(f"zero-shot accuracy on the 400 Rotten Tomatoes test sentences: {acc:.3f}  ({len(labels)} passes each)")
zs = pipeline("zero-shot-classification", model=nli, tokenizer=ntok, device="cpu")
pipe = [labels.index(r["labels"][0]) for r in zs(reviews, candidate_labels=labels,
                                                  hypothesis_template=template, batch_size=8)]
print("pipeline agrees on", int(np.sum(np.array(pipe) == P.argmax(1))), "of 400")

# %% [markdown]
# ## Four topics with no training labels

# %%
ag = load_dataset("fancyzhx/ag_news", split="test").shuffle(seed=0).select(range(400))
topics = ["world", "sports", "business", "science and technology"]   # ag_news labels 0 to 3, in order
Q = zero_shot(list(ag["text"]), topics, "This text is about {}.")
print(f"zero-shot accuracy on 400 AG News test articles, 4 topics: {np.mean(Q.argmax(1) == np.array(ag['label'])):.3f}")
conf = np.zeros((4, 4), int)
for g, p in zip(ag["label"], Q.argmax(1)): conf[g, p] += 1
print("rows gold, columns predicted:\n", conf)
