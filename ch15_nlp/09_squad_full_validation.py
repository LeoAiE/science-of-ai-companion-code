"""Section 15.5.5: the from-scratch span reader of the walkthrough, run on all 10,570 SQuAD v1.1
validation questions, to compare with the exact match and F1 on the checkpoint's model card.
Run from this folder. The run takes tens of minutes on the reference CPU, so it lives here.
torch.set_num_threads(2) pins the thread count.
"""
# %% [markdown]
# ## The reader and the SQuAD scores of 15.5.5, repeated so this script runs on its own

# %%
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
import re, string, time, warnings; warnings.filterwarnings("ignore")
import os; os.environ["TRANSFORMERS_VERBOSITY"] = "error"
from collections import Counter
import numpy as np, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, logging as hf_logging
hf_logging.set_verbosity_error(); hf_logging.disable_progress_bar()
torch.set_num_threads(2)
squad = load_dataset("rajpurkar/squad", split="validation")     # 10,570 questions on Wikipedia paragraphs
qa_id = "distilbert/distilbert-base-cased-distilled-squad"
tok = AutoTokenizer.from_pretrained(qa_id)
reader = AutoModelForQuestionAnswering.from_pretrained(qa_id).eval()
def answer(question, context, max_len=15):
    enc = tok(question, context, truncation="only_second", max_length=384, stride=128,
              return_overflowing_tokens=True, return_offsets_mapping=True, padding=True, return_tensors="pt")
    offsets = enc.pop("offset_mapping"); enc.pop("overflow_to_sample_mapping")
    with torch.no_grad():
        out = reader(**enc)
    best = (-1e9, "")
    for w in range(len(offsets)):                     # one window per 384-token slice of the passage
        ctx = [k for k, s in enumerate(enc.sequence_ids(w)) if s == 1]   # passage tokens only
        lo, hi = ctx[0], ctx[-1]
        s_log, e_log = out.start_logits[w], out.end_logits[w]
        for a in torch.topk(s_log[lo:hi + 1], 20).indices + lo:          # the 20 best starts and ends
            for b in torch.topk(e_log[lo:hi + 1], 20).indices + lo:
                if a <= b < a + max_len:
                    score = float(s_log[a] - s_log[lo:hi + 1].logsumexp(0) + e_log[b] - e_log[lo:hi + 1].logsumexp(0))
                    if score > best[0]:
                        best = (score, context[int(offsets[w][a][0]):int(offsets[w][b][1])])
    return best[1]
def normalize(s):                                     # the SQuAD v1.1 rule: lowercase, drop punctuation,
    s = "".join(ch for ch in s.lower() if ch not in set(string.punctuation))   # the articles a, an,
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())               # the, and extra spaces
def f1(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    same = sum((Counter(p) & Counter(g)).values())
    return 0.0 if same == 0 else 2 * same / (len(p) + len(g))

# %% [markdown]
# ## All 10,570 validation questions

# %%
t0 = time.time()
preds = [answer(q, c) for q, c in zip(squad["question"], squad["context"])]
golds = squad["answers"]
em = np.mean([max(normalize(p) == normalize(g) for g in a["text"]) for p, a in zip(preds, golds)])
fs = np.mean([max(f1(p, g) for g in a["text"]) for p, a in zip(preds, golds)])
print(f"{len(preds)} questions in {(time.time() - t0) / 60:.1f} min: exact match {100 * em:.2f}, F1 {100 * fs:.2f}")
windows = [len(tok(q, c, truncation="only_second", max_length=384, stride=128,
                   return_overflowing_tokens=True)["input_ids"]) for q, c in zip(squad["question"], squad["context"])]
print(f"questions whose passage needs more than one window: {sum(w > 1 for w in windows)}")
