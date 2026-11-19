# %% [markdown]
# # Section 25.6 Example 25-18: post-training weight quantization of Qwen2.5-0.5B-Instruct
#
# Every nn.Linear weight of Qwen/Qwen2.5-0.5B-Instruct except the tied output head (168 matrices, 357,826,560
# weights) is rounded to a symmetric integer grid with Example 25-14's rule and stored back in float32, under five
# settings: float32, int8 with one scale per output row, int4 per row, int4 with one scale per group of 32, and int4
# with one scale per matrix. Each setting is scored on the lab's 200 held-out Rotten Tomatoes reviews (Pang and Lee,
# 2005) by zero-shot accuracy (the logits of " positive" against " negative" after a chat prompt) and by the mean
# negative log likelihood per token of the review texts. Each setting costs 400 forward passes of the model on the
# CPU at two threads, which is why the listing lives here and not in the chapter walkthrough. Tested with
# transformers 5.22.1, datasets 5.1.0 and torch 2.15.0 (CPU).

# %% [markdown]
# Example 25-18. Post-training weight quantization of Qwen2.5-0.5B-Instruct, scored on 200 held-out reviews.

# %%
import warnings; warnings.filterwarnings("ignore")
import torch
torch.set_num_threads(2)                                    # two threads, the thread count of the book's builds
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# A fresh float32 copy of the chapter's model, because the loop below overwrites its weights.
qtok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
qlm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.float32).eval()
held_out = load_dataset("cornell-movie-review-data/rotten_tomatoes")["test"].shuffle(seed=0).select(range(200))

POS, NEG = qtok.encode(" positive"), qtok.encode(" negative")
assert len(POS) == len(NEG) == 1                            # each label is a single token
def prompt_ids(text):                                       # zero shot: the label is the next token after "The review is"
    msgs = [{"role": "user", "content": "Is this movie review positive or negative? Reply with one word.\nReview: " + text}]
    s = qtok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + "The review is"
    return qtok(s, return_tensors="pt").input_ids
prompts = [prompt_ids(r["text"]) for r in held_out]
texts = [qtok(r["text"], return_tensors="pt").input_ids for r in held_out]

@torch.no_grad()
def score():
    right = 0
    for ids, r in zip(prompts, held_out):                   # comparing logits is comparing log probabilities
        logits = qlm(ids).logits[0, -1]
        right += int((1 if logits[POS[0]] > logits[NEG[0]] else 0) == r["label"])   # a tie goes to negative
    nll = count = 0.0
    for ids in texts:                                       # mean negative log likelihood per predicted token
        loss = qlm(ids, labels=ids).loss.item()
        nll += loss * (ids.shape[1] - 1); count += ids.shape[1] - 1
    return right / len(held_out), nll / count

# Every linear layer except the output head; the head shares its matrix with the embedding (tied), which stays float32.
layers = [m for name, m in qlm.named_modules() if isinstance(m, torch.nn.Linear) and name != "lm_head"]
originals = [m.weight.data.clone() for m in layers]
n_weights = sum(w.numel() for w in originals)
print(f"linear layers quantized: {len(layers)}, weights {n_weights:,}; tied embedding and head: "
      f"{qlm.config.tie_word_embeddings}")
print(f"always answering negative would score {1 - sum(held_out['label']) / len(held_out):.3f}")

def fake_quant(W, bits, scheme):                            # Example 25-14's rule: s = max|w| / (2^(b-1) - 1)
    qmax = 2 ** (bits - 1) - 1
    if scheme == "matrix":
        s = W.abs().max() / qmax                            # one scale for the whole matrix
    elif scheme == "row":
        s = W.abs().amax(dim=1, keepdim=True) / qmax        # one scale per output row
    else:                                                   # one scale per group of 32 weights along a row
        Wg = W.reshape(W.shape[0], -1, 32)
        s = Wg.abs().amax(dim=2, keepdim=True) / qmax
        return (torch.round(Wg / s) * s).reshape(W.shape)
    return torch.round(W / s) * s                           # the integers times the scale, stored back as float32

def scales_of(W, scheme):
    return {"matrix": 1, "row": W.shape[0], "group32": W.numel() // 32}[scheme]

settings = [("float32", None, None), ("int8, one scale per row", 8, "row"), ("int4, one scale per row", 4, "row"),
            ("int4, one scale per 32", 4, "group32"), ("int4, one scale per matrix", 4, "matrix")]
print(f"{'setting':27s} {'bits/weight':>11s} {'mean |w - w_hat|':>17s} {'accuracy':>9s} {'NLL/token':>10s} {'perplexity':>11s}")
for name, bits, scheme in settings:
    for m, w in zip(layers, originals):
        m.weight.data = w.clone() if bits is None else fake_quant(w, bits, scheme)
    if bits is None:
        per_weight, err = 32.0, 0.0
    else:                                                   # integer bits plus the float32 scales, if packed
        per_weight = (bits * n_weights + 32 * sum(scales_of(w, scheme) for w in originals)) / n_weights
        err = sum((m.weight.data - w).abs().sum().item() for m, w in zip(layers, originals)) / n_weights
    acc, nll = score()
    print(f"{name:27s} {per_weight:11.3f} {err:17.6f} {acc:9.3f} {nll:10.4f} {torch.exp(torch.tensor(nll)).item():11.4g}")
for m, w in zip(layers, originals):                         # put the float32 weights back
    m.weight.data = w
