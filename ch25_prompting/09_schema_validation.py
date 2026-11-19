# %% [markdown]
# # Section 25.3.1 Example 25-4: structured output checked against a JSON Schema
#
# Qwen/Qwen2.5-0.5B-Instruct (float32, CPU, greedy) is asked for a JSON object with a sentiment and a quote on the
# first 50 of the lab's 200 held-out Rotten Tomatoes reviews (Pang and Lee, 2005). Each reply is parsed from its first
# brace with json.JSONDecoder().raw_decode and validated with jsonschema 4.26.0 (Draft202012Validator) against a schema
# with an enum, a maxLength, two required keys and no additional properties; failures are tallied by validator keyword
# and retried once with the validator's messages; then a constrained version writes the braces and keys itself and lets
# the model choose only the enum value (by the log probabilities of the two label tokens) and a capped quote, so every
# object validates by construction. About 130 generation calls take the run past the walkthrough's budget on the CPU at
# two threads, which is why the listing lives here. Tested with transformers 5.22.1, datasets 5.1.0, jsonschema 4.26.0
# and torch 2.15.0 (CPU).

# %% [markdown]
# Example 25-4. Structured output checked against a JSON Schema, with a retry and a constrained version.

# %%
import warnings; warnings.filterwarnings("ignore")
import json
from collections import Counter
from importlib.metadata import version
import torch
torch.set_num_threads(2)                                    # two threads, the thread count of the book's builds
import jsonschema
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# The model is loaded directly rather than through a pipeline, because the constrained version at the end
# needs its next-token log probabilities.
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.float32).eval()

# Real data: the first 50 of the lab's 200 held-out Rotten Tomatoes reviews (Section 25.7).
held_out = load_dataset("cornell-movie-review-data/rotten_tomatoes")["test"].shuffle(seed=0).select(range(200))
reviews = held_out.select(range(50))

schema = {"type": "object",
          "properties": {"sentiment": {"enum": ["positive", "negative"]},
                         "quote": {"type": "string", "maxLength": 60}},
          "required": ["sentiment", "quote"],
          "additionalProperties": False}
validator = jsonschema.Draft202012Validator(schema)
print("jsonschema", version("jsonschema"))

SYSTEM = ('Reply with only a JSON object with exactly two keys: "sentiment", either "positive" or "negative", '
          'and "quote", a phrase of at most 60 characters copied from the review.')

@torch.no_grad()
def reply(msgs, n=60):                                      # greedy decoding, as in every demo of this chapter
    enc = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), return_tensors="pt")
    out = model.generate(**enc, max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()

def first_object(text):                                     # decode one JSON value from the first brace on, or None
    i = text.find("{")
    if i < 0:
        return None
    try:
        return json.JSONDecoder().raw_decode(text[i:])[0]
    except json.JSONDecodeError:
        return None

def gold_of(r):
    return "positive" if r["label"] == 1 else "negative"

# 1. Free generation: ask for the JSON in words, then check every reply against the schema.
parsed = valid = right = 0
keywords, failures, first = Counter(), [], None
for k, r in enumerate(reviews):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": r["text"]}]
    raw = reply(msgs)
    obj = first_object(raw)
    errors = [] if obj is None else list(validator.iter_errors(obj))
    parsed += obj is not None
    if obj is not None and not errors:
        valid += 1
        right += obj["sentiment"] == gold_of(r)
        continue
    kws = ["not JSON"] if obj is None else sorted({e.validator for e in errors})
    keywords.update(kws)                                    # a failing reply counts once per keyword
    messages = ["the reply is not JSON"] if obj is None else [e.message for e in errors]
    failures.append((msgs, raw, messages, gold_of(r)))
    if first is None and isinstance(obj, dict):
        first = (k, kws[0], len(str(obj.get("quote", ""))))
print(f"free generation on 50 reviews: parsed as JSON {parsed}, valid against the schema {valid}, "
      f"valid with the right label {right}")
tally = sorted(keywords.items(), key=lambda kv: (-kv[1], kv[0]))      # ties in alphabetical order
print("failing replies by keyword:", ", ".join(f"{kw} {n}" for kw, n in tally))
if first is not None:
    print(f"first failing object: review {first[0]}, keyword {first[1]}, quote of {first[2]} characters")

# 2. One retry: show the model the validator's own messages and ask again.
repaired = repaired_right = 0
for msgs, raw, messages, gold in failures:
    retry = msgs + [{"role": "assistant", "content": raw},
                    {"role": "user", "content": "That reply fails the schema: " + "; ".join(messages)
                     + ". Reply again with only the JSON object."}]
    obj = first_object(reply(retry))
    if obj is not None and validator.is_valid(obj):
        repaired += 1
        repaired_right += obj["sentiment"] == gold
print(f"one retry with the error messages: {repaired} of {len(failures)} now valid, {repaired_right} with the right label")

# 3. Constrained: the program writes the braces, the keys and the quotation marks. The model picks the enum
#    value by comparing the log probabilities of the two legal next tokens, then writes the quote, which the
#    program cuts at the first double quote or at 60 characters, whichever comes first.
label_ids = [tok.encode(word) for word in ("positive", "negative")]
assert all(len(ids) == 1 for ids in label_ids)              # each label is one token, so one comparison picks it
pos_id, neg_id = label_ids[0][0], label_ids[1][0]
c_valid = c_right = verbatim = 0
for k, r in enumerate(reviews):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": r["text"]}]
    head = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + '{"sentiment": "'
    with torch.no_grad():
        lp = torch.log_softmax(model(tok(head, return_tensors="pt").input_ids).logits[0, -1], -1)
    label = "positive" if lp[pos_id] > lp[neg_id] else "negative"     # a tie would go to negative
    if k == 0:
        share = torch.exp(lp[pos_id]) / (torch.exp(lp[pos_id]) + torch.exp(lp[neg_id]))
        print(f"review 0: log p(positive) {lp[pos_id].item():.3f}, log p(negative) {lp[neg_id].item():.3f}, "
              f"positive's share of the two {share.item():.3f}")
    enc = tok(head + label + '", "quote": "', return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=20, do_sample=False, pad_token_id=tok.eos_token_id)
    quote = tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True).split('"')[0][:60]
    obj = {"sentiment": label, "quote": quote}
    c_valid += validator.is_valid(obj)
    c_right += label == gold_of(r)
    verbatim += bool(quote.strip()) and quote.strip().lower() in r["text"].lower()   # the schema cannot check this
print(f"constrained on the same 50: valid {c_valid} of 50, right label {c_right} of 50, "
      f"quote found word for word in its review {verbatim} of 50")
