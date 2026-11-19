# %% [markdown]
# # Section 23.6 check: chain of thought and self-consistency on GSM8K
#
# Runs Example 23-13 of Section 23.6: the first 50 problems of the GSM8K test split (openai/gsm8k,
# Cobbe et al., 2021) answered by Qwen/Qwen2.5-0.5B-Instruct on the CPU in three ways, a direct answer, one
# greedy chain of thought, and a majority vote over five sampled chains. It generates 350 replies, 300 of them
# chains of thought of up to 320 tokens, so it runs here rather than in the chapter walkthrough.

# %% [markdown]
# Example 23-13. A direct answer, one chain of thought, and a five-chain vote on 50 GSM8K problems.

# %%
import re, time
from collections import Counter
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers.utils import logging
logging.set_verbosity_error()                     # keep library notices out of the printed output
torch.set_num_threads(2)                          # the thread count of the book's builds

gsm = load_dataset("openai/gsm8k", "main", split="test")       # Cobbe et al. (2021), MIT license
gname = "Qwen/Qwen2.5-0.5B-Instruct"
gtok = AutoTokenizer.from_pretrained(gname)
# float32 on purpose: transformers 5.22.1 loads it as bfloat16; the chapter's models run in float32
glm = AutoModelForCausalLM.from_pretrained(gname, dtype=torch.float32).eval()
eos = glm.generation_config.eos_token_id
END = set(eos if isinstance(eos, list) else [eos])  # the token ids that end a reply

DIRECT = "\nReply with only the final number."
CHAIN = "\nThink step by step, then end with 'The answer is N.'"

def last_number(text):
    """The last number in a reply, commas and dollar signs removed, as a float (None if there is none)."""
    found = re.findall(r"-?\d[\d,]*\.?\d*", text.replace("$", ""))
    return float(found[-1].replace(",", "").rstrip(".")) if found else None

def show(x):
    return "none" if x is None else (str(int(x)) if x == int(x) else str(x))

def replies(question, suffix, max_new, sample, k=1):
    """k replies to one user turn, and how many of them ran to max_new tokens without an end token."""
    chat = gtok.apply_chat_template([{"role": "user", "content": question + suffix}],
                                    tokenize=False, add_generation_prompt=True)
    enc = gtok(chat, return_tensors="pt")
    # every decoding setting is named, so the model's generation_config.json fills none of them;
    # 1.1 is the repetition penalty that file ships (the penalty of Section 23.3), kept in all three runs
    if sample:
        settings = dict(do_sample=True, temperature=0.7, top_p=0.9, top_k=None)
    else:
        settings = dict(do_sample=False, temperature=None, top_p=None, top_k=None)
    with torch.no_grad():
        out = glm.generate(**enc, max_new_tokens=max_new, num_return_sequences=k,
                           repetition_penalty=1.1, pad_token_id=gtok.eos_token_id, **settings)
    new = out[:, enc.input_ids.shape[1]:]
    hit_limit = sum(not END & set(row.tolist()) for row in new)
    return [gtok.decode(row, skip_special_tokens=True) for row in new], hit_limit

N, K = 50, 5                                      # the first 50 test problems; five sampled chains each
right, seconds = Counter(), Counter()
cut_chain = cut_votes = vote_fixed = vote_broke = 0
for i in range(N):
    q = gsm[i]["question"]
    gold = float(gsm[i]["answer"].split("####")[-1].strip().replace(",", ""))
    t = time.time()
    direct, _ = replies(q, DIRECT, 16, sample=False)
    seconds["direct"] += time.time() - t
    t = time.time()
    chain, cut = replies(q, CHAIN, 320, sample=False)
    seconds["chain"] += time.time() - t
    cut_chain += cut
    t = time.time()
    set_seed(i)                                   # problem i draws the same five chains on every run
    chains, cut = replies(q, CHAIN, 320, sample=True, k=K)
    seconds["vote"] += time.time() - t
    cut_votes += cut
    answers = [last_number(x) for x in chains]
    tally = Counter(a for a in answers if a is not None)
    # most_common keeps first-seen order among equal counts, so a tie goes to the earliest chain
    vote = tally.most_common(1)[0][0] if tally else None
    d, c = last_number(direct[0]), last_number(chain[0])
    right["direct"] += d == gold
    right["chain"] += c == gold
    right["vote"] += vote == gold
    vote_fixed += vote == gold and c != gold
    vote_broke += c == gold and vote != gold
    if i < 10:
        print(f"{i}: gold {show(gold)} | direct {show(d)} | chain {show(c)} | "
              f"five chains {', '.join(show(a) for a in answers)} -> vote {show(vote)}")

print(f"\nfirst {N} GSM8K test problems, {gname}")
print(f"  direct answer   : {right['direct']:2d}/{N} correct")
print(f"  one greedy chain: {right['chain']:2d}/{N} correct, {cut_chain} of {N} stopped at 320 tokens")
print(f"  vote of {K} chains: {right['vote']:2d}/{N} correct, {cut_votes} of {N * K} stopped at 320 tokens")
print(f"  vote right where the greedy chain was wrong: {vote_fixed}; the reverse: {vote_broke}")
print(f"seconds in this run: direct {seconds['direct']:.0f}, greedy chain {seconds['chain']:.0f}, "
      f"five chains {seconds['vote']:.0f}")
