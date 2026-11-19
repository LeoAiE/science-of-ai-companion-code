"""Appendix H solutions for Chapter 25, as runnable cells.

Generated from Solutions.md (Appendix H). Every code block of this chapter's solutions appears here
verbatim, split into one cell per exercise or drill topic, so each answer can be run and checked against
the expected outputs it states. Keep Solutions.md as the source; regenerate after any change. The Section 25.7 lab variants that Appendix H 25.7 quotes run in
08_lab_lora_variants.py and have no cell here.
"""

# %% [markdown]
# ## 25.8 coding exercises

# %%
import warnings, re, json, copy, time; warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn
from transformers import pipeline
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# device="cpu" and dtype=torch.float32: transformers 5.22.1 would otherwise put the pipeline on an
# accelerator and load this checkpoint in bfloat16, and the printed text would change (Section 25.1)
llm = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct", device="cpu", dtype=torch.float32)
def chat(user, system="You are a helpful assistant.", n=40):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return llm(msgs, max_new_tokens=n, do_sample=False)[0]["generated_text"][-1]["content"].strip()

# %% [markdown]
# 1. Prompt anatomy: measure the specification gain

# %%
# 1. Prompt anatomy: measure the specification gain
bare = chat("Tell me about the water cycle.", n=90)
system = "You are a science teacher explaining to a 10-year-old child."
task = ("Explain the water cycle. Format: exactly 2 short bullets, each starting with '- '. "
        "Constraints: under 25 words total, no jargon.")
structured = chat(task, system, n=90)
def shape(text):                                           # word count and bullet count of one answer
    return len(text.split()), sum(l.strip().startswith("-") for l in text.splitlines())
for name, text in [("bare", bare), ("structured", structured)]:
    words, n_bul = shape(text)
    print(f"1) {name:10s} words: {words:3d}  bullets: {n_bul}")
words, n_bul = shape(structured)
print("   exactly 2 bullets:", "met" if n_bul == 2 else "missed",
      "| under 25 words:", "met" if words < 25 else "missed")
# expected output: 1) bare       words:  70  bullets: 0
# expected output: 1) structured words:  42  bullets: 6
# expected output:    exactly 2 bullets: missed | under 25 words: missed
# the structure imposed a parseable bullet shape and shortened the answer, but this 0.5B model wrote six
# bullets and 42 words where two bullets and under 25 words were asked for: it missed both exact constraints.

# %% [markdown]
# 2. Few-shot format locking on real Rotten Tomatoes reviews

# %%
# 2. Few-shot format locking on real Rotten Tomatoes reviews
from datasets import load_dataset
ds = load_dataset("cornell-movie-review-data/rotten_tomatoes")["test"]
revs = [ds[0]["text"], ds[1]["text"], ds[-1]["text"]]
labs = [ds[0]["label"], ds[1]["label"], ds[-1]["label"]]   # 1=positive, 0=negative
def few(r):
    return chat("Classify each review's sentiment. Follow the pattern exactly.\n"
                "Review: 'A masterpiece, I loved every minute.' Sentiment: positive\n"
                "Review: 'Total waste of money.' Sentiment: negative\n"
                "Review: 'The effects were stunning.' Sentiment: positive\n"
                f"Review: '{r}' Sentiment:", n=4).strip().lower()
def zero(r):
    return chat(f"Classify the sentiment of this review: '{r}'", n=20)
def is_bare(a):                                            # a single label word and nothing else
    return a.strip().lower() in ("positive", "negative")
def label_of(a):                                           # the first label word a reply names, or None
    m = re.search(r"positive|negative", a.lower())
    return m.group(0) if m else None
truth = ["positive" if l == 1 else "negative" for l in labs]
zs, fs = [zero(r) for r in revs], [few(r) for r in revs]
print("2) zero-shot reply to review 1:", repr(zs[0][:50]))
print("   bare labels: zero-shot", sum(map(is_bare, zs)), "of 3 | few-shot", sum(map(is_bare, fs)), "of 3")
print("   few-shot answers:", fs, "-> accuracy", sum(label_of(f) == t for f, t in zip(fs, truth)), "/ 3 (true", truth, ")")
print("   zero-shot accuracy by the first label word named:", sum(label_of(z) == t for z, t in zip(zs, truth)), "/ 3")
# expected output: 2) zero-shot reply to review 1: 'The sentiment of this review is positive. The revi'
# expected output:    bare labels: zero-shot 0 of 3 | few-shot 3 of 3
# expected output:    few-shot answers: ['positive', 'positive', 'positive'] -> accuracy 2 / 3 (true ['positive', 'positive', 'negative'] )
# expected output:    zero-shot accuracy by the first label word named: 3 / 3
# few-shot locks the terse label format (3 of 3 bare labels, none zero-shot), yet it misreads the one negative
# review that the zero-shot prose gets right: format reliability and label accuracy are separate things.

# %% [markdown]
# 3. Structured JSON output + a defensive parser + a tool call

# %%
# 3. Structured JSON output + a defensive parser + a tool call
def extract_json(text):                                    # decode one JSON object from the first '{'
    start = text.find("{")
    if start == -1:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text, start)
        return obj
    except json.JSONDecodeError:
        return None
raw = chat('Extract the fields name, age, city as a JSON object from: '
           '"Maria is 41 years old and lives in Lisbon." Output ONLY the JSON.',
           system="You output only valid JSON, nothing else.", n=60)
parsed = extract_json(raw)
print("3) reply starts with a brace:", raw.lstrip().startswith("{"), "| parsed:", parsed, "| city:", parsed["city"])
# expected output: 3) reply starts with a brace: False | parsed: {'name': 'Maria', 'age': 41, 'city': 'Lisbon'} | city: Lisbon
def calc(expr):                                            # whitelist + explicit '**' ban
    ok = re.fullmatch(r"[0-9.+\-*/() ]+", expr) and "**" not in expr
    return eval(expr) if ok else "error"
tool_sys = ('You can use a calculator. If a calculation is needed, reply with ONLY a JSON object '
            '{"tool": "calc", "expression": "<arithmetic only, no equals sign>"} and nothing else.')
call = extract_json(chat("What is 815 multiplied by 947?", tool_sys, n=40))
expr = call["expression"].split("=")[0].strip()            # ignore anything after '='
inline = call["expression"].split("=")[1].strip() if "=" in call["expression"] else None
print("   tool call:", call)
print("   calc(expr) =", calc(expr), "| true =", 815 * 947, "| model's inline guess:", inline,
      "(wrong)" if inline != str(815 * 947) else "(right)")
# expected output:    tool call: {'tool': 'calc', 'expression': '815 * 947 = 760245'}
# expected output:    calc(expr) = 771805 | true = 771805 | model's inline guess: 760245 (wrong)

# %% [markdown]
# 4. A prompt regression suite with four graders

# %%
# 4. A prompt regression suite with four graders
def label_grader(out, expected):  return expected in out.lower()
def json_grader(out, _=None):
    try: json.loads(out); return True
    except Exception: return False
def length_grader(out, limit):    return len(out.split()) <= limit
def grounding_grader(out, context):
    nums = re.findall(r"\d+", out)
    return len(nums) > 0 and all(n in context for n in nums)
ctx = "The tower is 324 meters tall and was built in 1889."
suites = {
    "label":     ([("positive.", "positive"), ("Sentiment: negative", "negative"),
                   ("I think it is positive", "positive"), ("neutral-ish", "negative")], label_grader),
    "json":      ([('{"a": 1}', None), ('{"a": 1', None), ("not json", None), ('{"x": [1,2]}', None)], json_grader),
    "length":    ([("one two three", 5), ("a b c d e f g", 5)], length_grader),
    "grounding": ([("It is 324 meters tall.", ctx), ("It is 500 meters tall.", ctx)], grounding_grader),
}
for name, (tests, grader) in suites.items():
    passed = sum(grader(o, arg) for o, arg in tests)
    print(f"4) {name:9s} grader: {passed}/{len(tests)}")
# expected output: 4) label     grader: 3/4
# expected output: 4) json      grader: 2/4
# expected output: 4) length    grader: 1/2
# expected output: 4) grounding grader: 1/2

# %% [markdown]
# 5. Exponential backoff + cost estimation

# %%
# 5. Exponential backoff + cost estimation
def with_retry(fn, max_tries=5, base=0.01):
    waits = []
    for attempt in range(max_tries):
        try:
            return fn(), waits
        except Exception as e:                             # a transient failure
            if attempt == max_tries - 1:
                raise RuntimeError("all retries failed") from e
            wait = round(base * (2 ** attempt), 3)
            waits.append(wait); time.sleep(wait)           # the backoff itself: wait, then try again
state = {"n": 0}
def flaky():
    state["n"] += 1
    if state["n"] < 3: raise ConnectionError("503 service busy")
    return "success"
result, waits = with_retry(flaky)
print("5) result:", result, "attempts:", state["n"], "backoffs:", waits)
# expected output: 5) result: success attempts: 3 backoffs: [0.01, 0.02]
from transformers import AutoTokenizer
gtok = AutoTokenizer.from_pretrained("gpt2")
prompt = "Summarize the French Revolution in two sentences."
n_in, n_out = len(gtok(prompt).input_ids), 60
price_in, price_out = 0.50 / 1e6, 1.50 / 1e6              # $ per token: an illustration, not a vendor price
cost = n_in * price_in + n_out * price_out
print("   input tokens:", n_in, "| illustrative cost per million calls: $%.2f" % (cost * 1e6))
# expected output:    input tokens: 10 | illustrative cost per million calls: $95.00

# %% [markdown]
# 6. LoRA from scratch: DeltaW = (alpha/r) B A

# %%
# 6. LoRA from scratch: DeltaW = (alpha/r) B A
torch.manual_seed(0)
d = 32
W_base = torch.randn(d, d) * 0.1                            # frozen weights
W_target = W_base + torch.randn(d, 2) @ torch.randn(2, d) * 0.1   # a RANK-2 target update
X = torch.randn(200, d); Y = X @ W_target.T
def fit_lora(r, alpha=16, steps=400):
    B = torch.zeros(d, r, requires_grad=True)              # B starts at zero: adapter starts as no-op
    A = (torch.randn(r, d) * 0.01).detach().requires_grad_(True)
    opt = torch.optim.Adam([B, A], lr=0.05); scale = alpha / r
    for _ in range(steps):
        loss = ((X @ (W_base + scale * (B @ A)).T - Y) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()
print(f"6) full update of the {d}x{d} layer: d*k = {d * d} params")
for r in [1, 2, 3]:
    print(f"   rank {r}: final MSE {fit_lora(r):.4f}, params r(d+k) = {r * (d + d)}")
print("   alpha/r scale for r=4, alpha=16:", 16 / 4)
# expected output: 6) full update of the 32x32 layer: d*k = 1024 params
# expected output:    rank 1: final MSE 0.1490, params r(d+k) = 64
# expected output:    rank 2: final MSE 0.0000, params r(d+k) = 128
# expected output:    rank 3: final MSE 0.0000, params r(d+k) = 192
# expected output:    alpha/r scale for r=4, alpha=16: 4.0
# rank 1 cannot hold a rank-2 update and leaves error; rank 2 recovers it exactly, and rank 3 adds parameters
# but nothing else. The scale alpha/r = 4 multiplies the learned product before it is added to the frozen weight.

# --- digits MLP shared by exercises 7-10 ---
torch.set_num_threads(2)                                    # two threads, the thread count of the book's builds
Xd, yd = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(Xd / 16.0, yd, test_size=0.3, random_state=0)
Xtr_t = torch.tensor(Xtr, dtype=torch.float32); ytr_t = torch.tensor(ytr)
Xte_t = torch.tensor(Xte, dtype=torch.float32)
torch.manual_seed(0)
net = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))
opt = torch.optim.Adam(net.parameters(), lr=0.01)
for _ in range(300):
    opt.zero_grad(); nn.functional.cross_entropy(net(Xtr_t), ytr_t).backward(); opt.step()
def acc(m):
    with torch.no_grad(): return float((m(Xte_t).argmax(1).numpy() == yte).mean())
base_acc = acc(net)

# %% [markdown]
# 7. Quantization tradeoff on load_digits

# %%
# 7. Quantization tradeoff on load_digits
def quantize(w, bits):
    qmax = 2 ** (bits - 1) - 1; scale = w.abs().max() / qmax
    return torch.round(w / scale) * scale
print("7) base accuracy:", round(base_acc, 3))              # expected output: 7) base accuracy: 0.97
for bits in [8, 4]:
    q = copy.deepcopy(net); errs = []
    for layer in [q[0], q[2]]:
        wq = quantize(layer.weight.data, bits)
        errs.append((wq - layer.weight.data).abs().mean().item()); layer.weight.data = wq
    print(f"   int{bits}: accuracy {acc(q):.3f}, recon error {np.mean(errs):.5f}")
# expected output:    int8: accuracy 0.972, recon error 0.00390
# expected output:    int4: accuracy 0.946, recon error 0.06788

# %% [markdown]
# 8. The pruning cliff on load_digits

# %%
# 8. The pruning cliff on load_digits
W1, W2 = net[0].weight.data.clone(), net[2].weight.data.clone()
def pruned(sp):                                            # zero the smallest |sp| fraction of each layer
    p = copy.deepcopy(net); masks = []
    for layer, orig in [(p[0], W1), (p[2], W2)]:
        thr = orig.abs().flatten().quantile(sp)
        m = (orig.abs() >= thr).float(); layer.weight.data = orig * m; masks.append(m)
    return p, masks
print("8) dense accuracy:", round(base_acc, 3))             # expected output: 8) dense accuracy: 0.97
for sp in [0.3, 0.6, 0.95]:
    p, _ = pruned(sp)
    print(f"   {int(sp * 100)}% zeroed -> accuracy {acc(p):.3f}")
# expected output:    30% zeroed -> accuracy 0.972
# expected output:    60% zeroed -> accuracy 0.894
# expected output:    95% zeroed -> accuracy 0.122
p, masks = pruned(0.95); o = torch.optim.Adam(p.parameters(), lr=0.01)
for _ in range(300):                                       # retrain the surviving weights only
    o.zero_grad(); nn.functional.cross_entropy(p(Xtr_t), ytr_t).backward(); o.step()
    with torch.no_grad():
        p[0].weight.mul_(masks[0]); p[2].weight.mul_(masks[1])   # keep the pruned weights at zero
print(f"   95% zeroed, then 300 Adam steps with the mask fixed -> accuracy {acc(p):.3f}")
# expected output:    95% zeroed, then 300 Adam steps with the mask fixed -> accuracy 0.509
# light pruning is free (30 percent even scores 0.972), 95 percent collapses to near chance, and retraining
# the 237 surviving weights recovers only part of the loss: with so few weights left there is little to retrain.

# %% [markdown]
# 9. Knowledge distillation on load_digits

# %%
# 9. Knowledge distillation on load_digits
T = 4.0
with torch.no_grad():
    soft = torch.softmax(net(Xtr_t) / T, dim=1)            # T=4 soft targets from the teacher
torch.manual_seed(0)
init = nn.Sequential(nn.Linear(64, 6), nn.ReLU(), nn.Linear(6, 10))   # one small student, copied twice
def train_student(kind, steps=300):
    st = copy.deepcopy(init); o = torch.optim.Adam(st.parameters(), lr=0.01)
    for _ in range(steps):
        out = st(Xtr_t)
        loss = (nn.functional.kl_div(torch.log_softmax(out / T, 1), soft, reduction="batchmean")
                if kind == "soft" else nn.functional.cross_entropy(out, ytr_t))
        o.zero_grad(); loss.backward(); o.step()
    return st
student, hard_student = train_student("soft"), train_student("hard")
tp = sum(p.numel() for p in net.parameters()); sp_ = sum(p.numel() for p in student.parameters())
print(f"9) teacher {tp} params acc {base_acc:.3f} -> student {sp_} params ({tp // sp_}x smaller):")
print(f"   soft targets (T=4) acc {acc(student):.3f} | the same student on hard labels acc {acc(hard_student):.3f}")
# expected output: 9) teacher 4810 params acc 0.970 -> student 460 params (10x smaller):
# expected output:    soft targets (T=4) acc 0.915 | the same student on hard labels acc 0.933
# the 6-unit student keeps most of the teacher's accuracy at a tenth of the size, but on this small task the
# soft targets gave it no edge: the copy trained on hard labels scored higher.

# %% [markdown]
# 10. Miniproject: a compression pipeline with a deploy recommendation

# %%
# 10. Miniproject: a compression pipeline with a deploy recommendation
q8 = copy.deepcopy(net)
for layer in [q8[0], q8[2]]: layer.weight.data = quantize(layer.weight.data, 8)
pr, _ = pruned(0.5)
lin = lambda m: [l for l in m if isinstance(l, nn.Linear)]
n_w = lambda m: sum(l.weight.numel() for l in lin(m)); n_b = lambda m: sum(l.bias.numel() for l in lin(m))
nonzero = lambda m: int(sum((l.weight != 0).sum() for l in lin(m)))
size = {"base fp32":  4 * (n_w(net) + n_b(net)),                   # 4 bytes per weight and per bias
        "int8 quant": n_w(q8) + 4 * n_b(q8) + 4 * len(lin(q8)),      # 1 byte per weight, fp32 biases, one fp32 scale per layer
        "50% pruned": 4 * (nonzero(pr) + n_b(pr)),                  # only the surviving weights are stored
        "distilled":  4 * sum(p.numel() for p in student.parameters())}
table = [("base fp32", net), ("int8 quant", q8), ("50% pruned", pr), ("distilled", student)]
print("10) compression summary (size proxy = bytes of stored numbers):")
for name, m in table:
    print(f"    {name:11s} size {size[name]:>6} B  acc {acc(m):.3f}")
# expected output:     base fp32   size  19240 B  acc 0.970
# expected output:     int8 quant  size   5040 B  acc 0.972
# expected output:     50% pruned  size   9768 B  acc 0.957
# expected output:     distilled   size   1840 B  acc 0.915
for alpha in [0.6, 0.8]:
    k = 4; ea = (1 - alpha ** (k + 1)) / (1 - alpha)
    print(f"    speculative accept {alpha}: ~{ea:.1f} tokens produced/pass (an expected count, not a measured speedup)")
# expected output:     speculative accept 0.6: ~2.3 tokens produced/pass (an expected count, not a measured speedup)
# expected output:     speculative accept 0.8: ~3.4 tokens produced/pass (an expected count, not a measured speedup)
within = sorted((size[n], n) for n, m in table if acc(m) >= acc(net) - 0.02)   # variants within 0.02 of the base
print(f"    smallest variant within 0.02 of the base accuracy: {within[0][1]} ({within[0][0]} B)")
print(f"    RECOMMENDATION: ship the quantized large model (int8 kept acc {acc(q8):.3f} at {size['int8 quant']} B against "
      f"{size['base fp32']} B here) and run the distilled student ({size['distilled']} B) as the draft for speculative decoding.")
# expected output:     smallest variant within 0.02 of the base accuracy: int8 quant (5040 B)
# expected output:     RECOMMENDATION: ship the quantized large model (int8 kept acc 0.972 at 5040 B against 19240 B here) and run the distilled student (1840 B) as the draft for speculative decoding.

# %% [markdown]
# ## 25.8 coding exercises

# %% [markdown]
# 11. and 12. Local Qwen stop string and pairwise letter. Not the provider APIs.

# %%
# 11. and 12. Local Qwen stop string and pairwise letter. Not the provider APIs.
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct', dtype=torch.float32).eval()
def inputs_for(user, system='You are a helpful assistant.'):
    msgs = [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(text, return_tensors='pt')
class StopOn(StoppingCriteria):
    def __init__(self, tokenizer, needle, start_len):
        self.tokenizer = tokenizer
        self.needle = needle
        self.start_len = start_len
    def __call__(self, input_ids, scores, **kwargs):
        text = self.tokenizer.decode(input_ids[0, self.start_len:], skip_special_tokens=True)
        return self.needle in text
def gen_colors(stop=None, n=40):
    inp = inputs_for('Name three primary colors separated by commas.')
    kw = dict(max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
    if stop is not None:
        kw['stopping_criteria'] = StoppingCriteriaList([StopOn(tok, stop, inp.input_ids.shape[1])])
    out = model.generate(**inp, **kw)
    new = out[0, inp.input_ids.shape[1]:]
    return tok.decode(new, skip_special_tokens=True), int(new.shape[0])
full, n_full = gen_colors()
cut, n_cut = gen_colors(stop=" Yellow")
print('11) unstopped:', n_full, repr(full), '| stop at " Yellow":', n_cut, repr(cut))
# expected output: 11) unstopped: 6 'Red, Yellow, Blue' | stop at " Yellow": 3 'Red, Yellow'
# the stopping rule halts the moment the decoded new text contains the needle, so the three tokens 'Red', ',' and
# ' Yellow' are produced and the last three never are.

# 12.
def chat12(user, n=6):
    inp = inputs_for(user)
    out = model.generate(**inp, max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, inp.input_ids.shape[1]:], skip_special_tokens=True).strip()
pair = chat12('Which answer is better? Reply with only A or B.\n'
              'Question: What is the capital of France?\n'
              'A: The capital of France is Berlin.\n'
              'B: Paris.', n=6)
print('12)', repr(pair), '| Paris is now B, so the letter',
      'followed the content' if pair.strip().startswith('B') else 'stayed with the position')
# expected output: 12) 'A' | Paris is now B, so the letter stayed with the position
# with the answers swapped the judge still says A, now the wrong Berlin answer: the letter followed the position,
# not the content, the position bias of a weak judge.

# 13.
import numpy as np
import torch, torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
torch.manual_seed(0)
X, y = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X / 16.0, y, test_size=0.3, random_state=0)
net13 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))
opt = torch.optim.Adam(net13.parameters(), lr=0.01)
Xtr_t = torch.tensor(Xtr, dtype=torch.float32); ytr_t = torch.tensor(ytr)
for _ in range(300):
    opt.zero_grad(); nn.functional.cross_entropy(net13(Xtr_t), ytr_t).backward(); opt.step()
row = net13[0].weight.detach().numpy()[1].astype(np.float32)   # the second row (index 1)
def quant_absmax(v, bits=4):
    qmax = 2 ** (bits - 1) - 1
    scale = np.abs(v).max() / qmax
    return np.round(v / scale) * scale
def quant_groups(v, group=4, bits=4):
    qmax = 2 ** (bits - 1) - 1
    out = np.empty_like(v)
    for i in range(0, len(v), group):
        chunk = v[i:i+group]
        scale = np.abs(chunk).max() / qmax
        out[i:i+group] = np.round(chunk / scale) * scale
    return out
gq = quant_absmax(row)
print(f'13) row 1: one scale {abs(gq-row).mean():.6f} | groups of 8 {abs(quant_groups(row, 8)-row).mean():.6f}'
      f' | groups of 16 {abs(quant_groups(row, 16)-row).mean():.6f}')
# expected output: 13) row 1: one scale 0.007099 | groups of 8 0.006313 | groups of 16 0.006569
# on this row both group sizes beat the single scale, and the smaller groups beat the larger ones, because each
# group's scale follows its own largest weight; Section 25.6 shows that this ordering is usual but not guaranteed.

# %% [markdown]
# ## 25.8 coding drills

# %%
import re, json
import numpy as np, torch

# --- Prompt anatomy and techniques ---
parts = ["role", "context", "task", "format", "constraints"]
print(parts)                                               # expected output: ['role', 'context', 'task', 'format', 'constraints']
p = ("Role: science teacher. Context: audience is a 10-year-old. Task: explain the water cycle. "
     "Format: 3 bullets. Constraint: under 40 words, no jargon.")
print(p[:11])                                              # expected output: Role: scien
fewshot = ("Review: 'Loved it.' Sentiment: positive\n"
           "Review: 'Awful.' Sentiment: negative\n"
           "Review: 'Stunning.' Sentiment: positive\n"
           "Review: 'It dragged.' Sentiment:")
print(fewshot.rstrip().endswith("Sentiment:"))             # expected output: True
print("Q: pens cost 3, buy 4, pay 20. Let's think step by step.".endswith("step by step."))  # expected output: True
logits = torch.tensor([2.0, 1.0, 0.1])
for T in [0.5, 2.0]:
    print(T, [round(float(x), 3) for x in torch.softmax(logits / T, 0)])
# expected output: 0.5 [0.864, 0.117, 0.019]
# expected output: 2.0 [0.502, 0.304, 0.194]

# --- Structured output and tools ---
def extract_json(t):                                       # one object from the first '{', as in Example 25-3
    i = t.find("{")
    try: return json.JSONDecoder().raw_decode(t, i)[0] if i != -1 else None
    except json.JSONDecodeError: return None
print(extract_json('Here you go: {"city": "Paris"} done'))  # expected output: {'city': 'Paris'}
def calc(e): return eval(e) if re.fullmatch(r"[0-9.+\-*/() ]+", e) and "**" not in e else "error"
print(calc("12*(3+4)"), calc("__import__('os')"), calc("9**9**9"))  # expected output: 84 error error
print("1234 * 5678 = 700".split("=")[0].strip())           # expected output: 1234 * 5678
tools = {"calc": calc}
call = {"tool": "calc", "expression": "6*7"}
print(tools[call["tool"]](call["expression"]))             # expected output: 42

# --- API engineering and evaluation ---
print([round(0.05 * 2 ** i, 3) for i in range(4)])         # expected output: [0.05, 0.1, 0.2, 0.4]
from transformers import AutoTokenizer
gtok = AutoTokenizer.from_pretrained("gpt2")
n_tok = len(gtok("Name three primary colors").input_ids)
print(n_tok, "tokens; per-call cost $%.8f at an illustrative $0.50 per million input tokens" % (n_tok * 0.50 / 1e6))
# expected output: 4 tokens; per-call cost $0.00000200 at an illustrative $0.50 per million input tokens
substring_grader = lambda out, exp: exp in out.lower(); print(substring_grader("It is Positive", "positive"))  # expected output: True
def json_ok(o):
    try: json.loads(o); return True
    except Exception: return False
print(json_ok('{"a":1}'), json_ok('{oops'))                # expected output: True False
print(len("a b c d".split()) <= 5)                         # expected output: True

# --- LoRA and fine-tuning ---
d = k = 4096; r = 8
print(r * (d + k), (d * k) // (r * (d + k)))               # expected output: 65536 256
alpha = 16; print(alpha / r)                               # expected output: 2.0
torch.manual_seed(0)
B = torch.randn(10, r); A = torch.randn(r, 10)             # small random factors, a 10x10 update
dW = (alpha / r) * (B @ A)
print(torch.linalg.matrix_rank(dW).item())                 # expected output: 8

# --- Compression ---
a = np.array([0.12, -0.34, 0.55, -0.9], dtype=np.float32)
s = np.abs(a).max() / (2 ** 7 - 1); aq = np.round(a / s) * s
print(round(float(np.abs(aq - a).mean()), 6))              # expected output: 0.000846
t = torch.tensor([0.1, -0.5, 0.02, 0.9, -0.03])
thr = t.abs().quantile(0.6); print([round(float(x), 2) for x in t * (t.abs() >= thr)])
# expected output: [0.0, -0.5, 0.0, 0.9, -0.0]
al, kk = 0.7, 4; print(round((1 - al ** (kk + 1)) / (1 - al), 2))   # expected output: 2.77
print([round(float(x), 3) for x in torch.softmax(torch.tensor([2.0, 1.0, 0.1]) / 2.0, 0)])
# expected output: [0.502, 0.304, 0.194]
