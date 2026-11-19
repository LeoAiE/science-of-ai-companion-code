"""Chapter 25: Prompt Engineering & Working with LLMs

Section-aligned walkthrough: every code block from this chapter of the book,
in book order, each in its own cell under its section heading so you can run
and check them one by one. Generated from Book.md; keep the book as the source.
"""

# %%
import os as _os
__file__ = _os.path.join(_os.getcwd(), "00_chapter25_walkthrough.py")

# %% [markdown]
# ## 25.1 Anatomy of a good prompt

# %% [markdown]
# the five ingredients turn a wish into a specification, and a specified output is one you can predict, parse, and build on, which is why prompt anatomy is the foundation for everything else in the chapter.

# %%
import warnings; warnings.filterwarnings("ignore")
import torch
from transformers import pipeline
# pin the CPU and 32-bit floats: left unset, transformers 5.22.1 puts the pipeline on the first
# accelerator it finds and loads this checkpoint in bfloat16, and the printed text would change
llm = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct", device="cpu", dtype=torch.float32)

def chat(system, user, max_new_tokens=130):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    out = llm(msgs, max_new_tokens=max_new_tokens, do_sample=False)[0]["generated_text"]
    return out[-1]["content"].strip()

vague = chat("You are a helpful assistant.", "Tell me about the water cycle.")
print("VAGUE PROMPT -> 'Tell me about the water cycle.'")
print(f"  {vague[:240]}...")
n_tok = len(llm.tokenizer(vague)["input_ids"])                 # the reply re-encoded as tokens
print(f"  (length: {len(vague.split())} words, {n_tok} tokens; chat caps a reply at 130 new tokens)")

system = "You are a science teacher explaining to a 10-year-old child."        # ROLE + CONTEXT
task = ("Explain the water cycle. "                                            # TASK
        "Format: exactly 3 short bullet points, each starting with '- '. "     # FORMAT
        "Constraints: under 40 words total, no technical jargon.")             # CONSTRAINTS
structured = chat(system, task)
print("\nSTRUCTURED PROMPT -> role + context + task + format + constraints")
print(structured)
bullets = [l for l in structured.splitlines() if l.strip().startswith("-")]
words = len(structured.split())
print(f"\n  -> {len(bullets)} bullets, {words} words vs the vague answer's {len(vague.split())} words.")
# check each constraint against the reply instead of asserting the outcome
print(f"  bullet count of exactly 3: {'met' if len(bullets) == 3 else 'missed'} ({len(bullets)} bullets)")
print(f"  word limit of under 40: {'met' if words < 40 else 'missed'} ({words} words)")

print("\nthe five ingredients (ROLE, CONTEXT, TASK, FORMAT, CONSTRAINTS) turn a wish into a spec.")
print("vague prompts make the model guess what you want; specifying format and constraints makes the")
print("output predictable and parseable: the foundation everything else in this chapter builds on.")

print("\nAll prompt-anatomy checks ran.")

# %% [markdown]
# ## 25.2 Core techniques: few-shot, chain of thought, and beyond

# %% [markdown]
# The progression trades more calls and complexity for more accuracy, and you climb it only as far as a task demands.

# %%
import warnings; warnings.filterwarnings("ignore")
import re
import torch
from transformers import pipeline
llm = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct",
               device="cpu", dtype=torch.float32)             # the CPU and 32-bit floats, as in Example 25-1
def chat(user, system="You are a helpful assistant.", n=80):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return llm(msgs, max_new_tokens=n, do_sample=False)[0]["generated_text"][-1]["content"].strip()

review = "The acting was wooden and the plot dragged on forever."
zero = chat(f"Classify the sentiment of this review: '{review}'")
fewshot = chat(
    "Classify each review's sentiment. Follow the pattern exactly.\n"
    "Review: 'A masterpiece, I loved every minute.' Sentiment: positive\n"
    "Review: 'Total waste of money.' Sentiment: negative\n"
    "Review: 'The effects were stunning.' Sentiment: positive\n"
    f"Review: '{review}' Sentiment:", n=6)
print("ZERO-SHOT (no examples):")
print(f"  {zero[:160]}")
print("FEW-SHOT (3 examples first):")
print(f"  {fewshot}")
print("  -> few-shot copied the terse 'Sentiment: <label>' pattern; zero-shot defaults to prose.")

problem = ("A shop sells pens at 3 dollars each. I buy 4 pens and pay with a 20-dollar bill. "
           "How much change do I get?")
direct = chat(problem + " Answer with just the number, nothing else.", n=12)
cot = chat(problem + " Let's think step by step, then give the final answer.", n=200)
print(f"\nproblem: {problem}  (correct: 20 - 4*3 = 8)")
print(f"  DIRECT ('just the number'): {direct}")
print(f"  CHAIN-OF-THOUGHT:\n    {cot[:340].replace(chr(10), ' ')}")
print(f"    last line: {cot.splitlines()[-1]}")              # where the chain states its answer
def last_number(text):                                      # the last integer written in a reply
    found = re.findall(r"\d+", text)
    return found[-1] if found else "none"
d, c = last_number(direct), last_number(cot)
print(f"  the direct answer {d} is {'right' if d == '8' else 'wrong'}; "
      f"the chain's last number, {c}, is {'right' if c == '8' else 'wrong'}")
print("  making the steps explicit raised accuracy on 50 GSM8K problems in Example 23-13 (1 right direct,")
print("  17 with one chain of thought), and it exposes WHERE a wrong answer went wrong.")

print("\nmore techniques:")
print("  self-consistency : sample N chains of thought, take the majority answer (Section 23.6 simulates 0.55 -> 0.99)")
print("  tree of thoughts : branch into multiple reasoning paths, evaluate, and expand the best")
print("  prompt chaining  : split a hard task into a pipeline of focused prompts (extract -> reason ->")
print("                     format), each simpler and more reliable than one mega-prompt")
print("  least-to-most    : decompose into ordered subproblems, solve in sequence, feed each answer forward")

print("\nAll core-technique checks ran.")

# %% [markdown]
# ## Least-to-most prompting and prompt chaining, on real data

# %% [markdown]
# The decomposition is fixed by hand here so the run is reproducible; in the full method the model proposes its own subquestions first and then solves them in order (Zhou et al., 2022).

# %%
facts = "A pack holds 4 pencils. Anna buys 3 packs and also picks up 5 loose pencils."
question = "How many pencils does Anna have in total?"
print("DIRECT, one call ('just the number'):")
print(" ", chat(facts + " " + question + " Answer with just the number, nothing else.", n=8))

subquestions = [
    "How many pencils are in Anna's 3 packs altogether?",
    "Adding the loose pencils, how many pencils does Anna have in total?",
]
print("\nLEAST-TO-MOST (fixed decomposition; each answer feeds the next step):")
notes = ""
for i, sub in enumerate(subquestions, 1):
    prompt = f"{facts}\n{notes}Answer in one short sentence: {sub}"  # earlier answers ride along
    ans = chat(prompt, n=40)
    notes += f"Known: {ans}\n"
    print(f"  step {i}: {sub}")
    print(f"    -> {ans}")

# %% [markdown]
# ## Prompt chaining, two calls

# %%
sentence = "A pack holds 4 pencils. Anna buys 3 packs and also picks up 5 loose pencils."
first = chat("List every integer in this sentence, separated by commas, and nothing else: " + sentence, n=20)
second = chat("Add these integers and reply with just the number: " + first, n=12)
direct = chat(sentence + " How many pencils does Anna have? Reply with just the number.", n=12)
print("CHAIN call 1:", repr(first))
print("CHAIN call 2:", repr(second))
print("ONE call    :", repr(direct))
print("correct total is 3*4+5 =", 3 * 4 + 5)
print("4+4+4+4+5+5 =", 4 + 4 + 4 + 4 + 5 + 5)

# %% [markdown]
# ## 25.3 System prompts, structured outputs, and tool calling

# %%
import warnings; warnings.filterwarnings("ignore")
import json, re
import torch
from transformers import pipeline
llm = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct",
               device="cpu", dtype=torch.float32)             # the CPU and 32-bit floats, as in Example 25-1
def chat(user, system="You are a helpful assistant.", n=80):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return llm(msgs, max_new_tokens=n, do_sample=False)[0]["generated_text"][-1]["content"].strip()

q = "What is 2 + 2?"
print("SAME question, different SYSTEM prompt:")
print(f"  pirate : {chat(q, 'You are a terse pirate. Answer in one short sentence with pirate slang.', 40)}")
print(f"  teacher: {chat(q, 'You are a formal mathematics tutor. State the answer precisely.', 40)}")
print("  the system prompt sets behavior that persists across the conversation, separate from the user turn.")

def extract_json(text):
    start = text.find("{")                                # the first opening brace
    if start == -1:
        return None
    try:                                                  # decode one object and stop at its end
        return json.JSONDecoder().raw_decode(text, start)[0]
    except json.JSONDecodeError:                          # no object starts there
        return None
raw = chat('Extract the fields name, age, city as a JSON object from: '
           '"John is 29 years old and lives in Paris." Output ONLY the JSON.',
           system="You output only valid JSON, nothing else.", n=60)
parsed = extract_json(raw)
print(f"\nmodel JSON: {raw[:120]}")
print(f"parsed into Python dict: {parsed}")
print(f"  -> code can now use parsed['city'] = {parsed.get('city') if parsed else '??'}: text became data.")

def calc(expr):                                           # a whitelist-guarded arithmetic evaluator
    # character whitelist PLUS an explicit ban on '**': the expression comes from the
    # model (an untrusted channel!), and 9**9**9**9 passes a character check yet hangs eval.
    if re.fullmatch(r"[0-9.+\-*/() ]+", expr) and "**" not in expr:
        return eval(expr)
    return "error"

tool_system = ('You can use a calculator. If a calculation is needed, reply with ONLY a JSON object '
               '{"tool": "calc", "expression": "<arithmetic only, no equals sign>"} and nothing else.')
question = "What is 1234 multiplied by 5678?"
step1 = chat(question, tool_system, n=40)
call = extract_json(step1)
print(f"\nuser: {question}")
print(f"  step 1, model emits tool call: {call}")
if call and call.get("tool") == "calc":
    expr = call["expression"].split("=")[0].strip()       # robustness: ignore anything after '='
    result = calc(expr)
    print(f"  step 2, WE run calc('{expr}') = {result}")
    final = chat(f"The calculator returned {result}. Now answer the user's question in one sentence.",
                 system="You are a helpful assistant.", n=40)
    print(f"  step 3, model's final answer: {final}")
    # compare the model's own numbers with the tool's result instead of asserting the outcome
    own = call["expression"].split("=")[1].strip() if "=" in call["expression"] else "none"
    print(f"  the model's own number after '=': {own}; "
          f"equal to the tool's {result}: {re.sub(r'[^0-9]', '', own) == str(result)}")
    said = re.findall(r"\d[\d,]*", final)                  # numbers the final answer writes
    print(f"  numbers in the final answer: {said}; "
          f"any equal to the tool's {result}: {any(s.replace(',', '') == str(result) for s in said)}")
print("\nthe model decides WHEN to call a tool and WITH WHAT arguments; the program executes it and")
print("returns the result. Looping this (think, call tool, observe, repeat) is exactly an agent (Ch27).")

print("\nAll structured-output / tool-calling checks ran.")

# %% [markdown]
# ## The real APIs: structured outputs and tool calling

# %% [markdown]
# Instead of asking for JSON in words and hoping, you hand the API a JSON Schema, and OpenAI's structured outputs guide (OpenAI, 2026d) states that a reply then follows that schema, with two exceptions it names and the snippet checks. Needs an API key: this calls OpenAI's hosted API, so it is shown for reference and not executed here. Set your own API key in the environment and run it to see live output.

# %%
from openai import OpenAI
import json
client = OpenAI()  # reads your OPENAI_API_KEY; run this with your own key

schema = {
    "type": "object",
    "properties": {"name": {"type": "string"},
                   "age":  {"type": "integer"},
                   "city": {"type": "string"}},
    "required": ["name", "age", "city"],
    "additionalProperties": False,
}
resp = client.chat.completions.create(
    model="gpt-5-mini",                                   # check the provider's model list for current ids
    messages=[{"role": "developer", "content": "Extract the person's fields."},
              {"role": "user", "content": "John is 29 and lives in Paris."}],
    response_format={"type": "json_schema",
                     "json_schema": {"name": "person", "schema": schema, "strict": True}},
)
choice = resp.choices[0]
if choice.message.refusal:                             # a refusal need not follow the schema
    print("refused:", choice.message.refusal)
elif choice.finish_reason != "stop":                   # "length" or "content_filter": the JSON can be partial
    print("incomplete reply, finish_reason:", choice.finish_reason)
else:                                                  # a completed reply that is not a refusal
    person = json.loads(choice.message.content)        # follows the schema

# %% [markdown]
# Example 25-4 asks Qwen/Qwen2.5-0.5B-Instruct (float32, CPU, greedy) for a JSON object on the first 50 of the lab's 200 held-out Rotten Tomatoes reviews, validates every reply against a JSON Schema with jsonschema 4.26.0, tallies the failures by keyword, retries once with the validator's messages, and then constrains the decoding so that every object validates by construction. Its 50 free replies, up to 33 retries and 50 constrained quotes are about 130 generation calls, so the listing runs in the companion script `09_schema_validation.py` rather than in this walkthrough.

# %% [markdown]
# You declare the available tools as a list of function schemas, the model replies with a structured tool call instead of prose, your program runs the named function, and you hand the result back as a tool message for the model to finish with (OpenAI, 2026a).

# %%
tools = [{
    "type": "function",
    "function": {
        "name": "calc",
        "description": "Evaluate an arithmetic expression.",
        "parameters": {"type": "object",
                       "properties": {"expression": {"type": "string"}},
                       "required": ["expression"]},
    },
}]
messages = [{"role": "user", "content": "What is 1234 multiplied by 5678?"}]
resp = client.chat.completions.create(model="gpt-5-mini", messages=messages, tools=tools)
msg = resp.choices[0].message
if msg.tool_calls:                                  # the model asked to call a tool
    call = msg.tool_calls[0]
    args = json.loads(call.function.arguments)      # a dict shaped by the parameters schema, e.g. {"expression": "1234 * 5678"}
    result = calc(args["expression"])               # your code runs the real tool (the calc from Example 25-3)
    messages.append(msg)                            # the assistant's tool-call turn
    messages.append({"role": "tool", "tool_call_id": call.id, "content": str(result)})
    final = client.chat.completions.create(model="gpt-5-mini", messages=messages, tools=tools)
    print(final.choices[0].message.content)         # answer that uses the tool result

# %% [markdown]
# Example 25-5 runs the typed tool-calling protocol locally: the chat template renders the calc schema into the system turn, the model answers with a <tool_call> block, and the program parses it, runs calc, and returns a role tool message, on the chapter's product and the first 10 GSM8K test problems, with and without a system message that asks for the calculator. Its replies run to 320 tokens for 22 questions and their follow-up rounds, so the listing runs in the companion script `10_tool_calling.py` rather than in this walkthrough.

# %% [markdown]
# ## 25.4 Using LLM APIs in code, and evaluating prompts

# %% [markdown]
# Keeping prompts under version control and running the suite on every edit makes a prompt change something you test before you ship it, the way you test a change to code.

# %%
import warnings; warnings.filterwarnings("ignore")
import time, random
from threading import Thread
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",
                                             dtype=torch.float32).eval()   # 32-bit floats on the CPU (Section 25.1)

def inputs_for(user, system="You are a helpful assistant."):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(text, return_tensors="pt")                 # robust: a BatchEncoding with input_ids + mask

def chat(user, system="You are a helpful assistant.", n=40):
    inp = inputs_for(user, system)
    out = model.generate(**inp, max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, inp.input_ids.shape[1]:], skip_special_tokens=True).strip()

streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
inp = inputs_for("Name three primary colors.")
Thread(target=model.generate, kwargs=dict(**inp, max_new_tokens=30, do_sample=False,
       streamer=streamer, pad_token_id=tok.eos_token_id)).start()
print("STREAMING (tokens as they arrive): ", end="")
for chunk in streamer:
    print(chunk, end="", flush=True)
print()

def with_retry(fn, max_tries=5):
    for attempt in range(max_tries):
        try:
            return fn()
        except Exception as e:
            wait = 0.05 * (2 ** attempt)                  # 0.05, 0.1, 0.2, 0.4 ... seconds
            print(f"  attempt {attempt+1} failed ({e}); backing off {wait:.2f}s")
            time.sleep(wait)
    raise RuntimeError("all retries failed")

calls = {"n": 0}
def flaky():                                              # fails twice, then succeeds (simulated API)
    calls["n"] += 1
    if calls["n"] < 3: raise ConnectionError("503 service busy")
    return "success"
print("\nERROR HANDLING:")
print(f"  result after backoff: {with_retry(flaky)}")

prompt = "Summarize the French Revolution in two sentences."
n_in = inputs_for(prompt).input_ids.shape[1]; n_out = 60
price_in, price_out = 0.50 / 1e6, 1.50 / 1e6              # $ per token (illustrative)
cost = n_in * price_in + n_out * price_out
print(f"\nCOST: prompt = {n_in} input tokens, ~{n_out} output tokens")
print(f"  estimated cost = ${cost*1e6:.2f} per million such calls; cache repeats to pay once.")
print("  the rates $0.50 and $1.50 per million tokens are an illustration, not a vendor price.")

SENTIMENT_PROMPT = "Reply with exactly one word, 'positive' or 'negative': "
tests = [("I absolutely loved this film!", "positive"),
         ("A boring, overlong mess.", "negative"),
         ("Best purchase I have made all year.", "positive"),
         ("It broke after one day.", "negative")]
passed = 0
print("\nPROMPT EVAL SUITE (run this on every prompt change):")
for text, expected in tests:
    out = chat(SENTIMENT_PROMPT + f"'{text}'", n=6).lower()
    ok = expected in out                                  # grader: does the label appear?
    passed += ok
    print(f"  [{'PASS' if ok else 'FAIL'}] expected {expected:8s} got '{out[:20]}'")
print(f"  score: {passed}/{len(tests)} = {passed/len(tests)*100:.0f}% accuracy")
print("graders can also check JSON validity, length limits, or GROUNDING (every claim traceable to")
print("the provided context). Versioning prompts + running this suite = prompt engineering as software.")

print("\nAll API/eval checks ran.")

# %% [markdown]
# ## Calling a real provider API

# %% [markdown]
# The OpenAI client takes a system message and a user message and returns the text on choices[0].message.content; the cell prints the finish reason beside it. Needs an API key: this calls OpenAI's hosted API, so it is shown for reference and not executed here. Set your own API key in the environment and run it to see live output.

# %%
# pip install openai ; needs OPENAI_API_KEY in your environment. Not run here.
from openai import OpenAI
client = OpenAI()                                         # reads OPENAI_API_KEY
MODEL = "gpt-5.6-terra"                                   # as of November 2026; check the provider's model list
MAX_OUT = 25000                                           # the cap counts hidden reasoning tokens and the reply
resp = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "system", "content": "You are a concise assistant."},
              {"role": "user", "content": "Name three primary colors."}],
    reasoning_effort="low",                               # little hidden reasoning for a three-word reply
    max_completion_tokens=MAX_OUT)                        # replaces the deprecated max_tokens
choice = resp.choices[0]
print(choice.finish_reason, choice.message.content)       # "length" means the cap cut the reply short

# %% [markdown]
# pass stream=True and iterate the chunks, reading each token off delta.content.

# %%
stream = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "Name three primary colors."}],
    reasoning_effort="low", max_completion_tokens=MAX_OUT, stream=True)
for chunk in stream:                                      # each chunk carries a small delta
    print(chunk.choices[0].delta.content or "", end="", flush=True)
print()

# %% [markdown]
# Anthropic's client has the same idea with a dedicated system field and typed content blocks; the cell reads the first block whose type is text, because thinking blocks can come first. Needs an API key: this calls Anthropic's hosted API, so it is shown for reference and not executed here. Set your own API key in the environment and run it to see live output.

# %%
# pip install anthropic ; needs ANTHROPIC_API_KEY. Not run here.
import anthropic
claude_client = anthropic.Anthropic()                     # reads ANTHROPIC_API_KEY; `client` stays the OpenAI one
msg = claude_client.messages.create(
    model="claude-sonnet-5-5",                            # as of November 2026; check the provider's model list
    system="You are a concise assistant.",
    messages=[{"role": "user", "content": "Name three primary colors."}],
    max_tokens=16000)                                     # thinking tokens count toward this cap too
print(next(block.text for block in msg.content if block.type == "text"))   # thinking blocks can come first

# %% [markdown]
# Real calls fail, and rate limits are the failure you meet first, so wrap the call in the same exponential-backoff retry used for the local stand-in, catching only the error classes that a retry can fix rather than a bare Exception. Needs an API key: this calls OpenAI's hosted API, so it is shown for reference and not executed here. Set your own API key in the environment and run it to see live output.

# %%
import time
from openai import RateLimitError, APIConnectionError, APITimeoutError, InternalServerError
RETRYABLE = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)  # 429, network, timeout, 500+
def call_with_backoff(fn, max_tries=5):
    for attempt in range(max_tries):
        try:
            return fn()
        except RETRYABLE as e:                            # transient: back off and retry
            if attempt == max_tries - 1:
                raise                                     # last attempt: no wait, pass the error on
            wait = 2 ** attempt                           # 1, 2, 4, 8 seconds
            print(f"attempt {attempt+1} failed ({type(e).__name__}); waiting {wait}s")
            time.sleep(wait)
answer = call_with_backoff(lambda: client.chat.completions.create(
    model=MODEL, messages=[{"role": "user", "content": "Hi"}],
    reasoning_effort="low", max_completion_tokens=MAX_OUT))

# %% [markdown]
# ## Sampling controls: temperature, top-p, and stop sequences

# %% [markdown]
# Section 15.5.1 defines temperature and top-p, works both by hand on small examples, and runs them on GPT-2; Section 23.3 covers which settings to use in production. A stop string is not one of those sampling rules; it is a cutoff, which the stop-string cell applies at a comma.

# %%
def chat(user, system="You are a helpful assistant.", n=40,
         do_sample=False, temperature=1.0, top_p=1.0):        # extended: forward the sampling knobs
    inp = inputs_for(user, system)
    kw = dict(max_new_tokens=n, do_sample=do_sample, pad_token_id=tok.eos_token_id)
    if do_sample:
        kw.update(temperature=temperature, top_p=top_p)       # only used when sampling
    out = model.generate(**inp, **kw)
    return tok.decode(out[0, inp.input_ids.shape[1]:], skip_special_tokens=True).strip()

torch.manual_seed(0)                                          # one realization; sampling still varies run-to-run in spirit
q = "Give three single words that mean happy. Reply with just the three words."
print("greedy        :", chat(q, n=20))                                     # do_sample=False
print("T=0.7         :", chat(q, n=20, do_sample=True, temperature=0.7, top_p=0.95))
print("T=1.2 (run A) :", chat(q, n=20, do_sample=True, temperature=1.2, top_p=0.95))
print("T=1.2 (run B) :", chat(q, n=20, do_sample=True, temperature=1.2, top_p=0.95))

# %% [markdown]
# ## A stop string

# %%
from transformers import StoppingCriteria, StoppingCriteriaList

class StopOn(StoppingCriteria):
    def __init__(self, tokenizer, needle, start_len):
        self.tokenizer = tokenizer
        self.needle = needle
        self.start_len = start_len
    def __call__(self, input_ids, scores, **kwargs):
        text = self.tokenizer.decode(input_ids[0, self.start_len:], skip_special_tokens=True)
        return self.needle in text

def gen_colors(stop=None, n=40):
    inp = inputs_for("Name three primary colors separated by commas.")
    kw = dict(max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
    if stop is not None:
        kw["stopping_criteria"] = StoppingCriteriaList([StopOn(tok, stop, inp.input_ids.shape[1])])
    out = model.generate(**inp, **kw)
    new = out[0, inp.input_ids.shape[1]:]
    pieces = [tok.decode([t]) for t in new.tolist()]        # each new token decoded alone, special tokens kept
    return tok.decode(new, skip_special_tokens=True), int(new.shape[0]), pieces

full, n_full, pieces = gen_colors()
cut, n_cut, _ = gen_colors(stop=",")
print("UNSTOPPED:", n_full, repr(full))
print("STOP at comma:", n_cut, repr(cut))
print("unstopped pieces:", pieces)

# %% [markdown]
# ## Evaluating beyond string match

# %% [markdown]
# In Example 25-9, Qwen grades three answers to a fixed question against a rubric that anchors the 1-to-5 scale.

# %%
import re
JUDGE_SYS = ("You grade an ANSWER to the question 'What is the capital of France?'. "
             "The correct capital is Paris. Output a single digit:\n"
             "1 = wrong or names a different city\n"
             "3 = mentions Paris but adds unrelated or incorrect extra claims\n"
             "5 = correctly says Paris and stays on topic\nReply with only the digit.")
answers = {"correct": "Paris.",
           "padded":  "Paris, which is also the largest city in Germany.",
           "wrong":   "The capital of France is Berlin."}
for label, ans in answers.items():
    raw = chat(f"ANSWER: {ans}\nScore:", system=JUDGE_SYS, n=6)
    score = re.search(r"[1-5]", raw)
    print(f"  {label:8s} -> score {score.group(0) if score else '?'}")

# %% [markdown]
# ## Pairwise preference

# %%
pair = chat(
    "Which answer is better? Reply with only A or B.\n"
    "Question: What is the capital of France?\n"
    "A: Paris.\n"
    "B: The capital of France is Berlin.",
    n=6)
print("PAIRWISE:", repr(pair.strip()))

# %% [markdown]
# Example 25-11 grades Example 25-6's prompt and a three-example version on the lab's 200 held-out reviews and runs an exact two-sided sign test on the discordant ones. It reuses chat, as Example 25-7 extended it, and SENTIMENT_PROMPT from Example 25-6.

# %% [markdown]
# Example 25-11. Two prompt versions graded on 200 held-out reviews, with a sign test on the discordant ones.

# %%
import warnings; warnings.filterwarnings("ignore")
from math import comb
import torch
torch.set_num_threads(2)                                    # two threads, the thread count of the book's builds
from datasets import load_dataset

# The offline held-out set: the lab's 200 Rotten Tomatoes reviews (Section 25.7), none of them used to write a prompt.
rt = load_dataset("cornell-movie-review-data/rotten_tomatoes")
held_out = rt["test"].shuffle(seed=0).select(range(200))

# Version A is Example 25-6's prompt. Version B puts three labeled training reviews first
# (train indices 0, 4500 and 1: the train split lists its 4,265 positive reviews first).
shots = [rt["train"][i] for i in (0, 4500, 1)]
FEW_SHOT_PROMPT = "".join(f"Review: '{s['text']}' Sentiment: {'positive' if s['label'] == 1 else 'negative'}\n"
                          for s in shots) + SENTIMENT_PROMPT
print("example labels:", [s["label"] for s in shots])

def grade(prefix):                                          # Example 25-6's grader, run on every held-out review
    passes, neither, both = [], 0, 0
    for r in held_out:
        out = chat(prefix + f"'{r['text']}'", n=6).lower()
        expected = "positive" if r["label"] == 1 else "negative"
        passes.append(expected in out)
        neither += "positive" not in out and "negative" not in out
        both += "positive" in out and "negative" in out     # a reply naming both labels passes either way
    return passes, neither, both

results = {}
for name, prefix in (("A, Example 25-6's prompt", SENTIMENT_PROMPT), ("B, three examples first", FEW_SHOT_PROMPT)):
    passes, neither, both = grade(prefix)
    results[name] = passes
    print(f"{name:26s}: pass rate {sum(passes)}/200 = {sum(passes) / 200:.3f}; "
          f"replies naming neither label {neither}, both labels {both}")

# Paired comparison: only the reviews on which exactly one version passes carry evidence.
a, b = results.values()
only_a = sum(x and not y for x, y in zip(a, b))
only_b = sum(y and not x for x, y in zip(a, b))
n = only_a + only_b
tail = sum(comb(n, k) for k in range(min(only_a, only_b) + 1))
p = min(1.0, 2 * tail / 2 ** n)
print(f"both pass {sum(x and y for x, y in zip(a, b))}, both fail {sum(not x and not y for x, y in zip(a, b))}, "
      f"only A passes {only_a}, only B passes {only_b}")
print(f"exact two-sided sign test: p = 2 x {tail:,} / 2^{n} = {p:.3f}")

# %% [markdown]
# ## 25.5 Fine-tuning in practice: LoRA, QLoRA, and the decision framework

# %% [markdown]
# ## QLoRA: a 4-bit base with 16-bit adapters

# %% [markdown]
# Gated families such as Meta's Llama require accepting a license on the Hub before the first download, which is a practical reason to prefer an ungated base for a first experiment. Needs a GPU: this uses bitsandbytes quantization, which requires an NVIDIA GPU with CUDA, so it is shown for reference and not executed; this book's code ran on CPU-only torch builds.

# %%
# illustrative: run on a GPU with bitsandbytes installed. Not executed here.
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
qcfg = BitsAndBytesConfig(load_in_4bit=True,               # store base weights in 4-bit
                          bnb_4bit_quant_type="nf4",       # 4-bit NormalFloat
                          bnb_4bit_use_double_quant=True,  # quantize the scale factors too
                          bnb_4bit_compute_dtype=torch.bfloat16)
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B", quantization_config=qcfg)  # an ungated open 7B base
base = prepare_model_for_kbit_training(base)
lora = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
model = get_peft_model(base, lora)                         # only the adapters train
model.print_trainable_parameters()

# %% [markdown]
# ## The LoRA merge identity

# %%
import torch
torch.manual_seed(0)
d, k, r, alpha = 4, 4, 2, 4
W = torch.randn(d, k)
B = torch.randn(d, r)
A = torch.randn(r, k)
x = torch.randn(k)
scale = alpha / r
merged = W + scale * (B @ A)
y_merged = merged @ x
y_two = W @ x + scale * (B @ (A @ x))
print("scale", scale)
print("max abs diff", float((y_merged - y_two).abs().max()))

def counts(d, r):
    full = d * d
    lora = r * (d + d)
    prompt = 2 * d                   # prompt tuning: a soft prompt of 2 vectors of width d at the input
    prefix = 2 * 2 * d               # prefix tuning, per layer: a key and a value of width d for 2 positions
    bottle = d * 2 + 2 * d           # one bottleneck adapter of inner width 2, weights only
    ia3 = d                          # one IA3 scaling vector
    print(f"d={d} r={r} full={full} lora={lora} prompt={prompt} prefix={prefix} bottle={bottle} ia3={ia3}")

counts(4, 2)
counts(8, 2)

# %% [markdown]
# ## Prompt, RAG, or fine-tune?

# %% [markdown]
# The rule of thumb is to prompt first, add retrieval for knowledge, and fine-tune for behavior, and to prefer retrieval over fine-tuning when the goal is to add facts (Ovadia et al., 2023; Gekhman et al., 2024).

# %%
import numpy as np
import torch
torch.manual_seed(0)

print("trainable parameters per weight matrix (LoRA rank r=8):")
print(f"  {'layer d x k':>14} {'full (d*k)':>14} {'LoRA r=8':>12} {'reduction':>10}")
for d in [1024, 4096, 8192]:
    full = d * d; lora = 8 * (d + d)
    print(f"  {f'{d} x {d}':>14} {full:>14,} {lora:>12,} {full//lora:>9}x")
print("This table is one matrix. Section 25.7 measures 0.18 percent trainable on DistilGPT-2.")

d = 64
W_base = torch.randn(d, d) * 0.1                          # frozen pretrained weights
B_true = torch.randn(d, 4); A_true = torch.randn(4, d)
W_target = W_base + B_true @ A_true * 0.1                 # the fine-tuned weights differ by a RANK-4 update
X = torch.randn(200, d)
Y = X @ W_target.T                                        # the behavior we want to match

def fit_lora(r, alpha=16, steps=400):
    B = torch.zeros(d, r, requires_grad=True)
    A = (torch.randn(r, d) * 0.01).detach().requires_grad_(True)   # make it a leaf with grad
    scale = alpha / r                                     # the LoRA scale: Delta W = (alpha/r) B A
    opt = torch.optim.Adam([B, A], lr=0.05)
    for _ in range(steps):
        pred = X @ (W_base + scale * (B @ A)).T           # frozen W plus the SCALED update
        loss = ((pred - Y) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return loss.item(), r * (d + d)

print("\nfitting a rank-4 behavior update with LoRA adapters of various rank:")
for r in [1, 2, 4, 8]:
    err, params = fit_lora(r)
    note = "(captures the update)" if err < 1e-3 else "(leaves error)"   # the label comes from the error, not from r
    print(f"  rank {r}: final MSE {err:.4f}, trainable params {params:>4}  {note}")
print(f"  full fine-tuning would train all d*d = {d*d} params; rank-4 LoRA matches it with {4*(d+d)}.")

print("\nthe alpha knob at fixed rank 4 (scale = alpha/r multiplies the update):")
for alpha in [1, 4, 16, 64]:
    err, _ = fit_lora(4, alpha=alpha, steps=40)           # a short 40-step budget exposes the effect
    print(f"  alpha {alpha:>2} (scale {alpha/4:>5.2f}): MSE after 40 steps {err:.4f}")
print("  bigger alpha = bigger effective step for the adapter, so it converges faster at a fixed budget.")

print("\nQLoRA = quantize the frozen base to 4-bit + train LoRA adapters in 16-bit bfloat16:")
print("  a 70B base in 4-bit is ~35GB (fits one 48GB GPU); only the tiny adapters are trained.")
print("\nWHEN to prompt vs RAG vs fine-tune (the decision framework):")
print("  PROMPT      : the model can already do it with guidance -> fastest, no training, iterate freely")
print("  RAG (Ch26)  : you need external / fresh / private KNOWLEDGE -> retrieve facts at query time")
print("  FINE-TUNE   : you need new BEHAVIOR/style/format/tone, a narrow skill, or shorter prompts")
print("  rule of thumb: prompt first; add RAG for knowledge; fine-tune for behavior.")
print("  to add facts, prefer RAG over fine-tuning (Ovadia et al., 2023; Gekhman et al., 2024).")

print("\nAll fine-tuning checks ran.")

# %% [markdown]
# ## 25.6 Making models smaller and faster

# %% [markdown]
# quantize the weights, optionally prune and distill, and estimate a speculative token count.

# %%
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch, torch.nn as nn
torch.set_num_threads(2)                                    # two threads, the thread count of the book's builds
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
torch.manual_seed(0); rng = np.random.default_rng(0)

W = rng.normal(0, 0.1, 100000).astype(np.float32)
def quantize(W, bits):
    qmax = 2 ** (bits - 1) - 1
    scale = np.abs(W).max() / qmax
    Wq = np.round(W / scale).astype(int)                 # the stored integers
    return Wq * scale                                    # dequantized approximation
print("quantizing weights from fp32:")
print(f"  {'precision':>10} {'bytes/wt':>9} {'memory vs fp32':>15} {'reconstruction error':>22}")
for bits, name in [(32, "fp32"), (8, "int8"), (4, "int4")]:
    if bits == 32:
        err = 0.0; bytes_ = 4
    else:
        Wd = quantize(W, bits); err = np.abs(Wd - W).mean(); bytes_ = bits / 8
    print(f"  {name:>10} {bytes_:>9} {4/bytes_:>13.0f}x {err:>22.5f}")
print("int8 stores one byte per weight, a quarter of fp32; int4 half a byte, an eighth, at a larger error.")

X, y = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X / 16.0, y, test_size=0.3, random_state=0)
Xtr_t = torch.tensor(Xtr, dtype=torch.float32); ytr_t = torch.tensor(ytr)
Xte_t = torch.tensor(Xte, dtype=torch.float32)
net = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))
opt = torch.optim.Adam(net.parameters(), lr=0.01)
for _ in range(300):
    opt.zero_grad(); nn.functional.cross_entropy(net(Xtr_t), ytr_t).backward(); opt.step()
def accuracy():
    with torch.no_grad(): return (net(Xte_t).argmax(1).numpy() == yte).mean()
base_acc = accuracy()
W1 = net[0].weight.data.clone(); W2 = net[2].weight.data.clone()    # save dense weights
print(f"\npruning (zero the smallest weights); dense accuracy = {base_acc:.3f}")
for sparsity in [0.5, 0.8, 0.9]:
    for w, orig in [(net[0].weight, W1), (net[2].weight, W2)]:
        thresh = orig.abs().flatten().quantile(sparsity)
        w.data = orig * (orig.abs() >= thresh)            # zero the smallest |sparsity| fraction
    print(f"  {int(sparsity*100)}% of weights zeroed -> accuracy {accuracy():.3f}")
net[0].weight.data = W1; net[2].weight.data = W2          # restore for distillation below

with torch.no_grad():
    teacher_logits = net(Xtr_t); soft = torch.softmax(teacher_logits / 2.0, dim=1)   # T=2 soft targets
student = nn.Sequential(nn.Linear(64, 12), nn.ReLU(), nn.Linear(12, 10))              # much smaller
so = torch.optim.Adam(student.parameters(), lr=0.01)
for _ in range(300):
    loss = nn.functional.kl_div(torch.log_softmax(student(Xtr_t) / 2.0, 1), soft, reduction="batchmean")
    so.zero_grad(); loss.backward(); so.step()
with torch.no_grad():
    student_acc = (student(Xte_t).argmax(1).numpy() == yte).mean()
t_params = sum(p.numel() for p in net.parameters()); s_params = sum(p.numel() for p in student.parameters())
print(f"\ndistillation: teacher {t_params} params (acc {base_acc:.3f}) -> student {s_params} params "
      f"(acc {student_acc:.3f})")
print(f"  the {t_params//s_params}x smaller student matches most of the teacher by mimicking its soft outputs.")

print("\nspeculative decoding (expected tokens per target pass):")
for alpha in [0.6, 0.8]:
    k = 4                                                 # draft proposes k tokens per step
    expected_tokens = (1 - alpha ** (k + 1)) / (1 - alpha)    # sum of alpha^i for i = 0..k
    print(f"  draft acceptance rate {alpha}: ~{expected_tokens:.1f} tokens produced per big-model pass "
          f"(vs 1 without it)")
print("the big model checks the draft's k tokens in ONE forward pass; accepted drafts are cheap, and")
print("the same pass yields one token of the big model's own, so the count is tokens PRODUCED per pass.")
print("those two counts are expected tokens per target pass if the draft is free, not a measured speedup.")

print("\nAll smaller-faster checks ran.")

# %% [markdown]
# ## Per-group scale

# %%
row = net[0].weight.detach().numpy()[0].astype(np.float32)

def quant_absmax(v, bits=4):
    qmax = 2 ** (bits - 1) - 1
    scale = np.abs(v).max() / qmax
    return np.round(v / scale) * scale, float(scale)

def quant_groups(v, group=4, bits=4):
    qmax = 2 ** (bits - 1) - 1
    out = np.empty_like(v)
    for i in range(0, len(v), group):
        chunk = v[i:i + group]
        scale = np.abs(chunk).max() / qmax
        out[i:i + group] = np.round(chunk / scale) * scale
    return out

global_q, global_scale = quant_absmax(row)
group_q = quant_groups(row)
print("first row of the trained digits MLP, length", len(row))
print(f"global mae {np.abs(global_q - row).mean():.6f} scale {global_scale:.6f}")
print(f"group mae  {np.abs(group_q - row).mean():.6f}")

# %% [markdown]
# ## Retraining the pruned MLP with the mask held fixed

# %%
import copy
torch.set_num_threads(2)                                    # two threads, the thread count of the book's builds
pruned = copy.deepcopy(net)                                 # a copy: net, W1 and W2 stay as Example 25-14 left them
masks = []
for w, orig in [(pruned[0].weight, W1), (pruned[2].weight, W2)]:
    thresh = orig.abs().flatten().quantile(0.8)             # the same 80 percent quantile rule as Example 25-14
    masks.append((orig.abs() >= thresh).float())            # 1 keeps a weight, 0 removes it
    w.data = orig * masks[-1]
def acc_of(m):
    with torch.no_grad(): return (m(Xte_t).argmax(1).numpy() == yte).mean()
print(f"80% pruned, before retraining: accuracy {acc_of(pruned):.3f}")
po = torch.optim.Adam(pruned.parameters(), lr=0.01)
for _ in range(100):                                        # 100 full-batch steps with the mask held fixed
    po.zero_grad(); nn.functional.cross_entropy(pruned(Xtr_t), ytr_t).backward(); po.step()
    with torch.no_grad():                                   # set the pruned weights back to zero after every step
        pruned[0].weight.mul_(masks[0]); pruned[2].weight.mul_(masks[1])
print(f"80% pruned, after 100 retraining steps: accuracy {acc_of(pruned):.3f}")

# %% [markdown]
# ## The same student on soft targets and on hard labels

# %%
torch.set_num_threads(2)                                    # two threads, the thread count of the book's builds
torch.manual_seed(1)
student0 = nn.Sequential(nn.Linear(64, 12), nn.ReLU(), nn.Linear(12, 10))   # one initialization
soft_student, hard_student = copy.deepcopy(student0), copy.deepcopy(student0)
def train_student(st, use_soft):
    o = torch.optim.Adam(st.parameters(), lr=0.01)
    for _ in range(300):
        out = st(Xtr_t)
        if use_soft:                                        # the teacher's T=2 soft targets from Example 25-14
            loss = nn.functional.kl_div(torch.log_softmax(out / 2.0, 1), soft, reduction="batchmean")
        else:                                               # the hard labels alone
            loss = nn.functional.cross_entropy(out, ytr_t)
        o.zero_grad(); loss.backward(); o.step()
    return acc_of(st)
print(f"same start, soft targets (T=2): accuracy {train_student(soft_student, True):.3f}")
print(f"same start, hard labels:        accuracy {train_student(hard_student, False):.3f}")

# %% [markdown]
# Example 25-18 rounds every nn.Linear weight of Qwen2.5-0.5B-Instruct except the tied output head to int8 and int4 grids with a scale per row, per group of 32, or per matrix, stores them back in float32, and scores each setting on the 200 held-out reviews by zero-shot accuracy and by negative log likelihood per token. Each of its five settings costs 400 forward passes of the 0.5B model on the CPU, so the listing runs in the companion script `11_llm_quantization.py` rather than in this walkthrough.

# %% [markdown]
# Example 25-19 runs speculative decoding with distilbert/distilgpt2 as the draft and openai-community/gpt2 as the target: the exact output distribution of one step, a seeded k = 4 sampling loop, the measured cost coefficient c, and the library's assisted generation against plain greedy decoding.

# %% [markdown]
# Example 25-19. Speculative decoding with a real draft and target: the exact step, a sampling loop, the cost coefficient, and assisted generation.

# %%
import warnings; warnings.filterwarnings("ignore")
import time
import torch
torch.set_num_threads(2)                                    # two threads, the thread count of the book's builds
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

gtok = AutoTokenizer.from_pretrained("openai-community/gpt2")                 # draft and target share this tokenizer
target = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").eval()  # p, the model we want to sample from
draft = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2").eval()   # q, the cheaper proposer
print(f"parameters: target {sum(t.numel() for t in target.parameters()):,} in {target.config.n_layer} layers, "
      f"draft {sum(t.numel() for t in draft.parameters()):,} in {draft.config.n_layer} layers")
rt_test = load_dataset("cornell-movie-review-data/rotten_tomatoes")["test"]
prompts = [" ".join(rt_test[i]["text"].split()[:6]) for i in range(0, 1066, 107)]   # 10 real prompts
print(f"{len(prompts)} prompts, the first: {prompts[0]!r}")

@torch.no_grad()
def next_probs(m, ids):                                     # next-token distribution, in float64 for the exact check
    return torch.softmax(m(ids).logits[0, -1].double(), -1)

# (a) One speculative step at a real prompt: accept x ~ q with probability min(1, p/q), else resample from
#     norm(max(0, p - q)). The output distribution is computed exactly and compared with p itself.
ids = gtok(prompts[0], return_tensors="pt").input_ids
p, q = next_probs(target, ids), next_probs(draft, ids)
accept = torch.clamp(p / q, max=1.0)
beta = (q * accept).sum()                                   # probability that the draft token is accepted
residual = torch.clamp(p - q, min=0); residual = residual / residual.sum()
step = q * accept + (1 - beta) * residual
tv = 0.5 * (p - q).abs().sum()
print(f"(a) beta {beta:.4f}, sum of min(p, q) {torch.minimum(p, q).sum():.4f}, 1 - beta {1 - beta:.4f}, "
      f"total variation {tv:.4f}; largest |step - p| over {p.numel():,} tokens {(step - p).abs().max():.1e}")

# (b) Speculative sampling with k = 4 draft tokens per target pass, seeded per prompt.
#     No key-value cache here: this loop counts tokens and passes, it is not a fast implementation.
@torch.no_grad()
def speculative(prompt, n_new=40, k=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = gtok(prompt, return_tensors="pt").input_ids
    start = ids.shape[1]; stats = {"passes": 0, "draft_runs": 0, "tested": 0, "accepted": 0}
    while ids.shape[1] - start < n_new:
        d_ids, qs = ids, []
        for _ in range(k):                                  # the draft proposes k tokens, one run each
            qj = next_probs(draft, d_ids); stats["draft_runs"] += 1
            x = torch.multinomial(qj.float(), 1, generator=g)
            qs.append(qj); d_ids = torch.cat([d_ids, x.view(1, 1)], 1)
        logits = target(d_ids).logits[0].double(); stats["passes"] += 1   # ONE target run scores all k positions
        n0 = ids.shape[1]
        for j in range(k):
            pj = torch.softmax(logits[n0 - 1 + j], -1); x = d_ids[0, n0 + j]
            stats["tested"] += 1
            if torch.rand(1, generator=g).item() < min(1.0, (pj[x] / qs[j][x]).item()):
                stats["accepted"] += 1
                continue
            r = torch.clamp(pj - qs[j], min=0); r = r / r.sum()          # rejected: resample from the residual
            y = torch.multinomial(r.float(), 1, generator=g)
            ids = torch.cat([d_ids[:, :n0 + j], y.view(1, 1)], 1)
            break
        else:                                               # all k accepted: one more token from the target itself
            y = torch.multinomial(torch.softmax(logits[-1], -1).float(), 1, generator=g)
            ids = torch.cat([d_ids, y.view(1, 1)], 1)
    return ids.shape[1] - start, stats

tokens, totals = 0, {"passes": 0, "draft_runs": 0, "tested": 0, "accepted": 0}
for i, prompt in enumerate(prompts):
    n_tok, stats = speculative(prompt, seed=i)
    tokens += n_tok
    for key in totals: totals[key] += stats[key]
alpha = totals["accepted"] / totals["tested"]
print(f"(b) k = 4: {tokens} tokens in {totals['passes']} target passes = {tokens / totals['passes']:.2f} per pass; "
      f"{totals['draft_runs']} draft runs; acceptance {alpha:.3f} ({totals['accepted']} of {totals['tested']} tested)")
print(f"    formula (1 - a^5) / (1 - a) at a = {alpha:.3f}: {(1 - alpha ** 5) / (1 - alpha):.2f} tokens per pass")

# (c) The cost coefficient c of Leviathan, Kalman, and Matias (2023): the time of one draft run divided by the
#     time of one target run at the same context, both with their key-value cache, averaged over 100 repeats.
@torch.no_grad()
def run_time(m, ids, new=1, reps=100):
    L = ids.shape[1] - new
    cache = m(ids[:, :L], use_cache=True).past_key_values
    m(ids[:, L:], past_key_values=cache, use_cache=True); cache.crop(-new)   # one warm-up run
    t0 = time.perf_counter()
    for _ in range(reps):
        m(ids[:, L:], past_key_values=cache, use_cache=True)
        cache.crop(-new)                                    # drop the new tokens: the same context every time
    return (time.perf_counter() - t0) / reps

enc0 = gtok(prompts[0], return_tensors="pt")
context = target.generate(**enc0, max_new_tokens=40, do_sample=False, pad_token_id=gtok.eos_token_id)
t_draft, t_target, t_verify = run_time(draft, context), run_time(target, context), run_time(target, context, new=5)
c = t_draft / t_target
print(f"(c) one run at a {context.shape[1]}-token context: draft {1e3 * t_draft:.2f} ms, target {1e3 * t_target:.2f} ms, "
      f"c = {c:.2f}; a target run over 5 new tokens {1e3 * t_verify:.2f} ms = {t_verify / t_target:.2f} single runs")

# Along the target's own greedy text, how often does the draft's top token agree? That rate plays the part of
# alpha for greedy assisted decoding, where a draft token is kept exactly when the two top tokens agree.
agree = total = 0
greedy = {}
for prompt in prompts:
    enc = gtok(prompt, return_tensors="pt")
    out = target.generate(**enc, max_new_tokens=40, do_sample=False, pad_token_id=gtok.eos_token_id)
    greedy[prompt] = out
    with torch.no_grad():
        guess = draft(out).logits[0, :-1].argmax(-1)        # the draft's top token at every position
    n0 = enc.input_ids.shape[1]
    agree += (guess[n0 - 1:] == out[0, n0:]).sum().item(); total += out.shape[1] - n0
a_g = agree / total
print(f"    greedy agreement of the draft with the target: {agree} of {total} = {a_g:.3f}")
for gamma in (1, 4):
    factor = (1 - a_g ** (gamma + 1)) / ((1 - a_g) * (gamma * c + 1))
    print(f"    expected improvement factor (Theorem 3.8) at a = {a_g:.3f}, c = {c:.2f}, gamma = {gamma}: {factor:.2f}")

# (d) The library's assisted generation: greedy with assistant_model against plain greedy, token for token and in
#     wall-clock time. The draft's token count and schedule are set explicitly for each setting; the last setting
#     is what transformers 5.22.1 uses when nothing is set.
def timed(**extra):
    same, seconds = 0, 0.0
    for prompt in prompts:
        enc = gtok(prompt, return_tensors="pt")
        t0 = time.perf_counter()
        out = target.generate(**enc, max_new_tokens=40, do_sample=False, pad_token_id=gtok.eos_token_id, **extra)
        seconds += time.perf_counter() - t0
        same += int(torch.equal(out, greedy[prompt]))
    return same, seconds

same, t_plain = timed()
print(f"(d) plain greedy: {same} of 10 equal to the reference run, {t_plain:.1f} s for 10 x 40 tokens")
for label, n_tok, schedule, threshold in (("1 draft token, constant, no confidence stop", 1, "constant", 0.0),
                                          ("4 draft tokens, constant, no confidence stop", 4, "constant", 0.0),
                                          ("20 draft tokens, constant, threshold 0.4 (the defaults)", 20, "constant", 0.4)):
    draft.generation_config.num_assistant_tokens = n_tok
    draft.generation_config.num_assistant_tokens_schedule = schedule
    draft.generation_config.assistant_confidence_threshold = threshold
    same, t_assist = timed(assistant_model=draft)
    print(f"    assisted, {label}: {same} of 10 equal to plain greedy, {t_assist:.1f} s ({t_plain / t_assist:.2f} x plain speed)")

# %% [markdown]
# ## 25.7 Hands-on lab: fine-tune a small open model with LoRA

# %% [markdown]
# Held-out accuracy in this lab is one six-epoch run of DistilGPT-2, not a maximum. QLoRA (Section 25.5.1) trains adapters like these on a base stored in 4 bits, and Section 23.7's weight table sizes such bases, weights only.

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

# %% [markdown]
# ## Coding exercise starters

# %%
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
def compare_prompts(bare, structured):
    ...
print("starter vocab", tok.vocab_size)

# %%
from datasets import load_dataset
ds = load_dataset("cornell-movie-review-data/rotten_tomatoes")
def few_shot_label(text):
    ...
print("starter splits", {k: len(v) for k, v in ds.items()})

# %%
import json, re
def extract_json(text):
    ...
def calc(expr):
    ...
print("starter product", 815 * 947)

# %%
def label_grader(out, expected):
    ...
def json_grader(out):
    ...
def length_grader(out, limit):
    ...
def grounding_grader(out, context):
    ...
print("starter graders", 4)

# %%
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("gpt2")
def with_retry(fn):
    ...
print("starter gpt2 vocab", tok.vocab_size)

# %%
def lora_counts(d, k, r):
    ...
print("starter scale", 16 / 4)

# %%
from sklearn.datasets import load_digits
X, y = load_digits(return_X_y=True)
def quantize_weights(W, bits):
    ...
print("starter digits", X.shape, y.shape)

# %%
from sklearn.datasets import load_digits
X, y = load_digits(return_X_y=True)
def prune(W, sparsity):
    ...
print("starter digits", X.shape)

# %%
from sklearn.datasets import load_digits
X, y = load_digits(return_X_y=True)
def distill(teacher, student):
    ...
print("starter digits", X.shape)

# %%
from sklearn.datasets import load_digits
X, y = load_digits(return_X_y=True)
def expected_tokens(alpha, k):
    ...
print("starter digits", X.shape)

# %%
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
def generate_until(prompt, stop):
    ...
print("starter vocab", tok.vocab_size)

# %%
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
def choose_letter(prompt):
    ...
print("starter vocab", tok.vocab_size)

# %%
from sklearn.datasets import load_digits
X, y = load_digits(return_X_y=True)
def group_mae(row):
    ...
print("starter digits", X.shape)

