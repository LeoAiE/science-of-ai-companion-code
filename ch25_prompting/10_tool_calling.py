# %% [markdown]
# # Section 25.3.1 Example 25-5: the typed tool-calling protocol run locally
#
# Qwen/Qwen2.5-0.5B-Instruct (float32, CPU, greedy) answers through its chat template with a tools list: the
# template renders the calc schema into the system turn, the model replies with a <tool_call> block, and the program
# parses it, runs Example 25-3's whitelist-guarded calc, and returns the result as a role tool message until a reply
# holds no call. Questions: the chapter's product and the first 10 GSM8K test problems (openai/gsm8k, Cobbe et al.,
# 2021), each run with no system message and then with a system message that asks for the calculator on every
# arithmetic step, at most four rounds and 320 new tokens per reply. Replies of up to 320 tokens take the run past the
# walkthrough's budget on the CPU at two threads, which is why the listing lives here. Tested with transformers 5.22.1,
# datasets 5.1.0 and torch 2.15.0 (CPU).

# %% [markdown]
# Example 25-5. The typed tool-calling protocol run locally through the chat template.

# %%
import warnings; warnings.filterwarnings("ignore")
import json, re
from collections import Counter
import torch
torch.set_num_threads(2)                                    # two threads, the thread count of the book's builds
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.float32).eval()

def calc(expr):                                             # Example 25-3's whitelist-guarded evaluator, repeated
    if re.fullmatch(r"[0-9.+\-*/() ]+", expr) and "**" not in expr:
        return eval(expr)
    return "error"

# The tool, declared as a JSON schema in the same function format the provider snippet uses.
tools = [{"type": "function",
          "function": {"name": "calc",
                       "description": "Evaluate an arithmetic expression.",
                       "parameters": {"type": "object",
                                      "properties": {"expression": {"type": "string",
                                                                    "description": "digits and + - * / ( ) only"}},
                                      "required": ["expression"]}}}]

# The chat template turns the tools list into text: a system turn that lists the schemas and asks for
# <tool_call> blocks. Printed once, so the protocol is visible as the plain text it is.
rendered = tok.apply_chat_template([{"role": "user", "content": "hi"}], tools=tools, tokenize=False)
print(rendered.split("<|im_end|>")[0])

@torch.no_grad()
def tool_turn(msgs, n=320):
    text = tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt")
    out = model.generate(**enc, max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
    new = out[0, enc.input_ids.shape[1]:]
    return tok.decode(new, skip_special_tokens=True).strip(), len(new) == n

CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)

def solve(question, system=None, max_rounds=4):
    """The typed loop: emit, parse, run calc, return a role 'tool' message, repeat until a reply has no call."""
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": question}]
    log = {"emitted": 0, "parsed": 0, "ran": 0, "calls": [], "final": None, "capped": False}
    for _ in range(max_rounds):
        text, capped = tool_turn(msgs)
        blocks = CALL.findall(text)
        if not blocks:                                      # a reply without a tool call is the final answer
            log["final"], log["capped"] = text, capped
            return log
        log["emitted"] += len(blocks)
        calls = []
        for block in blocks:
            try:
                call = json.loads(block)
                expression = str(call["arguments"]["expression"])
            except (json.JSONDecodeError, KeyError, TypeError):
                return log                                  # an unparsable call ends the attempt with no answer
            log["parsed"] += 1
            result = calc(expression) if call.get("name") == "calc" else "error"
            log["ran"] += result != "error"
            log["calls"].append((expression, result))
            calls.append((call, result))
        msgs.append({"role": "assistant", "content": "",   # the assistant turn that holds the typed calls
                     "tool_calls": [{"type": "function", "function": c} for c, _ in calls]})
        for _, result in calls:
            msgs.append({"role": "tool", "content": str(result)})   # each result goes back as a tool message
    return log

def last_number(text):
    nums = re.findall(r"-?\d[\d,]*\.?\d*", (text or "").replace("$", ""))
    return float(nums[-1].replace(",", "").rstrip(".")) if nums else None

# Real questions: the chapter's product, then the first 10 GSM8K test problems (Cobbe et al., 2021), whose
# reference answers follow the marker '####'.
gsm = load_dataset("openai/gsm8k", "main")["test"]
questions = [("What is 1234 multiplied by 5678?", float(1234 * 5678))] + \
            [(gsm[i]["question"], float(gsm[i]["answer"].split("####")[-1].replace(",", ""))) for i in range(10)]
TOOL_SYSTEM = ("You are a helpful assistant. Use the calc tool for every arithmetic step, "
               "then give the final answer as a number.")
for name, system in (("no system message", None), ("calculator instruction", TOOL_SYSTEM)):
    totals = Counter()
    print(f"\n{name}:")
    for i, (question, gold) in enumerate(questions):
        log = solve(question, system)
        got = last_number(log["final"])
        ok = got is not None and abs(got - gold) < 1e-6
        totals.update(emitted=log["emitted"], parsed=log["parsed"], ran=log["ran"], right=int(ok), capped=int(log["capped"]))
        label = "1234 x 5678" if i == 0 else f"GSM8K test {i - 1}"
        calls = "; ".join(f"calc({e!r}) = {r}" for e, r in log["calls"]) or "no tool call"
        shown = "none" if got is None else f"{got:.10g}"
        print(f"  {label:13s} {calls} | last number {shown}, gold {gold:.10g}, {'right' if ok else 'wrong'}")
        if i == 0:
            print(f"  {'':13s} final reply: {log['final']!r}")
    print(f"  11 questions: calls emitted {totals['emitted']}, parsed {totals['parsed']}, run {totals['ran']}, "
          f"final answers right {totals['right']}, final replies stopped at 320 tokens {totals['capped']}")
