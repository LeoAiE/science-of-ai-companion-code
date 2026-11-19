# %% [markdown]
# # Section 27.10 Example 27-29: Example 27-2's loop on 50 GSM8K problems
#
# Runs Example 27-2's ReAct loop (the first JSON object of each reply, a reply budget of 160 new tokens, four
# steps, the repeat and budget guards) on the first 50 problems of the GSM8K test split (openai/gsm8k, config
# main; Cobbe et al., 2021), the same 50 as Example 23-13, with Qwen/Qwen2.5-0.5B-Instruct on the CPU in 32-bit
# floats and greedy decoding. Grading is numeric, and every run's ending is recorded as a trajectory record. One
# run takes about 28 minutes on the reference machine, which is why the listing lives here rather than
# in the chapter walkthrough. Tested with transformers 5.22.1, datasets 5.1.0 and torch 2.15.0 (CPU).

# %% [markdown]
# Example 27-29. Example 27-2's loop on the first 50 GSM8K test problems, graded numerically, with every ending recorded.

# %%
import warnings; warnings.filterwarnings("ignore")
import json, re, time
from collections import Counter
import torch
torch.set_num_threads(2)                                  # two threads, the thread count of the book's builds
from datasets import load_dataset
from transformers import pipeline
llm = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct",
               device="cpu", dtype=torch.float32)         # the CPU and 32-bit floats, as in Example 25-1
t0 = time.time()

def ask(prompt, system, n=160):                           # Example 27-2's call, with its reply budget of 160
    return llm([{"role": "system", "content": system}, {"role": "user", "content": prompt}],
               max_new_tokens=n, do_sample=False)[0]["generated_text"][-1]["content"].strip()
def calc(expr):                                           # Example 27-2's restricted evaluator
    expr = expr.split("=")[0].strip()
    return eval(expr) if re.fullmatch(r"[0-9.+\-*/() ]+", expr) and "**" not in expr else "error"

SYSTEM = (                                                # Example 27-2's protocol, unchanged
    'Solve the problem step by step using a calculator. Each reply is ONE JSON object:\n'
    '{"thought": "<reasoning>", "action": "calc", "input": "<one arithmetic expression>"}\n'
    'or, when you have the numeric answer:\n'
    '{"thought": "<reasoning>", "action": "finish", "answer": "<number>"}\n'
    'Do only ONE calculation per step. Reply with ONLY the JSON.')

def parse_action(raw):                                    # the first JSON object of the reply, as Example 27-2 parses it
    raw = re.sub(r"```(?:json)?", "", raw).replace("```", "")
    i = raw.find("{")
    if i < 0:
        return None
    try:
        return json.JSONDecoder().raw_decode(raw[i:])[0]
    except Exception:
        return None

def react(problem, max_steps=4):
    """Example 27-2's loop with its prints removed; returns (answer, how the run ended, steps, calculator calls)."""
    scratch, last_obs, seen, calls = f"Problem: {problem}", None, set(), 0
    for step in range(1, max_steps + 1):
        a = parse_action(ask(scratch, SYSTEM))
        if a is None:                                     # an unparsed reply
            if last_obs is not None:
                return last_obs, "forced: unparsed reply", step, calls
            return None, "stopped: nothing observed", step, calls
        if a.get("action") == "finish" and a.get("answer") is not None:
            return a.get("answer"), "model finish", step, calls
        expr = str(a.get("input", "")).split("=")[0].strip()
        obs = calc(expr)
        calls += 1
        if isinstance(obs, (int, float)):                 # a real number came back
            if expr in seen or step == max_steps:         # the repeat guard and the step budget
                return obs, "forced: repeat or budget", step, calls
            seen.add(expr); last_obs = obs
        scratch += (f"\nStep {step}: computed {a.get('input')} = {obs}. "
                    f"If this is the final numeric answer, reply now with "
                    f'{{"thought": "...", "action": "finish", "answer": "{obs}"}}.')
    if last_obs is not None:
        return last_obs, "forced: loop exhausted", max_steps, calls
    return None, "stopped: nothing observed", max_steps, calls

def number(x):                                            # the first number in the answer, commas removed
    m = re.search(r"-?\d[\d,]*(?:\.\d+)?", str(x)) if x is not None else None
    return float(m.group().replace(",", "")) if m else None

gsm = load_dataset("openai/gsm8k", "main", split="test")  # Cobbe et al. (2021); the same 50 problems as Example 23-13
N = 50
right, ends, ends_right, steps_total, calls_total = 0, Counter(), Counter(), 0, 0
for i in range(N):
    gold = float(gsm[i]["answer"].split("####")[-1].strip().replace(",", ""))
    t = time.time()
    ans, how, steps, calls = react(gsm[i]["question"])
    ok = number(ans) is not None and abs(number(ans) - gold) < 1e-6     # numeric grading
    right += ok; ends[how] += 1; ends_right[how] += ok; steps_total += steps; calls_total += calls
    print(f"{i:2d} gold {gold:g} | agent {str(ans)[:14]:14s} {'OK' if ok else 'no'} | {how}, {steps} steps, "
          f"{calls} calc | {time.time() - t:.1f} s", flush=True)
print(f"ReAct with a calculator on the first {N} GSM8K test problems: {right}/{N} right")
for how in sorted(ends):
    print(f"  {how:26s} {ends[how]:2d} runs, {ends_right[how]:2d} right")
print(f"mean steps {steps_total / N:.2f}, mean calculator calls {calls_total / N:.2f}")
print(f"one run: {time.time() - t0:.1f} s")
