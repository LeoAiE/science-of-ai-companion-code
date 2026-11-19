"""Appendix H solutions for Chapter 1, as runnable cells.

Generated from Solutions.md (Appendix H). Every code block of this chapter's solutions appears here
verbatim, split into one cell per exercise or drill topic, so each answer can be run and checked against
the expected outputs it states. Keep Solutions.md as the source; regenerate after any change.
"""

# %% [markdown]
# ## 1.9 coding exercises

# %% [markdown]
# 1. Two-sum, O(n) with a hash map of value -> index

# %%
# 1. Two-sum, O(n) with a hash map of value -> index
def two_sum(nums, target):
    seen = {}
    for i, x in enumerate(nums):
        if target - x in seen:
            return [seen[target - x], i]
        seen[x] = i
    return None
print(two_sum([2, 7, 11, 15], 9))   # [0, 1]   because 2 + 7 == 9

# %% [markdown]
# 2. Palindrome, O(n) by comparing the string to its reverse

# %%
# 2. Palindrome, O(n) by comparing the string to its reverse
def is_palindrome(s):
    return s == s[::-1]
print(is_palindrome("racecar"), is_palindrome("hello"))   # True False

# %% [markdown]
# 3. First non-repeating character, O(n): count then scan

# %%
# 3. First non-repeating character, O(n): count then scan
from collections import Counter
def first_unique(s):
    counts = Counter(s)
    for ch in s:
        if counts[ch] == 1:
            return ch
    return None
print(first_unique("aabbcde"))      # c

# %% [markdown]
# 4. Find duplicates, single O(n) pass with a set

# %%
# 4. Find duplicates, single O(n) pass with a set
def has_duplicate(items):
    seen = set()
    for x in items:
        if x in seen:
            return True
        seen.add(x)
    return False
print(has_duplicate([1, 2, 3, 2]))  # True

# %% [markdown]
# 5. FizzBuzz, clean conditional logic (check the 15 case first)

# %%
# 5. FizzBuzz, clean conditional logic (check the 15 case first)
def fizzbuzz(n):
    for i in range(1, n + 1):
        if i % 15 == 0: print("FizzBuzz")
        elif i % 3 == 0: print("Fizz")
        elif i % 5 == 0: print("Buzz")
        else: print(i)
fizzbuzz(5)     # 1, 2, Fizz, 4, Buzz

# %% [markdown]
# 6. Best tipping day: load, filter, engineer, group (tips: 244 real bills)

# %%
# 6. Best tipping day: load, filter, engineer, group (tips: 244 real bills)
import seaborn as sns

tips = sns.load_dataset("tips")                          # 244 real restaurant bills

def tip_pct_by_day(tips):
    """Keep the dinner bills, add a tip percentage column, and return its mean per day, highest first."""
    dinner = tips[tips["time"] == "Dinner"].copy()       # 176 dinner bills remain
    dinner["tip_pct"] = dinner["tip"] / dinner["total_bill"] * 100
    by_day = dinner.groupby("day", observed=True)["tip_pct"].mean().round(2)
    return by_day.sort_values(ascending=False)

print(tip_pct_by_day(tips))
# day
# Sun     16.69
# Thur    15.97
# Fri     15.89
# Sat     15.32
# Name: tip_pct, dtype: float64
print(tips[tips["time"] == "Dinner"]["day"].value_counts())   # how many bills stand behind each mean
# day
# Sat     87
# Sun     76
# Fri     12
# Thur     1
# Name: count, dtype: int64
# Sunday diners tip best, at 16.69 percent on average. Thursday's second place rests on a
# single dinner bill and Friday's third place on 12, so only the Sunday and Saturday means,
# built from 76 and 87 bills, are solid enough to compare.

# %% [markdown]
# 7. Fares in a grid: pivot table plus a grouped bar chart saved headlessly

# %%
# 7. Fares in a grid: pivot table plus a grouped bar chart saved headlessly
import matplotlib
matplotlib.use("Agg")                                    # render to files only
import matplotlib.pyplot as plt
import seaborn as sns

titanic = sns.load_dataset("titanic")                    # the passenger table of the 1.8 lab

def fare_grid(titanic):
    """Return the pivot table of mean fare, with pclass as the rows and sex as the columns."""
    return titanic.pivot_table(values="fare", index="pclass",
                               columns="sex", aggfunc="mean").round(2)

def save_grouped_bars(table, path):
    """Draw the table as a grouped bar chart and save it to path as a PNG."""
    ax = table.plot(kind="bar")                          # one bar group per class
    ax.set_ylabel("mean fare (1912 pounds)")
    ax.set_title("Mean fare by class and sex")
    ax.figure.savefig(path)                              # writes the PNG, prints nothing
    plt.close(ax.figure)                                 # free the figure once it is saved

table = fare_grid(titanic)
print(table)
# sex     female   male
# pclass
# 1       106.13  67.23
# 2        21.97  19.74
# 3        16.12  12.66
save_grouped_bars(table, "fare_by_class_sex.png")
# Women's mean fares exceed men's in every class, most starkly in first.

# %% [markdown]
# 8. Engineer and compare: child and alone flags against survival

# %%
# 8. Engineer and compare: child and alone flags against survival
import seaborn as sns

titanic = sns.load_dataset("titanic")

def add_child_and_alone(t):
    """Median-fill age, then add boolean columns child (age under 16) and alone (family size of 1)."""
    t = t.copy()                                         # leave the caller's table untouched
    t["age"] = t["age"].fillna(t["age"].median())        # median-fill, as in the 1.8 lab
    t["child"] = t["age"] < 16
    t["alone"] = t["sibsp"] + t["parch"] + 1 == 1        # the family size counts the passenger
    return t

flags = add_child_and_alone(titanic)
print(flags.groupby("child")["survived"].mean().round(3))
# child
# False    0.363
# True     0.590
# Name: survived, dtype: float64
print(flags.groupby("alone")["survived"].mean().round(3))
# alone
# False    0.506
# True     0.304
# Name: survived, dtype: float64
# Being a child raises the survival rate by about 23 points (0.590 vs 0.363),
# traveling accompanied by about 20 (0.506 vs 0.304): the child flag separates
# survival slightly more strongly.

# %% [markdown]
# 9. A first fit: LinearRegression of tip on total_bill

# %%
# 9. A first fit: LinearRegression of tip on total_bill
import seaborn as sns
from sklearn.linear_model import LinearRegression

tips = sns.load_dataset("tips")

def fit_tip_on_bill(tips):
    """Fit LinearRegression predicting tip from total_bill and return the fitted model."""
    X = tips[["total_bill"]]                             # 2-D: (244, 1), as sklearn expects
    y = tips["tip"]
    return LinearRegression().fit(X, y)

model = fit_tip_on_bill(tips)
print(round(float(model.coef_[0]), 4), round(float(model.intercept_), 4))   # 0.105 0.9203
print(round(float(model.score(tips[["total_bill"]], tips["tip"])), 4))      # 0.4566
# Each extra dollar on the bill adds about 10.5 cents of tip on top of a 92-cent
# base, and the bill alone explains about 46 percent of the variance in tips.

# %% [markdown]
# 10. Retry and restore: a decorator factory, then a class-based context manager

# %%
# 10. Retry and restore: a decorator factory, then a class-based context manager
import functools
import numpy as np

def retry(times=3):                            # the factory takes the configuration
    def decorator(func):                       # the decorator takes the function
        @functools.wraps(func)
        def wrapper(*args, **kwargs):          # the wrapper runs the calls
            for attempt in range(1, times + 1):
                try:
                    result = func(*args, **kwargs)
                    print(f"{func.__name__} succeeded on attempt {attempt}")
                    return result
                except ConnectionError as e:
                    print(f"attempt {attempt} failed: {e}")
            raise RuntimeError(f"{func.__name__} failed {times} times")
        return wrapper
    return decorator

failures_left = 2                              # the flaky service fails twice, then works

@retry(times=3)
def fetch_weights():
    global failures_left                       # rebind the module-level counter (1.7.6)
    if failures_left > 0:
        failures_left -= 1
        raise ConnectionError("server busy")
    return np.array([0.5, 1.5, 2.5])

print(fetch_weights())
# attempt 1 failed: server busy
# attempt 2 failed: server busy
# fetch_weights succeeded on attempt 3
# [0.5 1.5 2.5]

class Precision:                               # a class-based context manager
    def __init__(self, n):
        self.n = n
    def __enter__(self):
        self.saved = np.get_printoptions()["precision"]   # remember the old setting
        np.set_printoptions(precision=self.n)
        return self
    def __exit__(self, exc_type, exc, tb):
        np.set_printoptions(precision=self.saved)          # restore it, error or not
        return False                                       # never swallow the exception

values = np.array([1 / 3, 2 / 3])
print(values)                     # [0.33333333 0.66666667]   the default, 8 digits
with Precision(2):
    print(values)                 # [0.33 0.67]
print(values)                     # [0.33333333 0.66666667]   restored
try:
    with Precision(1):
        print(values)             # [0.3 0.7]
        raise ValueError("boom")
except ValueError:
    pass
print(values)                     # [0.33333333 0.66666667]   restored even though the block raised

# %% [markdown]
# 11. Linked list surgery: reverse two ways, detect a cycle, and find the recursion limit

# %%
# 11. Linked list surgery: reverse two ways, detect a cycle, and find the recursion limit
import sys

class Node:
    def __init__(self, value, nxt=None):
        self.value = value
        self.next = nxt

def build(values):                     # a chain from a list, head first
    head = None
    for v in reversed(values):
        head = Node(v, head)
    return head

def to_list(head):
    out = []
    while head is not None:
        out.append(head.value); head = head.next
    return out

def reverse_iterative(head):           # O(n) time, O(1) extra space
    prev = None
    while head is not None:
        head.next, prev, head = prev, head, head.next   # flip one pointer per step
    return prev

def reverse_recursive(head):           # O(n) time, O(n) space: one frame per node
    if head is None or head.next is None:
        return head
    new_head = reverse_recursive(head.next)
    head.next.next = head              # the node after me now points back at me
    head.next = None
    return new_head

def has_cycle(head):                   # tortoise and hare: O(n) time, O(1) space
    slow = fast = head
    while fast is not None and fast.next is not None:
        slow, fast = slow.next, fast.next.next
        if slow is fast:               # the fast runner laps the slow one only on a loop
            return True
    return False

print(to_list(reverse_iterative(build([1, 2, 3, 4]))))    # [4, 3, 2, 1]
print(to_list(reverse_recursive(build([1, 2, 3, 4]))))    # [4, 3, 2, 1]
loop = build([1, 2, 3]); loop.next.next.next = loop.next    # 3 points back to 2
print(has_cycle(build([1, 2, 3])), has_cycle(loop))       # False True

big = build(list(range(10_000)))
print(to_list(reverse_iterative(big))[:3])                # [9999, 9998, 9997]
print(sys.getrecursionlimit())                            # 1000, CPython's default (1.4.10)
try:
    reverse_recursive(build(list(range(10_000))))
except RecursionError as e:
    print("RecursionError:", e)                           # RecursionError: maximum recursion depth exceeded
# The recursive reversal needs one stack frame per node, so a 10,000-node list exceeds
# the limit of 1,000 frames (and the 3,000 that older jedi releases set in a notebook,
# 1.4.10); the iterative version walks the same list in O(1) space.

# %% [markdown]
# 12. Sorting, slow and fast: insertion sort, a random-pivot quicksort, and the worst case measured

# %%
# 12. Sorting, slow and fast: insertion sort, a random-pivot quicksort, and the worst case measured
import random, sys, timeit

def insertion_sort(xs):
    out = list(xs)
    for i in range(1, len(out)):
        key, j = out[i], i - 1
        while j >= 0 and out[j] > key:     # shift the larger items one slot right
            out[j + 1] = out[j]; j -= 1
        out[j + 1] = key
    return out

def quicksort(xs):                         # the chapter's version: first-element pivot
    if len(xs) <= 1:
        return xs
    pivot, rest = xs[0], xs[1:]
    return (quicksort([x for x in rest if x < pivot]) + [pivot]
            + quicksort([x for x in rest if x >= pivot]))

def quicksort_random(xs):                  # a random pivot defuses the sorted-input case
    if len(xs) <= 1:
        return xs
    i = random.randrange(len(xs))
    pivot, rest = xs[i], xs[:i] + xs[i + 1:]
    return (quicksort_random([x for x in rest if x < pivot]) + [pivot]
            + quicksort_random([x for x in rest if x >= pivot]))

print(insertion_sort([38, 27, 43, 3, 9, 82, 10]))     # [3, 9, 10, 27, 38, 43, 82]
data = list(range(5000))                   # already sorted: the first-pivot worst case
shuffled = data[:]; random.seed(0); random.shuffle(shuffled)
sys.setrecursionlimit(10_000)              # the first-pivot version recurses 5,000 deep on sorted input
print(quicksort(shuffled) == quicksort_random(shuffled) == insertion_sort(shuffled) == data)   # True
for name, xs in [("sorted", data), ("shuffled", shuffled)]:
    for fn in (insertion_sort, quicksort, quicksort_random):
        t = min(timeit.repeat(lambda: fn(xs), number=1, repeat=3))
        print(f"{name:8s} {fn.__name__:16s} {t:.3f}s")
# One run saved in appendix_h_solutions_ch01.ipynb on the reference machine (Python 3.13.14,
# Apple M1 Max); the times vary by machine:
# sorted   insertion_sort   0.000s
# sorted   quicksort        0.561s
# sorted   quicksort_random 0.005s
# shuffled insertion_sort   0.429s
# shuffled quicksort        0.005s
# shuffled quicksort_random 0.006s
# On sorted input the first pivot is always the minimum, every split is maximally lopsided,
# the recursion is n deep (5,000 frames, beyond the default limit of 1,000, hence the raised
# limit) and the comprehensions touch about n squared / 2 elements: the quadratic worst case
# of 1.4.1, more than a hundred times slower here than the random pivot, whose expected cost is O(n log n)
# whatever the input order. Insertion sort runs in O(n) on sorted input, because every key
# stops after one comparison, and in O(n squared) on the shuffled copy, about 6 million shifts.

# %% [markdown]
# 13. Maze and components: breadth-first search for the shortest path, depth-first search for counting

# %%
# 13. Maze and components: breadth-first search for the shortest path, depth-first search for counting
from collections import deque

maze = [
    "S.#.....",
    ".##.###.",
    "....#...",
    ".##...#G",
]

def shortest_path(maze):
    rows, cols = len(maze), len(maze[0])
    start = next((r, c) for r in range(rows) for c in range(cols) if maze[r][c] == "S")
    dist = {start: 0}                      # steps from S to every cell reached so far
    queue = deque([start])
    while queue:
        r, c = queue.popleft()             # the queue makes it breadth first
        if maze[r][c] == "G":
            return dist[(r, c)]            # the first time G is dequeued is the shortest route
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and maze[nr][nc] != "#" and (nr, nc) not in dist:
                dist[(nr, nc)] = dist[(r, c)] + 1
                queue.append((nr, nc))
    return None                            # G was never reached

print(shortest_path(maze))                 # 12

def count_components(graph):
    seen, components = set(), 0
    for start in graph:
        if start in seen:
            continue
        components += 1                    # an unseen node starts a new component
        stack, seen = [start], seen | {start}
        while stack:                       # depth first: the stack takes the newest node
            node = stack.pop()
            for neighbor in graph[node]:
                if neighbor not in seen:
                    seen.add(neighbor); stack.append(neighbor)
    return components

graph = {"A": ["B"], "B": ["A", "C"], "C": ["B"], "D": ["E"], "E": ["D"], "F": []}
print(count_components(graph))             # 3
# The breadth-first search enqueues each open cell at most once and looks at four neighbors
# per cell, so it is O(cells) in time and space; the depth-first count visits every node and
# every edge once, O(V + E). The 12-step route runs down the left edge, along row 2, through
# the gap in row 3, and up and around the wall to G.

# %% [markdown]
# 14. Pool and loop: run as its own script, code/ch01_python/05_chapter1_concurrency_exercise.py

# %%
# 14. Pool and loop: run as its own script, code/ch01_python/05_chapter1_concurrency_exercise.py
# (ProcessPoolExecutor workers must import count_primes from a file, and asyncio.run needs
# no event loop already running, which a notebook has).
import asyncio, math, time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def download(i):
    time.sleep(0.1)                                   # one simulated network read
    return f"file{i}"

def count_primes(lo, hi):                             # pure Python arithmetic: CPU bound
    count = 0
    for n in range(max(lo, 2), hi):
        if all(n % d for d in range(2, math.isqrt(n) + 1)):
            count += 1
    return count

async def download_async(i):
    await asyncio.sleep(0.1)                          # the asynchronous cousin of time.sleep
    return f"file{i}"

async def download_all():
    return await asyncio.gather(*(download_async(i) for i in range(20)))

if __name__ == "__main__":                            # workers import this file again: guard the launch
    t0 = time.perf_counter(); seq = [download(i) for i in range(20)]; t_seq = time.perf_counter() - t0
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=20) as pool:
        thr = list(pool.map(download, range(20)))
    t_thr = time.perf_counter() - t0
    print(seq == thr, f"downloads: sequential {t_seq:.2f}s, threads {t_thr:.2f}s")
    # True downloads: sequential 2.17s, threads 0.11s

    step = 25_000                                     # sixteen equal slices of the numbers below 400,000
    los, his = [k * step for k in range(16)], [(k + 1) * step for k in range(16)]
    t0 = time.perf_counter(); total = sum(count_primes(lo, hi) for lo, hi in zip(los, his)); t_seq = time.perf_counter() - t0
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        total_thr = sum(pool.map(count_primes, los, his))
    t_thr = time.perf_counter() - t0
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=4) as pool:
        total_proc = sum(pool.map(count_primes, los, his))
    t_proc = time.perf_counter() - t0
    print(total, total_thr == total_proc == total,
          f"primes: sequential {t_seq:.2f}s, threads {t_thr:.2f}s, processes {t_proc:.2f}s")
    # 33860 True primes: sequential 0.65s, threads 0.65s, processes 0.27s

    t0 = time.perf_counter(); files = asyncio.run(download_all())
    print(len(files), files[:2], f"asyncio {time.perf_counter() - t0:.2f}s")
    # 20 ['file0', 'file1'] asyncio 0.10s
# (one run on the reference machine, an Apple M1 Max with 10 logical cores, under Python
# 3.13.14; times vary.) Twenty waits of 0.1 seconds cost 2.17 seconds in a row and 0.11
# seconds on twenty threads, because the GIL is released
# while a thread sleeps, so the waits overlap, and asyncio overlaps them the same way on one
# thread. The prime count is pure Python arithmetic: four threads take as long as the
# sequential run, since only one of them can execute bytecode at a time, while four
# processes, each with its own interpreter and GIL, finish in about two fifths of the time
# (0.27 against 0.65 seconds); the gap to the ideal quarter is the cost of starting the workers and shipping the slices and
# results between processes. That is the rule of Table 1-3 measured.

# %% [markdown]
# 15. Profile the lab: pandas groupby against a dictionary loop, at two table sizes

# %%
# 15. Profile the lab: pandas groupby against a dictionary loop, at two table sizes
import cProfile, pstats, io
import pandas as pd
import seaborn as sns

titanic = sns.load_dataset("titanic")[["pclass", "fare"]]           # the raw 891 rows, before the lab's two-row drop
big = pd.concat([titanic] * 100, ignore_index=True)                  # 89,100 rows: the table stacked 100 times

def mean_fare_pandas(df):
    return df.groupby("pclass")["fare"].mean()

def mean_fare_loop(df):
    totals, counts = {}, {}
    for pclass, fare in zip(df["pclass"].tolist(), df["fare"].tolist()):
        totals[pclass] = totals.get(pclass, 0.0) + fare
        counts[pclass] = counts.get(pclass, 0) + 1
    return {k: round(totals[k] / counts[k], 2) for k in sorted(totals)}

print(mean_fare_pandas(titanic).round(2).to_dict(), mean_fare_loop(titanic))
# {1: 84.15, 2: 20.66, 3: 13.68} {1: 84.15, 2: 20.66, 3: 13.68}   the two agree

def pipeline(df):
    for _ in range(20):
        mean_fare_pandas(df)
    for _ in range(20):
        mean_fare_loop(df)

for name, df in [("891 rows", titanic), ("89,100 rows", big)]:
    profiler = cProfile.Profile()
    profiler.enable(); pipeline(df); profiler.disable()
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats("mean_fare")
    print(name); print(stream.getvalue())
# 891 rows: 46424 function calls (46303 primitive calls) in 0.012 seconds (the run saved in
# appendix_h_solutions_ch01.ipynb on the reference machine, Python 3.13.14 on an Apple M1 Max,
# whose temporary file name is shortened here to profile_fares.py)
#    ncalls  tottime  percall  cumtime  percall filename:lineno(function)
#        20    0.004    0.000    0.007    0.000 profile_fares.py:12(mean_fare_loop)
#        20    0.000    0.000    0.005    0.000 profile_fares.py:9(mean_fare_pandas)
# 89,100 rows: 3575139 function calls (3575017 primitive calls) in 0.744 seconds
#        20    0.398    0.020    0.630    0.031 profile_fares.py:12(mean_fare_loop)
#        20    0.000    0.000    0.026    0.001 profile_fares.py:9(mean_fare_pandas)
# The loop is the target at both sizes. On 891 rows it already costs more than the groupby
# (about 0.35 ms against 0.25 ms per call, rough figures because the profiler prints only
# three decimals), and its time grows with the row count, about ninetyfold on the stacked
# table (0.007 to 0.630 seconds for twenty calls), while the groupby's cost grows only about
# fivefold (0.25 ms to 1.3 ms per call) because its work happens in compiled code. Replacing the loop with the
# groupby, which is the vectorization, is the only change worth making; the exact times vary
# by machine.

# %% [markdown]
# 16. Join the class names: an inner merge, then a left merge with a missing lookup row

# %%
# 16. Join the class names: an inner merge, then a left merge with a missing lookup row
import pandas as pd
import seaborn as sns

titanic = sns.load_dataset("titanic")[["pclass", "fare"]]
lookup = pd.DataFrame({"pclass": [1, 2, 3], "class_name": ["First", "Second", "Third"]})

def mean_fare_by_class_name(titanic, lookup, how="inner"):
    """Merge the class names onto titanic with the given join and return the mean fare per name."""
    merged = titanic.merge(lookup, on="pclass", how=how)        # inner by default: shared keys only
    return merged.groupby("class_name")["fare"].mean().round(2)

print(mean_fare_by_class_name(titanic, lookup))
# class_name
# First     84.15
# Second    20.66
# Third     13.68
# Name: fare, dtype: float64
partial = lookup[lookup["pclass"] != 3]                         # the lookup loses its third-class row
left = titanic.merge(partial, on="pclass", how="left")          # keep every passenger row
print(left.shape, left["class_name"].isna().sum())              # (891, 3) 491
print(titanic.merge(partial, on="pclass").shape)                # (400, 3)   the default inner join
print(mean_fare_by_class_name(titanic, partial, how="left"))
# class_name
# First     84.15
# Second    20.66
# Name: fare, dtype: float64
# A left merge keeps every row of the left table and fills the columns of the right table
# with NaN where the key found no partner, so the 491 third-class passengers survive with
# an empty class_name; the default inner join keeps only keys present on both sides and
# silently drops those 491 rows, leaving 400. The left-joined means still list only First
# and Second, because groupby leaves out rows whose key is NaN unless it is called with
# dropna=False.

# %% [markdown]
# 17. Two pipelines, one disagreement: sentiment and zero-shot classification with small pinned models

# %%
# 17. Two pipelines, one disagreement: sentiment and zero-shot classification with small pinned models
from transformers import pipeline

SENTIMENT_MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"   # pinned in 1.5.13
TOPIC_MODEL = "MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33"             # 142 MB, trained for zero-shot use
LABELS = ["bug", "feature request", "question"]
notes = [
    "The validation loss stopped improving after epoch 12.",       # the starter's two notes
    "Please let the trainer resume from a saved checkpoint.",
    "The loss dropped steadily for the first ten epochs.",         # and eight more
    "The GPU ran out of memory on the first batch.",
    "Checkpoints are saving correctly every epoch.",
    "Why does the loss jump every time the data loader restarts?",
    "The run finished early and the metrics look excellent.",
    "Training is twice as slow as yesterday on the same machine.",
    "Could we log the learning rate alongside the loss?",
    "The tokenizer silently truncated every input longer than 512 tokens.",
]

def label_sentiment(notes):
    """Run the pinned sentiment pipeline on notes and return one (label, score) pair per note."""
    sentiment = pipeline("sentiment-analysis", model=SENTIMENT_MODEL)
    return [(r["label"], r["score"]) for r in sentiment(notes)]

def label_topics(notes, labels):
    """Run a zero-shot-classification pipeline with a small pinned model; return each note's top label and score."""
    topic = pipeline("zero-shot-classification", model=TOPIC_MODEL)
    return [(r["labels"][0], r["scores"][0]) for r in topic(notes, candidate_labels=labels)]

for note, (label, score) in zip(notes, label_sentiment(notes)):
    print(f"{label:8s} {score:.3f}  {note}")
# NEGATIVE 0.999  The validation loss stopped improving after epoch 12.
# NEGATIVE 0.996  Please let the trainer resume from a saved checkpoint.
# NEGATIVE 0.999  The loss dropped steadily for the first ten epochs.
# NEGATIVE 1.000  The GPU ran out of memory on the first batch.
# POSITIVE 0.996  Checkpoints are saving correctly every epoch.
# NEGATIVE 0.999  Why does the loss jump every time the data loader restarts?
# POSITIVE 1.000  The run finished early and the metrics look excellent.
# NEGATIVE 0.999  Training is twice as slow as yesterday on the same machine.
# NEGATIVE 1.000  Could we log the learning rate alongside the loss?
# NEGATIVE 0.999  The tokenizer silently truncated every input longer than 512 tokens.
for note, (label, score) in zip(notes, label_topics(notes, LABELS)):
    print(f"{label:16s} {score:.3f}  {note}")
# bug              0.819  The validation loss stopped improving after epoch 12.
# question         0.776  Please let the trainer resume from a saved checkpoint.
# question         0.801  The loss dropped steadily for the first ten epochs.
# bug              0.969  The GPU ran out of memory on the first batch.
# bug              0.932  Checkpoints are saving correctly every epoch.
# question         0.783  Why does the loss jump every time the data loader restarts?
# question         0.679  The run finished early and the metrics look excellent.
# bug              0.739  Training is twice as slow as yesterday on the same machine.
# question         0.879  Could we log the learning rate alongside the loss?
# bug              0.967  The tokenizer silently truncated every input longer than 512 tokens.
# The sentiment model has no label for "neither": it calls the two neutral questions
# NEGATIVE with scores near 1.0, the polite request to resume from a checkpoint NEGATIVE at
# 0.996, and even the good news that the loss dropped steadily NEGATIVE at 0.999. The
# zero-shot model sorts the two questions and the memory and truncation reports correctly,
# but it calls the request to resume from a checkpoint a question (0.776) rather than a
# feature request, labels "Checkpoints are saving correctly every epoch" a bug at 0.932
# although the sentence reports that something works, and calls the good news about the
# loss a question. Each model is plainly wrong on at least one sentence, the usual reason
# to fine-tune on the texts a project actually sees rather than trust a borrowed label set.

# %% [markdown]
# ## 1.9 coding drills

# %% [markdown]
# Warm-ups

# %%
# Warm-ups
from collections import Counter
print(set("abcdefghijklmnopqrstuvwxyz") <= set("the quick brown fox jumps over the lazy dog"))  # True   pangram
a, b = 0, 1
fib = []
for _ in range(8): fib.append(a); a, b = b, a + b
print(fib)                                          # [0, 1, 1, 2, 3, 5, 8, 13]   Fibonacci by loop
print(Counter([1, 3, 2, 3, 4, 3]).most_common(1)[0][0])   # 3   most frequent element
n = 15
print("FizzBuzz" if n % 15 == 0 else "Fizz" if n % 3 == 0 else "Buzz" if n % 5 == 0 else n)   # FizzBuzz

# Strings: the first five (reverse, palindrome, count vowels, anagram, reverse
# word order) are worked with their outputs in the 1.1.6 code block, which owns
# them; only the three drills new to the bank are solved here.
print(len("this is a test".split()))                # 4            word count
print("data science".title())                       # Data Science title case
print(sum(int(d)**3 for d in str(153)) == 153)      # True         Armstrong number

# %% [markdown]
# Lists and dictionaries

# %%
# Lists and dictionaries
nums = [1, 2, 2, 3, 3, 3]
print([x for x in set(nums) if nums.count(x) > 1])    # [2, 3]        repeats
print(sorted(set([1, 2, 3, 4]))[-2])                  # 3            second largest distinct
print([x for sub in [[1, 2], [3, 4]] for x in sub])   # [1, 2, 3, 4] flatten
print(Counter(nums).most_common(2))                   # [(3, 3), (2, 2)] two most frequent
d = {"a": 2, "b": 1, "c": 3}
print(dict(sorted(d.items(), key=lambda kv: kv[1])))  # {'b': 1, 'a': 2, 'c': 3} sort by value
xs = [1, 2, 3, 4, 5, 6, 7]
print([(a, b) for i, a in enumerate(xs) for b in xs[i+1:] if a + b == 8])  # [(1, 7), (2, 6), (3, 5)]

# %% [markdown]
# Numbers

# %%
# Numbers
import math
n = 7; print(n >= 2 and all(n % i != 0 for i in range(2, math.isqrt(n) + 1)))   # True   prime test
print(math.gcd(12, 18))                                     # 6      gcd
y = 2024; print(y % 4 == 0 and (y % 100 != 0 or y % 400 == 0))   # True   leap year

# %% [markdown]
# NumPy

# %%
# NumPy
import numpy as np
print(np.arange(1, 7).reshape(2, 3).tolist())          # [[1, 2, 3], [4, 5, 6]]
arr = np.array([3, 7, 1, 9, 2])
print(arr.max(), arr.min(), arr.argmax())              # 9 1 3
print(np.unique(np.array([1, 2, 2, 3, 3])).tolist())   # [1, 2, 3]
v = np.array([-2, 5, -7, 8]); v[v < 0] = 0; print(v.tolist())   # [0, 5, 0, 8]
x = np.array([10.0, 20.0, 30.0])
print(((x - x.min()) / (x.max() - x.min())).tolist())  # [0.0, 0.5, 1.0]
