# %% [markdown]
# # Section 1.9 - Interview coding-drill bank
#
# Short, classic interview problems with one-line or few-line answers, grouped by topic.
# These mirror the 100-question style: each prints exactly the annotated output. The point
# is the idiom (slice, comprehension, Counter, set, NumPy) more than the problem.

# %%
import numpy as np
from collections import Counter, defaultdict

# %% [markdown]
# ## Strings
# %%
print("hello"[::-1])                                   # olleh           reverse
print("madam" == "madam"[::-1])                        # True            palindrome
print(sum(1 for c in "education" if c in "aeiou"))     # 5               count vowels
print(sorted("listen") == sorted("silent"))            # True            anagram
print(" ".join("hello world".split()[::-1]))           # world hello     reverse words
print(len("this is a test".split()))                   # 4               word count
print("data science".title())                          # Data Science    title case
print(sum(int(d)**3 for d in str(153)) == 153)         # True            Armstrong number

# %% [markdown]
# ## Lists and dictionaries
# %%
nums = [1, 2, 2, 3, 3, 3]
print([x for x in set(nums) if nums.count(x) > 1])     # [2, 3]          duplicates
print(sorted(set([1, 2, 3, 4]))[-2])                   # 3               second largest
print([x for sub in [[1, 2], [3, 4]] for x in sub])    # [1, 2, 3, 4]    flatten nested
print(Counter([1, 2, 2, 3, 3, 3]).most_common(2))      # [(3,3),(2,2)]   top-2 frequent
d = {"a": 2, "b": 1, "c": 3}
print(dict(sorted(d.items(), key=lambda kv: kv[1])))   # {'b':1,'a':2,'c':3} sort by value
print(dict([("a", 1), ("b", 2)]))                      # {'a':1,'b':2}   pairs to dict
grouped = defaultdict(list)
for w in ["apple", "ant", "bee", "bat"]:
    grouped[w[0]].append(w)
print(dict(grouped))                # {'a':['apple','ant'],'b':['bee','bat']} group by first letter
nums2 = [1, 2, 3, 4, 5, 6, 7]
print([(a, b) for i, a in enumerate(nums2) for b in nums2[i+1:] if a + b == 8])
# [(1,7),(2,6),(3,5)]   pairs summing to a target

# %% [markdown]
# ## Numbers
# %%
print(7 == 7 and all(7 % i for i in range(2, 3)))      # True            7 is prime (sqrt 7 < 3)
import math
print(math.gcd(12, 18))                                # 6               greatest common divisor
y = 2024
print(y % 4 == 0 and (y % 100 != 0 or y % 400 == 0))   # True            leap year

# %% [markdown]
# ## NumPy
# %%
print(np.arange(1, 7).reshape(2, 3).tolist())          # [[1,2,3],[4,5,6]]  arange + reshape
arr = np.array([3, 7, 1, 9, 2])
print(arr.max(), arr.min(), arr.argmax())              # 9 1 3           max, min, index of max
print(np.unique(np.array([1, 2, 2, 3, 3, 3])).tolist())   # [1, 2, 3]    unique values
v = np.array([-2, 5, -7, 8]); v[v < 0] = 0
print(v.tolist())                                      # [0, 5, 0, 8]   clamp negatives to 0
x = np.array([10.0, 20.0, 30.0])
print(((x - x.min()) / (x.max() - x.min())).tolist())  # [0.0,0.5,1.0]   min-max normalize
print(np.dot(np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])).tolist())
# [[19, 22], [43, 50]]   matrix multiply

print("\nAll interview-bank checks ran.")


# %% [markdown]
# ## 1.4.4 Hash maps: the hashability contract (`__hash__` and `__eq__`)
# %%
cache = {}
features = (0.5, 1.2, 3.0)             # an immutable feature vector
cache[features] = "cat"                # a tuple works as a key
print(cache[(0.5, 1.2, 3.0)])          # cat    equal contents, equal hash, same slot

try:
    cache[[0.5, 1.2, 3.0]] = "cat"     # a list is mutable, so it is unhashable
except TypeError as e:
    print(e)                           # unhashable type: 'list'

# deduplicating samples with a set rests on the same contract
samples = [(1, 2), (3, 4), (1, 2)]
print(set(samples))                    # {(1, 2), (3, 4)}

# %%
class Sample:
    def __init__(self, features, label):
        self.features = features       # a tuple, so it stays hashable
        self.label = label
    def __eq__(self, other):           # equal when the contents match, as in 1.2.1
        return (self.features, self.label) == (other.features, other.label)
    def __hash__(self):                # the contract: equal objects, equal hashes
        return hash((self.features, self.label))

a = Sample((1.0, 2.0), "cat")
b = Sample((1.0, 2.0), "cat")          # a different object with equal contents
print(a == b)                          # True
print(hash(a) == hash(b))              # True
print(len({a, b}))                     # 1    the set sees one sample, not two

# %% [markdown]
# ## Book 1.4.1: amortized O(1) append and the space bill
# Amortized analysis made visible: sys.getsizeof shows the capacity jumps of a
# growing list, then NumPy shows the hidden copy behind a vectorized one-liner.
# Same code as the two blocks in book subsection 1.4.1.

# %%
import sys

data = []
print(sys.getsizeof(data))            # 56    bytes for an empty list
last = sys.getsizeof(data)
for i in range(20):
    data.append(i)                    # one append at a time
    now = sys.getsizeof(data)
    if now != last:                   # speak only when the allocation jumps
        print(f"len={len(data):2d}  bytes={now}")   # len= 1 bytes=88 ... len=17 bytes=248
        last = now

# how rare are the resizes over a long run?
data, resizes, last = [], 0, sys.getsizeof([])
for i in range(100_000):
    data.append(i)
    now = sys.getsizeof(data)
    if now != last:
        resizes += 1
        last = now
print(resizes)                        # 66    resizes in 100,000 appends

# %%
import numpy as np

x = np.ones(10_000_000)               # ten million float64 values
print(x.nbytes // 1_000_000)          # 80     megabytes held by x
y = x * 2                             # the innocent one-liner
print(np.shares_memory(x, y))         # False  y is a brand-new 80 MB buffer
x *= 2                                # in place: rewrites x's own buffer instead
print(x[0], x.nbytes // 1_000_000)    # 2.0 80  new contents, same memory

# %% [markdown]
# ## 1.4.7 Dijkstra's shortest path on a weighted graph (greedy search with heapq)
# %%
import heapq

# the weighted graph of five nodes, as an adjacency list of (neighbor, weight) pairs
graph = {
    "A": [("B", 2), ("C", 5)],
    "B": [("A", 2), ("C", 1), ("D", 7)],
    "C": [("A", 5), ("B", 1), ("D", 3), ("E", 6)],
    "D": [("B", 7), ("C", 3), ("E", 1)],
    "E": [("D", 1), ("C", 6)],
}

def dijkstra(graph, start):
    dist = {start: 0}                      # best cost found so far to each node
    heap = [(0, start)]                    # the frontier: (cost so far, node)
    while heap:
        cost, node = heapq.heappop(heap)   # greedy step: expand the cheapest frontier node
        if cost > dist[node]:
            continue                       # a stale entry: a cheaper route was already found
        for neighbor, weight in graph[node]:
            new_cost = cost + weight       # cost of reaching neighbor through node
            if new_cost < dist.get(neighbor, float("inf")):
                dist[neighbor] = new_cost  # cheaper than anything known: record it
                heapq.heappush(heap, (new_cost, neighbor))
    return dist

print(dijkstra(graph, "A"))    # {'A': 0, 'B': 2, 'C': 3, 'D': 6, 'E': 7}

# %% [markdown]
# ## 1.4.11 Greedy vs dynamic programming: coin change
# %%
def greedy_change(amount, coins):
    coins = sorted(coins, reverse=True)   # always grab the biggest coin that fits
    used = []
    for coin in coins:
        while amount >= coin:             # take it as often as it still fits
            amount -= coin
            used.append(coin)
    return used                           # one pass, no choice ever reconsidered

def dp_change(amount, coins):
    best = [[]] + [None] * amount         # best[a] = fewest-coin list making amount a
    for a in range(1, amount + 1):        # build up from 1, exactly like fib_table
        for coin in coins:                # consider every coin at every subamount
            if coin <= a and best[a - coin] is not None:
                candidate = best[a - coin] + [coin]
                if best[a] is None or len(candidate) < len(best[a]):
                    best[a] = candidate   # keep the shortest way found so far
    return best[amount]

us = [25, 10, 5, 1]
print(greedy_change(63, us))              # [25, 25, 10, 1, 1, 1]   6 coins
print(dp_change(63, us))                  # [1, 1, 1, 10, 25, 25]   the same 6 coins
odd = [1, 3, 4]
print(greedy_change(6, odd))              # [4, 1, 1]   greedy: 3 coins
print(dp_change(6, odd))                  # [3, 3]      DP: 2 coins, the true optimum

# %%
# the experiment from the book: same coins, amounts 8 and 10
print(greedy_change(8, [1, 3, 4]), dp_change(8, [1, 3, 4]))     # [4, 4] [4, 4]           greedy lucky
print(greedy_change(10, [1, 3, 4]), dp_change(10, [1, 3, 4]))   # [4, 4, 1, 1] [4, 3, 3]  greedy wrong again

# %% [markdown]
# ## 1.4.9 aside - Rabin-Karp rolling hash: substring search in expected O(n+m)
#
# Reuse the previous window's hash to slide in O(1): subtract the outgoing character,
# shift by the base, add the incoming character. Hash matches are verified character
# by character because different windows can collide on one hash value.

# %%
base, mod = 256, 1_000_003          # characters are digits in base 256; a prime keeps hashes small

def fresh_hash(s):                  # hash a window from scratch: O(m) work
    h = 0
    for ch in s:
        h = (h * base + ord(ch)) % mod
    return h

text, m = "deduplicate", 4
h = fresh_hash(text[0:m])           # hash of the first window, "dedu"
print(h)                            # 361401

# slide one step to "edup" in O(1): drop the 'd', shift, append the 'p'
h = ((h - ord(text[0]) * pow(base, m - 1, mod)) * base + ord(text[m])) % mod
print(h)                            # 77377
print(h == fresh_hash(text[1:m + 1]))   # True   the rolled hash equals the from-scratch hash

# %% [markdown]
# ## Warm-ups (book 1.9, Coding drills)
# %%
from collections import Counter
print(set("abcdefghijklmnopqrstuvwxyz") <= set("the quick brown fox jumps over the lazy dog"))  # True   pangram
a, b = 0, 1
fib = []
for _ in range(8): fib.append(a); a, b = b, a + b
print(fib)                                          # [0, 1, 1, 2, 3, 5, 8, 13]   Fibonacci by loop
print(Counter([1, 3, 2, 3, 4, 3]).most_common(1)[0][0])   # 3   most frequent element
n = 15
print("FizzBuzz" if n % 15 == 0 else "Fizz" if n % 3 == 0 else "Buzz" if n % 5 == 0 else n)   # FizzBuzz

# %% [markdown]
# ## Real-dataset build exercises (book 1.9, coding exercises 6-9)
# The four build problems on the datasets the chapter already used: seaborn's
# tips (244 real restaurant bills) and titanic (891 real passengers).

# %%
# Exercise 6. Best tipping day: load, filter, engineer, group
import seaborn as sns

tips = sns.load_dataset("tips")
dinner = tips[tips["time"] == "Dinner"].copy()          # 176 dinner bills remain
dinner["tip_pct"] = dinner["tip"] / dinner["total_bill"] * 100
by_day = dinner.groupby("day", observed=True)["tip_pct"].mean().round(2)
print(by_day.sort_values(ascending=False))
# Sun 16.69 / Thur 15.97 / Fri 15.89 / Sat 15.32 -> Sunday diners tip best

# %%
# Exercise 7. Fares in a grid: pivot table plus a grouped bar chart, headless
import matplotlib
matplotlib.use("Agg")                                    # render to files only
import matplotlib.pyplot as plt

titanic = sns.load_dataset("titanic")
table = titanic.pivot_table(values="fare", index="pclass",
                            columns="sex", aggfunc="mean").round(2)
print(table)
# sex     female   male
# pclass
# 1       106.13  67.23
# 2        21.97  19.74
# 3        16.12  12.66
table.plot(kind="bar")                                   # one bar group per class
plt.ylabel("mean fare (1912 pounds)")
plt.title("Mean fare by class and sex")
plt.savefig("13_fare_by_class_sex.png")                  # writes the PNG beside this script
plt.close()

# %%
# Exercise 8. Engineer and compare: child and alone flags against survival
t = sns.load_dataset("titanic")[["survived", "age", "sibsp", "parch"]].copy()
t["age"] = t["age"].fillna(t["age"].median())            # median-fill, as in the 1.8 lab
t["family_size"] = t["sibsp"] + t["parch"] + 1
t["alone"] = t["family_size"] == 1
t["child"] = t["age"] < 16
print(t.groupby("child")["survived"].mean().round(3))    # False 0.363 / True 0.590
print(t.groupby("alone")["survived"].mean().round(3))    # False 0.506 / True 0.304
# the child flag separates survival slightly more strongly (23 vs 20 points)

# %%
# Exercise 9. A first fit: LinearRegression of tip on total_bill
from sklearn.linear_model import LinearRegression

X = tips[["total_bill"]]                                 # 2-D: (244, 1), as sklearn expects
y = tips["tip"]
model = LinearRegression().fit(X, y)
print(round(float(model.coef_[0]), 4), round(float(model.intercept_), 4))   # 0.105 0.9203
print(round(float(model.score(X, y)), 4))                                   # 0.4566
# each extra dollar on the bill adds ~10.5 cents of tip over a 92-cent base

# %% [markdown]
# ## Technique exercises (book 1.9, coding exercises 10-17)
# The eight build problems on the techniques of 1.2 through 1.5: decorators and
# context managers, linked lists, sorting, graph search, concurrency, profiling,
# merges, and Hugging Face pipelines. Same code as Appendix H.

# %%
# Exercise 10. Retry and restore: a decorator factory, then a class-based context manager
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

# %%
# Exercise 11. Linked list surgery: reverse two ways, detect a cycle, and find the recursion limit
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
print(sys.getrecursionlimit())                            # 1000 as a script, 3000 in a notebook (1.4.10)
try:
    reverse_recursive(build(list(range(10_000))))
except RecursionError as e:
    print("RecursionError:", e)                           # RecursionError: maximum recursion depth exceeded
# The recursive reversal needs one stack frame per node, so a 10,000-node list exceeds
# the limit, 1,000 frames by default and 3,000 in a notebook; the iterative version walks
# the same list in O(1) space.

# %%
# Exercise 12. Sorting, slow and fast: insertion sort, a random-pivot quicksort, and the worst case measured
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
# sorted   insertion_sort   0.002s      (the run saved in appendix_h_solutions_ch01.ipynb; times vary)
# sorted   quicksort        2.363s
# sorted   quicksort_random 0.026s
# shuffled insertion_sort   1.870s
# shuffled quicksort        0.018s
# shuffled quicksort_random 0.025s
# On sorted input the first pivot is always the minimum, every split is maximally lopsided,
# the recursion is n deep (5,000 frames, beyond the default limit of 1,000, hence the raised
# limit) and the comprehensions touch about n squared / 2 elements: the quadratic worst case
# of 1.4.1, about ninety times slower here than the random pivot, whose expected cost is O(n log n)
# whatever the input order. Insertion sort runs in O(n) on sorted input, because every key
# stops after one comparison, and in O(n squared) on the shuffled copy, about 6 million shifts.

# %%
# Exercise 13. Maze and components: breadth-first search for the shortest path, depth-first search for counting
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
# Exercise 14 (pool and loop) runs as its own script, 05_chapter1_concurrency_exercise.py
# in this folder: ProcessPoolExecutor workers must import count_primes from a file, and
# asyncio.run needs no event loop already running, which a notebook has. Run it with
# python 05_chapter1_concurrency_exercise.py and compare with the timings recorded there.

# %%
# Exercise 15. Profile the lab: pandas groupby against a dictionary loop, at two table sizes
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
# 891 rows: 46262 function calls (46142 primitive calls) in 0.069 seconds (the run saved in
# appendix_h_solutions_ch01.ipynb, whose temporary file name is shortened here to profile_fares.py)
#    ncalls  tottime  percall  cumtime  percall filename:lineno(function)
#        20    0.027    0.001    0.046    0.002 profile_fares.py:12(mean_fare_loop)
#        20    0.000    0.000    0.023    0.001 profile_fares.py:9(mean_fare_pandas)
# 89,100 rows: 3575081 function calls (3574956 primitive calls) in 4.879 seconds
#        20    2.979    0.149    4.726    0.236 profile_fares.py:12(mean_fare_loop)
#        20    0.000    0.000    0.124    0.006 profile_fares.py:9(mean_fare_pandas)
# The loop is the target at both sizes. On 891 rows it already costs more than the groupby
# (about 2.3 ms against 1.15 ms per call), and its time grows with the row count, about a
# hundredfold on the stacked table, while the groupby's cost grows only about fivefold (1.15 ms
# to 6.2 ms per call) because its work happens in compiled code. Replacing the loop with the
# groupby, which is the vectorization, is the only change worth making; the exact times vary
# by machine.

# %%
# Exercise 16. Join the class names: an inner merge, then a left merge with a missing lookup row
import pandas as pd
import seaborn as sns

titanic = sns.load_dataset("titanic")[["pclass", "fare"]]
lookup = pd.DataFrame({"pclass": [1, 2, 3], "class_name": ["First", "Second", "Third"]})
merged = titanic.merge(lookup, on="pclass")                     # inner join on the shared key
print(merged.shape)                                             # (891, 3)
print(merged.groupby("class_name")["fare"].mean().round(2))
# class_name
# First     84.15
# Second    20.66
# Third     13.68
# Name: fare, dtype: float64
partial = lookup[lookup["pclass"] != 3]                         # the lookup loses its third-class row
left = titanic.merge(partial, on="pclass", how="left")          # keep every passenger row
print(left.shape, left["class_name"].isna().sum())              # (891, 3) 491
inner = titanic.merge(partial, on="pclass")                     # the default inner join drops them
print(inner.shape)                                              # (400, 3)
# A left merge keeps every row of the left table and fills the columns of the right table
# with NaN where the key found no partner, so the 491 third-class passengers survive with
# an empty class_name; the default inner join keeps only keys present on both sides and
# silently drops those 491 rows, leaving 400.

# %%
# Exercise 17. Two pipelines, one disagreement: sentiment and zero-shot classification with small pinned models
from transformers import pipeline

notes = [
    "The loss dropped steadily for the first ten epochs.",
    "Validation accuracy collapsed after the learning rate warmup ended.",
    "The GPU ran out of memory on the first batch.",
    "Checkpoints are saving correctly every epoch.",
    "Why does the loss jump every time the data loader restarts?",
    "Please add a flag to resume training from the last checkpoint.",
    "The run finished early and the metrics look excellent.",
    "Training is twice as slow as yesterday on the same machine.",
    "Could we log the learning rate alongside the loss?",
    "The tokenizer silently truncated every input longer than 512 tokens.",
]

sentiment = pipeline("sentiment-analysis",
                     model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", device="cpu")
for note, result in zip(notes, sentiment(notes)):
    print(f"{result['label']:8s} {result['score']:.3f}  {note}")
# NEGATIVE 0.999  The loss dropped steadily for the first ten epochs.
# NEGATIVE 1.000  Validation accuracy collapsed after the learning rate warmup ended.
# NEGATIVE 1.000  The GPU ran out of memory on the first batch.
# POSITIVE 0.996  Checkpoints are saving correctly every epoch.
# NEGATIVE 0.999  Why does the loss jump every time the data loader restarts?
# NEGATIVE 0.950  Please add a flag to resume training from the last checkpoint.
# POSITIVE 1.000  The run finished early and the metrics look excellent.
# NEGATIVE 0.999  Training is twice as slow as yesterday on the same machine.
# NEGATIVE 1.000  Could we log the learning rate alongside the loss?
# NEGATIVE 0.999  The tokenizer silently truncated every input longer than 512 tokens.

topic = pipeline("zero-shot-classification",
                 model="MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33",
                 device="cpu")   # 142 MB, trained for zero-shot use
labels = ["bug", "feature request", "question"]
for note, result in zip(notes, topic(notes, candidate_labels=labels)):
    print(f"{result['labels'][0]:16s} {result['scores'][0]:.3f}  {note}")
# question         0.801  The loss dropped steadily for the first ten epochs.
# bug              0.754  Validation accuracy collapsed after the learning rate warmup ended.
# bug              0.969  The GPU ran out of memory on the first batch.
# bug              0.932  Checkpoints are saving correctly every epoch.
# question         0.783  Why does the loss jump every time the data loader restarts?
# feature request  0.988  Please add a flag to resume training from the last checkpoint.
# question         0.679  The run finished early and the metrics look excellent.
# bug              0.739  Training is twice as slow as yesterday on the same machine.
# question         0.879  Could we log the learning rate alongside the loss?
# bug              0.967  The tokenizer silently truncated every input longer than 512 tokens.
# The sentiment model has no label for "neither": it calls the
# two neutral questions NEGATIVE with scores near 1.0 and the polite feature request NEGATIVE
# at 0.950. The zero-shot model sorts the questions and the feature request correctly (0.988
# for the resume flag) but labels "Checkpoints are saving correctly every epoch" a bug at
# 0.932 although the sentence reports that something works, and it calls the good news about
# the loss a question. Each model is plainly wrong on at least one sentence, the usual reason
# to fine-tune on the texts a project actually sees rather than trust a borrowed label set.
