# %% [markdown]
# # Appendix C: Python, NumPy, and pandas cheat sheets (runnable)
#
# Compact, runnable reminders of the idioms used throughout the book. The three code cells of this
# notebook are the three code blocks of Appendix C, line for line, so the outputs saved here are the
# lines the appendix reports. The values are tiny literals chosen to show each idiom, not a
# dataset; the pandas expressions return Series and a DataFrame, which print with their index and,
# for a Series, a Name and dtype footer.

# %%
xs = [10, 20, 30, 40, 50]
print(xs[1:4])                           # [20, 30, 40]        slice [start:stop]
print(xs[::-1])                          # [50, 40, 30, 20, 10] reverse with step -1
print([x*x for x in xs if x > 20])       # [900, 1600, 2500]   list comprehension + filter
print(list(enumerate(["a", "b"])))       # [(0, 'a'), (1, 'b')]  index + value
print(list(zip([1, 2], ["x", "y"])))     # [(1, 'x'), (2, 'y')]  pair two sequences
print({k: v for k, v in zip("ab", [1, 2])})  # {'a': 1, 'b': 2}  dict comprehension
print(sorted([(2,"b"), (1,"a")], key=lambda t: t[0]))  # [(1, 'a'), (2, 'b')]  sort by a key

# %%
import numpy as np
a = np.array([[1, 2, 3], [4, 5, 6]])
print(a.shape)                           # (2, 3)
print(a * 2)                             # elementwise, no loop
print(a + np.array([10, 20, 30]))        # broadcasting: the row is added to each row of a
print(a.sum(axis=0))                     # [5 7 9]   column sums
print(a[a > 3])                          # [4 5 6]   boolean mask
print(a @ a.T)                           # matrix multiply: (2, 3) @ (3, 2) gives (2, 2)
print(a.mean(), a.std())                 # 3.5 1.707825127659933   aggregations

# %%
import pandas as pd
df = pd.DataFrame({"name": ["a","b","c","d"],
                   "group": ["x","x","y","y"],
                   "value": [10, 20, 30, 40]})
print(df["value"])                       # the value column as a Series
print(df[df.value > 15]["name"])         # boolean row filter: the names b, c, d
print(df.groupby("group")["value"].mean())  # split-apply-combine: x 15.0, y 35.0
print(df["value"] * 2)                   # vectorized arithmetic, returns a Series
other = pd.DataFrame({"group": ["x", "y"],
                      "label": ["low", "high"]})
print(df.merge(other, on="group"))       # join on a key: 4 rows gain a label column,
                                         # a/b get 'low' (group x), c/d get 'high' (group y)
