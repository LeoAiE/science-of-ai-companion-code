"""Section 16.11: how far the ratio printed by the attention timing block moves from run to run.
Run from this folder. It starts twenty fresh Python processes, one after another, so it takes a minute or two.
"""
# %% [markdown]
# # 16.11 Twenty fresh runs of the attention timing block
#
# The timing block of Section 16.11 times one full NumPy attention at n = 1024 and n = 2048, keeps the
# fastest of 20 calls at each length, and prints the second time divided by the first. One run is one
# sample. This script runs the same block, copied verbatim from the book, in twenty fresh Python processes,
# one after another, each with its matrix multiplications limited to two threads (the OpenMP, OpenBLAS, and
# MKL thread counts all set to 2), and prints every run's two times and the ratio the block printed. The
# second cell summarizes them: the minimum, maximum, and median ratio, how many ratios lie within 0.5 of
# the exact factor of 4 by which the score matrix grows, and, for each run outside that band, how far
# each of its two times sits from the median time at that length.

# %%
import os, re, statistics, subprocess, sys

TIMING_BLOCK = '''
import time
import numpy as np
rng = np.random.default_rng(0)
def softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)
d = 64
prev_ms = None
for n in [1024, 2048]:
    Q, K, V = (rng.normal(size=(n, d)).astype(np.float32) for _ in range(3))
    # keep the fastest of 20 calls, the best of several repeats habit of 1.3.4
    best = float("inf")
    for _ in range(20):
        t0 = time.perf_counter()
        out = softmax(Q @ K.T / np.sqrt(d)) @ V     # one full attention
        best = min(best, time.perf_counter() - t0)
    ms = best * 1000
    # the measured time divided by the time at the previous length
    ratio = "" if prev_ms is None else f"   ({ms / prev_ms:.1f}x)"
    print(f"n = {n}: one attention takes {ms:5.1f} ms{ratio}")
    prev_ms = ms
'''

env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
probe = subprocess.run([sys.executable, "-c", "import numpy, threadpoolctl; print([(d['internal_api'], "
                        "d['num_threads']) for d in threadpoolctl.threadpool_info()])"],
                       env=env, capture_output=True, text=True, check=True)
print("BLAS library and thread count in each fresh process:", probe.stdout.strip())
runs = []
for k in range(1, 21):
    out = subprocess.run([sys.executable, "-c", TIMING_BLOCK], env=env, capture_output=True,
                         text=True, check=True).stdout
    t_short, t_long = (float(x) for x in re.findall(r"takes\s+([\d.]+) ms", out))
    ratio = float(re.search(r"\(([\d.]+)x\)", out).group(1))   # the ratio the block itself printed
    runs.append((t_short, t_long, ratio))
    print(f"run {k:2d}: n = 1024 {t_short:5.1f} ms, n = 2048 {t_long:6.1f} ms, ratio {ratio:.1f}")

# %%
ratios = [r for _, _, r in runs]
lo, hi = 3.5, 4.5                                     # within 0.5 of the score matrix's exact factor of 4
inside = sum(lo <= r <= hi for r in ratios)
med_short = statistics.median(t for t, _, _ in runs)
med_long = statistics.median(t for _, t, _ in runs)
print(f"ratios: min {min(ratios):.1f}, max {max(ratios):.1f}, median {statistics.median(ratios):.2f}")
print(f"{inside} of {len(ratios)} ratios lie between {lo} and {hi}")
print(f"median times: {med_short:.1f} ms at n = 1024 and {med_long:.1f} ms at n = 2048, "
      f"a ratio of {med_long / med_short:.2f}")
for k, (t_short, t_long, r) in enumerate(runs, 1):
    if not lo <= r <= hi:
        print(f"run {k:2d} (ratio {r:.1f}): n = 1024 took {t_short / med_short:.2f} times its median, "
              f"n = 2048 took {t_long / med_long:.2f} times its median")
