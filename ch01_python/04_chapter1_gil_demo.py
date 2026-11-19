"""Chapter 1, section 1.3.3: the CPU-bound GIL experiment as a standalone script.

Run it with:  python 04_chapter1_gil_demo.py
It lives in its own file because ProcessPoolExecutor workers must import the
function they run, which a Jupyter kernel cannot provide on Windows.
The code below is the book's block, verbatim.
"""
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def crunch(n):
    return sum(i * i for i in range(n))     # pure Python arithmetic: all computing, no waiting

def best_of_three(run):                     # the best of several repeats habit of 1.3.4
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        run()
        best = min(best, time.perf_counter() - t0)
    return best

if __name__ == "__main__":                  # the guard 1.7.6 explains: workers import this file again
    tasks = [5_000_000] * 4                 # four equal jobs, about 0.23 seconds each on the reference machine
    t_seq = best_of_three(lambda: [crunch(n) for n in tasks])
    with ThreadPoolExecutor(max_workers=4) as pool:
        t_thr = best_of_three(lambda: list(pool.map(crunch, tasks)))
    with ProcessPoolExecutor(max_workers=4) as pool:
        t_proc = best_of_three(lambda: list(pool.map(crunch, tasks)))
    print(f"sequential {t_seq:.2f}s, threads {t_thr:.2f}s, processes {t_proc:.2f}s")
# sequential 0.89s, threads 0.86s, processes 0.22s   (one run on the reference machine)
