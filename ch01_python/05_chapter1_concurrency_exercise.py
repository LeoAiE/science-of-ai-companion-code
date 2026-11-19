"""Chapter 1.9, coding exercise 14: pool and loop.

Run this one as a plain script (python 05_chapter1_concurrency_exercise.py), not
in a notebook: ProcessPoolExecutor workers must import count_primes from a file,
and asyncio.run needs no event loop to be running already. Timings are from the
book's reference machine (an Apple M1 Max with 10 logical cores, Python 3.13.14) and
vary from run to run and by machine.
"""
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
