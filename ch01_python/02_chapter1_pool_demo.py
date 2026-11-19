"""Chapter 1, section 1.3.3: the multiprocessing.Pool example as a standalone script.

Run it with:  python 02_chapter1_pool_demo.py
It lives in its own file because spawned worker processes re-import the file
that defines `square`, which a Jupyter kernel cannot provide on Windows.
The code below is the book's block, verbatim.
"""
from multiprocessing import Pool

def square(x):
    return x * x

if __name__ == "__main__":
    with Pool(4) as pool:               # four separate processes, four cores
        results = pool.map(square, range(10))
    print(results)                      # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
