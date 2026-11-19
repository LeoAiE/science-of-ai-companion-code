"""Chapter 1, section 1.3.3: the asyncio example as a standalone script.

Run it with:  python 03_chapter1_async_demo.py
It lives in its own file because a Jupyter kernel already runs an event loop,
so asyncio.run() raises RuntimeError there (in a notebook, write `await main()`
instead of the last line). The code below is the book's block, verbatim.
"""
import asyncio, time

async def fetch(url, delay):
    await asyncio.sleep(delay)          # stands in for a network round trip
    return f"{url} done"

async def main():
    t0 = time.perf_counter()
    # run them concurrently: start all three, then await them together
    results = await asyncio.gather(fetch("a", 0.2), fetch("b", 0.2), fetch("c", 0.2))
    print(results, f"{time.perf_counter() - t0:.2f}s")

asyncio.run(main())
# ['a done', 'b done', 'c done'] 0.20s
