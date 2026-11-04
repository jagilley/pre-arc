"""Build the benchmark, compute every task's Bayes floor, and write `benchmark/{challenges,solutions,metadata}.json`.

    python scripts/export_benchmark.py [--workers 8] [--cache floors_cache.json]

The floors dominate the cost: depth 4 takes tens of seconds a task, so the whole set is about 4-7 CPU-hours (an hour
on 7 workers of a laptop). Finished floors go to the cache as they arrive, so an interrupted run resumes where it
stopped.
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pretrain_arc import benchmark as B  # noqa: E402
from pretrain_arc import floors as F  # noqa: E402


def _floor(key):
    split, depth, i = key
    return B.task_id(split, depth, i), F.task_floor(B.benchmark_task(0, split, depth, i))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--cache", default="floors_cache.json")
    ap.add_argument("--out", default=B.FILES_DIR)
    a = ap.parse_args()

    bench = B.build(0)
    fp = B.fingerprint(bench)
    assert fp == B.FINGERPRINT, (fp, B.FINGERPRINT)
    cache = json.load(open(a.cache)) if os.path.exists(a.cache) else {}
    todo = [(t["split"], t["depth"], int(t["id"][-4:])) for t in B.tasks(bench) if t["id"] not in cache]
    todo.sort(key=lambda k: -k[1])                     # the deepest (slowest) first
    print(f"{len(cache)} floors cached, {len(todo)} to compute on {a.workers} workers", flush=True)
    t0, last = time.time(), time.time()
    with Pool(a.workers) as pool:
        for n, (tid, f) in enumerate(pool.imap_unordered(_floor, todo, chunksize=1), 1):
            cache[tid] = f
            if time.time() - last > 30 or n == len(todo):
                with open(a.cache + ".tmp", "w") as fh:
                    json.dump(cache, fh)
                os.replace(a.cache + ".tmp", a.cache)
                last = time.time()
                print(f"  {n}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
    B.write_files(bench, cache, a.out)
    print(f"wrote {a.out} (fingerprint {fp})")


if __name__ == "__main__":
    main()
