"""The pretraining corpus: an unbounded, seeded stream of tasks from the training steps.

A corpus task is drawn exactly as an `in_dist` benchmark task is (`benchmark.py`): a palette, k + n_test grids from
the prior, and a rule of the given depth whose every step is a training step, drawn on all of them. Only the seeding
differs, and a draw that would contain any grid of the benchmark is redrawn, so the corpus and the benchmark share no
grid. Held-out steps never appear.

Tasks are addressed by (seed, shard, depth, index), each drawn from its own generator, so any slice of the corpus can
be produced independently, in any order, on any number of processes, and is the same every time.

`extra` asks for that many more input/output pairs of the same rule (`generator.fresh_grids`), appended to the test
pairs: more views of one rule, if your training wants them.
"""

import json
import os

import numpy as np

from . import benchmark as B
from . import generator as G

_TAG = 606
_EXCLUDE = {}


def benchmark_grids():
    """The benchmark's grids, built once a process (~15 s)."""
    if "keys" not in _EXCLUDE:
        _EXCLUDE["keys"] = B.grid_keys(B.build(0))
    return _EXCLUDE["keys"]


def _hits(t, exclude):
    return any(np.ascontiguousarray(g, np.int8).tobytes() in exclude
               for k in ("demo_in", "demo_out", "test_in", "test_out") for g in t[k])


def corpus_task(seed, shard, depth, i, k=3, n_test=1, extra=0, exclude=None):
    """One corpus task. exclude: grid keys to avoid (default: the benchmark's; pass an empty set to skip the check).
    -> a `generator.sample_task` dict with id, depth."""
    exclude = benchmark_grids() if exclude is None else exclude
    rng = np.random.default_rng([int(seed), _TAG, int(shard), int(depth), int(i)])
    while True:
        t = G.sample_task(rng, allowed_steps=[B.TRAINING_SET] * depth, k=k, n_test=n_test)
        if extra:
            xi, xo, _ = G.fresh_grids(rng, t["rule"], t["palette"], extra)
            t["test_in"] = np.concatenate([t["test_in"], xi])
            t["test_out"] = np.concatenate([t["test_out"], xo])
        if not _hits(t, exclude):
            break
    t.update(id=f"s{seed}-{shard:06d}-d{depth}-{i:06d}", depth=depth)
    return t


def shard(seed, shard_ix, per_depth, **kw):
    """One shard: per_depth {depth: count} tasks, in depth order."""
    return [corpus_task(seed, shard_ix, d, i, **kw) for d in sorted(per_depth) for i in range(per_depth[d])]


def stream(seed=0, depths=(1, 2, 3, 4), start_shard=0, per_shard=256, **kw):
    """Shard after shard, forever: per_shard tasks of each depth a shard."""
    s = start_shard
    while True:
        for t in shard(seed, s, {d: per_shard for d in depths}, **kw):
            yield t
        s += 1


def to_record(t):
    """A corpus task as one JSON-able record: ARC's train/test pairs, plus the rule."""
    r = B.to_arc(t)
    r["id"] = t["id"]
    r["depth"] = t["depth"]
    r["rule"] = [{"step": int(b), "code": G.step_code(b), "words": G.step_words(b)} for b in t["rule"]]
    return r


def write_shard(job):
    """(out dir, seed, shard, per_depth, extra) -> one shard as JSONL at out/shard_<shard>.jsonl, one task a line
    (`to_record`); skipped when the file exists. -> (path, tasks written)."""
    out, seed, s, per_depth, extra = job
    path = os.path.join(out, f"shard_{s:06d}.jsonl")
    if os.path.exists(path):
        return path, 0
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "w") as fh:
        for t in shard(seed, s, per_depth, extra=extra):
            fh.write(json.dumps(to_record(t), separators=(",", ":")) + "\n")
    os.replace(tmp, path)
    return path, sum(per_depth.values())
