"""Scoring a submission on the benchmark, beside the Bayes floor, and three baselines.

A submission is ARC Prize's Kaggle form: {task id: [{"attempt_1": grid, "attempt_2": grid, ...}]}, one dict per test
output (every benchmark task has one). A task counts as solved within k attempts when any of attempt_1 .. attempt_k
equals the test output exactly (pass@k; the ARC Prize scores pass@2). Results are reported by split and depth, with
`bayes@k` beside them: the expected pass@k of the best possible solver, the one that knows the generator and the
rule's depth (1 - the task's Bayes floor at k attempts, from metadata.json).

The baselines:
  identity    answers the test input unchanged. Every step changes every grid it acts on, so this is nearly always
              wrong; it is the zero line.
  colour_map  learns one colour-to-colour substitution from the demonstrations and applies it to the test input (the
              test input itself when the demonstrations admit none).
  one_step    searches the generator's own 251 steps for those that act on all four inputs and reproduce all three
              demonstrations, and answers the test outputs they produce, the output most steps agree on first. At
              depth 1 this is the Bayes solver exactly; deeper, it shows how much a single step explains.
"""

import numpy as np

from . import benchmark as B
from . import generator as G

ATTEMPTS = (1, 2, 4)


def _eq(a, b):
    try:
        return np.array_equal(np.asarray(a), np.asarray(b))
    except Exception:
        return False


def score(submission, solutions=None, metadata=None, attempts=ATTEMPTS):
    """-> {"by_set": {(split, depth): {n, missing, pass@k..., bayes@k...}}}."""
    if solutions is None or metadata is None:
        _, solutions, metadata = B.load()
    meta = metadata["tasks"]
    rows = {}
    for tid, sols in solutions.items():
        m = meta[tid]
        r = rows.setdefault((m["split"], m["depth"]), {"n": 0, "missing": 0, **{f"pass@{k}": 0 for k in attempts},
                                                         **{f"bayes@{k}": 0.0 for k in attempts}})
        r["n"] += 1
        for k in attempts:
            e = m["floor"].get(f"err@{k}")
            r[f"bayes@{k}"] += (1.0 - e) if e is not None else float("nan")
        sub = submission.get(tid)
        if not sub:
            r["missing"] += 1
            continue
        for k in attempts:
            ok = all(any(_eq(a.get(f"attempt_{j}"), sol) for j in range(1, k + 1)) for a, sol in zip(sub, sols))
            r[f"pass@{k}"] += int(ok and len(sub) >= len(sols))
    by_set = {}
    for key, r in sorted(rows.items(), key=lambda x: (B.SPLITS.index(x[0][0]), x[0][1])):
        n = r["n"]
        by_set[key] = {"n": n, "missing": r["missing"], **{k: v / n for k, v in r.items() if "@" in k}}
    return {"by_set": by_set}


def table(results, attempts=(1, 2)):
    """A markdown table of `score` results."""
    hdr = ["split", "depth", "tasks"] + [f"pass@{k}" for k in attempts] + [f"bayes@{k}" for k in attempts]
    out = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for (s, d), r in results["by_set"].items():
        out.append("| " + " | ".join([s, str(d), str(r["n"])] + [f"{r[f'pass@{k}']:.3f}" for k in attempts]
                                     + [f"{r[f'bayes@{k}']:.3f}" for k in attempts]) + " |")
    return "\n".join(out)


# ------------------------------------------------------------------------------------------------------------------ #
# baselines
# ------------------------------------------------------------------------------------------------------------------ #

def _arr(g):
    return np.asarray(g, np.int8)


def identity(task):
    return [{"attempt_1": p["input"]} for p in task["test"]]


def colour_map(task):
    m = {}
    for p in task["train"]:
        a, b = _arr(p["input"]), _arr(p["output"])
        if a.shape != b.shape:
            return identity(task)
        for x, y in zip(a.reshape(-1).tolist(), b.reshape(-1).tolist()):
            if m.setdefault(x, y) != y:
                return identity(task)
    lut = np.arange(10, dtype=np.int8)
    for x, y in m.items():
        lut[x] = y
    return [{"attempt_1": lut[_arr(p["input"])].astype(int).tolist()} for p in task["test"]]


def one_step(task, k=4):
    ins = [_arr(p["input"]) for p in task["train"]] + [_arr(p["input"]) for p in task["test"]]
    outs = [_arr(p["output"]) for p in task["train"]]
    n = len(outs)
    votes = {}
    for bi, res in G.admissible_steps(ins):
        if all(np.array_equal(res[j], outs[j]) for j in range(n)):
            key = tuple(r.tobytes() for r in res[n:])
            v = votes.setdefault(key, [0, res[n:]])
            v[0] += 1
    if not votes:
        return identity(task)
    ranked = sorted(votes.values(), key=lambda v: (-v[0], tuple(r.tobytes() for r in v[1])))[:k]
    return [{f"attempt_{a + 1}": v[1][j].astype(int).tolist() for a, v in enumerate(ranked)}
            for j in range(len(task["test"]))]


BASELINES = {"identity": identity, "colour_map": colour_map, "one_step": one_step}


def run_baseline(name, challenges=None):
    if challenges is None:
        challenges, _, _ = B.load()
    f = BASELINES[name]
    return {tid: f(t) for tid, t in challenges.items()}
