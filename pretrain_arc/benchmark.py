"""The held-out steps, the fixed benchmark, and the files it ships as.

HELD-OUT STEPS. Step (selector i, operator j) of `generator.STEPS` (i its index in SELECTORS, j in OPERATORS) is held
out when (i + j) % 5 == 0: 50 of the 251 steps. Every selector keeps 15-17 training steps and has 3-4 held out, every
operator keeps 10-11 and has 2-3 held out, and every selector kind and operator family is on both sides. The corpus
(`corpus.py`) never uses a held-out step.

THE BENCHMARK (`build`): 2,560 tasks in two splits, fixed by seed 0.
  in_dist   rules whose every step is a training step: the corpus's own distribution, on grids it never emits.
            512 tasks at depth 1, 256 at each of depths 2-4.
  held_out  rules with exactly one held-out step, at a uniformly drawn position, every other step a training step.
            Depth 1 is then a held-out step alone. Same counts.
Each task has three demonstration pairs and one test pair. Within a split, a position's step is uniform among the
admissible steps it permits (`generator.sample_rule_on`). The tasks are fingerprinted (`fingerprint`), and the shipped
files hold each task's Bayes floor (`floors.py`), computed under the generator's prior over all 251 steps: the floor's
solver does not know which steps are held out.

THE FILES (`benchmark/` at the repo root), in the ARC Prize's Kaggle layout:
  challenges.json  {task id: {"train": [{"input", "output"}] x 3, "test": [{"input"}]}}
  solutions.json   {task id: [test output]}
  metadata.json    {task id: {split, depth, rule (codes and words), held-out position, floor}}
Task ids are `<split>-d<depth>-<index>`, e.g. `held_out-d3-0042`.
"""

import hashlib
import json
import os

import numpy as np

from . import generator as G

SEL_IX = {s: i for i, s in enumerate(G.SELECTORS)}
OP_IX = {o: j for j, o in enumerate(G.OPERATORS)}


def is_held_out(b):
    s, o = G.STEPS[b]
    return (SEL_IX[s] + OP_IX[o]) % 5 == 0


HELD_OUT = tuple(b for b in range(G.N_STEPS) if is_held_out(b))
TRAINING = tuple(b for b in range(G.N_STEPS) if not is_held_out(b))
HELD_OUT_SET, TRAINING_SET = frozenset(HELD_OUT), frozenset(TRAINING)

SPLITS = ("in_dist", "held_out")                       # this order is part of the seeding
N_PER_DEPTH = {1: 512, 2: 256, 3: 256, 4: 256}
FINGERPRINT = "5d9f992ce03cdced"                       # build(0)
_TAG = 505
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES_DIR = os.path.join(REPO, "benchmark")


def task_id(split, depth, i):
    return f"{split}-d{depth}-{i:04d}"


def benchmark_task(seed, split, depth, i):
    """One benchmark task, drawn from its own generator seeded by (seed, split, depth, i)."""
    rng = np.random.default_rng([int(seed), _TAG, SPLITS.index(split), int(depth), int(i)])
    if split == "in_dist":
        steps, j = [TRAINING_SET] * depth, None
    else:
        j = int(rng.integers(depth))
        steps = [HELD_OUT_SET if t == j else TRAINING_SET for t in range(depth)]
    t = G.sample_task(rng, allowed_steps=steps, k=3, n_test=1)
    t.update(id=task_id(split, depth, i), split=split, depth=depth, held_position=j,
             held=[int(b in HELD_OUT_SET) for b in t["rule"]])
    return t


def build(seed=0, n_per_depth=None):
    """The benchmark: {split: {depth: [task]}}. About 15 s on one core."""
    n = dict(N_PER_DEPTH, **(n_per_depth or {}))
    return {s: {d: [benchmark_task(seed, s, d, i) for i in range(n[d])] for d in sorted(n)} for s in SPLITS}


def tasks(bench):
    """The benchmark's tasks in a flat list, in split, depth, index order."""
    return [t for s in SPLITS for d in sorted(bench[s]) for t in bench[s][d]]


def fingerprint(bench):
    h = hashlib.sha1()
    for t in tasks(bench):
        h.update(np.asarray(t["rule"], np.int64).tobytes())
        for k in ("demo_in", "demo_out", "test_in", "test_out"):
            h.update(np.ascontiguousarray(t[k]).tobytes())
    return h.hexdigest()[:16]


def grid_keys(bench):
    """Every grid of every benchmark task, as bytes (the corpus never emits one)."""
    return {np.ascontiguousarray(g, np.int8).tobytes()
            for t in tasks(bench) for k in ("demo_in", "demo_out", "test_in", "test_out") for g in t[k]}


# ------------------------------------------------------------------------------------------------------------------ #
# the files
# ------------------------------------------------------------------------------------------------------------------ #

def grid_list(g):
    return np.asarray(g).astype(int).tolist()


def to_arc(t, with_test_output=True):
    """A task dict in ARC's JSON form: {"train": [{"input", "output"}], "test": [{"input"[, "output"]}]}."""
    out = {"train": [{"input": grid_list(a), "output": grid_list(b)} for a, b in zip(t["demo_in"], t["demo_out"])],
           "test": [{"input": grid_list(a)} for a in t["test_in"]]}
    if with_test_output:
        for p, b in zip(out["test"], t["test_out"]):
            p["output"] = grid_list(b)
    return out


def rule_record(t):
    return [{"step": int(b), "code": G.step_code(b), "words": G.step_words(b), "held_out": bool(h)}
            for b, h in zip(t["rule"], t["held"])]


def write_files(bench, floors, out_dir=FILES_DIR):
    """The three files. floors: {task id: `floors.task_floor` dict}."""
    os.makedirs(out_dir, exist_ok=True)
    ch, so, meta = {}, {}, {}
    for t in tasks(bench):
        arc = to_arc(t, with_test_output=False)
        ch[t["id"]] = arc
        so[t["id"]] = [grid_list(b) for b in t["test_out"]]
        f = floors[t["id"]]
        meta[t["id"]] = {"split": t["split"], "depth": t["depth"], "held_position": t["held_position"],
                         "rule": rule_record(t), "palette": list(t["palette"]), "n_admissible": list(t["n_adm"]),
                         "floor": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in f.items()}}
    meta = {"fingerprint": fingerprint(bench), "seed": 0, "tasks": meta}
    for name, obj in (("challenges", ch), ("solutions", so), ("metadata", meta)):
        with open(os.path.join(out_dir, f"{name}.json"), "w") as fh:
            fh.write(_one_task_a_line(obj))


def _one_task_a_line(obj):
    """JSON with one task a line, so a diff of the files reads task by task."""
    if "tasks" in obj:
        head = ",".join(f"{json.dumps(k)}:{json.dumps(v)}" for k, v in obj.items() if k != "tasks")
        return "{" + head + ',"tasks":' + _one_task_a_line(obj["tasks"]).rstrip("\n") + "}\n"
    rows = [f"{json.dumps(k)}:{json.dumps(v, separators=(',', ':'))}" for k, v in obj.items()]
    return "{\n" + ",\n".join(rows) + "\n}\n"


def load(files_dir=FILES_DIR):
    """The shipped benchmark. -> (challenges, solutions, metadata), as in the files."""
    if not os.path.exists(os.path.join(files_dir, "metadata.json")):
        raise FileNotFoundError(
            f"the benchmark files are not at {files_dir}. They live in the repo's benchmark/ folder, beside the "
            "package: clone the repo and `pip install -e .`, or pass files_dir. (`build(0)` regenerates the tasks "
            "without the files, but not their floors.)")
    out = []
    for name in ("challenges", "solutions", "metadata"):
        with open(os.path.join(files_dir, f"{name}.json")) as fh:
            out.append(json.load(fh))
    return tuple(out)
