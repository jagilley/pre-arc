"""The package reproduces the published benchmark exactly, the files agree with it, and the corpus keeps its promises.

    pip install -e ".[test]" && pytest -q        (about a minute)
"""

import numpy as np
import pytest

from pretrain_arc import benchmark as B
from pretrain_arc import corpus as C
from pretrain_arc import floors as F
from pretrain_arc import generator as G
from pretrain_arc import score as S


@pytest.fixture(scope="module")
def bench():
    return B.build(0)


@pytest.fixture(scope="module")
def files():
    return B.load()


def test_benchmark_fingerprint(bench):
    assert B.fingerprint(bench) == B.FINGERPRINT


def test_files_match_the_generator(bench, files):
    ch, so, meta = files
    assert meta["fingerprint"] == B.FINGERPRINT
    ts = B.tasks(bench)
    assert len(ts) == len(ch) == len(so) == len(meta["tasks"]) == 2560
    for t in ts:
        assert B.to_arc(t, with_test_output=False) == ch[t["id"]]
        assert so[t["id"]] == [B.grid_list(g) for g in t["test_out"]]
        m = meta["tasks"][t["id"]]
        assert [s["step"] for s in m["rule"]] == list(t["rule"])
        assert [s["held_out"] for s in m["rule"]] == [bool(h) for h in t["held"]]


def test_splits(bench):
    assert len(B.HELD_OUT) == 50 and len(B.TRAINING) == 201
    sel_sides = {s: set() for s in G.SELECTORS}
    op_sides = {o: set() for o in G.OPERATORS}
    for b, (s, o) in enumerate(G.STEPS):
        sel_sides[s].add(b in B.HELD_OUT_SET)
        op_sides[o].add(b in B.HELD_OUT_SET)
    assert all(v == {True, False} for v in sel_sides.values())
    assert all(v == {True, False} for v in op_sides.values())
    for t in B.tasks(bench):
        n_held = sum(b in B.HELD_OUT_SET for b in t["rule"])
        assert n_held == (1 if t["split"] == "held_out" else 0)


def test_rules_reproduce_every_pair(bench):
    for t in B.tasks(bench)[::7]:
        for a, b in zip(list(t["demo_in"]) + list(t["test_in"]), list(t["demo_out"]) + list(t["test_out"])):
            assert np.array_equal(G.apply_rule(a, t["rule"]), b)


def test_floors_recompute(bench, files):
    meta = files[2]["tasks"]
    sample = [t for t in B.tasks(bench) if t["depth"] <= 2][::40]
    for t in sample:
        f = F.task_floor(t)
        for k in ("err@1", "err@2", "err@4"):
            assert f[k] == pytest.approx(meta[t["id"]]["floor"][k], abs=1e-6)


def test_corpus(bench):
    keys = B.grid_keys(bench)
    a = C.corpus_task(0, 3, 2, 7, exclude=keys)
    b = C.corpus_task(0, 3, 2, 7, exclude=keys)
    for k in ("demo_in", "demo_out", "test_in", "test_out"):
        assert np.array_equal(a[k], b[k])
    for t in C.shard(0, 0, {1: 20, 2: 20, 3: 10, 4: 10}, exclude=keys):
        assert all(s in B.TRAINING_SET for s in t["rule"])
        assert not any(np.ascontiguousarray(g, np.int8).tobytes() in keys
                       for k in ("demo_in", "demo_out", "test_in", "test_out") for g in t[k])


def test_score(files):
    ch, so, meta = files
    perfect = {tid: [{"attempt_1": s} for s in sols] for tid, sols in so.items()}
    r = S.score(perfect, so, meta)
    assert all(v["pass@1"] == 1.0 and v["missing"] == 0 for v in r["by_set"].values())
    r = S.score({}, so, meta)
    assert all(v["pass@2"] == 0.0 and v["missing"] == v["n"] for v in r["by_set"].values())
