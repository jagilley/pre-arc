"""Build the viewer's data: `site/data/<set>.json`, one file a set, fetched by `site/index.html` when the set is opened.

    python scripts/build_site.py [--arc-dir DIR] [--solvers FILE]
    cd site && python -m http.server                      (then open http://localhost:8000)

Ten sets:
  in_dist-d1 .. in_dist-d4, held_out-d1 .. held_out-d4
      The pre-ARC benchmark, rebuilt from the package (`benchmark.build(0)`, checked against its fingerprint and
      against the shipped `benchmark/challenges.json` and `solutions.json`). Per task: the pairs, the rule (each step in
      words, its code, whether it is a held-out step), the palette, the generator's own trace on every input (the
      object each step selects and the grid after it), and the Bayes floor from `benchmark/metadata.json`.
  arc1-training, arc1-evaluation
      ARC-AGI-1's 400 + 400 public tasks (github.com/fchollet/ARC-AGI, Apache-2.0), and for training tasks Michael
      Hodel's hand-written solver from arc-dsl (github.com/michaelhodel/arc-dsl, MIT).

By default both ARC sources are downloaded once, at pinned commits, into `.cache/` (git-ignored) and verified: the 800
task files against a digest of their bytes, `solvers.py` against its git blob hash. Their two licence texts, also
checked by blob hash, are written to `site/data/licenses/`, since the site redistributes what they cover. `--arc-dir` (a directory holding
`training/` and `evaluation/`) and `--solvers` point at local copies instead; they are verified the same way.

numpy only, like the package. Under a minute, most of it rebuilding the benchmark.
"""

import argparse
import ast
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tarfile
import urllib.request

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from pre_arc import benchmark as B  # noqa: E402
from pre_arc import generator as G  # noqa: E402

CACHE = os.path.join(REPO, ".cache")
OUT = os.path.join(REPO, "site", "data")

ARC_COMMIT = "399030444e0ab0cc8b4e199870fb20b863846f34"
ARC_URL = f"https://codeload.github.com/fchollet/ARC-AGI/tar.gz/{ARC_COMMIT}"
ARC_DIGEST = "894cabdc0e0c933d323e5ef8d9ba3e9a0c33cb42fd76c87debc105b64b0c0d99"   # the 800 task files (`arc_digest`)
DSL_COMMIT = "635de4902a5fb4e376f27333feaa396d3f5dfdcb"
DSL_URL = f"https://raw.githubusercontent.com/michaelhodel/arc-dsl/{DSL_COMMIT}/solvers.py"
DSL_BLOB = "f19052fcd9dde122ec793c46f44a44b2916d266b"   # git blob hash of solvers.py at DSL_COMMIT
LICENSES = {  # shipped beside the data they cover (site/data/licenses/), each checked against its git blob hash
    "ARC-AGI-LICENSE.txt": (f"https://raw.githubusercontent.com/fchollet/ARC-AGI/{ARC_COMMIT}/LICENSE",
                            "68ad4d4198faa02290b125d87ab94e2c23b5227f"),
    "arc-dsl-LICENSE.txt": (f"https://raw.githubusercontent.com/michaelhodel/arc-dsl/{DSL_COMMIT}/LICENSE",
                            "c7a5782dcde17a8004ce9dcf8fdd58336a565386"),
}
ARC_SPLITS = ("training", "evaluation")
ARC_NAME = re.compile(r"^[0-9a-f]{8}\.json$")


def gstr(g):
    """A grid as text: rows joined by commas, one digit a cell."""
    return ",".join("".join(str(int(x)) for x in row) for row in np.asarray(g))


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "pre-arc-build-site"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


# ------------------------------------------------------------------------------------------------------------------ #
# pre-ARC
# ------------------------------------------------------------------------------------------------------------------ #

def pretrain_task(t, meta):
    rule = [G.STEPS[b] for b in t["rule"]]
    recorded = [s["step"] for s in meta["rule"]]
    assert recorded == [int(b) for b in t["rule"]], (t["id"], recorded, t["rule"])
    assert [s["held_out"] for s in meta["rule"]] == [bool(h) for h in t["held"]], t["id"]
    traces = []
    for g in list(t["demo_in"]) + list(t["test_in"]):
        _, tr = G.apply_rule(g, t["rule"], trace=True)
        traces.append([{"sel": sorted(int(c) for c in obj[1]) if obj is not None else [], "g": gstr(gg)}
                       for obj, gg in tr])
    f = meta["floor"]
    return {
        "id": t["id"].rsplit("-", 1)[1],
        "task_id": t["id"],
        "train": [{"input": gstr(a), "output": gstr(b)} for a, b in zip(t["demo_in"], t["demo_out"])],
        "test": [{"input": gstr(a), "output": gstr(b)} for a, b in zip(t["test_in"], t["test_out"])],
        "rule": [{"sel": G.selector_words(s), "op": G.operator_words(o), "code": G.step_code(b), "held": int(h)}
                 for (s, o), b, h in zip(rule, t["rule"], t["held"])],
        "held_position": t["held_position"],
        "palette": [int(c) for c in t["palette"]],
        "trace": traces,
        "floor": {k: f.get(k) for k in ("err@1", "err@2", "err@4", "n_outputs", "map_is_true")},
    }


def build_pretrain():
    bench = B.build(0)
    fp = B.fingerprint(bench)
    assert fp == B.FINGERPRINT, (fp, B.FINGERPRINT)
    ch, so, meta = B.load()
    assert meta["fingerprint"] == fp, (meta["fingerprint"], fp)
    tasks = B.tasks(bench)
    assert set(ch) == set(so) == set(meta["tasks"]) == {t["id"] for t in tasks}
    for t in tasks:                                      # the shipped files are exactly what the package builds
        assert ch[t["id"]] == B.to_arc(t, with_test_output=False), t["id"]
        assert so[t["id"]] == [B.grid_list(b) for b in t["test_out"]], t["id"]
    out = {}
    for split in B.SPLITS:
        for d in sorted(bench[split]):
            out[f"{split}-d{d}"] = {
                "set": f"{split}-d{d}", "split": split, "depth": d, "fingerprint": fp,
                "source": "pre-ARC benchmark, seed 0 (benchmark/{challenges,solutions,metadata}.json)",
                "tasks": [pretrain_task(t, meta["tasks"][t["id"]]) for t in bench[split][d]],
            }
    return out


# ------------------------------------------------------------------------------------------------------------------ #
# ARC-AGI-1 and Hodel's solvers
# ------------------------------------------------------------------------------------------------------------------ #

def arc_digest(arc_dir):
    """sha256 over the task files, in split then name order: each contributes "<split>/<name>\\n" and its bytes."""
    h = hashlib.sha256()
    for split in ARC_SPLITS:
        for fn in sorted(os.listdir(os.path.join(arc_dir, split))):
            if ARC_NAME.match(fn):
                h.update(f"{split}/{fn}\n".encode())
                with open(os.path.join(arc_dir, split, fn), "rb") as fh:
                    h.update(fh.read())
    return h.hexdigest()


def download_arc():
    """ARC-AGI-1's task files at ARC_COMMIT -> .cache/arc-agi-1-<commit>/{training,evaluation}/. Only files named like
    a task id are taken from the archive, and nothing is extracted by the archive's own paths."""
    dest = os.path.join(CACHE, f"arc-agi-1-{ARC_COMMIT[:12]}")
    if os.path.isdir(dest):
        return dest
    print(f"downloading ARC-AGI-1 at {ARC_COMMIT[:12]} ...", flush=True)
    raw = fetch(ARC_URL)
    tmp = dest + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    for split in ARC_SPLITS:
        os.makedirs(os.path.join(tmp, split))
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:
        for m in tf.getmembers():
            parts = m.name.split("/")
            if (m.isfile() and len(parts) == 4 and parts[1] == "data" and parts[2] in ARC_SPLITS
                    and ARC_NAME.match(parts[3])):
                with open(os.path.join(tmp, parts[2], parts[3]), "wb") as fh:
                    fh.write(tf.extractfile(m).read())
    os.replace(tmp, dest)
    return dest


def download_solvers():
    dest = os.path.join(CACHE, f"arc-dsl-{DSL_COMMIT[:12]}-solvers.py")
    if not os.path.exists(dest):
        print(f"downloading arc-dsl solvers.py at {DSL_COMMIT[:12]} ...", flush=True)
        raw = fetch(DSL_URL)
        with open(dest + ".tmp", "wb") as fh:
            fh.write(raw)
        os.replace(dest + ".tmp", dest)
    return dest


def write_licenses(out):
    """The upstream licence texts of the ARC-AGI-1 tasks (Apache-2.0) and Hodel's solvers (MIT), which the site
    redistributes, into out/licenses/."""
    os.makedirs(os.path.join(out, "licenses"), exist_ok=True)
    for name, (url, blob) in LICENSES.items():
        cached = os.path.join(CACHE, name)
        if not os.path.exists(cached):
            with open(cached + ".tmp", "wb") as fh:
                fh.write(fetch(url))
            os.replace(cached + ".tmp", cached)
        with open(cached, "rb") as fh:
            raw = fh.read()
        assert git_blob(raw) == blob, f"{name}: blob {git_blob(raw)}, expected {blob}"
        with open(os.path.join(out, "licenses", name), "wb") as fh:
            fh.write(raw)


def git_blob(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def hodel_solvers(path):
    """{task id: the source of `solve_<task id>`} from arc-dsl's solvers.py (parsed, never run)."""
    with open(path, "rb") as fh:
        raw = fh.read()
    assert git_blob(raw) == DSL_BLOB, f"{path} is not arc-dsl's solvers.py at {DSL_COMMIT[:12]} (blob {git_blob(raw)})"
    src = raw.decode()
    out = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("solve_"):
            out[node.name[len("solve_"):]] = ast.get_source_segment(src, node)
    return out


def build_arc(arc_dir, solvers_path):
    dg = arc_digest(arc_dir)
    assert dg == ARC_DIGEST, f"{arc_dir} is not ARC-AGI-1 at {ARC_COMMIT[:12]} (digest {dg})"
    solvers = hodel_solvers(solvers_path)
    out = {}
    for split in ARC_SPLITS:
        d = os.path.join(arc_dir, split)
        tasks = []
        for fn in sorted(f for f in os.listdir(d) if ARC_NAME.match(f)):
            tid = fn[:-5]
            with open(os.path.join(d, fn)) as fh:
                raw = json.load(fh)
            tasks.append({
                "id": tid,
                "train": [{"input": gstr(p["input"]), "output": gstr(p["output"])} for p in raw["train"]],
                "test": [{"input": gstr(p["input"]), "output": gstr(p["output"])} for p in raw["test"]],
                "solver": solvers.get(tid) if split == "training" else None,
            })
        assert len(tasks) == 400, (split, len(tasks))
        out[f"arc1-{split}"] = {
            "set": f"arc1-{split}", "split": split,
            "source": f"ARC-AGI-1 {split} tasks, github.com/fchollet/ARC-AGI at {ARC_COMMIT[:12]} (Apache-2.0)"
                      + (f"; solvers from Michael Hodel's arc-dsl, github.com/michaelhodel/arc-dsl at {DSL_COMMIT[:12]}"
                         " (MIT, Copyright (c) 2023 Michael Hodel)" if split == "training" else ""),
            "tasks": tasks,
        }
    n_sol = sum(t["solver"] is not None for t in out["arc1-training"]["tasks"])
    print(f"Hodel's solvers: {n_sol} of 400 training tasks")
    return out


# ------------------------------------------------------------------------------------------------------------------ #

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arc-dir", help="a local ARC-AGI-1 data directory holding training/ and evaluation/")
    ap.add_argument("--solvers", help="a local copy of arc-dsl's solvers.py")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    os.makedirs(CACHE, exist_ok=True)
    arc_dir = a.arc_dir or download_arc()
    solvers = a.solvers or download_solvers()
    files = build_pretrain()
    files.update(build_arc(arc_dir, solvers))
    os.makedirs(a.out, exist_ok=True)
    write_licenses(a.out)
    for name, obj in files.items():
        p = os.path.join(a.out, f"{name}.json")
        with open(p, "w") as fh:
            json.dump(obj, fh, separators=(",", ":"))
        print(f"{name:18s} {len(obj['tasks']):4d} tasks  {os.path.getsize(p) / 1e6:5.2f} MB")


if __name__ == "__main__":
    main()
