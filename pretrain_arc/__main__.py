"""python -m pretrain_arc <command>

  corpus     write corpus shards as JSONL, one task a line
  score      score a submission (Kaggle form) on the benchmark, beside the best possible solver
  baselines  score the three baselines
  show       print one task: its grids and its rule
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

from . import benchmark as B
from . import corpus as C
from . import generator as G
from . import score as S


def _ints(s):
    """'0-9' or '0,3,5' -> list of ints."""
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def cmd_corpus(a):
    os.makedirs(a.out, exist_ok=True)
    per_depth = {d: a.per_depth for d in _ints(a.depths)}
    jobs = [(a.out, a.seed, s, per_depth, a.extra) for s in _ints(a.shards)]  # one shard a job
    t0, n = time.time(), 0
    with Pool(a.workers) as pool:
        for path, k in pool.imap_unordered(C.write_shard, jobs):
            n += k
            print(f"{path}  ({n} tasks, {time.time() - t0:.0f}s)", flush=True)


def cmd_score(a):
    sub = json.load(open(a.submission))
    r = S.score(sub, attempts=tuple(_ints(a.attempts)))
    print(S.table(r, attempts=tuple(_ints(a.attempts))))
    missing = sum(x["missing"] for x in r["by_set"].values())
    if missing:
        print(f"\n{missing} benchmark tasks have no answer in the submission (scored as wrong)")


def cmd_baselines(a):
    ch, so, meta = B.load()
    for name in a.names.split(","):
        r = S.score(S.run_baseline(name, ch), so, meta)
        print(f"\n### {name}\n\n" + S.table(r))


def _grid_text(g):
    return "\n".join("".join(".123456789"[x] if x else "." for x in row) for row in g)


def cmd_show(a):
    ch, so, meta = B.load()
    t, m = ch[a.task], meta["tasks"][a.task]
    print(f"{a.task}: depth {m['depth']}, {m['split']}")
    for i, s in enumerate(m["rule"]):
        print(f"  step {i + 1}: {s['words']}{'  [held out]' if s['held_out'] else ''}")
    print(f"  floor: err@1 {m['floor']['err@1']:.3f} over {m['floor']['n_outputs']} possible outputs")
    pairs = [(f"demo {i + 1}", p["input"], p["output"]) for i, p in enumerate(t["train"])]
    pairs.append(("test", t["test"][0]["input"], so[a.task][0]))
    for name, x, y in pairs:
        print(f"\n{name}")
        for l, r in zip(_grid_text(x).split("\n"), _grid_text(y).split("\n")):
            print(f"  {l}   ->   {r}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m pretrain_arc", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("corpus", help="write corpus shards as JSONL")
    c.add_argument("--out", required=True)
    c.add_argument("--shards", default="0-9", help="e.g. 0-99 or 0,5,7")
    c.add_argument("--depths", default="1-4")
    c.add_argument("--per-depth", type=int, default=256, help="tasks of each depth a shard")
    c.add_argument("--extra", type=int, default=0, help="extra input/output pairs of each rule")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--workers", type=int, default=os.cpu_count())
    c.set_defaults(fn=cmd_corpus)
    s = sub.add_parser("score", help="score a submission")
    s.add_argument("submission")
    s.add_argument("--attempts", default="1,2,4")
    s.set_defaults(fn=cmd_score)
    b = sub.add_parser("baselines", help="score the baselines")
    b.add_argument("--names", default=",".join(S.BASELINES))
    b.set_defaults(fn=cmd_baselines)
    w = sub.add_parser("show", help="print one task")
    w.add_argument("task", help="e.g. held_out-d2-0007")
    w.set_defaults(fn=cmd_show)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
