"""pretrain-ARC: an unbounded corpus of ARC-style tasks from a known rule grammar, a fixed benchmark with held-out
steps, and the exact Bayes floor of every benchmark task. See the README."""

from . import benchmark, corpus, floors, generator, score  # noqa: F401

__version__ = "0.1.0"
