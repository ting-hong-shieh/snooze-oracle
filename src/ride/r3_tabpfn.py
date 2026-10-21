"""Round 3 TabPFN ride predictions: GPU only, one fit per r3 retrieval context.

Usage (from the repo root; never wrap in `flock`, the code takes the GPU lock itself):
    python -m src.ride.r3_tabpfn run [--limit N] [--keys-from FILE]

Test rows are data/ride2/test.parquet (the round-2 sample). Each context's predictions go to
data/ride3/tabpfn/<key>.parquet (test_id + quantiles) and one line per context to
data/ride3/tabpfn_stats.jsonl; finished contexts are skipped on rerun. The GPU lock is held
for one batch of contexts (about LOCK_BATCH_S seconds) and released in between. There is
no CPU fallback: if CUDA fails, the process stops.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import resource
import time

import numpy as np
import pandas as pd

from .common import FEATURES, QUANTILES, qcol
from .r2_tabpfn import LOCK, fit_context
from .r3_context import OUT3, context_rows, key_name
from .retrieval import OUT2

PER_KEY = OUT3 / "tabpfn"
STATS = OUT3 / "tabpfn_stats.jsonl"
LOCK_BATCH_S = 120.0


def test_groups(test: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    keys = [key_name(a, b, c) for a, b, c in zip(test.pu_zone, test.do_zone, test.req_hour)]
    return sorted(test.assign(r3_key=keys).groupby("r3_key"), key=lambda kv: kv[0])


def run(limit: int | None, only: set[str] | None) -> None:
    import torch

    PER_KEY.mkdir(parents=True, exist_ok=True)
    test = pd.read_parquet(OUT2 / "test.parquet")
    todo = [(k, g) for k, g in test_groups(test)
            if not (PER_KEY / f"{k}.parquet").exists() and (only is None or k in only)]
    if limit:
        todo = todo[:limit]
    print(f"{len(todo)} contexts to do", flush=True)
    i = 0
    with open(LOCK, "a") as lock:
        while i < len(todo):
            t_wait = time.time()
            fcntl.flock(lock, fcntl.LOCK_EX)
            t_batch = time.time()
            try:
                while i < len(todo) and time.time() - t_batch < LOCK_BATCH_S:
                    k, g = todo[i]
                    r = g.iloc[0]
                    ctx, n_own, n_fill = context_rows(r.pu_zone, r.do_zone, r.req_hour)
                    torch.cuda.reset_peak_memory_stats()
                    t0 = time.time()
                    m = fit_context(ctx, "cuda")
                    t1 = time.time()
                    q = m.predict(g[FEATURES].to_numpy(np.float64), output_type="quantiles",
                                  quantiles=QUANTILES)
                    t2 = time.time()
                    pd.DataFrame({"test_id": g["test_id"].to_numpy(),
                                  **{qcol(qq): np.asarray(v, float)
                                     for qq, v in zip(QUANTILES, q)}}).to_parquet(
                        PER_KEY / f"{k}.parquet", index=False)
                    rec = {"key": k, "n_ctx": len(ctx), "n_own": n_own, "n_fill": n_fill,
                           "n_queries": len(g), "fit_s": round(t1 - t0, 2),
                           "predict_s": round(t2 - t1, 2), "device": "cuda",
                           "peak_alloc_mib": round(torch.cuda.max_memory_allocated() / 2**20),
                           "rss_mib": round(resource.getrusage(
                               resource.RUSAGE_SELF).ru_maxrss / 1024)}
                    with open(STATS, "a") as fh:
                        fh.write(json.dumps(rec) + "\n")
                    del m
                    i += 1
                torch.cuda.empty_cache()
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
            print(f"{i}/{len(todo)} done; batch {time.time() - t_batch:.0f}s, "
                  f"waited {t_batch - t_wait:.0f}s", flush=True)
            time.sleep(1)


def merge() -> None:
    test = pd.read_parquet(OUT2 / "test.parquet")
    parts = [pd.read_parquet(PER_KEY / f"{k}.parquet") for k, _ in test_groups(test)]
    p = pd.concat(parts).set_index("test_id").loc[test["test_id"]].reset_index()
    p.to_parquet(OUT3 / "preds_tabpfn.parquet", index=False)
    print(f"merged {len(p):,} rows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "merge"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--keys-from")
    a = ap.parse_args()
    if a.cmd == "run":
        only = set(open(a.keys_from).read().split()) if a.keys_from else None
        run(a.limit, only)
    else:
        merge()
