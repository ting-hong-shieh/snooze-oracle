"""Round 2 checks.

    python -m src.ride.r2_checks device      # CPU only, no lock
    python -m src.ride.r2_checks sampler     # takes the GPU lock itself (or set
                                             # SNOOZE_RIDE_DEVICE=cpu to run on the CPU)

device: the main run computed 329 of 707 contexts on the CPU because GPU predicts stalled
on the shared card. To size the CPU/GPU difference, rerun N_DEVICE_CHECK contexts that
were computed on the GPU (random, seeded, plus one borough context) on the CPU and
compare the five quantiles; also compare any contexts in data/ride2/tabpfn_gpu_redo/ (GPU
reruns of CPU contexts). Writes data/ride2/checks_device.json.

sampler: ride_total_samples smoke test on one early-morning route; checks that an
out-of-scope query raises. Writes data/ride2/checks_sampler.json.
"""

from __future__ import annotations

import fcntl
import json
import os
import resource
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

from .common import FEATURES, QUANTILES, qcol
from .r2_tabpfn import LOCK, OUT2, fit_context, key_tuple
from .retrieval import context_rows

N_DEVICE_CHECK = 20
QCOLS = [qcol(q) for q in QUANTILES]


def device() -> None:
    import torch

    torch.set_num_threads(4)
    test = pd.read_parquet(OUT2 / "test.parquet")
    st = pd.read_json(OUT2 / "tabpfn_stats.jsonl", lines=True).drop_duplicates(
        "ctx_key", keep="last")
    st["device"] = st["device"].fillna("cuda")
    gpu = st[(st["device"] == "cuda") & (st["n_ctx"] <= 5000)]  # CPU refuses > 5,000 rows
    rng = np.random.default_rng(0)
    keys = sorted(rng.choice(gpu["ctx_key"].to_numpy(), N_DEVICE_CHECK, replace=False))
    keys.append("borough_Bronx_138_6")  # largest borough context the CPU allows (<= 5,000 rows)
    rows = []
    for key in keys:
        g = test[test["ctx_key"] == key]
        ctx, _ = context_rows(key_tuple(g))
        m = fit_context(ctx, "cpu")
        q = np.column_stack(m.predict(g[FEATURES].to_numpy(np.float64),
                                      output_type="quantiles", quantiles=QUANTILES))
        old = pd.read_parquet(OUT2 / "tabpfn" / f"{key}.parquet")[QCOLS].to_numpy()
        d = np.abs(q - old)
        rows.append({"ctx_key": key, "compare": "gpu main vs cpu rerun", "n_ctx": len(ctx),
                     "rows": len(g), "max_abs_diff": float(d.max()),
                     "mean_abs_diff": float(d.mean())})
        print(rows[-1], flush=True)
    redo = OUT2 / "tabpfn_gpu_redo"
    for p in sorted(redo.glob("*.parquet")) if redo.exists() else []:
        a = pd.read_parquet(p)[QCOLS].to_numpy()
        b = pd.read_parquet(OUT2 / "tabpfn" / p.name)[QCOLS].to_numpy()
        d = np.abs(a - b)
        rows.append({"ctx_key": p.stem, "compare": "cpu main vs gpu redo", "n_ctx": None,
                     "rows": len(a), "max_abs_diff": float(d.max()),
                     "mean_abs_diff": float(d.mean())})
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    out = {"contexts": rows,
           "max_abs_diff_min": float(df["max_abs_diff"].max()),
           "mean_abs_diff_min (row-weighted)": float(np.average(df["mean_abs_diff"],
                                                                 weights=df["rows"])),
           "peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)}
    print(json.dumps({k: v for k, v in out.items() if k != "contexts"}, indent=2))
    (OUT2 / "checks_device.json").write_text(json.dumps(out, indent=2))


def sampler() -> None:
    lock = None
    if os.environ.get("SNOOZE_RIDE_DEVICE", "cuda") == "cuda":
        lock = open(LOCK, "w")
        fcntl.flock(lock, fcntl.LOCK_EX)
    from .sampler import ride_total_samples

    t = time.time()
    s = ride_total_samples(7, 138, datetime(2026, 8, 12, 5, 30), n=2000,
                           rng=np.random.default_rng(0))
    out = {
        "call": "ride_total_samples(7, 138, 2026-08-12 05:30, n=2000)",
        "device": os.environ.get("SNOOZE_RIDE_DEVICE", "cuda"),
        "seconds": round(time.time() - t, 1), "n": int(len(s)),
        "finite": bool(np.isfinite(s).all()), "min": float(s.min()),
        "p10_p50_p90": np.percentile(s, [10, 50, 90]).round(2).tolist(),
        "max": float(s.max()),
    }
    try:
        ride_total_samples(7, 1, datetime(2026, 8, 12, 5, 30))
        out["out_of_scope_raises"] = False
    except ValueError:
        out["out_of_scope_raises"] = True
    out["peak_rss_mib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    print(json.dumps(out, indent=2))
    (OUT2 / "checks_sampler.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    {"device": device, "sampler": sampler}[sys.argv[1]]()
