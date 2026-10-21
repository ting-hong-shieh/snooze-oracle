"""TabPFN-3.5 regressor (local, CUDA) fit on data/ride/context.parquet.

Usage (from the repo root; the GPU is shared, so always take the lock):
    flock /tmp/snooze-oracle-gpu.lock python -m src.ride.tabpfn_model

Predicts the QUANTILES for every test row in chunks of PREDICT_CHUNK rows. Each chunk is
written to data/ride/preds/tabpfn_chunks/ as soon as it is done, and finished chunks are
skipped on a rerun, so a crash loses at most one chunk. At the end the chunks are merged
into data/ride/preds/tabpfn.parquet and run stats (wall time, peak GPU memory) go to
data/ride/preds/tabpfn_stats.json.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from .common import (CATEGORICAL, FEATURES, PRED, QUANTILES, TARGET, load_context,
                     load_test, qcol, save_pred)

PREDICT_CHUNK = 500
# Batches attention internally to cut peak GPU memory (only ~1.9 GB is free on the
# shared card); it does not change the model or its settings.
MEMORY_SAVING_MODE = True
CHUNKS = PRED / "tabpfn_chunks"


def make_model(device: str = "cuda"):
    from tabpfn import TabPFNRegressor

    # Library defaults (model version 3.5, n_estimators="auto", random_state=0). Besides
    # the device and the memory switch, the only argument names which columns are
    # categorical (zone ids).
    return TabPFNRegressor(
        device=device,
        memory_saving_mode=MEMORY_SAVING_MODE,
        categorical_features_indices=[FEATURES.index(c) for c in CATEGORICAL],
    )


def fit_model(ctx: pd.DataFrame, device: str = "cuda"):
    m = make_model(device)
    m.fit(ctx[FEATURES].to_numpy(np.float64), ctx[TARGET].to_numpy(np.float64))
    return m


def main() -> None:
    import torch

    ctx, test = load_context(), load_test()
    CHUNKS.mkdir(parents=True, exist_ok=True)
    stats: dict = {"context_rows": len(ctx), "test_rows": len(test),
                   "predict_chunk": PREDICT_CHUNK,
                   "memory_saving_mode": MEMORY_SAVING_MODE, "quantiles": QUANTILES}
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    m = fit_model(ctx)
    torch.cuda.synchronize()
    stats["fit_s"] = round(time.time() - t0, 2)
    stats["fit_peak_alloc_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    stats["n_estimators"] = getattr(m, "n_estimators_", m.n_estimators)
    stats["max_number_of_samples"] = m.inference_config_.MAX_NUMBER_OF_SAMPLES
    print(json.dumps(stats), flush=True)

    t1 = time.time()
    for start in range(0, len(test), PREDICT_CHUNK):
        path = CHUNKS / f"chunk_{start:06d}.parquet"
        if path.exists():
            continue
        part = test.iloc[start:start + PREDICT_CHUNK]
        qs = m.predict(part[FEATURES].to_numpy(np.float64), output_type="quantiles",
                       quantiles=QUANTILES)
        df = pd.DataFrame({"test_id": part["test_id"].to_numpy(),
                           **{qcol(q): v for q, v in zip(QUANTILES, qs)}})
        df.to_parquet(path, index=False)
        print(f"chunk {start:6d}  {time.time() - t1:7.1f}s  peak "
              f"{torch.cuda.max_memory_allocated() / 2**20:7.1f} MiB", flush=True)
    stats["predict_s_this_run"] = round(time.time() - t1, 2)
    stats["total_s_this_run"] = round(time.time() - t0, 2)
    stats["peak_alloc_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    stats["peak_reserved_mib"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)

    pred = pd.concat([pd.read_parquet(p) for p in sorted(CHUNKS.glob("chunk_*.parquet"))])
    pred = pred.sort_values("test_id")
    assert len(pred) == len(test) and (pred["test_id"].to_numpy() == test["test_id"].to_numpy()).all()
    save_pred("tabpfn", pred["test_id"].to_numpy(), {q: pred[qcol(q)].to_numpy() for q in QUANTILES})
    (PRED / "tabpfn_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
