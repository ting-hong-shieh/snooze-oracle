"""GPU sanity checks (run under the shared lock):
    flock /tmp/snooze-oracle-gpu.lock python -m src.ride.checks

1. memory_saving_mode on vs off: same context, first CHECK_ROWS test rows, all QUANTILES;
   prints the max absolute difference in minutes.
2. ride_total_samples smoke test on one early-morning route.
Results go to data/ride/checks.json.
"""

from __future__ import annotations

import json
import time
from datetime import datetime

import numpy as np

from .common import FEATURES, OUT, QUANTILES, TARGET, load_context, load_test

CHECK_ROWS = 200


def main() -> None:
    import torch
    from tabpfn import TabPFNRegressor

    from . import tabpfn_model
    from .sampler import ride_total_samples

    ctx, test = load_context(), load_test()
    # Spread the rows over both strata.
    idx = np.r_[0:CHECK_ROWS // 2, len(test) - CHECK_ROWS // 2:len(test)]
    X = test.iloc[idx][FEATURES].to_numpy(np.float64)
    preds = {}
    for mode in (True, False):
        torch.cuda.reset_peak_memory_stats()
        m = TabPFNRegressor(device="cuda", memory_saving_mode=mode,
                            categorical_features_indices=[0, 1])
        m.fit(ctx[FEATURES].to_numpy(np.float64), ctx[TARGET].to_numpy(np.float64))
        preds[mode] = np.array(m.predict(X, output_type="quantiles", quantiles=QUANTILES))
        print(f"memory_saving_mode={mode}: peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB",
              flush=True)
        del m
        torch.cuda.empty_cache()
    diff = np.abs(preds[True] - preds[False])
    out = {"memory_mode_check_rows": int(len(idx)),
           "memory_mode_max_abs_diff_min": float(diff.max()),
           "memory_mode_mean_abs_diff_min": float(diff.mean())}
    print(json.dumps(out), flush=True)

    t = time.time()
    s = ride_total_samples(7, 138, datetime(2026, 8, 12, 5, 30), n=2000,
                           rng=np.random.default_rng(0))
    out["sampler"] = {
        "call": "ride_total_samples(7, 138, 2026-08-12 05:30, n=2000)",
        "seconds": round(time.time() - t, 1), "n": int(len(s)),
        "finite": bool(np.isfinite(s).all()), "min": float(s.min()),
        "p10_p50_p90": np.percentile(s, [10, 50, 90]).round(2).tolist(),
        "max": float(s.max()),
    }
    assert tabpfn_model.MEMORY_SAVING_MODE is True
    print(json.dumps(out, indent=2))
    (OUT / "checks.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
