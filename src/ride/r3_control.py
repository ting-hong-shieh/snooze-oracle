"""Round 3 retrieval control: lookup table and LightGBM fit on exactly the r3 retrieval
context TabPFN uses, for the test rows in the <30 and 30-199 density bins.

Usage (from the repo root; CPU only):
    python -m src.ride.r3_control

For each context (route, request hour): the round-1 lookup (cell (pu_zone, do_zone,
req_hour, is_weekend) with >= 10 rows, else route, else pickup zone, else the whole
context) and LightGBM quantile (defaults, one model per level, 4 threads) are fit on the
context rows and predict that context's test rows. Writes data/ride3/preds_lookup_ctx.parquet
and preds_lgbm_ctx.parquet (test_id, q10..q90, plus lookup_level), finished contexts are
cached under data/ride3/control/.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

from .baselines import lgbm_quantile, lookup
from .common import QUANTILES, qcol
from .r3_context import OUT3, context_rows
from .r3_tabpfn import test_groups
from .retrieval import OUT2

BINS = ("<30", "30-199")
CACHE = OUT3 / "control"


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    test = pd.read_parquet(OUT2 / "test.parquet")
    test = test[test["density"].isin(BINS)]
    groups = test_groups(test)
    t_start = time.time()
    for i, (k, g) in enumerate(groups):
        path = CACHE / f"{k}.parquet"
        if path.exists():
            continue
        r = g.iloc[0]
        ctx, _, _ = context_rows(r.pu_zone, r.do_zone, r.req_hour)
        lk, level = lookup(ctx, g)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lg = lgbm_quantile(ctx, g)
        pd.DataFrame({"test_id": g["test_id"].to_numpy(), "n_ctx": len(ctx),
                      "lookup_level": level,
                      **{f"lk_{qcol(q)}": lk[q] for q in QUANTILES},
                      **{f"lg_{qcol(q)}": lg[q] for q in QUANTILES}}).to_parquet(path,
                                                                               index=False)
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(groups)} contexts, {time.time() - t_start:.0f}s", flush=True)
    p = pd.concat([pd.read_parquet(CACHE / f"{k}.parquet") for k, _ in groups])
    p = p.set_index("test_id").loc[test["test_id"]].reset_index()
    for name, pre in [("lookup_ctx", "lk_"), ("lgbm_ctx", "lg_")]:
        out = p[["test_id"] + [pre + qcol(q) for q in QUANTILES]].rename(
            columns=lambda c: c.removeprefix(pre))
        if name == "lookup_ctx":
            out["lookup_level"] = p["lookup_level"].to_numpy()
        out.to_parquet(OUT3 / f"preds_{name}.parquet", index=False)
    print(f"done: {len(p):,} rows, {len(groups)} contexts, {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
