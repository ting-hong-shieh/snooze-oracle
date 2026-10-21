"""CPU baselines, all fit on data/ride/context.parquet (the TabPFN context rows).

Usage (from the repo root):
    python -m src.ride.baselines

Writes one file per model to data/ride/preds/<model>.parquet with columns
test_id, q10, q25, q50, q75, q90.

Models:
    fixed_buffer      human rule: always allow FIXED_BUFFER_MIN minutes (same number at
                      every quantile level, i.e. a point mass).
    lookup            empirical quantiles per (pu_zone, do_zone, req_hour, is_weekend),
                      backing off to coarser cells when a cell has < LOOKUP_MIN_ROWS rows.
    lgbm_quantile     LightGBM, objective="quantile", one model per quantile level,
                      library defaults except n_jobs=4 (machine resource cap).
    lookup_full_july  REFERENCE ONLY, not part of the same-rows comparison: the lookup
                      table built from every July trip instead of the context sample.
"""

from __future__ import annotations

import time

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

from .common import (CATEGORICAL, FEATURES, QUANTILES, ROOT, TARGET, load_context,
                     load_test, qcol, save_pred)

FIXED_BUFFER_MIN = 30.0
LOOKUP_MIN_ROWS = 10
# Back-off chain: the first level whose cell has >= LOOKUP_MIN_ROWS rows is used.
LOOKUP_LEVELS = [
    ["pu_zone", "do_zone", "req_hour", "is_weekend"],
    ["pu_zone", "do_zone"],
    ["pu_zone"],
    [],  # global
]
LGBM_THREADS = 4
ZONE_CATEGORIES = list(range(1, 264))  # TLC zones 1..263 (264/265 are filtered out)


def fixed_buffer(ctx: pd.DataFrame, test: pd.DataFrame) -> dict[float, np.ndarray]:
    return {q: np.full(len(test), FIXED_BUFFER_MIN) for q in QUANTILES}


def _lookup_from_table(table: pd.DataFrame, test: pd.DataFrame,
                       ) -> tuple[dict[float, np.ndarray], np.ndarray]:
    """`table` has one row per (level, key...) with n and q-columns. Returns predictions
    and the back-off level used per test row."""
    out = {q: np.full(len(test), np.nan) for q in QUANTILES}
    level_used = np.full(len(test), -1)
    for li, keys in enumerate(LOOKUP_LEVELS):
        todo = level_used < 0
        if not todo.any():
            break
        cells = table[(table["level"] == li) & (table["n"] >= LOOKUP_MIN_ROWS)]
        if keys:
            merged = test.loc[todo, keys].merge(cells, on=keys, how="left")
        else:
            merged = pd.DataFrame({qcol(q): np.repeat(cells[qcol(q)].to_numpy(), todo.sum())
                                   for q in QUANTILES})
        hit = merged[qcol(QUANTILES[0])].notna().to_numpy()
        idx = np.flatnonzero(todo)[hit]
        for q in QUANTILES:
            out[q][idx] = merged[qcol(q)].to_numpy()[hit]
        level_used[idx] = li
    return out, level_used


def lookup(ctx: pd.DataFrame, test: pd.DataFrame):
    rows = []
    for li, keys in enumerate(LOOKUP_LEVELS):
        g = ctx.groupby(keys)[TARGET] if keys else ctx.assign(_k=0).groupby("_k")[TARGET]
        t = g.quantile(QUANTILES).unstack()  # numpy "linear" interpolation
        t.columns = [qcol(q) for q in t.columns]
        t["n"] = g.size()
        t = t.reset_index()
        if not keys:
            t = t.drop(columns="_k")
        t["level"] = li
        rows.append(t)
    return _lookup_from_table(pd.concat(rows, ignore_index=True), test)


def lookup_full_july(test: pd.DataFrame):
    """Same lookup rule, but cells computed in DuckDB over all July trips (reference)."""
    train = ROOT / "data" / "processed" / "uber_trips_2026-07.parquet"
    con = duckdb.connect()
    con.execute("SET memory_limit='2GB'; SET threads=4;")
    con.register("test_keys", test[["pu_zone", "do_zone", "req_hour", "is_weekend"]])
    qs = ", ".join(str(q) for q in QUANTILES)
    rows = []
    for li, keys in enumerate(LOOKUP_LEVELS):
        sel = ", ".join(keys)
        # Only compute cells that some test row can use.
        join = (f"SEMI JOIN (SELECT DISTINCT {sel} FROM test_keys) k USING ({sel})"
                if keys else "")
        df = con.execute(f"""
            SELECT {sel + ',' if keys else ''} count(*) AS n,
                   quantile_cont(total_min, [{qs}]) AS qs
            FROM (SELECT pu_zone, do_zone, req_hour, CAST(is_weekend AS INTEGER) AS is_weekend,
                         total_min FROM read_parquet('{train}')
                  WHERE request_ts >= TIMESTAMP '2026-07-01'
                    AND request_ts < TIMESTAMP '2026-08-01') t
            {join}
            {'GROUP BY ' + sel if keys else ''}
        """).fetchdf()
        qarr = np.vstack(df.pop("qs").to_numpy())
        for i, q in enumerate(QUANTILES):
            df[qcol(q)] = qarr[:, i]
        df["level"] = li
        rows.append(df)
    return _lookup_from_table(pd.concat(rows, ignore_index=True), test)


def _lgbm_frame(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURES].copy()
    for c in CATEGORICAL:
        X[c] = pd.Categorical(X[c], categories=ZONE_CATEGORIES)
    return X


def lgbm_quantile(ctx: pd.DataFrame, test: pd.DataFrame) -> dict[float, np.ndarray]:
    Xtr, Xte = _lgbm_frame(ctx), _lgbm_frame(test)
    out = {}
    for q in QUANTILES:
        m = lgb.LGBMRegressor(objective="quantile", alpha=q, n_jobs=LGBM_THREADS,
                              verbose=-1)
        m.fit(Xtr, ctx[TARGET])  # pandas category dtype -> LightGBM categorical
        out[q] = m.predict(Xte)
    return out


def main() -> None:
    ctx, test = load_context(), load_test()
    print(f"context {len(ctx):,} rows, test {len(test):,} rows")
    for name, fn in [("fixed_buffer", fixed_buffer), ("lookup", lookup),
                     ("lgbm_quantile", lgbm_quantile)]:
        t = time.time()
        res = fn(ctx, test)
        pred, level = res if isinstance(res, tuple) else (res, None)
        save_pred(name, test["test_id"].to_numpy(), pred)
        msg = f"{name:16s} {time.time() - t:6.1f}s"
        if level is not None:
            msg += "  back-off levels used: " + str(np.bincount(level, minlength=4).tolist())
        print(msg, flush=True)
    t = time.time()
    pred, level = lookup_full_july(test)
    save_pred("lookup_full_july", test["test_id"].to_numpy(), pred)
    print(f"{'lookup_full_july':16s} {time.time() - t:6.1f}s  back-off levels used: "
          f"{np.bincount(level, minlength=4).tolist()}")


if __name__ == "__main__":
    main()
