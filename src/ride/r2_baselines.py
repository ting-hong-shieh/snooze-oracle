"""Round 2 CPU baselines, trained on ALL July trips in the product setting (no cap).

Usage (from the repo root):
    python -m src.ride.r2_baselines

Writes data/ride2/preds_<model>.parquet (test_id, q10..q90) for
    fixed_buffer   30 minutes at every quantile level (reference human rule)
    lookup         empirical quantiles per (pu_zone, do_zone, req_hour, is_weekend) over
                   every July row in the setting, round-1 back-off: a cell needs
                   >= LOOKUP_MIN_ROWS rows, else route, else origin zone, else global.
                   Also writes the back-off level and the rows in the cell used.
    lgbm_quantile  LightGBM objective="quantile", one model per level, defaults,
                   n_jobs=4 (resource cap), trained on every July row in the setting.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from .baselines import LGBM_THREADS, LOOKUP_LEVELS, LOOKUP_MIN_ROWS, FIXED_BUFFER_MIN
from .baselines import _lgbm_frame, _lookup_from_table
from .common import QUANTILES, TARGET, qcol
from .retrieval import OUT2, load_train


def save(name: str, test: pd.DataFrame, pred: dict, extra: dict | None = None) -> None:
    df = pd.DataFrame({"test_id": test["test_id"].to_numpy(),
                       **{qcol(q): np.asarray(pred[q], float) for q in QUANTILES},
                       **(extra or {})})
    df.to_parquet(OUT2 / f"preds_{name}.parquet", index=False)


def lookup_table(train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for li, keys in enumerate(LOOKUP_LEVELS):
        g = train.groupby(keys)[TARGET] if keys else train.assign(_k=0).groupby("_k")[TARGET]
        t = g.quantile(QUANTILES).unstack()  # numpy "linear" interpolation, as round 1
        t.columns = [qcol(q) for q in t.columns]
        t["n"] = g.size()
        t = t.reset_index()
        if not keys:
            t = t.drop(columns="_k")
        t["level"] = li
        rows.append(t)
    return pd.concat(rows, ignore_index=True)


def cell_rows(table: pd.DataFrame, test: pd.DataFrame, level: np.ndarray) -> np.ndarray:
    """Rows in the lookup cell each test row used."""
    out = np.zeros(len(test), int)
    for li, keys in enumerate(LOOKUP_LEVELS):
        idx = np.flatnonzero(level == li)
        if not len(idx):
            continue
        cells = table[table["level"] == li]
        if keys:
            out[idx] = test.iloc[idx][keys].merge(cells[keys + ["n"]], on=keys,
                                                  how="left")["n"].to_numpy()
        else:
            out[idx] = int(cells["n"].iloc[0])
    return out


def main() -> None:
    train = load_train()
    test = pd.read_parquet(OUT2 / "test.parquet")
    stats: dict = {"train_rows": len(train), "test_rows": len(test)}
    print(f"train {len(train):,} rows, test {len(test):,} rows")

    save("fixed_buffer", test, {q: np.full(len(test), FIXED_BUFFER_MIN) for q in QUANTILES})

    t = time.time()
    table = lookup_table(train)
    pred, level = _lookup_from_table(table, test)
    n_cell = cell_rows(table, test, level)
    save("lookup", test, pred, {"lookup_level": level, "lookup_cell_rows": n_cell})
    stats["lookup_levels_used"] = np.bincount(level, minlength=4).tolist()
    stats["lookup_s"] = round(time.time() - t, 1)
    print("lookup back-off levels used:", stats["lookup_levels_used"], flush=True)

    t = time.time()
    Xtr, Xte = _lgbm_frame(train), _lgbm_frame(test)
    out = {}
    import lightgbm as lgb
    for q in QUANTILES:
        m = lgb.LGBMRegressor(objective="quantile", alpha=q, n_jobs=LGBM_THREADS, verbose=-1)
        m.fit(Xtr, train[TARGET])
        out[q] = m.predict(Xte)
    save("lgbm_quantile", test, out)
    stats["lgbm_s"] = round(time.time() - t, 1)
    stats["lgbm_crossing_rows"] = int((np.diff(np.column_stack([out[q] for q in QUANTILES]),
                                               axis=1) < 0).any(axis=1).sum())
    (OUT2 / "baseline_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
