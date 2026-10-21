"""Round 2 data: product setting (do_zone in {132, 138}, request 03:00-08:59).

Usage (from the repo root):
    python -m src.ride.r2_data

Writes (git-ignored, under data/ride2/):
    train.parquet       every July trip in the setting (all models draw from these rows)
    route_density.parquet  July trips per route (pu_zone, do_zone) in the setting
    test_pool_counts.csv   August trips per (density bin, day) before subsampling
    test.parquet        the August test rows every model is scored on, with columns
                        test_id, density, day, weight (= pool rows / sampled rows in its
                        (bin, day) stratum), ctx_level, ctx_key, ...

Test subsample: every August row in the two sparse bins (<30, 30-199); in each of the two
dense bins, SAMPLE_FRAC_DENSE of the rows of every day (stratified by day, seeded).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .retrieval import (OUT2, TEST_FILE, TRAIN_FILE, connect, context_key, key_name,
                        load_train, window_sql)

DENSITY_BINS = ["<30", "30-199", "200-999", "1000+"]
KEEP_ALL_BINS = {"<30", "30-199"}
SAMPLE_FRAC_DENSE = 0.20
SEED = 20261021


def density_bin(n: np.ndarray) -> np.ndarray:
    return np.select([n < 30, n < 200, n < 1000], DENSITY_BINS[:3], DENSITY_BINS[3])


def main() -> None:
    OUT2.mkdir(parents=True, exist_ok=True)
    con = connect()
    con.execute(f"COPY ({window_sql(TRAIN_FILE, '2026-07-01', '2026-08-01')} "
                f"ORDER BY request_ts, pu_zone, do_zone) TO '{OUT2 / 'train.parquet'}' "
                f"(FORMAT parquet)")
    load_train.cache_clear()
    train = load_train()
    dens = (train.groupby(["pu_zone", "do_zone"]).size().rename("route_july_trips")
            .reset_index())
    dens.to_parquet(OUT2 / "route_density.parquet", index=False)

    pool = con.execute(window_sql(TEST_FILE, "2026-08-01", "2026-09-01")
                       + " ORDER BY request_ts, pu_zone, do_zone").fetchdf()
    pool = pool.merge(dens, on=["pu_zone", "do_zone"], how="left")
    pool["route_july_trips"] = pool["route_july_trips"].fillna(0).astype(int)
    pool["density"] = density_bin(pool["route_july_trips"].to_numpy())
    pool["day"] = pool["request_ts"].dt.day
    counts = pool.groupby(["density", "day"]).size().rename("pool_rows").reset_index()
    counts.to_csv(OUT2 / "test_pool_counts.csv", index=False)

    rng = np.random.default_rng(SEED)
    parts = []
    for (b, d), g in pool.groupby(["density", "day"], sort=True):
        if b in KEEP_ALL_BINS:
            s = g
        else:
            k = int(round(SAMPLE_FRAC_DENSE * len(g)))
            s = g.iloc[np.sort(rng.choice(len(g), k, replace=False))]
        parts.append(s.assign(weight=len(g) / len(s)))
    test = pd.concat(parts).sort_values(["request_ts", "pu_zone", "do_zone"])
    keys = [context_key(r.pu_zone, r.do_zone, r.req_hour, r.pu_borough)
            for r in test.itertuples()]
    test["ctx_level"] = [k[0] for k in keys]
    test["ctx_key"] = [key_name(k) for k in keys]
    test.insert(0, "test_id", np.arange(len(test)))
    test.to_parquet(OUT2 / "test.parquet", index=False)

    print(f"train rows {len(train):,}; test pool {len(pool):,}; test sample {len(test):,}")
    print(pool.groupby("density").size().rename("pool").to_frame()
          .join(test.groupby("density").size().rename("sampled")))
    print("context keys:", test["ctx_key"].nunique(), "| by level:",
          test.groupby("ctx_level")["ctx_key"].nunique().to_dict())


if __name__ == "__main__":
    main()
