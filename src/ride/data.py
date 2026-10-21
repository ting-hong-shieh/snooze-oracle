"""Build the shared training context and the test sample for the ride models.

Usage (from the repo root):
    python -m src.ride.data

Writes (git-ignored, under data/ride/):
    context.parquet      N_CONTEXT rows sampled uniformly from 2026-07 (requests in July).
                         Every model (fixed buffer, lookup, LightGBM, TabPFN) is fit on
                         exactly these rows.
    origin_density.parquet  full-July trip counts per (pu_zone, req_hour, is_weekend).
    test.parquet         N_TEST_PER_STRATUM rows per density stratum from 2026-08.

All filtering and sampling runs in DuckDB with a 2 GB memory cap; only the sampled rows
reach pandas.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "data" / "ride"

TRAIN = PROCESSED / "uber_trips_2026-07.parquet"
TEST = PROCESSED / "uber_trips_2026-08.parquet"

N_CONTEXT = 10_000
N_TEST_PER_STRATUM = 5_000
SEED = 20261021
# An origin cell (pu_zone, req_hour, is_weekend) is "sparse" if it had fewer than this
# many trips in the whole of July 2026.
SPARSE_MAX_TRIPS = 100

COLS = "request_ts, pu_zone, do_zone, req_hour, req_min_of_day, dow, " \
       "CAST(is_weekend AS INTEGER) AS is_weekend, total_min"


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET memory_limit='2GB'; SET threads=4;")
    return con


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    con = connect()

    # Strict month split on the request time: the processed files hold a few trips
    # requested in the last minutes of the previous month.
    con.execute(f"""
        CREATE TEMP TABLE density AS
        SELECT pu_zone, req_hour, CAST(is_weekend AS INTEGER) AS is_weekend,
               count(*) AS origin_cell_trips_july
        FROM read_parquet('{TRAIN}')
        WHERE request_ts >= TIMESTAMP '2026-07-01' AND request_ts < TIMESTAMP '2026-08-01'
        GROUP BY ALL
    """)
    con.execute(f"COPY density TO '{OUT / 'origin_density.parquet'}' (FORMAT parquet)")

    con.execute(f"""
        COPY (
          SELECT * FROM (
            SELECT {COLS} FROM read_parquet('{TRAIN}')
            WHERE request_ts >= TIMESTAMP '2026-07-01'
              AND request_ts < TIMESTAMP '2026-08-01'
          ) USING SAMPLE reservoir({N_CONTEXT} ROWS) REPEATABLE ({SEED})
          ORDER BY request_ts, pu_zone, do_zone
        ) TO '{OUT / 'context.parquet'}' (FORMAT parquet)
    """)

    # Test rows: label each August trip by its origin cell's July count (cells absent in
    # July count as 0 -> sparse), then sample the same number of rows per stratum.
    con.execute(f"""
        CREATE TEMP TABLE test_pool AS
        SELECT t.*, coalesce(d.origin_cell_trips_july, 0) AS origin_cell_trips_july,
               CASE WHEN coalesce(d.origin_cell_trips_july, 0) < {SPARSE_MAX_TRIPS}
                    THEN 'sparse' ELSE 'dense' END AS density
        FROM (
          SELECT {COLS} FROM read_parquet('{TEST}')
          WHERE request_ts >= TIMESTAMP '2026-08-01' AND request_ts < TIMESTAMP '2026-09-01'
        ) t
        LEFT JOIN density d USING (pu_zone, req_hour, is_weekend)
    """)
    print(con.execute("""
        SELECT density, count(*) AS trips, count(*) / sum(count(*)) OVER () AS share
        FROM test_pool GROUP BY 1 ORDER BY 1
    """).fetchdf())
    parts = []
    for i, stratum in enumerate(["dense", "sparse"]):
        parts.append(f"""
          SELECT * FROM (SELECT * FROM test_pool WHERE density = '{stratum}')
          USING SAMPLE reservoir({N_TEST_PER_STRATUM} ROWS) REPEATABLE ({SEED + 1 + i})
        """)
    con.execute(f"""
        COPY (
          SELECT row_number() OVER (ORDER BY density, request_ts, pu_zone, do_zone) - 1
                 AS test_id, *
          FROM ({' UNION ALL '.join(parts)})
          ORDER BY test_id
        ) TO '{OUT / 'test.parquet'}' (FORMAT parquet)
    """)
    ctx = con.execute(f"SELECT count(*), min(request_ts), max(request_ts) "
                      f"FROM '{OUT / 'context.parquet'}'").fetchall()
    tst = con.execute(f"SELECT density, count(*), count(*) FILTER (WHERE req_hour BETWEEN 3 AND 6) "
                      f"FROM '{OUT / 'test.parquet'}' GROUP BY 1 ORDER BY 1").fetchall()
    print("context rows, first/last request:", ctx)
    print("test rows per stratum (total, of which 03:00-06:59):", tst)


if __name__ == "__main__":
    main()
