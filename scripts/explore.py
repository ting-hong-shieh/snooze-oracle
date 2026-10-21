"""First look at the processed Uber trips: is there enough early-morning airport data?

Usage:
    python scripts/explore.py 2026-08

Prints:
  1. overall wait / ride / total minutes quantiles
  2. trip counts for airport routes by request hour (where the snooze story lives)
  3. for the busiest early-morning (4-7am) routes to JFK / LGA: total_min P10/P50/P90
  4. fare by request time within the hour for one busy route (is "wait 10 min" ever cheaper?)
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]


def show(rel) -> None:
    rel.show(max_rows=10_000)  # DuckDB truncates to 20 rows by default


def main(month: str) -> None:
    path = ROOT / "data" / "processed" / f"uber_trips_{month}.parquet"
    con = duckdb.connect()
    con.execute(f"CREATE VIEW t AS SELECT * FROM read_parquet('{path}')")

    print("\n== 1. Overall minutes (P10 / P50 / P90) ==")
    show(con.sql("""
        SELECT
          count(*) AS trips,
          round(quantile_cont(wait_min,  0.1), 1) AS wait_p10,
          round(quantile_cont(wait_min,  0.5), 1) AS wait_p50,
          round(quantile_cont(wait_min,  0.9), 1) AS wait_p90,
          round(quantile_cont(total_min, 0.5), 1) AS total_p50,
          round(quantile_cont(total_min, 0.9), 1) AS total_p90
        FROM t
    """))

    print("\n== 2. Trips TO an airport, by request hour ==")
    show(con.sql("""
        SELECT req_hour,
               count(*) FILTER (WHERE do_zone = 132) AS to_jfk,
               count(*) FILTER (WHERE do_zone = 138) AS to_lga,
               count(*) FILTER (WHERE do_zone = 1)   AS to_ewr
        FROM t GROUP BY req_hour ORDER BY req_hour
    """))

    print("\n== 3. Busiest 4-7am routes to JFK / LGA: total_min quantiles ==")
    show(con.sql("""
        SELECT pu_zone, do_zone, count(*) AS trips,
               round(quantile_cont(total_min, 0.1), 1) AS p10,
               round(quantile_cont(total_min, 0.5), 1) AS p50,
               round(quantile_cont(total_min, 0.9), 1) AS p90,
               round(quantile_cont(total_min, 0.9) - quantile_cont(total_min, 0.5), 1) AS p90_minus_p50
        FROM t
        WHERE do_zone IN (132, 138) AND req_hour BETWEEN 4 AND 6
        GROUP BY pu_zone, do_zone
        ORDER BY trips DESC LIMIT 15
    """))

    print("\n== 4. Weekday fare per mile by 10-min request slot, busiest route 6-10am ==")
    route = con.sql("""
        SELECT pu_zone, do_zone FROM t
        WHERE req_hour BETWEEN 6 AND 9 AND NOT is_weekend
        GROUP BY pu_zone, do_zone ORDER BY count(*) DESC LIMIT 1
    """).fetchone()
    print(f"route {route[0]} -> {route[1]}")
    show(con.sql(f"""
        SELECT req_min_of_day // 10 * 10 AS slot_start_min,
               lpad(CAST(req_min_of_day // 60 AS VARCHAR), 2, '0') || ':' ||
               lpad(CAST(req_min_of_day // 10 * 10 % 60 AS VARCHAR), 2, '0') AS slot,
               count(*) AS trips,
               round(median(fare_per_mile), 2) AS med_fare_per_mile,
               round(median(fare), 2) AS med_fare
        FROM t
        WHERE pu_zone = {route[0]} AND do_zone = {route[1]}
          AND req_hour BETWEEN 6 AND 9 AND NOT is_weekend
        GROUP BY 1, 2 ORDER BY 1
    """))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "2026-08")
