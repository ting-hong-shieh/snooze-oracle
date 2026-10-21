"""First look at the processed Uber trips: is there enough early-morning airport data?

Usage:
    python scripts/explore.py 2026-08

Prints:
  1. overall wait / ride / total minutes quantiles
  2. trip counts for airport routes by request hour (where the snooze story lives)
  3. for the busiest early-morning (4-7am) routes to JFK / LGA: total_min P10/P50/P90
  4. fare by 10-min slot (3-7am) and by hour for the 5 busiest 3-7am routes to an
     airport (pu_zone <> do_zone): is "wait 10 min" ever cheaper?
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb

from prep_trips import AIRPORT_ZONES, show

ROOT = Path(__file__).resolve().parents[1]


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
        ORDER BY trips DESC, pu_zone, do_zone LIMIT 15
    """))

    airports = ", ".join(str(z) for z in AIRPORT_ZONES)
    con.execute(f"""
        CREATE TEMP TABLE routes AS
        SELECT pu_zone, do_zone, count(*) AS trips_3to7
        FROM t
        WHERE do_zone IN ({airports}) AND pu_zone <> do_zone AND req_hour BETWEEN 3 AND 6
        GROUP BY ALL ORDER BY trips_3to7 DESC, pu_zone, do_zone LIMIT 5
    """)
    con.execute("""
        CREATE TEMP VIEW rt AS
        SELECT t.* FROM t JOIN routes USING (pu_zone, do_zone)
    """)

    print("\n== 4. Fare by request time, 5 busiest 3-7am routes to an airport (all days) ==")
    show(con.sql("SELECT * FROM routes ORDER BY trips_3to7 DESC"))

    print("\n-- 4a. By 10-min request slot, 03:00-06:59 --")
    show(con.sql(bucket_sql("req_min_of_day // 10 * 10", "req_hour BETWEEN 3 AND 6")))
    print("\n-- 4b. Slot summary per route --")
    show(con.sql(summary_sql("req_min_of_day // 10 * 10", "req_hour BETWEEN 3 AND 6", 10)))

    print("\n-- 4c. By request hour, 0-23 --")
    show(con.sql(bucket_sql("req_hour * 60", "TRUE")))
    print("\n-- 4d. Hour summary per route --")
    show(con.sql(summary_sql("req_hour * 60", "TRUE", 60)))


def bucket_sql(bucket: str, where: str) -> str:
    """Per route and time bucket (start minute of day): trips, fare and fare_per_mile."""
    return f"""
        SELECT pu_zone, do_zone,
               lpad(CAST(b // 60 AS VARCHAR), 2, '0') || ':' ||
               lpad(CAST(b % 60 AS VARCHAR), 2, '0') AS start,
               trips, med_fare, fare_p10, fare_p90, med_fare_per_mile
        FROM (
          SELECT pu_zone, do_zone, {bucket} AS b, count(*) AS trips,
                 round(median(fare), 2) AS med_fare,
                 round(quantile_cont(fare, 0.1), 2) AS fare_p10,
                 round(quantile_cont(fare, 0.9), 2) AS fare_p90,
                 round(median(fare_per_mile), 2) AS med_fare_per_mile
          FROM rt WHERE {where} GROUP BY ALL
        )
        ORDER BY pu_zone, do_zone, b
    """


def summary_sql(bucket: str, where: str, step: int) -> str:
    """Per route: range of bucket medians, mean |change| between adjacent buckets
    (only buckets exactly `step` minutes apart), and mean within-bucket P90-P10 fare spread."""
    return f"""
        WITH b AS (
          SELECT pu_zone, do_zone, {bucket} AS b, count(*) AS trips,
                 median(fare) AS med_fare, median(fare_per_mile) AS med_fpm,
                 quantile_cont(fare, 0.9) - quantile_cont(fare, 0.1) AS spread
          FROM rt WHERE {where} GROUP BY ALL
        ), d AS (
          SELECT *,
                 b - lag(b) OVER w = {step} AS adjacent,
                 abs(med_fare - lag(med_fare) OVER w) AS d_fare,
                 abs(med_fpm - lag(med_fpm) OVER w) AS d_fpm
          FROM b WINDOW w AS (PARTITION BY pu_zone, do_zone ORDER BY b)
        )
        SELECT pu_zone, do_zone, count(*) AS buckets, min(trips) AS min_trips,
               round(min(med_fare), 2) AS min_med_fare, round(max(med_fare), 2) AS max_med_fare,
               round(avg(d_fare) FILTER (WHERE adjacent), 2) AS mean_abs_adj_fare,
               round(avg(d_fpm) FILTER (WHERE adjacent), 2) AS mean_abs_adj_fpm,
               round(avg(spread), 2) AS mean_within_p90_p10
        FROM d GROUP BY ALL ORDER BY pu_zone, do_zone
    """

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "2026-08")
