"""Download BTS Reporting Carrier On-Time Performance and keep arrivals into JFK and LGA.

Usage:
    python scripts/prep_flights.py 2026-07 2026-08

Writes data/processed/flights_YYYY-MM.parquet (schema in SPEC.md) and prints a report:
arrivals scheduled 04:00-07:59, and ArrDelay P10/P50/P90 overall and for the 10 busiest
early-morning origin airports. Raw zips are cached in data/raw/bts/.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import duckdb

from prep_trips import download, show

BASE = "https://transtats.bts.gov/PREZIP"
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "bts"
OUT = ROOT / "data" / "processed"
AIRPORTS = ("JFK", "LGA")
EARLY = (400, 759)  # scheduled arrival, local hhmm


def build_month(month: str) -> Path:
    year, mon = month.split("-")
    name = f"On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{int(mon)}"
    zpath = download(f"{BASE}/{name}.zip", RAW / f"{name}.zip")
    with zipfile.ZipFile(zpath) as zf:
        member = next(m for m in zf.namelist() if m.lower().endswith(".csv"))
        csv = RAW / f"ontime_{month}.csv"
        if not csv.exists():
            with zf.open(member) as src, open(csv, "wb") as dst:
                while chunk := src.read(1 << 20):
                    dst.write(chunk)

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"flights_{month}.parquet"
    dests = ", ".join(f"'{a}'" for a in AIRPORTS)
    con = duckdb.connect()
    con.execute(f"""
        COPY (
          WITH f AS (
            SELECT
              CAST(FlightDate AS DATE)                        AS flight_date,
              Reporting_Airline                               AS airline,
              CAST(Flight_Number_Reporting_Airline AS INTEGER) AS flight_num,
              Origin                                          AS origin,
              Dest                                            AS dest,
              CAST(CRSDepTime AS INTEGER)                     AS crs_dep_hhmm,
              CAST(CRSArrTime AS INTEGER)                     AS crs_arr_hhmm,
              TRY_CAST(ArrTime AS INTEGER)                    AS arr_hhmm,
              TRY_CAST(ArrDelay AS DOUBLE)                    AS arr_delay,
              CAST(Cancelled AS DOUBLE) = 1                   AS cancelled,
              CAST(Diverted AS DOUBLE) = 1                    AS diverted
            FROM read_csv('{csv}', header = true, all_varchar = true)
            WHERE Dest IN ({dests})
          )
          SELECT *,
            (crs_arr_hhmm // 100 * 60 + crs_arr_hhmm % 100) % 1440 AS crs_arr_min_of_day,
            -- FlightDate is the departure date; a scheduled arrival clock time earlier than
            -- the departure clock time means the flight lands the next day (red-eyes).
            flight_date + CAST(crs_arr_hhmm < crs_dep_hhmm OR crs_arr_hhmm = 2400 AS INTEGER)
              + INTERVAL ((crs_arr_hhmm // 100 * 60 + crs_arr_hhmm % 100) % 1440) MINUTE
              AS crs_arr_ts
          FROM f
        ) TO '{out}' (FORMAT PARQUET)
    """)
    n = con.execute(f"SELECT count(*) FROM read_parquet('{out}')").fetchone()[0]
    print(f"wrote   {out.relative_to(ROOT)}  ({n:,} arrivals into {'/'.join(AIRPORTS)})")
    return out


def report(months: list[str]) -> None:
    files = ", ".join(f"'{OUT / f'flights_{m}.parquet'}'" for m in months)
    con = duckdb.connect()
    con.execute(f"""
        CREATE VIEW f AS
        SELECT strftime(flight_date, '%Y-%m') AS month, *,
               crs_arr_hhmm BETWEEN {EARLY[0]} AND {EARLY[1]} AS early
        FROM read_parquet([{files}])
    """)
    q = """round(quantile_cont(arr_delay, 0.1), 1) AS delay_p10,
           round(quantile_cont(arr_delay, 0.5), 1) AS delay_p50,
           round(quantile_cont(arr_delay, 0.9), 1) AS delay_p90"""

    print("\n== 1. Arrivals into JFK / LGA per month (month = FlightDate, the departure date) ==")
    show(con.sql("""
        SELECT month, dest, count(*) AS arrivals,
               count(*) FILTER (WHERE early) AS sched_0400_0759,
               count(*) FILTER (WHERE early AND cancelled) AS early_cancelled,
               count(*) FILTER (WHERE early AND diverted) AS early_diverted
        FROM f GROUP BY ALL ORDER BY month, dest
    """))

    print("\n== 2. ArrDelay minutes (flown, not cancelled/diverted) ==")
    show(con.sql(f"""
        SELECT month, CASE WHEN early THEN '04:00-07:59' ELSE 'other hours' END AS sched_arr,
               dest, count(arr_delay) AS flown, {q}
        FROM f GROUP BY ALL ORDER BY month, sched_arr, dest
    """))
    show(con.sql(f"""
        SELECT month, 'all hours' AS sched_arr, count(arr_delay) AS flown, {q}
        FROM f GROUP BY ALL
        UNION ALL
        SELECT month, '04:00-07:59', count(arr_delay), {q}
        FROM f WHERE early GROUP BY ALL
        ORDER BY 1, 2
    """))

    print("\n== 3. 10 busiest origins for 04:00-07:59 arrivals: ArrDelay per month ==")
    show(con.sql(f"""
        WITH top AS (
          SELECT origin FROM f WHERE early GROUP BY origin ORDER BY count(*) DESC, origin LIMIT 10
        )
        SELECT origin, month, count(*) AS sched,
               count(*) FILTER (WHERE dest = 'JFK') AS to_jfk,
               count(*) FILTER (WHERE dest = 'LGA') AS to_lga,
               count(*) FILTER (WHERE cancelled OR diverted) AS canc_div, {q}
        FROM f JOIN top USING (origin) WHERE early
        GROUP BY ALL ORDER BY origin, month
    """))


def main(months: list[str]) -> None:
    if not months:
        sys.exit("usage: python scripts/prep_flights.py 2026-07 2026-08")
    for month in months:
        build_month(month)
    report(months)


if __name__ == "__main__":
    main(sys.argv[1:])
