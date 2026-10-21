"""Download NYC TLC High Volume FHV trips and build the processed Uber trip table.

Usage:
    python scripts/prep_trips.py 2026-07 2026-08

Writes data/processed/uber_trips_YYYY-MM.parquet (schema in SPEC.md).
Raw downloads are cached in data/raw/ and skipped if already present.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import duckdb

BASE = "https://d37ci6vzurychx.cloudfront.net"
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
AIRPORT_ZONES = (1, 132, 138)  # Newark, JFK, LaGuardia


def show(rel) -> None:
    rel.show(max_rows=10_000, max_width=10_000)  # DuckDB truncates rows and columns by default


def download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"cached  {dest.name}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"fetch   {url}")

    def progress(blocks: int, block_size: int, total: int) -> None:
        if total > 0:
            pct = min(100, blocks * block_size * 100 // total)
            print(f"\r        {pct:3d}% of {total / 1e6:,.0f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, tmp, reporthook=progress)
    print()
    tmp.rename(dest)
    return dest


def build_month(month: str, zones_csv: Path) -> Path:
    raw = download(f"{BASE}/trip-data/fhvhv_tripdata_{month}.parquet",
                   RAW / f"fhvhv_tripdata_{month}.parquet")
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"uber_trips_{month}.parquet"

    con = duckdb.connect()
    con.execute(f"CREATE VIEW zones AS SELECT * FROM read_csv_auto('{zones_csv}')")
    airports = ", ".join(str(z) for z in AIRPORT_ZONES)
    con.execute(f"""
        COPY (
          WITH t AS (
            SELECT
              request_datetime                                   AS request_ts,
              pickup_datetime                                    AS pickup_ts,
              CAST(PULocationID AS INTEGER)                      AS pu_zone,
              CAST(DOLocationID AS INTEGER)                      AS do_zone,
              date_diff('second', request_datetime, pickup_datetime) / 60.0 AS wait_min,
              trip_time / 60.0                                   AS ride_min,
              base_passenger_fare                                AS fare,
              trip_miles
            FROM read_parquet('{raw}')
            WHERE hvfhs_license_num = 'HV0003'
              AND shared_request_flag = 'N'
              AND wav_request_flag = 'N'
              AND access_a_ride_flag = 'N'
              AND request_datetime IS NOT NULL
              AND pickup_datetime IS NOT NULL
          )
          SELECT
            t.request_ts, t.pickup_ts, t.pu_zone, t.do_zone,
            zp.Borough AS pu_borough, zd.Borough AS do_borough,
            t.wait_min, t.ride_min, t.wait_min + t.ride_min AS total_min,
            t.fare, t.trip_miles, t.fare / t.trip_miles AS fare_per_mile,
            hour(t.request_ts)                                 AS req_hour,
            hour(t.request_ts) * 60 + minute(t.request_ts)     AS req_min_of_day,
            isodow(t.request_ts) - 1                           AS dow,
            isodow(t.request_ts) >= 6                          AS is_weekend,
            t.pu_zone IN ({airports})                          AS pu_airport,
            t.do_zone IN ({airports})                          AS do_airport
          FROM t
          LEFT JOIN zones zp ON zp.LocationID = t.pu_zone
          LEFT JOIN zones zd ON zd.LocationID = t.do_zone
          WHERE t.wait_min BETWEEN 0 AND 60
            AND t.ride_min BETWEEN 1 AND 180
            AND t.fare > 0
            AND t.trip_miles > 0.1
            AND t.pu_zone NOT IN (264, 265)
            AND t.do_zone NOT IN (264, 265)
        ) TO '{out}' (FORMAT PARQUET)
    """)
    n = con.execute(f"SELECT count(*) FROM read_parquet('{out}')").fetchone()[0]
    print(f"wrote   {out.relative_to(ROOT)}  ({n:,} Uber trips)")
    return out


def main(months: list[str]) -> None:
    if not months:
        sys.exit("usage: python scripts/prep_trips.py 2026-07 2026-08")
    zones_csv = download(f"{BASE}/misc/taxi_zone_lookup.csv", RAW / "taxi_zone_lookup.csv")
    for month in months:
        build_month(month, zones_csv)


if __name__ == "__main__":
    main(sys.argv[1:])
