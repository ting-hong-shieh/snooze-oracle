# Project spec (shared by all sessions — read this first)

Working name: **Snooze Oracle** — an alarm that lets you snooze only while an open
tabular model (TabPFN-3.5, run locally) says you will still make it on time: to a fixed
deadline, or to pick someone up from an early-morning flight at JFK / LGA.

Dropped: a "wait 10 minutes to save on the fare" feature. The data says snoozing 10 minutes
does not save money; see `docs/findings.md`.

Challenge: Hacktoberfest 2026 DEV Open-Source AI Challenge, Week 3 "Slow and Steady".
Deadline: 2026-10-25 23:59 PDT (= 2026-10-26 14:59 Taipei). Target categories:
Best Use of TabPFN (primary), Gemma, Temporal, Sentry (only if actually used).

## Data sources (all free, no keys)

| Source | URL | Notes |
|---|---|---|
| NYC TLC High Volume FHV trips | `https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_YYYY-MM.parquet` | Monthly parquet, latest 2026-08. Times are NYC local, naive. |
| NYC taxi zone lookup | `https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv` | 265 zones. JFK = 132, LaGuardia = 138, Newark = 1. |
| Open-Meteo historical weather | `https://archive-api.open-meteo.com/v1/archive` | Hourly precipitation for NYC (later step). |
| BTS on-time performance | `https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_YYYY_M.zip` | Monthly zip of one CSV, US carriers' domestic flights only (no foreign airlines, no international arrivals). Local times as hhmm. Airport-pickup scenario. |

Months (time-based holdout; never shuffle across months):
- Rides: **train = 2026-07, test = 2026-08**.
- Flights: **train = 2025-08 … 2026-07** (12 months), **test = 2026-08**.

## Processed trip table (`data/processed/uber_trips_YYYY-MM.parquet`)

Produced by `scripts/prep_trips.py`. One row per Uber trip.

| Column | Type | Definition |
|---|---|---|
| `request_ts` | timestamp | `request_datetime` |
| `pickup_ts` | timestamp | `pickup_datetime` |
| `pu_zone`, `do_zone` | int | `PULocationID`, `DOLocationID` |
| `pu_borough`, `do_borough` | str | from zone lookup |
| `wait_min` | float | `(pickup_ts - request_ts)` in minutes |
| `ride_min` | float | `trip_time / 60` |
| `total_min` | float | `wait_min + ride_min` — time from pressing "request" to arrival |
| `fare` | float | `base_passenger_fare` (pre tolls/tips/taxes; includes surge) |
| `trip_miles` | float | `trip_miles` |
| `fare_per_mile` | float | `fare / trip_miles` |
| `req_hour` | int | 0–23 |
| `req_min_of_day` | int | 0–1439 |
| `dow` | int | 0 = Monday … 6 = Sunday |
| `is_weekend` | bool | `dow >= 5` |
| `pu_airport`, `do_airport` | bool | zone in {1, 132, 138} |

Filters applied (document any change here):
- `hvfhs_license_num = 'HV0003'` (Uber only)
- `shared_request_flag = 'N'`, `wav_request_flag = 'N'`, `access_a_ride_flag = 'N'`
- `0 <= wait_min <= 60`, `1 <= ride_min <= 180`, `fare > 0`, `trip_miles > 0.1`
- `pu_zone`, `do_zone` not in {264, 265} (unknown / outside NYC)

## Processed flight table (`data/processed/flights_YYYY-MM.parquet`)

Produced by `scripts/prep_flights.py`. One row per scheduled flight arriving at JFK or LGA,
from the BTS file for that month (the month is `FlightDate`, i.e. the departure date).

| Column | Type | Definition |
|---|---|---|
| `flight_date` | date | `FlightDate` (scheduled departure date, origin local) |
| `airline` | str | `Reporting_Airline` |
| `flight_num` | int | `Flight_Number_Reporting_Airline` |
| `origin`, `dest` | str | `Origin`, `Dest` (IATA); `dest` in {JFK, LGA} |
| `crs_dep_hhmm` | int | `CRSDepTime` (origin local); needed only to date the arrival |
| `crs_arr_hhmm` | int | `CRSArrTime` (scheduled arrival, NYC local) |
| `arr_hhmm` | int | `ArrTime` (actual arrival, NYC local); null if cancelled/diverted |
| `arr_delay` | float | `ArrDelay` minutes (negative = early); null if cancelled/diverted |
| `cancelled`, `diverted` | bool | `Cancelled = 1`, `Diverted = 1` |
| `distance` | float | `Distance` (miles, origin to dest) |
| `crs_arr_min_of_day` | int | `crs_arr_hhmm` as minutes, 0–1439 |
| `crs_arr_ts` | timestamp | scheduled arrival, NYC local: `flight_date` + 1 day if `crs_arr_hhmm < crs_dep_hhmm` (red-eyes), + `crs_arr_min_of_day` |

"Early-morning arrival" = `crs_arr_hhmm` between 0400 and 0759. Delay quantiles use flown
flights only (cancelled and diverted are counted separately).

## Exploration definitions (`scripts/explore.py`)

- "4–7am" for trips = `req_hour` 4–6 (request 04:00–06:59); "3–7am" = `req_hour` 3–6.
- Fare-by-time check (section 4): the 5 busiest routes with `do_zone` in {1, 132, 138} and
  `pu_zone <> do_zone`, ranked by trips requested 03:00–06:59, all days of the week.
  For each route: median fare and `fare_per_mile` per 10-minute request slot (03:00–06:59)
  and per request hour (0–23); mean absolute change of the bucket median between adjacent
  buckets; mean within-bucket fare P90–P10 spread.
- Ties in "busiest" rankings are broken by `pu_zone`, `do_zone` (or `origin`).

## Code layout and ownership

| Path | Owner | Branch |
|---|---|---|
| `scripts/`, `SPEC.md`, `docs/` | shared (main) | `main` |
| `src/ride/` | ride workstream | `ride-model` |
| `src/flight/` | flight workstream | `flight-model` |

A workstream edits only its own `src/` folder. Changes to shared files go through main.

## Interfaces

Ride workstream (`src/ride/`):

```python
def ride_total_samples(pu_zone: int, do_zone: int, request_ts: datetime,
                       n: int = 2000) -> np.ndarray:
    """n sampled total_min values (request -> drop-off, minutes) for an Uber requested
    at request_ts from pu_zone to do_zone."""

def p_on_time(pu_zone: int, do_zone: int, wake_time: datetime, deadline: datetime,
              prep_min: float = 15.0, prep_sd: float = 5.0) -> float:
    """Probability of arriving by `deadline` if you get up at `wake_time`,
    spend ~prep_min getting ready, then request an Uber."""

def snooze_curve(pu_zone, do_zone, deadline, start, end, step_min=1, **prep) -> pd.DataFrame:
    """Columns: wake_time, p_on_time. Precomputed the night before; the alarm only reads it."""
```

Flight workstream (`src/flight/`):

```python
def flight_arrival_samples(origin: str, carrier: str, crs_arr_ts: datetime,
                           n: int = 2000) -> np.ndarray:
    """n sampled ArrDelay minutes for a flight from `origin` on `carrier`
    (`airline` column) scheduled to arrive at crs_arr_ts (NYC local),
    conditional on the flight not being cancelled."""

def p_cancel(origin: str, carrier: str, crs_arr_ts: datetime) -> float:
    """Probability the flight is cancelled. Reported separately, never mixed into the
    ArrDelay samples."""
```

Snooze rule: allow a snooze of length `s` only if `p_on_time(wake_time = now + s) >= 0.90`;
`s` = largest such value up to 9 minutes. Otherwise refuse and keep ringing.

Pickup scenario (meeting a flight): you are on time if

    wake + prep + ride_total <= actual_arrival + exit_buffer

where `prep ~ N(prep_min, prep_sd)`, the ride is requested at `wake + prep`,
`ride_total` comes from `ride_total_samples`, `actual_arrival = crs_arr_ts + ArrDelay`
from `flight_arrival_samples`, and `exit_buffer` (deplaning and bags) is a user parameter,
default 20 minutes. Being early is fine; being late is the failure. `p_cancel` is shown
next to `p_on_time`, not folded into it.

## Evaluation (both workstreams)

Compare on the 2026-08 holdout, same features, default settings, no tuning:
1. Human rule: fixed buffer (e.g. "leave 30 min early"); for flights, schedule only (delay = 0).
2. Lookup table: empirical quantiles per (pu_zone, do_zone, req_hour, is_weekend) cell for
   rides; per origin and per origin x carrier for flights.
3. LightGBM quantile regression, defaults.
4. TabPFN-3.5 regressor (local), quantile outputs.

Report: coverage (P90 for rides; P10 and P90 for flights), pinball loss, and the decision
metrics — average extra minutes of sleep and late rate. Break results down by data density
(dense routes vs sparse early-morning / rare routes; for flights, training flights per
origin x carrier) — that is where TabPFN is expected to matter.

## Constraints

- TabPFN on CPU accepts ≤ 5,000 training rows per fit by default: filter by route/time
  window, then sample. Dev machine: WSL2 (Ubuntu) with an NVIDIA RTX 3080 Ti (CUDA).
  `data/` is a symlink to `/mnt/d/Data/snooze-oracle` (kept out of git).
  The Windows NVIDIA driver supports CUDA 12.6, so on WSL install torch from the cu126 index
  after `requirements.txt`: `pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126`.
- TabPFN-3.5 weights are non-commercial and may not be offered as a hosted service:
  never commit weights; never deploy a public endpoint that serves predictions.
  Public demo = video + static precomputed page.
- An LLM (local Gemma via Ollama) may only phrase results; every number comes from code.
- Repo must be created within 2026-10-19 … 2026-10-25.
