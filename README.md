# Snooze Oracle

An alarm that works out, the night before, the latest time you can get up and still be on
time with 90% probability, and then lets you snooze only while you would still make it.

Two mornings are covered:

- **Commute:** an Uber to a meeting in Midtown, the Financial District or Hudson Yards.
- **Pickup:** meeting someone off an early flight (landing 03:00–08:59) at JFK or LaGuardia.

Ride times come from TabPFN-3.5 run locally on NYC TLC Uber trips; flight delays from
TabPFN-3.5 on BTS on-time data. The backtests behind it are in `docs/backtest.md` and
`docs/commute_backtest.md`.

## Replay dates

The public data ends in August 2026, so "tomorrow" is a replay: the planner plans a morning
in August 2026 (default 2026-08-12, a Wednesday) using only what was known the night
before. The `.ics` file it writes puts the same clock times on a real date
(`--alarm-date`, default tomorrow on your computer) so the alarms can ring.

## Run the planner

The planner reads precomputed tables only: no GPU, no model call, under 2 seconds.

```bash
python -m src.agent index      # once, after the tables below exist
python -m src.agent plan --scenario commute --from "Cobble Hill" --to "Midtown North" --deadline 09:00
python -m src.agent flights --airport JFK --date 2026-08-12
python -m src.agent plan --scenario pickup --flight DL709 --airport JFK --from "South Ozone Park"
```

It writes `out/plan.json` (wake time, first ring, latest time to request the ride,
probability of being on time, and the lookup-table and gut-rule alternatives) and
`out/alarms.ics` (a first ring 30 minutes before the wake time and a final alarm at it).

Place names are fuzzy-matched to TLC taxi zones; when a name fits several zones you are
asked to choose. Only precomputed routes work:

- commute: 12 routes, listed in `docs/commute_backtest.md`;
- pickup: 8 pickup zones per airport, listed in `src/alarm/README.md`, and the August flights
  that `python -m src.agent flights` lists.

## Rebuild everything

Python 3.11 or 3.12. On WSL with a driver that supports CUDA 12.6, install torch from the
cu126 index after the requirements (see `SPEC.md`).

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
python scripts/prep_trips.py 2026-07 2026-08
python scripts/prep_flights.py --no-report 2025-08 2025-09 2025-10 2025-11 2025-12 2026-01 2026-02 2026-03 2026-04 2026-05 2026-06 2026-07 2026-08
```

Then follow the reproduce sections in `src/flight/RESULTS_round2.md`,
`src/ride/RESULTS_round3.md` and `src/alarm/README.md`, and
`python -m src.alarm.commute commute-routes|tables|run|report` for the commute tables.
TabPFN steps need a GPU and take the lock in `/tmp/snooze-oracle-gpu.lock` themselves.

## TabPFN-3.5 license

TabPFN-3.5's weights are free under Prior Labs' non-commercial license; you accept it once
at first use and the weights are cached on your machine. The weights are not in this
repository, and Snooze Oracle does not offer a hosted prediction service: the public demo
uses precomputed results only.

## License

Code: Apache-2.0 (`LICENSE`). Data: NYC TLC trip records and BTS on-time performance, both
public; they are downloaded by the scripts and not stored in git.
