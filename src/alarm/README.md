# Alarm: pickup scenario and the August 2026 backtest

`p_on_time`, `snooze_curve` and `latest_safe_wake` (SPEC "Alarm") read quantile tables that
are precomputed the night before (`tables.py`); the alarm itself runs no model. Results of
the backtest are in `docs/backtest.md` and `docs/frontier.png`.

## Reproduce

```bash
python -m src.alarm.backtest routes
python -m src.alarm.tables flight
python -m src.alarm.tables ride --routes data/ride3/backtest_routes.csv
python -m src.alarm.backtest run
python -m src.alarm.backtest report
```

`tables ride --models tabpfn` takes the GPU lock itself per batch (96 contexts, ~9 min);
everything else is CPU (flight tables 10 s, ride lookup/LightGBM 34 s, run ~2.5 min,
report ~20 s).

## Pickup zones

| airport | kind | pu_zone | zone | borough | July trips | August trips |
|---|---|---|---|---|---|---|
| JFK | dense | 216 | South Ozone Park | Queens | 3,351 | 3,393 |
| JFK | dense | 10 | Baisley Park | Queens | 2,558 | 2,639 |
| JFK | dense | 230 | Times Sq/Theatre District | Manhattan | 2,403 | 1,944 |
| JFK | dense | 76 | East New York | Brooklyn | 1,798 | 1,878 |
| JFK | sparse | 245 | West Brighton | Staten Island | 23 | 28 |
| JFK | sparse | 190 | Prospect Park | Brooklyn | 15 | 27 |
| JFK | sparse | 251 | Westerleigh | Staten Island | 28 | 27 |
| JFK | sparse | 96 | Forest Park/Highland Park | Queens | 23 | 23 |
| LGA | dense | 230 | Times Sq/Theatre District | Manhattan | 3,841 | 3,309 |
| LGA | dense | 164 | Midtown South | Manhattan | 2,568 | 2,343 |
| LGA | dense | 7 | Astoria | Queens | 2,065 | 2,189 |
| LGA | dense | 48 | Clinton East | Manhattan | 2,041 | 1,703 |
| LGA | sparse | 123 | Homecrest | Brooklyn | 29 | 39 |
| LGA | sparse | 222 | Starrett City | Brooklyn | 27 | 35 |
| LGA | sparse | 29 | Brighton Beach | Brooklyn | 26 | 31 |
| LGA | sparse | 178 | Ocean Parkway South | Brooklyn | 29 | 31 |

Trips counted in the product setting (to the airport, requested 03:00–08:59).

## Judgment calls (not in SPEC.md)

Models and tables
1. One quantile grid for every model: 50 levels, 0.01 … 0.99. Monte Carlo draws
   u ~ U(0.01, 0.99) and interpolates linearly, so the outer 1% tails of every model are
   cut the same way. Quantile vectors are sorted per row (LightGBM levels can cross).
2. Ride tables are on a request-time grid every 10 minutes, 03:00 … 08:30 plus 08:59,
   per route and date; between grid points the quantiles are interpolated linearly. A
   request outside 03:00–08:59 uses the nearest edge (gut rules with large B request
   before 03:00).
3. Strategy (c) uses the full-data July lookup for rides and the origin x carrier lookup
   (≥ 20 rows, else origin, else all) for flights; (d) uses LightGBM on all July rides and
   on all flight training rows with the "dest only" features; (e) uses TabPFN with the r3
   retrieval context for rides and the round-2 "dest only" TabPFN grid for flights (reused,
   not recomputed). LightGBM was refit at the 50 levels (round 2 had only 3 for flights).
4. Prep draws are clipped at 0 minutes.
5. The same random draws (flight level, prep, ride level) are reused for every wake time
   and every model of a scenario (common random numbers).
6. `latest_safe_wake` is a bisection over whole minutes from 300 min before to 120 min
   after the scheduled arrival, assuming p_on_time does not increase with the wake time;
   it equalled a full 1-minute scan on 120 of 120 sampled scenarios. No scenario was
   infeasible.

Backtest
7. Scenario flights: the flight test month as defined in the flight workstream (FlightDate
   in August) and scheduled to land in August, 04:00–07:59. 32 red-eyes landing on 09-01
   were dropped; red-eyes landing on 08-01 departed on 07-31 (a training month) and are not
   in the test set. Cancelled (91) and diverted (5) flights skipped.
8. Pickup zones: rule in `choose_routes()` (table above); airports are not pickup zones;
   "sparse" = 10–29 July trips, picking those with the most August trips so the reality pool
   is rarely empty. No scenario was skipped for an empty pool.
9. Reality ride: the u-quantile of real August trips on the same route with request minute
   of day within ±60 min of the strategy's request time (rounded to the minute, any day of
   August, wrapping at midnight). Five draws per scenario.
10. Waiting = minutes between the driver's arrival and the passenger coming out
    (actual arrival + 20 min); 0 when late.
11. Gut rule and schedule-only use the July lookup q50 / q90 for the route at the request
    hour of `crs_arr + exit_buffer − 30 min` (one fixed guess, not iterated).
12. Gap to the gut curve: the curve is the gut rule every 5 minutes (B = 0, 5, …, 90),
    linear between points; with only the six reported B values, chords of the convex curve
    made schedule-only look 10 points below the curve. The six B values in the request are
    the reported strategies; the 5-minute ones are only the reference curve.
13. Snooze: the alarm first rings 30 minutes before the TabPFN wake time; the user takes
    the largest allowed snooze (1–9 min) each time and gets up at the first refusal.
14. Bootstrap: 2,000 resamples of the 31 arrival days; medians are weighted medians over
    the resampled rows; paired differences use the same resamples.

## Fairness

- All three model strategies target the same 0.90; their late rates differ because their
  calibration differs, so "below the curve" mixes calibration with sharpness. The headline
  comparison is against the gut curve at equal waiting, which removes the calibration part.
- The lookup and LightGBM baselines use all training rows; TabPFN uses a 10k retrieval
  context per route and hour. Ride TabPFN for sparse zones is mostly borough rows.
- The reality ride pool mixes all August days at the same time of day; the models never
  see the scenario's own day, but the pool includes rides from days the models were not
  trained on in the same way for every strategy.
- Flight delays come from the actual BTS record; flight models never saw August.
- Eight zones per airport and one month; sparse zones are four per airport. Scenarios share
  flights across zones, so the effective sample is closer to the number of flights and days
  than to 12,600 (the day bootstrap accounts for the day clustering).
- The snooze simulation is deterministic given the wake time (p_on_time is monotone in
  practice), so it adds no information beyond strategy (e).
