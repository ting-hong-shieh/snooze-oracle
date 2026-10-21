# Flight workstream: results (holdout 2026-08)

ArrDelay (minutes) for flights into JFK and LGA scheduled to arrive 03:00-08:59
(`crs_arr_hhmm` 300-859), flown flights only. Train = flights dated 2025-08 ... 2026-07
(28,672 flown rows), test = flights dated 2026-08 (2,487 flown rows). Every model was fit on
the same 28,672 rows (TabPFN used all of them as context; no subsample) and scored on the same
2,487 rows.

Columns:
- **P10 cov** = share of test flights with `arr_delay <= predicted 10th percentile` (target 0.10).
- **P90 cov** = share with `arr_delay <= predicted 90th percentile` (target 0.90).
- **pin10 / pin50 / pin90** = mean pinball (quantile) loss in minutes at quantiles 0.1 / 0.5 / 0.9;
  **pin mean** = their average. Lower is better.
- **bin** = number of flown training flights (12 months, 03:00-08:59) for the test flight's
  origin x carrier. Bins fixed before looking at results: 0, 1-49, 50-199, 200-799, 800+;
  0 (11 test flights) and 1-49 (36) are merged in the table because each alone is too small.

## Delay quantiles

| model | bin | n_test | P10 cov | P90 cov | pin10 | pin50 | pin90 | pin mean(10,50,90) |
|---|---|---|---|---|---|---|---|---|
| schedule (delay=0) | <50 (incl. unseen) | 47 | 0.766 | 0.766 | 13.915 | 12.979 | 12.043 | 12.979 |
| lookup origin | <50 (incl. unseen) | 47 | 0.149 | 0.915 | 3.495 | 10.574 | 10.293 | 8.121 |
| lookup origin x carrier | <50 (incl. unseen) | 47 | 0.128 | 0.894 | 3.204 | 10.181 | 10.206 | 7.863 |
| LightGBM quantile | <50 (incl. unseen) | 47 | 0.191 | 0.915 | 3.594 | 10.702 | 10.090 | 8.129 |
| TabPFN-3.5 | <50 (incl. unseen) | 47 | 0.191 | 0.915 | 3.557 | 11.261 | 11.235 | 8.684 |
| schedule (delay=0) | 50-199 | 476 | 0.662 | 0.662 | 13.572 | 20.661 | 27.749 | 20.661 |
| lookup origin | 50-199 | 476 | 0.074 | 0.836 | 4.843 | 19.102 | 21.005 | 14.984 |
| lookup origin x carrier | 50-199 | 476 | 0.080 | 0.847 | 4.919 | 19.117 | 20.515 | 14.850 |
| LightGBM quantile | 50-199 | 476 | 0.149 | 0.807 | 4.893 | 19.018 | 21.227 | 15.046 |
| TabPFN-3.5 | 50-199 | 476 | 0.101 | 0.834 | 4.848 | 19.023 | 20.330 | 14.734 |
| schedule (delay=0) | 200-799 | 1580 | 0.692 | 0.692 | 11.144 | 14.048 | 16.952 | 14.048 |
| lookup origin | 200-799 | 1580 | 0.082 | 0.860 | 3.829 | 13.224 | 13.803 | 10.285 |
| lookup origin x carrier | 200-799 | 1580 | 0.084 | 0.863 | 3.803 | 13.138 | 13.580 | 10.174 |
| LightGBM quantile | 200-799 | 1580 | 0.120 | 0.826 | 3.790 | 13.018 | 13.758 | 10.189 |
| TabPFN-3.5 | 200-799 | 1580 | 0.101 | 0.857 | 3.781 | 12.954 | 13.357 | 10.031 |
| schedule (delay=0) | 800+ | 384 | 0.672 | 0.672 | 10.160 | 11.017 | 11.874 | 11.017 |
| lookup origin | 800+ | 384 | 0.057 | 0.867 | 3.284 | 10.327 | 8.065 | 7.225 |
| lookup origin x carrier | 800+ | 384 | 0.057 | 0.867 | 3.323 | 10.306 | 7.938 | 7.189 |
| LightGBM quantile | 800+ | 384 | 0.089 | 0.802 | 3.121 | 9.981 | 7.916 | 7.006 |
| TabPFN-3.5 | 800+ | 384 | 0.076 | 0.859 | 3.066 | 9.919 | 7.433 | 6.806 |
| schedule (delay=0) | all | 2487 | 0.684 | 0.684 | 11.509 | 14.826 | 18.142 | 14.826 |
| lookup origin | all | 2487 | 0.078 | 0.858 | 3.933 | 13.852 | 14.229 | 10.671 |
| lookup origin x carrier | all | 2487 | 0.080 | 0.861 | 3.931 | 13.789 | 13.973 | 10.564 |
| LightGBM quantile | all | 2487 | 0.122 | 0.820 | 3.894 | 13.654 | 14.216 | 10.588 |
| TabPFN-3.5 | all | 2487 | 0.099 | 0.854 | 3.870 | 13.615 | 13.737 | 10.407 |

TabPFN minus each other model, in mean pinball over quantiles 0.1/0.5/0.9 (negative = TabPFN
better). The 95% interval resamples whole arrival days (2,000 bootstrap draws), because delays
on the same morning are correlated.


| other model | bin | TabPFN minus other (mean pinball) | 95% day-bootstrap CI |
|---|---|---|---|
| schedule (delay=0) | <50 (incl. unseen) | -4.295 | [-5.209, -3.282] |
| lookup origin | <50 (incl. unseen) | +0.563 | [+0.152, +0.961] |
| lookup origin x carrier | <50 (incl. unseen) | +0.821 | [+0.399, +1.261] |
| LightGBM quantile | <50 (incl. unseen) | +0.556 | [+0.286, +0.859] |
| schedule (delay=0) | 50-199 | -5.927 | [-6.603, -5.241] |
| lookup origin | 50-199 | -0.250 | [-0.480, +0.017] |
| lookup origin x carrier | 50-199 | -0.117 | [-0.294, +0.075] |
| LightGBM quantile | 50-199 | -0.313 | [-0.511, -0.123] |
| schedule (delay=0) | 200-799 | -4.018 | [-4.328, -3.682] |
| lookup origin | 200-799 | -0.255 | [-0.376, -0.139] |
| lookup origin x carrier | 200-799 | -0.143 | [-0.243, -0.033] |
| LightGBM quantile | 200-799 | -0.158 | [-0.237, -0.082] |
| schedule (delay=0) | 800+ | -4.211 | [-4.728, -3.758] |
| lookup origin | 800+ | -0.419 | [-0.675, -0.199] |
| lookup origin x carrier | 800+ | -0.383 | [-0.632, -0.145] |
| LightGBM quantile | 800+ | -0.200 | [-0.392, -0.037] |
| schedule (delay=0) | all | -4.418 | [-4.694, -4.110] |
| lookup origin | all | -0.264 | [-0.376, -0.150] |
| lookup origin x carrier | all | -0.157 | [-0.264, -0.046] |
| LightGBM quantile | all | -0.181 | [-0.265, -0.104] |

Reading it:
- TabPFN-3.5 has the lowest mean pinball loss overall and in the 50-199, 200-799 and 800+ bins;
  the margins over the lookup tables and LightGBM are small (0.15-0.4 min) but their day-bootstrap
  intervals exclude zero except against the lookups in the 50-199 bin.
- In the sparse bin (<50 training flights) TabPFN is **worse** than the lookups and LightGBM
  (+0.56 to +0.82 min). That bin is only 47 test flights from 6 cells, 23 of them RDU-AA
  (48 training flights) and 10 PDX-AS (unseen), so it says little either way. The expectation
  that TabPFN helps most on sparse routes is not supported by this test month.
- Every model's P90 is too low in 2026-08 (coverage 0.82-0.86 vs 0.90): the month had a heavier
  late tail than most training months (test P90 of ArrDelay is about 50 min; in 11 of the 12
  training months it was 13-37 min, and 50 min only in 2026-07). LightGBM is the most over-confident at P90 (0.820) and the
  least at P10 (0.122); TabPFN's P10 coverage (0.099) is the closest to target.
- Schedule only (delay = 0) is far behind on every metric. 68% of flown early-morning flights
  arrived at or before schedule, so "delay = 0" is a P68, not a P10 or a P90.

## p_cancel calibration (2026-08)

Model: shrunken empirical cancellation rate per origin x carrier (see Judgment calls). Scored on
all 2,630 scheduled 03:00-08:59 arrivals dated 2026-08 (cancelled = 1, everything else, including
diverted, = 0).

| | value |
|---|---|
| observed cancellation rate | 0.0502 (132 / 2,630) |
| mean predicted p_cancel | 0.0335 |
| Brier score, p_cancel | 0.04765 |
| Brier score, constant training rate 0.0333 | 0.04796 |
| log loss, p_cancel | 0.1990 |
| log loss, constant training rate | 0.2029 |

Reliability by quartile of predicted p_cancel:

| quartile | predicted range | n | mean predicted | observed |
|---|---|---|---|---|
| 1 | 0.0105-0.0255 | 647 | 0.0187 | 0.0232 |
| 2 | 0.0256-0.0336 | 650 | 0.0287 | 0.0462 |
| 3 | 0.0338-0.0395 | 624 | 0.0365 | 0.0593 |
| 4 | 0.0405-0.0620 | 709 | 0.0486 | 0.0705 |

The ranking is right (observed rate rises with every quartile) but the level is too low by
about a third in every quartile: 2026-08 was a bad month. Cancellations cluster on a few days
(08-07: 26 of 93 arrivals cancelled; 08-03: 20 of 94; 08-21: 15 of 93), and the monthly training
rate swings from 0.3% (2026-04) to 9.1% (2026-02), so a 12-month average cannot know the
month's level. p_cancel is a long-run base rate per route and airline, not a forecast for a given
morning; it would need weather or same-day operational data to do better.

## How to reproduce

From the worktree root, with the project venv
(`PY=/home/elias-shieh/projects/competition/hackathon/snooze-oracle/.venv/bin/python`):

```bash
flock /tmp/snooze-oracle-gpu.lock $PY -m src.flight.evaluate tabpfn   # ~26 min on the shared GPU
$PY -m src.flight.evaluate baselines                                   # CPU, seconds
$PY -m src.flight.evaluate report      # writes data/flight/results_table.md
$PY -m src.flight.cancel               # writes data/flight/cancel_calibration.json
flock /tmp/snooze-oracle-gpu.lock $PY -m src.flight.model   # smoke test of the interface
```

Inputs: `data/processed/flights_2025-08.parquet` ... `flights_2026-08.parquet`. Outputs go to
`data/flight/` (git-ignored). TabPFN `random_state=0`, LightGBM `random_state=0`.

TabPFN step (full context, 28,672 rows x 6 features, 2,487 test rows, predicted in chunks of
500): fit 1.8 s, total 1,585 s (26.4 min) wall time; peak GPU memory allocated 2,563 MiB,
reserved 4,050 MiB; peak process RSS 2.87 GB; n_estimators = 8 (package default "auto").
The other job on the GPU left only ~1.7 GB free, so WSL most likely spilled part of TabPFN's
memory into shared system memory, which explains the slow run; nothing hit CUDA OOM.

Interface smoke test (`python -m src.flight.model`, after the evaluation, same shared GPU):
`precompute` of 3 queries took 188 s (refit + one predict call), then 2,000 samples each:

| query (origin, carrier, crs_arr_ts) | sample P10 / P50 / P90 | p_cancel |
|---|---|---|
| BQN, B6, 2026-10-26 05:10 | -32.5 / -12.4 / +45.1 | 0.0192 |
| ATL, DL, 2026-10-26 07:00 | -24.0 / -10.3 / +16.9 | 0.0286 |
| XXX, ZZ, 2026-10-26 06:00 (unknown) | -28.0 / -12.9 / +19.8 | 0.0333 |

## Judgment calls

Choices that SPEC.md does not dictate:

1. **Time window 03:00-08:59 by `crs_arr_hhmm`** (300-859), as the task asked, not SPEC's
   "early-morning arrival" 04:00-07:59. Wider window = more context rows; the pickup use case
   covers 03:00-09:00 arrivals.
2. **Months by `flight_date`** (the processed file the row is in), as SPEC's split implies. A
   red-eye dated 2026-07-31 arriving 2026-08-01 is a training row; one dated 2026-08-31 arriving
   2026-09-01 is a test row.
3. **`dow` and `month` from `crs_arr_ts`** (scheduled arrival, NYC local), not from
   `flight_date`: the user of `flight_arrival_samples` gives `crs_arr_ts`, so the features must
   be computable from it.
4. **Diverted flights** are excluded from the delay models (their ArrDelay is null) and count as
   "not cancelled" for p_cancel. 0 diverted rows have an ArrDelay in this window.
5. **Extreme delays kept**: no clipping or winsorising of ArrDelay (max 1,524 min in training).
   The late tail is exactly what the pickup decision cares about.
6. **TabPFN context = all 28,672 flown training rows**, no subsample. It fit on the GPU with the
   default settings; the "5,000 rows" limit in SPEC is for CPU.
7. **Categorical encoding.** TabPFN: origin and carrier as integer codes (sorted alphabetically
   on the training rows), declared via `categorical_features_indices=[0, 1]`. LightGBM: pandas
   `category` dtype (its native categorical handling). Lookup: raw strings. Codes not seen in
   training become NaN for both TabPFN and LightGBM (11 test flights from PDX-AS and STL-YX; no
   test origin or carrier is entirely unseen, only those origin x carrier pairs).
8. **TabPFN quantiles**: `predict(output_type="quantiles")` at 0.1/0.5/0.9 (read off a 203-point
   grid: 0.001, 0.0025, 0.005, 0.010, ..., 0.995, 0.9975, 0.999). Default `n_estimators`
   ("auto" -> 8), default `softmax_temperature`, `random_state=0`. Prediction in chunks of 500
   test rows (TabPFN scores test rows independently, so chunking does not change predictions).
9. **Samples from TabPFN**: inverse-CDF sampling. `flight_arrival_samples` draws
   u ~ Uniform(0.001, 0.999) and linearly interpolates the 203-point quantile grid (made
   monotone with a running max). Tails beyond the 0.1% / 99.9% quantiles are cut off.
10. **Runtime features for a query**: the interface has no destination, so `distance` is the most
    common training distance for the (origin, carrier) cell, else for the origin, else the
    overall median (659 mi). This mixes JFK and LGA distances, which differ by a few miles.
11. **Runtime batching and cache**: one TabPFN predict call takes minutes on the shared GPU even
    for one row (the 28,672-row context dominates), so `precompute([...])` predicts many flights
    in one call and caches the quantile grids in `data/flight/grid_cache.parquet`;
    `flight_arrival_samples` reads that cache and only calls TabPFN on a miss. The model is
    refitted per process from the parquet files; nothing pickled, so no weights on disk.
12. **Lookup tables**: two variants. "lookup origin": empirical quantiles (pandas linear
    interpolation) of the origin's training rows; "lookup origin x carrier": the origin x
    carrier cell. A cell is used only if it has >= 20 training rows; otherwise fall back to the
    origin, then to all training rows. On the test set the origin x carrier lookup used its own
    cell for 2,472 flights and fell back to origin for 15; the origin lookup never needed the
    global fallback.
13. **LightGBM**: `LGBMRegressor(objective="quantile", alpha=q)`, one model per quantile
    (0.1, 0.5, 0.9), all other parameters default, `random_state=0`. `num_threads=4` is a resource
    cap for the shared machine (OOM history), not tuning. Quantile crossing is not corrected.
14. **Schedule-only baseline scored as a point forecast at 0 for every quantile**: P10 and P90
    "coverage" are both the share of flights with `arr_delay <= 0`, and pinball loss uses 0 at
    each quantile.
15. **Pinball quantiles**: 0.1, 0.5, 0.9, reported separately and as their mean.
16. **Density bins**: 0, 1-49, 50-199, 200-799, 800+ flown training flights per origin x
    carrier, chosen from the cell-size distribution (median cell 204 flights) before scoring;
    0 and 1-49 merged in the table (11 + 36 flights).
17. **Day-block bootstrap** for the paired comparison (resample the 32 scheduled-arrival dates, 2026-08-01 ... 2026-09-01), because
    errors within a morning are correlated (weather, ground stops).
18. **p_cancel**: hierarchical shrinkage of counts on all scheduled training rows (03:00-08:59,
    2025-08 ... 2026-07): global rate -> carrier rate shrunk to global -> origin x carrier rate
    shrunk to carrier, each with K = 100 pseudo-flights. Unknown carrier -> global rate; unknown
    cell -> carrier rate. Time of day, day of week and month are ignored (`crs_arr_ts` is
    accepted but unused). Reason: cancellations are rare (3.3% in training) and driven by
    weather days, so per-cell counts are tiny; shrinkage keeps a 7-flight cell from reading
    0% or 14%. K = 100 was set once, not tuned.
19. **No dest feature**: following the task's feature list. JFK and LGA are pooled in all
    models (distance partly tells them apart).

## Fairness

Things that could tilt the comparison:

- **Same rows, same features, but not the same use of them.** The lookups use only origin (and
  carrier); LightGBM and TabPFN also use time of day, day of week, month and distance. That is
  the design of the comparison, not a leak, but the lookups' simplicity is part of what is
  being measured.
- **`month` is nearly constant information for the test**: training has one August (2025-08), so
  "month = 8" tells the models about one past August. Tree models can split on it sharply;
  TabPFN treats it as a number. Neither is tuned for that.
- **Defaults are not equally suited.** LightGBM's defaults (100 trees, 31 leaves, lr 0.1) are a
  generic setting for quantile loss and may overfit or underfit here; TabPFN's defaults are
  designed for small tables. Neither was tuned, as SPEC requires, but "default vs default" is
  not "best vs best".
- **LightGBM fits three separate models** (one per quantile) and can cross; TabPFN and the
  lookups give one coherent distribution. Crossing was not fixed, which can only hurt LightGBM.
- **Categorical handling differs** (LightGBM native categories vs TabPFN integer codes flagged as
  categorical); unseen cells are NaN for both. The lookups fall back explicitly to origin
  instead, which is a hand-made advantage on unseen cells.
- **TabPFN used a 203-point quantile grid read-off** while the others predict 0.1/0.5/0.9
  directly. The grid contains 0.1, 0.5 and 0.9 exactly, so this does not move the numbers.
- **Tiny sparse bin**: 47 test flights, 6 cells, one cell (RDU-AA) is half of them. Sparse-route
  conclusions in either direction are not reliable from one test month.
- **One test month with a regime shift** (heavy late tail, high cancellations). Results could
  differ in a calm month; the ranking between the close models (TabPFN, lookups, LightGBM) is the
  part most likely to change.
- **Schedule-only cannot express spread**, so its P10/P90 coverage is not comparable in kind;
  it is shown as the "human rule" reference.

## Proposed SPEC.md changes (for main, not applied here)

- The flight interface has no destination; JFK vs LGA matters for a pickup. Proposed:
  `flight_arrival_samples(origin, carrier, crs_arr_ts, n=2000, dest=None)` and the same for
  `p_cancel`, so `distance` can be exact and a `dest` feature can be added.
- State the flight window used for modelling (03:00-08:59 by `crs_arr_hhmm`) next to the
  04:00-07:59 "early-morning" definition used in exploration.
- Note that one TabPFN predict call with the ~29k-row flight context costs minutes on the shared
  GPU, so flight predictions should be precomputed in a batch the night before (as `snooze_curve`
  already is).
