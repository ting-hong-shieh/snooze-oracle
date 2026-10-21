# Ride model, round 2: product-aligned evaluation

Round 1 (`RESULTS.md`) scored every model on one uniform 10k-row context over all of NYC.
Round 2 scores only what the product uses: Uber trips to JFK (zone 132) or LaGuardia (138)
requested 03:00-08:59. Train = July 2026 trips in that setting (224,923 rows), test =
August 2026 trips in that setting (220,955 rows, of which 59,321 are scored; see
judgment call 11).

- **TabPFN-3.5** (package defaults) is fit once per retrieval context: same
  `pu_zone -> do_zone` within +-2 request hours; if that has fewer than 500 rows, the same
  `pu_borough -> do_zone` in the same hours; capped at 10,000 rows. 707 contexts.
- **Lookup** (round-1 rule) and **LightGBM quantile** (defaults, 4 threads) use every July
  row in the setting. **30 minutes** is kept as a reference.
- Features as in round 1: `pu_zone`, `do_zone`, `req_min_of_day`, `dow`, `is_weekend`.
- Density bin = July trips on the route (`pu_zone -> do_zone`) in the setting.

## Results

95% intervals in brackets come from a day-level bootstrap (2,000 resamples of the 31
August days, judgment call 15). Pinball loss in minutes; lower is better. `pinball_mean`
averages the levels 0.1, 0.25, 0.5, 0.75, 0.9. `mean_q90` = the average predicted P90.
`train_rows` = rows each model learned from for these test rows: LightGBM always uses all
224,923; for the lookup it is the rows in the cell that answered (median [min-max]); for
TabPFN the context size per test row (median [min-max]).

| density | model | n_test | train_rows | p90_coverage | pinball_q90 | pinball_mean | mean_q90 |
|---|---|---|---|---|---|---|---|
| <30 | fixed_buffer | 903 | 0 | 0.169 [0.146, 0.192] | 15.23 [13.94, 16.65] | 8.99 [8.31, 9.74] | 30.0 [30.0, 30.0] |
| <30 | lookup | 903 | median 21 [10-224,923] | 0.866 [0.834, 0.893] | 2.48 [2.12, 2.98] | 3.13 [2.86, 3.50] | 57.6 [56.9, 58.3] |
| <30 | lgbm_quantile | 903 | 224,923 (all) | 0.827 [0.795, 0.859] | 2.37 [1.96, 2.90] | 2.96 [2.67, 3.34] | 55.1 [53.7, 56.5] |
| <30 | **tabpfn** | 903 | median 10,000 [66-10,000] | 0.895 [0.865, 0.918] | **2.03** [1.76, 2.36] | **2.57** [2.35, 2.83] | 57.1 [55.4, 58.8] |
| 30-199 | fixed_buffer | 18,006 | 0 | 0.338 [0.321, 0.354] | 8.32 [7.80, 8.84] | 5.52 [5.27, 5.77] | 30.0 [30.0, 30.0] |
| 30-199 | lookup | 18,006 | median 27 [10-198] | 0.882 [0.865, 0.898] | 1.67 [1.61, 1.75] | 2.11 [2.05, 2.17] | 46.6 [46.4, 46.8] |
| 30-199 | lgbm_quantile | 18,006 | 224,923 (all) | 0.876 [0.858, 0.892] | 1.61 [1.52, 1.71] | 2.02 [1.94, 2.10] | 45.5 [44.6, 46.4] |
| 30-199 | **tabpfn** | 18,006 | median 10,000 [218-10,000] | 0.889 [0.872, 0.904] | **1.57** [1.48, 1.67] | **1.99** [1.92, 2.08] | 46.0 [45.1, 46.9] |
| 200-999 | fixed_buffer | 20,933 | 0 | 0.382 [0.367, 0.396] | 7.26 [6.76, 7.73] | 5.23 [4.99, 5.46] | 30.0 [30.0, 30.0] |
| 200-999 | lookup | 20,933 | median 65 [10-985] | 0.877 [0.858, 0.896] | 1.55 [1.44, 1.69] | 1.92 [1.84, 2.02] | 43.5 [42.7, 44.3] |
| 200-999 | lgbm_quantile | 20,933 | 224,923 (all) | 0.885 [0.865, 0.902] | 1.50 [1.39, 1.64] | 1.88 [1.79, 1.98] | 43.5 [42.6, 44.3] |
| 200-999 | tabpfn | 20,933 | median 10,000 [500-10,000] | 0.888 [0.870, 0.905] | 1.50 [1.38, 1.64] | 1.88 [1.79, 1.99] | 43.6 [42.7, 44.5] |
| 1000+ | fixed_buffer | 19,479 | 0 | 0.609 [0.589, 0.630] | 3.92 [3.63, 4.20] | 4.46 [4.36, 4.58] | 30.0 [30.0, 30.0] |
| 1000+ | lookup | 19,479 | median 193 [10-1,827] | 0.890 [0.871, 0.906] | 1.31 [1.23, 1.40] | 1.61 [1.55, 1.69] | 36.1 [35.4, 36.8] |
| 1000+ | lgbm_quantile | 19,479 | 224,923 (all) | 0.896 [0.879, 0.911] | 1.29 [1.20, 1.39] | 1.60 [1.54, 1.68] | 36.2 [35.5, 36.9] |
| 1000+ | tabpfn | 19,479 | median 1,111 [502-10,000] | 0.894 [0.877, 0.909] | 1.29 [1.20, 1.39] | 1.61 [1.54, 1.68] | 36.2 [35.5, 36.9] |
| all (weighted to pool) | fixed_buffer | 59,321 | 0 | 0.478 [0.461, 0.494] | 5.90 [5.52, 6.27] | 4.93 [4.77, 5.10] | 30.0 [30.0, 30.0] |
| all (weighted to pool) | lookup | 59,321 | median 77 [10-224,923] | 0.883 [0.866, 0.899] | 1.46 [1.37, 1.56] | 1.81 [1.74, 1.88] | 40.6 [39.9, 41.2] |
| all (weighted to pool) | lgbm_quantile | 59,321 | 224,923 (all) | 0.889 [0.871, 0.904] | 1.42 [1.33, 1.53] | 1.77 [1.70, 1.86] | 40.5 [39.7, 41.2] |
| all (weighted to pool) | tabpfn | 59,321 | median 3,535 [66-10,000] | 0.891 [0.874, 0.905] | 1.41 [1.32, 1.54] | 1.77 [1.70, 1.86] | 40.6 [39.8, 41.4] |

"all (weighted to pool)" weights each scored row by (August rows in its density bin and
day) / (scored rows in that bin and day), so it estimates the full 220,955-row August
pool. Its `train_rows` medians are over scored rows, unweighted.

### Paired differences: TabPFN minus baseline (same test rows)

Negative pinball = TabPFN better; positive coverage = TabPFN covers more often.

| density | comparison | n_test | p90_coverage | pinball_q90 | pinball_mean |
|---|---|---|---|---|---|
| <30 | tabpfn - fixed_buffer | 903 | 0.725 [0.698, 0.751] | -13.21 [-14.53, -12.02] | -6.42 [-7.02, -5.87] |
| <30 | tabpfn - lookup | 903 | 0.029 [0.003, 0.051] | -0.45 [-0.77, -0.21] | -0.56 [-0.74, -0.39] |
| <30 | tabpfn - lgbm_quantile | 903 | 0.068 [0.046, 0.090] | -0.34 [-0.64, -0.12] | -0.39 [-0.55, -0.27] |
| 30-199 | tabpfn - fixed_buffer | 18,006 | 0.551 [0.530, 0.570] | -6.75 [-7.24, -6.26] | -3.53 [-3.72, -3.32] |
| 30-199 | tabpfn - lookup | 18,006 | 0.007 [-0.010, 0.023] | -0.11 [-0.17, -0.03] | -0.11 [-0.16, -0.06] |
| 30-199 | tabpfn - lgbm_quantile | 18,006 | 0.013 [0.010, 0.017] | -0.04 [-0.06, -0.02] | -0.03 [-0.04, -0.02] |
| 200-999 | tabpfn - fixed_buffer | 20,933 | 0.507 [0.486, 0.525] | -5.76 [-6.21, -5.30] | -3.35 [-3.54, -3.16] |
| 200-999 | tabpfn - lookup | 20,933 | 0.011 [0.005, 0.016] | -0.05 [-0.08, -0.03] | -0.04 [-0.06, -0.02] |
| 200-999 | tabpfn - lgbm_quantile | 20,933 | 0.004 [0.001, 0.006] | -0.00 [-0.01, 0.01] | 0.00 [-0.01, 0.01] |
| 1000+ | tabpfn - fixed_buffer | 19,479 | 0.285 [0.265, 0.304] | -2.63 [-2.88, -2.36] | -2.86 [-2.94, -2.77] |
| 1000+ | tabpfn - lookup | 19,479 | 0.005 [-0.001, 0.011] | -0.01 [-0.03, 0.00] | -0.01 [-0.02, 0.01] |
| 1000+ | tabpfn - lgbm_quantile | 19,479 | -0.002 [-0.004, 0.001] | 0.00 [-0.00, 0.01] | 0.00 [-0.00, 0.01] |
| all (weighted to pool) | tabpfn - fixed_buffer | 59,321 | 0.413 [0.395, 0.431] | -4.49 [-4.83, -4.13] | -3.16 [-3.29, -3.03] |
| all (weighted to pool) | tabpfn - lookup | 59,321 | 0.008 [0.003, 0.013] | -0.04 [-0.06, -0.02] | -0.03 [-0.05, -0.01] |
| all (weighted to pool) | tabpfn - lgbm_quantile | 59,321 | 0.002 [0.001, 0.004] | -0.00 [-0.01, 0.00] | -0.00 [-0.01, 0.00] |

### TabPFN contexts per density bin

| density | n_test | contexts | context rows p10/p50/p90 (per scored test row) | widened to borough | hit 10k cap | computed on CPU | lookup used its full cell |
|---|---|---|---|---|---|---|---|
| <30 | 903 | 58 | 129/10,000/10,000 | 100.0% | 50.2% | 0.0% | 0.7% |
| 30-199 | 18,006 | 54 | 3,381/10,000/10,000 | 100.0% | 55.2% | 0.0% | 63.3% |
| 200-999 | 20,933 | 310 | 559/10,000/10,000 | 61.8% | 58.7% | 10.5% | 96.5% |
| 1000+ | 19,479 | 391 | 727/1,111/1,926 | 1.0% | 1.0% | 75.7% | 99.8% |
| all scored rows (unweighted) | 59,321 | 707 | 644/3,535/10,000 | 54.0% | 38.6% | 28.6% | 86.1% |

Contexts: 647 route-level (500-3,572 rows, median 768, none capped) and 60 borough-level
(66-10,000 rows; 36 capped at 10,000 from up to 13,648 matches; 12 Staten Island contexts
stayed under 500 rows, 66-450, covering 728 test rows). No test query had an empty context.

Lookup back-off over the 59,321 scored rows: 51,049 full cell, 8,096 route, 142 origin
zone, 34 global.

### Reading

- **Sparse routes (<30 trips/month, 70 routes, 903 August trips):** TabPFN is the only
  model whose CI covers 0.90 coverage, and it has the lowest pinball loss with paired CIs
  that exclude zero: mean pinball 0.56 min lower than the lookup and 0.39 min lower than
  LightGBM (18% / 13%); q90 pinball 0.45 / 0.34 min lower. LightGBM under-covers there
  (0.83).
- **30-199:** TabPFN is still ahead, but by little: -0.11 min vs lookup, -0.03 min vs
  LightGBM (both CIs exclude zero).
- **200+ trips/month (91% of August trips in the setting):** TabPFN and LightGBM are tied
  (paired differences 0.00, CIs spanning zero). TabPFN is 0.01-0.05 min better than the
  lookup (CIs exclude zero in 200-999, touch zero in 1000+). Differences this small are
  of the same order as the CPU/GPU difference below.
- Overall (weighted to the real August mix) TabPFN and LightGBM are tied (1.77 vs 1.77,
  paired difference -0.00); both beat the lookup by 0.03-0.04 min. All three learned models
  cover 0.88-0.89, slightly under the 0.90 target, while the 30-minute rule covers 0.48:
  half of these early-morning airport trips take longer than 30 minutes.
- So retrieval puts TabPFN level with a LightGBM that sees 225k rows, and ahead of it
  where data is thin, which is where the product's rare routes live. It does not beat the
  full-data baselines on busy routes.

## TabPFN run

- Fits + predicts that finished: 707 contexts. 378 on the GPU: 2,172 s in total
  (1,523 s without one context, `route_107_138_5`, whose predict took 650 s while the card
  was paging; median 0.8 s per context; the 36 10k-row contexts took 13-98 s each, median 23 s). 329 on
  the CPU (4 threads): 9,749 s across all CPU runs, including 54 contexts computed twice
  (judgment call 19).
- GPU lock time was far higher than the compute: 73 GPU attempts stalled and were killed
  by the worker's watchdog after 60 s (66) or 90 s (7), about 4,600 s; one earlier stall
  held the lock about 25 minutes before I killed it; plus about 6 minutes of benchmarks.
  Roughly 2.4 hours of lock time in total, mostly stalls. Apart from the 650 s paging
  case and the 25-minute stall, each worker held the lock for at most ~150 s of work plus
  at most one 60-90 s stall, and the driver waited 10-120 s between workers; the flight
  workstream ran its jobs in between.
- Peak GPU memory: 2,567 MiB allocated, 3,750 MiB reserved by PyTorch (10k-row contexts).
  Small route contexts: 1.1-2.5 GB reserved; real footprint ~1.5 GB for a 640-row context
  (CUDA context ~290 MiB + weights ~850 MiB + activations).
- Peak RSS: GPU workers 3,745 MiB before the built-model cache was switched on (judgment
  call 17), 2,683 MiB after; CPU workers up to 3,998 MiB; `r2_checks device` 3,727 MiB;
  `r2_checks sampler` 2,639 MiB; data/baselines/evaluate under 0.5 GB. **The GPU workers
  before the change, the CPU workers and the device check went over the ~3 GB per-process
  limit** (the TabPFN checkpoint is ~0.9 GB and is held twice while the model is built).
- CPU vs GPU check (`r2_checks device`): 21 GPU-computed contexts rerun on the CPU plus 6
  CPU-computed contexts rerun on the GPU: mean absolute difference 0.008 min per predicted
  quantile; per-context max 0.01-0.62 min (3 of 27 contexts above 0.2 min, all
  route contexts: 0.56, 0.33 and 0.62 min).
- `ride_total_samples(7, 138, 2026-08-12 05:30, n=2000)` on the GPU with the round-2
  retrieval: 13.2 s including the fit; 2,000 finite values, min 5.4, P10/P50/P90 =
  11.7 / 16.4 / 22.6, max 44.9 min. A query outside the setting (do_zone 1) raises
  `ValueError`.

## How to reproduce

From the repo root, with the project venv's python as `python`:

```bash
python -m src.ride.r2_data          # train/test in the setting, density bins, test sample -> data/ride2/
python -m src.ride.r2_baselines     # fixed 30 min, full-data lookup, LightGBM (all July rows)
python -m src.ride.r2_tabpfn drive  # TabPFN per context; workers take the GPU lock themselves
python -m src.ride.r2_evaluate      # tables -> data/ride2/{results_table,paired_diffs,context_sizes}.md
python -m src.ride.r2_checks device   # CPU vs GPU difference (CPU only)
python -m src.ride.r2_checks sampler  # ride_total_samples smoke test (takes the GPU lock)
```

`r2_tabpfn drive` starts one short worker at a time; a worker computes contexts for about
150 s and exits, finished contexts (`data/ride2/tabpfn/<key>.parquet`) are skipped on a
rerun, and contexts the GPU cannot run go to a CPU pass at the end. Extra CPU workers can
run alongside with `python -m src.ride.r2_tabpfn work --device cpu [--reverse]` (claim
files prevent double work). Because the device split depended on how full the shared GPU
was, a rerun may put different contexts on the CPU; the check above bounds what that
changes. Delete `data/ride2/tabpfn*` to start over.

## Judgment calls

Choices SPEC.md and the task do not make, with the reason for each.

1. **Setting filter:** `do_zone IN (132, 138)` and `req_min_of_day` 180-539 (requests
   03:00:00-08:59:59), on the request time; strict month split on `request_ts` as in
   round 1.
2. **"+-2 request hours" = request-hour buckets `req_hour` in [h-2, h+2]**, not +-120
   minutes around the query minute. All queries in the same hour then share one context,
   so there is one fit per (route or borough, destination, hour). The band spans 2-3
   hours on each side of a query rather than exactly 2.
3. **No wrap and nothing outside the window.** Step 1 restricts training rows to
   03:00-08:59, so retrieval only sees those rows: a 03:xx query sees hours 3-5 and an
   08:xx query hours 6-8 (3 hours instead of 5). Night trips (01:00-02:59) and late-morning
   trips (09:00-10:59) are never in a context.
4. **The 500-row threshold counts rows in the hour band**, not the whole route. The cap
   check also counts the band.
5. **Borough widening** uses `pu_borough` from the processed table (the taxi zone lookup
   in `ride_total_samples`). The borough context includes the query's own route. There is
   no third level: when the borough is still under 500 rows (12 Staten Island contexts,
   66-450 rows) it is used as is.
6. **Cap:** uniform random sample without replacement, seeded by the CRC32 of the context
   name, so a context always gets the same rows. 36 borough contexts were capped.
7. **Weekdays and weekends share a context**; `dow` and `is_weekend` are features. This
   follows the retrieval rule as written (route + hours only).
8. **Features unchanged from round 1**, zone ids categorical. In a route context `pu_zone`
   and `do_zone` are constant, and `do_zone` is constant in every context; they were kept
   so that one feature set serves both levels (they matter in borough contexts).
9. **Density** = July trips on the route in the setting, both day types, all six hours.
   Routes with no July trips (2 routes, 4 August trips) fall in `<30`.
10. **Default `memory_saving_mode` ("auto")**, as the task asked; round 1 forced it on.
    One predict call per context (chunk 1,000 rows; only the three largest contexts with
    1,005-1,119 queries needed two). A GPU benchmark showed the fixed cost per predict call
    (the context passes through the model again on every call) dominates: a 10k context
    took 22-35 s for 10 or 587 rows alike.
11. **Test subsample:** every August row in the `<30` and `30-199` bins; in `200-999` and
    `1000+`, round(0.2 x rows) per day, uniform within the day (seed 20261021). 59,321 of
    220,955 rows; every model is scored on these rows. Sizing: GPU time depends on the
    number of contexts (707, same with or without subsampling) more than on test rows; 20%
    kept the estimate near 30 minutes on an idle card. The "all" row reweights to the full
    pool.
12. **Lookup back-off** = round-1 rule (cell needs >= 10 rows; then route, origin zone,
    global), computed on all July rows in the setting. The origin-zone and global levels
    mix both airports and all six hours. A route with no July rows gets the origin zone's
    quantiles, or the global ones.
13. **LightGBM** trained on all 224,923 rows with the round-1 zone category list (1-263).
    A pickup zone unseen in July falls into LightGBM's default branch for unseen
    categories. 222 scored rows have crossing quantiles; left as predicted.
14. **Fixed rule = 30 minutes**, same as round 1, reference only.
15. **Bootstrap:** 2,000 replicates; each resamples the 31 August days with replacement
    and keeps every scored row of a drawn day; metrics are ratio estimators (sum over drawn
    rows / count); the same day draws serve all models and bins; 2.5/97.5 percentiles
    (seed 0). Paired differences bootstrap the per-row difference the same way.
16. **Coverage and pinball conventions** as round 1 (`y <= q90`; pinball on the five
    levels).
17. **Built-model cache on after 99 contexts** (`TABPFN_MODEL_CACHE_SIZE=1`, and the
    package's cached CPU checkpoint cleared after the first build): the worker kept one
    built model instead of rebuilding it from the checkpoint for every context. That cut
    host RSS from 3.7 to 2.7 GB and fit time from ~0.5-3 s to ~0.15 s. Weights are not
    changed by inference; the CPU/GPU check covers contexts from before and after.
18. **CPU fallback for 329 contexts (28.6% of scored rows, 75.7% of the `1000+` bin).**
    After about 100 contexts, GPU predicts started hanging inside a CUDA allocation even
    when `nvidia-smi` showed 2+ GB free (WSL pages GPU memory instead of raising OOM). A
    context that hung twice went to the CPU, and from then on every route context with
    >= 1,000 rows (later >= 700) went to the CPU directly. Same package, settings and
    seed; only the device differs (difference above: mean 0.008 min). This was decided by
    how the shared card behaved, not by any result.
19. **Duplicates:** 54 contexts were computed twice (two of my drivers ran at once by
    mistake for ~1 hour, and two CPU workers overlapped once); the last result is used.
    Both copies came from the same code and settings.
20. **`ride_total_samples` now uses the round-2 retrieval** and raises `ValueError` outside
    the setting (do_zone not 132/138, or request outside 03:00-08:59). It keeps one fitted
    model (the last context) in memory. Round-1 `checks.py` now calls this sampler; its
    recorded round-1 numbers in `RESULTS.md` came from the old sampler.

## Fairness

- **Training data differ by design:** LightGBM and the lookup see all 224,923 July rows;
  TabPFN sees at most 10,000 per query, chosen by retrieval. On busy routes TabPFN's
  context is that route only (median 1,111 rows), so it cannot borrow from other routes,
  while LightGBM can. In sparse bins TabPFN's borough context (often 10,000 rows) carries
  more nearby-route data than the lookup's cell (median 21-27 rows).
- **Retrieval was given, not tuned**, but its windows (+-2 h, 500, 10k) differ from the
  lookup's cell (exact hour x weekend). Some of TabPFN's edge in sparse bins may come from
  the wider pooling rather than the model; a lookup on the same retrieval context was not
  run.
- **Mixed devices for TabPFN** (judgment call 18): 29% of scored rows came from the CPU.
  The measured difference (mean 0.008 min) is small next to the paired differences in the
  sparse bins (0.03-0.56 min) but of the same order as the near-zero differences in the
  dense bins, so read the dense-bin ties as ties.
- **The `<30` bin is small:** 903 trips on 70 routes; its CIs are wide and a few routes
  dominate it. The 12 Staten Island contexts (728 test rows, across bins) had under 500
  rows even after widening.
- **Density bins come from July counts**, the same data the models learn from. Routes
  absent in July (4 test rows) are in `<30`.
- **LightGBM defaults** (`min_child_samples=20`, categorical `min_data_per_group=100`)
  limit how finely it can split on rare zones; the lookup's 10-row rule is from round 1.
  Neither was tuned, same as TabPFN.
- **The 30-minute rule** is a point mass, so its mean pinball is structurally poor;
  compare it on coverage and q90 pinball.
- **Dense-bin test rows are a 20% sample** (same rows for every model); the weighted "all"
  row corrects the mix, but its CI reflects the sample.
- **Cost is not compared:** the baselines finished in 3.5 s on the CPU; TabPFN needed
  ~0.6 GPU-hours of compute plus ~2.7 CPU-hours here. Predictions are precomputed nightly
  in the product, so this is a batch cost.
