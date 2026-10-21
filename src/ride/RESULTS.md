# Ride model: Uber request-to-drop-off time (`total_min`)

Train: trips requested in July 2026. Test: trips requested in August 2026.
All four models are fit on the same 10,000 July rows (`data/ride/context.parquet`) and scored
on the same 10,000 August rows (`data/ride/test.parquet`, 5,000 dense + 5,000 sparse).
Features for the learned models: `pu_zone`, `do_zone`, `req_min_of_day`, `dow`, `is_weekend`.

## Results

Pinball loss is in minutes, averaged over the quantile levels 0.1, 0.25, 0.5, 0.75 and 0.9
(`pinball_mean`, with a 95% bootstrap interval from 1,000 row resamples) and at 0.9 alone
(`pinball_q90`). `p90_coverage` = share of test trips that arrived within the predicted P90
(target 0.90). `mean_q90` = the average predicted P90, i.e. how many minutes the model
tells you to allow. Lower pinball is better.

| split | model | n | p90_coverage | pinball_mean | ci95 | pinball_q90 | mean_q90 |
|---|---|---|---|---|---|---|---|
| dense | fixed_buffer | 5,000 | 0.750 | 6.28 | 6.17-6.39 | 3.72 | 30.0 |
| dense | lookup | 5,000 | 0.895 | 3.34 | 3.26-3.43 | 2.84 | 40.3 |
| dense | lgbm_quantile | 5,000 | 0.876 | 2.78 | 2.70-2.86 | 2.41 | 36.3 |
| dense | **tabpfn** | 5,000 | 0.937 | **2.00** | 1.94-2.05 | **1.58** | 35.5 |
| sparse | fixed_buffer | 5,000 | 0.737 | 6.03 | 5.93-6.13 | 3.63 | 30.0 |
| sparse | lookup | 5,000 | 0.884 | 3.50 | 3.41-3.60 | 2.94 | 40.6 |
| sparse | lgbm_quantile | 5,000 | 0.858 | 3.27 | 3.18-3.36 | 2.74 | 36.8 |
| sparse | **tabpfn** | 5,000 | 0.907 | **2.59** | 2.52-2.66 | **2.02** | 37.3 |
| all (50/50 strata) | fixed_buffer | 10,000 | 0.743 | 6.16 | 6.08-6.23 | 3.68 | 30.0 |
| all (50/50 strata) | lookup | 10,000 | 0.889 | 3.42 | 3.36-3.49 | 2.89 | 40.4 |
| all (50/50 strata) | lgbm_quantile | 10,000 | 0.867 | 3.03 | 2.97-3.09 | 2.57 | 36.5 |
| all (50/50 strata) | **tabpfn** | 10,000 | 0.922 | **2.29** | 2.25-2.34 | **1.80** | 36.4 |
| early 03:00-06:59, dense | fixed_buffer | 369 | 0.740 | 5.51 | 5.16-5.86 | 2.93 | 30.0 |
| early 03:00-06:59, dense | lookup | 369 | 0.905 | 3.08 | 2.83-3.35 | 2.37 | 39.8 |
| early 03:00-06:59, dense | lgbm_quantile | 369 | 0.875 | 2.50 | 2.28-2.71 | 1.95 | 35.5 |
| early 03:00-06:59, dense | **tabpfn** | 369 | 0.959 | **1.76** | 1.64-1.87 | **1.33** | 35.8 |
| early 03:00-06:59, sparse | fixed_buffer | 1,593 | 0.742 | 5.82 | 5.64-6.00 | 3.30 | 30.0 |
| early 03:00-06:59, sparse | lookup | 1,593 | 0.886 | 3.36 | 3.21-3.49 | 2.64 | 39.9 |
| early 03:00-06:59, sparse | lgbm_quantile | 1,593 | 0.876 | 2.90 | 2.78-3.02 | 2.31 | 36.6 |
| early 03:00-06:59, sparse | **tabpfn** | 1,593 | 0.923 | **2.22** | 2.12-2.31 | **1.70** | 36.1 |

Reference only, **not** part of the same-rows comparison: the same lookup rule built from
every July trip (about 14.0M rows, 1,400x TabPFN's context).

| split | model | n | p90_coverage | pinball_mean | ci95 | pinball_q90 | mean_q90 |
|---|---|---|---|---|---|---|---|
| dense | lookup_full_july | 5,000 | 0.887 | 1.81 | 1.76-1.87 | 1.48 | 31.8 |
| sparse | lookup_full_july | 5,000 | 0.864 | 2.45 | 2.37-2.54 | 1.93 | 33.1 |
| all (50/50 strata) | lookup_full_july | 10,000 | 0.876 | 2.13 | 2.08-2.18 | 1.71 | 32.5 |
| early 03:00-06:59, dense | lookup_full_july | 369 | 0.932 | 1.76 | 1.60-1.91 | 1.38 | 34.2 |
| early 03:00-06:59, sparse | lookup_full_july | 1,593 | 0.896 | 2.34 | 2.23-2.45 | 1.76 | 34.8 |

Reading:
- On the same 10k rows, TabPFN has the lowest pinball loss in every split, with
  non-overlapping 95% intervals against LightGBM and the lookup table. It over-covers
  slightly (0.92 vs the 0.90 target overall; 0.94 dense, 0.91 sparse); the baselines
  under-cover (LightGBM 0.87, lookup 0.89).
- The 30-minute rule is too short for about a quarter of trips (coverage 0.74).
- The full-July lookup (reference) still beats TabPFN-on-10k on pinball loss in the dense,
  sparse and overall splits (1.81 vs 2.00, 2.45 vs 2.59, 2.13 vs 2.29). In the early-morning
  splits TabPFN ties it (dense 1.76 vs 1.76) or edges it (sparse 2.22 vs 2.34, intervals
  overlap). TabPFN's P90 coverage is higher than the full lookup's in every split.
  So with 10k rows TabPFN gets most of the way to a 14M-row table, but does not beat it
  outright.

## TabPFN run and checks

- Main run (`tabpfn_model`, memory-saving mode on, predict chunks of 500 rows):
  fit 3.2 s; predicting the 10,000 test rows took 277 s; 281 s of GPU time in total.
  Peak GPU memory: 2,595 MiB allocated, 3,580 MiB reserved by PyTorch. Peak process RSS
  2.9 GB. (The 454 s wall time from `time` includes waiting for the shared GPU lock.)
- Memory-saving mode did **not** lower peak GPU memory (200-row check: 2,521 MiB on vs
  2,520 MiB off). It also did not give bit-identical predictions: across 200 test rows x
  5 quantiles the max absolute difference between on and off was 0.26 min and the mean
  0.02 min. That is about 1% of the pinball loss and does not change any ranking above,
  but it was not run on/on to tell mode effects apart from GPU run-to-run noise.
- `ride_total_samples(7, 138, 2026-08-12 05:30, n=2000)` (Astoria -> LaGuardia, Wednesday):
  2,000 finite values, min 3.3, P10/P50/P90 = 11.1 / 16.0 / 24.1, max 52.6 min; 40 s per
  call including the fit. The checks process peaked at 3.8 GB RSS (two models plus the
  sampler's model in one process).

## How to reproduce

From the repo root, with the project venv's python as `python`:

```bash
python -m src.ride.data                                          # context + test samples -> data/ride/
python -m src.ride.baselines                                     # fixed buffer, lookup, LightGBM, full-July lookup
flock /tmp/snooze-oracle-gpu.lock python -u -m src.ride.tabpfn_model   # TabPFN (GPU, shared lock)
python -m src.ride.evaluate                                      # results table -> data/ride/results_table.md
flock /tmp/snooze-oracle-gpu.lock python -u -m src.ride.checks   # memory-mode check + sampler smoke test
```

`tabpfn_model` writes one file per 500-row chunk and skips finished chunks, so after a crash
rerun the same command. Delete `data/ride/preds/tabpfn_chunks/` to start over.

## Judgment calls

Choices SPEC.md does not make, with the reason for each.

1. **Context size: 10,000 rows**, sampled uniformly at random from July trips (DuckDB reservoir
   sample, seed 20261021). Big enough to help every model; at 10k rows TabPFN peaks at about
   2.6 GB of GPU memory, already more than the ~1.9 GB free on the shared card. Uniform is the
   honest default: no oversampling of sparse cells.
2. **Strict month split on the request time.** The processed files hold a few trips requested
   in the last minutes of the previous month; they are dropped from both train and test.
3. **Density is defined per origin cell (`pu_zone`, `req_hour`, `is_weekend`)**, counted over
   all of July. A cell with fewer than 100 trips is **sparse** (0.5% of August trips); every
   other cell is **dense**. Zone alone cannot separate them: every pickup zone had at least
   44 trips in July and the median zone had 42,000. Adding hour picks out the rare
   early-morning and outlying cases the product is about.
4. **Test sample: 5,000 rows per stratum**, uniform within the stratum (seeds 20261022/23).
   The "all" row therefore weights the strata 50/50, not by real trip share. Also reported:
   requests at 03:00-06:59 in each stratum (369 dense, 1,593 sparse rows).
5. **Quantile levels 0.1, 0.25, 0.5, 0.75, 0.9.** Pinball loss is their average; pinball at
   0.9 alone is also reported, since P90 is the level the snooze rule uses. A 95% bootstrap
   interval (1,000 row resamples, seed 0) is given for the average.
6. **Human rule = a constant 30 minutes**, from SPEC.md's "leave 30 min early" example. It is
   the same number at every quantile level, i.e. a point mass.
7. **Lookup back-off:** a cell needs at least 10 rows. Otherwise use the route
   (`pu_zone`, `do_zone`), then the origin zone, then all rows. Quantiles use linear
   interpolation (numpy and DuckDB default). On the 10k context no test row found a full
   cell: 159 used the route, 6,414 the origin zone, 3,427 the global quantiles.
8. **Zone ids are declared categorical** for both TabPFN (`categorical_features_indices`) and
   LightGBM (pandas category, fixed category list for zones 1-263). They are labels, not
   magnitudes. Other features are numeric.
9. **LightGBM:** one model per quantile level, `objective="quantile"`, `alpha=q`, defaults
   otherwise. `n_jobs=4` is a resource cap for this shared machine, not tuning. Crossing
   quantiles are left as predicted.
10. **TabPFN:** package defaults (model version 3.5, `n_estimators="auto"` -> 8,
    `random_state=0`, autocast precision). Quantiles from `predict(output_type="quantiles")`.
    `memory_saving_mode=True` and 500-row predict chunks were set to fit the shared GPU; the
    on/off check above shows differences of at most 0.26 min.
11. **A reference row: the lookup built from all of July**, so readers can see what the
    subsample costs the lookup table. It is labelled and kept out of the main table.
12. **`ride_total_samples`:** TabPFN predictive quantiles at levels (i + 0.5) / n for
    i = 0..n-1 (inverse-CDF sampling on an even grid), raised to at least 1.0 min (the
    processed table has `ride_min >= 1`), then shuffled. Shuffling matters downstream: sorted
    arrays paired index-wise with other samples would be perfectly correlated. An optional
    `rng` argument (not in the SPEC signature) makes the order reproducible; do not reuse one
    seeded order for two arrays you pair. The model is fit lazily on the same context and
    cached in the process.

## Fairness

- **The 10k subsample hurts the lookup table most.** It needs data in each cell and almost
  never had any, so it fell back to coarse cells for every test row. With all of July it
  beats TabPFN on most splits (reference table). "Lookup" in the main table is a 10k-row
  lookup, not the rule as normally used.
- **LightGBM defaults are tuned for larger data.** `min_child_samples=20` and categorical
  `min_data_per_group=100` mean most zones cannot get their own split with 10k rows.
- **The 30-minute rule is a point, not a distribution**, so its average pinball loss is
  structurally poor. Compare it on P90 coverage and `pinball_q90`.
- **Only TabPFN got `memory_saving_mode` and chunked prediction.** Neither changes its
  settings, but predictions moved by up to 0.26 min in the on/off check.
- **Uniform context sampling** gives sparse cells almost no context rows. That is the same
  for all models, but it is a choice: a sample stratified by origin would likely help every
  model on sparse origins, and might help some more than others.
- **Coverage vs. loss trade-off.** TabPFN over-covers a little (0.92) while LightGBM and the
  lookup under-cover. A model that is late less often looks better on coverage alone;
  read coverage and pinball together.
- **TabPFN gets a GPU and 8 ensemble members by default**; the CPU baselines finished in
  under 1 s. The comparison is accuracy at defaults, not cost.
- **Test strata are weighted 50/50**, so the "all" row overweights sparse origins (0.5% of
  real trips).
