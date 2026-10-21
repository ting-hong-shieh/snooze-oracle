# Ride round 3: retrieval control and GPU-only TabPFN rerun

Test rows: the round-2 sample (59,321 August trips to JFK/LGA requested 03:00–08:59; every
row in <30 and 30–199, 20% per day in the denser bins). Metrics and day-level bootstrap
(2,000 resamples of the 31 August days) as in round 2. Pinball in minutes, mean over the
0.1/0.25/0.5/0.75/0.9 quantiles.

TabPFN context (SPEC "Ride TabPFN context"): all July rows of the query's own route in the
±2 h band first, then pu_borough -> do_zone rows sampled uniformly up to 10,000. One context
per (route, request hour): 2,914 contexts, all on the GPU (no CPU fallback), median 10,000
rows (own-route rows median 177; 657 contexts under 10,000 because the borough pool ran out,
minimum 66). GPU fit+predict 7,128 s; wall 7,470 s; peak GPU 2,460 MiB allocated; peak RSS
2,743 MiB.

## Step 1: retrieval control (<30 and 30–199)

`lookup_ctx` / `lgbm_ctx` are fit on exactly the TabPFN context of each (route, hour);
`*_full` are the round-2 baselines on all 224,923 July rows.

| density | model | n_test | p90_coverage | pinball_q90 | pinball_mean |
|---|---|---|---|---|---|
| <30 | tabpfn_r3 | 903 | 0.895 [0.867, 0.918] | 1.96 [1.69, 2.29] | 2.49 [2.27, 2.75] |
| <30 | lookup_ctx | 903 | 0.862 [0.825, 0.892] | 2.35 [1.97, 2.89] | 2.97 [2.69, 3.36] |
| <30 | lgbm_ctx | 903 | 0.808 [0.773, 0.839] | 2.34 [1.98, 2.82] | 2.78 [2.52, 3.11] |
| <30 | lookup_full | 903 | 0.866 [0.834, 0.893] | 2.48 [2.12, 2.98] | 3.13 [2.86, 3.50] |
| <30 | lgbm_full | 903 | 0.827 [0.795, 0.859] | 2.37 [1.96, 2.90] | 2.96 [2.67, 3.34] |
| 30-199 | tabpfn_r3 | 18,006 | 0.891 [0.875, 0.907] | 1.56 [1.47, 1.66] | 1.98 [1.91, 2.06] |
| 30-199 | lookup_ctx | 18,006 | 0.880 [0.864, 0.896] | 1.66 [1.59, 1.73] | 2.09 [2.04, 2.16] |
| 30-199 | lgbm_ctx | 18,006 | 0.851 [0.832, 0.869] | 1.65 [1.55, 1.76] | 2.04 [1.97, 2.13] |
| 30-199 | lookup_full | 18,006 | 0.882 [0.865, 0.898] | 1.67 [1.61, 1.75] | 2.11 [2.05, 2.17] |
| 30-199 | lgbm_full | 18,006 | 0.876 [0.858, 0.892] | 1.61 [1.52, 1.71] | 2.02 [1.94, 2.10] |

| density | comparison | n_test | p90_coverage | pinball_q90 | pinball_mean |
|---|---|---|---|---|---|
| <30 | tabpfn_r3 - lookup_ctx | 903 | 0.033 [0.011, 0.056] | -0.40 [-0.76, -0.11] | -0.48 [-0.68, -0.30] |
| <30 | tabpfn_r3 - lgbm_ctx | 903 | 0.086 [0.068, 0.106] | -0.39 [-0.64, -0.19] | -0.29 [-0.39, -0.21] |
| <30 | lookup_ctx - lookup_full | 903 | -0.004 [-0.022, 0.012] | -0.13 [-0.28, 0.04] | -0.16 [-0.28, -0.04] |
| <30 | lgbm_ctx - lgbm_full | 903 | -0.019 [-0.041, 0.003] | -0.03 [-0.16, 0.11] | -0.18 [-0.27, -0.10] |
| <30 | tabpfn_r3 - lookup_full | 903 | 0.029 [0.003, 0.054] | -0.52 [-0.81, -0.29] | -0.64 [-0.82, -0.47] |
| <30 | tabpfn_r3 - lgbm_full | 903 | 0.068 [0.046, 0.090] | -0.41 [-0.66, -0.23] | -0.47 [-0.62, -0.35] |
| 30-199 | tabpfn_r3 - lookup_ctx | 18,006 | 0.011 [-0.004, 0.025] | -0.10 [-0.16, -0.02] | -0.11 [-0.16, -0.06] |
| 30-199 | tabpfn_r3 - lgbm_ctx | 18,006 | 0.041 [0.036, 0.046] | -0.09 [-0.13, -0.06] | -0.06 [-0.07, -0.05] |
| 30-199 | lookup_ctx - lookup_full | 18,006 | -0.001 [-0.003, 0.001] | -0.02 [-0.02, -0.01] | -0.01 [-0.02, -0.00] |
| 30-199 | lgbm_ctx - lgbm_full | 18,006 | -0.025 [-0.029, -0.020] | 0.04 [0.02, 0.07] | 0.02 [0.01, 0.03] |
| 30-199 | tabpfn_r3 - lookup_full | 18,006 | 0.010 [-0.007, 0.026] | -0.11 [-0.18, -0.04] | -0.13 [-0.17, -0.07] |
| 30-199 | tabpfn_r3 - lgbm_full | 18,006 | 0.016 [0.013, 0.019] | -0.05 [-0.07, -0.03] | -0.04 [-0.05, -0.03] |

Lookup on the r3 context, back-off level used per test row (cell / route / pickup zone / whole context): [11396, 7210, 0, 303]

Reading: fitting the lookup or LightGBM on the retrieval context barely changes them
(lookup −0.16, LightGBM −0.18 mean pinball on <30; ≈0 on 30–199), while TabPFN on the same
context is better than both by −0.48 and −0.29 on <30 and −0.11 and −0.06 on 30–199, with
intervals excluding zero. The sparse-route gain comes from TabPFN, not from the retrieval.

## Step 2: GPU-only rerun vs round 2, every density bin

| density | model | n_test | p90_coverage | pinball_q90 | pinball_mean |
|---|---|---|---|---|---|
| <30 | tabpfn_r3 | 903 | 0.895 [0.867, 0.918] | 1.96 [1.69, 2.29] | 2.49 [2.27, 2.75] |
| <30 | tabpfn_r2 | 903 | 0.895 [0.865, 0.918] | 2.03 [1.76, 2.36] | 2.57 [2.35, 2.83] |
| <30 | lookup_full | 903 | 0.866 [0.834, 0.893] | 2.48 [2.12, 2.98] | 3.13 [2.86, 3.50] |
| <30 | lgbm_full | 903 | 0.827 [0.795, 0.859] | 2.37 [1.96, 2.90] | 2.96 [2.67, 3.34] |
| 30-199 | tabpfn_r3 | 18,006 | 0.891 [0.875, 0.907] | 1.56 [1.47, 1.66] | 1.98 [1.91, 2.06] |
| 30-199 | tabpfn_r2 | 18,006 | 0.889 [0.872, 0.904] | 1.57 [1.48, 1.67] | 1.99 [1.92, 2.08] |
| 30-199 | lookup_full | 18,006 | 0.882 [0.865, 0.898] | 1.67 [1.61, 1.75] | 2.11 [2.05, 2.17] |
| 30-199 | lgbm_full | 18,006 | 0.876 [0.858, 0.892] | 1.61 [1.52, 1.71] | 2.02 [1.94, 2.10] |
| 200-999 | tabpfn_r3 | 20,933 | 0.888 [0.869, 0.905] | 1.50 [1.38, 1.64] | 1.87 [1.79, 1.98] |
| 200-999 | tabpfn_r2 | 20,933 | 0.888 [0.870, 0.905] | 1.50 [1.38, 1.64] | 1.88 [1.79, 1.99] |
| 200-999 | lookup_full | 20,933 | 0.877 [0.858, 0.896] | 1.55 [1.44, 1.69] | 1.92 [1.84, 2.02] |
| 200-999 | lgbm_full | 20,933 | 0.885 [0.865, 0.902] | 1.50 [1.39, 1.64] | 1.88 [1.79, 1.98] |
| 1000+ | tabpfn_r3 | 19,479 | 0.896 [0.879, 0.912] | 1.28 [1.19, 1.39] | 1.60 [1.53, 1.68] |
| 1000+ | tabpfn_r2 | 19,479 | 0.894 [0.877, 0.909] | 1.29 [1.20, 1.39] | 1.61 [1.54, 1.68] |
| 1000+ | lookup_full | 19,479 | 0.890 [0.871, 0.906] | 1.31 [1.23, 1.40] | 1.61 [1.55, 1.69] |
| 1000+ | lgbm_full | 19,479 | 0.896 [0.879, 0.911] | 1.29 [1.20, 1.39] | 1.60 [1.54, 1.68] |
| all (weighted to pool) | tabpfn_r3 | 59,321 | 0.892 [0.875, 0.907] | 1.41 [1.31, 1.53] | 1.76 [1.69, 1.85] |
| all (weighted to pool) | tabpfn_r2 | 59,321 | 0.891 [0.874, 0.905] | 1.41 [1.32, 1.54] | 1.77 [1.70, 1.86] |
| all (weighted to pool) | lookup_full | 59,321 | 0.883 [0.866, 0.899] | 1.46 [1.37, 1.56] | 1.81 [1.74, 1.88] |
| all (weighted to pool) | lgbm_full | 59,321 | 0.889 [0.871, 0.904] | 1.42 [1.33, 1.53] | 1.77 [1.70, 1.86] |

| density | comparison | n_test | p90_coverage | pinball_q90 | pinball_mean |
|---|---|---|---|---|---|
| <30 | tabpfn_r3 - tabpfn_r2 | 903 | 0.000 [-0.009, 0.009] | -0.07 [-0.17, 0.01] | -0.08 [-0.14, -0.03] |
| <30 | tabpfn_r3 - lookup_full | 903 | 0.029 [0.003, 0.054] | -0.52 [-0.81, -0.29] | -0.64 [-0.82, -0.47] |
| <30 | tabpfn_r3 - lgbm_full | 903 | 0.068 [0.046, 0.090] | -0.41 [-0.66, -0.23] | -0.47 [-0.62, -0.35] |
| 30-199 | tabpfn_r3 - tabpfn_r2 | 18,006 | 0.003 [0.001, 0.004] | -0.01 [-0.01, -0.00] | -0.01 [-0.02, -0.01] |
| 30-199 | tabpfn_r3 - lookup_full | 18,006 | 0.010 [-0.007, 0.026] | -0.11 [-0.18, -0.04] | -0.13 [-0.17, -0.07] |
| 30-199 | tabpfn_r3 - lgbm_full | 18,006 | 0.016 [0.013, 0.019] | -0.05 [-0.07, -0.03] | -0.04 [-0.05, -0.03] |
| 200-999 | tabpfn_r3 - tabpfn_r2 | 20,933 | 0.000 [-0.002, 0.002] | -0.00 [-0.01, 0.01] | -0.00 [-0.01, 0.00] |
| 200-999 | tabpfn_r3 - lookup_full | 20,933 | 0.011 [0.005, 0.017] | -0.05 [-0.08, -0.02] | -0.05 [-0.07, -0.02] |
| 200-999 | tabpfn_r3 - lgbm_full | 20,933 | 0.004 [0.002, 0.006] | -0.00 [-0.01, 0.01] | -0.00 [-0.01, 0.01] |
| 1000+ | tabpfn_r3 - tabpfn_r2 | 19,479 | 0.002 [-0.001, 0.005] | -0.01 [-0.01, -0.00] | -0.01 [-0.02, -0.00] |
| 1000+ | tabpfn_r3 - lookup_full | 19,479 | 0.007 [-0.001, 0.015] | -0.02 [-0.04, -0.00] | -0.02 [-0.03, 0.00] |
| 1000+ | tabpfn_r3 - lgbm_full | 19,479 | 0.001 [-0.001, 0.003] | -0.00 [-0.01, 0.01] | -0.01 [-0.01, -0.00] |
| all (weighted to pool) | tabpfn_r3 - tabpfn_r2 | 59,321 | 0.001 [-0.000, 0.003] | -0.00 [-0.01, 0.00] | -0.01 [-0.01, -0.00] |
| all (weighted to pool) | tabpfn_r3 - lookup_full | 59,321 | 0.009 [0.003, 0.016] | -0.05 [-0.07, -0.02] | -0.04 [-0.06, -0.02] |
| all (weighted to pool) | tabpfn_r3 - lgbm_full | 59,321 | 0.004 [0.002, 0.005] | -0.01 [-0.02, 0.00] | -0.01 [-0.02, -0.00] |

TabPFN r3: 2914 contexts, all on cuda; context rows median 10,000 (min 66), own-route rows median 177; 657 contexts under 10,000 rows; GPU fit+predict 7,128 s, peak alloc 2,460 MiB, peak RSS 2,743 MiB

Reading: the new context rule and GPU-only runs change TabPFN very little: −0.08 mean
pinball on <30 (interval excludes zero), −0.01 or less elsewhere. The round-2 conclusions
hold: TabPFN wins on sparse routes and ties LightGBM on full data at 200+ trips a month.

## Reproduce

```bash
python -m src.ride.r3_control
python -m src.ride.r3_tabpfn run
python -m src.ride.r3_tabpfn merge
python -m src.ride.r3_evaluate
```

## Judgment calls

1. Test rows reused from round 2 (same sample, same weights), so round 2 and round 3 are
   paired row by row. Full run estimated at 2.1 h from a 32-context benchmark (2.6 s per
   context), under 3 h, so the test set was not shrunk.
2. Borough fill excludes own-route rows already taken; it is a uniform sample without
   replacement seeded by the CRC32 of the key. When the borough pool is smaller than the
   remaining slots, every row is used (657 contexts, mostly Staten Island and Bronx).
3. Own-route rows are never subsampled (the July maximum in a band is ~3.6k, under 10k).
4. The control's lookup inside a context uses the round-1 back-off; in an r3 context the
   "pickup zone" level equals the route level (do_zone is fixed), so it never fires; the
   last level is the whole context (303 rows).
5. Contexts are fit once per (route, hour), so weekday and weekend queries share one.
6. The 32 benchmark contexts are part of the final results (same code and settings).
7. Package defaults otherwise (TabPFN-3.5, memory_saving_mode "auto",
   TABPFN_MODEL_CACHE_SIZE=1 as in round 2).

## Fairness

- TabPFN and the `_ctx` baselines see identical rows; the `_full` baselines see all 224,923
  July rows. LightGBM defaults (20 rows per leaf, 100 per category group) still limit it on
  10k-row contexts; nothing was tuned for any model.
- Density bins come from July, the training month.
- <30 is 903 trips on 70 routes; intervals are wide.
