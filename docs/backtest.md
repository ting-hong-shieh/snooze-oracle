# Backtest on 2026-08: wake-time strategies vs reality

Flights scheduled 04:00-07:59: 1671 (plus 32 landing on 09-01, dropped), cancelled 91 and diverted 5 (skipped), flown 1575. Scenarios kept: 12600 (flown flights x 8 pickup zones), skipped for an empty reality ride pool: {'dense': 0, 'sparse': 0}. Wake ranges with no time reaching 0.90: {'lookup': 0, 'lgbm': 0, 'tabpfn': 0}. 5 reality draws per scenario.

95% intervals: day-level bootstrap (2,000 resamples of the 31 arrival days). Method, judgment calls and fairness notes: `src/alarm/README.md`. Chart: `docs/frontier.png`.

## all zones

| strategy | late rate | median wait (min) | mean wake vs gut B=30 (min) |
|---|---|---|---|
| gut B=0 | 0.652 [0.605, 0.698] | 0.0 [0.0, 0.0] | 30.0 [30.0, 30.0] |
| gut B=15 | 0.377 [0.335, 0.420] | 6.0 [3.7, 8.5] | 15.0 [15.0, 15.0] |
| gut B=30 | 0.111 [0.090, 0.133] | 21.3 [18.9, 23.7] | 0.0 [0.0, 0.0] |
| gut B=45 | 0.016 [0.011, 0.021] | 36.5 [34.1, 39.0] | -15.0 [-15.0, -15.0] |
| gut B=60 | 0.001 [0.001, 0.002] | 51.8 [49.4, 54.2] | -30.0 [-30.0, -30.0] |
| gut B=90 | 0.000 [0.000, 0.000] | 82.3 [79.9, 84.7] | -60.0 [-60.0, -60.0] |
| schedule only | 0.462 [0.417, 0.508] | 1.9 [0.0, 4.4] | 19.1 [18.9, 19.4] |
| lookup p>=0.90 | 0.091 [0.073, 0.109] | 22.8 [20.6, 25.3] | -1.9 [-2.1, -1.7] |
| LightGBM p>=0.90 | 0.117 [0.095, 0.139] | 20.5 [18.4, 22.9] | 0.5 [0.1, 0.9] |
| TabPFN p>=0.90 | 0.103 [0.083, 0.123] | 21.4 [19.3, 23.9] | -0.3 [-0.7, 0.2] |
| TabPFN + snooze | 0.103 [0.083, 0.123] | 21.4 [19.3, 23.9] | -0.3 [-0.7, 0.2] |

Gap to the gut-rule curve (late rate minus the curve's late rate at the same median wait, curve = gut rules every 5 min, linear between them; negative = below the curve):

- schedule only: 0.001 [-0.018, 0.003]
- lookup p>=0.90: -0.005 [-0.009, 0.000]
- LightGBM p>=0.90: -0.005 [-0.009, 0.000]
- TabPFN p>=0.90: -0.007 [-0.012, -0.001]
- TabPFN + snooze: -0.007 [-0.012, -0.001]

Paired differences (same scenarios and draws):

- TabPFN minus lookup: late rate 0.012 [0.006, 0.018], median wait -1.5 [-2.0, -0.9] min, wake time 1.6 [1.1, 2.1] min
- TabPFN minus LightGBM: late rate -0.014 [-0.017, -0.011], median wait 0.9 [0.7, 1.1] min, wake time -0.8 [-0.9, -0.6] min

Snooze simulation: 4.00 snoozes granted per scenario (min 4, max 4), 1 refusal per scenario (the alarm keeps ringing).

## dense zones

| strategy | late rate | median wait (min) | mean wake vs gut B=30 (min) |
|---|---|---|---|
| gut B=0 | 0.638 [0.591, 0.685] | 0.0 [0.0, 0.0] | 30.0 [30.0, 30.0] |
| gut B=15 | 0.354 [0.310, 0.397] | 7.2 [4.9, 9.6] | 15.0 [15.0, 15.0] |
| gut B=30 | 0.102 [0.081, 0.123] | 22.3 [20.0, 24.8] | 0.0 [0.0, 0.0] |
| gut B=45 | 0.015 [0.010, 0.021] | 37.4 [35.1, 39.9] | -15.0 [-15.0, -15.0] |
| gut B=60 | 0.002 [0.001, 0.002] | 52.6 [50.2, 55.0] | -30.0 [-30.0, -30.0] |
| gut B=90 | 0.000 [0.000, 0.000] | 82.8 [80.4, 85.2] | -60.0 [-60.0, -60.0] |
| schedule only | 0.450 [0.402, 0.498] | 2.5 [0.1, 5.0] | 19.5 [19.2, 19.9] |
| lookup p>=0.90 | 0.085 [0.067, 0.104] | 23.3 [21.2, 25.7] | -1.5 [-1.8, -1.3] |
| LightGBM p>=0.90 | 0.110 [0.089, 0.132] | 20.7 [18.6, 23.2] | 1.0 [0.7, 1.4] |
| TabPFN p>=0.90 | 0.103 [0.083, 0.123] | 21.3 [19.1, 23.8] | 0.5 [0.1, 1.0] |
| TabPFN + snooze | 0.103 [0.083, 0.123] | 21.3 [19.1, 23.8] | 0.5 [0.1, 1.0] |

Gap to the gut-rule curve (late rate minus the curve's late rate at the same median wait, curve = gut rules every 5 min, linear between them; negative = below the curve):

- schedule only: 0.001 [-0.004, 0.005]
- lookup p>=0.90: -0.009 [-0.013, -0.003]
- LightGBM p>=0.90: -0.013 [-0.018, -0.007]
- TabPFN p>=0.90: -0.012 [-0.018, -0.006]
- TabPFN + snooze: -0.012 [-0.018, -0.006]

Paired differences (same scenarios and draws):

- TabPFN minus lookup: late rate 0.018 [0.014, 0.023], median wait -2.0 [-2.5, -1.5] min, wake time 2.0 [1.7, 2.4] min
- TabPFN minus LightGBM: late rate -0.007 [-0.011, -0.003], median wait 0.6 [0.3, 0.8] min, wake time -0.5 [-0.7, -0.3] min

Snooze simulation: 4.00 snoozes granted per scenario (min 4, max 4), 1 refusal per scenario (the alarm keeps ringing).

## sparse zones

| strategy | late rate | median wait (min) | mean wake vs gut B=30 (min) |
|---|---|---|---|
| gut B=0 | 0.666 [0.619, 0.713] | 0.0 [0.0, 0.0] | 30.0 [30.0, 30.0] |
| gut B=15 | 0.400 [0.356, 0.445] | 4.8 [2.5, 7.3] | 15.0 [15.0, 15.0] |
| gut B=30 | 0.120 [0.097, 0.144] | 20.1 [17.8, 22.6] | 0.0 [0.0, 0.0] |
| gut B=45 | 0.016 [0.012, 0.022] | 35.5 [33.1, 38.0] | -15.0 [-15.0, -15.0] |
| gut B=60 | 0.001 [0.001, 0.002] | 50.9 [48.5, 53.4] | -30.0 [-30.0, -30.0] |
| gut B=90 | 0.000 [0.000, 0.000] | 81.6 [79.2, 84.1] | -60.0 [-60.0, -60.0] |
| schedule only | 0.474 [0.428, 0.521] | 1.3 [0.0, 3.8] | 18.8 [18.7, 18.9] |
| lookup p>=0.90 | 0.097 [0.079, 0.117] | 22.4 [20.1, 25.0] | -2.3 [-2.4, -2.2] |
| LightGBM p>=0.90 | 0.123 [0.101, 0.147] | 20.3 [18.1, 22.7] | -0.0 [-0.9, 0.9] |
| TabPFN p>=0.90 | 0.103 [0.082, 0.124] | 21.5 [19.4, 24.0] | -1.1 [-1.9, -0.2] |
| TabPFN + snooze | 0.103 [0.082, 0.124] | 21.5 [19.4, 24.0] | -1.1 [-1.9, -0.2] |

Gap to the gut-rule curve (late rate minus the curve's late rate at the same median wait, curve = gut rules every 5 min, linear between them; negative = below the curve):

- schedule only: 0.000 [-0.030, 0.004]
- lookup p>=0.90: 0.000 [-0.005, 0.005]
- LightGBM p>=0.90: 0.005 [-0.003, 0.012]
- TabPFN p>=0.90: -0.003 [-0.009, 0.004]
- TabPFN + snooze: -0.003 [-0.009, 0.004]

Paired differences (same scenarios and draws):

- TabPFN minus lookup: late rate 0.006 [-0.005, 0.018], median wait -0.9 [-1.9, 0.1] min, wake time 1.2 [0.3, 2.1] min
- TabPFN minus LightGBM: late rate -0.021 [-0.025, -0.016], median wait 1.2 [0.9, 1.6] min, wake time -1.0 [-1.2, -0.8] min

Snooze simulation: 4.00 snoozes granted per scenario (min 4, max 4), 1 refusal per scenario (the alarm keeps ringing).

## Risk by flight type

Late rate and median airport wait per group (95% day-bootstrap intervals). Groups are by scenario; every scenario has the same 5 reality draws.

### By origin airport (top 10 by scenarios)

Each cell: late rate [95% CI]; median wait in minutes [95% CI].

| group | scenarios | gut B=30 | schedule only | lookup p>=0.90 | LightGBM p>=0.90 | TabPFN p>=0.90 |
|---|---|---|---|---|---|---|
| LAX | 1,216 | 7.5% [4.9, 10.3]; wait 26 [22, 33] | 38.0% [30.7, 45.0]; wait 7 [2, 14] | 4.6% [2.7, 7.0]; wait 30 [26, 37] | 5.6% [3.5, 8.2]; wait 28 [23, 35] | 4.4% [2.5, 6.8]; wait 29 [25, 35] |
| BOS | 1,056 | 9.9% [7.3, 12.7]; wait 18 [16, 20] | 54.2% [48.6, 59.9]; wait 0 [0, 0] | 8.2% [5.7, 11.0]; wait 18 [17, 20] | 10.1% [7.1, 13.6]; wait 17 [15, 19] | 7.8% [5.3, 10.6]; wait 18 [17, 20] |
| SFO | 904 | 20.5% [13.6, 28.1]; wait 22 [15, 29] | 45.8% [36.5, 55.4]; wait 3 [0, 9] | 16.9% [11.1, 23.4]; wait 27 [20, 34] | 19.6% [13.1, 26.4]; wait 24 [17, 30] | 19.3% [12.9, 26.5]; wait 23 [16, 29] |
| DCA | 856 | 10.5% [8.2, 13.1]; wait 17 [15, 20] | 55.4% [48.1, 62.3]; wait 0 [0, 1] | 12.1% [9.3, 14.9]; wait 16 [14, 19] | 11.3% [8.6, 14.2]; wait 17 [14, 19] | 8.9% [6.7, 11.3]; wait 18 [16, 21] |
| SEA | 656 | 10.6% [6.8, 14.8]; wait 25 [20, 34] | 38.9% [29.5, 47.8]; wait 7 [1, 16] | 9.8% [6.3, 13.9]; wait 26 [20, 35] | 14.0% [9.2, 18.9]; wait 23 [17, 31] | 16.6% [10.9, 22.5]; wait 20 [15, 29] |
| LAS | 624 | 8.4% [5.3, 12.1]; wait 27 [20, 34] | 38.4% [29.3, 48.3]; wait 8 [1, 15] | 1.2% [0.5, 2.1]; wait 35 [29, 44] | 2.9% [1.7, 4.5]; wait 32 [25, 39] | 3.3% [1.9, 5.1]; wait 30 [23, 37] |
| PIT | 600 | 7.2% [5.1, 9.4]; wait 21 [19, 24] | 44.9% [37.4, 52.1]; wait 2 [0, 5] | 7.9% [5.6, 10.2]; wait 19 [17, 23] | 11.0% [8.2, 14.0]; wait 17 [14, 20] | 9.4% [6.7, 12.4]; wait 18 [15, 21] |
| PHX | 576 | 7.1% [4.0, 10.6]; wait 27 [21, 38] | 37.3% [27.0, 47.5]; wait 8 [1, 18] | 4.8% [2.4, 7.7]; wait 29 [23, 40] | 4.5% [2.3, 6.9]; wait 30 [23, 40] | 3.5% [1.8, 5.3]; wait 31 [26, 42] |
| HNL | 472 | 10.2% [5.5, 15.5]; wait 24 [19, 30] | 39.4% [30.1, 49.3]; wait 5 [0, 11] | 5.7% [2.2, 10.4]; wait 31 [26, 36] | 8.9% [4.0, 14.9]; wait 27 [22, 32] | 8.7% [3.8, 14.8]; wait 28 [23, 33] |
| SAN | 448 | 16.8% [10.2, 23.9]; wait 17 [11, 24] | 54.9% [43.8, 65.5]; wait 0 [0, 4] | 6.0% [2.2, 10.7]; wait 24 [19, 31] | 9.7% [5.0, 15.3]; wait 21 [16, 28] | 6.7% [3.2, 11.1]; wait 25 [20, 32] |
| other | 5,192 | 11.5% [9.4, 13.6]; wait 20 [18, 23] | 48.0% [43.3, 52.7]; wait 1 [0, 4] | 10.5% [8.6, 12.5]; wait 21 [18, 24] | 14.2% [11.6, 16.7]; wait 18 [16, 21] | 12.2% [9.9, 14.5]; wait 19 [17, 21] |

### By stage length

Each cell: late rate [95% CI]; median wait in minutes [95% CI].

| group | scenarios | gut B=30 | schedule only | lookup p>=0.90 | LightGBM p>=0.90 | TabPFN p>=0.90 |
|---|---|---|---|---|---|---|
| short (<500 mi) | 4,992 | 9.7% [8.2, 11.3]; wait 19 [17, 20] | 51.5% [48.0, 55.1]; wait 0 [0, 1] | 10.6% [9.0, 12.3]; wait 18 [16, 19] | 12.4% [10.5, 14.6]; wait 16 [15, 18] | 10.1% [8.4, 12.0]; wait 18 [16, 19] |
| medium (500-1500 mi) | 696 | 11.4% [8.3, 15.0]; wait 16 [14, 20] | 56.3% [47.9, 64.8]; wait 0 [0, 1] | 15.3% [11.4, 19.6]; wait 14 [11, 17] | 19.0% [13.9, 24.7]; wait 13 [10, 16] | 15.5% [11.2, 20.5]; wait 14 [11, 17] |
| long/red-eye (>1500 mi) | 6,912 | 12.1% [9.2, 15.1]; wait 25 [21, 30] | 41.4% [35.5, 47.2]; wait 6 [2, 10] | 7.4% [5.3, 9.6]; wait 30 [26, 35] | 10.4% [7.8, 13.0]; wait 27 [23, 31] | 9.9% [7.5, 12.4]; wait 27 [23, 31] |

### By pickup zone density

Each cell: late rate [95% CI]; median wait in minutes [95% CI].

| group | scenarios | gut B=30 | schedule only | lookup p>=0.90 | LightGBM p>=0.90 | TabPFN p>=0.90 |
|---|---|---|---|---|---|---|
| dense | 6,300 | 10.2% [8.1, 12.3]; wait 22 [20, 25] | 45.0% [40.2, 49.8]; wait 2 [0, 5] | 8.5% [6.7, 10.4]; wait 23 [21, 26] | 11.0% [8.9, 13.2]; wait 21 [19, 23] | 10.3% [8.3, 12.3]; wait 21 [19, 24] |
| sparse | 6,300 | 12.0% [9.7, 14.4]; wait 20 [18, 23] | 47.4% [42.8, 52.1]; wait 1 [0, 4] | 9.7% [7.9, 11.7]; wait 22 [20, 25] | 12.3% [10.1, 14.7]; wait 20 [18, 23] | 10.3% [8.2, 12.4]; wait 22 [19, 24] |

### By destination

Each cell: late rate [95% CI]; median wait in minutes [95% CI].

| group | scenarios | gut B=30 | schedule only | lookup p>=0.90 | LightGBM p>=0.90 | TabPFN p>=0.90 |
|---|---|---|---|---|---|---|
| JFK | 10,000 | 11.1% [8.8, 13.5]; wait 22 [19, 25] | 45.3% [40.2, 50.2]; wait 3 [0, 6] | 7.9% [6.1, 9.8]; wait 25 [22, 28] | 10.6% [8.3, 12.9]; wait 22 [20, 25] | 9.7% [7.5, 11.8]; wait 23 [20, 26] |
| LGA | 2,600 | 11.1% [9.2, 13.2]; wait 19 [17, 21] | 49.9% [45.8, 54.5]; wait 0 [0, 2] | 13.5% [11.3, 16.0]; wait 17 [15, 18] | 15.9% [13.2, 18.9]; wait 15 [13, 17] | 12.7% [10.2, 15.5]; wait 17 [15, 19] |

### Spread of late rates across the origin groups

Across the 11 origin groups (top 10 + other): min, max, and the standard deviation weighted by scenario count, in percentage points. Interval for the weighted SD from the same day bootstrap.

| strategy | min | max | weighted SD [95% CI] | groups with CI excluding 10% |
|---|---|---|---|---|
| gut B=30 | 7.1% (PHX) | 20.5% (SFO) | 3.3 [2.3, 5.0] | 3 of 11 |
| schedule only | 37.3% (PHX) | 55.4% (DCA) | 5.7 [4.6, 8.3] | 11 of 11 |
| lookup p>=0.90 | 1.2% (LAS) | 16.9% (SFO) | 3.5 [2.8, 4.8] | 4 of 11 |
| LightGBM p>=0.90 | 2.9% (LAS) | 19.6% (SFO) | 4.2 [3.4, 5.5] | 5 of 11 |
| TabPFN p>=0.90 | 3.3% (LAS) | 19.3% (SFO) | 4.3 [3.3, 5.7] | 5 of 11 |

- Weighted SD, lookup p>=0.90 minus gut B=30: +0.3 [-0.5, +0.9] points (no clear difference).
- Weighted SD, LightGBM p>=0.90 minus gut B=30: +0.9 [+0.0, +1.6] points (less even).
- Weighted SD, TabPFN p>=0.90 minus gut B=30: +1.0 [+0.0, +1.7] points (less even).

### TabPFN wake time minus gut B=30 wake time, per scenario

12,600 scenarios. Percentiles (5/25/50/75/95): -9.7 / -4.2 / -0.2 / +3.7 / +8.6 min. Sleep 10+ min longer: 2.0% (255); woken 10+ min earlier: 4.3% (541).

**sleep 10+ min longer** (255 scenarios, 29 days). Late rate in these scenarios: TabPFN 16.9%, gut B=30 5.6%.

- origins: RIC 48, ORF 43, BUF 40, SEA 32, PIT 25, DTW 10, SFO 10, SMF 10
- carriers: YX 164, AS 41, B6 36, DL 14
- stage: short (<500 mi) 170, long/red-eye (>1500 mi) 68, medium (500-1500 mi) 17
- destination: LGA 160, JFK 95
- pickup zones: dense 194, sparse 61
- scheduled arrival hour: 5:00 7, 6:00 9, 7:00 239

**woken 10+ min earlier** (541 scenarios, 30 days). Late rate in these scenarios: TabPFN 3.8%, gut B=30 17.8%.

- origins: LAS 114, SAN 109, PHX 104, HNL 92, LAX 29, SFO 26, PDX 19, ANC 8
- carriers: DL 306, AA 169, B6 30, YX 28, AS 8
- stage: long/red-eye (>1500 mi) 507, short (<500 mi) 34
- destination: JFK 508, LGA 33
- pickup zones: dense 277, sparse 264
- scheduled arrival hour: 5:00 41, 6:00 228, 7:00 272

