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

