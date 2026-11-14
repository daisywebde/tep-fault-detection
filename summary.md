# Results summary

- PCA: k = 31 components (90.2% of variance)
- Control limits: empirical (held-out normal runs), 99% level
- Training: 80 fault-free runs; thresholds from 20 held-out runs
- Testing: 10 runs per fault; fault starts after sample 20
- Delay = minutes from onset to 3 consecutive alarms (median over detected runs)

False alarm rate on fault-free test runs: Isolation Forest 0.96%, PCA (T2 or SPE) 1.94%, PCA SPE 1.00%, PCA T2 0.95%

| Fault | PCA FDR % | PCA FAR % | PCA runs detected | PCA delay (min) | IForest FDR % | IForest FAR % | IForest runs detected | IForest delay (min) |
|---|---|---|---|---|---|---|---|---|
| 1 | 99.62 | 1 | 10/10 | 6 | 93.04 | 0 | 10/10 | 21 |
| 2 | 97.81 | 1 | 10/10 | 30 | 94.9 | 0 | 10/10 | 78 |
| 3 | 2 | 1 | 0/10 |  | 1.46 | 0 | 3/10 | 345 |
| 4 | 99.98 | 1 | 10/10 | 0 | 2.92 | 0 | 6/10 | 484.5 |
| 5 | 39.52 | 1 | 10/10 | 0 | 34.06 | 0 | 10/10 | 39 |
| 6 | 100 | 1 | 10/10 | 0 | 98.12 | 0 | 10/10 | 30 |
| 7 | 100 | 1 | 10/10 | 0 | 61.5 | 0 | 10/10 | 6 |
| 8 | 95.92 | 1 | 10/10 | 57 | 93.73 | 0 | 10/10 | 69 |
| 9 | 1.9 | 1 | 0/10 |  | 1.67 | 0 | 4/10 | 249 |
| 10 | 37.65 | 1 | 10/10 | 232.5 | 11.31 | 0 | 9/10 | 390 |
| 11 | 77.08 | 1 | 10/10 | 31.5 | 4.38 | 0 | 7/10 | 642 |
| 12 | 97.81 | 1 | 10/10 | 18 | 92.04 | 0 | 10/10 | 63 |
| 13 | 90.42 | 1 | 10/10 | 147 | 88.44 | 0 | 10/10 | 159 |
| 14 | 99.9 | 1 | 10/10 | 1.5 | 7.88 | 0 | 8/10 | 715.5 |
| 15 | 2.19 | 1 | 1/10 | 222 | 1.23 | 0 | 3/10 | 291 |
| 16 | 27.42 | 1 | 10/10 | 139.5 | 7.77 | 0 | 8/10 | 678 |
| 17 | 89.62 | 1 | 10/10 | 118.5 | 7.96 | 0 | 8/10 | 310.5 |
| 18 | 89.54 | 1 | 10/10 | 144 | 86.94 | 0 | 10/10 | 142.5 |
| 19 | 17.62 | 1 | 10/10 | 286.5 | 3.23 | 0 | 6/10 | 651 |
| 20 | 52.42 | 1 | 10/10 | 147 | 8.9 | 0 | 9/10 | 579 |
