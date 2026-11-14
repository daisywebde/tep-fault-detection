# Fault Detection on the Tennessee Eastman Process (PCA + Isolation Forest)

Multivariate statistical process monitoring on the Tennessee Eastman Process (TEP), a standard
benchmark for chemical-plant fault detection. The project learns what *normal* operation looks like,
flags abnormal behaviour, measures how well it works, and shows **which variables** caused each alarm.

- **PCA monitoring:** Hotelling's T² and SPE (Q) with control limits set on held-out normal runs
- **ML baseline:** Isolation Forest, thresholded the same way so the comparison is fair
- **Root cause:** per-variable contribution plots (contributions sum exactly to T² / SPE)
- **Metrics:** fault detection rate (FDR), false alarm rate (FAR), detection delay, per fault
- **Dashboard-ready export:** `results/monitoring_timeseries.csv` loads straight into Power BI / Tableau

## 1. Get the data

Dataset: Rieth, Amsel, Tran, Cook (2017), *Additional Tennessee Eastman Process Simulation Data for
Anomaly Detection Evaluation*, Harvard Dataverse (search that title; it is also mirrored on Kaggle).
The underlying process is from Downs & Vogel (1993). Check the license on the data page.

You need three files (put them in `data/`):

| File | Used for |
|---|---|
| `TEP_FaultFree_Training` | learn normal behaviour, set thresholds |
| `TEP_FaultFree_Testing`  | false alarm rate on unseen normal runs |
| `TEP_Faulty_Testing`     | detection rate / delay for faults 1-20 |

`.csv` or `.RData` both work. Columns: `faultNumber, simulationRun, sample, xmeas_1..xmeas_41, xmv_1..xmv_11`.

**Memory tip.** The faulty testing file is about 4 GB. CSV files are streamed and filtered while
loading, so only the runs you ask for are kept. `.RData` files must be loaded whole (`pip install pyreadr`);
if that is too heavy, convert a subset to CSV in R:

```r
load("TEP_Faulty_Testing.RData")                 # object name may differ
sub <- faulty_testing[faulty_testing$simulationRun <= 5, ]
write.csv(sub, "data/TEP_Faulty_Testing.csv", row.names = FALSE)
```

## 2. Run

```bash
pip install -r requirements.txt
python run_pipeline.py --data-dir data/                    # all 20 faults, 100 training runs, 5 test runs/fault
python run_pipeline.py --data-dir data/ --faults 1 2 4 5 6 7 --test-runs 10
python run_pipeline.py --synthetic                          # smoke test on FAKE data (no download needed)
python tests/test_core.py                                   # unit tests
```

Or open `TEP_fault_detection.ipynb` for the same pipeline step by step with explanations.

Outputs go to `results/`: `metrics.csv`, `summary.md`, `contributions_top.csv`,
`monitoring_timeseries.csv` and `figures/*.png`.

> The `--synthetic` mode generates fake data only to check that the code runs. Never quote
> numbers from it. Only results from the real dataset belong on a resume.

## 3. How it works

1. **Standardise and fit PCA** on fault-free training runs. The number of components `k` keeps 90% of the variance (`--variance`).
2. **Two scores per sample**
   - **T²**: distance from normal *inside* the PCA subspace (normal pattern, unusual magnitude).
   - **SPE (Q)**: size of the part PCA cannot reconstruct (the normal correlation pattern is broken).
3. **Control limits** at the 99% level, taken from **held-out normal runs** (20% of the training runs are never used to fit PCA).
   Theoretical limits (F-distribution for T², Box chi-square approximation for SPE) are computed too and printed for comparison.
   Empirical limits are the default because TEP samples are autocorrelated, which makes theoretical limits unreliable.
4. **Alarm** = T² or SPE above its limit. **Isolation Forest** gets its threshold the same way (99th percentile of held-out normal scores).
5. **Metrics**, per fault, averaged over test runs (the fault starts after sample 160, i.e. hour 8, with 3-minute sampling):
   - **FDR**: % of post-fault samples that raise an alarm.
   - **FAR**: % of pre-fault samples that raise an alarm (and % over whole fault-free runs).
   - **Delay**: minutes from fault onset to the first 3 consecutive alarms (median over runs that were detected).
6. **Root cause**: each alarm's score is split into per-variable contributions
   (`c_j = z_j * (P Λ⁻¹ Pᵀ z)_j` for T², squared residual per variable for SPE). The top variables point to where the fault shows up.

## 4. Things to say honestly

- A few faults (3, 9 and 15 in the TEP literature) barely change the measurements and are reported as
  essentially undetectable with standard PCA monitoring. Expect low FDR there for every method.
- Even with a 99% limit the combined T²/SPE rule produces roughly 1-2% false alarms, because the data are autocorrelated.
- Hyperparameters (variance level, forest size) are defaults, not tuned.
- TEP is a **continuous** process. Batch processes (e.g. fermentation) need an extra step: align batches in time and
  normalise against a reference trajectory such as a golden batch. The monitoring ideas carry over; the preprocessing does not come for free.
- Variable descriptions in `tep_fd.py` are approximate labels; verify against Downs & Vogel (1993) before quoting them.

## 5. Optional: dashboard

Load `results/monitoring_timeseries.csv` into Power BI or Tableau. Useful views: T² and SPE vs time with limit
lines (columns `t2_limit`, `spe_limit`), a fault selector (`faultNumber`), alarm flags (`alarm_pca`, `alarm_if`),
and `contributions_top.csv` as a bar chart of top variables.

## 6. Project layout

```
tep_fd.py            loading, PCA monitor (T2/SPE/contributions), Isolation Forest, metrics
plots.py             figures
run_pipeline.py      command-line pipeline
make_synthetic.py    fake data in the TEP layout, for smoke tests only
tests/test_core.py   correctness checks (contributions sum to the statistics, false-alarm rate, ...)
TEP_fault_detection.ipynb
```

## 7. Resume bullets (fill in the numbers from YOUR run)

- Built a multivariate fault-detection pipeline on the Tennessee Eastman Process benchmark in Python (pandas, NumPy, scikit-learn), using PCA with Hotelling's T² and SPE statistics and control limits calibrated on held-out normal runs.
- Benchmarked against Isolation Forest across N fault types: PCA detected X of N faults with a mean false alarm rate of Y%, median detection delay Z minutes.
- Implemented contribution analysis to rank the process variables behind each alarm (root cause identification) and exported monitoring results for dashboarding.
