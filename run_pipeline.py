#!/usr/bin/env python3
"""
run_pipeline.py - end-to-end fault detection on the Tennessee Eastman Process.

    python run_pipeline.py --data-dir data/            # real TEP data
    python run_pipeline.py --synthetic                 # smoke test, fake data

Outputs (in --out, default results/):
    metrics.csv                 per fault x method: FDR, FAR, runs detected, delay
    summary.csv / summary.md    one row per fault, PCA vs Isolation Forest
    contributions_top.csv       top contributing variables per fault (root cause)
    monitoring_timeseries.csv   T2 / SPE / alarms for a few runs (feed to Power BI)
    figures/*.png               scree, detection rates, monitoring and contribution plots
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import plots
import tep_fd as T


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data", help="folder with the TEP CSV / RData files")
    ap.add_argument("--synthetic", action="store_true", help="generate and use synthetic data (smoke test only)")
    ap.add_argument("--out", default="results", help="output folder")
    ap.add_argument("--train-runs", type=int, default=100, help="fault-free training runs to use (80%% fit, 20%% threshold)")
    ap.add_argument("--test-runs", type=int, default=5, help="test runs per fault (and fault-free)")
    ap.add_argument("--faults", type=int, nargs="+", default=list(range(1, 21)), help="fault numbers to evaluate")
    ap.add_argument("--variance", type=float, default=0.90, help="cumulative variance for choosing k")
    ap.add_argument("--n-components", type=int, default=None, help="fix k instead of using --variance")
    ap.add_argument("--alpha", type=float, default=0.99, help="confidence level of the control limits")
    ap.add_argument("--consec", type=int, default=3, help="consecutive alarms needed to confirm a fault (delay metric)")
    ap.add_argument("--faulty-split", choices=["test", "train"], default="test",
                    help="which faulty file to evaluate on: 'test' (960 samples/run, fault at sample 160) or "
                         "'train' (500 samples/run, fault at sample 20; about half the size, use it if the "
                         "faulty testing file does not fit in memory)")
    ap.add_argument("--fault-start", type=int, default=None,
                    help="last normal sample of the faulty runs (default: 160 for 'test', 20 for 'train')")
    ap.add_argument("--plot-faults", type=int, nargs="+", default=[1, 6, 3], help="faults to draw monitoring/contribution plots for")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else (f"{v:g}" if isinstance(v, (float, np.floating)) else str(v)) for v in r) + " |")
    return "\n".join(lines)


def build_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    def part(method, tag):
        m = metrics[(metrics["method"] == method) & (metrics["fault"] > 0)].set_index("fault")
        return pd.DataFrame({
            f"{tag} FDR %": m["FDR_pct"],
            f"{tag} FAR %": m["FAR_pct"],
            f"{tag} runs detected": m["runs_detected"].astype("Int64").astype(str) + "/" + m["runs"].astype(str),
            f"{tag} delay (min)": m["median_delay_min"],
        })
    s = part("PCA (T2 or SPE)", "PCA").join(part("Isolation Forest", "IForest"))
    return s.reset_index().rename(columns={"fault": "Fault"})


def main():
    args = parse_args()
    t0 = time.time()
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        from make_synthetic import make_synthetic
        data_dir = make_synthetic(out / "synthetic_data", seed=args.seed)
        print(f"[synthetic] wrote FAKE data to {data_dir} - results are for testing only")
        args.faults = [f for f in args.faults if f in (1, 2, 3, 4, 5, 6)]
        args.train_runs = min(args.train_runs, 60)
        args.test_runs = min(args.test_runs, 6)
    else:
        data_dir = Path(args.data_dir)

    if args.fault_start is None:
        args.fault_start = T.FAULT_START_TEST if args.faulty_split == "test" else T.FAULT_START_TRAIN

    # ---- 1. normal training data -------------------------------------------
    p_train = T.find_tep_file(data_dir, faulty=False, split="train")
    print(f"Loading fault-free training data: {p_train.name}")
    train = T.load_tep(p_train, faults=[0], max_run=args.train_runs)
    runs = sorted(train["simulationRun"].unique())
    n_fit = max(1, int(0.8 * len(runs)))
    fit_runs, val_runs = runs[:n_fit], runs[n_fit:]
    if not val_runs:
        raise SystemExit("Need at least 2 training runs (one is held out to set thresholds).")
    X_fit = train.loc[train["simulationRun"].isin(fit_runs), T.VARS].to_numpy(np.float64)
    X_val = train.loc[train["simulationRun"].isin(val_runs), T.VARS].to_numpy(np.float64)
    print(f"  {len(runs)} runs: {len(fit_runs)} to fit ({len(X_fit):,} samples), "
          f"{len(val_runs)} held out for thresholds ({len(X_val):,} samples)")

    # ---- 2. fit the models --------------------------------------------------
    pca = T.PCAMonitor(variance=args.variance, n_components=args.n_components, alpha=args.alpha)
    pca.fit(X_fit, X_val)
    iforest = T.IForestMonitor(alpha=args.alpha, random_state=args.seed).fit(X_fit, X_val)
    print(f"PCA: k = {pca.k} components ({pca.cum_var[pca.k - 1] * 100:.1f}% variance); "
          f"limits [{pca.limit_method}]: T2 = {pca.t2_limit:.1f}, SPE = {pca.spe_limit:.1f}  "
          f"(theoretical: T2 = {pca.t2_limit_theory:.1f}, SPE = {pca.spe_limit_theory:.1f})")

    # ---- 3. test data ---------------------------------------------------------
    p_ff_test = T.find_tep_file(data_dir, faulty=False, split="test")
    p_fy_test = T.find_tep_file(data_dir, faulty=True, split=args.faulty_split)
    print(f"Loading test data: {p_ff_test.name}, {p_fy_test.name}")
    test0 = T.load_tep(p_ff_test, faults=[0], max_run=args.test_runs)
    testf = T.load_tep(p_fy_test, faults=args.faults, max_run=args.test_runs)
    found = [int(f) for f in sorted(testf["faultNumber"].unique())]
    missing = sorted(set(args.faults) - set(found))
    if missing:
        print(f"  note: faults not found in the file: {missing}")
    test = pd.concat([test0, testf], ignore_index=True)
    print(f"  {len(test):,} test samples, faults {found}, {args.test_runs} runs each")
    if args.faulty_split == "train":
        print("  note: faulty TRAINING runs have only 20 normal samples before the fault, so the per-fault FAR "
              "below rests on 20 samples per run; use the fault-free test FAR for the overall false alarm rate.")

    X = test[T.VARS].to_numpy(np.float64)
    a = pca.alarms(X)
    b = iforest.alarms(X)
    for k, v in {**a, **b}.items():
        test[k] = v

    # ---- 4. metrics -------------------------------------------------------------
    metrics = T.evaluate(test, fault_start=args.fault_start, consec=args.consec)
    metrics.to_csv(out / "metrics.csv", index=False)
    summary = build_summary(metrics)
    summary.to_csv(out / "summary.csv", index=False)

    far0 = metrics[(metrics["fault"] == 0)].set_index("method")["FAR_pct"]
    print("\nFalse alarm rate on fault-free test runs (whole run):")
    for m_, v in far0.items():
        print(f"  {m_:<18} {v:5.2f} %")
    print("\nPer-fault results (FDR = detection rate after onset, FAR = false alarms before onset):")
    print(summary.to_string(index=False))

    # ---- 5. root cause: contributions -------------------------------------------
    rows = []
    for f in found:
        g = test[(test["faultNumber"] == f) & (test["simulationRun"] == 1) &
                 (test["sample"] > args.fault_start)]
        if g.empty:
            continue
        Xf = g[T.VARS].to_numpy(np.float64)
        which = "spe" if g["alarm_spe"].mean() >= g["alarm_t2"].mean() else "t2"
        top = T.top_contributors(pca, Xf, which=which, top=10)
        for rank, r in enumerate(top.itertuples(), 1):
            rows.append((f, which.upper(), rank, r.variable, r.description, r[3]))
    contrib = pd.DataFrame(rows, columns=["fault", "statistic", "rank", "variable", "description", "contribution_pct"])
    contrib.to_csv(out / "contributions_top.csv", index=False)

    # ---- 6. dashboard export ----------------------------------------------------
    dash = test[test["simulationRun"] <= 2][T.ID_COLS + ["t2", "spe", "alarm_t2", "alarm_spe", "alarm_pca", "if_score", "alarm_if"]].copy()
    dash["time_h"] = dash["sample"] * T.SAMPLE_MINUTES / 60
    dash["t2_limit"], dash["spe_limit"], dash["if_limit"] = pca.t2_limit, pca.spe_limit, iforest.limit
    dash["phase"] = np.where((dash["faultNumber"] > 0) & (dash["sample"] > args.fault_start), "fault active", "normal")
    dash.to_csv(out / "monitoring_timeseries.csv", index=False)

    # ---- 7. figures -------------------------------------------------------------
    figs = out / "figures"
    plots.plot_scree(pca, figs / "01_pca_variance.png")
    plots.plot_fdr(metrics, figs / "02_detection_rate_by_fault.png")
    for f in args.plot_faults:
        if f not in found:
            continue
        g = test[(test["faultNumber"] == f) & (test["simulationRun"] == 1)].sort_values("sample")
        plots.plot_monitoring(g, pca, f, 1, args.fault_start, figs / f"03_monitoring_fault{f}.png")
        sub = contrib[contrib["fault"] == f]
        if not sub.empty:
            top = sub.rename(columns={"contribution_pct": "contribution_%"})
            plots.plot_contrib(top, f, sub["statistic"].iloc[0], figs / f"04_contributions_fault{f}.png")

    # ---- 8. summary.md ----------------------------------------------------------
    with open(out / "summary.md", "w") as fh:
        fh.write("# Results summary\n\n")
        if args.synthetic:
            fh.write("> **SYNTHETIC DATA - smoke test only. Do not quote these numbers.**\n\n")
        fh.write(f"- PCA: k = {pca.k} components ({pca.cum_var[pca.k - 1] * 100:.1f}% of variance)\n")
        fh.write(f"- Control limits: {pca.limit_method}, {args.alpha * 100:g}% level\n")
        fh.write(f"- Training: {len(fit_runs)} fault-free runs; thresholds from {len(val_runs)} held-out runs\n")
        fh.write(f"- Testing: {args.test_runs} runs per fault; fault starts after sample {args.fault_start}\n")
        fh.write(f"- Delay = minutes from onset to {args.consec} consecutive alarms (median over detected runs)\n\n")
        fh.write("False alarm rate on fault-free test runs: " +
                 ", ".join(f"{m_} {v:.2f}%" for m_, v in far0.items()) + "\n\n")
        fh.write(md_table(summary) + "\n")

    print(f"\nDone in {time.time() - t0:.0f}s. Results written to {out.resolve()}")


if __name__ == "__main__":
    main()