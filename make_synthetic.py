"""
make_synthetic.py - generate SYNTHETIC data in the same layout as the real TEP files.

Purpose: smoke-test the pipeline (and let you run it before the real data is
downloaded). The data are NOT from the Tennessee Eastman process. Never quote
results from this data on a resume - only results from the real dataset.

Fault ids used here (just for testing):
    1 shift along the normal correlation direction  (T2 should react)
    2 bias on one sensor                            (SPE should react)
    3 no change in the measurements                 (undetectable by design)
    4 slow ramp on two sensors
    5 decoupling of six correlated sensors          (SPE)
    6 frozen sensor                                 (SPE)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tep_fd import ID_COLS, VARS, FAULT_START_TEST

P, R = len(VARS), 6


def _build_model(rng):
    scale = np.linspace(2.0, 1.0, R)
    A = rng.normal(size=(P, R)) * scale / np.sqrt(R)
    mu = rng.uniform(10, 100, P)
    sd = rng.uniform(1, 10, P)
    return A, mu, sd


def _run(rng, A, mu, sd, n, fault=0, start=FAULT_START_TEST):
    phi = 0.9
    f = np.zeros((n, R))
    f[0] = rng.normal(size=R)
    e = rng.normal(size=(n, R)) * np.sqrt(1 - phi ** 2)
    for t in range(1, n):
        f[t] = phi * f[t - 1] + e[t]
    X = f @ A.T + 0.25 * rng.normal(size=(n, P))
    if fault:
        idx = np.arange(n) >= start
        m = int(idx.sum())
        if fault == 1:
            X[idx] += 5.0 * A[:, 0]
        elif fault == 2:
            X[idx, 9] += 6.0
        elif fault == 4:
            ramp = np.linspace(0, 8, m)
            X[idx, 19] += ramp
            X[idx, 20] += 0.8 * ramp
        elif fault == 5:
            s = np.sqrt((A[29:35] ** 2).sum(1) + 0.25 ** 2)
            X[idx, 29:35] = rng.normal(size=(m, 6)) * s
        elif fault == 6:
            X[idx, 4] = X[start - 1, 4]
        # fault 3: nothing changes
    return mu + sd * X


def _frame(X, fault, run):
    n = X.shape[0]
    df = pd.DataFrame(X, columns=VARS)
    df.insert(0, "sample", np.arange(1, n + 1))
    df.insert(0, "simulationRun", run)
    df.insert(0, "faultNumber", fault)
    return df[ID_COLS + VARS]


def make_synthetic(out_dir, n_train_runs: int = 60, n_test_runs: int = 6,
                   faults=(1, 2, 3, 4, 5, 6), seed: int = 0):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    A, mu, sd = _build_model(rng)

    train = [_frame(_run(rng, A, mu, sd, 500), 0, r) for r in range(1, n_train_runs + 1)]
    pd.concat(train).to_csv(out_dir / "TEP_FaultFree_Training.csv", index=False)

    test0 = [_frame(_run(rng, A, mu, sd, 960), 0, r) for r in range(1, n_test_runs + 1)]
    pd.concat(test0).to_csv(out_dir / "TEP_FaultFree_Testing.csv", index=False)

    faulty = []
    for fl in faults:
        for r in range(1, n_test_runs + 1):
            faulty.append(_frame(_run(rng, A, mu, sd, 960, fault=fl), fl, r))
    pd.concat(faulty).to_csv(out_dir / "TEP_Faulty_Testing.csv", index=False)
    return out_dir


if __name__ == "__main__":
    import sys
    d = make_synthetic(sys.argv[1] if len(sys.argv) > 1 else "synthetic_data")
    print("Synthetic data written to", d)
