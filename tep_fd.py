"""
tep_fd.py - Fault detection on the Tennessee Eastman Process (TEP) benchmark.

Methods
-------
* PCAMonitor     : PCA model of normal operation, monitored with Hotelling's T^2
                   and SPE (Q). Includes per-variable contribution analysis.
* IForestMonitor : Isolation Forest baseline (scikit-learn).
* Evaluation     : fault detection rate (FDR), false alarm rate (FAR) and
                   detection delay per fault.

Data format expected (Rieth et al., 2017, "Additional Tennessee Eastman Process
Simulation Data for Anomaly Detection Evaluation", Harvard Dataverse):
    columns: faultNumber, simulationRun, sample, xmeas_1..xmeas_41, xmv_1..xmv_11
    * sampling interval: 3 minutes
    * fault-free training : 500 runs x 500 samples  (25 h)
    * faulty training     : faults 1-20, fault starts after hour 1  (sample 20)
    * fault-free testing  : 500 runs x 960 samples  (48 h)
    * faulty testing      : faults 1-20, fault starts after hour 8  (sample 160)
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
XMEAS = [f"xmeas_{i}" for i in range(1, 42)]   # 41 measured variables
XMV = [f"xmv_{i}" for i in range(1, 12)]       # 11 manipulated variables
VARS = XMEAS + XMV                              # 52 process variables
ID_COLS = ["faultNumber", "simulationRun", "sample"]

SAMPLE_MINUTES = 3
FAULT_START_TRAIN = 20     # faulty TRAINING runs: fault introduced after 1 h = 20 samples (500 samples/run)
FAULT_START_TEST = 160     # fault introduced after 8 h = 160 samples (testing set)

# Approximate descriptive labels (Downs & Vogel, 1993). Please double-check
# against the paper before quoting them in an interview.
VAR_LABELS = {
    "xmeas_1": "A feed (stream 1)", "xmeas_2": "D feed (stream 2)",
    "xmeas_3": "E feed (stream 3)", "xmeas_4": "A+C feed (stream 4)",
    "xmeas_5": "Recycle flow (stream 8)", "xmeas_6": "Reactor feed rate (stream 6)",
    "xmeas_7": "Reactor pressure", "xmeas_8": "Reactor level",
    "xmeas_9": "Reactor temperature", "xmeas_10": "Purge rate (stream 9)",
    "xmeas_11": "Product separator temp", "xmeas_12": "Product separator level",
    "xmeas_13": "Product separator pressure", "xmeas_14": "Product separator underflow",
    "xmeas_15": "Stripper level", "xmeas_16": "Stripper pressure",
    "xmeas_17": "Stripper underflow", "xmeas_18": "Stripper temperature",
    "xmeas_19": "Stripper steam flow", "xmeas_20": "Compressor work",
    "xmeas_21": "Reactor cooling water outlet temp",
    "xmeas_22": "Separator cooling water outlet temp",
    "xmv_1": "D feed flow valve", "xmv_2": "E feed flow valve",
    "xmv_3": "A feed flow valve", "xmv_4": "A+C feed flow valve",
    "xmv_5": "Compressor recycle valve", "xmv_6": "Purge valve",
    "xmv_7": "Separator liquid flow valve", "xmv_8": "Stripper liquid flow valve",
    "xmv_9": "Stripper steam valve", "xmv_10": "Reactor cooling water valve",
    "xmv_11": "Condenser cooling water valve",
}


def var_label(name: str) -> str:
    """Readable label for a variable; composition analysers (xmeas_23-41) stay generic."""
    if name in VAR_LABELS:
        return f"{name} - {VAR_LABELS[name]}"
    n = int(name.split("_")[1])
    if name.startswith("xmeas") and n >= 23:
        return f"{name} - composition analyser"
    return name


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def find_tep_file(data_dir: str | Path, faulty: bool, split: str) -> Path:
    """Locate one of the four TEP files in `data_dir` by (loose) name matching.

    split: "train" or "test". Prefers CSV over RData.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    cands = []
    for p in data_dir.iterdir():
        name = _norm(p.name)
        if not p.is_file():
            continue
        if not name.endswith(("csv", "csvgz", "rdata", "rda")):
            continue
        is_ff = "faultfree" in name
        is_fy = ("faulty" in name) and not is_ff
        if faulty and not is_fy:
            continue
        if not faulty and not is_ff:
            continue
        if split == "train" and "train" not in name:
            continue
        if split == "test" and "test" not in name:
            continue
        cands.append(p)
    if not cands:
        kind = "faulty" if faulty else "fault-free"
        raise FileNotFoundError(
            f"No {kind} {split}ing file (.csv / .RData) found in {data_dir}. "
            "Expected names like TEP_FaultFree_Training.csv, TEP_Faulty_Testing.csv."
        )
    cands.sort(key=lambda p: (0 if p.name.lower().endswith((".csv", ".csv.gz")) else 1, p.name))
    return cands[0]


def _read_rdata(path: Path, faults, max_run):
    """Read an .RData table with as little peak memory as possible.

    pyreadr's default output is one big pandas DataFrame, which needs roughly twice
    the table size in RAM (the 9.6M-row faulty testing file wants ~8 GB). Asking
    pyreadr for a plain dict of column arrays skips that extra copy, and we keep
    only the requested rows straight away. If this pyreadr version does not offer
    that format, we fall back to the default DataFrame.
    """
    import pyreadr

    obj = None
    try:
        res = pyreadr.read_r(str(path), output_format="dict")
        obj = res if "faultNumber" in res else next(iter(res.values()))
    except MemoryError as e:
        raise MemoryError(
            f"Not enough free RAM to load {path.name}. Close other programs and retry, or use the "
            "smaller faulty file (--faulty-split train), or cut the file down on another machine "
            "with convert_rdata_to_csv.py."
        ) from e
    except Exception:
        obj = None

    if isinstance(obj, dict):
        n = len(next(iter(obj.values())))
        mask = np.ones(n, dtype=bool)
        if faults is not None:
            mask &= np.isin(np.asarray(obj["faultNumber"]), list(faults))
        if max_run is not None:
            mask &= np.asarray(obj["simulationRun"]) <= max_run
        wanted = set(ID_COLS + VARS)
        return pd.DataFrame({c: np.asarray(v)[mask] for c, v in obj.items() if c in wanted})

    try:
        res = pyreadr.read_r(str(path))
    except MemoryError as e:
        raise MemoryError(
            f"Not enough free RAM to load {path.name}. Close other programs and retry, or use the "
            "smaller faulty file (--faulty-split train), or cut the file down on another machine "
            "with convert_rdata_to_csv.py."
        ) from e
    df = next(iter(res.values()))
    if faults is not None:
        df = df[df["faultNumber"].isin(set(int(f) for f in faults))]
    if max_run is not None:
        df = df[df["simulationRun"] <= max_run]
    return df


def load_tep(path: str | Path, faults=None, max_run: int | None = None,
             chunksize: int = 250_000) -> pd.DataFrame:
    """Load a TEP file, keeping only the requested faults and simulation runs.

    CSV files are read in chunks and filtered on the fly, so the 4 GB faulty
    testing file does not have to fit in memory. (.RData files must be loaded
    fully by pyreadr first - convert to CSV if memory is tight, see README.)
    """
    path = Path(path)
    faults = None if faults is None else set(int(f) for f in faults)

    def _filter(df: pd.DataFrame) -> pd.DataFrame:
        if faults is not None:
            df = df[df["faultNumber"].isin(faults)]
        if max_run is not None:
            df = df[df["simulationRun"] <= max_run]
        return df

    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith((".csv", ".csv.gz")):
        parts = [_filter(c) for c in pd.read_csv(path, chunksize=chunksize)]
        df = pd.concat(parts, ignore_index=True)
    elif suffixes.endswith((".rdata", ".rda")):
        try:
            import pyreadr  # optional dependency
        except ImportError as e:
            raise ImportError(
                "Reading .RData needs `pip install pyreadr` (or convert the files to CSV in R)."
            ) from e
        df = _read_rdata(path, faults, max_run)
    else:
        raise ValueError(f"Unsupported file type: {path.name}")

    missing = [c for c in ID_COLS + VARS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing expected columns: {missing[:6]} ...")
    df = df[ID_COLS + VARS].copy()
    for c in ["faultNumber", "simulationRun", "sample"]:
        df[c] = df[c].astype(int)
    df[VARS] = df[VARS].astype(np.float32)
    return df.sort_values(ID_COLS).reset_index(drop=True)


# --------------------------------------------------------------------------
# PCA monitor (Hotelling's T^2 and SPE / Q)
# --------------------------------------------------------------------------
class PCAMonitor:
    """PCA process-monitoring model.

    Parameters
    ----------
    variance : float
        Cumulative explained variance used to pick the number of components k
        (ignored if `n_components` is given).
    n_components : int | None
        Fix k directly.
    alpha : float
        Confidence level of the control limits (0.99 -> ~1 % false alarms
        on normal data).
    """

    def __init__(self, variance: float = 0.90, n_components: int | None = None,
                 alpha: float = 0.99):
        self.variance = variance
        self.n_components = n_components
        self.alpha = alpha
        self.scaler = StandardScaler()

    # ---- fitting ---------------------------------------------------------
    def fit(self, X_fit: np.ndarray, X_val: np.ndarray | None = None):
        Z = self.scaler.fit_transform(X_fit)
        pca = PCA(n_components=None, svd_solver="full").fit(Z)
        self.eigvals_all = pca.explained_variance_
        self.cum_var = np.cumsum(pca.explained_variance_ratio_)
        if self.n_components is not None:
            k = int(self.n_components)
        else:
            k = int(np.searchsorted(self.cum_var, self.variance) + 1)
        self.k = k
        self.P = pca.components_[:k].T               # loadings, shape (p, k)
        self.lam = self.eigvals_all[:k]              # variance of each score
        n = Z.shape[0]

        # Theoretical limits (assume independent, Gaussian samples)
        F = stats.f.ppf(self.alpha, k, n - k)
        self.t2_limit_theory = k * (n ** 2 - 1) / (n * (n - k)) * F
        # SPE limit: Box (1954) chi-square approximation, SPE ~ g * chi2(h) with
        # g = theta2/theta1 and h = theta1^2/theta2. (The Jackson-Mudholkar
        # formula breaks down when one residual eigenvalue dominates, so the
        # Box form is the safer "theoretical" reference.)
        res = self.eigvals_all[k:]
        th1, th2 = res.sum(), (res ** 2).sum()
        g, h = th2 / th1, th1 ** 2 / th2
        self.spe_limit_theory = float(g * stats.chi2.ppf(self.alpha, h))

        # Empirical limits from held-out normal data (preferred: TEP samples are
        # autocorrelated, so the theoretical limits give too many false alarms).
        self.t2_limit = self.t2_limit_theory
        self.spe_limit = self.spe_limit_theory
        self.limit_method = "theoretical"
        if X_val is not None:
            t2, spe = self.stats(X_val)
            self.t2_limit = float(np.quantile(t2, self.alpha))
            self.spe_limit = float(np.quantile(spe, self.alpha))
            self.limit_method = "empirical (held-out normal runs)"
        return self

    # ---- monitoring statistics ------------------------------------------
    def stats(self, X: np.ndarray):
        """Return (T2, SPE) for each row of X."""
        Z = self.scaler.transform(X)
        T = Z @ self.P
        t2 = np.sum(T ** 2 / self.lam, axis=1)
        R = Z - T @ self.P.T
        spe = np.sum(R ** 2, axis=1)
        return t2, spe

    def alarms(self, X: np.ndarray):
        """Return dict with T2, SPE and boolean alarm arrays."""
        t2, spe = self.stats(X)
        a_t2, a_spe = t2 > self.t2_limit, spe > self.spe_limit
        return {"t2": t2, "spe": spe, "alarm_t2": a_t2, "alarm_spe": a_spe,
                "alarm_pca": a_t2 | a_spe}

    # ---- root cause: variable contributions -----------------------------
    def contributions(self, X: np.ndarray):
        """Per-variable contributions to T2 and SPE (each row sums to its statistic).

        T2  : complete-decomposition contribution  c_j = z_j * (P L^-1 P^T z)_j
        SPE : squared residual of each variable    c_j = (z_j - zhat_j)^2
        """
        Z = self.scaler.transform(X)
        M = self.P @ np.diag(1.0 / self.lam) @ self.P.T
        c_t2 = Z * (Z @ M)
        R = Z - (Z @ self.P) @ self.P.T
        c_spe = R ** 2
        return c_t2, c_spe


# --------------------------------------------------------------------------
# Isolation Forest monitor (ML baseline)
# --------------------------------------------------------------------------
class IForestMonitor:
    """Isolation Forest trained on normal data; threshold set on held-out normal
    data at the same confidence level as the PCA limits (fair comparison)."""

    def __init__(self, alpha: float = 0.99, n_estimators: int = 200,
                 random_state: int = 0):
        self.alpha = alpha
        self.scaler = StandardScaler()
        self.model = IsolationForest(n_estimators=n_estimators,
                                     random_state=random_state, n_jobs=-1)

    def fit(self, X_fit: np.ndarray, X_val: np.ndarray):
        self.model.fit(self.scaler.fit_transform(X_fit))
        self.limit = float(np.quantile(self.score(X_val), self.alpha))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Anomaly score: higher = more abnormal."""
        return -self.model.score_samples(self.scaler.transform(X))

    def alarms(self, X: np.ndarray):
        s = self.score(X)
        return {"if_score": s, "alarm_if": s > self.limit}


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
METHODS = {
    "PCA T2": "alarm_t2",
    "PCA SPE": "alarm_spe",
    "PCA (T2 or SPE)": "alarm_pca",
    "Isolation Forest": "alarm_if",
}


def first_sustained(alarm: np.ndarray, start: int, consec: int = 3):
    """Index (relative to `start`) of the first run of `consec` consecutive alarms
    at or after `start`; None if the fault is never confirmed."""
    a = alarm[start:].astype(int)
    if len(a) < consec:
        return None
    run = np.convolve(a, np.ones(consec, dtype=int), mode="valid")
    hits = np.flatnonzero(run == consec)
    return int(hits[0]) if hits.size else None


def evaluate(df: pd.DataFrame, fault_start: int = FAULT_START_TEST,
             consec: int = 3) -> pd.DataFrame:
    """Per-fault metrics from a frame holding the alarm columns of METHODS.

    For every (fault, run, method):
      FAR   % of alarms among samples <= fault_start (fault-free part of the run;
            for fault 0 the whole run).
      FDR   % of alarms among samples  > fault_start.
      delay minutes from fault onset to the first 3 consecutive alarms.
    Results are averaged over runs.
    """
    rows = []
    for (fault, run), g in df.groupby(["faultNumber", "simulationRun"], sort=True):
        g = g.sort_values("sample")
        pre = (g["sample"].to_numpy() <= fault_start)
        for method, col in METHODS.items():
            if col not in g:
                continue
            a = g[col].to_numpy().astype(bool)
            if fault == 0:
                far, fdr, delay = a.mean() * 100, np.nan, np.nan
            else:
                far = a[pre].mean() * 100
                fdr = a[~pre].mean() * 100
                d = first_sustained(a, int(pre.sum()), consec)
                delay = np.nan if d is None else d * SAMPLE_MINUTES
            rows.append((fault, run, method, far, fdr, delay))
    per_run = pd.DataFrame(rows, columns=["fault", "run", "method", "FAR_%", "FDR_%", "delay_min"])
    out = (per_run.groupby(["fault", "method"])
           .agg(runs=("run", "nunique"),
                FAR_pct=("FAR_%", "mean"),
                FDR_pct=("FDR_%", "mean"),
                runs_detected=("delay_min", lambda s: int(s.notna().sum())),
                median_delay_min=("delay_min", "median"))
           .reset_index())
    out.loc[out["fault"] == 0, "runs_detected"] = np.nan
    return out.round(2)


def top_contributors(monitor: PCAMonitor, X_fault: np.ndarray, which: str = "spe",
                     top: int = 5, n_samples: int = 40) -> pd.DataFrame:
    """Average contribution of each variable over the first `n_samples` flagged
    samples, normalised to % of the total absolute contribution."""
    c_t2, c_spe = monitor.contributions(X_fault)
    c = c_spe if which == "spe" else c_t2
    t2, spe = monitor.stats(X_fault)
    flagged = np.flatnonzero((spe if which == "spe" else t2) >
                             (monitor.spe_limit if which == "spe" else monitor.t2_limit))
    if flagged.size == 0:
        return pd.DataFrame(columns=["variable", "contribution_%"])
    sel = c[flagged[:n_samples]].mean(axis=0)
    share = 100 * np.abs(sel) / np.abs(sel).sum()
    order = np.argsort(share)[::-1][:top]
    return pd.DataFrame({"variable": [VARS[i] for i in order],
                         "description": [var_label(VARS[i]).split(" - ", 1)[-1] if " - " in var_label(VARS[i]) else "" for i in order],
                         "contribution_%": share[order].round(1)})
