"""
plots.py - static figures for the TEP fault-detection project.

Colours follow a validated two-series palette (blue / orange on a light
surface); text always uses neutral ink, never the series colour.
"""
from __future__ import annotations

import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tep_fd import SAMPLE_MINUTES

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASE = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 10, "text.color": INK,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": BASE, "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.axisbelow": True,
})


def _grid(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.tick_params(length=0)


def _titles(fig, title, subtitle=None, top=0.88):
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold", y=0.985)
    if subtitle:
        fig.text(0.01, 0.925, subtitle, ha="left", va="top", fontsize=9.5, color=INK2)
    return top


def _save(fig, path, top):
    fig.tight_layout(rect=[0, 0, 1, top])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scree(monitor, path, n_show: int = 30):
    cum = monitor.cum_var[:n_show] * 100
    x = np.arange(1, len(cum) + 1)
    k, cum_k = monitor.k, monitor.cum_var[monitor.k - 1] * 100
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.plot(x, cum, color=BLUE, lw=1.4, solid_capstyle="round")
    ax.plot([k], [cum_k], "o", ms=8, color=BLUE, mec=SURFACE, mew=2, zorder=3)
    ax.annotate(f"k = {k} components keep {cum_k:.1f}% of the variance",
                xy=(k, cum_k), xytext=(k + 1.5, max(cum_k - 18, 5)),
                color=INK2, fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative variance explained (%)")
    _grid(ax)
    top = _titles(fig, "Variance explained by the PCA model",
                  "Fitted on normal operation data only")
    _save(fig, path, top)


def plot_monitoring(g, monitor, fault, run, fault_start, path):
    """T2 and SPE over time for one test run, with control limits."""
    hrs = g["sample"].to_numpy() * SAMPLE_MINUTES / 60
    onset = fault_start * SAMPLE_MINUTES / 60
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    panels = ((axes[0], "t2", monitor.t2_limit, "Hotelling T²"),
              (axes[1], "spe", monitor.spe_limit, "SPE (Q)"))
    for ax, col, lim, name in panels:
        ax.plot(hrs, g[col].to_numpy(), color=BLUE, lw=1.0, solid_capstyle="round")
        ax.set_yscale("log")
        ax.axhline(lim, color=INK2, lw=0.9)
        ax.text(hrs.min(), lim * 1.12, f"{monitor.alpha * 100:g}% control limit",
                ha="left", va="bottom", color=INK2, fontsize=8.5)
        ax.axvline(onset, color=MUTED, lw=1.0)
        ax.set_ylabel(name)
        _grid(ax)
    axes[0].text(onset - 0.2, 0.93, "fault starts", transform=axes[0].get_xaxis_transform(),
                 color=INK2, fontsize=8.5, va="top", ha="right")
    axes[1].set_xlabel("Time (hours)")
    top = _titles(fig, f"Fault {fault}: monitoring statistics (test run {run})",
                  "Log scale. A sample is an alarm when it is above the control limit.")
    _save(fig, path, top)


def plot_fdr(metrics, path):
    """Grouped bars: fault detection rate per fault, PCA vs Isolation Forest."""
    m = metrics[metrics["fault"] > 0]
    pca = m[m["method"] == "PCA (T2 or SPE)"].set_index("fault")["FDR_pct"]
    iff = m[m["method"] == "Isolation Forest"].set_index("fault")["FDR_pct"]
    faults = sorted(pca.index)
    x = np.arange(len(faults))
    w = 0.36
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(faults) + 3), 4.8))
    ax.bar(x - w / 2 - 0.02, pca.loc[faults].to_numpy(), w, color=BLUE, label="PCA (T² or SPE)")
    ax.bar(x + w / 2 + 0.02, iff.loc[faults].to_numpy(), w, color=ORANGE, label="Isolation Forest")
    ax.set_xticks(x)
    ax.set_xticklabels(faults)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Fault number")
    ax.set_ylabel("Detection rate after fault onset (%)")
    ax.legend(frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.0))
    _grid(ax)
    top = _titles(fig, "Fault detection rate by fault",
                  "Share of post-fault samples flagged as alarms. Higher is better.", top=0.88)
    _save(fig, path, top)


def plot_contrib(top_df, fault, which, path):
    """Top contributing variables for one fault (root-cause view)."""
    d = top_df.iloc[::-1]
    labels = [f"{r.variable}  {textwrap.shorten(str(r.description), 34, placeholder='…')}".strip()
              for r in d.itertuples()]
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(d) + 1.8))
    y = np.arange(len(d))
    ax.barh(y, d["contribution_%"].to_numpy(), height=0.55, color=BLUE)
    for yi, v in zip(y, d["contribution_%"].to_numpy()):
        ax.text(v + 0.4, yi, f"{v:.0f}%", va="center", color=INK2, fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=INK2)
    ax.set_xlabel("Share of total contribution (%)")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.tick_params(length=0)
    ax.set_xlim(0, max(d["contribution_%"].max() * 1.15, 5))
    stat = "T²" if which.lower() == "t2" else "SPE"
    top = _titles(fig, f"Fault {fault}: which variables drive the {stat} alarm?",
                  "Average contribution over the first flagged post-fault samples", top=0.86)
    _save(fig, path, top)
