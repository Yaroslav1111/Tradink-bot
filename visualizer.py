"""
Aegis-Quant-Lab — Visualisation Engine
========================================
Generates publication-quality charts:
  • Correlation heatmap (Pearson / Spearman)
  • Cluster dendrogram
  • Top indicator ranking bar chart
  • Equity curves for top strategies
All outputs saved as PNG to /output/.
"""

from __future__ import annotations

import logging
import os

import matplotlib
matplotlib.use("Agg")  # headless — no GUI backend needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

import config

logger = logging.getLogger("aegis.visualizer")
sns.set_theme(style="darkgrid", palette="deep")


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    title: str = "Cross-Asset Pearson Correlation",
    filename: str = "correlation_heatmap.png",
) -> str:
    """Save a correlation heatmap to output/."""
    if corr_matrix.empty:
        return ""

    n = len(corr_matrix)
    fig_size = max(10, n * 0.4)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=n <= 25,
        fmt=".2f" if n <= 25 else "",
        cmap="RdYlGn",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.3,
        cbar_kws={"shrink": 0.7},
        ax=ax,
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(config.OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")
    return path


def plot_top_indicators(
    top_strategies: list,
    title: str = "Top Indicators by OOS Sharpe Ratio",
    filename: str = "top_indicators.png",
    max_items: int = 25,
) -> str:
    """Horizontal bar chart of top strategies by Sharpe."""
    if not top_strategies:
        return ""

    items = top_strategies[:max_items]
    names = [f"{m.name} ({m.symbol})" for m in items]
    sharpes = [m.sharpe_ratio for m in items]
    colors = ["#2ecc71" if s > 0 else "#e74c3c" for s in sharpes]

    fig, ax = plt.subplots(figsize=(12, max(6, len(items) * 0.4)))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, sharpes, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Out-of-Sample Sharpe Ratio")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axvline(x=0, color="grey", linewidth=0.8, linestyle="--")
    plt.tight_layout()

    path = os.path.join(config.OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")
    return path


def plot_equity_curves(
    top_strategies: list,
    title: str = "Equity Curves — Top Strategies",
    filename: str = "equity_curves.png",
    max_curves: int = 10,
) -> str:
    """Cumulative PnL curves for the top strategies."""
    if not top_strategies:
        return ""

    fig, ax = plt.subplots(figsize=(14, 7))
    for m in top_strategies[:max_curves]:
        if not m.trades:
            continue
        pnls = [t.pnl_net for t in m.trades]
        cum = np.cumsum(pnls) * 100  # convert to percentage
        label = f"{m.name} ({m.symbol}) SR={m.sharpe_ratio:.2f}"
        ax.plot(cum, label=label, linewidth=1.2)

    ax.set_xlabel("Trade #")
    ax.set_ylabel("Cumulative PnL (%)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=7, loc="upper left")
    ax.axhline(y=0, color="grey", linewidth=0.8, linestyle="--")
    plt.tight_layout()

    path = os.path.join(config.OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")
    return path


def plot_cluster_dendrogram(
    corr_matrix: pd.DataFrame,
    title: str = "Asset Cluster Dendrogram",
    filename: str = "cluster_dendrogram.png",
) -> str:
    """Hierarchical clustering dendrogram based on correlation distance."""
    if corr_matrix.empty or len(corr_matrix) < 3:
        return ""

    dist = 1 - corr_matrix.abs().values
    np.fill_diagonal(dist, 0)
    dist = np.clip(dist, 0, None)
    dist = (dist + dist.T) / 2

    try:
        condensed = squareform(dist, checks=False)
        Z = linkage(condensed, method="ward")
    except Exception as e:
        logger.warning(f"Dendrogram linkage failed: {e}")
        return ""

    fig, ax = plt.subplots(figsize=(max(12, len(corr_matrix) * 0.3), 7))
    dendrogram(
        Z,
        labels=corr_matrix.columns.tolist(),
        leaf_rotation=90,
        leaf_font_size=8,
        ax=ax,
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("Distance (1 - |correlation|)")
    plt.tight_layout()

    path = os.path.join(config.OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")
    return path


def plot_losing_patterns(
    losing: list,
    title: str = "Anti-Strategies — Losing Patterns",
    filename: str = "losing_patterns.png",
    max_items: int = 20,
) -> str:
    """Bar chart of worst-performing indicator/symbol combos."""
    if not losing:
        return ""

    items = sorted(losing, key=lambda m: m.total_pnl)[:max_items]
    names = [f"{m.name} ({m.symbol})" for m in items]
    pnls = [m.total_pnl * 100 for m in items]

    fig, ax = plt.subplots(figsize=(12, max(6, len(items) * 0.4)))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, pnls, color="#e74c3c", edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Total PnL (%)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axvline(x=0, color="grey", linewidth=0.8, linestyle="--")
    plt.tight_layout()

    path = os.path.join(config.OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")
    return path


def generate_all_charts(
    backtest_results: dict,
    cross_asset_results: dict | None = None,
) -> list[str]:
    """Generate the full suite of charts. Returns list of file paths."""
    paths: list[str] = []

    # Correlation heatmaps
    if cross_asset_results:
        pcorr = cross_asset_results.get("pearson_corr")
        if pcorr is not None and not pcorr.empty:
            p = plot_correlation_heatmap(pcorr, "Pearson Correlation", "pearson_heatmap.png")
            if p:
                paths.append(p)

        scorr = cross_asset_results.get("spearman_corr")
        if scorr is not None and not scorr.empty:
            p = plot_correlation_heatmap(scorr, "Spearman Correlation", "spearman_heatmap.png")
            if p:
                paths.append(p)

        # Dendrogram
        if pcorr is not None and not pcorr.empty:
            p = plot_cluster_dendrogram(pcorr)
            if p:
                paths.append(p)

    # Top indicators
    top = backtest_results.get("top_strategies", [])
    if top:
        p = plot_top_indicators(top)
        if p:
            paths.append(p)
        p = plot_equity_curves(top)
        if p:
            paths.append(p)

    # Losing patterns
    losers = backtest_results.get("losing_patterns", [])
    if losers:
        p = plot_losing_patterns(losers)
        if p:
            paths.append(p)

    return paths
