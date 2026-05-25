"""
Aegis-Quant-Lab — Cross-Asset Dependency Analyst
==================================================
Portfolio-level analysis layer.
  • Rolling Pearson & Spearman correlations
  • Engle–Granger cointegration test
  • Hierarchical clustering of asset return profiles
  • Beta-coefficient estimation vs. BTC

All computations are vectorised (pandas / numpy / scipy).
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from statsmodels.tsa.stattools import coint

import config

logger = logging.getLogger("aegis.cross_asset")

# ──────────────────────────────────────────────
# Return matrices
# ──────────────────────────────────────────────

def _build_return_matrix(
    data: dict[str, pd.DataFrame],
    col: str = "close",
) -> pd.DataFrame:
    """Build an aligned returns matrix (symbols as columns)."""
    closes: dict[str, pd.Series] = {}
    for sym, df in data.items():
        if col in df.columns and len(df) > 0:
            closes[sym] = df[col].astype(np.float64)  # need float64 for corr precision
    if not closes:
        return pd.DataFrame()
    price_df = pd.DataFrame(closes)
    price_df.sort_index(inplace=True)
    price_df.ffill(inplace=True)
    returns = price_df.pct_change().dropna(how="all")
    return returns


# ──────────────────────────────────────────────
# Correlation matrices
# ──────────────────────────────────────────────

def pearson_correlation(
    data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Full Pearson correlation matrix on log-returns."""
    returns = _build_return_matrix(data)
    if returns.empty:
        return pd.DataFrame()
    return returns.corr(method="pearson")


def spearman_correlation(
    data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Full Spearman rank correlation matrix on log-returns."""
    returns = _build_return_matrix(data)
    if returns.empty:
        return pd.DataFrame()
    return returns.corr(method="spearman")


def rolling_correlation(
    data: dict[str, pd.DataFrame],
    window: int | None = None,
    method: str = "pearson",
) -> dict[tuple[str, str], pd.Series]:
    """
    Pairwise rolling correlations between all assets.
    Returns dict  {(symA, symB): Series_of_rolling_corr}.
    """
    if window is None:
        window = config.ROLLING_CORR_WINDOW
    returns = _build_return_matrix(data)
    if returns.empty:
        return {}

    symbols = list(returns.columns)
    results: dict[tuple[str, str], pd.Series] = {}
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            a, b = symbols[i], symbols[j]
            rc = returns[a].rolling(window).corr(returns[b])
            results[(a, b)] = rc
    return results


# ──────────────────────────────────────────────
# Cointegration
# ──────────────────────────────────────────────

def cointegration_matrix(
    data: dict[str, pd.DataFrame],
    progress_cb: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """
    Engle-Granger cointegration p-values for every pair.
    Returns a symmetric DataFrame of p-values.
    """
    closes: dict[str, pd.Series] = {}
    for sym, df in data.items():
        if "close" in df.columns and len(df) > 100:
            closes[sym] = df["close"].dropna().astype(np.float64)

    symbols = list(closes.keys())
    n = len(symbols)
    pval_matrix = pd.DataFrame(np.ones((n, n)), index=symbols, columns=symbols)

    total_pairs = n * (n - 1) // 2
    done = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = symbols[i], symbols[j]
            # Align series
            aligned = pd.concat([closes[a], closes[b]], axis=1, join="inner").dropna()
            if len(aligned) < 100:
                continue
            try:
                _, pvalue, _ = coint(aligned.iloc[:, 0].values, aligned.iloc[:, 1].values)
                pval_matrix.loc[a, b] = pvalue
                pval_matrix.loc[b, a] = pvalue
            except Exception as e:
                logger.debug(f"Coint test failed for {a}/{b}: {e}")

            done += 1
            if done % 50 == 0 and progress_cb:
                progress_cb(f"  Cointegration: {done}/{total_pairs} pairs tested")

    return pval_matrix


# ──────────────────────────────────────────────
# Highly correlated pairs
# ──────────────────────────────────────────────

def find_high_corr_pairs(
    corr_matrix: pd.DataFrame,
    threshold: float | None = None,
) -> list[dict]:
    """
    Extract pairs with |correlation| > threshold.
    Returns list of {symA, symB, corr_value}.
    """
    if threshold is None:
        threshold = config.CORRELATION_THRESHOLD
    pairs: list[dict] = []
    symbols = corr_matrix.columns.tolist()
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            val = corr_matrix.iloc[i, j]
            if abs(val) >= threshold:
                pairs.append({
                    "symbol_a": symbols[i],
                    "symbol_b": symbols[j],
                    "correlation": round(float(val), 4),
                })
    pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    return pairs


def find_cointegrated_pairs(
    coint_pvalues: pd.DataFrame,
    threshold: float = 0.05,
) -> list[dict]:
    """Return pairs that are cointegrated at the given significance level."""
    pairs: list[dict] = []
    symbols = coint_pvalues.columns.tolist()
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            pval = coint_pvalues.iloc[i, j]
            if pval < threshold:
                pairs.append({
                    "symbol_a": symbols[i],
                    "symbol_b": symbols[j],
                    "coint_pvalue": round(float(pval), 6),
                })
    pairs.sort(key=lambda x: x["coint_pvalue"])
    return pairs


# ──────────────────────────────────────────────
# Clustering
# ──────────────────────────────────────────────

def hierarchical_clustering(
    corr_matrix: pd.DataFrame,
    n_clusters: int = 5,
) -> dict[str, int]:
    """
    Hierarchical (Ward) clustering of assets based on correlation distance.
    Returns {symbol: cluster_id}.
    """
    if corr_matrix.empty or len(corr_matrix) < 3:
        return {}
    # Distance = 1 - |correlation|
    dist = 1 - corr_matrix.abs().values
    np.fill_diagonal(dist, 0)
    dist = np.clip(dist, 0, None)  # remove negatives from rounding

    # Make symmetric
    dist = (dist + dist.T) / 2

    try:
        condensed = squareform(dist, checks=False)
        Z = linkage(condensed, method="ward")
        labels = fcluster(Z, t=n_clusters, criterion="maxclust")
        return dict(zip(corr_matrix.columns, labels.tolist()))
    except Exception as e:
        logger.warning(f"Hierarchical clustering failed: {e}")
        return {}


def kmeans_clustering(
    data: dict[str, pd.DataFrame],
    n_clusters: int = 5,
) -> dict[str, int]:
    """K-Means clustering of assets based on normalized return statistics."""
    stats_rows = []
    syms = []
    for sym, df in data.items():
        if "close" not in df.columns or len(df) < 50:
            continue
        ret = df["close"].pct_change().dropna()
        stats_rows.append([
            float(ret.mean()),
            float(ret.std()),
            float(ret.skew()),
            float(ret.kurtosis()),
            float(ret.quantile(0.05)),
            float(ret.quantile(0.95)),
        ])
        syms.append(sym)

    if len(stats_rows) < n_clusters:
        return {}

    X = np.array(stats_rows, dtype=np.float64)
    # Normalize
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)

    km = KMeans(n_clusters=min(n_clusters, len(X)), random_state=42, n_init=10)
    labels = km.fit_predict(X)
    return dict(zip(syms, labels.tolist()))


# ──────────────────────────────────────────────
# Beta coefficients (vs BTC)
# ──────────────────────────────────────────────

def compute_betas(
    data: dict[str, pd.DataFrame],
    benchmark: str = "BTC/USDT",
) -> dict[str, float]:
    """Compute beta coefficient of each asset relative to a benchmark."""
    returns = _build_return_matrix(data)
    if benchmark not in returns.columns:
        logger.warning(f"Benchmark {benchmark} not found in data")
        return {}

    bench = returns[benchmark].dropna()
    betas: dict[str, float] = {}
    var_bench = bench.var()
    if var_bench == 0:
        return {}

    for sym in returns.columns:
        if sym == benchmark:
            betas[sym] = 1.0
            continue
        aligned = pd.concat([returns[sym], bench], axis=1, join="inner").dropna()
        if len(aligned) < 30:
            continue
        cov_val = aligned.iloc[:, 0].cov(aligned.iloc[:, 1])
        betas[sym] = round(float(cov_val / var_bench), 4)
    return betas


# ──────────────────────────────────────────────
# Master analysis runner
# ──────────────────────────────────────────────

def run_full_analysis(
    data: dict[str, pd.DataFrame],
    progress_cb: Callable[[str], None] | None = None,
) -> dict:
    """
    Run the complete cross-asset analysis pipeline.
    Returns a dict with all results.
    """
    results: dict = {}

    # 1. Correlations
    msg = "Cross-Asset: Computing Pearson correlation matrix …"
    logger.info(msg)
    if progress_cb:
        progress_cb(msg)
    results["pearson_corr"] = pearson_correlation(data)

    msg = "Cross-Asset: Computing Spearman correlation matrix …"
    logger.info(msg)
    if progress_cb:
        progress_cb(msg)
    results["spearman_corr"] = spearman_correlation(data)

    # 2. High-correlation pairs
    if not results["pearson_corr"].empty:
        results["high_corr_pairs"] = find_high_corr_pairs(results["pearson_corr"])
        msg = f"  Found {len(results['high_corr_pairs'])} highly correlated pairs (|r| >= {config.CORRELATION_THRESHOLD})"
        logger.info(msg)
        if progress_cb:
            progress_cb(msg)

    # 3. Cointegration
    msg = "Cross-Asset: Running Engle-Granger cointegration tests …"
    logger.info(msg)
    if progress_cb:
        progress_cb(msg)
    results["coint_pvalues"] = cointegration_matrix(data, progress_cb)
    results["cointegrated_pairs"] = find_cointegrated_pairs(results["coint_pvalues"])
    msg = f"  Found {len(results['cointegrated_pairs'])} cointegrated pairs (p < 0.05)"
    logger.info(msg)
    if progress_cb:
        progress_cb(msg)

    # 4. Clustering
    msg = "Cross-Asset: Clustering assets …"
    logger.info(msg)
    if progress_cb:
        progress_cb(msg)
    if not results["pearson_corr"].empty:
        results["hierarchical_clusters"] = hierarchical_clustering(results["pearson_corr"])
        results["kmeans_clusters"] = kmeans_clustering(data)

    # 5. Beta coefficients
    results["betas"] = compute_betas(data)

    # 6. Rolling correlations (summary — mean of last window)
    rolling = rolling_correlation(data)
    rolling_summary = {}
    for (a, b), series in rolling.items():
        last_vals = series.dropna().tail(config.ROLLING_CORR_WINDOW)
        if len(last_vals) > 0:
            rolling_summary[f"{a}|{b}"] = round(float(last_vals.mean()), 4)
    results["rolling_corr_summary"] = rolling_summary

    return results
