"""
Aegis-Quant-Lab — Risk Block Generator (Anti-Strategy Engine)
==============================================================
Aggregates losing patterns from the backtest and cross-asset analysis
to produce a defensive risk_blocks.json configuration.

Focus: identify conditions that consistently DESTROY capital, then emit
machine-readable prohibition rules.

Output format per block:
{
  "symbol": "TONUSDT",
  "forbidden_conditions": {
    "indicator_name": "RSI_14",
    "indicator_threshold": {"gt": 70},
    "hour_range": [2, 6],
    "day_of_week": [0, 6],
    "direction": "LONG",
    "highly_correlated_active_trade": "HYPEUSDT",
    "market_regime": "low_atr",
    "cluster_id": 2,
    "reason": "High false breakout rate during Asia session ..."
  }
}
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter, defaultdict
from typing import Any, Callable

import numpy as np
import pandas as pd

import config
from backtest_simulator import StrategyMetrics, TradeResult

logger = logging.getLogger("aegis.risk_gen")

# ──────────────────────────────────────────────
# Pattern detectors
# ──────────────────────────────────────────────

def _detect_toxic_hours(trades: list[TradeResult], loss_threshold: float = -0.003) -> list[int]:
    """Find hours where average PnL is consistently negative."""
    hour_pnls: dict[int, list[float]] = defaultdict(list)
    for t in trades:
        hour_pnls[t.hour].append(t.pnl_net)

    toxic_hours: list[int] = []
    for h, pnls in hour_pnls.items():
        if len(pnls) >= 5 and np.mean(pnls) < loss_threshold:
            toxic_hours.append(h)
    return sorted(toxic_hours)


def _detect_toxic_days(trades: list[TradeResult], loss_threshold: float = -0.003) -> list[int]:
    """Find days of week where average PnL is consistently negative."""
    day_pnls: dict[int, list[float]] = defaultdict(list)
    for t in trades:
        day_pnls[t.day_of_week].append(t.pnl_net)

    toxic_days: list[int] = []
    for d, pnls in day_pnls.items():
        if len(pnls) >= 5 and np.mean(pnls) < loss_threshold:
            toxic_days.append(d)
    return sorted(toxic_days)


def _detect_toxic_direction(trades: list[TradeResult]) -> str | None:
    """If one direction is significantly worse, flag it."""
    long_pnls = [t.pnl_net for t in trades if t.direction == "LONG"]
    short_pnls = [t.pnl_net for t in trades if t.direction == "SHORT"]

    long_avg = np.mean(long_pnls) if long_pnls else 0.0
    short_avg = np.mean(short_pnls) if short_pnls else 0.0

    # Flag direction that is significantly worse
    if len(long_pnls) >= 10 and long_avg < -0.005 and long_avg < short_avg * 0.5:
        return "LONG"
    if len(short_pnls) >= 10 and short_avg < -0.005 and short_avg < long_avg * 0.5:
        return "SHORT"
    return None


def _detect_hour_range(toxic_hours: list[int]) -> list[int] | None:
    """Compress individual toxic hours into contiguous ranges."""
    if not toxic_hours:
        return None
    # Find the longest contiguous block
    if len(toxic_hours) == 1:
        return toxic_hours
    # Simple: return the full list; consumer can interpret as forbidden zone
    return toxic_hours


def _detect_market_regime_failures(
    df: pd.DataFrame,
    trades: list[TradeResult],
) -> str | None:
    """
    Detect if losses concentrate in a specific ATR regime.
    Returns 'high_atr' or 'low_atr' or None.
    """
    if "atr_pct" not in df.columns or len(trades) < 10:
        return None

    atr_median = df["atr_pct"].median()
    high_atr_losses = []
    low_atr_losses = []

    for t in trades:
        if t.entry_bar < len(df):
            atr_val = df["atr_pct"].iloc[t.entry_bar] if t.entry_bar < len(df) else np.nan
            if np.isnan(atr_val):
                continue
            if atr_val > atr_median:
                high_atr_losses.append(t.pnl_net)
            else:
                low_atr_losses.append(t.pnl_net)

    if high_atr_losses and np.mean(high_atr_losses) < -0.005 and len(high_atr_losses) >= 5:
        if low_atr_losses and np.mean(low_atr_losses) > np.mean(high_atr_losses) * 2:
            return "high_atr"

    if low_atr_losses and np.mean(low_atr_losses) < -0.005 and len(low_atr_losses) >= 5:
        if high_atr_losses and np.mean(high_atr_losses) > np.mean(low_atr_losses) * 2:
            return "low_atr"

    return None


# ──────────────────────────────────────────────
# Block builder
# ──────────────────────────────────────────────

def _build_risk_block(
    metric: StrategyMetrics,
    df: pd.DataFrame | None,
    high_corr_pairs: list[dict] | None = None,
    clusters: dict[str, int] | None = None,
) -> dict[str, Any] | None:
    """Build a single risk block from a losing strategy."""
    trades = metric.trades
    if not trades or metric.total_trades < 10:
        return None

    sym_clean = metric.symbol.replace("/", "")

    block: dict[str, Any] = {
        "symbol": sym_clean,
        "forbidden_conditions": {},
    }
    fc = block["forbidden_conditions"]

    # Indicator info
    fc["indicator_name"] = metric.name
    fc["profit_factor"] = round(metric.profit_factor, 3)
    fc["sharpe_ratio"] = round(metric.sharpe_ratio, 3)
    fc["total_pnl_pct"] = round(metric.total_pnl * 100, 2)
    fc["total_trades"] = metric.total_trades
    fc["win_rate"] = round(metric.win_rate, 3)

    # Toxic hours
    toxic_h = _detect_toxic_hours(trades)
    if toxic_h:
        fc["hour_range"] = toxic_h

    # Toxic days
    toxic_d = _detect_toxic_days(trades)
    if toxic_d:
        fc["day_of_week"] = toxic_d

    # Toxic direction
    toxic_dir = _detect_toxic_direction(trades)
    if toxic_dir:
        fc["direction"] = toxic_dir

    # Market regime
    if df is not None:
        regime = _detect_market_regime_failures(df, trades)
        if regime:
            fc["market_regime"] = regime

    # Highly correlated assets
    if high_corr_pairs:
        correlated_with = []
        for pair in high_corr_pairs:
            if metric.symbol in (pair["symbol_a"], pair["symbol_b"]):
                other = pair["symbol_b"] if pair["symbol_a"] == metric.symbol else pair["symbol_a"]
                correlated_with.append(other.replace("/", ""))
        if correlated_with:
            fc["highly_correlated_active_trade"] = correlated_with[0]
            if len(correlated_with) > 1:
                fc["correlated_assets"] = correlated_with

    # Cluster info
    if clusters and metric.symbol in clusters:
        fc["cluster_id"] = clusters[metric.symbol]

    # Build human-readable reason
    reason_parts = []
    if toxic_h:
        reason_parts.append(f"losses concentrated in hours {toxic_h}")
    if toxic_d:
        day_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
        days_str = ", ".join(day_names.get(d, str(d)) for d in toxic_d)
        reason_parts.append(f"weak days: {days_str}")
    if toxic_dir:
        reason_parts.append(f"consistently losing on {toxic_dir} entries")
    if fc.get("market_regime"):
        reason_parts.append(f"fails in {fc['market_regime']} regime")
    if fc.get("highly_correlated_active_trade"):
        reason_parts.append(f"high cluster exposure risk with {fc['highly_correlated_active_trade']}")

    reason_parts.append(
        f"PF={metric.profit_factor:.2f}, WR={metric.win_rate:.1%}, DD={metric.max_drawdown:.2%}"
    )
    fc["reason"] = "; ".join(reason_parts)

    return block


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def generate_risk_blocks(
    backtest_results: dict,
    enriched_data: dict[str, pd.DataFrame] | None = None,
    cross_asset_results: dict | None = None,
    progress_cb: Callable[[str], None] | None = None,
) -> list[dict]:
    """
    Master risk block generator.

    Analyses losing patterns from backtest and cross-asset analysis
    to produce defensive prohibition rules.
    """
    blocks: list[dict] = []

    # Extract cross-asset info
    high_corr_pairs = []
    clusters: dict[str, int] = {}
    if cross_asset_results:
        high_corr_pairs = cross_asset_results.get("high_corr_pairs", [])
        clusters = cross_asset_results.get("hierarchical_clusters", {})

    # 1. Process all losing patterns from backtest
    losing_patterns = backtest_results.get("losing_patterns", [])
    msg = f"Risk Generator: Processing {len(losing_patterns)} losing patterns …"
    logger.info(msg)
    if progress_cb:
        progress_cb(msg)

    for metric in losing_patterns:
        df = enriched_data.get(metric.symbol) if enriched_data else None
        block = _build_risk_block(metric, df, high_corr_pairs, clusters)
        if block:
            blocks.append(block)

    # 2. Add correlation-based blocks
    # If two assets have corr > threshold, generate mutual blocking rules
    for pair in high_corr_pairs:
        for sym_key in ["symbol_a", "symbol_b"]:
            other_key = "symbol_b" if sym_key == "symbol_a" else "symbol_a"
            blocks.append({
                "symbol": pair[sym_key].replace("/", ""),
                "forbidden_conditions": {
                    "rule_type": "correlation_cluster_block",
                    "highly_correlated_active_trade": pair[other_key].replace("/", ""),
                    "correlation": pair["correlation"],
                    "reason": (
                        f"Correlation {pair['correlation']:.3f} with {pair[other_key]} — "
                        f"simultaneous entries duplicate cluster risk"
                    ),
                },
            })

    # 3. Aggregate: group blocks by symbol
    symbol_blocks: dict[str, list[dict]] = defaultdict(list)
    for b in blocks:
        symbol_blocks[b["symbol"]].append(b["forbidden_conditions"])

    # Final output structure
    output = []
    for sym, conditions in symbol_blocks.items():
        output.append({
            "symbol": sym,
            "forbidden_conditions": conditions if len(conditions) > 1 else conditions[0],
        })

    msg = f"Risk Generator: {len(output)} risk blocks generated for {len(symbol_blocks)} symbols"
    logger.info(msg)
    if progress_cb:
        progress_cb(msg)

    return output


def save_risk_blocks(blocks: list[dict], path: str | None = None) -> str:
    """Save risk blocks to JSON file."""
    if path is None:
        path = os.path.join(config.OUTPUT_DIR, "risk_blocks.json")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(blocks, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"Risk blocks saved to {path}")
    return path


def generate_summary_report(
    backtest_results: dict,
    cross_asset_results: dict | None = None,
) -> str:
    """Generate a human-readable summary report."""
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("  AEGIS-QUANT-LAB — ANALYSIS SUMMARY REPORT")
    lines.append("=" * 70)

    # Top strategies
    top = backtest_results.get("top_strategies", [])
    lines.append(f"\n{'─' * 50}")
    lines.append(f"  TOP STATISTICALLY VALID STRATEGIES: {len(top)}")
    lines.append(f"{'─' * 50}")
    for i, m in enumerate(top[:20], 1):
        lines.append(
            f"  {i:2d}. {m.name:20s} | {m.symbol:12s} | "
            f"PF={m.profit_factor:5.2f} | SR={m.sharpe_ratio:6.2f} | "
            f"WR={m.win_rate:5.1%} | Trades={m.total_trades:4d} | "
            f"MDD={m.max_drawdown:7.2%} | MW-p={m.mann_whitney_p:.4f}"
        )

    # Losing patterns
    losers = backtest_results.get("losing_patterns", [])
    lines.append(f"\n{'─' * 50}")
    lines.append(f"  LOSING PATTERNS (ANTI-STRATEGIES): {len(losers)}")
    lines.append(f"{'─' * 50}")
    for i, m in enumerate(losers[:20], 1):
        lines.append(
            f"  {i:2d}. {m.name:20s} | {m.symbol:12s} | "
            f"PF={m.profit_factor:5.2f} | PnL={m.total_pnl:+7.2%} | "
            f"Trades={m.total_trades:4d}"
        )

    # Cross-asset highlights
    if cross_asset_results:
        hcp = cross_asset_results.get("high_corr_pairs", [])
        lines.append(f"\n{'─' * 50}")
        lines.append(f"  HIGH CORRELATION PAIRS (|r| >= {config.CORRELATION_THRESHOLD}): {len(hcp)}")
        lines.append(f"{'─' * 50}")
        for p in hcp[:15]:
            lines.append(f"  {p['symbol_a']:12s} ↔ {p['symbol_b']:12s}  r = {p['correlation']:+.4f}")

        coint = cross_asset_results.get("cointegrated_pairs", [])
        lines.append(f"\n  COINTEGRATED PAIRS (p < 0.05): {len(coint)}")
        for p in coint[:10]:
            lines.append(f"  {p['symbol_a']:12s} ↔ {p['symbol_b']:12s}  p = {p['coint_pvalue']:.6f}")

    lines.append(f"\n{'=' * 70}")
    return "\n".join(lines)
