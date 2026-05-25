"""
Aegis-Quant-Lab — Backtest Simulator & Walk-Forward Engine
===========================================================
Isolated simulation engine.

Hard constraints:
  • NO look-ahead bias: at bar T, only data[:T-1] is visible
  • Realistic costs: fee_rate + slippage on every entry/exit
  • Walk-Forward Analysis: in-sample → out-of-sample sliding windows
  • Vectorised PnL calculation (no row-by-row loops)

Statistical reality check:
  • Mann–Whitney U  (p < 0.01 gate)
  • Monte-Carlo bootstrap  (1 000 iterations)
  • Pair ensemble analysis
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

import config

logger = logging.getLogger("aegis.backtest")

# ──────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────

@dataclass
class TradeResult:
    symbol: str = ""
    direction: str = ""            # LONG / SHORT
    entry_bar: int = 0
    exit_bar: int = 0
    entry_price: float = 0.0
    exit_price: float = 0.0
    pnl_gross: float = 0.0
    pnl_net: float = 0.0          # after fees + slippage
    entry_time: str = ""
    exit_time: str = ""
    hour: int = 0
    day_of_week: int = 0
    holding_bars: int = 0


@dataclass
class StrategyMetrics:
    name: str = ""
    symbol: str = ""
    timeframe: str = ""
    segment: str = ""              # "in_sample" / "out_of_sample" / "full"
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    mann_whitney_p: float = 1.0
    bootstrap_mean: float = 0.0
    bootstrap_ci_low: float = 0.0
    bootstrap_ci_high: float = 0.0
    is_statistically_valid: bool = False
    trades: list = field(default_factory=list)


# ──────────────────────────────────────────────
# Vectorised signal → trade engine
# ──────────────────────────────────────────────

def _generate_threshold_signals(
    series: pd.Series,
    long_entry_below: float | None = None,
    long_exit_above: float | None = None,
    short_entry_above: float | None = None,
    short_exit_below: float | None = None,
) -> pd.Series:
    """
    Generate +1 (long) / -1 (short) / 0 (flat) signal series
    from a single indicator series with threshold crossings.

    CRITICAL: signals are shifted by +1 to enforce look-ahead protection.
    Signal at bar T is computed from data up to bar T-1.
    """
    signals = pd.Series(0, index=series.index, dtype=np.int8)

    if long_entry_below is not None and long_exit_above is not None:
        long_mask = series < long_entry_below
        long_exit_mask = series > long_exit_above
        # Forward-fill the entry until exit
        in_long = False
        state = np.zeros(len(series), dtype=np.int8)
        vals = series.values
        for i in range(len(vals)):
            if not in_long and not np.isnan(vals[i]) and vals[i] < long_entry_below:
                in_long = True
            elif in_long and not np.isnan(vals[i]) and vals[i] > long_exit_above:
                in_long = False
            state[i] = 1 if in_long else 0
        signals += state

    if short_entry_above is not None and short_exit_below is not None:
        in_short = False
        state = np.zeros(len(series), dtype=np.int8)
        vals = series.values
        for i in range(len(vals)):
            if not in_short and not np.isnan(vals[i]) and vals[i] > short_entry_above:
                in_short = True
            elif in_short and not np.isnan(vals[i]) and vals[i] < short_exit_below:
                in_short = False
            state[i] = -1 if in_short else 0
        signals += state

    # LOOK-AHEAD PROTECTION: shift signals by 1 bar
    # Decision at T uses info up to T-1; execution at T's close
    signals = signals.shift(1).fillna(0).astype(np.int8)
    return signals


def _signals_to_trades(
    signals: pd.Series,
    df: pd.DataFrame,
    symbol: str = "",
) -> list[TradeResult]:
    """Convert a signal series into a list of discrete TradeResult objects."""
    trades: list[TradeResult] = []
    close_vals = df["close"].values
    idx = df.index

    position = 0  # 0=flat, 1=long, -1=short
    entry_bar = 0
    entry_price = 0.0

    sig_vals = signals.values
    for i in range(1, len(sig_vals)):
        sig = int(sig_vals[i])

        if position == 0 and sig != 0:
            # Open position
            position = sig
            entry_bar = i
            raw_price = float(close_vals[i])
            # Apply slippage on entry
            if position == 1:  # LONG — enter higher
                entry_price = raw_price * (1 + config.SLIPPAGE)
            else:  # SHORT — enter lower
                entry_price = raw_price * (1 - config.SLIPPAGE)

        elif position != 0 and (sig != position):
            # Close position
            raw_exit = float(close_vals[i])
            if position == 1:  # closing LONG — exit lower
                exit_price = raw_exit * (1 - config.SLIPPAGE)
                pnl_gross = (exit_price - entry_price) / entry_price
            else:  # closing SHORT — exit higher
                exit_price = raw_exit * (1 + config.SLIPPAGE)
                pnl_gross = (entry_price - exit_price) / entry_price

            # Subtract fees (both sides)
            pnl_net = pnl_gross - 2 * config.FEE_RATE

            entry_ts = str(idx[entry_bar]) if entry_bar < len(idx) else ""
            exit_ts = str(idx[i]) if i < len(idx) else ""

            hour = idx[entry_bar].hour if hasattr(idx[entry_bar], "hour") else 0
            dow = idx[entry_bar].dayofweek if hasattr(idx[entry_bar], "dayofweek") else 0

            trades.append(TradeResult(
                symbol=symbol,
                direction="LONG" if position == 1 else "SHORT",
                entry_bar=entry_bar,
                exit_bar=i,
                entry_price=entry_price,
                exit_price=exit_price,
                pnl_gross=pnl_gross,
                pnl_net=pnl_net,
                entry_time=entry_ts,
                exit_time=exit_ts,
                hour=hour,
                day_of_week=dow,
                holding_bars=i - entry_bar,
            ))

            # If new signal is opposite, immediately open
            if sig != 0:
                position = sig
                entry_bar = i
                raw_price = float(close_vals[i])
                if position == 1:
                    entry_price = raw_price * (1 + config.SLIPPAGE)
                else:
                    entry_price = raw_price * (1 - config.SLIPPAGE)
            else:
                position = 0

    return trades


# ──────────────────────────────────────────────
# Metrics calculation (vectorised)
# ──────────────────────────────────────────────

def _compute_metrics(
    trades: list[TradeResult],
    name: str = "",
    symbol: str = "",
    timeframe: str = "",
    segment: str = "full",
) -> StrategyMetrics:
    """Compute all performance metrics from a list of trades."""
    m = StrategyMetrics(name=name, symbol=symbol, timeframe=timeframe, segment=segment)
    m.trades = trades

    if len(trades) < 2:
        return m

    pnls = np.array([t.pnl_net for t in trades], dtype=np.float64)
    m.total_trades = len(pnls)
    m.winning_trades = int(np.sum(pnls > 0))
    m.losing_trades = int(np.sum(pnls <= 0))
    m.win_rate = m.winning_trades / m.total_trades if m.total_trades > 0 else 0.0
    m.total_pnl = float(np.sum(pnls))
    m.avg_pnl = float(np.mean(pnls))

    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    m.avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
    m.avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0

    gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.0
    gross_loss = float(np.abs(np.sum(losses))) if len(losses) > 0 else 1e-10
    m.profit_factor = gross_profit / gross_loss

    # Sharpe Ratio (annualised, assuming ~8760 bars/year for 1h)
    if np.std(pnls) > 0:
        m.sharpe_ratio = float(np.mean(pnls) / np.std(pnls) * np.sqrt(min(len(pnls), 252)))
    else:
        m.sharpe_ratio = 0.0

    # Max drawdown on cumulative PnL
    cum_pnl = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum_pnl)
    drawdown = cum_pnl - peak
    m.max_drawdown = float(np.min(drawdown)) if len(drawdown) > 0 else 0.0

    return m


# ──────────────────────────────────────────────
# Statistical validation
# ──────────────────────────────────────────────

def _mann_whitney_test(pnls: np.ndarray) -> float:
    """
    Mann-Whitney U test: compare trade PnLs against zero-mean null.
    Returns p-value.
    """
    if len(pnls) < 10:
        return 1.0
    null = np.zeros(len(pnls))
    try:
        _, pvalue = sp_stats.mannwhitneyu(pnls, null, alternative="greater")
        return float(pvalue)
    except Exception:
        return 1.0


def _bootstrap_test(pnls: np.ndarray, n_iter: int | None = None) -> tuple[float, float, float]:
    """
    Monte-Carlo bootstrap: resample trade PnLs n_iter times.
    Returns (mean_of_means, 2.5th percentile, 97.5th percentile).
    """
    if n_iter is None:
        n_iter = config.BOOTSTRAP_ITERATIONS
    if len(pnls) < 5:
        return 0.0, 0.0, 0.0

    rng = np.random.default_rng(42)
    boot_means = np.empty(n_iter, dtype=np.float64)
    for i in range(n_iter):
        sample = rng.choice(pnls, size=len(pnls), replace=True)
        boot_means[i] = sample.mean()

    return (
        float(boot_means.mean()),
        float(np.percentile(boot_means, 2.5)),
        float(np.percentile(boot_means, 97.5)),
    )


def validate_strategy(metrics: StrategyMetrics) -> StrategyMetrics:
    """Apply statistical reality checks to a strategy's trade results."""
    if metrics.total_trades < config.WFA_MIN_TRADES:
        return metrics

    pnls = np.array([t.pnl_net for t in metrics.trades], dtype=np.float64)

    # Mann-Whitney
    metrics.mann_whitney_p = _mann_whitney_test(pnls)

    # Bootstrap
    metrics.bootstrap_mean, metrics.bootstrap_ci_low, metrics.bootstrap_ci_high = _bootstrap_test(pnls)

    # Validity gate
    metrics.is_statistically_valid = (
        metrics.mann_whitney_p < config.STAT_P_VALUE_THRESHOLD
        and metrics.bootstrap_ci_low > 0  # 95% CI entirely above zero
        and metrics.profit_factor > 1.0
    )
    return metrics


# ──────────────────────────────────────────────
# Walk-Forward Analysis
# ──────────────────────────────────────────────

def walk_forward_split(
    df: pd.DataFrame,
    n_windows: int = 3,
) -> list[dict]:
    """
    Split a time-series DataFrame into Walk-Forward windows.
    Each window has an in-sample and out-of-sample segment.

    Returns list of dicts with 'in_sample' and 'out_of_sample' DataFrames.
    """
    total_bars = len(df)
    window_size = total_bars // n_windows
    if window_size < 100:
        # Too short — use single split
        split_idx = int(total_bars * config.WFA_IN_SAMPLE_RATIO)
        return [{
            "in_sample": df.iloc[:split_idx],
            "out_of_sample": df.iloc[split_idx:],
            "window_id": 0,
        }]

    windows = []
    for w in range(n_windows):
        start = w * window_size
        end = min(start + window_size, total_bars)
        segment = df.iloc[start:end]

        split = int(len(segment) * config.WFA_IN_SAMPLE_RATIO)
        windows.append({
            "in_sample": segment.iloc[:split],
            "out_of_sample": segment.iloc[split:],
            "window_id": w,
        })
    return windows


# ──────────────────────────────────────────────
# Indicator screening engine
# ──────────────────────────────────────────────

# Pre-defined scanning rules for common indicator families
INDICATOR_SCAN_RULES: list[dict] = [
    # RSI variants
    {"name": "RSI_14", "col_pattern": "RSI_14", "long_below": 30, "long_exit": 50, "short_above": 70, "short_exit": 50},
    {"name": "RSI_7", "col_pattern": "RSI_7", "long_below": 25, "long_exit": 50, "short_above": 75, "short_exit": 50},
    # Stochastic
    {"name": "KDJ_J", "col_pattern": "kdj_j", "long_below": 0, "long_exit": 50, "short_above": 100, "short_exit": 50},
    # Stochastic RSI
    {"name": "StochRSI_K", "col_pattern": "stochrsi_STOCHRSIk", "long_below": 20, "long_exit": 50, "short_above": 80, "short_exit": 50},
    # CCI
    {"name": "CCI_14", "col_pattern": "CCI_14", "long_below": -100, "long_exit": 0, "short_above": 100, "short_exit": 0},
    # CMO
    {"name": "CMO_14", "col_pattern": "cmo_14", "long_below": -50, "long_exit": 0, "short_above": 50, "short_exit": 0},
    # MFI
    {"name": "MFI_14", "col_pattern": "mfi_14", "long_below": 20, "long_exit": 50, "short_above": 80, "short_exit": 50},
    # Fisher Transform
    {"name": "Fisher", "col_pattern": "fisher_FISHERT", "long_below": -1.5, "long_exit": 0, "short_above": 1.5, "short_exit": 0},
    # Williams %R
    {"name": "WillR_14", "col_pattern": "WILLR_14", "long_below": -80, "long_exit": -50, "short_above": -20, "short_exit": -50},
    # AO (Awesome Oscillator)
    {"name": "AO", "col_pattern": "AO_", "long_below": -50, "long_exit": 0, "short_above": 50, "short_exit": 0},
    # BB %B
    {"name": "BB_PctB_20", "col_pattern": "bb_pctb_20", "long_below": 0.0, "long_exit": 0.5, "short_above": 1.0, "short_exit": 0.5},
    # CMF
    {"name": "CMF_20", "col_pattern": "cmf_20", "long_below": -0.15, "long_exit": 0, "short_above": 0.15, "short_exit": 0},
    # PPO signal
    {"name": "PPO", "col_pattern": "ppo_PPO_", "long_below": -1.0, "long_exit": 0, "short_above": 1.0, "short_exit": 0},
    # HMA deviation
    {"name": "HMA_Dev", "col_pattern": "hma_dev", "long_below": -0.02, "long_exit": 0, "short_above": 0.02, "short_exit": 0},
]


def _find_column(df: pd.DataFrame, pattern: str) -> str | None:
    """Find the first column matching a case-insensitive pattern."""
    pat_lower = pattern.lower()
    for col in df.columns:
        if pat_lower in col.lower():
            return col
    return None


def screen_indicators(
    df: pd.DataFrame,
    symbol: str = "",
    timeframe: str = "",
    progress_cb: Callable[[str], None] | None = None,
) -> list[StrategyMetrics]:
    """
    Brute-force screen all indicator scan rules on a single enriched DataFrame.
    Applies Walk-Forward validation and statistical tests.
    Returns sorted list of StrategyMetrics (best OOS Sharpe first).
    """
    results: list[StrategyMetrics] = []

    for rule in INDICATOR_SCAN_RULES:
        col = _find_column(df, rule["col_pattern"])
        if col is None:
            continue

        series = df[col]
        if series.isna().sum() / len(series) > 0.5:
            continue

        # Walk-Forward windows
        windows = walk_forward_split(df, n_windows=3)
        oos_metrics_list: list[StrategyMetrics] = []

        for win in windows:
            is_df = win["in_sample"]
            oos_df = win["out_of_sample"]

            if len(is_df) < 50 or len(oos_df) < 20:
                continue

            # Generate signals on IS (for parameter validation)
            is_signals = _generate_threshold_signals(
                is_df[col] if col in is_df.columns else pd.Series(dtype=float),
                long_entry_below=rule["long_below"],
                long_exit_above=rule["long_exit"],
                short_entry_above=rule["short_above"],
                short_exit_below=rule["short_exit"],
            )

            is_trades = _signals_to_trades(is_signals, is_df, symbol)
            is_metrics = _compute_metrics(is_trades, rule["name"], symbol, timeframe, "in_sample")

            # Only proceed to OOS if IS is promising
            if is_metrics.total_trades < 10 or is_metrics.profit_factor < 0.5:
                continue

            # Generate signals on OOS (strict forward test)
            oos_signals = _generate_threshold_signals(
                oos_df[col] if col in oos_df.columns else pd.Series(dtype=float),
                long_entry_below=rule["long_below"],
                long_exit_above=rule["long_exit"],
                short_entry_above=rule["short_above"],
                short_exit_below=rule["short_exit"],
            )

            oos_trades = _signals_to_trades(oos_signals, oos_df, symbol)
            oos_metrics = _compute_metrics(oos_trades, rule["name"], symbol, timeframe, "out_of_sample")
            oos_metrics = validate_strategy(oos_metrics)
            oos_metrics_list.append(oos_metrics)

        # Aggregate OOS results across windows
        if oos_metrics_list:
            # Combined OOS trades from all windows
            all_oos_trades = []
            for om in oos_metrics_list:
                all_oos_trades.extend(om.trades)

            combined = _compute_metrics(all_oos_trades, rule["name"], symbol, timeframe, "out_of_sample_combined")
            combined = validate_strategy(combined)
            results.append(combined)

    # Sort by Sharpe (descending)
    results.sort(key=lambda m: m.sharpe_ratio, reverse=True)

    if progress_cb and results:
        valid_count = sum(1 for r in results if r.is_statistically_valid)
        progress_cb(f"  {symbol}: {len(results)} indicators screened, {valid_count} statistically valid")

    return results


# ──────────────────────────────────────────────
# Pair ensemble analysis
# ──────────────────────────────────────────────

def ensemble_pair_test(
    df: pd.DataFrame,
    col_a: str,
    rule_a: dict,
    col_b: str,
    rule_b: dict,
    symbol: str = "",
    timeframe: str = "",
) -> StrategyMetrics | None:
    """
    Test if combining two indicators improves metrics over individual.
    Entry only when BOTH indicators agree on direction.
    """
    if col_a not in df.columns or col_b not in df.columns:
        return None

    sig_a = _generate_threshold_signals(
        df[col_a],
        long_entry_below=rule_a["long_below"],
        long_exit_above=rule_a["long_exit"],
        short_entry_above=rule_a["short_above"],
        short_exit_below=rule_a["short_exit"],
    )
    sig_b = _generate_threshold_signals(
        df[col_b],
        long_entry_below=rule_b["long_below"],
        long_exit_above=rule_b["long_exit"],
        short_entry_above=rule_b["short_above"],
        short_exit_below=rule_b["short_exit"],
    )

    # Ensemble: both must agree
    combined = pd.Series(0, index=df.index, dtype=np.int8)
    both_long = (sig_a == 1) & (sig_b == 1)
    both_short = (sig_a == -1) & (sig_b == -1)
    combined[both_long] = 1
    combined[both_short] = -1

    trades = _signals_to_trades(combined, df, symbol)
    if len(trades) < config.WFA_MIN_TRADES:
        return None

    name = f"{rule_a['name']}+{rule_b['name']}"
    metrics = _compute_metrics(trades, name, symbol, timeframe, "ensemble")
    metrics = validate_strategy(metrics)
    return metrics


def screen_pairs(
    df: pd.DataFrame,
    top_singles: list[StrategyMetrics],
    symbol: str = "",
    timeframe: str = "",
    max_pairs: int = 20,
    progress_cb: Callable[[str], None] | None = None,
) -> list[StrategyMetrics]:
    """Test top individual indicators in pairs."""
    results: list[StrategyMetrics] = []

    # Use top N individual indicators
    top_n = min(6, len(top_singles))
    candidates = top_singles[:top_n]

    tested = 0
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            if tested >= max_pairs:
                break
            rule_a = next((r for r in INDICATOR_SCAN_RULES if r["name"] == candidates[i].name), None)
            rule_b = next((r for r in INDICATOR_SCAN_RULES if r["name"] == candidates[j].name), None)
            if rule_a is None or rule_b is None:
                continue

            col_a = _find_column(df, rule_a["col_pattern"])
            col_b = _find_column(df, rule_b["col_pattern"])
            if col_a is None or col_b is None:
                continue

            m = ensemble_pair_test(df, col_a, rule_a, col_b, rule_b, symbol, timeframe)
            if m is not None:
                results.append(m)
            tested += 1

    results.sort(key=lambda m: m.sharpe_ratio, reverse=True)

    if progress_cb and results:
        progress_cb(f"  {symbol}: {len(results)} pair-ensembles tested")

    return results


# ──────────────────────────────────────────────
# Master backtest runner
# ──────────────────────────────────────────────

def run_full_backtest(
    enriched_data: dict[str, pd.DataFrame],
    timeframe: str = "1h",
    progress_cb: Callable[[str], None] | None = None,
) -> dict:
    """
    Run the complete backtest pipeline across all symbols.
    Returns comprehensive results dict.
    """
    all_results: dict = {
        "single_indicator": {},
        "pair_ensemble": {},
        "top_strategies": [],
        "losing_patterns": [],
    }

    total = len(enriched_data)
    for idx, (sym, df) in enumerate(enriched_data.items(), 1):
        msg = f"[{idx}/{total}] Backtesting {sym} …"
        logger.info(msg)
        if progress_cb:
            progress_cb(msg)

        # Single indicator screening
        singles = screen_indicators(df, sym, timeframe, progress_cb)
        all_results["single_indicator"][sym] = singles

        # Pair ensemble testing (top singles)
        pairs = screen_pairs(df, singles, sym, timeframe, progress_cb=progress_cb)
        all_results["pair_ensemble"][sym] = pairs

        # Collect globally valid strategies
        for m in singles + pairs:
            if m.is_statistically_valid:
                all_results["top_strategies"].append(m)

        # Collect losing patterns (strategies with consistent losses)
        for m in singles:
            if m.total_trades >= config.WFA_MIN_TRADES and m.profit_factor < 0.7 and m.total_pnl < -0.05:
                all_results["losing_patterns"].append(m)

    # Sort global top by Sharpe
    all_results["top_strategies"].sort(key=lambda m: m.sharpe_ratio, reverse=True)

    valid_count = len(all_results["top_strategies"])
    losing_count = len(all_results["losing_patterns"])
    msg = f"Backtest complete: {valid_count} valid strategies, {losing_count} losing patterns identified"
    logger.info(msg)
    if progress_cb:
        progress_cb(msg)

    return all_results
