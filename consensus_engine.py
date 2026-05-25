"""
Aegis-Quant-Lab — Consensus Voting Engine
===========================================
The "brain" of the live trading system.

Logic:
1. Load indicator_weights.json (pre-computed by generate_weight_matrix.py)
2. For each bar, ask the top 15 indicators: "LONG, SHORT, or neutral?"
3. Each indicator votes with its accuracy_weight (historical expected return)
4. Calculate:
   - Total LONG weight = sum of weights from indicators voting LONG
   - Total SHORT weight = sum of weights from indicators voting SHORT
   - Net signal strength = |LONG_weight - SHORT_weight| / total_weight
   - Dissonance index = min(LONG_weight, SHORT_weight) / max(LONG_weight, SHORT_weight)
5. Decision:
   - If dissonance > threshold (e.g. 0.55) → CHAOS → no trade (sit in cash)
   - If net signal > agreement threshold AND enough indicators agree → ENTER
   - Otherwise → no trade

This ensures the bot only trades when there's a STRONG consensus.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import pandas_ta as ta

import config
from generate_weight_matrix import _compute_indicator_signals

logger = logging.getLogger("aegis.consensus")


# ──────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────

@dataclass
class ConsensusSignal:
    """Result of the consensus voting for a single bar."""
    timestamp: str = ""
    symbol: str = ""
    direction: str = "NEUTRAL"       # LONG / SHORT / NEUTRAL
    strength: float = 0.0            # 0.0 to 1.0 (how strong the consensus is)
    dissonance: float = 1.0          # 0.0 (full agreement) to 1.0 (total chaos)
    long_weight: float = 0.0         # total weight voting LONG
    short_weight: float = 0.0        # total weight voting SHORT
    total_voters: int = 0            # how many indicators voted
    agreeing_voters: int = 0         # how many voted for the winning direction
    is_tradeable: bool = False       # final decision: trade or not
    blocked_reason: str = ""         # why it was blocked (if not tradeable)

    @property
    def confidence_pct(self) -> float:
        """Confidence as percentage."""
        return round(self.strength * 100, 1)


# ──────────────────────────────────────────────
# Core voting engine
# ──────────────────────────────────────────────

class ConsensusEngine:
    """
    Weighted consensus voting engine.
    
    Usage:
        engine = ConsensusEngine(symbol_weights)
        signal = engine.evaluate(df, bar_index=T)
    """

    def __init__(
        self,
        symbol_weights: dict[str, dict],
        min_agreement: float | None = None,
        dissonance_threshold: float | None = None,
        top_n: int | None = None,
    ):
        """
        Parameters
        ----------
        symbol_weights : dict from indicator_weights.json for ONE symbol
        min_agreement : fraction of top indicators that must agree (default 0.70)
        dissonance_threshold : max allowed dissonance before blocking (default 0.55)
        top_n : use only top N indicators by weight (default 15)
        """
        self.weights = symbol_weights
        self.min_agreement = min_agreement or config.CONSENSUS_MIN_AGREEMENT
        self.dissonance_threshold = dissonance_threshold or config.CONSENSUS_DISSONANCE_THRESHOLD
        self.top_n = top_n or config.CONSENSUS_TOP_N_INDICATORS

        # Pre-select top indicators (sorted by accuracy_weight)
        self._top_indicators = self._select_top()

    def _select_top(self) -> list[dict]:
        """Select top N indicators by accuracy_weight."""
        candidates = []
        for name, info in self.weights.items():
            if info.get("accuracy_weight", 0) > 0:
                candidates.append({"name": name, **info})
        candidates.sort(key=lambda x: x["accuracy_weight"], reverse=True)
        return candidates[:self.top_n]

    @property
    def active_indicators(self) -> list[str]:
        """Names of active (voting) indicators."""
        return [ind["name"] for ind in self._top_indicators]

    def evaluate_bar(
        self,
        df: pd.DataFrame,
        bar_idx: int,
        symbol: str = "",
    ) -> ConsensusSignal:
        """
        Evaluate consensus at a specific bar index.
        
        IMPORTANT: Uses only data up to bar_idx (no look-ahead).
        """
        signal = ConsensusSignal(symbol=symbol)

        if bar_idx < 50 or bar_idx >= len(df):
            signal.blocked_reason = "Insufficient history"
            return signal

        # Get slice up to current bar (LOOK-AHEAD PROTECTION)
        df_slice = df.iloc[:bar_idx + 1]
        signal.timestamp = str(df_slice.index[-1]) if hasattr(df_slice.index, '__getitem__') else ""

        # Compute all indicator conditions on the historical slice
        all_signals = _compute_indicator_signals(df_slice)

        # Vote
        long_weight = 0.0
        short_weight = 0.0
        long_voters = 0
        short_voters = 0
        total_voters = 0

        for ind_info in self._top_indicators:
            ind_name = ind_info["name"]
            weight = ind_info["accuracy_weight"]
            expected_dir = ind_info["direction"]

            if ind_name not in all_signals:
                continue

            condition, direction = all_signals[ind_name]
            if condition is None or len(condition) == 0:
                continue

            # Check if the indicator is currently firing (last bar)
            last_val = condition.iloc[-1] if len(condition) > 0 else False
            if pd.isna(last_val):
                last_val = False

            if bool(last_val):
                total_voters += 1
                if direction == "LONG":
                    long_weight += weight
                    long_voters += 1
                elif direction == "SHORT":
                    short_weight += weight
                    short_voters += 1

        signal.long_weight = long_weight
        signal.short_weight = short_weight
        signal.total_voters = long_voters + short_voters

        # Calculate dissonance and strength
        total_weight = long_weight + short_weight
        if total_weight <= 0:
            signal.direction = "NEUTRAL"
            signal.dissonance = 1.0
            signal.strength = 0.0
            signal.blocked_reason = "No indicators firing"
            return signal

        # Dissonance: how evenly split are the votes?
        max_w = max(long_weight, short_weight)
        min_w = min(long_weight, short_weight)
        signal.dissonance = min_w / max_w if max_w > 0 else 1.0

        # Net strength: normalized difference
        signal.strength = (max_w - min_w) / total_weight

        # Determine direction
        if long_weight > short_weight:
            signal.direction = "LONG"
            signal.agreeing_voters = long_voters
        elif short_weight > long_weight:
            signal.direction = "SHORT"
            signal.agreeing_voters = short_voters
        else:
            signal.direction = "NEUTRAL"
            signal.agreeing_voters = 0

        # ── Decision gates ──
        # Gate 1: Dissonance filter (chaos detection)
        if signal.dissonance > self.dissonance_threshold:
            signal.is_tradeable = False
            signal.blocked_reason = (
                f"High dissonance ({signal.dissonance:.2f} > {self.dissonance_threshold}) — "
                f"market is unpredictable, sitting in cash"
            )
            return signal

        # Gate 2: Minimum agreement
        agreement_ratio = signal.agreeing_voters / max(len(self._top_indicators), 1)
        if agreement_ratio < self.min_agreement:
            signal.is_tradeable = False
            signal.blocked_reason = (
                f"Insufficient agreement ({signal.agreeing_voters}/{len(self._top_indicators)} = "
                f"{agreement_ratio:.1%} < {self.min_agreement:.0%} required)"
            )
            return signal

        # Gate 3: Minimum absolute strength
        if signal.strength < 0.3:
            signal.is_tradeable = False
            signal.blocked_reason = f"Weak signal strength ({signal.strength:.2f} < 0.30)"
            return signal

        # All gates passed — TRADEABLE
        signal.is_tradeable = True
        return signal


# ──────────────────────────────────────────────
# Vectorized backtest of consensus system
# ──────────────────────────────────────────────

def backtest_consensus(
    df: pd.DataFrame,
    symbol_weights: dict[str, dict],
    symbol: str = "",
    forward_bars: int | None = None,
    progress_cb: Callable[[str], None] | None = None,
) -> dict:
    """
    Backtest the consensus voting system on historical data.
    
    Instead of evaluating every single bar (too slow for 2000+ bars),
    we use a vectorized approach:
    1. Compute all indicator signals for the full history
    2. For each bar, sum weights of firing indicators
    3. Apply consensus rules vectorized
    4. Calculate PnL
    """
    if forward_bars is None:
        forward_bars = config.FORWARD_BARS

    if len(df) < 200:
        return {"error": "Insufficient data"}

    # 1. Compute all indicator signals
    all_signals = _compute_indicator_signals(df)

    # 2. Select top indicators by weight
    top_inds = []
    for name, info in symbol_weights.items():
        if info.get("accuracy_weight", 0) > 0:
            top_inds.append({"name": name, **info})
    top_inds.sort(key=lambda x: x["accuracy_weight"], reverse=True)
    top_inds = top_inds[:config.CONSENSUS_TOP_N_INDICATORS]

    if not top_inds:
        return {"error": "No profitable indicators found"}

    # 3. Build weight matrices (vectorized)
    n_bars = len(df)
    long_weights = np.zeros(n_bars, dtype=np.float64)
    short_weights = np.zeros(n_bars, dtype=np.float64)
    long_count = np.zeros(n_bars, dtype=np.int32)
    short_count = np.zeros(n_bars, dtype=np.int32)

    for ind_info in top_inds:
        ind_name = ind_info["name"]
        weight = ind_info["accuracy_weight"]

        if ind_name not in all_signals:
            continue

        condition, direction = all_signals[ind_name]
        if condition is None:
            continue

        mask = condition.fillna(False).values.astype(bool)

        if direction == "LONG":
            long_weights[mask] += weight
            long_count[mask] += 1
        elif direction == "SHORT":
            short_weights[mask] += weight
            short_count[mask] += 1

    # 4. Calculate consensus metrics (vectorized)
    total_weight = long_weights + short_weights
    max_weight = np.maximum(long_weights, short_weights)
    min_weight = np.minimum(long_weights, short_weights)

    # Dissonance = min/max (1.0 = total chaos, 0.0 = full agreement)
    with np.errstate(divide="ignore", invalid="ignore"):
        dissonance = np.where(max_weight > 0, min_weight / max_weight, 1.0)
        strength = np.where(total_weight > 0, (max_weight - min_weight) / total_weight, 0.0)

    # Direction
    direction_arr = np.where(long_weights > short_weights, 1,
                             np.where(short_weights > long_weights, -1, 0))

    # Agreement count
    agree_count = np.where(direction_arr == 1, long_count,
                           np.where(direction_arr == -1, short_count, 0))
    n_top = len(top_inds)
    agreement_ratio = agree_count / max(n_top, 1)

    # 5. Apply trading filters
    tradeable = (
        (dissonance < config.CONSENSUS_DISSONANCE_THRESHOLD) &
        (agreement_ratio >= config.CONSENSUS_MIN_AGREEMENT) &
        (strength >= 0.3) &
        (direction_arr != 0)
    )

    # LOOK-AHEAD PROTECTION: shift signals by 1 bar
    tradeable = np.roll(tradeable, 1)
    tradeable[0] = False
    direction_arr = np.roll(direction_arr, 1)
    direction_arr[0] = 0

    # 6. Calculate PnL
    close = df["close"].astype(np.float64).values
    fwd_return = np.zeros(n_bars)
    fwd_return[:-forward_bars] = (close[forward_bars:] - close[:-forward_bars]) / close[:-forward_bars]

    # Apply direction
    trade_returns = fwd_return * direction_arr * tradeable

    # Subtract Bybit fees on trades
    trade_returns[tradeable] -= config.ROUNDTRIP_FEE

    # 7. Collect results
    trade_mask = tradeable & (trade_returns != 0)
    trade_pnls = trade_returns[trade_mask]

    results = {
        "symbol": symbol,
        "total_bars": n_bars,
        "total_signals": int(tradeable.sum()),
        "total_trades": len(trade_pnls),
        "top_indicators_used": len(top_inds),
    }

    if len(trade_pnls) > 0:
        wins = trade_pnls[trade_pnls > 0]
        losses = trade_pnls[trade_pnls <= 0]

        results.update({
            "win_rate_pct": round(len(wins) / len(trade_pnls) * 100, 2),
            "total_pnl_pct": round(float(trade_pnls.sum()) * 100, 4),
            "avg_pnl_per_trade_pct": round(float(trade_pnls.mean()) * 100, 4),
            "avg_win_pct": round(float(wins.mean()) * 100, 4) if len(wins) > 0 else 0.0,
            "avg_loss_pct": round(float(losses.mean()) * 100, 4) if len(losses) > 0 else 0.0,
            "profit_factor": round(float(wins.sum()) / max(float(abs(losses.sum())), 1e-10), 3),
            "sharpe_ratio": round(float(trade_pnls.mean() / (trade_pnls.std() + 1e-10) * np.sqrt(252)), 2),
            "max_drawdown_pct": round(float(np.min(np.cumsum(trade_pnls) - np.maximum.accumulate(np.cumsum(trade_pnls)))) * 100, 4),
            "dissonance_blocked_pct": round(float((dissonance >= config.CONSENSUS_DISSONANCE_THRESHOLD).sum()) / n_bars * 100, 1),
        })
    else:
        results.update({
            "win_rate_pct": 0.0,
            "total_pnl_pct": 0.0,
            "avg_pnl_per_trade_pct": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
        })

    # 8. Estimate monthly P&L on $2000 capital
    if results.get("total_trades", 0) > 0 and results.get("avg_pnl_per_trade_pct", 0) > 0:
        # Assume ~100 consensus signals per month across all coins
        avg_return_per_trade = results["avg_pnl_per_trade_pct"] / 100
        risk_per_trade = config.DEFAULT_CAPITAL * config.RISK_PER_TRADE_PCT
        monthly_trades_estimate = min(results["total_trades"] * 30 / max(n_bars / 96, 1), 100)
        expected_monthly_pnl = avg_return_per_trade * risk_per_trade * config.RISK_REWARD_RATIO * monthly_trades_estimate
        results["estimated_monthly_pnl_usdt"] = round(expected_monthly_pnl, 2)
        results["estimated_monthly_roi_pct"] = round(expected_monthly_pnl / config.DEFAULT_CAPITAL * 100, 2)

    if progress_cb:
        msg = (f"  Consensus BT {symbol}: {results.get('total_trades', 0)} trades, "
               f"WR={results.get('win_rate_pct', 0):.1f}%, "
               f"PnL={results.get('total_pnl_pct', 0):.2f}%, "
               f"PF={results.get('profit_factor', 0):.2f}")
        progress_cb(msg)

    return results


def backtest_consensus_batch(
    data: dict[str, pd.DataFrame],
    library: dict[str, dict],
    progress_cb: Callable[[str], None] | None = None,
) -> dict[str, dict]:
    """Run consensus backtest across all symbols."""
    results: dict[str, dict] = {}
    total = len(data)

    for idx, (sym, df) in enumerate(data.items(), 1):
        sym_clean = sym.replace("/", "")
        msg = f"[{idx}/{total}] Consensus backtest: {sym}"
        logger.info(msg)
        if progress_cb:
            progress_cb(msg)

        if sym_clean not in library:
            continue

        result = backtest_consensus(df, library[sym_clean], sym, progress_cb=progress_cb)
        results[sym] = result

    return results
