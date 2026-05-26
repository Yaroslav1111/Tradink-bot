"""
Aegis-Quant-Lab v3.0 -- Trade Diagnostics Module
===================================================
Comprehensive logging and analysis for every trade decision.

Diagnostics tracked:
  - PYRAMID_DANGER  : Detects re-entry attempts into same symbol
  - TREND_EXHAUSTION: Blocks entry after 7+ consecutive candles in same direction
  - SIGNAL_DECAY    : Detects weakening momentum before entry
  - LATE_ENTRY      : Warns if price moved significantly since first signal
  - ENTRY_QUALITY   : Logs detailed reasoning for each entry/skip decision
  - FILL_ANALYSIS   : Compares expected vs actual fill price
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

import config

logger = logging.getLogger("aegis.diagnostics")


# ======================================================================
# DIAGNOSTICS DATA STRUCTURES
# ======================================================================

@dataclass
class DiagnosticEntry:
    """Single diagnostic record for a trade decision."""
    timestamp: str
    symbol: str
    action: str           # ENTRY, SKIP, BLOCK, CLOSE
    reason: str
    details: dict = field(default_factory=dict)


class TradeDiagnostics:
    """
    Full trade diagnostics and logging system.
    Tracks every decision the bot makes with detailed reasoning.
    """

    TREND_EXHAUSTION_BARS: int = 7     # block after N consecutive same-direction candles
    SIGNAL_DECAY_LOOKBACK: int = 3     # check RSI over last N bars for decay
    LATE_ENTRY_THRESHOLD: float = 0.008  # 0.8% price movement = late entry warning

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.entries: list[DiagnosticEntry] = []
        self._signal_prices: dict[str, float] = {}  # first signal price per symbol
        self._diag_logger = self._setup_diagnostics_logger()

    def _setup_diagnostics_logger(self) -> logging.Logger:
        """Create a dedicated diagnostics file logger."""
        diag_log = logging.getLogger("aegis.diag_file")
        diag_log.setLevel(logging.DEBUG)
        diag_log.handlers.clear()

        log_file = os.path.join(
            self.log_dir,
            f"diagnostics_{datetime.now().strftime('%Y%m%d')}.log"
        )
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        diag_log.addHandler(fh)
        return diag_log

    # ------------------------------------------------------------------
    # TREND EXHAUSTION FILTER
    # ------------------------------------------------------------------

    def check_trend_exhaustion(
        self,
        df: pd.DataFrame,
        direction: str,
    ) -> tuple[bool, int]:
        """
        Check if the trend is exhausted (7+ consecutive candles in same direction).

        Parameters
        ----------
        df : OHLCV DataFrame
        direction : "LONG" or "SHORT"

        Returns
        -------
        (is_exhausted: bool, consecutive_count: int)
        """
        if len(df) < self.TREND_EXHAUSTION_BARS + 1:
            return False, 0

        close = df["close"].astype(float).values

        # Count consecutive candles in same direction from the end
        count = 0
        for i in range(len(close) - 1, 0, -1):
            if direction == "LONG":
                # For LONG entry: check if price has been going UP already
                if close[i] > close[i - 1]:
                    count += 1
                else:
                    break
            else:
                # For SHORT entry: check if price has been going DOWN already
                if close[i] < close[i - 1]:
                    count += 1
                else:
                    break

        is_exhausted = count >= self.TREND_EXHAUSTION_BARS
        return is_exhausted, count

    # ------------------------------------------------------------------
    # SIGNAL DECAY DETECTION
    # ------------------------------------------------------------------

    def check_signal_decay(
        self,
        df: pd.DataFrame,
        direction: str,
    ) -> tuple[bool, str]:
        """
        Detect weakening momentum before entry.

        For LONG: RSI should be turning UP (rsi[-1] > rsi[-2]), not just low.
        For SHORT: RSI should be turning DOWN (rsi[-1] < rsi[-2]), not just high.

        Returns
        -------
        (is_decaying: bool, reason: str)
        """
        if len(df) < 20:
            return False, "Insufficient data"

        try:
            import pandas_ta as ta
            close = df["close"].astype(np.float64)
            rsi = ta.rsi(close, length=14)

            if rsi is None or len(rsi) < 3:
                return False, "RSI unavailable"

            rsi_values = rsi.dropna().values
            if len(rsi_values) < 3:
                return False, "RSI too short"

            rsi_now = rsi_values[-1]
            rsi_prev = rsi_values[-2]
            rsi_prev2 = rsi_values[-3]

            if direction == "LONG":
                # For LONG: we want RSI to be TURNING UP from oversold
                if rsi_now < rsi_prev:
                    return True, (
                        f"SIGNAL_DECAY: RSI falling ({rsi_now:.1f} < {rsi_prev:.1f}) "
                        f"- momentum weakening for LONG"
                    )
                # Also check: if RSI was oversold but now rising fast = good entry
                # If RSI > 50 already = we missed the boat
                if rsi_now > 50 and rsi_prev > 45:
                    return True, (
                        f"SIGNAL_DECAY: RSI already above 50 ({rsi_now:.1f}) "
                        f"- LONG entry is late"
                    )
            else:
                # For SHORT: we want RSI to be TURNING DOWN from overbought
                if rsi_now > rsi_prev:
                    return True, (
                        f"SIGNAL_DECAY: RSI rising ({rsi_now:.1f} > {rsi_prev:.1f}) "
                        f"- momentum weakening for SHORT"
                    )
                if rsi_now < 50 and rsi_prev < 55:
                    return True, (
                        f"SIGNAL_DECAY: RSI already below 50 ({rsi_now:.1f}) "
                        f"- SHORT entry is late"
                    )

            return False, f"RSI momentum OK: {rsi_prev2:.1f} -> {rsi_prev:.1f} -> {rsi_now:.1f}"

        except Exception as e:
            return False, f"Decay check error: {e}"

    # ------------------------------------------------------------------
    # MOMENTUM CONFIRMATION (RSI must be TURNING)
    # ------------------------------------------------------------------

    def check_momentum_confirmation(
        self,
        df: pd.DataFrame,
        direction: str,
    ) -> tuple[bool, str]:
        """
        Confirm that indicator is TURNING in our direction, not just at extreme.

        For LONG:  RSI[-1] > RSI[-2] (turning up from oversold)
        For SHORT: RSI[-1] < RSI[-2] (turning down from overbought)

        Returns
        -------
        (is_confirmed: bool, reason: str)
        """
        if len(df) < 20:
            return True, "Insufficient data - allowing"

        try:
            import pandas_ta as ta
            close = df["close"].astype(np.float64)
            rsi = ta.rsi(close, length=14)

            if rsi is None or len(rsi) < 3:
                return True, "RSI unavailable - allowing"

            rsi_values = rsi.dropna().values
            if len(rsi_values) < 3:
                return True, "RSI too short - allowing"

            rsi_now = rsi_values[-1]
            rsi_prev = rsi_values[-2]

            if direction == "LONG":
                # RSI must be turning UP (momentum confirming the reversal)
                if rsi_now > rsi_prev:
                    return True, (
                        f"MOMENTUM OK: RSI turning up "
                        f"({rsi_prev:.1f} -> {rsi_now:.1f})"
                    )
                else:
                    return False, (
                        f"MOMENTUM FAIL: RSI still falling "
                        f"({rsi_prev:.1f} -> {rsi_now:.1f}) - wait for turn"
                    )
            else:
                # RSI must be turning DOWN
                if rsi_now < rsi_prev:
                    return True, (
                        f"MOMENTUM OK: RSI turning down "
                        f"({rsi_prev:.1f} -> {rsi_now:.1f})"
                    )
                else:
                    return False, (
                        f"MOMENTUM FAIL: RSI still rising "
                        f"({rsi_prev:.1f} -> {rsi_now:.1f}) - wait for turn"
                    )

        except Exception:
            return True, "Momentum check error - allowing"

    # ------------------------------------------------------------------
    # LATE ENTRY DETECTION
    # ------------------------------------------------------------------

    def check_late_entry(
        self,
        symbol: str,
        current_price: float,
    ) -> tuple[bool, float]:
        """
        Detect if we're entering too late (price already moved since first signal).

        Returns
        -------
        (is_late: bool, price_move_pct: float)
        """
        first_price = self._signal_prices.get(symbol)
        if first_price is None:
            # First signal - record price
            self._signal_prices[symbol] = current_price
            return False, 0.0

        move_pct = abs(current_price - first_price) / first_price
        is_late = move_pct > self.LATE_ENTRY_THRESHOLD

        if is_late:
            # Reset for next signal
            self._signal_prices.pop(symbol, None)

        return is_late, move_pct

    def clear_signal_price(self, symbol: str):
        """Clear tracked signal price after entry or skip."""
        self._signal_prices.pop(symbol, None)

    # ------------------------------------------------------------------
    # FILL ANALYSIS
    # ------------------------------------------------------------------

    def log_fill_analysis(
        self,
        symbol: str,
        expected_price: float,
        actual_fill_price: float,
        direction: str,
    ):
        """Log the difference between expected and actual fill price."""
        slippage_pct = (actual_fill_price - expected_price) / expected_price * 100
        is_favorable = (
            (direction == "LONG" and actual_fill_price <= expected_price) or
            (direction == "SHORT" and actual_fill_price >= expected_price)
        )

        entry = DiagnosticEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            symbol=symbol,
            action="FILL_ANALYSIS",
            reason=f"{'GOOD' if is_favorable else 'BAD'} fill",
            details={
                "expected_price": expected_price,
                "actual_fill_price": actual_fill_price,
                "slippage_pct": round(slippage_pct, 4),
                "direction": direction,
                "is_favorable": is_favorable,
            }
        )
        self.entries.append(entry)
        self._diag_logger.info(
            f"FILL | {symbol} {direction} | "
            f"Expected={expected_price:.6f} Actual={actual_fill_price:.6f} | "
            f"Slippage={slippage_pct:+.4f}% | "
            f"{'FAVORABLE' if is_favorable else 'UNFAVORABLE'}"
        )

    # ------------------------------------------------------------------
    # COMPREHENSIVE ENTRY DIAGNOSTICS
    # ------------------------------------------------------------------

    def log_entry_decision(
        self,
        symbol: str,
        action: str,
        reason: str,
        score: float = 0.0,
        direction: str = "",
        price: float = 0.0,
        extra: dict | None = None,
    ):
        """Log a trade entry decision with full details."""
        entry = DiagnosticEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            symbol=symbol,
            action=action,
            reason=reason,
            details={
                "score": round(score, 4),
                "direction": direction,
                "price": price,
                **(extra or {}),
            }
        )
        self.entries.append(entry)

        emoji = {
            "ENTRY": ">>>",
            "SKIP": "---",
            "BLOCK": "XXX",
            "CLOSE": "<<<",
        }.get(action, "???")

        self._diag_logger.info(
            f"{emoji} | {symbol:12s} | {action:6s} | "
            f"Score={score:.3f} {direction:5s} @ {price:.6f} | {reason}"
        )

    # ------------------------------------------------------------------
    # RUN ALL PRE-ENTRY CHECKS
    # ------------------------------------------------------------------

    def run_pre_entry_checks(
        self,
        symbol: str,
        direction: str,
        df: pd.DataFrame,
        current_price: float,
    ) -> tuple[bool, str]:
        """
        Run all diagnostic pre-entry checks.
        Returns (allowed, reason).
        """
        # 1. Trend exhaustion
        exhausted, consec = self.check_trend_exhaustion(df, direction)
        if exhausted:
            reason = (
                f"TREND_EXHAUSTION: {consec} consecutive candles in {direction} direction "
                f"(max {self.TREND_EXHAUSTION_BARS})"
            )
            self.log_entry_decision(
                symbol, "BLOCK", reason,
                direction=direction, price=current_price,
                extra={"consecutive_candles": consec}
            )
            return False, reason

        # 2. Signal decay
        decaying, decay_reason = self.check_signal_decay(df, direction)
        if decaying:
            self.log_entry_decision(
                symbol, "BLOCK", decay_reason,
                direction=direction, price=current_price,
            )
            return False, decay_reason

        # 3. Momentum confirmation (RSI must be turning)
        confirmed, momentum_reason = self.check_momentum_confirmation(df, direction)
        if not confirmed:
            self.log_entry_decision(
                symbol, "BLOCK", momentum_reason,
                direction=direction, price=current_price,
            )
            return False, momentum_reason

        # 4. Late entry
        is_late, move_pct = self.check_late_entry(symbol, current_price)
        if is_late:
            reason = (
                f"LATE_ENTRY: Price moved {move_pct*100:.2f}% since first signal "
                f"(threshold {self.LATE_ENTRY_THRESHOLD*100:.1f}%)"
            )
            self.log_entry_decision(
                symbol, "BLOCK", reason,
                direction=direction, price=current_price,
                extra={"price_move_pct": round(move_pct * 100, 2)}
            )
            return False, reason

        return True, "All pre-entry checks passed"

    # ------------------------------------------------------------------
    # EXPORT
    # ------------------------------------------------------------------

    def export_diagnostics(self, path: str | None = None) -> str:
        """Export all diagnostics to JSON."""
        if path is None:
            path = os.path.join(
                self.log_dir,
                f"diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

        data = [
            {
                "timestamp": e.timestamp,
                "symbol": e.symbol,
                "action": e.action,
                "reason": e.reason,
                "details": e.details,
            }
            for e in self.entries
        ]

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Diagnostics exported: {path} ({len(data)} entries)")
        return path

    def get_summary(self) -> dict:
        """Get diagnostics summary statistics."""
        actions = {}
        for e in self.entries:
            actions[e.action] = actions.get(e.action, 0) + 1

        return {
            "total_decisions": len(self.entries),
            "by_action": actions,
            "tracked_signals": len(self._signal_prices),
        }
