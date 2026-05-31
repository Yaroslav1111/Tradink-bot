"""
v5.0 — Pure Strategy Logic (The "Brain")
══════════════════════════════════════════════
ZERO I/O. ZERO imports of pybit, requests, or time.
This file runs identically in Live mode and Backtest mode.

It receives data via the Broker protocol and emits Signals.
The Broker handles execution — strategy doesn't care how.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from engine.models import (
    Direction, Signal, Order, Position, PositionPhase,
    AccountState, TradeResult, OrderStatus, Candle,
)

logger = logging.getLogger("aegis.strategy")


# ══════════════════════════════════════════════════════════════════
# CONFIGURATION (injected at init — allows optimizer to override)
# ══════════════════════════════════════════════════════════════════

@dataclass
class StrategyConfig:
    """All tunable parameters. Optimizer generates variants of this."""
    # Oscillator thresholds
    rsi_oversold: float = 15.0
    rsi_overbought: float = 85.0
    cci_oversold: float = -250.0
    cci_overbought: float = 250.0
    willr_oversold: float = -95.0
    willr_overbought: float = -5.0
    min_oscillators: int = 2

    # Fibonacci
    fibo_primary: float = 0.618
    fibo_secondary: float = 0.50
    fibo_ext_1: float = 1.618
    fibo_ext_2: float = 2.618
    swing_lookback: int = 50

    # Orders
    order_ttl_seconds: int = 600
    cascade_enabled: bool = True
    cascade_risk_split: float = 0.5

    # Position management (3-phase)
    sl_atr_multiplier: float = 2.0
    breakeven_trigger_pct: float = 0.015
    breakeven_fee_buffer: float = 0.0004
    trailing_trigger_pct: float = 0.030
    trailing_distance_pct: float = 0.015
    trailing_atr_cushion: float = 0.2

    # TP repositioning
    maker_tp_reposition_trigger: float = 0.8

    # Volume Profile
    poc_enabled: bool = True
    poc_1m_lookback: int = 60
    poc_cluster_width_atr: float = 0.3
    poc_weight: float = 0.6

    # Risk
    risk_pct: float = 0.01
    risk_reduced: float = 0.005
    loss_streak_threshold: int = 3
    max_risk_pct: float = 0.02
    leverage: float = 5.0
    max_concurrent: int = 5
    max_entries_per_cycle: int = 3

    # Fees
    maker_fee: float = 0.0002
    taker_fee: float = 0.00055


# ══════════════════════════════════════════════════════════════════
# INDICATORS (pure math, no I/O)
# ══════════════════════════════════════════════════════════════════

class Indicators:
    """Stateless indicator calculations."""

    @staticmethod
    def rsi(close: pd.Series, length: int = 14) -> Optional[pd.Series]:
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=length).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=length).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def cci(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> Optional[pd.Series]:
        tp = (high + low + close) / 3
        ma = tp.rolling(window=length).mean()
        md = tp.rolling(window=length).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        return (tp - ma) / (0.015 * md.replace(0, np.nan))

    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> Optional[pd.Series]:
        hh = high.rolling(window=length).max()
        ll = low.rolling(window=length).min()
        return -100 * (hh - close) / (hh - ll).replace(0, np.nan)

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> Optional[pd.Series]:
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=length).mean()


# ══════════════════════════════════════════════════════════════════
# VOLUME PROFILER (pure math)
# ══════════════════════════════════════════════════════════════════

class VolumeProfiler:
    """POC calculation from 1m data — pure numpy, no I/O."""

    @staticmethod
    def calculate_poc(
        df_1m: pd.DataFrame, atr_15m: float,
        direction: Direction, current_price: float,
        cluster_width_atr: float = 0.3,
    ) -> Optional[float]:
        if df_1m is None or len(df_1m) < 20:
            return None

        close = df_1m["close"].astype(float).values
        high = df_1m["high"].astype(float).values
        low = df_1m["low"].astype(float).values
        volume = df_1m["volume"].astype(float).values

        typical = (high + low + close) / 3.0
        bin_width = atr_15m * cluster_width_atr
        if bin_width <= 0:
            bin_width = current_price * 0.002

        price_min = float(np.min(low))
        price_max = float(np.max(high))
        n_bins = max(int((price_max - price_min) / bin_width) + 1, 5)
        bins = np.linspace(price_min, price_max, n_bins + 1)

        vol_per_bin = np.zeros(n_bins)
        for i in range(len(typical)):
            idx = int((typical[i] - price_min) / bin_width)
            idx = max(0, min(idx, n_bins - 1))
            vol_per_bin[idx] += volume[i]

        poc_idx = int(np.argmax(vol_per_bin))
        poc = (bins[poc_idx] + bins[poc_idx + 1]) / 2.0

        # Direction filter
        if direction == Direction.LONG and poc >= current_price:
            return None
        if direction == Direction.SHORT and poc <= current_price:
            return None

        return poc

    @staticmethod
    def blend(poc: Optional[float], fibo: float, weight: float = 0.6) -> float:
        if poc is None or poc <= 0:
            return fibo
        return poc * weight + fibo * (1 - weight)


# ══════════════════════════════════════════════════════════════════
# FIBONACCI CALCULATOR (pure math)
# ══════════════════════════════════════════════════════════════════

class FiboCalc:
    """Fibonacci retracement and extension — pure math."""

    @staticmethod
    def cascade_entries(
        swing_high: float, swing_low: float, direction: Direction,
        current_price: float, fibo_50: float = 0.50, fibo_618: float = 0.618,
    ) -> tuple[float, float]:
        diff = swing_high - swing_low
        if diff <= 0:
            mid = (swing_high + swing_low) / 2
            return mid, mid

        if direction == Direction.LONG:
            e50 = swing_high - diff * fibo_50
            e618 = swing_high - diff * fibo_618
            max_buy = current_price * 0.999
            e50 = min(e50, max_buy)
            e618 = min(e618, max_buy)
            if e618 >= e50:
                e618 = e50 * 0.998
        else:
            e50 = swing_low + diff * fibo_50
            e618 = swing_low + diff * fibo_618
            min_sell = current_price * 1.001
            e50 = max(e50, min_sell)
            e618 = max(e618, min_sell)
            if e618 <= e50:
                e618 = e50 * 1.002

        return e50, e618

    @staticmethod
    def extensions(
        swing_high: float, swing_low: float, direction: Direction,
        ext_1: float = 1.618, ext_2: float = 2.618,
    ) -> tuple[float, float]:
        diff = swing_high - swing_low
        if diff <= 0:
            return swing_high, swing_high
        if direction == Direction.LONG:
            return swing_high + diff * (ext_1 - 1), swing_high + diff * (ext_2 - 1)
        else:
            return swing_low - diff * (ext_1 - 1), swing_low - diff * (ext_2 - 1)


# ══════════════════════════════════════════════════════════════════
# COMPOUND CALCULATOR (pure math)
# ══════════════════════════════════════════════════════════════════

class CompoundCalc:
    """Risk sizing with compound growth — pure math."""

    def __init__(self, balance: float, cfg: StrategyConfig):
        self.balance = balance
        self.cfg = cfg
        self.consecutive_losses = 0
        self.total_trades = 0
        self.wins = 0
        self.losses = 0
        self.peak_balance = balance
        self.max_drawdown_pct = 0.0

    @property
    def risk_pct(self) -> float:
        if self.consecutive_losses >= self.cfg.loss_streak_threshold:
            return self.cfg.risk_reduced
        return self.cfg.risk_pct

    @property
    def risk_usdt(self) -> float:
        return min(self.balance * self.risk_pct, self.balance * self.cfg.max_risk_pct)

    def record(self, pnl: float):
        self.total_trades += 1
        self.balance += pnl
        if pnl > 0:
            self.wins += 1
            self.consecutive_losses = 0
        else:
            self.losses += 1
            self.consecutive_losses += 1
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        dd = (self.peak_balance - self.balance) / self.peak_balance if self.peak_balance > 0 else 0
        self.max_drawdown_pct = max(self.max_drawdown_pct, dd)


# ══════════════════════════════════════════════════════════════════
# THE STRATEGY (The "Brain")
# ══════════════════════════════════════════════════════════════════

class FiboReversalStrategy:
    """
    v5.0 Fibonacci Reversal Sniper — PURE LOGIC.

    This class has NO knowledge of:
      - Network/API calls
      - File I/O
      - System time (uses broker.now())
      - Whether it's trading real money or backtesting

    It receives a Broker interface and calls broker methods.
    """

    def __init__(self, cfg: StrategyConfig, initial_balance: float = 2000.0):
        self.cfg = cfg
        self.compound = CompoundCalc(initial_balance, cfg)
        self.trade_history: list[TradeResult] = []

    # ──────────────────────────────────────────
    # SIGNAL DETECTION (15m scan)
    # ──────────────────────────────────────────

    def detect_signal(self, symbol: str, df: pd.DataFrame, current_price: float) -> Optional[Direction]:
        """
        Detect reversal signal from 15m OHLCV data.
        Returns Direction if signal, None otherwise.
        """
        if df is None or len(df) < 100:
            return None

        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)

        rsi = Indicators.rsi(close, 14)
        cci = Indicators.cci(high, low, close, 14)
        willr = Indicators.williams_r(high, low, close, 14)

        if rsi is None or cci is None or willr is None:
            return None

        rsi_val = float(rsi.iloc[-1])
        cci_val = float(cci.iloc[-1])
        willr_val = float(willr.iloc[-1])

        if np.isnan(rsi_val) or np.isnan(cci_val) or np.isnan(willr_val):
            return None

        long_count = short_count = 0
        if rsi_val < self.cfg.rsi_oversold: long_count += 1
        elif rsi_val > self.cfg.rsi_overbought: short_count += 1
        if cci_val < self.cfg.cci_oversold: long_count += 1
        elif cci_val > self.cfg.cci_overbought: short_count += 1
        if willr_val < self.cfg.willr_oversold: long_count += 1
        elif willr_val > self.cfg.willr_overbought: short_count += 1

        if long_count >= self.cfg.min_oscillators:
            return Direction.LONG
        if short_count >= self.cfg.min_oscillators:
            return Direction.SHORT
        return None

    # ──────────────────────────────────────────
    # BUILD SIGNAL (compute entries, SL, TP, sizing)
    # ──────────────────────────────────────────

    def build_signal(
        self,
        symbol: str,
        direction: Direction,
        df_15m: pd.DataFrame,
        df_1m: Optional[pd.DataFrame],
        current_price: float,
        account: AccountState,
    ) -> Optional[Signal]:
        """Build a complete entry Signal with cascade levels + sizing."""
        high = df_15m["high"].astype(float)
        low = df_15m["low"].astype(float)
        close = df_15m["close"].astype(float)

        # ATR
        atr_s = Indicators.atr(high, low, close, 14)
        if atr_s is None:
            return None
        atr_val = float(atr_s.iloc[-1])
        if np.isnan(atr_val) or atr_val <= 0:
            atr_val = current_price * 0.02

        # Swing
        lookback = self.cfg.swing_lookback
        swing_high = float(high.tail(lookback).max())
        swing_low = float(low.tail(lookback).min())

        # Cascade entries
        e50, e618 = FiboCalc.cascade_entries(
            swing_high, swing_low, direction, current_price,
            self.cfg.fibo_secondary, self.cfg.fibo_primary,
        )

        # POC blend
        poc = None
        if self.cfg.poc_enabled and df_1m is not None:
            poc = VolumeProfiler.calculate_poc(
                df_1m, atr_val, direction, current_price,
                self.cfg.poc_cluster_width_atr,
            )
            if poc:
                e50 = VolumeProfiler.blend(poc, e50, self.cfg.poc_weight)
                e618 = VolumeProfiler.blend(poc, e618, self.cfg.poc_weight)
                # Re-apply spread protection
                if direction == Direction.LONG:
                    cap = current_price * 0.999
                    e50 = min(e50, cap)
                    e618 = min(e618, cap)
                    if e618 >= e50:
                        e618 = e50 * 0.998
                else:
                    floor = current_price * 1.001
                    e50 = max(e50, floor)
                    e618 = max(e618, floor)
                    if e618 <= e50:
                        e618 = e50 * 1.002

        # Extensions
        ext_1, ext_2 = FiboCalc.extensions(
            swing_high, swing_low, direction,
            self.cfg.fibo_ext_1, self.cfg.fibo_ext_2,
        )

        # SL/TP
        sl_dist = atr_val * self.cfg.sl_atr_multiplier
        if direction == Direction.LONG:
            stop_loss = e618 - sl_dist
            take_profit = ext_1
        else:
            stop_loss = e618 + sl_dist
            take_profit = ext_1

        # Sizing (split 50/50)
        half_risk = self.compound.risk_usdt * self.cfg.cascade_risk_split

        sl_dist_50 = abs(e50 - stop_loss) / e50 if e50 > 0 else 0.02
        if sl_dist_50 <= 0: sl_dist_50 = 0.02
        qty_50 = (half_risk / sl_dist_50) / e50

        sl_dist_618 = abs(e618 - stop_loss) / e618 if e618 > 0 else 0.02
        if sl_dist_618 <= 0: sl_dist_618 = 0.02
        qty_618 = (half_risk / sl_dist_618) / e618

        # Margin check
        margin_needed = (qty_50 * e50 + qty_618 * e618) / self.cfg.leverage
        if margin_needed > account.free_margin:
            logger.info(f"  💸 {symbol}: Insufficient margin ({margin_needed:.2f} > {account.free_margin:.2f})")
            return None

        # RSI/CCI/WillR for logging
        rsi_val = float(Indicators.rsi(close, 14).iloc[-1])
        cci_val = float(Indicators.cci(high, low, close, 14).iloc[-1])
        willr_val = float(Indicators.williams_r(high, low, close, 14).iloc[-1])

        osc_count = 0
        if direction == Direction.LONG:
            if rsi_val < self.cfg.rsi_oversold: osc_count += 1
            if cci_val < self.cfg.cci_oversold: osc_count += 1
            if willr_val < self.cfg.willr_oversold: osc_count += 1
        else:
            if rsi_val > self.cfg.rsi_overbought: osc_count += 1
            if cci_val > self.cfg.cci_overbought: osc_count += 1
            if willr_val > self.cfg.willr_overbought: osc_count += 1

        return Signal(
            symbol=symbol, direction=direction,
            entry_price_50=e50, entry_price_618=e618,
            stop_loss=stop_loss, take_profit=take_profit,
            qty_50=qty_50, qty_618=qty_618,
            risk_usdt_50=half_risk, risk_usdt_618=half_risk,
            atr=atr_val, fibo_ext_1=ext_1, fibo_ext_2=ext_2,
            swing_high=swing_high, swing_low=swing_low,
            oscillators_firing=osc_count,
            rsi_value=rsi_val, cci_value=cci_val, willr_value=willr_val,
            poc_price=poc,
        )

    # ──────────────────────────────────────────
    # POSITION MANAGEMENT (3-phase trailing)
    # ──────────────────────────────────────────

    def update_position(
        self,
        pos: Position,
        current_price: float,
        prev_candle_low: float = 0.0,
        prev_candle_high: float = 0.0,
        reversal_against: bool = False,
    ) -> Position:
        """
        3-phase position management. Returns updated position.
        If pos.phase == CLOSED, the caller should finalize it.
        """
        if pos.phase == PositionPhase.CLOSED:
            return pos

        # Track extremes
        if current_price > pos.highest_price:
            pos.highest_price = current_price
        if current_price < pos.lowest_price:
            pos.lowest_price = current_price

        # Unrealized PnL
        if pos.direction == Direction.LONG:
            unr_pct = (current_price - pos.entry_price) / pos.entry_price
            peak_pct = (pos.highest_price - pos.entry_price) / pos.entry_price
        else:
            unr_pct = (pos.entry_price - current_price) / pos.entry_price
            peak_pct = (pos.entry_price - pos.lowest_price) / pos.entry_price

        pos.highest_profit_pct = max(pos.highest_profit_pct, peak_pct)

        # Check SL hit
        if pos.direction == Direction.LONG and current_price <= pos.stop_loss:
            return self._close_position(pos, current_price, "Stop Loss hit")
        if pos.direction == Direction.SHORT and current_price >= pos.stop_loss:
            return self._close_position(pos, current_price, "Stop Loss hit")

        # Reversal exit (only in profit)
        if reversal_against and unr_pct > 0:
            return self._close_position(pos, current_price, "Oscillator reversal")

        # Phase transitions
        if pos.phase == PositionPhase.BREATHING:
            if unr_pct >= self.cfg.breakeven_trigger_pct:
                fee_buf = pos.entry_price * self.cfg.breakeven_fee_buffer
                if pos.direction == Direction.LONG:
                    pos.stop_loss = pos.entry_price + fee_buf
                else:
                    pos.stop_loss = pos.entry_price - fee_buf
                pos.phase = PositionPhase.BREAKEVEN

        if pos.phase == PositionPhase.BREAKEVEN:
            if peak_pct >= self.cfg.trailing_trigger_pct:
                pos.phase = PositionPhase.TRAILING

        if pos.phase == PositionPhase.TRAILING:
            atr = pos.entry_atr if pos.entry_atr > 0 else pos.entry_price * 0.01
            cushion = atr * self.cfg.trailing_atr_cushion

            if pos.direction == Direction.LONG:
                if prev_candle_low > 0:
                    shadow_sl = prev_candle_low - cushion
                else:
                    shadow_sl = pos.highest_price * (1 - self.cfg.trailing_distance_pct)
                if shadow_sl > pos.stop_loss:
                    pos.stop_loss = shadow_sl
            else:
                if prev_candle_high > 0:
                    shadow_sl = prev_candle_high + cushion
                else:
                    shadow_sl = pos.lowest_price * (1 + self.cfg.trailing_distance_pct)
                if shadow_sl < pos.stop_loss:
                    pos.stop_loss = shadow_sl

        return pos

    def _close_position(self, pos: Position, exit_price: float, reason: str) -> Position:
        """Mark position as closed and compute PnL."""
        if pos.direction == Direction.LONG:
            raw = (exit_price - pos.entry_price) * pos.quantity
        else:
            raw = (pos.entry_price - exit_price) * pos.quantity
        fees = (pos.entry_price + exit_price) * pos.quantity * self.cfg.maker_fee
        pos.pnl_usdt = raw - fees
        pos.phase = PositionPhase.CLOSED
        pos.close_reason = reason
        return pos

    def check_reversal_against(self, pos: Position, df: pd.DataFrame) -> bool:
        """Check if oscillators flipped against our direction."""
        if df is None or len(df) < 50:
            return False
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)

        rsi = Indicators.rsi(close, 14)
        willr = Indicators.williams_r(high, low, close, 14)
        if rsi is None or willr is None:
            return False

        rsi_val = float(rsi.iloc[-1])
        willr_val = float(willr.iloc[-1])
        if np.isnan(rsi_val) or np.isnan(willr_val):
            return False

        if pos.direction == Direction.LONG:
            count = 0
            if rsi_val > self.cfg.rsi_overbought: count += 1
            if willr_val > self.cfg.willr_overbought: count += 1
            return count >= 2
        else:
            count = 0
            if rsi_val < self.cfg.rsi_oversold: count += 1
            if willr_val < self.cfg.willr_oversold: count += 1
            return count >= 2

    def should_reposition_tp(self, pos: Position, current_price: float) -> bool:
        """Check if TP should jump from ext_1 to ext_2."""
        if pos.tp_repositioned or pos.fibo_ext_1 <= 0:
            return False
        if pos.phase != PositionPhase.TRAILING:
            return False
        if pos.direction == Direction.LONG:
            denom = pos.fibo_ext_1 - pos.entry_price
            if denom <= 0: return False
            progress = (current_price - pos.entry_price) / denom
        else:
            denom = pos.entry_price - pos.fibo_ext_1
            if denom <= 0: return False
            progress = (pos.entry_price - current_price) / denom
        return progress >= self.cfg.maker_tp_reposition_trigger
