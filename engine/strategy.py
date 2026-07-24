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

    # Trend Invalidation Exit (SuperTrend-based emergency evacuation)
    trend_invalidation_enabled: bool = True
    supertrend_period: int = 10        # ATR period for SuperTrend
    supertrend_multiplier: float = 3.0  # ATR multiplier for bands
    trend_confirm_bars: int = 2         # consecutive bars against to confirm invalidation

    # ══════════════════════════════════════════════════════════════
    # v5.5 — "TWO-WINGED" DUAL-LOT DYNAMIC EXIT ENGINE
    # (These control EXIT only — entry thresholds above are untouched.)
    # ══════════════════════════════════════════════════════════════
    dual_lot_enabled: bool = True
    dual_lot_split: float = 0.5              # 50/50 split A/B

    # Lot A — Maker Fix: tight limit maker to cover fees + secure fractional profit.
    # Target NET profit band after fees: +1.5% .. +2.0%.
    maker_fix_net_target_low: float = 0.015  # +1.5% net
    maker_fix_net_target_high: float = 0.020  # +2.0% net (upper edge of band)

    # Lot B — Momentum Float: NO static TP. Close on momentum decay.
    # RSI crossing back to the neutral 50 line validates local reversal.
    momentum_rsi_neutral: float = 50.0
    momentum_rsi_len: int = 14
    momentum_decay_confirm_bars: int = 1     # bars past neutral needed to confirm decay
    momentum_min_profit_pct: float = 0.0     # only momentum-exit once at/above this PnL

    # Protection Cascade: when Lot A FILLS, snap Lot B SL to breakeven + buffer.
    breakeven_snap_buffer: float = 0.0006    # BE + 0.06% buffer (covers fees, eliminates risk)

    # Structural Invalidation Stop: tight SL behind the extreme pivot/fibo grid level
    # that triggered the trade (replaces arbitrary ATR stop distance).
    structural_stop_enabled: bool = True
    structural_stop_buffer_atr: float = 0.25  # buffer beyond pivot = 0.25 × ATR (Lot A, tight)
    # Lot B ("float") gets extra room below the structural pivot so it can
    # survive long enough for the momentum-decay exit to be the deciding signal.
    momentum_float_extra_atr: float = 1.25    # additional buffer for Lot B stop


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

    @staticmethod
    def supertrend(
        high: pd.Series, low: pd.Series, close: pd.Series,
        period: int = 10, multiplier: float = 3.0,
    ) -> Optional[pd.Series]:
        """
        SuperTrend indicator — returns a Series of +1 (uptrend) / -1 (downtrend).

        Logic:
          - Upper band = HL2 + multiplier * ATR
          - Lower band = HL2 - multiplier * ATR
          - Trend flips when close crosses the band

        Lightweight, ATR-based, binary output — ideal for trend invalidation.
        """
        n = len(close)
        if n < period + 1:
            return None

        hl2 = (high + low) / 2.0

        # ATR calculation (RMA / Wilder's smoothing)
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

        # Basic bands
        basic_upper = hl2 + multiplier * atr
        basic_lower = hl2 - multiplier * atr

        # Final bands (with clamping logic)
        close_vals = close.values.astype(float).copy()
        upper_vals = basic_upper.values.astype(float).copy()
        lower_vals = basic_lower.values.astype(float).copy()
        dir_vals = np.ones(n, dtype=float)  # 1 = uptrend

        for i in range(1, n):
            # Clamp lower band: can only go UP in uptrend
            if lower_vals[i] < lower_vals[i - 1] and close_vals[i - 1] > lower_vals[i - 1]:
                lower_vals[i] = lower_vals[i - 1]

            # Clamp upper band: can only go DOWN in downtrend
            if upper_vals[i] > upper_vals[i - 1] and close_vals[i - 1] < upper_vals[i - 1]:
                upper_vals[i] = upper_vals[i - 1]

            # Direction logic
            if dir_vals[i - 1] == 1:  # was uptrend
                if close_vals[i] < lower_vals[i]:
                    dir_vals[i] = -1  # flip to downtrend
                else:
                    dir_vals[i] = 1
            else:  # was downtrend
                if close_vals[i] > upper_vals[i]:
                    dir_vals[i] = 1  # flip to uptrend
                else:
                    dir_vals[i] = -1

        # Set NaN for warmup period
        dir_vals[:period] = np.nan
        return pd.Series(dir_vals, index=close.index)


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

    def __init__(
        self,
        cfg: StrategyConfig,
        initial_balance: float = 2000.0,
        exchange_rules: Optional[dict[str, dict]] = None,
    ):
        self.cfg = cfg
        self.compound = CompoundCalc(initial_balance, cfg)
        self.trade_history: list[TradeResult] = []
        # Exchange rules: {"BTCUSDT": {"qtyStep": 0.001, "minOrderQty": 0.001, "tickSize": 0.1}}
        self._exchange_rules = exchange_rules or {}

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

        # ─── Exchange Rules Precision (Exact-in-Exact) ───
        rules = self._exchange_rules.get(symbol)
        if rules:
            from broker.exchange_rules import floor_qty, round_to_tick

            qty_step = rules.get("qtyStep", 0)
            min_order_qty = rules.get("minOrderQty", 0)
            tick_size = rules.get("tickSize", 0)

            # Floor quantities to qtyStep
            if qty_step > 0:
                qty_50 = floor_qty(qty_50, qty_step)
                qty_618 = floor_qty(qty_618, qty_step)

            # Check minOrderQty — cancel signal if below minimum
            if min_order_qty > 0:
                if qty_50 < min_order_qty or qty_618 < min_order_qty:
                    logger.info(
                        f"  📏 {symbol}: Qty below minimum "
                        f"(qty_50={qty_50}, qty_618={qty_618}, min={min_order_qty})"
                    )
                    return None

            # Round prices to tickSize
            if tick_size > 0:
                e50 = round_to_tick(e50, tick_size)
                e618 = round_to_tick(e618, tick_size)
                stop_loss = round_to_tick(stop_loss, tick_size)
                take_profit = round_to_tick(take_profit, tick_size)
                ext_1 = round_to_tick(ext_1, tick_size)
                ext_2 = round_to_tick(ext_2, tick_size)

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
        trend_invalidated: bool = False,
    ) -> Position:
        """
        3-phase position management. Returns updated position.
        If pos.phase == CLOSED, the caller should finalize it.

        trend_invalidated: if True, emergency market exit (TREND_INVALIDATION).
        """
        if pos.phase == PositionPhase.CLOSED:
            return pos

        # Track extremes
        if current_price > pos.highest_price:
            pos.highest_price = current_price
        if current_price < pos.lowest_price:
            pos.lowest_price = current_price

    # Unrealized PnL
        if pos.entry_price <= 0:
            unr_pct = 0.0
            peak_pct = 0.0
        else:
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

        # TREND INVALIDATION — emergency evacuation (any PnL state)
        if trend_invalidated:
            return self._close_position(pos, current_price, "TREND_INVALIDATION")

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
                # Cap shadow_sl so it never exceeds current_price (0.1% buffer)
                shadow_sl = min(shadow_sl, current_price * 0.999)
                if shadow_sl > pos.stop_loss:
                    pos.stop_loss = shadow_sl
            else:
                if prev_candle_high > 0:
                    shadow_sl = prev_candle_high + cushion
                else:
                    shadow_sl = pos.lowest_price * (1 + self.cfg.trailing_distance_pct)
                # Floor shadow_sl so it never goes below current_price (0.1% buffer)
                shadow_sl = max(shadow_sl, current_price * 1.001)
                if shadow_sl < pos.stop_loss:
                    pos.stop_loss = shadow_sl

        return pos

    def _close_position(self, pos: Position, exit_price: float, reason: str) -> Position:
        """Mark position as closed and compute PnL."""
        if pos.direction == Direction.LONG:
            raw = (exit_price - pos.entry_price) * pos.quantity
        else:
            raw = (pos.entry_price - exit_price) * pos.quantity
        # Market exits (TREND_INVALIDATION, MOMENTUM_DECAY, Stop Loss, Market close)
        # use taker fee; limit-maker TP exits use maker fee.
        if reason in ("TREND_INVALIDATION", "Stop Loss hit", "MOMENTUM_DECAY", "Market close"):
            fee_rate = self.cfg.taker_fee
        else:
            fee_rate = self.cfg.maker_fee
        fees = (pos.entry_price + exit_price) * pos.quantity * fee_rate
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

    def check_trend_invalidation(self, pos: Position, df: pd.DataFrame) -> bool:
        """
        Check if the global trend has broken against our position.

        Uses SuperTrend: if the last N bars (trend_confirm_bars) show
        trend direction OPPOSITE to our position → evacuate immediately.

        This catches slow trend reversals that oscillators miss
        (RSI can sit at oversold for weeks in a strong downtrend).
        """
        if not self.cfg.trend_invalidation_enabled:
            return False
        if df is None or len(df) < self.cfg.supertrend_period + self.cfg.trend_confirm_bars + 1:
            return False

        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)

        st = Indicators.supertrend(
            high, low, close,
            period=self.cfg.supertrend_period,
            multiplier=self.cfg.supertrend_multiplier,
        )
        if st is None:
            return False

        # Check last N bars for confirmed trend against us
        confirm_bars = self.cfg.trend_confirm_bars
        recent = st.iloc[-confirm_bars:]

        # All must be valid (not NaN)
        if recent.isna().any():
            return False

        if pos.direction == Direction.LONG:
            # LONG invalidated when SuperTrend shows DOWNTREND for N bars
            return (recent == -1).all()
        else:
            # SHORT invalidated when SuperTrend shows UPTREND for N bars
            return (recent == 1).all()

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

    # ══════════════════════════════════════════════════════════════
    # v5.5 — "TWO-WINGED" DUAL-LOT DYNAMIC EXIT ENGINE (pure logic)
    # Entry logic above is NOT touched. These methods only shape exits.
    # ══════════════════════════════════════════════════════════════

    def compute_structural_stop(
        self, direction: Direction, pivot_level: float, atr: float,
    ) -> float:
        """
        Tight structural invalidation stop placed immediately behind the
        extreme pivot / Fibo grid level that triggered the trade.

        For a LONG the pivot is the swing_low (the level being defended);
        stop sits just below it. For a SHORT the pivot is the swing_high;
        stop sits just above it. Replaces arbitrary ATR-multiple stops.
        """
        buffer = max(atr, 0.0) * self.cfg.structural_stop_buffer_atr
        if buffer <= 0:
            buffer = pivot_level * 0.001
        if direction == Direction.LONG:
            return pivot_level - buffer
        return pivot_level + buffer

    def compute_float_stop(
        self, direction: Direction, pivot_level: float, atr: float,
    ) -> float:
        """
        Lot B ("Momentum Float") stop — the structural stop plus extra ATR
        breathing room, so the float lot is exited by momentum decay rather
        than being prematurely stopped at the same tight level as Lot A.
        """
        base_buf = max(atr, 0.0) * self.cfg.structural_stop_buffer_atr
        extra = max(atr, 0.0) * self.cfg.momentum_float_extra_atr
        buffer = base_buf + extra
        if buffer <= 0:
            buffer = pivot_level * 0.002
        if direction == Direction.LONG:
            return pivot_level - buffer
        return pivot_level + buffer

    def compute_maker_fix_tp(
        self, direction: Direction, entry_price: float,
    ) -> float:
        """
        Lot A limit-maker target price.

        We solve for a gross move that lands NET profit inside the
        +1.5%..+2.0% band after paying maker fees on both legs. We aim at
        the LOW edge (+1.5% net) so the maker order is as tight as possible
        (fastest, highest fill probability) while still guaranteed positive.
        """
        # Round-trip maker fee (entry maker + exit maker), as a fraction of notional.
        roundtrip_fee = 2.0 * self.cfg.maker_fee
        gross_move = self.cfg.maker_fix_net_target_low + roundtrip_fee
        if direction == Direction.LONG:
            return entry_price * (1.0 + gross_move)
        return entry_price * (1.0 - gross_move)

    def maker_fix_net_pct(self, direction: Direction, entry_price: float, exit_price: float) -> float:
        """Net profit % for Lot A after round-trip maker fees."""
        if entry_price <= 0:
            return 0.0
        if direction == Direction.LONG:
            gross = (exit_price - entry_price) / entry_price
        else:
            gross = (entry_price - exit_price) / entry_price
        return gross - 2.0 * self.cfg.maker_fee

    def check_momentum_decay(self, pos: Position, df: pd.DataFrame) -> bool:
        """
        Lot B momentum-decay exit trigger (localized recovery exhaustion).

        A reversal entry begins with RSI at an extreme (e.g. deeply oversold
        for a LONG). The recovery is only "alive" once RSI has pushed THROUGH
        the neutral 50 line in our favour. Momentum is declared DEAD when,
        after that recovery, RSI crosses BACK across the neutral 50 line —
        validating a local reversal over `momentum_decay_confirm_bars` bars.

          - LONG  : RSI rose past 50 (bounce worked) then falls back below 50.
          - SHORT : RSI fell past 50 (drop worked) then rises back above 50.

        This prevents a false decay signal on the very first bars, when RSI is
        still sitting at the entry extreme.
        """
        need = self.cfg.momentum_rsi_len + self.cfg.momentum_decay_confirm_bars + 2
        if df is None or len(df) < need:
            return False

        close = df["close"].astype(float)
        rsi = Indicators.rsi(close, self.cfg.momentum_rsi_len)
        if rsi is None:
            return False

        rsi = rsi.dropna()
        if len(rsi) < self.cfg.momentum_decay_confirm_bars + 2:
            return False

        neutral = self.cfg.momentum_rsi_neutral
        confirm = max(1, self.cfg.momentum_decay_confirm_bars)
        recent = rsi.iloc[-confirm:]
        # look back over a short window to confirm the recovery actually happened
        window = rsi.iloc[-(confirm + 6):-confirm] if len(rsi) > confirm + 6 else rsi.iloc[:-confirm]
        if recent.isna().any() or len(window) == 0:
            return False

        if pos.direction == Direction.LONG:
            recovered = bool((window > neutral).any())      # bounced above 50 earlier
            faded = bool((recent < neutral).all())          # now back below 50
            return recovered and faded
        else:
            recovered = bool((window < neutral).any())       # dropped below 50 earlier
            faded = bool((recent > neutral).all())           # now back above 50
            return recovered and faded

    def apply_breakeven_cascade(self, pos: Position) -> Position:
        """
        Protection Cascade — invoked the moment Lot A's limit order state
        becomes FILLED. Snaps Lot B's stop loss to breakeven + a minor buffer,
        eliminating residual risk on the floating lot.
        """
        buf = pos.entry_price * self.cfg.breakeven_snap_buffer
        if pos.direction == Direction.LONG:
            new_sl = pos.entry_price + buf
            # only ratchet upward
            if new_sl > pos.stop_loss:
                pos.stop_loss = new_sl
        else:
            new_sl = pos.entry_price - buf
            if new_sl < pos.stop_loss or pos.stop_loss <= 0:
                pos.stop_loss = new_sl
        if pos.phase == PositionPhase.BREATHING:
            pos.phase = PositionPhase.BREAKEVEN
        return pos

    def update_momentum_float(
        self,
        pos: Position,
        current_price: float,
        momentum_dead: bool = False,
        trend_invalidated: bool = False,
    ) -> Position:
        """
        Lot B ("Momentum Float") manager — NO static take-profit.

        Priority of exits:
          1. Structural / breakeven stop hit  → close (protective).
          2. Trend invalidation               → emergency market close.
          3. Momentum decay (RSI back to 50)  → market close once in profit.

        Returns the (possibly CLOSED) position.
        """
        if pos.phase == PositionPhase.CLOSED:
            return pos

        # Track extremes
        if current_price > pos.highest_price:
            pos.highest_price = current_price
        if current_price < pos.lowest_price:
            pos.lowest_price = current_price

        if pos.entry_price > 0:
            if pos.direction == Direction.LONG:
                unr_pct = (current_price - pos.entry_price) / pos.entry_price
            else:
                unr_pct = (pos.entry_price - current_price) / pos.entry_price
        else:
            unr_pct = 0.0

        # 1. Protective stop (structural or snapped breakeven)
        if pos.stop_loss > 0:
            if pos.direction == Direction.LONG and current_price <= pos.stop_loss:
                return self._close_position(pos, pos.stop_loss, "Stop Loss hit")
            if pos.direction == Direction.SHORT and current_price >= pos.stop_loss:
                return self._close_position(pos, pos.stop_loss, "Stop Loss hit")

        # 2. Momentum decay — PRIMARY dynamic exit for the float lot.
        #    (Evaluated BEFORE trend invalidation: the localized recovery dying
        #     out is Lot B's intended signal; RSI crossing back to neutral 50.)
        if momentum_dead and unr_pct >= self.cfg.momentum_min_profit_pct:
            return self._close_position(pos, current_price, "MOMENTUM_DECAY")

        # 3. Trend invalidation — EMERGENCY backstop only. Requires a real
        #    give-back scenario (meaningfully in profit) so it does not
        #    hair-trigger on every mean-reversion bounce.
        if trend_invalidated and unr_pct >= self.cfg.trailing_trigger_pct:
            return self._close_position(pos, current_price, "TREND_INVALIDATION")

        return pos
