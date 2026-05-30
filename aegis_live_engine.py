#!/usr/bin/env python3
"""
Aegis-Quant-Lab v4.0 — CORE ENGINE
════════════════════════════════════════════════════════════════
Architecture: Fibonacci Reversal Sniper

Components:
  1. ReversalEngine     — Detects oscillator exhaustion (RSI/CCI/WillR)
  2. FiboCalculator     — Computes Fibonacci retracement/extension levels
  3. PositionManager    — Single-entry lock, trailing stop, breakeven
  4. BybitConnector     — Limit orders (PostOnly), cancel, ticker, positions
  5. CompoundCalculator — Dynamic position sizing (1% risk, compound growth)

Flow:
  Oscillator Resonance → Swing Detection → Fibo Level → Limit Order → Trail
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

import config

logger = logging.getLogger("aegis.engine")


# ══════════════════════════════════════════════════════════════════
# DATA MODELS
# ══════════════════════════════════════════════════════════════════

class TradeDirection(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class PositionState(Enum):
    OPEN = "OPEN"
    BREAKEVEN = "BREAKEVEN"
    TRAILING = "TRAILING"
    CLOSED = "CLOSED"


@dataclass
class ReversalSignal:
    """Output of the Reversal Engine."""
    symbol: str
    direction: TradeDirection
    oscillators_firing: int          # how many (out of 3) are at extremes
    rsi_value: float
    cci_value: float
    willr_value: float
    swing_high: float                # recent swing high (50-bar)
    swing_low: float                 # recent swing low (50-bar)
    fibo_entry_price: float          # calculated Fibonacci entry level
    current_price: float
    atr_value: float                 # for SL/TP calculation
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class LivePosition:
    """Tracked open position."""
    symbol: str
    direction: TradeDirection
    entry_price: float
    entry_time: datetime
    quantity: float
    risk_usdt: float
    stop_loss: float
    take_profit: float
    initial_stop_loss: float         # original SL (never moves wider)
    state: PositionState = PositionState.OPEN
    highest_price: float = 0.0       # highest since entry (for trailing)
    lowest_price: float = float("inf")  # lowest since entry (for trailing)
    highest_profit_pct: float = 0.0
    pnl_usdt: float = 0.0
    close_reason: str = ""
    close_time: Optional[datetime] = None
    # Fibonacci extension targets
    fibo_ext_1_price: float = 0.0
    fibo_ext_2_price: float = 0.0


@dataclass
class PendingOrder:
    """Pending limit order waiting to be filled."""
    symbol: str
    order_id: str
    direction: TradeDirection
    limit_price: float
    stop_loss: float
    take_profit: float
    quantity: float
    risk_usdt: float
    placed_at: float                 # time.time() when placed
    ttl_seconds: int = config.ORDER_TTL_SECONDS
    max_deviation_pct: float = config.PRICE_DEVIATION_CANCEL_PCT
    # Fibo context
    swing_high: float = 0.0
    swing_low: float = 0.0
    fibo_ext_1: float = 0.0
    fibo_ext_2: float = 0.0
    # Cascade twin tracking
    cascade_group_id: str = ""       # shared ID linking twin orders
    cascade_level: str = ""          # "0.50" or "0.618"


# ══════════════════════════════════════════════════════════════════
# COMPONENT 1: REVERSAL ENGINE
# ══════════════════════════════════════════════════════════════════

class ReversalEngine:
    """
    Detects trend exhaustion via oscillator resonance at extremes.

    Signal fires when 2+ of 3 oscillators (RSI14, CCI14, Williams%R)
    hit extreme levels simultaneously → reversal is imminent.
    """

    def detect(self, symbol: str, df: pd.DataFrame) -> Optional[ReversalSignal]:
        """
        Analyze DataFrame for reversal signal.

        Returns ReversalSignal if conditions met, None otherwise.
        """
        if df is None or len(df) < 100:
            return None

        try:
            import pandas_ta as ta
        except ImportError:
            logger.error("pandas_ta not installed!")
            return None

        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)

        # ── Compute oscillators ──
        rsi = ta.rsi(close, length=14)
        cci = ta.cci(high, low, close, length=14)
        willr = ta.willr(high, low, close, length=14)

        if rsi is None or cci is None or willr is None:
            return None

        rsi_val = float(rsi.iloc[-1])
        cci_val = float(cci.iloc[-1])
        willr_val = float(willr.iloc[-1])

        if np.isnan(rsi_val) or np.isnan(cci_val) or np.isnan(willr_val):
            return None

        # ── Count oscillators at extremes ──
        long_count = 0
        short_count = 0

        # RSI
        if rsi_val < config.RSI_OVERSOLD:
            long_count += 1
        elif rsi_val > config.RSI_OVERBOUGHT:
            short_count += 1

        # CCI
        if cci_val < config.CCI_OVERSOLD:
            long_count += 1
        elif cci_val > config.CCI_OVERBOUGHT:
            short_count += 1

        # Williams %R
        if willr_val < config.WILLR_OVERSOLD:
            long_count += 1
        elif willr_val > config.WILLR_OVERBOUGHT:
            short_count += 1

        # ── Check resonance (2+ oscillators at same extreme) ──
        if long_count >= config.MIN_OSCILLATORS_FIRING:
            direction = TradeDirection.LONG
            oscillators_firing = long_count
        elif short_count >= config.MIN_OSCILLATORS_FIRING:
            direction = TradeDirection.SHORT
            oscillators_firing = short_count
        else:
            return None  # No resonance — no signal

        # ── Compute ATR for SL/TP ──
        atr_series = ta.atr(high, low, close, length=14)
        if atr_series is None:
            return None
        atr_val = float(atr_series.iloc[-1])
        if np.isnan(atr_val) or atr_val <= 0:
            atr_val = float(close.iloc[-1]) * 0.02

        # ── Find Swing High/Low (impulse detection) ──
        lookback = config.SWING_LOOKBACK
        recent_high = float(high.tail(lookback).max())
        recent_low = float(low.tail(lookback).min())

        # ── Calculate Fibonacci entry price ──
        current_price = float(close.iloc[-1])
        fibo_entry = FiboCalculator.calculate_entry(
            recent_high, recent_low, direction
        )

        return ReversalSignal(
            symbol=symbol,
            direction=direction,
            oscillators_firing=oscillators_firing,
            rsi_value=rsi_val,
            cci_value=cci_val,
            willr_value=willr_val,
            swing_high=recent_high,
            swing_low=recent_low,
            fibo_entry_price=fibo_entry,
            current_price=current_price,
            atr_value=atr_val,
        )


# ══════════════════════════════════════════════════════════════════
# COMPONENT 2: FIBONACCI CALCULATOR
# ══════════════════════════════════════════════════════════════════

class FiboCalculator:
    """
    Computes Fibonacci retracement and extension levels.

    v4.1: CASCADE MODE — two entry levels (0.50 and 0.618).
    Extensions: 1.618 and 2.618 for trailing TP targets.
    """

    @staticmethod
    def calculate_entry(
        swing_high: float, swing_low: float, direction: TradeDirection
    ) -> float:
        """
        Calculate the 0.618 Fibonacci retracement entry price.
        (Legacy single-entry for compatibility)
        """
        diff = swing_high - swing_low
        if diff <= 0:
            return (swing_high + swing_low) / 2

        fib_level = config.FIBO_LEVEL_PRIMARY  # 0.618

        if direction == TradeDirection.LONG:
            return swing_high - (diff * fib_level)
        else:
            return swing_low + (diff * fib_level)

    @staticmethod
    def calculate_cascade_entries(
        swing_high: float, swing_low: float, direction: TradeDirection,
        current_price: float,
    ) -> tuple[float, float]:
        """
        Calculate CASCADE entry prices (Order Laddering).

        Returns (entry_50, entry_618):
          - entry_50:  50% retracement (closer to market, catches V-reversals)
          - entry_618: 61.8% retracement (deeper, golden ratio)

        Includes NEGATIVE SPREAD PROTECTION:
          - LONG:  if limit_price > current_price → clamp to current - 0.1%
          - SHORT: if limit_price < current_price → clamp to current + 0.1%
        """
        diff = swing_high - swing_low
        if diff <= 0:
            mid = (swing_high + swing_low) / 2
            return mid, mid

        if direction == TradeDirection.LONG:
            entry_50 = swing_high - (diff * config.FIBO_LEVEL_SECONDARY)   # 0.50
            entry_618 = swing_high - (diff * config.FIBO_LEVEL_PRIMARY)    # 0.618

            # NEGATIVE SPREAD FIX: limit buy must be BELOW current price
            max_buy = current_price * 0.999  # at least 0.1% below market
            entry_50 = min(entry_50, max_buy)
            entry_618 = min(entry_618, max_buy)
            # Ensure 618 is always deeper than 50
            if entry_618 >= entry_50:
                entry_618 = entry_50 * 0.998
        else:
            entry_50 = swing_low + (diff * config.FIBO_LEVEL_SECONDARY)    # 0.50
            entry_618 = swing_low + (diff * config.FIBO_LEVEL_PRIMARY)     # 0.618

            # NEGATIVE SPREAD FIX: limit sell must be ABOVE current price
            min_sell = current_price * 1.001  # at least 0.1% above market
            entry_50 = max(entry_50, min_sell)
            entry_618 = max(entry_618, min_sell)
            # Ensure 618 is always deeper than 50
            if entry_618 <= entry_50:
                entry_618 = entry_50 * 1.002

        return entry_50, entry_618

    @staticmethod
    def calculate_extensions(
        swing_high: float, swing_low: float, direction: TradeDirection
    ) -> tuple[float, float]:
        """
        Calculate Fibonacci extension levels for take-profit targets.

        Returns (ext_1.618, ext_2.618).
        """
        diff = swing_high - swing_low
        if diff <= 0:
            return swing_high, swing_high

        if direction == TradeDirection.LONG:
            ext_1 = swing_high + diff * (config.FIBO_EXT_1 - 1)
            ext_2 = swing_high + diff * (config.FIBO_EXT_2 - 1)
        else:
            ext_1 = swing_low - diff * (config.FIBO_EXT_1 - 1)
            ext_2 = swing_low - diff * (config.FIBO_EXT_2 - 1)

        return ext_1, ext_2


# ══════════════════════════════════════════════════════════════════
# COMPONENT 3: POSITION MANAGER (Trailing + Breakeven)
# ══════════════════════════════════════════════════════════════════

class PositionManager:
    """
    Manages open positions with:
      - Breakeven at +1.5% (SL → entry + fees)
      - Trailing stop at +3.0% (trail 1.5% behind peak)
      - Exit on oscillator reversal
    """

    def update(
        self,
        pos: LivePosition,
        current_price: float,
        reversal_detected: bool = False,
    ) -> LivePosition:
        """
        Update position state based on current price.

        State machine: OPEN → BREAKEVEN → TRAILING → CLOSED
        """
        if pos.state == PositionState.CLOSED:
            return pos

        # Track highest/lowest prices
        if current_price > pos.highest_price:
            pos.highest_price = current_price
        if current_price < pos.lowest_price:
            pos.lowest_price = current_price

        # Calculate unrealized profit
        if pos.direction == TradeDirection.LONG:
            unrealized_pct = (current_price - pos.entry_price) / pos.entry_price
            peak_pct = (pos.highest_price - pos.entry_price) / pos.entry_price
        else:
            unrealized_pct = (pos.entry_price - current_price) / pos.entry_price
            peak_pct = (pos.entry_price - pos.lowest_price) / pos.entry_price

        if peak_pct > pos.highest_profit_pct:
            pos.highest_profit_pct = peak_pct

        # ── Check Stop Loss hit ──
        if pos.direction == TradeDirection.LONG and current_price <= pos.stop_loss:
            return self._close(pos, current_price, "Stop Loss hit")
        elif pos.direction == TradeDirection.SHORT and current_price >= pos.stop_loss:
            return self._close(pos, current_price, "Stop Loss hit")

        # ── Exit on oscillator reversal (if enabled) ──
        if config.EXIT_ON_REVERSAL and reversal_detected:
            if unrealized_pct > 0:  # only exit in profit on reversal
                return self._close(pos, current_price, "Oscillator reversal detected")

        # ── State transitions ──
        # OPEN → BREAKEVEN (at +1.5%)
        if pos.state == PositionState.OPEN:
            if unrealized_pct >= config.BREAKEVEN_TRIGGER_PCT:
                # Move SL to breakeven + fees
                fee_buffer = pos.entry_price * config.BREAKEVEN_FEE_BUFFER
                if pos.direction == TradeDirection.LONG:
                    pos.stop_loss = pos.entry_price + fee_buffer
                else:
                    pos.stop_loss = pos.entry_price - fee_buffer
                pos.state = PositionState.BREAKEVEN
                logger.info(
                    f"    🟡 {pos.symbol} → BREAKEVEN | SL={pos.stop_loss:.6f}"
                )

        # BREAKEVEN → TRAILING (at +3.0%)
        if pos.state == PositionState.BREAKEVEN:
            if peak_pct >= config.TRAILING_TRIGGER_PCT:
                pos.state = PositionState.TRAILING
                logger.info(
                    f"    🟢 {pos.symbol} → TRAILING | Peak: +{peak_pct*100:.2f}%"
                )

        # TRAILING: move SL behind price
        if pos.state == PositionState.TRAILING:
            trail_distance = config.TRAILING_DISTANCE_PCT
            if pos.direction == TradeDirection.LONG:
                trail_sl = pos.highest_price * (1 - trail_distance)
                if trail_sl > pos.stop_loss:
                    pos.stop_loss = trail_sl
            else:
                trail_sl = pos.lowest_price * (1 + trail_distance)
                if trail_sl < pos.stop_loss:
                    pos.stop_loss = trail_sl

        return pos

    def _close(
        self, pos: LivePosition, exit_price: float, reason: str
    ) -> LivePosition:
        """Close position and calculate PnL."""
        if pos.direction == TradeDirection.LONG:
            raw_pnl = (exit_price - pos.entry_price) * pos.quantity
        else:
            raw_pnl = (pos.entry_price - exit_price) * pos.quantity

        # Subtract fees (maker both sides)
        fee_cost = (pos.entry_price + exit_price) * pos.quantity * config.MAKER_FEE_RATE
        pos.pnl_usdt = raw_pnl - fee_cost
        pos.state = PositionState.CLOSED
        pos.close_reason = reason
        pos.close_time = datetime.now(timezone.utc)
        return pos

    def check_reversal_for_exit(
        self, pos: LivePosition, df: pd.DataFrame
    ) -> bool:
        """
        Check if oscillators now show reversal AGAINST our position.
        If we're LONG and oscillators hit OVERBOUGHT → exit signal.
        If we're SHORT and oscillators hit OVERSOLD → exit signal.
        """
        if df is None or len(df) < 50:
            return False

        try:
            import pandas_ta as ta
        except ImportError:
            return False

        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)

        rsi = ta.rsi(close, length=14)
        willr = ta.willr(high, low, close, length=14)

        if rsi is None or willr is None:
            return False

        rsi_val = float(rsi.iloc[-1])
        willr_val = float(willr.iloc[-1])

        if np.isnan(rsi_val) or np.isnan(willr_val):
            return False

        if pos.direction == TradeDirection.LONG:
            # Exit LONG when market becomes extremely overbought
            overbought_count = 0
            if rsi_val > config.RSI_OVERBOUGHT:
                overbought_count += 1
            if willr_val > config.WILLR_OVERBOUGHT:
                overbought_count += 1
            return overbought_count >= 2
        else:
            # Exit SHORT when market becomes extremely oversold
            oversold_count = 0
            if rsi_val < config.RSI_OVERSOLD:
                oversold_count += 1
            if willr_val < config.WILLR_OVERSOLD:
                oversold_count += 1
            return oversold_count >= 2


# ══════════════════════════════════════════════════════════════════
# COMPONENT 5: COMPOUND CALCULATOR
# ══════════════════════════════════════════════════════════════════

class CompoundCalculator:
    """
    Dynamic position sizing with compound interest.

    Risk is 1% of current balance normally.
    After 3 consecutive losses, reduces to 0.5%.
    Hard cap at 2%.
    """

    def __init__(self, initial_capital: float = config.INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.current_balance = initial_capital
        self.consecutive_losses = 0
        self.total_trades = 0
        self.wins = 0
        self.losses = 0
        self.max_drawdown_pct = 0.0
        self.peak_balance = initial_capital
        self.trade_history: list[dict] = []

    @property
    def current_risk_pct(self) -> float:
        """Current risk percentage based on streak."""
        if self.consecutive_losses >= config.COMPOUND_LOSS_STREAK_THRESHOLD:
            return config.COMPOUND_RISK_REDUCED
        return config.COMPOUND_RISK_NORMAL

    @property
    def current_risk_usdt(self) -> float:
        """Current risk in USDT."""
        risk = self.current_balance * self.current_risk_pct
        max_risk = self.current_balance * config.COMPOUND_MAX_RISK
        return min(risk, max_risk)

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss_price: float,
        leverage: float = config.LEVERAGE,
    ) -> dict:
        """
        Calculate position size based on risk.

        Returns dict with quantity, risk_usdt, position_value, etc.
        """
        risk_usdt = self.current_risk_usdt
        sl_distance_pct = abs(entry_price - stop_loss_price) / entry_price

        if sl_distance_pct <= 0:
            sl_distance_pct = 0.02  # fallback 2%

        # Position value = risk / SL_distance
        position_value = risk_usdt / sl_distance_pct
        # With leverage: margin needed = position_value / leverage
        margin_needed = position_value / leverage
        # Quantity = position_value / entry_price
        quantity = position_value / entry_price

        return {
            "quantity": quantity,
            "risk_usdt": risk_usdt,
            "risk_pct": self.current_risk_pct * 100,
            "position_value": position_value,
            "margin_needed": margin_needed,
            "sl_distance_pct": sl_distance_pct * 100,
        }

    def record_trade(self, pnl_usdt: float, symbol: str = ""):
        """Record a completed trade result."""
        self.total_trades += 1
        self.current_balance += pnl_usdt

        if pnl_usdt > 0:
            self.wins += 1
            self.consecutive_losses = 0
        else:
            self.losses += 1
            self.consecutive_losses += 1

        # Track peak and drawdown
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
        drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
        if drawdown > self.max_drawdown_pct:
            self.max_drawdown_pct = drawdown

        self.trade_history.append({
            "symbol": symbol,
            "pnl": round(pnl_usdt, 2),
            "balance": round(self.current_balance, 2),
            "time": datetime.now(timezone.utc).isoformat(),
        })

    def get_status(self) -> dict:
        """Return current status summary."""
        roi = (self.current_balance - self.initial_capital) / self.initial_capital * 100
        win_rate = (self.wins / self.total_trades * 100) if self.total_trades > 0 else 0
        return {
            "current_balance": self.current_balance,
            "initial_capital": self.initial_capital,
            "roi_pct": round(roi, 2),
            "total_trades": self.total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate_pct": round(win_rate, 1),
            "max_drawdown_pct": round(self.max_drawdown_pct * 100, 2),
            "consecutive_losses": self.consecutive_losses,
            "current_risk_pct": round(self.current_risk_pct * 100, 2),
            "current_risk_usdt": round(self.current_risk_usdt, 2),
        }


# ══════════════════════════════════════════════════════════════════
# COMPONENT 4: BYBIT DEMO CONNECTOR
# ══════════════════════════════════════════════════════════════════

class BybitConnector:
    """
    Bybit Demo API connector for linear perpetual futures.

    Supports: Limit orders (PostOnly), cancellation, ticker, positions, klines.
    """

    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self._session = None
        self._instrument_cache: dict[str, dict] = {}

    def connect(self) -> bool:
        """Initialize pybit HTTP session."""
        try:
            from pybit.unified_trading import HTTP
            self._session = HTTP(
                testnet=False,
                api_key=self.api_key,
                api_secret=self.api_secret,
                demo=True,             # → api-demo.bybit.com
            )
            # Test connection
            self._session.get_wallet_balance(accountType="UNIFIED")
            logger.info("  ✅ Bybit Demo API connected")
            return True
        except Exception as e:
            logger.error(f"  ❌ Bybit connection failed: {e}")
            return False

    def get_balance(self) -> float:
        """Get USDT balance from unified account."""
        try:
            resp = self._session.get_wallet_balance(accountType="UNIFIED")
            coins = resp["result"]["list"][0]["coin"]
            for coin in coins:
                if coin["coin"] == "USDT":
                    return float(coin["walletBalance"])
            return 0.0
        except Exception as e:
            logger.debug(f"Balance fetch error: {e}")
            return 0.0

    def get_klines(self, symbol: str, interval: str = "15", limit: int = 200) -> Optional[pd.DataFrame]:
        """Fetch kline/candlestick data."""
        try:
            resp = self._session.get_kline(
                category="linear",
                symbol=symbol,
                interval=interval,
                limit=limit,
            )
            rows = resp["result"]["list"]
            if not rows:
                return None

            df = pd.DataFrame(rows, columns=[
                "timestamp", "open", "high", "low", "close", "volume", "turnover"
            ])
            df = df.iloc[::-1].reset_index(drop=True)  # oldest first
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            return df
        except Exception as e:
            logger.debug(f"Kline fetch error for {symbol}: {e}")
            return None

    def get_ticker_price(self, symbol: str) -> float:
        """Get current last traded price."""
        try:
            resp = self._session.get_tickers(category="linear", symbol=symbol)
            tickers = resp["result"]["list"]
            if tickers:
                return float(tickers[0]["lastPrice"])
            return 0.0
        except Exception:
            return 0.0

    def get_instrument_info(self, symbol: str) -> Optional[dict]:
        """Get instrument details (lot size, tick size)."""
        if symbol in self._instrument_cache:
            return self._instrument_cache[symbol]
        try:
            resp = self._session.get_instruments_info(
                category="linear", symbol=symbol
            )
            items = resp["result"]["list"]
            if items:
                self._instrument_cache[symbol] = items[0]
                return items[0]
            return None
        except Exception:
            return None

    def get_lot_size_precision(self, symbol: str) -> int:
        """Get decimal precision for quantity from instrument info."""
        info = self.get_instrument_info(symbol)
        if info:
            try:
                qty_step = info.get("lotSizeFilter", {}).get("qtyStep", "1")
                if "." in qty_step:
                    return len(qty_step.rstrip("0").split(".")[1])
                return 0
            except Exception:
                pass
        # Fallback
        return 3

    def get_tick_size(self, symbol: str) -> float:
        """Get price tick size for proper rounding."""
        info = self.get_instrument_info(symbol)
        if info:
            try:
                tick = info.get("priceFilter", {}).get("tickSize", "0.01")
                return float(tick)
            except Exception:
                pass
        return 0.01

    def round_price(self, price: float, symbol: str) -> float:
        """Round price to valid tick size."""
        tick = self.get_tick_size(symbol)
        if tick > 0:
            return round(round(price / tick) * tick, 10)
        return round(price, 6)

    def place_limit_order(
        self,
        symbol: str,
        side: str,          # "Buy" or "Sell"
        qty: float,
        price: float,
        stop_loss: float = 0,
        take_profit: float = 0,
    ) -> Optional[dict]:
        """
        Place a LIMIT order with PostOnly time-in-force.
        Guarantees maker fee. Rejected if would be taker.
        """
        try:
            params = {
                "category": "linear",
                "symbol": symbol,
                "side": side,
                "orderType": "Limit",
                "price": str(price),
                "qty": str(qty),
                "timeInForce": "PostOnly",
            }
            if stop_loss > 0:
                params["stopLoss"] = str(stop_loss)
            if take_profit > 0:
                params["takeProfit"] = str(take_profit)

            resp = self._session.place_order(**params)

            if resp["retCode"] == 0:
                result = resp["result"]
                logger.info(f"  ✅ Limit order placed: {side} {symbol} @ {price}")
                return result
            else:
                # PostOnly rejected or other API error — log exact reason
                ret_msg = resp.get("retMsg", "unknown")
                ret_code = resp.get("retCode", -1)
                logger.warning(
                    f"  ⚠️ Order rejected for {symbol}: "
                    f"[{ret_code}] {ret_msg}"
                )
                return None
        except Exception as e:
            logger.error(f"  ❌ Order placement error for {symbol}: {e}")
            return None

    def place_market_order(
        self,
        symbol: str,
        side: str,
        qty: float,
    ) -> Optional[dict]:
        """Place a MARKET order (for closing positions)."""
        try:
            resp = self._session.place_order(
                category="linear",
                symbol=symbol,
                side=side,
                orderType="Market",
                qty=str(qty),
                timeInForce="IOC",
            )
            if resp["retCode"] == 0:
                return resp["result"]
            return None
        except Exception as e:
            logger.error(f"  ❌ Market order error: {e}")
            return None

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel a pending order."""
        try:
            resp = self._session.cancel_order(
                category="linear",
                symbol=symbol,
                orderId=order_id,
            )
            return resp["retCode"] == 0
        except Exception:
            return False

    def get_order_detail(self, symbol: str, order_id: str) -> Optional[dict]:
        """Get order status and fill details."""
        try:
            resp = self._session.get_order_history(
                category="linear",
                symbol=symbol,
                orderId=order_id,
            )
            orders = resp["result"]["list"]
            if orders:
                return orders[0]
            # Try open orders
            resp2 = self._session.get_open_orders(
                category="linear",
                symbol=symbol,
                orderId=order_id,
            )
            open_orders = resp2["result"]["list"]
            if open_orders:
                return open_orders[0]
            return None
        except Exception:
            return None

    def get_positions(self) -> list[dict]:
        """Get all open positions from exchange."""
        try:
            resp = self._session.get_positions(
                category="linear",
                settleCoin="USDT",
            )
            return resp["result"]["list"]
        except Exception:
            return []
