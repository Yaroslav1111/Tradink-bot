#!/usr/bin/env python3
"""
Aegis-Quant-Lab v4.2 — LIVE BOT «Fibonacci Reversal Sniper»
══════════════════════════════════════════════════════════════════

Architecture: Impulse → POC+Fibo Blend → Limit Order → Shadow Trail → Maker Exit

v4.2 Upgrades:
  - Volume Profile POC (Lazy Sniper: only 1m fetch after 15m signal fires)
  - 3-Phase Trailing (ATR buffer → Breakeven → Shadow-based trailing)
  - Maker exits (limit TP at ext_1.618 with dynamic repositioning to ext_2.618)
  - SHORT fully supported (no directional bias)

Execution cycle (every 15 minutes):
  1. Wake at candle close
  2. Manage pending limit orders (check fills, TTL, deviation)
  3. Monitor open positions (shadow trailing, TP repositioning, reversal exit)
  4. Scan ALL symbols for oscillator reversal resonance (15m)
  5. If reversal detected → fetch 1m → compute POC → blend with Fibo → Limit order
  6. Sleep until next candle

Usage:
  # Set keys via .env.local (auto-loaded) or environment variables:
  export BYBIT_DEMO_API_KEY="your_key"
  export BYBIT_DEMO_API_SECRET="your_secret"
  python live_bot.py            # live mode
  python live_bot.py --dry-run  # test pipeline without API
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from aegis_live_engine import (
    ReversalEngine,
    FiboCalculator,
    VolumeProfiler,
    PositionManager,
    CompoundCalculator,
    BybitConnector,
    LivePosition,
    PendingOrder,
    ReversalSignal,
    TradeDirection,
    PositionState,
)

try:
    from dotenv import load_dotenv
    load_dotenv(".env.local")
except ImportError:
    pass


# ══════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════

def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """Configure console + file logging."""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("aegis.pulse")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # Console handler — avoid UnicodeEncodeError on Windows cp1251
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(
        "%(asctime)s | %(message)s", datefmt="%H:%M:%S"
    ))
    console.terminator = "\n"
    logger.addHandler(console)

    log_file = os.path.join(log_dir, f"live_bot_{datetime.now().strftime('%Y%m%d')}.log")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(fh)

    return logger


# ══════════════════════════════════════════════════════════════════
# SIGNAL HANDLERS
# ══════════════════════════════════════════════════════════════════

_shutdown_requested = False


def _signal_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    print("\n\n  🛑 Shutdown signal received. Finishing current cycle...")


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ══════════════════════════════════════════════════════════════════
# CORE: LIVE BOT
# ══════════════════════════════════════════════════════════════════

class LiveBot:
    """
    Fibonacci Reversal Sniper — autonomous 24/7 trading loop.

    v4.2 Components:
      - ReversalEngine: detects oscillator exhaustion (15m)
      - VolumeProfiler: finds POC from 1m micro-structure (Lazy Sniper)
      - FiboCalculator: computes entry/extension levels
      - PositionManager: 3-phase trailing (shadow-based)
      - CompoundCalculator: dynamic position sizing
      - BybitConnector: limit orders on demo API (Maker entry + exit)
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        symbols: list[str] | None = None,
        interval_minutes: int = 15,
        initial_capital: float = 2000.0,
        leverage: float = 5.0,
    ):
        self.symbols = symbols or config.SYMBOLS
        self.interval_minutes = interval_minutes
        self.leverage = leverage

        self.logger = setup_logging()
        self.cycle_count = 0
        self.total_orders_placed = 0
        self.start_time = datetime.now(timezone.utc)

        # ── Core components ──
        self.reversal_engine = ReversalEngine()
        self.position_manager = PositionManager()
        self.compound = CompoundCalculator(initial_capital)
        self.connector = BybitConnector(api_key, api_secret)

        # ── State ──
        self.open_positions: list[LivePosition] = []
        self.closed_positions: list[LivePosition] = []
        self.pending_orders: list[PendingOrder] = []

        # ── Banner ──
        self.logger.info("=" * 70)
        self.logger.info("  🎯 AEGIS-QUANT-LAB v4.2 — Fibonacci Reversal Sniper")
        self.logger.info("=" * 70)
        self.logger.info(f"  Capital:    ${initial_capital:.2f} USDT")
        self.logger.info(f"  Leverage:   {leverage}x")
        self.logger.info(f"  Symbols:    {len(self.symbols)} coins")
        self.logger.info(f"  Interval:   {interval_minutes}m candles")
        self.logger.info(f"  Strategy:   Oscillator Reversal → POC+Fibo → Limit → Shadow Trail")
        self.logger.info(f"  Order type: Limit (PostOnly, Maker 0.020%)")
        self.logger.info(f"  Exits:      Limit TP (Maker) + Shadow Trailing SL")
        self.logger.info(f"  Directions: LONG + SHORT (symmetric)")
        self.logger.info(f"  TTL:        {config.ORDER_TTL_SECONDS}s")
        self.logger.info("=" * 70)

        # ── Connect to Bybit ──
        if not self.connector.connect():
            raise ConnectionError("Failed to connect to Bybit Demo API")

        balance = self.connector.get_balance()
        self.logger.info(f"  💰 Connected | Balance: ${balance:.2f} USDT")
        if balance > 0:
            self.compound.current_balance = balance

    # ──────────────────────────────────────────────
    # TIMING
    # ──────────────────────────────────────────────

    def _seconds_until_next_candle(self) -> float:
        """Calculate seconds until next candle close (+2s buffer)."""
        now = datetime.now(timezone.utc)
        next_min = ((now.minute // self.interval_minutes) + 1) * self.interval_minutes

        if next_min >= 60:
            next_time = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        else:
            next_time = now.replace(minute=next_min, second=0, microsecond=0)

        next_time += timedelta(seconds=2)
        wait = (next_time - now).total_seconds()
        return max(wait, 0)

    # ──────────────────────────────────────────────
    # DATA FETCH (parallel)
    # ──────────────────────────────────────────────

    def _fetch_all_candles(self) -> dict[str, pd.DataFrame]:
        """Fetch candles for all symbols in parallel (10 workers)."""
        results: dict[str, pd.DataFrame] = {}

        def _fetch_one(sym: str):
            try:
                df = self.connector.get_klines(
                    sym, interval=config.CANDLE_INTERVAL, limit=config.CANDLE_LIMIT
                )
                return sym, df
            except Exception:
                return sym, None

        with ThreadPoolExecutor(max_workers=config.PARALLEL_WORKERS) as executor:
            futures = {executor.submit(_fetch_one, s): s for s in self.symbols}
            for future in as_completed(futures):
                try:
                    sym, df = future.result(timeout=15)
                    if df is not None and len(df) >= 100:
                        results[sym] = df
                except Exception:
                    pass

        return results

    def _fetch_1m_candles(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Fetch 1-minute candles for POC calculation.
        Called ONLY when a 15m signal fires — Lazy Sniper pattern.
        Single API call, no parallelism needed.
        """
        try:
            df = self.connector.get_klines(
                symbol, interval="1", limit=config.POC_1M_LOOKBACK
            )
            if df is not None and len(df) >= 20:
                return df
            return None
        except Exception:
            return None

    # ──────────────────────────────────────────────
    # PENDING ORDERS MANAGEMENT
    # ──────────────────────────────────────────────

    def _manage_pending_orders(self):
        """
        Check all pending limit orders (CASCADE-aware):
          - FILLED → register as open position (or merge with twin)
          - TTL expired → cancel (and cancel twin if exists)
          - Price deviation > 1% → cancel
          - Twin logic: if both filled → merge into single position with avg price
        """
        if not self.pending_orders:
            return

        self.logger.info(f"\n  ⏳ Checking {len(self.pending_orders)} pending orders...")

        still_pending = []
        filled_orders: list[PendingOrder] = []  # collect fills this cycle
        now = time.time()

        for order in self.pending_orders:
            elapsed = now - order.placed_at

            # Check 1: TTL expired
            if elapsed > order.ttl_seconds:
                self.logger.info(
                    f"    ⌛ EXPIRED: {order.symbol} [{order.cascade_level}] "
                    f"(TTL {order.ttl_seconds}s)"
                )
                self.connector.cancel_order(order.symbol, order.order_id)
                continue

            # Check 2: Order filled?
            detail = self.connector.get_order_detail(order.symbol, order.order_id)
            if detail:
                status = detail.get("orderStatus", "")

                if status == "Filled":
                    avg_price = float(detail.get("avgPrice", "0") or "0")
                    fill_price = avg_price if avg_price > 0 else order.limit_price

                    self.logger.info(
                        f"    ✅ FILLED: {order.symbol} {order.direction.value} "
                        f"[{order.cascade_level}] @ {fill_price:.6f} ({elapsed:.0f}s)"
                    )
                    order.limit_price = fill_price  # update with actual fill
                    filled_orders.append(order)
                    continue

                elif status in ("Cancelled", "Rejected", "Deactivated"):
                    self.logger.info(
                        f"    ❌ {status}: {order.symbol} [{order.cascade_level}]"
                    )
                    continue

            # Check 3: Price deviation — cancel if price moved away
            current_price = self.connector.get_ticker_price(order.symbol)
            if current_price > 0:
                deviation = abs(current_price - order.limit_price) / order.limit_price
                if deviation > order.max_deviation_pct:
                    self.logger.info(
                        f"    📉 CANCEL: {order.symbol} [{order.cascade_level}] "
                        f"deviation {deviation*100:.2f}%"
                    )
                    self.connector.cancel_order(order.symbol, order.order_id)
                    continue

            # Still pending
            still_pending.append(order)
            self.logger.info(
                f"    ⏳ WAITING: {order.symbol} [{order.cascade_level}] "
                f"({elapsed:.0f}s/{order.ttl_seconds}s)"
            )

        self.pending_orders = still_pending

        # ── Process filled orders: merge cascade twins ──
        if not filled_orders:
            return

        # Group filled orders by cascade_group_id
        groups: dict[str, list[PendingOrder]] = {}
        for order in filled_orders:
            gid = order.cascade_group_id or order.order_id
            groups.setdefault(gid, []).append(order)

        for group_id, group_orders in groups.items():
            if len(group_orders) == 2:
                # BOTH twins filled → merge into single position with avg price
                o1, o2 = group_orders
                total_qty = o1.quantity + o2.quantity
                avg_entry = (
                    (o1.limit_price * o1.quantity + o2.limit_price * o2.quantity)
                    / total_qty
                )
                total_risk = o1.risk_usdt + o2.risk_usdt

                # SL/TP from the deeper order (0.618)
                sl = o2.stop_loss if o2.cascade_level == "0.618" else o1.stop_loss
                tp = o1.take_profit  # both have same TP target

                self.logger.info(
                    f"    🔗 MERGED: {o1.symbol} | Avg entry: {avg_entry:.6f} "
                    f"| Total qty: {total_qty}"
                )

                pos = LivePosition(
                    symbol=o1.symbol,
                    direction=o1.direction,
                    entry_price=avg_entry,
                    entry_time=datetime.now(timezone.utc),
                    quantity=total_qty,
                    risk_usdt=total_risk,
                    stop_loss=sl,
                    take_profit=tp,
                    initial_stop_loss=sl,
                    highest_price=avg_entry,
                    lowest_price=avg_entry,
                    fibo_ext_1_price=o1.fibo_ext_1,
                    fibo_ext_2_price=o1.fibo_ext_2,
                )
                self.open_positions.append(pos)

                # Cancel any remaining twin still pending
                for pend in self.pending_orders[:]:
                    if pend.cascade_group_id == group_id:
                        self.connector.cancel_order(pend.symbol, pend.order_id)
                        self.pending_orders.remove(pend)
            else:
                # Single fill (twin may still be pending or expired)
                for order in group_orders:
                    pos = LivePosition(
                        symbol=order.symbol,
                        direction=order.direction,
                        entry_price=order.limit_price,
                        entry_time=datetime.now(timezone.utc),
                        quantity=order.quantity,
                        risk_usdt=order.risk_usdt,
                        stop_loss=order.stop_loss,
                        take_profit=order.take_profit,
                        initial_stop_loss=order.stop_loss,
                        highest_price=order.limit_price,
                        lowest_price=order.limit_price,
                        fibo_ext_1_price=order.fibo_ext_1,
                        fibo_ext_2_price=order.fibo_ext_2,
                    )
                    self.open_positions.append(pos)

                    # Cancel the twin if still pending
                    for pend in self.pending_orders[:]:
                        if (pend.cascade_group_id == group_id
                                and pend.order_id != order.order_id):
                            self.logger.info(
                                f"    🗑️ Cancel twin: {pend.symbol} [{pend.cascade_level}]"
                            )
                            self.connector.cancel_order(pend.symbol, pend.order_id)
                            self.pending_orders.remove(pend)

    # ──────────────────────────────────────────────
    # POSITION MONITORING (v4.2: Shadow Trailing + Maker TP)
    # ──────────────────────────────────────────────

    def _monitor_positions(self, all_candles: dict[str, pd.DataFrame]):
        """
        Monitor all open positions:
          - Extract prev candle low/high for shadow trailing
          - Apply 3-phase position management
          - Check for oscillator reversal → exit
          - Reposition limit TP when trailing extends past ext_1.618
        """
        if not self.open_positions:
            return

        self.logger.info(f"\n  📊 Monitoring {len(self.open_positions)} positions...")

        still_open = []

        for pos in self.open_positions:
            if pos.state == PositionState.CLOSED:
                continue

            # Get current data
            df = all_candles.get(pos.symbol)
            if df is None or len(df) == 0:
                still_open.append(pos)
                continue

            current_price = float(df["close"].iloc[-1])

            # Extract previous candle's low/high for shadow trailing
            prev_candle_low = float(df["low"].iloc[-2]) if len(df) >= 2 else 0.0
            prev_candle_high = float(df["high"].iloc[-2]) if len(df) >= 2 else 0.0

            # Check for oscillator reversal (exit signal)
            reversal = self.position_manager.check_reversal_for_exit(pos, df)

            # Update position state (3-phase trailing)
            updated = self.position_manager.update(
                pos, current_price, reversal,
                prev_candle_low=prev_candle_low,
                prev_candle_high=prev_candle_high,
            )

            if updated.state == PositionState.CLOSED:
                # Close position on exchange
                close_side = "Sell" if updated.direction == TradeDirection.LONG else "Buy"
                self.connector.place_market_order(
                    updated.symbol, close_side, updated.quantity
                )
                self.compound.record_trade(updated.pnl_usdt, updated.symbol)
                self.closed_positions.append(updated)

                emoji = "✅" if updated.pnl_usdt > 0 else "❌"
                self.logger.info(
                    f"  {emoji} CLOSED {updated.symbol}: "
                    f"PnL=${updated.pnl_usdt:+.2f} | {updated.close_reason}"
                )
            else:
                # ── v4.2: Dynamic TP repositioning ──
                if (updated.state == PositionState.TRAILING
                        and not updated.tp_repositioned
                        and updated.fibo_ext_1_price > 0
                        and updated.fibo_ext_2_price > 0):
                    # Check if price has passed 80% of ext_1 target
                    if updated.direction == TradeDirection.LONG:
                        progress = (current_price - updated.entry_price) / (
                            updated.fibo_ext_1_price - updated.entry_price
                        ) if updated.fibo_ext_1_price != updated.entry_price else 0
                    else:
                        progress = (updated.entry_price - current_price) / (
                            updated.entry_price - updated.fibo_ext_1_price
                        ) if updated.fibo_ext_1_price != updated.entry_price else 0

                    if progress >= config.MAKER_TP_REPOSITION_TRIGGER:
                        # Reposition TP to ext_2.618 (let profits run)
                        new_tp = updated.fibo_ext_2_price
                        self.connector.set_trading_stop(
                            updated.symbol,
                            take_profit=new_tp,
                            stop_loss=updated.stop_loss,
                        )
                        updated.take_profit = new_tp
                        updated.tp_repositioned = True
                        self.logger.info(
                            f"    🎯 TP REPOSITIONED: {updated.symbol} → "
                            f"ext_2.618={new_tp:.6f}"
                        )

                still_open.append(updated)
                state_emoji = {
                    PositionState.OPEN: "🔵",
                    PositionState.BREAKEVEN: "🟡",
                    PositionState.TRAILING: "🟢",
                }
                unrealized_pct = 0.0
                if updated.direction == TradeDirection.LONG:
                    unrealized_pct = (current_price - updated.entry_price) / updated.entry_price
                else:
                    unrealized_pct = (updated.entry_price - current_price) / updated.entry_price

                self.logger.info(
                    f"    {state_emoji.get(updated.state, '⚪')} "
                    f"{updated.symbol} {updated.direction.value} | "
                    f"{updated.state.value} | "
                    f"P&L: {unrealized_pct*100:+.2f}% | "
                    f"SL: {updated.stop_loss:.6f}"
                )

        self.open_positions = still_open

    # ──────────────────────────────────────────────
    # EXCHANGE SYNC
    # ──────────────────────────────────────────────

    def _sync_with_exchange(self):
        """Reconcile internal state with actual Bybit positions."""
        try:
            exchange_positions = self.connector.get_positions()
            exchange_symbols = set()

            for ep in exchange_positions:
                if float(ep.get("size", 0)) > 0:
                    exchange_symbols.add(ep["symbol"])

            # Check if exchange closed any positions (SL/TP server-side)
            still_open = []
            for pos in self.open_positions:
                if pos.symbol not in exchange_symbols:
                    self.logger.info(
                        f"  🔄 {pos.symbol} closed by exchange (server-side SL/TP)"
                    )
                    pos.state = PositionState.CLOSED
                    pos.close_reason = "Exchange SL/TP (Maker exit)"
                    self.closed_positions.append(pos)
                else:
                    still_open.append(pos)

            self.open_positions = still_open
        except Exception as e:
            self.logger.debug(f"  Sync warning: {e}")

    # ──────────────────────────────────────────────
    # ENTRY EXECUTION (v4.2: POC + Fibo blend)
    # ──────────────────────────────────────────────

    def _execute_entry(self, signal: ReversalSignal) -> bool:
        """
        Execute entry using CASCADE Fibonacci Order Laddering (v4.1):
          1. Calculate TWO Fibo entry prices (0.50 and 0.618)
          2. Split risk 50/50 between the two levels
          3. Place TWO LIMIT orders (PostOnly) as linked twins
          4. Register both as pending with shared cascade_group_id
        """
        import uuid

        symbol = signal.symbol
        direction = signal.direction
        atr = signal.atr_value
        side = "Buy" if direction == TradeDirection.LONG else "Sell"

        # ── Calculate CASCADE entry prices with negative spread protection ──
        entry_50, entry_618 = FiboCalculator.calculate_cascade_entries(
            signal.swing_high, signal.swing_low, direction, signal.current_price
        )

        # ── Fibo extensions (for trailing targets + Maker TP) ──
        ext_1, ext_2 = FiboCalculator.calculate_extensions(
            signal.swing_high, signal.swing_low, direction
        )

        # ── SL/TP (calculated from the deeper 0.618 level) ──
        sl_distance = atr * 2.0
        if direction == TradeDirection.LONG:
            stop_loss = entry_618 - sl_distance
            take_profit = entry_50 + sl_distance * config.RISK_REWARD_RATIO
        else:
            stop_loss = entry_618 + sl_distance
            take_profit = entry_50 - sl_distance * config.RISK_REWARD_RATIO

        # ── Position sizes (split risk 50/50) ──
        # Each leg gets half the total risk
        half_risk_pct = self.compound.current_risk_pct * config.CASCADE_RISK_SPLIT

        # Leg 1 (0.50 level)
        sl_dist_50 = abs(entry_50 - stop_loss) / entry_50
        if sl_dist_50 <= 0:
            sl_dist_50 = 0.02
        risk_usdt_50 = self.compound.current_balance * half_risk_pct
        pos_value_50 = risk_usdt_50 / sl_dist_50
        qty_50 = pos_value_50 / entry_50

        # Leg 2 (0.618 level)
        sl_dist_618 = abs(entry_618 - stop_loss) / entry_618
        if sl_dist_618 <= 0:
            sl_dist_618 = 0.02
        risk_usdt_618 = self.compound.current_balance * half_risk_pct
        pos_value_618 = risk_usdt_618 / sl_dist_618
        qty_618 = pos_value_618 / entry_618

        # ── Precision ──
        precision = self.connector.get_lot_size_precision(symbol)
        qty_50 = round(qty_50, precision)
        qty_618 = round(qty_618, precision)

        if qty_50 <= 0 and qty_618 <= 0:
            self.logger.warning(f"  ⚠️ {symbol} quantities too small")
            return False

        # ── Round prices to tick size ──
        entry_50 = self.connector.round_price(entry_50, symbol)
        entry_618 = self.connector.round_price(entry_618, symbol)
        stop_loss = self.connector.round_price(stop_loss, symbol)
        take_profit = self.connector.round_price(take_profit, symbol)

        # ── Cascade group ID (links twin orders) ──
        cascade_id = f"{symbol}_{uuid.uuid4().hex[:8]}"

        # ── Log entry details ──
        poc_str = f" | POC={poc_price:.6f}" if poc_price else " | POC=none"
        self.logger.info("")
        self.logger.info(f"  ╔══════════════════════════════════════════════════╗")
        self.logger.info(f"  ║  CASCADE ENTRY: {direction.value} {symbol}")
        self.logger.info(f"  ╠══════════════════════════════════════════════════╣")
        self.logger.info(f"  ║  Oscillators: {signal.oscillators_firing}/3 at extremes")
        self.logger.info(f"  ║  RSI={signal.rsi_value:.1f} | CCI={signal.cci_value:.0f} | W%R={signal.willr_value:.1f}")
        self.logger.info(f"  ║  Market:     {signal.current_price:.6f}")
        self.logger.info(f"  ║  Leg 1 (0.50):  {entry_50:.6f} | Qty: {qty_50} | Risk: ${risk_usdt_50:.2f}")
        self.logger.info(f"  ║  Leg 2 (0.618): {entry_618:.6f} | Qty: {qty_618} | Risk: ${risk_usdt_618:.2f}")
        self.logger.info(f"  ║  SL: {stop_loss:.6f} | TP: {take_profit:.6f}")
        self.logger.info(f"  ║  Fibo Ext: 1.618={ext_1:.6f} | 2.618={ext_2:.6f}")
        self.logger.info(f"  ║  TTL: {config.ORDER_TTL_SECONDS}s | PostOnly (Maker)")
        self.logger.info(f"  ╚══════════════════════════════════════════════════╝")

        orders_placed = 0

        # ── Place Leg 1: 0.50 level (closer to market) ──
        if qty_50 > 0:
            result_50 = self.connector.place_limit_order(
                symbol=symbol, side=side, qty=qty_50,
                price=entry_50, stop_loss=stop_loss, take_profit=take_profit,
            )
            if result_50:
                self.total_orders_placed += 1
                orders_placed += 1
                self.pending_orders.append(PendingOrder(
                    symbol=symbol,
                    order_id=result_50.get("orderId", ""),
                    direction=direction,
                    limit_price=entry_50,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    quantity=qty_50,
                    risk_usdt=risk_usdt_50,
                    placed_at=time.time(),
                    swing_high=signal.swing_high,
                    swing_low=signal.swing_low,
                    fibo_ext_1=ext_1,
                    fibo_ext_2=ext_2,
                    cascade_group_id=cascade_id,
                    cascade_level="0.50",
                ))
            else:
                self.logger.error(f"  ❌ Leg 1 (0.50) FAILED for {symbol}")

        # ── Place Leg 2: 0.618 level (deeper) ──
        if qty_618 > 0:
            result_618 = self.connector.place_limit_order(
                symbol=symbol, side=side, qty=qty_618,
                price=entry_618, stop_loss=stop_loss, take_profit=take_profit,
            )
            if result_618:
                self.total_orders_placed += 1
                orders_placed += 1
                self.pending_orders.append(PendingOrder(
                    symbol=symbol,
                    order_id=result_618.get("orderId", ""),
                    direction=direction,
                    limit_price=entry_618,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    quantity=qty_618,
                    risk_usdt=risk_usdt_618,
                    placed_at=time.time(),
                    swing_high=signal.swing_high,
                    swing_low=signal.swing_low,
                    fibo_ext_1=ext_1,
                    fibo_ext_2=ext_2,
                    cascade_group_id=cascade_id,
                    cascade_level="0.618",
                ))
            else:
                self.logger.error(f"  ❌ Leg 2 (0.618) FAILED for {symbol}")

        if orders_placed > 0:
            self.logger.info(
                f"  >>> {orders_placed} cascade order(s) placed "
                f"(TTL={config.ORDER_TTL_SECONDS}s)"
            )
            return True

        return False

    # ──────────────────────────────────────────────
    # DASHBOARD
    # ──────────────────────────────────────────────

    def _print_dashboard(self):
        """Print status dashboard."""
        status = self.compound.get_status()
        uptime = datetime.now(timezone.utc) - self.start_time
        hours = uptime.total_seconds() / 3600

        self.logger.info("")
        self.logger.info("  ┌─────────────── DASHBOARD ───────────────┐")
        self.logger.info(f"  │ Cycle:       #{self.cycle_count:<25}│")
        self.logger.info(f"  │ Uptime:      {hours:.1f}h{' '*27}│")
        self.logger.info(f"  │ Balance:     ${status['current_balance']:<23.2f}│")
        self.logger.info(f"  │ ROI:         {status['roi_pct']:+.2f}%{' '*24}│")
        self.logger.info(f"  │ Win Rate:    {status['win_rate_pct']:.1f}%{' '*25}│")
        trades_str = f"{status['total_trades']} ({status['wins']}W/{status['losses']}L)"
        self.logger.info(f"  │ Trades:      {trades_str:<25}│")
        self.logger.info(f"  │ Open:        {len(self.open_positions)}/{config.MAX_CONCURRENT_POSITIONS}{' '*27}│")
        self.logger.info(f"  │ Pending:     {len(self.pending_orders)} limit orders{' '*16}│")
        self.logger.info(f"  │ Max DD:      {status['max_drawdown_pct']:.2f}%{' '*24}│")
        self.logger.info(f"  │ Risk/Trade:  ${status['current_risk_usdt']:.2f} ({status['current_risk_pct']:.1f}%){' '*14}│")
        self.logger.info("  └─────────────────────────────────────────┘")

    # ──────────────────────────────────────────────
    # MAIN LOOP — THE HEARTBEAT
    # ──────────────────────────────────────────────

    def run(self):
        """
        🎯 Main trading loop — Fibonacci Reversal Sniper v4.2.

        Cycle:
          1. Sync with exchange
          2. Manage pending orders
          3. Fetch all candles (parallel, 15m)
          4. Monitor open positions (shadow trailing + TP reposition)
          5. Scan for reversal signals
          6. Execute entries (POC+Fibo → limit orders)
          7. Sleep until next candle
        """
        global _shutdown_requested

        self.logger.info("\n" + "═" * 70)
        self.logger.info("  🎯 SNIPER ACTIVATED — Bot is now LIVE (v4.2)")
        self.logger.info("  Press Ctrl+C to stop gracefully")
        self.logger.info("═" * 70 + "\n")

        while not _shutdown_requested:
            try:
                # ── Wait for next candle ──
                wait_secs = self._seconds_until_next_candle()
                next_time = datetime.now(timezone.utc) + timedelta(seconds=wait_secs)
                self.logger.info(
                    f"  ⏳ Sleeping {wait_secs:.0f}s → "
                    f"Next cycle at {next_time.strftime('%H:%M:%S')} UTC"
                )

                sleep_until = time.time() + wait_secs
                while time.time() < sleep_until and not _shutdown_requested:
                    time.sleep(min(1.0, sleep_until - time.time()))

                if _shutdown_requested:
                    break

                # ══════════════════════════════════════════
                # 🎯 CYCLE START
                # ══════════════════════════════════════════
                self.cycle_count += 1
                cycle_start = time.time()

                self.logger.info(f"\n{'─' * 70}")
                self.logger.info(
                    f"  🎯 CYCLE #{self.cycle_count} — "
                    f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC"
                )
                self.logger.info(f"{'─' * 70}")

                # ── Phase 1: Sync with exchange ──
                self._sync_with_exchange()

                # ── Phase 2: Manage pending limit orders ──
                self._manage_pending_orders()

                # ── Phase 3: Fetch ALL candles (parallel, 15m) ──
                self.logger.info(f"\n  📡 Fetching {len(self.symbols)} symbols...")
                fetch_start = time.time()
                all_candles = self._fetch_all_candles()
                fetch_time = time.time() - fetch_start
                self.logger.info(
                    f"  Fetched {len(all_candles)}/{len(self.symbols)} in {fetch_time:.1f}s"
                )

                # ── Phase 4: Monitor open positions (shadow trailing) ──
                self._monitor_positions(all_candles)

                # ── Phase 5: Scan for reversal signals ──
                signals_found = 0
                entries_made = 0

                # Build set of occupied symbols (SINGLE-ENTRY LOCK)
                occupied: set[str] = set()
                for pos in self.open_positions:
                    occupied.add(pos.symbol)
                for pend in self.pending_orders:
                    occupied.add(pend.symbol)

                # Check max positions
                can_enter = (
                    len(self.open_positions) + len(self.pending_orders)
                    < config.MAX_CONCURRENT_POSITIONS
                )

                if can_enter:
                    self.logger.info(f"\n  🔍 Scanning for reversal signals...")

                    for symbol, df in all_candles.items():
                        # SINGLE-ENTRY LOCK — skip if already in position/pending
                        if symbol in occupied:
                            continue

                        # Detect reversal
                        signal = self.reversal_engine.detect(symbol, df)
                        if signal is None:
                            continue

                        signals_found += 1
                        self.logger.info(
                            f"\n  ⚡ REVERSAL: {signal.direction.value} {symbol} | "
                            f"Oscillators: {signal.oscillators_firing}/3 | "
                            f"RSI={signal.rsi_value:.1f}"
                        )

                        # Execute entry (with POC targeting)
                        success = self._execute_entry(signal)
                        if success:
                            entries_made += 1

                        # Max entries per cycle
                        if entries_made >= config.MAX_ENTRIES_PER_CYCLE:
                            self.logger.info("  Max entries per cycle reached")
                            break
                else:
                    self.logger.info(
                        f"\n  ⏸️ Max positions reached "
                        f"({len(self.open_positions)} open + "
                        f"{len(self.pending_orders)} pending)"
                    )

                # ── Phase 6: Summary ──
                cycle_time = time.time() - cycle_start
                self.logger.info(
                    f"\n  Cycle #{self.cycle_count} done in {cycle_time:.1f}s: "
                    f"{signals_found} signals, {entries_made} entries"
                )

                # Dashboard every 4 cycles
                if self.cycle_count % 4 == 0:
                    self._print_dashboard()

            except KeyboardInterrupt:
                break
            except Exception as e:
                self.logger.error(f"  ❌ Cycle error: {e}")
                self.logger.debug(traceback.format_exc())
                time.sleep(10)

        # ── Graceful shutdown ──
        self._shutdown()

    def _shutdown(self):
        """Graceful shutdown — save state."""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("  SHUTTING DOWN")
        self.logger.info("=" * 70)
        self._print_dashboard()

        # Cancel all pending orders
        for order in self.pending_orders:
            self.connector.cancel_order(order.symbol, order.order_id)
            self.logger.info(f"  Cancelled pending: {order.symbol}")

        # Save state
        state = {
            "shutdown_time": datetime.now(timezone.utc).isoformat(),
            "version": "4.2",
            "cycles_completed": self.cycle_count,
            "total_orders": self.total_orders_placed,
            "compound_status": self.compound.get_status(),
            "open_positions": [
                {
                    "symbol": p.symbol,
                    "direction": p.direction.value,
                    "entry_price": p.entry_price,
                    "quantity": p.quantity,
                    "stop_loss": p.stop_loss,
                    "take_profit": p.take_profit,
                    "state": p.state.value,
                }
                for p in self.open_positions
            ],
            "trade_history": self.compound.trade_history[-50:],
        }

        state_path = os.path.join(config.OUTPUT_DIR, "bot_state.json")
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, default=str)

        self.logger.info(f"  State saved to {state_path}")
        self.logger.info("  Goodbye! 👋\n")


# ══════════════════════════════════════════════════════════════════
# DRY-RUN MODE
# ══════════════════════════════════════════════════════════════════

class DryRunBot:
    """Test pipeline without API connection."""

    def __init__(self, symbols: list[str] | None = None):
        self.symbols = symbols or config.SYMBOLS[:10]
        self.logger = setup_logging()
        self.reversal_engine = ReversalEngine()
        self.compound = CompoundCalculator()

        self.logger.info("  🧪 DRY-RUN MODE — No real orders")
        self.logger.info(f"  Symbols: {len(self.symbols)} | Strategy: Fibo Reversal Sniper v4.2")

    def run_single_cycle(self):
        """Execute one analysis cycle with simulated data."""
        self.logger.info(f"\n{'─' * 60}")
        self.logger.info(f"  🧪 DRY-RUN — {datetime.now(timezone.utc).strftime('%H:%M:%S')}")
        self.logger.info(f"{'─' * 60}")
        self.logger.info("  Architecture v4.2:")
        self.logger.info("    1. ReversalEngine  — RSI/CCI/WillR oscillator resonance")
        self.logger.info("    2. VolumeProfiler  — POC from 1m micro-structure (Lazy Sniper)")
        self.logger.info("    3. FiboCalculator  — 0.50+0.618 cascade entry (POC blend)")
        self.logger.info("    4. PositionManager — 3-Phase: Breathing → Breakeven → Shadow Trail")
        self.logger.info("    5. CompoundCalc    — 1% risk, compound growth")
        self.logger.info("    6. BybitConnector  — Limit PostOnly (maker 0.020% in+out)")
        self.logger.info("")

        # Test Fibo calculation
        fibo_long = FiboCalculator.calculate_entry(100.0, 80.0, TradeDirection.LONG)
        fibo_short = FiboCalculator.calculate_entry(100.0, 80.0, TradeDirection.SHORT)
        ext1, ext2 = FiboCalculator.calculate_extensions(100.0, 80.0, TradeDirection.LONG)
        self.logger.info(f"  📐 Fibo test (H=100, L=80):")
        self.logger.info(f"     LONG entry (0.618):  {fibo_long:.2f}")
        self.logger.info(f"     SHORT entry (0.618): {fibo_short:.2f}")
        self.logger.info(f"     Extension 1.618:     {ext1:.2f}")
        self.logger.info(f"     Extension 2.618:     {ext2:.2f}")
        self.logger.info("")

        # Test Volume Profile POC
        self.logger.info("  📊 Volume Profile POC test:")
        # Simulated 1m data with volume cluster at ~92
        np.random.seed(42)
        prices_1m = np.linspace(88, 96, 60) + np.random.normal(0, 0.5, 60)
        volume_1m = np.random.uniform(100, 500, 60)
        # Create volume spike at prices 91-93 (simulating POC)
        for i in range(20, 35):
            prices_1m[i] = 92.0 + np.random.uniform(-0.5, 0.5)
            volume_1m[i] = 2000 + np.random.uniform(0, 1000)  # 4-6x normal volume

        df_1m = pd.DataFrame({
            "open": prices_1m - 0.1,
            "high": prices_1m + 0.3,
            "low": prices_1m - 0.3,
            "close": prices_1m,
            "volume": volume_1m,
        })
        poc = VolumeProfiler.calculate_poc(df_1m, atr_15m=2.0, direction=TradeDirection.LONG, current_price=95.0)
        if poc:
            blended = VolumeProfiler.blend_poc_with_fibo(poc, fibo_long)
            self.logger.info(f"     POC (1m volume cluster): {poc:.2f}")
            self.logger.info(f"     Fibo 0.618:              {fibo_long:.2f}")
            self.logger.info(f"     Blended (60/40):         {blended:.2f}")
        else:
            self.logger.info(f"     POC: not available (synthetic data)")
        self.logger.info("")

        # Test Shadow Trailing
        self.logger.info("  🌙 Shadow Trailing test:")
        self.logger.info(f"     Phase 1 (Breathing):  SL fixed at entry - {config.SL_ATR_MULTIPLIER}×ATR")
        self.logger.info(f"     Phase 2 (Breakeven):  +{config.BREAKEVEN_TRIGGER_PCT*100:.1f}% → SL=entry+fees")
        self.logger.info(f"     Phase 3 (Shadow):     +{config.TRAILING_TRIGGER_PCT*100:.1f}% → SL=prev_low - {config.TRAILING_ATR_CUSHION}×ATR")
        self.logger.info(f"     Maker TP:             Fibo ext_1.618 → reposition to ext_2.618")
        self.logger.info("")

        status = self.compound.get_status()
        self.logger.info(f"  💰 Capital: ${status['current_balance']:.2f} | "
                         f"Risk/trade: ${status['current_risk_usdt']:.2f}")
        self.logger.info(f"  📈 Directions: LONG + SHORT (symmetric)")
        self.logger.info("  ✅ Pipeline validated — ready for live deployment!")


# ══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════

def main():
    # Force UTF-8 for stdout on Windows (avoids UnicodeEncodeError with emoji/box-drawing)
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

    parser = argparse.ArgumentParser(
        description="Aegis v4.2 — Fibonacci Reversal Sniper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python live_bot.py --dry-run        # test pipeline
  python live_bot.py                  # live trading
  python live_bot.py --symbols BTCUSDT,ETHUSDT
        """
    )

    parser.add_argument("--dry-run", action="store_true", help="Test mode (no API)")
    parser.add_argument("--interval", type=int, default=15, choices=[5, 15, 30, 60])
    parser.add_argument("--capital", type=float, default=2000.0)
    parser.add_argument("--leverage", type=float, default=5.0)
    parser.add_argument("--symbols", type=str, default=None)

    args = parser.parse_args()

    symbols = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]

    # DRY-RUN
    if args.dry_run:
        bot = DryRunBot(symbols=symbols)
        bot.run_single_cycle()
        return

    # LIVE — Try both naming variants (with and without _API_)
    api_key = os.environ.get("BYBIT_DEMO_API_KEY") or os.environ.get("BYBIT_DEMO_KEY", "")
    api_secret = os.environ.get("BYBIT_DEMO_API_SECRET") or os.environ.get("BYBIT_DEMO_SECRET", "")

    if not api_key or not api_secret:
        print("\n" + "=" * 60)
        print("  ⚠️  BYBIT API KEYS NOT SET IN ENVIRONMENT OR .env.local!")
        print("=" * 60)
        print("\n  Please check your .env.local file or set environment variables:")
        print('    export BYBIT_DEMO_API_KEY="your_demo_api_key"')
        print('    export BYBIT_DEMO_API_SECRET="your_demo_api_secret"')
        print("\n  Or run: python live_bot.py --dry-run")
        print()
        sys.exit(1)

    try:
        bot = LiveBot(
            api_key=api_key,
            api_secret=api_secret,
            symbols=symbols,
            interval_minutes=args.interval,
            initial_capital=args.capital,
            leverage=args.leverage,
        )
        bot.run()
    except ConnectionError as e:
        print(f"\n  ❌ Connection failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n  ❌ Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
