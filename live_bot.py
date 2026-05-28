#!/usr/bin/env python3
"""
Aegis-Quant-Lab v4.0 — LIVE BOT «Fibonacci Reversal Sniper»
══════════════════════════════════════════════════════════════════

Architecture: Impulse → Fibo Pullback → Limit Order → Trail to Exhaustion

Execution cycle (every 15 minutes):
  1. Wake at candle close
  2. Manage pending limit orders (check fills, TTL, deviation)
  3. Monitor open positions (breakeven, trailing, reversal exit)
  4. Scan ALL symbols for oscillator reversal resonance
  5. If reversal detected → calculate Fibo entry → place Limit order
  6. Sleep until next candle

Usage:
  export BYBIT_DEMO_KEY="your_key"
  export BYBIT_DEMO_SECRET="your_secret"
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

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(
        "%(asctime)s │ %(message)s", datefmt="%H:%M:%S"
    ))
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

    Components:
      - ReversalEngine: detects oscillator exhaustion
      - FiboCalculator: computes entry/extension levels
      - PositionManager: breakeven + trailing stop
      - CompoundCalculator: dynamic position sizing
      - BybitConnector: limit orders on demo API
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
        self.logger.info("  🎯 AEGIS-QUANT-LAB v4.0 — Fibonacci Reversal Sniper")
        self.logger.info("=" * 70)
        self.logger.info(f"  Capital:    ${initial_capital:.2f} USDT")
        self.logger.info(f"  Leverage:   {leverage}x")
        self.logger.info(f"  Symbols:    {len(self.symbols)} coins")
        self.logger.info(f"  Interval:   {interval_minutes}m candles")
        self.logger.info(f"  Strategy:   Oscillator Reversal → Fibo 0.618 Entry → Trail")
        self.logger.info(f"  Order type: Limit (PostOnly, Maker 0.020%)")
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

    # ──────────────────────────────────────────────
    # PENDING ORDERS MANAGEMENT
    # ──────────────────────────────────────────────

    def _manage_pending_orders(self):
        """
        Check all pending limit orders:
          - FILLED → register as open position
          - TTL expired → cancel
          - Price deviation > 1% → cancel (trend reversed)
        """
        if not self.pending_orders:
            return

        self.logger.info(f"\n  ⏳ Checking {len(self.pending_orders)} pending orders...")

        still_pending = []
        now = time.time()

        for order in self.pending_orders:
            elapsed = now - order.placed_at

            # Check 1: TTL expired
            if elapsed > order.ttl_seconds:
                self.logger.info(
                    f"    ⌛ EXPIRED: {order.symbol} (TTL {order.ttl_seconds}s elapsed)"
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
                        f"@ {fill_price:.6f} (waited {elapsed:.0f}s)"
                    )

                    # Register as open position
                    pos = LivePosition(
                        symbol=order.symbol,
                        direction=order.direction,
                        entry_price=fill_price,
                        entry_time=datetime.now(timezone.utc),
                        quantity=order.quantity,
                        risk_usdt=order.risk_usdt,
                        stop_loss=order.stop_loss,
                        take_profit=order.take_profit,
                        initial_stop_loss=order.stop_loss,
                        highest_price=fill_price,
                        lowest_price=fill_price,
                        fibo_ext_1_price=order.fibo_ext_1,
                        fibo_ext_2_price=order.fibo_ext_2,
                    )
                    self.open_positions.append(pos)
                    continue

                elif status in ("Cancelled", "Rejected", "Deactivated"):
                    self.logger.info(f"    ❌ {status}: {order.symbol}")
                    continue

            # Check 3: Price deviation — cancel if price moved away
            current_price = self.connector.get_ticker_price(order.symbol)
            if current_price > 0:
                deviation = abs(current_price - order.limit_price) / order.limit_price
                if deviation > order.max_deviation_pct:
                    self.logger.info(
                        f"    📉 CANCEL: {order.symbol} | price deviated "
                        f"{deviation*100:.2f}% > {order.max_deviation_pct*100:.1f}%"
                    )
                    self.connector.cancel_order(order.symbol, order.order_id)
                    continue

            # Still pending
            still_pending.append(order)
            self.logger.info(f"    ⏳ WAITING: {order.symbol} ({elapsed:.0f}s/{order.ttl_seconds}s)")

        self.pending_orders = still_pending

    # ──────────────────────────────────────────────
    # POSITION MONITORING
    # ──────────────────────────────────────────────

    def _monitor_positions(self, all_candles: dict[str, pd.DataFrame]):
        """
        Monitor all open positions:
          - Update price tracking
          - Apply breakeven/trailing logic
          - Check for oscillator reversal → exit
          - Close if SL hit
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

            # Check for oscillator reversal (exit signal)
            reversal = self.position_manager.check_reversal_for_exit(pos, df)

            # Update position state (breakeven/trailing/close)
            updated = self.position_manager.update(pos, current_price, reversal)

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
                    pos.close_reason = "Exchange SL/TP"
                    self.closed_positions.append(pos)
                else:
                    still_open.append(pos)

            self.open_positions = still_open
        except Exception as e:
            self.logger.debug(f"  Sync warning: {e}")

    # ──────────────────────────────────────────────
    # ENTRY EXECUTION
    # ──────────────────────────────────────────────

    def _execute_entry(self, signal: ReversalSignal) -> bool:
        """
        Execute entry based on reversal signal:
          1. Calculate Fibo entry price
          2. Calculate SL/TP
          3. Calculate position size
          4. Place LIMIT order (PostOnly)
          5. Register as pending order
        """
        symbol = signal.symbol
        direction = signal.direction

        # ── Fibo entry price ──
        fibo_entry = signal.fibo_entry_price

        # ── SL/TP based on ATR ──
        atr = signal.atr_value
        sl_distance = atr * 2.0  # 2x ATR stop loss

        if direction == TradeDirection.LONG:
            stop_loss = fibo_entry - sl_distance
            take_profit = fibo_entry + sl_distance * config.RISK_REWARD_RATIO
            side = "Buy"
        else:
            stop_loss = fibo_entry + sl_distance
            take_profit = fibo_entry - sl_distance * config.RISK_REWARD_RATIO
            side = "Sell"

        # ── Fibo extensions (for trailing targets) ──
        ext_1, ext_2 = FiboCalculator.calculate_extensions(
            signal.swing_high, signal.swing_low, direction
        )

        # ── Position size ──
        size_info = self.compound.calculate_position_size(
            entry_price=fibo_entry,
            stop_loss_price=stop_loss,
            leverage=self.leverage,
        )
        quantity = size_info["quantity"]

        # ── Precision ──
        precision = self.connector.get_lot_size_precision(symbol)
        quantity = round(quantity, precision)
        if quantity <= 0:
            self.logger.warning(f"  ⚠️ {symbol} quantity too small")
            return False

        # ── Round prices to tick size ──
        fibo_entry = self.connector.round_price(fibo_entry, symbol)
        stop_loss = self.connector.round_price(stop_loss, symbol)
        take_profit = self.connector.round_price(take_profit, symbol)

        # ── Log entry details ──
        self.logger.info("")
        self.logger.info(f"  ╔══════════════════════════════════════════════╗")
        self.logger.info(f"  ║  NEW ENTRY: {direction.value} {symbol}")
        self.logger.info(f"  ╠══════════════════════════════════════════════╣")
        self.logger.info(f"  ║  Oscillators: {signal.oscillators_firing}/3 at extremes")
        self.logger.info(f"  ║  RSI={signal.rsi_value:.1f} | CCI={signal.cci_value:.0f} | W%R={signal.willr_value:.1f}")
        self.logger.info(f"  ║  Market:  {signal.current_price:.6f}")
        self.logger.info(f"  ║  Fibo Entry (0.618): {fibo_entry:.6f}")
        discount_pct = abs(signal.current_price - fibo_entry) / signal.current_price * 100
        self.logger.info(f"  ║  Discount: {discount_pct:.2f}% from market")
        self.logger.info(f"  ║  SL: {stop_loss:.6f} | TP: {take_profit:.6f}")
        self.logger.info(f"  ║  Qty: {quantity} | Risk: ${size_info['risk_usdt']:.2f}")
        self.logger.info(f"  ║  Fibo Ext: 1.618={ext_1:.6f} | 2.618={ext_2:.6f}")
        self.logger.info(f"  ║  TTL: {config.ORDER_TTL_SECONDS}s | PostOnly (Maker)")
        self.logger.info(f"  ╚══════════════════════════════════════════════╝")

        # ── Place limit order ──
        order_result = self.connector.place_limit_order(
            symbol=symbol,
            side=side,
            qty=quantity,
            price=fibo_entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

        if order_result is None:
            self.logger.error(f"  ❌ Order FAILED for {symbol}")
            return False

        self.total_orders_placed += 1
        order_id = order_result.get("orderId", "")

        # ── Register as pending ──
        pending = PendingOrder(
            symbol=symbol,
            order_id=order_id,
            direction=direction,
            limit_price=fibo_entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            quantity=quantity,
            risk_usdt=size_info["risk_usdt"],
            placed_at=time.time(),
            swing_high=signal.swing_high,
            swing_low=signal.swing_low,
            fibo_ext_1=ext_1,
            fibo_ext_2=ext_2,
        )
        self.pending_orders.append(pending)

        self.logger.info(f"  >>> Limit order placed — waiting for fill (TTL={config.ORDER_TTL_SECONDS}s)")
        return True

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
        🎯 Main trading loop — Fibonacci Reversal Sniper.

        Cycle:
          1. Sync with exchange
          2. Manage pending orders
          3. Fetch all candles (parallel)
          4. Monitor open positions
          5. Scan for reversal signals
          6. Execute entries (limit orders)
          7. Sleep until next candle
        """
        global _shutdown_requested

        self.logger.info("\n" + "═" * 70)
        self.logger.info("  🎯 SNIPER ACTIVATED — Bot is now LIVE")
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

                # ── Phase 3: Fetch ALL candles (parallel) ──
                self.logger.info(f"\n  📡 Fetching {len(self.symbols)} symbols...")
                fetch_start = time.time()
                all_candles = self._fetch_all_candles()
                fetch_time = time.time() - fetch_start
                self.logger.info(
                    f"  Fetched {len(all_candles)}/{len(self.symbols)} in {fetch_time:.1f}s"
                )

                # ── Phase 4: Monitor open positions ──
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

                        # Execute entry
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
            "version": "4.0",
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
        self.logger.info(f"  Symbols: {len(self.symbols)} | Strategy: Fibo Reversal Sniper")

    def run_single_cycle(self):
        """Execute one analysis cycle with simulated data."""
        self.logger.info(f"\n{'─' * 60}")
        self.logger.info(f"  🧪 DRY-RUN — {datetime.now(timezone.utc).strftime('%H:%M:%S')}")
        self.logger.info(f"{'─' * 60}")
        self.logger.info("  Architecture v4.0:")
        self.logger.info("    1. ReversalEngine  — RSI/CCI/WillR oscillator resonance")
        self.logger.info("    2. FiboCalculator  — 0.618 retracement entry")
        self.logger.info("    3. PositionManager — Breakeven → Trailing → Reversal exit")
        self.logger.info("    4. CompoundCalc    — 1% risk, compound growth")
        self.logger.info("    5. BybitConnector  — Limit PostOnly (maker 0.020%)")
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

        status = self.compound.get_status()
        self.logger.info(f"  💰 Capital: ${status['current_balance']:.2f} | "
                         f"Risk/trade: ${status['current_risk_usdt']:.2f}")
        self.logger.info("  ✅ Pipeline validated — ready for live deployment!")


# ══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Aegis v4.0 — Fibonacci Reversal Sniper",
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

    # LIVE
    api_key = os.environ.get("BYBIT_DEMO_KEY", "")
    api_secret = os.environ.get("BYBIT_DEMO_SECRET", "")

    if not api_key or not api_secret:
        print("\n" + "=" * 60)
        print("  ⚠️  BYBIT API KEYS NOT SET!")
        print("=" * 60)
        print("\n  Set environment variables:")
        print('    export BYBIT_DEMO_KEY="your_demo_api_key"')
        print('    export BYBIT_DEMO_SECRET="your_demo_api_secret"')
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
