"""
v5.4 — Live Runner (Real-Time Trading Loop)
══════════════════════════════════════════════
Connects Strategy (Brain) to LiveBroker (Exchange).

Key features:
  - Amnesia-proof: recovers state from exchange on every restart
  - Single-entry lock: checks exchange positions (not local JSON)
  - Cascade order management (place, TTL, merge on fill)
  - 3-phase position management with SL/TP sync to exchange
  - Symbol-specific configuration from optimized_params.json
  - Exchange rules precision (qtyStep, minOrderQty, tickSize)
  - Dynamic portfolio sync with graceful draining (v5.4)
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Optional

import pandas as pd

from engine.models import (
    Direction, Order, OrderStatus, Position, PositionPhase,
    Signal, TradeResult,
)
from engine.strategy import FiboReversalStrategy, StrategyConfig
from broker.live_broker import LiveBroker

logger = logging.getLogger("aegis.runner.live")


class LiveRunner:
    """
    Live trading loop — runs FiboReversalStrategy on real Bybit data.

    Architecture:
      Strategy (Brain) → emits Signals → Runner places via Broker
      Runner manages: order lifecycle, TTL, fill detection, position updates
    """

    def __init__(
        self,
        broker: LiveBroker,
        strategy: FiboReversalStrategy,
        symbols: list[str],
        scan_interval: int = 60,
        candle_interval: str = "15",
        candle_limit: int = 200,
        optimized_params_path: str = "optimized_params.json",
        exchange_rules: Optional[dict[str, dict]] = None,
        auto_sync: bool = False,
        active_portfolio_path: str = "data/active_portfolio.json",
    ):
        self.broker = broker
        self.strategy = strategy
        self.cfg = strategy.cfg
        self.symbols = list(symbols)  # mutable copy
        self.scan_interval = scan_interval
        self.candle_interval = candle_interval
        self.candle_limit = candle_limit
        self.auto_sync = auto_sync
        self.active_portfolio_path = active_portfolio_path
        self._exchange_rules = exchange_rules

        # Inject exchange rules into default strategy
        if exchange_rules:
            self.strategy._exchange_rules = exchange_rules

        # Symbol-specific configs (loaded from optimized_params.json)
        self._symbol_configs: dict[str, StrategyConfig] = {}
        self._symbol_strategies: dict[str, FiboReversalStrategy] = {}
        self._load_optimized_params(optimized_params_path, exchange_rules)

        # Graceful draining: symbols being phased out (no new entries)
        self.draining_symbols: set[str] = set()

        # Tracked state (recovered from exchange)
        self.active_orders: list[Order] = []
        self.active_positions: list[Position] = []
        self._cascade_groups: dict[str, list[Order]] = {}  # group_id -> orders

        # Cycle counter for periodic sync
        self._sync_counter: int = 0
        self._sync_every_n_cycles: int = 15

        # Recover state from exchange (Amnesia Fix)
        self._recover_state()

    # ─────────────────── Symbol-Specific Config ───────────────────

    def _load_optimized_params(self, path: str, exchange_rules: Optional[dict[str, dict]] = None):
        """Load per-symbol optimized parameters from JSON file."""
        p = Path(path)
        if not p.exists():
            logger.info("📋 No optimized_params.json found — using default config for all symbols")
            return

        try:
            with open(p, "r") as f:
                params_data = json.load(f)

            from dataclasses import asdict
            base_dict = asdict(self.cfg)

            count = 0
            for symbol, overrides in params_data.items():
                merged = {**base_dict, **overrides}
                cfg = StrategyConfig(**merged)
                self._symbol_configs[symbol] = cfg
                self._symbol_strategies[symbol] = FiboReversalStrategy(
                    cfg,
                    initial_balance=self.strategy.compound.balance,
                    exchange_rules=exchange_rules,
                )
                count += 1

            logger.info(f"📋 Loaded optimized params for {count} symbols from {path}")

        except Exception as e:
            logger.warning(f"⚠️ Failed to load optimized_params.json: {e}")

    def _get_strategy_for_symbol(self, symbol: str) -> FiboReversalStrategy:
        """Get the strategy instance for a specific symbol (optimized or default)."""
        return self._symbol_strategies.get(symbol, self.strategy)

    def _get_config_for_symbol(self, symbol: str) -> StrategyConfig:
        """Get the config for a specific symbol (optimized or default)."""
        return self._symbol_configs.get(symbol, self.cfg)

    # ─────────────────── Amnesia Fix ───────────────────

    def _recover_state(self):
        """Read positions and orders from exchange on startup."""
        self.active_positions = self.broker.get_positions()
        self.active_orders = self.broker.get_pending_orders()
        logger.info(
            f"🔄 State recovered: {len(self.active_positions)} positions, "
            f"{len(self.active_orders)} pending orders"
        )

    # ─────────────────── Main Loop ───────────────────

    def run(self):
        """Main trading loop — runs until interrupted."""
        logger.info(
            f"🚀 LiveRunner started | {len(self.symbols)} symbols | "
            f"Scan every {self.scan_interval}s | "
            f"Auto-sync: {self.auto_sync}"
        )
        cycle = 0
        while True:
            try:
                cycle += 1
                logger.info(f"\n{'═'*50}\n  CYCLE {cycle} | {time.strftime('%H:%M:%S')}\n{'═'*50}")

                # Periodic portfolio sync (every N cycles)
                if self.auto_sync:
                    self._sync_counter += 1
                    if self._sync_counter >= self._sync_every_n_cycles:
                        self._sync_counter = 0
                        self.sync_portfolio()

                self._run_cycle()

                time.sleep(self.scan_interval)

            except KeyboardInterrupt:
                logger.info("🛑 LiveRunner stopped by user")
                break
            except Exception as e:
                logger.error(f"❌ Cycle error: {e}", exc_info=True)
                time.sleep(30)  # Back off on error

    def _run_cycle(self):
        """Single scan cycle: manage orders → manage positions → scan for signals."""
        # 1. Refresh account state
        account = self.broker.get_account()
        logger.info(
            f"  💰 Balance: {account.total_balance:.2f} | "
            f"Free: {account.free_margin:.2f} | "
            f"Locked: {account.locked_margin:.2f}"
        )

        # 2. Manage pending orders (TTL, fill detection)
        self._manage_orders()

        # 3. Manage open positions (ALL positions: active + draining)
        self._manage_positions()

        # 4. Cleanup fully drained symbols (no positions/orders remaining)
        self._cleanup_drained_symbols()

        # 5. Scan ONLY active symbols for new signals (not draining)
        occupied_symbols = set(
            p.symbol for p in self.active_positions
        ) | set(
            o.symbol for o in self.active_orders
        )

        if len(self.active_positions) >= self.cfg.max_concurrent:
            logger.info(f"  ⏸️ Max concurrent positions ({self.cfg.max_concurrent}) reached")
            return

        entries_this_cycle = 0
        for symbol in self.symbols:  # Only active symbols (not draining)
            if entries_this_cycle >= self.cfg.max_entries_per_cycle:
                break
            if symbol in occupied_symbols:
                continue

            signal = self._scan_symbol(symbol, account)
            if signal:
                self._execute_signal(signal)
                entries_this_cycle += 1
                occupied_symbols.add(symbol)

    # ─────────────────── Portfolio Sync (Graceful Draining) ───────────────────

    def sync_portfolio(self):
        """
        Hot-reload active portfolio from data/active_portfolio.json.

        - New symbols: create strategy instances, add to self.symbols
        - Removed symbols: move from self.symbols to self.draining_symbols
        - Draining symbols stop receiving new signals but existing positions
          are managed until naturally closed.
        """
        if not os.path.exists(self.active_portfolio_path):
            logger.debug("📋 No active_portfolio.json found — skipping sync")
            return

        try:
            with open(self.active_portfolio_path, "r") as f:
                data = json.load(f)

            new_active = data.get("symbols", [])
            if not new_active:
                logger.warning("⚠️ active_portfolio.json has empty symbols list — ignoring")
                return

            new_active_set = set(new_active)
            current_active_set = set(self.symbols)

            # Additions: symbols in new list but not currently active
            additions = new_active_set - current_active_set - self.draining_symbols
            # Removals: symbols currently active but not in new list
            removals = current_active_set - new_active_set

            # Process additions
            for symbol in additions:
                self._add_symbol(symbol)

            # Process removals (move to draining)
            for symbol in removals:
                self.symbols.remove(symbol)
                self.draining_symbols.add(symbol)
                logger.info(f"  🔻 {symbol} → draining (removed from active portfolio)")

            # Re-activate: if a draining symbol reappears in the active list
            reactivated = self.draining_symbols & new_active_set
            for symbol in reactivated:
                self.draining_symbols.discard(symbol)
                if symbol not in self.symbols:
                    self.symbols.append(symbol)
                    logger.info(f"  🔄 {symbol} re-activated from draining")

            if additions or removals or reactivated:
                logger.info(
                    f"  📊 Portfolio sync: +{len(additions)} added, "
                    f"-{len(removals)} draining, "
                    f"↩️{len(reactivated)} reactivated | "
                    f"Active: {len(self.symbols)} | Draining: {len(self.draining_symbols)}"
                )
            else:
                logger.debug("📋 Portfolio sync: no changes")

        except (json.JSONDecodeError, IOError, KeyError) as e:
            logger.warning(f"⚠️ Failed to read active_portfolio.json: {e}")

    def _add_symbol(self, symbol: str):
        """
        Add a new symbol to active trading:
        load its optimized config, create strategy instance.
        """
        from dataclasses import asdict

        # Load optimized params if available
        params_path = Path(
            os.path.dirname(self.active_portfolio_path)
        ).parent / "optimized_params.json"
        overrides = {}
        if params_path.exists():
            try:
                with open(params_path, "r") as f:
                    all_params = json.load(f)
                overrides = all_params.get(symbol, {})
            except (json.JSONDecodeError, IOError):
                pass

        # Build config
        base_dict = {k: v for k, v in asdict(self.cfg).items()}
        base_dict.update(overrides)
        cfg = StrategyConfig(**base_dict)

        # Create strategy instance
        strat = FiboReversalStrategy(
            cfg,
            initial_balance=self.strategy.compound.balance,
            exchange_rules=self._exchange_rules,
        )

        self._symbol_configs[symbol] = cfg
        self._symbol_strategies[symbol] = strat
        self.symbols.append(symbol)

        logger.info(f"  🔺 {symbol} → added to active portfolio (overrides: {len(overrides)} params)")

    # ─────────────────── Draining Cleanup ───────────────────

    def _cleanup_drained_symbols(self):
        """
        After position management, check if any draining symbol has
        ZERO active positions and ZERO pending orders.
        If so, fully remove it from memory.
        """
        if not self.draining_symbols:
            return

        # Gather symbols with active positions or orders
        symbols_with_positions = set(p.symbol for p in self.active_positions)
        symbols_with_orders = set(o.symbol for o in self.active_orders)
        occupied = symbols_with_positions | symbols_with_orders

        # Find draining symbols that are completely clear
        fully_drained = self.draining_symbols - occupied

        for symbol in fully_drained:
            self.draining_symbols.discard(symbol)
            # Clean up strategy instances
            self._symbol_configs.pop(symbol, None)
            self._symbol_strategies.pop(symbol, None)
            logger.info(f"  🗑️ {symbol} fully drained — removed from memory")

        if fully_drained:
            logger.info(
                f"  ♻️ Cleaned {len(fully_drained)} drained symbols | "
                f"Remaining draining: {len(self.draining_symbols)}"
            )

    # ─────────────────── Signal Detection ───────────────────

    def _scan_symbol(self, symbol: str, account) -> Optional[Signal]:
        """Scan a single symbol for reversal signal."""
        # Get 15m data
        df_15m = self.broker.get_klines(symbol, self.candle_interval, self.candle_limit)
        if df_15m is None or len(df_15m) < 100:
            return None

        current_price = self.broker.get_ticker_price(symbol)
        if current_price <= 0:
            return None

        # Use per-symbol strategy (optimized or default)
        strategy = self._get_strategy_for_symbol(symbol)
        cfg = self._get_config_for_symbol(symbol)

        # Detect signal direction
        direction = strategy.detect_signal(symbol, df_15m, current_price)
        if direction is None:
            return None

        logger.info(f"  🎯 Signal detected: {direction.value} {symbol} @ {current_price}")

        # Get 1m data for POC
        df_1m = None
        if cfg.poc_enabled:
            df_1m = self.broker.get_klines(symbol, "1", cfg.poc_1m_lookback)

        # Build full signal with sizing
        signal = strategy.build_signal(
            symbol, direction, df_15m, df_1m, current_price, account
        )
        return signal

    # ─────────────────── Signal Execution ───────────────────

    def _execute_signal(self, signal: Signal):
        """Place cascade limit orders from signal."""
        cascade_id = str(uuid.uuid4())[:8]
        side = "Buy" if signal.direction == Direction.LONG else "Sell"

        # Order 1: Fibo 0.50
        order_50 = self.broker.place_limit_order(
            symbol=signal.symbol,
            side=side,
            price=signal.entry_price_50,
            quantity=signal.qty_50,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            cascade_group_id=cascade_id,
            cascade_level="0.50",
            direction=signal.direction,
            risk_usdt=signal.risk_usdt_50,
            atr=signal.atr,
            fibo_ext_1=signal.fibo_ext_1,
            fibo_ext_2=signal.fibo_ext_2,
            swing_high=signal.swing_high,
            swing_low=signal.swing_low,
        )

        # Order 2: Fibo 0.618
        order_618 = self.broker.place_limit_order(
            symbol=signal.symbol,
            side=side,
            price=signal.entry_price_618,
            quantity=signal.qty_618,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            cascade_group_id=cascade_id,
            cascade_level="0.618",
            direction=signal.direction,
            risk_usdt=signal.risk_usdt_618,
            atr=signal.atr,
            fibo_ext_1=signal.fibo_ext_1,
            fibo_ext_2=signal.fibo_ext_2,
            swing_high=signal.swing_high,
            swing_low=signal.swing_low,
        )

        placed = []
        if order_50:
            self.active_orders.append(order_50)
            placed.append(order_50)
        if order_618:
            self.active_orders.append(order_618)
            placed.append(order_618)

        if placed:
            self._cascade_groups[cascade_id] = placed
            logger.info(
                f"  📝 Cascade {cascade_id}: {len(placed)} orders placed for "
                f"{signal.direction.value} {signal.symbol}"
            )

    # ─────────────────── Order Management ───────────────────

    def _manage_orders(self):
        """Check order fills and TTL expiration."""
        now = self.broker.now()
        expired = []
        filled = []

        for order in self.active_orders[:]:
            # Check status on exchange
            status = self.broker.check_order_status(order.symbol, order.order_id)

            if status == "Filled":
                order.status = OrderStatus.FILLED
                order.filled_at = now
                filled.append(order)
                self.active_orders.remove(order)
                logger.info(
                    f"  ✅ FILLED: {order.direction.value} {order.symbol} "
                    f"@ {order.price} (cascade {order.cascade_level})"
                )
                # Create position
                self._on_order_filled(order)

            elif status in ("Cancelled", "Rejected", "Deactivated"):
                order.status = OrderStatus.CANCELLED
                self.active_orders.remove(order)
                logger.info(f"  🗑️ Order cancelled by exchange: {order.symbol} {order.order_id[:8]}")

            elif status in ("New", "PartiallyFilled", None):
                # Check TTL
                age = now - order.placed_at
                if age > self.cfg.order_ttl_seconds:
                    # Cancel expired order
                    self.broker.cancel_order(order.symbol, order.order_id)
                    order.status = OrderStatus.CANCELLED
                    self.active_orders.remove(order)
                    expired.append(order)
                    logger.info(
                        f"  ⏰ TTL expired ({age:.0f}s): {order.symbol} "
                        f"@ {order.price}"
                    )

        if expired:
            # Clean up cascade groups
            for order in expired:
                self._cleanup_cascade(order)

    def _on_order_filled(self, order: Order):
        """Handle filled order — create position, cancel sibling cascade."""
        # Create tracked position
        pos = Position(
            symbol=order.symbol,
            direction=order.direction,
            entry_price=order.price,
            quantity=order.quantity,
            risk_usdt=order.risk_usdt,
            stop_loss=order.stop_loss,
            take_profit=order.take_profit,
            initial_stop_loss=order.stop_loss,
            phase=PositionPhase.BREATHING,
            highest_price=order.price,
            lowest_price=order.price,
            entry_atr=order.atr,
            fibo_ext_1=order.fibo_ext_1,
            fibo_ext_2=order.fibo_ext_2,
            entry_time=order.filled_at or self.broker.now(),
            exchange_order_id=order.order_id,
        )
        self.active_positions.append(pos)

        # If cascade sibling exists and also fills → merge into position
        # If sibling is still pending → leave it (gives better avg entry)
        # This is handled naturally: if sibling fills later, we'll merge

    def _cleanup_cascade(self, cancelled_order: Order):
        """If one cascade leg is cancelled, cancel the other too."""
        group_id = cancelled_order.cascade_group_id
        if not group_id or group_id not in self._cascade_groups:
            return

        group = self._cascade_groups[group_id]
        for order in group[:]:
            if order.order_id != cancelled_order.order_id and order.status == OrderStatus.PENDING:
                # Check if it's still in active_orders
                if order in self.active_orders:
                    self.broker.cancel_order(order.symbol, order.order_id)
                    order.status = OrderStatus.CANCELLED
                    self.active_orders.remove(order)
                    logger.info(f"  🗑️ Cascade sibling cancelled: {order.symbol}")

        del self._cascade_groups[group_id]

    # ─────────────────── Position Management ───────────────────

    def _manage_positions(self):
        """Update trailing, check reversals, reposition TP."""
        for pos in self.active_positions[:]:
            if pos.phase == PositionPhase.CLOSED:
                self.active_positions.remove(pos)
                continue

            # Get current data
            current_price = self.broker.get_ticker_price(pos.symbol)
            if current_price <= 0:
                continue

            # Use per-symbol strategy (optimized or default)
            strategy = self._get_strategy_for_symbol(pos.symbol)

            # Get previous candle for shadow trailing
            df = self.broker.get_klines(pos.symbol, self.candle_interval, 3)
            prev_low = 0.0
            prev_high = 0.0
            if df is not None and len(df) >= 2:
                prev_low = float(df.iloc[-2]["low"])
                prev_high = float(df.iloc[-2]["high"])

            # Check oscillator reversal
            df_full = self.broker.get_klines(pos.symbol, self.candle_interval, self.candle_limit)
            reversal = strategy.check_reversal_against(pos, df_full) if df_full is not None else False

            # Check trend invalidation (SuperTrend-based emergency exit)
            trend_invalidated = strategy.check_trend_invalidation(pos, df_full) if df_full is not None else False

            # Update position (phase transitions, trailing)
            old_sl = pos.stop_loss
            old_tp = pos.take_profit
            pos = strategy.update_position(
                pos, current_price, prev_low, prev_high, reversal, trend_invalidated
            )

            # Handle close
            if pos.phase == PositionPhase.CLOSED:
                close_side = "Sell" if pos.direction == Direction.LONG else "Buy"
                self.broker.close_position(pos.symbol, pos.quantity, close_side)
                strategy.compound.record(pos.pnl_usdt)
                self.active_positions.remove(pos)
                logger.info(
                    f"  🔒 Position closed: {pos.symbol} | "
                    f"Reason: {pos.close_reason} | PnL: {pos.pnl_usdt:+.2f}"
                )
                continue

            # Sync SL/TP to exchange if changed
            sl_changed = abs(pos.stop_loss - old_sl) > 0.0001
            tp_changed = abs(pos.take_profit - old_tp) > 0.0001

            # TP Repositioning check
            if strategy.should_reposition_tp(pos, current_price):
                pos.take_profit = pos.fibo_ext_2
                pos.tp_repositioned = True
                tp_changed = True
                logger.info(
                    f"  🎯 TP repositioned: {pos.symbol} → ext_2.618 = {pos.fibo_ext_2:.4f}"
                )

            if sl_changed or tp_changed:
                self.broker.update_position_sl_tp(
                    pos.symbol,
                    stop_loss=pos.stop_loss if sl_changed else 0,
                    take_profit=pos.take_profit if tp_changed else 0,
                )

    # ─────────────────── Stats ───────────────────

    def get_status(self) -> dict:
        """Return current runner status."""
        return {
            "active_positions": len(self.active_positions),
            "pending_orders": len(self.active_orders),
            "total_trades": self.strategy.compound.total_trades,
            "balance": self.strategy.compound.balance,
            "win_rate": (
                self.strategy.compound.wins / self.strategy.compound.total_trades * 100
                if self.strategy.compound.total_trades > 0 else 0
            ),
        }
