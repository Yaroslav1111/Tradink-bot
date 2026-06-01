"""
v5.0 — Backtest Runner (Time Machine)
══════════════════════════════════════════════
Bar-by-bar replay with honest fills and indicator warm-up.

Key features:
  - Indicator warm-up: feeds N bars before test start
  - Honest fills: via SimBroker (candle high/low crossing)
  - Same Strategy code as live — zero modifications
  - Fast: months of data in seconds
  - Symbol-specific configuration from optimized_params.json
"""
from __future__ import annotations

import json
import logging
import time as time_module
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from engine.models import (
    Direction, Position, PositionPhase, Signal, AccountState,
)
from engine.strategy import FiboReversalStrategy, StrategyConfig
from broker.sim_broker import SimBroker

logger = logging.getLogger("aegis.runner.backtest")


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""
    symbol: str = "BTCUSDT"
    initial_balance: float = 2000.0
    leverage: float = 5.0
    candle_interval: str = "15"
    warmup_bars: int = 100        # bars fed before test starts
    maker_fee: float = 0.0002
    taker_fee: float = 0.00055


class BacktestRunner:
    """
    Bar-by-bar backtest engine.

    Uses the SAME FiboReversalStrategy as live trading.
    SimBroker handles fills with honest candle logic.
    Supports per-symbol optimized config from optimized_params.json.
    """

    def __init__(
        self,
        strategy_cfg: StrategyConfig,
        backtest_cfg: BacktestConfig,
        optimized_params: Optional[dict[str, dict]] = None,
        exchange_rules: Optional[dict[str, dict]] = None,
    ):
        self.strategy_cfg = strategy_cfg
        self.bt_cfg = backtest_cfg

        # If optimized params exist for this symbol, apply overrides
        if optimized_params and backtest_cfg.symbol in optimized_params:
            overrides = optimized_params[backtest_cfg.symbol]
            base_dict = asdict(strategy_cfg)
            base_dict.update(overrides)
            self.strategy_cfg = StrategyConfig(**base_dict)
            logger.info(
                f"📋 Using optimized params for {backtest_cfg.symbol}: "
                f"{list(overrides.keys())}"
            )

        # Create components
        self.broker = SimBroker(
            initial_balance=backtest_cfg.initial_balance,
            leverage=backtest_cfg.leverage,
            maker_fee=backtest_cfg.maker_fee,
            taker_fee=backtest_cfg.taker_fee,
        )
        self.strategy = FiboReversalStrategy(
            cfg=self.strategy_cfg,
            initial_balance=backtest_cfg.initial_balance,
            exchange_rules=exchange_rules,
        )

    def run(
        self,
        df_15m: pd.DataFrame,
        df_1m: Optional[pd.DataFrame] = None,
    ) -> dict:
        """
        Execute backtest on provided data.

        Args:
            df_15m: 15-minute OHLCV DataFrame (columns: timestamp, open, high, low, close, volume)
            df_1m: Optional 1-minute data for POC calculations

        Returns:
            dict with performance metrics
        """
        start_time = time_module.time()
        symbol = self.bt_cfg.symbol

        # Reset broker state
        self.broker.reset()

        # Load data into sim broker
        self.broker.load_data(symbol, df_15m, self.bt_cfg.candle_interval)
        if df_1m is not None:
            self.broker.load_1m_data(symbol, df_1m)

        total_bars = len(df_15m)
        warmup = min(self.bt_cfg.warmup_bars, total_bars - 1)

        logger.info(
            f"📊 Backtest: {symbol} | {total_bars} bars | "
            f"Warmup: {warmup} | Balance: {self.bt_cfg.initial_balance}"
        )

        # ─── Bar-by-bar replay ───
        for bar_idx in range(total_bars):
            # Update broker time and bar position
            bar = df_15m.iloc[bar_idx]
            bar_time = float(bar.get("timestamp", bar_idx * 900))
            self.broker.set_time(bar_time)

            # Process fills FIRST (honest: uses this bar's high/low)
            self.broker.process_bar(symbol, bar_idx)

            # Skip signal generation during warm-up
            if bar_idx < warmup:
                continue

            # Get data visible to strategy (up to current bar)
            df_visible = self.broker.get_klines(symbol, self.bt_cfg.candle_interval, 200)
            if df_visible is None or len(df_visible) < 100:
                continue

            current_price = float(bar["close"])

            # ─── Manage existing positions ───
            self._manage_positions(symbol, df_visible, current_price, bar)

            # ─── Check order TTL ───
            self._check_order_ttl()

            # ─── Single-entry lock ───
            positions = self.broker.get_positions()
            pending = self.broker.get_pending_orders()
            has_position = any(p.symbol == symbol for p in positions)
            has_order = any(o.symbol == symbol for o in pending)

            if has_position or has_order:
                continue
            if len(positions) >= self.strategy_cfg.max_concurrent:
                continue

            # ─── Detect new signal ───
            direction = self.strategy.detect_signal(symbol, df_visible, current_price)
            if direction is None:
                continue

            # Build signal
            account = self.broker.get_account()
            df_1m_visible = self.broker.get_1m_klines(symbol, self.strategy_cfg.poc_1m_lookback)
            signal = self.strategy.build_signal(
                symbol, direction, df_visible, df_1m_visible, current_price, account
            )
            if signal is None:
                continue

            # Execute signal (place orders)
            self._execute_signal(signal)

        # ─── Close remaining positions at last price ───
        last_price = float(df_15m.iloc[-1]["close"])
        for pos in self.broker.get_positions():
            close_side = "Sell" if pos.direction == Direction.LONG else "Buy"
            self.broker.close_position(pos.symbol, pos.quantity, close_side)

        # Cancel remaining orders
        for order in self.broker.get_pending_orders():
            self.broker.cancel_order(order.symbol, order.order_id)

        # Record all trades to strategy compound calc
        for trade in self.broker.closed_trades:
            self.strategy.compound.record(trade.pnl_usdt)

        elapsed = time_module.time() - start_time
        results = self.broker.get_results_summary()
        results["elapsed_seconds"] = elapsed
        results["bars_processed"] = total_bars - warmup
        results["config"] = self._config_to_dict()

        logger.info(
            f"✅ Backtest done in {elapsed:.1f}s | "
            f"Trades: {results['total_trades']} | "
            f"PnL: {results['total_pnl']:+.2f} | "
            f"Win rate: {results['win_rate']:.1f}% | "
            f"MaxDD: {results['max_drawdown_pct']:.1f}%"
        )

        return results

    # ─────────────────── Position Management ───────────────────

    def _manage_positions(self, symbol: str, df: pd.DataFrame, current_price: float, bar):
        """Run 3-phase management on all open positions."""
        for pos in self.broker.positions[:]:
            if pos.symbol != symbol or pos.phase == PositionPhase.CLOSED:
                continue

            # Previous candle for trailing
            prev_low = float(df.iloc[-2]["low"]) if len(df) >= 2 else 0
            prev_high = float(df.iloc[-2]["high"]) if len(df) >= 2 else 0

            # Check reversal
            reversal = self.strategy.check_reversal_against(pos, df)

            # Check trend invalidation (SuperTrend emergency exit)
            trend_invalidated = self.strategy.check_trend_invalidation(pos, df)

            # Update position
            old_sl = pos.stop_loss
            old_tp = pos.take_profit
            pos = self.strategy.update_position(
                pos, current_price, prev_low, prev_high, reversal, trend_invalidated
            )

            # If strategy says close (reversal exit)
            if pos.phase == PositionPhase.CLOSED:
                close_side = "Sell" if pos.direction == Direction.LONG else "Buy"
                self.broker.close_position(pos.symbol, pos.quantity, close_side)
                continue

            # TP Repositioning
            if self.strategy.should_reposition_tp(pos, current_price):
                pos.take_profit = pos.fibo_ext_2
                pos.tp_repositioned = True

            # Sync SL/TP to broker
            if abs(pos.stop_loss - old_sl) > 0.0001 or abs(pos.take_profit - old_tp) > 0.0001:
                self.broker.update_position_sl_tp(
                    pos.symbol,
                    stop_loss=pos.stop_loss,
                    take_profit=pos.take_profit,
                )

    def _check_order_ttl(self):
        """Cancel expired orders."""
        now = self.broker.now()
        for order in self.broker.pending_orders[:]:
            if now - order.placed_at > self.strategy_cfg.order_ttl_seconds:
                self.broker.cancel_order(order.symbol, order.order_id)

    # ─────────────────── Signal Execution ───────────────────

    def _execute_signal(self, signal: Signal):
        """Place cascade orders from signal."""
        side = "Buy" if signal.direction == Direction.LONG else "Sell"

        import uuid
        cascade_id = str(uuid.uuid4())[:8]

        # Leg 1: Fibo 0.50
        self.broker.place_limit_order(
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

        # Leg 2: Fibo 0.618
        self.broker.place_limit_order(
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

    # ─────────────────── Helpers ───────────────────

    def _config_to_dict(self) -> dict:
        """Serialize config for results output."""
        from dataclasses import asdict
        return {
            "strategy": asdict(self.strategy_cfg),
            "backtest": {
                "symbol": self.bt_cfg.symbol,
                "initial_balance": self.bt_cfg.initial_balance,
                "leverage": self.bt_cfg.leverage,
                "warmup_bars": self.bt_cfg.warmup_bars,
            },
        }

    @staticmethod
    def load_optimized_params(path: str = "optimized_params.json") -> dict[str, dict]:
        """
        Load optimized params from JSON file.
        Returns: {"BTCUSDT": {"fibo_primary": 0.786, ...}, ...}
        """
        p = Path(path)
        if not p.exists():
            return {}
        with open(p, "r") as f:
            return json.load(f)
