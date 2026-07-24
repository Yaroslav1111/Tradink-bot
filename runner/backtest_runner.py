"""
v5.5 — Backtest Runner (Time Machine + Two-Winged Dual-Lot Exit)
════════════════════════════════════════════════════════════════
Bar-by-bar replay with honest fills and indicator warm-up.

Key features:
  - Indicator warm-up: feeds N bars before test start
  - Honest fills: via SimBroker (candle high/low crossing)
  - Same ENTRY Strategy code as live — zero modifications
  - v5.5 dual-lot dynamic exit: Lot A (maker fix) + Lot B (momentum float)
  - v5.5 race-safe state machine + breakeven protection cascade
  - v5.5 Trade Capsule assembly for LLM meta-analysis
  - Fast: months of data in seconds
"""
from __future__ import annotations

import json
import logging
import time as time_module
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from engine.models import (
    Direction, Position, PositionPhase, Signal, AccountState,
    LotRole, LotState, TradeCapsule, ExecutionReality,
)
from engine.strategy import FiboReversalStrategy, StrategyConfig
from engine.trade_capsule import (
    CapsuleLogger, build_pre_trade, build_internal_thoughts, build_post_trade,
    capsule_to_json,
)
from broker.sim_broker import SimBroker

logger = logging.getLogger("aegis.runner.backtest")


@dataclass
class DualLotTrade:
    """
    Runtime tracker linking Lot A (maker fix) + Lot B (momentum float)
    that were split from a single filled entry, plus the in-progress
    Trade Capsule for this cycle.
    """
    capsule_id: str
    symbol: str
    direction: Direction
    entry_price: float
    entry_time: float
    lot_a_id: str = ""            # exchange_order_id of Lot A position
    lot_b_id: str = ""            # exchange_order_id of Lot B position
    lot_a_state: LotState = LotState.ACTIVE
    lot_b_state: LotState = LotState.ACTIVE
    maker_tp_price: float = 0.0
    structural_stop: float = 0.0
    pivot_level: float = 0.0
    cascade_applied: bool = False
    # capsule pillars (assembled progressively)
    capsule: Optional[TradeCapsule] = None
    lot_a_pnl: float = 0.0
    lot_b_pnl: float = 0.0
    close_reason_a: str = ""
    close_reason_b: str = ""
    lot_b_exit_price: float = 0.0
    lot_b_exit_time: float = 0.0
    finalized: bool = False


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
        capsule_logger: Optional[CapsuleLogger] = None,
        enable_capsules: bool = True,
    ):
        self.strategy_cfg = strategy_cfg
        self.bt_cfg = backtest_cfg

        # v5.5 dual-lot + capsule state
        self.enable_capsules = enable_capsules
        self.capsule_logger = capsule_logger or (
            CapsuleLogger(async_mode=False) if enable_capsules else None
        )
        self._dual_trades: list[DualLotTrade] = []          # active dual-lot trackers
        self._pending_capsules: dict[str, DualLotTrade] = {}  # awaiting post-trade audit
        self.capsules: list[TradeCapsule] = []              # finalized capsules (this run)
        # per-signal context captured at entry, keyed by cascade_group_id
        self._entry_context: dict[str, dict] = {}

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

            # v5.5: split any freshly-filled entry into Lot A + Lot B
            if self.strategy_cfg.dual_lot_enabled:
                self._split_new_fills(symbol, bar_time)

            # v5.5: run post-trade audit on capsules awaiting lookahead candles
            self._audit_pending_capsules(df_15m, bar_idx)

            # Skip signal generation during warm-up
            if bar_idx < warmup:
                continue

            # Get data visible to strategy (up to current bar)
            df_visible = self.broker.get_klines(symbol, self.bt_cfg.candle_interval, 200)
            if df_visible is None or len(df_visible) < 100:
                continue

            current_price = float(bar["close"])

            # ─── Manage existing positions ───
            if self.strategy_cfg.dual_lot_enabled:
                self._manage_dual_lots(symbol, df_visible, current_price, bar, bar_idx)
            else:
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

            # Execute signal (place orders) — pass context for capsule assembly
            self._execute_signal(signal, df_visible, current_price, bar_time)

        # ─── Close remaining positions at last price ───
        last_price = float(df_15m.iloc[-1]["close"])
        last_time = float(df_15m.iloc[-1].get("timestamp", 0.0))
        for pos in self.broker.get_positions():
            close_side = "Sell" if pos.direction == Direction.LONG else "Buy"
            self.broker.close_position(pos.symbol, pos.quantity, close_side)

        # v5.5: reconcile + finalize any dual-lot trades still open at EOD
        if self.strategy_cfg.dual_lot_enabled:
            self._reconcile_dual_lots(last_time, force=True)
            # flush remaining pending capsules with whatever lookahead exists
            self._audit_pending_capsules(df_15m, total_bars - 1, force=True)

        # Cancel remaining orders
        for order in self.broker.get_pending_orders():
            self.broker.cancel_order(order.symbol, order.order_id)

        # Record all trades to strategy compound calc
        for trade in self.broker.closed_trades:
            self.strategy.compound.record(trade.pnl_usdt)

        # Flush capsule logger (non-blocking writer)
        if self.capsule_logger is not None:
            self.capsule_logger.flush()

        elapsed = time_module.time() - start_time
        results = self.broker.get_results_summary()
        results["elapsed_seconds"] = elapsed
        results["bars_processed"] = total_bars - warmup
        results["config"] = self._config_to_dict()
        # v5.5 dual-lot exit metrics
        results.update(self._dual_lot_metrics())
        results["capsules_generated"] = len(self.capsules)

        logger.info(
            f"✅ Backtest done in {elapsed:.1f}s | "
            f"Trades: {results['total_trades']} | "
            f"PnL: {results['total_pnl']:+.2f} | "
            f"Win rate: {results['win_rate']:.1f}% | "
            f"MaxDD: {results['max_drawdown_pct']:.1f}% | "
            f"Capsules: {len(self.capsules)}"
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

    # ══════════════════════════════════════════════════════════════
    # v5.5 — TWO-WINGED DUAL-LOT ORCHESTRATION
    # ══════════════════════════════════════════════════════════════

    def _split_new_fills(self, symbol: str, bar_time: float):
        """
        Detect freshly-filled entry positions and split each 50/50 into:
          - Lot A (Maker Fix): tight limit-maker TP for guaranteed +1.5–2.0% net.
          - Lot B (Momentum Float): NO static TP; closes on momentum decay.

        The single filled sim-position becomes Lot A; a second sibling
        position (Lot B) is spawned at the same entry. The structural
        invalidation stop replaces any arbitrary ATR stop on both lots.
        """
        tracked_ids = {t.lot_a_id for t in self._dual_trades} | {t.lot_b_id for t in self._dual_trades}
        for pos in self.broker.positions[:]:
            if pos.symbol != symbol or pos.phase == PositionPhase.CLOSED:
                continue
            if pos.lot_role is not None or pos.exchange_order_id in tracked_ids:
                continue  # already split / already a lot

            ctx = self._entry_context.get(pos.capsule_id)
            signal = ctx["signal"] if ctx else None

            # ── Structural invalidation stop (tight, behind the pivot) ──
            if pos.direction == Direction.LONG:
                pivot = signal.swing_low if signal else pos.stop_loss
            else:
                pivot = signal.swing_high if signal else pos.stop_loss
            atr = pos.entry_atr if pos.entry_atr > 0 else pos.entry_price * 0.01
            if self.strategy_cfg.structural_stop_enabled:
                structural_stop = self.strategy.compute_structural_stop(
                    pos.direction, pivot, atr
                )
            else:
                structural_stop = pos.stop_loss

            # ── 50/50 size split (honor lot precision) ──
            precision = self.broker.get_lot_precision(symbol)
            half = round(pos.quantity * self.strategy_cfg.dual_lot_split, precision)
            if half <= 0:
                half = pos.quantity  # too small to split → single lot behaves as Lot B
            lot_b_qty = round(pos.quantity - half, precision)

            # ── Lot A (maker fix): reuse this position, set tight maker TP ──
            maker_tp = self.strategy.compute_maker_fix_tp(pos.direction, pos.entry_price)
            maker_tp = self.broker.round_price(maker_tp, symbol)
            pos.lot_role = LotRole.MAKER_FIX
            pos.lot_state = LotState.PENDING       # maker TP resting on the book
            pos.quantity = half
            pos.stop_loss = structural_stop
            pos.initial_stop_loss = structural_stop
            pos.take_profit = maker_tp             # honest sim: fills when price crosses
            pos.maker_tp_price = maker_tp
            pos.structural_stop = structural_stop
            pos.pivot_level = pivot

            lot_a_id = pos.exchange_order_id
            lot_b_id = ""

            # ── Lot B (momentum float): spawn sibling with NO static TP ──
            # Lot B floats with extra breathing room so momentum-decay (RSI→50),
            # not a tight structural stop, is the deciding exit.
            if self.strategy_cfg.structural_stop_enabled:
                float_stop = self.strategy.compute_float_stop(pos.direction, pivot, atr)
            else:
                float_stop = pos.stop_loss
            if lot_b_qty > 0:
                lot_b = Position(
                    symbol=pos.symbol,
                    direction=pos.direction,
                    entry_price=pos.entry_price,
                    quantity=lot_b_qty,
                    risk_usdt=pos.risk_usdt * 0.5,
                    stop_loss=float_stop,
                    take_profit=0.0,               # NO static target
                    initial_stop_loss=float_stop,
                    phase=PositionPhase.BREATHING,
                    highest_price=pos.entry_price,
                    lowest_price=pos.entry_price,
                    entry_atr=atr,
                    entry_time=bar_time,
                    exchange_order_id=str(uuid.uuid4())[:12],
                    capsule_id=pos.capsule_id,
                    lot_role=LotRole.MOMENTUM_FLOAT,
                    lot_state=LotState.ACTIVE,
                    structural_stop=float_stop,
                    pivot_level=pivot,
                )
                self.broker.positions.append(lot_b)
                lot_b_id = lot_b.exchange_order_id

            # ── Build capsule (pillars 1-2) ──
            capsule = None
            if self.enable_capsules and signal is not None:
                capsule = TradeCapsule(
                    capsule_id=pos.capsule_id,
                    symbol=symbol,
                    direction=pos.direction.value,
                    pre_trade=build_pre_trade(
                        symbol, pos.direction,
                        ctx.get("df_snapshot"), ctx.get("current_price", pos.entry_price),
                        ctx.get("entry_time", bar_time),
                    ),
                    internal_thoughts=build_internal_thoughts(signal, pivot, structural_stop),
                    execution_reality=ExecutionReality(
                        lot_a_intended_price=pos.entry_price,
                        lot_a_fill_price=pos.entry_price,
                        lot_a_fill_time=pos.entry_time,
                        lot_b_intended_price=pos.entry_price,
                        lot_b_fill_price=pos.entry_price,
                        lot_b_fill_time=bar_time,
                        partial_fill=(lot_b_qty <= 0),
                        filled_size_ratio=1.0,
                    ),
                )

            self._dual_trades.append(DualLotTrade(
                capsule_id=pos.capsule_id,
                symbol=symbol,
                direction=pos.direction,
                entry_price=pos.entry_price,
                entry_time=pos.entry_time,
                lot_a_id=lot_a_id,
                lot_b_id=lot_b_id,
                lot_a_state=LotState.PENDING,
                lot_b_state=(LotState.ACTIVE if lot_b_id else LotState.CLOSED),
                maker_tp_price=maker_tp,
                structural_stop=structural_stop,
                pivot_level=pivot,
                capsule=capsule,
            ))

    def _manage_dual_lots(self, symbol: str, df: pd.DataFrame,
                          current_price: float, bar, bar_idx: int):
        """
        Drive the Two-Winged exit engine for every active dual-lot trade.

        Order of operations each bar (race-safe):
          1. Detect Lot A maker-TP fill  → protection cascade (snap Lot B to BE+).
          2. Evaluate Lot B momentum decay / trend invalidation.
          3. If Lot B exits BEFORE Lot A fills → cancel Lot A resting maker,
             verify status, then market-close remaining size.
        """
        reversal_trend = self.strategy.check_trend_invalidation
        momentum_dead_fn = self.strategy.check_momentum_decay

        for trade in self._dual_trades[:]:
            if trade.symbol != symbol or trade.finalized:
                continue

            lot_a = self.broker.get_position_by_id(trade.lot_a_id) if trade.lot_a_id else None
            lot_b = self.broker.get_position_by_id(trade.lot_b_id) if trade.lot_b_id else None

            # ── (1) Lot A fill detection → PROTECTION CASCADE ──
            if trade.lot_a_state == LotState.PENDING and lot_a is None:
                # Lot A no longer open → it filled its maker TP (or closed).
                trade.lot_a_state = LotState.FILLED
                if lot_b is not None and not trade.cascade_applied:
                    # Dynamic size sync before mutating (prevents ErrCode 10001).
                    live_size = self.broker.get_filled_size(trade.lot_b_id)
                    if live_size > 0:
                        lot_b = self.strategy.apply_breakeven_cascade(lot_b)
                        self.broker.update_position_sl_tp_by_id(
                            trade.lot_b_id, stop_loss=lot_b.stop_loss, take_profit=0.0,
                        )
                        trade.cascade_applied = True
            elif lot_a is not None and trade.lot_a_state == LotState.PENDING:
                # still resting — keep structural stop synced (size-safe)
                pass

            # ── (2 & 3) Manage Lot B (momentum float) ──
            if lot_b is not None:
                momentum_dead = momentum_dead_fn(lot_b, df)
                # Trend invalidation is an EMERGENCY backstop only — the entry is
                # a deliberate counter-trend reversal, so it must NOT fire on the
                # opening bars. Require the lot to have breathed into profit first.
                if lot_b.direction == Direction.LONG:
                    unr = (current_price - lot_b.entry_price) / lot_b.entry_price
                else:
                    unr = (lot_b.entry_price - current_price) / lot_b.entry_price
                trend_invalidated = reversal_trend(lot_b, df) if unr > 0 else False

                a_still_pending = (trade.lot_a_state == LotState.PENDING
                                   and self.broker.get_position_by_id(trade.lot_a_id) is not None)

                lot_b = self.strategy.update_momentum_float(
                    lot_b, current_price,
                    momentum_dead=momentum_dead,
                    trend_invalidated=trend_invalidated,
                )

                if lot_b.phase == PositionPhase.CLOSED:
                    # Lot B decided to exit.
                    # RACE CONDITION: if Lot A's maker is still resting, we must
                    # Cancel(A) → Verify → Market-close remaining, atomically.
                    if a_still_pending:
                        self._resolve_race_exit(trade, lot_b, current_price)
                    else:
                        # Lot A already filled — just realize Lot B at market.
                        self.broker.close_position_by_id(
                            trade.lot_b_id, current_price, lot_b.close_reason,
                        )
                    trade.lot_b_state = LotState.CLOSED
                else:
                    # keep structural / snapped stop synced to broker (size-safe)
                    self.broker.update_position_sl_tp_by_id(
                        trade.lot_b_id, stop_loss=lot_b.stop_loss, take_profit=0.0,
                    )

            # ── Finalize when both lots are done ──
            a_open = self.broker.get_position_by_id(trade.lot_a_id) is not None
            b_open = (trade.lot_b_id != "" and
                      self.broker.get_position_by_id(trade.lot_b_id) is not None)
            if not a_open and not b_open:
                self._finalize_dual_trade(trade, bar_idx)

    def _resolve_race_exit(self, trade: "DualLotTrade", lot_b: Position, price: float):
        """
        Bulletproof race handler: Lot B momentum-exit fired while Lot A's
        limit maker is still resting on the book.

        Pipeline:  Cancel Order (Lot A) → Verify Status → Market Close (remaining)
        """
        # Mark A as cancelling
        trade.lot_a_state = LotState.CANCELLING
        lot_a = self.broker.get_position_by_id(trade.lot_a_id)

        if lot_a is not None:
            # (a) Cancel the resting maker TP so it can't fill mid-flight.
            #     In sim, the maker TP is modeled as pos.take_profit; clearing
            #     it is equivalent to cancelling the resting limit order.
            self.broker.update_position_sl_tp_by_id(
                trade.lot_a_id, stop_loss=lot_a.stop_loss, take_profit=0.0,
            )
            # (b) Verify status via dynamic size lookup (no phantom size).
            live_size = self.broker.get_filled_size(trade.lot_a_id)
            # (c) Market-close the remaining size of Lot A at market.
            if live_size > 0:
                self.broker.close_position_by_id(
                    trade.lot_a_id, price, "RACE_CANCEL_MARKET_CLOSE",
                )
            trade.lot_a_state = LotState.CLOSED
            if trade.capsule and trade.capsule.execution_reality:
                trade.capsule.execution_reality.race_condition_triggered = True

        # Now realize Lot B itself (its close was decided by update_momentum_float).
        self.broker.close_position_by_id(trade.lot_b_id, price, lot_b.close_reason)

    def _reconcile_dual_lots(self, now_time: float, force: bool = False):
        """Finalize any dual-lot trades whose lots have all closed (EOD sweep)."""
        for trade in self._dual_trades[:]:
            if trade.finalized:
                continue
            a_open = trade.lot_a_id and self.broker.get_position_by_id(trade.lot_a_id) is not None
            b_open = trade.lot_b_id and self.broker.get_position_by_id(trade.lot_b_id) is not None
            if force or (not a_open and not b_open):
                self._finalize_dual_trade(trade, bar_idx=None)

    def _finalize_dual_trade(self, trade: "DualLotTrade", bar_idx: Optional[int]):
        """
        Both lots closed → gather realized PnL from broker.closed_trades,
        stamp the capsule execution reality, and queue it for the
        post-trade lookahead audit.
        """
        if trade.finalized:
            return
        trade.finalized = True

        # Look up realized PnL by lot id from the broker's per-lot close registry.
        a_pnl, a_reason = self._lookup_realized(trade.lot_a_id)
        b_pnl, b_reason, b_exit_price, b_exit_time = self._lookup_realized_full(trade.lot_b_id)
        trade.lot_a_pnl = a_pnl
        trade.lot_b_pnl = b_pnl
        trade.close_reason_a = a_reason
        trade.close_reason_b = b_reason
        trade.lot_b_exit_price = b_exit_price or trade.entry_price
        trade.lot_b_exit_time = b_exit_time

        if trade.capsule is not None:
            trade.capsule.lot_a_pnl_usdt = round(a_pnl, 6)
            trade.capsule.lot_b_pnl_usdt = round(b_pnl, 6)
            trade.capsule.total_pnl_usdt = round(a_pnl + b_pnl, 6)
            trade.capsule.close_reason_a = a_reason
            trade.capsule.close_reason_b = b_reason
            # queue for post-trade audit (needs future candles)
            self._pending_capsules[trade.capsule_id] = trade

        self._dual_trades.remove(trade)

    def _lookup_realized(self, lot_id: str) -> tuple[float, str]:
        pnl, reason, _, _ = self._lookup_realized_full(lot_id)
        return pnl, reason

    def _lookup_realized_full(self, lot_id: str) -> tuple[float, str, float, float]:
        """Find realized PnL for a lot by scanning the sim's position registry."""
        if not lot_id:
            return 0.0, "", 0.0, 0.0
        # Closed sim positions are removed from broker.positions but their
        # TradeResult is appended; match the most recent one for this lot via
        # the parallel _closed_by_id map maintained on the broker.
        rec = getattr(self.broker, "_closed_by_id", {}).get(lot_id)
        if rec:
            return rec["pnl"], rec["reason"], rec["exit_price"], rec["exit_time"]
        return 0.0, "", 0.0, 0.0

    def _audit_pending_capsules(self, df_15m: pd.DataFrame, bar_idx: int, force: bool = False):
        """
        Pillar 4 — Post-Trade Reality audit. For each capsule awaiting
        lookahead, once 10–15 candles exist beyond the exit (or on force),
        build the post-trade audit, log the capsule, and clear it.
        """
        if not self._pending_capsules:
            return
        LOOKAHEAD = 15
        for cid, trade in list(self._pending_capsules.items()):
            # find index of the exit bar within df_15m
            exit_time = trade.lot_b_exit_time or trade.entry_time
            # locate first bar strictly after exit_time
            ts = df_15m["timestamp"].values if "timestamp" in df_15m.columns else None
            if ts is not None and len(ts):
                after_idx = int(np.searchsorted(ts, exit_time, side="right"))
            else:
                after_idx = bar_idx + 1
            available = bar_idx - after_idx + 1
            if not force and available < LOOKAHEAD:
                continue  # wait for more candles

            end = min(after_idx + LOOKAHEAD, len(df_15m))
            df_after = df_15m.iloc[after_idx:end] if after_idx < len(df_15m) else None
            if trade.capsule is not None:
                trade.capsule.post_trade = build_post_trade(
                    trade.direction, trade.lot_b_exit_price,
                    trade.lot_b_exit_time, df_after,
                )
                self.capsules.append(trade.capsule)
                if self.capsule_logger is not None:
                    self.capsule_logger.log(trade.capsule)
            del self._pending_capsules[cid]

    def _dual_lot_metrics(self) -> dict:
        """Aggregate exit-performance metrics from finalized capsules."""
        if not self.capsules:
            return {
                "dual_lot_trades": 0,
                "lot_a_total_pnl": 0.0,
                "lot_b_total_pnl": 0.0,
                "maker_fix_win_rate": 0.0,
                "momentum_exit_count": 0,
                "race_conditions": 0,
                "premature_exits": 0,
                "max_impulse_catches": 0,
            }
        lot_a = sum(c.lot_a_pnl_usdt for c in self.capsules)
        lot_b = sum(c.lot_b_pnl_usdt for c in self.capsules)
        a_wins = sum(1 for c in self.capsules if c.lot_a_pnl_usdt > 0)
        momentum = sum(1 for c in self.capsules if c.close_reason_b == "MOMENTUM_DECAY")
        races = sum(1 for c in self.capsules
                    if c.execution_reality and c.execution_reality.race_condition_triggered)
        premature = sum(1 for c in self.capsules
                        if c.post_trade and c.post_trade.exited_prematurely)
        caught = sum(1 for c in self.capsules
                     if c.post_trade and c.post_trade.caught_max_impulse)
        return {
            "dual_lot_trades": len(self.capsules),
            "lot_a_total_pnl": round(lot_a, 4),
            "lot_b_total_pnl": round(lot_b, 4),
            "maker_fix_win_rate": round(a_wins / len(self.capsules) * 100, 1),
            "momentum_exit_count": momentum,
            "race_conditions": races,
            "premature_exits": premature,
            "max_impulse_catches": caught,
        }

    # ─────────────────── Signal Execution ───────────────────

    def _execute_signal(self, signal: Signal, df_15m_visible: Optional[pd.DataFrame] = None,
                        current_price: float = 0.0, entry_time: float = 0.0):
        """Place cascade orders from signal. ENTRY logic unchanged."""
        side = "Buy" if signal.direction == Direction.LONG else "Sell"

        cascade_id = str(uuid.uuid4())[:8]

        # v5.5: capture the pre-trade context + internal thoughts for this signal.
        # This does NOT alter entry — it only records what the entry engine saw.
        self._entry_context[cascade_id] = {
            "signal": signal,
            "df_snapshot": (df_15m_visible.copy() if df_15m_visible is not None else None),
            "current_price": current_price,
            "entry_time": entry_time,
        }

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
