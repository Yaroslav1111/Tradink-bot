"""
v5.0 — Simulated Broker (Honest Backtester)
══════════════════════════════════════════════
Implements the Broker Protocol for backtesting.

Key features:
  - Honest fills: orders execute ONLY when candle high/low crosses level
  - Simulated time: advances bar-by-bar
  - Margin tracking: maintains free_margin like exchange
  - Zero network I/O — pure in-memory state
"""
from __future__ import annotations

import logging
import uuid
from typing import Optional

import pandas as pd
import numpy as np

from engine.models import (
    AccountState, Direction, Order, OrderStatus, Position,
    PositionPhase, TradeResult,
)

logger = logging.getLogger("aegis.broker.sim")


class SimBroker:
    """
    Simulated broker for backtesting — implements Broker Protocol.

    Processes fills bar-by-bar using candle high/low (honest execution).
    """

    def __init__(
        self,
        initial_balance: float = 2000.0,
        leverage: float = 5.0,
        maker_fee: float = 0.0002,
        taker_fee: float = 0.00055,
    ):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.leverage = leverage
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee

        # State
        self.positions: list[Position] = []
        self.pending_orders: list[Order] = []
        self.closed_trades: list[TradeResult] = []
        # v5.5: per-lot realized-PnL registry keyed by exchange_order_id
        self._closed_by_id: dict[str, dict] = {}

        # Simulated time & candle data
        self._current_time: float = 0.0
        self._current_candle: Optional[dict] = None  # {open, high, low, close, volume, timestamp}

        # Market data store (symbol -> full DataFrame)
        self._market_data: dict[str, pd.DataFrame] = {}
        self._current_bar_idx: dict[str, int] = {}
        self._1m_data: dict[str, pd.DataFrame] = {}

        # Instrument specs (simplified)
        self._lot_precision: dict[str, int] = {}
        self._tick_size: dict[str, float] = {}

    # ─────────────────── Data Loading ───────────────────

    def load_data(self, symbol: str, df: pd.DataFrame, interval: str = "15"):
        """Load historical data for a symbol."""
        self._market_data[symbol] = df.copy().reset_index(drop=True)
        self._current_bar_idx[symbol] = 0
        # Guess precision from price
        if len(df) > 0:
            price = float(df["close"].iloc[0])
            if price > 1000:
                self._lot_precision[symbol] = 3
                self._tick_size[symbol] = 0.01
            elif price > 10:
                self._lot_precision[symbol] = 2
                self._tick_size[symbol] = 0.001
            else:
                self._lot_precision[symbol] = 0
                self._tick_size[symbol] = 0.00001

    def load_1m_data(self, symbol: str, df_1m: pd.DataFrame):
        """Load 1-minute data for Volume Profile calculations."""
        self._1m_data[symbol] = df_1m.copy().reset_index(drop=True)

    # ─────────────────── Simulation Control ───────────────────

    def set_time(self, timestamp: float):
        """Set current simulation time."""
        self._current_time = timestamp

    def process_bar(self, symbol: str, bar_idx: int):
        """
        Process a single bar — check pending orders for fills.
        HONEST: fills only if candle high/low crosses order price.
        """
        df = self._market_data.get(symbol)
        if df is None or bar_idx >= len(df):
            return

        self._current_bar_idx[symbol] = bar_idx
        bar = df.iloc[bar_idx]
        bar_high = float(bar["high"])
        bar_low = float(bar["low"])
        bar_close = float(bar["close"])
        bar_time = float(bar.get("timestamp", self._current_time))
        self._current_time = bar_time

        # 1. Check pending orders for fills
        filled_orders = []
        for order in self.pending_orders[:]:
            if order.symbol != symbol:
                continue

            filled = False
            if order.direction == Direction.LONG:
                # Buy limit fills when price drops to order level
                if bar_low <= order.price:
                    filled = True
            else:
                # Sell limit fills when price rises to order level
                if bar_high >= order.price:
                    filled = True

            if filled:
                order.status = OrderStatus.FILLED
                order.filled_at = bar_time
                order.filled_price = order.price
                filled_orders.append(order)
                self.pending_orders.remove(order)
                self._create_position_from_order(order, bar_time)

        # 2. Check position SL/TP hits
        for pos in self.positions[:]:
            if pos.symbol != symbol or pos.phase == PositionPhase.CLOSED:
                continue

            # Track extremes
            if bar_high > pos.highest_price:
                pos.highest_price = bar_high
            if bar_low < pos.lowest_price:
                pos.lowest_price = bar_low

            # Check stop loss
            sl_hit = False
            tp_hit = False

            if pos.direction == Direction.LONG:
                if pos.stop_loss > 0 and bar_low <= pos.stop_loss:
                    sl_hit = True
                if pos.take_profit > 0 and bar_high >= pos.take_profit:
                    tp_hit = True
            else:
                if pos.stop_loss > 0 and bar_high >= pos.stop_loss:
                    sl_hit = True
                if pos.take_profit > 0 and bar_low <= pos.take_profit:
                    tp_hit = True

            if sl_hit and tp_hit:
                # Both hit in same bar — assume SL hit first (conservative)
                self._close_sim_position(pos, pos.stop_loss, "Stop Loss hit", bar_time)
            elif sl_hit:
                self._close_sim_position(pos, pos.stop_loss, "Stop Loss hit", bar_time)
            elif tp_hit:
                self._close_sim_position(pos, pos.take_profit, "Take Profit hit", bar_time)

    def _create_position_from_order(self, order: Order, fill_time: float):
        """Convert filled order to open position."""
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
            entry_time=fill_time,
            exchange_order_id=order.order_id,
            capsule_id=order.cascade_group_id,   # v5.5: link back to entry context
        )
        self.positions.append(pos)

        # Deduct margin
        margin = (order.price * order.quantity) / self.leverage
        self.balance -= margin * self.maker_fee  # Entry fee

        logger.debug(
            f"SIM FILL: {order.direction.value} {order.symbol} "
            f"{order.quantity} @ {order.price}"
        )

    def _close_sim_position(self, pos: Position, exit_price: float, reason: str, close_time: float):
        """Close a sim position and record PnL."""
        if pos.direction == Direction.LONG:
            raw_pnl = (exit_price - pos.entry_price) * pos.quantity
        else:
            raw_pnl = (pos.entry_price - exit_price) * pos.quantity

        # Fee: maker for limit TP, taker for SL
        if "Take Profit" in reason:
            fee = (pos.entry_price + exit_price) * pos.quantity * self.maker_fee
        else:
            fee = (pos.entry_price + exit_price) * pos.quantity * self.taker_fee

        net_pnl = raw_pnl - fee
        self.balance += net_pnl

        # Also return margin
        margin = (pos.entry_price * pos.quantity) / self.leverage
        # margin was never actually deducted as a block in sim — pnl is applied directly

        pos.phase = PositionPhase.CLOSED
        pos.pnl_usdt = net_pnl
        pos.close_reason = reason
        pos.close_time = close_time

        trade = TradeResult(
            symbol=pos.symbol,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            quantity=pos.quantity,
            pnl_usdt=net_pnl,
            risk_usdt=pos.risk_usdt,
            entry_time=pos.entry_time,
            exit_time=close_time,
            close_reason=reason,
            duration_seconds=close_time - pos.entry_time,
        )
        self.closed_trades.append(trade)
        # v5.5: register realized PnL against this lot's exchange_order_id
        if pos.exchange_order_id:
            self._closed_by_id[pos.exchange_order_id] = {
                "pnl": net_pnl,
                "reason": reason,
                "exit_price": exit_price,
                "exit_time": close_time,
            }
        self.positions.remove(pos)

        logger.debug(
            f"SIM CLOSE: {pos.symbol} {reason} PnL={net_pnl:+.2f} USDT"
        )

    # ─────────────────── Broker Protocol ───────────────────

    def get_account(self) -> AccountState:
        """Simulated account state."""
        locked = sum(
            (p.entry_price * p.quantity) / self.leverage
            for p in self.positions
        )
        pending = sum(
            (o.price * o.quantity) / self.leverage
            for o in self.pending_orders
        )
        unrealized = sum(
            self._unrealized_pnl(p) for p in self.positions
        )
        available = self.balance - locked
        return AccountState(
            total_balance=self.balance,
            available_balance=available,
            locked_margin=locked,
            pending_margin=pending,
            unrealized_pnl=unrealized,
        )

    def _unrealized_pnl(self, pos: Position) -> float:
        """Compute unrealized PnL for a position."""
        price = self.get_ticker_price(pos.symbol)
        if price <= 0:
            return 0.0
        if pos.direction == Direction.LONG:
            return (price - pos.entry_price) * pos.quantity
        else:
            return (pos.entry_price - price) * pos.quantity

    def get_positions(self) -> list[Position]:
        """Return open positions."""
        return list(self.positions)

    def get_pending_orders(self) -> list[Order]:
        """Return pending orders."""
        return list(self.pending_orders)

    def place_limit_order(
        self,
        symbol: str,
        side: str,
        price: float,
        quantity: float,
        stop_loss: float,
        take_profit: float,
        cascade_group_id: str = "",
        cascade_level: str = "",
        direction: Optional[Direction] = None,
        risk_usdt: float = 0.0,
        atr: float = 0.0,
        fibo_ext_1: float = 0.0,
        fibo_ext_2: float = 0.0,
        swing_high: float = 0.0,
        swing_low: float = 0.0,
    ) -> Optional[Order]:
        """Place a simulated limit order."""
        precision = self.get_lot_precision(symbol)
        quantity = round(quantity, precision)
        price = self.round_price(price, symbol)

        if quantity <= 0 or price <= 0:
            return None

        # Margin check
        margin_needed = (price * quantity) / self.leverage
        account = self.get_account()
        if margin_needed > account.free_margin:
            logger.debug(f"SIM: Insufficient margin for {symbol}")
            return None

        order = Order(
            order_id=str(uuid.uuid4())[:12],
            symbol=symbol,
            direction=direction or (Direction.LONG if side == "Buy" else Direction.SHORT),
            side=side,
            price=price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            status=OrderStatus.PENDING,
            placed_at=self._current_time,
            cascade_group_id=cascade_group_id,
            cascade_level=cascade_level,
            risk_usdt=risk_usdt,
            atr=atr,
            fibo_ext_1=fibo_ext_1,
            fibo_ext_2=fibo_ext_2,
            swing_high=swing_high,
            swing_low=swing_low,
        )
        self.pending_orders.append(order)
        return order

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel a pending order."""
        for order in self.pending_orders[:]:
            if order.order_id == order_id:
                order.status = OrderStatus.CANCELLED
                self.pending_orders.remove(order)
                return True
        return False

    def close_position(self, symbol: str, quantity: float, side: str) -> bool:
        """Close a position at current market price."""
        for pos in self.positions[:]:
            if pos.symbol == symbol:
                price = self.get_ticker_price(symbol)
                if price <= 0:
                    return False
                self._close_sim_position(pos, price, "Market close", self._current_time)
                return True
        return False

    # ─── v5.5 lot-precise helpers (dual-lot "Two-Winged" engine) ───

    def get_position_by_id(self, exchange_order_id: str) -> Optional[Position]:
        """Look up an open position by the exchange_order_id it was created from."""
        for pos in self.positions:
            if pos.exchange_order_id == exchange_order_id:
                return pos
        return None

    def get_filled_size(self, exchange_order_id: str) -> float:
        """
        Dynamic Size Sync — return the ACTUAL open size of a lot.
        Live trading uses this before firing trailing/breakeven payloads to
        prevent size-mismatch API rejections (e.g. Bybit ErrCode 10001)
        during partial fills. In sim, the full quantity is always filled.
        """
        pos = self.get_position_by_id(exchange_order_id)
        return pos.quantity if pos else 0.0

    def close_position_by_id(
        self, exchange_order_id: str, exit_price: float = 0.0, reason: str = "Market close",
    ) -> bool:
        """Close a specific lot (by its exchange_order_id) at market/given price."""
        for pos in self.positions[:]:
            if pos.exchange_order_id == exchange_order_id:
                price = exit_price if exit_price > 0 else self.get_ticker_price(pos.symbol)
                if price <= 0:
                    return False
                self._close_sim_position(pos, price, reason, self._current_time)
                return True
        return False

    def update_position_sl_tp_by_id(
        self, exchange_order_id: str, stop_loss: float = 0, take_profit: float = 0,
    ) -> bool:
        """Update SL/TP on a specific lot identified by exchange_order_id."""
        for pos in self.positions:
            if pos.exchange_order_id == exchange_order_id:
                if stop_loss > 0:
                    pos.stop_loss = stop_loss
                # allow clearing TP (set to 0) for the momentum-float lot
                pos.take_profit = take_profit
                return True
        return False

    def update_position_sl_tp(
        self, symbol: str, stop_loss: float = 0, take_profit: float = 0
    ) -> bool:
        """Update SL/TP on sim position."""
        for pos in self.positions:
            if pos.symbol == symbol:
                if stop_loss > 0:
                    pos.stop_loss = stop_loss
                if take_profit > 0:
                    pos.take_profit = take_profit
                return True
        return False

    def get_klines(
        self, symbol: str, interval: str, limit: int
    ) -> Optional[pd.DataFrame]:
        """Return historical data up to current bar index."""
        df = self._market_data.get(symbol)
        if df is None:
            return None

        bar_idx = self._current_bar_idx.get(symbol, 0)
        # Return up to current bar (inclusive)
        end = bar_idx + 1
        start = max(0, end - limit)
        result = df.iloc[start:end].copy().reset_index(drop=True)
        if len(result) == 0:
            return None
        return result

    def get_1m_klines(self, symbol: str, limit: int = 60) -> Optional[pd.DataFrame]:
        """Return 1-minute data (for POC). Uses pre-loaded data."""
        df_1m = self._1m_data.get(symbol)
        if df_1m is None or len(df_1m) == 0:
            return None

        # In sim, return the last `limit` bars of 1m data relative to current time
        if "timestamp" in df_1m.columns:
            mask = df_1m["timestamp"] <= self._current_time
            available = df_1m[mask]
            if len(available) == 0:
                return df_1m.tail(limit)
            return available.tail(limit).reset_index(drop=True)
        return df_1m.tail(limit).reset_index(drop=True)

    def get_ticker_price(self, symbol: str) -> float:
        """Get current price (close of current bar)."""
        df = self._market_data.get(symbol)
        if df is None:
            return 0.0
        bar_idx = self._current_bar_idx.get(symbol, 0)
        if bar_idx < len(df):
            return float(df.iloc[bar_idx]["close"])
        return 0.0

    def get_lot_precision(self, symbol: str) -> int:
        return self._lot_precision.get(symbol, 3)

    def get_tick_size(self, symbol: str) -> float:
        return self._tick_size.get(symbol, 0.01)

    def round_price(self, price: float, symbol: str) -> float:
        tick = self.get_tick_size(symbol)
        if tick > 0:
            return round(round(price / tick) * tick, 10)
        return round(price, 6)

    def now(self) -> float:
        """Return simulated time."""
        return self._current_time

    # ─────────────────── Results ───────────────────

    def get_results_summary(self) -> dict:
        """Get backtest results summary."""
        trades = self.closed_trades
        if not trades:
            return {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "max_drawdown_pct": 0.0,
                "profit_factor": 0.0,
                "final_balance": self.balance,
                "return_pct": 0.0,
            }

        wins = [t for t in trades if t.pnl_usdt > 0]
        losses = [t for t in trades if t.pnl_usdt <= 0]
        total_pnl = sum(t.pnl_usdt for t in trades)

        gross_profit = sum(t.pnl_usdt for t in wins)
        gross_loss = abs(sum(t.pnl_usdt for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Max drawdown
        peak = self.initial_balance
        max_dd = 0.0
        running = self.initial_balance
        for t in trades:
            running += t.pnl_usdt
            if running > peak:
                peak = running
            dd = (peak - running) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        return {
            "total_trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(trades) * 100 if trades else 0,
            "total_pnl": total_pnl,
            "max_drawdown_pct": max_dd * 100,
            "profit_factor": profit_factor,
            "final_balance": self.balance,
            "return_pct": (self.balance - self.initial_balance) / self.initial_balance * 100,
            "avg_trade_pnl": total_pnl / len(trades),
            "avg_win": gross_profit / len(wins) if wins else 0,
            "avg_loss": -gross_loss / len(losses) if losses else 0,
        }

    def reset(self):
        """Reset broker state for new backtest run."""
        self.balance = self.initial_balance
        self.positions.clear()
        self.pending_orders.clear()
        self.closed_trades.clear()
        self._closed_by_id.clear()
        self._current_time = 0.0
        self._current_bar_idx = {s: 0 for s in self._market_data}
