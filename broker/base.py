"""
v5.0 — Broker Protocol (Abstract Interface)
══════════════════════════════════════════════
The Strategy never imports this — it receives a broker via DI.
LiveBroker and SimBroker both implement this protocol.
"""
from __future__ import annotations
from typing import Protocol, Optional
import pandas as pd

from engine.models import (
    Order, Position, AccountState, Direction, Candle
)


class Broker(Protocol):
    """
    Abstract broker interface.
    Strategy calls these methods — implementation decides if it's live or sim.
    """

    def get_account(self) -> AccountState:
        """Return current account state (balance, margin, etc.)."""
        ...

    def get_positions(self) -> list[Position]:
        """Return all open positions (from exchange or sim state)."""
        ...

    def get_pending_orders(self) -> list[Order]:
        """Return all pending limit orders."""
        ...

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
        """Place a limit order. Returns Order if accepted, None if rejected."""
        ...

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel a pending order."""
        ...

    def close_position(self, symbol: str, quantity: float, side: str) -> bool:
        """Close (partially or fully) a position via market order."""
        ...

    def update_position_sl_tp(
        self, symbol: str, stop_loss: float = 0, take_profit: float = 0
    ) -> bool:
        """Update SL/TP on an open position (server-side)."""
        ...

    def get_klines(
        self, symbol: str, interval: str, limit: int
    ) -> Optional[pd.DataFrame]:
        """Fetch historical candle data."""
        ...

    def get_ticker_price(self, symbol: str) -> float:
        """Get current price of a symbol."""
        ...

    def get_lot_precision(self, symbol: str) -> int:
        """Get quantity decimal precision."""
        ...

    def get_tick_size(self, symbol: str) -> float:
        """Get price tick size."""
        ...

    def round_price(self, price: float, symbol: str) -> float:
        """Round price to valid tick."""
        ...

    def now(self) -> float:
        """Current time (unix epoch). Sim can return simulated time."""
        ...
