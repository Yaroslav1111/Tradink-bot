"""
v5.0 — Domain Models (Pure Data, No I/O)
═════════════════════════════════════════════
All dataclasses used across Strategy, Broker, and Runner layers.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Direction(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class PositionPhase(Enum):
    BREATHING = "BREATHING"      # Phase 1: SL fixed, don't touch
    BREAKEVEN = "BREAKEVEN"      # Phase 2: SL at entry + fees
    TRAILING = "TRAILING"        # Phase 3: Shadow trailing
    CLOSED = "CLOSED"


@dataclass
class Candle:
    """Single OHLCV bar."""
    timestamp: float  # unix epoch seconds
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Signal:
    """Strategy output — request to enter a trade."""
    symbol: str
    direction: Direction
    entry_price_50: float        # cascade leg 1 (0.50 fibo)
    entry_price_618: float       # cascade leg 2 (0.618 fibo)
    stop_loss: float
    take_profit: float
    qty_50: float
    qty_618: float
    risk_usdt_50: float
    risk_usdt_618: float
    atr: float
    fibo_ext_1: float
    fibo_ext_2: float
    swing_high: float
    swing_low: float
    oscillators_firing: int = 0
    rsi_value: float = 0.0
    cci_value: float = 0.0
    willr_value: float = 0.0
    poc_price: Optional[float] = None


@dataclass
class Order:
    """Represents a limit order (pending or filled)."""
    order_id: str
    symbol: str
    direction: Direction
    side: str                    # "Buy" or "Sell"
    price: float
    quantity: float
    stop_loss: float
    take_profit: float
    status: OrderStatus = OrderStatus.PENDING
    placed_at: float = 0.0      # unix timestamp
    filled_at: float = 0.0
    filled_price: float = 0.0
    risk_usdt: float = 0.0
    # Cascade
    cascade_group_id: str = ""
    cascade_level: str = ""     # "0.50" or "0.618"
    # Context for position creation
    atr: float = 0.0
    fibo_ext_1: float = 0.0
    fibo_ext_2: float = 0.0
    swing_high: float = 0.0
    swing_low: float = 0.0


@dataclass
class Position:
    """Tracked open position with 3-phase management state."""
    symbol: str
    direction: Direction
    entry_price: float
    quantity: float
    risk_usdt: float
    stop_loss: float
    take_profit: float
    initial_stop_loss: float
    phase: PositionPhase = PositionPhase.BREATHING
    # Tracking
    highest_price: float = 0.0
    lowest_price: float = float("inf")
    highest_profit_pct: float = 0.0
    entry_atr: float = 0.0
    # Fibo targets
    fibo_ext_1: float = 0.0
    fibo_ext_2: float = 0.0
    tp_repositioned: bool = False
    # Timestamps
    entry_time: float = 0.0
    close_time: float = 0.0
    # Result
    pnl_usdt: float = 0.0
    close_reason: str = ""
    # Exchange sync
    exchange_order_id: str = ""  # TP order on exchange


@dataclass
class TradeResult:
    """Completed trade for record keeping."""
    symbol: str
    direction: Direction
    entry_price: float
    exit_price: float
    quantity: float
    pnl_usdt: float
    risk_usdt: float
    entry_time: float
    exit_time: float
    close_reason: str
    duration_seconds: float = 0.0


@dataclass
class AccountState:
    """Snapshot of account for margin calculations."""
    total_balance: float = 0.0
    available_balance: float = 0.0     # free margin
    locked_margin: float = 0.0         # margin in positions
    pending_margin: float = 0.0        # margin reserved for pending orders
    unrealized_pnl: float = 0.0

    @property
    def free_margin(self) -> float:
        """Margin available for new trades."""
        return self.available_balance - self.pending_margin
