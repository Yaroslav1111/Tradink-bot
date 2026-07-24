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


# ══════════════════════════════════════════════════════════════════
# v5.5 — DUAL-LOT ("Two-Winged") EXIT ARCHITECTURE
# ══════════════════════════════════════════════════════════════════

class LotRole(Enum):
    """Role of a lot inside a dual-lot ("Two-Winged") trade."""
    MAKER_FIX = "MAKER_FIX"        # Lot A: tight limit-maker, guaranteed fractional profit
    MOMENTUM_FLOAT = "MOMENTUM_FLOAT"  # Lot B: no static TP, closes on momentum decay


class LotState(Enum):
    """Fine-grained lifecycle state of an individual lot (for the race-safe state machine)."""
    PENDING = "PENDING"            # limit order resting on the book (Lot A only)
    ACTIVE = "ACTIVE"              # lot is an open position, being managed
    FILLED = "FILLED"              # Lot A limit maker filled → triggers breakeven cascade
    CANCELLING = "CANCELLING"      # Lot A cancel-in-flight (race: Lot B exited first)
    CLOSED = "CLOSED"              # lot fully closed / realized


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

    # ── v5.5 Dual-Lot ("Two-Winged") linkage ──
    # A dual-lot trade is two Position objects sharing a capsule_id.
    lot_role: Optional[LotRole] = None    # None = legacy single-lot behaviour
    capsule_id: str = ""                  # links Lot A + Lot B to one Trade Capsule
    lot_state: LotState = LotState.ACTIVE
    # Lot A (maker fix): the resting limit-maker TP order id + target price
    maker_tp_order_id: str = ""
    maker_tp_price: float = 0.0
    # structural invalidation stop (tight, behind the pivot / fibo grid level)
    structural_stop: float = 0.0
    pivot_level: float = 0.0              # the extreme pivot/fibo level that triggered entry


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


# ══════════════════════════════════════════════════════════════════
# v5.5 — "TRADE CAPSULE" META-ANALYSIS SUBSYSTEM
# ══════════════════════════════════════════════════════════════════
# A token-optimized, LLM-ready JSON record of a full trade cycle,
# built from exactly four pillars for later Gemini 1.5 Pro evaluation.


@dataclass
class PreTradeContext:
    """Pillar 1 — OHLCV snapshot immediately prior to entry."""
    symbol: str = ""
    direction: str = ""
    entry_time: float = 0.0
    ohlcv: list[list[float]] = field(default_factory=list)  # last N bars [ts,o,h,l,c,v]
    current_price: float = 0.0


@dataclass
class InternalThoughts:
    """Pillar 2 — exact oscillator/structural params at the decision millisecond."""
    rsi: float = 0.0
    cci: float = 0.0
    willr: float = 0.0
    oscillators_firing: int = 0
    atr: float = 0.0
    swing_high: float = 0.0
    swing_low: float = 0.0
    fibo_ext_1: float = 0.0
    fibo_ext_2: float = 0.0
    poc_price: Optional[float] = None
    pivot_level: float = 0.0
    structural_stop: float = 0.0
    rationale: str = ""


@dataclass
class ExecutionReality:
    """Pillar 3 — slippage, partial fills, limit order execution time."""
    lot_a_intended_price: float = 0.0
    lot_a_fill_price: float = 0.0
    lot_a_fill_time: float = 0.0
    lot_a_slippage_pct: float = 0.0
    lot_a_maker_exec_seconds: float = 0.0   # time from placement to maker TP fill
    lot_b_intended_price: float = 0.0
    lot_b_fill_price: float = 0.0
    lot_b_fill_time: float = 0.0
    lot_b_slippage_pct: float = 0.0
    partial_fill: bool = False
    filled_size_ratio: float = 1.0          # actual filled / intended
    race_condition_triggered: bool = False  # Lot B decayed before Lot A hit


@dataclass
class PostTradeReality:
    """Pillar 4 — audit of the next 10–15 candles after full closure."""
    lot_b_exit_price: float = 0.0
    lot_b_exit_time: float = 0.0
    lookahead_candles: int = 0
    max_favorable_after_exit: float = 0.0   # best price the market reached post-exit
    max_favorable_pct: float = 0.0          # % beyond exit that Lot B could have caught
    exited_prematurely: bool = False        # True if market kept running our way
    caught_max_impulse: bool = False        # True if exit was near the local extreme
    post_exit_ohlcv: list[list[float]] = field(default_factory=list)


@dataclass
class TradeCapsule:
    """
    v5.5 LLM-ready Trade Capsule. One per completed dual-lot trade cycle.
    Serialized (token-optimized) to data/ai_analysis/<capsule_id>.json.
    """
    capsule_id: str
    symbol: str
    direction: str
    version: str = "5.5"
    # realized results
    total_pnl_usdt: float = 0.0
    lot_a_pnl_usdt: float = 0.0
    lot_b_pnl_usdt: float = 0.0
    close_reason_a: str = ""
    close_reason_b: str = ""
    # four pillars
    pre_trade: Optional[PreTradeContext] = None
    internal_thoughts: Optional[InternalThoughts] = None
    execution_reality: Optional[ExecutionReality] = None
    post_trade: Optional[PostTradeReality] = None
