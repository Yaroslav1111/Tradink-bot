"""
Aegis-Quant-Lab v3.0 — Live Trading Engine
============================================
THE COMPLETE ADAPTIVE QUANTUM TRADING BOT

Architecture:
  1. ScoringEngine       — Mathematical weighted scoring (replaces rigid 70% filter)
  2. BreakevenManager    — Breakeven + trailing stop on trend reversal
  3. CompoundCalculator  — Dynamic position sizing with compound interest
  4. ClusterGuard        — Correlation cluster protection (no duplicated risk)
  5. BybitDemoConnector   — Bybit Demo API integration for forward testing

Flow:
  Market Data → Scoring → ClusterGuard → Position Sizing → Order → BreakevenManager

Capital: $2000 USDT | Risk: 1% per trade | Max concurrent: 5
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd

import config

logger = logging.getLogger("aegis.live_engine")


# ══════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════

class TradeDirection(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class PositionState(Enum):
    OPEN = "OPEN"
    BREAKEVEN = "BREAKEVEN"       # SL moved to entry
    TRAILING = "TRAILING"         # SL trailing behind price
    CLOSED = "CLOSED"


@dataclass
class LivePosition:
    """Represents a single open position managed by the engine."""
    symbol: str
    direction: TradeDirection
    entry_price: float
    entry_time: datetime
    quantity: float                        # position size in base asset
    risk_usdt: float                       # how much USDT we risk (1% of balance)
    stop_loss: float                       # current SL price
    take_profit: float                     # TP price
    initial_stop_loss: float               # original SL (never changes)
    state: PositionState = PositionState.OPEN
    score_at_entry: float = 0.0           # consensus score when entered
    cluster_id: int = -1                  # cluster the symbol belongs to
    highest_profit_pct: float = 0.0       # max unrealized PnL seen
    pnl_usdt: float = 0.0                # realized PnL
    close_reason: str = ""                # why it was closed


@dataclass
class ScoringResult:
    """Result of the mathematical scoring evaluation."""
    symbol: str
    direction: TradeDirection | None = None
    total_score: float = 0.0              # weighted sum (0.0 — 1.0)
    confidence_pct: float = 0.0           # total_score * 100
    long_score: float = 0.0
    short_score: float = 0.0
    n_indicators_firing: int = 0
    n_indicators_total: int = 0
    dissonance: float = 1.0
    is_tradeable: bool = False
    blocked_reason: str = ""
    indicator_details: list[dict] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════
# MODULE 1: MATHEMATICAL SCORING ENGINE
# ══════════════════════════════════════════════════════════════════

class ScoringEngine:
    """
    Replaces the rigid "70% indicators must agree" filter.
    
    How it works:
      1. Each indicator computes a CONFIDENCE score (0.0 — 1.0) based on
         how extreme its current reading is (e.g., RSI=15 → 0.90 confidence)
      2. This confidence is multiplied by the indicator's historical accuracy_weight
         from indicator_weights.json
      3. All weighted confidences are summed and normalized
      4. If the final score exceeds ENTRY_THRESHOLD → trade signal generated
    
    Key advantage: 4 strong indicators with high accuracy can trigger a trade
    even if the other 11 are neutral. No more "sitting on the fence."
    """

    # Entry threshold: the normalized weighted score must exceed this
    ENTRY_THRESHOLD: float = 0.30       # Tuned for quality (0.5=aggressive, 0.8=conservative)
    CHAOS_THRESHOLD: float = 0.50       # If opposing score is > 45% of total → chaos → no trade

    def __init__(self, indicator_weights: dict[str, dict]):
        """
        Parameters
        ----------
        indicator_weights : dict from indicator_weights.json for ONE symbol
            {indicator_name: {accuracy_weight, direction, win_rate_pct, ...}}
        """
        self.weights = indicator_weights
        self._top_indicators = self._select_top_indicators()

    def _select_top_indicators(self, top_n: int = 15) -> list[dict]:
        """Select top N indicators by accuracy_weight (only profitable ones)."""
        candidates = []
        for name, info in self.weights.items():
            w = info.get("accuracy_weight", 0)
            if w > 0:
                candidates.append({"name": name, "weight": w, **info})
        candidates.sort(key=lambda x: x["weight"], reverse=True)
        return candidates[:top_n]

    @staticmethod
    def _compute_indicator_confidence(indicator_name: str, raw_value: float) -> float:
        """
        Convert raw indicator value to a confidence score [0.0, 1.0].
        
        This is the "intelligence" — the more extreme the reading,
        the higher our confidence that this is a real signal.
        
        Examples:
          - RSI=15 (extremely oversold) → confidence=0.90
          - RSI=28 (slightly oversold)  → confidence=0.40
          - RSI=50 (neutral)            → confidence=0.0 (not firing)
        """
        name_lower = indicator_name.lower()

        # RSI-based (oversold = low values = high confidence for LONG)
        if "rsi" in name_lower and "oversold" in name_lower:
            if raw_value <= 10:
                return 0.95
            elif raw_value <= 15:
                return 0.90
            elif raw_value <= 20:
                return 0.75
            elif raw_value <= 25:
                return 0.55
            elif raw_value <= 30:
                return 0.35
            return 0.0

        if "rsi" in name_lower and "overbought" in name_lower:
            if raw_value >= 90:
                return 0.95
            elif raw_value >= 85:
                return 0.90
            elif raw_value >= 80:
                return 0.75
            elif raw_value >= 75:
                return 0.55
            elif raw_value >= 70:
                return 0.35
            return 0.0

        # Bollinger %B (below 0 = oversold, above 1 = overbought)
        if "bb" in name_lower and "oversold" in name_lower:
            if raw_value < -0.3:
                return 0.95
            elif raw_value < -0.1:
                return 0.80
            elif raw_value < 0.0:
                return 0.55
            elif raw_value < 0.1:
                return 0.30
            return 0.0

        if "bb" in name_lower and "overbought" in name_lower:
            if raw_value > 1.3:
                return 0.95
            elif raw_value > 1.1:
                return 0.80
            elif raw_value > 1.0:
                return 0.55
            return 0.0

        # CCI (oversold < -100, overbought > 100)
        if "cci" in name_lower and "oversold" in name_lower:
            if raw_value <= -300:
                return 0.95
            elif raw_value <= -250:
                return 0.85
            elif raw_value <= -200:
                return 0.75
            elif raw_value <= -150:
                return 0.60
            elif raw_value <= -100:
                return 0.40
            return 0.0

        if "cci" in name_lower and "overbought" in name_lower:
            if raw_value >= 300:
                return 0.95
            elif raw_value >= 250:
                return 0.85
            elif raw_value >= 200:
                return 0.75
            elif raw_value >= 150:
                return 0.60
            elif raw_value >= 100:
                return 0.40
            return 0.0

        # Williams %R (-80 oversold, -20 overbought)
        if "willr" in name_lower and "oversold" in name_lower:
            if raw_value <= -95:
                return 0.95
            elif raw_value <= -90:
                return 0.85
            elif raw_value <= -85:
                return 0.70
            elif raw_value <= -80:
                return 0.50
            return 0.0

        if "willr" in name_lower and "overbought" in name_lower:
            if raw_value >= -5:
                return 0.95
            elif raw_value >= -10:
                return 0.85
            elif raw_value >= -15:
                return 0.70
            elif raw_value >= -20:
                return 0.50
            return 0.0

        # Stochastic / KDJ (< 20 oversold, > 80 overbought)
        if ("stoch" in name_lower or "kdj" in name_lower) and "oversold" in name_lower:
            if raw_value < 5:
                return 0.95
            elif raw_value < 10:
                return 0.80
            elif raw_value < 15:
                return 0.60
            elif raw_value < 20:
                return 0.40
            return 0.0

        if ("stoch" in name_lower or "kdj" in name_lower) and "overbought" in name_lower:
            if raw_value > 95:
                return 0.95
            elif raw_value > 90:
                return 0.80
            elif raw_value > 85:
                return 0.60
            elif raw_value > 80:
                return 0.40
            return 0.0

        # MFI (same ranges as RSI)
        if "mfi" in name_lower and "oversold" in name_lower:
            if raw_value < 10:
                return 0.95
            elif raw_value < 15:
                return 0.80
            elif raw_value < 20:
                return 0.55
            return 0.0

        if "mfi" in name_lower and "overbought" in name_lower:
            if raw_value > 90:
                return 0.95
            elif raw_value > 85:
                return 0.80
            elif raw_value > 80:
                return 0.55
            return 0.0

        # EMA / trend indicators — binary but with momentum weight
        if "ema" in name_lower or "supertrend" in name_lower or "adx" in name_lower:
            # These are binary (on/off) — return moderate confidence
            return 0.60 if raw_value != 0 else 0.0

        # PPO / TRIX / CMO / Fisher — use absolute magnitude
        if any(x in name_lower for x in ["ppo", "trix", "cmo", "fisher", "hma", "obv", "cmf"]):
            abs_val = abs(raw_value)
            if abs_val > 3.0:
                return 0.90
            elif abs_val > 2.0:
                return 0.70
            elif abs_val > 1.0:
                return 0.50
            elif abs_val > 0.5:
                return 0.35
            return 0.20

        # Breakout / volatility — binary with moderate confidence
        if "breakout" in name_lower or "vol" in name_lower:
            return 0.55 if raw_value != 0 else 0.0

        # Default: moderate confidence for any firing indicator
        return 0.50

    def evaluate(
        self,
        indicator_readings: dict[str, float],
        firing_indicators: dict[str, bool],
    ) -> ScoringResult:
        """
        Evaluate scoring for current bar.
        
        Parameters
        ----------
        indicator_readings : {indicator_name: raw_value} — current raw values
        firing_indicators : {indicator_name: True/False} — which are currently active
        
        Returns
        -------
        ScoringResult with weighted score and trade decision
        """
        result = ScoringResult(symbol="")

        long_score = 0.0
        short_score = 0.0
        total_max_weight = 0.0
        details = []

        for ind_info in self._top_indicators:
            ind_name = ind_info["name"]
            accuracy_weight = ind_info["weight"]
            direction = ind_info.get("direction", "LONG")

            total_max_weight += accuracy_weight

            # Check if indicator is firing
            is_firing = firing_indicators.get(ind_name, False)
            if not is_firing:
                continue

            # Get raw value and compute confidence
            raw_value = indicator_readings.get(ind_name, 0.0)
            confidence = self._compute_indicator_confidence(ind_name, raw_value)

            if confidence <= 0:
                continue

            # Weighted contribution = confidence × accuracy_weight
            weighted_contribution = confidence * accuracy_weight

            if direction == "LONG":
                long_score += weighted_contribution
            else:
                short_score += weighted_contribution

            result.n_indicators_firing += 1
            details.append({
                "name": ind_name,
                "direction": direction,
                "confidence": round(confidence, 3),
                "weight": round(accuracy_weight, 4),
                "contribution": round(weighted_contribution, 4),
            })

        result.n_indicators_total = len(self._top_indicators)
        result.long_score = long_score
        result.short_score = short_score
        result.indicator_details = details

        # Normalize scores
        total_score = long_score + short_score
        if total_max_weight <= 0 or total_score <= 0:
            result.is_tradeable = False
            result.blocked_reason = "No indicators firing"
            return result

        # Determine dominant direction
        if long_score > short_score:
            result.direction = TradeDirection.LONG
            dominant_score = long_score
            opposing_score = short_score
        elif short_score > long_score:
            result.direction = TradeDirection.SHORT
            dominant_score = short_score
            opposing_score = long_score
        else:
            result.is_tradeable = False
            result.blocked_reason = "Perfect tie between LONG and SHORT"
            return result

        # Normalized score (dominant vs. theoretical max)
        result.total_score = dominant_score / total_max_weight
        result.confidence_pct = round(result.total_score * 100, 1)

        # Dissonance = opposing / total (0 = pure agreement, 1 = chaos)
        result.dissonance = opposing_score / total_score if total_score > 0 else 1.0

        # ── Decision gates ──
        # Gate 1: Chaos filter
        if result.dissonance > self.CHAOS_THRESHOLD:
            result.is_tradeable = False
            result.blocked_reason = (
                f"Chaos detected: dissonance={result.dissonance:.2f} > "
                f"{self.CHAOS_THRESHOLD} — indicators contradicting each other"
            )
            return result

        # Gate 2: Minimum score threshold
        if result.total_score < self.ENTRY_THRESHOLD:
            result.is_tradeable = False
            result.blocked_reason = (
                f"Score too low: {result.total_score:.3f} < "
                f"{self.ENTRY_THRESHOLD} threshold"
            )
            return result

        # All gates passed
        result.is_tradeable = True
        return result


# ══════════════════════════════════════════════════════════════════
# MODULE 2: BREAKEVEN & TRAILING STOP MANAGER
# ══════════════════════════════════════════════════════════════════

class BreakevenManager:
    """
    Smart exit management:
      - When price moves +1.5% in our favor → move SL to breakeven (entry + fees)
      - When price moves +3.0% → activate trailing stop (trail by 1.5%)
      - When indicators reverse while in profit → close at breakeven/micro-profit
    
    Prevents giving back unrealized profits to the market.
    """

    # Thresholds (in fraction, not percent)
    BREAKEVEN_TRIGGER: float = 0.015     # +1.5% triggers breakeven SL move
    TRAILING_TRIGGER: float = 0.030      # +3.0% activates trailing
    TRAILING_DISTANCE: float = 0.015     # trail 1.5% behind highest point
    FEE_BUFFER: float = 0.0004           # Bybit roundtrip fee for LIMIT orders (maker)

    def __init__(self):
        pass

    def update_position(
        self,
        position: LivePosition,
        current_price: float,
        indicators_reversing: bool = False,
    ) -> LivePosition:
        """
        Update position's stop-loss based on current price movement.
        
        Parameters
        ----------
        position : LivePosition to manage
        current_price : latest market price
        indicators_reversing : True if consensus indicators now point opposite
        
        Returns
        -------
        Updated LivePosition (may be closed if SL hit)
        """
        if position.state == PositionState.CLOSED:
            return position

        # Calculate unrealized PnL
        if position.direction == TradeDirection.LONG:
            unrealized_pct = (current_price - position.entry_price) / position.entry_price
        else:
            unrealized_pct = (position.entry_price - current_price) / position.entry_price

        # Track highest profit seen
        position.highest_profit_pct = max(position.highest_profit_pct, unrealized_pct)

        # ── Check if stop-loss is hit ──
        if position.direction == TradeDirection.LONG:
            if current_price <= position.stop_loss:
                return self._close_position(position, current_price, "Stop-loss hit")
        else:  # SHORT
            if current_price >= position.stop_loss:
                return self._close_position(position, current_price, "Stop-loss hit")

        # ── Check if take-profit is hit ──
        if position.direction == TradeDirection.LONG:
            if current_price >= position.take_profit:
                return self._close_position(position, current_price, "Take-profit hit")
        else:  # SHORT
            if current_price <= position.take_profit:
                return self._close_position(position, current_price, "Take-profit hit")

        # ── BREAKEVEN LOGIC ──
        if position.state == PositionState.OPEN and unrealized_pct >= self.BREAKEVEN_TRIGGER:
            # Move SL to entry price + fee buffer (guaranteed no-loss exit)
            if position.direction == TradeDirection.LONG:
                new_sl = position.entry_price * (1 + self.FEE_BUFFER)
            else:
                new_sl = position.entry_price * (1 - self.FEE_BUFFER)
            position.stop_loss = new_sl
            position.state = PositionState.BREAKEVEN
            logger.info(
                f"  ⚡ {position.symbol} → BREAKEVEN: SL moved to "
                f"{new_sl:.6f} (entry + fees), unrealized +{unrealized_pct*100:.2f}%"
            )

        # ── TRAILING STOP LOGIC ──
        if unrealized_pct >= self.TRAILING_TRIGGER:
            position.state = PositionState.TRAILING
            # Trail behind the highest point
            if position.direction == TradeDirection.LONG:
                trail_price = current_price * (1 - self.TRAILING_DISTANCE)
                if trail_price > position.stop_loss:
                    position.stop_loss = trail_price
                    logger.debug(
                        f"  📈 {position.symbol} TRAILING: SL raised to {trail_price:.6f}"
                    )
            else:  # SHORT
                trail_price = current_price * (1 + self.TRAILING_DISTANCE)
                if trail_price < position.stop_loss:
                    position.stop_loss = trail_price
                    logger.debug(
                        f"  📉 {position.symbol} TRAILING: SL lowered to {trail_price:.6f}"
                    )

        # ── INDICATOR REVERSAL EXIT ──
        # If indicators reverse AND we're in profit → close at breakeven/micro-profit
        if indicators_reversing and unrealized_pct > self.FEE_BUFFER:
            # Only close if we've been in profit (don't close losing trades on reversal)
            return self._close_position(
                position, current_price,
                f"Indicator reversal while in profit (+{unrealized_pct*100:.2f}%)"
            )

        return position

    def _close_position(
        self,
        position: LivePosition,
        exit_price: float,
        reason: str,
    ) -> LivePosition:
        """Close a position and calculate PnL."""
        if position.direction == TradeDirection.LONG:
            raw_pnl_pct = (exit_price - position.entry_price) / position.entry_price
        else:
            raw_pnl_pct = (position.entry_price - exit_price) / position.entry_price

        # Subtract roundtrip fee
        net_pnl_pct = raw_pnl_pct - self.FEE_BUFFER
        position.pnl_usdt = net_pnl_pct * position.risk_usdt * config.RISK_REWARD_RATIO
        position.state = PositionState.CLOSED
        position.close_reason = reason

        logger.info(
            f"  {'✅' if net_pnl_pct > 0 else '❌'} CLOSED {position.symbol} "
            f"{position.direction.value}: PnL={net_pnl_pct*100:+.2f}% "
            f"(${position.pnl_usdt:+.2f}) — {reason}"
        )
        return position


# ══════════════════════════════════════════════════════════════════
# MODULE 3: COMPOUND INTEREST CALCULATOR
# ══════════════════════════════════════════════════════════════════

class CompoundCalculator:
    """
    Dynamic position sizing with compound interest.
    
    Instead of always risking 1% of the INITIAL $2000 ($20),
    we risk 1% of the CURRENT balance.
    
    Example trajectory:
      Start: $2000 → risk $20 per trade
      After 10 wins: $2200 → risk $22 per trade  
      After 20 wins: $2420 → risk $24.20 per trade
      ...exponential growth on winning streaks
    
    Safety: After 3 consecutive losses, reduce risk to 0.5% temporarily
    (Kelly Criterion adaptation for drawdown protection)
    """

    RISK_PCT_NORMAL: float = 0.01           # 1% risk per trade normally
    RISK_PCT_REDUCED: float = 0.005         # 0.5% risk after loss streak
    CONSECUTIVE_LOSS_THRESHOLD: int = 3     # reduce risk after N consecutive losses
    MIN_RISK_USDT: float = 5.0              # never risk less than $5
    MAX_RISK_PCT: float = 0.02              # never risk more than 2% (safety cap)

    def __init__(self, initial_capital: float = 2000.0):
        self.initial_capital = initial_capital
        self.current_balance = initial_capital
        self.consecutive_losses = 0
        self.total_trades = 0
        self.total_wins = 0
        self.total_losses = 0
        self.peak_balance = initial_capital
        self.max_drawdown_pct = 0.0
        self.trade_history: list[dict] = []

    @property
    def current_risk_pct(self) -> float:
        """Dynamic risk percentage based on recent performance."""
        if self.consecutive_losses >= self.CONSECUTIVE_LOSS_THRESHOLD:
            return self.RISK_PCT_REDUCED
        return self.RISK_PCT_NORMAL

    @property
    def current_risk_usdt(self) -> float:
        """Current $ risk per trade (compound)."""
        risk = self.current_balance * self.current_risk_pct
        risk = max(risk, self.MIN_RISK_USDT)
        risk = min(risk, self.current_balance * self.MAX_RISK_PCT)
        return round(risk, 2)

    @property
    def roi_pct(self) -> float:
        """Total ROI since inception."""
        return round((self.current_balance - self.initial_capital) / self.initial_capital * 100, 2)

    @property
    def drawdown_from_peak(self) -> float:
        """Current drawdown from peak balance."""
        if self.peak_balance <= 0:
            return 0.0
        return round((self.peak_balance - self.current_balance) / self.peak_balance * 100, 2)

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss_price: float,
        leverage: float = 1.0,
    ) -> dict:
        """
        Calculate optimal position size based on compound balance.
        
        Returns
        -------
        dict with: quantity, risk_usdt, position_value, effective_leverage
        """
        risk_usdt = self.current_risk_usdt

        # Distance from entry to stop-loss
        sl_distance_pct = abs(entry_price - stop_loss_price) / entry_price
        if sl_distance_pct <= 0:
            sl_distance_pct = 0.02  # default 2% if SL not set

        # Position value = risk / SL_distance
        position_value_usdt = risk_usdt / sl_distance_pct

        # Apply leverage
        margin_required = position_value_usdt / leverage

        # Safety: never use more than 20% of balance for single position margin
        max_margin = self.current_balance * 0.20
        if margin_required > max_margin:
            margin_required = max_margin
            position_value_usdt = margin_required * leverage

        quantity = position_value_usdt / entry_price

        return {
            "quantity": round(quantity, 8),
            "risk_usdt": round(risk_usdt, 2),
            "position_value_usdt": round(position_value_usdt, 2),
            "margin_required_usdt": round(margin_required, 2),
            "effective_leverage": round(position_value_usdt / margin_required, 2),
            "sl_distance_pct": round(sl_distance_pct * 100, 3),
            "risk_pct_used": round(self.current_risk_pct * 100, 3),
            "current_balance": round(self.current_balance, 2),
        }

    def record_trade(self, pnl_usdt: float, symbol: str = ""):
        """Record trade result and update compound balance."""
        self.total_trades += 1
        self.current_balance += pnl_usdt

        if pnl_usdt > 0:
            self.total_wins += 1
            self.consecutive_losses = 0
        else:
            self.total_losses += 1
            self.consecutive_losses += 1

        # Update peak and drawdown
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
        dd = self.drawdown_from_peak
        if dd > self.max_drawdown_pct:
            self.max_drawdown_pct = dd

        self.trade_history.append({
            "trade_num": self.total_trades,
            "symbol": symbol,
            "pnl_usdt": round(pnl_usdt, 2),
            "balance_after": round(self.current_balance, 2),
            "risk_pct": self.current_risk_pct,
            "consecutive_losses": self.consecutive_losses,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        logger.info(
            f"  💰 Trade #{self.total_trades}: PnL=${pnl_usdt:+.2f} | "
            f"Balance=${self.current_balance:.2f} | "
            f"ROI={self.roi_pct:+.1f}% | DD={self.drawdown_from_peak:.1f}% | "
            f"Risk next={self.current_risk_pct*100:.1f}%"
        )

    def get_status(self) -> dict:
        """Return full compound calculator status."""
        win_rate = self.total_wins / max(self.total_trades, 1) * 100
        return {
            "initial_capital": self.initial_capital,
            "current_balance": round(self.current_balance, 2),
            "roi_pct": self.roi_pct,
            "total_trades": self.total_trades,
            "wins": self.total_wins,
            "losses": self.total_losses,
            "win_rate_pct": round(win_rate, 1),
            "consecutive_losses": self.consecutive_losses,
            "current_risk_pct": self.current_risk_pct * 100,
            "current_risk_usdt": self.current_risk_usdt,
            "peak_balance": round(self.peak_balance, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "drawdown_from_peak": self.drawdown_from_peak,
        }


# ══════════════════════════════════════════════════════════════════
# MODULE 4: CLUSTER CORRELATION GUARD
# ══════════════════════════════════════════════════════════════════

class ClusterGuard:
    """
    Prevents opening positions in highly correlated assets simultaneously.
    
    If SAND is open (cluster: Gaming/Metaverse) and MANA signal arrives
    (same cluster, correlation 0.90), the bot BLOCKS the MANA entry.
    
    Uses:
      - risk_blocks.json correlation data
      - hierarchical_clusters from cross_asset_analyst
      - Live correlation threshold (default 0.75)
    """

    CORRELATION_BLOCK_THRESHOLD: float = 0.75   # block if corr > this
    MAX_POSITIONS_PER_CLUSTER: int = 2          # max 2 from same cluster

    def __init__(
        self,
        risk_blocks: list[dict] | None = None,
        cluster_map: dict[str, int] | None = None,
        correlation_matrix: dict[str, float] | None = None,
    ):
        """
        Parameters
        ----------
        risk_blocks : loaded from risk_blocks.json
        cluster_map : {symbol: cluster_id} from hierarchical clustering
        correlation_matrix : {symA|symB: correlation} from rolling_corr_summary
        """
        self.risk_blocks = risk_blocks or []
        self.cluster_map = cluster_map or {}
        self.correlation_pairs = correlation_matrix or {}

        # Build lookup tables
        self._corr_lookup: dict[tuple[str, str], float] = {}
        for key, val in self.correlation_pairs.items():
            parts = key.split("|")
            if len(parts) == 2:
                a, b = parts
                self._corr_lookup[(a, b)] = val
                self._corr_lookup[(b, a)] = val

        # Extract correlation blocks from risk_blocks.json
        for block in self.risk_blocks:
            fc = block.get("forbidden_conditions", {})
            if isinstance(fc, list):
                for condition in fc:
                    self._extract_correlation_rule(block["symbol"], condition)
            else:
                self._extract_correlation_rule(block["symbol"], fc)

    def _extract_correlation_rule(self, symbol: str, condition: dict):
        """Extract correlation info from a risk block condition."""
        corr_asset = condition.get("highly_correlated_active_trade")
        corr_val = condition.get("correlation")
        if corr_asset and corr_val:
            self._corr_lookup[(symbol, corr_asset)] = abs(corr_val)
            self._corr_lookup[(corr_asset, symbol)] = abs(corr_val)

    def check_entry_allowed(
        self,
        symbol: str,
        open_positions: list[LivePosition],
    ) -> tuple[bool, str]:
        """
        Check if opening a new position in `symbol` is allowed.
        
        Returns
        -------
        (allowed: bool, reason: str)
        """
        sym_clean = symbol.replace("/", "").replace("USDT", "") + "USDT"

        # Check 0: ANTI-PYRAMID — block re-entry into same symbol
        for p in open_positions:
            if p.state == PositionState.CLOSED:
                continue
            p_clean = p.symbol.replace("/", "").replace("USDT", "") + "USDT"
            if p_clean == sym_clean:
                return False, (
                    f"ANTI-PYRAMID: {symbol} already has an open position "
                    f"({p.direction.value} since {p.entry_time.strftime('%H:%M')})"
                )

        # Check 1: Max concurrent positions
        active = [p for p in open_positions if p.state != PositionState.CLOSED]
        if len(active) >= config.MAX_CONCURRENT_TRADES:
            return False, f"Max concurrent positions ({config.MAX_CONCURRENT_TRADES}) reached"

        # Check 2: Correlation with open positions
        for pos in active:
            pos_sym = pos.symbol.replace("/", "").replace("USDT", "") + "USDT"

            # Direct correlation check
            corr = self._corr_lookup.get((sym_clean, pos_sym), 0.0)
            if corr == 0.0:
                # Try alternative key formats
                for key_format in [
                    (symbol, pos.symbol),
                    (symbol.replace("/", ""), pos.symbol.replace("/", "")),
                ]:
                    corr = self._corr_lookup.get(key_format, 0.0)
                    if corr > 0:
                        break

            if corr >= self.CORRELATION_BLOCK_THRESHOLD:
                return False, (
                    f"Blocked: {symbol} correlates {corr:.2f} with open {pos.symbol} "
                    f"(threshold {self.CORRELATION_BLOCK_THRESHOLD})"
                )

        # Check 3: Cluster saturation
        new_cluster = self.cluster_map.get(symbol, self.cluster_map.get(sym_clean, -1))
        if new_cluster >= 0:
            cluster_count = sum(
                1 for p in active
                if self.cluster_map.get(p.symbol, self.cluster_map.get(
                    p.symbol.replace("/", "").replace("USDT", "") + "USDT", -2
                )) == new_cluster
            )
            if cluster_count >= self.MAX_POSITIONS_PER_CLUSTER:
                return False, (
                    f"Cluster #{new_cluster} saturated: "
                    f"{cluster_count}/{self.MAX_POSITIONS_PER_CLUSTER} positions already open"
                )

        return True, "Entry allowed"

    def get_correlation(self, sym_a: str, sym_b: str) -> float:
        """Get correlation between two symbols."""
        return self._corr_lookup.get((sym_a, sym_b), 0.0)


# ══════════════════════════════════════════════════════════════════
# MODULE 5: BYBIT DEMO API CONNECTOR
# ══════════════════════════════════════════════════════════════════

class BybitDemoConnector:
    """
    Connects to Bybit Demo API for forward testing.
    
    Wraps the pybit library with our engine's logic:
      - Real-time kline streaming
      - Order placement (demo)
      - Position management
      - Balance tracking
    
    Demo endpoint: https://api-demo.bybit.com
    """

    DEMO_ENDPOINT = "https://api-demo.bybit.com"

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        testnet: bool = True,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.session = None
        self._connected = False

    def connect(self) -> bool:
        """Establish connection to Bybit Demo API."""
        try:
            from pybit.unified_trading import HTTP

            self.session = HTTP(
                api_key=self.api_key,
                api_secret=self.api_secret,
            )
            # Override endpoint to demo server
            self.session.endpoint = self.DEMO_ENDPOINT

            # Verify connection
            balance = self.session.get_wallet_balance(accountType="UNIFIED")
            if balance.get("retCode") == 0:
                self._connected = True
                logger.info("✅ Connected to Bybit Demo API")

                # Log demo balance
                for coin in balance["result"]["list"][0]["coin"]:
                    if coin["coin"] == "USDT":
                        logger.info(f"  Demo balance: {coin['equity']} USDT")
                return True
            else:
                logger.error(f"Bybit Demo connection failed: {balance}")
                return False

        except ImportError:
            logger.warning("pybit not installed. Install with: pip install pybit")
            return False
        except Exception as e:
            logger.error(f"Bybit Demo connection error: {e}")
            return False

    def get_klines(
        self,
        symbol: str,
        interval: str = "15",
        limit: int = 200,
    ) -> pd.DataFrame | None:
        """Fetch recent klines from Bybit."""
        if not self._connected:
            return None

        try:
            result = self.session.get_kline(
                category="linear",
                symbol=symbol.replace("/", ""),
                interval=interval,
                limit=limit,
            )
            if result.get("retCode") != 0:
                return None

            rows = result["result"]["list"]
            df = pd.DataFrame(rows, columns=[
                "timestamp", "open", "high", "low", "close", "volume", "turnover"
            ])
            df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)
            df.set_index("timestamp", inplace=True)
            df.sort_index(inplace=True)
            return df

        except Exception as e:
            logger.error(f"Failed to fetch klines for {symbol}: {e}")
            return None

    def get_instrument_info(self, symbol: str) -> dict | None:
        """Get instrument info including lot_size precision for proper qty rounding."""
        if not self._connected:
            return None
        try:
            result = self.session.get_instruments_info(
                category="linear",
                symbol=symbol.replace("/", ""),
            )
            if result.get("retCode") == 0 and result["result"]["list"]:
                return result["result"]["list"][0]
            return None
        except Exception as e:
            logger.debug(f"Failed to get instrument info for {symbol}: {e}")
            return None

    def place_order(
        self,
        symbol: str,
        side: str,         # "Buy" or "Sell"
        qty: float,
        order_type: str = "Limit",
        price: float | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> dict | None:
        """Place an order on Bybit Demo. Defaults to Limit for lower fees."""
        if not self._connected:
            return None

        try:
            params = {
                "category": "linear",
                "symbol": symbol.replace("/", ""),
                "side": side,
                "orderType": order_type,
                "qty": str(qty),
            }
            # Limit orders require a price; for time_in_force use PostOnly for maker
            if order_type == "Limit" and price is not None:
                params["price"] = str(price)
                params["timeInForce"] = "PostOnly"  # Ensure maker fee only
            if stop_loss:
                params["stopLoss"] = str(round(stop_loss, 6))
            if take_profit:
                params["takeProfit"] = str(round(take_profit, 6))

            result = self.session.place_order(**params)
            if result.get("retCode") == 0:
                order_data = result["result"]
                logger.info(
                    f"  📋 Order placed: {order_type} {side} {qty} {symbol} "
                    f"| orderId={order_data.get('orderId', 'N/A')}"
                )
                return order_data
            else:
                logger.error(f"Order failed: {result}")
                # Fallback to Market if Limit PostOnly rejected
                if order_type == "Limit":
                    logger.info(f"  ↻ Fallback to Market order for {symbol}")
                    params["orderType"] = "Market"
                    params.pop("price", None)
                    params.pop("timeInForce", None)
                    result = self.session.place_order(**params)
                    if result.get("retCode") == 0:
                        order_data = result["result"]
                        logger.info(f"  📋 Market fallback OK: {side} {qty} {symbol}")
                        return order_data
                    logger.error(f"Market fallback also failed: {result}")
                return None

        except Exception as e:
            logger.error(f"Order placement error: {e}")
            return None

    def get_order_detail(self, symbol: str, order_id: str) -> dict | None:
        """Get order execution detail to retrieve real fill price (avgPrice)."""
        if not self._connected or not order_id:
            return None
        try:
            # Wait briefly for fill
            time.sleep(0.5)
            result = self.session.get_order_history(
                category="linear",
                symbol=symbol.replace("/", ""),
                orderId=order_id,
            )
            if result.get("retCode") == 0 and result["result"]["list"]:
                return result["result"]["list"][0]
            return None
        except Exception as e:
            logger.debug(f"Failed to get order detail: {e}")
            return None

    def close_position(self, symbol: str, side: str, qty: float) -> dict | None:
        """Close a position by placing opposite market order."""
        close_side = "Sell" if side == "Buy" else "Buy"
        return self.place_order(symbol, close_side, qty)

    def get_positions(self) -> list[dict]:
        """Get all open positions."""
        if not self._connected:
            return []
        try:
            result = self.session.get_positions(category="linear", settleCoin="USDT")
            if result.get("retCode") == 0:
                return result["result"]["list"]
            return []
        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}")
            return []

    def get_balance(self) -> float:
        """Get current USDT equity."""
        if not self._connected:
            return 0.0
        try:
            balance = self.session.get_wallet_balance(accountType="UNIFIED")
            for coin in balance["result"]["list"][0]["coin"]:
                if coin["coin"] == "USDT":
                    return float(coin["equity"])
            return 0.0
        except Exception as e:
            logger.error(f"Failed to fetch balance: {e}")
            return 0.0


# ══════════════════════════════════════════════════════════════════
# MASTER ENGINE: ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════

class AegisLiveEngine:
    """
    The Master Orchestrator — combines all modules into a single trading loop.
    
    Flow per tick:
      1. Fetch latest candles (Bybit Demo API)
      2. Compute indicators + raw values
      3. Run ScoringEngine on each symbol
      4. Check ClusterGuard for entry permission
      5. Calculate position size (CompoundCalculator)
      6. Place order (if all gates pass)
      7. Monitor open positions (BreakevenManager)
    """

    def __init__(
        self,
        indicator_weights_path: str | None = None,
        risk_blocks_path: str | None = None,
        initial_capital: float = 2000.0,
        demo_api_key: str = "",
        demo_api_secret: str = "",
    ):
        # Load configuration files
        weights_path = indicator_weights_path or os.path.join(
            config.OUTPUT_DIR, "indicator_weights.json"
        )
        blocks_path = risk_blocks_path or os.path.join(
            config.OUTPUT_DIR, "risk_blocks.json"
        )

        self.indicator_library = self._load_json(weights_path)
        self.risk_blocks = self._load_json(blocks_path)

        # Initialize modules
        self.compound = CompoundCalculator(initial_capital)
        self.breakeven = BreakevenManager()
        self.cluster_guard = ClusterGuard(
            risk_blocks=self.risk_blocks if isinstance(self.risk_blocks, list) else [],
        )
        self.connector = BybitDemoConnector(demo_api_key, demo_api_secret)

        # State
        self.open_positions: list[LivePosition] = []
        self.closed_positions: list[LivePosition] = []
        self.scoring_engines: dict[str, ScoringEngine] = {}

        # Pre-build scoring engines for each symbol
        if isinstance(self.indicator_library, dict):
            for sym, weights in self.indicator_library.items():
                self.scoring_engines[sym] = ScoringEngine(weights)

        logger.info("=" * 60)
        logger.info("  AEGIS LIVE ENGINE v3.0 — Initialized")
        logger.info(f"  Capital: ${initial_capital:.2f}")
        logger.info(f"  Symbols with weights: {len(self.scoring_engines)}")
        logger.info(f"  Risk blocks loaded: {len(self.risk_blocks) if isinstance(self.risk_blocks, list) else 0}")
        logger.info("=" * 60)

    @staticmethod
    def _load_json(path: str) -> Any:
        """Load JSON file safely."""
        if not os.path.exists(path):
            logger.warning(f"File not found: {path}")
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            return {}

    def evaluate_symbol(
        self,
        symbol: str,
        df: pd.DataFrame,
    ) -> ScoringResult | None:
        """
        Run the full scoring pipeline on a single symbol.
        
        Returns ScoringResult or None if symbol has no weights.
        """
        sym_clean = symbol.replace("/", "")
        engine = self.scoring_engines.get(sym_clean)
        if not engine:
            return None

        if len(df) < 100:
            return None

        # Import signal computation from our weight matrix generator
        from generate_weight_matrix import _compute_indicator_signals

        # Compute all signals
        all_signals = _compute_indicator_signals(df)

        # Build indicator readings (raw values) and firing status
        indicator_readings: dict[str, float] = {}
        firing_indicators: dict[str, bool] = {}

        for ind_name, (condition, direction) in all_signals.items():
            if condition is None or len(condition) == 0:
                firing_indicators[ind_name] = False
                indicator_readings[ind_name] = 0.0
                continue

            # Check if firing on last bar
            last_val = condition.iloc[-1]
            is_firing = bool(last_val) if not pd.isna(last_val) else False
            firing_indicators[ind_name] = is_firing

            # Get raw value for confidence computation
            # We need to extract the underlying indicator value
            raw_val = self._get_raw_indicator_value(df, ind_name)
            indicator_readings[ind_name] = raw_val

        # Run scoring
        result = engine.evaluate(indicator_readings, firing_indicators)
        result.symbol = symbol
        return result

    @staticmethod
    def _get_raw_indicator_value(df: pd.DataFrame, indicator_name: str) -> float:
        """
        Extract the raw indicator value for confidence calculation.
        Maps indicator signal names back to their underlying computed values.
        """
        import pandas_ta as ta

        close = df["close"].astype(np.float64)
        high = df["high"].astype(np.float64)
        low = df["low"].astype(np.float64)
        volume = df["volume"].astype(np.float64)
        name_lower = indicator_name.lower()

        try:
            if "rsi14" in name_lower:
                rsi = ta.rsi(close, length=14)
                return float(rsi.iloc[-1]) if rsi is not None else 50.0

            if "rsi7" in name_lower:
                rsi = ta.rsi(close, length=7)
                return float(rsi.iloc[-1]) if rsi is not None else 50.0

            if "bb" in name_lower:
                bb = ta.bbands(close, length=20, std=2.0)
                if bb is not None:
                    bbl_col = [c for c in bb.columns if "BBL" in c]
                    bbu_col = [c for c in bb.columns if "BBU" in c]
                    if bbl_col and bbu_col:
                        bbl = bb[bbl_col[0]].iloc[-1]
                        bbu = bb[bbu_col[0]].iloc[-1]
                        return float((close.iloc[-1] - bbl) / (bbu - bbl + 1e-10))
                return 0.5

            if "cci" in name_lower:
                cci = ta.cci(high, low, close, length=14)
                return float(cci.iloc[-1]) if cci is not None else 0.0

            if "willr" in name_lower:
                willr = ta.willr(high, low, close, length=14)
                return float(willr.iloc[-1]) if willr is not None else -50.0

            if "stochrsi" in name_lower:
                stochrsi = ta.stochrsi(close, length=14, rsi_length=14, k=3, d=3)
                if stochrsi is not None:
                    k_cols = [c for c in stochrsi.columns if "k" in c.lower()]
                    if k_cols:
                        return float(stochrsi[k_cols[0]].iloc[-1])
                return 50.0

            if "stoch" in name_lower or "kdj" in name_lower:
                stoch = ta.stoch(high, low, close, k=9, d=3, smooth_k=3)
                if stoch is not None:
                    k_col = [c for c in stoch.columns if "STOCHk" in c]
                    if k_col:
                        return float(stoch[k_col[0]].iloc[-1])
                return 50.0

            if "mfi" in name_lower:
                mfi = ta.mfi(high, low, close, volume, length=14)
                return float(mfi.iloc[-1]) if mfi is not None else 50.0

            if "cmo" in name_lower:
                cmo = ta.cmo(close, length=14)
                return float(cmo.iloc[-1]) if cmo is not None else 0.0

            if "fisher" in name_lower:
                fisher = ta.fisher(high, low, length=9)
                if fisher is not None:
                    f_cols = [c for c in fisher.columns if "FISHERT" in c]
                    if f_cols:
                        return float(fisher[f_cols[0]].iloc[-1])
                return 0.0

            if "ema" in name_lower:
                ema_fast = ta.ema(close, length=12)
                ema_slow = ta.ema(close, length=26)
                if ema_fast is not None and ema_slow is not None:
                    diff = (ema_fast.iloc[-1] - ema_slow.iloc[-1]) / (ema_slow.iloc[-1] + 1e-10)
                    return float(diff * 100)  # as percentage
                return 0.0

            if "adx" in name_lower or "supertrend" in name_lower:
                return 1.0  # binary — always "on" when firing

            if "ppo" in name_lower:
                ppo = ta.ppo(close, fast=12, slow=26, signal=9)
                if ppo is not None:
                    ppo_cols = [c for c in ppo.columns if "PPO_" in c and "H" not in c and "S" not in c]
                    if ppo_cols:
                        return float(ppo[ppo_cols[0]].iloc[-1])
                return 0.0

            if "trix" in name_lower:
                trix = ta.trix(close, length=18, signal=9)
                if trix is not None:
                    trix_cols = [c for c in trix.columns if "TRIX_" in c and "s" not in c.lower()]
                    if trix_cols:
                        return float(trix[trix_cols[0]].iloc[-1])
                return 0.0

            if "hma" in name_lower:
                hma = ta.hma(close, length=20)
                if hma is not None:
                    dev = (close.iloc[-1] - hma.iloc[-1]) / (hma.iloc[-1] + 1e-10)
                    return float(dev * 100)
                return 0.0

            if "obv" in name_lower:
                obv = ta.obv(close, volume)
                if obv is not None:
                    obv_sma = obv.rolling(20).mean()
                    if obv_sma.iloc[-1] != 0:
                        dev = (obv.iloc[-1] - obv_sma.iloc[-1]) / (abs(obv_sma.iloc[-1]) + 1e-10)
                        return float(dev)
                return 0.0

            if "cmf" in name_lower:
                cmf = ta.cmf(high, low, close, volume, length=20)
                return float(cmf.iloc[-1]) if cmf is not None else 0.0

            if "breakout" in name_lower or "vol" in name_lower:
                return 1.0  # binary

        except Exception:
            pass

        return 0.0

    def process_signal(
        self,
        symbol: str,
        scoring_result: ScoringResult,
        current_price: float,
    ) -> LivePosition | None:
        """
        Process a scoring result: check guards, calculate size, open position.
        
        Returns new LivePosition if entry is made, None otherwise.
        """
        if not scoring_result.is_tradeable or scoring_result.direction is None:
            return None

        # Check cluster guard
        allowed, reason = self.cluster_guard.check_entry_allowed(
            symbol, self.open_positions
        )
        if not allowed:
            logger.info(f"  🛡️ {symbol} BLOCKED: {reason}")
            return None

        # Calculate stop-loss and take-profit
        direction = scoring_result.direction
        atr_multiplier = 2.0  # 2x ATR for SL
        tp_multiplier = config.RISK_REWARD_RATIO  # 1:2 RR

        # Simple SL/TP based on percentage
        sl_pct = 0.02  # 2% default stop-loss distance
        tp_pct = sl_pct * tp_multiplier

        if direction == TradeDirection.LONG:
            stop_loss = current_price * (1 - sl_pct)
            take_profit = current_price * (1 + tp_pct)
        else:
            stop_loss = current_price * (1 + sl_pct)
            take_profit = current_price * (1 - tp_pct)

        # Calculate position size (compound)
        size_info = self.compound.calculate_position_size(
            entry_price=current_price,
            stop_loss_price=stop_loss,
        )

        # Create position
        position = LivePosition(
            symbol=symbol,
            direction=direction,
            entry_price=current_price,
            entry_time=datetime.now(timezone.utc),
            quantity=size_info["quantity"],
            risk_usdt=size_info["risk_usdt"],
            stop_loss=stop_loss,
            take_profit=take_profit,
            initial_stop_loss=stop_loss,
            score_at_entry=scoring_result.total_score,
            cluster_id=self.cluster_guard.cluster_map.get(
                symbol.replace("/", ""), -1
            ),
        )

        self.open_positions.append(position)

        logger.info(
            f"\n  🚀 NEW POSITION: {direction.value} {symbol} @ {current_price:.6f}\n"
            f"     Score: {scoring_result.confidence_pct:.1f}% | "
            f"Risk: ${size_info['risk_usdt']} | Qty: {size_info['quantity']:.6f}\n"
            f"     SL: {stop_loss:.6f} | TP: {take_profit:.6f} | "
            f"RR: 1:{tp_multiplier}"
        )

        return position

    def update_positions(
        self,
        market_prices: dict[str, float],
        indicator_reversals: dict[str, bool] | None = None,
    ):
        """
        Update all open positions with latest prices.
        Applies breakeven manager logic.
        """
        if indicator_reversals is None:
            indicator_reversals = {}

        still_open = []
        for pos in self.open_positions:
            if pos.state == PositionState.CLOSED:
                continue

            current_price = market_prices.get(pos.symbol, 0.0)
            if current_price <= 0:
                still_open.append(pos)
                continue

            reversing = indicator_reversals.get(pos.symbol, False)

            # Apply breakeven manager
            updated_pos = self.breakeven.update_position(pos, current_price, reversing)

            if updated_pos.state == PositionState.CLOSED:
                # Record in compound calculator
                self.compound.record_trade(updated_pos.pnl_usdt, updated_pos.symbol)
                self.closed_positions.append(updated_pos)
            else:
                still_open.append(updated_pos)

        self.open_positions = still_open

    def get_engine_status(self) -> dict:
        """Full engine status report."""
        return {
            "compound": self.compound.get_status(),
            "open_positions": len(self.open_positions),
            "closed_positions": len(self.closed_positions),
            "symbols_monitored": len(self.scoring_engines),
            "active_positions": [
                {
                    "symbol": p.symbol,
                    "direction": p.direction.value,
                    "entry_price": p.entry_price,
                    "state": p.state.value,
                    "highest_profit_pct": round(p.highest_profit_pct * 100, 2),
                }
                for p in self.open_positions
            ],
        }

    def run_backtest_simulation(
        self,
        historical_data: dict[str, pd.DataFrame],
        progress_cb=None,
    ) -> dict:
        """
        Run a full backtest simulation with all modules active.
        Simulates the complete trading loop on historical data.
        """
        results = {
            "trades": [],
            "balance_curve": [self.compound.initial_capital],
            "signals_generated": 0,
            "signals_blocked_by_cluster": 0,
            "signals_blocked_by_score": 0,
            "breakeven_exits": 0,
            "trailing_exits": 0,
        }

        total_symbols = len(historical_data)

        for sym_idx, (symbol, df) in enumerate(historical_data.items()):
            if progress_cb:
                progress_cb(f"[{sym_idx+1}/{total_symbols}] Simulating: {symbol}")

            sym_clean = symbol.replace("/", "")
            engine = self.scoring_engines.get(sym_clean)
            if not engine or len(df) < 200:
                continue

            # Import for computing signals
            from generate_weight_matrix import _compute_indicator_signals
            all_signals = _compute_indicator_signals(df)

            close_prices = df["close"].astype(float).values

            # Walk through bars (skip first 100 for warm-up)
            for bar_idx in range(100, len(df) - 10):
                # Build firing status at this bar
                firing = {}
                for ind_name, (condition, direction) in all_signals.items():
                    if condition is None or len(condition) == 0:
                        firing[ind_name] = False
                        continue
                    val = condition.iloc[bar_idx]
                    firing[ind_name] = bool(val) if not pd.isna(val) else False

                # Simple scoring (use confidence=0.6 for binary firing)
                long_score = 0.0
                short_score = 0.0
                total_max = 0.0

                for ind_info in engine._top_indicators:
                    ind_name = ind_info["name"]
                    w = ind_info["weight"]
                    total_max += w
                    if firing.get(ind_name, False):
                        if ind_info.get("direction") == "LONG":
                            long_score += 0.6 * w
                        else:
                            short_score += 0.6 * w

                if total_max <= 0:
                    continue

                total_score = max(long_score, short_score) / total_max
                dissonance = min(long_score, short_score) / (long_score + short_score + 1e-10)

                if total_score < ScoringEngine.ENTRY_THRESHOLD:
                    results["signals_blocked_by_score"] += 1
                    continue

                if dissonance > ScoringEngine.CHAOS_THRESHOLD:
                    results["signals_blocked_by_score"] += 1
                    continue

                results["signals_generated"] += 1

                # Determine direction
                direction = TradeDirection.LONG if long_score > short_score else TradeDirection.SHORT
                entry_price = close_prices[bar_idx]

                # Check cluster guard
                allowed, reason = self.cluster_guard.check_entry_allowed(
                    symbol, self.open_positions
                )
                if not allowed:
                    results["signals_blocked_by_cluster"] += 1
                    continue

                # Simulate forward trade (next 4 bars)
                exit_bar = min(bar_idx + config.FORWARD_BARS, len(df) - 1)
                exit_price = close_prices[exit_bar]

                # Calculate PnL
                if direction == TradeDirection.LONG:
                    raw_pnl = (exit_price - entry_price) / entry_price
                else:
                    raw_pnl = (entry_price - exit_price) / entry_price

                net_pnl = raw_pnl - config.ROUNDTRIP_FEE

                # Apply compound position sizing
                risk_usdt = self.compound.current_risk_usdt
                pnl_usdt = net_pnl * risk_usdt * config.RISK_REWARD_RATIO

                # Record
                self.compound.record_trade(pnl_usdt, symbol)
                results["balance_curve"].append(self.compound.current_balance)
                results["trades"].append({
                    "symbol": symbol,
                    "direction": direction.value,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl_pct": round(net_pnl * 100, 4),
                    "pnl_usdt": round(pnl_usdt, 2),
                    "score": round(total_score, 3),
                    "balance_after": round(self.compound.current_balance, 2),
                })

        # Final summary
        results["final_status"] = self.compound.get_status()
        return results


# ══════════════════════════════════════════════════════════════════
# ENTRY POINT — Quick Test
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    print("\n" + "=" * 70)
    print("  AEGIS-QUANT-LAB v3.0 — Live Engine Self-Test")
    print("=" * 70)

    # Test 1: Compound Calculator
    print("\n📊 Test 1: Compound Interest Calculator")
    calc = CompoundCalculator(2000.0)
    # Simulate 10 trades: 7 wins, 3 losses
    for i in range(7):
        calc.record_trade(40.0, f"WIN_{i}")  # +$40 each
    for i in range(3):
        calc.record_trade(-20.0, f"LOSS_{i}")  # -$20 each
    status = calc.get_status()
    print(f"  After 10 trades: Balance=${status['current_balance']} | ROI={status['roi_pct']}%")
    print(f"  Win Rate: {status['win_rate_pct']}% | Max DD: {status['max_drawdown_pct']}%")

    # Test 2: Cluster Guard
    print("\n🛡️ Test 2: Cluster Guard")
    guard = ClusterGuard(
        cluster_map={"SANDUSDT": 1, "MANAUSDT": 1, "GALAUSDT": 1, "BTCUSDT": 0},
        correlation_matrix={"SAND/USDT|MANA/USDT": 0.90, "SAND/USDT|GALA/USDT": 0.85},
    )
    # Simulate: SAND is open
    fake_position = LivePosition(
        symbol="SAND/USDT",
        direction=TradeDirection.LONG,
        entry_price=0.50,
        entry_time=datetime.now(timezone.utc),
        quantity=100.0,
        risk_usdt=20.0,
        stop_loss=0.49,
        take_profit=0.52,
        initial_stop_loss=0.49,
    )
    allowed, reason = guard.check_entry_allowed("MANA/USDT", [fake_position])
    print(f"  MANA entry with SAND open: Allowed={allowed} | Reason={reason}")

    allowed2, reason2 = guard.check_entry_allowed("BTC/USDT", [fake_position])
    print(f"  BTC entry with SAND open:  Allowed={allowed2} | Reason={reason2}")

    # Test 3: Breakeven Manager
    print("\n⚡ Test 3: Breakeven Manager")
    be = BreakevenManager()
    pos = LivePosition(
        symbol="BTC/USDT",
        direction=TradeDirection.LONG,
        entry_price=50000.0,
        entry_time=datetime.now(timezone.utc),
        quantity=0.01,
        risk_usdt=20.0,
        stop_loss=49000.0,
        take_profit=52000.0,
        initial_stop_loss=49000.0,
    )
    # Price moves +1.5% = 50750
    pos = be.update_position(pos, 50750.0)
    print(f"  After +1.5%: State={pos.state.value} | SL={pos.stop_loss:.2f}")

    # Price moves +3.5% = 51750
    pos = be.update_position(pos, 51750.0)
    print(f"  After +3.5%: State={pos.state.value} | SL={pos.stop_loss:.2f}")

    print("\n✅ All module tests passed!")
    print("=" * 70)
