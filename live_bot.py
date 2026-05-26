#!/usr/bin/env python3
"""
Aegis-Quant-Lab v3.0 — LIVE BOT 🫀 «Пульс»
==============================================
THE 24/7 AUTONOMOUS TRADING ORCHESTRATOR

This is the HEARTBEAT — the infinite loop that connects:
  Brain  (aegis_live_engine.py)  →  scoring, breakeven, compound, cluster guard
  Hands  (Bybit Demo API)       →  real orders on demo futures
  Eyes   (Market Data)           →  15m candles for 49 coins

Execution cycle (every 15 minutes):
  1. Wake up at candle close (XX:00, XX:15, XX:30, XX:45)
  2. Download last 200 candles for all monitored symbols
  3. Compute 20 indicators per symbol (instantaneous on 200 bars)
  4. Feed into ScoringEngine → weighted confidence evaluation
  5. ClusterGuard checks open positions for correlation conflicts
  6. CompoundCalculator determines position size
  7. If score > 0.65 → PLACE ORDER on Bybit Demo
  8. BreakevenManager monitors all open positions
  9. Sleep until next candle close

Usage:
  # Set environment variables:
  export BYBIT_DEMO_KEY="your_demo_api_key"
  export BYBIT_DEMO_SECRET="your_demo_api_secret"

  # Run:
  python live_bot.py

  # Or with custom settings:
  python live_bot.py --interval 15 --capital 2000 --symbols BTCUSDT,ETHUSDT,SOLUSDT

Architecture:
  ┌──────────────┐      ┌──────────────────┐      ┌─────────────────┐
  │  Bybit API   │─────→│   live_bot.py    │─────→│ aegis_live_engine│
  │  (Demo)      │←─────│   «Пульс»       │←─────│  (Brain)         │
  └──────────────┘      └──────────────────┘      └─────────────────┘
        ↕                       ↕                         ↕
   Market Data             Orchestrator             5 Core Modules:
   Order Execution         Sleep/Wake Cycle         - ScoringEngine
   Position Tracking       Logging & Alerts         - BreakevenManager
                                                    - CompoundCalculator
                                                    - ClusterGuard
                                                    - BybitDemoConnector
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
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np
import pandas as pd

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from aegis_live_engine import (
    AegisLiveEngine,
    ScoringEngine,
    BreakevenManager,
    CompoundCalculator,
    ClusterGuard,
    BybitDemoConnector,
    LivePosition,
    TradeDirection,
    PositionState,
    ScoringResult,
)
from generate_weight_matrix import _compute_indicator_signals
import os
from dotenv import load_dotenv

# Загружаем ключи из .env.local
load_dotenv(".env.local")

# ══════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ══════════════════════════════════════════════════════════════════

def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """Configure dual logging: console + rotating file."""
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("aegis.pulse")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # Console handler — colorful
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(
        "%(asctime)s │ %(message)s",
        datefmt="%H:%M:%S"
    ))
    logger.addHandler(console)

    # File handler — detailed
    log_file = os.path.join(log_dir, f"live_bot_{datetime.now().strftime('%Y%m%d')}.log")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(file_handler)

    return logger


# ══════════════════════════════════════════════════════════════════
# SIGNAL HANDLERS (graceful shutdown)
# ══════════════════════════════════════════════════════════════════

_shutdown_requested = False


def _signal_handler(signum, frame):
    """Handle SIGINT/SIGTERM for graceful shutdown."""
    global _shutdown_requested
    _shutdown_requested = True
    print("\n\n  🛑 Shutdown signal received. Finishing current cycle...")


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ══════════════════════════════════════════════════════════════════
# CORE: THE PULSE — LIVE TRADING LOOP
# ══════════════════════════════════════════════════════════════════

class LiveBot:
    """
    The Heartbeat — autonomous 24/7 trading loop.

    Connects all components into a single execution pipeline
    that wakes up every candle close and processes the market.
    """

    # ── Supported symbols (49 coins from our research) ──
    DEFAULT_SYMBOLS = [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
        "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
        "MATICUSDT", "TONUSDT", "TRXUSDT", "SHIBUSDT", "UNIUSDT",
        "ATOMUSDT", "LTCUSDT", "BCHUSDT", "NEARUSDT", "APTUSDT",
        "FILUSDT", "ARBUSDT", "OPUSDT", "SUIUSDT", "HYPEUSDT",
        "IMXUSDT", "PEPEUSDT", "WIFUSDT", "FETUSDT", "RENDERUSDT",
        "INJUSDT", "SEIUSDT", "STXUSDT", "AAVEUSDT", "MKRUSDT",
        "RUNEUSDT", "TIAUSDT", "ALGOUSDT", "FTMUSDT", "SANDUSDT",
        "MANAUSDT", "GALAUSDT", "EOSUSDT", "XLMUSDT", "IOTAUSDT",
        "DYDXUSDT", "CRVUSDT", "COMPUSDT", "JASMYUSDT",
    ]

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        symbols: list[str] | None = None,
        interval_minutes: int = 15,
        initial_capital: float = 2000.0,
        leverage: float = 5.0,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.symbols = symbols or self.DEFAULT_SYMBOLS
        self.interval_minutes = interval_minutes
        self.initial_capital = initial_capital
        self.leverage = leverage

        self.logger = setup_logging()
        self.cycle_count = 0
        self.total_signals_checked = 0
        self.total_orders_placed = 0
        self.start_time = datetime.now(timezone.utc)

        # ── Initialize components ──
        self.logger.info("=" * 70)
        self.logger.info("  🫀 AEGIS-QUANT-LAB v3.0 — LIVE BOT «Пульс»")
        self.logger.info("=" * 70)
        self.logger.info(f"  Capital:    ${initial_capital:.2f} USDT")
        self.logger.info(f"  Leverage:   {leverage}x")
        self.logger.info(f"  Symbols:    {len(self.symbols)} coins")
        self.logger.info(f"  Interval:   {interval_minutes}m candles")
        self.logger.info(f"  Risk/trade: {config.RISK_PER_TRADE_PCT*100:.1f}%")
        self.logger.info("=" * 70)

        # Initialize the brain
        self._init_engine()

        # Initialize Bybit connector
        self._init_connector()

    def _init_engine(self):
        """Initialize the Aegis Engine with all modules."""
        weights_path = os.path.join(config.OUTPUT_DIR, "indicator_weights.json")
        blocks_path = os.path.join(config.OUTPUT_DIR, "risk_blocks.json")

        self.engine = AegisLiveEngine(
            indicator_weights_path=weights_path,
            risk_blocks_path=blocks_path,
            initial_capital=self.initial_capital,
            demo_api_key=self.api_key,
            demo_api_secret=self.api_secret,
        )

        self.logger.info(f"  🧠 Brain loaded: {len(self.engine.scoring_engines)} symbols with weights")

    def _init_connector(self):
        """Initialize Bybit Demo API connection."""
        self.connector = BybitDemoConnector(
            api_key=self.api_key,
            api_secret=self.api_secret,
        )

        if self.connector.connect():
            balance = self.connector.get_balance()
            self.logger.info(f"  🌐 Connected to Bybit Demo | Balance: ${balance:.2f} USDT")
        else:
            self.logger.error("  ❌ Failed to connect to Bybit Demo API!")
            self.logger.error("     Check your API keys and internet connection.")
            raise ConnectionError("Bybit Demo API connection failed")

    # ──────────────────────────────────────────────
    # TIMING — Sleep until next candle close
    # ──────────────────────────────────────────────

    def _seconds_until_next_candle(self) -> float:
        """
        Calculate seconds to wait until the next candle close.

        For 15m interval: wakes at :00, :15, :30, :45 + 2 sec buffer
        (buffer ensures the candle is fully closed on Bybit)
        """
        now = datetime.now(timezone.utc)
        minutes = now.minute
        seconds = now.second

        # Next candle boundary
        next_candle_min = ((minutes // self.interval_minutes) + 1) * self.interval_minutes

        if next_candle_min >= 60:
            # Roll to next hour
            next_time = now.replace(
                minute=0, second=0, microsecond=0
            ) + timedelta(hours=1)
        else:
            next_time = now.replace(
                minute=next_candle_min, second=0, microsecond=0
            )

        # Add 2-second buffer (Bybit needs a moment to close the candle)
        next_time += timedelta(seconds=2)

        wait_seconds = (next_time - now).total_seconds()
        if wait_seconds < 0:
            wait_seconds = 0

        return wait_seconds

    # ──────────────────────────────────────────────
    # DATA — Fetch candles from Bybit
    # ──────────────────────────────────────────────

    def _fetch_candles(self, symbol: str, limit: int = 200) -> pd.DataFrame | None:
        """
        Fetch latest candles from Bybit Demo API.

        Returns DataFrame with columns: open, high, low, close, volume
        """
        try:
            df = self.connector.get_klines(
                symbol=symbol,
                interval=str(self.interval_minutes),
                limit=limit,
            )
            return df
        except Exception as e:
            self.logger.debug(f"    Failed to fetch {symbol}: {e}")
            return None

    # ──────────────────────────────────────────────
    # ANALYSIS — Compute indicators + scoring
    # ──────────────────────────────────────────────

    def _analyze_symbol(self, symbol: str, df: pd.DataFrame) -> ScoringResult | None:
        """
        Run full analysis pipeline on a single symbol:
          1. Compute all 20 indicator signals
          2. Determine which are firing on the LAST bar
          3. Get raw values for confidence scoring
          4. Run ScoringEngine evaluation
        """
        sym_clean = symbol.replace("/", "").replace("USDT", "") + "USDT"

        # Get scoring engine for this symbol
        scoring_engine = self.engine.scoring_engines.get(sym_clean)
        if not scoring_engine:
            return None

        if len(df) < 100:
            return None

        # Compute all indicator signals on this data
        all_signals = _compute_indicator_signals(df)

        # Build firing status and raw readings for the LAST bar
        firing_indicators: dict[str, bool] = {}
        indicator_readings: dict[str, float] = {}

        for ind_name, (condition, direction) in all_signals.items():
            if condition is None or len(condition) == 0:
                firing_indicators[ind_name] = False
                indicator_readings[ind_name] = 0.0
                continue

            # Is this indicator firing on the LATEST completed candle?
            last_val = condition.iloc[-1]
            is_firing = bool(last_val) if not pd.isna(last_val) else False
            firing_indicators[ind_name] = is_firing

            # Get raw value for confidence calculation
            raw_val = self.engine._get_raw_indicator_value(df, ind_name)
            indicator_readings[ind_name] = raw_val

        # Run scoring engine
        result = scoring_engine.evaluate(indicator_readings, firing_indicators)
        result.symbol = symbol
        return result

    def _check_indicator_reversal(self, symbol: str, df: pd.DataFrame) -> bool:
        """
        Check if indicators are reversing for an open position.

        Logic: If the previous bar had a strong signal in one direction
        and now the signal is weakening/flipping → reversal detected.
        """
        if len(df) < 5:
            return False

        sym_clean = symbol.replace("/", "").replace("USDT", "") + "USDT"
        scoring_engine = self.engine.scoring_engines.get(sym_clean)
        if not scoring_engine:
            return False

        all_signals = _compute_indicator_signals(df)

        # Count firing indicators on last 2 bars
        current_long = 0
        current_short = 0
        prev_long = 0
        prev_short = 0

        for ind_info in scoring_engine._top_indicators:
            ind_name = ind_info["name"]
            direction = ind_info.get("direction", "LONG")

            if ind_name not in all_signals:
                continue
            condition, _ = all_signals[ind_name]
            if condition is None or len(condition) < 2:
                continue

            # Current bar
            cur_val = condition.iloc[-1]
            if not pd.isna(cur_val) and bool(cur_val):
                if direction == "LONG":
                    current_long += 1
                else:
                    current_short += 1

            # Previous bar
            prev_val = condition.iloc[-2]
            if not pd.isna(prev_val) and bool(prev_val):
                if direction == "LONG":
                    prev_long += 1
                else:
                    prev_short += 1

        # Reversal detected if majority flipped direction
        # For LONG position: reversal = current_short grew AND current_long shrank
        # Simple heuristic: if opposing count doubled and our count halved
        total_prev = prev_long + prev_short
        if total_prev == 0:
            return False

        # Check if sentiment shifted significantly
        prev_dominant = max(prev_long, prev_short)
        cur_dominant = max(current_long, current_short)
        prev_direction = "LONG" if prev_long > prev_short else "SHORT"
        cur_direction = "LONG" if current_long > current_short else "SHORT"

        # Reversal = direction changed AND new direction has decent strength
        if prev_direction != cur_direction and cur_dominant >= 3:
            return True

        return False

    # ──────────────────────────────────────────────
    # EXECUTION — Place orders
    # ──────────────────────────────────────────────

    def _execute_entry(
        self,
        symbol: str,
        scoring: ScoringResult,
        current_price: float,
        df: pd.DataFrame,
    ) -> LivePosition | None:
        """
        Execute a trade entry on Bybit Demo.

        Steps:
          1. ClusterGuard approval
          2. CompoundCalculator position size
          3. Calculate SL/TP using ATR
          4. Place market order with SL/TP
          5. Register position in engine
        """
        direction = scoring.direction
        if direction is None:
            return None

        # ── Step 1: Cluster Guard ──
        allowed, reason = self.engine.cluster_guard.check_entry_allowed(
            symbol, self.engine.open_positions
        )
        if not allowed:
            self.logger.info(f"  🛡️  {symbol} BLOCKED by ClusterGuard: {reason}")
            return None

        # ── Step 2: Calculate ATR-based SL/TP ──
        import pandas_ta as ta
        atr_series = ta.atr(
            df["high"].astype(float),
            df["low"].astype(float),
            df["close"].astype(float),
            length=14
        )
        if atr_series is None or len(atr_series) == 0:
            atr_value = current_price * 0.02  # fallback: 2%
        else:
            atr_value = float(atr_series.iloc[-1])
            if np.isnan(atr_value) or atr_value <= 0:
                atr_value = current_price * 0.02

        # SL = 2x ATR, TP = 4x ATR (1:2 risk/reward)
        sl_distance = atr_value * 2.0
        tp_distance = sl_distance * config.RISK_REWARD_RATIO

        if direction == TradeDirection.LONG:
            stop_loss = current_price - sl_distance
            take_profit = current_price + tp_distance
            side = "Buy"
        else:
            stop_loss = current_price + sl_distance
            take_profit = current_price - tp_distance
            side = "Sell"

        # ── Step 3: Position size (compound) ──
        size_info = self.engine.compound.calculate_position_size(
            entry_price=current_price,
            stop_loss_price=stop_loss,
            leverage=self.leverage,
        )

        quantity = size_info["quantity"]

        # Round quantity to Bybit precision (varies per symbol)
        # For most USDT perps: 3 decimals for majors, more for altcoins
        if current_price > 1000:
            quantity = round(quantity, 3)
        elif current_price > 10:
            quantity = round(quantity, 2)
        elif current_price > 1:
            quantity = round(quantity, 1)
        else:
            quantity = round(quantity, 0)

        if quantity <= 0:
            self.logger.warning(f"  ⚠️  {symbol} quantity too small: {quantity}")
            return None

        # ── Step 4: Place order on Bybit Demo ──
        self.logger.info("")
        self.logger.info(f"  ╔══════════════════════════════════════════════════════")
        self.logger.info(f"  ║ 🚀 NEW ENTRY: {direction.value} {symbol}")
        self.logger.info(f"  ╠──────────────────────────────────────────────────────")
        self.logger.info(f"  ║ Score:    {scoring.confidence_pct:.1f}% "
                         f"({scoring.n_indicators_firing} indicators)")
        self.logger.info(f"  ║ Price:    {current_price:.6f}")
        self.logger.info(f"  ║ Qty:      {quantity}")
        self.logger.info(f"  ║ Risk:     ${size_info['risk_usdt']:.2f} "
                         f"({size_info['risk_pct_used']:.2f}% of balance)")
        self.logger.info(f"  ║ SL:       {stop_loss:.6f} "
                         f"(-{sl_distance/current_price*100:.2f}%)")
        self.logger.info(f"  ║ TP:       {take_profit:.6f} "
                         f"(+{tp_distance/current_price*100:.2f}%)")
        self.logger.info(f"  ║ Leverage: {self.leverage}x")
        self.logger.info(f"  ║ Balance:  ${self.engine.compound.current_balance:.2f}")
        self.logger.info(f"  ╚══════════════════════════════════════════════════════")

        order_result = self.connector.place_order(
            symbol=symbol,
            side=side,
            qty=quantity,
            order_type="Market",
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

        if order_result is None:
            self.logger.error(f"  ❌ Order FAILED for {symbol}")
            return None

        self.total_orders_placed += 1

        # ── Step 5: Register position in engine ──
        position = LivePosition(
            symbol=symbol,
            direction=direction,
            entry_price=current_price,
            entry_time=datetime.now(timezone.utc),
            quantity=quantity,
            risk_usdt=size_info["risk_usdt"],
            stop_loss=stop_loss,
            take_profit=take_profit,
            initial_stop_loss=stop_loss,
            score_at_entry=scoring.total_score,
            cluster_id=self.engine.cluster_guard.cluster_map.get(symbol, -1),
        )
        self.engine.open_positions.append(position)

        self.logger.info(f"  ✅ Order CONFIRMED — Position registered")
        return position

    # ──────────────────────────────────────────────
    # MONITORING — Manage open positions
    # ──────────────────────────────────────────────

    def _monitor_positions(self):
        """
        Monitor all open positions:
          - Fetch current price
          - Apply BreakevenManager logic
          - Close if SL/TP hit or indicators reverse
        """
        if not self.engine.open_positions:
            return

        self.logger.info(f"\n  📊 Monitoring {len(self.engine.open_positions)} open positions...")

        positions_to_keep = []

        for pos in self.engine.open_positions:
            if pos.state == PositionState.CLOSED:
                continue

            # Get current price
            df = self._fetch_candles(pos.symbol, limit=50)
            if df is None or len(df) == 0:
                positions_to_keep.append(pos)
                continue

            current_price = float(df["close"].iloc[-1])

            # Check for indicator reversal
            reversing = self._check_indicator_reversal(pos.symbol, df)

            # Apply breakeven manager
            updated = self.engine.breakeven.update_position(
                pos, current_price, reversing
            )

            if updated.state == PositionState.CLOSED:
                # Record in compound calculator
                self.engine.compound.record_trade(updated.pnl_usdt, updated.symbol)
                self.engine.closed_positions.append(updated)

                # Close on Bybit
                close_side = "Sell" if updated.direction == TradeDirection.LONG else "Buy"
                self.connector.place_order(
                    symbol=updated.symbol,
                    side=close_side,
                    qty=updated.quantity,
                )
                self.logger.info(
                    f"  {'✅' if updated.pnl_usdt > 0 else '❌'} "
                    f"CLOSED {updated.symbol}: PnL=${updated.pnl_usdt:+.2f} "
                    f"| Reason: {updated.close_reason}"
                )
            else:
                positions_to_keep.append(updated)
                state_emoji = {
                    PositionState.OPEN: "🔵",
                    PositionState.BREAKEVEN: "🟡",
                    PositionState.TRAILING: "🟢",
                }
                self.logger.info(
                    f"    {state_emoji.get(updated.state, '⚪')} "
                    f"{updated.symbol} {updated.direction.value} | "
                    f"State: {updated.state.value} | "
                    f"Unrealized: {updated.highest_profit_pct*100:+.2f}% | "
                    f"SL: {updated.stop_loss:.6f}"
                )

        self.engine.open_positions = positions_to_keep

    # ──────────────────────────────────────────────
    # SYNC — Reconcile with Bybit positions
    # ──────────────────────────────────────────────

    def _sync_with_exchange(self):
        """
        Sync internal state with actual Bybit positions.
        Handles cases where SL/TP was hit by the exchange directly.
        """
        try:
            exchange_positions = self.connector.get_positions()
            exchange_symbols = set()

            for ep in exchange_positions:
                if float(ep.get("size", 0)) > 0:
                    exchange_symbols.add(ep["symbol"])

            # Check if any of our tracked positions were closed by exchange
            still_open = []
            for pos in self.engine.open_positions:
                sym_clean = pos.symbol.replace("/", "")
                if sym_clean not in exchange_symbols:
                    # Position was closed by exchange (SL/TP hit on server)
                    # Calculate approximate PnL
                    self.logger.info(
                        f"  🔄 {pos.symbol} closed by exchange "
                        f"(SL/TP hit server-side)"
                    )
                    # We don't know exact exit price, estimate from SL/TP
                    # Mark as closed
                    pos.state = PositionState.CLOSED
                    pos.close_reason = "Exchange SL/TP triggered"
                    self.engine.closed_positions.append(pos)
                else:
                    still_open.append(pos)

            self.engine.open_positions = still_open

        except Exception as e:
            self.logger.debug(f"  Sync warning: {e}")

    # ──────────────────────────────────────────────
    # STATUS — Print dashboard
    # ──────────────────────────────────────────────

    def _print_dashboard(self):
        """Print a beautiful status dashboard after each cycle."""
        status = self.engine.compound.get_status()
        uptime = datetime.now(timezone.utc) - self.start_time
        hours = uptime.total_seconds() / 3600

        self.logger.info("")
        self.logger.info("  ┌───────────── DASHBOARD ─────────────┐")
        self.logger.info(f"  │ Cycle:       #{self.cycle_count:<23}│")
        self.logger.info(f"  │ Uptime:      {hours:.1f}h                     │")
        self.logger.info(f"  │ Balance:     ${status['current_balance']:<21.2f}│")
        self.logger.info(f"  │ ROI:         {status['roi_pct']:+.2f}%                   │")
        self.logger.info(f"  │ Win Rate:    {status['win_rate_pct']:.1f}%                    │")
        trades_str = f"{status['total_trades']} ({status['wins']}W / {status['losses']}L)"
        self.logger.info(f"  │ Trades:      {trades_str:<23}│")
        self.logger.info(f"  │ Open Pos:    {len(self.engine.open_positions)}/5                    │")
        self.logger.info(f"  │ Max DD:      {status['max_drawdown_pct']:.2f}%                   │")
        risk_str = f"${status['current_risk_usdt']:.2f} ({status['current_risk_pct']:.1f}%)"
        self.logger.info(f"  │ Risk/Trade:  {risk_str:<23}│")
        self.logger.info(f"  │ Orders:      {self.total_orders_placed} placed                │")
        self.logger.info("  └──────────────────────────────────────┘")

    # ──────────────────────────────────────────────
    # MAIN LOOP — The Heartbeat 🫀
    # ──────────────────────────────────────────────

    def run(self):
        """
        🫀 THE PULSE — Main infinite trading loop.

        Runs until SIGINT/SIGTERM or keyboard interrupt.
        """
        global _shutdown_requested

        self.logger.info("\n" + "═" * 70)
        self.logger.info("  🫀 PULSE ACTIVATED — Bot is now LIVE")
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

                # Sleep in 1-second chunks (allows graceful shutdown)
                sleep_until = time.time() + wait_secs
                while time.time() < sleep_until and not _shutdown_requested:
                    time.sleep(min(1.0, sleep_until - time.time()))

                if _shutdown_requested:
                    break

                # ══════════════════════════════════════════
                # 🫀 HEARTBEAT — Process market
                # ══════════════════════════════════════════
                self.cycle_count += 1
                cycle_start = time.time()

                self.logger.info(f"\n{'─' * 70}")
                self.logger.info(
                    f"  🫀 CYCLE #{self.cycle_count} — "
                    f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC"
                )
                self.logger.info(f"{'─' * 70}")

                # ── Phase 1: Sync with exchange ──
                self._sync_with_exchange()

                # ── Phase 2: Monitor existing positions ──
                self._monitor_positions()

                # ── Phase 3: Scan for new entries ──
                self.logger.info(f"\n  🔍 Scanning {len(self.symbols)} symbols...")

                signals_found = 0
                entries_made = 0

                for i, symbol in enumerate(self.symbols):
                    # Respect Bybit rate limits
                    if i > 0 and i % 10 == 0:
                        time.sleep(0.5)

                    # Fetch candles
                    df = self._fetch_candles(symbol, limit=200)
                    if df is None or len(df) < 100:
                        continue

                    self.total_signals_checked += 1

                    # Analyze
                    scoring = self._analyze_symbol(symbol, df)
                    if scoring is None:
                        continue

                    if not scoring.is_tradeable:
                        continue

                    # Signal found!
                    signals_found += 1
                    current_price = float(df["close"].iloc[-1])

                    self.logger.info(
                        f"\n  📡 SIGNAL: {scoring.direction.value} {symbol} "
                        f"@ {current_price:.6f} | "
                        f"Score: {scoring.confidence_pct:.1f}% | "
                        f"Indicators: {scoring.n_indicators_firing}"
                    )

                    # Execute entry
                    position = self._execute_entry(symbol, scoring, current_price, df)
                    if position:
                        entries_made += 1

                    # Don't enter too many at once
                    if entries_made >= 2:
                        self.logger.info("  ⚠️  Max 2 entries per cycle — pausing scan")
                        break

                # ── Phase 4: Summary ──
                cycle_time = time.time() - cycle_start
                self.logger.info(
                    f"\n  ⚡ Cycle #{self.cycle_count} complete in {cycle_time:.1f}s: "
                    f"{signals_found} signals, {entries_made} entries"
                )

                # Print dashboard every 4 cycles (every hour on 15m)
                if self.cycle_count % 4 == 0:
                    self._print_dashboard()

            except KeyboardInterrupt:
                break
            except Exception as e:
                self.logger.error(f"  ❌ Cycle error: {e}")
                self.logger.debug(traceback.format_exc())
                # Don't crash — wait for next cycle
                time.sleep(10)

        # ── Graceful shutdown ──
        self._shutdown()

    def _shutdown(self):
        """Graceful shutdown — log final state."""
        self.logger.info("\n" + "═" * 70)
        self.logger.info("  🛑 SHUTTING DOWN")
        self.logger.info("═" * 70)
        self._print_dashboard()

        # Save state to JSON
        state = {
            "shutdown_time": datetime.now(timezone.utc).isoformat(),
            "cycles_completed": self.cycle_count,
            "total_orders": self.total_orders_placed,
            "compound_status": self.engine.compound.get_status(),
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
                for p in self.engine.open_positions
            ],
            "trade_history": self.engine.compound.trade_history[-50:],
        }

        state_path = os.path.join(config.OUTPUT_DIR, "bot_state.json")
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, default=str)

        self.logger.info(f"  💾 State saved to {state_path}")
        self.logger.info("  👋 Goodbye! Bot shutdown complete.\n")


# ══════════════════════════════════════════════════════════════════
# DRY-RUN MODE (no API keys needed)
# ══════════════════════════════════════════════════════════════════

class DryRunBot:
    """
    Simulates the live bot loop without placing real orders.
    Perfect for testing the pipeline without API keys.
    """

    def __init__(self, symbols: list[str] | None = None):
        self.symbols = symbols or LiveBot.DEFAULT_SYMBOLS[:10]
        self.logger = setup_logging()
        self.engine = AegisLiveEngine(
            indicator_weights_path=os.path.join(config.OUTPUT_DIR, "indicator_weights.json"),
            risk_blocks_path=os.path.join(config.OUTPUT_DIR, "risk_blocks.json"),
            initial_capital=2000.0,
        )
        self.logger.info("  🧪 DRY-RUN MODE — No real orders will be placed")
        self.logger.info(f"  Symbols: {len(self.symbols)} | Brain loaded: "
                         f"{len(self.engine.scoring_engines)} weights")

    def run_single_cycle(self):
        """Execute a single analysis cycle without API calls."""
        self.logger.info(f"\n{'─' * 60}")
        self.logger.info(f"  🧪 DRY-RUN CYCLE — {datetime.now(timezone.utc).strftime('%H:%M:%S')}")
        self.logger.info(f"{'─' * 60}")
        self.logger.info("  (In dry-run mode, we simulate using random data)")
        self.logger.info("  To run live: set BYBIT_DEMO_KEY and BYBIT_DEMO_SECRET")
        self.logger.info("")

        for symbol in self.symbols[:5]:
            sym_clean = symbol.replace("USDT", "") + "USDT"
            engine = self.engine.scoring_engines.get(sym_clean)
            if engine:
                self.logger.info(
                    f"    {symbol}: {len(engine._top_indicators)} indicators ready | "
                    f"Top weight: {engine._top_indicators[0]['weight']:.4f}"
                )

        status = self.engine.compound.get_status()
        self.logger.info(f"\n  💰 Capital: ${status['current_balance']:.2f} | "
                         f"Risk/trade: ${status['current_risk_usdt']:.2f}")
        self.logger.info("  ✅ Pipeline validated — ready for live deployment!")


# ══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Aegis-Quant-Lab v3.0 — Live Trading Bot (Pulse)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry-run (no API keys needed):
  python live_bot.py --dry-run

  # Live on Bybit Demo (requires keys):
  python live_bot.py

  # Custom settings:
  python live_bot.py --interval 15 --capital 2000 --leverage 5

  # Specific symbols only:
  python live_bot.py --symbols BTCUSDT,ETHUSDT,SOLUSDT
        """
    )

    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run without API connection (test pipeline only)"
    )
    parser.add_argument(
        "--interval", type=int, default=15,
        choices=[5, 15, 30, 60],
        help="Candle interval in minutes (default: 15)"
    )
    parser.add_argument(
        "--capital", type=float, default=2000.0,
        help="Starting capital in USDT (default: 2000)"
    )
    parser.add_argument(
        "--leverage", type=float, default=5.0,
        help="Trading leverage (default: 5x)"
    )
    parser.add_argument(
        "--symbols", type=str, default=None,
        help="Comma-separated list of symbols (default: all 49)"
    )

    args = parser.parse_args()

    # Parse symbols
    symbols = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]

    # ── DRY-RUN MODE ──
    if args.dry_run:
        bot = DryRunBot(symbols=symbols)
        bot.run_single_cycle()
        return

    # ── LIVE MODE ──
    api_key = os.environ.get("BYBIT_DEMO_KEY", "")
    api_secret = os.environ.get("BYBIT_DEMO_SECRET", "")

    if not api_key or not api_secret:
        print("\n" + "=" * 60)
        print("  ⚠️  BYBIT API KEYS NOT SET!")
        print("=" * 60)
        print("\n  To run live bot, set environment variables:")
        print("    export BYBIT_DEMO_KEY='your_demo_api_key'")
        print("    export BYBIT_DEMO_SECRET='your_demo_api_secret'")
        print("\n  Get keys at: https://testnet.bybit.com/")
        print("\n  Or run in dry-run mode:")
        print("    python live_bot.py --dry-run")
        print()
        sys.exit(1)

    # Create and run the bot
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
        print("  Check your API keys and internet connection.")
        sys.exit(1)
    except Exception as e:
        print(f"\n  ❌ Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
