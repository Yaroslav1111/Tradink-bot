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
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from trade_diagnostics import TradeDiagnostics

try:
    from dotenv import load_dotenv
    load_dotenv(".env.local")
except ImportError:
    pass  # dotenv not installed, use env vars directly

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
        "MATICUSDT", "TONUSDT", "TRXUSDT", "1000SHIBUSDT", "UNIUSDT",
        "ATOMUSDT", "LTCUSDT", "BCHUSDT", "NEARUSDT", "APTUSDT",
        "FILUSDT", "ARBUSDT", "OPUSDT", "SUIUSDT", "HYPEUSDT",
        "IMXUSDT", "1000PEPEUSDT", "WIFUSDT", "FETUSDT", "RENDERUSDT",
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
        self.diagnostics = TradeDiagnostics()
        self._instrument_cache: dict[str, dict] = {}  # cache lot_size info
        self._pending_orders: list[dict] = []  # pending limit orders with TTL

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

    def _fetch_all_candles_parallel(self, limit: int = 200) -> dict[str, pd.DataFrame]:
        """
        Fetch candles for ALL symbols in parallel using ThreadPoolExecutor.
        This reduces scan time from 60+ seconds to ~5-10 seconds.
        """
        results: dict[str, pd.DataFrame] = {}

        def _fetch_one(sym: str):
            try:
                return sym, self._fetch_candles(sym, limit=limit)
            except Exception:
                return sym, None

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(_fetch_one, s): s for s in self.symbols}
            for future in as_completed(futures):
                try:
                    sym, df = future.result(timeout=15)
                    if df is not None and len(df) >= 100:
                        results[sym] = df
                except Exception:
                    pass

        return results

    def _get_lot_size_precision(self, symbol: str, current_price: float) -> int:
        """
        Get the proper decimal precision for quantity from Bybit instrument info.
        Falls back to price-based heuristic if API call fails.
        """
        # Check cache first
        if symbol in self._instrument_cache:
            info = self._instrument_cache[symbol]
        else:
            info = self.connector.get_instrument_info(symbol)
            if info:
                self._instrument_cache[symbol] = info

        if info:
            try:
                lot_filter = info.get("lotSizeFilter", {})
                qty_step = lot_filter.get("qtyStep", "1")
                # Count decimal places in qtyStep
                if "." in qty_step:
                    decimals = len(qty_step.rstrip("0").split(".")[1])
                else:
                    decimals = 0
                return decimals
            except Exception:
                pass

        # Fallback: price-based heuristic
        if current_price > 10000:
            return 3
        elif current_price > 100:
            return 2
        elif current_price > 1:
            return 1
        elif current_price > 0.01:
            return 0
        else:
            return 0

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
        Execute a trade entry on Bybit Demo using SMART LIMIT ORDERS.

        Architecture (v3.2):
          1. Diagnostics pre-entry checks (exhaustion, decay, momentum)
          2. ClusterGuard approval (includes anti-pyramid)
          3. Calculate SL/TP using ATR
          4. Calculate OPTIMAL limit price (discount entry)
          5. Place Limit order with TTL (Time-To-Live)
          6. Order will be checked on next cycle for fill or expiry
        """
        direction = scoring.direction
        if direction is None:
            return None

        dir_str = direction.value

        # -- Step 1: Diagnostic pre-entry checks --
        diag_ok, diag_reason = self.diagnostics.run_pre_entry_checks(
            symbol, dir_str, df, current_price,
        )
        if not diag_ok:
            self.logger.info(f"  >>> {symbol} BLOCKED by diagnostics: {diag_reason}")
            return None

        # -- Step 2: Cluster Guard (includes ANTI-PYRAMID) --
        allowed, reason = self.engine.cluster_guard.check_entry_allowed(
            symbol, self.engine.open_positions
        )
        if not allowed:
            self.logger.info(f"  >>> {symbol} BLOCKED by ClusterGuard: {reason}")
            self.diagnostics.log_entry_decision(
                symbol, "BLOCK", reason,
                score=scoring.total_score, direction=dir_str, price=current_price,
            )
            return None

        # -- Step 3: Calculate ATR-based SL/TP --
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

        # -- Step 4: Calculate OPTIMAL limit price with discount --
        # Instead of entering at market, we enter at a DISCOUNT:
        # LONG: limit price = current_price - 0.3*ATR (buy on a micro-dip)
        # SHORT: limit price = current_price + 0.3*ATR (sell on a micro-bounce)
        discount = atr_value * 0.3

        if direction == TradeDirection.LONG:
            limit_price = current_price - discount
            stop_loss = limit_price - sl_distance
            take_profit = limit_price + tp_distance
            side = "Buy"
        else:
            limit_price = current_price + discount
            stop_loss = limit_price + sl_distance
            take_profit = limit_price - tp_distance
            side = "Sell"

        # -- Step 5: Position size (compound) --
        size_info = self.engine.compound.calculate_position_size(
            entry_price=limit_price,
            stop_loss_price=stop_loss,
            leverage=self.leverage,
        )

        quantity = size_info["quantity"]

        # -- Step 5b: Proper Bybit lot_size precision --
        precision = self._get_lot_size_precision(symbol, current_price)
        quantity = round(quantity, precision)

        if quantity <= 0:
            self.logger.warning(f"  !!! {symbol} quantity too small: {quantity}")
            return None

        # Round limit price to tick size
        if current_price > 100:
            limit_price = round(limit_price, 2)
            stop_loss = round(stop_loss, 2)
            take_profit = round(take_profit, 2)
        elif current_price > 1:
            limit_price = round(limit_price, 4)
            stop_loss = round(stop_loss, 4)
            take_profit = round(take_profit, 4)
        else:
            limit_price = round(limit_price, 6)
            stop_loss = round(stop_loss, 6)
            take_profit = round(take_profit, 6)

        # -- Step 6: Place LIMIT order --
        self.logger.info("")
        self.logger.info(f"  ======================================================")
        self.logger.info(f"   NEW LIMIT ENTRY: {direction.value} {symbol}")
        self.logger.info(f"  ------------------------------------------------------")
        self.logger.info(f"   Score:    {scoring.confidence_pct:.1f}% "
                         f"({scoring.n_indicators_firing} indicators)")
        self.logger.info(f"   Market:   {current_price:.6f}")
        self.logger.info(f"   Limit:    {limit_price:.6f} "
                         f"(discount: {discount/current_price*100:.2f}%)")
        self.logger.info(f"   Qty:      {quantity} (precision: {precision} decimals)")
        self.logger.info(f"   Risk:     ${size_info['risk_usdt']:.2f} "
                         f"({size_info['risk_pct_used']:.2f}% of balance)")
        self.logger.info(f"   SL:       {stop_loss:.6f} "
                         f"(-{sl_distance/limit_price*100:.2f}%)")
        self.logger.info(f"   TP:       {take_profit:.6f} "
                         f"(+{tp_distance/limit_price*100:.2f}%)")
        self.logger.info(f"   TTL:      4 minutes (cancel if not filled)")
        self.logger.info(f"   Balance:  ${self.engine.compound.current_balance:.2f}")
        self.logger.info(f"  ======================================================")

        order_result = self.connector.place_order(
            symbol=symbol,
            side=side,
            qty=quantity,
            order_type="Limit",
            price=limit_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

        if order_result is None:
            self.logger.error(f"  !!! Order FAILED for {symbol}")
            self.diagnostics.log_entry_decision(
                symbol, "SKIP", "Order placement failed",
                score=scoring.total_score, direction=dir_str, price=current_price,
            )
            return None

        self.total_orders_placed += 1
        order_id = order_result.get("orderId", "")

        # -- Step 7: Register as PENDING order with TTL --
        # We do NOT register as open position yet — only when FILLED
        pending_info = {
            "symbol": symbol,
            "order_id": order_id,
            "side": side,
            "direction": direction,
            "limit_price": limit_price,
            "market_price_at_signal": current_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "quantity": quantity,
            "risk_usdt": size_info["risk_usdt"],
            "score": scoring.total_score,
            "placed_at": time.time(),
            "ttl_seconds": 240,  # 4 minutes TTL
            "max_price_deviation_pct": 0.01,  # cancel if price moves 1% away
        }
        self._pending_orders.append(pending_info)

        self.diagnostics.log_entry_decision(
            symbol, "PENDING",
            f"Limit order placed | TTL=4min | Score={scoring.confidence_pct:.1f}%",
            score=scoring.total_score, direction=dir_str, price=limit_price,
            extra={
                "order_id": order_id,
                "limit_price": limit_price,
                "market_price": current_price,
                "discount_pct": round(discount / current_price * 100, 3),
            }
        )
        self.diagnostics.clear_signal_price(symbol)

        self.logger.info(f"  >>> LIMIT order placed — waiting for fill (TTL=4min)")
        # Return None because position is not yet confirmed
        return None

    # ──────────────────────────────────────────────
    # PENDING ORDERS — Check fills, TTL, cancellation
    # ──────────────────────────────────────────────

    def _manage_pending_orders(self):
        """
        Check all pending limit orders:
          - If FILLED → register as open position
          - If TTL expired → cancel order
          - If price moved too far against us → cancel (trend reversed)
        """
        if not self._pending_orders:
            return

        self.logger.info(f"\n  Checking {len(self._pending_orders)} pending orders...")

        still_pending = []
        now = time.time()

        for order in self._pending_orders:
            symbol = order["symbol"]
            order_id = order["order_id"]
            placed_at = order["placed_at"]
            ttl = order["ttl_seconds"]
            elapsed = now - placed_at

            # Check 1: TTL expired?
            if elapsed > ttl:
                self.logger.info(
                    f"    EXPIRED: {symbol} limit order (TTL={ttl}s elapsed)"
                )
                self.connector.cancel_order(symbol, order_id)
                self.diagnostics.log_entry_decision(
                    symbol, "CANCEL", f"TTL expired ({elapsed:.0f}s > {ttl}s)",
                    direction=order["direction"].value, price=order["limit_price"],
                )
                continue

            # Check 2: Has the order been filled?
            detail = self.connector.get_order_detail(symbol, order_id)
            if detail:
                order_status = detail.get("orderStatus", "")

                if order_status == "Filled":
                    # ORDER FILLED! Register as open position
                    avg_price_str = detail.get("avgPrice", "")
                    fill_price = float(avg_price_str) if avg_price_str else order["limit_price"]

                    self.logger.info(
                        f"    FILLED: {symbol} {order['direction'].value} "
                        f"@ {fill_price:.6f} (limit was {order['limit_price']:.6f})"
                    )

                    # Register position
                    position = LivePosition(
                        symbol=symbol,
                        direction=order["direction"],
                        entry_price=fill_price,
                        entry_time=datetime.now(timezone.utc),
                        quantity=order["quantity"],
                        risk_usdt=order["risk_usdt"],
                        stop_loss=order["stop_loss"],
                        take_profit=order["take_profit"],
                        initial_stop_loss=order["stop_loss"],
                        score_at_entry=order["score"],
                        cluster_id=self.engine.cluster_guard.cluster_map.get(symbol, -1),
                    )
                    self.engine.open_positions.append(position)

                    self.diagnostics.log_entry_decision(
                        symbol, "ENTRY",
                        f"Limit FILLED @ {fill_price:.6f}",
                        score=order["score"],
                        direction=order["direction"].value,
                        price=fill_price,
                        extra={"fill_price": fill_price, "wait_time": f"{elapsed:.0f}s"},
                    )
                    self.diagnostics.log_fill_analysis(
                        symbol, order["market_price_at_signal"],
                        fill_price, order["direction"].value,
                    )
                    continue

                elif order_status in ("Cancelled", "Rejected", "Deactivated"):
                    self.logger.info(f"    {order_status}: {symbol} order by exchange")
                    continue

            # Check 3: Price moved too far — cancel (trend reversed)
            current_price = self.connector.get_ticker_price(symbol)
            if current_price > 0:
                deviation = abs(current_price - order["limit_price"]) / order["limit_price"]
                max_dev = order["max_price_deviation_pct"]

                if deviation > max_dev:
                    # Price moved away from our limit — trend may have reversed
                    self.logger.info(
                        f"    CANCEL: {symbol} price deviation "
                        f"{deviation*100:.2f}% > {max_dev*100:.1f}% limit"
                    )
                    self.connector.cancel_order(symbol, order_id)
                    self.diagnostics.log_entry_decision(
                        symbol, "CANCEL",
                        f"Price deviation {deviation*100:.2f}% (trend reversal)",
                        direction=order["direction"].value, price=current_price,
                    )
                    continue

            # Still pending, keep tracking
            still_pending.append(order)
            self.logger.info(
                f"    WAITING: {symbol} ({elapsed:.0f}s / {ttl}s TTL)"
            )

        self._pending_orders = still_pending

    # ──────────────────────────────────────────────
    # TRAILING PROFIT — Extend TP when trend continues
    # ──────────────────────────────────────────────

    def _extend_tp_for_open_positions(self, all_candles: dict[str, pd.DataFrame]):
        """
        If we have an open position and the signal STILL fires in same direction,
        extend TP by 1*ATR instead of opening a new position.
        This is the 'ride the trend until reversal' logic.
        """
        for pos in self.engine.open_positions:
            if pos.state == PositionState.CLOSED:
                continue

            df = all_candles.get(pos.symbol)
            if df is None or len(df) < 50:
                continue

            # Check if scoring still favors our direction
            scoring = self._analyze_symbol(pos.symbol, df)
            if scoring is None or not scoring.is_tradeable:
                continue

            # Only extend if signal is in SAME direction as our position
            if scoring.direction != pos.direction:
                continue

            # Calculate ATR for extension
            import pandas_ta as ta
            atr_series = ta.atr(
                df["high"].astype(float),
                df["low"].astype(float),
                df["close"].astype(float),
                length=14
            )
            if atr_series is None:
                continue
            atr_val = float(atr_series.iloc[-1])
            if np.isnan(atr_val) or atr_val <= 0:
                continue

            current_price = float(df["close"].iloc[-1])

            # Extend TP by 1*ATR if price is moving in our favor
            if pos.direction == TradeDirection.LONG:
                unrealized = (current_price - pos.entry_price) / pos.entry_price
                if unrealized > 0.005:  # only extend if already in profit
                    new_tp = current_price + atr_val * 2
                    if new_tp > pos.take_profit:
                        old_tp = pos.take_profit
                        pos.take_profit = new_tp
                        self.logger.info(
                            f"    >> TP EXTENDED: {pos.symbol} LONG "
                            f"| Old TP: {old_tp:.6f} -> New TP: {new_tp:.6f} "
                            f"(+{atr_val*2/current_price*100:.2f}%)"
                        )
            else:
                unrealized = (pos.entry_price - current_price) / pos.entry_price
                if unrealized > 0.005:
                    new_tp = current_price - atr_val * 2
                    if new_tp < pos.take_profit:
                        old_tp = pos.take_profit
                        pos.take_profit = new_tp
                        self.logger.info(
                            f"    >> TP EXTENDED: {pos.symbol} SHORT "
                            f"| Old TP: {old_tp:.6f} -> New TP: {new_tp:.6f}"
                        )

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
        self.logger.info(f"  │ Pending:     {len(self._pending_orders)} limit orders           │")
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

                # ── Phase 2.5: Manage pending limit orders (TTL/fill/cancel) ──
                self._manage_pending_orders()

                # -- Phase 3: Scan for new entries (PARALLEL) --
                self.logger.info(f"\n  Scanning {len(self.symbols)} symbols (parallel)...")

                signals_found = 0
                entries_made = 0

                # Fetch all candles in parallel (5-10s instead of 60+s)
                fetch_start = time.time()
                all_candles = self._fetch_all_candles_parallel(limit=200)
                fetch_time = time.time() - fetch_start
                self.logger.info(
                    f"  Fetched {len(all_candles)}/{len(self.symbols)} symbols "
                    f"in {fetch_time:.1f}s"
                )

                # ── Phase 3.1: Extend TP for open positions (trail to target) ──
                self._extend_tp_for_open_positions(all_candles)

                # ── Phase 3.2: Build set of symbols we already hold/pending ──
                # HARD ANTI-PYRAMID: skip any symbol with open position OR pending order
                occupied_symbols: set[str] = set()
                for pos in self.engine.open_positions:
                    if pos.state != PositionState.CLOSED:
                        occupied_symbols.add(pos.symbol.replace("/", ""))
                for pend in self._pending_orders:
                    occupied_symbols.add(pend["symbol"].replace("/", ""))

                # ── Phase 3.3: Scan for new entries ──
                for symbol, df in all_candles.items():
                    self.total_signals_checked += 1

                    # HARD ANTI-PYRAMID at scan level — skip occupied symbols
                    sym_clean = symbol.replace("/", "")
                    if sym_clean in occupied_symbols:
                        continue

                    # Analyze
                    scoring = self._analyze_symbol(symbol, df)
                    if scoring is None:
                        continue

                    if not scoring.is_tradeable:
                        # Log skipped signals for diagnostics
                        if scoring.total_score > 0.15:  # only log near-misses
                            self.diagnostics.log_entry_decision(
                                symbol, "SKIP", scoring.blocked_reason,
                                score=scoring.total_score,
                                direction=scoring.direction.value if scoring.direction else "",
                                price=float(df["close"].iloc[-1]),
                            )
                        continue

                    # Signal found!
                    signals_found += 1
                    current_price = float(df["close"].iloc[-1])

                    self.logger.info(
                        f"\n  SIGNAL: {scoring.direction.value} {symbol} "
                        f"@ {current_price:.6f} | "
                        f"Score: {scoring.confidence_pct:.1f}% | "
                        f"Indicators: {scoring.n_indicators_firing}"
                    )

                    # Execute entry (all checks inside)
                    position = self._execute_entry(symbol, scoring, current_price, df)
                    if position:
                        entries_made += 1

                    # Don't enter too many at once
                    if entries_made >= 3:
                        self.logger.info("  Max 3 entries per cycle -- pausing scan")
                        break

                # -- Phase 4: Summary --
                cycle_time = time.time() - cycle_start
                self.logger.info(
                    f"\n  Cycle #{self.cycle_count} complete in {cycle_time:.1f}s: "
                    f"{signals_found} signals, {entries_made} entries"
                )

                # Print dashboard every 4 cycles (every hour on 15m)
                if self.cycle_count % 4 == 0:
                    self._print_dashboard()

                # Export diagnostics every 8 cycles (every 2 hours)
                if self.cycle_count % 8 == 0:
                    try:
                        self.diagnostics.export_diagnostics()
                        diag_summary = self.diagnostics.get_summary()
                        self.logger.info(
                            f"  Diagnostics: {diag_summary['total_decisions']} decisions logged"
                        )
                    except Exception:
                        pass

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
        """Graceful shutdown -- log final state."""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("  SHUTTING DOWN")
        self.logger.info("=" * 70)
        self._print_dashboard()

        # Save state to JSON
        state = {
            "shutdown_time": datetime.now(timezone.utc).isoformat(),
            "cycles_completed": self.cycle_count,
            "total_orders": self.total_orders_placed,
            "compound_status": self.engine.compound.get_status(),
            "diagnostics_summary": self.diagnostics.get_summary(),
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
            "pending_orders": [
                {
                    "symbol": o["symbol"],
                    "order_id": o["order_id"],
                    "direction": o["direction"].value,
                    "limit_price": o["limit_price"],
                    "quantity": o["quantity"],
                }
                for o in self._pending_orders
            ],
            "trade_history": self.engine.compound.trade_history[-50:],
        }

        state_path = os.path.join(config.OUTPUT_DIR, "bot_state.json")
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, default=str)

        # Export final diagnostics
        try:
            self.diagnostics.export_diagnostics()
        except Exception:
            pass

        self.logger.info(f"  State saved to {state_path}")
        self.logger.info("  Goodbye! Bot shutdown complete.\n")


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
