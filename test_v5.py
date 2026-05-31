#!/usr/bin/env python3
"""
v5.0 — Comprehensive Test Suite
══════════════════════════════════
Tests all v5.0 components: Models, Strategy, SimBroker, BacktestRunner, Optimizer.

Run: python test_v5.py
"""
import sys
import os
import unittest
import numpy as np
import pandas as pd
from dataclasses import asdict

# Ensure project root in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.models import (
    Direction, OrderStatus, PositionPhase, Candle, Signal, Order,
    Position, TradeResult, AccountState,
)
from engine.strategy import (
    StrategyConfig, Indicators, VolumeProfiler, FiboCalc,
    CompoundCalc, FiboReversalStrategy,
)
from broker.sim_broker import SimBroker
from runner.backtest_runner import BacktestRunner, BacktestConfig
from runner.optimizer import Optimizer


def make_ohlcv_df(n: int = 200, base_price: float = 100.0, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing with valid OHLC relationships."""
    np.random.seed(seed)
    timestamps = np.arange(n) * 900.0 + 1700000000  # 15m bars
    prices = np.cumsum(np.random.randn(n) * 0.5) + base_price
    prices = np.maximum(prices, base_price * 0.5)  # floor

    opens = prices.copy()
    closes = prices + np.random.randn(n) * 0.2
    # Ensure high >= max(open, close) and low <= min(open, close)
    highs = np.maximum(opens, closes) + np.abs(np.random.randn(n) * 0.3)
    lows = np.minimum(opens, closes) - np.abs(np.random.randn(n) * 0.3)
    volumes = np.random.randint(100, 10000, n).astype(float)

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })


def make_trending_df(n: int = 200, direction: str = "up", base: float = 100.0) -> pd.DataFrame:
    """Generate trending data (up/down) for signal testing."""
    np.random.seed(123)
    if direction == "up":
        trend = np.linspace(0, 20, n)
    else:
        trend = np.linspace(0, -20, n)

    prices = base + trend + np.random.randn(n) * 0.3
    timestamps = np.arange(n) * 900.0 + 1700000000

    closes = prices.copy()
    opens = prices - np.random.randn(n) * 0.1
    highs = np.maximum(opens, closes) + np.abs(np.random.randn(n) * 0.5)
    lows = np.minimum(opens, closes) - np.abs(np.random.randn(n) * 0.5)

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": np.random.randint(100, 5000, n).astype(float),
    })


# ══════════════════════════════════════════════════════════════════
# TEST: Models
# ══════════════════════════════════════════════════════════════════

class TestModels(unittest.TestCase):
    """Test domain model dataclasses."""

    def test_direction_enum(self):
        self.assertEqual(Direction.LONG.value, "LONG")
        self.assertEqual(Direction.SHORT.value, "SHORT")

    def test_account_state_free_margin(self):
        acc = AccountState(
            total_balance=2000,
            available_balance=1500,
            locked_margin=500,
            pending_margin=200,
        )
        self.assertAlmostEqual(acc.free_margin, 1300.0)

    def test_account_state_zero(self):
        acc = AccountState()
        self.assertEqual(acc.free_margin, 0.0)

    def test_position_phase_transitions(self):
        pos = Position(
            symbol="BTCUSDT",
            direction=Direction.LONG,
            entry_price=100.0,
            quantity=1.0,
            risk_usdt=20.0,
            stop_loss=95.0,
            take_profit=110.0,
            initial_stop_loss=95.0,
        )
        self.assertEqual(pos.phase, PositionPhase.BREATHING)
        pos.phase = PositionPhase.BREAKEVEN
        self.assertEqual(pos.phase, PositionPhase.BREAKEVEN)

    def test_order_defaults(self):
        order = Order(
            order_id="test123",
            symbol="ETHUSDT",
            direction=Direction.SHORT,
            side="Sell",
            price=2000.0,
            quantity=0.5,
            stop_loss=2100.0,
            take_profit=1800.0,
        )
        self.assertEqual(order.status, OrderStatus.PENDING)
        self.assertEqual(order.cascade_group_id, "")


# ══════════════════════════════════════════════════════════════════
# TEST: Indicators
# ══════════════════════════════════════════════════════════════════

class TestIndicators(unittest.TestCase):
    """Test pure indicator calculations."""

    def setUp(self):
        self.df = make_ohlcv_df(200)

    def test_rsi_range(self):
        rsi = Indicators.rsi(self.df["close"], 14)
        self.assertIsNotNone(rsi)
        valid = rsi.dropna()
        self.assertTrue((valid >= 0).all())
        self.assertTrue((valid <= 100).all())

    def test_rsi_length(self):
        rsi = Indicators.rsi(self.df["close"], 14)
        self.assertEqual(len(rsi), 200)

    def test_cci_not_none(self):
        cci = Indicators.cci(self.df["high"], self.df["low"], self.df["close"], 14)
        self.assertIsNotNone(cci)
        valid = cci.dropna()
        self.assertTrue(len(valid) > 0)

    def test_williams_r_range(self):
        willr = Indicators.williams_r(self.df["high"], self.df["low"], self.df["close"], 14)
        self.assertIsNotNone(willr)
        valid = willr.dropna()
        self.assertTrue((valid >= -100).all())
        self.assertTrue((valid <= 0).all())

    def test_atr_positive(self):
        atr = Indicators.atr(self.df["high"], self.df["low"], self.df["close"], 14)
        self.assertIsNotNone(atr)
        valid = atr.dropna()
        self.assertTrue((valid > 0).all())


# ══════════════════════════════════════════════════════════════════
# TEST: Volume Profiler
# ══════════════════════════════════════════════════════════════════

class TestVolumeProfiler(unittest.TestCase):
    """Test POC calculation."""

    def test_poc_returns_float(self):
        df_1m = make_ohlcv_df(60, base_price=100.0, seed=99)
        poc = VolumeProfiler.calculate_poc(df_1m, 2.0, Direction.LONG, 102.0)
        # POC should be below current price for LONG
        if poc is not None:
            self.assertLess(poc, 102.0)

    def test_poc_none_for_insufficient_data(self):
        df_short = make_ohlcv_df(5)
        result = VolumeProfiler.calculate_poc(df_short, 1.0, Direction.LONG, 100.0)
        self.assertIsNone(result)

    def test_blend_no_poc(self):
        result = VolumeProfiler.blend(None, 50.0, 0.6)
        self.assertAlmostEqual(result, 50.0)

    def test_blend_with_poc(self):
        result = VolumeProfiler.blend(40.0, 50.0, 0.6)
        expected = 40.0 * 0.6 + 50.0 * 0.4
        self.assertAlmostEqual(result, expected)


# ══════════════════════════════════════════════════════════════════
# TEST: Fibonacci Calculator
# ══════════════════════════════════════════════════════════════════

class TestFiboCalc(unittest.TestCase):
    """Test Fibonacci calculations."""

    def test_cascade_entries_long(self):
        e50, e618 = FiboCalc.cascade_entries(
            swing_high=110, swing_low=100, direction=Direction.LONG,
            current_price=108, fibo_50=0.50, fibo_618=0.618,
        )
        # For LONG: entry below current price
        self.assertLess(e50, 108)
        self.assertLess(e618, e50)  # 0.618 deeper than 0.50

    def test_cascade_entries_short(self):
        e50, e618 = FiboCalc.cascade_entries(
            swing_high=110, swing_low=100, direction=Direction.SHORT,
            current_price=102, fibo_50=0.50, fibo_618=0.618,
        )
        # For SHORT: entry above current price
        self.assertGreater(e50, 102)
        self.assertGreater(e618, e50)

    def test_negative_spread_protection_long(self):
        # If fibo would be above current price, clamp it
        e50, e618 = FiboCalc.cascade_entries(
            swing_high=100, swing_low=90, direction=Direction.LONG,
            current_price=96, fibo_50=0.50, fibo_618=0.618,
        )
        max_buy = 96 * 0.999
        self.assertLessEqual(e50, max_buy)
        self.assertLessEqual(e618, max_buy)

    def test_extensions_long(self):
        ext1, ext2 = FiboCalc.extensions(
            swing_high=110, swing_low=100, direction=Direction.LONG,
        )
        self.assertGreater(ext1, 110)
        self.assertGreater(ext2, ext1)

    def test_extensions_short(self):
        ext1, ext2 = FiboCalc.extensions(
            swing_high=110, swing_low=100, direction=Direction.SHORT,
        )
        self.assertLess(ext1, 100)
        self.assertLess(ext2, ext1)


# ══════════════════════════════════════════════════════════════════
# TEST: Compound Calculator
# ══════════════════════════════════════════════════════════════════

class TestCompoundCalc(unittest.TestCase):
    """Test risk sizing and compound tracking."""

    def setUp(self):
        self.cfg = StrategyConfig()
        self.calc = CompoundCalc(2000.0, self.cfg)

    def test_initial_risk(self):
        # 1% of 2000 = 20
        self.assertAlmostEqual(self.calc.risk_usdt, 20.0)

    def test_risk_after_loss_streak(self):
        for _ in range(3):
            self.calc.record(-10.0)
        # After 3 losses, risk_pct should be reduced
        self.assertEqual(self.calc.risk_pct, self.cfg.risk_reduced)

    def test_win_resets_streak(self):
        for _ in range(3):
            self.calc.record(-10.0)
        self.calc.record(30.0)
        self.assertEqual(self.calc.consecutive_losses, 0)
        self.assertEqual(self.calc.risk_pct, self.cfg.risk_pct)

    def test_drawdown_tracking(self):
        self.calc.record(100)  # balance = 2100, peak = 2100
        self.calc.record(-200)  # balance = 1900, dd = 200/2100 ≈ 9.5%
        self.assertGreater(self.calc.max_drawdown_pct, 0.09)


# ══════════════════════════════════════════════════════════════════
# TEST: Strategy (The Brain)
# ══════════════════════════════════════════════════════════════════

class TestStrategy(unittest.TestCase):
    """Test FiboReversalStrategy logic."""

    def setUp(self):
        self.cfg = StrategyConfig()
        self.strategy = FiboReversalStrategy(self.cfg, initial_balance=2000.0)

    def test_detect_signal_returns_none_insufficient_data(self):
        df = make_ohlcv_df(50)  # Less than 100
        result = self.strategy.detect_signal("BTCUSDT", df, 100.0)
        self.assertIsNone(result)

    def test_detect_signal_valid_data(self):
        df = make_ohlcv_df(200)
        # May or may not detect signal depending on random data
        result = self.strategy.detect_signal("BTCUSDT", df, 100.0)
        if result is not None:
            self.assertIn(result, [Direction.LONG, Direction.SHORT])

    def test_build_signal_returns_signal(self):
        df_15m = make_ohlcv_df(200, base_price=100)
        account = AccountState(
            total_balance=2000, available_balance=2000,
            locked_margin=0, pending_margin=0,
        )
        signal = self.strategy.build_signal(
            "BTCUSDT", Direction.LONG, df_15m, None, 100.0, account
        )
        if signal is not None:
            self.assertEqual(signal.symbol, "BTCUSDT")
            self.assertEqual(signal.direction, Direction.LONG)
            self.assertGreater(signal.qty_50, 0)
            self.assertGreater(signal.qty_618, 0)

    def test_build_signal_insufficient_margin(self):
        df_15m = make_ohlcv_df(200, base_price=100)
        account = AccountState(
            total_balance=1, available_balance=0.01,
            locked_margin=0.99, pending_margin=0,
        )
        signal = self.strategy.build_signal(
            "BTCUSDT", Direction.LONG, df_15m, None, 100.0, account
        )
        # Should reject due to margin
        self.assertIsNone(signal)

    def test_update_position_breathing_to_breakeven(self):
        pos = Position(
            symbol="BTCUSDT", direction=Direction.LONG,
            entry_price=100.0, quantity=1.0, risk_usdt=20.0,
            stop_loss=95.0, take_profit=115.0, initial_stop_loss=95.0,
            phase=PositionPhase.BREATHING,
            highest_price=100.0, lowest_price=100.0,
        )
        # Price rises 2% → should trigger breakeven
        pos = self.strategy.update_position(pos, 102.0)
        self.assertEqual(pos.phase, PositionPhase.BREAKEVEN)
        self.assertGreater(pos.stop_loss, 99.9)  # SL moved above entry

    def test_update_position_stop_loss_hit(self):
        pos = Position(
            symbol="BTCUSDT", direction=Direction.LONG,
            entry_price=100.0, quantity=1.0, risk_usdt=20.0,
            stop_loss=95.0, take_profit=115.0, initial_stop_loss=95.0,
            phase=PositionPhase.BREATHING,
            highest_price=100.0, lowest_price=100.0,
        )
        pos = self.strategy.update_position(pos, 94.0)
        self.assertEqual(pos.phase, PositionPhase.CLOSED)
        self.assertLess(pos.pnl_usdt, 0)

    def test_update_position_short_breakeven(self):
        pos = Position(
            symbol="BTCUSDT", direction=Direction.SHORT,
            entry_price=100.0, quantity=1.0, risk_usdt=20.0,
            stop_loss=105.0, take_profit=90.0, initial_stop_loss=105.0,
            phase=PositionPhase.BREATHING,
            highest_price=100.0, lowest_price=100.0,
        )
        # Price drops 2% → breakeven for SHORT
        pos = self.strategy.update_position(pos, 98.0)
        self.assertEqual(pos.phase, PositionPhase.BREAKEVEN)
        self.assertLess(pos.stop_loss, 100.0)  # SL below entry

    def test_trailing_phase(self):
        pos = Position(
            symbol="BTCUSDT", direction=Direction.LONG,
            entry_price=100.0, quantity=1.0, risk_usdt=20.0,
            stop_loss=100.04, take_profit=115.0, initial_stop_loss=95.0,
            phase=PositionPhase.BREAKEVEN,
            highest_price=103.0, lowest_price=100.0,
            highest_profit_pct=0.03,
            entry_atr=2.0,
        )
        # Peak is 3% above → should enter trailing
        pos = self.strategy.update_position(pos, 103.5, prev_candle_low=103.0)
        self.assertEqual(pos.phase, PositionPhase.TRAILING)

    def test_should_reposition_tp(self):
        pos = Position(
            symbol="BTCUSDT", direction=Direction.LONG,
            entry_price=100.0, quantity=1.0, risk_usdt=20.0,
            stop_loss=103.0, take_profit=116.18, initial_stop_loss=95.0,
            phase=PositionPhase.TRAILING,
            highest_price=113.0, lowest_price=100.0,
            fibo_ext_1=116.18, fibo_ext_2=126.18,
            tp_repositioned=False,
        )
        # 80% of (116.18 - 100) = 12.944; price at 113 → progress ~80%
        result = self.strategy.should_reposition_tp(pos, 113.0)
        self.assertTrue(result)


# ══════════════════════════════════════════════════════════════════
# TEST: Trend Invalidation Exit (SuperTrend)
# ══════════════════════════════════════════════════════════════════

class TestTrendInvalidation(unittest.TestCase):
    """Test SuperTrend-based trend invalidation exit."""

    def setUp(self):
        self.cfg = StrategyConfig(
            trend_invalidation_enabled=True,
            supertrend_period=10,
            supertrend_multiplier=3.0,
            trend_confirm_bars=2,
        )
        self.strategy = FiboReversalStrategy(self.cfg, initial_balance=2000.0)

    def test_supertrend_returns_series(self):
        """SuperTrend returns a valid Series with +1/-1 values."""
        df = make_ohlcv_df(200)
        st = Indicators.supertrend(
            df["high"], df["low"], df["close"],
            period=10, multiplier=3.0,
        )
        self.assertIsNotNone(st)
        self.assertEqual(len(st), 200)
        # Valid values should be +1 or -1
        valid = st.dropna()
        self.assertTrue(len(valid) > 0)
        unique_vals = set(valid.unique())
        self.assertTrue(unique_vals.issubset({1.0, -1.0}))

    def test_supertrend_none_insufficient_data(self):
        """SuperTrend returns None when data is too short."""
        df = make_ohlcv_df(5)
        st = Indicators.supertrend(df["high"], df["low"], df["close"], period=10)
        self.assertIsNone(st)

    def test_supertrend_warmup_nan(self):
        """First 'period' bars should be NaN."""
        df = make_ohlcv_df(100)
        st = Indicators.supertrend(df["high"], df["low"], df["close"], period=10)
        self.assertTrue(st.iloc[:10].isna().all())
        self.assertFalse(st.iloc[10:].isna().all())

    def test_trend_invalidation_long_downtrend(self):
        """LONG position exits when SuperTrend confirms downtrend."""
        # Create strongly declining data
        n = 100
        np.random.seed(555)
        timestamps = np.arange(n) * 900.0 + 1700000000
        # Strong downtrend: starts at 100, drops to 60
        prices = np.linspace(100, 60, n)
        opens = prices.copy()
        closes = prices - 0.5  # close below open in downtrend
        highs = np.maximum(opens, closes) + 0.2
        lows = np.minimum(opens, closes) - 0.2

        df = pd.DataFrame({
            "timestamp": timestamps,
            "open": opens, "high": highs, "low": lows,
            "close": closes,
            "volume": np.ones(n) * 1000,
        })

        pos = Position(
            symbol="BTCUSDT", direction=Direction.LONG,
            entry_price=95.0, quantity=1.0, risk_usdt=20.0,
            stop_loss=85.0, take_profit=115.0, initial_stop_loss=85.0,
            highest_price=95.0, lowest_price=60.0,
        )

        result = self.strategy.check_trend_invalidation(pos, df)
        # In a strong downtrend, LONG should be invalidated
        self.assertTrue(result)

    def test_trend_invalidation_long_uptrend_no_exit(self):
        """LONG position does NOT exit during uptrend."""
        n = 100
        np.random.seed(666)
        timestamps = np.arange(n) * 900.0 + 1700000000
        # Strong uptrend
        prices = np.linspace(100, 140, n)
        opens = prices.copy()
        closes = prices + 0.5
        highs = np.maximum(opens, closes) + 0.3
        lows = np.minimum(opens, closes) - 0.3

        df = pd.DataFrame({
            "timestamp": timestamps,
            "open": opens, "high": highs, "low": lows,
            "close": closes,
            "volume": np.ones(n) * 1000,
        })

        pos = Position(
            symbol="BTCUSDT", direction=Direction.LONG,
            entry_price=100.0, quantity=1.0, risk_usdt=20.0,
            stop_loss=90.0, take_profit=150.0, initial_stop_loss=90.0,
            highest_price=140.0, lowest_price=100.0,
        )

        result = self.strategy.check_trend_invalidation(pos, df)
        self.assertFalse(result)

    def test_trend_invalidation_short_uptrend(self):
        """SHORT position exits when SuperTrend confirms uptrend."""
        n = 100
        np.random.seed(777)
        timestamps = np.arange(n) * 900.0 + 1700000000
        # Strong uptrend: starts at 100, rises to 140
        prices = np.linspace(100, 140, n)
        opens = prices.copy()
        closes = prices + 0.5
        highs = np.maximum(opens, closes) + 0.2
        lows = np.minimum(opens, closes) - 0.2

        df = pd.DataFrame({
            "timestamp": timestamps,
            "open": opens, "high": highs, "low": lows,
            "close": closes,
            "volume": np.ones(n) * 1000,
        })

        pos = Position(
            symbol="BTCUSDT", direction=Direction.SHORT,
            entry_price=105.0, quantity=1.0, risk_usdt=20.0,
            stop_loss=115.0, take_profit=90.0, initial_stop_loss=115.0,
            highest_price=105.0, lowest_price=100.0,
        )

        result = self.strategy.check_trend_invalidation(pos, df)
        self.assertTrue(result)

    def test_trend_invalidation_disabled(self):
        """No invalidation when feature is disabled."""
        cfg = StrategyConfig(trend_invalidation_enabled=False)
        strategy = FiboReversalStrategy(cfg)

        n = 100
        prices = np.linspace(100, 60, n)
        opens = prices.copy()
        closes = prices - 0.5
        highs = np.maximum(opens, closes) + 0.2
        lows = np.minimum(opens, closes) - 0.2
        df = pd.DataFrame({
            "timestamp": np.arange(n) * 900.0 + 1700000000,
            "open": opens, "high": highs, "low": lows,
            "close": closes,
            "volume": np.ones(n) * 1000,
        })

        pos = Position(
            symbol="BTCUSDT", direction=Direction.LONG,
            entry_price=95.0, quantity=1.0, risk_usdt=20.0,
            stop_loss=85.0, take_profit=115.0, initial_stop_loss=85.0,
            highest_price=95.0, lowest_price=60.0,
        )

        result = strategy.check_trend_invalidation(pos, df)
        self.assertFalse(result)

    def test_update_position_with_trend_invalidation(self):
        """update_position() closes immediately on trend_invalidated=True."""
        pos = Position(
            symbol="BTCUSDT", direction=Direction.LONG,
            entry_price=100.0, quantity=1.0, risk_usdt=20.0,
            stop_loss=90.0, take_profit=120.0, initial_stop_loss=90.0,
            phase=PositionPhase.BREATHING,
            highest_price=100.0, lowest_price=97.0,
        )
        # Even though price is above SL, trend invalidation forces exit
        pos = self.strategy.update_position(
            pos, 97.0, trend_invalidated=True
        )
        self.assertEqual(pos.phase, PositionPhase.CLOSED)
        self.assertEqual(pos.close_reason, "TREND_INVALIDATION")
        # PnL should reflect taker fee (market exit)
        self.assertLess(pos.pnl_usdt, 0)  # Loss since price dropped

    def test_update_position_trend_invalidation_priority(self):
        """TREND_INVALIDATION fires even if price is above SL (hasn't hit SL yet)."""
        pos = Position(
            symbol="BTCUSDT", direction=Direction.LONG,
            entry_price=100.0, quantity=1.0, risk_usdt=20.0,
            stop_loss=85.0, take_profit=120.0, initial_stop_loss=85.0,
            phase=PositionPhase.TRAILING,
            highest_price=110.0, lowest_price=100.0,
            highest_profit_pct=0.10,
            entry_atr=2.0,
        )
        # Price is 95 (above SL of 85), but trend is broken
        pos = self.strategy.update_position(
            pos, 95.0, trend_invalidated=True
        )
        self.assertEqual(pos.phase, PositionPhase.CLOSED)
        self.assertEqual(pos.close_reason, "TREND_INVALIDATION")

    def test_taker_fee_on_trend_invalidation(self):
        """TREND_INVALIDATION uses taker fee (market exit)."""
        pos = Position(
            symbol="BTCUSDT", direction=Direction.LONG,
            entry_price=100.0, quantity=1.0, risk_usdt=20.0,
            stop_loss=90.0, take_profit=120.0, initial_stop_loss=90.0,
            phase=PositionPhase.BREATHING,
            highest_price=100.0, lowest_price=100.0,
        )
        # Exit at entry = 0 raw PnL, only fees
        pos = self.strategy.update_position(
            pos, 100.0, trend_invalidated=True
        )
        # Fee = (100 + 100) * 1.0 * 0.00055 = 0.11 (taker)
        expected_fee = 200 * 1.0 * self.cfg.taker_fee
        self.assertAlmostEqual(pos.pnl_usdt, -expected_fee, places=4)

    def test_config_params_in_strategy_config(self):
        """All trend invalidation params are in StrategyConfig (optimizer can search)."""
        from dataclasses import fields
        field_names = {f.name for f in fields(StrategyConfig)}
        self.assertIn("trend_invalidation_enabled", field_names)
        self.assertIn("supertrend_period", field_names)
        self.assertIn("supertrend_multiplier", field_names)
        self.assertIn("trend_confirm_bars", field_names)


# ══════════════════════════════════════════════════════════════════
# TEST: SimBroker
# ══════════════════════════════════════════════════════════════════

class TestSimBroker(unittest.TestCase):
    """Test simulated broker with honest fills."""

    def setUp(self):
        self.broker = SimBroker(initial_balance=2000.0, leverage=5.0)
        self.df = make_ohlcv_df(200)
        self.broker.load_data("BTCUSDT", self.df)

    def test_initial_account(self):
        acc = self.broker.get_account()
        self.assertAlmostEqual(acc.total_balance, 2000.0)
        self.assertAlmostEqual(acc.available_balance, 2000.0)

    def test_place_limit_order(self):
        order = self.broker.place_limit_order(
            symbol="BTCUSDT", side="Buy", price=95.0, quantity=1.0,
            stop_loss=90.0, take_profit=110.0,
            direction=Direction.LONG, risk_usdt=10.0, atr=2.0,
        )
        self.assertIsNotNone(order)
        self.assertEqual(order.status, OrderStatus.PENDING)
        self.assertEqual(len(self.broker.pending_orders), 1)

    def test_cancel_order(self):
        order = self.broker.place_limit_order(
            symbol="BTCUSDT", side="Buy", price=95.0, quantity=1.0,
            stop_loss=90.0, take_profit=110.0,
        )
        result = self.broker.cancel_order("BTCUSDT", order.order_id)
        self.assertTrue(result)
        self.assertEqual(len(self.broker.pending_orders), 0)

    def test_honest_fill_long(self):
        """Order fills ONLY when candle low touches order price."""
        # Place buy at 95.0
        order = self.broker.place_limit_order(
            symbol="BTCUSDT", side="Buy", price=95.0, quantity=1.0,
            stop_loss=90.0, take_profit=110.0,
            direction=Direction.LONG, risk_usdt=10.0, atr=2.0,
        )

        # Process a bar that doesn't reach 95
        self.broker._market_data["BTCUSDT"].iloc[10] = pd.Series({
            "timestamp": 1700009000.0, "open": 100, "high": 102,
            "low": 98, "close": 101, "volume": 1000,
        })
        self.broker.process_bar("BTCUSDT", 10)
        self.assertEqual(len(self.broker.positions), 0)  # Not filled

        # Process a bar that reaches 95
        self.broker._market_data["BTCUSDT"].iloc[11] = pd.Series({
            "timestamp": 1700009900.0, "open": 99, "high": 100,
            "low": 94, "close": 96, "volume": 1500,
        })
        self.broker.process_bar("BTCUSDT", 11)
        self.assertEqual(len(self.broker.positions), 1)  # Filled!
        self.assertEqual(self.broker.positions[0].entry_price, 95.0)

    def test_honest_fill_short(self):
        """Sell limit fills when candle high touches order price."""
        order = self.broker.place_limit_order(
            symbol="BTCUSDT", side="Sell", price=105.0, quantity=1.0,
            stop_loss=110.0, take_profit=95.0,
            direction=Direction.SHORT, risk_usdt=10.0, atr=2.0,
        )

        # Bar that reaches 105
        self.broker._market_data["BTCUSDT"].iloc[10] = pd.Series({
            "timestamp": 1700009000.0, "open": 102, "high": 106,
            "low": 101, "close": 103, "volume": 1000,
        })
        self.broker.process_bar("BTCUSDT", 10)
        self.assertEqual(len(self.broker.positions), 1)
        self.assertEqual(self.broker.positions[0].direction, Direction.SHORT)

    def test_stop_loss_execution(self):
        """SL triggers when price crosses stop level."""
        # Create position manually
        pos = Position(
            symbol="BTCUSDT", direction=Direction.LONG,
            entry_price=100.0, quantity=1.0, risk_usdt=10.0,
            stop_loss=95.0, take_profit=115.0, initial_stop_loss=95.0,
            highest_price=100.0, lowest_price=100.0,
        )
        self.broker.positions.append(pos)

        # Bar with low below SL
        self.broker._market_data["BTCUSDT"].iloc[5] = pd.Series({
            "timestamp": 1700004500.0, "open": 97, "high": 98,
            "low": 93, "close": 94, "volume": 2000,
        })
        self.broker.process_bar("BTCUSDT", 5)
        self.assertEqual(len(self.broker.positions), 0)
        self.assertEqual(len(self.broker.closed_trades), 1)
        self.assertLess(self.broker.closed_trades[0].pnl_usdt, 0)

    def test_take_profit_execution(self):
        """TP triggers when price crosses target level."""
        pos = Position(
            symbol="BTCUSDT", direction=Direction.LONG,
            entry_price=100.0, quantity=1.0, risk_usdt=10.0,
            stop_loss=95.0, take_profit=110.0, initial_stop_loss=95.0,
            highest_price=100.0, lowest_price=100.0,
        )
        self.broker.positions.append(pos)

        # Bar with high above TP
        self.broker._market_data["BTCUSDT"].iloc[5] = pd.Series({
            "timestamp": 1700004500.0, "open": 108, "high": 112,
            "low": 107, "close": 111, "volume": 2000,
        })
        self.broker.process_bar("BTCUSDT", 5)
        self.assertEqual(len(self.broker.positions), 0)
        self.assertEqual(len(self.broker.closed_trades), 1)
        self.assertGreater(self.broker.closed_trades[0].pnl_usdt, 0)

    def test_margin_check(self):
        """Cannot place order if insufficient margin."""
        # Use up all margin
        self.broker.balance = 1.0
        order = self.broker.place_limit_order(
            symbol="BTCUSDT", side="Buy", price=100.0, quantity=100.0,
            stop_loss=90.0, take_profit=110.0,
        )
        self.assertIsNone(order)

    def test_get_klines_respects_bar_idx(self):
        """get_klines only returns data up to current bar."""
        self.broker._current_bar_idx["BTCUSDT"] = 50
        df = self.broker.get_klines("BTCUSDT", "15", 200)
        self.assertIsNotNone(df)
        self.assertLessEqual(len(df), 51)

    def test_results_summary(self):
        """Results summary works with no trades."""
        summary = self.broker.get_results_summary()
        self.assertEqual(summary["total_trades"], 0)
        self.assertAlmostEqual(summary["final_balance"], 2000.0)

    def test_reset(self):
        """Reset clears all state."""
        self.broker.place_limit_order(
            "BTCUSDT", "Buy", 95.0, 1.0, 90.0, 110.0
        )
        self.broker.reset()
        self.assertEqual(len(self.broker.pending_orders), 0)
        self.assertAlmostEqual(self.broker.balance, 2000.0)


# ══════════════════════════════════════════════════════════════════
# TEST: Backtest Runner
# ══════════════════════════════════════════════════════════════════

class TestBacktestRunner(unittest.TestCase):
    """Test full backtest execution."""

    def test_backtest_runs_without_error(self):
        """Backtest completes on synthetic data."""
        df_15m = make_ohlcv_df(300, base_price=100.0)
        strategy_cfg = StrategyConfig()
        bt_cfg = BacktestConfig(
            symbol="BTCUSDT",
            initial_balance=2000.0,
            warmup_bars=100,
        )
        runner = BacktestRunner(strategy_cfg, bt_cfg)
        results = runner.run(df_15m)

        self.assertIn("total_trades", results)
        self.assertIn("final_balance", results)
        self.assertIn("elapsed_seconds", results)
        self.assertGreater(results["elapsed_seconds"], 0)

    def test_backtest_with_poc(self):
        """Backtest with 1m data for POC."""
        df_15m = make_ohlcv_df(300, base_price=100.0)
        df_1m = make_ohlcv_df(1000, base_price=100.0, seed=77)

        strategy_cfg = StrategyConfig(poc_enabled=True)
        bt_cfg = BacktestConfig(symbol="BTCUSDT", warmup_bars=100)
        runner = BacktestRunner(strategy_cfg, bt_cfg)
        results = runner.run(df_15m, df_1m)

        self.assertIn("total_trades", results)

    def test_warmup_bars_skipped(self):
        """Warmup bars don't generate trades."""
        df_15m = make_ohlcv_df(150, base_price=100.0)
        strategy_cfg = StrategyConfig()
        bt_cfg = BacktestConfig(
            symbol="BTCUSDT",
            warmup_bars=140,  # Only 10 bars for trading
        )
        runner = BacktestRunner(strategy_cfg, bt_cfg)
        results = runner.run(df_15m)
        # With only 10 bars and 100-bar minimum for detect_signal, no trades
        self.assertEqual(results["total_trades"], 0)


# ══════════════════════════════════════════════════════════════════
# TEST: Optimizer
# ══════════════════════════════════════════════════════════════════

class TestOptimizer(unittest.TestCase):
    """Test grid search optimizer."""

    def test_optimizer_generates_combinations(self):
        param_grid = {
            "fibo_primary": [0.5, 0.618],
            "sl_atr_multiplier": [1.5, 2.0],
        }
        opt = Optimizer(param_grid=param_grid)
        self.assertEqual(len(opt.combinations), 4)  # 2 * 2

    def test_optimizer_invalid_param(self):
        param_grid = {"nonexistent_param": [1, 2, 3]}
        with self.assertRaises(ValueError):
            Optimizer(param_grid=param_grid)

    def test_optimizer_runs(self):
        """Optimizer runs sequentially on synthetic data."""
        df_15m = make_ohlcv_df(300, base_price=100.0)
        param_grid = {
            "sl_atr_multiplier": [1.5, 2.0],
        }
        bt_cfg = BacktestConfig(symbol="BTCUSDT", warmup_bars=100)
        opt = Optimizer(
            param_grid=param_grid,
            backtest_config=bt_cfg,
            max_workers=1,
        )
        results = opt.run(df_15m, parallel=False)
        self.assertEqual(len(results), 2)
        self.assertIn("total_pnl", results.columns)
        self.assertIn("param_sl_atr_multiplier", results.columns)


# ══════════════════════════════════════════════════════════════════
# TEST: Integration (Strategy + SimBroker)
# ══════════════════════════════════════════════════════════════════

class TestIntegration(unittest.TestCase):
    """Integration tests: Strategy uses SimBroker correctly."""

    def test_exact_in_exact(self):
        """
        Core requirement: Strategy code is IDENTICAL for live and backtest.
        Verify strategy can work with SimBroker without modifications.
        """
        cfg = StrategyConfig()
        strategy = FiboReversalStrategy(cfg, initial_balance=2000.0)
        broker = SimBroker(initial_balance=2000.0, leverage=cfg.leverage)

        df = make_ohlcv_df(200)
        broker.load_data("BTCUSDT", df)

        # Strategy uses broker's data
        broker._current_bar_idx["BTCUSDT"] = 150
        df_visible = broker.get_klines("BTCUSDT", "15", 200)

        # Strategy calls are identical to live
        direction = strategy.detect_signal("BTCUSDT", df_visible, 100.0)
        account = broker.get_account()

        if direction:
            signal = strategy.build_signal(
                "BTCUSDT", direction, df_visible, None, 100.0, account
            )
            # Signal can be executed via broker (same interface as live)
            if signal:
                side = "Buy" if signal.direction == Direction.LONG else "Sell"
                order = broker.place_limit_order(
                    symbol=signal.symbol, side=side,
                    price=signal.entry_price_50, quantity=signal.qty_50,
                    stop_loss=signal.stop_loss, take_profit=signal.take_profit,
                    direction=signal.direction,
                    risk_usdt=signal.risk_usdt_50, atr=signal.atr,
                )
                # Order accepted or rejected — same behavior as live
                if order:
                    self.assertEqual(order.status, OrderStatus.PENDING)

    def test_margin_rejection(self):
        """Signal rejected when insufficient margin (Requirement 3)."""
        cfg = StrategyConfig()
        strategy = FiboReversalStrategy(cfg, initial_balance=2000.0)

        df = make_ohlcv_df(200, base_price=50000)  # BTC-like prices
        account = AccountState(
            total_balance=10, available_balance=0.1,
            locked_margin=9.9, pending_margin=0,
        )

        signal = strategy.build_signal(
            "BTCUSDT", Direction.LONG, df, None, 50000.0, account
        )
        # Should be None due to margin check
        self.assertIsNone(signal)


# ══════════════════════════════════════════════════════════════════
# SYMBOL-SPECIFIC CONFIGURATION TESTS
# ══════════════════════════════════════════════════════════════════

class TestSymbolSpecificConfig(unittest.TestCase):
    """Tests for per-symbol optimized parameter system."""

    def test_optimizer_get_best_params(self):
        """Optimizer.get_best_params() extracts top-1 overrides from results."""
        from runner.optimizer import Optimizer

        param_grid = {
            "fibo_primary": [0.5, 0.618, 0.786],
            "sl_atr_multiplier": [1.5, 2.0],
        }
        optimizer = Optimizer(param_grid=param_grid)

        # Simulate results DataFrame
        rows = [
            {"total_pnl": 100, "return_pct": 5, "max_drawdown_pct": 3,
             "total_trades": 10, "win_rate": 60, "profit_factor": 2.0,
             "final_balance": 2100, "avg_trade_pnl": 10,
             "param_fibo_primary": 0.786, "param_sl_atr_multiplier": 1.5},
            {"total_pnl": 50, "return_pct": 2.5, "max_drawdown_pct": 5,
             "total_trades": 8, "win_rate": 55, "profit_factor": 1.5,
             "final_balance": 2050, "avg_trade_pnl": 6.25,
             "param_fibo_primary": 0.618, "param_sl_atr_multiplier": 2.0},
        ]
        df = pd.DataFrame(rows)

        best = optimizer.get_best_params(df)
        self.assertEqual(best["fibo_primary"], 0.786)
        self.assertEqual(best["sl_atr_multiplier"], 1.5)

    def test_optimizer_save_and_load_params(self):
        """save_optimized_params() writes JSON and load reads it back."""
        import tempfile
        from runner.optimizer import Optimizer

        params = {
            "BTCUSDT": {"fibo_primary": 0.786, "sl_atr_multiplier": 1.5},
            "ETHUSDT": {"fibo_primary": 0.618, "sl_atr_multiplier": 2.0},
        }

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            tmp_path = f.name

        try:
            Optimizer.save_optimized_params(params, tmp_path)
            loaded = Optimizer.load_optimized_params(tmp_path)

            self.assertEqual(loaded["BTCUSDT"]["fibo_primary"], 0.786)
            self.assertEqual(loaded["ETHUSDT"]["sl_atr_multiplier"], 2.0)
            self.assertEqual(len(loaded), 2)
        finally:
            os.unlink(tmp_path)

    def test_optimizer_load_nonexistent_returns_empty(self):
        """load_optimized_params() returns empty dict if file doesn't exist."""
        from runner.optimizer import Optimizer

        result = Optimizer.load_optimized_params("/tmp/nonexistent_test_file_xyz.json")
        self.assertEqual(result, {})

    def test_build_symbol_config_with_overrides(self):
        """build_symbol_config() applies overrides for known symbols."""
        from runner.optimizer import Optimizer

        optimized = {
            "BTCUSDT": {"fibo_primary": 0.786, "sl_atr_multiplier": 1.5},
        }
        base = StrategyConfig()  # fibo_primary=0.618 by default

        cfg = Optimizer.build_symbol_config("BTCUSDT", optimized, base)
        self.assertEqual(cfg.fibo_primary, 0.786)
        self.assertEqual(cfg.sl_atr_multiplier, 1.5)
        # Non-overridden params stay default
        self.assertEqual(cfg.fibo_secondary, base.fibo_secondary)

    def test_build_symbol_config_unknown_symbol_uses_defaults(self):
        """build_symbol_config() returns base config for unknown symbol."""
        from runner.optimizer import Optimizer

        optimized = {
            "BTCUSDT": {"fibo_primary": 0.786},
        }
        base = StrategyConfig()

        cfg = Optimizer.build_symbol_config("XLMUSDT", optimized, base)
        self.assertEqual(cfg.fibo_primary, base.fibo_primary)
        self.assertEqual(cfg.sl_atr_multiplier, base.sl_atr_multiplier)

    def test_backtest_runner_with_optimized_params(self):
        """BacktestRunner applies per-symbol optimized config."""
        from runner.backtest_runner import BacktestRunner, BacktestConfig

        optimized = {
            "BTCUSDT": {"fibo_primary": 0.786, "sl_atr_multiplier": 1.5},
        }

        bt_cfg = BacktestConfig(symbol="BTCUSDT", initial_balance=2000.0)
        runner = BacktestRunner(StrategyConfig(), bt_cfg, optimized_params=optimized)

        # Verify the runner applied the overrides
        self.assertEqual(runner.strategy_cfg.fibo_primary, 0.786)
        self.assertEqual(runner.strategy_cfg.sl_atr_multiplier, 1.5)
        # Strategy was created with overridden config
        self.assertEqual(runner.strategy.cfg.fibo_primary, 0.786)

    def test_backtest_runner_without_optimized_params(self):
        """BacktestRunner uses defaults when no optimized params."""
        from runner.backtest_runner import BacktestRunner, BacktestConfig

        bt_cfg = BacktestConfig(symbol="BTCUSDT", initial_balance=2000.0)
        runner = BacktestRunner(StrategyConfig(), bt_cfg, optimized_params=None)

        self.assertEqual(runner.strategy_cfg.fibo_primary, 0.618)  # default
        self.assertEqual(runner.strategy_cfg.sl_atr_multiplier, 2.0)  # default

    def test_backtest_runner_symbol_not_in_optimized(self):
        """BacktestRunner uses defaults when symbol not in optimized dict."""
        from runner.backtest_runner import BacktestRunner, BacktestConfig

        optimized = {
            "ETHUSDT": {"fibo_primary": 0.786},
        }

        bt_cfg = BacktestConfig(symbol="BTCUSDT", initial_balance=2000.0)
        runner = BacktestRunner(StrategyConfig(), bt_cfg, optimized_params=optimized)

        # BTCUSDT not in optimized → should use default
        self.assertEqual(runner.strategy_cfg.fibo_primary, 0.618)

    def test_backtest_runner_load_optimized_params_static(self):
        """BacktestRunner.load_optimized_params() reads JSON correctly."""
        import tempfile
        import json
        from runner.backtest_runner import BacktestRunner

        params = {"SOLUSDT": {"fibo_primary": 0.5, "order_ttl_seconds": 300}}

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(params, f)
            tmp_path = f.name

        try:
            loaded = BacktestRunner.load_optimized_params(tmp_path)
            self.assertEqual(loaded["SOLUSDT"]["fibo_primary"], 0.5)
            self.assertEqual(loaded["SOLUSDT"]["order_ttl_seconds"], 300)
        finally:
            os.unlink(tmp_path)

    def test_live_runner_get_strategy_for_symbol(self):
        """LiveRunner._get_strategy_for_symbol() returns per-symbol or default."""
        import tempfile
        import json
        from unittest.mock import MagicMock, patch
        from runner.live_runner import LiveRunner

        # Create mock broker that returns empty lists for recovery
        mock_broker = MagicMock()
        mock_broker.get_positions.return_value = []
        mock_broker.get_pending_orders.return_value = []

        # Create temp optimized_params.json
        params = {"BTCUSDT": {"fibo_primary": 0.786, "sl_atr_multiplier": 1.5}}
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(params, f)
            tmp_path = f.name

        try:
            strategy = FiboReversalStrategy(StrategyConfig(), initial_balance=2000.0)
            runner = LiveRunner(
                broker=mock_broker,
                strategy=strategy,
                symbols=["BTCUSDT", "ETHUSDT"],
                optimized_params_path=tmp_path,
            )

            # BTCUSDT should get custom strategy
            btc_strategy = runner._get_strategy_for_symbol("BTCUSDT")
            self.assertEqual(btc_strategy.cfg.fibo_primary, 0.786)
            self.assertEqual(btc_strategy.cfg.sl_atr_multiplier, 1.5)

            # ETHUSDT should get default strategy
            eth_strategy = runner._get_strategy_for_symbol("ETHUSDT")
            self.assertEqual(eth_strategy.cfg.fibo_primary, 0.618)  # default
            self.assertIs(eth_strategy, strategy)  # same object as default
        finally:
            os.unlink(tmp_path)

    def test_live_runner_no_optimized_file(self):
        """LiveRunner works fine when optimized_params.json doesn't exist."""
        from unittest.mock import MagicMock
        from runner.live_runner import LiveRunner

        mock_broker = MagicMock()
        mock_broker.get_positions.return_value = []
        mock_broker.get_pending_orders.return_value = []

        strategy = FiboReversalStrategy(StrategyConfig(), initial_balance=2000.0)
        runner = LiveRunner(
            broker=mock_broker,
            strategy=strategy,
            symbols=["BTCUSDT"],
            optimized_params_path="/tmp/nonexistent_params_xyz.json",
        )

        # Should fallback to default for all symbols
        self.assertEqual(len(runner._symbol_configs), 0)
        self.assertEqual(len(runner._symbol_strategies), 0)

        # All symbols should get the default strategy
        s = runner._get_strategy_for_symbol("BTCUSDT")
        self.assertIs(s, strategy)

    def test_live_runner_get_config_for_symbol(self):
        """LiveRunner._get_config_for_symbol() returns correct configs."""
        import tempfile
        import json
        from unittest.mock import MagicMock
        from runner.live_runner import LiveRunner

        mock_broker = MagicMock()
        mock_broker.get_positions.return_value = []
        mock_broker.get_pending_orders.return_value = []

        params = {"SOLUSDT": {"sl_atr_multiplier": 2.5, "trailing_trigger_pct": 0.04}}
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(params, f)
            tmp_path = f.name

        try:
            base_cfg = StrategyConfig()
            strategy = FiboReversalStrategy(base_cfg, initial_balance=2000.0)
            runner = LiveRunner(
                broker=mock_broker,
                strategy=strategy,
                symbols=["SOLUSDT", "BTCUSDT"],
                optimized_params_path=tmp_path,
            )

            # SOLUSDT should have custom config
            sol_cfg = runner._get_config_for_symbol("SOLUSDT")
            self.assertEqual(sol_cfg.sl_atr_multiplier, 2.5)
            self.assertEqual(sol_cfg.trailing_trigger_pct, 0.04)

            # BTCUSDT should have default config
            btc_cfg = runner._get_config_for_symbol("BTCUSDT")
            self.assertEqual(btc_cfg.sl_atr_multiplier, 2.0)  # default
            self.assertIs(btc_cfg, base_cfg)  # same object
        finally:
            os.unlink(tmp_path)

    def test_optimizer_build_symbol_config_no_base(self):
        """build_symbol_config() uses default StrategyConfig when base is None."""
        from runner.optimizer import Optimizer

        optimized = {"BTCUSDT": {"fibo_primary": 0.5}}
        cfg = Optimizer.build_symbol_config("BTCUSDT", optimized, None)
        self.assertEqual(cfg.fibo_primary, 0.5)
        # Other fields from default
        self.assertEqual(cfg.sl_atr_multiplier, 2.0)

    def test_full_backtest_with_optimized_params(self):
        """End-to-end: BacktestRunner with optimized params produces valid results."""
        from runner.backtest_runner import BacktestRunner, BacktestConfig

        df = make_ohlcv_df(300, base_price=100.0, seed=123)

        optimized = {
            "TESTUSDT": {"fibo_primary": 0.5, "sl_atr_multiplier": 1.5},
        }

        bt_cfg = BacktestConfig(symbol="TESTUSDT", initial_balance=2000.0)
        runner = BacktestRunner(StrategyConfig(), bt_cfg, optimized_params=optimized)

        # Verify config was applied
        self.assertEqual(runner.strategy_cfg.fibo_primary, 0.5)
        self.assertEqual(runner.strategy_cfg.sl_atr_multiplier, 1.5)

        # Run backtest (should complete without errors)
        results = runner.run(df)
        self.assertIn("total_trades", results)
        self.assertIn("total_pnl", results)
        self.assertIn("final_balance", results)
        self.assertGreater(results["final_balance"], 0)


# ══════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Run with verbosity
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Exit code for CI
    sys.exit(0 if result.wasSuccessful() else 1)
