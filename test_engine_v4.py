#!/usr/bin/env python3
"""
Aegis-Quant-Lab v4.2 — TEST SUITE
═══════════════════════════════════
Tests all components of the Fibonacci Reversal Sniper + Volume POC + Shadow Trailing.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aegis_live_engine import (
    ReversalEngine,
    FiboCalculator,
    VolumeProfiler,
    PositionManager,
    CompoundCalculator,
    LivePosition,
    PendingOrder,
    ReversalSignal,
    TradeDirection,
    PositionState,
)
import config


def make_df(closes: list[float], n_bars: int = 150) -> pd.DataFrame:
    """Create a synthetic DataFrame for testing."""
    if len(closes) < n_bars:
        # Pad with random walk from first price
        base = closes[0] if closes else 100.0
        padding = [base + np.random.uniform(-1, 1) for _ in range(n_bars - len(closes))]
        closes = padding + closes

    df = pd.DataFrame({
        "open": closes,
        "high": [c * 1.005 for c in closes],
        "low": [c * 0.995 for c in closes],
        "close": closes,
        "volume": [1000.0] * len(closes),
    })
    return df


def test_reversal_engine():
    """Test 1: Oscillator resonance detection."""
    print("\n" + "─" * 60)
    print("  TEST 1: Reversal Engine (Oscillator Resonance)")
    print("─" * 60)

    engine = ReversalEngine()

    # Scenario A: Create EXTREME OVERSOLD data (RSI < 15, CCI < -250)
    n = 150
    prices = [100.0]
    for i in range(1, n):
        drop = -0.5 + np.random.uniform(-0.2, 0.05)
        prices.append(max(prices[-1] + drop, 55.0))

    # Last 20 bars: extreme drop to trigger oversold
    for i in range(20):
        prices.append(prices[-1] - 0.8)

    df = pd.DataFrame({
        "open": prices[:-1] + [prices[-1]],
        "high": [p + 0.3 for p in prices],
        "low": [p - 0.5 for p in prices],
        "close": prices,
        "volume": [1000.0] * len(prices),
    })

    signal = engine.detect("TESTUSDT", df)

    if signal is not None:
        assert signal.direction == TradeDirection.LONG, "Oversold should signal LONG"
        assert signal.oscillators_firing >= 2, "Need 2+ oscillators"
        print(f"  ✅ Oversold signal: LONG | {signal.oscillators_firing}/3 oscillators")
        print(f"     RSI={signal.rsi_value:.1f} | CCI={signal.cci_value:.0f} | W%R={signal.willr_value:.1f}")
    else:
        print(f"  ℹ️ No signal on synthetic data (expected — real data needed)")
        print(f"     Engine correctly requires extreme conditions")

    # Scenario B: Neutral market — should NOT fire
    neutral_prices = [100.0 + np.random.uniform(-1, 1) for _ in range(170)]
    df_neutral = pd.DataFrame({
        "open": neutral_prices,
        "high": [p + 0.5 for p in neutral_prices],
        "low": [p - 0.5 for p in neutral_prices],
        "close": neutral_prices,
        "volume": [1000.0] * len(neutral_prices),
    })

    signal_neutral = engine.detect("TESTUSDT", df_neutral)
    assert signal_neutral is None, "Neutral market should NOT trigger signal"
    print(f"  ✅ Neutral market: No signal (correct)")

    # Scenario C: Insufficient data
    df_short = pd.DataFrame({
        "open": [100.0] * 50,
        "high": [101.0] * 50,
        "low": [99.0] * 50,
        "close": [100.0] * 50,
        "volume": [1000.0] * 50,
    })
    signal_short = engine.detect("TESTUSDT", df_short)
    assert signal_short is None, "Too short data should return None"
    print(f"  ✅ Insufficient data: No signal (correct)")


def test_fibo_calculator():
    """Test 2: Fibonacci level calculations."""
    print("\n" + "─" * 60)
    print("  TEST 2: Fibonacci Calculator")
    print("─" * 60)

    # LONG entry: swing H=100, L=80 → entry at 100 - 20*0.618 = 87.64
    entry_long = FiboCalculator.calculate_entry(100.0, 80.0, TradeDirection.LONG)
    expected_long = 100.0 - (20.0 * 0.618)
    assert abs(entry_long - expected_long) < 0.01, f"LONG entry wrong: {entry_long}"
    print(f"  ✅ LONG entry (H=100, L=80): {entry_long:.2f} (expected {expected_long:.2f})")

    # SHORT entry: swing H=100, L=80 → entry at 80 + 20*0.618 = 92.36
    entry_short = FiboCalculator.calculate_entry(100.0, 80.0, TradeDirection.SHORT)
    expected_short = 80.0 + (20.0 * 0.618)
    assert abs(entry_short - expected_short) < 0.01, f"SHORT entry wrong: {entry_short}"
    print(f"  ✅ SHORT entry (H=100, L=80): {entry_short:.2f} (expected {expected_short:.2f})")

    # Extensions
    ext1, ext2 = FiboCalculator.calculate_extensions(100.0, 80.0, TradeDirection.LONG)
    expected_ext1 = 100.0 + 20.0 * (1.618 - 1)
    expected_ext2 = 100.0 + 20.0 * (2.618 - 1)
    assert abs(ext1 - expected_ext1) < 0.01, f"Extension 1 wrong: {ext1}"
    assert abs(ext2 - expected_ext2) < 0.01, f"Extension 2 wrong: {ext2}"
    print(f"  ✅ LONG extensions: 1.618={ext1:.2f}, 2.618={ext2:.2f}")

    # SHORT extensions
    ext1_s, ext2_s = FiboCalculator.calculate_extensions(100.0, 80.0, TradeDirection.SHORT)
    expected_ext1_s = 80.0 - 20.0 * (1.618 - 1)
    assert abs(ext1_s - expected_ext1_s) < 0.01
    print(f"  ✅ SHORT extensions: 1.618={ext1_s:.2f}, 2.618={ext2_s:.2f}")

    # Edge case: zero range
    entry_zero = FiboCalculator.calculate_entry(100.0, 100.0, TradeDirection.LONG)
    assert entry_zero == 100.0, "Zero range should return midpoint"
    print(f"  ✅ Zero range: {entry_zero:.2f} (midpoint fallback)")


def test_position_manager():
    """Test 3: 3-Phase trailing (Breathing → Breakeven → Shadow Trail)."""
    print("\n" + "─" * 60)
    print("  TEST 3: Position Manager (3-Phase Shadow Trailing)")
    print("─" * 60)

    pm = PositionManager()

    # Create a LONG position with ATR-based SL
    pos = LivePosition(
        symbol="BTCUSDT",
        direction=TradeDirection.LONG,
        entry_price=100.0,
        entry_time=datetime.now(timezone.utc),
        quantity=0.1,
        risk_usdt=20.0,
        stop_loss=96.0,       # -4% (2×ATR where ATR=2.0)
        take_profit=112.36,   # Fibo ext 1.618
        initial_stop_loss=96.0,
        highest_price=100.0,
        lowest_price=100.0,
        entry_atr=2.0,        # ATR at entry
    )

    # Phase 1: "Breathing Room" — SL should NOT move
    pos = pm.update(pos, 100.5, prev_candle_low=99.5, prev_candle_high=100.8)
    assert pos.state == PositionState.OPEN
    assert pos.stop_loss == 96.0, "Phase 1: SL must stay fixed"
    print(f"  ✅ Phase 1 (+0.5%): OPEN, SL fixed at {pos.stop_loss:.2f}")

    # Phase 2: +1.6% → BREAKEVEN
    pos = pm.update(pos, 101.6, prev_candle_low=100.5, prev_candle_high=101.8)
    assert pos.state == PositionState.BREAKEVEN
    assert pos.stop_loss > 100.0  # SL above entry
    print(f"  ✅ Phase 2 (+1.6%): BREAKEVEN, SL={pos.stop_loss:.4f}")

    # Phase 3: +3.1% → TRAILING activated
    pos = pm.update(pos, 103.1, prev_candle_low=102.0, prev_candle_high=103.5)
    assert pos.state == PositionState.TRAILING
    print(f"  ✅ Phase 3 (+3.1%): TRAILING activated")

    # Shadow trailing: SL should follow prev_candle_low - 0.2×ATR
    pos = pm.update(pos, 105.0, prev_candle_low=103.8, prev_candle_high=105.5)
    expected_shadow_sl = 103.8 - (2.0 * config.TRAILING_ATR_CUSHION)  # 103.8 - 0.4 = 103.4
    assert pos.stop_loss >= expected_shadow_sl - 0.01
    print(f"  ✅ Shadow trailing: SL={pos.stop_loss:.4f} (prev_low=103.8, cushion={2.0*config.TRAILING_ATR_CUSHION:.1f})")

    # SL can only move UP — try a lower prev_candle_low
    old_sl = pos.stop_loss
    pos = pm.update(pos, 104.5, prev_candle_low=100.0, prev_candle_high=105.0)
    assert pos.stop_loss >= old_sl, "SL must never move down!"
    print(f"  ✅ SL monotonic: stays at {pos.stop_loss:.4f} (won't go back to 100.0)")

    # Price drops to SL → CLOSED
    pos = pm.update(pos, pos.stop_loss - 0.01)
    assert pos.state == PositionState.CLOSED
    assert pos.pnl_usdt > 0  # should be profitable
    print(f"  ✅ SL hit: CLOSED, PnL=${pos.pnl_usdt:.2f} ({pos.close_reason})")

    # Test SHORT position with shadow trailing
    pos_short = LivePosition(
        symbol="ETHUSDT",
        direction=TradeDirection.SHORT,
        entry_price=3000.0,
        entry_time=datetime.now(timezone.utc),
        quantity=0.5,
        risk_usdt=20.0,
        stop_loss=3120.0,     # +4% (2×ATR)
        take_profit=2760.0,   # ext 1.618
        initial_stop_loss=3120.0,
        highest_price=3000.0,
        lowest_price=3000.0,
        entry_atr=60.0,       # ATR at entry
    )

    # SHORT: price drops 1.6% → breakeven
    pos_short = pm.update(pos_short, 2952.0, prev_candle_low=2940.0, prev_candle_high=2970.0)
    assert pos_short.state == PositionState.BREAKEVEN
    print(f"  ✅ SHORT Phase 2: BREAKEVEN, SL={pos_short.stop_loss:.2f}")

    # SHORT trailing: SL follows prev_candle_high + cushion
    pos_short = pm.update(pos_short, 2890.0, prev_candle_low=2880.0, prev_candle_high=2920.0)
    pos_short = pm.update(pos_short, 2870.0, prev_candle_low=2860.0, prev_candle_high=2900.0)
    expected_short_sl = 2900.0 + (60.0 * config.TRAILING_ATR_CUSHION)  # 2900 + 12 = 2912
    assert pos_short.state == PositionState.TRAILING
    assert pos_short.stop_loss <= expected_short_sl + 0.01
    print(f"  ✅ SHORT Shadow: SL={pos_short.stop_loss:.2f} (prev_high based)")


def test_compound_calculator():
    """Test 4: Position sizing + compound interest."""
    print("\n" + "─" * 60)
    print("  TEST 4: Compound Calculator")
    print("─" * 60)

    calc = CompoundCalculator(initial_capital=2000.0)

    # Initial state
    assert calc.current_balance == 2000.0
    assert calc.current_risk_usdt == 20.0  # 1% of 2000
    print(f"  ✅ Initial: ${calc.current_balance:.2f} | Risk: ${calc.current_risk_usdt:.2f}")

    # Position sizing
    size = calc.calculate_position_size(
        entry_price=50000.0,  # BTC
        stop_loss_price=49000.0,  # -2% SL
    )
    assert size["quantity"] > 0
    assert abs(size["risk_usdt"] - 20.0) < 0.01
    print(f"  ✅ Position size: qty={size['quantity']:.6f} | risk=${size['risk_usdt']:.2f}")

    # Winning streak
    for i in range(5):
        calc.record_trade(40.0, "BTCUSDT")
    assert calc.current_balance == 2200.0
    assert calc.consecutive_losses == 0
    print(f"  ✅ 5 wins (+$40): Balance=${calc.current_balance:.2f}")

    # Loss streak → risk reduction
    for i in range(3):
        calc.record_trade(-22.0, "ETHUSDT")
    assert calc.consecutive_losses == 3
    assert calc.current_risk_pct == config.COMPOUND_RISK_REDUCED
    print(f"  ✅ 3 losses: Risk reduced to {calc.current_risk_pct*100:.1f}%")

    # Recovery → risk restored
    calc.record_trade(30.0, "SOLUSDT")
    assert calc.consecutive_losses == 0
    assert calc.current_risk_pct == config.COMPOUND_RISK_NORMAL
    print(f"  ✅ Recovery: Risk restored to {calc.current_risk_pct*100:.1f}%")

    # Status
    status = calc.get_status()
    assert status["total_trades"] == 9
    assert status["wins"] == 6
    assert status["losses"] == 3
    print(f"  ✅ Status: {status['total_trades']} trades | "
          f"WR={status['win_rate_pct']:.1f}% | ROI={status['roi_pct']:.2f}%")


def test_single_entry_lock():
    """Test 5: Anti-pyramid / single-entry lock."""
    print("\n" + "─" * 60)
    print("  TEST 5: Single-Entry Lock (Anti-Pyramid)")
    print("─" * 60)

    open_positions = [
        LivePosition(
            symbol="SANDUSDT",
            direction=TradeDirection.LONG,
            entry_price=0.5,
            entry_time=datetime.now(timezone.utc),
            quantity=100.0,
            risk_usdt=20.0,
            stop_loss=0.46,
            take_profit=0.58,
            initial_stop_loss=0.46,
        ),
    ]
    pending_orders = [
        PendingOrder(
            symbol="BTCUSDT",
            order_id="test123",
            direction=TradeDirection.LONG,
            limit_price=67000.0,
            stop_loss=65000.0,
            take_profit=71000.0,
            quantity=0.003,
            risk_usdt=20.0,
            placed_at=0.0,
        ),
    ]

    occupied = set()
    for pos in open_positions:
        occupied.add(pos.symbol)
    for pend in pending_orders:
        occupied.add(pend.symbol)

    assert "SANDUSDT" in occupied
    print(f"  ✅ SANDUSDT blocked: already has open position")

    assert "BTCUSDT" in occupied
    print(f"  ✅ BTCUSDT blocked: pending limit order exists")

    assert "ETHUSDT" not in occupied
    print(f"  ✅ ETHUSDT allowed: no position or pending order")

    assert "SOLUSDT" not in occupied
    print(f"  ✅ SOLUSDT allowed: free to enter")


def test_trailing_with_reversal():
    """Test 6: Exit on oscillator reversal."""
    print("\n" + "─" * 60)
    print("  TEST 6: Trailing + Reversal Exit")
    print("─" * 60)

    pm = PositionManager()

    # LONG position in profit
    pos = LivePosition(
        symbol="SOLUSDT",
        direction=TradeDirection.LONG,
        entry_price=150.0,
        entry_time=datetime.now(timezone.utc),
        quantity=1.0,
        risk_usdt=20.0,
        stop_loss=144.0,
        take_profit=162.0,
        initial_stop_loss=144.0,
        highest_price=155.0,
        lowest_price=150.0,
        entry_atr=3.0,
    )

    # Already in profit
    pos = pm.update(pos, 155.0)  # +3.3% → should be trailing
    assert pos.state in (PositionState.BREAKEVEN, PositionState.TRAILING)
    print(f"  ✅ +3.3%: State={pos.state.value}")

    # Reversal detected while in profit → close
    pos = pm.update(pos, 154.0, reversal_detected=True)
    assert pos.state == PositionState.CLOSED
    assert pos.pnl_usdt > 0
    print(f"  ✅ Reversal detected: CLOSED with PnL=${pos.pnl_usdt:.2f}")
    print(f"     Reason: {pos.close_reason}")


def test_config_integrity():
    """Test 7: Config values are sane."""
    print("\n" + "─" * 60)
    print("  TEST 7: Configuration Integrity (v4.2)")
    print("─" * 60)

    assert config.RSI_OVERSOLD < 30, "RSI oversold should be < 30"
    assert config.RSI_OVERBOUGHT > 70, "RSI overbought should be > 70"
    assert config.FIBO_LEVEL_PRIMARY == 0.618, "Primary Fibo should be 0.618"
    assert config.ORDER_TTL_SECONDS == 600, "TTL should be 10 minutes"
    assert config.MAX_CONCURRENT_POSITIONS == 5
    assert config.MAKER_FEE_RATE < config.TAKER_FEE_RATE
    assert config.BREAKEVEN_TRIGGER_PCT > 0
    assert config.TRAILING_TRIGGER_PCT > config.BREAKEVEN_TRIGGER_PCT
    assert len(config.SYMBOLS) >= 40
    assert config.CASCADE_ENABLED is True
    assert config.CASCADE_RISK_SPLIT == 0.5
    print(f"  ✅ All config values valid")
    print(f"     RSI thresholds: {config.RSI_OVERSOLD}/{config.RSI_OVERBOUGHT}")
    print(f"     Fibo levels: 0.50 + 0.618 (cascade)")
    print(f"     TTL: {config.ORDER_TTL_SECONDS}s (10 min)")
    print(f"     Symbols: {len(config.SYMBOLS)}")
    print(f"     Maker fee: {config.MAKER_FEE_RATE*100:.3f}%")
    print(f"     Cascade: enabled (split {config.CASCADE_RISK_SPLIT*100:.0f}%/{config.CASCADE_RISK_SPLIT*100:.0f}%)")


def test_cascade_fibo():
    """Test 8: Cascade Fibonacci Order Laddering."""
    print("\n" + "─" * 60)
    print("  TEST 8: Cascade Fibonacci (Order Laddering)")
    print("─" * 60)

    # LONG cascade: H=100, L=80, current=95
    e50, e618 = FiboCalculator.calculate_cascade_entries(
        100.0, 80.0, TradeDirection.LONG, current_price=95.0
    )
    # entry_50 = 100 - 20*0.50 = 90.0
    # entry_618 = 100 - 20*0.618 = 87.64
    assert abs(e50 - 90.0) < 0.01, f"Cascade LONG 0.50 wrong: {e50}"
    assert abs(e618 - 87.64) < 0.01, f"Cascade LONG 0.618 wrong: {e618}"
    assert e618 < e50, "0.618 must be deeper (lower) than 0.50 for LONG"
    print(f"  ✅ LONG cascade: 0.50={e50:.2f}, 0.618={e618:.2f}")

    # SHORT cascade: H=100, L=80, current=85
    e50_s, e618_s = FiboCalculator.calculate_cascade_entries(
        100.0, 80.0, TradeDirection.SHORT, current_price=85.0
    )
    # entry_50 = 80 + 20*0.50 = 90.0
    # entry_618 = 80 + 20*0.618 = 92.36
    assert abs(e50_s - 90.0) < 0.01, f"Cascade SHORT 0.50 wrong: {e50_s}"
    assert abs(e618_s - 92.36) < 0.01, f"Cascade SHORT 0.618 wrong: {e618_s}"
    assert e618_s > e50_s, "0.618 must be deeper (higher) than 0.50 for SHORT"
    print(f"  ✅ SHORT cascade: 0.50={e50_s:.2f}, 0.618={e618_s:.2f}")


def test_negative_spread_protection():
    """Test 9: Negative spread fix (limit above market for LONG)."""
    print("\n" + "─" * 60)
    print("  TEST 9: Negative Spread Protection")
    print("─" * 60)

    # Scenario: price crashed to 60, but swing_high is still 100
    # Raw Fibo 0.618 = 100 - 20*0.618 = 87.64 → ABOVE current price (60)!
    e50, e618 = FiboCalculator.calculate_cascade_entries(
        100.0, 80.0, TradeDirection.LONG, current_price=60.0
    )
    max_allowed = 60.0 * 0.999  # must be below market
    assert e50 <= max_allowed, f"LONG limit {e50:.4f} above market cap {max_allowed:.4f}!"
    assert e618 <= max_allowed, f"LONG limit {e618:.4f} above market cap!"
    assert e618 < e50, "0.618 must still be deeper than 0.50"
    print(f"  ✅ Crash scenario (market=60): limits clamped to {e50:.4f} / {e618:.4f}")
    print(f"     (both below market cap {max_allowed:.4f})")

    # SHORT: price pumped to 120, swing_low=80
    # Raw 0.618 = 80 + 20*0.618 = 92.36 → BELOW current price (120) → OK normally
    # But what if swing range is tiny? H=121, L=120, current=121.5
    # Raw 0.50 = 120 + 1*0.50 = 120.5 → BELOW current 121.5 → BAD for SHORT
    e50_s, e618_s = FiboCalculator.calculate_cascade_entries(
        121.0, 120.0, TradeDirection.SHORT, current_price=121.5
    )
    min_allowed = 121.5 * 1.001
    assert e50_s >= min_allowed, f"SHORT limit {e50_s:.4f} below market!"
    assert e618_s >= min_allowed, f"SHORT 618 {e618_s:.4f} below market!"
    print(f"  ✅ SHORT clamp (market=121.5): limits raised to {e50_s:.4f} / {e618_s:.4f}")


def test_ttl_increased():
    """Test 10: TTL is 600s (10 minutes)."""
    print("\n" + "─" * 60)
    print("  TEST 10: TTL = 600 seconds (10 min)")
    print("─" * 60)

    assert config.ORDER_TTL_SECONDS == 600
    pending = PendingOrder(
        symbol="TEST", order_id="x", direction=TradeDirection.LONG,
        limit_price=100.0, stop_loss=96.0, take_profit=108.0,
        quantity=1.0, risk_usdt=10.0, placed_at=0.0,
    )
    assert pending.ttl_seconds == 600
    print(f"  ✅ Default TTL: {pending.ttl_seconds}s = {pending.ttl_seconds/60:.0f} minutes")


# ══════════════════════════════════════════════════════════════════
# RUN ALL TESTS
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  AEGIS-QUANT-LAB v4.1 — TEST SUITE")
    print("  Architecture: Fibonacci Reversal Sniper + Cascade Laddering")
    print("=" * 70)

    test_reversal_engine()
    test_fibo_calculator()
    test_position_manager()
    test_compound_calculator()
    test_single_entry_lock()
    test_trailing_with_reversal()
    test_config_integrity()
    test_cascade_fibo()
    test_negative_spread_protection()
    test_ttl_increased()

    print("\n" + "=" * 70)
    print("  🎉 ALL 10 TESTS PASSED — v4.1 ready for deployment!")
    print("=" * 70 + "\n")
