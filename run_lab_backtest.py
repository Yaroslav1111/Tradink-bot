#!/usr/bin/env python3
"""
v5.5 — Laboratory Backtest (offline, no network)
═══════════════════════════════════════════════════
Runs the Two-Winged dual-lot exit engine over a synthetic but realistic
14-day volatile series when the live Bybit API is unreachable (sandbox).

It exercises the FULL v5.5 pipeline:
  - unchanged entry engine (oscillator + fibo cascade + POC)
  - dual-lot 50/50 split (Lot A maker fix + Lot B momentum float)
  - protection cascade, race-safe state machine, structural stop
  - Trade Capsule assembly → data/ai_analysis/

Usage: python run_lab_backtest.py [--symbol SOLUSDT] [--days 14] [--seed 42]
"""
import argparse
import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("aegis.lab")


def synth_volatile_series(days: int, seed: int, base_price: float = 150.0) -> pd.DataFrame:
    """Generate a volatile 15m OHLCV series with reversals (SOL-like behaviour)."""
    np.random.seed(seed)
    n = days * 24 * 4  # 15m bars/day = 96
    t = np.arange(n) * 900.0 + 1_700_000_000

    # regime-switching drift + multi-frequency oscillation → many reversals
    drift = np.cumsum(np.random.randn(n) * 0.9)
    wave = (0.05 * base_price) * np.sin(np.arange(n) / 14.0)
    micro = (0.02 * base_price) * np.sin(np.arange(n) / 3.3)
    shocks = np.random.randn(n) * 0.6
    close = base_price + drift + wave + micro + np.cumsum(shocks) * 0.1
    close = np.maximum(close, base_price * 0.4)

    open_ = close + np.random.randn(n) * 0.3
    high = np.maximum(open_, close) + np.abs(np.random.randn(n)) * (0.006 * base_price)
    low = np.minimum(open_, close) - np.abs(np.random.randn(n)) * (0.006 * base_price)
    vol = np.random.randint(2000, 40000, n).astype(float)
    return pd.DataFrame({"timestamp": t, "open": open_, "high": high,
                         "low": low, "close": close, "volume": vol})


def main():
    ap = argparse.ArgumentParser(description="Aegis v5.5 Laboratory Backtest (offline)")
    ap.add_argument("--symbol", default="SOLUSDT")
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--balance", type=float, default=2000.0)
    ap.add_argument("--leverage", type=float, default=5.0)
    ap.add_argument("--warmup", type=int, default=100)
    args = ap.parse_args()

    from engine.strategy import StrategyConfig
    from engine.trade_capsule import capsule_to_json
    from runner.backtest_runner import BacktestRunner, BacktestConfig

    logger.info(f"🧪 Lab backtest: {args.symbol} | {args.days} volatile days | seed={args.seed}")
    df_15m = synth_volatile_series(args.days, args.seed)
    logger.info(f"✅ Generated {len(df_15m)} synthetic 15m bars ({len(df_15m)/96:.1f} days)")

    strategy_cfg = StrategyConfig()
    bt_cfg = BacktestConfig(symbol=args.symbol, initial_balance=args.balance,
                            leverage=args.leverage, warmup_bars=args.warmup)
    runner = BacktestRunner(strategy_cfg, bt_cfg, enable_capsules=True)
    results = runner.run(df_15m, None)

    print(f"\n{'═'*60}")
    print(f"  LABORATORY BACKTEST RESULTS — {args.symbol}  (14-day, offline)")
    print(f"{'═'*60}")
    print(f"  Period:       {args.days} days ({len(df_15m)} bars)")
    print(f"  Balance:      {args.balance} → {results['final_balance']:.2f} USDT")
    print(f"  Return:       {results['return_pct']:+.2f}%")
    print(f"  Total PnL:    {results['total_pnl']:+.2f} USDT")
    print(f"  Trades:       {results['total_trades']}")
    print(f"  Win Rate:     {results['win_rate']:.1f}%")
    print(f"  Profit Factor:{results['profit_factor']:.2f}")
    print(f"  Max Drawdown: {results['max_drawdown_pct']:.1f}%")
    print(f"  Time:         {results['elapsed_seconds']:.1f}s")
    print(f"{'═'*60}")

    print(f"\n{'═'*60}")
    print(f"  v5.5 TWO-WINGED DUAL-LOT EXIT PERFORMANCE")
    print(f"{'═'*60}")
    print(f"  Dual-Lot Trades:   {results['dual_lot_trades']}")
    print(f"  Lot A (Maker Fix): {results['lot_a_total_pnl']:+.2f} USDT "
          f"(win rate {results['maker_fix_win_rate']:.1f}%)")
    print(f"  Lot B (Momentum):  {results['lot_b_total_pnl']:+.2f} USDT")
    print(f"  Momentum Exits:    {results['momentum_exit_count']}")
    print(f"  Race Conditions:   {results['race_conditions']} (resolved safely)")
    print(f"  Premature Exits:   {results['premature_exits']} (Lot B left profit on table)")
    print(f"  Max-Impulse Catch: {results['max_impulse_catches']} (Lot B near local extreme)")
    print(f"  Capsules Written:  {results['capsules_generated']} → data/ai_analysis/")
    print(f"{'═'*60}")

    if runner.capsules:
        preview = runner.capsules[0]
        print(f"\n{'─'*60}")
        print(f"  RAW TRADE CAPSULE PREVIEW  (capsule_id={preview.capsule_id})")
        print(f"  4 pillars: Pre-Trade | Internal Thoughts | Execution | Post-Trade")
        print(f"{'─'*60}")
        print(capsule_to_json(preview, indent=2))
        print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()
