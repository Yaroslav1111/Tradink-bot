#!/usr/bin/env python3
"""
v5.0 — Backtest Entry Point
══════════════════════════════════
Run: python run_backtest.py [--symbol BTCUSDT] [--days 30]

Downloads historical data from Bybit and runs backtest.
No API keys required for public kline data.
"""
import argparse
import logging
import os
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("aegis.backtest")


def fetch_historical_data(symbol: str, interval: str, limit: int) -> "pd.DataFrame":
    """Fetch historical klines from Bybit public API."""
    import pandas as pd
    from pybit.unified_trading import HTTP

    session = HTTP(testnet=False)
    all_rows = []
    end_time = int(time.time() * 1000)

    # Fetch in chunks (max 200 per call)
    remaining = limit
    while remaining > 0:
        batch = min(remaining, 200)
        resp = session.get_kline(
            category="linear",
            symbol=symbol,
            interval=interval,
            limit=batch,
            end=end_time,
        )
        rows = resp["result"]["list"]
        if not rows:
            break
        all_rows.extend(rows)
        # Move end_time to oldest bar in this batch
        end_time = int(rows[-1][0]) - 1
        remaining -= len(rows)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows, columns=[
        "timestamp", "open", "high", "low", "close", "volume", "turnover"
    ])
    df = df.iloc[::-1].reset_index(drop=True)  # oldest first
    # Remove duplicates
    df = df.drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce") / 1000.0
    return df


def main():
    parser = argparse.ArgumentParser(description="Aegis v5.0 Backtester")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading pair")
    parser.add_argument("--days", type=int, default=30, help="Days of history")
    parser.add_argument("--balance", type=float, default=2000.0, help="Initial balance")
    parser.add_argument("--leverage", type=float, default=5.0, help="Leverage")
    parser.add_argument("--warmup", type=int, default=100, help="Warmup bars")
    args = parser.parse_args()

    import pandas as pd
    from engine.strategy import StrategyConfig
    from runner.backtest_runner import BacktestRunner, BacktestConfig

    logger.info(f"📊 Fetching {args.days} days of {args.symbol} 15m data...")
    bars_needed = args.days * 24 * 4  # 15m bars per day = 96
    df_15m = fetch_historical_data(args.symbol, "15", bars_needed)

    if df_15m.empty or len(df_15m) < 200:
        logger.error(f"❌ Insufficient data: got {len(df_15m)} bars")
        sys.exit(1)

    logger.info(f"✅ Got {len(df_15m)} bars ({len(df_15m)/96:.1f} days)")

    # Optionally fetch 1m data for POC
    logger.info(f"📊 Fetching 1m data for Volume Profile...")
    df_1m = fetch_historical_data(args.symbol, "1", min(bars_needed * 15, 10000))
    if len(df_1m) < 60:
        df_1m = None
        logger.info("  (1m data insufficient, POC disabled)")

    # Config
    strategy_cfg = StrategyConfig()
    bt_cfg = BacktestConfig(
        symbol=args.symbol,
        initial_balance=args.balance,
        leverage=args.leverage,
        warmup_bars=args.warmup,
    )

    # Run backtest
    runner = BacktestRunner(strategy_cfg, bt_cfg)
    results = runner.run(df_15m, df_1m)

    # Print results
    print(f"\n{'═'*60}")
    print(f"  BACKTEST RESULTS — {args.symbol}")
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
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
