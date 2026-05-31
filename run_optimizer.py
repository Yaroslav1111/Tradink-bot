#!/usr/bin/env python3
"""
v5.0 — Optimizer Entry Point (Parameter Grid Search)
══════════════════════════════════════════════════════
Run: python run_optimizer.py [--symbol BTCUSDT] [--days 30] [--workers 4]

Iterates over parameter combinations and finds the best config.
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
logger = logging.getLogger("aegis.optimizer")


def fetch_historical_data(symbol: str, interval: str, limit: int) -> "pd.DataFrame":
    """Fetch historical klines from Bybit public API."""
    import pandas as pd
    from pybit.unified_trading import HTTP

    session = HTTP(testnet=False)
    all_rows = []
    end_time = int(time.time() * 1000)

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
        end_time = int(rows[-1][0]) - 1
        remaining -= len(rows)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows, columns=[
        "timestamp", "open", "high", "low", "close", "volume", "turnover"
    ])
    df = df.iloc[::-1].reset_index(drop=True)
    df = df.drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce") / 1000.0
    return df


def main():
    parser = argparse.ArgumentParser(description="Aegis v5.0 Parameter Optimizer")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading pair")
    parser.add_argument("--days", type=int, default=30, help="Days of history")
    parser.add_argument("--balance", type=float, default=2000.0, help="Initial balance")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--output", default="optimization_results.csv", help="Output CSV")
    args = parser.parse_args()

    import pandas as pd
    from engine.strategy import StrategyConfig
    from runner.backtest_runner import BacktestConfig
    from runner.optimizer import Optimizer

    # Fetch data
    logger.info(f"📊 Fetching {args.days} days of {args.symbol} 15m data...")
    bars_needed = args.days * 96
    df_15m = fetch_historical_data(args.symbol, "15", bars_needed)

    if df_15m.empty or len(df_15m) < 200:
        logger.error(f"❌ Insufficient data: got {len(df_15m)} bars")
        sys.exit(1)

    logger.info(f"✅ Got {len(df_15m)} bars")

    # 1m data for POC
    df_1m = fetch_historical_data(args.symbol, "1", min(bars_needed * 15, 5000))
    if len(df_1m) < 60:
        df_1m = None

    # ─── Define parameter grid ───
    # Customize this grid to explore different parameter spaces
    param_grid = {
        "fibo_primary": [0.5, 0.618, 0.786],
        "fibo_secondary": [0.382, 0.50, 0.618],
        "sl_atr_multiplier": [1.5, 2.0, 2.5],
        "order_ttl_seconds": [300, 600, 900],
    }

    total_combos = 1
    for v in param_grid.values():
        total_combos *= len(v)
    logger.info(f"🔬 Parameter grid: {total_combos} combinations")

    # Config
    bt_cfg = BacktestConfig(
        symbol=args.symbol,
        initial_balance=args.balance,
        leverage=5.0,
        warmup_bars=100,
    )

    # Run optimizer
    optimizer = Optimizer(
        param_grid=param_grid,
        base_config=StrategyConfig(),
        backtest_config=bt_cfg,
        max_workers=args.workers,
    )

    results_df = optimizer.run(df_15m, df_1m, parallel=(args.workers > 1))

    # Print top results
    Optimizer.print_results(results_df, top_n=20)

    # Save to CSV
    results_df.to_csv(args.output, index=False)
    logger.info(f"💾 Results saved to {args.output}")


if __name__ == "__main__":
    main()
