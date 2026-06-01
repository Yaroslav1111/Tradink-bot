#!/usr/bin/env python3
"""
v5.2 — Multi-Symbol Optimizer Entry Point
══════════════════════════════════════════════
Run: python run_optimizer.py [--symbol BTCUSDT] [--symbols BTC,ETH,...] [--all] [--days 30]

Modes:
  --symbol BTCUSDT       Single symbol optimization (legacy)
  --symbols BTC,ETH      Comma-separated list of symbols
  --all                  All 46 monitored coins from config.py

Output:
  - optimization_results.csv  (detailed results per combo)
  - optimized_params.json     (Top-1 best params per symbol)
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


def optimize_single_symbol(
    symbol: str,
    param_grid: dict,
    days: int,
    balance: float,
    workers: int,
) -> tuple[dict, "pd.DataFrame"]:
    """
    Run grid optimization for a single symbol.
    Returns: (best_params_dict, full_results_df)
    """
    import pandas as pd
    from engine.strategy import StrategyConfig
    from runner.backtest_runner import BacktestConfig
    from runner.optimizer import Optimizer

    # Fetch data
    bars_needed = days * 96  # 15m bars per day
    logger.info(f"  📊 Fetching {days} days of {symbol} 15m data...")
    df_15m = fetch_historical_data(symbol, "15", bars_needed)

    if df_15m.empty or len(df_15m) < 200:
        logger.warning(f"  ⚠️ Insufficient data for {symbol}: {len(df_15m)} bars — skipping")
        return {}, pd.DataFrame()

    logger.info(f"  ✅ Got {len(df_15m)} bars for {symbol}")

    # 1m data for POC
    df_1m = fetch_historical_data(symbol, "1", min(bars_needed * 15, 5000))
    if len(df_1m) < 60:
        df_1m = None

    # Config
    bt_cfg = BacktestConfig(
        symbol=symbol,
        initial_balance=balance,
        leverage=5.0,
        warmup_bars=100,
    )

    # Run optimizer
    optimizer = Optimizer(
        param_grid=param_grid,
        base_config=StrategyConfig(),
        backtest_config=bt_cfg,
        max_workers=workers,
    )

    results_df = optimizer.run(df_15m, df_1m, parallel=(workers > 1))

    # Extract best params
    best_params = optimizer.get_best_params(results_df)

    return best_params, results_df


def main():
    parser = argparse.ArgumentParser(description="Aegis v5.4 Multi-Dimensional Entry/Exit Optimizer")

    # Symbol selection (mutually exclusive group)
    symbol_group = parser.add_mutually_exclusive_group()
    symbol_group.add_argument("--symbol", default=None, help="Single trading pair (e.g. BTCUSDT)")
    symbol_group.add_argument("--symbols", default=None, help="Comma-separated list (e.g. BTCUSDT,ETHUSDT,SOLUSDT)")
    symbol_group.add_argument("--all", action="store_true", help="Optimize all 46 monitored coins from config.py")

    parser.add_argument("--days", type=int, default=14, help="Days of history (default: 14 for regime fitting)")
    parser.add_argument("--balance", type=float, default=2000.0, help="Initial balance for backtest")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers per symbol")
    parser.add_argument("--output", default="optimized_params.json", help="Output JSON path")
    parser.add_argument("--csv", default="optimization_results.csv", help="Output CSV (detailed)")
    args = parser.parse_args()

    import pandas as pd
    from runner.optimizer import Optimizer

    # Resolve symbol list
    if args.all:
        from config import SYMBOLS
        symbols = SYMBOLS
    elif args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    elif args.symbol:
        symbols = [args.symbol.upper()]
    else:
        # Default to single BTCUSDT
        symbols = ["BTCUSDT"]

    logger.info(f"🚀 Multi-Dimensional Optimizer | {len(symbols)} symbols | {args.days} days each")

    # ─── Define parameter grid (Multi-Dimensional Entry/Exit) ───
    # Uses the default grid with entry sensitivity + parameter symmetry
    from runner.optimizer import get_default_param_grid, count_grid_combinations
    param_grid = get_default_param_grid()

    total_combos = count_grid_combinations(param_grid)
    logger.info(
        f"🔬 Parameter grid: {total_combos} combinations × {len(symbols)} symbols "
        f"= {total_combos * len(symbols)} total backtests"
    )
    logger.info(f"  Dimensions: {', '.join(f'{k}({len(v)})' for k, v in param_grid.items())}")
    logger.info(f"  Symmetry: rsi_oversold → rsi_overbought (auto-mirrored)")

    # ─── Run optimization per symbol ───
    all_best_params: dict[str, dict] = {}
    all_results: list[pd.DataFrame] = []
    start_time = time.time()

    for i, symbol in enumerate(symbols, 1):
        logger.info(f"\n{'═'*60}")
        logger.info(f"  [{i}/{len(symbols)}] Optimizing {symbol}")
        logger.info(f"{'═'*60}")

        best_params, results_df = optimize_single_symbol(
            symbol=symbol,
            param_grid=param_grid,
            days=args.days,
            balance=args.balance,
            workers=args.workers,
        )

        if best_params:
            all_best_params[symbol] = best_params
            logger.info(f"  🏆 Best params for {symbol}: {best_params}")

            # Tag results with symbol
            if not results_df.empty:
                results_df["symbol"] = symbol
                all_results.append(results_df)
        else:
            logger.warning(f"  ⚠️ No valid results for {symbol}")

    elapsed = time.time() - start_time

    # ─── Save optimized_params.json ───
    if all_best_params:
        Optimizer.save_optimized_params(all_best_params, args.output)
        logger.info(f"\n💾 Saved best params for {len(all_best_params)} symbols → {args.output}")
    else:
        logger.warning("⚠️ No symbols produced valid results — JSON not saved")

    # ─── Save combined CSV ───
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_df.to_csv(args.csv, index=False)
        logger.info(f"💾 Detailed results → {args.csv}")

    # ─── Summary ───
    logger.info(f"\n{'═'*60}")
    logger.info(f"  OPTIMIZATION COMPLETE")
    logger.info(f"{'═'*60}")
    logger.info(f"  Symbols processed: {len(symbols)}")
    logger.info(f"  Symbols with results: {len(all_best_params)}")
    logger.info(f"  Total time: {elapsed:.1f}s ({elapsed/max(len(symbols),1):.1f}s/symbol)")
    logger.info(f"  Output: {args.output}")
    logger.info(f"{'═'*60}\n")

    # Print best params summary
    if all_best_params:
        print(f"\n{'═'*60}")
        print(f"  TOP-1 PARAMS PER SYMBOL")
        print(f"{'═'*60}")
        for symbol, params in sorted(all_best_params.items()):
            params_str = " | ".join(f"{k}={v}" for k, v in params.items())
            print(f"  {symbol:12s} → {params_str}")
        print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
