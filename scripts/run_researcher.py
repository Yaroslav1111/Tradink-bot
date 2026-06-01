#!/usr/bin/env python3
"""
v5.4 — Portfolio Researcher (Continuous Daemon)
═══════════════════════════════════════════════════
Continuously scans Bybit for the highest-turnover USDT linear perpetuals,
ensures all Top 30 have optimized parameters, then selects a Beta-Neutral
elite portfolio (Top 8-10 best-per-sector).

Loop:
  1. Fetch Top 30 by turnover24h from Bybit public API
  2. Ensure ALL Top 30 have optimized_params.json entries (run optimizer if missing)
  3. Run portfolio_selector logic to pick Beta-Neutral Top-8/10
  4. Save filtered elite list → data/active_portfolio.json
  5. Sleep 1 hour, repeat

Usage:
  python scripts/run_researcher.py [--once]  # --once = single iteration (no loop)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.portfolio_selector import SECTOR_CLUSTERS, select_best_per_sector

logger = logging.getLogger("aegis.researcher")

# ══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════

TOP_N_BY_TURNOVER = 30
SLEEP_INTERVAL_SECONDS = 3600  # 1 hour
OPTIMIZED_PARAMS_PATH = os.path.join(PROJECT_ROOT, "optimized_params.json")
ACTIVE_PORTFOLIO_PATH = os.path.join(PROJECT_ROOT, "data", "active_portfolio.json")
OPTIMIZATION_DAYS = 14


# ══════════════════════════════════════════════════════════════════
# STEP 1: Fetch Top 30 by turnover24h from Bybit
# ══════════════════════════════════════════════════════════════════

def fetch_top_by_turnover(top_n: int = TOP_N_BY_TURNOVER) -> list[str]:
    """
    Fetch top N USDT linear perpetual symbols ranked by 24h turnover from Bybit.

    Returns:
        List of symbols, e.g. ["BTCUSDT", "ETHUSDT", "SOLUSDT", ...]
    """
    from pybit.unified_trading import HTTP

    session = HTTP(testnet=False)

    try:
        # Get all tickers for linear category
        resp = session.get_tickers(category="linear")
        tickers = resp.get("result", {}).get("list", [])

        # Filter USDT perpetuals only (exclude inverse, USDC, etc.)
        usdt_perps = [
            t for t in tickers
            if t.get("symbol", "").endswith("USDT")
            and "PERP" not in t.get("symbol", "")  # exclude named perps like BTCPERP
        ]

        # Sort by turnover24h descending
        usdt_perps.sort(
            key=lambda t: float(t.get("turnover24h", "0")),
            reverse=True,
        )

        # Take top N
        top_symbols = [t["symbol"] for t in usdt_perps[:top_n]]

        logger.info(
            f"📊 Fetched Top {len(top_symbols)} by turnover24h | "
            f"#1: {top_symbols[0] if top_symbols else 'N/A'} | "
            f"#30: {top_symbols[-1] if len(top_symbols) >= 30 else 'N/A'}"
        )
        return top_symbols

    except Exception as e:
        logger.error(f"❌ Failed to fetch tickers from Bybit: {e}")
        return []


# ══════════════════════════════════════════════════════════════════
# STEP 2: Ensure All Top 30 Have Optimized Parameters
# ══════════════════════════════════════════════════════════════════

def load_existing_params() -> dict[str, dict]:
    """Load existing optimized_params.json."""
    if not os.path.exists(OPTIMIZED_PARAMS_PATH):
        return {}
    try:
        with open(OPTIMIZED_PARAMS_PATH, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def save_merged_params(params: dict[str, dict]):
    """Save merged optimized params back to JSON."""
    # Ensure all values are JSON-serializable
    clean = {}
    for symbol, p in params.items():
        clean[symbol] = {
            k: v.item() if hasattr(v, "item") else v
            for k, v in p.items()
        }

    with open(OPTIMIZED_PARAMS_PATH, "w") as f:
        json.dump(clean, f, indent=2)
    logger.info(f"💾 Saved optimized params: {len(clean)} symbols total")


def optimize_symbol(symbol: str) -> dict:
    """
    Run optimizer for a single symbol: fetch 14d data, run grid search,
    return best params dict.
    """
    import pandas as pd
    from pybit.unified_trading import HTTP
    from engine.strategy import StrategyConfig
    from runner.backtest_runner import BacktestConfig
    from runner.optimizer import Optimizer, get_default_param_grid, apply_symmetry

    logger.info(f"  🔬 Optimizing {symbol} (14d backtest)...")

    try:
        # Fetch 14 days of 15m candles from Bybit
        session = HTTP(testnet=False)

        # Calculate start time (14 days ago)
        end_ms = int(time.time() * 1000)
        start_ms = end_ms - (OPTIMIZATION_DAYS * 24 * 3600 * 1000)

        all_candles = []
        cursor_start = start_ms

        while cursor_start < end_ms:
            resp = session.get_kline(
                category="linear",
                symbol=symbol,
                interval="15",
                start=cursor_start,
                end=end_ms,
                limit=1000,
            )
            klines = resp.get("result", {}).get("list", [])
            if not klines:
                break

            all_candles.extend(klines)

            # Bybit returns newest first — get oldest timestamp
            oldest_ts = int(klines[-1][0])
            # Move cursor forward past what we fetched
            if oldest_ts <= cursor_start:
                break
            cursor_start = int(klines[0][0]) + 1  # newest + 1

            if len(klines) < 1000:
                break

        if len(all_candles) < 200:
            logger.warning(f"  ⚠️ Insufficient data for {symbol}: {len(all_candles)} candles")
            return {}

        # Convert to DataFrame
        df = pd.DataFrame(all_candles, columns=[
            "timestamp", "open", "high", "low", "close", "volume", "turnover"
        ])
        df = df.astype({
            "timestamp": "int64",
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "float64",
            "turnover": "float64",
        })
        # Sort ascending (Bybit returns descending)
        df = df.sort_values("timestamp").reset_index(drop=True)
        # Remove duplicates
        df = df.drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

        # Run optimizer
        param_grid = get_default_param_grid()
        bt_cfg = BacktestConfig(symbol=symbol)
        optimizer = Optimizer(
            param_grid=param_grid,
            base_config=StrategyConfig(),
            backtest_config=bt_cfg,
            max_workers=2,
        )

        results_df = optimizer.run(df, df_1m=None, parallel=True)
        best_params = optimizer.get_best_params(results_df)

        if best_params:
            logger.info(f"  ✅ {symbol}: best params found (PnL={results_df.iloc[0]['total_pnl']:+.2f})")
        else:
            logger.warning(f"  ⚠️ {symbol}: optimization returned empty params")

        return best_params

    except Exception as e:
        logger.error(f"  ❌ Optimization failed for {symbol}: {e}")
        return {}


def ensure_all_optimized(symbols: list[str]) -> dict[str, dict]:
    """
    Ensure all given symbols have entries in optimized_params.
    Run optimizer for any that are missing. Returns full merged params.
    """
    existing = load_existing_params()
    missing = [s for s in symbols if s not in existing]

    if not missing:
        logger.info(f"✅ All {len(symbols)} symbols already optimized")
        return existing

    logger.info(f"🔬 {len(missing)} symbols need optimization: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    for symbol in missing:
        params = optimize_symbol(symbol)
        if params:
            existing[symbol] = params
        else:
            # Store empty dict so we don't retry every hour
            existing[symbol] = {}

    # Save merged results
    save_merged_params(existing)
    return existing


# ══════════════════════════════════════════════════════════════════
# STEP 3: Beta-Neutral Portfolio Selection
# ══════════════════════════════════════════════════════════════════

def select_portfolio(top_symbols: list[str], all_params: dict[str, dict]) -> list[str]:
    """
    Apply Beta-Neutral sector selection logic on the Top 30 symbols.
    Uses portfolio_selector.select_best_per_sector with a synthetic DataFrame.

    Strategy:
      - For each symbol in Top 30, we have optimized params (may be empty).
      - Rank by "presence in sector clusters" → pick best per sector.
      - If a symbol isn't in any cluster, skip it (unclassified).
      - Maximum: 10 symbols (best-of-sector from Top 30).
    """
    import pandas as pd

    # Build a ranking DataFrame
    # Since we don't have live backtest PnL per symbol at this point,
    # we rank by turnover position (earlier = higher turnover = better liquidity)
    rows = []
    for rank, symbol in enumerate(top_symbols):
        # Higher rank (lower index) = higher turnover = better score
        # Score: normalized 0-100, higher = better
        score = 100.0 - (rank * (100.0 / len(top_symbols)))
        params = all_params.get(symbol, {})

        rows.append({
            "symbol": symbol,
            "total_pnl": score,  # Use turnover rank as proxy for PnL selection
            "win_rate": 50.0 + (score / 10.0),  # Slight bias for higher-volume coins
            "return_pct": score * 0.5,
            "max_drawdown_pct": 5.0,
        })

    df = pd.DataFrame(rows)

    # Use portfolio_selector logic
    selections = select_best_per_sector(df)

    # Extract selected symbols (non-None sectors)
    portfolio = []
    for sector, data in selections.items():
        if data and data["symbol"] in top_symbols:
            portfolio.append(data["symbol"])

    # Cap at 10 symbols maximum
    portfolio = portfolio[:10]

    logger.info(
        f"🎯 Beta-Neutral selection: {len(portfolio)} symbols from "
        f"{len(SECTOR_CLUSTERS)} sectors"
    )
    for i, sym in enumerate(portfolio, 1):
        logger.info(f"   {i}. {sym}")

    return portfolio


# ══════════════════════════════════════════════════════════════════
# STEP 4: Save Active Portfolio
# ══════════════════════════════════════════════════════════════════

def save_active_portfolio(symbols: list[str]):
    """Save the active portfolio list to data/active_portfolio.json."""
    os.makedirs(os.path.dirname(ACTIVE_PORTFOLIO_PATH), exist_ok=True)

    payload = {
        "symbols": symbols,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(symbols),
    }

    with open(ACTIVE_PORTFOLIO_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    logger.info(f"💾 Saved active portfolio: {len(symbols)} symbols → {ACTIVE_PORTFOLIO_PATH}")


# ══════════════════════════════════════════════════════════════════
# MAIN DAEMON LOOP
# ══════════════════════════════════════════════════════════════════

def run_research_cycle():
    """Execute one full research cycle."""
    logger.info(f"\n{'═'*60}")
    logger.info(f"  RESEARCHER CYCLE | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    logger.info(f"{'═'*60}")

    # Step 1: Fetch top symbols by turnover
    top_symbols = fetch_top_by_turnover(TOP_N_BY_TURNOVER)
    if not top_symbols:
        logger.error("❌ No symbols fetched — skipping cycle")
        return False

    # Step 2: Ensure all have optimized params
    all_params = ensure_all_optimized(top_symbols)

    # Step 3: Beta-Neutral selection
    portfolio = select_portfolio(top_symbols, all_params)
    if not portfolio:
        logger.error("❌ Portfolio selection returned empty — skipping save")
        return False

    # Step 4: Save active portfolio
    save_active_portfolio(portfolio)

    logger.info(f"✅ Research cycle complete | Next run in {SLEEP_INTERVAL_SECONDS // 60} minutes")
    return True


def main():
    parser = argparse.ArgumentParser(description="Portfolio Researcher Daemon")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single cycle and exit (no loop)",
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("🔬 Portfolio Researcher starting...")
    logger.info(f"   Top N: {TOP_N_BY_TURNOVER} | Sleep: {SLEEP_INTERVAL_SECONDS}s | Days: {OPTIMIZATION_DAYS}")

    if args.once:
        run_research_cycle()
        return

    # Continuous daemon loop
    while True:
        try:
            run_research_cycle()
        except KeyboardInterrupt:
            logger.info("🛑 Researcher stopped by user")
            break
        except Exception as e:
            logger.error(f"❌ Research cycle error: {e}", exc_info=True)

        # Sleep between cycles
        logger.info(f"💤 Sleeping {SLEEP_INTERVAL_SECONDS // 60} minutes...")
        try:
            time.sleep(SLEEP_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            logger.info("🛑 Researcher stopped by user during sleep")
            break


if __name__ == "__main__":
    main()
