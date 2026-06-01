#!/usr/bin/env python3
"""
v5.3 — Beta-Neutral Portfolio Selector
══════════════════════════════════════════════
Parses optimization_results.csv and selects the BEST performing coin
per sector to eliminate cross-asset correlation.

Usage:
  python scripts/portfolio_selector.py [--csv optimization_results.csv]

Output:
  Prints a formatted SYMBOLS = [...] list for config.py
"""
import argparse
import sys
import os

import pandas as pd

# ══════════════════════════════════════════════════════════════════
# SECTOR CLUSTERS (hardcoded)
# ══════════════════════════════════════════════════════════════════

SECTOR_CLUSTERS = {
    "Majors": ["BTCUSDT", "ETHUSDT"],
    "L1/L2": [
        "SOLUSDT", "AVAXUSDT", "APTUSDT", "SUIUSDT", "SEIUSDT",
        "NEARUSDT", "TONUSDT", "ADAUSDT", "DOTUSDT", "MATICUSDT",
        "TRXUSDT", "BCHUSDT", "LTCUSDT", "ARBUSDT", "OPUSDT",
    ],
    "DeFi": ["UNIUSDT", "AAVEUSDT", "COMPUSDT", "MKRUSDT", "RUNEUSDT"],
    "Memes": ["DOGEUSDT", "WIFUSDT"],
    "Gaming/Metaverse": ["MANAUSDT", "GALAUSDT", "PORTALUSDT", "IMXUSDT", "SANDUSDT"],
    "AI/DePIN": ["RENDERUSDT", "FETUSDT", "FILUSDT"],
    "Infra/Oracles": ["LINKUSDT", "ATOMUSDT", "STXUSDT", "TIAUSDT"],
    "Payments/Misc": ["XRPUSDT", "XLMUSDT", "ALGOUSDT", "IOTAUSDT", "EOSUSDT", "BRUSDT"],
}


def select_best_per_sector(df: pd.DataFrame) -> dict[str, dict]:
    """
    For each sector, find the coin with:
      Primary: highest total_pnl
      Secondary: highest win_rate (tiebreaker)

    Returns:
        {sector_name: {"symbol": "BTCUSDT", "total_pnl": 150.0, "win_rate": 65.0}}
    """
    results = {}

    for sector, symbols in SECTOR_CLUSTERS.items():
        # Filter rows for this sector
        sector_df = df[df["symbol"].isin(symbols)].copy()

        if sector_df.empty:
            results[sector] = None
            continue

        # Group by symbol, take the BEST result per symbol (top-1 from optimizer)
        # If optimization_results.csv has multiple rows per symbol (from grid search),
        # take the best row per symbol first
        best_per_symbol = sector_df.sort_values(
            by=["total_pnl", "win_rate"],
            ascending=[False, False],
        ).drop_duplicates(subset=["symbol"], keep="first")

        if best_per_symbol.empty:
            results[sector] = None
            continue

        # Now pick the best symbol in this sector
        best_row = best_per_symbol.iloc[0]
        results[sector] = {
            "symbol": best_row["symbol"],
            "total_pnl": float(best_row["total_pnl"]),
            "win_rate": float(best_row.get("win_rate", 0)),
            "return_pct": float(best_row.get("return_pct", 0)),
            "max_drawdown_pct": float(best_row.get("max_drawdown_pct", 0)),
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Beta-Neutral Portfolio Selector")
    parser.add_argument(
        "--csv",
        default="optimization_results.csv",
        help="Path to optimization_results.csv",
    )
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"ERROR: File not found: {args.csv}")
        print("Run the multi-symbol optimizer first:")
        print("  python run_optimizer.py --all --days 30")
        sys.exit(1)

    # Load results
    df = pd.read_csv(args.csv)

    # Verify required columns
    required_cols = ["symbol", "total_pnl"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"ERROR: Missing columns in CSV: {missing}")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    print(f"\n{'═'*70}")
    print(f"  BETA-NEUTRAL PORTFOLIO SELECTOR")
    print(f"{'═'*70}")
    print(f"  Source: {args.csv} ({len(df)} rows, {df['symbol'].nunique()} symbols)")
    print(f"{'═'*70}\n")

    # Select best per sector
    selections = select_best_per_sector(df)

    # Display results
    print(f"{'Sector':<20} {'Symbol':<12} {'PnL':>10} {'WR%':>7} {'Return%':>9} {'MaxDD%':>8}")
    print(f"{'─'*70}")

    selected_symbols = []
    for sector, data in selections.items():
        if data is None:
            print(f"{sector:<20} {'(no data)':<12}")
        else:
            symbol = data["symbol"]
            selected_symbols.append(symbol)
            print(
                f"{sector:<20} {symbol:<12} "
                f"{data['total_pnl']:>+10.2f} "
                f"{data['win_rate']:>6.1f}% "
                f"{data['return_pct']:>+8.1f}% "
                f"{data['max_drawdown_pct']:>7.1f}%"
            )

    print(f"{'─'*70}")
    print(f"  Selected: {len(selected_symbols)} symbols from {len(SECTOR_CLUSTERS)} sectors\n")

    # Output formatted Python list
    print(f"{'═'*70}")
    print(f"  COPY-PASTE INTO config.py:")
    print(f"{'═'*70}\n")

    # Format nicely in rows of 5
    print("SYMBOLS: list[str] = [")
    for i in range(0, len(selected_symbols), 5):
        chunk = selected_symbols[i:i+5]
        line = ", ".join(f'"{s}"' for s in chunk)
        comma = "," if i + 5 < len(selected_symbols) else ""
        print(f"    {line}{comma}")
    print("]")
    print()

    # Also print sector reasoning
    print(f"# Portfolio rationale ({len(selected_symbols)} coins, {len(SECTOR_CLUSTERS)} sectors):")
    for sector, data in selections.items():
        if data:
            print(f"#   {sector}: {data['symbol']} (PnL={data['total_pnl']:+.2f}, WR={data['win_rate']:.1f}%)")
        else:
            print(f"#   {sector}: (no data available)")
    print()


if __name__ == "__main__":
    main()
