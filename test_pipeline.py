#!/usr/bin/env python3
"""
Aegis-Quant-Lab — Integration Test
====================================
Validates the full pipeline with synthetic OHLCV data.
No Bybit API calls needed.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import data_loader
import feature_factory
import cross_asset_analyst
import backtest_simulator
import risk_generator
import visualizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("test")


def generate_synthetic_ohlcv(n_bars: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate realistic synthetic OHLCV data."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_bars, freq="1h", tz="UTC")

    # Random walk with drift
    log_returns = rng.normal(0.0001, 0.015, n_bars)
    close = 100 * np.exp(np.cumsum(log_returns))

    high = close * (1 + rng.uniform(0.001, 0.02, n_bars))
    low = close * (1 - rng.uniform(0.001, 0.02, n_bars))
    opn = close * (1 + rng.normal(0, 0.005, n_bars))
    volume = rng.lognormal(10, 1, n_bars)

    df = pd.DataFrame({
        "open": opn.astype(np.float32),
        "high": high.astype(np.float32),
        "low": low.astype(np.float32),
        "close": close.astype(np.float32),
        "volume": volume.astype(np.float32),
    }, index=dates)
    df.index.name = "timestamp"
    return df


def main():
    print("=" * 60)
    print("  AEGIS-QUANT-LAB — INTEGRATION TEST")
    print("=" * 60)

    # Generate test data for 5 synthetic "coins"
    symbols = ["SYN1/USDT", "SYN2/USDT", "SYN3/USDT", "SYN4/USDT", "SYN5/USDT"]
    raw_data = {}
    for i, sym in enumerate(symbols):
        raw_data[sym] = generate_synthetic_ohlcv(n_bars=2000, seed=42 + i)
        print(f"  Generated {sym}: {len(raw_data[sym])} bars")

    # Phase 2: Feature Factory
    print("\n--- PHASE 2: Feature Factory ---")
    enriched = feature_factory.compute_features_batch(raw_data, progress_cb=print)
    for sym, df in enriched.items():
        print(f"  {sym}: {len(df)} bars x {len(df.columns)} features")

    assert len(enriched) > 0, "Feature factory produced no output"
    first_df = list(enriched.values())[0]
    assert len(first_df.columns) > 20, f"Too few features: {len(first_df.columns)}"
    print(f"  PASS: {len(first_df.columns)} features generated")

    # Phase 3: Cross-asset
    print("\n--- PHASE 3: Cross-Asset Analysis ---")
    cross_results = cross_asset_analyst.run_full_analysis(raw_data, progress_cb=print)
    pcorr = cross_results.get("pearson_corr")
    assert pcorr is not None and not pcorr.empty, "No Pearson correlation matrix"
    print(f"  Pearson corr matrix: {pcorr.shape}")
    print(f"  High corr pairs: {len(cross_results.get('high_corr_pairs', []))}")
    print(f"  Cointegrated pairs: {len(cross_results.get('cointegrated_pairs', []))}")
    print(f"  Clusters: {cross_results.get('hierarchical_clusters', {})}")
    print(f"  Betas: {cross_results.get('betas', {})}")
    print("  PASS")

    # Phase 4: Backtest
    print("\n--- PHASE 4: Backtest Simulation ---")
    bt_results = backtest_simulator.run_full_backtest(enriched, "1h", progress_cb=print)
    top = bt_results.get("top_strategies", [])
    losers = bt_results.get("losing_patterns", [])
    print(f"  Top valid strategies: {len(top)}")
    print(f"  Losing patterns: {len(losers)}")
    if top:
        m = top[0]
        print(f"  Best: {m.name} ({m.symbol}) | SR={m.sharpe_ratio:.2f} PF={m.profit_factor:.2f} WR={m.win_rate:.1%}")
    print("  PASS")

    # Phase 5: Risk blocks
    print("\n--- PHASE 5: Risk Block Generation ---")
    blocks = risk_generator.generate_risk_blocks(bt_results, enriched, cross_results, progress_cb=print)
    path = risk_generator.save_risk_blocks(blocks)
    print(f"  Generated {len(blocks)} risk blocks")
    print(f"  Saved to: {path}")

    # Verify JSON structure
    with open(path, "r") as f:
        loaded = json.load(f)
    assert isinstance(loaded, list), "risk_blocks.json should be a list"
    print("  PASS: Valid JSON structure")

    # Phase 6: Report
    report = risk_generator.generate_summary_report(bt_results, cross_results)
    report_path = os.path.join(config.OUTPUT_DIR, "summary_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n  Report saved: {report_path}")

    # Phase 7: Charts
    print("\n--- PHASE 6: Visualisation ---")
    charts = visualizer.generate_all_charts(bt_results, cross_results)
    for c in charts:
        assert os.path.isfile(c), f"Chart file not found: {c}"
        print(f"  Chart: {c}")
    print(f"  PASS: {len(charts)} charts generated")

    print("\n" + "=" * 60)
    print("  ALL INTEGRATION TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
