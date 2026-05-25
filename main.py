#!/usr/bin/env python3
"""
Aegis-Quant-Lab — Main Entry Point
====================================
Launch the application:
    python main.py          →  GUI mode (default)
    python main.py --cli    →  headless CLI mode (for servers / CI)

Architecture:
    gui_interface.py        Lightweight GUI (customtkinter / tkinter)
    data_loader.py          Bybit OHLCV download & CSV cache
    feature_factory.py      200+ indicator generation (pandas-ta)
    cross_asset_analyst.py  Correlation / cointegration / clustering
    backtest_simulator.py   Walk-Forward + statistical validation
    risk_generator.py       Anti-strategy risk_blocks.json export
    visualizer.py           Chart generation (matplotlib / seaborn)
    config.py               Central configuration constants
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config


def setup_root_logging():
    """Configure root 'aegis' logger for console + file."""
    root = logging.getLogger("aegis")
    root.setLevel(logging.INFO)

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S"))
    root.addHandler(ch)

    # File
    fh = logging.FileHandler(os.path.join(config.LOG_DIR, "aegis_quant_lab.log"), encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s — %(message)s"))
    root.addHandler(fh)
    return root


def run_cli(symbols: list[str], timeframe: str, start_str: str, end_str: str):
    """Run the full pipeline in headless CLI mode."""
    from datetime import datetime, timezone

    import data_loader
    import feature_factory
    import cross_asset_analyst
    import backtest_simulator
    import risk_generator
    import visualizer

    logger = logging.getLogger("aegis.cli")
    start = datetime.strptime(start_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(end_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    def log(msg: str):
        logger.info(msg)

    # Phase 1: Load / download data
    log(f"\n{'=' * 60}")
    log("PHASE 1: Loading data …")
    raw_data = data_loader.load_all_local(symbols, timeframe, start, end)
    if not raw_data:
        log("No cached data. Downloading from Bybit …")
        raw_data = data_loader.download_batch(symbols, timeframe, start, end, progress_cb=log)
    log(f"Loaded {len(raw_data)} symbols")

    if not raw_data:
        log("ERROR: No data available. Exiting.")
        return

    # Phase 2: Features
    log(f"\n{'=' * 60}")
    log("PHASE 2: Feature Factory …")
    enriched = feature_factory.compute_features_batch(raw_data, progress_cb=log)
    log(f"Features computed for {len(enriched)} symbols")

    # Phase 3: Cross-asset
    log(f"\n{'=' * 60}")
    log("PHASE 3: Cross-Asset Analysis …")
    cross_results = cross_asset_analyst.run_full_analysis(raw_data, progress_cb=log)

    # Phase 4: Backtest
    log(f"\n{'=' * 60}")
    log("PHASE 4: Backtest Simulation …")
    bt_results = backtest_simulator.run_full_backtest(enriched, timeframe, progress_cb=log)

    # Phase 5: Risk blocks
    log(f"\n{'=' * 60}")
    log("PHASE 5: Risk Block Generation …")
    blocks = risk_generator.generate_risk_blocks(bt_results, enriched, cross_results, progress_cb=log)
    risk_path = risk_generator.save_risk_blocks(blocks)
    log(f"Risk blocks saved: {risk_path}")

    # Report
    report = risk_generator.generate_summary_report(bt_results, cross_results)
    log(report)
    report_path = os.path.join(config.OUTPUT_DIR, "summary_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    # Charts
    log(f"\n{'=' * 60}")
    log("PHASE 6: Generating charts …")
    charts = visualizer.generate_all_charts(bt_results, cross_results)
    for c in charts:
        log(f"  Chart: {c}")

    log(f"\nDone. Outputs in: {config.OUTPUT_DIR}")


def main():
    parser = argparse.ArgumentParser(
        description="Aegis-Quant-Lab — Ultimate Quantitative Backtesting Laboratory",
    )
    parser.add_argument("--cli", action="store_true", help="Run in headless CLI mode")
    parser.add_argument("--symbols", nargs="*", default=None, help="Symbol list (CLI mode)")
    parser.add_argument("--timeframe", default="1h", choices=config.SUPPORTED_TIMEFRAMES)
    parser.add_argument("--start", default="2024-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2025-01-01", help="End date YYYY-MM-DD")
    args = parser.parse_args()

    setup_root_logging()
    logger = logging.getLogger("aegis.main")
    logger.info("Aegis-Quant-Lab starting …")

    if args.cli:
        symbols = args.symbols if args.symbols else config.DEFAULT_SYMBOLS[:10]
        run_cli(symbols, args.timeframe, args.start, args.end)
    else:
        # GUI mode
        try:
            from gui_interface import AegisQuantLabGUI
            app = AegisQuantLabGUI()
            app.run()
        except Exception as e:
            logger.error(f"GUI failed to start: {e}")
            logger.info("Falling back to CLI mode …")
            run_cli(config.DEFAULT_SYMBOLS[:10], "1h", "2024-01-01", "2025-01-01")


if __name__ == "__main__":
    main()
