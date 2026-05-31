#!/usr/bin/env python3
"""
v5.0 — Live Trading Entry Point
══════════════════════════════════
Run: python run_live.py

Environment variables required:
  BYBIT_API_KEY    — Bybit Demo API key
  BYBIT_API_SECRET — Bybit Demo API secret
"""
import logging
import os
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/live_v5.log", mode="a"),
    ],
)
logger = logging.getLogger("aegis.main")

# Ensure logs dir
os.makedirs("logs", exist_ok=True)


def main():
    from engine.strategy import FiboReversalStrategy, StrategyConfig
    from broker.live_broker import LiveBroker
    from runner.live_runner import LiveRunner
    from config import SYMBOLS, BYBIT_DEMO_ENDPOINT

    # API credentials
    api_key = os.environ.get("BYBIT_API_KEY", "")
    api_secret = os.environ.get("BYBIT_API_SECRET", "")

    if not api_key or not api_secret:
        logger.error("❌ Missing BYBIT_API_KEY or BYBIT_API_SECRET environment variables")
        sys.exit(1)

    # Config
    cfg = StrategyConfig()

    # Initialize components
    logger.info("═" * 50)
    logger.info("  AEGIS v5.0 — Live Trading Mode")
    logger.info("═" * 50)

    broker = LiveBroker(api_key, api_secret, leverage=cfg.leverage)
    strategy = FiboReversalStrategy(cfg, initial_balance=2000.0)

    runner = LiveRunner(
        broker=broker,
        strategy=strategy,
        symbols=SYMBOLS,
        scan_interval=60,
        candle_interval="15",
        candle_limit=200,
    )

    # Run
    runner.run()


if __name__ == "__main__":
    main()
