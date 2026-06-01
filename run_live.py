#!/usr/bin/env python3
"""
v5.4 — Live Trading Entry Point
══════════════════════════════════
Run: python run_live.py [--auto-sync]

Flags:
  --auto-sync   Enable dynamic portfolio rebalancing (reads data/active_portfolio.json)

Environment variables required (from .env.local):
  BYBIT_DEMO_KEY    — Bybit Demo API key
  BYBIT_DEMO_SECRET — Bybit Demo API secret
"""
import argparse
import logging
import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv(".env.local")
except ImportError:
    pass

# ── Safe Stream Wrapper ─────────────────────────────────────────────
# Windows cp1251 cannot encode many Unicode characters (emoji, arrows, etc.).
# We wrap sys.stdout and sys.stderr so ALL output is safe.

class _SafeStream:
    """Wraps a text stream, replacing characters that can't be encoded."""
    def __init__(self, stream):
        self._stream = stream
        self.encoding = stream.encoding or "utf-8"

    def write(self, data):
        safe = data.encode(self.encoding, errors="replace").decode(self.encoding)
        self._stream.write(safe)

    def flush(self):
        self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


sys.stdout = _SafeStream(sys.stdout)
sys.stderr = _SafeStream(sys.stderr)


# ── Safe Logging Handler ────────────────────────────────────────────
# Python 3.14's StreamHandler.emit() has its own try/except Exception
# that catches UnicodeEncodeError and calls self.handleError(record),
# which prints the full traceback to stderr. We override emit() to
# bypass this and do encoding-safe writes directly.

class _SafeHandler(logging.StreamHandler):
    """StreamHandler that replaces unsupported characters instead of raising."""

    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            # Replace characters that can't be encoded
            safe = msg.encode(stream.encoding, errors="replace").decode(stream.encoding)
            stream.write(safe + self.terminator)
            self.flush()
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)


def _setup_logging():
    """Configure root logger with safe console + file handlers."""
    root = logging.getLogger()
    # Remove any pre-existing handlers
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter(
        "%(asctime)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = _SafeHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)

    os.makedirs("logs", exist_ok=True)
    file_h = logging.FileHandler("logs/live_v5.log", mode="a", encoding="utf-8")
    file_h.setLevel(logging.DEBUG)
    file_h.setFormatter(fmt)

    root.setLevel(logging.DEBUG)
    root.addHandler(console)
    root.addHandler(file_h)


_setup_logging()
logger = logging.getLogger("aegis.main")


def main():
    parser = argparse.ArgumentParser(description="Aegis v5.4 — Live Trading")
    parser.add_argument(
        "--auto-sync",
        action="store_true",
        help="Enable dynamic portfolio sync (reads data/active_portfolio.json every 15 cycles)",
    )
    args = parser.parse_args()

    from engine.strategy import FiboReversalStrategy, StrategyConfig
    from broker.live_broker import LiveBroker
    from broker.exchange_rules import ensure_exchange_rules
    from runner.live_runner import LiveRunner
    from config import SYMBOLS, BYBIT_DEMO_ENDPOINT

    # API credentials — try both naming variants for compatibility
    api_key = os.environ.get("BYBIT_DEMO_API_KEY") or os.environ.get("BYBIT_DEMO_KEY", "")
    api_secret = os.environ.get("BYBIT_DEMO_API_SECRET") or os.environ.get("BYBIT_DEMO_SECRET", "")

    if not api_key or not api_secret:
        logger.error("Missing BYBIT_DEMO_KEY / BYBIT_DEMO_SECRET environment variables")
        logger.error("Ensure .env.local exists with the correct keys, or set them manually.")
        sys.exit(1)

    # Config
    cfg = StrategyConfig()

    # Load exchange rules (auto-fetch if missing or stale)
    exchange_rules = ensure_exchange_rules(symbols=SYMBOLS)

    # Initialize components
    logger.info("=" * 50)
    logger.info("  AEGIS v5.4 -- Live Trading Mode")
    logger.info("=" * 50)
    logger.info(f"  Exchange rules: {len(exchange_rules)} symbols loaded")
    logger.info(f"  Auto-sync: {args.auto_sync}")

    broker = LiveBroker(api_key, api_secret, leverage=cfg.leverage)
    strategy = FiboReversalStrategy(cfg, initial_balance=2000.0, exchange_rules=exchange_rules)

    runner = LiveRunner(
        broker=broker,
        strategy=strategy,
        symbols=SYMBOLS,
        scan_interval=60,
        candle_interval="15",
        candle_limit=200,
        exchange_rules=exchange_rules,
        auto_sync=args.auto_sync,
    )

    # Run
    runner.run()


if __name__ == "__main__":
    main()
