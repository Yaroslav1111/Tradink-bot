"""
v5.3 — Exchange Rules (Precision Database)
══════════════════════════════════════════════
Fetches and caches instrument specs from Bybit:
  - qtyStep: minimum quantity increment
  - minOrderQty: minimum order quantity
  - tickSize: minimum price increment

Saves to data/exchange_rules.json (auto-refreshed if older than 7 days).
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("aegis.exchange_rules")

# Default path for exchange rules cache
DEFAULT_RULES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "exchange_rules.json"
)

# Max age before refresh (7 days in seconds)
RULES_MAX_AGE_SECONDS = 7 * 24 * 3600


def fetch_exchange_rules(symbols: Optional[list[str]] = None) -> dict[str, dict]:
    """
    Fetch instrument info from Bybit public API.
    
    Returns:
        {
            "BTCUSDT": {"qtyStep": 0.001, "minOrderQty": 0.001, "tickSize": 0.1},
            "ETHUSDT": {"qtyStep": 0.01, "minOrderQty": 0.01, "tickSize": 0.01},
            ...
        }
    """
    from pybit.unified_trading import HTTP

    session = HTTP(testnet=False)
    rules = {}

    try:
        # Fetch all linear instruments
        resp = session.get_instruments_info(category="linear")
        instruments = resp.get("result", {}).get("list", [])

        for inst in instruments:
            symbol = inst.get("symbol", "")
            if symbols and symbol not in symbols:
                continue

            lot_filter = inst.get("lotSizeFilter", {})
            price_filter = inst.get("priceFilter", {})

            qty_step = float(lot_filter.get("qtyStep", "0.001"))
            min_order_qty = float(lot_filter.get("minOrderQty", "0.001"))
            tick_size = float(price_filter.get("tickSize", "0.01"))

            rules[symbol] = {
                "qtyStep": qty_step,
                "minOrderQty": min_order_qty,
                "tickSize": tick_size,
            }

        logger.info(f"📐 Fetched exchange rules for {len(rules)} instruments")

    except Exception as e:
        logger.error(f"❌ Failed to fetch exchange rules: {e}")

    return rules


def save_exchange_rules(rules: dict[str, dict], path: str = DEFAULT_RULES_PATH):
    """Save exchange rules to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "timestamp": time.time(),
        "rules": rules,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"💾 Saved exchange rules → {path} ({len(rules)} symbols)")


def load_exchange_rules(path: str = DEFAULT_RULES_PATH) -> dict[str, dict]:
    """
    Load exchange rules from JSON cache.
    Returns empty dict if file doesn't exist.
    """
    p = Path(path)
    if not p.exists():
        return {}
    try:
        with open(p, "r") as f:
            data = json.load(f)
        return data.get("rules", {})
    except Exception as e:
        logger.warning(f"⚠️ Failed to load exchange rules: {e}")
        return {}


def rules_need_refresh(path: str = DEFAULT_RULES_PATH) -> bool:
    """Check if rules file is missing or older than 7 days."""
    p = Path(path)
    if not p.exists():
        return True
    try:
        with open(p, "r") as f:
            data = json.load(f)
        ts = data.get("timestamp", 0)
        age = time.time() - ts
        return age > RULES_MAX_AGE_SECONDS
    except Exception:
        return True


def ensure_exchange_rules(
    path: str = DEFAULT_RULES_PATH,
    symbols: Optional[list[str]] = None,
) -> dict[str, dict]:
    """
    Load exchange rules from cache, refresh if needed.
    This is the main entry point for runners.
    
    Returns: {"BTCUSDT": {"qtyStep": ..., "minOrderQty": ..., "tickSize": ...}, ...}
    """
    if not rules_need_refresh(path):
        rules = load_exchange_rules(path)
        if rules:
            logger.info(f"📐 Exchange rules loaded from cache ({len(rules)} symbols)")
            return rules

    # Need to fetch fresh rules
    logger.info("📐 Fetching fresh exchange rules from Bybit...")
    try:
        rules = fetch_exchange_rules(symbols)
        if rules:
            save_exchange_rules(rules, path)
            return rules
    except Exception as e:
        logger.warning(f"⚠️ Could not refresh exchange rules: {e}")

    # Fallback: load whatever we have cached
    return load_exchange_rules(path)


# ══════════════════════════════════════════════════════════════════
# PRECISION HELPERS (used by engine/strategy.py)
# ══════════════════════════════════════════════════════════════════

def floor_qty(qty: float, qty_step: float) -> float:
    """
    Floor quantity to nearest valid step (rounds DOWN).
    
    Example: floor_qty(0.1234, 0.001) → 0.123
    """
    if qty_step <= 0:
        return qty
    # Use Decimal for precise floor division to avoid floating point artifacts
    # (e.g., 4.637 / 0.001 = 4636.999... in IEEE-754)
    decimals = _step_decimals(qty_step)
    # Round qty to enough precision first to avoid float drift
    qty_rounded = round(qty, decimals + 4)
    steps = int(qty_rounded / qty_step + 1e-9)  # add epsilon for "almost exact" values
    # But we must floor, so check if we overshot
    candidate = round(steps * qty_step, decimals)
    if candidate > qty_rounded + qty_step * 1e-9:
        steps -= 1
        candidate = round(steps * qty_step, decimals)
    return candidate


def round_to_tick(price: float, tick_size: float) -> float:
    """
    Round price to nearest valid tick (nearest, not floor).
    
    Example: round_to_tick(45123.456, 0.1) → 45123.5
    """
    if tick_size <= 0:
        return price
    ticks = round(price / tick_size)
    result = ticks * tick_size
    decimals = _step_decimals(tick_size)
    return round(result, decimals)


def _step_decimals(step: float) -> int:
    """Determine number of decimal places in a step value."""
    s = f"{step:.10f}".rstrip("0")
    if "." in s:
        return len(s.split(".")[1])
    return 0
