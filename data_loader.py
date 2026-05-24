"""
Aegis-Quant-Lab — Data Loader & History Manager
=================================================
Two operating modes:
  1. LOCAL  — reads cached CSV files from  data/history/
  2. BYBIT  — downloads OHLCV candles via ccxt (public, no API key)
             with strict pagination + rate-limiting to avoid HTTP 429.

All downloaded data is cached as CSV so subsequent runs are instant.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Callable

import ccxt
import numpy as np
import pandas as pd

import config

logger = logging.getLogger("aegis.data_loader")

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _symbol_to_filename(symbol: str, timeframe: str) -> str:
    """Convert 'BTC/USDT' + '1h' → 'BTCUSDT_1h.csv'"""
    safe = symbol.replace("/", "")
    return f"{safe}_{timeframe}.csv"


def _csv_path(symbol: str, timeframe: str) -> str:
    return os.path.join(config.DATA_DIR, _symbol_to_filename(symbol, timeframe))


def _parse_ohlcv(raw: list[list]) -> pd.DataFrame:
    """Convert ccxt OHLCV list → clean DataFrame."""
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    # Downcast to float32 for memory efficiency
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(np.float32)
    return df


# ──────────────────────────────────────────────
# Local CSV loader
# ──────────────────────────────────────────────

def load_local(symbol: str, timeframe: str,
               start: datetime | None = None,
               end: datetime | None = None) -> pd.DataFrame | None:
    """Read cached CSV. Returns None if file doesn't exist."""
    path = _csv_path(symbol, timeframe)
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(np.float32)
    if start is not None:
        start_utc = start if start.tzinfo else start.replace(tzinfo=timezone.utc)
        df = df[df.index >= start_utc]
    if end is not None:
        end_utc = end if end.tzinfo else end.replace(tzinfo=timezone.utc)
        df = df[df.index <= end_utc]
    return df if len(df) > 0 else None


def save_local(df: pd.DataFrame, symbol: str, timeframe: str) -> str:
    """Persist DataFrame to CSV cache."""
    path = _csv_path(symbol, timeframe)
    df.to_csv(path)
    return path


# ──────────────────────────────────────────────
# Bybit downloader with pagination & rate limit
# ──────────────────────────────────────────────

def _create_exchange() -> ccxt.bybit:
    """Instantiate a Bybit exchange object (public, no keys)."""
    return ccxt.bybit({
        "enableRateLimit": True,
        "options": {"defaultType": "linear"},
    })


def download_symbol(
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    progress_cb: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """
    Download OHLCV candles from Bybit with pagination.
    Handles rate-limiting with exponential back-off.
    Merges with existing local cache to avoid re-downloading.
    """
    exchange = _create_exchange()
    exchange.load_markets()

    # Resolve Bybit symbol mapping
    bybit_symbol = symbol
    if symbol not in exchange.markets:
        # Try linear perp variant
        alt = symbol.replace("/", "") + ":USDT"
        if alt in exchange.markets:
            bybit_symbol = alt

    since_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    all_candles: list[list] = []
    current_since = since_ms
    retries = 0

    while current_since < end_ms:
        try:
            candles = exchange.fetch_ohlcv(
                bybit_symbol,
                timeframe=timeframe,
                since=current_since,
                limit=config.BYBIT_BATCH_SIZE,
            )
            if not candles:
                break

            all_candles.extend(candles)
            last_ts = candles[-1][0]

            if last_ts <= current_since:
                break  # no progress — stop
            current_since = last_ts + 1  # next batch starts after last candle

            retries = 0  # reset on success
            msg = (f"  {symbol} [{timeframe}]: fetched {len(all_candles)} candles "
                   f"up to {datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')}")
            logger.info(msg)
            if progress_cb:
                progress_cb(msg)

            time.sleep(config.BYBIT_RATE_LIMIT_SLEEP)

        except ccxt.RateLimitExceeded:
            retries += 1
            wait = min(2 ** retries, 30)
            msg = f"  Rate limit hit for {symbol}. Waiting {wait}s …"
            logger.warning(msg)
            if progress_cb:
                progress_cb(msg)
            time.sleep(wait)
            if retries >= config.BYBIT_MAX_RETRIES:
                logger.error(f"Max retries exceeded for {symbol}")
                break

        except ccxt.NetworkError as e:
            retries += 1
            wait = min(2 ** retries, 30)
            logger.warning(f"Network error for {symbol}: {e}. Retry in {wait}s")
            time.sleep(wait)
            if retries >= config.BYBIT_MAX_RETRIES:
                break

        except Exception as e:
            logger.error(f"Unexpected error downloading {symbol}: {e}")
            break

    if not all_candles:
        logger.warning(f"No data downloaded for {symbol} [{timeframe}]")
        return pd.DataFrame()

    df_new = _parse_ohlcv(all_candles)
    # Filter to requested range
    start_utc = start if start.tzinfo else start.replace(tzinfo=timezone.utc)
    end_utc = end if end.tzinfo else end.replace(tzinfo=timezone.utc)
    df_new = df_new[(df_new.index >= start_utc) & (df_new.index <= end_utc)]

    # Merge with existing cache
    existing = load_local(symbol, timeframe)
    if existing is not None and len(existing) > 0:
        df_merged = pd.concat([existing, df_new])
        df_merged = df_merged[~df_merged.index.duplicated(keep="last")]
        df_merged.sort_index(inplace=True)
    else:
        df_merged = df_new

    save_local(df_merged, symbol, timeframe)
    return df_new.copy()


def download_batch(
    symbols: list[str],
    timeframe: str,
    start: datetime,
    end: datetime,
    progress_cb: Callable[[str], None] | None = None,
) -> dict[str, pd.DataFrame]:
    """Download OHLCV for multiple symbols sequentially with rate-limiting."""
    results: dict[str, pd.DataFrame] = {}
    total = len(symbols)
    for idx, sym in enumerate(symbols, 1):
        msg = f"[{idx}/{total}] Downloading {sym} ({timeframe}) …"
        logger.info(msg)
        if progress_cb:
            progress_cb(msg)
        try:
            df = download_symbol(sym, timeframe, start, end, progress_cb)
            if len(df) > 0:
                results[sym] = df
                msg_ok = f"  ✓ {sym}: {len(df)} candles"
            else:
                msg_ok = f"  ✗ {sym}: no data"
            logger.info(msg_ok)
            if progress_cb:
                progress_cb(msg_ok)
        except Exception as e:
            msg_err = f"  ✗ {sym}: ERROR — {e}"
            logger.error(msg_err)
            if progress_cb:
                progress_cb(msg_err)
        # Extra sleep between symbols
        time.sleep(config.BYBIT_RATE_LIMIT_SLEEP * 2)
    return results


def load_all_local(
    symbols: list[str],
    timeframe: str,
    start: datetime | None = None,
    end: datetime | None = None,
) -> dict[str, pd.DataFrame]:
    """Load cached data for multiple symbols."""
    results: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = load_local(sym, timeframe, start, end)
        if df is not None and len(df) > 0:
            results[sym] = df
    return results
