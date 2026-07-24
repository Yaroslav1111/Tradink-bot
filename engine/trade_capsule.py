"""
v5.5 — "Trade Capsule" Meta-Analysis Subsystem
════════════════════════════════════════════════
A lightweight, NON-BLOCKING logging module that records a token-optimized
JSON "Trade Capsule" to data/ai_analysis/ for every completed trade cycle.

Each capsule contains EXACTLY four pillars, ready for Gemini 1.5 Pro
evaluation of the "Two-Winged" dual-lot exit engine:

  1. Pre-Trade Context   — OHLCV snapshot immediately prior to entry.
  2. Internal Thoughts   — exact oscillator/structural params at decision ms.
  3. Execution Reality   — slippage, partial fill states, limit exec time.
  4. Post-Trade Reality  — next 10–15 candle audit (premature vs. max impulse).

Design constraints:
  - Non-blocking: writes are best-effort; any I/O failure is swallowed so
    the trading/backtest loop never stalls or crashes on a logging error.
  - Token-optimized: floats are rounded, OHLCV is capped, nulls dropped.
  - Pure of trading logic: this module ONLY assembles + serializes.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import pandas as pd

from engine.models import (
    Direction, Position, TradeCapsule, PreTradeContext,
    InternalThoughts, ExecutionReality, PostTradeReality, Signal,
)

logger = logging.getLogger("aegis.capsule")

DEFAULT_CAPSULE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "ai_analysis",
)

# How many bars of context / lookahead to embed (token budget control)
PRE_TRADE_BARS = 12
POST_TRADE_BARS = 15


def _round(x: float, nd: int = 6) -> float:
    try:
        return round(float(x), nd)
    except (TypeError, ValueError):
        return 0.0


def _ohlcv_rows(df: Optional[pd.DataFrame], n: int) -> list[list[float]]:
    """Extract the last n OHLCV rows as compact [ts,o,h,l,c,v] lists."""
    if df is None or len(df) == 0:
        return []
    tail = df.tail(n)
    rows: list[list[float]] = []
    for _, r in tail.iterrows():
        rows.append([
            _round(r.get("timestamp", 0.0), 1),
            _round(r.get("open", 0.0), 6),
            _round(r.get("high", 0.0), 6),
            _round(r.get("low", 0.0), 6),
            _round(r.get("close", 0.0), 6),
            _round(r.get("volume", 0.0), 2),
        ])
    return rows


# ══════════════════════════════════════════════════════════════════
# CAPSULE BUILDERS (pure — no I/O)
# ══════════════════════════════════════════════════════════════════

def build_pre_trade(symbol: str, direction: Direction, df_15m: Optional[pd.DataFrame],
                    current_price: float, entry_time: float) -> PreTradeContext:
    return PreTradeContext(
        symbol=symbol,
        direction=direction.value,
        entry_time=_round(entry_time, 1),
        ohlcv=_ohlcv_rows(df_15m, PRE_TRADE_BARS),
        current_price=_round(current_price, 6),
    )


def build_internal_thoughts(signal: Signal, pivot_level: float,
                            structural_stop: float) -> InternalThoughts:
    return InternalThoughts(
        rsi=_round(signal.rsi_value, 3),
        cci=_round(signal.cci_value, 3),
        willr=_round(signal.willr_value, 3),
        oscillators_firing=int(signal.oscillators_firing),
        atr=_round(signal.atr, 6),
        swing_high=_round(signal.swing_high, 6),
        swing_low=_round(signal.swing_low, 6),
        fibo_ext_1=_round(signal.fibo_ext_1, 6),
        fibo_ext_2=_round(signal.fibo_ext_2, 6),
        poc_price=(None if signal.poc_price is None else _round(signal.poc_price, 6)),
        pivot_level=_round(pivot_level, 6),
        structural_stop=_round(structural_stop, 6),
        rationale=(
            f"{signal.oscillators_firing} oscillators fired at extremes; "
            f"entered {signal.direction.value} at fibo grid, structural stop "
            f"behind pivot {pivot_level:.4f}."
        ),
    )


def build_post_trade(direction: Direction, exit_price: float, exit_time: float,
                     df_after: Optional[pd.DataFrame]) -> PostTradeReality:
    """
    Audit the next 10–15 candles after full closure to evaluate whether
    Lot B exited prematurely or caught the maximum impulse.
    """
    rows = _ohlcv_rows(df_after, POST_TRADE_BARS)
    ptr = PostTradeReality(
        lot_b_exit_price=_round(exit_price, 6),
        lot_b_exit_time=_round(exit_time, 1),
        lookahead_candles=len(rows),
        post_exit_ohlcv=rows,
    )
    if not rows or exit_price <= 0:
        return ptr

    highs = [r[2] for r in rows]
    lows = [r[3] for r in rows]
    if direction == Direction.LONG:
        best = max(highs) if highs else exit_price
        fav_pct = (best - exit_price) / exit_price
    else:
        best = min(lows) if lows else exit_price
        fav_pct = (exit_price - best) / exit_price

    ptr.max_favorable_after_exit = _round(best, 6)
    ptr.max_favorable_pct = _round(fav_pct, 5)
    # If the market kept moving >0.5% in our favor after we exited → premature.
    ptr.exited_prematurely = bool(fav_pct > 0.005)
    # If almost nothing was left on the table (<0.2%) → we caught the impulse.
    ptr.caught_max_impulse = bool(fav_pct <= 0.002)
    return ptr


# ══════════════════════════════════════════════════════════════════
# TOKEN-OPTIMIZED SERIALIZATION
# ══════════════════════════════════════════════════════════════════

def _prune(obj):
    """Recursively drop None values and empty containers for token economy."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            pv = _prune(v)
            if pv is None:
                continue
            if isinstance(pv, (list, dict)) and len(pv) == 0:
                continue
            out[k] = pv
        return out
    if isinstance(obj, list):
        return [_prune(v) for v in obj]
    return obj


def capsule_to_dict(capsule: TradeCapsule) -> dict:
    """Convert a TradeCapsule to a token-optimized dict."""
    raw = asdict(capsule)
    return _prune(raw)


def capsule_to_json(capsule: TradeCapsule, indent: Optional[int] = None) -> str:
    return json.dumps(capsule_to_dict(capsule), separators=(",", ":") if indent is None else (",", ": "),
                      indent=indent)


# ══════════════════════════════════════════════════════════════════
# NON-BLOCKING WRITER
# ══════════════════════════════════════════════════════════════════

class CapsuleLogger:
    """
    Non-blocking Trade Capsule writer.

    Capsules are pushed onto an in-memory queue and flushed by a background
    daemon thread. If the queue/disk fails, errors are logged and swallowed —
    the trading loop is never blocked or crashed by capsule logging.

    In backtests (where determinism matters) call flush() at the end, or set
    async_mode=False to write inline (still best-effort / exception-safe).
    """

    def __init__(self, out_dir: str = DEFAULT_CAPSULE_DIR, async_mode: bool = True):
        self.out_dir = out_dir
        self.async_mode = async_mode
        self._queue: "queue.Queue[Optional[TradeCapsule]]" = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._written: list[str] = []
        self._lock = threading.Lock()
        try:
            os.makedirs(self.out_dir, exist_ok=True)
        except OSError as e:  # non-blocking: never raise
            logger.warning(f"⚠️ Capsule dir create failed: {e}")
        if async_mode:
            self._worker = threading.Thread(target=self._run, daemon=True)
            self._worker.start()

    # ── public API ──

    def log(self, capsule: TradeCapsule) -> None:
        """Enqueue (async) or write (sync) a capsule. Never raises."""
        try:
            if self.async_mode:
                self._queue.put_nowait(capsule)
            else:
                self._write_one(capsule)
        except Exception as e:  # noqa: BLE001 — best-effort logging
            logger.warning(f"⚠️ Capsule log failed (swallowed): {e}")

    def flush(self, timeout: float = 5.0) -> None:
        """Block until the queue drains (used at end of a backtest)."""
        if not self.async_mode:
            return
        try:
            self._queue.join()
        except Exception:  # noqa: BLE001
            pass

    def written_files(self) -> list[str]:
        with self._lock:
            return list(self._written)

    # ── internals ──

    def _run(self):
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                self._write_one(item)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"⚠️ Capsule write error (swallowed): {e}")
            finally:
                self._queue.task_done()

    def _write_one(self, capsule: TradeCapsule):
        path = Path(self.out_dir) / f"{capsule.capsule_id}.json"
        payload = capsule_to_dict(capsule)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        with self._lock:
            self._written.append(str(path))
        logger.debug(f"🧬 Trade Capsule saved → {path}")
