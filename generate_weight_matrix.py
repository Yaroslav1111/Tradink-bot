"""
Aegis-Quant-Lab — Indicator Weight Matrix Generator
=====================================================
Core concept: each indicator has its own "accuracy weight" (confidence coefficient)
calculated SEPARATELY for each coin.

This module:
1. Takes historical OHLCV data (15m candles) for 50+ assets
2. Computes a comprehensive set of indicators
3. For each indicator signal, calculates forward return (after N bars)
4. Subtracts Bybit roundtrip fees (0.11% = taker entry + taker exit)
5. Produces a compact indicator_weights.json — the "brain" of the consensus system

Output format per symbol:
{
    "BTCUSDT": {
        "RSI_Oversold_Long": {
            "win_rate_pct": 62.3,
            "expected_return_per_trade_pct": 0.18,
            "total_historical_trades": 847,
            "accuracy_weight": 0.18,
            "direction": "LONG",
            "avg_win_pct": 0.45,
            "avg_loss_pct": -0.22,
            "profit_factor": 1.85,
            "max_consecutive_losses": 5
        },
        ...
    }
}
"""

from __future__ import annotations

import json
import logging
import os
from typing import Callable

import numpy as np
import pandas as pd
import pandas_ta as ta

import config

logger = logging.getLogger("aegis.weight_matrix")

# ──────────────────────────────────────────────
# Indicator Signal Definitions (15+ indicators)
# ──────────────────────────────────────────────

def _compute_indicator_signals(df: pd.DataFrame) -> dict[str, tuple[pd.Series, str]]:
    """
    Compute all indicator signals and return dict of {name: (boolean_mask, direction)}.
    Each signal indicates: "when this condition is TRUE, go LONG or SHORT".
    
    Returns dict mapping indicator_name -> (condition_series, direction_str)
    """
    close = df["close"].astype(np.float64)
    high = df["high"].astype(np.float64)
    low = df["low"].astype(np.float64)
    volume = df["volume"].astype(np.float64)

    signals: dict[str, tuple[pd.Series, str]] = {}

    # ── 1. RSI (14) ──
    rsi_14 = ta.rsi(close, length=14)
    if rsi_14 is not None:
        signals["RSI14_Oversold_Long"] = (rsi_14 < 30, "LONG")
        signals["RSI14_Overbought_Short"] = (rsi_14 > 70, "SHORT")
        signals["RSI14_DeepOversold_Long"] = (rsi_14 < 20, "LONG")
        signals["RSI14_ExtremeOverbought_Short"] = (rsi_14 > 80, "SHORT")

    # ── 2. RSI (7) — faster ──
    rsi_7 = ta.rsi(close, length=7)
    if rsi_7 is not None:
        signals["RSI7_Oversold_Long"] = (rsi_7 < 25, "LONG")
        signals["RSI7_Overbought_Short"] = (rsi_7 > 75, "SHORT")

    # ── 3. EMA Cross (12/26 = MACD proxy) ──
    ema_fast = ta.ema(close, length=12)
    ema_slow = ta.ema(close, length=26)
    if ema_fast is not None and ema_slow is not None:
        ema_diff = (ema_fast - ema_slow) / (ema_slow + 1e-10)
        signals["EMA_Bullish_Cross_Long"] = (ema_diff > 0.002, "LONG")
        signals["EMA_Bearish_Cross_Short"] = (ema_diff < -0.002, "SHORT")
        signals["EMA_StrongBull_Long"] = (ema_diff > 0.005, "LONG")
        signals["EMA_StrongBear_Short"] = (ema_diff < -0.005, "SHORT")

    # ── 4. Bollinger Bands %B ──
    bb = ta.bbands(close, length=20, std=2.0)
    if bb is not None:
        bbl_col = [c for c in bb.columns if "BBL" in c]
        bbu_col = [c for c in bb.columns if "BBU" in c]
        if bbl_col and bbu_col:
            bbl = bb[bbl_col[0]]
            bbu = bb[bbu_col[0]]
            pctb = (close - bbl) / (bbu - bbl + 1e-10)
            signals["BB_Oversold_Long"] = (pctb < 0.0, "LONG")
            signals["BB_Overbought_Short"] = (pctb > 1.0, "SHORT")
            signals["BB_LowerBand_Bounce_Long"] = (pctb < 0.1, "LONG")

    # ── 5. Stochastic (KDJ-J line) ──
    stoch = ta.stoch(high, low, close, k=9, d=3, smooth_k=3)
    if stoch is not None:
        k_col = [c for c in stoch.columns if "STOCHk" in c]
        d_col = [c for c in stoch.columns if "STOCHd" in c]
        if k_col and d_col:
            k_val = stoch[k_col[0]]
            d_val = stoch[d_col[0]]
            j_val = 3 * k_val - 2 * d_val
            signals["KDJ_J_Oversold_Long"] = (j_val < 0, "LONG")
            signals["KDJ_J_Overbought_Short"] = (j_val > 100, "SHORT")
            signals["Stoch_K_Oversold_Long"] = (k_val < 20, "LONG")
            signals["Stoch_K_Overbought_Short"] = (k_val > 80, "SHORT")

    # ── 6. CCI (14) ──
    cci = ta.cci(high, low, close, length=14)
    if cci is not None:
        signals["CCI14_Oversold_Long"] = (cci < -100, "LONG")
        signals["CCI14_Overbought_Short"] = (cci > 100, "SHORT")
        signals["CCI14_DeepOversold_Long"] = (cci < -200, "LONG")

    # ── 7. Williams %R ──
    willr = ta.willr(high, low, close, length=14)
    if willr is not None:
        signals["WillR_Oversold_Long"] = (willr < -80, "LONG")
        signals["WillR_Overbought_Short"] = (willr > -20, "SHORT")

    # ── 8. MFI (Money Flow Index) ──
    mfi = ta.mfi(high, low, close, volume, length=14)
    if mfi is not None:
        signals["MFI_Oversold_Long"] = (mfi < 20, "LONG")
        signals["MFI_Overbought_Short"] = (mfi > 80, "SHORT")

    # ── 9. Stochastic RSI ──
    stochrsi = ta.stochrsi(close, length=14, rsi_length=14, k=3, d=3)
    if stochrsi is not None:
        k_cols = [c for c in stochrsi.columns if "k" in c.lower()]
        if k_cols:
            srsi_k = stochrsi[k_cols[0]]
            signals["StochRSI_Oversold_Long"] = (srsi_k < 20, "LONG")
            signals["StochRSI_Overbought_Short"] = (srsi_k > 80, "SHORT")

    # ── 10. CMO (Chande Momentum Oscillator) ──
    cmo = ta.cmo(close, length=14)
    if cmo is not None:
        signals["CMO_Oversold_Long"] = (cmo < -50, "LONG")
        signals["CMO_Overbought_Short"] = (cmo > 50, "SHORT")

    # ── 11. PPO (Percentage Price Oscillator) ──
    ppo = ta.ppo(close, fast=12, slow=26, signal=9)
    if ppo is not None:
        ppo_cols = [c for c in ppo.columns if "PPO_" in c and "H" not in c and "S" not in c]
        if ppo_cols:
            ppo_line = ppo[ppo_cols[0]]
            signals["PPO_Bullish_Long"] = (ppo_line > 1.0, "LONG")
            signals["PPO_Bearish_Short"] = (ppo_line < -1.0, "SHORT")

    # ── 12. Fisher Transform ──
    fisher = ta.fisher(high, low, length=9)
    if fisher is not None:
        f_cols = [c for c in fisher.columns if "FISHERT" in c]
        if f_cols:
            ft = fisher[f_cols[0]]
            signals["Fisher_Oversold_Long"] = (ft < -1.5, "LONG")
            signals["Fisher_Overbought_Short"] = (ft > 1.5, "SHORT")

    # ── 13. ADX (trend strength) ──
    adx_result = ta.adx(high, low, close)
    if adx_result is not None:
        adx_col = [c for c in adx_result.columns if "ADX_" in c]
        dmp_col = [c for c in adx_result.columns if "DMP" in c]
        dmn_col = [c for c in adx_result.columns if "DMN" in c]
        if adx_col and dmp_col and dmn_col:
            adx_val = adx_result[adx_col[0]]
            dmp = adx_result[dmp_col[0]]
            dmn = adx_result[dmn_col[0]]
            signals["ADX_StrongTrend_Bull_Long"] = ((adx_val > 25) & (dmp > dmn), "LONG")
            signals["ADX_StrongTrend_Bear_Short"] = ((adx_val > 25) & (dmn > dmp), "SHORT")

    # ── 14. Breakout (Donchian Channel) ──
    high_20 = high.rolling(20).max().shift(1)
    low_20 = low.rolling(20).min().shift(1)
    signals["Breakout_High20_Long"] = (close > high_20, "LONG")
    signals["Breakout_Low20_Short"] = (close < low_20, "SHORT")

    # ── 15. HMA deviation ──
    hma = ta.hma(close, length=20)
    if hma is not None:
        hma_dev = (close - hma) / (hma + 1e-10)
        signals["HMA_Oversold_Long"] = (hma_dev < -0.02, "LONG")
        signals["HMA_Overbought_Short"] = (hma_dev > 0.02, "SHORT")

    # ── 16. OBV trend ──
    obv = ta.obv(close, volume)
    if obv is not None:
        obv_sma = obv.rolling(20).mean()
        obv_dev = (obv - obv_sma) / (obv_sma.abs() + 1e-10)
        signals["OBV_Bullish_Long"] = (obv_dev > 0.3, "LONG")
        signals["OBV_Bearish_Short"] = (obv_dev < -0.3, "SHORT")

    # ── 17. CMF (Chaikin Money Flow) ──
    cmf = ta.cmf(high, low, close, volume, length=20)
    if cmf is not None:
        signals["CMF_Bullish_Long"] = (cmf > 0.15, "LONG")
        signals["CMF_Bearish_Short"] = (cmf < -0.15, "SHORT")

    # ── 18. TRIX ──
    trix = ta.trix(close, length=18, signal=9)
    if trix is not None:
        trix_cols = [c for c in trix.columns if "TRIX_" in c and "s" not in c.lower()]
        if trix_cols:
            trix_val = trix[trix_cols[0]]
            signals["TRIX_Bullish_Long"] = (trix_val > 0, "LONG")
            signals["TRIX_Bearish_Short"] = (trix_val < 0, "SHORT")

    # ── 19. Supertrend ──
    st = ta.supertrend(high, low, close)
    if st is not None:
        dir_col = [c for c in st.columns if "d_" in c.lower() or "SUPERTd" in c]
        if dir_col:
            st_dir = st[dir_col[0]]
            signals["Supertrend_Bull_Long"] = (st_dir == 1, "LONG")
            signals["Supertrend_Bear_Short"] = (st_dir == -1, "SHORT")

    # ── 20. ATR Volatility Filter ──
    atr = ta.atr(high, low, close, length=14)
    if atr is not None:
        atr_pct = atr / (close + 1e-10)
        atr_median = atr_pct.rolling(100).median()
        signals["HighVol_Breakout_Long"] = ((atr_pct > atr_median * 1.5) & (close > close.shift(1)), "LONG")
        signals["HighVol_Breakdown_Short"] = ((atr_pct > atr_median * 1.5) & (close < close.shift(1)), "SHORT")

    return signals


# ──────────────────────────────────────────────
# Weight calculation engine
# ──────────────────────────────────────────────

def calculate_weights_for_symbol(
    df: pd.DataFrame,
    symbol: str,
    forward_bars: int | None = None,
    roundtrip_fee: float | None = None,
) -> dict[str, dict]:
    """
    Calculate accuracy weights for all indicators on a single symbol.
    
    For each indicator signal:
    1. Find all bars where signal fires
    2. Calculate forward return (close[T+N] - close[T]) / close[T]
    3. Subtract roundtrip Bybit fee (0.11%)
    4. Compute: win_rate, expected_return, profit_factor, max_consecutive_losses
    5. Assign accuracy_weight = max(0, expected_return)
    """
    if forward_bars is None:
        forward_bars = config.FORWARD_BARS
    if roundtrip_fee is None:
        roundtrip_fee = config.ROUNDTRIP_FEE

    if len(df) < 500:
        logger.warning(f"Insufficient data for {symbol}: {len(df)} bars (need 500+)")
        return {}

    # Compute forward returns (look-ahead is OK here — this is training/scoring)
    close = df["close"].astype(np.float64)
    fwd_return = (close.shift(-forward_bars) - close) / (close + 1e-10)

    # Get all indicator signals
    signals = _compute_indicator_signals(df)
    
    weight_matrix: dict[str, dict] = {}

    for ind_name, (condition, direction) in signals.items():
        # Ensure condition is valid
        if condition is None or condition.dtype != bool:
            continue

        # Get bars where signal fires
        mask = condition.fillna(False)
        signal_bars = mask[mask].index

        if len(signal_bars) < 15:
            continue  # need minimum 15 historical trades to be statistically meaningful

        # Get forward returns for signal bars
        signal_fwd = fwd_return.loc[signal_bars].dropna()
        if len(signal_fwd) < 15:
            continue

        # Apply direction: LONG profits from positive fwd_return, SHORT from negative
        if direction == "SHORT":
            signal_fwd = -signal_fwd  # invert for shorts

        # Subtract Bybit roundtrip fee (0.11%)
        returns_after_fees = signal_fwd - roundtrip_fee

        # Calculate metrics
        total_trades = len(returns_after_fees)
        wins = returns_after_fees[returns_after_fees > 0]
        losses = returns_after_fees[returns_after_fees <= 0]

        win_rate = len(wins) / total_trades
        expected_return = float(returns_after_fees.mean())
        avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
        avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0

        gross_profit = float(wins.sum()) if len(wins) > 0 else 0.0
        gross_loss = float(abs(losses.sum())) if len(losses) > 0 else 1e-10
        profit_factor = gross_profit / gross_loss

        # Max consecutive losses
        signs = (returns_after_fees > 0).astype(int).values
        max_consec_loss = 0
        current_streak = 0
        for s in signs:
            if s == 0:
                current_streak += 1
                max_consec_loss = max(max_consec_loss, current_streak)
            else:
                current_streak = 0

        # Accuracy weight = expected return (only if positive, else 0)
        accuracy_weight = max(0.0, expected_return * 100)  # in percentage points

        weight_matrix[ind_name] = {
            "direction": direction,
            "win_rate_pct": round(win_rate * 100, 2),
            "expected_return_per_trade_pct": round(expected_return * 100, 4),
            "total_historical_trades": total_trades,
            "accuracy_weight": round(accuracy_weight, 4),
            "avg_win_pct": round(avg_win * 100, 4),
            "avg_loss_pct": round(avg_loss * 100, 4),
            "profit_factor": round(profit_factor, 3),
            "max_consecutive_losses": max_consec_loss,
        }

    return weight_matrix


# ──────────────────────────────────────────────
# Batch processing
# ──────────────────────────────────────────────

def generate_weight_library(
    data: dict[str, pd.DataFrame],
    forward_bars: int | None = None,
    progress_cb: Callable[[str], None] | None = None,
) -> dict[str, dict]:
    """
    Generate the complete indicator weight library for all symbols.
    Returns compact dict: {symbol_clean: {indicator_name: weight_info}}.
    """
    library: dict[str, dict] = {}
    total = len(data)

    for idx, (sym, df) in enumerate(data.items(), 1):
        sym_clean = sym.replace("/", "")
        msg = f"[{idx}/{total}] Computing weights for {sym} …"
        logger.info(msg)
        if progress_cb:
            progress_cb(msg)

        weights = calculate_weights_for_symbol(df, sym, forward_bars)
        if weights:
            library[sym_clean] = weights
            # Count profitable indicators
            profitable = sum(1 for v in weights.values() if v["accuracy_weight"] > 0)
            msg_done = f"  ✓ {sym}: {len(weights)} signals tested, {profitable} profitable after fees"
            logger.info(msg_done)
            if progress_cb:
                progress_cb(msg_done)
        else:
            msg_fail = f"  ✗ {sym}: insufficient data or no valid signals"
            logger.warning(msg_fail)
            if progress_cb:
                progress_cb(msg_fail)

    return library


def save_weight_library(library: dict[str, dict], path: str | None = None) -> str:
    """Save the weight library to JSON."""
    if path is None:
        path = os.path.join(config.OUTPUT_DIR, "indicator_weights.json")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(library, f, indent=2, ensure_ascii=False)

    logger.info(f"Weight library saved: {path} ({os.path.getsize(path) / 1024:.1f} KB)")
    return path


def load_weight_library(path: str | None = None) -> dict[str, dict]:
    """Load existing weight library from JSON."""
    if path is None:
        path = os.path.join(config.OUTPUT_DIR, "indicator_weights.json")

    if not os.path.isfile(path):
        logger.warning(f"Weight library not found: {path}")
        return {}

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ──────────────────────────────────────────────
# Top-N indicator selector
# ──────────────────────────────────────────────

def select_top_indicators(
    weights: dict[str, dict],
    top_n: int | None = None,
    direction: str | None = None,
) -> list[dict]:
    """
    Select the top N indicators for a given symbol by accuracy_weight.
    
    Returns sorted list of dicts with indicator info.
    """
    if top_n is None:
        top_n = config.CONSENSUS_TOP_N_INDICATORS

    candidates = []
    for ind_name, info in weights.items():
        if info["accuracy_weight"] <= 0:
            continue
        if direction and info["direction"] != direction:
            continue
        candidates.append({"name": ind_name, **info})

    # Sort by accuracy_weight descending
    candidates.sort(key=lambda x: x["accuracy_weight"], reverse=True)
    return candidates[:top_n]


def get_symbol_summary(library: dict[str, dict]) -> pd.DataFrame:
    """Create a summary DataFrame of all symbols and their top indicators."""
    rows = []
    for sym, weights in library.items():
        profitable = [v for v in weights.values() if v["accuracy_weight"] > 0]
        if profitable:
            best = max(profitable, key=lambda x: x["accuracy_weight"])
            rows.append({
                "symbol": sym,
                "total_indicators": len(weights),
                "profitable_indicators": len(profitable),
                "best_indicator": [k for k, v in weights.items()
                                   if v["accuracy_weight"] == best["accuracy_weight"]][0],
                "best_weight": best["accuracy_weight"],
                "best_win_rate": best["win_rate_pct"],
                "best_pf": best["profit_factor"],
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df.sort_values("best_weight", ascending=False, inplace=True)
    return df
