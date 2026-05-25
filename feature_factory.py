"""
Aegis-Quant-Lab — Feature Factory
===================================
Pure indicator-generation layer.
Input  : raw OHLCV DataFrame (single symbol / single timeframe).
Output : enriched DataFrame with 200+ technical features.

Uses pandas-ta AllStudy (v0.4.71b+) with multi-core acceleration,
forced float32 downcasting, NaN / inf sanitisation.
"""

from __future__ import annotations

import logging
import warnings
from typing import Callable

import numpy as np
import pandas as pd
import pandas_ta as ta

import config

logger = logging.getLogger("aegis.feature_factory")
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ──────────────────────────────────────────────
# pandas-ta global settings
# ──────────────────────────────────────────────
ta.cores = config.TA_CORES


# ──────────────────────────────────────────────
# Core helpers
# ──────────────────────────────────────────────

def _downcast(df: pd.DataFrame) -> pd.DataFrame:
    """Force all float64 columns → float32 to save ~50 % RAM on i5."""
    float_cols = df.select_dtypes(include=["float64"]).columns
    if len(float_cols):
        df[float_cols] = df[float_cols].astype(np.float32)
    return df


def _sanitize(df: pd.DataFrame) -> pd.DataFrame:
    """Replace inf / -inf with NaN, then forward-fill, then drop residual NaN rows at head."""
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.ffill(inplace=True)
    # Drop leading rows that are entirely NaN (warm-up period of indicators)
    first_valid = df.dropna(how="all").index.min()
    if first_valid is not None:
        df = df.loc[first_valid:]
    return df


# ──────────────────────────────────────────────
# Custom indicators (beyond pandas-ta AllStrategy)
# ──────────────────────────────────────────────

def _add_custom_features(df: pd.DataFrame) -> pd.DataFrame:
    """Hand-crafted features that pandas-ta AllStrategy may not cover or that
    we want with specific parameters."""

    c = df["close"]
    h = df["high"]
    l = df["low"]  # noqa: E741
    v = df["volume"]

    # ── Returns & log-returns ──
    df["returns_1"] = c.pct_change(1)
    df["returns_3"] = c.pct_change(3)
    df["returns_5"] = c.pct_change(5)
    df["log_return"] = np.log(c / c.shift(1))

    # ── Volatility ratios ──
    df["atr_pct"] = ta.atr(h, l, c, length=14) / c  # ATR as % of price
    df["atr_pct_5"] = ta.atr(h, l, c, length=5) / c
    df["bb_width"] = np.nan  # placeholder — filled from BB if available

    # Bollinger %B and Width (multi-period)
    for period in [14, 20, 30]:
        bb = ta.bbands(c, length=period, std=2.0)
        if bb is not None and len(bb.columns) >= 3:
            upper_col = [col for col in bb.columns if "BBU" in col]
            lower_col = [col for col in bb.columns if "BBL" in col]
            mid_col = [col for col in bb.columns if "BBM" in col]
            if upper_col and lower_col:
                bbu = bb[upper_col[0]]
                bbl = bb[lower_col[0]]
                bbm = bb[mid_col[0]] if mid_col else (bbu + bbl) / 2
                df[f"bb_width_{period}"] = (bbu - bbl) / bbm
                df[f"bb_pctb_{period}"] = (c - bbl) / (bbu - bbl + 1e-10)

    # ── Momentum extras ──
    # KDJ
    stoch = ta.stoch(h, l, c, k=9, d=3, smooth_k=3)
    if stoch is not None:
        k_col = [col for col in stoch.columns if "STOCHk" in col]
        d_col = [col for col in stoch.columns if "STOCHd" in col]
        if k_col and d_col:
            df["kdj_k"] = stoch[k_col[0]]
            df["kdj_d"] = stoch[d_col[0]]
            df["kdj_j"] = 3 * stoch[k_col[0]] - 2 * stoch[d_col[0]]

    # Stochastic RSI
    stochrsi = ta.stochrsi(c, length=14, rsi_length=14, k=3, d=3)
    if stochrsi is not None:
        for col in stochrsi.columns:
            df[f"stochrsi_{col}"] = stochrsi[col]

    # Fisher Transform
    fisher = ta.fisher(h, l, length=9)
    if fisher is not None:
        for col in fisher.columns:
            df[f"fisher_{col}"] = fisher[col]

    # Chande Momentum Oscillator
    cmo = ta.cmo(c, length=14)
    if cmo is not None:
        df["cmo_14"] = cmo

    # PPO (Percentage Price Oscillator) — scale-invariant across coins
    ppo = ta.ppo(c, fast=12, slow=26, signal=9)
    if ppo is not None:
        for col in ppo.columns:
            df[f"ppo_{col}"] = ppo[col]

    # TRIX
    trix = ta.trix(c, length=18, signal=9)
    if trix is not None:
        for col in trix.columns:
            df[f"trix_{col}"] = trix[col]

    # Hull Moving Average deviation
    hma = ta.hma(c, length=20)
    if hma is not None:
        df["hma_20"] = hma
        df["hma_dev"] = (c - hma) / hma  # deviation from HMA in %

    # ── Volume-based ──
    # Money Flow Index
    mfi = ta.mfi(h, l, c, v, length=14)
    if mfi is not None:
        df["mfi_14"] = mfi

    # OBV
    obv = ta.obv(c, v)
    if obv is not None:
        df["obv"] = obv
        df["obv_sma20"] = obv.rolling(20).mean()
        df["obv_dev"] = (obv - df["obv_sma20"]) / (df["obv_sma20"].abs() + 1e-10)

    # Chaikin Money Flow
    cmf = ta.cmf(h, l, c, v, length=20)
    if cmf is not None:
        df["cmf_20"] = cmf

    # Accumulation / Distribution
    ad = ta.ad(h, l, c, v)
    if ad is not None:
        df["ad_line"] = ad

    # ── Session / calendar features (for risk_blocks) ──
    if hasattr(df.index, "hour"):
        df["hour"] = df.index.hour
        df["day_of_week"] = df.index.dayofweek
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(np.int8)
        # Session buckets
        df["session_asia"] = ((df["hour"] >= 0) & (df["hour"] < 8)).astype(np.int8)
        df["session_europe"] = ((df["hour"] >= 8) & (df["hour"] < 16)).astype(np.int8)
        df["session_us"] = ((df["hour"] >= 16) & (df["hour"] < 24)).astype(np.int8)

    return df


# ──────────────────────────────────────────────
# Manual bulk indicator runner (fallback)
# ──────────────────────────────────────────────

def _run_all_indicators_manual(df: pd.DataFrame) -> None:
    """
    Manually invoke every major pandas-ta indicator when the bulk
    Study/Strategy API is unavailable. Modifies df in-place.
    """
    c = df["close"]
    h = df["high"]
    l = df["low"]  # noqa: E741
    o = df["open"]
    v = df["volume"]

    # ── Momentum ──
    _safe_add(df, "RSI_7", lambda: ta.rsi(c, length=7))
    _safe_add(df, "RSI_14", lambda: ta.rsi(c, length=14))
    _safe_add(df, "RSI_21", lambda: ta.rsi(c, length=21))
    _safe_multi(df, "MACD", lambda: ta.macd(c, fast=12, slow=26, signal=9))
    _safe_multi(df, "STOCH", lambda: ta.stoch(h, l, c))
    _safe_multi(df, "STOCHF", lambda: ta.stochf(h, l, c))
    _safe_add(df, "MOM_10", lambda: ta.mom(c, length=10))
    _safe_add(df, "MOM_20", lambda: ta.mom(c, length=20))
    _safe_add(df, "ROC_10", lambda: ta.roc(c, length=10))
    _safe_add(df, "ROC_20", lambda: ta.roc(c, length=20))
    _safe_add(df, "WILLR_14", lambda: ta.willr(h, l, c, length=14))
    _safe_add(df, "WILLR_21", lambda: ta.willr(h, l, c, length=21))
    _safe_add(df, "CCI_14", lambda: ta.cci(h, l, c, length=14))
    _safe_add(df, "CCI_20", lambda: ta.cci(h, l, c, length=20))
    _safe_add(df, "AO", lambda: ta.ao(h, l))
    _safe_add(df, "APO", lambda: ta.apo(c))
    _safe_add(df, "BOP", lambda: ta.bop(o, h, l, c))
    _safe_add(df, "UO", lambda: ta.uo(h, l, c))
    _safe_add(df, "TSI", lambda: ta.tsi(c))
    _safe_multi(df, "KDJ", lambda: ta.kdj(h, l, c))
    _safe_multi(df, "QQE", lambda: ta.qqe(c))

    # ── Trend ──
    _safe_add(df, "EMA_9", lambda: ta.ema(c, length=9))
    _safe_add(df, "EMA_21", lambda: ta.ema(c, length=21))
    _safe_add(df, "EMA_50", lambda: ta.ema(c, length=50))
    _safe_add(df, "SMA_20", lambda: ta.sma(c, length=20))
    _safe_add(df, "SMA_50", lambda: ta.sma(c, length=50))
    _safe_add(df, "SMA_100", lambda: ta.sma(c, length=100))
    _safe_add(df, "HMA_14", lambda: ta.hma(c, length=14))
    _safe_add(df, "DEMA_20", lambda: ta.dema(c, length=20))
    _safe_add(df, "TEMA_20", lambda: ta.tema(c, length=20))
    _safe_add(df, "WMA_20", lambda: ta.wma(c, length=20))
    _safe_add(df, "KAMA_10", lambda: ta.kama(c, length=10))
    _safe_add(df, "T3_10", lambda: ta.t3(c, length=10))
    _safe_multi(df, "ADX", lambda: ta.adx(h, l, c))
    _safe_multi(df, "AROON", lambda: ta.aroon(h, l))
    _safe_multi(df, "SUPERTREND", lambda: ta.supertrend(h, l, c))
    _safe_multi(df, "PSAR", lambda: ta.psar(h, l, c))
    _safe_multi(df, "ICHIMOKU", lambda: ta.ichimoku(h, l, c))
    _safe_add(df, "SLOPE_14", lambda: ta.slope(c, length=14))
    _safe_add(df, "DPO_14", lambda: ta.dpo(c, length=14))
    _safe_add(df, "CHOP_14", lambda: ta.chop(h, l, c, length=14))
    _safe_add(df, "VHF_14", lambda: ta.vhf(c, length=14))

    # ── Volatility ──
    _safe_multi(df, "BBANDS", lambda: ta.bbands(c, length=20))
    _safe_add(df, "ATR_14", lambda: ta.atr(h, l, c, length=14))
    _safe_add(df, "NATR_14", lambda: ta.natr(h, l, c, length=14))
    _safe_add(df, "ATR_7", lambda: ta.atr(h, l, c, length=7))
    _safe_multi(df, "KC", lambda: ta.kc(h, l, c))
    _safe_multi(df, "DONCHIAN", lambda: ta.donchian(h, l))
    _safe_add(df, "MASSI", lambda: ta.massi(h, l))
    _safe_add(df, "UI_14", lambda: ta.ui(c, length=14))

    # ── Volume ──
    _safe_add(df, "MFI_14", lambda: ta.mfi(h, l, c, v, length=14))
    _safe_add(df, "OBV", lambda: ta.obv(c, v))
    _safe_add(df, "CMF_20", lambda: ta.cmf(h, l, c, v, length=20))
    _safe_add(df, "AD", lambda: ta.ad(h, l, c, v))
    _safe_add(df, "ADOSC", lambda: ta.adosc(h, l, c, v))
    _safe_add(df, "EFI_13", lambda: ta.efi(c, v, length=13))
    _safe_add(df, "NVI", lambda: ta.nvi(c, v))
    _safe_add(df, "PVI", lambda: ta.pvi(c, v))
    _safe_multi(df, "KVO", lambda: ta.kvo(h, l, c, v))

    # ── Statistics ──
    _safe_add(df, "ZSCORE_20", lambda: ta.zscore(c, length=20))
    _safe_add(df, "ENTROPY_10", lambda: ta.entropy(c, length=10))
    _safe_add(df, "KURTOSIS_20", lambda: ta.kurtosis(c, length=20))
    _safe_add(df, "SKEW_20", lambda: ta.skew(c, length=20))
    _safe_add(df, "VARIANCE_20", lambda: ta.variance(c, length=20))
    _safe_add(df, "STDEV_20", lambda: ta.stdev(c, length=20))
    _safe_add(df, "MAD_20", lambda: ta.mad(c, length=20))
    _safe_add(df, "LINREG_14", lambda: ta.linreg(c, length=14))
    _safe_add(df, "MIDPOINT_14", lambda: ta.midpoint(c, length=14))
    _safe_add(df, "MIDPRICE_14", lambda: ta.midprice(h, l, length=14))


def _safe_add(df: pd.DataFrame, name: str, func) -> None:
    """Safely compute a single-column indicator and add to df."""
    try:
        result = func()
        if result is not None:
            if isinstance(result, pd.Series):
                df[name] = result
            elif isinstance(result, pd.DataFrame):
                for col in result.columns:
                    df[col] = result[col]
    except Exception:
        pass  # silently skip failed indicators


def _safe_multi(df: pd.DataFrame, prefix: str, func) -> None:
    """Safely compute a multi-column indicator and add all columns to df."""
    try:
        result = func()
        if result is not None:
            if isinstance(result, pd.DataFrame):
                for col in result.columns:
                    df[col] = result[col]
            elif isinstance(result, tuple):
                # Some indicators return (df, df) like ichimoku
                for part in result:
                    if isinstance(part, pd.DataFrame):
                        for col in part.columns:
                            df[col] = part[col]
                    elif isinstance(part, pd.Series):
                        df[f"{prefix}_{part.name}"] = part
            elif isinstance(result, pd.Series):
                df[prefix] = result
    except Exception:
        pass


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def compute_features(
    df: pd.DataFrame,
    symbol: str = "",
    progress_cb: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """
    Master feature generator.

    1. Run pandas-ta AllStrategy (200+ indicators automatically)
    2. Add custom hand-crafted features
    3. Downcast + sanitise

    Parameters
    ----------
    df : raw OHLCV DataFrame with DatetimeIndex
    symbol : optional, for logging
    progress_cb : optional callback for GUI log

    Returns
    -------
    Enriched DataFrame (original columns + all indicator columns).
    """
    if df is None or len(df) < 50:
        logger.warning(f"Insufficient data for {symbol}: {len(df) if df is not None else 0} rows")
        return pd.DataFrame()

    msg = f"  Computing features for {symbol} ({len(df)} bars) …"
    logger.info(msg)
    if progress_cb:
        progress_cb(msg)

    # Work on a copy — upcast to float64 for pandas-ta compatibility
    enriched = df.copy()
    for col in ["open", "high", "low", "close", "volume"]:
        if col in enriched.columns:
            enriched[col] = enriched[col].astype(np.float64)

    # ── 1. pandas-ta bulk indicators ──
    # v0.4.71b uses Study instead of Strategy; detect API automatically
    try:
        if hasattr(ta, 'AllStudy'):
            enriched.ta.study(ta.AllStudy, verbose=False)
        elif hasattr(ta, 'AllStrategy'):
            enriched.ta.strategy(ta.AllStrategy, verbose=False)
        else:
            _run_all_indicators_manual(enriched)
    except Exception as e:
        logger.warning(f"Bulk indicator failure for {symbol}: {e}")
        try:
            _run_all_indicators_manual(enriched)
        except Exception as e2:
            logger.error(f"Manual indicator fallback also failed for {symbol}: {e2}")

    # ── 2. Custom features ──
    enriched = _add_custom_features(enriched)

    # ── 3. Downcast + clean ──
    enriched = _downcast(enriched)
    enriched = _sanitize(enriched)

    msg_done = f"  ✓ {symbol}: {len(enriched)} bars × {len(enriched.columns)} features"
    logger.info(msg_done)
    if progress_cb:
        progress_cb(msg_done)

    return enriched


def compute_features_batch(
    data: dict[str, pd.DataFrame],
    progress_cb: Callable[[str], None] | None = None,
) -> dict[str, pd.DataFrame]:
    """Run feature computation for a dict of {symbol: ohlcv_df}."""
    results: dict[str, pd.DataFrame] = {}
    total = len(data)
    for idx, (sym, df) in enumerate(data.items(), 1):
        msg = f"[{idx}/{total}] Feature factory: {sym}"
        logger.info(msg)
        if progress_cb:
            progress_cb(msg)
        enriched = compute_features(df, symbol=sym, progress_cb=progress_cb)
        if len(enriched) > 0:
            results[sym] = enriched
    return results


def list_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return only the indicator/feature columns (exclude OHLCV + meta)."""
    base_cols = {"open", "high", "low", "close", "volume"}
    return [c for c in df.columns if c not in base_cols]
