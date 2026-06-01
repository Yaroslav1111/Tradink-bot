"""
Aegis-Quant-Lab v4.2 — Global Configuration
═══════════════════════════════════════════════
Architecture: Fibonacci Reversal Sniper + Volume POC + Shadow Trailing

v4.2 Upgrades:
  - Volume Profile POC (Lazy Sniper: 1m precision after 15m signal)
  - 3-Phase Trailing (ATR buffer → Breakeven → Shadow-based trailing)
  - Maker exits (dynamic limit TP repositioning)
  - SHORT fully supported (no directional bias)
"""

import os

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "history")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LOG_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# Bybit Demo API
# ──────────────────────────────────────────────
BYBIT_DEMO_ENDPOINT: str = "https://api-demo.bybit.com"

# ──────────────────────────────────────────────
# Monitored symbols (linear perpetual, Bybit format)
# ──────────────────────────────────────────────

SYMBOLS: list[str] = [
    "BTCUSDT", "OPUSDT", "COMPUSDT", "DOGEUSDT", "GALAUSDT",
    "FILUSDT", "STXUSDT", "BRUSDT", "MANAUSDT", "PORTALUSDT", "BSBUSDT", "NEARUSDT", "HYPEUSDT"
]

# ──────────────────────────────────────────────
# Timeframe
# ──────────────────────────────────────────────
CANDLE_INTERVAL: str = "15"      # minutes (Bybit kline interval)
CANDLE_LIMIT: int = 200          # bars to fetch per symbol

# ──────────────────────────────────────────────
# Component 1: Reversal Engine (oscillator extremes)
# ──────────────────────────────────────────────
# Oscillator thresholds for REVERSAL detection
RSI_OVERSOLD: float = 15.0       # RSI14 < 15 → extreme oversold
RSI_OVERBOUGHT: float = 85.0     # RSI14 > 85 → extreme overbought
CCI_OVERSOLD: float = -250.0     # CCI14 < -250 → extreme oversold
CCI_OVERBOUGHT: float = 250.0    # CCI14 > 250 → extreme overbought
WILLR_OVERSOLD: float = -95.0    # Williams%R < -95 → extreme oversold
WILLR_OVERBOUGHT: float = -5.0   # Williams%R > -5 → extreme overbought

# Minimum oscillators in resonance for signal (out of 3)
MIN_OSCILLATORS_FIRING: int = 2  # need 2/3 oscillators at extremes

# Swing detection
SWING_LOOKBACK: int = 50         # bars to scan for swing high/low

# ──────────────────────────────────────────────
# Component 2: Fibonacci Entry (Maker Sniper)
# ──────────────────────────────────────────────
FIBO_LEVEL_PRIMARY: float = 0.618     # golden ratio (main entry)
FIBO_LEVEL_SECONDARY: float = 0.50    # 50% retracement (aggressive)
FIBO_IMPULSE_BARS: int = 50           # bars lookback for impulse swing

# Limit order settings
ORDER_TYPE: str = "Limit"             # Limit = Maker fee
TIME_IN_FORCE: str = "PostOnly"       # guaranteed maker (rejected if would be taker)
ORDER_TTL_SECONDS: int = 600          # 10 minutes TTL (full candle + buffer)
PRICE_DEVIATION_CANCEL_PCT: float = 0.01  # 1% — cancel if price moves away

# Cascade order laddering
CASCADE_ENABLED: bool = True          # True = dual orders at 0.50 and 0.618
CASCADE_RISK_SPLIT: float = 0.5       # each leg gets 50% of total risk

# Fees
MAKER_FEE_RATE: float = 0.0002       # 0.020% per side (Bybit VIP0 Maker)
TAKER_FEE_RATE: float = 0.00055      # 0.055% per side (Bybit VIP0 Taker)
ROUNDTRIP_FEE_MAKER: float = 0.0004  # 0.04% total roundtrip (maker both sides)

# ──────────────────────────────────────────────
# Component 2b: Volume Profile POC (Lazy Sniper)
# ──────────────────────────────────────────────
# After 15m signal fires, fetch 1m data for precision entry
POC_ENABLED: bool = True              # enable Volume Profile targeting
POC_1M_LOOKBACK: int = 60            # 1m bars to fetch (= 1 hour of micro-structure)
POC_CLUSTER_WIDTH_ATR: float = 0.3   # VWAP cluster width = 0.3 × ATR14
POC_WEIGHT_VS_FIBO: float = 0.6     # 60% POC + 40% Fibo blend (0=pure Fibo, 1=pure POC)

# ──────────────────────────────────────────────
# Component 3: Single-Entry Lock (Anti-Pyramid)
# ──────────────────────────────────────────────
MAX_CONCURRENT_POSITIONS: int = 12    # max simultaneous open positions
MAX_ENTRIES_PER_CYCLE: int = 3       # don't enter more than 3 per scan
# Rule: 1 symbol = 1 position. No averaging, no grid, no pyramiding.

# ──────────────────────────────────────────────
# Component 4: 3-Phase Position Management (v4.2)
# ──────────────────────────────────────────────

# Phase 1: "Breathing Room" (initial buffer — don't touch SL)
# SL set at entry - 2×ATR. No movement until breakeven triggers.
SL_ATR_MULTIPLIER: float = 2.0       # initial SL distance = 2 × ATR

# Phase 2: Breakeven
BREAKEVEN_TRIGGER_PCT: float = 0.015  # +1.5% → move SL to entry + fees
BREAKEVEN_FEE_BUFFER: float = 0.0004  # add maker roundtrip fee to breakeven SL

# Phase 3: Shadow-based Trailing (candle-low tracking)
TRAILING_TRIGGER_PCT: float = 0.030   # +3.0% → activate trailing
TRAILING_DISTANCE_PCT: float = 0.015  # fallback: trail 1.5% behind peak
TRAILING_ATR_CUSHION: float = 0.2     # cushion below prev candle low = 0.2 × ATR
# Rule: new_sl = max(old_sl, prev_candle_low - cushion) for LONG
#        new_sl = min(old_sl, prev_candle_high + cushion) for SHORT

# Fibonacci extension targets
FIBO_EXT_1: float = 1.618            # first extension target (primary TP)
FIBO_EXT_2: float = 2.618            # second extension target (extended TP)

# Exit on reversal: if oscillators flip to opposite extreme → close position
EXIT_ON_REVERSAL: bool = True

# ──────────────────────────────────────────────
# Component 4b: Maker Exit (Limit Take-Profit)
# ──────────────────────────────────────────────
MAKER_EXIT_ENABLED: bool = True       # place limit TP instead of market close
MAKER_TP_REPOSITION_TRIGGER: float = 0.8  # reposition TP when price reaches 80% of target
# When trailing activates past ext_1.618, TP jumps to ext_2.618

# ──────────────────────────────────────────────
# Risk Management
# ──────────────────────────────────────────────
INITIAL_CAPITAL: float = 2000.0       # starting balance USDT
LEVERAGE: float = 5.0                 # trading leverage
RISK_PER_TRADE_PCT: float = 0.01      # 1% of balance per trade
RISK_REWARD_RATIO: float = 2.0        # initial TP = 2x SL distance
MAX_DAILY_LOSS_PCT: float = 0.05      # 5% daily loss → stop trading

# Compound interest
COMPOUND_ENABLED: bool = True
COMPOUND_LOSS_STREAK_THRESHOLD: int = 3   # reduce risk after N consecutive losses
COMPOUND_RISK_REDUCED: float = 0.005      # 0.5% risk after loss streak
COMPOUND_RISK_NORMAL: float = 0.01        # 1% risk normally
COMPOUND_MAX_RISK: float = 0.02           # hard cap: never risk more than 2%

# ──────────────────────────────────────────────
# Bybit API rate-limiting
# ──────────────────────────────────────────────
BYBIT_RATE_LIMIT_SLEEP: float = 0.35
BYBIT_MAX_RETRIES: int = 3
PARALLEL_WORKERS: int = 10            # ThreadPoolExecutor workers for candle fetch
