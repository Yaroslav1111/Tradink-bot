"""
Aegis-Quant-Lab — Global Configuration
Central registry of all constants, asset lists, and simulation parameters.
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
# Bybit — supported linear perpetual tickers
# ──────────────────────────────────────────────
DEFAULT_SYMBOLS: list[str] = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "DOT/USDT", "LINK/USDT",
    "MATIC/USDT", "TON/USDT", "TRX/USDT", "SHIB/USDT", "UNI/USDT",
    "ATOM/USDT", "LTC/USDT", "BCH/USDT", "NEAR/USDT", "APT/USDT",
    "FIL/USDT", "ARB/USDT", "OP/USDT", "SUI/USDT", "HYPE/USDT",
    "IMX/USDT", "PEPE/USDT", "WIF/USDT", "FET/USDT", "RENDER/USDT",
    "INJ/USDT", "SEI/USDT", "STX/USDT", "AAVE/USDT", "MKR/USDT",
    "RUNE/USDT", "TIA/USDT", "ALGO/USDT", "FTM/USDT", "SAND/USDT",
    "MANA/USDT", "GALA/USDT", "EOS/USDT", "XLM/USDT", "IOTA/USDT",
    "DYDX/USDT", "CRV/USDT", "COMP/USDT", "JASMY/USDT", "1000BONK/USDT",
    "WLD/USDT", "JUP/USDT", "ENA/USDT", "PENDLE/USDT", "ORDI/USDT",
]

# ──────────────────────────────────────────────
# Timeframes
# ──────────────────────────────────────────────
SUPPORTED_TIMEFRAMES: list[str] = ["15m", "1h", "3h"]
DEFAULT_TIMEFRAME: str = "1h"

# ──────────────────────────────────────────────
# Bybit API rate-limiting
# ──────────────────────────────────────────────
BYBIT_RATE_LIMIT_SLEEP: float = 0.35        # seconds between requests
BYBIT_BATCH_SIZE: int = 1000                 # candles per request (max 1000)
BYBIT_MAX_RETRIES: int = 5

# ──────────────────────────────────────────────
# Simulation — Realistic costs (pessimistic)
# ──────────────────────────────────────────────
FEE_RATE: float = 0.0004          # 0.04 % taker fee (per side)
SLIPPAGE: float = 0.0002          # 0.02 % price slippage (per side)
TOTAL_COST_PER_SIDE: float = FEE_RATE + SLIPPAGE   # 0.06 % per entry/exit

# ──────────────────────────────────────────────
# Walk-Forward Analysis
# ──────────────────────────────────────────────
WFA_IN_SAMPLE_RATIO: float = 0.70   # 70 % for training
WFA_OUT_SAMPLE_RATIO: float = 0.30  # 30 % for validation
WFA_MIN_TRADES: int = 30            # minimum trades to consider result valid

# ──────────────────────────────────────────────
# Statistics
# ──────────────────────────────────────────────
STAT_P_VALUE_THRESHOLD: float = 0.01
BOOTSTRAP_ITERATIONS: int = 1000

# ──────────────────────────────────────────────
# Cross-asset
# ──────────────────────────────────────────────
CORRELATION_THRESHOLD: float = 0.75
ROLLING_CORR_WINDOW: int = 60       # bars

# ──────────────────────────────────────────────
# Feature Factory
# ──────────────────────────────────────────────
TA_CORES: int = 4
DOWNCAST_DTYPE: str = "float32"
