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
# Simulation — Realistic Bybit costs (VIP 0)
# Official: Maker 0.020%, Taker 0.055%
# Worst case: Taker on entry + Taker on exit
# ──────────────────────────────────────────────
FEE_RATE: float = 0.00055         # 0.055 % taker fee (per side) — Bybit VIP 0
SLIPPAGE: float = 0.0000          # slippage embedded into taker fee model
TOTAL_COST_PER_SIDE: float = FEE_RATE + SLIPPAGE   # 0.055 % per side
ROUNDTRIP_FEE: float = 0.0011     # 0.11% total (entry taker + exit taker)

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

# ──────────────────────────────────────────────
# Weighted Consensus Voting System
# ──────────────────────────────────────────────
CONSENSUS_TOP_N_INDICATORS: int = 15        # number of indicators that vote
CONSENSUS_MIN_AGREEMENT: float = 0.70       # 70% = 10.5/15 must agree for signal
CONSENSUS_DISSONANCE_THRESHOLD: float = 0.55  # if split is 55/45 or closer → chaos → no trade
FORWARD_BARS: int = 4                       # evaluate trade outcome after N bars (4 bars = 1h on 15m)
MIN_WEIGHT_THRESHOLD: float = 0.0           # indicators with EV <= 0 get weight = 0

# ──────────────────────────────────────────────
# Risk Management (for $2000 USDT deposit)
# ──────────────────────────────────────────────
DEFAULT_CAPITAL: float = 2000.0             # starting capital USDT
RISK_PER_TRADE_PCT: float = 0.01            # 1% risk per trade
RISK_REWARD_RATIO: float = 2.0              # 1:2 risk/reward
MAX_CONCURRENT_TRADES: int = 5              # max simultaneous positions
MAX_DAILY_LOSS_PCT: float = 0.05            # 5% daily loss limit

# ──────────────────────────────────────────────
# v3.0 — Adaptive Scoring Engine Parameters
# ──────────────────────────────────────────────
SCORING_ENTRY_THRESHOLD: float = 0.65       # min weighted score for entry (0.5=aggro, 0.8=conserv)
SCORING_CHAOS_THRESHOLD: float = 0.45       # max opposing score ratio before chaos block
SCORING_TOP_N: int = 15                     # number of top indicators used in scoring

# ──────────────────────────────────────────────
# v3.0 — Breakeven & Trailing Stop
# ──────────────────────────────────────────────
BREAKEVEN_TRIGGER_PCT: float = 0.015        # +1.5% unrealized → move SL to breakeven
TRAILING_TRIGGER_PCT: float = 0.030         # +3.0% unrealized → activate trailing stop
TRAILING_DISTANCE_PCT: float = 0.015        # trail 1.5% behind highest/lowest

# ──────────────────────────────────────────────
# v3.0 — Compound Interest
# ──────────────────────────────────────────────
COMPOUND_ENABLED: bool = True               # enable dynamic position sizing
COMPOUND_RISK_NORMAL: float = 0.01          # 1% risk normally
COMPOUND_RISK_REDUCED: float = 0.005        # 0.5% risk after loss streak
COMPOUND_LOSS_STREAK_THRESHOLD: int = 3     # reduce risk after N consecutive losses
COMPOUND_MAX_RISK: float = 0.02             # hard cap: never risk more than 2%

# ──────────────────────────────────────────────
# v3.0 — Cluster Guard
# ──────────────────────────────────────────────
CLUSTER_CORRELATION_BLOCK: float = 0.75     # block if corr > this
CLUSTER_MAX_PER_GROUP: int = 2              # max positions in same cluster

# ──────────────────────────────────────────────
# v3.0 — Bybit Demo API
# ──────────────────────────────────────────────
BYBIT_DEMO_ENDPOINT: str = "https://api-demo.bybit.com"
BYBIT_DEMO_API_KEY: str = ""                # set via environment variable
BYBIT_DEMO_API_SECRET: str = ""             # set via environment variable
