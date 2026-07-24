# Aegis-Quant-Lab

A modular crypto trading bot (Bybit) built on a **clean architecture** — pure strategy
logic, a broker abstraction (Protocol / dependency injection), and an orchestration
runner. This document covers the **v5.5 "Two-Winged" Dynamic Exit Engine** and the
LLM-ready **Trade Capsule** auditing subsystem.

---

## Architecture Overview

```
config.py                 Global config (entry thresholds — DO NOT change)
engine/
  models.py               Dataclasses & enums (Position, LotRole, LotState, TradeCapsule …)
  strategy.py             Pure decision logic (entry + v5.5 dual-lot exits)
  trade_capsule.py        v5.5 non-blocking Trade Capsule logger (→ data/ai_analysis/)
  indicators.py           RSI / CCI / Williams%R / ATR / SuperTrend / Fibo / POC
broker/
  base.py                 Broker Protocol interface (DI seam)
  sim_broker.py           Backtest broker (honest candle-crossing fills)
  live_broker.py          Live Bybit broker (pybit)
  exchange_rules.py       floor_qty / round_to_tick helpers
runner/
  backtest_runner.py      Orchestration + dual-lot state machine + capsule assembly
run_backtest.py           Standard backtest entry (needs history CSV)
run_lab_backtest.py       Offline 14-day synthetic-volatility lab backtest
run_live.py               Live trading entry
run_optimizer.py          Parameter optimizer
test_v5.py                Full test suite (131 tests)
```

**Separation of concerns:** the strategy never touches the network; the runner decides
*when* to act and delegates *how* to the broker via the `Broker` protocol. The same
strategy code drives both `SimBroker` (backtest) and `LiveBroker` (Bybit).

---

## v5.5 — The "Two-Winged" Dynamic Exit Engine

> **Design invariant:** the entry logic (oscillator thresholds, Fibo cascade, volume POC,
> trend filter) is **completely untouched** by v5.5. A dedicated regression test,
> `test_entry_thresholds_unchanged`, guards this.

On entry the position is split **50/50** into two lots with opposite exit philosophies.
There is **zero full static take-profit**.

### Lot A — Maker Fix (guaranteed fractional profit)
Immediately places a **Limit Maker** order at a tight, mathematically derived distance
that covers the exchange round-trip fees plus a guaranteed net profit, targeting
**+1.5% → +2.0% NET**.

```
gross_move = maker_fix_net_target_low (1.5%) + round-trip fee
```

### Lot B — Momentum Float (no static target)
No fixed price target. It rides the recovery impulse and closes **at market** the moment
localized recovery momentum dies:

- **Momentum decay:** RSI must first *recover through* the neutral 50 line (validating a
  real local reversal on the short timeframe), then *cross back*. Only then does the
  float exit fire (`MOMENTUM_DECAY`). This "require recovery first" rule prevents an
  already-oversold entry from instantly triggering.
- Lot B's protective stop is a **structural stop with extra ATR breathing room**
  (`momentum_float_extra_atr = 1.25`) so normal noise doesn't stop it out prematurely.

**Exit priority inside `update_momentum_float` (order matters):**
1. Protective stop (structural).
2. `MOMENTUM_DECAY` — once at/above `momentum_min_profit_pct`.
3. `TREND_INVALIDATION` — only when already in profit ≥ `trailing_trigger_pct`.

### Breakeven Protection Cascade
The instant **Lot A's limit fills**, Lot B's stop **snaps to breakeven + a minor buffer**
(`breakeven_snap_buffer = 0.06%`, covering fees) — risk on the remaining size is
eliminated.

### Structural Invalidation Stop
Arbitrary ATR stops are replaced by a tight **structural stop** placed immediately behind
the extreme pivot / Fibo level that triggered the trade
(`structural_stop_enabled = True`, `structural_stop_buffer_atr = 0.25`).

### Exchange-State Safety / Race-Condition Prevention
- **Race pipeline:** if Lot B's momentum exit triggers *before* Lot A's limit fills →
  `Cancel Order (Lot A) → Verify Status → Market Close (remaining size)`.
- **Dynamic size sync:** the actual filled size is looked up (`get_filled_size`) before
  firing any breakeven / close payload — prevents Bybit **ErrCode 10001** size-mismatch
  rejections during partial fills.

### Key exit config (`engine/strategy.py → StrategyConfig`)
| Param | Default | Meaning |
|---|---|---|
| `dual_lot_enabled` | `True` | Master switch for two-winged exits |
| `dual_lot_split` | `0.5` | 50/50 Lot A / Lot B |
| `maker_fix_net_target_low` | `0.015` | +1.5% net floor for Lot A |
| `maker_fix_net_target_high` | `0.020` | +2.0% net upper band |
| `momentum_rsi_neutral` | `50.0` | Neutral line for decay detection |
| `momentum_min_profit_pct` | `0.0` | Min PnL before momentum exit allowed |
| `breakeven_snap_buffer` | `0.0006` | BE + 0.06% after Lot A fills |
| `structural_stop_enabled` | `True` | Use pivot-based stop, not ATR |
| `momentum_float_extra_atr` | `1.25` | Extra ATR breathing room for Lot B |

---

## v5.5 — "Trade Capsule" Meta-Analysis Subsystem

A **lightweight, non-blocking** logger (`engine/trade_capsule.py`) writes a
token-optimized JSON "Trade Capsule" to `data/ai_analysis/` for **every completed trade
cycle**, ready for later Gemini 1.5 Pro evaluation.

- **Non-blocking:** a background daemon thread + queue; all I/O is best-effort and wrapped
  in exception-swallowing try/except, so logging can never stall or crash the trade loop.
- **Token-optimized:** null/empty pruning, float rounding, OHLCV row capping.

### The four pillars
1. **Pre-Trade Context** — OHLCV snapshot immediately prior to entry.
2. **Internal Thoughts** — exact oscillator / structural parameters at the instant of the
   entry decision (RSI, CCI, Williams%R, ATR, swing high/low, Fibo extensions, POC, pivot,
   structural stop, rationale).
3. **Execution Reality** — slippage metrics, partial-fill states, maker execution time,
   race-condition flag.
4. **Post-Trade Reality** — audits the next 10–15 candles after full closure to judge
   whether Lot B exited *prematurely* or *caught the max impulse*.

---

## Running Backtests

### Offline laboratory backtest (no network / no CSV needed)
Generates synthetic volatile data and exercises the full dual-lot engine + capsules:

```bash
python run_lab_backtest.py --symbol SOLUSDT --days 14 --seed 99
```
Options: `--symbol --days --seed --balance --leverage --warmup`.
Prints standard metrics, the **v5.5 dual-lot exit performance** block, and a **raw Trade
Capsule JSON preview**. A canonical run is saved at
`output/lab_backtest_SOLUSDT_14d.log`.

### Standard backtest (requires history CSV in `data/history/`)
```bash
python run_backtest.py
```
Also prints the dual-lot metrics + capsule JSON preview.

### Sample lab result (SOLUSDT, 14 days, seed 99)
```
Trades: 140 | Return: -2.46% | Win Rate 30.0% | MaxDD 5.2%
Dual-Lot Trades:   39
Lot A (Maker Fix): win rate 38.5%
Momentum Exits:    5
Race Conditions:   3 (resolved safely)
Capsules Written:  39 → data/ai_analysis/
```
> The negative return is expected — this is pure synthetic noise with no real edge,
> used only to exercise every exit path end-to-end.

---

## Testing

```bash
python test_v5.py
```
**131 tests** (107 original + 24 new for v5.5): dual-lot exit engine, Trade Capsule
serialization, broker integration, end-to-end dual-lot backtest, and the
`test_entry_thresholds_unchanged` guard.

---

## Data Models & Storage
- **Models:** `engine/models.py` — `Position` (extended with `lot_role`, `lot_state`,
  `maker_tp_order_id`, `structural_stop`, `pivot_level`, `capsule_id`), `LotRole`,
  `LotState`, and the capsule dataclasses (`PreTradeContext`, `InternalThoughts`,
  `ExecutionReality`, `PostTradeReality`, `TradeCapsule`).
- **Storage:** Trade Capsules are written as JSON files under `data/ai_analysis/`
  (gitignored — generated artifacts). No database is used.

---

## Tech Stack
Python 3, pandas / numpy, `pybit` (live Bybit API). No web framework — this is a CLI /
service bot.

## Status
- ✅ v5.5 Two-Winged Dynamic Exit Engine
- ✅ v5.5 Trade Capsule meta-analysis subsystem
- ✅ 131/131 tests passing
- ✅ Entry logic unchanged (regression-guarded)
