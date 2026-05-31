"""
v5.0 — Parameter Optimizer (Grid Search)
══════════════════════════════════════════════
Runs backtests with different StrategyConfig combinations.

Key features:
  - Grid search over any config parameters
  - Multi-symbol batch optimization (individual best params per coin)
  - Parallel execution via ProcessPoolExecutor
  - Results sorted by profit and drawdown
  - JSON export of best params per symbol (optimized_params.json)
"""
from __future__ import annotations

import itertools
import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, fields
from pathlib import Path
from typing import Optional

import pandas as pd

from engine.strategy import StrategyConfig
from runner.backtest_runner import BacktestRunner, BacktestConfig

logger = logging.getLogger("aegis.runner.optimizer")


def _run_single_backtest(args: tuple) -> dict:
    """
    Worker function for parallel optimization.
    Must be at module level for pickling.
    """
    cfg_dict, bt_cfg_dict, df_15m_records, df_1m_records = args

    # Reconstruct objects
    strategy_cfg = StrategyConfig(**cfg_dict)
    bt_cfg = BacktestConfig(**bt_cfg_dict)

    df_15m = pd.DataFrame.from_records(df_15m_records)
    df_1m = pd.DataFrame.from_records(df_1m_records) if df_1m_records else None

    runner = BacktestRunner(strategy_cfg, bt_cfg)
    result = runner.run(df_15m, df_1m)

    # Add config params to result for identification
    result["params"] = cfg_dict
    return result


class Optimizer:
    """
    Grid search optimizer over StrategyConfig parameters.

    Usage:
        optimizer = Optimizer(
            param_grid={
                "fibo_primary": [0.5, 0.618, 0.786],
                "order_ttl_seconds": [300, 600, 900],
                "sl_atr_multiplier": [1.5, 2.0, 2.5],
            },
            base_config=StrategyConfig(),
            backtest_config=BacktestConfig(symbol="BTCUSDT"),
        )
        results = optimizer.run(df_15m, df_1m)
    """

    def __init__(
        self,
        param_grid: dict[str, list],
        base_config: Optional[StrategyConfig] = None,
        backtest_config: Optional[BacktestConfig] = None,
        max_workers: int = 4,
    ):
        self.param_grid = param_grid
        self.base_config = base_config or StrategyConfig()
        self.bt_config = backtest_config or BacktestConfig()
        self.max_workers = max_workers

        # Generate all combinations
        self.combinations = self._generate_combinations()
        logger.info(f"🔬 Optimizer: {len(self.combinations)} parameter combinations")

    def _generate_combinations(self) -> list[dict]:
        """Generate all parameter combinations from grid."""
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())

        # Validate keys exist in StrategyConfig
        valid_fields = {f.name for f in fields(StrategyConfig)}
        for key in keys:
            if key not in valid_fields:
                raise ValueError(f"Invalid parameter: '{key}' not in StrategyConfig")

        combinations = []
        for combo in itertools.product(*values):
            param_dict = dict(zip(keys, combo))
            combinations.append(param_dict)

        return combinations

    def _build_config(self, overrides: dict) -> StrategyConfig:
        """Create StrategyConfig with overrides applied."""
        base_dict = asdict(self.base_config)
        base_dict.update(overrides)
        return StrategyConfig(**base_dict)

    def run(
        self,
        df_15m: pd.DataFrame,
        df_1m: Optional[pd.DataFrame] = None,
        parallel: bool = True,
    ) -> pd.DataFrame:
        """
        Run all backtests and return sorted results.

        Args:
            df_15m: Historical 15m OHLCV data
            df_1m: Optional 1m data for POC
            parallel: Use parallel execution (default True)

        Returns:
            DataFrame with results sorted by profit, then drawdown
        """
        start_time = time.time()
        total = len(self.combinations)

        logger.info(f"🚀 Starting optimization: {total} backtests...")

        # Prepare serializable data
        df_15m_records = df_15m.to_dict("records")
        df_1m_records = df_1m.to_dict("records") if df_1m is not None else None
        bt_cfg_dict = asdict(self.bt_config)

        results = []

        if parallel and total > 1:
            results = self._run_parallel(df_15m_records, df_1m_records, bt_cfg_dict)
        else:
            results = self._run_sequential(df_15m_records, df_1m_records, bt_cfg_dict)

        elapsed = time.time() - start_time

        # Build results DataFrame
        rows = []
        for r in results:
            row = {
                "total_pnl": r.get("total_pnl", 0),
                "return_pct": r.get("return_pct", 0),
                "max_drawdown_pct": r.get("max_drawdown_pct", 0),
                "total_trades": r.get("total_trades", 0),
                "win_rate": r.get("win_rate", 0),
                "profit_factor": r.get("profit_factor", 0),
                "final_balance": r.get("final_balance", 0),
                "avg_trade_pnl": r.get("avg_trade_pnl", 0),
            }
            # Add parameter columns
            params = r.get("params", {})
            for key in self.param_grid.keys():
                row[f"param_{key}"] = params.get(key, "")
            rows.append(row)

        df_results = pd.DataFrame(rows)

        # Sort: best profit first, then least drawdown
        if not df_results.empty:
            df_results = df_results.sort_values(
                by=["total_pnl", "max_drawdown_pct"],
                ascending=[False, True],
            ).reset_index(drop=True)

        logger.info(
            f"✅ Optimization complete: {total} runs in {elapsed:.1f}s "
            f"({elapsed/total:.2f}s/run)"
        )

        if not df_results.empty:
            best = df_results.iloc[0]
            logger.info(
                f"🏆 Best: PnL={best['total_pnl']:+.2f} | "
                f"Return={best['return_pct']:.1f}% | "
                f"MaxDD={best['max_drawdown_pct']:.1f}% | "
                f"WR={best['win_rate']:.1f}%"
            )

        return df_results

    def _run_parallel(self, df_15m_records, df_1m_records, bt_cfg_dict) -> list[dict]:
        """Run backtests in parallel."""
        # Build tasks
        tasks = []
        for combo in self.combinations:
            cfg = self._build_config(combo)
            cfg_dict = asdict(cfg)
            tasks.append((cfg_dict, bt_cfg_dict, df_15m_records, df_1m_records))

        results = []
        completed = 0
        total = len(tasks)

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(_run_single_backtest, task): i
                for i, task in enumerate(tasks)
            }
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1
                    if completed % 10 == 0 or completed == total:
                        logger.info(f"  Progress: {completed}/{total}")
                except Exception as e:
                    completed += 1
                    logger.warning(f"  Backtest failed: {e}")
                    results.append({"total_pnl": 0, "params": {}})

        return results

    def _run_sequential(self, df_15m_records, df_1m_records, bt_cfg_dict) -> list[dict]:
        """Run backtests sequentially (for debugging or small grids)."""
        results = []
        total = len(self.combinations)

        for i, combo in enumerate(self.combinations):
            try:
                cfg = self._build_config(combo)
                cfg_dict = asdict(cfg)
                result = _run_single_backtest(
                    (cfg_dict, bt_cfg_dict, df_15m_records, df_1m_records)
                )
                results.append(result)
            except Exception as e:
                logger.warning(f"  Run {i+1}/{total} failed: {e}")
                results.append({"total_pnl": 0, "params": asdict(self._build_config(combo))})

            if (i + 1) % 10 == 0 or (i + 1) == total:
                logger.info(f"  Progress: {i+1}/{total}")

        return results

    @staticmethod
    def print_results(df: pd.DataFrame, top_n: int = 20):
        """Pretty-print top results."""
        if df.empty:
            print("No results to display.")
            return

        print(f"\n{'═'*80}")
        print(f"  TOP {min(top_n, len(df))} OPTIMIZATION RESULTS")
        print(f"{'═'*80}")
        print(
            f"{'#':>3} | {'PnL':>10} | {'Return%':>8} | {'MaxDD%':>7} | "
            f"{'Trades':>6} | {'WinRate':>7} | {'PF':>6} | Parameters"
        )
        print(f"{'─'*80}")

        param_cols = [c for c in df.columns if c.startswith("param_")]

        for i, row in df.head(top_n).iterrows():
            params_str = " | ".join(
                f"{col.replace('param_', '')}={row[col]}"
                for col in param_cols
            )
            print(
                f"{i+1:>3} | {row['total_pnl']:>+10.2f} | "
                f"{row['return_pct']:>7.1f}% | {row['max_drawdown_pct']:>6.1f}% | "
                f"{int(row['total_trades']):>6} | {row['win_rate']:>6.1f}% | "
                f"{row['profit_factor']:>6.2f} | {params_str}"
            )

        print(f"{'═'*80}\n")

    def get_best_params(self, df_results: pd.DataFrame) -> dict:
        """
        Extract the best (Top-1) parameter overrides from results.
        Returns only the grid-searched params (not the full StrategyConfig).
        """
        if df_results.empty:
            return {}

        best_row = df_results.iloc[0]
        params = {}
        for key in self.param_grid.keys():
            col = f"param_{key}"
            if col in best_row:
                val = best_row[col]
                # Convert numpy types to Python native for JSON serialization
                if hasattr(val, 'item'):
                    val = val.item()
                params[key] = val
        return params

    @staticmethod
    def save_optimized_params(
        results: dict[str, dict],
        output_path: str = "optimized_params.json",
    ):
        """
        Save best params per symbol to JSON.

        Args:
            results: {"BTCUSDT": {"fibo_primary": 0.786, ...}, "ETHUSDT": {...}}
            output_path: path to save JSON file
        """
        # Ensure all values are JSON-serializable
        clean = {}
        for symbol, params in results.items():
            clean[symbol] = {
                k: v.item() if hasattr(v, 'item') else v
                for k, v in params.items()
            }

        with open(output_path, "w") as f:
            json.dump(clean, f, indent=2)

        logger.info(f"💾 Saved optimized params for {len(clean)} symbols → {output_path}")

    @staticmethod
    def load_optimized_params(path: str = "optimized_params.json") -> dict[str, dict]:
        """
        Load optimized params from JSON file.
        Returns: {"BTCUSDT": {"fibo_primary": 0.786, ...}, ...}
        """
        p = Path(path)
        if not p.exists():
            return {}
        with open(p, "r") as f:
            return json.load(f)

    @staticmethod
    def build_symbol_config(
        symbol: str,
        optimized_params: dict[str, dict],
        base_config: Optional[StrategyConfig] = None,
    ) -> StrategyConfig:
        """
        Build StrategyConfig for a specific symbol.
        If symbol has optimized params → apply overrides.
        Otherwise → return base config (defaults).
        """
        base = base_config or StrategyConfig()
        if symbol not in optimized_params:
            return base

        base_dict = asdict(base)
        base_dict.update(optimized_params[symbol])
        return StrategyConfig(**base_dict)
