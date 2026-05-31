"""
v5.0 — Live Broker (Bybit Demo API)
══════════════════════════════════════════════
Implements the Broker Protocol for real exchange interaction.

Key features:
  - Amnesia fix: on init, reads ALL positions/orders from exchange
  - Margin tracking: free_margin = available - pending
  - Enhanced error logging (retCode + retMsg)
  - Instrument caching (lot size, tick size)
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import pandas as pd

from engine.models import (
    AccountState, Direction, Order, OrderStatus, Position,
    PositionPhase,
)

logger = logging.getLogger("aegis.broker.live")


class LiveBroker:
    """
    Bybit Demo API broker — implements Broker Protocol.
    On construction, recovers state from exchange (positions + orders).
    """

    def __init__(self, api_key: str, api_secret: str, leverage: float = 5.0):
        self.api_key = api_key
        self.api_secret = api_secret
        self.leverage = leverage
        self._session = None
        self._instrument_cache: dict[str, dict] = {}
        self._connect()

    # ─────────────────── Connection ───────────────────

    def _connect(self):
        """Initialize pybit HTTP session to Demo API."""
        from pybit.unified_trading import HTTP
        self._session = HTTP(
            testnet=False,
            api_key=self.api_key,
            api_secret=self.api_secret,
            demo=True,
        )
        # Verify connection
        resp = self._session.get_wallet_balance(accountType="UNIFIED")
        if resp["retCode"] != 0:
            raise ConnectionError(
                f"Bybit connection failed: [{resp['retCode']}] {resp.get('retMsg', '')}"
            )
        logger.info("✅ LiveBroker connected to Bybit Demo API")

    # ─────────────────── Broker Protocol ───────────────────

    def get_account(self) -> AccountState:
        """Fetch real account state from Bybit wallet API."""
        try:
            resp = self._session.get_wallet_balance(accountType="UNIFIED")
            if resp["retCode"] != 0:
                logger.warning(f"Wallet error: [{resp['retCode']}] {resp.get('retMsg', '')}")
                return AccountState()

            account_info = resp["result"]["list"][0]
            total_equity = float(account_info.get("totalEquity", 0))
            available = float(account_info.get("totalAvailableBalance", 0))

            # Get USDT coin details
            usdt_balance = 0.0
            unrealized = 0.0
            for coin in account_info.get("coin", []):
                if coin["coin"] == "USDT":
                    usdt_balance = float(coin.get("walletBalance", 0))
                    unrealized = float(coin.get("unrealisedPnl", 0))
                    break

            # Locked = total - available
            locked = max(0, usdt_balance - available)

            # Pending margin from open limit orders
            pending = self._calc_pending_margin()

            return AccountState(
                total_balance=usdt_balance,
                available_balance=available,
                locked_margin=locked,
                pending_margin=pending,
                unrealized_pnl=unrealized,
            )
        except Exception as e:
            logger.error(f"Account fetch error: {e}")
            return AccountState()

    def _calc_pending_margin(self) -> float:
        """Calculate margin reserved by pending limit orders."""
        try:
            resp = self._session.get_open_orders(
                category="linear", settleCoin="USDT"
            )
            if resp["retCode"] != 0:
                return 0.0
            total = 0.0
            for o in resp["result"]["list"]:
                price = float(o.get("price", 0))
                qty = float(o.get("qty", 0))
                if price > 0 and qty > 0:
                    total += (price * qty) / self.leverage
            return total
        except Exception:
            return 0.0

    def get_positions(self) -> list[Position]:
        """
        Read ALL open positions from Bybit exchange.
        This is the Amnesia Fix — no local state dependency.
        """
        positions = []
        try:
            resp = self._session.get_positions(
                category="linear", settleCoin="USDT"
            )
            if resp["retCode"] != 0:
                logger.warning(f"Position fetch error: [{resp['retCode']}] {resp.get('retMsg', '')}")
                return positions

            for p in resp["result"]["list"]:
                size = float(p.get("size", 0))
                if size <= 0:
                    continue

                side = p.get("side", "")
                direction = Direction.LONG if side == "Buy" else Direction.SHORT
                entry = float(p.get("avgPrice", 0))
                sl = float(p.get("stopLoss", 0))
                tp = float(p.get("takeProfit", 0))
                unrealized = float(p.get("unrealisedPnl", 0))
                mark_price = float(p.get("markPrice", entry))

                # Determine phase from current state
                if direction == Direction.LONG:
                    pnl_pct = (mark_price - entry) / entry if entry > 0 else 0
                else:
                    pnl_pct = (entry - mark_price) / entry if entry > 0 else 0

                if pnl_pct >= 0.030:
                    phase = PositionPhase.TRAILING
                elif pnl_pct >= 0.015:
                    phase = PositionPhase.BREAKEVEN
                else:
                    phase = PositionPhase.BREATHING

                pos = Position(
                    symbol=p["symbol"],
                    direction=direction,
                    entry_price=entry,
                    quantity=size,
                    risk_usdt=0.0,  # unknown from exchange
                    stop_loss=sl,
                    take_profit=tp,
                    initial_stop_loss=sl,
                    phase=phase,
                    highest_price=mark_price if direction == Direction.LONG else entry,
                    lowest_price=mark_price if direction == Direction.SHORT else entry,
                    highest_profit_pct=max(0, pnl_pct),
                    entry_time=float(p.get("createdTime", 0)) / 1000.0,
                )
                positions.append(pos)

        except Exception as e:
            logger.error(f"Position recovery error: {e}")

        if positions:
            logger.info(f"📋 Recovered {len(positions)} positions from exchange")
        return positions

    def get_pending_orders(self) -> list[Order]:
        """Read ALL pending orders from exchange — Amnesia Fix."""
        orders = []
        try:
            resp = self._session.get_open_orders(
                category="linear", settleCoin="USDT"
            )
            if resp["retCode"] != 0:
                logger.warning(f"Orders fetch error: [{resp['retCode']}] {resp.get('retMsg', '')}")
                return orders

            for o in resp["result"]["list"]:
                side = o.get("side", "Buy")
                direction = Direction.LONG if side == "Buy" else Direction.SHORT
                order = Order(
                    order_id=o.get("orderId", ""),
                    symbol=o.get("symbol", ""),
                    direction=direction,
                    side=side,
                    price=float(o.get("price", 0)),
                    quantity=float(o.get("qty", 0)),
                    stop_loss=float(o.get("stopLoss", 0)),
                    take_profit=float(o.get("takeProfit", 0)),
                    status=OrderStatus.PENDING,
                    placed_at=float(o.get("createdTime", 0)) / 1000.0,
                )
                orders.append(order)

        except Exception as e:
            logger.error(f"Order recovery error: {e}")

        if orders:
            logger.info(f"📋 Recovered {len(orders)} pending orders from exchange")
        return orders

    def place_limit_order(
        self,
        symbol: str,
        side: str,
        price: float,
        quantity: float,
        stop_loss: float,
        take_profit: float,
        cascade_group_id: str = "",
        cascade_level: str = "",
        direction: Optional[Direction] = None,
        risk_usdt: float = 0.0,
        atr: float = 0.0,
        fibo_ext_1: float = 0.0,
        fibo_ext_2: float = 0.0,
        swing_high: float = 0.0,
        swing_low: float = 0.0,
    ) -> Optional[Order]:
        """Place limit order with PostOnly TIF on Bybit."""
        try:
            # Round to valid tick/lot
            price = self.round_price(price, symbol)
            precision = self.get_lot_precision(symbol)
            quantity = round(quantity, precision)

            if quantity <= 0 or price <= 0:
                logger.warning(f"Invalid order params: {symbol} price={price} qty={quantity}")
                return None

            # Round SL/TP
            if stop_loss > 0:
                stop_loss = self.round_price(stop_loss, symbol)
            if take_profit > 0:
                take_profit = self.round_price(take_profit, symbol)

            params = {
                "category": "linear",
                "symbol": symbol,
                "side": side,
                "orderType": "Limit",
                "price": str(price),
                "qty": str(quantity),
                "timeInForce": "PostOnly",
            }
            if stop_loss > 0:
                params["stopLoss"] = str(stop_loss)
            if take_profit > 0:
                params["takeProfit"] = str(take_profit)

            resp = self._session.place_order(**params)

            if resp["retCode"] == 0:
                result = resp["result"]
                order = Order(
                    order_id=result.get("orderId", ""),
                    symbol=symbol,
                    direction=direction or (Direction.LONG if side == "Buy" else Direction.SHORT),
                    side=side,
                    price=price,
                    quantity=quantity,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    status=OrderStatus.PENDING,
                    placed_at=time.time(),
                    cascade_group_id=cascade_group_id,
                    cascade_level=cascade_level,
                    risk_usdt=risk_usdt,
                    atr=atr,
                    fibo_ext_1=fibo_ext_1,
                    fibo_ext_2=fibo_ext_2,
                    swing_high=swing_high,
                    swing_low=swing_low,
                )
                logger.info(f"✅ Limit order: {side} {symbol} {quantity} @ {price}")
                return order
            else:
                logger.warning(
                    f"⚠️ Order rejected {symbol}: "
                    f"[{resp['retCode']}] {resp.get('retMsg', 'unknown')}"
                )
                return None

        except Exception as e:
            logger.error(f"❌ Order error {symbol}: {e}")
            return None

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel a pending order on exchange."""
        try:
            resp = self._session.cancel_order(
                category="linear", symbol=symbol, orderId=order_id
            )
            if resp["retCode"] == 0:
                logger.info(f"🗑️ Cancelled order {order_id[:8]}.. ({symbol})")
                return True
            else:
                logger.warning(
                    f"Cancel failed {symbol}: [{resp['retCode']}] {resp.get('retMsg', '')}"
                )
                return False
        except Exception as e:
            logger.error(f"Cancel error: {e}")
            return False

    def close_position(self, symbol: str, quantity: float, side: str) -> bool:
        """Close position via market order."""
        try:
            precision = self.get_lot_precision(symbol)
            quantity = round(quantity, precision)
            resp = self._session.place_order(
                category="linear",
                symbol=symbol,
                side=side,
                orderType="Market",
                qty=str(quantity),
                timeInForce="IOC",
            )
            if resp["retCode"] == 0:
                logger.info(f"🔒 Closed position: {side} {symbol} {quantity}")
                return True
            else:
                logger.warning(
                    f"Close failed {symbol}: [{resp['retCode']}] {resp.get('retMsg', '')}"
                )
                return False
        except Exception as e:
            logger.error(f"Close error: {e}")
            return False

    def update_position_sl_tp(
        self, symbol: str, stop_loss: float = 0, take_profit: float = 0
    ) -> bool:
        """Update SL/TP via set_trading_stop."""
        try:
            params = {
                "category": "linear",
                "symbol": symbol,
                "positionIdx": 0,
            }
            if stop_loss > 0:
                params["stopLoss"] = str(self.round_price(stop_loss, symbol))
            if take_profit > 0:
                params["takeProfit"] = str(self.round_price(take_profit, symbol))

            resp = self._session.set_trading_stop(**params)
            if resp["retCode"] == 0:
                logger.info(f"🎯 SL/TP updated: {symbol} SL={stop_loss:.4f} TP={take_profit:.4f}")
                return True
            else:
                logger.warning(
                    f"SL/TP update failed {symbol}: [{resp['retCode']}] {resp.get('retMsg', '')}"
                )
                return False
        except Exception as e:
            logger.error(f"SL/TP error: {e}")
            return False

    def get_klines(
        self, symbol: str, interval: str, limit: int
    ) -> Optional[pd.DataFrame]:
        """Fetch kline data from Bybit."""
        try:
            resp = self._session.get_kline(
                category="linear", symbol=symbol,
                interval=interval, limit=limit,
            )
            if resp["retCode"] != 0:
                logger.warning(f"Kline error {symbol}: [{resp['retCode']}] {resp.get('retMsg', '')}")
                return None

            rows = resp["result"]["list"]
            if not rows:
                return None

            df = pd.DataFrame(rows, columns=[
                "timestamp", "open", "high", "low", "close", "volume", "turnover"
            ])
            df = df.iloc[::-1].reset_index(drop=True)  # oldest first
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce") / 1000.0
            return df

        except Exception as e:
            logger.error(f"Kline fetch error {symbol}: {e}")
            return None

    def get_ticker_price(self, symbol: str) -> float:
        """Get current last traded price."""
        try:
            resp = self._session.get_tickers(category="linear", symbol=symbol)
            if resp["retCode"] == 0:
                tickers = resp["result"]["list"]
                if tickers:
                    return float(tickers[0]["lastPrice"])
            return 0.0
        except Exception:
            return 0.0

    def get_lot_precision(self, symbol: str) -> int:
        """Get quantity decimal precision."""
        info = self._get_instrument(symbol)
        if info:
            qty_step = info.get("lotSizeFilter", {}).get("qtyStep", "1")
            if "." in qty_step:
                return len(qty_step.rstrip("0").split(".")[1])
        return 3

    def get_tick_size(self, symbol: str) -> float:
        """Get price tick size."""
        info = self._get_instrument(symbol)
        if info:
            return float(info.get("priceFilter", {}).get("tickSize", "0.01"))
        return 0.01

    def round_price(self, price: float, symbol: str) -> float:
        """Round price to valid tick size."""
        tick = self.get_tick_size(symbol)
        if tick > 0:
            return round(round(price / tick) * tick, 10)
        return round(price, 6)

    def now(self) -> float:
        """Current real time."""
        return time.time()

    # ─────────────────── Internal helpers ───────────────────

    def _get_instrument(self, symbol: str) -> Optional[dict]:
        """Cached instrument info."""
        if symbol in self._instrument_cache:
            return self._instrument_cache[symbol]
        try:
            resp = self._session.get_instruments_info(
                category="linear", symbol=symbol
            )
            if resp["retCode"] == 0:
                items = resp["result"]["list"]
                if items:
                    self._instrument_cache[symbol] = items[0]
                    return items[0]
        except Exception:
            pass
        return None

    def check_order_status(self, symbol: str, order_id: str) -> Optional[str]:
        """Check order status: 'New', 'Filled', 'Cancelled', 'PartiallyFilled'."""
        try:
            # Check open orders first
            resp = self._session.get_open_orders(
                category="linear", symbol=symbol, orderId=order_id
            )
            if resp["retCode"] == 0 and resp["result"]["list"]:
                return resp["result"]["list"][0].get("orderStatus", "New")

            # Check history
            resp2 = self._session.get_order_history(
                category="linear", symbol=symbol, orderId=order_id
            )
            if resp2["retCode"] == 0 and resp2["result"]["list"]:
                return resp2["result"]["list"][0].get("orderStatus", "Unknown")

            return None
        except Exception:
            return None
