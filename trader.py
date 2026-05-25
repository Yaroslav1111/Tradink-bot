"""
Aegis-Quant-Lab — Bybit Demo API Connection Test
===================================================
Tests connectivity to Bybit Demo futures API.

Usage:
    Set environment variables:
        export BYBIT_DEMO_KEY="your_demo_api_key"
        export BYBIT_DEMO_SECRET="your_demo_api_secret"
    
    Then run:
        python trader.py
"""

import os
import sys

from pybit.unified_trading import HTTP


def connect_demo():
    """Connect to Bybit Demo API and verify."""
    # Load keys from environment (NEVER hardcode in source!)
    api_key = os.environ.get("BYBIT_DEMO_KEY", "")
    api_secret = os.environ.get("BYBIT_DEMO_SECRET", "")

    if not api_key or not api_secret:
        print("⚠️  Установите переменные окружения:")
        print("    export BYBIT_DEMO_KEY='your_key'")
        print("    export BYBIT_DEMO_SECRET='your_secret'")
        print("\n  Ключи создаются на https://testnet.bybit.com/")
        sys.exit(1)

    # Create session
    session = HTTP(
        api_key=api_key,
        api_secret=api_secret,
    )

    # Override endpoint to demo server
    session.endpoint = "https://api-demo.bybit.com"

    # Verify connection
    try:
        balance = session.get_wallet_balance(accountType="UNIFIED")
        
        if balance.get("retCode") == 0:
            print("🎉 Успех! Вы подключились к Демо-рынку Bybit!")
            
            for coin in balance["result"]["list"][0]["coin"]:
                if coin["coin"] == "USDT":
                    print(f"  💰 Ваш демо-баланс: {coin['equity']} USDT")

            # Test market data
            klines = session.get_kline(
                category="linear",
                symbol="BTCUSDT",
                interval="15",
                limit=5,
            )
            if klines.get("retCode") == 0:
                last_price = klines["result"]["list"][0][4]  # close price
                print(f"  📈 BTC текущая цена: ${last_price}")
                print("\n  ✅ API полностью рабочий — готов к интеграции с Aegis Engine!")
            
            return session
        else:
            print(f"❌ Ошибка API: {balance.get('retMsg', 'Unknown')}")
            return None

    except Exception as e:
        print(f"❌ Ошибка подключения: {e}")
        return None


if __name__ == "__main__":
    connect_demo()