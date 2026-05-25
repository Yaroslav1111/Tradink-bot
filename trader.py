from pybit.unified_trading import HTTP


# Вставляем КЛЮЧИ, созданные ВНУТРИ ДЕМО-РЕЖИМА
DEMO_API_KEY = "4Q80TBT7uHmFzfgC95"
DEMO_API_SECRET = "GozL8P6vVgTU8HIlAx0ov5AA6QjJIziYsKAh"

# 1. Создаем сессию
session = HTTP(
    api_key=DEMO_API_KEY,
    api_secret=DEMO_API_SECRET
)

# 2. Перезаписываем скрытый эндпоинт напрямую в демо-сервер
session.endpoint = "https://api-demo.bybit.com"

# Проверяем работу
try:
    balance = session.get_wallet_balance(accountType="UNIFIED")
    print("🎉 Успех! Вы подключились к Демо-рынку Bybit!")
    
    # Красиво выведем доступный баланс USDT для проверки
    for coin in balance['result']['list'][0]['coin']:
        if coin['coin'] == 'USDT':
            print(f"Ваш демо-баланс: {coin['equity']} USDT")
            
except Exception as e:
    print(f"Ошибка подключения: {e}")