import os
from dotenv import load_dotenv
from logging_config import setup_logging
import sys
import threading
import time
import asyncio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.telegram_bot.bot import TelegramAssistantBot

# Настройка логирования
setup_logging()

# Загрузка переменных окружения
load_dotenv()

def main():
    # Получаем токен из переменных окружения
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    
    # Проверяем наличие токена
    if not bot_token:
        raise ValueError("Telegram Bot Token не найден. Установите переменную окружения TELEGRAM_BOT_TOKEN")
    
    # Создаем и запускаем бота
    async def run_bot():
        bot = TelegramAssistantBot()
        await bot.start()
    asyncio.run(run_bot())

if __name__ == "__main__":
    threading.stack_size(200000000)
    thread = threading.Thread(target=main)
    thread.daemon = True  # Поток демон
    thread.start()
    
    # Бесконечный цикл для поддержания работы
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Завершение программы")
    except Exception as e:
        print(f"Непредвиденная ошибка: {e}")
    finally:
        sys.exit(0)