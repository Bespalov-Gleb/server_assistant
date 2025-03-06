import os
from dotenv import load_dotenv
from src.logging_config import setup_logging
import sys
import asyncio


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

async def main():
    # Получаем токен из переменных окружения
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')

    # Проверяем наличие токена
    if not bot_token:
        raise ValueError("Telegram Bot Token не найден. Установите переменную окружения TELEGRAM_BOT_TOKEN")

    # Создаем и запускаем бота
    bot = TelegramAssistantBot()

    try:
        await bot.start()
    except KeyboardInterrupt:
        print("Завершение работы бота...")

if __name__ == "__main__":
    asyncio.run(main())