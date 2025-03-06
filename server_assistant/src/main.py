import asyncio

from config import get_config
from src.logging_config import setup_logging
from src.telegram_bot.bot import TelegramAssistantBot

# Настройка логирования
setup_logging()

# Загрузка переменных окружения


async def main():
    # Получаем токен из переменных окружения
    telegram_config = get_config().telegram
    bot_token = telegram_config.token

    # Проверяем наличие токена
    if not bot_token:
        raise ValueError("Telegram Bot Token не найден. Установите переменную окружения TELEGRAM_BOT_TOKEN")

    # Создаем и запускаем бота
    bot = TelegramAssistantBot(bot_token)

    try:
        await bot.start()
    except KeyboardInterrupt:
        print("Завершение работы бота...")

if __name__ == "__main__":
    asyncio.run(main())