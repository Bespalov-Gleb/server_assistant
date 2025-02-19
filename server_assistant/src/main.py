import os
from dotenv import load_dotenv
from logging_config import setup_logging
import sys
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
    telegram_bot = TelegramAssistantBot()
    telegram_bot.start()  # Изменили с start() на run()

if __name__ == "__main__":
    main()