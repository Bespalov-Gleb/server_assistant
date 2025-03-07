from datetime import datetime
import json
import logging
import re
import traceback
import asyncio
from aiogram import types

from src.neural_networks.openai_processor import OpenAIProcessor
from src.utils.user_preferences import UserPreferences


class ReminderNetwork:
    """
    Обработчик напоминаний. 
    Создает и управляет напоминаниями пользователя.
    """

    def __init__(self, bot, chat_id):
        """
        :param bot: Экземпляр бота для отправки напоминаний
        :param chat_id: ID чата
        """
        self.logger = logging.getLogger(__name__)
        self.user_preferences = UserPreferences()
        selected_model = self.user_preferences.get_llm_model(chat_id=chat_id)
        
        self.openai_processor = OpenAIProcessor(task_type="REMINDER", chat_id=chat_id)
        self.bot = bot
        self.chat_id = chat_id

    def generate_response(self, message: types.Message, transcribe=None):
        """
        Генерирует структурированный ответ с деталями напоминания.

        :param message: Сообщение с запросом напоминания
        :return: JSON-структура с деталями напоминания
        """
        current_datetime = datetime.now()
        time_message = f"Текущая дата и время: {current_datetime.isoformat()}"
        system_message = """
        Системное сообщение:
        Ты помощник, который создает напоминания. 
        Всегда отвечай ТОЛЬКО в JSON формате с полями:
        {
            "text": "текст напоминания",
            "time": "время напоминания в ISO формате",
            "type": "one-time или constant"
        }
        Примеры:
        1. Напомнить купить хлеб -> {"text": "Купить хлеб", "time": "2025-02-27T18:00:00", "type": "one-time"}
        2. Ежедневная зарядка -> {"text": "Зарядка", "time": "2025-02-27T07:00:00", "type": "constant"}
        Запрос пользователя:
        """
        
        try:
            # Добавляем контекст сообщения в системное сообщение
            if transcribe == None:
                full_prompt = system_message + '\n' + time_message + '\n' + f'Запрос пользователя {message.from_user.username}: ' + message.text
            else:
                full_prompt = system_message + '\n' + time_message + '\n' + f'Запрос пользователя {message.from_user.username}: ' + transcribe
            # Получаем ответ от OpenAI
            response = self.openai_processor.process_with_retry(
                prompt=full_prompt,
                temperature=0.2,
                use_context=True
            )
            
            # Логируем полный ответ
            self.logger.info(f"Полный ответ от OpenAI: {response}")
            
            # Извлекаем JSON из ответа
            return self.parse_reminder_json(response)
        
        except Exception as e:
            self.logger.error(f"Ошибка в generate_response: {e}")
            self.logger.error(traceback.format_exc())
            return None

    def parse_reminder_json(self, response: str):
        """
        Извлекает данные напоминания из JSON-ответа.

        :param response: JSON-строка с данными напоминания
        :return: Кортеж (текст, время, тип) или None при ошибке
        """
        try:
            # Извлечение JSON из текста с помощью регулярного выражения
            json_match = re.search(r'\{[^{}]+\}', response)
            if json_match:
                json_str = json_match.group(0)
                reminder_data = json.loads(json_str)
                
                # Валидация полей
                if not all(key in reminder_data for key in ['text', 'time', 'type']):
                    self.logger.error(f"Неполный JSON: {reminder_data}")
                    return None
                
                # Парсинг времени
                reminder_time = datetime.fromisoformat(reminder_data['time'])
                
                return (
                    reminder_data['text'], 
                    reminder_time, 
                    reminder_data['type']
                )
            
            self.logger.error(f"Не найден JSON в ответе: {response}")
            return None
        
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.error(f"Ошибка парсинга JSON: {e}")
            self.logger.error(f"Проблемный ответ: {response}")
            return None

    async def create_reminder(self, message: str, transcribe=None):
        """
        Создает новое напоминание из сообщения пользователя.

        :param message: Сообщение с запросом напоминания
        :return: Список с параметрами напоминания или сообщение об ошибке
        """
        try:
            # Получаем детали напоминания
            if transcribe == None:
                reminder_details = self.generate_response(message)
            else:
                reminder_details = self.generate_response(message, transcribe)
            
            if reminder_details:
                reminder_text, reminder_time, reminder_type = reminder_details
                return ["Запуск", reminder_text, reminder_time, reminder_type]
            
            return "Не удалось распознать детали напоминания."
        
        except Exception as e:
            self.logger.error(f"Ошибка в process_reminder: {e}")
            return "Произошла ошибка при создании напоминания."