import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from aiogram import types
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import os.path
import pickle
import os
import json
import re

from src.neural_networks.openai_processor import OpenAIProcessor
from src.utils.user_preferences import UserPreferences
from config import get_config

#TODO: Нужно автоматизировать загрузку json с секретами в .env. Сделать это через кнопку. Подключение аккаунта пользователя к боту для авторизации.
class TodoNetwork:
    """
    Класс для создания списка дел на основе сообщения пользователя.
    Использует нейросеть для генерации списка.
    Полученный список дел загружает в Google Календарь.
    """

    SCOPES = ['https://www.googleapis.com/auth/calendar']
    OAUTH_PORT = 8080

    def __init__(self, chat_id: int):
        """
        Инициализация обработчика задач

        :param chat_id: ID чата для идентификации пользователя
        :type chat_id: int
        """
        self.logger = logging.getLogger(__name__)
        self.user_preferences = UserPreferences()
        self.openai_processor = OpenAIProcessor(chat_id=chat_id)
        self.chat_id = chat_id
        self.config = get_config()
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

        # Обновляем пути к файлам на абсолютные
        self.config.google_calendar.credentials_path = os.path.join(project_root, self.config.google_calendar.credentials_path)
        self.config.google_calendar.token_path = os.path.join(project_root, self.config.google_calendar.token_path)
        self.calendar_service = self._get_calendar_service()


    def _get_calendar_service(self):
        """
        Получение сервиса Google Calendar

        :return: Сервис Google Calendar
        """
        creds = None
        token_path = self.config.google_calendar.token_path
        credentials_path = self.config.google_calendar.credentials_path

        if os.path.exists(token_path):
            with open(token_path, 'rb') as token:
                creds = pickle.load(token)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    credentials_path,
                    self.SCOPES,
                    redirect_uri=f'http://localhost:{self.OAUTH_PORT}/'
                )
                creds = flow.run_local_server(port=self.OAUTH_PORT)
            with open(token_path, 'wb') as token:
                pickle.dump(creds, token)

        return build('calendar', 'v3', credentials=creds)

    def _parse_tasks_from_response(self, response: str) -> List[Dict]:
        """
        Парсинг задач из ответа нейросети

        :param response: Ответ от нейросети
        :type response: str
        :return: Список задач
        :rtype: List[Dict]
        """
        tasks = []
        try:
            # Логируем ответ для отладки
            self.logger.info(f"Получен ответ от нейросети: {response}")

            # Пытаемся найти JSON в ответе
            import json
            import re

            # Ищем JSON в тексте
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                self.logger.info(f"Найден JSON: {json_str}")
                tasks_data = json.loads(json_str)
            else:
                # Если JSON не найден, пробуем распарсить весь ответ
                self.logger.info("JSON не найден, пробуем распарсить весь ответ")
                tasks_data = json.loads(response)

            # Проверяем, что tasks_data это список
            if not isinstance(tasks_data, list):
                tasks_data = [tasks_data]

            for task in tasks_data:
                if not isinstance(task, dict):
                    self.logger.error(f"Задача не является словарем: {task}")
                    continue

                # Проверяем наличие обязательных полей
                title = task.get('task') or task.get('title', 'Без названия')
                start_time = task.get('time') or task.get('start_time')
                duration = task.get('duration', '1 час')

                if not start_time:
                    self.logger.error(f"Задача не содержит времени: {task}")
                    continue

                # Создаем объект задачи
                task_obj = {
                    'summary': title,
                    'description': task.get('description', 'Описание отсутствует'),
                    'start': {
                        'dateTime': start_time,
                        'timeZone': self.config.google_calendar.timezone,
                    },
                    'end': {
                        'dateTime': task.get('end_time', start_time),
                        # Если end_time не указан, используем start_time
                        'timeZone': self.config.google_calendar.timezone,
                    }
                }
                tasks.append(task_obj)
                self.logger.info(f"Успешно создана задача: {task_obj}")

        except json.JSONDecodeError as e:
            self.logger.error(f"Ошибка декодирования JSON: {e}")
            self.logger.error(f"Ответ, который вызвал ошибку: {response}")
        except Exception as e:
            self.logger.error(f"Ошибка парсинга задач: {e}")
            self.logger.error(f"Ответ, который вызвал ошибку: {response}")

        return tasks

    def _add_to_calendar(self, tasks: List[Dict]) -> List[bool]:
        """
        Добавление задач в Google Calendar

        :param tasks: Список задач
        :type tasks: List[Dict]
        :return: Список результатов добавления
        :rtype: List[bool]
        """
        results = []
        for task in tasks:
            try:
                event = self.calendar_service.events().insert(
                    calendarId='primary',
                    body=task
                ).execute()
                results.append(True)
                self.logger.info(f"Задача добавлена в календарь: {event.get('htmlLink')}")
            except Exception as e:
                self.logger.error(f"Ошибка добавления задачи в календарь: {e}")
                results.append(False)
        return results

    def generate_response(self, message: types.Message, transcribe: Optional[str] = None) -> str:
        """
        Генерация ответа на сообщение пользователя

        :param message: Сообщение пользователя
        :type message: types.Message
        :param transcribe: Транскрипция голосового сообщения
        :type transcribe: Optional[str]
        :return: Ответ на сообщение
        :rtype: str
        """
        system_message = """
            Ты - ассистент по управлению задачами. Твоя задача - извлекать из сообщений пользователя задачи
            и возвращать их в формате JSON. Каждая задача должна содержать следующие поля:
            - title: название задачи
            - description: описание задачи
            - start_time: время начала в формате ISO 8601
            - end_time: время окончания в формате ISO 8601
            
            Если время не указано, используй текущую дату и предполагаемую длительность 1 час.
        
            Пример правильного ответа:
            [
                {
                    "title": "Встреча с клиентом",
                    "description": "Обсуждение проекта",
                    "start_time": "2024-03-08T15:00:00",
                    "end_time": "2024-03-08T16:00:00"
                }
            ]
            Задача без времени должна быть в конце списка.
            Обязательно заполни все поля! 
            Если задача не имеет описания, например "С 10 до 11 я буду занят уборкой", в поле description запиши
            "Описание отсутствует".
            Все поля должны быть заполнены!
            Не меняй название полей, записывай их в таком же порядке, как написано в примере выше!
            """
        t = str({datetime.now().isoformat()})
        system_message = system_message + '\n' + t
        # Получаем текст сообщения
        if transcribe:
            text = f"{message.from_user.username}: {transcribe}"
        else:
            text = f"{message.from_user.username}: {message.text}"

        # Получаем ответ от нейросети
        response = self.openai_processor.process_with_retry(
            prompt=text,
            system_message=system_message,
            temperature=0.3,
            use_context=True
        )

        if not response:
            return "Извините, не удалось обработать ваше сообщение."

        # Парсим задачи
        tasks = self._parse_tasks_from_response(response)

        if not tasks:
            return "Извините, не удалось извлечь задачи из вашего сообщения."

        # Добавляем задачи в календарь
        results = self._add_to_calendar(tasks)

        # Формируем ответ
        success_count = sum(1 for r in results if r)
        if success_count == len(tasks):
            return f"Отлично! Я добавил {success_count} задач в ваш календарь."
        else:
            return f"Я добавил {success_count} из {len(tasks)} задач в ваш календарь. Некоторые задачи не удалось добавить."