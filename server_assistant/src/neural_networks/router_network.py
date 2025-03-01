import logging
from enum import Enum, auto
from dotenv import load_dotenv
from ..utils.user_preferences import UserPreferences
from .deepseek_processor import DeepSeekProcessor  # Изменили импорт
from .openai_processor import OpenAIProcessor

load_dotenv()

class TaskType(Enum):
    SMALL_TALK = auto()
    COMPLEX_DIALOG = auto()
    FUNCTIONAL = auto()
    INFORMATION = auto()
    REMINDER = auto()

class RouterNetwork:
    def __init__(self, user_id):
        self.logger = logging.getLogger(__name__)
        self.user_preferences = UserPreferences()
        selected_model = self.user_preferences.get_llm_model(user_id=user_id)
        
        self.openai_processor = OpenAIProcessor()
        

    def detect_task_type(self, message: str) -> TaskType:
        system_message = """
        Ты - профессиональный классификатор сообщений.
        В ответе укажи только тип сообщения. 
        Определи тип сообщения ТОЧНО:

        1. SMALL_TALK: 
           - Короткие, дежурные фразы
           - Светская беседа
           - Приветствия и общение без глубокого смысла
           - Например: "Привет", "Как дела?", "Чем занимаешься?"

        2. COMPLEX_DIALOG: 
           - Развернутые диалоги
           - Требуют глубокого, содержательного ответа
           - Обсуждение сложных тем
           - Анализ и размышления
           - Например: "Расскажи о философии искусственного интеллекта"

        3. FUNCTIONAL: 
           - Четкие инструкции и алгоритмы
           - Просьбы о помощи в выполнении задач
           - Пошаговые руководства
           - Например: "Как настроить Wi-Fi?", "Помоги составить план"

        4. INFORMATION: 
           - Информационные запросы
           - Справочные вопросы
           - Получение конкретных знаний
           - Например: "Что такое квантовая физика?", "Сколько планет в солнечной системе?"

        5. REMINDER:
            - Запрос на создание напоминания
            - Просьба напомнить что-либо
            - Просба написать через какое-то время
            - Например: "Напомни сегодня в 16 часов встретить жену с салона"
        """

        classification = self.openai_processor.process_with_retry(
            prompt=system_message + '\n' + message, 
            temperature=0.5,
            max_tokens=2000
        )
        self.logger.info(f"Классификация типа задачи: {classification}")
        # Логика распознавания типа задачи
        if classification:
            classification = classification.upper()
            if 'SMALL_TALK' in classification:
                return TaskType.SMALL_TALK
            elif 'COMPLEX_DIALOG' in classification:
                return TaskType.COMPLEX_DIALOG
            elif 'FUNCTIONAL' in classification:
                return TaskType.FUNCTIONAL
            elif 'INFORMATION' in classification:
                return TaskType.INFORMATION
            elif 'REMINDER' in classification:
                return TaskType.REMINDER

        # Fallback - по умолчанию SMALL_TALK
        self.logger.warning(f"Не удалось определить тип задачи для: {message}")
        return TaskType.SMALL_TALK