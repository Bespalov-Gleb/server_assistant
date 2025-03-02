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
    ADD_MEMORY = auto()
    RECALL_MEMORY = auto()

class OutputType(Enum):
    TEXT = auto()
    AUDIO = auto()
    MULTI = auto()
    DEFAULT = auto()

class RouterNetwork:
    def __init__(self, user_id):
        self.logger = logging.getLogger(__name__)
        self.user_preferences = UserPreferences()
        selected_model = self.user_preferences.get_llm_model(user_id=user_id)
        
        self.openai_processor = OpenAIProcessor()
    
    def detect_output_type(self, message: str) -> OutputType:
        system_message = """
        Ты - профессиональный классификатор.
        Твоя задача - определить тип ответа, который хочет пользователь.
        В ответе укажи только тип. 
        Определи тип ТОЧНО:

        1. TEXT: 
           - Пользователь явно говорит о том, что ответ должен быть текстовым
           - Например:"Напиши текстом", "Ответь текстовым сообщением"

        2. AUDIO:
           - Пользователь явно говорит о том, что ответ должен быть в виде голосового сообщения
           - Например: "Ответь голосовым", "Расскажи в голосовом", "Ответь голосовым сообщением"
        3. MULTI:
           - ОБЯЗАТЕЛЬНО используй этот тип, когда пользователь просит что-то напомнить, составить план
        4. DEFAULT:
            - Если пользователь не указывает желаемый тип ответа, не просит напомнить что-то, составить план и т.д.
        """
        classification = self.openai_processor.process_with_retry(
            prompt=system_message + '\n' + message, 
            temperature=0.5,
            max_tokens=2000
        )
        self.logger.info(f"Классификация типа ответа: {classification}")
        # Логика распознавания типа задачи
        if classification:
            classification = classification.upper()
            if 'TEXT' in classification:
                return OutputType.TEXT
            elif 'AUDIO' in classification:
                return OutputType.AUDIO
            elif 'MULTI' in classification:
                return OutputType.MULTI
            elif 'DEFAULT' in classification:
                return OutputType.DEFAULT

        # Fallback - по умолчанию SMALL_TALK
        self.logger.warning(f"Не удалось определить тип ответа для: {message}")
        return OutputType.TEXT

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
        6. ADD_MEMORY:
            - Пользователь просит запомнить информацию, при этом не указывает время
            - Фиксация каких-то данных
            - Создание заметки
            - Например: "Запомни, мне нравится шоколад Alpen Gold"
        7. RECALL_MEMORY:
            - Пользователь просит вспомнить что-либо
            - Если пользователь говорит "напомни", при этом не указывая время, то
            в большинстве случаев этот запрос относится к данному типу
            - Например: "Напомни мне, какой шоколад мне понравился?"

        ВАЖНО! Основное различие между типами ADD_MEMORY и REMINDER:
        Тип ADD_MEMORY - это тип для создания заметок, записей. В заметках не указывается дата и время
        Тип REMINDER - это тип для создания напоминаний. В них указывается дата и время обязательно!
        Тип RECALL_MEMORY - это тип для поиска в памяти информации по запросу пользователя. 
        Обращай особое вниание на различие ADD_MEMORY и RECALL_MEMORY.
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
            elif 'ADD_MEMORY' in classification:
                return TaskType.ADD_MEMORY
            elif 'RECALL_MEMORY' in classification:
                return TaskType.RECALL_MEMORY

        # Fallback - по умолчанию SMALL_TALK
        self.logger.warning(f"Не удалось определить тип задачи для: {message}")
        return TaskType.SMALL_TALK