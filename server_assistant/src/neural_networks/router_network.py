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
    DELETE_MEMORY = auto()
    DELETE_ALL_MEMORIES = auto()
    CHANGE_MEMORY = auto()
    VIEW_MEMORIES = auto()

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
        
        self.openai_processor = OpenAIProcessor(user_id=user_id)
    
    def detect_output_type(self, message: str) -> OutputType:
        system_message = """
        Системное сообщение:
        Ты - профессиональный классификатор.
        Твоя задача - определить тип ответа, который хочет пользователь.
        В ответе укажи только тип AUDIO, TEXT, MULTI или DEFAULT. 
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

        Запрос пользователя:
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
        Системное сообщение:
        Ты - профессиональный классификатор сообщений.
        Выполни запрос максимально качественно, иначе пользователь будет очень расстроен.
        Ты должен уловить суть запроса и сопопставить его смысл с одним из типов.
        Не обращай внимания на просьбу ответить текстом или голосом. Твоя задача - определить тип запроса.
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
           - Просьбы о помощи в чем-то практическом 
           - Четкие инструкции и алгоритмы
           - Просьбы о помощи в выполнении задач
           - Пошаговые руководства
           - Например: "Как настроить Wi-Fi?", "Помоги составить план"

        4. INFORMATION: 
           - Просьбы прочитать стихи или рассказать о чем-либо
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

        8. DELETE_MEMORY:
            - Запрос на удаление заметки
            - Пользователь просит удалить какую-то одну заметку.
            - Например: "Удали заметку о шоколаде"

        9. DELETE_ALL_MEMORIES:
            - Запрос на удаление ВСЕХ заметок
            - Пользователь просит удалить ВСЕ заметки.
            - Например: "Удали все заметки" или "Очисти список заметок" или "Очисти память"

        10. CHANGE_MEMORY:
            - Пользователь просит отредактировать уже существующую заметку
            - Запрос на изменение содержания заметки.
            - Запрос этого типа содержит две смысловые части:
            Информация о существующей заметке, которую нужно откорректировать
            Информация о том, какие именно изменения нужно внести.
            - Например: "Дополни заметку про шоколад, запиши еще, что мне очень нравится MAXFUN с мармеладом"

        11. VIEW_MEMORIES:
            - Пользователь просит показать все заметки.
            - Пользователь хочет увидеть все записи.
            - Например: "Что у нас записано?" или "Покажи все заметки"

        Вместе с запросом пользователя ты получаешь контекст диалога.
        Тебе очень важно понимать, когда пользователь хочет обратиться к заметке,
        а когда просит что-то повторить или обратиться к тому, о чем шла речь совсем недавно.
        Для этого опирайся на контекст диалога, пожалуйста.
        ВАЖНО! Основное различие между типами ADD_MEMORY и REMINDER:
        Если речь идет о контексте диалога, а не заметках, не используй RECALL_MEMORY.
        Если пользователь явно говорит о том, что в заметках искать не нужно, верни COMPLEX_DIALOG.
        Если запрос выглядит образом "расскажи это в голосовом", просто верни тип "COMPLEX_DIALOG"
        Ты должен очень хорошо различать заметки - долгосрочная память и контекст - кратковременная память.
        Тип ADD_MEMORY - это тип для создания заметок, записей. В заметках не указывается дата и время
        Тип REMINDER - это тип для создания напоминаний. В них указывается дата и время обязательно!
        Тип RECALL_MEMORY - это тип для поиска в памяти информации по запросу пользователя. 
        Обращай особое вниание на различие ADD_MEMORY и RECALL_MEMORY.

        Запрос пользователя:
        """

        classification = self.openai_processor.process_with_retry(
            prompt=system_message + '\n' + message, 
            temperature=0.5,
            max_tokens=2000,
            use_context=True
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
            elif 'DELETE_MEMORY' in classification:
                return TaskType.DELETE_MEMORY
            elif 'DELETE_ALL_MEMORIES' in classification:
                return TaskType.DELETE_ALL_MEMORIES
            elif 'CHANGE_MEMORY' in classification:
                return TaskType.CHANGE_MEMORY
            elif 'VIEW_MEMORIES' in classification:
                return TaskType.VIEW_MEMORIES

        # Fallback - по умолчанию SMALL_TALK
        self.logger.warning(f"Не удалось определить тип задачи для: {message}")
        return TaskType.INFORMATION