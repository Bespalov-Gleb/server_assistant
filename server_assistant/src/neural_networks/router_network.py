import logging
from enum import Enum, auto

from src.neural_networks.openai_processor import OpenAIProcessor
from src.utils.user_preferences import UserPreferences


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
    TODO = auto()


class OutputType(Enum):
    TEXT = auto()
    AUDIO = auto()
    MULTI = auto()
    DEFAULT = auto()


class RouterNetwork:
    """
    Классификатор запросов пользователя.
    Определяет тип задачи и желаемый формат вывода.
    Используется внутри GuideNetwork для классификации запросов.
    """

    def __init__(self, chat_id):
        """
        :param chat_id: ID чата для персонализации настроек
        """
       
        self.logger = logging.getLogger(__name__)
        self.user_preferences = UserPreferences()
        
        self.openai_processor = OpenAIProcessor(chat_id=chat_id)
    
    def detect_output_type(self, message: str) -> OutputType:
        """
        Определяет желаемый тип вывода на основе сообщения пользователя.

        :param message: Сообщение пользователя
        :return: Тип вывода из enum OutputType
        """
        system_message = f"""
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
        """
        Определяет тип задачи на основе сообщения пользователя.

        :param message: Сообщение пользователя
        :return: Тип задачи из enum TaskType
        """
        system_message = f"""
        Системное сообщение:
        Ты - профессиональный классификатор сообщений.
        Выполни запрос максимально качественно, иначе пользователь будет очень расстроен.
        Ты должен уловить суть запроса и сопоставить его смысл с одним из типов.
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
            - Просьба написать через какое-то время
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
            - Например: "Дополни заметку про шоколад, запиши еще, что мне очень нравится MAX FUN с мармеладом"

        11. VIEW_MEMORIES:
            - Пользователь просит показать все заметки.
            - Пользователь хочет увидеть все записи.
            - Например: "Что у нас записано?" или "Покажи все заметки"
            
        12. TODO:
            - Запрос на создание списка дел или расписания на день
            - В тексте присутствуют задачи с началом и концом по времени (необязательно)
            - Например: "Давай составим план на сегодня. В 8 утра мне нужно отвезти дочь в школу,
            нужно позавтракать, заехать к маме примерно в 11. С 13 до до 15 у меня совещание в офисе."
            - Чаще всего в запросах такого типа будет фигурировать просьба составить план дел или расписание дня.
            - Не нужно указывать этот тип, если пользователь просит показать ему список дел. 
            - Этот тип исключительно для создания списка дел или расписания дня!  

        Вместе с запросом пользователя ты получаешь контекст диалога.
        Тебе очень важно понимать, когда пользователь хочет обратиться к заметке,
        а когда просит что-то повторить или обратиться к тому, о чем шла речь совсем недавно.
        Для этого опирайся на контекст диалога, пожалуйста.
        ВАЖНО! Основное различие между типами ADD_MEMORY и REMINDER:
        Если речь идет о контексте диалога, а не заметках, не используй RECALL_MEMORY.
        Если пользователь явно говорит о том, что в заметках искать не нужно, верни COMPLEX_DIALOG.
        Если запрос выглядит образом "расскажи это в голосовом", просто верни тип COMPLEX_DIALOG.
        Ты должен очень хорошо различать заметки - долгосрочная память и контекст - кратковременная память.
        Если речь идет об информации, которая обсуждалась в беседе, не нужно возвращать типы с заметками. 
        Проанализируй сообщения с учетом контекста, который ты также получаешь при запросе. Уделяй особое внимание различию между
        контекстной памятью и заметками! Это очень важно!
        Тип ADD_MEMORY - это тип для создания заметок, записей. В заметках не указывается дата и время
        Тип REMINDER - это тип для создания напоминаний. В них указывается дата и время обязательно!
        Тип RECALL_MEMORY - это тип для поиска в памяти информации по запросу пользователя. 
        Обращай особое внимание на различие ADD_MEMORY и RECALL_MEMORY.

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
            elif 'TODO' in classification:
                return TaskType.TODO
        # Fallback - по умолчанию SMALL_TALK
        self.logger.warning(f"Не удалось определить тип задачи для: {message}")
        return TaskType.COMPLEX_DIALOG
