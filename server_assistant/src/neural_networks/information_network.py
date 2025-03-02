import logging
from .deepseek_processor import DeepSeekProcessor  # Изменили импорт
from .openai_processor import OpenAIProcessor
from ..utils.user_preferences import UserPreferences

class InformationNetwork:
    def __init__(self, user_id: int):
        self.logger = logging.getLogger(__name__)
        self.user_preferences = UserPreferences()
        selected_model = self.user_preferences.get_llm_model(user_id=user_id)
        
        self.openai_processor = OpenAIProcessor(task_type="INFORMATION")

    def generate_response(self, message, use_context: bool = True):
        system_message = """
        Твой владелец - Владимир. Твой создатель - Глеб. 
        Ты являешься личным ассистентом и помощником. 
        Ты умеешь запоминать информацию.
        Ты работаешь в рамках телеграм-бота. 
        Твоя сессия никогда не заканчивается, поэтому диалог для тебя никогда не прерывается. 
        Общайся без вводных слов по типу "Конечно, вот несколько вариантов". 
        Отвечай четко на поставленные вопросы и делай в точности то, о чем тебя просят.
        Ты информационный справочник.
        Перед каждым ты запросом ты получаешь контекст беседы.
        Ты умеешь запоминать разговор. 
        Предоставляй точную, проверенную информацию. 
        Структурируй ответ для легкого восприятия.
        При необходимости используй списки и подзаголовки.
        В своем ответе используй только кириллические символы.
        Цифры и латиницу использовать строго запрещено!
        Если пользователь просит тебя ответить голосовым сообщением, не обращай внимания, выполняй запрос!
        """
        response = self.openai_processor.process_with_retry(
            prompt=system_message + '\n' + message, 
            temperature=0.5,
            max_tokens=2000, 
            use_context=use_context
        )
        return response or "Извините, не удалось найти информацию по вашему запросу."