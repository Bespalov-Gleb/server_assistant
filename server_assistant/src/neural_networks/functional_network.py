import logging
from .deepseek_processor import DeepSeekProcessor  # Изменили импорт
from .openai_processor import OpenAIProcessor
from ..utils.user_preferences import UserPreferences

class FunctionalNetwork:
    def __init__(self, user_id: int):
        self.logger = logging.getLogger(__name__)
        self.user_preferences = UserPreferences()
        selected_model = self.user_preferences.get_llm_model(user_id=user_id)

            
        self.openai_processor = OpenAIProcessor(task_type="FUNCTIONAL")

    def generate_response(self, message):
        system_message = """
        Твой владелец - Владимир. Твой создатель - Глеб. 
        Ты являешься личным ассистентом и помощником. 
        Ты умеешь запоминать информацию.
        Ты работаешь в рамках телеграм-бота. 
        Твоя сессия никогда не заканчивается, поэтому диалог для тебя никогда не прерывается. 
        Общайся без вводных слов по типу "Конечно, вот несколько вариантов". 
        Отвечай четко на поставленные вопросы и делай в точности то, о чем тебя просят.
        Ты помощник для выполнения конкретных задач.
        Перед каждым ты запросом ты получаешь контекст беседы.
        Ты умеешь запоминать разговор. 
        Четко и лаконично объясняй алгоритм действий. 
        Давай пошаговые инструкции.
        В своем ответе используй только кириллические символы.
        Цифры и латиницу использовать строго запрещено!
        """

        response = self.openai_processor.process_with_retry(
            prompt=system_message + '\n' + message, 
            max_tokens=2000, 
            temperature=0.4,
            use_context=True
        )

        return response or "Извините, не могу помочь с выполнением этой задачи."