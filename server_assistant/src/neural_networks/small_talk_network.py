import logging
from .deepseek_processor import DeepSeekProcessor  # Изменили импорт
from .openai_processor import OpenAIProcessor
from ..utils.user_preferences import UserPreferences

class SmallTalkNetwork:
    def __init__(self, user_id):
        self.logger = logging.getLogger(__name__)
        self.user_preferences = UserPreferences()
        self.openai_processor = OpenAIProcessor(task_type="SMALL_TALK")
    

    def generate_response(self, message, use_context: bool = True):
        system_message = """
        Ты дружелюбный ассистент.
        Перед каждым ты запросом ты получаешь контекст беседы.
        Ты умеешь запоминать разговор. 
        Общайся в неформальном стиле, 
        кратко и позитивно.
        В своем ответе используй только кириллические символы.
        Цифры и латиницу использовать строго запрещено!
        """

        response = self.openai_processor.process_with_retry(
            prompt=system_message + '\n' + message, 
            max_tokens=2000, 
            temperature=0.7,
            use_context=use_context
        )

        return response or "Извините, не могу сформулировать ответ."