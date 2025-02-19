import logging
from .deepseek_processor import DeepSeekProcessor  # Изменили импорт
from .openai_processor import OpenAIProcessor

class SmallTalkNetwork:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.openai_processor = DeepSeekProcessor()

    def generate_response(self, message):
        system_message = """
        Ты дружелюбный ассистент. 
        Общайся в неформальном стиле, 
        кратко и позитивно.
        """

        response = self.openai_processor.process_with_retry(
            prompt=message, 
            system_message=system_message,
            max_tokens=100,
            temperature=0.7
        )

        return response or "Извините, не могу сформулировать ответ."