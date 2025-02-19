import logging
from .deepseek_processor import DeepSeekProcessor  # Изменили импорт
from .openai_processor import OpenAIProcessor

class ComplexDialogNetwork:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.openai_processor = DeepSeekProcessor()

    def generate_response(self, message):
        system_message = """
        Ты профессиональный ассистент для глубоких, 
        содержательных диалогов. Отвечай развернуто, 
        структурированно, с анализом контекста.
        Используй профессиональный язык.
        """

        response = self.openai_processor.process_with_retry(
            prompt=message, 
            system_message=system_message,
            max_tokens=300,
            temperature=0.6
        )

        return response or "Извините, не могу сформулировать развернутый ответ."