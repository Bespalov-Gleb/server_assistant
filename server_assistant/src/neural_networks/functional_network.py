import logging
from .deepseek_processor import DeepSeekProcessor  # Изменили импорт
from .openai_processor import OpenAIProcessor

class FunctionalNetwork:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.openai_processor = DeepSeekProcessor()

    def generate_response(self, message):
        system_message = """
        Ты помощник для выполнения конкретных задач. 
        Четко и лаконично объясняй алгоритм действий. 
        Давай пошаговые инструкции.
        """

        response = self.openai_processor.process_with_retry(
            prompt=message, 
            system_message=system_message,
            max_tokens=200,
            temperature=0.4
        )

        return response or "Извините, не могу помочь с выполнением этой задачи."