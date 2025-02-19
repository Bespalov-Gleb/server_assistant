import logging
from .deepseek_processor import DeepSeekProcessor  # Изменили импорт
from .openai_processor import OpenAIProcessor

class InformationNetwork:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.openai_processor = DeepSeekProcessor()

    def generate_response(self, message):
        system_message = """
        Ты информационный справочник. 
        Предоставляй точную, проверенную информацию. 
        Структурируй ответ для легкого восприятия.
        При необходимости используй списки и подзаголовки.
        """

        response = self.openai_processor.process_with_retry(
            prompt=message, 
            system_message=system_message,
            max_tokens=250,
            temperature=0.5
        )

        return response or "Извините, не удалось найти информацию по вашему запросу."