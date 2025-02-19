import os
import logging
from dotenv import load_dotenv
from openai import OpenAI
from typing import Dict, Any, Optional
from .llm_processor import LLMProcessor

load_dotenv()

class OpenAIProcessor(LLMProcessor):
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            self.logger.error("OpenAI API ключ не найден!")
            raise ValueError("Необходимо установить OPENAI_API_KEY в .env файле")
        
        self.client = OpenAI(api_key=api_key)

    def process_with_retry(
        self, 
        prompt: str, 
        system_message: str = '', 
        model: str = 'gpt-3.5-turbo',
        max_tokens: int = 200, 
        temperature: float = 0.7
    ) -> Optional[str]:
        """
        Обработка запроса через OpenAI
        """
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            generated_text = response.choices[0].message.content.strip()
            
            # Проверка на пустой ответ
            if not generated_text:
                self.logger.warning("Сгенерирован пустой ответ")
                return "❌ Не удалось сгенерировать ответ. Возможны временные проблемы с OpenAI."
            
            return generated_text
        
        except Exception as e:
            error_message = str(e)
            
            # Специфические сообщения об ошибках
            if "quota" in error_message.lower():
                self.logger.error("Превышена квота OpenAI")
                return "❌ Извините, исчерпан лимит использования OpenAI."
            
            self.logger.error(f"Ошибка при запросе к OpenAI: {e}")
            return "❌ Технические неполадки с OpenAI. Попробуйте позже."

    def get_model_info(self) -> Dict[str, Any]:
        """
        Возвращает информацию о текущей модели OpenAI
        """
        return {
            "name": "GPT-3.5 Turbo",
            "provider": "OpenAI",
            "base_url": "https://api.openai.com/v1",
            "default_model": "gpt-3.5-turbo"
        }

    def validate_api_key(self) -> bool:
        """
        Проверяет валидность API-ключа OpenAI
        """
        try:
            # Пробуем сделать простой запрос для проверки ключа
            test_response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            return True
        except Exception as e:
            self.logger.error(f"Ошибка валидации API-ключа OpenAI: {e}")
            return False