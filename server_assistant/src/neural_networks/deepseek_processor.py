import os
import logging
from dotenv import load_dotenv
from openai import OpenAI
from typing import Dict, Any, Optional
from .llm_processor import LLMProcessor

load_dotenv()

class DeepSeekProcessor(LLMProcessor):
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        api_key = os.getenv('DEEPSEEK_API_KEY')
        
        if not api_key:
            self.logger.error("DeepSeek API ключ не найден!")
            raise ValueError("Необходимо установить DEEPSEEK_API_KEY в .env файле")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1"
        )

    def process_with_retry(
        self, 
        prompt: str, 
        system_message: str = '', 
        model: str = "deepseek/chat",
        max_tokens: int = 200, 
        temperature: float = 0.7
    ) -> Optional[str]:
        """
        Обработка запроса через DeepSeek
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
                return None
            
            return generated_text
        
        except Exception as e:
            error_message = str(e)
            
            # Специфические сообщения об ошибках
            if "Insufficient Balance" in error_message:
                self.logger.error("Недостаточно средств на счете DeepSeek")
                return "❌ Извините, закончились средства на DeepSeek. Переключаюсь на резервную модель."
            
            self.logger.error(f"Ошибка при запросе к DeepSeek: {e}")
            return None

    def get_model_info(self) -> Dict[str, Any]:
        """
        Возвращает информацию о текущей модели DeepSeek
        """
        return {
            "name": "DeepSeek Chat",
            "provider": "DeepSeek",
            "base_url": "https://api.deepseek.com/v1",
            "default_model": "deepseek/chat"
        }

    def validate_api_key(self) -> bool:
        """
        Проверяет валидность API-ключа DeepSeek
        """
        try:
            # Пробуем сделать простой запрос для проверки ключа
            test_response = self.client.chat.completions.create(
                model="deepseek/chat",
                messages=[{"role": "user", "content": "Привет"}],
                max_tokens=10
            )
            return True
        except Exception as e:
            error_message = str(e)
            
            if "Insufficient Balance" in error_message:
                self.logger.warning("Недостаточно средств на счете DeepSeek")
                return False
            
            self.logger.error(f"Ошибка валидации API-ключа DeepSeek: {e}")
            return False