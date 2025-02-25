import os
import logging
from dotenv import load_dotenv
from openai import OpenAI
from typing import Dict, Any, Optional
from .llm_processor import LLMProcessor
from .dialog_manager import DialogManager

load_dotenv()

class DeepSeekProcessor(LLMProcessor):
    def __init__(self, user_id: int = None):
        self.logger = logging.getLogger(__name__)
        api_key = os.getenv('DEEPSEEK_API_KEY')
        
        if not api_key:
            self.logger.error("DeepSeek API ключ не найден!")
            raise ValueError("Необходимо установить DEEPSEEK_API_KEY в .env файле")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1"
        )
        
        self.dialog_manager = DialogManager(
            max_context_length=5,
            max_tokens=1000,
            context_file=f'temp/dialog_context_{user_id}.json' if user_id else None
        )

    def process_with_retry(
        self, 
        prompt: str, 
        system_message: str = '', 
        model: str = "deepseek/chat",
        max_tokens: int = 2000, 
        temperature: float = 0.7,
        use_context: bool = True
    ) -> Optional[str]:
        """
        Обработка запроса через DeepSeek с поддержкой контекста
        
        :param prompt: Текст запроса
        :param system_message: Системное сообщение
        :param model: Модель для генерации
        :param max_tokens: Максимальная длина ответа
        :param temperature: Креативность ответа
        :param use_context: Использовать ли диалоговый контекст
        :return: Сгенерированный текст
        """
        try:
            # Подготовка контекста
            messages = []
            
            if use_context:
                # Добавляем сообщение пользователя в историю
                self.dialog_manager.add_message("user", prompt)
                
                # Получаем контекст
                messages = self.dialog_manager.get_context()
            else:
                # Если контекст не нужен, создаем базовое сообщение
                messages = [
                    {"role": "system", "content": system_message or "Ты - helpful ассистент"},
                    {"role": "user", "content": prompt}
                ]
            
            # Генерация ответа
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Извлечение текста ответа
            generated_text = response.choices[0].message.content.strip()
            
            # Добавляем ответ в историю, если используется контекст
            if use_context:
                self.dialog_manager.add_message("assistant", generated_text)
            
            return generated_text
        
        except Exception as e:
            self.logger.error(f"Ошибка генерации: {e}")
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