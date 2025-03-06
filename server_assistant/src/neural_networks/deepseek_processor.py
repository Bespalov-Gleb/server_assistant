import os
import logging
from dotenv import load_dotenv
from openai import OpenAI
from typing import Dict, Any, Optional
from src.neural_networks.llm_processor import LLMProcessor
from src.neural_networks.dialog_manager import dialog_manager

load_dotenv()

class DeepSeekProcessor(LLMProcessor):
    def __init__(self, task_type: str = None):
        self.logger = logging.getLogger(__name__)
        api_key = os.getenv('DEEPSEEK_API_KEY')
        
        if not api_key:
            self.logger.error("Deepseek API ключ не найден!")
            raise ValueError("Необходимо установить DEEPSEEK_API_KEY в .env файле")
        
        self.client = OpenAI(api_key=api_key)
        self.dialog_manager = dialog_manager  # Added DialogManager instance
        self.task_type = task_type

    def process_with_retry(
        self, 
        prompt: str, 
        system_message: str = """ Твой владелец - Владимир. Твой создатель - Глеб. 
        Ты являешься личным ассистентом и помощником. 
        Ты умеешь запоминать информацию.
        Ты работаешь в рамках телеграм-бота. 
        Твоя сессия никогда не заканчивается, поэтому диалог для тебя никогда не прерывается. 
        Общайся без вводных слов по типу "Конечно, вот несколько вариантов". 
        Отвечай четко на поставленные вопросы и делай в точности то, о чем тебя просят.""", 
        model: str = 'gpt-4o-mini',
        max_tokens: int = 2000, 
        temperature: float = 0.7,
        use_context: bool = True
    ) -> Optional[str]:
        try:
            # Подготовка контекста
            context = self.dialog_manager.get_context() if use_context else []
            
            # Добавление системного сообщения            
            if system_message:
                context.insert(0, {'role': 'system', 'content': system_message})
            
            # Добавление текущего промпта
            context.append({'role': 'user', 'content': prompt})
            
            # Вызов OpenAI API с контекстом
            response = self.client.chat.completions.create(
                model=model,
                messages=context,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Извлечение ответа
            assistant_response = response.choices[0].message.content
            
            # Добавление ответа в контекст
            self.dialog_manager.add_message(prompt, role='user')
            self.dialog_manager.add_message(assistant_response, role='assistant')
            
            return assistant_response
        
        except Exception as e:
            self.logger.error(f"Ошибка обработки: {e}")
            return None

    def get_model_info(self) -> Dict[str, Any]:
        """
        Возвращает информацию о текущей модели Deepseek
        """
        return {
            "name": "deepseek_chat",
            "provider": "Deepseek",
            "base_url": "https://api.deepseek.com",
            "default_model": "deepseek_chat"
        }

    def validate_api_key(self) -> bool:
        """
        Проверяет валидность API-ключа OpenAI
        """
        try:
            # Пробуем сделать простой запрос для проверки ключа
            test_response = self.client.chat.completions.create(
                model="deepseek_chat",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            return True
        except Exception as e:
            self.logger.error(f"Ошибка валидации API-ключа Deepseek: {e}")
            return False