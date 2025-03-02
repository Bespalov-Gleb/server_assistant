import os
import logging
from dotenv import load_dotenv
from openai import OpenAI
from typing import Dict, Any, Optional
from .llm_processor import LLMProcessor
from .dialog_manager import dialog_manager  # Added import for DialogManager

load_dotenv()

class OpenAIProcessor(LLMProcessor):
    def __init__(self, task_type: str = None, user_id: int = None):
        self.logger = logging.getLogger(__name__)
        api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            self.logger.error("OpenAI API ключ не найден!")
            raise ValueError("Необходимо установить OPENAI_API_KEY в .env файле")
        
        self.client = OpenAI(api_key=api_key)
        self.dialog_manager = dialog_manager  # Added DialogManager instance
        self.task_type = task_type

    def process_with_retry(
        self, 
        prompt: str, 
        system_message: str = "", 
        model: str = 'gpt-4o-mini',
        max_tokens: int = 2000, 
        temperature: float = 0.7,
        use_context = True,
        context_file =  None,
    ) -> Optional[str]:
        if use_context == "MEM":
            try:
                if not isinstance(context_file, list):
                    raise ValueError("Context file должен быть списком сообщений")
                memory = context_file
                memory.append({'role': 'user', 'content': prompt})              
                response = self.client.chat.completions.create(
                    model=model,
                    messages=memory,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                assistant_response = response.choices[0].message.content
                return assistant_response
            except Exception as e:
                self.logger.error(f"Ошибка при сопоставлении памяти: {e}")
                return None
        else:

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
        Возвращает информацию о текущей модели OpenAI
        """
        return {
            "name": "GPT-4o-mini",
            "provider": "OpenAI",
            "base_url": "https://api.openai.com/v1",
            "default_model": "gpt-4o-mini"
        }

    def validate_api_key(self) -> bool:
        """
        Проверяет валидность API-ключа OpenAI
        """
        try:
            # Пробуем сделать простой запрос для проверки ключа
            test_response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            return True
        except Exception as e:
            self.logger.error(f"Ошибка валидации API-ключа OpenAI: {e}")
            return False