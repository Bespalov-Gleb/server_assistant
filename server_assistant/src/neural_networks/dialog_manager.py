import os
import json
import time
import logging
from typing import List, Dict, Optional

class DialogManager:
    def __init__(
        self, 
        max_context_length: int = 50, 
        max_tokens: int = 4000,
        context_file: str = None
    ):
        """
        Менеджер диалогового контекста
        
        :param max_context_length: Максимальное количество сообщений в контексте
        :param max_tokens: Максимальное количество токенов в контексте
        :param context_file: Путь для персистентного хранения контекста
        """
        self.logger = logging.getLogger(__name__)
        self.max_context_length = max_context_length
        self.max_tokens = max_tokens
        
        # Путь для хранения контекста
        self.context_file = context_file or os.path.join(
            os.path.dirname(__file__), 
            '..', 
            '..', 
            'temp', 
            'dialogue_context.json'
        )
        
        # Создаем директорию, если не существует
        os.makedirs(os.path.dirname(self.context_file), exist_ok=True)
        
        # Инициализация контекста
        self.conversation_history: List[Dict] = self._load_context()
    
    def _load_context(self) -> List[Dict]:
        """Загрузка контекста из файла"""
        try:
            if os.path.exists(self.context_file):
                with open(self.context_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return []
        except Exception as e:
            self.logger.error(f"Ошибка загрузки контекста: {e}")
            return []
    
    def _save_context(self):
        """Сохранение контекста в файл"""
        try:
            with open(self.context_file, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"Ошибка сохранения контекста: {e}")
    
    def add_message(self, content: str, role: str = 'user'  ):
        """
        Добавление сообщения в историю
        
        :param role: Роль отправителя (user/assistant)
        :param content: Текст сообщения
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": time.time()
        }
        
        self.conversation_history.append(message)
        
        # Усечение истории по длине
        if len(self.conversation_history) > self.max_context_length:
            self.conversation_history.pop(0)
        
        self._save_context()
    
    def get_context(self, include_system_prompt: bool = True) -> List[Dict]:
        """
        Получение контекста для генерации
        
        :param include_system_prompt: Добавлять ли системный промпт
        :return: Список сообщений контекста
        """
        context = self.conversation_history.copy()
        
        if include_system_prompt:
            system_prompt = {
                "role": "system",
                "content": """
                Твой владелец - Владимир. Твой создатель - Глеб. 
        Ты являешься личным ассистентом и помощником. 
        Ты умеешь запоминать информацию. 
        Общайся без вводных слов по типу "Конечно, вот несколько вариантов". 
        Отвечай четко на поставленные вопросы и делай в точности то, о чем тебя просят
                """
            }
            context.insert(0, system_prompt)
        
        return context
    
    def clear_context(self):
        """Очистка истории диалога"""
        self.conversation_history = []
        self._save_context()
    
    def get_last_message(self, role: Optional[str] = None) -> Optional[Dict]:
        """
        Получение последнего сообщения
        
        :param role: Фильтр по роли (user/assistant)
        :return: Последнее сообщение или None
        """
        if not self.conversation_history:
            return None
        
        filtered_history = [
            msg for msg in self.conversation_history 
            if role is None or msg['role'] == role
        ]
        
        return filtered_history[-1] if filtered_history else None
DialogManager = DialogManager