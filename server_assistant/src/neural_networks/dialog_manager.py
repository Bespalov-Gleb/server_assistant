import os
import json
import time
import logging
from datetime import datetime
from typing import List, Dict, Optional

class DialogManager:
    def __init__(
        self, 
        max_context_length: int = 100000, 
        max_tokens: int = 10000,
        context_file: str = 'temp/dialogue_context.json'
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
        
        self.context_file = context_file or os.path.join('temp', 'dialogue_context.json')
        # Создаем директорию, если не существует
        os.makedirs(os.path.dirname(self.context_file), exist_ok=True)
        
        self.context = {
            'messages': [],
            'task_types': {}
        }
        # Инициализация контекста
        self.load_context()
    
    def load_context(self):
        """
        Загрузка контекста из файла
        """
        try:
            if os.path.exists(self.context_file):
                with open(self.context_file, 'r', encoding='utf-8') as f:
                    loaded_context = json.load(f)
                    
                    # Проверка и восстановление структуры контекста
                    if isinstance(loaded_context, list):
                        # Если контекст - список, преобразуем его
                        self.context['messages'] = loaded_context
                        self.context['task_types'] = {}
                    elif isinstance(loaded_context, dict):
                        # Если контекст - словарь, используем его как есть
                        self.context = loaded_context
                    else:
                        # Если структура неизвестна, сбрасываем к дефолту
                        self.context = {
                            'messages': [],
                            'task_types': {}
                        }
            
            return self.context
        
        except Exception as e:
            self.logger.error(f"Ошибка загрузки контекста: {e}")
            return self.context

    def save_context(self):
        """
        Сохранение контекста в файл
        """
        try:
            with open(self.context_file, 'w', encoding='utf-8') as f:
                json.dump(self.context, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"Ошибка сохранения контекста: {e}")

    def add_message(self, message, role='user', task_type=None):
        """
        Добавление сообщения в контекст
        """
        try:
            # Создаем объект сообщения
            message_entry = {
                'role': role,
                'content': message,
                'task_type': task_type,
                'timestamp': datetime.now().timestamp()
            }

            # Добавляем в общий список сообщений
            self.context['messages'].append(message_entry)

            # Добавляем в список сообщений по типу задачи
            if task_type:
                if task_type not in self.context['task_types']:
                    self.context['task_types'][task_type] = []
                self.context['task_types'][task_type].append(message_entry)

            # Обрезаем контекст, если превышена максимальная длина
            if len(self.context['messages']) > self.max_context_length:
                self.context['messages'] = self.context['messages'][-self.max_context_length:]
                
                # Обрезаем контекст для каждого типа задачи
                for task_type in list(self.context['task_types'].keys()):
                    if len(self.context['task_types'][task_type]) > self.max_context_length:
                        self.context['task_types'][task_type] = self.context['task_types'][task_type][-self.max_context_length:]

            # Сохраняем контекст
            self.save_context()

            self.logger.info(f"Добавлено сообщение. Роль: {role}, Тип задачи: {task_type}")
            self.logger.info(f"Общее количество сообщений: {len(self.context['messages'])}")
        
        except Exception as e:
            self.logger.error(f"Ошибка при добавлении сообщения: {e}")

    def get_context(self, task_type=None, include_general=True, hours_to_include=24):
        """
        Получение контекста с возможностью фильтрации
        """
        try:
            # Текущее время
            current_time = datetime.now().timestamp()
            
            # Фильтр по времени (сообщения за последние hours_to_include часов)
            time_threshold = current_time - (hours_to_include * 3600)
            
            # Базовый список сообщений
            context = []
            
            # Добавляем общие сообщения, если включено
            if include_general:
                context.extend([
                    msg for msg in self.context['messages'] 
                    if msg['timestamp'] >= time_threshold
                ])
            
            # Добавляем сообщения конкретного типа задачи
            if task_type and task_type in self.context['task_types']:
                context.extend([
                    msg for msg in self.context['task_types'][task_type] 
                    if msg['timestamp'] >= time_threshold
                ])
            
            # Сортируем по времени
            context.sort(key=lambda x: x['timestamp'])
            
            return context
        
        except Exception as e:
            self.logger.error(f"Ошибка при получении контекста: {e}")
            return []

# Создаем глобальный экземпляр DialogManager
dialog_manager = DialogManager()