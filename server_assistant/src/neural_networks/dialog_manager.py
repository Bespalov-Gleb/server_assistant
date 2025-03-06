import json
import logging
import os
from datetime import datetime


class DialogManager:
    """
    Менеджер для управления контекстом диалога.
    
    Обеспечивает сохранение и загрузку истории диалога,
    фильтрацию по типам задач и временным периодам.
    """

    def __init__(
        self, 
        max_context_length: int = 100000, 
        max_tokens: int = 10000,
        context_file: str = 'temp/dialogue_context.json'
    ):
        """
        Инициализация менеджера диалога

        :param max_context_length: Максимальное количество сообщений в контексте
        :type max_context_length: int
        :param max_tokens: Максимальное количество токенов в контексте
        :type max_tokens: int
        :param context_file: Путь для сохранения контекста
        :type context_file: str
        """
        self.logger = logging.getLogger(__name__)
        self.max_context_length = max_context_length
        self.max_tokens = max_tokens
        
        self.context_file = context_file
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

        :return: Загруженный контекст или пустой контекст при ошибке
        :rtype: dict
        """
        try:
            if os.path.exists(self.context_file):
                with open(self.context_file, 'r', encoding='utf-8') as f:
                    loaded_context = json.load(f)
                    self.logger.info(f"Загружен контекст")
                    
                    # Проверка и восстановление структуры контекста
                    if isinstance(loaded_context, list):
                        # Если контекст - список, преобразуем его
                        self.context['messages'] = loaded_context
                        self.context['task_types'] = {}
                        self.logger.info(f"Контекст список")
                    elif isinstance(loaded_context, dict):
                        # Если контекст - словарь, используем его как есть
                        self.logger.info(f"Контекст словарь")
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
        
        :raises Exception: При ошибке сохранения
        """
        try:
            with open(self.context_file, 'w', encoding='utf-8') as f:
                json.dump(self.context, f, ensure_ascii=False, indent=2)
                self.logger.info("Функция save_context завершена")
        except Exception as e:
            self.logger.error(f"Ошибка сохранения контекста: {e}")

    def add_message(self, message, role='user', task_type=None):
        """
        Добавление сообщения в контекст

        :param message: Текст сообщения
        :type message: str
        :param role: Роль отправителя (user/assistant)
        :type role: str
        :param task_type: Тип задачи для группировки сообщений
        :type task_type: str | None
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
            self.logger.info(f"Добавлено сообщение: {message_entry}")
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
            
            #if len(all_mes) > 140:
            #    all_mes.pop(0)
            # Сохраняем контекст
            self.logger.info(f"Сохранение контекста")
            self.save_context()
            self.logger.info("Сохранение контекста завершено")
            self.logger.info(f"Добавлено сообщение. Роль: {role}, Тип задачи: {task_type}")
            self.logger.info(f"Общее количество сообщений: {len(self.context['messages'])}")
        
        except Exception as e:
            self.logger.error(f"Ошибка при добавлении сообщения: {e}")

    def get_context(self, task_type=None, include_general=True, hours_to_include=24):
        """
        Получение отфильтрованного контекста диалога

        :param task_type: Тип задачи для фильтрации
        :type task_type: str | None
        :param include_general: Включать ли общие сообщения
        :type include_general: bool
        :param hours_to_include: Временной период в часах
        :type hours_to_include: int
        :return: Список отфильтрованных сообщений
        :rtype: list[dict]
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