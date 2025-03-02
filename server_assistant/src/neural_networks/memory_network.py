import os
import json
import logging
from typing import Dict, List, Optional
from .deepseek_processor import DeepSeekProcessor  # Изменили импорт
from .openai_processor import OpenAIProcessor

class MemoryNetwork:
    def __init__(self, user_id: int):
        self.logger = logging.getLogger(__name__)
        self.memory_file = os.path.join('temp', f'memories_{user_id}.json')
        # Создаем директорию, если не существует
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
        self.openai_processor = OpenAIProcessor()
        self.user_id = user_id
        self.memory = {
            'user_id': self.user_id,
            'text': ''
        }

    async def extract_memory_details(self, message: str):
        """
        Извлечение деталей памяти с помощью OpenAI
        
        :param message: Сообщение пользователя
        :return: Извлеченный текст заметки или None
        """
        system_message = """
        Тебе дается текст сообщения пользователя.
        Сообщение содержит в себе просьбу запомнить какую-либо информацию.
        Твоя задача - извлечь информацию, которую необходимо запомнить.
        Например: сообщение пользователя "Запомни, мне понравилось вино Кагор".
        Твой ответ: Понравилось вино Кагор.
        В ответе верни только текст напоминания, без лишних вводных.
        """
        self.logger.info("Функция extract_memory_details запущена")
        try:
            response = self.openai_processor.process_with_retry(
                prompt=system_message + '\n' + message, 
                temperature=0.5,
                max_tokens=2000, 
                use_context=True
            )
            return response.strip() if response else None
        except Exception as e:
            self.logger.error("Не удалось извлечь информацию заметки.")
            return None
        
        

    async def search_memories(self, message: str):
        """
        Извлечение деталей памяти для поиска с помощью OpenAI
        
        :param message: Сообщение пользователя
        :return: Извлеченный текст заметки или None
        """
        system_message = """
        Тебе дается текст сообщения пользователя.
        В виде контекста тебе также подается список сохраненных заметок.
        Твоя задача - найти в списке заметку, которая подходит под запрос пользователя.
        В ответе верни только текст подходящей заметки, без лишних вводных.
        Например: "Бот, вспомни, какой шоколад мне понравился"
        В списке ты находишь заметку "Понравился шоколад Alpen Gold"
        Твой ответ: "Понравился шоколад Alpen Gold
        """
        memories = [
        {"role": "system", "content": "Ты помощник, который умеет извлекать информацию из памяти."},
        {"role": "system", "content": str(self._load_memories())},
        {"role": "user", "content": message}
        ]   
        
        try:
            response = self.openai_processor.process_with_retry(
                prompt=system_message + '\n' + message, 
                temperature=0.5,
                max_tokens=2000, 
                use_context="MEM",
                context_file=memories
            )
            return response.strip() if response else None
        except Exception as e:
            self.logger.error("Не удалось найти подходящую заметку")
            return None
        
        return response
            
    def _load_memories(self) -> List[Dict[str, str]]:
        """
        Загрузка заметок из файла
        
        :return: Список заметок
        """
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return []
        except Exception as e:
            self.logger.error(f"Ошибка загрузки памяти: {e}")
            return []


    def _save_memories(self, memories: List[Dict[str, str]]):
        """
        Сохранение заметок в файл
        
        :param memories: Список заметок для сохранения
        """
        try:
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(memories, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"Ошибка сохранения памяти: {e}")

    async def add_memory(self, message: str) -> str:
        """
        Добавление новой заметки
        
        :param message: Сообщение пользователя
        :return: Результат добавления заметки
        """
        # Извлечение текста заметки
        self.logger.info("Функция add_memory запущена")
        memory_text = await self.extract_memory_details(message)
        self.logger.info(f"Извлечен текст заметки: {memory_text}")
        if not memory_text:
            return "Не удалось извлечь информацию для заметки."
        
        # Загрузка существующих заметок
        self.logger.info("Загрузка существующих заметок")
        try:
            memories = self._load_memories()
            self.logger.info(f"Существующие заметки извлечены")
            if any(mem['text'] == memory_text for mem in memories):
                return "Такая заметка уже существует."
        except Exception as e:
            self.logger.error(f"Ошибка загрузки памяти: {e}")
        # Создание новой заметки
        new_memory = {
            'user_id': self.user_id,
            'text': memory_text
        }
        
        memories.append(new_memory)
        self.logger.info(f"Сохранение новой заметки: {new_memory}")
        self._save_memories(memories)
        self.logger.info(f"Заметка успешно добавлена")
        
        return f"Запомнил: {memory_text}"

    async def recall_memory(self, message: str) -> str:
        """
        Поиск заметок по запросу
        
        :param message: Запрос пользователя
        :return: Найденные заметки
        """
        response = await self.search_memories(message)
        if response:
            return f"Заметка: {response}"
        else:
            return "Не нашел заметок по вашему запросу."

    async def delete_memory(self, message: str) -> str:
        """Удаление памяти"""
        try:
            # Извлечение текста для удаления после "Удали" или "удали"
            if not message:
                return "Укажите, что именно удалить."
            
            # Поиск и удаление
            initial_count = len(self.memories)
            self.memories = [mem for mem in self.memories if message.lower() not in mem.lower()]
            
            if len(self.memories) < initial_count:
                self.save_memories()
                return f"Удалил заметки, содержащие: {message}"
            
            return "Не нашел заметок для удаления."
        
        except Exception as e:
            self.logger.error(f"Ошибка удаления памяти: {e}")
            return "Не удалось удалить заметку."

    def get_memory(self):
        """
        Получение контекста с возможностью фильтрации
        """
        try:
            # Базовый список сообщений
            memories = self._load_memories()
            memory = [
                {'role': 'system', 'content': 'Это список сохраненных заметок:'},
                *[{'role': 'user', 'content': mem['text']} for mem in memories]
            ]
            return memory
        
        except Exception as e:
            self.logger.error(f"Ошибка при получении заметок: {e}")
            return []