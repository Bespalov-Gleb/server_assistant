import os
import json
import logging
from re import I, T
from typing import Dict, List, Optional
from .deepseek_processor import DeepSeekProcessor  # Изменили импорт
from .openai_processor import OpenAIProcessor
from aiogram import types

class MemoryNetwork:
    def __init__(self, chat_id: int):
        self.logger = logging.getLogger(__name__)
        self.memory_file = os.path.join('temp', f'memories_{chat_id}.json')
        # Создаем директорию, если не существует
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
        self.openai_processor = OpenAIProcessor(chat_id=chat_id)
        self.chat_id = chat_id
        self.memory = {
            'chat_id': self.chat_id,
            'text': ''
        }

    async def extract_memory_details(self, message: types.Message, transcribe=None):
        """
        Извлечение деталей памяти с помощью OpenAI
        
        :param message: Сообщение пользователя
        :return: Извлеченный текст заметки или None
        """
        system_message = """
        Системное сообщение:
        Тебе дается текст сообщения пользователя.
        Сообщение содержит в себе просьбу запомнить какую-либо информацию.
        Твоя задача - извлечь информацию, которую необходимо запомнить.
        Например: сообщение пользователя "Запомни, мне понравилось вино Кагор".
        Твой ответ: Понравилось вино Кагор.
        В ответе верни только текст напоминания, без лишних вводных.
        Запрос пользователя:
        """
        self.logger.info("Функция extract_memory_details запущена")
        if transcribe == None:
            text = message.from_user.username + ': ' + message.text
        else:
            text = message.from_user.username + ': ' + transcribe
        try:
            response = self.openai_processor.process_with_retry(
                prompt=system_message + '\n' + text, 
                temperature=0.5,
                max_tokens=2000, 
            )
            return response.strip() if response else None
        except Exception as e:
            self.logger.error("Не удалось извлечь информацию заметки.")
            return None
        
        

    async def search_memories(self, search_type: str, message: types.Message, transcribe=None):
        """
        Извлечение деталей памяти для поиска с помощью OpenAI
        
        :param message: Сообщение пользователя
        :return: Извлеченный текст заметки или None
        """
        if search_type == "SEARCH":
            self.logger.info("Активирована ветка SEARCH метода search_memories")
            system_message = """
            Системное сообщение:
            Тебе дается текст сообщения пользователя.
            В виде контекста тебе также подается список сохраненных заметок.
            Твоя задача - найти в списке заметку, которая подходит под запрос пользователя.
            В ответе верни только текст подходящей заметки, без лишних вводных.
            Например: "Бот, вспомни, какой шоколад мне понравился"
            В списке ты находишь заметку "Понравился шоколад Alpen Gold"
            Твой ответ: "Понравился шоколад Alpen Gold
            Запрос пользователя:
            """
            if transcribe == None:
                text = message.from_user.username + ': ' + message.text
            else:
                text = message.from_user.username + ': ' + transcribe
            memories = [
            {"role": "system", "content": "Ты помощник, который умеет извлекать информацию из памяти."},
            {"role": "system", "content": str(self._load_memories())},
            {"role": "user", "content": text}
            ]   
        if search_type == "DELETE":

            self.logger.info("Активирована ветка DELETE метода search_memories")
            system_message = """
            Системное сообщение:
            Ты профессиональный ассистент для анализа списка заметок.
            Тебе предоставляется список заметок и сообщение пользователя.
            На основе содержания сообщения пользователя ты должен вернуть заметку, которую пользователь просит удалить.
            Хорошо проанализируй список заметок и выбери наиболее подходящую.
            Выбранную для удаления заметку верни в формате json по шаблону:
            {
            "chat_id": <id пользователя>,
            "text": "<текст заметки>"
            }
            В ответе верни заметку в точности, как она записана в списке заметок!
            Запрос пользователя:
            """
            if transcribe == None:
                text = message.from_user.username + ': ' + message.text
            else:
                text = message.from_user.username + ': ' + transcribe
            memories = [
            {"role": "system", "content": "Ты помощник, который профессионально сопоставляет заметку из запроса пользователя с заметкой из списка."},
            {"role": "system", "content": str(self._load_memories())},
            {"role": "user", "content": text}
            ]   

        if search_type == "CHANGE":
            system_message = """
            Системное сообщение:
            Ты профессиональный редактор заметок.
            Тебе дается текст сообщения пользователя.
            В виде контекста тебе также подается список сохраненных заметок.
            Твоя задача - найти в списке заметку, которая подходит под запрос пользователя.
            Далее тебе нужно выполнить запрос, касаемо текста этой заметки.
            Если пользователь говорит что-то дописать, удалить какую-то часть из заметки, 
            ты должен опираться и на запрос пользователя, и на существующий текст заметки.
            В ответе тебе нужно вернуть изначальную заметку и обновленную заметку в формате json по шаблону:
            [
            {
            "chat_id": <id пользователя>,
            "text": "<текст заметки>"
            }
            {
            "chat_id": <id пользователя>,
            "text": "<обновленный текст заметки>"
            }
            ]
            Запрос пользователя:
            """
            if transcribe == None:
                text = message.from_user.username + ': ' + message.text
            else:
                text = message.from_user.username + ': ' + transcribe
            memories = [
            {"role": "system", "content": "Ты помощник, который профессионально редактирует заметки пользователя"},
            {"role": "system", "content": str(self._load_memories())},
            {"role": "user", "content": text}
            ]   

        try:
            response = self.openai_processor.process_with_retry(
                prompt=system_message + '\n' + text, 
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

    async def add_memory(self, message: types.Message, transcribe=None) -> str:
        """
        Добавление новой заметки
        
        :param message: Сообщение пользователя
        :return: Результат добавления заметки
        """
        # Извлечение текста заметки
        self.logger.info("Функция add_memory запущена")
        memory_text = await self.extract_memory_details(message, transcribe)
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
            'chat_id': self.chat_id,
            'text': memory_text
        }
        
        memories.append(new_memory)
        self.logger.info(f"Сохранение новой заметки: {new_memory}")
        self._save_memories(memories)
        self.logger.info(f"Заметка успешно добавлена")
        
        return f"Запомнил: {memory_text}"

    async def recall_memory(self, message: types.Message, transcribe=None) -> str:
        """
        Поиск заметок по запросу
        
        :param message: Запрос пользователя
        :return: Найденные заметки
        """
        response = await self.search_memories(search_type="SEARCH", message=message, transcribe=transcribe)
        if response:
            return f"Заметка: {response}"
        else:
            return "Не нашел заметок по вашему запросу."

    async def delete_memory(self, message: types.Message, transcribe=None) -> str:
        """Удаление памяти"""
        try:
            response = await self.search_memories(search_type="DELETE", message=message, transcribe=transcribe)
            self.logger.info(f"Заметка на удаление: {response}")
            if response:
                # Преобразуем JSON-строку в объект
                memory_to_delete = json.loads(response)  # Предполагаем, что message содержит JSON
                chat_id = memory_to_delete['chat_id']
                text_to_delete = memory_to_delete['text']
                
                # Загружаем существующие заметки
                memories = self._load_memories()
                
                # Поиск и удаление заметки
                initial_count = len(memories)
                memories = [mem for mem in memories if not (mem['chat_id'] == chat_id and mem['text'] == text_to_delete)]
                
                if len(memories) < initial_count:
                    self._save_memories(memories)
                    return f"Удалил заметку: {text_to_delete}"
                
                return "Не нашел заметок для удаления."
        
        except Exception as e:
            self.logger.error(f"Ошибка удаления памяти: {e}")
            return "Не удалось удалить заметку."
    
    async def change_memory(self, message: types.Message, transcribe=None) -> str:
        """Изменения заметок"""
        try:
            # Получаем обновленную заметку через search_memories
            response = await self.search_memories(search_type="CHANGE", message=message, transcribe=transcribe)
            
            # Преобразуем ответ в JSON
            updated_memory = json.loads(response)
            
            # Извлекаем chat_id и текст обновленной заметки
            chat_id = updated_memory[1]['chat_id']  # Обновленная заметка
            updated_text = updated_memory[1]['text']
            
            # Загружаем существующие заметки
            memories = self._load_memories()
            
            # Обновляем заметку
            for mem in memories:
                if mem['chat_id'] == chat_id and mem['text'] == updated_memory[0]['text']:
                    mem['text'] = updated_text
                    break
            
            # Сохраняем обновленный список заметок
            self._save_memories(memories)
            
            return f"Заметка обновлена: {updated_text}"
        except Exception as e:
            self.logger.error(f"Ошибка изменения памяти: {e}")
            return "Не удалось изменить заметку."

    def delete_all(self):
        """Полное очищение списка заметок"""
        try:
            if os.path.exists(self.memory_file):
                os.remove(self.memory_file)  # Удаляем файл заметок
                return "Все заметки были удалены."
            else:
                return "Файл заметок не найден."
        except Exception as e:
            self.logger.error(f"Ошибка при удалении всех заметок: {e}")
            return "Произошла ошибка при удалении всех заметок."


    def get_all_notes(self) -> str:
        """Возвращает текст всех заметок, каждая заметка на новой строке"""
        try:
            memories = self._load_memories()  # Загружаем все заметки
            if not memories:
                return "Нет заметок для отображения."
            
            # Формируем строку с текстами заметок
            notes_text = "\n".join(mem['text'] for mem in memories)
            return notes_text
        except Exception as e:
            self.logger.error(f"Ошибка при получении всех заметок: {e}")
            return "Произошла ошибка при получении заметок."