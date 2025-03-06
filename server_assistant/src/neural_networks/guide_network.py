import logging
from typing import Tuple
import asyncio
from aiogram import types
from enum import Enum

from src.neural_networks.complex_dialog_network import ComplexDialogNetwork
from src.neural_networks.functional_network import FunctionalNetwork
from src.neural_networks.information_network import InformationNetwork
from src.neural_networks.memory_network import MemoryNetwork
from src.neural_networks.reminder_network import ReminderNetwork
from src.neural_networks.router_network import RouterNetwork, TaskType, OutputType
from src.neural_networks.small_talk_network import SmallTalkNetwork


class GuideNetwork:
    """
    Центральный маршрутизатор для обработки сообщений.
    
    Управляет распределением запросов между специализированными нейронными сетями
    на основе типа задачи.
    """

    def __init__(self, bot, chat_id):
        """
        :param bot: Экземпляр бота для отправки сообщений
        :param chat_id: Идентификатор чата
        :type chat_id: int
        """
        self.logger = logging.getLogger(__name__)

        # Инициализация сетей
        self.functional_network = FunctionalNetwork(chat_id=chat_id)
        self.router_network = RouterNetwork(chat_id=chat_id)
        self.small_talk_network = SmallTalkNetwork(chat_id=chat_id)
        self.complex_dialog_network = ComplexDialogNetwork(chat_id=chat_id)
        self.information_network = InformationNetwork(chat_id=chat_id)
        self.reminder_network = ReminderNetwork(bot=bot, chat_id=chat_id)
        self.memory_network = MemoryNetwork(chat_id=chat_id)

    async def _route_to_network(self, task_type: TaskType, message: types.Message, transcribe=None) -> str:
        """
        Маршрутизация сообщения в соответствующую нейронную сеть
        
        :param task_type: Тип задачи для обработки
        :type task_type: TaskType
        :param message: Текст сообщения
        :type message: str
        :return: Ответ от соответствующей сети
        :rtype: str
        """
        try:
            if task_type == TaskType.SMALL_TALK:
                return self.small_talk_network.generate_response(message, transcribe=transcribe)
            elif task_type == TaskType.COMPLEX_DIALOG:
                return self.complex_dialog_network.generate_response(message, transcribe=transcribe)
            elif task_type == TaskType.INFORMATION:
                self.logger.info("Приступил к генерации информационного ответа")
                return self.information_network.generate_response(message, transcribe=transcribe)
            elif task_type == TaskType.FUNCTIONAL:
                return self.functional_network.generate_response(message, transcribe=transcribe)
            elif task_type == TaskType.REMINDER:
                return await self.reminder_network.create_reminder(message, transcribe=transcribe)
            elif task_type == TaskType.RECALL_MEMORY:
                return await self.memory_network.recall_memory(message, transcribe=transcribe)
            elif task_type == TaskType.ADD_MEMORY:
                self.logger.info("Приступил к добавлению заметки")
                return await self.memory_network.add_memory(message, transcribe=transcribe)
            elif task_type == TaskType.DELETE_MEMORY:
                return await self.memory_network.delete_memory(message, transcribe=transcribe)
            elif task_type == TaskType.DELETE_ALL_MEMORIES:
                return self.memory_network.delete_all()
            elif task_type == TaskType.CHANGE_MEMORY:
                return await self.memory_network.change_memory(message, transcribe=transcribe)
            elif task_type == TaskType.VIEW_MEMORIES:
                return self.memory_network.get_all_notes()
            # Fallback для функциональных задач
            return "Извините, я не могу обработать это сообщение."
        
        except Exception as e:
            self.logger.error(f"Ошибка при маршрутизации: {e}")
            return "Произошла ошибка при обработке сообщения."

          
    async def process_message(self, message: types.Message, transcribe=None) -> str:
        """
        Основной метод обработки входящего сообщения
        """
        # Определение типа задачи
        if transcribe == None:
            text = message.text
        else:
            text = transcribe
        task_type = self.router_network.detect_task_type(text)
        output_type = self.router_network.detect_output_type(text)
        
        
        # Выбор и генерация ответа
        response = await self._route_to_network(task_type, message, transcribe)
        self.logger.info(f"Получено от нейросети: {response}")
        return response, output_type
