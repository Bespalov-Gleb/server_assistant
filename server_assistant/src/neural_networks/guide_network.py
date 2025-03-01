import logging
import asyncio
from enum import Enum
from .router_network import RouterNetwork, TaskType
from .small_talk_network import SmallTalkNetwork
from .complex_dialog_network import ComplexDialogNetwork
from .information_network import InformationNetwork
from .reminder_network import ReminderNetwork
from .functional_network import FunctionalNetwork

class GuideNetwork:
    def __init__(self, bot, user_id):
        self.logger = logging.getLogger(__name__)        
        # Инициализация сетей
        self.functional_network = FunctionalNetwork(user_id=user_id)
        self.router_network = RouterNetwork(user_id=user_id)
        self.small_talk_network = SmallTalkNetwork(user_id=user_id)
        self.complex_dialog_network = ComplexDialogNetwork(user_id=user_id)
        self.information_network = InformationNetwork(user_id=user_id)
        self.reminder_network = ReminderNetwork(bot=bot, user_id=user_id)

    async def _route_to_network(self, task_type: TaskType, message: str) -> str:
        """
        Маршрутизация сообщения в соответствующую нейронную сеть
        """
        try:
            if task_type == TaskType.SMALL_TALK:
                return self.small_talk_network.generate_response(message)
            elif task_type == TaskType.COMPLEX_DIALOG:
                return self.complex_dialog_network.generate_response(message)
            elif task_type == TaskType.INFORMATION:
                self.logger.info("Приступил к генерации информационного ответа")
                return self.information_network.generate_response(message)
            elif task_type == TaskType.FUNCTIONAL:
                return self.functional_network.generate_response(message)
            elif task_type == TaskType.REMINDER:
                return await self.reminder_network.create_reminder(message)
            else:
                # Fallback для функциональных задач
                return "Извините, я не могу обработать это сообщение."
        
        except Exception as e:
            self.logger.error(f"Ошибка при маршрутизации: {e}")
            return "Произошла ошибка при обработке сообщения."

    async def process_message(self, message: str) -> str:
        """
        Основной метод обработки входящего сообщения
        """
        # Определение типа задачи
        task_type = self.router_network.detect_task_type(message)
        
        
        # Выбор и генерация ответа
        response = await self._route_to_network(task_type, message)
        return response