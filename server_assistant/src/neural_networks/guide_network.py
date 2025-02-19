import logging
from .deepseek_processor import DeepSeekProcessor  # Изменили импорт
from .openai_processor import OpenAIProcessor
from .router_network import RouterNetwork, TaskType
from .small_talk_network import SmallTalkNetwork
from .complex_dialog_network import ComplexDialogNetwork
from .information_network import InformationNetwork

class GuideNetwork:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Инициализация сетей
        self.openai_processor = DeepSeekProcessor()
        self.router_network = RouterNetwork()
        self.small_talk_network = SmallTalkNetwork()
        self.complex_dialog_network = ComplexDialogNetwork()
        self.information_network = InformationNetwork()

    def _route_to_network(self, task_type: TaskType, message: str) -> str:
        """
        Маршрутизация сообщения в соответствующую нейронную сеть
        """
        try:
            if task_type == TaskType.SMALL_TALK:
                return self.small_talk_network.generate_response(message)
            elif task_type == TaskType.COMPLEX_DIALOG:
                return self.complex_dialog_network.generate_response(message)
            elif task_type == TaskType.INFORMATION:
                return self.information_network.generate_response(message)
            else:
                # Fallback для функциональных задач
                return "Извините, я не могу обработать это сообщение."
        
        except Exception as e:
            self.logger.error(f"Ошибка при маршрутизации: {e}")
            return "Произошла ошибка при обработке сообщения."

    def process_message(self, message: str) -> str:
        """
        Основной метод обработки входящего сообщения
        """
        # Определение типа задачи
        task_type = self.router_network.detect_task_type(message)
        
        # Выбор и генерация ответа
        response = self._route_to_network(task_type, message)
        return response