from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

class LLMProcessor(ABC):
    """
    Абстрактный базовый класс для Language Model процессоров
    
    Определяет общий интерфейс для работы с различными языковыми моделями.
    Обеспечивает единообразие взаимодействия с различными LLM.
    """

    @abstractmethod
    def process_with_retry(
        self, 
        prompt: str, 
        system_message: str = '', 
        model: str = 'default',
        max_tokens: int = 2000, 
        temperature: float = 0.7
    ) -> Optional[str]:
        """
        Абстрактный метод обработки запроса к LLM
        
        :param prompt: Пользовательский запрос
        :param system_message: Системное сообщение для контекста
        :param model: Название модели
        :param max_tokens: Максимальная длина ответа
        :param temperature: Креативность ответа
        :return: Сгенерированный текст или None при ошибке
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Возвращает информацию о текущей модели
        
        :return: Словарь с информацией о модели
        """
        pass

    @abstractmethod
    def validate_api_key(self) -> bool:
        """
        Проверяет валидность API-ключа
        
        :return: True, если ключ валиден, иначе False
        """
        pass

    def __str__(self) -> str:
        """
        Строковое представление процессора
        
        :return: Название класса процессора
        """
        return self.__class__.__name__