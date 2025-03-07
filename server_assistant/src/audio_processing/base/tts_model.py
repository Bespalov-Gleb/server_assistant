from abc import ABC, abstractmethod
import logging
from typing import Optional

from src.audio_processing.base.tts_parameters import Parameters


class TTSModel(ABC):
    """
    Абстрактный базовый класс для моделей синтеза речи.
    """

    def __init__(self):
        """
        Инициализация базового класса синтезатора речи.
        """
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def text_to_speech(self, text: str, params: Parameters, output_file: Optional[str] = None) -> str:
        """
        Преобразование текста в речь.

        :param text: Текст для синтеза
        :type text: str
        :param params: Параметры синтеза речи
        :type params: Parameters
        :param output_file: Путь для сохранения файла
        :type output_file: Optional[str]
        :return: Путь к сгенерированному аудиофайлу или пустая строка при ошибке
        :rtype: str
        """
        pass
