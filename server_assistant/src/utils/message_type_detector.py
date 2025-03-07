import os
import logging
import magic  # python-magic для определения MIME-типа
import soundfile as sf
import numpy as np

class MessageTypeDetector:
    """
    Определяет тип входящего сообщения (текст или голос).
    Поддерживает проверку файлов и бинарных данных.
    """

    def __init__(self):
        """Инициализация детектора типов сообщений"""
        self.logger = logging.getLogger(__name__)
    
    def detect_message_type(self, message: str):
        """
        Определяет тип входящего сообщения.

        :param message: Входящее сообщение (путь к файлу, текст или бинарные данные)
        :return: 'text' или 'voice'
        """
        # Если передан текст
        if isinstance(message, str) and os.path.exists(message):
            # Определение MIME-типа файла
            mime = magic.Magic(mime=True)
            file_type = mime.from_file(message)
            
            # Проверка аудио-файлов
            if 'audio' in file_type.lower():
                return self._validate_audio_file(message)
            
            return 'text'
        
        # Если передан текст напрямую
        if isinstance(message, str):
            return 'text'
        
        # Для бинарных данных (например, из Telegram)
        if isinstance(message, bytes):
            return self._validate_audio_bytes(message)
        
        return 'text'
    
    def _validate_audio_file(self, file_path):
        """
        Проверяет аудиофайл на соответствие критериям голосового сообщения.

        :param file_path: Путь к аудиофайлу
        :return: 'voice' если файл соответствует критериям, иначе 'text'
        """
        try:
            # Чтение аудио-файла
            audio_data, sample_rate = sf.read(file_path)
            
            # Проверка длительности и громкости
            duration = len(audio_data) / sample_rate
            rms = np.sqrt(np.mean(audio_data**2))
            
            # Критерии для голосового сообщения
            is_voice = (
                duration > 0.5 and  # Минимальная длительность 0.5 сек
                duration < 300 and  # Максимальная длительность 5 минут
                rms > 0.01  # Минимальный уровень громкости
            )
            
            return 'voice' if is_voice else 'text'
        
        except Exception as e:
            self.logger.error(f"Ошибка проверки аудио-файла: {e}")
            return 'text'
    
    def _validate_audio_bytes(self, audio_bytes):
        """
        Проверяет бинарные аудиоданные на соответствие критериям голосового сообщения.

        :param audio_bytes: Бинарные данные аудио
        :return: 'voice' если данные соответствуют критериям, иначе 'text'
        """
        try:
            # Создаем временный файл
            import tempfile
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                temp_file.write(audio_bytes)
                temp_file_path = temp_file.name
            
            # Проверяем тип файла
            result = self._validate_audio_file(temp_file_path)
            
            # Удаляем временный файл
            os.unlink(temp_file_path)
            
            return result
        
        except Exception as e:
            self.logger.error(f"Ошибка проверки аудио-байтов: {e}")
            return 'text'