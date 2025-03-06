import logging
import os

import librosa
import numpy as np
import soundfile as sf


def find_ffmpeg_path():
    """
    Найти путь к FFmpeg в системе с расширенной диагностикой

    :return: Путь к исполняемому файлу FFmpeg или None, если не найден
    :rtype: str | None
    """
    import shutil
    
    # Список приоритетных путей
    possible_paths = [
        r'C:\Users\ArdorPC\AppData\Local\Microsoft\WinGet\Links\ffmpeg.exe',
        r'C:\Tools\FFmpeg\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe',
        shutil.which('ffmpeg'),
        r'C:\Program Files\FFmpeg\bin\ffmpeg.exe',
        r'C:\Program Files (x86)\FFmpeg\bin\ffmpeg.exe',
        r'C:\ProgramData\chocolatey\bin\ffmpeg.exe',
        r'C:\FFmpeg\bin\ffmpeg.exe'
    ]
    
    # Добавляем пути из PATH
    path_env = os.environ.get('PATH', '').split(os.pathsep)
    possible_paths.extend([os.path.join(path, 'ffmpeg.exe') for path in path_env])
    
    for path in possible_paths:
        if path and os.path.exists(path):
            return path
    
    return None

# Глобальная переменная для пути FFmpeg
FFMPEG_PATH = find_ffmpeg_path()

class AudioTranscriber:
    """
    Класс для распознавания речи с использованием модели Whisper.
    
    Поддерживает конвертацию различных аудиоформатов в WAV и их транскрибацию.
    Требует установленного FFmpeg для обработки аудио.
    """

    def __init__(self, language: str = 'ru'):
        self.logger = logging.getLogger(__name__)
        
        # Логируем путь к FFmpeg
        self.logger.info(f"Путь к FFmpeg: {FFMPEG_PATH}")
        
        try:
            # Используем базовую модель для экономии ресурсов
            import whisper
            whisper.audio.ffmpeg_path = FFMPEG_PATH  # Явно устанавливаем путь
            self.model = whisper.load_model("base")
            self.language = language
        except Exception as e:
            self.logger.error(f"Ошибка загрузки модели Whisper: {e}")
            raise

    def _convert_audio(self, input_path: str, output_path: str = None) -> str:
        """
        Конвертация аудио в формат WAV для обработки Whisper.
        Включает нормализацию, приведение к моно и ресемплинг до 16кГц.

        :param input_path: Путь к исходному аудиофайлу
        :type input_path: str
        :param output_path: Опциональный путь для сохранения WAV файла
        :type output_path: str | None
        :return: Путь к конвертированному WAV файлу или исходному файлу при ошибке
        :rtype: str
        """
        # Расширенная диагностика входного файла
        self.logger.info(f"Начало конвертации файла: {input_path}")
        
        # Проверка существования входного файла
        if not os.path.exists(input_path):
            self.logger.error(f"Входной файл не существует: {input_path}")
            return input_path
        
        # Проверка размера входного файла
        input_file_size = os.path.getsize(input_path)
        self.logger.info(f"Размер входного файла: {input_file_size} байт")
        
        if input_file_size == 0:
            self.logger.error("Входной файл пустой")
            return input_path
        
        # Генерация пути для выходного файла .wav
        if not output_path:
            base_dir = os.path.dirname(input_path)
            filename = os.path.splitext(os.path.basename(input_path))[0]
            output_path = os.path.join(base_dir, f'{filename}.wav')
        
        try:
            # Чтение исходного аудио
            data, samplerate = sf.read(input_path)
            
            # Логирование параметров аудио
            self.logger.info(f"Частота дискретизации: {samplerate} Гц")
            self.logger.info(f"Количество каналов: {data.ndim}")
            
            # Преобразование в моно, если стерео
            if data.ndim > 1:
                data = data.mean(axis=1)
            
            # Принудительная частота дискретизации 16 кГц
            if samplerate != 16000:
                self.logger.info(f"Пересемплирование с {samplerate} до 16000 Гц")
                data = librosa.resample(data, orig_sr=samplerate, target_sr=16000)
                samplerate = 16000
            
            # Запись в .wav
            sf.write(output_path, data, samplerate)
            
            # Проверка результата
            output_file_size = os.path.getsize(output_path)
            self.logger.info(f"Размер выходного файла: {output_file_size} байт")
            
            if output_file_size == 0:
                self.logger.error("Выходной файл пустой")
                return input_path
            
            return output_path
        
        except Exception as e:
            self.logger.error(f"Критическая ошибка конвертации: {e}", exc_info=True)
            return input_path

    def transcribe_audio(self, audio_path: str) -> str:
        """
        Распознавание речи из аудиофайла с использованием Whisper.

        :param audio_path: Путь к аудиофайлу для распознавания
        :type audio_path: str
        :return: Распознанный текст или пустая строка при ошибке
        :rtype: str
        """
        try:
            # Расширенная диагностика входного файла
            self.logger.info(f"Начало распознавания файла: {audio_path}")
            
            # Проверка существования файла
            if not os.path.exists(audio_path):
                self.logger.error(f"Файл не существует: {audio_path}")
                return ""
            
            # Проверка размера файла
            file_size = os.path.getsize(audio_path)
            self.logger.info(f"Размер файла: {file_size} байт")
            
            if file_size == 0:
                self.logger.error("Файл пустой")
                return ""
            
            # Конвертация файла, если необходимо
            if not audio_path.lower().endswith('.wav'):
                self.logger.info("Конвертация файла в формат .wav")
                audio_path = self._convert_audio(audio_path)
            
            # Прямая загрузка аудио с использованием soundfile
            self.logger.info("Конвертация завершена")
            audio_data, sample_rate = sf.read(audio_path)
            
            # Логирование параметров аудио
            self.logger.info(f"Частота дискретизации: {sample_rate} Гц")
            self.logger.info(f"Количество каналов: {audio_data.ndim}")
            
            # Преобразование в моно, если стерео
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)
            
            # Принудительная нормализация к диапазону [-1, 1]
            audio_data = audio_data.astype(np.float32)
            audio_data /= np.max(np.abs(audio_data))
            
            # Распознавание речи
            result = self.model.transcribe(
                audio_data, 
                language='ru',  # Указываем русский язык
                fp16=False      # Отключаем float16 для совместимости
            )
            
            # Извлечение и логирование результата
            transcribed_text = result['text'].strip()
            self.logger.info(f"Распознанный текст: {transcribed_text}")
            
            return transcribed_text
        
        except Exception as e:
            self.logger.error(f"Критическая ошибка распознавания речи: {e}", exc_info=True)
            return ""