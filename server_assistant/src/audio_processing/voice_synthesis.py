import logging
import os

import numpy as np
import soundfile as sf
import torch
import torchaudio


class VoiceSynthesizer:
    """
    Класс для синтеза речи с использованием модели SileroTTS.
    
    Поддерживает несколько голосов и форматов аудио вывода.
    """

    def __init__(self, language: str = 'ru'):
        """
        Инициализация синтезатора речи с SileroTTS
        
        :param language: Язык синтеза речи
        :type language: str
        :raises Exception: При ошибке загрузки модели
        """
        self.logger = logging.getLogger(__name__)
        self.language = language
        
        try:
            # Загрузка модели Silero
            torch.set_num_threads(4)  # Оптимизация для CPU
            self.device = torch.device('cpu')
            
            # Альтернативный метод загрузки
            model_url = 'https://models.silero.ai/models/tts/ru/v3_1_ru.pt'
            model_path = os.path.join(os.path.expanduser('~'), '.cache', 'torch', 'silero_tts_model.pt')
            
            # Создаем директорию, если не существует
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Загружаем модель, если она еще не скачана
            if not os.path.exists(model_path):
                torch.hub.download_url_to_file(model_url, model_path)
            
            # Загрузка модели
            self.model = torch.package.PackageImporter(model_path).load_pickle("tts_models", "model")
            
            # Список доступных дикторов
            self.speakers = {
                'xenia': 'xenia',  # Женский голос
                'eugene': 'eugene',  # Мужской голос
                'aidar': 'aidar'  # Альтернативный мужской голос
            }
            
            # Выбираем женский голос по умолчанию
            self.speaker = self.speakers['xenia']
            
        except Exception as e:
            self.logger.error(f"Ошибка загрузки модели Silero: {e}")
            raise

    def save_audio_file(self, audio_data: np.ndarray, output_path: str, sample_rate: int = 16000):
        """
        Сохранение аудио с автоматическим определением формата
        
        :param audio_data: Numpy массив с аудиоданными
        :type audio_data: np.ndarray
        :param output_path: Путь для сохранения файла
        :type output_path: str
        :param sample_rate: Частота дискретизации
        :type sample_rate: int
        :return: Путь к сохраненному файлу или None при ошибке
        :rtype: str | None
        """
        try:
            # Определение формата по расширению
            file_ext = os.path.splitext(output_path)[1].lower()
            
            # Список поддерживаемых форматов
            supported_formats = {
                '.wav': 'wav',
                '.ogg': 'ogg',
                '.oga': 'oga',
                '.mp3': 'mp3'
            }
            
            # Автоматическое определение формата, если не указан
            if file_ext not in supported_formats:
                # По умолчанию используем .wav
                output_path = os.path.splitext(output_path)[0] + '.wav'
                file_ext = '.wav'
            
            # Нормализация аудио
            audio_data = audio_data.astype(np.float32)
            audio_data /= np.max(np.abs(audio_data))
            
            # Преобразование в torch тензор
            audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)
            
            # Сохранение с учетом формата
            if file_ext in ['.oga', '.ogg']:
                torchaudio.save(
                    output_path, 
                    audio_tensor, 
                    sample_rate, 
                    format='oga'
                )
            else:
                sf.write(output_path, audio_data, sample_rate)
            
            self.logger.info(f"Аудио сохранено: {output_path}")
            return output_path
        
        except Exception as e:
            self.logger.error(f"Ошибка сохранения аудио: {e}", exc_info=True)
            return None

    def text_to_speech(self, text: str, output_file: str = 'response.wav') -> str:
        """
        Преобразование текста в речь с помощью SileroTTS
        
        :param text: Текст для синтеза речи
        :type text: str
        :param output_file: Путь для сохранения файла
        :type output_file: str
        :return: Путь к сгенерированному аудиофайлу или пустая строка при ошибке
        :rtype: str
        :raises Exception: При критических ошибках синтеза речи
        """
        try:
            # Проверка текста
            # Отладочная информация
            self.logger.info(f"Входной текст: {text}")
            self.logger.info(f"Путь файла: {output_file}")
            if not text or len(text.strip()) == 0:
                self.logger.warning("Пустой текст для синтеза речи")
                return ""
            
            # Генерация пути для файлов
            if not output_file:
                base_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'temp')
                output_file = os.path.join(base_dir, f'tts_response_{int(time.time())}.wav')
            else:
                # Если указан относительный путь, делаем его абсолютным
                output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'temp', output_file))
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Путь для .wav файла
            wav_output_path = os.path.splitext(output_file)[0] + '.wav'
            oga_output_path = os.path.splitext(output_file)[0] + '.oga'
            
            # Генерация речи
            audio = self.model.apply_tts(
                text=text,
                speaker=self.speaker,
                sample_rate=24000
            )
            
            # Сохранение в .wav
            full_output_path = wav_output_path
            
            # Нормализация аудио
            audio_data = audio.numpy().astype(np.float32)
            audio_data /= np.max(np.abs(audio_data))
            
            # Сохранение .wav
            sf.write(full_output_path, audio_data, 24000)
            
            # Конвертация в .oga через soundfile
            try:
                # Чтение исходного .wav файла
                audio_data, sample_rate = sf.read(full_output_path)
                
                # Сохранение в .oga с указанием формата
                sf.write(
                    oga_output_path, 
                    audio_data, 
                    sample_rate, 
                    format='ogg'  # Используем 'ogg' вместо 'oga'
                )
                
                # Логирование успешной конвертации
                self.logger.info(f"Аудио сконвертировано в .oga: {oga_output_path}")
            
            except Exception as e:
                self.logger.error(f"Ошибка конвертации в .oga через soundfile: {e}")
                return full_output_path
            
            # Удаление промежуточного .wav файла
            try:
                os.remove(full_output_path)
            except Exception as e:
                self.logger.warning(f"Не удалось удалить временный .wav файл: {e}")
            
            # Логирование успешной генерации
            self.logger.info(f"Голосовой ответ сгенерирован: {oga_output_path}")
            
            return oga_output_path
        
        except Exception as e:
            self.logger.error(f"Ошибка синтеза речи: {e}", exc_info=True)
            return ""