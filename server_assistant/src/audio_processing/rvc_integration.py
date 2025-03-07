import logging
import tempfile

import requests
from dotenv import load_dotenv

from config import get_config
from src.audio_processing.base.tts_model import TTSModel
from src.audio_processing.base.tts_parameters import Parameters

load_dotenv()


class YandexSpeechConverter(TTSModel):
    def __init__(self):
        super().__init__()
        yandex_config = get_config().neural_networks.yspeechkit # config

        self.oauth_token = yandex_config.oauth_token
        self.folder_id = yandex_config.oauth_token
        self.logger = logging.getLogger(__name__)
        
        #Стандартные настройки голоса
        self.voice_settings = {
            'voice': 'anton',  # Женский голос
            'emotion': 'good',  # Эмоциональная окраска
            'speed': 1.0,  # Скорость речи
            'format': 'mp3'  # Формат аудио
        }

    def text_to_speech(self, text: str, params: Parameters | None = None, output_file: str | None = None) -> str:
        """
        Реализация абстрактного метода для преобразования текста в речь.

        :param text: Текст для синтеза
        :type text: str
        :param params: Параметры для синтеза
        :type params: Parameters
        :param output_file: Путь для сохранения файла
        :type output_file: Optional[str]
        :return: Путь к сгенерированному аудиофайлу или пустая строка при ошибке
        :rtype: str
        """
        try:
            # Применяем параметры к настройкам голоса
            if params:
                if params.voice:
                    self.voice_settings['voice'] = params.voice
                if params.emotion:
                    self.voice_settings['emotion'] = params.emotion
                if params.speed:
                    self.voice_settings['speed'] = params.speed
                if params.format:
                    self.voice_settings['format'] = params.format
            
            # Генерируем аудио через внутренний метод
            result = self._generate_audio(
                text, 
                output_file,
                language=params.language or 'ru-RU'
            )
            return result if result else ""
            
        except Exception as e:
            self.logger.error(f"Ошибка в text_to_speech: {e}")
            return ""

    def _get_iam_token(self):
        """
        Получение IAM-токена для авторизации в Yandex Cloud.

        :return: IAM-токен в случае успеха, None при ошибке
        :rtype: str | None
        """
        try:
            response = requests.post(
                'https://iam.api.cloud.yandex.net/iam/v1/tokens',
                json={'yandexPassportOauthToken': self.oauth_token}
            )
            return response.json()['iamToken']
        except Exception as e:
            self.logger.error(f"Ошибка получения IAM-токена: {e}")
            return None

    def _generate_audio(self, text, output_path=None, language='ru-RU'):
        """
        Внутренний метод для синтеза речи через Яндекс.Speechkit.
        """
        # Генерация временного пути, если не указан
        if not output_path:
            output_path = tempfile.mktemp(suffix='.mp3')

        # Получение IAM-токена
        iam_token = self._get_iam_token()
        if not iam_token:
            self.logger.error("Не удалось получить IAM-токен")
            return None

        # Параметры запроса
        url = 'https://tts.api.cloud.yandex.net/speech/v1/tts:synthesize'
        headers = {
            'Authorization': f'Bearer {iam_token}',
            'Content-Type': 'application/x-www-form-urlencoded',
        }
        data = {
            'text': text,
            'voice': self.voice_settings['voice'],
            'emotion': self.voice_settings['emotion'],
            'speed': str(self.voice_settings['speed']),
            'format': self.voice_settings['format'],
            'folderId': self.folder_id,
            'lang': language
        }

        try:
            # Отправка запроса
            response = requests.post(url, headers=headers, data=data)
            
            if response.status_code == 200:
                # Сохранение аудио
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                
                return output_path
            else:
                self.logger.error(f"Ошибка синтеза: {response.text}")
                return None

        except Exception as e:
            self.logger.error(f"Ошибка генерации аудио: {e}")
            return None

# Обратная совместимость
RVCVoiceConverter = YandexSpeechConverter