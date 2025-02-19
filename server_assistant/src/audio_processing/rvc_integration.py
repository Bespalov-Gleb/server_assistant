import os
import logging
import tempfile
from dotenv import load_dotenv
import requests
import soundfile as sf

load_dotenv()

class YandexSpeechConverter:
    def __init__(self):
        self.oauth_token = os.getenv('OAUTH')
        self.folder_id = os.getenv('YANDEX_FOLDER_ID', 'b1g3k9jqrid8********')
        self.logger = logging.getLogger(__name__)
        
        # Настройки голоса
        self.voice_settings = {
            'voice': 'anton',  # Женский голос
            'emotion': 'good',  # Эмоциональная окраска
            'speed': 1.0,  # Скорость речи
            'format': 'mp3'  # Формат аудио
        }

    def get_iam_token(self):
        """
        Получение IAM-токена для авторизации
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

    def generate_audio(self, text, output_path=None, language='ru-RU'):
        """
        Синтез речи с использованием Яндекс.Speechkit
        """
        # Генерация временного пути, если не указан
        if not output_path:
            output_path = tempfile.mktemp(suffix='.mp3')

        # Получение IAM-токена
        iam_token = self.get_iam_token()
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