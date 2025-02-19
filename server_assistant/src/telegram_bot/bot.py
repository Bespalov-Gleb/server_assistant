import os
import logging
from dotenv import load_dotenv
import telebot
from telebot.types import Message, ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove
import sys

# Добавляем путь к корневой директории проекта
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Импорты компонентов
from src.neural_networks.guide_network import GuideNetwork
from src.neural_networks.deepseek_processor import DeepSeekProcessor
from src.neural_networks.openai_processor import OpenAIProcessor
from src.audio_processing.speech_recognition import AudioTranscriber
from src.audio_processing.voice_synthesis import VoiceSynthesizer
from src.utils.user_preferences import UserPreferences

# Загрузка переменных окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TelegramAssistantBot:
    def __init__(self):
        # Инициализация токена и бота
        self.token = os.getenv('TELEGRAM_BOT_TOKEN')
        if not self.token:
            raise ValueError("Telegram Bot Token не найден в переменных окружения")
        
        self.bot = telebot.TeleBot(self.token)
        
        # Инициализация компонентов
        self.guide_network = GuideNetwork()
        self.audio_transcriber = AudioTranscriber()
        self.voice_synthesizer = VoiceSynthesizer()
        self.user_preferences = UserPreferences()
        
        # Словарь процессоров
        self.llm_processors = {
            'deepseek': DeepSeekProcessor(),
            'openai': OpenAIProcessor()
        }
        
        # Регистрация обработчиков
        self._register_handlers()

    def _process_message(self, text: str, user_id: int) -> str:
        """
        Обработка сообщения с учетом выбранной модели и автоматическим переключением
        
        :param text: Текст сообщения
        :param user_id: ID пользователя
        :return: Сгенерированный ответ
        """
        # Получаем предпочтительную модель пользователя
        model = self.user_preferences.get_llm_model(user_id)
        
        # Выбираем процессор
        processor = self.llm_processors.get(model, self.llm_processors['deepseek'])
        
        # Обработка сообщения через выбранную модель
        response = processor.process_with_retry(text)
        
        # Если ответ None или содержит сообщение о переключении, пробуем OpenAI
        if (response is None or 
            (isinstance(response, str) and "❌ Извините, закончились средства" in response)):
            
            self.logger.warning(f"Не удалось получить ответ от {model}, переключаемся на OpenAI")
            
            # Принудительное переключение на OpenAI
            processor = self.llm_processors['openai']
            response = processor.process_with_retry(text)
            
            # Автоматическое обновление модели пользователя
            if response is not None:
                self.user_preferences.set_llm_model(user_id, 'openai')
                self.logger.info(f"Модель для пользователя {user_id} автоматически переключена на OpenAI")
        
        return response

    def _register_handlers(self):
        """Регистрация обработчиков команд и сообщений"""
        @self.bot.message_handler(commands=['start', 'help'])
        def send_welcome(message: Message):
            """Обработчик команд /start и /help"""
            welcome_text = (
                "👋 Привет! Я твой умный ассистент. \n\n"
                "Могу помочь с:\n"
                "• Информационными запросами\n"
                "• Решением задач\n"
                "• Голосовым взаимодействием\n\n"
                "Доступные команды:\n"
                "• /select_model - выбрать модель ИИ\n"
                "• /switch_model - переключить модель\n"
                "• /check_balance - проверить баланс API\n\n"
                "Отправь голосовое или текстовое сообщение для общения!"
            )
            
            # Создаем клавиатуру с режимами
            markup = ReplyKeyboardMarkup(row_width=2, resize_keyboard=True)
            markup.add(
                KeyboardButton('🔊 Голосовой режим'),
                KeyboardButton('📝 Текстовый режим')
            )
            
            self.bot.reply_to(message, welcome_text, reply_markup=markup)

        @self.bot.message_handler(commands=['select_model'])
        def select_model(message: Message):
            """Выбор модели LLM"""
            markup = ReplyKeyboardMarkup(row_width=2, resize_keyboard=True)
            markup.add(
                KeyboardButton('🤖 DeepSeek'),
                KeyboardButton('🌐 ChatGPT')
            )
            self.bot.reply_to(
                message, 
                "Выберите модель для генерации ответов:", 
                reply_markup=markup
            )

        @self.bot.message_handler(func=lambda message: message.text in ['🤖 DeepSeek', '🌐 ChatGPT'])
        def set_model(message: Message):
            """Установка выбранной модели"""
            model_map = {
                '🤖 DeepSeek': 'deepseek',
                '🌐 ChatGPT': 'openai'
            }
            model_name = model_map.get(message.text, 'deepseek')
            
            # Проверка валидности API-ключа
            processor = self.llm_processors[model_name]
            if processor.validate_api_key():
                self.user_preferences.set_llm_model(message.from_user.id, model_name)
                model_info = processor.get_model_info()
                
                response = (
                    f"✅ Выбрана модель: {model_info['name']}\n"
                    f"Провайдер: {model_info['provider']}\n"
                    f"Модель по умолчанию: {model_info['default_model']}"
                )
                
                # Возвращаем стандартную клавиатуру
                markup = ReplyKeyboardRemove()
                self.bot.reply_to(message, response, reply_markup=markup)
            else:
                self.bot.reply_to(
                    message, 
                    "❌ Не удалось проверить API-ключ. Пожалуйста, проверьте настройки."
                )

        @self.bot.message_handler(commands=['switch_model'])
        def switch_model(message: Message):
            """Принудительное переключение между моделями"""
            user_id = message.from_user.id
            current_model = self.user_preferences.get_llm_model(user_id)
            
            # Переключаем на альтернативную модель
            new_model = 'openai' if current_model == 'deepseek' else 'deepseek'
            
            # Проверяем валидность нового API-ключа
            new_processor = self.llm_processors[new_model]
            if new_processor.validate_api_key():
                self.user_preferences.set_llm_model(user_id, new_model)
                model_info = new_processor.get_model_info()
                
                response = (
                    f"🔄 Модель переключена на {model_info['name']}\n"
                    f"Провайдер: {model_info['provider']}"
                )
                self.bot.reply_to(message, response)
            else:
                self.bot.reply_to(
                    message, 
                    f"❌ Не удалось переключиться на {new_model}. Проверьте API-ключ."
                )

        @self.bot.message_handler(commands=['check_balance'])
        def check_balance(message: Message):
            """Проверка баланса текущей модели"""
            user_id = message.from_user.id
            model = self.user_preferences.get_llm_model(user_id)
            
            processor = self.llm_processors.get(model, self.llm_processors['deepseek'])
            
            if model == 'deepseek':
                if processor.validate_api_key():
                    self.bot.reply_to(message, "✅ DeepSeek: API-ключ активен")
                else:
                    self.bot.reply_to(message, "❌ DeepSeek: Недостаточно средств или проблемы с ключом")
            elif model == 'openai':
                if processor.validate_api_key():
                    self.bot.reply_to(message, "✅ OpenAI: API-ключ активен")
                else:
                    self.bot.reply_to(message, "❌ OpenAI: Проблемы с ключом")

        @self.bot.message_handler(func=lambda message: message.text == '🔊 Голосовой режим')
        def voice_mode(message: Message):
            """Включение голосового режима"""
            self.user_preferences.set_user_mode(message.from_user.id, 'voice')
            self.bot.reply_to(
                message, 
                "✅ Включен голосовой режим. Теперь я буду отвечать голосовыми сообщениями."
            )

        @self.bot.message_handler(func=lambda message: message.text == '📝 Текстовый режим')
        def text_mode(message: Message):
            """Включение текстового режима"""
            self.user_preferences.set_user_mode(message.from_user.id, 'text')
            self.bot.reply_to(
                message, 
                "✅ Включен текстовый режим. Я буду отвечать текстовыми сообщениями."
            )

        @self.bot.message_handler(content_types=['text'])
        def handle_text_message(message: Message):
            """Обработчик текстовых сообщений"""
            try:
                # Логирование входящего сообщения
                logger.info(f"Получено текстовое сообщение от {message.from_user.username}: {message.text}")
                
                # Обработка сообщения
                response = self._process_message(message.text, message.from_user.id)
                
                # Проверяем, что ответ не пустой
                if not response:
                    response = "Извините, не удалось сгенерировать ответ. Попробуйте позже."
                
                # Проверяем режим пользователя
                user_mode = self.user_preferences.get_user_mode(message.from_user.id)
                
                if user_mode == 'voice':
                    # Генерируем голосовой ответ
                    voice_response = self.voice_synthesizer.text_to_speech(
                        response, 
                        output_file=f'temp_response_{message.message_id}.ogg'
                    )
                    
                    # Отправляем голосовой ответ
                    if voice_response:
                        with open(voice_response, 'rb') as voice:
                            self.bot.send_voice(message.chat.id, voice)
                        os.remove(voice_response)
                else:
                    # Отправляем текстовый ответ
                    self.bot.reply_to(message, response)
                
                # Логирование исходящего сообщения
                logger.info(f"Отправлен ответ: {response}")
            
            except Exception as e:
                # Обработка непредвиденных ошибок
                error_message = "Извините, произошла ошибка при обработке сообщения."
                self.bot.reply_to(message, error_message)
                logger.error(f"Ошибка при обработке текстового сообщения: {e}", exc_info=True)

        @self.bot.message_handler(content_types=['voice'])
        def handle_voice_message(message: Message):
            try:
                # Создаем директорию для временных файлов с использованием абсолютного пути
                base_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.abspath(os.path.join(base_dir, '..', '..'))
                temp_dir = os.path.join(project_root, 'temp')
                
                # Создаем директорию с полными правами
                os.makedirs(temp_dir, mode=0o777, exist_ok=True)
                
                # Скачиваем голосовое сообщение
                voice_file = self.bot.get_file(message.voice.file_id)
                downloaded_file = self.bot.download_file(voice_file.file_path)
                
                # Сохраняем файл с полным путем во временной директории
                voice_path = os.path.join(temp_dir, f'temp_voice_{message.message_id}.oga')
                
                # Диагностика путей
                logger.info(f"Base directory: {base_dir}")
                logger.info(f"Project root: {project_root}")
                logger.info(f"Temp directory: {temp_dir}")
                logger.info(f"Voice file path: {voice_path}")
                
                try:
                    with open(voice_path, 'wb') as new_file:
                        new_file.write(downloaded_file)
                    
                    # Проверка существования и размера файла
                    if not os.path.exists(voice_path):
                        raise FileNotFoundError(f"Файл не создан: {voice_path}")
                    
                    file_size = os.path.getsize(voice_path)
                    logger.info(f"Размер файла: {file_size} байт")
                    
                    if file_size == 0:
                        raise ValueError("Файл пустой")
                
                except Exception as e:
                    logger.error(f"Ошибка сохранения файла: {e}")
                    raise
                
                # Распознаем текст
                text = self.audio_transcriber.transcribe_audio(voice_path)
                
                # Диагностика распознавания
                logger.info(f"Распознанный текст: {text}")
                
                # Обрабатываем сообщение
                response = self._process_message(text, message.from_user.id)
                
                # Проверяем, что ответ не пустой
                if not response:
                    response = "Извините, не удалось сгенерировать ответ. Попробуйте позже."
                
                # Генерируем голосовой ответ
                voice_response = self.voice_synthesizer.text_to_speech(
                    response, 
                    output_file=os.path.join(temp_dir, f'temp_response_{message.message_id}.oga')
                )
                
                # Отправляем голосовой ответ
                if voice_response:
                    with open(voice_response, 'rb') as voice:
                        self.bot.send_voice(message.chat.id, voice)
                    os.remove(voice_response)
                
                # Очищаем временные файлы
                os.remove(voice_path)
                
                # Очистка временной директории от аудиофайлов
                self._cleanup_temp_audio_files(temp_dir)
            
            except Exception as e:
                logger.error(f"Ошибка обработки голосового сообщения: {e}", exc_info=True)
                self.bot.reply_to(message, "Извините, не удалось обработать голосовое сообщение")

    def _cleanup_temp_audio_files(self, temp_dir: str):
        """
        Очистка временной директории от аудиофайлов
        
        :param temp_dir: Путь к временной директории
        """
        try:
            # Список расширений для удаления
            audio_extensions = ['.wav', '.oga', '.ogg', '.mp3']
            
            # Счетчики для логирования
            deleted_files = 0
            total_files = 0
            
            # Перебираем все файлы во временной директории
            for filename in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, filename)
                
                # Проверяем, что это файл и имеет аудио расширение
                if os.path.isfile(file_path) and any(filename.lower().endswith(ext) for ext in audio_extensions):
                    total_files += 1
                    try:
                        os.remove(file_path)
                        deleted_files += 1
                    except Exception as e:
                        logger.warning(f"Не удалось удалить файл {filename}: {e}")
            
            # Логируем результат
            if total_files > 0:
                logger.info(f"Очищено {deleted_files} из {total_files} аудиофайлов во временной директории")
        
        except Exception as e:
            logger.error(f"Ошибка при очистке временной директории: {e}", exc_info=True)

    def start(self):
        """Запуск бота"""
        try:
            logger.info("Запуск Telegram-бота")
            self.bot.polling(none_stop=True)
        except Exception as e:
            logger.error(f"Ошибка при запуске бота: {e}", exc_info=True)

def main():
    """Точка входа для запуска бота"""
    try:
        bot = TelegramAssistantBot()
        bot.start()
    except Exception as e:
        logger.critical(f"Критическая ошибка при запуске бота: {e}", exc_info=True)

if __name__ == '__main__':
    main()