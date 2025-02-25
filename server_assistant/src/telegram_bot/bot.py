import os
import logging
from dotenv import load_dotenv
import telebot
from telebot.types import Message, ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove
import sys
import json
from datetime import datetime, timedelta
import threading
from src.utils.message_type_detector import MessageTypeDetector

# Добавляем путь к корневой директории проекта
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Импорты компонентов
from src.neural_networks.guide_network import GuideNetwork
from src.neural_networks.deepseek_processor import DeepSeekProcessor
from src.neural_networks.openai_processor import OpenAIProcessor
from src.audio_processing.speech_recognition import AudioTranscriber
from src.audio_processing.voice_synthesis import VoiceSynthesizer
from src.utils.user_preferences import UserPreferences
from src.neural_networks.dialog_manager import DialogManager

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
        self.message_type_detector = MessageTypeDetector()
        if not self.token:
            raise ValueError("Telegram Bot Token не найден в переменных окружения")
        
        self.bot = telebot.TeleBot(self.token)
        self.logger = logger
        
        # Инициализация компонентов
        self.guide_network = GuideNetwork(bot=self.bot)
        self.audio_transcriber = AudioTranscriber()
        self.voice_synthesizer = VoiceSynthesizer()
        self.user_preferences = UserPreferences()

        self.reminder_file = 'temp/reminders.json'
        self.reminders = self.load_reminders()
        #self.start_reminder_thread()
        
        # Словарь процессоров
        self.llm_processors = {
            'deepseek': DeepSeekProcessor(),
            'openai': OpenAIProcessor()
        }
        # Добавляем инициализацию dialog_manager
        self.dialog_manager = DialogManager(
            max_context_length=50,  # Соответствует текущей настройке
            context_file=f'temp/dialogue_context.json'
        )
        
        # Регистрация обработчиков
        self._register_handlers()

    def load_reminders(self):
        if os.path.exists(self.reminder_file):
            with open(self.reminder_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if content:  # Проверяем, что файл не пустой
                    return json.loads(content)
                else:
                    return []  # Возвращаем пустой список, если файл пустой
        else:
            # Создаем файл с пустым массивом, если он не существует
            with open(self.reminder_file, 'w', encoding='utf-8') as f:
                json.dump([], f)
            return []

    def save_reminders(self):
        with open(self.reminder_file, 'w', encoding='utf-8') as f:
            json.dump(self.reminders, f)

    def add_reminder(self, reminder_text, reminder_time, reminder_type='one-time', user_id=None):
        reminder = {
            'id': len(self.reminders) + 1,
            'text': reminder_text,
            'time': reminder_time.isoformat(),
            'type': reminder_type,
            'user_id': user_id  # Сохраняем ID пользователя
        }
        self.reminders.append(reminder)
        self.save_reminders()

        # Запуск потока для уведомления
        threading.Thread(target=self.wait_and_notify, args=(reminder,)).start()

    def wait_and_notify(self, reminder):
        reminder_time = datetime.fromisoformat(reminder['time'])
        time_to_wait = (reminder_time - datetime.now()).total_seconds()
        if time_to_wait > 0:
            time.sleep(time_to_wait)
            message_text = f"Напоминание: {reminder['text']}"

            # Отправка сообщения пользователю
            user_id = reminder['user_id']  # Получаем ID пользователя
            self.bot.send_message(user_id, message_text)  # Используйте правильный метод для отправки сообщения
            if reminder['type'] == 'one-time':
                self.reminders.remove(reminder)  # Удаляем напоминание
                self.save_reminders()
            elif reminder['type'] == 'constant':
                # Устанавливаем новое напоминание на сутки вперед
                new_reminder_time = reminder_time + timedelta(days=1)
                self.add_reminder(reminder['text'], new_reminder_time, 'constant')
                self.reminders.remove(reminder)  # Удаляем старое напоминание
                self.save_reminders()
    def delete_reminder(self, reminder_id):
        self.reminders = [rem for rem in self.reminders if rem['id'] != reminder_id]
        self.save_reminders()

    def list_reminders(self):
        return self.reminders

    def start_reminder_thread(self):
        threading.Thread(target=self.check_reminders).start()

    def check_reminders(self):
        while True:
            now = datetime.now()
            if self.reminders:
                for reminder in self.reminders:
                    reminder_time = datetime.fromisoformat(reminder['time'])
                    if reminder_time <= now:
                        print(f"Напоминание: {reminder['text']}")  # Здесь можно добавить отправку сообщения в Telegram
                        self.reminders.remove(reminder)
                        self.save_reminders()
                time.sleep(60)  # Проверяем каждую минуту
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
        # response = processor.process_with_retry(text)
        response = self.guide_network.process_message(text)
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
            
            self.bot.reply_to(message, welcome_text)

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

        @self.bot.message_handler(content_types=['text', 'voice'])
        def handle_message(message: Message):
            """Универсальный обработчик сообщений"""
            if message.content_type == 'text':
                # Существующая логика для текстовых сообщений
                handle_text_message(message)
            elif message.content_type == 'voice':
                # Существующая логика для голосовых сообщений
                handle_voice_message(message)

        @self.bot.message_handler(content_types=['text'])
        def handle_text_message(message: Message):
            """Обработчик текстовых сообщений"""
            response = "Произошла ошибка при обработке вашего запроса."
            try:
                # Генерация текстового ответа
                response = self._process_message(message.text, message.from_user.id)
                
                if response:
                    self.bot.reply_to(message, response)
                else:
                    response = "Извините, не удалось сгенерировать ответ. Попробуйте позже."
                    self.bot.reply_to(message, response)
            
            except Exception as e:
                logger.error(f"Ошибка обработки текстового сообщения: {e}")
                self.bot.reply_to(message, "Произошла ошибка при обработке сообщения.")
            
                

        @self.bot.message_handler(content_types=['voice'])
        def handle_voice_message(message: Message):
            """Обработчик голосовых сообщений"""
            try:
                # Получаем информацию о файле
                file_info = self.bot.get_file(message.voice.file_id)
                downloaded_file = self.bot.download_file(file_info.file_path)
                
                # Сохраняем временный файл
                with open('temp_voice_message.oga', 'wb') as new_file:
                    new_file.write(downloaded_file)
                
                # Транскрибация аудио
                transcribed_text = self.audio_transcriber.transcribe_audio('temp_voice_message.oga')
                
                if transcribed_text:
                    self.dialog_manager.add_message(transcribed_text, role='user_voice')
                    # Генерация ответа на основе транскрибированного текста
                    response = self._process_message(transcribed_text, message.from_user.id)
                    
                    if response:
                        # Синтез голосового ответа
                        voice_response_path = self.voice_synthesizer.text_to_speech(response)
                        
                        # Отправка голосового ответа
                        with open(voice_response_path, 'rb') as voice_file:
                            self.bot.send_voice(message.chat.id, voice_file)
                        
                        # Удаление временных файлов
                        os.remove(voice_response_path)
                
                # Удаление временного файла
                os.remove('temp_voice_message.oga')
            
            except Exception as e:
                logger.error(f"Ошибка обработки голосового сообщения: {e}")
                self.bot.reply_to(message, "Произошла ошибка при обработке голосового сообщения.")

    def start(self):
        """Запуск бота"""
        logger.info("Telegram бот запущен")
        self.bot.polling(none_stop=True)

def main():
    """Точка входа для запуска бота"""
    bot = TelegramAssistantBot()
    bot.start()

if __name__ == '__main__':
    main()