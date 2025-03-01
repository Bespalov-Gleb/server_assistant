import os
import logging
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import BufferedInputFile
import aiofiles
import asyncio
import sys
import json
from datetime import datetime, timedelta

from src.neural_networks.router_network import RouterNetwork
from src.neural_networks.guide_network import GuideNetwork
from src.neural_networks.small_talk_network import SmallTalkNetwork
from src.neural_networks.information_network import InformationNetwork
from src.neural_networks.functional_network import FunctionalNetwork
from src.neural_networks.reminder_network import ReminderNetwork
from src.neural_networks.complex_dialog_network import ComplexDialogNetwork
from src.neural_networks.deepseek_processor import DeepSeekProcessor
from src.neural_networks.openai_processor import OpenAIProcessor

from src.neural_networks.dialog_manager import DialogManager
from src.utils.user_preferences import UserPreferences
from src.audio_processing.speech_recognition import AudioTranscriber
from src.audio_processing.voice_synthesis import VoiceSynthesizer

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()

class TelegramAssistantBot:
    def __init__(self):
        # Загрузка токена бота
        self.token = os.getenv('TELEGRAM_BOT_TOKEN')
        if not self.token:
            raise ValueError("Telegram Bot Token не найден в переменных окружения")
        
        self.bot = Bot(token=self.token)
        self.dp = Dispatcher()
        self.logger = logger
        
        self.reminder_file = 'temp\\reminders.json'
        self.reminders = []  # Инициализируем пустым списком    
        asyncio.create_task(self.initialize_reminders())  
        # Инициализация компонентов
        self.user_preferences = UserPreferences()
        self.audio_transcriber = AudioTranscriber()
        self.voice_synthesizer = VoiceSynthesizer()

        # Инициализация нейронных сетей
        self.dialog_manager = DialogManager()
        # Инициализация сети-гида
        self.llm_processors = {
            'deepseek': DeepSeekProcessor(),
            'openai': OpenAIProcessor()
        }

        # Регистрация обработчиков
        self._register_handlers()
        asyncio.create_task(self.start_reminder_monitoring())

    async def initialize_reminders(self):
        """Асинхронная инициализация напоминаний"""
        try:
            # Загружаем напоминания
            self.reminders = await self.load_reminders()
            self.logger.info(f"Загружено {len(self.reminders)} напоминаний")
        except Exception as e:
            self.logger.error(f"Ошибка инициализации напоминаний: {e}")
            self.reminders = []

    async def load_reminders(self):
        try:
            if os.path.exists(self.reminder_file):
                async with aiofiles.open(self.reminder_file, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    if content:  # Проверяем, что файл не пустой
                        return json.loads(content)
                    else:
                        return []  # Возвращаем пустой список, если файл пустой
            else:
                # Создаем файл с пустым массивом, если он не существует
                async with aiofiles.open(self.reminder_file, 'w', encoding='utf-8') as f:
                    json.dump([], f)
                return []
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке напоминаний: {e}")
            return []

    async def save_reminders(self, reminders=None):
        try:
            reminders_to_save = reminders if reminders is not None else self.reminders
            async with aiofiles.open(self.reminder_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(reminders_to_save, ensure_ascii=False))
            
            self.logger.info(f"Сохранено {len(reminders_to_save)} напоминаний")
        except Exception as e:
            self.logger.error(f"Ошибка сохранения напоминаний: {e}")

    async def add_reminder(self, reminder_text, reminder_time, reminder_type='one-time', user_id=None):
        try:
            
            self.logger.info(f"Добавление напоминания: {reminder_text} на {reminder_time}")
            reminder = {
                'id': len(self.reminders) + 1,
                'text': reminder_text,
                'time': reminder_time.isoformat(),
                'type': reminder_type,
                'user_id': user_id  # Сохраняем ID пользователя
            }
            self.reminders.append(reminder)
            await self.save_reminders()
            # Запуск потока для уведомления
            asyncio.create_task(self.wait_and_notify(reminder))
            return reminder
        except Exception as e:
            self.logger.error(f"Ошибка при добавлении напоминания: {e}")
            return None

    async def wait_and_notify(self, reminder):
        try:
            reminder_time = datetime.fromisoformat(reminder['time'])
            time_to_wait = max((reminder_time - datetime.now()).total_seconds(), 0)
            await asyncio.sleep(time_to_wait)
            message_text = f"Напоминание: {reminder['text']}"

            # Отправка сообщения пользователю
            user_id = reminder['user_id']  # Получаем ID пользователя
            try:
                await self.bot.send_message(user_id, message_text)
            except Exception as send_error:
                self.logger.error(f"Ошибка отправки напоминания: {send_error}")
            self.reminders = [rem for rem in self.reminders if rem['id'] != reminder['id']]
            await self.save_reminders()
            
            # Для постоянных напоминаний создаем новое
            if reminder['type'] == 'constant':
                new_reminder_time = reminder_time + timedelta(days=1)
                await self.add_reminder(
                    reminder['text'], 
                    new_reminder_time, 
                    'constant', 
                    user_id
                )   
        except Exception as e:
            self.logger.error(f"Ошибка в wait_and_notify: {e}")

    async def delete_reminder(self, reminder_id):
        self.reminders = [rem for rem in self.reminders if rem['id'] != reminder_id]
        await self.save_reminders()

    def list_reminders(self):
        return self.reminders

    async def start_reminder_monitoring(self):
        # Небольшая задержка для инициализации
        await asyncio.sleep(5)
        """Асинхронная инициализация напоминаний"""
        while True:
            try:
                now = datetime.now()
                active_reminders = [
                    reminder for reminder in self.reminders 
                    if datetime.fromisoformat(reminder['time']) <= now
                ]
                
                for reminder in active_reminders:
                    asyncio.create_task(self.wait_and_notify(reminder))
                    
                # Удаляем обработанные напоминания
                self.reminders = [
                    rem for rem in self.reminders 
                    if datetime.fromisoformat(rem['time']) > now
                ]
                await self.save_reminders()
                
                # Ожидание перед следующей проверкой
                await asyncio.sleep(60)
            
            except Exception as e:
                self.logger.error(f"Ошибка в мониторинге напоминаний: {e}")
                await asyncio.sleep(60)

    async def _process_message(self, text: str, user_id: int) -> str:
        """Обработка сообщения с учетом выбранной модели и автоматическим переключением"""
        model = self.user_preferences.get_llm_model(user_id)
        
        # Выбираем процессор
        processor = self.llm_processors.get(model, self.llm_processors['deepseek'])
        
        # Обработка сообщения через выбранную модель
        # response = processor.process_with_retry(text)
        guide_network = GuideNetwork(self.bot, user_id)
        response = await guide_network.process_message(text)
        if isinstance(response, list) and response[0] == "Запуск":
            result = await self.add_reminder(
                    response[1], 
                    response[2], 
                    response[3], 
                    user_id=user_id
                )
                
            if result:
                return f"Напоминание '{response[1]}' установлено на {response[2].strftime('%Y-%m-%d %H:%M:%S')}."
            else:
                return "Не удалось создать напоминание."
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
        @self.dp.message(Command('start'))
        async def send_welcome(message: types.Message):
            """Обработчик команд /start и /help"""
            welcome_text = (
                "👋 Привет! Я твой умный ассистент. \n\n"
                "Я могу:\n"
                "✉️ Общаться на разные темы\n"
                "🔍 Искать информацию\n"
                "📅 Создавать напоминания\n"
                "🔊 Принимать голосовые сообщения\n\n"
                "Отправь голосовое или текстовое сообщение для общения!"
            )
            
            await message.answer(welcome_text)

        @self.dp.message(Command('select_model'))
        async def select_model(message: types.Message):
            """Выбор модели LLM"""
            markup = types.ReplyKeyboardMarkup(
                keyboard=[
                    [types.KeyboardButton(text='🤖 DeepSeek'), types.KeyboardButton(text='🌐 ChatGPT')]
                ],
                resize_keyboard=True
            )
            await message.answer(
                "Выберите модель:", 
                reply_markup=markup
            )

        @self.dp.message(lambda message: message.text in ['🤖 DeepSeek', '🌐 ChatGPT'])
        async def set_model(message: types.Message):
            """Установка выбранной модели"""
            model_map = {
                '🤖 DeepSeek': 'deepseek',
                '🌐 ChatGPT': 'openai'
            }
            selected_model = model_map.get(message.text, 'openai')
            
            user_id = message.from_user.id
            try:
                self.user_preferences.set_llm_model(user_id, selected_model)
                await message.answer(f"✅ Выбрана модель: {message.text}")
            except Exception as e:
                await message.answer(
                    "❌ Не удалось проверить API-ключ. Пожалуйста, проверьте настройки."
                )

        @self.dp.message(Command('switch_model'))
        async def switch_model(message: types.Message):
            """Принудительное переключение между моделями"""
            user_id = message.from_user.id
            current_model = self.user_preferences.get_llm_model(user_id)
            
            new_model = 'deepseek' if current_model == 'openai' else 'openai'
            
            try:
                self.user_preferences.set_llm_model(user_id, new_model)
                model_name = '🤖 DeepSeek' if new_model == 'deepseek' else '🌐 ChatGPT'
                await message.answer(f"✅ Переключено на модель: {model_name}")
            except Exception as e:
                await message.answer(
                    f"❌ Не удалось переключиться на {new_model}. Проверьте API-ключ."
                )

        @self.dp.message(Command('check_balance'))
        async def check_balance(message: types.Message):
            """Проверка баланса текущей модели"""
            user_id = message.from_user.id
            model = self.user_preferences.get_llm_model(user_id)
            
            try:
                if model == 'openai':
                    balance = self.user_preferences.check_openai_balance()
                    await message.answer(f"💰 OpenAI: {balance}")
                else:
                    await message.answer("❌ OpenAI: Проблемы с ключом")
            except Exception as e:
                await message.answer("❌ OpenAI: Проблемы с ключом")

        @self.dp.message()
        async def handle_message(message: types.Message):
            """Универсальный обработчик сообщений"""
            response = "Произошла ошибка при обработке вашего запроса."
            if message.content_type == 'text':
                try:
                    # Генерация текстового ответа
                    response = await self._process_message(message.text, message.from_user.id)
                    
                    if response:
                        await message.reply(response)
                
                except Exception as e:
                    self.logger.error(f"Ошибка обработки текстового сообщения: {e}")
                    await message.reply("Произошла ошибка при обработке сообщения.")

            elif message.content_type == 'voice':
                await handle_voice_message(message)
                

        @self.dp.message(lambda message: message.content_type == 'voice')
        async def handle_voice_message(message: types.Message):
            """Обработчик голосовых сообщений"""
            try:
                # Получаем информацию о файле
                voice_file = message.voice
                self.logger.info('Скачивание голосового сообщения')
                destination = os.path.join('temp', f'voice_{message.from_user.id}_{message.message_id}.oga')
                await self.bot.download(voice_file.file_id, destination=destination)
                self.logger.info(f'Скачивание голосового сообщения завершено. Путь: {destination}')
                # Проверяем, что файл существует
                if not os.path.exists(destination):
                    raise FileNotFoundError(f"Файл не найден: {destination}")

                # Транскрибация аудио
                transcribed_text = self.audio_transcriber.transcribe_audio(destination)
                self.logger.info('Транскрибация аудио завершена')
                
                if transcribed_text:
                    self.dialog_manager.add_message(transcribed_text, role='user')
                    # Генерация ответа на основе транскрибированного текста
                    response =await self._process_message(transcribed_text, message.from_user.id)
                    
                    if response:
                        # Синтез голосового ответа
                        voice_response_path = self.voice_synthesizer.text_to_speech(response)
                        
                        # Отправка голосового ответа
                        with open(voice_response_path, 'rb') as voice_file:
                            voice_bytes = voice_file.read()
                            await message.answer_voice(BufferedInputFile(voice_bytes, 'voice.oga'))
                        
                        # Удаление временных файлов
                        os.remove(voice_response_path)
                
                # Удаление временного файла
                if destination and os.path.exists(destination):
                    os.remove(destination)
            
            except Exception as e:
                logger.error(f"Ошибка обработки голосового сообщения: {e}")
                await message.reply("Не удалось обработать голосовое сообщение.")

    async def start(self):
        """Запуск бота"""
        self.logger.info("Telegram бот запущен")
        await self.dp.start_polling(self.bot)

async def main():
    # Настройка логирования
    logging.basicConfig(level=logging.INFO)
    
    # Создание и запуск бота
    bot = TelegramAssistantBot()
    await bot.start()

if __name__ == '__main__':
    asyncio.run(main())