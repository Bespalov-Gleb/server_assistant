import os
import logging
import asyncio
import json
from datetime import datetime, timedelta

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import BufferedInputFile

import aiofiles

from src.neural_networks.router_network import OutputType
from src.neural_networks.guide_network import GuideNetwork
from src.neural_networks.dialog_manager import DialogManager
from src.utils.user_preferences import UserPreferences
from src.audio_processing.speech_recognition import AudioTranscriber
from src.audio_processing.voice_synthesis import VoiceSynthesizer
import glob

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TelegramAssistantBot:
    def __init__(self, token):
        # Загрузка токена бота
        self.token = os.getenv('TELEGRAM_BOT_TOKEN')
        if not self.token:
            raise ValueError("Telegram Bot Token не найден в переменных окружения")
        
        self.bot = Bot(token=token)
        self.dp = Dispatcher()
        self.logger = logger
        
        self.reminder_file = 'temp/reminders.json'
        self.reminders = []  # Инициализируем пустым списком    
        
        # Инициализация компонентов
        self.user_preferences = UserPreferences()
        self.audio_transcriber = AudioTranscriber()
        self.voice_synthesizer = VoiceSynthesizer()
        self.dialog_manager = DialogManager()
        
        # Регистрация обработчиков
        self._register_handlers()
        
        # Запуск мониторинга напоминаний
        asyncio.create_task(self.initialize_reminders())
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
                    # Явная проверка на пустую строку или пустой JSON
                    if content.strip() in ['', '[]', 'null']:
                        return []
                    try:
                        return json.loads(content)
                    except json.JSONDecodeError:
                        self.logger.error(f"Ошибка декодирования JSON: {content}")
                        return []
            else:
                # Создаем файл с пустым списком, если он не существует
                async with aiofiles.open(self.reminder_file, 'w', encoding='utf-8') as f:
                    await f.write('[]')
                return []
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке напоминаний: {e}")
            return []

    async def save_reminders(self):
        try:
            async with aiofiles.open(self.reminder_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(self.reminders, ensure_ascii=False))
            
            self.logger.info(f"Сохранено {len(self.reminders)} напоминаний")
        except Exception as e:
            self.logger.error(f"Ошибка сохранения напоминаний: {e}")

    async def add_reminder(self, reminder_text, reminder_time, reminder_type='one-time', user_id=None):
        try:
            self.logger.info(f"Добавление напоминания: {reminder_text}, время: {reminder_time}, тип: {reminder_type}")
            
            reminder = {
                'id': len(self.reminders) + 1,
                'text': reminder_text,
                'time': reminder_time.isoformat(),
                'type': reminder_type,
                'user_id': user_id
            }
            
            self.reminders.append(reminder)
            await self.save_reminders()
            
            self.logger.info(f"Напоминание успешно добавлено. Текущий список: {self.reminders}")
            
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

    async def _process_message(self, text: str, user_id: int):
        """Обработка сообщения с учетом выбранной модели и автоматическим переключением"""
        guide_network = GuideNetwork(bot=self.bot, user_id=user_id)
        response, output_type = await guide_network.process_message(text)
        if isinstance(response, list):
            if response[0] == "Запуск":
                await self.add_reminder(response[1], response[2], response[3], user_id=user_id)
                response = f"Установлено напоминание {response[1]} на {response[2]}"
        
        return response, output_type

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

        @self.dp.message()
        async def handle_message(message: types.Message):
            """Универсальный обработчик сообщений"""
            response = "Произошла ошибка при обработке вашего запроса."
            if message.content_type == types.ContentType.TEXT:
                try:
                    # Генерация текстового ответа
                    response, output_type = await self._process_message(message.text, message.from_user.id)
                    
                    if output_type == OutputType.TEXT:  
                        if response:
                            await message.reply(response)
                        else:
                            response = "Извините, не удалось сгенерировать ответ. Попробуйте позже."
                            await message.reply(response)
                    elif output_type == OutputType.AUDIO:
                        if response:
                            # Синтез голосового ответа
                            voice_response_path = self.voice_synthesizer.text_to_speech(response)
                            
                            # Отправка голосового ответа
                            with open(voice_response_path, 'rb') as voice_file:
                                await message.answer_voice(BufferedInputFile(voice_file.read(), 'voice.oga'))
                            
                            # Удаление временных файлов
                            os.remove(voice_response_path)

                    elif output_type == OutputType.MULTI:
                        if response:
                            await message.reply(response)
                            # Синтез голосового ответа
                            voice_response_path = self.voice_synthesizer.text_to_speech(response)
                            
                            # Отправка голосового ответа
                            with open(voice_response_path, 'rb') as voice_file:
                                await message.answer_voice(BufferedInputFile(voice_file.read(), 'voice.oga'))
                            
                            # Удаление временных файлов
                            os.remove(voice_response_path)
                    elif output_type == OutputType.DEFAULT:
                        if response:
                            await message.reply(response)
                        else:
                            response = "Извините, не удалось сгенерировать ответ. Попробуйте позже."
                            await message.reply(response)
                
                except Exception as e:
                    self.logger.error(f"Ошибка обработки текстового сообщения: {e}")
                    await message.reply("Произошла ошибка при обработке сообщения.")

            elif message.content_type == types.ContentType.VOICE:
                await handle_voice_message(message)

        @self.dp.message(lambda message: message.content_type == types.ContentType.VOICE)
        async def handle_voice_message(message: types.Message):
            """Обработчик голосовых сообщений"""
            try:
                # Получаем информацию о файле
                voice_file = message.voice
                self.logger.info('Скачивание голосового сообщения')
                destination = os.path.join('temp', f'voice_{message.from_user.id}_{message.message_id}.oga')
                await self.bot.download(voice_file.file_id, destination=destination)
                self.logger.info(f'Скачивание голосового сообщения завершено. Путь: {destination}')
                
                # Транскрибация аудио
                transcribed_text = self.audio_transcriber.transcribe_audio(destination)
                self.logger.info('Транскрибация аудио завершена')
                
                if transcribed_text:
                    self.dialog_manager.add_message(transcribed_text, role='user_voice')
                    # Генерация ответа на основе транскрибированного текста
                    response, output_type = await self._process_message(transcribed_text, message.from_user.id)
                    
                    if output_type == OutputType.TEXT:  
                        if response:
                            await message.reply(response)
                        else:
                            response = "Извините, не удалось сгенерировать ответ. Попробуйте позже."
                            await message.reply(response)
                    elif output_type == OutputType.AUDIO:
                        if response:
                            # Синтез голосового ответа
                            voice_response_path = self.voice_synthesizer.text_to_speech(response)
                            
                            # Отправка голосового ответа
                            with open(voice_response_path, 'rb') as voice_file:
                                await message.answer_voice(BufferedInputFile(voice_file.read(), 'voice.oga'))
                            
                            # Удаление временных файлов
                            os.remove(voice_response_path)
                    elif output_type == OutputType.MULTI:
                        if response:
                            await message.reply(response)
                            # Синтез голосового ответа
                            voice_response_path = self.voice_synthesizer.text_to_speech(response)
                            
                            # Отправка голосового ответа
                            with open(voice_response_path, 'rb') as voice_file:
                                await message.answer_voice(BufferedInputFile(voice_file.read(), 'voice.oga'))
                            
                            # Удаление временных файлов
                            os.remove(voice_response_path)
                    elif output_type == OutputType.DEFAULT:
                        if response:
                            # Синтез голосового ответа
                            voice_response_path = self.voice_synthesizer.text_to_speech(response)
                            
                            # Отправка голосового ответа
                            with open(voice_response_path, 'rb') as voice_file:
                                await message.answer_voice(BufferedInputFile(voice_file.read(), 'voice.oga'))
                            
                            # Удаление временных файлов
                            os.remove(voice_response_path)
                
                # Удаление временного файла
                os.remove(destination)
            
            except Exception as e:
                logger.error(f"Ошибка обработки голосового сообщения: {e}")
                await message.reply("Не удалось обработать голосовое сообщение.")
    async def _cleanup_temp_audio_files(self):
        """
        Удаляет все .wav файлы из временной директории после отправки
        """
        try:
            wav_files = glob.glob(os.path.join('temp', '*.wav'))
            for file_path in wav_files:
                try:
                    os.remove(file_path)
                    self.logger.info(f"Удален временный аудиофайл: {file_path}")
                except Exception as e:
                    self.logger.error(f"Ошибка удаления файла {file_path}: {e}")
        except Exception as e:
            self.logger.error(f"Ошибка при очистке временных аудиофайлов: {e}")


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