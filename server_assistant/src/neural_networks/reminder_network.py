import json
from datetime import datetime
from .openai_processor import OpenAIProcessor

class ReminderNetwork:
    def __init__(self, bot):
        self.openai_processor = OpenAIProcessor()
        self.bot = bot # Инициализация менеджера напоминаний

    def generate_response(self, message: str):
        """
        Отправляет запрос к нейросети для извлечения деталей напоминания.
        
        :param message: Входящее сообщение от пользователя
        :return: текст напоминания, время напоминания, тип напоминания
        """
        response = self.openai_processor.process_with_retry(
            prompt=message,
            system_message=
            '''Твой владелец - Владимир. Твой создатель - Глеб. 
        Ты являешься личным ассистентом и помощником. 
        Ты умеешь запоминать информацию.
        Ты работаешь в рамках телеграм-бота. 
        Твоя сессия никогда не заканчивается, поэтому диалог для тебя никогда не прерывается. 
        Общайся без вводных слов по типу "Конечно, вот несколько вариантов". 
        Отвечай четко на поставленные вопросы и делай в точности то, о чем тебя просят.
        Тебе нужно извлечь данные для создания напоминания. А именно:
        Текст напоминания, то есть что нужно напомнить.
        Дату и время напоминания.
        Тип напоминания: однократно(one-time) или на постоянной основе(constant).
        Верни данные в следующем формате:
        text = ""
        date = ДД.ММ.ГГГГ ЧЧ:ММ
        type = "one-time" или "constant"''',
            temperature=0.2
        )
        self.extract_reminder_details(response)


    
    def extract_reminder_details(self, response):
        """
        Извлекает детали напоминания из ответа нейросети.
        
        :param response: Ответ от нейросети
        :return: текст напоминания, время напоминания, тип напоминания
        """
        # Предполагаем, что response является JSON-строкой
        try:
            data = json.loads(response)
            reminder_text = data.get("reminder_text", "Нет текста напоминания")
            reminder_time_str = data.get("reminder_time", None)
            reminder_type = data.get("reminder_type", "one-time")

            # Преобразование строки времени в объект datetime
            reminder_time = datetime.fromisoformat(reminder_time_str) if reminder_time_str else datetime.now()

            return reminder_text, reminder_time, reminder_type
        except (json.JSONDecodeError, ValueError) as e:
            # Обработка ошибок парсинга
            print(f"Ошибка при извлечении данных о напоминании: {e}")
            return None, None, None

    def create_reminder(self, message: str):
        """
        Создает напоминание на основе входящего сообщения.
        
        :param message: Входящее сообщение от пользователя
        """
        reminder_text, reminder_time, reminder_type = self.generate_response(message)
        if reminder_text and reminder_time:
            self.bot.add_reminder(reminder_text, reminder_time, reminder_type)
            return f"Напоминание '{reminder_text}' установлено на {reminder_time.strftime('%Y-%m-%d %H:%M:%S')}."
        else:
            return "Не удалось создать напоминание."