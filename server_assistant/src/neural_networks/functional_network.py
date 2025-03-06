import logging
from .deepseek_processor import DeepSeekProcessor  # Изменили импорт
from .openai_processor import OpenAIProcessor
from ..utils.user_preferences import UserPreferences
from aiogram import types

class FunctionalNetwork:
    def __init__(self, chat_id: int):
        self.logger = logging.getLogger(__name__)
        self.user_preferences = UserPreferences()
        selected_model = self.user_preferences.get_llm_model(chat_id=chat_id)

            
        self.openai_processor = OpenAIProcessor(chat_id=chat_id)

    def generate_response(self, message, transcribe=None):
        system_message = """
        Системное сообщение:
        Твой владелец - Владимир. Твой создатель - Глеб. 
        Ты являешься личным ассистентом и помощником. 
        Ты умеешь запоминать информацию.
        Ты работаешь в рамках телеграм-бота. 
        Твоя сессия никогда не заканчивается, поэтому диалог для тебя никогда не прерывается. 
        Общайся без вводных слов по типу "Конечно, вот несколько вариантов". 
        Отвечай четко на поставленные вопросы и делай в точности то, о чем тебя просят.
        Программа, в которой ты работаешь, может преобразовать твой ответ в аудио сообщение, когда это требуется.
        У тебя всегда есть запись о том, о чем до этого шла речь. Ориентируйся на этот контекст. Это важно!
        Если пользователь просит тебя что-то прочитать в голосовом и речь идет о том, что ты прислал несколько сообщений назад,
        просто продублируй свой ответ. Программа преобразует его в аудио. Не забывай об этом!
        Ты помощник для выполнения конкретных задач.
        Перед каждым ты запросом ты получаешь контекст беседы.
        Ты умеешь запоминать разговор. 
        Четко и лаконично объясняй алгоритм действий. 
        Давай пошаговые инструкции.
        В своем ответе используй только кириллические символы.
        Цифры и латиницу использовать строго запрещено!
        Если пользователь просит тебя ответить голосовым сообщением, не обращай внимания, выполняй запрос!
        Не используй фразы по типу "Если нужно что-то еще - дай знать".
        Если запрос пользователя просто требует твоего подтверждения - ответь "окей" или "хорошо".
        Старайся общаться как человек. Говори так, чтобы у пользователя не возникало мысли, что он говорит с нейросетью.
        Запрос пользователя:
        """
        if transcribe == None:
            text = message.from_user.username + ': ' + message.text
        else:
            text = message.from_user.username + ': ' + transcribe
        response = self.openai_processor.process_with_retry(
            prompt=system_message + '\n' + text, 
            max_tokens=2000, 
            temperature=0.4,
            use_context=True
        )

        return response or "Извините, не могу помочь с выполнением этой задачи."