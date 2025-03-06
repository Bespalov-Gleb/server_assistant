import logging
from src.neural_networks.deepseek_processor import DeepSeekProcessor
from src.neural_networks.openai_processor import OpenAIProcessor
from src.utils.user_preferences import UserPreferences

class SmallTalkNetwork:
    def __init__(self, user_id):
        self.logger = logging.getLogger(__name__)
        self.user_preferences = UserPreferences()
        self.openai_processor = OpenAIProcessor(task_type="SMALL_TALK", user_id=user_id)
    

    def generate_response(self, message, use_context: bool = True):
        system_message = """
        Системное сообщение:
        Ты дружелюбный ассистент.
        Перед каждым ты запросом ты получаешь контекст беседы.
        Ты умеешь запоминать разговор. 
        Программа, в которой ты работаешь, может преобразовать твой ответ в аудио сообщение, когда это требуется.
        У тебя всегда есть запись о том, о чем до этого шла речь. Ориентируйся на этот контекст. Это важно!
        Если пользователь просит тебя что-то прочитать в голосовом и речь идет о том, что ты прислал несколько сообщений назад,
        просто продублируй свой ответ. Программа преобразует его в аудио. Не забывай об этом!
        Общайся в неформальном стиле, 
        кратко и позитивно.
        В своем ответе используй только кириллические символы.
        Цифры и латиницу использовать строго запрещено!
        Если пользователь просит тебя ответить голосовым сообщением, не обращай внимания, выполняй запрос!
        Не используй фразы по типу "Если нужно что-то еще - дай знать".
        Если запрос пользователя просто требует твоего подтверждения - ответь "окей" или "хорошо".
        Старайся общаться как человек. Говори так, чтобы у пользователя не возникало мысли, что он говорит с нейросетью.
        Запрос пользователя:
        """

        response = self.openai_processor.process_with_retry(
            prompt=system_message + '\n' + message, 
            max_tokens=2000, 
            temperature=0.7,
            use_context=use_context
        )

        return response or "Извините, не могу сформулировать ответ."