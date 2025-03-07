from dataclasses import dataclass
from dotenv import load_dotenv
from os import getenv


@dataclass
class Telegram:
    token: str
#


@dataclass
class OpenAI:
    api_key: str
    model: str


@dataclass
class DeepSeek:
    api_key: str
    model: str


@dataclass
class YSpeechKit:
    oauth_token: str
    folder_id: str


@dataclass
class NeuralNetworks:
    openai: OpenAI
    deepseek: DeepSeek
    yspeechkit: YSpeechKit


@dataclass
class GoogleCalendar:
    credentials_path: str
    token_path: str
    timezone: str = 'Europe/Moscow'


@dataclass
class Config:
    telegram: Telegram
    neural_networks: NeuralNetworks
    google_calendar: GoogleCalendar


def get_config():
    load_dotenv()

    return Config(
        telegram=Telegram(
            token=getenv('TELEGRAM_BOT_TOKEN')
        ),
        neural_networks=NeuralNetworks(
            openai=OpenAI(
                api_key=getenv('OPENAI_API_KEY'),
                model=getenv('OPENAI_MODEL')
            ),
            deepseek=DeepSeek(
                api_key=getenv('DEEPSEEK_API_KEY'),
                model=getenv('DEEPSEEK_MODEL')
            ),
            yspeechkit=YSpeechKit(
                oauth_token=getenv('OAUTH'),
                folder_id=getenv('YANDEX_FOLDER_ID')
            )
        ),
        google_calendar=GoogleCalendar(
            credentials_path=getenv('GOOGLE_CALENDAR_CREDENTIALS_PATH', 'credentials.json'),
            token_path=getenv('GOOGLE_CALENDAR_TOKEN_PATH', 'token.pickle'),
            timezone=getenv('GOOGLE_CALENDAR_TIMEZONE', 'Europe/Moscow')
        )
    )
