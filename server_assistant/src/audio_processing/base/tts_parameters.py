from dataclasses import dataclass


@dataclass
class Parameters:
    voice: str | None = None
    emotion: str | None = None
    speed: float | None = None
    language: str | None = "ru-RU"
    format: str | None = "mp3"  # auido format, e.g. mp3, wav, ogg...
