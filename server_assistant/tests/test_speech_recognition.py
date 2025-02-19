import pytest
from src.audio_processing.speech_recognition import AudioTranscriber

@pytest.fixture
def audio_transcriber():
    return AudioTranscriber()

def test_audio_transcriber_initialization(audio_transcriber):
    assert audio_transcriber.language == 'ru-RU'
    assert audio_transcriber.recognizer is not None