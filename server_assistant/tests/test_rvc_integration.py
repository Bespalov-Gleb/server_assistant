import os
import pytest
from src.audio_processing.rvc_integration import RVCVoiceConverter

@pytest.fixture
def rvc_converter():
    return RVCVoiceConverter()

def test_rvc_audio_generation(rvc_converter, tmp_path):
    test_text = "Привет, это тестовый голосовой синтез"
    output_path = tmp_path / "test_output.wav"
    
    generated_path = rvc_converter.generate_audio(
        text=test_text, 
        output_path=str(output_path)
    )
    
    assert os.path.exists(generated_path)
    assert generated_path.endswith('.wav')

def test_rvc_audio_generation_different_language(rvc_converter, tmp_path):
    test_text = "Hello, this is a test voice synthesis"
    output_path = tmp_path / "test_output_en.wav"
    
    generated_path = rvc_converter.generate_audio(
        text=test_text, 
        output_path=str(output_path),
        language='en'
    )
    
    assert os.path.exists(generated_path)
    assert generated_path.endswith('.wav')