import os
import pytest
import openai
from src.neural_networks.complex_dialog_network import ComplexDialogNetwork

def test_complex_dialog_network_initialization():
    # Проверка инициализации с API-ключом
    os.environ['OPENAI_API_KEY'] = 'test_key'
    network = ComplexDialogNetwork()
    assert openai.api_key == 'test_key'

def test_complex_dialog_network_no_api_key(monkeypatch):
    # Проверка поведения при отсутствии API-ключа
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    
    with pytest.raises(ValueError, match="Необходимо установить OPENAI_API_KEY"):
        ComplexDialogNetwork()