import pytest
from src.neural_networks.small_talk_network import SmallTalkNetwork

@pytest.fixture
def small_talk_network():
    return SmallTalkNetwork()

def test_small_talk_network_initialization(small_talk_network):
    assert small_talk_network.model is not None
    assert small_talk_network.tokenizer is not None

def test_small_talk_network_response(small_talk_network):
    test_messages = [
        "Привет",
        "Как дела?",
        "Что ты умеешь?",
        "Расскажи о себе"
    ]
    
    for message in test_messages:
        response = small_talk_network.generate_response(message)
        assert isinstance(response, str)
        assert len(response) > 0
        assert len(response) <= 200