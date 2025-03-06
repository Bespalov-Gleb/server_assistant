import pytest
from src.neural_networks.router_network import RouterNetwork, TaskType
from src.neural_networks.small_talk_network import SmallTalkNetwork
from src.neural_networks.complex_dialog_network import ComplexDialogNetwork

@pytest.fixture
def router_network():
    networks = {
        TaskType.SMALL_TALK: SmallTalkNetwork(),
        TaskType.COMPLEX_DIALOG: ComplexDialogNetwork(),
    }
    return RouterNetwork(networks)

def test_router_network_task_detection(router_network):
    # Тесты определения типа задачи
    assert router_network.detect_task_type("Привет") == TaskType.SMALL_TALK
    assert router_network.detect_task_type("Расскажи мне историю") == TaskType.COMPLEX_DIALOG