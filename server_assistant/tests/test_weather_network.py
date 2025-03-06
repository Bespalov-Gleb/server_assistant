import pytest

@pytest.fixture
def weather_network():
    return WeatherNetwork()

def test_weather_network_city_extraction(weather_network):
    cities_to_test = [
        "Какая погода в Москве?",
        "Температура в СПб",
        "Новосибирск, прогноз погоды",
        "Екатеринбург сегодня"
    ]
    
    for city_message in cities_to_test:
        response = weather_network.generate_response(city_message)
        assert "город" in response.lower()

def test_weather_network_no_city(weather_network):
    response = weather_network.generate_response("Какая погода?")
    assert "не удалось определить город" in response.lower()

def test_weather_network_no_api_key(weather_network, monkeypatch):
    monkeypatch.setenv('OPENWEATHER_API_KEY', '')
    response = weather_network.generate_response("Погода в Москве")
    assert "требуется api-ключ" in response.lower()