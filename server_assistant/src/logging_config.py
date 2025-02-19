import logging
import os

def setup_logging():
    """
    Настройка подробного логирования для отладки
    """
    # Создаем директорию для логов, если она не существует
    log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, 'assistant.log')
    
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Вывод в консоль
            logging.FileHandler(log_file, encoding='utf-8')  # Логи в файл
        ]
    )
    
    # Настройка логгеров для внешних библиотек
    logging.getLogger('telegram').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)