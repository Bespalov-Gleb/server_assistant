import json
import os

class UserPreferences:
    """
    Управляет пользовательскими настройками.
    Сохраняет и загружает предпочтения пользователей из JSON файла.
    """

    def __init__(self, preferences_file='user_preferences.json'):
        """
        :param preferences_file: Путь к файлу с настройками пользователей
        """
        self.preferences_file = preferences_file
        self.preferences = self._load_preferences()

    def _load_preferences(self):
        """
        Загружает настройки пользователей из файла.

        :return: Словарь с настройками пользователей
        """
        if os.path.exists(self.preferences_file):
            try:
                with open(self.preferences_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _save_preferences(self):
        """
        Сохраняет текущие настройки пользователей в файл.
        """
        try:
            with open(self.preferences_file, 'w') as f:
                json.dump(self.preferences, f, indent=4)
        except IOError:
            print(f"Не удалось сохранить настройки в {self.preferences_file}")

    def set_llm_model(self, chat_id: int, model: str):
        """
        Устанавливает предпочитаемую модель для пользователя.

        :param chat_id: ID чата
        :param model: Название модели
        """
        user_id_str = str(user_id)
        chat_id_str = str(chat_id)
        if chat_id_str not in self.preferences:
            self.preferences[chat_id_str] = {}
        
        # Гарантируем, что значение - словарь
        if not isinstance(self.preferences[chat_id_str], dict):
            self.preferences[chat_id_str] = {}
        
        self.preferences[chat_id_str]['model'] = model
        self._save_preferences()

        
    def get_llm_model(self, chat_id: int, default: str = 'deepseek'):
        """
        Возвращает предпочитаемую модель пользователя.

        :param chat_id: ID пользователя
        :param default: Модель по умолчанию
        :return: Название модели
        """
        chat_id_str = str(chat_id)
        user_prefs = self.preferences.get(chat_id_str, {})
        
        # Если user_prefs не словарь, возвращаем дефолтное значение
        if not isinstance(user_prefs, dict):
            return default
        
        return user_prefs.get('model', default)

