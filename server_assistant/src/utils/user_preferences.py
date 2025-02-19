import json
import os

class UserPreferences:
    def __init__(self, preferences_file='user_preferences.json'):
        self.preferences_file = preferences_file
        self.preferences = self._load_preferences()

    def _load_preferences(self):
        """Загрузка настроек пользователей"""
        if os.path.exists(self.preferences_file):
            try:
                with open(self.preferences_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _save_preferences(self):
        """Сохранение настроек пользователей"""
        try:
            with open(self.preferences_file, 'w') as f:
                json.dump(self.preferences, f, indent=4)
        except IOError:
            print(f"Не удалось сохранить настройки в {self.preferences_file}")

    def set_llm_model(self, user_id: int, model: str):
        """Установка модели для пользователя"""
        user_id_str = str(user_id)
        if user_id_str not in self.preferences:
            self.preferences[user_id_str] = {}
        
        # Гарантируем, что значение - словарь
        if not isinstance(self.preferences[user_id_str], dict):
            self.preferences[user_id_str] = {}
        
        self.preferences[user_id_str]['model'] = model
        self._save_preferences()

    def get_llm_model(self, user_id: int, default: str = 'deepseek'):
        """Получение модели для пользователя"""
        user_id_str = str(user_id)
        user_prefs = self.preferences.get(user_id_str, {})
        
        # Если user_prefs не словарь, возвращаем дефолтное значение
        if not isinstance(user_prefs, dict):
            return default
        
        return user_prefs.get('model', default)

    def set_user_mode(self, user_id: int, mode: str):
        """Установка режима для пользователя"""
        user_id_str = str(user_id)
        if user_id_str not in self.preferences:
            self.preferences[user_id_str] = {}
        
        # Гарантируем, что значение - словарь
        if not isinstance(self.preferences[user_id_str], dict):
            self.preferences[user_id_str] = {}
        
        self.preferences[user_id_str]['mode'] = mode
        self._save_preferences()

    def get_user_mode(self, user_id: int, default: str = 'text'):
        """Получение режима для пользователя"""
        user_id_str = str(user_id)
        user_prefs = self.preferences.get(user_id_str, {})
        
        # Если user_prefs не словарь, возвращаем дефолтное значение
        if not isinstance(user_prefs, dict):
            return default
        
        return user_prefs.get('mode', default)