import os
import subprocess
import whisper
import traceback

def diagnose_audio_conversion(input_file):
    print(f"🔍 Диагностика файла: {input_file}")
    
    # Базовая информация о файле
    print(f"Размер файла: {os.path.getsize(input_file)} байт")
    
    # Чтение заголовка файла
    try:
        with open(input_file, 'rb') as f:
            header = f.read(16)
            print(f"Заголовок файла (hex): {header.hex()}")
    except Exception as e:
        print(f"Ошибка чтения заголовка: {e}")
    
    # Попытка определить формат через FFmpeg
    try:
        ffmpeg_info = subprocess.run([
            'ffmpeg', 
            '-i', input_file, 
            '-show_entries', 'format=format_name', 
            '-v', 'quiet', 
            '-of', 'default=noprint_wrappers=1:nokey=1'
        ], capture_output=True, text=True)
        
        print(f"Формат файла (FFmpeg): {ffmpeg_info.stdout.strip()}")
    except Exception as e:
        print(f"Ошибка определения формата: {e}")
    
    # Попытка загрузки через Whisper
    try:
        model = whisper.load_model("base")
        audio = whisper.load_audio(input_file)
        print("✅ Аудио успешно загружено Whisper")
        
        # Дополнительная информация о загруженном аудио
        print(f"Длина аудио: {len(audio)} сэмплов")
        print(f"Тип данных: {audio.dtype}")
        print(f"Мин/макс значения: {audio.min()}, {audio.max()}")
    
    except Exception as e:
        print("❌ Ошибка загрузки аудио Whisper")
        print(traceback.format_exc())

# Путь к вашему аудиофайлу
audio_file = r"C:\Users\ArdorPC\CascadeProjects\server_assistant\server_assistant\temp\temp_voice_204_converted.wav"
diagnose_audio_conversion(audio_file)