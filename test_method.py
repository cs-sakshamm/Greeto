import sys
sys.path.insert(0, '/c/Users/saksh/OneDrive/Desktop/AI_ADMISSION_ROBOT')

from app.speech.tts_service import TTSService
from app.core.container import ServiceContainer

container = ServiceContainer()
tts = TTSService(container)

print("Has _play_audio_non_blocking:", hasattr(tts, '_play_audio_non_blocking'))
print("Method:", tts._play_audio_non_blocking if hasattr(tts, '_play_audio_non_blocking') else "NOT FOUND")