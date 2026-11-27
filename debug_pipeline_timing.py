from pathlib import Path
from app.speech.tts_service import TTSService
from app.core.container import ServiceContainer
from app.brain.conversation_manager import ConversationManager
from app.brain.llm import LocalLLM
from app.core.event_bus import EventBus
import time
import sys
import subprocess

print("=" * 60)
print("TEST: Full pipeline with incremental TTS")
print("=" * 60)

# Test 1: LLM generation timing
print("\n--- Test 1: LLM generation ---")
llm = LocalLLM()
prompt = "Hello, how can I help you with admission information at I.T.S Engineering College?"

start = time.time()
tokens = list(llm.stream_generate(prompt))
llm_time = time.time() - start
print(f'LLM generation: {llm_time:.2f}s, tokens: {len(tokens)}')

# Test 2: TTS incremental with non-blocking playback
print("\n--- Test 2: TTS incremental with non-blocking playback ---")

container = ServiceContainer()
tts = TTSService(container)

# Copy real model
import tempfile, shutil
tmpdir = tempfile.TemporaryDirectory()
tmp = Path(tmpdir.name)
model_path = tmp / 'en_US-lessac-medium.onnx'
json_path = tmp / 'en_US-lessac-medium.onnx.json'
shutil.copy('models/tts/en_US-lessac-medium.onnx', model_path)
shutil.copy('models/tts/en_US-lessac-medium.onnx.json', json_path)
tts.model_path = model_path

# Monkey-patch timing
original_speak_incremental = tts._speak_incremental
sentence_times = []
playback_start_times = []

def traced_speak_incremental(full_text, sentences):
    import time as time_module
    print(f'  Splitting into {len(sentences)} sentences: {sentences}')
    
    base_cmd = tts._resolve_piper_command()
    
    for i, sentence in enumerate(sentences):
        t0 = time_module.perf_counter()
        
        wav_path = ''
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as audio_file:
                wav_path = audio_file.name
            
            command = base_cmd + [
                '--model', str(tts.model_path),
                '--output_file', wav_path,
            ]
            
            process = subprocess.Popen(command, stdin=subprocess.PIPE, text=True)
            process.communicate(sentence)
            
            gen_time = time_module.perf_counter() - t0
            sentence_times.append(gen_time)
            print(f'  Sentence {i+1} generation ({gen_time:.2f}s): {sentence[:50]}')
            
            # Non-blocking playback
            play_start = time_module.perf_counter()
            if not tts._play_audio_non_blocking(wav_path):
                raise RuntimeError('Playback failed')
            play_duration = time_module.perf_counter() - play_start
            playback_start_times.append(play_duration)
            print(f'  Playback started immediately ({play_duration:.2f}s setup)')
            
        except Exception as e:
            print(f'  Sentence {i+1} error: {type(e).__name__}: {e}')
            import traceback
            traceback.print_exc()
    
    return original_speak_incremental(full_text, sentences)

tts._speak_incremental = traced_speak_incremental

print("\nSpeaking: 'Hello! Welcome to I.T.S Engineering College. How can I help you today?'")
start = time.time()
result = tts.speak('Hello! Welcome to I.T.S Engineering College. How can I help you today?')
total_time = time.time() - start
print(f'\nTotal speak() time: {total_time:.2f}s')
print(f'Sentence generation times: {sentence_times}')
print(f'Playback setup times: {playback_start_times}')
print(f'Result: {result}')

tmpdir.cleanup()