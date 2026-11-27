from pathlib import Path
from app.speech.tts_service import TTSService
from app.core.container import ServiceContainer
from app.brain.llm import LocalLLM
from app.core.event_bus import EventBus
import time
import sys
import subprocess

# Test the complete pipeline timing
print("=" * 60)
print("TEST 1: LLM generation timing")
print("=" * 60)

llm = LocalLLM()
prompt = "Hello, how can I help you today with admission information?"

start = time.time()
tokens = list(llm.stream_generate(prompt))
llm_time = time.time() - start
print(f'LLM generation: {llm_time:.2f}s, tokens: {len(tokens)}')

print("\n" + "=" * 60)
print("TEST 2: TTS with real model (incremental)")
print("=" * 60)

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

# Monkey-patch to trace
original_speak_incremental = tts._speak_incremental

sentence_times = []

def traced_speak_incremental(full_text, sentences):
    import time as time_module
    print(f'  Splitting into {len(sentences)} sentences')
    
    # Initialize Piper once
    base_cmd = tts._resolve_piper_command()
    
    for i, sentence in enumerate(sentences):
        t0 = time_module.perf_counter()
        
        wav_path = ''
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as audio_file:
                wav_path = audio_file.name
            
            command = base_cmd + ['--model', str(tts.model_path), '--output_file', wav_path]
            process = subprocess.Popen(command, stdin=subprocess.PIPE, text=True)
            process.communicate(sentence)
            
            gen_time = time_module.perf_counter() - t0
            sentence_times.append(gen_time)
            print(f'  Sentence {i+1} generation: {gen_time:.2f}s')
            
            # Play immediately (just verify WAV exists)
            if os.path.exists(wav_path):
                size = os.path.getsize(wav_path)
                print(f'  WAV ready ({size} bytes), would play now')
            
        except Exception as e:
            print(f'  Sentence {i+1} error: {e}')
    
    return original_speak_incremental(full_text, sentences)

import os
tts._speak_incremental = traced_speak_incremental

print("\nSpeaking: 'Hello! Welcome to I.T.S Engineering College. How can I help you today?'")
start = time.time()
sentences = ['Hello!', 'Welcome to I.T.S Engineering College.', 'How can I help you today?']
result = tts.speak('Hello! Welcome to I.T.S Engineering College. How can I help you today?')
total_time = time.time() - start
print(f'\nTotal: {total_time:.2f}s')
print(f'Sentence times: {sentence_times}')

tmpdir.cleanup()