from pathlib import Path
from app.speech.tts_service import TTSService
from app.core.container import ServiceContainer
import time
import sys

container = ServiceContainer()
tts = TTSService(container)

# Monkey-patch _speak_incremental to trace timing
original_method = tts._speak_incremental

call_count = [0]

def traced_speak_incremental(full_text, sentences):
    import time as time_module
    call_count[0] += 1
    print(f'  _speak_incremental call #{call_count[0]} with {len(sentences)} sentences')
    print(f'  Text: {full_text[:80]}...')
    
    for i, sentence in enumerate(sentences):
        t0 = time_module.perf_counter()
        # Call the actual WAV generation
        wav_path = ''
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as audio_file:
                wav_path = audio_file.name
            
            base_cmd = tts._resolve_piper_command()
            command = base_cmd + ['--model', str(tts.model_path), '--output_file', wav_path]
            
            import subprocess
            process = subprocess.Popen(command, stdin=subprocess.PIPE, text=True)
            result_text = process.communicate(sentence)
            
            gen_time = time_module.perf_counter() - t0
            print(f'  Sentence {i+1} generation ({gen_time:.2f}s): {sentence[:50]}...')
            
            # Check WAV file
            import os
            if os.path.exists(wav_path):
                size = os.path.getsize(wav_path)
                print(f'  WAV file size: {size} bytes')
            else:
                print('  WAV file NOT created')
            
        except Exception as e:
            print(f'  Sentence {i+1} error: {type(e).__name__}: {e}')
            import traceback
            traceback.print_exc()
    
    return original_method(full_text, sentences)

tts._speak_incremental = traced_speak_incremental

import tempfile
tmpdir = tempfile.TemporaryDirectory()
tmp = Path(tmpdir.name)
model_path = tmp / 'voice.onnx'
model_path.write_bytes(b'not a real model')
tts.model_path = model_path

start = time.time()
result = tts.speak('Hello! Welcome to I.T.S Engineering College. How can I help you today?')
elapsed = time.time() - start
print(f'\nTotal speak() time: {elapsed:.2f}s')
print(f'Result: {result}')

tmpdir.cleanup()