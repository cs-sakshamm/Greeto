"""
Benchmark script for Phase 4 Speech Pipeline performance measurement.

Measures:
1. Microphone initialization / device resolution time.
2. Whisper model loading time.
3. Audio preprocessing / resampling duration.
4. End-of-speech detection delay.
5. Whisper inference latency on 1s, 3s, and 5s audio samples.
6. Total turnaround time from end-of-speech to transcription.
7. Process memory before and after Whisper model load.
"""

from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

def get_memory_mb():
    try:
        import subprocess
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", f"(Get-Process -Id {os.getpid()}).WorkingSet64 / 1MB"],
            stderr=subprocess.DEVNULL,
        )
        return float(out.strip())
    except Exception:
        return 0.0


def benchmark():
    print("==================================================================")
    print("Greeto Phase 4 Speech Pipeline Performance Benchmark")
    print("==================================================================")

    mem_baseline = get_memory_mb()
    print(f"Initial Process Memory: {mem_baseline:.2f} MB")

    # 1. Device resolution & capture init
    from app.speech.audio_capture import AudioCapture, resample_audio
    t0 = time.perf_counter()
    devices = AudioCapture.list_input_devices()
    t_enum = (time.perf_counter() - t0) * 1000
    print(f"1. Audio Device Enumeration Time: {t_enum:.2f} ms (Found {len(devices)} input devices)")

    t0 = time.perf_counter()
    dev_resolved = AudioCapture.resolve_device(None)
    t_resolve = (time.perf_counter() - t0) * 1000
    print(f"2. Audio Device Resolution Time: {t_resolve:.2f} ms (Default device: {dev_resolved})")

    # 2. Resampling performance (48kHz -> 16kHz for 5 seconds)
    sr_48k = 48000
    audio_48k = np.random.randn(sr_48k * 5).astype(np.float32)
    t0 = time.perf_counter()
    resampled = resample_audio(audio_48k, 48000, 16000)
    t_resample = (time.perf_counter() - t0) * 1000
    print(f"3. Audio Resampling (5s 48kHz -> 16kHz): {t_resample:.2f} ms ({len(resampled)} samples)")

    # 3. VAD processing latency
    from app.speech.vad import VoiceActivityDetector
    vad = VoiceActivityDetector(sample_rate=16000, energy_threshold=0.01)
    chunk_1024 = np.random.randn(1024).astype(np.float32) * 0.1
    t0 = time.perf_counter()
    for _ in range(100):
        _ = vad.is_speech(chunk_1024)
    t_vad = ((time.perf_counter() - t0) / 100.0) * 1000
    print(f"4. VAD Decision Latency (per 64ms chunk): {t_vad:.3f} ms")

    # 4. Whisper Model Loading
    from app.speech.stt import SpeechToText
    model_path = Path("models/whisper/ggml-tiny.en.bin")
    if not model_path.exists():
        print("ERROR: Whisper model not found at", model_path)
        return

    mem_before_load = get_memory_mb()
    t0 = time.perf_counter()
    stt = SpeechToText(model_path=str(model_path), n_threads=4)
    t_model_load = (time.perf_counter() - t0) * 1000
    mem_after_load = get_memory_mb()
    print(f"5. Whisper Model Load Time (ggml-tiny.en.bin): {t_model_load:.2f} ms ({t_model_load/1000:.3f}s)")
    print(f"   Memory Delta for Whisper Model: {mem_after_load - mem_before_load:.2f} MB (Total: {mem_after_load:.2f} MB)")

    # 5. Whisper Inference Latency on synthetic question audio
    # Synthetic speech tone + noise at 16kHz
    for duration_s in (1.0, 3.0, 5.0):
        t = np.linspace(0, duration_s, int(16000 * duration_s), endpoint=False)
        audio_speech = (np.sin(2 * np.pi * 300 * t) * 0.4 + np.random.randn(len(t)) * 0.01).astype(np.float32)

        t0 = time.perf_counter()
        _ = stt.transcribe(audio_speech)
        t_infer = (time.perf_counter() - t0) * 1000
        rtf = (t_infer / 1000.0) / duration_s
        print(f"6. Whisper Inference Latency ({duration_s:.1f}s audio): {t_infer:.2f} ms (RTF: {rtf:.3f}x)")

    # 6. Pipeline Turnaround Benchmark
    from app.speech.speech_service import SpeechService
    from app.core.container import ServiceContainer
    container = ServiceContainer()
    speech_svc = SpeechService(container)
    speech_svc.stt = stt

    # Test processing turnaround
    test_audio = np.random.randn(16000 * 2).astype(np.float32) * 0.2
    t0 = time.perf_counter()
    speech_svc._process_recorded_audio(test_audio)
    t_turnaround = (time.perf_counter() - t0) * 1000
    print(f"7. Total Pipeline Processing Turnaround (2s audio): {t_turnaround:.2f} ms")

    print("==================================================================")
    print("Benchmark Complete")
    print("==================================================================")

if __name__ == "__main__":
    benchmark()
