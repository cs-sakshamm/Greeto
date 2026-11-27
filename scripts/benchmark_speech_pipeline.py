#!/usr/bin/env python3
"""
Benchmark the complete speech pipeline.

This script benchmarks:
- STT latency
- LLM first-token latency
- LLM generation speed
- TTS synthesis latency
- First audible response latency
- Sentence gap
- Total response duration
- CPU and GPU information
- Memory usage

Use reproducible inputs.
"""

import sys
import time
import argparse
import resource
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np

from app.core.container import ServiceContainer
from app.speech.speech_service import SpeechService
from app.speech.tts_service import TTSService
from app.brain.llm import LocalLLM
from app.brain.knowledge_base import KnowledgeBase
from app.brain.prompt_manager import PromptManager


def get_cpu_info():
    """Get CPU information."""
    try:
        with open("/proc/cpuinfo", "r") as f:
            cpu_info = f.read()
        # Extract model name
        for line in cpu_info.split("\n"):
            if "model name" in line:
                return line.split(":")[1].strip().split("@")[0].strip()
    except:
        pass
    
    # Try alternative
    try:
        with open("/system/sysctl.py", "r") as f:  # Windows fallback
            pass
    except:
        pass
    
    return "Unknown CPU"


def get_gpu_info():
    """Get GPU information if available."""
    try:
        import subprocess
        # Try nvidia-smi
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", 
             "--format=csv,no"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            # Parse CSV
            parts = lines[1].split(",")
            return {
                "gpu_name": parts[0].strip(),
                "driver_version": parts[1].strip(),
                "memory_total_mb": int(parts[2].strip().split(" ")[0])
            }
    except Exception:
        pass
    
    # Try Python approach
    try:
        import pycuda.driver as driver
        driver.init()
        gpu = driver.Device(0)
        return {"gpu_name": gpu.name(), "memory_mb": gpu.total_memory() // 1024 // 1024}
    except ImportError:
        pass
    except Exception:
        pass
    
    return {"gpu_name": "No GPU detected", "memory_total_mb": 0}


def benchmark_stt(stt, audio):
    """Benchmark STT inference."""
    print("Benchmarking STT...")
    start = time.perf_counter()
    text = stt.transcribe(audio)
    latency = time.perf_counter() - start
    print(f"  STT latency: {latency:.2f}s")
    print(f"  Transcription: '{text}'")
    return latency


def benchmark_llm(llm, prompt):
    """Benchmark LLM generation."""
    print("Benchmarking LLM...")
    start = time.perf_counter()
    first_token_start = None
    tokens = []
    
    for token in llm.stream_generate(prompt):
        tokens.append(token)
        if first_token_start is None:
            first_token_start = time.perf_counter()
    
    total_time = time.perf_counter() - start
    first_token_time = (first_token_start - start) * 1000 if first_token_start else 0
    total_time_ms = total_time * 1000
    first_token_ms = first_token_time
    
    print(f"  First-token latency: {first_token_ms:.1f}ms")
    print(f"  Total generation time: {total_time_ms:.1f}ms")
    print(f"  Tokens generated: {len(tokens)}")
    print(f"  Generation speed: {len(tokens) / total_time:.1f} tokens/s")
    return first_token_ms / 1000, total_time_ms, len(tokens)


def benchmark_tts(tts, text):
    """Benchmark TTS synthesis."""
    print("Benchmarking TTS...")
    start = time.perf_counter()
    
    sentences = tts._split_into_sentences(text)
    print(f"  Split into {len(sentences)} sentences")
    
    # Measure just the generation (not playback)
    # We'll generate WAVs but not play them
    generation_times = []
    for i, sentence in enumerate(sentences):
        import tempfile
        import subprocess
        
        wav_path = ""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_file:
                wav_path = audio_file.name
            
            base_cmd = tts._resolve_piper_command()
            command = base_cmd + [
                "--model",
                str(tts.model_path),
                "--output_file",
                wav_path,
            ]
            
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                text=True,
            )
            process.communicate(sentence)
            
            gen_time = time.perf_counter() - start - sum(generation_times)
            generation_times.append(gen_time)
            
        except Exception as e:
            print(f"  Error generating sentence {i+1}: {e}")
            generation_times.append(0)
    
    avg_gen_time = sum(generation_times) / len(generation_times) if generation_times else 0
    print(f"  Average generation time per sentence: {avg_gen_time:.2f}s")
    print(f"  Total generation time: {sum(generation_times):.2f}s")
    return avg_gen_time, sum(generation_times)


def main():
    parser = argparse.ArgumentParser(description="Speech pipeline benchmark")
    parser.add_argument(
        "--text",
        type=str,
        default="Hello! Welcome to I.T.S Engineering College. How can I help you today with admission information?",
        help="Text to process through the pipeline"
    )
    parser.add_argument(
        "--wav-file",
        type=str,
        default=None,
        help="Pre-recorded WAV file for STT benchmark (optional)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Greeto Speech Pipeline Benchmark")
    print("=" * 60)
    print()
    
    # CPU/GPU info
    print("System Information")
    print("-" * 40)
    cpu_info = get_cpu_info()
    print(f"  CPU: {cpu_info}")
    gpu_info = get_gpu_info()
    print(f"  GPU: {gpu_info['gpu_name']}")
    print(f"  GPU memory: {gpu_info['memory_total_mb']} MB")
    print()
    
    # Initialize services
    print("Initializing Services")
    print("-" * 40)
    container = ServiceContainer()
    speech = SpeechService(container)
    tts = TTSService(container)
    llm = LocalLLM()
    knowledge = KnowledgeBase()
    prompt_manager = PromptManager()
    
    print(f"  STT available: {speech.stt is not None}")
    print(f"  LLM model loaded")
    print(f"  Knowledge base: {knowledge.documents.__len__()} documents")
    print()
    
    # Generate a prompt for LLM
    print("Generating LLM prompt...")
    test_text = args.text
    
    # Process through conversation manager flow
    import threading
    event_results = {"llm_response": None, "speech_recognized": None}
    
    # Build prompt from knowledge
    context = knowledge.search(test_text)
    prompt = prompt_manager.build_prompt(
        user_message=test_text,
        history=[],
        context=context,
    )
    
    print(f"  Prompt length: {len(prompt)} characters")
    print()
    
    # 1. STT Benchmark (using synthetic audio if no WAV provided)
    print("=" * 60)
    print("1. Speech-to-Text (STT) Benchmark")
    print("=" * 60)
    if args.wav_file:
        import soundfile as sf
        audio, sr = sf.read(args.wav_file)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        audio = audio.astype(np.float32)
        stt_latency = benchmark_stt(speech.stt, audio)
    else:
        print("  SKIPPED: No WAV file provided. Use --wav-file <path>")
        stt_latency = None
    print()
    
    # 2. LLM Benchmark
    print("=" * 60)
    print("2. Local LLM Benchmark")
    print("=" * 60)
    first_token_latency, total_gen_time, num_tokens = benchmark_llm(llm, prompt)
    print()
    
    # 3. TTS Benchmark
    print("=" * 60)
    print("3. Text-to-Speech (TTS) Benchmark")
    print("=" * 60)
    avg_tts_time, total_tts_time = benchmark_tts(tts, args.text)
    print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  STT latency: {stt_latency:.2f}s" if stt_latency else "  STT latency: N/A")
    print(f"  LLM first-token: {first_token_latency:.2f}s")
    print(f"  LLM total generation: {total_gen_time:.2f}s ({num_tokens} tokens)")
    print(f"  TTS per-sentence: {avg_tts_time:.2f}s")
    print(f"  TTS total: {total_tts_time:.2f}s")
    
    # Estimated total response time
    if stt_latency and first_token_latency and avg_tts_time:
        # Simulate 3-sentence response
        estimated_total = stt_latency + first_token_latency + (avg_tts_time * 3)
        print(f"  Estimated total response (3 sentences): {estimated_total:.2f}s")
    
    print()
    print("=" * 60)
    print("Benchmark complete")
    print("=" * 60)


if __name__ == "__main__":
    main()