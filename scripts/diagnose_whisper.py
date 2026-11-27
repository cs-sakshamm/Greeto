#!/usr/bin/env python3
"""
Diagnose Whisper STT backend.

This script:
- Accepts a WAV file
- Runs the configured STT backend
- Shows raw transcription
- Shows cleaned transcription
- Shows rejection reason
- Shows inference latency
- Shows GPU backend information
"""

import sys
import time
import argparse
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.speech.stt import SpeechToText


def main():
    parser = argparse.ArgumentParser(description="Whisper STT diagnostics")
    parser.add_argument(
        "wav_file",
        type=str,
        help="Path to WAV file to transcribe"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/whisper/ggml-tiny.en.bin",
        help="Path to Whisper model"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language code for STT"
    )
    parser.add_argument(
        "--n-threads",
        type=int,
        default=4,
        help="Number of threads for Whisper"
    )
    
    args = parser.parse_args()
    
    wav_path = Path(args.wav_file)
    if not wav_path.exists():
        print(f"Error: WAV file not found: {wav_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("Whisper STT Diagnostic")
    print("=" * 60)
    print(f"  WAV file: {wav_path}")
    print(f"  Model: {args.model_path}")
    print(f"  Language: {args.language}")
    print(f"  Threads: {args.n_threads}")
    print()
    
    # Load the WAV file
    try:
        import soundfile as sf
        audio, sr = sf.read(str(wav_path))
        # Convert to mono if needed
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        # Ensure float32
        audio = audio.astype(np.float32)
        # Resample if needed
        if sr != 16000:
            print(f"  Note: Input sample rate is {sr}Hz, resampling to 16000Hz")
            # Simple resampling
            num_target = int(len(audio) * 16000 / sr)
            orig_indices = np.linspace(0, len(audio) - 1, len(audio))
            target_indices = np.linspace(0, len(audio) - 1, num_target)
            from scipy import signal
            audio = signal.resample_poly(audio, 16000, sr).astype(np.float32)
        print(f"  Audio loaded: {len(audio)} samples at effectively {16000 if sr != 16000 else sr}Hz")
    except Exception as e:
        print(f"Error loading WAV file: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Initialize STT
    print("\n--- Initializing Whisper STT ---")
    try:
        stt = SpeechToText(
            model_path=args.model_path,
            n_threads=args.n_threads,
            language=args.language,
        )
        print(f"  STT available: {stt.is_available()}")
        print(f"  Model path: {stt.model_path}")
        print(f"  Language: {stt.language}")
    except Exception as e:
        print(f"Error initializing STT: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Transcribe
    print("\n--- Running Transcription ---")
    start_time = time.perf_counter()
    
    try:
        raw_text = stt.transcribe(audio)
    except Exception as e:
        print(f"Error during transcription: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    elapsed = time.perf_counter() - start_time
    
    # Determine if text was cleaned/ rejected
    cleaned = stt.clean_text(raw_text)
    
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"  Inference latency: {elapsed:.2f} seconds")
    print(f"  Raw transcription: '{raw_text}'")
    print(f"  Cleaned transcription: '{cleaned}'")
    print()
    
    # Report rejection reason
    if not cleaned:
        print("  Rejection reason: Transcription was filtered as hallucination/noise or too short")
        print("  - Check if the raw text matches known hallucination tokens")
        print("  - Check if the text is below min_text_length (3 chars)")
        print("  - Check if the text contains only repetitive words")
    else:
        print("  Status: Transcription accepted")
    
    # Show GPU/CPU backend info
    print("\n  Model information:")
    print(f"    Model file: {stt.model_path.name}")
    # Note: pywhispercpp backend info would require checking the model
    print(f"    Language code: {stt.language}")
    print(f"  Initial prompt length: {len(stt.initial_prompt) if stt.initial_prompt else 0} chars")
    
    print("\n" + "=" * 60)
    print("Diagnostic complete")
    print("=" * 60)


if __name__ == "__main__":
    main()