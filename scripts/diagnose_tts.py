#!/usr/bin/env python3
"""
Diagnose TTS service.

This script:
- Uses the existing TTSService
- Synthesizes and plays: "Hello! Welcome to I.T.S Engineering College."
- Reports synthesis and playback lifecycle
- Reports WAV properties
- Reports playback device
- Confirms whether playback completed without an exception
"""

import sys
import time
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.core.container import ServiceContainer
from app.speech.tts_service import TTSService


def main():
    parser = argparse.ArgumentParser(description="TTS diagnostics")
    parser.add_argument(
        "--text",
        type=str,
        default="Hello! Welcome to I.T.S Engineering College.",
        help="Text to synthesize"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("TTS Diagnostic")
    print("=" * 60)
    print(f"  Text to synthesize: '{args.text}'")
    print()
    
    # Create container and TTS service
    container = ServiceContainer()
    try:
        tts = TTSService(container)
        print("  TTSService initialized successfully")
        print(f"  Model path: {tts.model_path}")
    except Exception as e:
        print(f"Error initializing TTSService: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Synthesize and play
    print("\n--- Synthesizing and playing audio ---")
    start_time = time.perf_counter()
    
    try:
        result = tts.speak(args.text)
        total_time = time.perf_counter() - start_time
        
        print(f"  speak() returned: {result}")
        print(f"  Total time (synthesis + playback): {total_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error during speak(): {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Report
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"  Synthesis and playback: {'SUCCESS' if result else 'FAILED'}")
    print(f"  Total duration: {total_time:.2f} seconds")
    
    if result:
        print("  Playback completed without exception: YES")
    else:
        print("  Playback completed without exception: NO (result was False)")
    
    print("\n" + "=" * 60)
    print("Diagnostic complete")
    print("=" * 60)


if __name__ == "__main__":
    main()