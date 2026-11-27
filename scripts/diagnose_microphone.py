#!/usr/bin/env python3
"""
Diagnose microphone input devices and audio capture quality.

This script:
- Lists available input devices
- Shows the selected device
- Records a short sample
- Measures RMS, peak and clipping
- Saves a WAV file with verified sample rate and channels
- Reports audio duration and sample count
"""

import sys
import time
import argparse
from pathlib import Path

import numpy as np
import sounddevice as sd


def list_input_devices():
    """List all available audio input devices."""
    print("=" * 60)
    print("Available Audio Input Devices")
    print("=" * 60)
    
    devices = sd.query_devices()
    default_idx = sd.default.device[0] if sd.default.device else None
    
    for idx, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) > 0:
            is_default = " (DEFAULT)" if idx == default_idx else ""
            print(f"  Index {idx}: {dev.get('name', f'Device {idx}')}{is_default}")
            print(f"    Input channels: {dev.get('max_input_channels', 1)}")
            print(f"    Default sample rate: {dev.get('default_samplerate', 16000)} Hz")
    
    print()
    return default_idx


def record_sample(device_idx=None, duration=3.0, sample_rate=16000):
    """Record a short audio sample and measure quality metrics."""
    print("=" * 60)
    print(f"Recording {duration}s sample at {sample_rate}Hz...")
    print("=" * 60)
    
    try:
        # Record audio
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            device=device_idx,
            dtype='float32'
        )
        sd.wait()  # Wait until recording is finished
        
        # Calculate audio quality metrics
        audio = recording[:, 0] if recording.ndim > 1 else recording.flatten()
        rms = np.sqrt(np.mean(audio ** 2))
        peak = np.max(np.abs(audio))
        # Clipping detection
        clipping_pct = np.sum(np.abs(audio) >= 0.95) / len(audio) * 100
        duration_s = len(audio) / sample_rate
        sample_count = len(audio)
        
        print("=" * 60)
        print("Audio Quality Metrics")
        print("=" * 60)
        print(f"  Duration: {duration_s:.2f} seconds")
        print(f"  Sample count: {sample_count}")
        print(f"  RMS amplitude: {rms:.6f}")
        print(f"  Peak amplitude: {peak:.6f}")
        print(f"  Clipping percentage: {clipping_pct:.2f}%")
        print(f"  Sample values range: [{np.min(audio):.6f}, {np.max(audio):.6f}]")
        
        # Save WAV file
        wav_path = Path("mic_sample.wav")
        import soundfile as sf
        sf.write(str(wav_path), audio, sample_rate)
        print(f"  Sample saved to: {wav_path}")
        print(f"  WAV properties: {sample_rate}Hz, 1 channel, {duration_s:.2f}s")
        
        return {
            "duration_s": duration_s,
            "sample_count": sample_count,
            "rms": float(rms),
            "peak": float(peak),
            "clipping_pct": float(clipping_pct),
            "sample_rate": sample_rate,
            "channels": 1,
        }
        
    except Exception as e:
        print(f"Error during recording: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Microphone diagnostics")
    parser.add_argument(
        "--device", 
        type=int, 
        default=None,
        help="Input device index (None for default)"
    )
    parser.add_argument(
        "--duration", 
        type=float, 
        default=3.0,
        help="Recording duration in seconds"
    )
    parser.add_argument(
        "--sample-rate", 
        type=int, 
        default=16000,
        help="Sample rate in Hz"
    )
    
    args = parser.parse_args()
    
    print("\n--- Step 1: List input devices ---")
    default_idx = list_input_devices()
    
    # Resolve device
    device_idx = args.device
    if device_idx is None:
        device_idx = default_idx
    
    print(f"\n--- Step 2: Record sample using device {device_idx} ---")
    metrics = record_sample(
        device_idx=device_idx,
        duration=args.duration,
        sample_rate=args.sample_rate
    )
    
    if metrics:
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"Device: {device_idx}")
        print(f"Sample rate: {metrics['sample_rate']} Hz")
        print(f"Channels: {metrics['channels']}")
        print(f"Duration: {metrics['duration_s']:.2f} s")
        print(f"RMS: {metrics['rms']:.6f}")
        print(f"Peak: {metrics['peak']:.6f}")
        print(f"Clipping: {metrics['clipping_pct']:.2f}%")
        
        # Check if audio is usable
        if metrics['rms'] < 0.001:
            print("\nWARNING: Very low RMS - microphone may be too quiet or not picking up speech")
        if metrics['clipping_pct'] > 50:
            print("\nWARNING: High clipping percentage - microphone signal may be too loud")
    
    print("\nDone.")


if __name__ == "__main__":
    main()