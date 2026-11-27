# Phase 4 Speech Recognition and Audio Pipeline Report

Offline AI Admission Receptionist Robot — I.T.S Engineering College

---

## Executive Summary

Phase 4 stabilized Greeto's complete speech-recognition pipeline, transitioning it from a rudimentary sounddevice hook into an offline, fault-tolerant speech ingestion and transcription engine.

All milestones were implemented, benchmarked, and verified through automated tests without breaking existing working functionality or introducing cloud dependencies.

---

## A. Initial Audit

### 1. Existing Microphone Implementation
- Previously, `AudioCapture` utilized `sounddevice.InputStream` with no input device enumeration, device resolution, or validation.
- Unbounded queue growth was present: if consumption lagged, memory grew indefinitely until failure.
- No automatic resampling: devices requiring native 44.1kHz or 48kHz caused PortAudio initialization crashes.
- Lacked graceful error callbacks when microphone devices were disconnected or unavailable.

### 2. Existing Whisper STT Implementation
- Model used: `pywhispercpp` with `models/whisper/ggml-tiny.en.bin` (offline GGML format).
- Had no domain adaptation: lacked college terminology initial prompts, causing inaccurate recognition of Indian names, courses (AKTU, B.Tech, CSE), and departments.
- Hallucination vulnerability: Whisper's silence tokens (`[BLANK_AUDIO]`, `[silence]`, `(music)`, `Thank you.`, `Thanks for watching!`) were returned as valid text, triggering unwarranted AI responses.
- Lacked thread synchronization around model inference calls.

### 3. Existing Pipeline & Event Flow
- `SpeechService` ran a continuous polling loop.
- No pre-speech buffer: first syllables of visitor questions were clipped before VAD triggered.
- No streak verification: single noise clicks/coughs triggered spurious `RECORDING` state transitions.
- Emitted `speech.recording_started` repeatedly on every audio chunk while speaking.
- Missing `speech.recording_finished` lifecycle event.
- No acoustic echo suppression: when Greeto spoke via `TTSService`, the robot's microphone captured its own voice and transcribed it as a user question (acoustic feedback loop).
- Synchronous transcription inside the capture loop blocked chunk processing during inference.

### 4. Architecture Preserved
- `TTSService` remains the sole shared speech-output owner.
- `ConversationManager` event contracts preserved (`speech_recognized` with `text`).
- `EventBus` decoupled publish/subscribe architecture strictly preserved.
- Full offline operation maintained with no external network requests or runtime downloads.

---

## B. Implementation

### Files Created
- `tests/test_audio_phase4.py`: 20 comprehensive unit and integration checks covering all milestones.
- `scripts/benchmark_speech_phase4.py`: Deterministic latency, throughput, and memory profiling suite.

### Files Modified
- `app/speech/audio_capture.py`:
  - Device enumeration (`list_input_devices`) and fuzzy/indexed resolution (`resolve_device`).
  - Automatic software resampling (48kHz/44.1kHz to 16kHz) via `scipy.signal.resample_poly` and linear fallback.
  - Multi-channel to mono averaging.
  - Bounded FIFO queue with oldest-chunk drop on saturation.
  - Clean lifecycle, stream management, and `on_error` callback.
- `app/speech/vad.py`:
  - WebRTC VAD integration with configurable aggressiveness (0–3) and frame duration (10/20/30 ms).
  - Energy threshold (RMS) gating to reject silent and low-energy ambient room noise.
  - Detailed frame analysis and chunk metrics (`calculate_energy`, `analyze_chunk`).
- `app/speech/stt.py`:
  - Offline model loading with project-relative path resolution.
  - College domain prompt injection (`initial_prompt`) for I.T.S Engineering College terminology.
  - Thread-safe inference locking (`_lock`).
  - Strict text normalization, silence suppression, hallucination rejection, and length bounds.
- `app/speech/speech_service.py`:
  - State machine: `IDLE`, `LISTENING`, `RECORDING`, `PROCESSING`, `ERROR`.
  - Circular pre-speech buffer (ring buffer deque) preserving initial sentence phonemes.
  - Speech streak counter preventing brief ambient impulse noises from triggering recordings.
  - End-of-speech silence segmentation with configurable duration.
  - Acoustic feedback suppression: pauses capture and adds echo cooldown during TTS speaking.
  - Asynchronous background transcription queue ensuring audio capture never drops frames.
  - Emits full lifecycle events: `speech.listening_started`, `speech.recording_started`, `speech.recording_finished`, `speech.transcription_started`, `speech.transcription_completed`, `speech.transcription_failed`, `speech.no_speech_detected`, `speech.error`.
- `app/gui/kiosk_presenter.py`:
  - Subscribes to `speech.recording_started`, `speech.transcription_started`, and `speech.no_speech_detected`.
  - Sets mascot face to "listening" when speech begins, and "thinking" during STT transcription.
  - Cleans up subscriptions and state on shutdown.
- `config/speech.yaml`:
  - Full configuration parameters for audio, VAD, STT, and pipeline coordination.

---

## C. Configuration

File: `config/speech.yaml`

```yaml
audio:
  sample_rate: 16000
  channels: 1
  dtype: float32
  chunk_size: 1024
  device: null
  max_recording_seconds: 15.0
  min_recording_seconds: 0.3
  queue_max_chunks: 100
  auto_resample: true

vad:
  aggressiveness: 2
  frame_ms: 30
  energy_threshold: 0.01
  min_speech_ms: 200
  silence_ms: 700
  pre_speech_ms: 300

stt:
  model_path: models/whisper/ggml-tiny.en.bin
  language: en
  n_threads: 4
  initial_prompt: "I.T.S Engineering College, Greater Noida, APJ Abdul Kalam Technical University, AKTU, B.Tech, CSE, Computer Science, ECE, Electronics, ME, Mechanical, Civil, MBA, MCA, admission, fees, eligibility, hostel, placement, registrar, scholarship, campus, Greeto."
  max_text_length: 500
  min_text_length: 3

pipeline:
  publish_events: true
  max_concurrent_transcriptions: 1
  mute_during_speech: true
  speech_cooldown_seconds: 0.3
```

---

## D. Testing

### 1. Phase 4 Test Suite (`tests/test_audio_phase4.py`)
- **M1. AudioCapture initialization & defaults**: PASS (0.00s)
- **M2. AudioCapture device enumeration**: PASS (0.00s)
- **M3. AudioCapture invalid device resolution**: PASS (0.00s)
- **M4. AudioCapture bounded queue drop-oldest**: PASS (0.00s)
- **M5. AudioCapture resampling (48kHz -> 16kHz)**: PASS (0.00s)
- **M6. AudioCapture stop & queue draining**: PASS (0.00s)
- **M7. AudioCapture error callback propagation**: PASS (0.00s)
- **V1. VAD empty and None input safety**: PASS (0.00s)
- **V2. VAD silence below energy threshold rejected**: PASS (0.00s)
- **V3. VAD ambient room noise rejected**: PASS (0.00s)
- **V4. VAD human speech tone detection**: PASS (0.00s)
- **V5. VAD detailed chunk frame analysis**: PASS (0.00s)
- **W1. SpeechToText missing model file error handling**: PASS (0.00s)
- **W2. SpeechToText text normalization & hallucination rejection**: PASS (0.15s)
- **W3. SpeechToText live offline Whisper transcription**: PASS (0.49s)
- **S1. SpeechService lifecycle states (IDLE/LISTENING/STOP)**: PASS (0.13s)
- **S2. SpeechService VAD segmentation & pre-speech buffer**: PASS (0.10s)
- **S3. SpeechService acoustic feedback suppression during TTS**: PASS (0.15s)
- **S4. SpeechService dispatches valid speech to ConversationManager**: PASS (0.11s)
- **S5. SpeechService rejects empty/hallucinated transcription**: PASS (0.11s)

Total: **20 passed, 0 failed, 0 skipped**.

### 2. Regression Test Results
- `tests/test_speech.py`: 4 checks PASSED (0.13s)
- `tests/test_storage.py`: 10 checks PASSED (0.50s)
- `tests/test_enrollment.py`: 8 checks PASSED (0.15s)
- `tests/test_recognition.py`: 20 checks PASSED (0.10s)
- `tests/test_camera.py`: 9 checks PASSED, 1 skipped (live hardware camera flag) (3.20s)
- `tests/test_kiosk.py`: 13 checks PASSED, 1 skipped (stability flag) (83.0s)
- Compilation: `python -m compileall app config tests scripts` — 0 errors.

---

## E. Performance Measurements

Benchmarked using `scripts/benchmark_speech_phase4.py` on the development workstation (AMD Ryzen 7 7840HS, 16GB RAM):

| Metric | Measurement | Target Limit | Status |
|---|---|---|---|
| Audio Device Enumeration | 0.19 ms | < 50 ms | Well within budget |
| Audio Device Resolution | 0.00 ms | < 5 ms | Instantaneous |
| 5s Audio Resampling (48kHz -> 16kHz) | 3.81 ms | < 20 ms | 0.00076x real-time |
| VAD Decision Latency (64ms chunk) | 0.019 ms | < 2 ms | 0.0003x real-time |
| Whisper Model Loading (`ggml-tiny.en.bin`) | 253.86 ms | < 2.0 s | Fast startup (~0.25s) |
| Whisper Inference Latency (1.0s audio) | 350.07 ms | < 600 ms | RTF: 0.350x |
| Whisper Inference Latency (3.0s audio) | 389.20 ms | < 1000 ms | RTF: 0.130x |
| Whisper Inference Latency (5.0s audio) | 406.47 ms | < 1500 ms | RTF: 0.081x |
| Total Pipeline Turnaround (end of 2s speech) | 374.34 ms | < 800 ms | Sub-second response |
| Process Working Set Memory (after model load) | 232.96 MB | < 1.0 GB | Lightweight |
| Memory Delta for Whisper Model | 127.89 MB | < 300 MB | Highly compact |

---

## F. Compatibility

1. **TTSService Compatibility**:
   - Preserved singleton ownership in `app/main.py`.
   - Coordinated via EventBus (`speech.speaking_started` / `speech.speaking_finished`) to suppress microphone capture while the robot speaks, eliminating acoustic feedback loops.

2. **ConversationManager Compatibility**:
   - Valid questions are dispatched once via `speech_recognized`.
   - Empty, short, and hallucinated audio outputs are filtered before dispatch.
   - Question context generation, LLM generation, and history tracking remain intact.

3. **EventBus Compatibility**:
   - All event names and payloads follow established conventions.
   - Handlers clean up upon service `stop()`.

4. **Vision System Compatibility**:
   - Camera management, face recognition, and visitor sessions are unaffected.
   - Kiosk presentation maps both vision and speech states seamlessly.

5. **Jetson Xavier NX Readiness Assessment**:
   - `ggml-tiny.en.bin` requires ~128 MB RAM and under 400ms inference on 4 CPU threads.
   - The Jetson Xavier NX 8GB CPU (6-core Carmel ARMv8.2) is expected to execute this model in ~500–700ms on 4 threads, well within acceptable interactive limits for reception kiosk dialogue.

---

## G. Outstanding Issues

- Physical deployment on the Jetson Xavier NX 8GB hardware will require testing with the final USB microphone/array to determine ambient noise calibration in the physical reception hall.
- All hardware-independent tests pass deterministically.

---

## H. Phase 5 Readiness

The speech-input pipeline is stable, robust against noise and acoustic feedback, produces clean transcriptions, and communicates reliably with `ConversationManager`.

The system is fully prepared to proceed with the Qwen3-0.6B LLM migration in Phase 5.

### Verdict: **PHASE 4 COMPLETE**
