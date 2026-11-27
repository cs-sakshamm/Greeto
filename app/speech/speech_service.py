"""
Speech Service.

Continuous speech recognition pipeline with Voice Activity Detection,
end-of-speech segmentation, offline Whisper transcription, and
EventBus coordination.
"""

from __future__ import annotations

import collections
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from app.core.container import ServiceContainer
from app.speech.audio_capture import AudioCapture
from app.speech.vad import VoiceActivityDetector
from app.speech.stt import SpeechToText

# Repository root, independent of the current working directory. Used when
# writing the last recorded utterance for microphone-level diagnosis.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

logger = logging.getLogger(__name__)


class SpeechService:
    """
    Manages continuous microphone listening, voice segmentation,
    offline Whisper speech-to-text, and conversational event dispatching.
    """

    STATE_IDLE = "IDLE"
    STATE_LISTENING = "LISTENING"
    STATE_RECORDING = "RECORDING"
    STATE_PROCESSING = "PROCESSING"
    STATE_ERROR = "ERROR"

    def __init__(self, container: ServiceContainer) -> None:
        self.container = container
        self.logger = getattr(container, "logger", logger)
        self.event_bus = getattr(container, "event_bus", None)
        self.config = getattr(container, "config", None)

        # Audio capture settings
        sr = self._cfg("speech", "audio", "sample_rate", default=16000)
        chunk = self._cfg("speech", "audio", "chunk_size", default=1024)
        device = self._cfg("speech", "audio", "device", default=None)
        qmax = self._cfg("speech", "audio", "queue_max_chunks", default=100)

        self.sample_rate = sr
        self.chunk_size = chunk
        self.chunk_duration_s = chunk / sr

        self.audio_capture = AudioCapture(
            sample_rate=sr,
            chunk_size=chunk,
            device=device,
            queue_max_chunks=qmax,
            on_error=self._on_audio_error,
        )

        # VAD settings
        vad_agg = self._cfg("speech", "vad", "aggressiveness", default=2)
        frame_ms = self._cfg("speech", "vad", "frame_ms", default=30)
        energy_thresh = self._cfg("speech", "vad", "energy_threshold", default=0.0035)
        self.vad = VoiceActivityDetector(
            sample_rate=sr,
            aggressiveness=vad_agg,
            frame_ms=frame_ms,
            energy_threshold=energy_thresh,
        )

        # STT settings
        model_path = self._cfg(
            "speech", "stt", "model_path", default="models/whisper/ggml-tiny.en.bin"
        )
        n_threads = self._cfg("speech", "stt", "n_threads", default=4)
        language = self._cfg("speech", "stt", "language", default="en")
        initial_prompt = self._cfg("speech", "stt", "initial_prompt", default=None)
        min_text_len = self._cfg("speech", "stt", "min_text_length", default=3)
        max_text_len = self._cfg("speech", "stt", "max_text_length", default=500)

        try:
            self.stt = SpeechToText(
                model_path=model_path,
                n_threads=n_threads,
                language=language,
                initial_prompt=initial_prompt,
                min_text_length=min_text_len,
                max_text_length=max_text_len,
            )
        except Exception as error:  # noqa: BLE001
            self.logger.warning("STT initialization failed: %s", error)
            self.stt = None

        # Pipeline timing & segmentation thresholds
        # NOTE: These values come from config/speech.yaml and can be
        # adjusted at runtime using set_vad_parameters() for optimization.
        
        silence_ms = self._cfg("speech", "vad", "silence_ms", default=600)
        # 600ms (9 chunks) avoids clipping soft trailing syllables while
        # still ending an utterance promptly enough for natural turn-taking.
        self.max_silence_chunks = max(1, int((silence_ms / 1000.0) / self.chunk_duration_s))

        min_speech_ms = self._cfg("speech", "vad", "min_speech_ms", default=120)
        # 120ms (2 chunks) is short enough to capture brief greetings
        # ("hello", "hi") while still rejecting isolated noise bursts.
        self.min_speech_chunks = max(1, int((min_speech_ms / 1000.0) / self.chunk_duration_s))

        pre_speech_ms = self._cfg("speech", "vad", "pre_speech_ms", default=384)
        # 384ms (6 chunks) of look-back preserves unvoiced leading
        # consonants such as the "h" in "hello".
        pre_chunks = max(1, int((pre_speech_ms / 1000.0) / self.chunk_duration_s))
        self.pre_speech_buffer: collections.deque[np.ndarray] = collections.deque(maxlen=pre_chunks)

        min_rec_sec = self._cfg("speech", "audio", "min_recording_seconds", default=0.3)
        self.min_audio_length = int(sr * min_rec_sec)

        max_rec_sec = self._cfg("speech", "audio", "max_recording_seconds", default=15.0)
        self.max_audio_length = int(sr * max_rec_sec)

        # Acoustic feedback / echo suppression
        self.mute_during_speech = self._cfg("speech", "pipeline", "mute_during_speech", default=True)
        self.speech_cooldown_s = self._cfg("speech", "pipeline", "speech_cooldown_seconds", default=0.3)
        self._tts_speaking = False
        self._tts_cooldown_until = 0.0

        # State management
        self.state = self.STATE_IDLE
        self.running = False
        self.microphone_ok = True
        self.thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Buffers for recording
        self.audio_buffer: list[np.ndarray] = []
        self.silence_counter = 0
        self.speech_streak = 0
        self.total_samples_recorded = 0

        # VAD performance tracking
        self._vad_stats = {
            "chunks_processed": 0,
            "speech_frames_total": 0,
            "silence_frames_total": 0,
            "first_speech_delay_ms": None,
        }

        # Dedicated transcription worker. Items are (audio, speech_end_t)
        # so the latency log measures from the real end of speech, not
        # from after the VAD tail and transcription.
        self._transcription_queue: queue.Queue = queue.Queue(maxsize=10)
        self._transcription_thread: Optional[threading.Thread] = None

        self._register_event_handlers()

    def _cfg(self, *keys, default=None):
        if self.config is not None and hasattr(self.config, "get_value"):
            try:
                val = self.config.get_value(*keys, default=default)
                return val if val is not None else default
            except Exception:  # noqa: BLE001
                return default
        return default

    def _register_event_handlers(self) -> None:
        """Subscribe to TTS events to avoid recording the robot's own voice."""
        if self.event_bus is not None:
            try:
                self.event_bus.subscribe("speech.synthesis_started", self._on_tts_started)
                self.event_bus.subscribe("speech.speaking_started", self._on_tts_started)
                self.event_bus.subscribe("speech.speaking_finished", self._on_tts_finished)
                self.event_bus.subscribe("speech.error", self._on_tts_error)
            except Exception as error:  # noqa: BLE001
                self.logger.warning("Failed to register speech event listeners: %s", error)

    def _on_tts_error(self, **kwargs) -> None:
        if kwargs.get("kind") == "tts":
            self._tts_speaking = False

    def _on_tts_started(self, **kwargs) -> None:
        if self.mute_during_speech:
            self._tts_speaking = True
            with self._lock:
                self.audio_buffer.clear()
                self.pre_speech_buffer.clear()
                self.silence_counter = 0
                self.speech_streak = 0
                self.total_samples_recorded = 0
                if self.state == self.STATE_RECORDING:
                    self._set_state(self.STATE_LISTENING)

    def _on_tts_finished(self, **kwargs) -> None:
        if self.mute_during_speech:
            self._tts_speaking = False
            self._tts_cooldown_until = time.monotonic() + self.speech_cooldown_s

    def set_vad_parameters(self, silence_ms: Optional[int] = None, 
                          min_speech_ms: Optional[int] = None, 
                          pre_speech_ms: Optional[int] = None) -> Dict[str, Any]:
        """
        Adjust VAD parameters at runtime for latency/accuracy optimization.

        Parameters:
            silence_ms: Silence threshold in ms before speech ends.
                Lower values reduce latency but may truncate soft trailing words.
                Increase for noisy environments, decrease for faster response.
            min_speech_ms: Minimum speech duration before starting to record.
                Lower values start sooner but may capture transient noise.
                Increase for noisy environments to avoid false starts.
            pre_speech_ms: Pre-speech buffer duration in ms.
                Captures audio before voice detection to preserve sentence starts.
                Increase if beginning of words are being truncated.

        Returns:
            Dictionary with the updated parameters and their computed chunk counts.
        """
        import time
        
        updated = {
            "silence_ms": silence_ms,
            "min_speech_ms": min_speech_ms,
            "pre_speech_ms": pre_speech_ms,
        }
        
        if silence_ms is not None:
            self.max_silence_chunks = max(1, int((silence_ms / 1000.0) / self.chunk_duration_s))
            updated["silence_chunks"] = self.max_silence_chunks
            updated["silence_ms_actual"] = self.max_silence_chunks * self.chunk_duration_s * 1000
        
        if min_speech_ms is not None:
            self.min_speech_chunks = max(1, int((min_speech_ms / 1000.0) / self.chunk_duration_s))
            updated["min_speech_chunks"] = self.min_speech_chunks
            updated["min_speech_ms_actual"] = self.min_speech_chunks * self.chunk_duration_s * 1000
        
        if pre_speech_ms is not None:
            # Recalculate pre-speech buffer chunks
            pre_chunks_new = max(1, int((pre_speech_ms / 1000.0) / self.chunk_duration_s))
            self.pre_speech_buffer = collections.deque(
                list(self.pre_speech_buffer), maxlen=pre_chunks_new
            )
            self.pre_speech_buffer.maxlen = pre_chunks_new
            updated["pre_speech_chunks"] = pre_chunks_new
            updated["pre_speech_ms_actual"] = pre_chunks_new * self.chunk_duration_s * 1000
        
        # Log the changes for monitoring
        self.logger.info(
            "VAD parameters updated: %s", updated
        )
        
        return updated

    def _on_audio_error(self, message: str) -> None:
        self.microphone_ok = False
        self._set_state(self.STATE_ERROR)
        self.logger.error("Audio capture error: %s", message)
        self._publish("speech.error", kind="microphone", message=message)

    def _publish(self, event_name: str, **payload) -> None:
        if self.event_bus is not None:
            try:
                self.event_bus.publish(event_name, **payload)
            except Exception as error:  # noqa: BLE001
                self.logger.warning("Event publish failed for %s: %s", event_name, error)

    def _set_state(self, state: str) -> None:
        self.state = state

    def start(self) -> None:
        """Start audio capture and listening threads."""
        if self.running:
            return
        self.logger.info("Starting Speech Service...")
        self.running = True

        # Start transcription worker thread
        self._transcription_thread = threading.Thread(
            target=self._transcription_worker,
            daemon=True,
            name="STTWorker",
        )
        self._transcription_thread.start()

        # Start audio capture
        self.audio_capture.start()

        # Start listen loop
        self.thread = threading.Thread(
            target=self._listen_loop,
            daemon=True,
            name="SpeechService",
        )
        self.thread.start()

        self._set_state(self.STATE_LISTENING)
        self._publish("speech.listening_started")
        self.logger.info("Speech Service started and listening.")

    def stop(self) -> None:
        """Stop listening and release all resources."""
        self.running = False
        self.audio_capture.stop()

        # Signal transcription worker to stop
        try:
            self._transcription_queue.put_nowait(None)
        except queue.Full:
            pass

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.thread = None

        if self._transcription_thread is not None and self._transcription_thread.is_alive():
            self._transcription_thread.join(timeout=2.0)
        self._transcription_thread = None

        with self._lock:
            self.audio_buffer.clear()
            self.pre_speech_buffer.clear()
            self._set_state(self.STATE_IDLE)

        # Unsubscribe event listeners
        if self.event_bus is not None:
            for ev, handler in (
                ("speech.synthesis_started", self._on_tts_started),
                ("speech.speaking_started", self._on_tts_started),
                ("speech.speaking_finished", self._on_tts_finished),
                ("speech.error", self._on_tts_error),
            ):
                try:
                    self.event_bus.unsubscribe(ev, handler)
                except Exception:
                    pass

        self.logger.info("Speech Service stopped.")

    def _listen_loop(self) -> None:
        """Main listening loop polling audio chunks from capture queue."""
        while self.running:
            audio = self.audio_capture.get_audio(timeout=0.2)
            if audio is None:
                continue

            try:
                self._handle_chunk(audio)
            except Exception as error:  # noqa: BLE001
                self.logger.exception("Error processing audio chunk: %s", error)
                self._publish("speech.error", kind="pipeline", message=str(error))

    def _handle_chunk(self, audio: np.ndarray) -> None:
        """Process one incoming audio chunk through VAD and state transitions."""
        # Check acoustic feedback suppression
        if self.mute_during_speech:
            if self._tts_speaking or (time.monotonic() < self._tts_cooldown_until):
                return

        speaking = self.vad.is_speech(audio)

        with self._lock:
            if self.state == self.STATE_LISTENING:
                if speaking:
                    self.speech_streak += 1
                    if self.speech_streak >= self.min_speech_chunks:
                        # Meaningful speech start confirmed!
                        self._set_state(self.STATE_RECORDING)
                        self._publish("speech.recording_started")

                        # Prepend pre-speech ring buffer so beginning of sentence is preserved
                        self.audio_buffer = list(self.pre_speech_buffer)
                        self.audio_buffer.append(audio)
                        self.total_samples_recorded = sum(c.size for c in self.audio_buffer)
                        self.silence_counter = 0
                else:
                    self.speech_streak = 0
                    self.pre_speech_buffer.append(audio)

            elif self.state == self.STATE_RECORDING:
                self.audio_buffer.append(audio)
                self.total_samples_recorded += audio.size

                if speaking:
                    self.silence_counter = 0
                else:
                    self.silence_counter += 1

                # Check max duration cutoff
                reached_max = self.total_samples_recorded >= self.max_audio_length
                # Check end-of-speech silence cutoff
                reached_silence = self.silence_counter >= self.max_silence_chunks

                if reached_max or reached_silence:
                    recorded_audio = np.concatenate(self.audio_buffer)
                    duration_s = len(recorded_audio) / self.sample_rate

                    # Endpointing confirms silence only after the VAD tail;
                    # the visitor actually stopped speaking that much
                    # earlier. Subtracting it keeps the latency log honest.
                    if reached_silence and not reached_max:
                        speech_end_t = time.perf_counter() - (
                            self.max_silence_chunks * self.chunk_duration_s
                        )
                    else:
                        speech_end_t = time.perf_counter()

                    self._publish("speech.recording_finished", duration_seconds=duration_s)
                    self._set_state(self.STATE_PROCESSING)

                    # Reset recording state
                    self.audio_buffer.clear()
                    self.pre_speech_buffer.clear()
                    self.silence_counter = 0
                    self.speech_streak = 0
                    self.total_samples_recorded = 0

                    # Offload to async transcription queue
                    try:
                        self._transcription_queue.put_nowait(
                            (recorded_audio, speech_end_t)
                        )
                    except queue.Full:
                        self.logger.warning("Transcription queue full. Dropping audio.")
                        self._publish("speech.transcription_failed", message="Queue full")
                        self._set_state(self.STATE_LISTENING)

    def _transcription_worker(self) -> None:
        """Background worker consuming audio from queue and invoking Whisper STT."""
        while self.running:
            try:
                item = self._transcription_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if item is None:
                # Stop sentinel
                break

            # Backwards compatibility: tests may enqueue a bare array.
            if isinstance(item, tuple):
                audio, speech_end_t = item
            else:
                audio, speech_end_t = item, None

            try:
                self._process_recorded_audio(audio, speech_end_t)
            except Exception as error:  # noqa: BLE001
                self.logger.exception("Unexpected error in transcription worker: %s", error)
                self._publish("speech.transcription_failed", message=str(error))
                self._publish("speech.error", kind="stt", message=str(error))
            finally:
                with self._lock:
                    if self.state == self.STATE_PROCESSING:
                        self._set_state(self.STATE_LISTENING)

    def _process_recorded_audio(
        self,
        audio: np.ndarray,
        speech_end_t: float | None = None,
    ) -> None:
        """Execute transcription and validation on recorded speech."""
        duration_s = len(audio) / self.sample_rate
        rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) > 0 else 0.0

        # Save the last recorded utterance for physical microphone verification.
        # This gives ground truth (duration / RMS / actual waveform) whenever
        # a short greeting is misrecognized. Disable via config in production.
        if self._cfg("speech", "pipeline", "debug_save_audio", default=True):
            try:
                import scipy.io.wavfile as wavfile

                log_dir = PROJECT_ROOT / "logs"
                log_dir.mkdir(exist_ok=True)
                wav_pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
                wavfile.write(
                    str(log_dir / "debug_last_recorded.wav"), self.sample_rate, wav_pcm
                )
            except Exception as error:  # noqa: BLE001 - diagnosis must never break STT
                self.logger.debug("Could not write debug audio: %s", error)

        if len(audio) < self.min_audio_length:
            self.logger.info("Speech too short (%.2fs < min, rms=%.4f). Discarding.", duration_s, rms)
            self._publish("speech.no_speech_detected")
            return

        self._publish("speech.transcription_started")

        if self.stt is None or not self.stt.is_available():
            self.logger.error("Whisper STT engine is not available.")
            self._publish("speech.transcription_failed", message="STT unavailable")
            self._publish("speech.error", kind="stt", message="STT unavailable")
            return

        t_stt_started = time.perf_counter()

        try:
            text = self.stt.transcribe(audio)
        except Exception as error:  # noqa: BLE001
            self.logger.exception("Whisper transcription failed: %s", error)
            self._publish("speech.transcription_failed", message=str(error))
            self._publish("speech.error", kind="stt", message=str(error))
            return

        text = (text or "").strip()
        if not text:
            self.logger.info(
                "No meaningful speech detected in audio (duration=%.2fs, rms=%.4f). Debug audio saved to logs/debug_last_recorded.wav",
                duration_s,
                rms,
            )
            self._publish("speech.no_speech_detected")
            return

        t_stt_done = time.perf_counter()

        if speech_end_t is None:
            speech_end_t = t_stt_started

        self.logger.info(
            "Recognized Speech: '%s' (duration=%.2fs, rms=%.4f, stt=%.0fms)",
            text,
            duration_s,
            rms,
            (t_stt_done - t_stt_started) * 1000.0,
        )
        self._publish("speech.transcription_completed", text=text)

        # Dispatch valid transcription to ConversationManager with the
        # stage timestamps so it can log speech_end -> playback latency.
        if self.event_bus is not None:
            try:
                self.event_bus.publish(
                    "speech_recognized",
                    text=text,
                    t_speech_end=speech_end_t,
                    t_stt_done=t_stt_done,
                )
            except Exception as error:  # noqa: BLE001
                self.logger.exception("Failed to publish speech_recognized: %s", error)
