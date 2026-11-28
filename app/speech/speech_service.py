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
from app.speech.transcript_normalizer import normalize_transcript

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

        # Input gain. A visitor speaking softly ~1 m from the kiosk mic
        # produces RMS around 0.004, which is barely above the noise floor
        # and below the elevated barge-in gate. A modest fixed gain lifts
        # genuine speech into the VAD/Whisper range without amplifying the
        # speaker bleed more than the playback gate already tolerates.
        try:
            self.input_gain = float(
                self._cfg("speech", "audio", "input_gain", default=1.0)
            )
        except (TypeError, ValueError):
            self.input_gain = 1.0

        self.input_gain = max(0.25, min(8.0, self.input_gain))

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
        model_path = self._resolve_stt_model_path()
        n_threads = self._cfg("speech", "stt", "n_threads", default=4)
        language = self._cfg("speech", "stt", "language", default="en")
        initial_prompt = self._cfg("speech", "stt", "initial_prompt", default=None)
        hotwords = self._cfg("speech", "stt", "hotwords", default="")
        if hotwords:
            initial_prompt = "%s %s" % (
                (initial_prompt or "").strip(),
                str(hotwords).strip(),
            )
            initial_prompt = initial_prompt.strip() or None
        min_text_len = self._cfg("speech", "stt", "min_text_length", default=3)
        max_text_len = self._cfg("speech", "stt", "max_text_length", default=500)

        # Quiet utterances are peak-normalised before Whisper so a ~0.004
        # RMS capture is not transcribed as noise. Gain is capped so a
        # silent buffer cannot be amplified into a hallucination.
        self.stt_normalize_audio = bool(
            self._cfg("speech", "stt", "normalize_audio", default=True)
        )
        try:
            self.stt_target_peak = float(
                self._cfg("speech", "stt", "normalize_target_peak", default=0.9)
            )
        except (TypeError, ValueError):
            self.stt_target_peak = 0.9
        self.stt_target_peak = max(0.1, min(1.0, self.stt_target_peak))
        try:
            self.stt_max_gain = float(
                self._cfg("speech", "stt", "normalize_max_gain", default=64.0)
            )
        except (TypeError, ValueError):
            self.stt_max_gain = 64.0
        self.stt_max_gain = max(1.0, min(32.0, self.stt_max_gain))
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

        # Barge-in (visitor interrupts the robot while it is speaking).
        self.barge_in_enabled = bool(
            self._cfg("speech", "barge_in", "enabled", default=True)
        )
        self.barge_in_require_visitor = bool(
            self._cfg(
                "speech",
                "barge_in",
                "require_visitor_in_frame",
                default=True,
            )
        )
        self.barge_in_min_speech_ms = max(
            1,
            int(self._cfg("speech", "barge_in", "min_speech_ms", default=160)),
        )
        # A laptop/desk mic hears a visitor at RMS ~0.004-0.015. The old
        # 1.8x gate on top of the VAD threshold put the bar above normal
        # speech, so a live barge-in could never fire. 1.3x above the
        # recent noise/playback floor is reachable while still rejecting
        # Greeto's own speaker bleed.
        self.barge_in_energy_multiplier = max(
            1.0,
            float(
                self._cfg(
                    "speech",
                    "barge_in",
                    "playback_vad_threshold_multiplier",
                    default=1.3,
                )
            ),
        )
        self.barge_in_cooldown_s = max(
            0.0,
            float(self._cfg("speech", "barge_in", "cooldown_ms", default=300))
            / 1000.0,
        )

        # Optional absolute playback gate (RMS after input gain). Used for
        # a calibrated microphone whose speaker bleed is known; when unset
        # the gate is the VAD threshold scaled by the multiplier above.
        try:
            floor = self._cfg(
                "speech", "barge_in", "energy_threshold", default=None
            )
            self.barge_in_energy_floor = 0.0 if floor in (None, "") else float(floor)
        except (TypeError, ValueError):
            self.barge_in_energy_floor = 0.0

        self.barge_in_energy_floor = max(0.0, self.barge_in_energy_floor)

        self._visitor_in_frame = False
        self._barge_in_candidate = False
        self._barge_in_fired = False
        self._barge_in_streak = 0
        self._barge_in_silence = 0
        self._barge_in_started = 0.0
        self._last_barge_in = 0.0
        self._barge_in_utterance = False
        self._barge_in_pre: collections.deque[np.ndarray] = collections.deque(maxlen=2)

        # Rolling RMS window used to estimate the live noise/playback floor
        # while Greeto is speaking (~3 s at a 1024-sample chunk). The
        # barge-in gate tracks this instead of a hard-coded constant, so a
        # quiet room is easy to interrupt in and a loud one does not
        # self-interrupt.
        self._playback_rms_window: collections.deque[float] = collections.deque(
            maxlen=47
        )
        self._last_barge_in_diag = 0.0

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

    def _resolve_stt_model_path(self) -> str:
        """Return the Whisper model path, honouring ``stt.model``.

        ``stt.model`` is a convenience switch (``tiny`` / ``base`` /
        ``small`` / ``medium``) that maps to the conventional
        ``models/whisper/ggml-<model>.en.bin`` filename. When the file is
        not present the configured ``model_path`` is used unchanged, so a
        typo degrades to the known model instead of breaking startup.
        """

        model_path = self._cfg(
            "speech", "stt", "model_path", default="models/whisper/ggml-tiny.en.bin"
        )

        name = self._cfg("speech", "stt", "model", default=None)

        if not name:
            return model_path

        candidate = "models/whisper/ggml-%s.en.bin" % str(name).strip().lower()
        resolved = Path(candidate)
        if not resolved.is_absolute():
            resolved = PROJECT_ROOT / resolved

        if resolved.exists():
            self.logger.info("Whisper model selected by stt.model: %s", candidate)
            return candidate

        self.logger.warning(
            "stt.model=%r maps to %s, which does not exist; using %s.",
            name,
            candidate,
            model_path,
        )
        return model_path

    def _apply_input_gain(self, audio: np.ndarray) -> np.ndarray:
        """Apply the configured microphone gain with hard clipping."""

        if self.input_gain == 1.0 or audio is None or len(audio) == 0:
            return audio

        return np.clip(
            np.asarray(audio, dtype=np.float32) * self.input_gain, -1.0, 1.0
        )

    def _normalize_for_stt(self, audio: np.ndarray) -> tuple[np.ndarray, float]:
        """Peak-normalise a quiet utterance before Whisper.

        Returns the (possibly amplified) audio and the gain that was
        applied. Gain is capped by ``stt.normalize_max_gain`` and skipped
        for an effectively silent buffer, so background noise is never
        amplified into a false transcript.
        """

        if (
            not self.stt_normalize_audio
            or audio is None
            or len(audio) == 0
        ):
            return audio, 1.0

        arr = np.asarray(audio, dtype=np.float32)
        peak = float(np.max(np.abs(arr))) if arr.size else 0.0

        if peak <= 1e-4 or peak >= self.stt_target_peak:
            return audio, 1.0

        gain = min(self.stt_max_gain, self.stt_target_peak / peak)

        if gain <= 1.01:
            return audio, 1.0

        return np.clip(arr * gain, -1.0, 1.0), gain
    def _register_event_handlers(self) -> None:
        """Subscribe to TTS events to avoid recording the robot's own voice."""
        if self.event_bus is not None:
            try:
                self.event_bus.subscribe("speech.synthesis_started", self._on_tts_started)
                self.event_bus.subscribe("speech.speaking_started", self._on_tts_started)
                self.event_bus.subscribe("speech.speaking_finished", self._on_tts_finished)
                self.event_bus.subscribe("speech.error", self._on_tts_error)
                self.event_bus.subscribe("tts.started", self._on_tts_started)
                self.event_bus.subscribe("tts.stopped", self._on_tts_stopped)
                self.event_bus.subscribe("visitor.detected", self._on_visitor_present)
                self.event_bus.subscribe("visitor.unknown", self._on_visitor_present)
                self.event_bus.subscribe("visitor.left", self._on_visitor_left)
                self.event_bus.subscribe("visitor.departed", self._on_visitor_left)
            except Exception as error:  # noqa: BLE001
                self.logger.warning("Failed to register speech event listeners: %s", error)

    def _on_tts_error(self, **kwargs) -> None:
        if kwargs.get("kind") == "tts":
            self._tts_speaking = False
            self._reset_barge_in_candidate()

    def _on_tts_started(self, **kwargs) -> None:
        # Always track playback state; mute_during_speech only controls the
        # legacy "drop everything" behaviour when barge-in is disabled.
        self._tts_speaking = True
        self._reset_barge_in_candidate()

        if self.mute_during_speech and not self.barge_in_enabled:
            with self._lock:
                self.audio_buffer.clear()
                self.pre_speech_buffer.clear()
                self.silence_counter = 0
                self.speech_streak = 0
                self.total_samples_recorded = 0
                if self.state == self.STATE_RECORDING:
                    self._set_state(self.STATE_LISTENING)

    def _on_tts_finished(self, **kwargs) -> None:
        self._tts_speaking = False
        self._reset_barge_in_candidate()
        # If the finish came from an interrupted turn, the pipeline is
        # already recording the visitor's new utterance. Do not apply the
        # post-TTS mute cooldown, or the first 300 ms would be dropped.
        if self.state != self.STATE_RECORDING:
            self._tts_cooldown_until = time.monotonic() + self.speech_cooldown_s

    def _on_tts_stopped(self, **kwargs) -> None:
        self._tts_speaking = False
        if kwargs.get("interrupted"):
            # The interruption handler has already moved the pipeline into
            # RECORDING; just clear the candidate bookkeeping.
            self._reset_barge_in_candidate()

    def _on_visitor_present(self, **kwargs) -> None:
        self._visitor_in_frame = True

    def _on_visitor_left(self, **kwargs) -> None:
        self._visitor_in_frame = False
        self._reset_barge_in_candidate()

    def _reset_barge_in_candidate(self) -> None:
        self._barge_in_candidate = False
        self._barge_in_fired = False
        self._barge_in_streak = 0
        self._barge_in_silence = 0
        self._barge_in_pre.clear()
        self._playback_rms_window.clear()

    @property
    def visitor_in_frame(self) -> bool:
        return self._visitor_in_frame

    def _visitor_gate_open(self) -> bool:
        if not self.barge_in_require_visitor:
            return True
        return self._visitor_in_frame

    def _playback_noise_floor(self) -> float:
        """Estimate the current playback/noise floor from recent chunks.

        The 20th percentile of the last ~3 s is used so a single loud
        syllable (or a burst of Greeto's own speech) cannot raise the gate
        for the whole utterance. Only frames observed during playback feed
        this window, so it measures what the mic actually hears while the
        speaker is on.
        """

        if not self._playback_rms_window:

            return 0.0

        ordered = sorted(self._playback_rms_window)

        return ordered[max(0, len(ordered) // 5)]

    def _barge_in_threshold(self) -> float:
        """Return the RMS gate used while Greeto is speaking."""

        if self.barge_in_energy_floor > 0.0:
            return self.barge_in_energy_floor

        base = max(self.vad.energy_threshold, self._playback_noise_floor())

        return base * self.barge_in_energy_multiplier

    def _handle_barge_in_chunk(self, audio: np.ndarray) -> None:
        """Monitor mic input while Piper is playing and fire a barge-in.

        Returns via side effects only; the caller always returns after this
        method so playback echo is never fed into the normal VAD path.
        """

        now = time.perf_counter()

        if now - self._last_barge_in < self.barge_in_cooldown_s:
            return

        if not self._visitor_gate_open():
            # Keep the pre-roll fresh but do not interrupt without a face.
            self._barge_in_pre.append(audio)
            return

        rms = self.vad.calculate_energy(audio)

        # Only quiet chunks describe the room: feeding the visitor's own
        # speech into the floor would raise the gate above the interruption
        # it is meant to detect. The pre-gate uses the current floor.
        if rms < self._barge_in_threshold():

            self._playback_rms_window.append(rms)

        threshold = self._barge_in_threshold()
        loud = rms >= threshold
        voiced = loud and self.vad.is_speech(audio)

        self._log_barge_in_diagnostics(rms, threshold, voiced, now)

        if not self._barge_in_candidate:
            if not voiced:
                self._barge_in_pre.append(audio)
                return

            # Speech onset: preserve the two chunks immediately before the
            # onset (often the unvoiced consonant) and the onset chunk.
            self._barge_in_candidate = True
            self._barge_in_fired = False
            self._barge_in_streak = 1
            self._barge_in_silence = 0
            self._barge_in_started = now - self.chunk_duration_s
            self.audio_buffer = list(self._barge_in_pre)
            self.audio_buffer.append(audio)
            self.total_samples_recorded = sum(c.size for c in self.audio_buffer)
            self._barge_in_pre.clear()
            return

        # Candidate in progress: keep every chunk from onset onward.
        self.audio_buffer.append(audio)
        self.total_samples_recorded += audio.size

        if voiced:
            self._barge_in_silence = 0
            self._barge_in_streak += 1
            # Count confirmed chunks as well as wall time. Chunk counting
            # keeps the gate deterministic under a simulated/offline mic
            # while wall time still covers irregular audio callbacks.
            sustained_ms = max(
                (now - self._barge_in_started) * 1000.0,
                self._barge_in_streak * self.chunk_duration_s * 1000.0,
            )
            if not self._barge_in_fired and sustained_ms >= self.barge_in_min_speech_ms:
                self._fire_barge_in(now, rms)
            return

        self._barge_in_silence += 1
        # A cough or click shorter than the confirmation window is discarded.
        if (
            not self._barge_in_fired
            and self._barge_in_silence * self.chunk_duration_s * 1000.0 >= 300.0
        ):
            self._barge_in_candidate = False
            self.audio_buffer.clear()
            self.total_samples_recorded = 0
            self._barge_in_silence = 0

    def _log_barge_in_diagnostics(
        self,
        rms: float,
        threshold: float,
        voiced: bool,
        now: float,
    ) -> None:
        """Explain barge-in gating at most once every two seconds.

        A live miss is usually one of two things: the visitor is quieter
        than the gate, or the gate is audible but the VAD rejects it. This
        line makes which one obvious without flooding the terminal.
        """

        if now - self._last_barge_in_diag < 2.0:

            return

        approaching = threshold > 0 and rms >= 0.7 * threshold

        if not approaching:

            return

        self._last_barge_in_diag = now

        self.logger.info(
            "Barge-in probe: rms=%.4f gate=%.4f floor=%.4f voiced=%s "
            "candidate=%s streak=%d",
            rms,
            threshold,
            self._playback_noise_floor(),
            voiced,
            self._barge_in_candidate,
            self._barge_in_streak,
        )

    def _fire_barge_in(self, now: float, rms: float) -> None:
        self._barge_in_fired = True
        self._last_barge_in = now
        self._barge_in_utterance = True
        # Stop the normal self-echo suppression immediately; TTS is about to
        # be purged and the new utterance must be recorded without clipping.
        self._tts_speaking = False

        with self._lock:
            self.silence_counter = 0
            self.speech_streak = 0
            self._set_state(self.STATE_RECORDING)

        self._publish(
            "speech.barge_in",
            t_barge_in=now,
            t_speech_onset=self._barge_in_started,
            min_speech_ms=self.barge_in_min_speech_ms,
            rms=rms,
            visitor_in_frame=self._visitor_in_frame,
        )
        self._publish("speech.recording_started")
        self.logger.info(
            "Barge-in accepted after %.0fms of speech (rms=%.4f, "
            "gate=%.4f, floor=%.4f).",
            self.barge_in_min_speech_ms,
            rms,
            self._barge_in_threshold(),
            self._playback_noise_floor(),
        )

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
                ("tts.started", self._on_tts_started),
                ("tts.stopped", self._on_tts_stopped),
                ("visitor.detected", self._on_visitor_present),
                ("visitor.unknown", self._on_visitor_present),
                ("visitor.left", self._on_visitor_left),
                ("visitor.departed", self._on_visitor_left),
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

        # Input gain lifts quiet speech for the barge-in gate and the
        # Whisper buffer. The normal onset VAD keeps seeing the raw signal,
        # so the calibrated noise-floor threshold is unchanged and a gain
        # cannot turn room noise into a false start.
        raw_audio = audio
        audio = self._apply_input_gain(audio)
        # While the robot is speaking, keep the microphone open in
        # barge-in mode and look for sustained human speech above the
        # elevated playback threshold. Otherwise retain the old mute gate.
        if self._tts_speaking:
            if self.barge_in_enabled:
                self._handle_barge_in_chunk(audio)
                return
            if self.mute_during_speech:
                return

        if time.monotonic() < self._tts_cooldown_until:
            return

        speaking = self.vad.is_speech(raw_audio)

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

        # Consume the barge-in marker exactly once. The conversation layer
        # uses it to bypass the duplicate-utterance rate limit.
        is_barge_in = self._barge_in_utterance
        self._barge_in_utterance = False

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

        # Whisper is fed a peak-normalised copy; the debug WAV above stays
        # the raw microphone signal so calibration is measured honestly.
        audio, normalize_gain = self._normalize_for_stt(audio)
        if normalize_gain > 1.01:
            self.logger.info(
                "Normalised quiet utterance for Whisper (gain=%.2fx, rms=%.4f).",
                normalize_gain,
                rms,
            )

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
        raw_text = text
        text, repairs = normalize_transcript(text)
        if repairs:
            self.logger.info(
                "Transcript repaired: %s (raw=%r).",
                "; ".join(repairs),
                raw_text,
            )

        self._publish(
            "speech.transcription_completed", text=text, raw_text=raw_text
        )

        # Dispatch valid transcription to ConversationManager with the
        # stage timestamps so it can log speech_end -> playback latency.
        if self.event_bus is not None:
            try:
                self.event_bus.publish(
                    "speech_recognized",
                    text=text,
                    raw_text=raw_text,
                    t_speech_end=speech_end_t,
                    t_stt_done=t_stt_done,
                    barge_in=is_barge_in,
                )
            except Exception as error:  # noqa: BLE001
                self.logger.exception("Failed to publish speech_recognized: %s", error)
