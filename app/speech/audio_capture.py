"""
Continuous microphone capture and audio device management.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import sounddevice as sd

try:
    from math import gcd
    from scipy import signal

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

logger = logging.getLogger(__name__)


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample a 1D float32 audio array from orig_sr to target_sr.
    """
    if orig_sr == target_sr or len(audio) == 0:
        return audio.astype(np.float32)

    if _HAS_SCIPY:
        common = gcd(orig_sr, target_sr)
        up = target_sr // common
        down = orig_sr // common
        resampled = signal.resample_poly(audio, up, down)
        return np.asarray(resampled, dtype=np.float32)

    # Linear interpolation fallback
    num_target_samples = int(round(len(audio) * target_sr / orig_sr))
    orig_indices = np.linspace(0, len(audio) - 1, len(audio))
    target_indices = np.linspace(0, len(audio) - 1, num_target_samples)
    resampled = np.interp(target_indices, orig_indices, audio)
    return np.asarray(resampled, dtype=np.float32)


class AudioCapture:
    """
    Captures microphone audio in a background thread with device resolution,
    bounded queue buffering, channel mixing and automatic resampling.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 1024,
        channels: int = 1,
        device: Optional[Union[int, str]] = None,
        queue_max_chunks: int = 100,
        on_error: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.device = device
        self.queue_max_chunks = queue_max_chunks
        self.on_error = on_error

        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue(
            maxsize=queue_max_chunks
        )

        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.stream: Optional[object] = None
        self._lock = threading.Lock()
        self.actual_device_index: Optional[int] = None
        self.actual_device_name: Optional[str] = None
        self.hardware_sample_rate: int = sample_rate

        # Latency instrumentation
        self._correlation_id: Optional[str] = None
        self._capture_start_ms: Optional[float] = None
        self._capture_end_ms: Optional[float] = None
        self._capture_chunks: List[float] = []

    @staticmethod
    def list_input_devices() -> List[Dict[str, Any]]:
        """
        Return a list of available audio input devices.
        """
        devices = []
        try:
            device_list = sd.query_devices()
            default_input = sd.default.device[0] if sd.default.device else None
            for idx, dev in enumerate(device_list):
                if dev.get("max_input_channels", 0) > 0:
                    devices.append({
                        "index": idx,
                        "name": dev.get("name", f"Device {idx}"),
                        "channels": dev.get("max_input_channels", 1),
                        "default_samplerate": dev.get("default_samplerate", 16000),
                        "hostapi": dev.get("hostapi", 0),
                        "is_default": (idx == default_input),
                    })
        except Exception as error:  # noqa: BLE001
            logger.warning("Failed to query audio devices: %s", error)
        return devices

    @staticmethod
    def resolve_device(device_spec: Optional[Union[int, str]]) -> Optional[int]:
        """
        Resolve a device index or substring name to a valid input device index.
        Returns None to use default device, or device index.
        Raises ValueError if device cannot be resolved.
        """
        if device_spec is None:
            return None

        if isinstance(device_spec, int):
            try:
                info = sd.query_devices(device_spec)
                if info.get("max_input_channels", 0) <= 0:
                    raise ValueError(
                        f"Device index {device_spec} ('{info.get('name')}') has no input channels."
                    )
                return device_spec
            except Exception as error:  # noqa: BLE001
                raise ValueError(
                    f"Invalid audio device index {device_spec}: {error}"
                ) from error

        if isinstance(device_spec, str):
            device_spec_stripped = device_spec.strip()
            if not device_spec_stripped:
                return None
            try:
                # If numeric string
                if device_spec_stripped.isdigit():
                    return AudioCapture.resolve_device(int(device_spec_stripped))

                devices = AudioCapture.list_input_devices()
                query_lower = device_spec_stripped.lower()
                for dev in devices:
                    if query_lower in dev["name"].lower():
                        return dev["index"]
                raise ValueError(
                    f"No input audio device matches name '{device_spec}'."
                )
            except Exception as error:  # noqa: BLE001
                if isinstance(error, ValueError):
                    raise
                raise ValueError(
                    f"Failed resolving audio device '{device_spec}': {error}"
                ) from error

        return None

    def start(self) -> None:
        """
        Start the background capture loop.
        """
        if self.running:
            return
        with self._lock:
            self.running = True
            self._correlation_id = f"capture_{int(time.monotonic() * 1000):06d}"
            self._capture_start_ms = time.monotonic()
        self.thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name="AudioCapture",
        )
        self.thread.start()

    def stop(self) -> None:
        """
        Stop the capture loop and release resources.
        """
        with self._lock:
            self.running = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.thread = None
        self._drain_queue()
        self._capture_end_ms = time.monotonic()

    def get_audio(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        Retrieve an audio chunk from the queue.
        """
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _drain_queue(self) -> None:
        """
        Empty all items in the audio queue.
        """
        try:
            while True:
                self.audio_queue.get_nowait()
        except queue.Empty:
            pass

    def _push_chunk(self, audio: np.ndarray) -> None:
        """
        Push audio to the bounded queue, dropping the oldest if full.
        """
        try:
            self.audio_queue.put_nowait(audio)
        except queue.Full:
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.audio_queue.put_nowait(audio)
            except queue.Full:
                pass

    def _capture_loop(self) -> None:
        """
        Thread target that initializes and streams from sounddevice.InputStream.
        """
        resolved_device = None
        try:
            resolved_device = self.resolve_device(self.device)
            self.actual_device_index = resolved_device
            if resolved_device is not None:
                info = sd.query_devices(resolved_device)
                self.actual_device_name = info.get("name")
            else:
                default_idx = sd.default.device[0] if sd.default.device else None
                if default_idx is not None and default_idx >= 0:
                    info = sd.query_devices(default_idx)
                    self.actual_device_name = info.get("name")
        except Exception as error:  # noqa: BLE001
            logger.error("Failed to resolve audio input device: %s", error)
            if self.on_error is not None:
                self.on_error(str(error))
            with self._lock:
                self.running = False
            return

        target_sr = self.sample_rate

        # Attempt to determine supported sample rate
        stream_sr = target_sr
        stream_channels = self.channels

        # Get initial device info for quality metrics
        dev_info = sd.query_devices(resolved_device if resolved_device is not None else sd.default.device[0])
        self._device_info = {
            "name": info.get("name"),
            "host_api": info.get("hostapi"),
            "max_input_channels": info.get("max_input_channels", 1),
            "default_samplerate": info.get("default_samplerate", 16000),
        }

        def callback(indata, frames, time_info, status):
            if status:
                logger.warning("Audio input status: %s", status)
            try:
                # Convert to mono float32
                if indata.ndim > 1 and indata.shape[1] > 1:
                    audio = np.mean(indata, axis=1, dtype=np.float32)
                else:
                    audio = indata[:, 0].astype(np.float32) if indata.ndim > 1 else indata.astype(np.float32)

                # Resample if hardware sample rate differs
                if self.hardware_sample_rate != target_sr:
                    audio = resample_audio(audio, self.hardware_sample_rate, target_sr)

                # Track capture timestamp per chunk for latency measurement
                chunk_ms = time.monotonic() * 1000
                self._capture_chunks.append(chunk_ms)

                self._push_chunk(audio)
            except Exception as error:  # noqa: BLE001
                logger.exception("Audio callback error: %s", error)

        # First attempt: open with target sample rate
        stream = None
        try:
            kwargs = dict(
                samplerate=stream_sr,
                channels=stream_channels,
                dtype="float32",
                callback=callback,
                blocksize=self.chunk_size,
            )
            if resolved_device is not None:
                kwargs["device"] = resolved_device

            try:
                stream = sd.InputStream(**kwargs)
                self.hardware_sample_rate = stream_sr
            except Exception as sr_err:
                # If target sample rate unsupported, fallback to device default sample rate
                logger.warning(
                    "Cannot open audio stream at %d Hz (%s). Attempting fallback to device default rate.",
                    stream_sr,
                    sr_err,
                )
                dev_info = sd.query_devices(resolved_device if resolved_device is not None else sd.default.device[0])
                fallback_sr = int(dev_info.get("default_samplerate", 44100))
                kwargs["samplerate"] = fallback_sr
                kwargs["blocksize"] = int(self.chunk_size * fallback_sr / target_sr)
                stream = sd.InputStream(**kwargs)
                self.hardware_sample_rate = fallback_sr
                logger.info("Microphone opened with fallback rate %d Hz (resampling to %d Hz).", fallback_sr, target_sr)

            self.stream = stream
            with stream:
                logger.info(
                    "Audio capture active on device [%s: %s] at %d Hz.",
                    self.actual_device_index,
                    self.actual_device_name,
                    self.hardware_sample_rate,
                )
                while self.running:
                    time.sleep(0.05)
        except Exception as error:  # noqa: BLE001
            logger.exception("Audio capture failed: %s", error)
            if self.on_error is not None:
                try:
                    self.on_error(str(error))
                except Exception:  # noqa: BLE001
                    logger.exception("on_error callback raised an exception.")
        finally:
            self.stream = None
            with self._lock:
                self.running = False
            self._drain_queue()

    def get_audio_quality_metrics(self) -> Dict[str, Any]:
        """
        Return audio quality metrics for this capture session.
        """
        metrics: Dict[str, Any] = {
            "correlation_id": self._correlation_id,
            "capture_start_ms": self._capture_start_ms,
            "capture_end_ms": self._capture_end_ms,
            "device_name": self._device_info.get("name") if self._device_info else None,
            "device_host_api": self._device_info.get("host_api") if self._device_info else None,
            "max_input_channels": self._device_info.get("max_input_channels", 1) if self._device_info else 1,
            "default_samplerate": self._device_info.get("default_samplerate", 16000) if self._device_info else 16000,
            "hardware_sample_rate": self.hardware_sample_rate,
            "capture_chunks": self._capture_chunks,
        }
        # Calculate chunk continuity
        if len(self._capture_chunks) > 1:
            intervals = [self._capture_chunks[i] - self._capture_chunks[i - 1] for i in range(1, len(self._capture_chunks))]
            metrics["avg_chunk_interval_ms"] = sum(intervals) / len(intervals) if intervals else 0
            metrics["chunk_jitter_ms"] = max(intervals) - min(intervals) if intervals else 0
        else:
            metrics["avg_chunk_interval_ms"] = 0
            metrics["chunk_jitter_ms"] = 0
        return metrics

    def get_capture_metrics(self) -> Dict[str, Any]:
        """
        Return timing metrics for this capture session.
        """
        return {
            "correlation_id": self._correlation_id,
            "capture_start_ms": self._capture_start_ms,
            "capture_end_ms": self._capture_end_ms,
            "capture_chunks": self._capture_chunks,
        }
