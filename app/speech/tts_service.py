"""Text To Speech Service.

Converts AI responses into speech.

Playback lifecycle events consumed by the kiosk:

    speech.speaking_started    text=, utterance=    playback is starting
    speech.speaking_finished   utterance=           playback has ended
    speech.error               kind="tts", message=

``speech.speaking_started`` is published after Piper has produced the WAV
and immediately before playback, so the face only enters SPEAKING when
audio is actually being played, never while the WAV is still being
generated.
"""

from __future__ import annotations

import logging
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time as time_module
import wave
from pathlib import Path
from typing import Optional

import numpy as np

from app.core.container import ServiceContainer

# Repository root, independent of the current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

logger = logging.getLogger(__name__)


class TTSService:
    """Offline Piper TTS wrapper."""

    def __init__(
        self,
        container: ServiceContainer,
        model_path=None,
        engine: str | None = None,
    ) -> None:
        self.container = container

        self.logger = container.logger

        self.event_bus = getattr(
            container,
            "event_bus",
            None,
        )

        # Piper can run in-process with a cached voice (fast, no per-
        # utterance process/model start) or as a subprocess (compatible
        # fallback). "auto" tries the in-process engine first.
        self._engine = self._resolve_engine(engine)

        self._voice = None
        self._voice_lock = threading.Lock()
        self._synth_config = None

        # Identifies one utterance, so a late "finished" event can never
        # end a newer one in the UI.
        self._utterance = 0
        self._is_speaking = False
        self._cancelled = False
        self._active_turn_id: Optional[int] = None

        # One TTSService is shared by the application. Serializing the
        # whole generate-and-play cycle keeps two callers from speaking
        # over each other and keeps utterance numbers in playback order.
        self._speak_lock = threading.Lock()

        # Barge-in support: stop() must be callable from the VAD/event-bus
        # thread while speak() owns the speak lock. These fields are
        # deliberately separate from that lock.
        self._playback_lock = threading.Lock()
        self._active_player = None
        self._active_synth_process = None
        self._active_stream = None
        self._stop_event = threading.Event()
        self._stop_event.set()

        # Playback is written to the output device in small blocks so
        # stop() can cut the speaker within one block (~20-40 ms), which
        # a single blocking winsound call cannot do.
        self._playback_block_frames = 1024

        self.model_path = (
            Path(model_path)
            if model_path is not None
            else PROJECT_ROOT / "models" / "tts" / "en_US-lessac-medium.onnx"
        )

        if not self.model_path.exists():
            raise FileNotFoundError(f"TTS model missing: {self.model_path}")

    # -------------------------------------------------

    def _resolve_engine(self, override: str | None) -> str:
        """Return auto, python or subprocess."""

        value = override

        if value is None:
            config = getattr(self.container, "config", None)
            try:
                value = config.get_value(
                    "speech", "tts", "engine", default="auto"
                )
            except (AttributeError, TypeError, ValueError):
                value = "auto"

        value = str(value or "auto").strip().lower()

        if value not in ("auto", "python", "subprocess"):
            value = "auto"

        return value

    # -------------------------------------------------

    def warm_up(self) -> None:
        """Load the Piper voice in the background.

        Called once at startup so the first greeting or answer does not
        pay the model-load cost inside the visitor's response latency.
        """

        if self._engine == "subprocess":
            return

        threading.Thread(
            target=self._load_voice,
            daemon=True,
            name="TTSWarmup",
        ).start()

    def _load_voice(self):
        """Load and cache the in-process Piper voice, or fall back."""

        if self._engine == "subprocess":
            return None

        if self._voice is not None:
            return self._voice

        with self._voice_lock:

            if self._voice is not None:
                return self._voice

            try:

                from piper import PiperVoice, SynthesisConfig

                voice = PiperVoice.load(str(self.model_path))

                length_scale = 1.0

                config = getattr(self.container, "config", None)

                try:
                    length_scale = float(
                        config.get_value(
                            "speech", "tts", "length_scale", default=1.0
                        )
                    )
                except (AttributeError, TypeError, ValueError):
                    length_scale = 1.0

                self._synth_config = SynthesisConfig(
                    length_scale=max(0.5, min(2.0, length_scale))
                )

                self._voice = voice

                self.logger.info(
                    "Piper voice loaded in-process: %s",
                    self.model_path.name,
                )

                return voice

            except Exception as error:  # noqa: BLE001

                if self._engine == "python":

                    raise

                self.logger.warning(
                    "In-process Piper unavailable (%s); using the "
                    "subprocess fallback.",
                    error,
                )

                self._engine = "subprocess"

                return None

    def _synthesize_to_wav(self, text: str, wav_path: str) -> None:
        """Synthesize *text* into *wav_path* with a warm or cold Piper."""

        voice = self._load_voice()

        if voice is not None:

            import wave

            with wave.open(wav_path, "wb") as wav_file:

                voice.synthesize_wav(text, wav_file, self._synth_config)

            return

        command = self._resolve_piper_command() + [
            "--model",
            str(self.model_path),
            "--output_file",
            wav_path,
        ]

        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            text=True,
        )

        # stop() cancels an in-flight synthesis process as well as the
        # player, so a barge-in never waits for an obsolete sentence to
        # finish rendering before audio is purged.
        with self._playback_lock:
            self._active_synth_process = process

        try:
            process.communicate(text)
        finally:
            with self._playback_lock:
                if self._active_synth_process is process:
                    self._active_synth_process = None

        if process.returncode not in (0, None):
            raise RuntimeError("Piper exited with code %s" % process.returncode)

    # -------------------------------------------------

    def _resolve_piper_command(self) -> list[str]:
        """Locate the Piper binary or Python module."""
        piper_exe = shutil.which("piper")
        if piper_exe:
            return [piper_exe]

        candidate_dirs = [
            Path(sys.executable).parent,
            PROJECT_ROOT / ".venv" / ("Scripts" if sys.platform.startswith("win") else "bin"),
        ]
        exe_name = "piper.exe" if sys.platform.startswith("win") else "piper"
        for d in candidate_dirs:
            p = d / exe_name
            if p.exists() and p.is_file():
                return [str(p)]

        # Python module invocation fallback
        return [sys.executable, "-m", "piper"]

    # -------------------------------------------------

    # -------------------------------------------------

    def is_speaking(self) -> bool:
        """Return True if TTS is currently synthesizing or playing audio."""
        return self._is_speaking

    def stop(self, interrupted: bool = True) -> bool:
        """Stop synthesis/playback immediately. Idempotent and thread-safe.

        Called from the VAD thread on barge-in. The Windows purge and the
        POSIX player terminate both return in well under 50 ms; synthesis
        that has not reached playback is abandoned by the cancellation
        flag before a WAV is played.
        """

        started = time_module.perf_counter()

        self._cancelled = True
        self._stop_event.set()

        with self._playback_lock:
            player = self._active_player
            synth_process = self._active_synth_process
            stream = self._active_stream

        # Abort the audio output stream first: it is what the visitor is
        # actually hearing, and aborting unblocks the playback thread's
        # pending write() immediately.
        if stream is not None:
            try:
                stream.abort()
            except Exception:
                pass

        for process in (player, synth_process):
            if process is None:
                continue
            try:
                process.terminate()
            except Exception:
                pass

        if sys.platform.startswith("win"):
            try:
                import winsound

                # Fallback for the winsound playback path. None + 0 stops
                # asynchronous playback; SND_PURGE covers the blocking
                # path on the Windows versions that still honour it.
                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception:
                pass
            try:
                import winsound

                winsound.PlaySound(None, 0)
            except Exception:
                pass

        elapsed_ms = (time_module.perf_counter() - started) * 1000.0

        self._publish(
            "tts.stopped",
            interrupted=bool(interrupted),
            stop_ms=elapsed_ms,
            turn_id=self._active_turn_id,
        )
        self.logger.info(
            "TTS stopped (interrupted=%s) in %.1fms.",
            bool(interrupted),
            elapsed_ms,
        )
        return True

    def cancel(self) -> None:
        """Backwards-compatible alias for :meth:`stop`."""

        self.stop(interrupted=True)

    # -------------------------------------------------

    def speak(self, text: str, turn_id: Optional[int] = None) -> bool:
        """Convert text into speech and play it.

        Returns True when the audio was played. Calls on a shared
        instance run one at a time.

        For standard conversational replies (<= 250 characters), synthesizes
        in a single Piper invocation to eliminate inter-sentence gaps.
        """
        if not text:
            return False

        clean_text = text.strip()
        if not clean_text:
            return False

        with self._speak_lock:
            self._is_speaking = True
            self._cancelled = False
            self._stop_event.clear()
            self._active_turn_id = turn_id

            # Publish synthesis start so VAD echo suppression begins immediately
            self._publish("speech.synthesis_started", turn_id=turn_id, text=clean_text)
            self._publish("tts.started", turn_id=turn_id, text=clean_text)

            try:
                # If conversational length (<= 250 chars), synthesize in a single pass
                # to prevent cold subprocess startup delay between sentences.
                if len(clean_text) <= 250:
                    return self._speak_single_chunk(clean_text, turn_id=turn_id)

                sentences = self._split_into_sentences(clean_text)
                if not sentences:
                    return False
                return self._speak_sentences(clean_text, sentences, turn_id=turn_id)
            finally:
                self._is_speaking = False
                if not self._cancelled:
                    self._publish(
                        "tts.stopped",
                        interrupted=False,
                        turn_id=turn_id,
                    )

    # -------------------------------------------------

    def _speak_single_chunk(self, full_text: str, turn_id: Optional[int] = None) -> bool:
        """Synthesize and play the full response in a single Piper pass (0s sentence gap)."""
        if self._cancelled:
            return False

        utterance = self._utterance + 1
        self._utterance = utterance
        wav_path = ""
        played_successfully = False

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_file:
                wav_path = audio_file.name

            synth_started = time_module.perf_counter()

            self._synthesize_to_wav(full_text, wav_path)

            tts_synth_ms = (time_module.perf_counter() - synth_started) * 1000.0

            wav_file = Path(wav_path)
            if not wav_file.exists() or wav_file.stat().st_size == 0:
                raise RuntimeError("Piper failed to produce a valid WAV file.")

            if self._cancelled:
                return False

            self._publish(
                "speech.speaking_started",
                text=full_text,
                utterance=utterance,
                turn_id=turn_id,
                sentence_index=0,
                tts_synth_ms=tts_synth_ms,
                tts_ready_at=time_module.perf_counter(),
            )

            if not self._play_audio(wav_path):
                raise RuntimeError("No audio player could play the generated speech.")

            played_successfully = True

        except Exception as error:
            self.logger.exception("TTS error: %s", error)
            self._publish(
                "speech.error",
                kind="tts",
                message=str(error),
                turn_id=turn_id,
            )
            return False

        finally:
            if wav_path:
                try:
                    Path(wav_path).unlink()
                except OSError:
                    pass

        if played_successfully:
            self._publish(
                "speech.speaking_finished",
                utterance=utterance,
                turn_id=turn_id,
                sentence_index=0,
            )

        return True

    # -------------------------------------------------

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences/phrases for incremental playback."""
        import re

        sentences = re.split(r"(?<=[.!?])\s+", text.strip())

        result = []
        for s in sentences:
            s = s.strip()
            if s:
                result.append(s)

        if not result and text.strip():
            return [text.strip()]

        return result

    # -------------------------------------------------

    def _speak_sentences(
        self,
        full_text: str,
        sentences: list[str],
        turn_id: Optional[int] = None,
    ) -> bool:
        """Synthesize with one-sentence lookahead for gap-free playback.

        The first sentence is synthesized synchronously so playback starts as
        soon as possible. While it plays, a prefetch worker synthesizes the
        next sentence into a temp WAV. By the time playback ends, the next
        WAV is ready and the next sentence starts immediately.
        """
        total = len(sentences)
        first_synth_ms = 0.0
        # At most two synthesized-but-unplayed WAVs are kept on disk.
        ready: "queue.Queue[tuple[int, str, Optional[BaseException]]]" = queue.Queue(
            maxsize=2
        )

        def cleanup(wav_path: str) -> None:
            if not wav_path:
                return
            try:
                Path(wav_path).unlink()
            except OSError:
                pass

        def synthesize(sentence: str) -> str:
            """Synthesize one sentence into a unique temp WAV and return its path."""
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_file:
                wav_path = audio_file.name

            try:
                self._synthesize_to_wav(sentence, wav_path)

                wav_file = Path(wav_path)
                if not wav_file.exists() or wav_file.stat().st_size == 0:
                    raise RuntimeError("Piper failed to produce a valid WAV file.")
                return wav_path
            except BaseException:
                cleanup(wav_path)
                raise

        def enqueue(
            index: int, wav_path: str, error: Optional[BaseException]
        ) -> bool:
            """Hand an item to playback, unblocking promptly on cancellation."""
            while not self._cancelled:
                try:
                    ready.put((index, wav_path, error), timeout=0.2)
                    return True
                except queue.Full:
                    continue
            return False

        def prefetch_worker() -> None:
            for i in range(1, total):
                if self._cancelled:
                    return
                try:
                    path = synthesize(sentences[i])
                    error = None
                except BaseException as exc:  # reported by the playback thread
                    path, error = "", exc
                if not enqueue(i, path, error):
                    cleanup(path)

        # Sentence 0 first: no avoidable latency before playback begins.
        first_synth_started = time_module.perf_counter()
        try:
            first_path = synthesize(sentences[0])
            first_synth_ms = (
                time_module.perf_counter() - first_synth_started
            ) * 1000.0
        except Exception as error:
            self.logger.exception("TTS error for sentence 0: %s", error)
            self._publish(
                "speech.error", kind="tts", message=str(error), turn_id=turn_id
            )
            return False
        ready.put((0, first_path, None))

        worker = threading.Thread(
            target=prefetch_worker,
            daemon=True,
            name="TTSPrefetch",
        )
        worker.start()

        try:
            for _ in range(total):
                if self._cancelled:
                    return False

                # Normally already on the queue: it was synthesized while
                # the previous sentence was playing.
                item = None
                while item is None and not self._cancelled:
                    try:
                        item = ready.get(timeout=0.2)
                    except queue.Empty:
                        continue
                if item is None:
                    return False

                index, wav_path, error = item
                if error is not None:
                    raise error

                utterance = self._utterance + 1
                self._utterance = utterance
                played_successfully = False
                try:
                    if self._cancelled:
                        return False

                    # A single-sentence reply must keep the exact input text.
                    display_text = full_text if total == 1 else sentences[index]
                    self._publish(
                        "speech.speaking_started",
                        text=display_text,
                        utterance=utterance,
                        turn_id=turn_id,
                        sentence_index=index,
                        tts_synth_ms=first_synth_ms if index == 0 else 0.0,
                        tts_ready_at=time_module.perf_counter(),
                    )

                    if not self._play_audio(wav_path):
                        raise RuntimeError(
                            "No audio player could play the generated speech."
                        )
                    played_successfully = True
                finally:
                    cleanup(wav_path)

                if played_successfully:
                    self._publish(
                        "speech.speaking_finished",
                        utterance=utterance,
                        turn_id=turn_id,
                        sentence_index=index,
                    )
        except Exception as error:
            self.logger.exception("TTS error: %s", error)
            self._publish("speech.error", kind="tts", message=str(error), turn_id=turn_id)
            return False
        finally:
            # A cancelled or failed run may leave the worker mid-synthesis.
            # Cancellation unblocks enqueue(); draining the queue unblocks it
            # if it was waiting on a full queue.
            worker.join(timeout=1.0)
            while True:
                try:
                    _, leftover_path, _ = ready.get_nowait()
                except queue.Empty:
                    break
                cleanup(leftover_path)

        return True

    # -------------------------------------------------

    @staticmethod
    def _read_wav_float32(path: str):
        """Decode a PCM WAV into a float32 ``(frames, channels)`` array."""

        with wave.open(path, "rb") as wav_file:

            channels = wav_file.getnchannels()
            width = wav_file.getsampwidth()
            rate = wav_file.getframerate()
            raw = wav_file.readframes(wav_file.getnframes())

        if not raw:
            return None, rate, channels

        if width == 1:

            data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0

        elif width == 2:

            data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0

        elif width == 3:

            triplets = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
            values = triplets[:, 0] | (triplets[:, 1] << 8) | (triplets[:, 2] << 16)
            values = np.where(values >= 1 << 23, values - (1 << 24), values)
            data = values.astype(np.float32) / float(1 << 23)

        elif width == 4:

            data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0

        else:

            raise ValueError("unsupported WAV sample width: %s" % width)

        return data.reshape(-1, channels), rate, channels

    @staticmethod
    def _wav_duration_s(path: str) -> float:
        """Return the playback length of a WAV file in seconds."""

        try:

            with wave.open(path, "rb") as wav_file:

                rate = wav_file.getframerate() or 1
                return wav_file.getnframes() / float(rate)

        except Exception:  # noqa: BLE001 - duration is only a timeout hint

            return 0.0

    def _play_with_output_stream(self, path: str) -> bool:
        """Play a WAV in small blocks on an interruptible output stream.

        This is the primary Windows path. ``winsound.PlaySound`` blocks the
        playing thread for the whole file and cannot be reliably purged
        from another thread on current Windows, which is why a live
        barge-in never actually silenced Greeto. Writing blocks to a
        sounddevice stream lets :meth:`stop` abort the stream (and thus the
        pending write) in a few milliseconds.

        Returns False when no output device is usable, so the caller can
        fall back to a system player.
        """

        try:

            import sounddevice as sd

        except Exception:  # noqa: BLE001 - optional at runtime

            return False

        try:

            samples, rate, channels = self._read_wav_float32(path)

        except Exception as error:  # noqa: BLE001

            self.logger.warning("Could not decode the generated WAV: %s", error)

            return False

        if samples is None or not len(samples):

            return False

        try:

            stream = sd.OutputStream(
                samplerate=rate,
                channels=channels,
                dtype="float32",
            )

        except Exception as error:  # noqa: BLE001 - no device / busy device

            self.logger.warning(
                "sounddevice output unavailable (%s); using a system player.",
                error,
            )

            return False

        block = self._playback_block_frames

        try:

            with self._playback_lock:

                self._active_stream = stream

            stream.start()

            for start in range(0, len(samples), block):

                if self._stop_event.is_set() or self._cancelled:

                    break

                stream.write(np.ascontiguousarray(samples[start : start + block]))

        except Exception as error:  # noqa: BLE001

            if self._stop_event.is_set() or self._cancelled:

                # stop() aborted the stream underneath write(); that is the
                # interruption working, not a playback failure.
                return True

            self.logger.warning(
                "sounddevice playback failed (%s); using a system player.",
                error,
            )

            return False

        finally:

            with self._playback_lock:

                if self._active_stream is stream:

                    self._active_stream = None

            try:

                stream.stop()

            except Exception:

                pass

            try:

                stream.close()

            except Exception:

                pass

        return True

    def _play_with_winsound(self, path: str) -> bool:
        """Chunked asynchronous winsound playback for a missing stream."""

        import winsound

        duration = self._wav_duration_s(path)

        winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)

        deadline = time_module.perf_counter() + duration + 1.0

        try:

            while time_module.perf_counter() < deadline:

                if self._stop_event.wait(0.02):

                    break

        finally:

            try:

                winsound.PlaySound(None, winsound.SND_PURGE)

            except Exception:

                pass

            try:

                winsound.PlaySound(None, 0)

            except Exception:

                pass

        return True

    def _play_audio(self, path: str) -> bool:
        """Play generated WAV file, interruptible by :meth:`stop`.

        Returns True when a player handled it.
        """
        if self._cancelled:
            return False

        if self._play_with_output_stream(path):

            return True

        if self._cancelled or self._stop_event.is_set():

            return False

        if sys.platform.startswith("win"):

            return self._play_with_winsound(path)

        for player in ("aplay", "paplay", "ffplay"):
            if shutil.which(player) is None:
                continue

            if player == "ffplay":
                command = [
                    player,
                    "-nodisp",
                    "-autoexit",
                    "-loglevel",
                    "quiet",
                    path,
                ]
            else:
                command = [player, path]

            process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            with self._playback_lock:
                self._active_player = process

            try:
                process.wait()
            finally:
                with self._playback_lock:
                    self._active_player = None

            return True

        logger.warning("No audio player found. Tried aplay, paplay and ffplay.")
        return False

    # -------------------------------------------------

    def _publish(self, event_name: str, **payload) -> None:
        """Publish an event when an event bus is available."""
        if self.event_bus is not None:
            try:
                self.event_bus.publish(event_name, **payload)
            except Exception as error:  # noqa: BLE001
                self.logger.warning("Event publish failed for %s: %s", event_name, error)