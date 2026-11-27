"""Local LLM wrapper.

Runs a small GGUF model through llama.cpp. The active default is a
~0.5B Qwen instruct model (Qwen2.5-0.5B / Qwen3-0.6B class) so the kiosk
can start speaking within the 2-3 second target after the visitor stops
talking. Every setting that affects latency or output length is read from
``config/llm.yaml``:

    model.path / fallback_paths / n_ctx / n_batch / n_threads /
    n_gpu_layers
    generation.max_tokens / temperature / top_p / repeat_penalty /
    enable_thinking

Gemma-3-1B can still be selected through the same config for A/B
comparison; nothing in the conversation layer is Gemma-specific.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from llama_cpp import Llama

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _as_bool(value, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "yes", "1", "on"):
            return True
        if lowered in ("false", "no", "0", "off"):
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _as_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class LocalLLM:
    """Config-driven local llama.cpp model."""

    # Stop sequences keep a small instruct model from rambling past the
    # JSON object or emitting another role turn.
    CHATML_STOPS = ("<|im_end|>", "<|endoftext|>", "<|im_start|>")

    def __init__(self, config=None, model_path: str | Path | None = None) -> None:

        self.lock = threading.Lock()

        self.config = config if config is not None else self._load_config()

        self.model_path = self._resolve_model_path(model_path)

        self.prompt_style = (
            "qwen" if "qwen" in self.model_path.name.lower() else "gemma"
        )

        self._enable_thinking = _as_bool(
            self._setting("generation", "enable_thinking", default=False),
            False,
        )

        # Qwen3 supports a soft reasoning switch. Qwen2.5 does not, so it
        # is only appended when the selected file is a Qwen3 GGUF.
        self.disable_thinking = (
            self.prompt_style == "qwen"
            and "qwen3" in self.model_path.name.lower()
            and not self._enable_thinking
        )

        self.max_tokens = max(
            16,
            _as_int(self._setting("generation", "max_tokens", default=96), 96),
        )
        self.temperature = _as_float(
            self._setting("generation", "temperature", default=0.2), 0.2
        )
        self.top_p = _as_float(
            self._setting("generation", "top_p", default=0.8), 0.8
        )
        self.repeat_penalty = _as_float(
            self._setting("generation", "repeat_penalty", default=1.05), 1.05
        )

        n_ctx = max(
            256,
            _as_int(self._setting("model", "n_ctx", default=1024), 1024),
        )
        n_batch = max(
            32,
            _as_int(self._setting("model", "n_batch", default=256), 256),
        )
        n_threads = max(
            1,
            _as_int(self._setting("model", "n_threads", default=6), 6),
        )
        n_gpu_layers = _as_int(
            self._setting("model", "n_gpu_layers", default=-1), -1
        )

        logger.info(
            "Loading LLM %s (style=%s n_ctx=%d n_batch=%d n_threads=%d "
            "n_gpu_layers=%d max_tokens=%d thinking=%s)",
            self.model_path.name,
            self.prompt_style,
            n_ctx,
            n_batch,
            n_threads,
            n_gpu_layers,
            self.max_tokens,
            "on" if self._enable_thinking else "off",
        )

        self.model = Llama(
            model_path=str(self.model_path),
            n_ctx=n_ctx,
            n_batch=n_batch,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

    # -------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------

    @staticmethod
    def _load_config():
        try:
            from app.core.config_manager import ConfigManager

            config = ConfigManager()
            config.load()
            return config
        except Exception as error:  # noqa: BLE001 - config is optional
            logger.warning("LLM configuration could not be loaded: %s", error)
            return None

    def _setting(self, section: str, key: str, default):
        if self.config is None:
            return default
        try:
            value = self.config.get_value("llm", section, key, default=default)
        except (AttributeError, TypeError, ValueError):
            return default
        return default if value is None else value

    def _resolve_model_path(self, explicit: str | Path | None) -> Path:
        configured = self._setting("model", "path", default=None)
        fallbacks = self._setting("model", "fallback_paths", default=[]) or []

        candidates: list[Path] = []
        for raw in (explicit, configured, *fallbacks):
            if not raw:
                continue
            path = Path(raw)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            if path not in candidates:
                candidates.append(path)

        for path in candidates:
            if path.exists():
                if configured:
                    requested = Path(configured)
                    if not requested.is_absolute():
                        requested = PROJECT_ROOT / requested
                    if path != requested and not requested.exists():
                        logger.warning(
                            "Configured LLM %s is missing; using %s instead.",
                            requested,
                            path,
                        )
                return path

        raise FileNotFoundError(
            "No LLM GGUF model found. Checked: %s. Download a Qwen "
            "0.5B/0.6B instruct GGUF into models/llm/ and update "
            "config/llm.yaml model.path."
            % ", ".join(str(path) for path in candidates)
        )

    # -------------------------------------------------------------
    # Generation
    # -------------------------------------------------------------

    def _sampling_options(self, max_tokens: int | None = None) -> dict:
        options = {
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repeat_penalty": self.repeat_penalty,
        }
        if self.prompt_style == "qwen":
            options["stop"] = list(self.CHATML_STOPS)
        return options

    def generate(self, prompt: str) -> str:
        """Generate a complete response."""

        with self.lock:

            output = self.model(prompt, **self._sampling_options())

        return output["choices"][0]["text"].strip()

    def stream_generate(self, prompt: str, max_tokens: int | None = None):
        """Stream a response token by token.

        The lock is held for the whole generation on purpose: one shared
        llama.cpp context cannot serve two conversations at once.
        """

        with self.lock:

            stream = self.model(
                prompt,
                stream=True,
                **self._sampling_options(max_tokens),
            )

            for chunk in stream:

                token = chunk["choices"][0]["text"]

                if token:
                    yield token
