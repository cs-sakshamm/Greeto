"""
Conversation Manager.

Controls the AI receptionist conversation and records stage-by-stage
latency for every turn:

    speech_end -> stt_done -> kb_done -> llm_first_token -> llm_done
    -> tts_first_audio -> playback_started

Flow:

Speech
 ↓
Conversation Manager
 ↓
Knowledge Base
 ↓
Prompt Manager
 ↓
Local LLM
 ↓
TTS
"""

from __future__ import annotations

import queue
import re
import threading
import time

from app.core.container import ServiceContainer

from app.brain.llm import LocalLLM
from app.brain.prompt_manager import PromptManager
from app.brain.parser import ResponseParser, LLMResponse
from app.brain.knowledge_base import KnowledgeBase
from app.brain.streaming import ReplyStreamExtractor, SentenceChunker


def _ms(seconds: float | None) -> str:
    """Format a duration in milliseconds for the latency log."""

    if seconds is None:
        return "n/a"

    return "%.0fms" % (max(0.0, seconds) * 1000.0)


class _SentenceSpeaker:
    """Play streamed sentences without blocking token generation.

    ``TTSService.speak`` synthesizes and then plays audio synchronously, so
    calling it inside the token loop would stall generation. A tiny worker
    queue lets the first sentence start while the rest of the answer is
    still being generated, while keeping utterance order intact.
    """

    def __init__(self, tts, turn_id: int) -> None:
        self.tts = tts
        self.turn_id = turn_id
        self.spoken_any = False
        self._queue: "queue.Queue[str | None]" = queue.Queue()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="SentenceSpeaker-%s" % turn_id,
        )
        self._closed = False

    def start(self) -> None:
        self._thread.start()

    def submit(self, text: str) -> None:
        if self._closed or not text:
            return
        self._queue.put(text)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)
        self._thread.join()

    def _run(self) -> None:
        while True:
            text = self._queue.get()

            if text is None:
                return

            try:
                try:
                    played = self.tts.speak(text, turn_id=self.turn_id)
                except TypeError:
                    played = self.tts.speak(text)

                if played:
                    self.spoken_any = True

            except Exception:  # noqa: BLE001 - one sentence must not kill TTS
                continue


class ConversationManager:
    """
    Main AI conversation controller.

    ``tts_service`` is the application's single shared speech output
    service, owned by the startup code rather than constructed here.
    """

    MAX_TIMING_ENTRIES = 32

    # Numeric facts a 0.5B model likes to invent. Any such value in the
    # reply must also appear in the retrieved knowledge, otherwise the
    # answer is replaced with a grounded fallback.
    _GROUNDING_PATTERNS = (
        re.compile(r"(?:₹|rs\.?\s*|inr\s*)[\d,]+", re.IGNORECASE),
        re.compile(
            r"\b[\d,.]+\s*(?:lakh|lakhs|lac|lacs|crore|crores|thousand|lpa)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\s+"
            r"(?:lakh|lakhs|lac|lacs|crore|crores|thousand)\b",
            re.IGNORECASE,
        ),
        re.compile(r"\b\d+(?:\.\d+)?\s*%"),
        re.compile(
            r"\b\d+(?:\.\d+)?\s*(?:percent|per\s*cent)\b",
            re.IGNORECASE,
        ),
        # Indian digit grouping for a bare amount without a currency symbol.
        re.compile(r"\b\d{1,3}(?:,\d{2,3}){1,3}\b"),
        re.compile(r"\b(?:19|20)\d{2}\b"),
        re.compile(r"\b\d{3,5}[- ]?\d{3}[- ]?\d{4}\b"),
    )

    GROUNDED_FALLBACK = (
        "I couldn't find that exact detail in the college database. "
        "Please contact the admission office at 1800-180-0840."
    )

    _FACTUAL_CUES = (
        "fee", "fees", "eligibility", "admission", "document", "hostel",
        "placement", "scholarship", "refund", "seat", "intake", "cutoff",
        "contact", "phone", "helpline", "address", "course", "branch",
        "percentage", "lpa", "deadline", "date", "counseling", "aktu",
        "uptac", "jee", "cat", "mat", "syllabus", "attendance",
    )

    @staticmethod
    def _prepare_spoken_answer(answer: str) -> str:
        """Make an answer pleasant to hear, not to read."""

        if not answer:

            return answer

        answer = re.sub(r"[*_`#]+", "", answer)
        answer = re.sub(
            r"^(?:Your answer|Answer|Greeto|Assistant|[AQ])\s*:\s*",
            "",
            answer,
            flags=re.IGNORECASE,
        ).strip()

        # A bare phone number is accurate but not a sentence. Give the
        # visitor the context the model left implicit.
        if re.fullmatch(r"[\d\s\-()+.]{7,}", answer):

            return "The admission helpline number is %s." % answer.strip()

        if "\n" not in answer and not re.search(
            r"^\s*[-*•]", answer, re.MULTILINE
        ):

            return answer

        parts = []

        for line in answer.splitlines():

            line = line.strip()

            line = re.sub(r"^(?:\d+[.)]|[-*•])\s*", "", line).strip()

            if not line:

                continue

            if line.endswith(":"):

                line = line[:-1]

            parts.append(line)

        flattened = ", ".join(parts)

        flattened = re.sub(r"\s+", " ", flattened)

        flattened = re.sub(r",\s*,", ",", flattened).strip(" ,")

        if flattened and not flattened.endswith((".", "?", "!")):

            flattened += "."

        return flattened

    @classmethod
    def _ground_reply(cls, reply: str, context: str) -> str:
        """Reject invented fees, percentages, years or phone numbers."""

        if not reply:

            return reply

        def normalise(value: str) -> str:
            return re.sub(r"[\s,]", "", value).casefold()

        haystack = normalise(context or "")

        for pattern in cls._GROUNDING_PATTERNS:

            for match in pattern.findall(reply):

                token = match if isinstance(match, str) else match[0]
                normalised = normalise(token)

                if normalised in haystack:
                    continue

                # The admission helpline is the sanctioned fallback and is
                # safe to speak even when the retrieved chunk does not repeat
                # it. Every other number still needs evidence.
                compact = re.sub(r"\D", "", normalised)
                if compact and (
                    "18001800840" in compact or compact in "18001800840"
                ):
                    continue

                return cls.GROUNDED_FALLBACK

        return reply

    def __init__(
        self,
        container: ServiceContainer,
        tts_service,
        llm=None,
    ) -> None:

        self.container = container

        self.logger = container.logger

        self.event_bus = container.event_bus

        self.llm = llm if llm is not None else LocalLLM()

        style = getattr(self.llm, "prompt_style", "qwen")

        config = getattr(container, "config", None)

        try:

            self.prompt_manager = PromptManager(style=style, config=config)

        except TypeError:

            # Test doubles may accept no constructor arguments.
            self.prompt_manager = PromptManager()

        if hasattr(self.prompt_manager, "style"):

            self.prompt_manager.style = style

        if hasattr(self.prompt_manager, "disable_thinking"):

            self.prompt_manager.disable_thinking = bool(
                getattr(self.llm, "disable_thinking", False)
            )

        self.parser = ResponseParser()

        self.stream_sentences = bool(
            self._llm_setting("streaming", "sentence_tts", default=True)
        )
        self.sentence_min_chars = int(
            self._llm_setting("streaming", "min_sentence_chars", default=8)
        )
        self.sentence_max_chars = int(
            self._llm_setting("streaming", "max_sentence_chars", default=240)
        )

        try:
            self.knowledge = KnowledgeBase(config=config)
        except TypeError:
            # Test doubles may accept no constructor arguments.
            self.knowledge = KnowledgeBase()

        rag = getattr(self.knowledge, "rag", {}) or {}
        retrieval = rag.get("retrieval", {}) if isinstance(rag, dict) else {}
        self.extractive_floor = float(
            retrieval.get("answer_confidence_floor", 0.55)
        )

        self.tts = tts_service

        self.history = []

        self.processing_lock = threading.Lock()

        self._turn_lock = threading.Lock()
        self._current_turn_id = 0

        self.last_request = 0

        # turn_id -> monotonic stage timestamps used by the latency log.
        self._timings: dict[int, dict] = {}

        self.register_events()

        self.logger.info(
            "Conversation Manager started (LLM style=%s).",
            style,
        )

    # -------------------------------------------------

    def register_events(
        self,
    ):
        self.event_bus.subscribe(
            "speech_recognized",
            self.handle_speech,
        )
        self.event_bus.subscribe(
            "greeting.speak_requested",
            self.handle_greeting,
        )
        # Playback start is published by TTSService immediately before
        # audio is played; it closes the speech_end -> audible measurement.
        self.event_bus.subscribe(
            "speech.speaking_started",
            self._on_playback_started,
        )

    # -------------------------------------------------

    @staticmethod
    def _is_valid_known_name(name) -> bool:
        """A greeting is only valid for a real, recognized identity."""

        if not isinstance(name, str):

            return False

        cleaned = name.strip()

        if not cleaned:

            return False

        return cleaned.casefold() not in {
            "unknown",
            "none",
            "visitor",
            "null",
            "n/a",
        }

    def _llm_setting(self, section: str, key: str, default=None):
        config = getattr(self.container, "config", None)

        if config is None:
            return default

        try:
            return config.get_value("llm", section, key, default=default)
        except (AttributeError, TypeError, ValueError):
            return default

    @classmethod
    def _looks_factual(cls, text: str) -> bool:
        """Return True when the visitor is asking for institutional facts."""

        lowered = (text or "").casefold()

        return any(cue in lowered for cue in cls._FACTUAL_CUES)

    def _speak_text(self, text: str, turn_id: int):
        """Call TTS across both the current and legacy test-double API."""

        try:
            return self.tts.speak(text, turn_id=turn_id)
        except TypeError:
            return self.tts.speak(text)

    def _retrieve(self, text: str):
        """Return (context, confidence, has_evidence) from the KB.

        Real :class:`KnowledgeBase` instances expose
        ``search_with_confidence``; lightweight test doubles only implement
        ``search``. Checking the class (not the instance) keeps a
        ``MagicMock`` from being mistaken for a real retriever.
        """

        if hasattr(type(self.knowledge), "search_with_confidence"):
            result = self.knowledge.search_with_confidence(text)
            return (
                getattr(result, "context", "") or "",
                getattr(result, "confidence", None),
                bool(getattr(result, "has_evidence", False)),
            )

        context = self.knowledge.search(text)

        return context or "", None, bool(context)

    def _stream_sentence(self, sentence: str, context: str) -> str:
        """Ground and format one streamed sentence before it is spoken."""

        grounded = self._ground_reply(sentence, context)

        if grounded != sentence:
            self.logger.warning(
                "Ungrounded numeric claim in streamed sentence rejected: %r",
                sentence[:160],
            )

        return self._prepare_spoken_answer(grounded)

    @staticmethod
    def _is_refusal(answer: str) -> bool:
        lowered = (answer or "").casefold()

        return (
            "couldn't find" in lowered and "1800-180-0840" in lowered
        ) or (
            "contact the admission office" in lowered
            and "1800-180-0840" in lowered
        )

    @staticmethod
    def _looks_like_refusal_start(sentence: str) -> bool:
        """Detect the first half of a refusal before the phone number."""

        lowered = (sentence or "").casefold()

        return (
            "couldn't find" in lowered
            or "could not find" in lowered
            or "contact the admission office" in lowered
        )

    @staticmethod
    def _extractive_answer(context: str) -> str:
        """Return the first atomic Q&A answer from retrieved context.

        This is the deterministic safety net for the invariant "never hand
        off when factual knowledge exists". The answer is copied verbatim
        from the retrieved chunk, so it cannot invent a number.
        """

        if not context:
            return ""

        match = re.search(
            r"Q:\s*.*?\nA:\s*(.+?)(?:\n\s*\nSource:|\Z)",
            context,
            re.DOTALL,
        )

        if not match:
            return ""

        answer = re.sub(r"\s+", " ", match.group(1)).strip()

        # Keep the first two sentences: a spoken kiosk answer should stay
        # short even when the source answer is a long checklist.
        sentences = re.split(r"(?<=[.!?])\s+", answer)

        return " ".join(sentences[:2]).strip()

    def handle_greeting(
        self,
        text: str = "",
        name: str | None = None,
        **kwargs,
    ):
        """Speak a visitor greeting requested for a recognized person.

        Unknown faces never reach this method: KioskPresenter only
        publishes ``greeting.speak_requested`` for a valid known name and
        VisionService only announces a recognized identity once per
        sitting. The guards here are defence in depth.
        """

        if not text or not text.strip():

            return

        if not self._is_valid_known_name(name):

            self.logger.debug(
                "Ignoring greeting without a recognized identity: %r",
                name,
            )

            return

        # Do not interrupt an answer that is already being spoken or
        # generated. The visitor is already being served, so the greeting
        # is redundant at that point.
        if self.processing_lock.locked():

            self.logger.info(
                "Skipping greeting for %s: a conversation turn is active.",
                name,
            )

            return

        if getattr(self.tts, "is_speaking", lambda: False)():

            self.logger.info(
                "Skipping greeting for %s: TTS is already speaking.",
                name,
            )

            return

        with self._turn_lock:

            self._current_turn_id += 1
            turn_id = self._current_turn_id

        threading.Thread(
            target=self._speak_greeting,
            args=(text.strip(), name.strip(), turn_id),
            daemon=True,
        ).start()

    def _speak_greeting(self, text: str, name: str, turn_id: int):
        with self.processing_lock:

            if self._current_turn_id != turn_id:
                return

            try:
                self.logger.info(
                    f"Speaking visitor greeting for {name} [Turn {turn_id}]: {text}"
                )
                try:
                    self.tts.speak(text, turn_id=turn_id)
                except TypeError:
                    self.tts.speak(text)
            except Exception as error:
                self.logger.exception("Failed to speak visitor greeting: %s", error)

    # -------------------------------------------------

    def handle_speech(
        self,
        text: str,
        t_speech_end: float | None = None,
        t_stt_done: float | None = None,
        **kwargs,
    ):
        if not text:
            return

        text = text.strip()

        ignored = [
            "[blank_audio]",
            "blank_audio",
            "silence",
            "[silence]",
        ]

        if text.lower() in ignored:
            return

        if len(text) < 3:
            return

        now = time.perf_counter()

        if t_stt_done is None:
            t_stt_done = now

        if t_speech_end is None:
            t_speech_end = t_stt_done

        with self._turn_lock:
            now_wall = time.time()
            if now_wall - self.last_request < 0.3:
                return

            self.last_request = now_wall
            self._current_turn_id += 1
            turn_id = self._current_turn_id

        # If previous turn is synthesizing or playing, interrupt it
        if hasattr(self.tts, "cancel"):
            try:
                self.tts.cancel()
            except Exception:
                pass

        threading.Thread(
            target=self.process_message,
            args=(text, turn_id, t_speech_end, t_stt_done),
            daemon=True,
        ).start()

    # -------------------------------------------------

    def process_message(
        self,
        text: str,
        turn_id: int = 0,
        t_speech_end: float | None = None,
        t_stt_done: float | None = None,
    ):
        # Allow manual direct calls where turn_id might default to 0
        if turn_id == 0:
            with self._turn_lock:
                self._current_turn_id += 1
                turn_id = self._current_turn_id

        if self._current_turn_id != turn_id:
            self.logger.info(
                f"Turn {turn_id} superseded before lock (active={self._current_turn_id}). Dropping."
            )
            return

        with self.processing_lock:
            if self._current_turn_id != turn_id:
                self.logger.info(
                    f"Turn {turn_id} superseded after lock (active={self._current_turn_id}). Dropping."
                )
                return

            t_start = time.perf_counter()

            if t_speech_end is None:
                t_speech_end = t_start

            if t_stt_done is None:
                t_stt_done = t_start

            speaker = None

            try:
                self.logger.info(
                    f"User [Turn {turn_id}]: {text}"
                )

                self.event_bus.publish(
                    "conversation.processing",
                    text=text,
                    turn_id=turn_id,
                )

                self.history.append(
                    {
                        "role": "user",
                        "content": text,
                    }
                )

                if len(self.history) > 6:
                    self.history = self.history[-6:]

                t_kb_start = time.perf_counter()

                context, confidence, has_evidence = self._retrieve(text)

                t_kb_done = time.perf_counter()

                self.logger.info(
                    "Retrieved context length: %d confidence=%s (%.1fms)",
                    len(context),
                    "n/a" if confidence is None else "%.3f" % confidence,
                    (t_kb_done - t_kb_start) * 1000.0,
                )

                if context:
                    self.logger.debug(f"Context preview:\n{context[:500]}")

                self._timings[turn_id] = {
                    "speech_end": t_speech_end,
                    "stt_done": t_stt_done,
                    "kb_done": t_kb_done,
                    "llm_first_token": None,
                    "llm_done": None,
                    "playback_logged": False,
                }

                if len(self._timings) > self.MAX_TIMING_ENTRIES:

                    for stale in list(self._timings)[:-self.MAX_TIMING_ENTRIES]:

                        self._timings.pop(stale, None)

                # Confidence gate: a factual question with no retrieved
                # evidence must not be handed to a 0.5B model to improvise.
                # General conversation still goes through the LLM.
                if (
                    confidence is not None
                    and not has_evidence
                    and self._looks_factual(text)
                    and getattr(self.knowledge, "confidence_gate", True)
                ):

                    answer = self.GROUNDED_FALLBACK
                    response = LLMResponse(
                        reply=answer,
                        emotion="neutral",
                        action="answer",
                    )

                    t_llm_done = time.perf_counter()
                    self._timings[turn_id]["llm_first_token"] = t_llm_done
                    self._timings[turn_id]["llm_done"] = t_llm_done

                    self.logger.info(
                        "LATENCY turn=%s retrieval confidence below gate; "
                        "answering with grounded helpline fallback (%s)",
                        turn_id,
                        _ms(t_llm_done - t_speech_end),
                    )

                    self.history.append(
                        {"role": "assistant", "content": answer}
                    )

                    self.event_bus.publish(
                        "llm_response_ready",
                        response=response,
                        turn_id=turn_id,
                    )

                    if self._current_turn_id == turn_id:
                        self._speak_text(answer, turn_id)

                    return

                if self._current_turn_id != turn_id:
                    self.logger.info(f"Turn {turn_id} superseded before LLM prompt build.")
                    return

                prompt = self.prompt_manager.build_prompt(
                    user_message=text,
                    history=self.history,
                    context=context,
                )

                self.logger.info("Generating AI response...")

                response_text = ""

                t_first_token: float | None = None

                # Stream tokens and speak completed sentences while the rest
                # of the answer is still being generated. The extractor
                # understands the JSON "reply" field, so raw prompt tokens
                # never reach the speaker.
                extractor = ReplyStreamExtractor()
                chunker = SentenceChunker(
                    min_chars=self.sentence_min_chars,
                    max_chars=self.sentence_max_chars,
                )
                submitted_sentences = 0
                skipped_refusal = False

                if self.stream_sentences:
                    speaker = _SentenceSpeaker(self.tts, turn_id)
                    speaker.start()

                for token in self.llm.stream_generate(prompt):
                    if self._current_turn_id != turn_id:

                        self.logger.info(
                            f"Turn {turn_id} superseded during LLM generation. Aborting."
                        )
                        return
                    if t_first_token is None:
                        t_first_token = time.perf_counter()
                        self._timings[turn_id]["llm_first_token"] = t_first_token
                    response_text += token

                    if speaker is not None:
                        decoded = extractor.feed(token)

                        if decoded:
                            for sentence in chunker.feed(decoded):
                                spoken = self._stream_sentence(sentence, context)

                                # Do not speak a refusal while confident
                                # evidence exists; the extractive answer is
                                # submitted after generation instead.
                                if (
                                    has_evidence
                                    and confidence is not None
                                    and confidence >= self.extractive_floor
                                    and (
                                        self._looks_like_refusal_start(spoken)
                                        or skipped_refusal
                                    )
                                ):
                                    skipped_refusal = True
                                    continue

                                speaker.submit(spoken)
                                submitted_sentences += 1

                t_llm_done = time.perf_counter()

                if t_first_token is None:
                    t_first_token = t_llm_done

                self._timings[turn_id]["llm_first_token"] = t_first_token
                self._timings[turn_id]["llm_done"] = t_llm_done

                if speaker is not None:
                    tail = chunker.flush()

                    if tail:
                        speaker.submit(self._stream_sentence(tail, context))
                        submitted_sentences += 1

                self.logger.info(
                    "LATENCY turn=%s speech_end->stt_done=%s "
                    "stt_done->kb_done=%s kb_done->llm_first_token=%s "
                    "llm_first_token->llm_done=%s "
                    "speech_end->llm_done=%s tokens=%d streamed_sentences=%d",
                    turn_id,
                    _ms(t_stt_done - t_speech_end),
                    _ms(t_kb_done - t_stt_done),
                    _ms(t_first_token - t_kb_done),
                    _ms(t_llm_done - t_first_token),
                    _ms(t_llm_done - t_speech_end),
                    len(response_text.split()),
                    submitted_sentences,
                )

                self.logger.info(
                    f"LLM Raw [Turn {turn_id}]: {response_text}"
                )

                response = self.parser.parse(response_text)

                if response is None:
                    return

                if self._current_turn_id != turn_id:
                    self.logger.info(
                        f"Turn {turn_id} superseded after parsing. Dropping response."
                    )
                    return

                answer = response.reply.strip()

                grounded = self._ground_reply(answer, context)

                if grounded != answer:

                    self.logger.warning(
                        "Ungrounded numeric claim rejected [Turn %s]: %r",
                        turn_id,
                        answer[:200],
                    )

                    answer = grounded

                answer = self._prepare_spoken_answer(answer)

                # A 0.5B model sometimes refuses even when the retrieved
                # context contains a precise answer. If the evidence is
                # confident and atomic, answer from it verbatim instead of
                # handing the visitor to the helpline.
                if (
                    self._is_refusal(answer)
                    and has_evidence
                    and confidence is not None
                    and confidence >= self.extractive_floor
                ):
                    extracted = self._extractive_answer(context)
                    if extracted:
                        self.logger.info(
                            "Model refused a confident retrieval; using "
                            "extractive grounded answer (confidence=%.3f).",
                            confidence,
                        )
                        answer = self._prepare_spoken_answer(
                            self._ground_reply(extracted, context)
                        )

                response.reply = answer

                self.history.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

                self.event_bus.publish(
                    "llm_response_ready",
                    response=response,
                    turn_id=turn_id,
                )

                if self._current_turn_id != turn_id:
                    self.logger.info(
                        f"Turn {turn_id} superseded before TTS. Skipping speech."
                    )
                    return

                self.logger.info(
                    f"Speaking [Turn {turn_id}]: {answer}"
                )

                if speaker is None:
                    self._speak_text(answer, turn_id)
                elif submitted_sentences == 0 or skipped_refusal:
                    # The stream did not contain a decodable reply (for
                    # example a repaired/truncated JSON object), or it was a
                    # refusal that has been replaced by a grounded extract.
                    speaker.submit(answer)

            except Exception as error:
                self.logger.exception(error)

                self.event_bus.publish(
                    "conversation.error",
                    message=str(error),
                    turn_id=turn_id,
                )

            finally:
                if speaker is not None:
                    speaker.close()

    # -------------------------------------------------

    def _on_playback_started(
        self,
        turn_id=None,
        tts_synth_ms=None,
        tts_ready_at=None,
        **kwargs,
    ):
        """Close the latency trace when audio actually starts playing."""

        if turn_id is None:

            return

        timing = self._timings.get(turn_id)

        if not timing:

            return

        # Long answers are spoken sentence by sentence. Only the first
        # playback start is the user-visible time-to-first-audio marker.
        if timing.get("playback_logged"):
            return

        timing["playback_logged"] = True

        now = time.perf_counter()

        synth_s = max(0.0, float(tts_synth_ms or 0.0)) / 1000.0

        ready = tts_ready_at if tts_ready_at is not None else now

        synth_start = ready - synth_s

        llm_done = timing.get("llm_done")

        self.logger.info(
            "LATENCY turn=%s llm_done->tts_synth_start=%s tts_synth=%s "
            "tts_ready->playback_started=%s "
            "TOTAL speech_end->playback_started=%s",
            turn_id,
            _ms(None if llm_done is None else synth_start - llm_done),
            _ms(synth_s),
            _ms(now - ready),
            _ms(now - timing["speech_end"]),
        )

    # -------------------------------------------------

    def clear_history(
        self,
    ):

        self.history.clear()
