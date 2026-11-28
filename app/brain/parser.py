"""
LLM Response Parser.

Converts Gemma output into safe structured data.
Extracts valid JSON objects from fenced or surrounding text, repairs minor
JSON/brace issues, recovers plain-text replies when appropriate, and reserves
handover for explicit handoff requests.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

from app.brain.streaming import strip_placeholders, strip_reply_prefix

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """
    Structured response from the LLM.
    """

    reply: str
    emotion: str
    action: str


class ResponseParser:
    """
    Parses and validates LLM responses with multi-tier recovery.
    """

    DEFAULT_RESPONSE = LLMResponse(
        reply="I will connect you with the admission team.",
        emotion="neutral",
        action="handover",
    )

    VALID_EMOTIONS = {"neutral", "happy", "thinking", "speaking", "listening", "error"}
    VALID_ACTIONS = {"answer", "handover", "clarify"}

    EXPLICIT_HANDOVER_PATTERNS = [
        r"connect (?:you )?with (?:the )?admission",
        r"contact (?:the )?admission (?:team|office|department|cell)",
        r"visit (?:the )?admission (?:office|cell|counter|desk)",
        r"hand(?:ing)? over to (?:the )?admission",
        r"transfer(?:ring)? (?:you )?to (?:the )?admission",
        r"speak with (?:an|the) admission counselor",
    ]

    def parse(self, text: str) -> LLMResponse:
        """
        Parse raw LLM output into an LLMResponse.

        Args:
            text: Raw output string from LLM.

        Returns:
            Valid LLMResponse object.
        """
        if not text or not text.strip():
            return LLMResponse(
                reply="I'm sorry, I didn't catch that. Could you please repeat your question?",
                emotion="neutral",
                action="clarify",
            )

        raw = text.strip()

        # Tier 1: Try structured JSON extraction and parsing
        parsed = self._try_parse_json(raw)
        if parsed is not None:
            return parsed

        # Tier 2: Try regex field extraction for partial/broken JSON
        parsed = self._try_regex_fields(raw)
        if parsed is not None:
            return parsed

        # Tier 3: Plain text recovery
        return self._recover_plain_text(raw)

    def _try_parse_json(self, text: str) -> Optional[LLMResponse]:
        """Extract and parse a JSON object from text."""
        json_candidate = self._extract_json_candidate(text)
        if not json_candidate:
            return None

        # Clean common JSON issues (trailing commas, trailing junk)
        cleaned = re.sub(r",\s*([}\]])", r"\1", json_candidate)

        # Attempt direct JSON load
        data = None
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # Try repairing unclosed braces / truncated output
            data = self._repair_and_load_json(cleaned)

        if isinstance(data, dict):
            return self._build_response_from_dict(data)

        return None

    def _extract_json_candidate(self, text: str) -> Optional[str]:
        """Extract JSON substring, handling markdown fences and prose."""
        # Check for markdown code blocks: ```json ... ``` or ``` ... ```
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fence_match:
            return fence_match.group(1).strip()

        # Find first { and last }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start:end + 1].strip()

        # If there's an opening { with no closing }, might be truncated
        if start != -1 and end == -1:
            return text[start:].strip()

        return None

    def _repair_and_load_json(self, text: str) -> Optional[dict]:
        """Attempt to repair common JSON truncations and decode."""
        candidates = [
            text + "}",
            text + '"}',
            text + '"}' + "\n}",
            text.rstrip(",") + "}",
            text.rstrip(',"') + '"}',
        ]
        for candidate in candidates:
            try:
                data = json.loads(candidate)
                if isinstance(data, dict):
                    return data
            except (json.JSONDecodeError, ValueError):
                continue
        return None

    def _try_regex_fields(self, text: str) -> Optional[LLMResponse]:
        """Extract 'reply', 'emotion', and 'action' fields using regex when JSON fails."""
        reply_match = re.search(r'"reply"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
        if not reply_match:
            # Try single quotes or multiline unescaped
            reply_match = re.search(r'["\']reply["\']\s*:\s*["\'](.*?)["\'](?:\s*,\s*|\s*})', text, re.DOTALL)

        if reply_match:
            raw_reply = reply_match.group(1)
            # Unescape basic escapes
            reply = raw_reply.replace('\\"', '"').replace("\\n", " ").strip()

            emotion_match = re.search(r'"emotion"\s*:\s*"([^"]+)"', text)
            emotion = emotion_match.group(1).lower().strip() if emotion_match else "neutral"

            action_match = re.search(r'"action"\s*:\s*"([^"]+)"', text)
            action = action_match.group(1).lower().strip() if action_match else "answer"

            return self._build_response_from_dict({
                "reply": reply,
                "emotion": emotion,
                "action": action,
            })

        return None

    def _recover_plain_text(self, text: str) -> LLMResponse:
        """
        Recover plain text response when no JSON structure could be parsed.
        Never blindly default to handover if the LLM provided an answer.
        """
        cleaned = text.strip()

        # Strip code block wrappers if any
        cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.MULTILINE).strip()

        # A small model sometimes runs past its answer into the next
        # knowledge Q&A block. Keep only the first answer.
        cleaned = re.split(
            r"\n\s*-{3,}\s*\n|\n\s*(?:Question|User|Assistant|System)\s*:",
            cleaned,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip()

        # Strip typical prompt echoes / speaker tags with the same helper
        # the streaming path uses, so the two can never drift apart.
        cleaned = strip_placeholders(strip_reply_prefix(cleaned).strip())

        if not cleaned:
            return LLMResponse(
                reply="I'm sorry, I could not understand that. Could you please rephrase?",
                emotion="neutral",
                action="clarify",
            )

        # Determine if the text represents an explicit handover request
        is_handover = self._is_explicit_handover(cleaned)

        return LLMResponse(
            reply=cleaned,
            emotion="neutral",
            action="handover" if is_handover else "answer",
        )

    def _build_response_from_dict(self, data: dict) -> LLMResponse:
        """Normalize dict keys and values into an LLMResponse."""
        reply = str(data.get("reply", "") or "").strip()
        reply = strip_placeholders(strip_reply_prefix(reply).strip())
        emotion = str(data.get("emotion", "neutral") or "neutral").lower().strip()
        action = str(data.get("action", "answer") or "answer").lower().strip()

        # If reply is empty, check if action was handover or provide fallback
        if not reply:
            if action == "handover":
                reply = self.DEFAULT_RESPONSE.reply
            else:
                reply = "I'm sorry, I don't have enough information on that."

        # Validate emotion
        if emotion not in self.VALID_EMOTIONS:
            emotion = "neutral"

        # Validate action
        if action not in self.VALID_ACTIONS:
            action = "answer"

        # Check if the reply itself explicitly signals a handover even if action said answer
        if action == "answer" and self._is_explicit_handover(reply):
            action = "handover"

        return LLMResponse(
            reply=reply,
            emotion=emotion,
            action=action,
        )

    def _is_explicit_handover(self, text: str) -> bool:
        """Check if text genuinely asks for human admission team escalation."""
        text_lower = text.lower()
        return any(re.search(pat, text_lower) for pat in self.EXPLICIT_HANDOVER_PATTERNS)
