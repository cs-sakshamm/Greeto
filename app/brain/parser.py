"""
LLM Response Parser.

Converts Gemma output into safe structured data.
"""

from __future__ import annotations

import json

from dataclasses import dataclass


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
    Parses and validates LLM JSON responses.
    """

    DEFAULT_RESPONSE = LLMResponse(
        reply=(
            "I will connect you with "
            "the admission team."
        ),
        emotion="neutral",
        action="handover",
    )


    def parse(
        self,
        text: str,
    ) -> LLMResponse:
        """
        Parse raw LLM output.

        Args:
            text:
                Raw Gemma output.

        Returns:
            Valid LLMResponse object.
        """

        try:

            json_text = self._extract_json(
                text
            )

            data = json.loads(
                json_text
            )


            return LLMResponse(

                reply=data.get(
                    "reply",
                    self.DEFAULT_RESPONSE.reply,
                ),

                emotion=data.get(
                    "emotion",
                    "neutral",
                ),

                action=data.get(
                    "action",
                    "answer",
                ),
            )


        except Exception:

            return self.DEFAULT_RESPONSE



    def _extract_json(
        self,
        text: str,
    ) -> str:
        """
        Extract JSON block from model output.
        """

        start = text.find("{")

        end = text.rfind("}")


        if start == -1 or end == -1:

            raise ValueError(
                "No JSON found"
            )


        return text[
            start:end + 1
        ]