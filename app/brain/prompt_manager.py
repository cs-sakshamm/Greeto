"""Prompt Manager.

Builds compact prompts for the local small model.

A 0.5B-0.6B instruct model is fast only when the prompt stays small, so
the system rules are short, retrieved knowledge is hard-capped, and only
a few recent turns are included. Two formats are supported:

    qwen   ChatML (Qwen2.5 / Qwen3 instruct)
    gemma  plain role-tagged text (Gemma-3 fallback / A-B tests)

``/no_think`` is appended for Qwen3 models when thinking is disabled.
"""

from __future__ import annotations


class PromptManager:
    """Builds prompts for the configured local model."""

    SYSTEM_RULES = (
        "You are Greeto, the admission receptionist for I.T.S Engineering "
        "College, Greater Noida.\n"
        "Answer ONLY from the college knowledge provided. Never guess a fee, "
        "date or percentage that is not written there.\n"
        "If the knowledge does not explicitly contain the answer, reply with "
        '"I couldn\'t find that information in the college database. Please '
        'contact the admission office at 1800-180-0840."\n'
        "Reply with one JSON object only, no markdown:\n"
        '{"reply": "your answer", "emotion": "happy", "action": "answer"}\n'
        "emotion must be happy, neutral or empathetic. action must be "
        "answer.\n"
        "The reply must be 1-2 short, natural spoken sentences. Never use "
        "bullet points, lists or headings, and do not repeat the question."
    )

    @classmethod
    def system_prompt(cls, disable_thinking: bool = False) -> str:
        """Return the static system block used at the top of every prompt.

        Keeping this string identical across turns is what allows
        llama.cpp to reuse the KV-cache prefix. Training-data preparation
        uses the same helper so the fine-tuned model sees exactly the
        system rules it will see at inference time.
        """

        prompt = cls.SYSTEM_RULES

        if disable_thinking:
            prompt += "\n/no_think"

        return prompt

    def __init__(self, style: str = "qwen", config=None) -> None:

        self.style = "gemma" if style == "gemma" else "qwen"

        self.disable_thinking = False

        self.max_context_chars = 1100

        self.max_history_messages = 4

        if config is not None:

            try:

                self.max_context_chars = int(
                    config.get_value(
                        "llm",
                        "prompt",
                        "max_context_chars",
                        default=self.max_context_chars,
                    )
                )

                self.max_history_messages = int(
                    config.get_value(
                        "llm",
                        "prompt",
                        "max_history_messages",
                        default=self.max_history_messages,
                    )
                )

            except (AttributeError, TypeError, ValueError):

                pass

        self.max_context_chars = max(200, self.max_context_chars)

        self.max_history_messages = max(0, self.max_history_messages)

    # -------------------------------------------------------------

    @staticmethod
    def _is_followup(question: str) -> bool:
        """Return True only for questions that genuinely need context."""

        text = " ".join((question or "").casefold().split())

        if not text:
            return False

        cues = (
            "what about",
            "how about",
            "tell me more",
            "more about",
            "and the",
            "and what",
            "and how",
            "also",
            "the same",
            "that one",
            "those",
            "they",
            "there",
        )

        if any(cue in text for cue in cues):
            return True

        # Very short pronoun-led questions ("and hostels?", "for MBA?")
        # need the previous turn; anything longer should stand on its own
        # against freshly retrieved knowledge.
        return len(text.split()) <= 4 and text.startswith(
            ("it ", "that ", "this ", "they ", "and ", "also ", "for ")
        )

    def build_prompt(
        self,
        user_message: str,
        history: list[dict[str, str]],
        context: str = "",
    ) -> str:

        knowledge = (context or "").strip()

        if len(knowledge) > self.max_context_chars:

            knowledge = knowledge[: self.max_context_chars].rstrip() + "…"

        if not knowledge:

            knowledge = "No relevant college knowledge found."

        # A 0.5B model will happily copy the previous answer into a new
        # factual question if that answer is in the prompt. History is only
        # useful for genuine follow-ups ("what about MBA?", "tell me more"),
        # so it is included only when the question is short and refers back
        # to the previous turn.
        include_history = self.max_history_messages and self._is_followup(
            user_message
        )

        recent = history[-self.max_history_messages :] if include_history else []

        conversation = []

        for message in recent:

            role = str(message.get("role", "user")).capitalize()

            content = str(message.get("content", "")).strip()

            if content:

                conversation.append("%s: %s" % (role, content))

        question = (user_message or "").strip()

        if self.style == "gemma":

            system = self.system_prompt(self.disable_thinking)

            prompt = "System:\n%s\n\n" % system

            prompt += "College knowledge:\n%s\n\n" % knowledge

            if conversation:

                prompt += "Conversation so far:\n%s\n\n" % "\n".join(conversation)

            prompt += "User: %s\nAssistant:\n" % question

            return prompt

        system = self.system_prompt(self.disable_thinking)

        prompt = "<|im_start|>system\n%s<|im_end|>\n" % system

        prompt += "<|im_start|>user\nCollege knowledge:\n%s\n\n" % knowledge

        if conversation:

            prompt += "Conversation so far:\n%s\n\n" % "\n".join(conversation)

        prompt += "Question: %s<|im_end|>\n" % question

        prompt += "<|im_start|>assistant\n"

        return prompt
