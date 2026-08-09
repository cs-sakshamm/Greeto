"""
Prompt Manager.

Builds prompts for the local LLM.
"""

from __future__ import annotations


class PromptManager:
    """
    Builds prompts for Gemma.
    """

    def __init__(self) -> None:

        self.system_prompt = """
You are the official AI Admission Receptionist of ITS Engineering College.

You MUST answer ONLY in valid JSON.

The JSON format is EXACTLY:

{
  "reply": "your answer",
  "emotion": "happy",
  "action": "answer"
}

Rules:

1. Output ONLY JSON.
2. Do NOT output markdown.
3. Do NOT output ```json.
4. Do NOT explain anything.
5. Do NOT output any text before or after JSON.
6. reply must contain a natural English sentence.
7. emotion must be one of:
   happy
   neutral
   empathetic
8. action must always be:
   answer
9. Use ONLY the College Knowledge provided below.
10. Never invent facts.
11. If the answer is not found in College Knowledge, reply:

{
  "reply":"I couldn't find that information in the college database.",
  "emotion":"neutral",
  "action":"answer"
}
"""

    # -------------------------------------------------------------

    def build_prompt(
        self,
        user_message: str,
        history: list[dict[str, str]],
        context: str = "",
    ) -> str:

        prompt = self.system_prompt.strip()

        prompt += "\n\n========== COLLEGE KNOWLEDGE ==========\n"

        if context.strip():
            prompt += context.strip()
        else:
            prompt += "No relevant college knowledge found."

        prompt += "\n\n========== CONVERSATION ==========\n"

        recent_history = history[-6:]

        for message in recent_history:

            role = message.get("role", "user").capitalize()

            content = message.get("content", "").strip()

            prompt += f"{role}: {content}\n"

        prompt += f"\nUser: {user_message.strip()}\n"

        prompt += """
Assistant:

Remember:

Return ONLY this JSON format.

{
  "reply":"...",
  "emotion":"happy",
  "action":"answer"
}
"""

        return prompt