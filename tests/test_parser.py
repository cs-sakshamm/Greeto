"""
Unit tests for ResponseParser and LLMResponse.
"""

import pytest
from app.brain.parser import ResponseParser, LLMResponse


@pytest.fixture
def parser():
    return ResponseParser()


def test_clean_json(parser):
    text = '{"reply": "The CSE department is in the main academic block.", "emotion": "happy", "action": "answer"}'
    resp = parser.parse(text)
    assert resp.reply == "The CSE department is in the main academic block."
    assert resp.emotion == "happy"
    assert resp.action == "answer"


def test_markdown_fenced_json(parser):
    text = """```json
{
  "reply": "B.Tech admission requires 50% in PCM in Class 12.",
  "emotion": "neutral",
  "action": "answer"
}
```"""
    resp = parser.parse(text)
    assert resp.reply == "B.Tech admission requires 50% in PCM in Class 12."
    assert resp.emotion == "neutral"
    assert resp.action == "answer"


def test_prose_around_json(parser):
    text = """Sure! Here is the admission details:
{"reply": "The annual tuition fee for B.Tech is INR 1,25,000.", "emotion": "happy", "action": "answer"}
Hope this information helps your inquiry!"""
    resp = parser.parse(text)
    assert resp.reply == "The annual tuition fee for B.Tech is INR 1,25,000."
    assert resp.emotion == "happy"
    assert resp.action == "answer"


def test_trailing_comma_json(parser):
    text = '{"reply": "Hostel facilities include 2-seater and 3-seater rooms.", "emotion": "neutral", "action": "answer",}'
    resp = parser.parse(text)
    assert resp.reply == "Hostel facilities include 2-seater and 3-seater rooms."
    assert resp.action == "answer"


def test_truncated_json_missing_brace(parser):
    text = '{"reply": "The college offers scholarships based on UPSEE rank.", "emotion": "happy", "action": "answer"'
    resp = parser.parse(text)
    assert resp.reply == "The college offers scholarships based on UPSEE rank."
    assert resp.emotion == "happy"
    assert resp.action == "answer"


def test_plain_text_recovery_no_handover(parser):
    text = "The college library has over 50,000 books and access to IEEE journals."
    resp = parser.parse(text)
    assert resp.reply == "The college library has over 50,000 books and access to IEEE journals."
    assert resp.action == "answer"
    assert resp.emotion == "neutral"


def test_plain_text_with_assistant_prefix(parser):
    text = "Assistant: Direct admission for MBA is available through management quota."
    resp = parser.parse(text)
    assert resp.reply == "Direct admission for MBA is available through management quota."
    assert resp.action == "answer"


def test_explicit_handover_plain_text(parser):
    text = "For fee concession details, I will connect you with the admission team at counter 3."
    resp = parser.parse(text)
    assert resp.action == "handover"
    assert "connect you with the admission team" in resp.reply


def test_empty_and_whitespace_input(parser):
    resp = parser.parse("")
    assert resp.action == "clarify"
    assert len(resp.reply) > 0

    resp2 = parser.parse("   \n\t  ")
    assert resp2.action == "clarify"
    assert len(resp2.reply) > 0


def test_invalid_emotion_and_action_normalized(parser):
    text = '{"reply": "Welcome to I.T.S.", "emotion": "super_excited", "action": "do_something"}'
    resp = parser.parse(text)
    assert resp.reply == "Welcome to I.T.S."
    assert resp.emotion == "neutral"  # Normalized to valid default
    assert resp.action == "answer"   # Normalized to valid default
