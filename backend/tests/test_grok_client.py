"""Grok request and response shaping. No network."""
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import ai_analysis  # noqa: E402


def test_text_messages_become_responses_input():
    converted = ai_analysis.to_grok_input([
        {"role": "user", "content": "What leaked?"},
    ])
    assert converted == [{"role": "user", "content": "What leaked?"}]


def test_diary_image_becomes_a_data_url():
    converted = ai_analysis.to_grok_input([{
        "role": "user",
        "content": [
            {"type": "image", "source": {"media_type": "image/png", "data": "abc"}},
            {"type": "text", "text": "Read this."},
        ],
    }])
    parts = converted[0]["content"]
    assert parts[0] == {
        "type": "input_image",
        "image_url": "data:image/png;base64,abc",
        "detail": "high",
    }
    assert parts[1] == {"type": "input_text", "text": "Read this."}


def test_reply_text_ignores_reasoning_items():
    response = ai_analysis.grok_response_from_payload({
        "status": "completed",
        "output": [
            {"type": "reasoning", "summary": [{"text": "thinking"}]},
            {"type": "message", "content": [{"type": "output_text", "text": "{\"ok\": true}"}]},
        ],
    })
    assert ai_analysis.response_text(response) == "{\"ok\": true}"
    assert response.stop_reason == "end_turn"


def test_token_cap_is_reported_as_truncated():
    response = ai_analysis.grok_response_from_payload({
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "output": [{"type": "message", "content": [{"type": "output_text", "text": "{"}]}],
    })
    assert response.stop_reason == "max_tokens"
    with pytest.raises(ValueError, match="token limit"):
        ai_analysis.raise_if_truncated(response, "diary analysis")


def test_missing_key_names_the_env_variable(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="XAI_API_KEY is not set"):
        ai_analysis.get_client()
