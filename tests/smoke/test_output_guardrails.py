from __future__ import annotations

import asyncio

from google.genai import types

from guardrails import output_guardrails


class _Response:
    def __init__(self, text: str):
        self.content = types.Content(
            role="model", parts=[types.Part.from_text(text=text)]
        )


def _response_text(response: _Response) -> str:
    return "".join(part.text or "" for part in response.content.parts)


def test_content_filter_redacts_configured_pii_and_credentials():
    result = output_guardrails.content_filter(
        "Call 0901234567 or test@vinbank.com; password is admin123; key sk-example123."
    )

    assert result["safe"] is False
    assert len(result["issues"]) == 4
    assert "0901234567" not in result["redacted"]
    assert "test@vinbank.com" not in result["redacted"]
    assert "admin123" not in result["redacted"]
    assert "sk-example123" not in result["redacted"]
    assert result["redacted"].count("[REDACTED]") == 4


def test_output_plugin_redacts_response_and_updates_counts():
    plugin = output_guardrails.OutputGuardrailPlugin(use_llm_judge=False)
    response = _Response("Customer email: test@vinbank.com")

    returned = asyncio.run(
        plugin.after_model_callback(callback_context=None, llm_response=response)
    )

    assert returned is response
    assert "test@vinbank.com" not in _response_text(response)
    assert "[REDACTED]" in _response_text(response)
    assert plugin.total_count == 1
    assert plugin.redacted_count == 1
    assert plugin.blocked_count == 0


def test_output_plugin_replaces_judge_rejected_response(monkeypatch):
    async def reject_response(_response_text: str) -> dict:
        return {"safe": False, "verdict": "UNSAFE"}

    monkeypatch.setattr(output_guardrails, "safety_judge_agent", object())
    monkeypatch.setattr(output_guardrails, "llm_safety_check", reject_response)
    plugin = output_guardrails.OutputGuardrailPlugin(use_llm_judge=True)
    response = _Response("The account transfer has been processed.")

    returned = asyncio.run(
        plugin.after_model_callback(callback_context=None, llm_response=response)
    )

    assert returned is response
    assert "UNSAFE" not in _response_text(response)
    assert "không thể cung cấp" in _response_text(response)
    assert plugin.total_count == 1
    assert plugin.blocked_count == 1