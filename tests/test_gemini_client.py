import pytest

from src.llm.gemini_client import GeminiClient


def test_gemini_requires_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(
        ValueError,
        match="GEMINI_API_KEY",
    ):
        GeminiClient(model_name="test-model")