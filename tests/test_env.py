import os

from dotenv import load_dotenv


load_dotenv()


def test_gemini_api_key_is_configured():
    api_key = os.getenv("GEMINI_API_KEY")

    assert api_key is not None
    assert api_key.strip() != ""