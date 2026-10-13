"""Shared settings and the Anthropic client.

The API key is read from FUND_AI_ANTHROPIC_KEY in .env (never ANTHROPIC_API_KEY)
and passed explicitly to the client.
"""

import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODEL = "claude-haiku-5-5"
COMPARISON_MODEL = "claude-sonnet-5-5"
SEED = 42


def get_client() -> anthropic.Anthropic:
    """Load the key from .env and return an Anthropic client."""
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("FUND_AI_ANTHROPIC_KEY")
    if not api_key:
        raise RuntimeError("FUND_AI_ANTHROPIC_KEY is not set. Copy .env.example to .env and add your key.")
    return anthropic.Anthropic(api_key=api_key)
