"""System prompt and output schema for the AI-exposure classifier.

The system prompt is docs/codebook.md, used as is. The prompt version is read from the
codebook heading ("# AI Exposure Codebook — v1"), so the two cannot drift apart.

OUTPUT_SCHEMA is a plain JSON Schema dict. It is deliberately not tied to one API
method, so the same schema can be used for both Haiku and Sonnet.
"""

import re

from config import PROJECT_ROOT

CODEBOOK_PATH = PROJECT_ROOT / "docs" / "codebook.md"

CATEGORIES = ["ENABLER", "DISRUPTED", "BENEFICIARY", "LIMITED"]
CONFIDENCE_LEVELS = ["high", "medium", "low"]

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "enum": CATEGORIES,
            "description": "The first category in codebook rule order that fits.",
        },
        "confidence": {
            "type": "string",
            "enum": CONFIDENCE_LEVELS,
            "description": "How confident you are in the category. Must be low if knows_company is false.",
        },
        "knows_company": {
            "type": "boolean",
            "description": "True if you know what this specific company does; false if inferring from name, country and industry.",
        },
        "rationale": {
            "type": "string",
            "description": "One sentence, at most 25 words, naming the concrete mechanism.",
        },
    },
    "required": ["category", "confidence", "knows_company", "rationale"],
    "additionalProperties": False,
}


def read_codebook() -> str:
    return CODEBOOK_PATH.read_text(encoding="utf-8")


def get_prompt_version(codebook: str) -> str:
    """Return the version from the codebook heading, e.g. 'v1'."""
    first_line = codebook.splitlines()[0]
    match = re.search(r"\b(v\d+(?:\.\d+)?)\s*$", first_line)
    if not match:
        raise ValueError(f"No version found in codebook heading: {first_line!r}")
    return match.group(1)


def build_system_prompt() -> str:
    return read_codebook().strip()


PROMPT_VERSION = get_prompt_version(read_codebook())


if __name__ == "__main__":
    import json

    prompt = build_system_prompt()
    print(f"Prompt version: {PROMPT_VERSION}")
    print(f"System prompt:  {len(prompt):,} characters, {len(prompt.split()):,} words")
    print(json.dumps(OUTPUT_SCHEMA, indent=2))
