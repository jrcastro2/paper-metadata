"""
Module 1 — Metadata Extractor

Takes raw paper text and uses Claude to extract structured metadata as JSON.
Demonstrates: schema-in-prompt, output validation, and graceful error handling.

Run:
    python extract.py
"""

import json
import os
import anthropic
from dotenv import load_dotenv
from data.sample_papers import SAMPLES

load_dotenv()

MODEL = "claude-haiku-4-5-20251001"

# The schema we want back. Giving the model a concrete example of the exact
# shape — including types and nullable fields — is the most reliable way to
# get consistent output. It acts as a contract.
SCHEMA = """
{
  "title": "string or null",
  "authors": ["list of strings, or empty list"],
  "affiliations": ["list of strings, or empty list"],
  "year": "integer or null",
  "keywords": ["list of strings, or empty list"],
  "identifiers": {
    "doi": "string or null",
    "arxiv_id": "string or null"
  }
}
"""

SYSTEM_PROMPT = """You are a precise scientific metadata extraction assistant.
Extract metadata from the provided text and return it as a JSON object.
Follow these rules strictly:
- Return ONLY the JSON object. No explanation, no markdown, no code fences.
- Use null for any field that cannot be found in the text. Do not guess or invent values.
- Do not add fields that are not in the schema.
- authors and affiliations should be lists even if there is only one.
- year must be an integer (e.g. 2017), not a string.
- arxiv_id should be just the ID portion (e.g. "1706.03762"), not the full URL."""


def build_prompt(text: str) -> str:
    return f"""Extract metadata from the following paper text.

Return a JSON object with exactly this schema:
{SCHEMA}

Paper text:
{text}"""


def validate(data: dict) -> list[str]:
    """
    Check the parsed dict against our expected schema.
    Returns a list of validation errors (empty = valid).
    """
    errors = []
    required_keys = {"title", "authors", "affiliations", "year", "keywords", "identifiers"}
    missing = required_keys - data.keys()
    if missing:
        errors.append(f"Missing fields: {missing}")

    for list_field in ("authors", "affiliations", "keywords"):
        if list_field in data and not isinstance(data[list_field], list):
            errors.append(f"'{list_field}' must be a list, got {type(data[list_field]).__name__}")

    if "year" in data and data["year"] is not None and not isinstance(data["year"], int):
        errors.append(f"'year' must be an integer or null, got {type(data['year']).__name__}: {data['year']!r}")

    if "identifiers" in data and not isinstance(data["identifiers"], dict):
        errors.append(f"'identifiers' must be a dict, got {type(data['identifiers']).__name__}")

    return errors


def extract_metadata(text: str, client: anthropic.Anthropic) -> dict | None:
    """
    Call Claude and parse the response as JSON.
    Returns the parsed dict, or None if parsing fails after retrying.
    """
    for attempt in range(2):  # one retry on parse failure
        response = client.messages.create(
            model=MODEL,
            max_tokens=600,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_prompt(text)}],
        )
        raw = response.content[0].text.strip()

        # Guard against the model wrapping output in markdown code fences
        if raw.startswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            if attempt == 0:
                print(f"    [warn] JSON parse error on attempt 1: {e} — retrying")
            else:
                print(f"    [error] JSON parse failed after retry. Raw output:\n{raw}")
                return None


def print_result(label: str, data: dict | None, errors: list[str]) -> None:
    print(f"\n{'=' * 60}")
    print(f"Sample: {label.upper()}")
    print(f"{'=' * 60}")
    if data is None:
        print("  FAILED: could not parse response as JSON")
        return
    if errors:
        print(f"  VALIDATION ERRORS: {errors}")
    else:
        print("  Validation: OK")
    print(json.dumps(data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set — add it to your .env file")

    client = anthropic.Anthropic(api_key=api_key)

    for label, text in SAMPLES:
        data = extract_metadata(text, client)
        errors = validate(data) if data is not None else []
        print_result(label, data, errors)
