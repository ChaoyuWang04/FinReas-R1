"""Minimal OpenAI-backed LLM wrapper for local data scripts.

This wrapper is intentionally small: it standardizes how repository utilities
call a hosted model without depending on notebook-specific code. It is currently
used by synthetic-data and reasoning-chain utilities, and returns plain text so
callers can own their own parsing logic.
"""

from __future__ import annotations

import os
from typing import Iterator, Optional

from openai import OpenAI


def _build_client() -> OpenAI:
    """Construct an OpenAI-compatible client from environment variables."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Please export your OpenAI API key before running this script."
        )

    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def _extract_text_from_response(response) -> str:
    """Normalize OpenAI Responses API output into a plain string."""
    text_parts: list[str] = []

    output = getattr(response, "output", None) or []
    for item in output:
        content = getattr(item, "content", None) or []
        for part in content:
            part_text = getattr(part, "text", None)
            if part_text:
                text_parts.append(part_text)

    if text_parts:
        return "".join(text_parts).strip()

    fallback_text = getattr(response, "output_text", None)
    if fallback_text:
        return fallback_text.strip()

    raise RuntimeError("OpenAI response did not contain any text output.")


def _stream_text(client: OpenAI, model_name: str, prompt: str) -> Iterator[str]:
    """Yield streamed text deltas while preserving the final accumulated text."""
    with client.responses.stream(
        model=model_name,
        input=prompt,
    ) as stream:
        for event in stream:
            if event.type == "response.output_text.delta":
                yield event.delta
        stream.until_done()


def call_llm(
    model_name: str,
    prompt: str,
    stream: bool = False,
    show_thinking: bool = False,
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
) -> str:
    """Call an OpenAI model and return plain text.

    Args:
        model_name: OpenAI model identifier, such as ``gpt-5``.
        prompt: Prompt text sent as the user input.
        stream: If True, prints streamed tokens and returns the full text.
        show_thinking: Compatibility argument for older wrappers. Ignored here.
        temperature: Optional sampling temperature.
        max_output_tokens: Optional cap for generated tokens.

    Returns:
        The model output as a plain string, suitable for downstream JSON parsing
        or batch request generation.
    """

    del show_thinking  # Compatibility-only parameter.

    client = _build_client()
    request_kwargs = {
        "model": model_name,
        "input": prompt,
    }
    if temperature is not None:
        request_kwargs["temperature"] = temperature
    if max_output_tokens is not None:
        request_kwargs["max_output_tokens"] = max_output_tokens

    if stream:
        chunks: list[str] = []
        for chunk in _stream_text(client, model_name=model_name, prompt=prompt):
            print(chunk, end="", flush=True)
            chunks.append(chunk)
        print()
        return "".join(chunks).strip()

    response = client.responses.create(**request_kwargs)
    return _extract_text_from_response(response)
