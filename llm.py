"""One small wrapper over the LLM providers used by the project."""
from __future__ import annotations

import os
from functools import lru_cache


class MissingAPIKey(RuntimeError):
    """Raised when the API key for a provider is not configured."""


def _key(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    raise MissingAPIKey(
        f"{names[0]} is not set. Add it to your .env file or enter it in the app sidebar."
    )


@lru_cache(maxsize=4)
def _gemini(api_key: str):
    from google import genai
    return genai.Client(api_key=api_key)


@lru_cache(maxsize=4)
def _groq(api_key: str):
    from groq import Groq
    return Groq(api_key=api_key)


def generate(prompt, provider, model, system=None, temperature=None, max_tokens=None) -> str:
    provider = provider.lower()

    if provider == "gemini":
        from google.genai import types
        cfg = {}
        if system:
            cfg["system_instruction"] = system
        if temperature is not None:
            cfg["temperature"] = temperature
        if max_tokens:
            cfg["max_output_tokens"] = max_tokens
        client = _gemini(_key("GOOGLE_API_KEY", "GEMINI_API_KEY"))
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(**cfg) if cfg else None,
        )
        return response.text or ""

    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]

    if provider == "groq":
        kwargs = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        client = _groq(_key("GROQ_API_KEY"))
        out = client.chat.completions.create(model=model, messages=messages, **kwargs)
        return out.choices[0].message.content or ""

    if provider == "ollama":
        import ollama
        options = {"temperature": temperature} if temperature is not None else None
        out = ollama.chat(model=model, messages=messages, options=options)
        return out["message"]["content"]

    raise ValueError(f"Unknown LLM provider '{provider}' (use gemini, groq or ollama).")
