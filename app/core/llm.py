"""The only place that talks to OpenAI. Swap providers by editing this file."""
import base64
import json
import logging
from functools import lru_cache

from app.core.config import settings

log = logging.getLogger("athleteiq.llm")


class LLMError(Exception):
    pass


def llm_available() -> bool:
    return bool(settings.OPENAI_API_KEY)


@lru_cache
def _client():
    from openai import OpenAI

    return OpenAI(api_key=settings.OPENAI_API_KEY, timeout=180, max_retries=2)


def extract_json(text: str) -> dict:
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e == -1:
        raise ValueError("no JSON object in model output")
    return json.loads(text[s : e + 1])


def _complete(messages: list, model: str, max_tokens: int) -> dict:
    kwargs = dict(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},  # guarantees syntactically valid JSON
        max_completion_tokens=max_tokens,  # GPT-5 family: reasoning tokens count too, so keep this generous
    )
    if settings.LLM_REASONING_EFFORT:
        kwargs["reasoning_effort"] = settings.LLM_REASONING_EFFORT
    resp = _client().chat.completions.create(**kwargs)
    choice = resp.choices[0]
    text = choice.message.content or ""
    if not text.strip():
        raise ValueError(f"empty response (finish_reason={choice.finish_reason}); raise max tokens or lower reasoning effort")
    if choice.finish_reason == "length":
        log.warning("LLM output was cut off at the token limit (%s)", model)
    return extract_json(text)


def chat_json(system: str, user: str, model: str, max_tokens: int = 8000) -> dict:
    """Text in, JSON object out. The prompts must mention JSON (OpenAI requires it for json mode)."""
    return _complete([{"role": "system", "content": system}, {"role": "user", "content": user}], model, max_tokens)


def vision_json(images_jpeg: list[bytes], prompt: str, model: str, max_tokens: int = 1500) -> dict:
    content: list = [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(b).decode(), "detail": "low"}}
        for b in images_jpeg
    ]
    content.append({"type": "text", "text": prompt})
    return _complete([{"role": "user", "content": content}], model, max_tokens)
