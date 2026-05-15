"""Anthropic Claude adapter.

Why this is shaped the way it is:

- ``temperature``/``top_p``/``top_k`` are *not* sent. Opus 4.7 returns 400
  if any of them appear; for pre-4.7 models they're harmless to omit and
  benchmarks should be reproducible without sampler tuning.
- The system prompt is wrapped in a ``cache_control: ephemeral`` block.
  For short prompts (< 4096 tokens on Opus 4.7) this is a no-op, but the
  shape is forward-compatible if we add few-shot examples to the system
  prompt later.
- The ``anthropic`` package is imported lazily inside ``__init__`` so
  ``italiabench`` can be imported without it (it's an optional extra).
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any

from italiabench.providers.base import (
    ProviderError,
    ProviderResponse,
    estimate_cost_usd,
)

if TYPE_CHECKING:
    pass


DEFAULT_SYSTEM_PROMPT = (
    "Sei un esperto di diritto, fisco, geografia, storia, cultura e procedure "
    "della pubblica amministrazione italiana. Rispondi in italiano alla domanda "
    "con una risposta fattuale, concisa e diretta. Se non sei certo di un dato "
    "preciso, dichiaralo apertamente invece di inventare. Non aggiungere "
    "disclaimer, commenti meta sulla domanda o riferimenti al fatto di essere "
    "un'IA."
)


class AnthropicProvider:
    """Adapter around the official ``anthropic`` Python SDK.

    Parameters
    ----------
    model:
        Bare model ID (no date suffix). Defaults to ``claude-opus-4-7``.
    api_key:
        Optional explicit key; otherwise the SDK reads ``ANTHROPIC_API_KEY``.
    system_prompt:
        Override the default Italian-knowledge system prompt.
    client:
        Inject a pre-built ``anthropic.Anthropic`` (or compatible mock) for
        tests. When provided, ``api_key`` is ignored.
    input_price_per_mtok / output_price_per_mtok:
        Override the cached pricing table for this instance.
    """

    name = "anthropic"

    def __init__(
        self,
        model: str = "claude-opus-4-7",
        *,
        api_key: str | None = None,
        system_prompt: str | None = None,
        client: Any | None = None,
        input_price_per_mtok: float | None = None,
        output_price_per_mtok: float | None = None,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self._input_price_per_mtok = input_price_per_mtok
        self._output_price_per_mtok = output_price_per_mtok

        if client is not None:
            self._client = client
            return

        try:
            import anthropic
        except ImportError as e:
            raise ProviderError(
                "AnthropicProvider requires the 'anthropic' package. "
                "Install with: pip install italiabench[anthropic]"
            ) from e

        self._client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def ask(self, prompt: str, *, max_tokens: int = 1024) -> ProviderResponse:
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            # Wrapping the system prompt in a list with cache_control is a no-op
            # for short prompts but lets longer prompts cache automatically.
            "system": [
                {
                    "type": "text",
                    "text": self.system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [{"role": "user", "content": prompt}],
        }

        t0 = time.monotonic()
        try:
            message = self._client.messages.create(**request_kwargs)
        except Exception as e:
            raise ProviderError(f"Anthropic API call failed: {e}") from e
        latency_ms = (time.monotonic() - t0) * 1000.0

        answer = "".join(
            block.text for block in message.content if getattr(block, "type", None) == "text"
        ).strip()

        usage = message.usage
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        cache_creation = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
        cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)

        cost_usd = estimate_cost_usd(
            self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_tokens=cache_creation,
            cache_read_tokens=cache_read,
            input_price_per_mtok=self._input_price_per_mtok,
            output_price_per_mtok=self._output_price_per_mtok,
        )

        return ProviderResponse(
            answer=answer,
            model=getattr(message, "model", self.model),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_tokens=cache_creation,
            cache_read_tokens=cache_read,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            stop_reason=getattr(message, "stop_reason", None),
        )
