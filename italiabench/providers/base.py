"""Provider interface and shared cost-estimation helpers.

A provider is anything that, given a question prompt, returns a
:class:`ProviderResponse` carrying the answer text plus enough metadata
(token counts, latency, cost) to populate a benchmark report.

Cost estimation is intentionally pluggable: the table below carries the
per-million-token prices we know about today; callers may override
``input_price_per_mtok`` and ``output_price_per_mtok`` per provider
instance to track price changes without waiting for a release.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

PRICING_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    # Anthropic (per the claude-api skill, cached 2026-04-29):
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}
"""``model_id -> (input_usd_per_mtok, output_usd_per_mtok)`` lookup.

Values are best-effort and should be re-checked before publishing a
report. Cache reads are billed at ~0.1x input, cache writes at ~1.25x
input -- :func:`estimate_cost_usd` handles those multipliers.
"""


@dataclass(frozen=True)
class ProviderResponse:
    """Outcome of one provider call."""

    answer: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: float | None = None
    latency_ms: float | None = None
    stop_reason: str | None = None


class ProviderError(RuntimeError):
    """Raised when a provider cannot produce an answer (auth, rate limit, etc.)."""


class Provider(Protocol):
    """Minimal contract every provider implements."""

    name: str
    model: str

    def ask(self, prompt: str, *, max_tokens: int = 1024) -> ProviderResponse: ...


def estimate_cost_usd(
    model: str,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
    input_price_per_mtok: float | None = None,
    output_price_per_mtok: float | None = None,
) -> float | None:
    """Return the dollar cost for a single call, or ``None`` if the model is unknown
    and no explicit prices were passed in.

    Accounts for prompt-cache pricing: cache writes cost 1.25x the base input
    rate, cache reads cost 0.1x.
    """
    if input_price_per_mtok is None or output_price_per_mtok is None:
        defaults = PRICING_USD_PER_MTOK.get(model)
        if defaults is None:
            return None
        default_in, default_out = defaults
        input_price_per_mtok = default_in if input_price_per_mtok is None else input_price_per_mtok
        output_price_per_mtok = (
            default_out if output_price_per_mtok is None else output_price_per_mtok
        )

    return (
        input_tokens / 1_000_000 * input_price_per_mtok
        + cache_creation_tokens / 1_000_000 * input_price_per_mtok * 1.25
        + cache_read_tokens / 1_000_000 * input_price_per_mtok * 0.1
        + output_tokens / 1_000_000 * output_price_per_mtok
    )
