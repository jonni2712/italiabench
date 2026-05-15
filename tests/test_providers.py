from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from italiabench.providers import (
    PRICING_USD_PER_MTOK,
    ProviderError,
    ProviderResponse,
    estimate_cost_usd,
)
from italiabench.providers.anthropic import DEFAULT_SYSTEM_PROMPT, AnthropicProvider

# ---------------------------------------------------------------------------
# Pricing table + cost estimator
# ---------------------------------------------------------------------------


def test_known_models_have_pricing() -> None:
    assert "claude-opus-4-7" in PRICING_USD_PER_MTOK
    assert "claude-haiku-4-5" in PRICING_USD_PER_MTOK
    for in_price, out_price in PRICING_USD_PER_MTOK.values():
        assert in_price > 0
        assert out_price > 0
        # Output should always cost at least as much as input.
        assert out_price >= in_price


def test_estimate_cost_uses_default_pricing_for_known_model() -> None:
    cost = estimate_cost_usd(
        "claude-opus-4-7",
        input_tokens=1_000_000,
        output_tokens=500_000,
    )
    # 1M input * $5 + 0.5M output * $25 = 5 + 12.5 = 17.5
    assert cost == pytest.approx(17.5)


def test_estimate_cost_returns_none_for_unknown_model_without_overrides() -> None:
    assert estimate_cost_usd("imaginary-model-99", input_tokens=1000, output_tokens=100) is None


def test_estimate_cost_accepts_explicit_overrides() -> None:
    cost = estimate_cost_usd(
        "imaginary-model-99",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        input_price_per_mtok=2.0,
        output_price_per_mtok=10.0,
    )
    assert cost == pytest.approx(12.0)


def test_estimate_cost_applies_cache_multipliers() -> None:
    # Cache reads = 0.1x input, cache writes = 1.25x input.
    cost = estimate_cost_usd(
        "claude-opus-4-7",
        input_tokens=0,
        output_tokens=0,
        cache_creation_tokens=1_000_000,  # 1M * $5 * 1.25 = 6.25
        cache_read_tokens=1_000_000,  # 1M * $5 * 0.1 = 0.5
    )
    assert cost == pytest.approx(6.75)


# ---------------------------------------------------------------------------
# AnthropicProvider — request shape + response parsing
# ---------------------------------------------------------------------------


def _fake_message(
    *,
    text: str = "L'aliquota IVA ridotta sui libri è del 4%.",
    model: str = "claude-opus-4-7",
    input_tokens: int = 200,
    output_tokens: int = 30,
    cache_creation: int = 0,
    cache_read: int = 0,
    stop_reason: str = "end_turn",
) -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        model=model,
        stop_reason=stop_reason,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=cache_creation,
            cache_read_input_tokens=cache_read,
        ),
    )


def _provider_with_mock_client(**message_kwargs: Any) -> tuple[AnthropicProvider, MagicMock]:
    fake_client = MagicMock()
    fake_client.messages.create.return_value = _fake_message(**message_kwargs)
    provider = AnthropicProvider(model="claude-opus-4-7", client=fake_client)
    return provider, fake_client


def test_ask_returns_provider_response_with_parsed_text() -> None:
    provider, _ = _provider_with_mock_client(text="Il 4%.")
    response = provider.ask("Qual è l'aliquota IVA ridotta sui libri?")
    assert isinstance(response, ProviderResponse)
    assert response.answer == "Il 4%."
    assert response.model == "claude-opus-4-7"
    assert response.stop_reason == "end_turn"


def test_ask_does_not_send_sampling_parameters() -> None:
    """Opus 4.7 returns 400 if temperature/top_p/top_k are present.

    Even on older models we don't want them — benchmarks should be reproducible
    without sampler tuning. This test pins that promise.
    """
    provider, fake_client = _provider_with_mock_client()
    provider.ask("Domanda di prova")

    kwargs = fake_client.messages.create.call_args.kwargs
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs
    assert "top_k" not in kwargs


def test_ask_includes_cache_control_on_system_prompt() -> None:
    provider, fake_client = _provider_with_mock_client()
    provider.ask("Domanda di prova")

    kwargs = fake_client.messages.create.call_args.kwargs
    system = kwargs["system"]
    assert isinstance(system, list)
    assert system[0]["text"] == DEFAULT_SYSTEM_PROMPT
    assert system[0]["cache_control"] == {"type": "ephemeral"}


def test_ask_passes_max_tokens_through() -> None:
    provider, fake_client = _provider_with_mock_client()
    provider.ask("Domanda di prova", max_tokens=42)

    assert fake_client.messages.create.call_args.kwargs["max_tokens"] == 42


def test_ask_uses_custom_system_prompt_when_provided() -> None:
    fake_client = MagicMock()
    fake_client.messages.create.return_value = _fake_message()
    provider = AnthropicProvider(
        model="claude-opus-4-7",
        client=fake_client,
        system_prompt="Custom prompt.",
    )
    provider.ask("Domanda di prova")

    assert fake_client.messages.create.call_args.kwargs["system"][0]["text"] == "Custom prompt."


def test_ask_records_token_counts_and_latency() -> None:
    provider, _ = _provider_with_mock_client(input_tokens=512, output_tokens=128)
    response = provider.ask("Domanda di prova")
    assert response.input_tokens == 512
    assert response.output_tokens == 128
    assert response.latency_ms is not None and response.latency_ms >= 0


def test_ask_computes_cost_with_default_pricing() -> None:
    provider, _ = _provider_with_mock_client(input_tokens=1_000_000, output_tokens=500_000)
    response = provider.ask("Domanda di prova")
    # 1M input * $5 + 0.5M output * $25 = 17.5
    assert response.cost_usd == pytest.approx(17.5)


def test_ask_propagates_cache_token_counts() -> None:
    provider, _ = _provider_with_mock_client(cache_creation=10_000, cache_read=20_000)
    response = provider.ask("Domanda di prova")
    assert response.cache_creation_tokens == 10_000
    assert response.cache_read_tokens == 20_000


def test_ask_handles_multiple_text_blocks() -> None:
    fake_client = MagicMock()
    fake_client.messages.create.return_value = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="parte 1 "),
            SimpleNamespace(type="thinking", thinking="ignored"),
            SimpleNamespace(type="text", text="parte 2"),
        ],
        model="claude-opus-4-7",
        stop_reason="end_turn",
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
    )
    provider = AnthropicProvider(model="claude-opus-4-7", client=fake_client)
    response = provider.ask("Domanda di prova")
    assert response.answer == "parte 1 parte 2"


def test_ask_wraps_sdk_errors_in_provider_error() -> None:
    fake_client = MagicMock()
    fake_client.messages.create.side_effect = RuntimeError("boom")
    provider = AnthropicProvider(model="claude-opus-4-7", client=fake_client)
    with pytest.raises(ProviderError, match="Anthropic API call failed"):
        provider.ask("Domanda di prova")
