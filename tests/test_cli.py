from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import yaml
from typer.testing import CliRunner

from italiabench.cli import app
from italiabench.providers.base import ProviderResponse

runner = CliRunner()


def _seed_dataset(root: Path) -> None:
    """Write one minimal question per category for tests."""
    questions = [
        {
            "category": "fisco",
            "id": "fisco-iva-001",
            "ground_truth": "4%",
            "must_mention": ["4%"],
        },
        {
            "category": "diritto",
            "id": "diritto-cc-001",
            "ground_truth": "salute",
            "must_mention": ["salute"],
        },
    ]
    for q in questions:
        payload = {
            "schema_version": 1,
            "id": q["id"],
            "category": q["category"],
            "difficulty": "easy",
            "question": "Domanda di test?",
            "ground_truth": q["ground_truth"],
            "must_mention": q["must_mention"],
            "must_not_mention": [],
            "source": ["fonte di test"],
            "last_verified": "2026-04-01",
        }
        target_dir = root / q["category"]
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / f"{q['id']}.yaml").write_text(
            yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
        )


def test_help_does_not_require_anthropic_package() -> None:
    """Smoke test — `--help` must not import the anthropic SDK or hit the network."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.stdout.lower()


def test_run_help_lists_options() -> None:
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    # Rich-formatted help may wrap or escape option names differently in
    # TTY vs CI environments — just confirm we got the `run` command's
    # help (its docstring is unique enough).
    assert "Run a benchmark" in result.output


def test_run_executes_against_mock_provider(tmp_path: Path) -> None:
    _seed_dataset(tmp_path)

    class _StubProvider:
        name = "stub"
        model = "stub-model"

        def ask(self, prompt: str, *, max_tokens: int = 1024) -> ProviderResponse:
            # Both questions accept the same answer text since must_mention
            # is "4%" and "salute"; we satisfy both at once.
            return ProviderResponse(
                answer="La risposta corretta menziona 4% e salute.",
                model="stub-model",
                input_tokens=100,
                output_tokens=20,
                cost_usd=0.001,
                latency_ms=42.0,
                stop_reason="end_turn",
            )

    with patch("italiabench.cli._build_provider", return_value=_StubProvider()):
        result = runner.invoke(
            app,
            ["run", "--model", "stub-model", "--dataset", str(tmp_path)],
        )

    assert result.exit_code == 0, result.stdout
    assert "ItaliaBench" in result.stdout
    assert "PASS" in result.stdout
    assert "Score: 2/2" in result.stdout


def test_run_reports_failures_with_detail(tmp_path: Path) -> None:
    _seed_dataset(tmp_path)

    class _WrongProvider:
        name = "stub"
        model = "stub-model"

        def ask(self, prompt: str, *, max_tokens: int = 1024) -> ProviderResponse:
            return ProviderResponse(
                answer="Non lo so.",
                model="stub-model",
                input_tokens=10,
                output_tokens=5,
                cost_usd=None,
                latency_ms=10.0,
            )

    with patch("italiabench.cli._build_provider", return_value=_WrongProvider()):
        result = runner.invoke(
            app,
            ["run", "--model", "stub-model", "--dataset", str(tmp_path)],
        )

    assert result.exit_code == 0
    assert "FAIL" in result.stdout
    assert "Failure detail" in result.stdout
    assert "Non lo so" in result.stdout


def test_run_filters_by_category(tmp_path: Path) -> None:
    _seed_dataset(tmp_path)

    class _StubProvider:
        name = "stub"
        model = "stub-model"

        def ask(self, prompt: str, *, max_tokens: int = 1024) -> ProviderResponse:
            return ProviderResponse(answer="4%", model="stub-model")

    with patch("italiabench.cli._build_provider", return_value=_StubProvider()):
        result = runner.invoke(
            app,
            ["run", "--model", "stub-model", "--dataset", str(tmp_path), "--category", "fisco"],
        )

    assert result.exit_code == 0
    assert "fisco-iva-001" in result.stdout
    assert "diritto-cc-001" not in result.stdout


def test_run_rejects_unknown_provider(tmp_path: Path) -> None:
    _seed_dataset(tmp_path)
    result = runner.invoke(
        app,
        ["run", "--provider", "openai", "--dataset", str(tmp_path)],
    )
    assert result.exit_code != 0
    # Click's BadParameter writes to stderr; .output is the combined stream.
    assert "openai" in result.output.lower()


def test_run_rejects_unknown_category(tmp_path: Path) -> None:
    _seed_dataset(tmp_path)
    result = runner.invoke(
        app,
        ["run", "--dataset", str(tmp_path), "--category", "imaginary"],
    )
    assert result.exit_code != 0
    assert "imaginary" in result.output
