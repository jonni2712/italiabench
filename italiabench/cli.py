"""Minimal CLI for ItaliaBench.

Single command for v0.1.0:

    italiabench run --model claude-opus-4-7 [--category fisco] [--limit 5]

Loads ``data/questions/``, queries the model once per question, scores the
answer with the deterministic engine, prints a Rich table, and lists the
detail of every failure. Provider is hard-coded to Anthropic for now —
adding OpenAI/Google/Ollama is task #4 follow-up work.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from italiabench.providers.base import ProviderError
from italiabench.scoring import score_answer
from italiabench.validator import DatasetError, validate_dataset

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()

# Module-level so it's not re-evaluated on every import (and so ruff B008 is happy
# about not having a function call inside an Option default).
_DEFAULT_DATASET_DIR = Path("data/questions")


@app.callback()
def _entrypoint() -> None:
    """ItaliaBench: open benchmark for LLM hallucinations on Italian knowledge."""
    # Empty callback — its only purpose is to force Typer to keep the
    # subcommand pattern even when there is just one command (otherwise
    # Typer inlines `run` and `italiabench run --model X` becomes
    # `italiabench --model X`, which is confusing and breaks the future
    # `italiabench compare` / `italiabench report` workflow).


def _build_provider(provider: str, model: str):
    if provider != "anthropic":
        raise typer.BadParameter(
            f"Provider {provider!r} is not implemented yet (only 'anthropic').",
            param_hint="--provider",
        )
    # Imported lazily so `italiabench --help` doesn't require `anthropic`.
    from italiabench.providers.anthropic import AnthropicProvider

    return AnthropicProvider(model=model)


@app.command()
def run(
    model: str = typer.Option(
        "claude-opus-4-7",
        "--model",
        "-m",
        help="Model ID. Default: claude-opus-4-7.",
    ),
    category: str | None = typer.Option(
        None,
        "--category",
        "-c",
        help="Restrict to one category (diritto, fisco, geografia_pa, storia_cultura, procedure).",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        "-n",
        help="Stop after N questions. Useful for quick smoke tests.",
    ),
    dataset: Path = typer.Option(
        _DEFAULT_DATASET_DIR,
        "--dataset",
        "-d",
        help="Path to the YAML dataset directory.",
    ),
    provider: str = typer.Option(
        "anthropic",
        "--provider",
        help="Provider name. Only 'anthropic' is wired in v0.1.0.",
    ),
    max_tokens: int = typer.Option(
        1024,
        "--max-tokens",
        help="Max output tokens per answer.",
    ),
) -> None:
    """Run a benchmark against a model and print a scoreboard."""
    try:
        questions = validate_dataset(dataset)
    except DatasetError as e:
        console.print(f"[red]Dataset validation failed:[/red] {e}")
        raise typer.Exit(1) from None

    questions = [q for q in questions if not q.canary]
    if category:
        questions = [q for q in questions if q.category.value == category]
        if not questions:
            console.print(f"[red]No public questions in category {category!r}.[/red]")
            raise typer.Exit(1)

    if limit:
        questions = questions[:limit]

    if not questions:
        console.print("[red]No questions to run.[/red]")
        raise typer.Exit(1)

    provider_obj = _build_provider(provider, model)

    console.print(
        f"[bold]Running ItaliaBench[/bold] on [cyan]{model}[/cyan] "
        f"({len(questions)} question{'s' if len(questions) != 1 else ''})"
    )

    results = []
    total_cost = 0.0
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        for q in questions:
            task_id = progress.add_task(f"[{q.category.value}] {q.id}", total=None)
            try:
                response = provider_obj.ask(q.question, max_tokens=max_tokens)
            except ProviderError as e:
                console.print(f"[red]Provider error on {q.id}:[/red] {e}")
                raise typer.Exit(2) from None
            score = score_answer(q, response.answer)
            results.append((q, response, score))
            if response.cost_usd:
                total_cost += response.cost_usd
            progress.remove_task(task_id)

    table = Table(title=f"ItaliaBench — {model}", show_lines=False)
    table.add_column("ID", style="dim")
    table.add_column("Cat", no_wrap=True)
    table.add_column("Diff", no_wrap=True, justify="center")
    table.add_column("Result", no_wrap=True, justify="center")
    table.add_column("Score", justify="right")
    table.add_column("Latency", justify="right")
    table.add_column("Tokens (in/out)", justify="right")

    for q, response, score in results:
        if score.passed:
            result_cell = "[green]PASS[/green]"
        elif score.forbidden_violations:
            result_cell = "[red]FAIL (forbidden)[/red]"
        else:
            result_cell = "[red]FAIL[/red]"
        latency_str = f"{response.latency_ms:.0f}ms" if response.latency_ms else "-"
        tokens_str = f"{response.input_tokens}/{response.output_tokens}"
        table.add_row(
            q.id,
            q.category.value,
            q.difficulty.value,
            result_cell,
            f"{score.score:.2f}",
            latency_str,
            tokens_str,
        )

    console.print(table)

    passed = sum(1 for _, _, s in results if s.passed)
    total = len(results)
    pct = (passed / total * 100) if total else 0
    cost_str = f"${total_cost:.4f}" if total_cost > 0 else "n/a"
    console.print(
        f"\n[bold]Score: {passed}/{total} ({pct:.0f}%)[/bold]   "
        f"[dim]Total cost: {cost_str}[/dim]"
    )

    failures = [(q, r, s) for q, r, s in results if not s.passed]
    if failures:
        console.print("\n[bold red]Failure detail[/bold red]")
        for q, response, score in failures:
            console.print(f"\n[bold]{q.id}[/bold] — {q.category.value} / {q.difficulty.value}")
            console.print(f"  [dim]Question:[/dim] {q.question.strip()}")
            console.print(f"  [dim]Expected:[/dim] {q.ground_truth}")
            console.print(f"  [dim]Got:[/dim] {response.answer}")
            if score.forbidden_violations:
                console.print(
                    f"  [red]Forbidden terms hit:[/red] {', '.join(score.forbidden_violations)}"
                )
            unmet = [m for m in score.mentions if not m.satisfied]
            for m in unmet:
                console.print(f"  [yellow]Missing constraint:[/yellow] {m.constraint}")


if __name__ == "__main__":
    app()
