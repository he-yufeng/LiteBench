from __future__ import annotations

from rich.console import Console
from rich.table import Table

from litebench.core.models import RunSummary, SampleResult

console = Console()


def print_summary(summary: RunSummary) -> None:
    table = Table(title=f"{summary.task} · {summary.model}", show_header=False, box=None)
    table.add_column(style="bold cyan")
    table.add_column()
    table.add_row("Samples", str(summary.n_samples))
    table.add_row(
        "Accuracy",
        f"[bold green]{summary.accuracy * 100:.1f}%[/]  "
        f"({summary.n_correct}/{summary.n_samples})",
    )
    table.add_row("Mean latency", f"{summary.mean_latency_ms:.0f} ms")
    table.add_row(
        "Tokens",
        f"prompt={summary.total_prompt_tokens:,}  "
        f"completion={summary.total_completion_tokens:,}",
    )
    duration = (summary.finished_at - summary.started_at).total_seconds()
    table.add_row("Duration", f"{duration:.1f}s")
    table.add_row("Run ID", summary.run_id[:8])
    console.print()
    console.print(table)
    console.print()


def print_run_list(runs: list[RunSummary]) -> None:
    if not runs:
        console.print("[dim]No runs yet. Try: litebench run gsm8k --samples 10[/]")
        return
    table = Table(title="Recent runs")
    table.add_column("Run", style="dim")
    table.add_column("Task")
    table.add_column("Model")
    table.add_column("Samples", justify="right")
    table.add_column("Accuracy", justify="right")
    table.add_column("When", style="dim")
    for r in runs:
        table.add_row(
            r.run_id[:8],
            r.task,
            r.model,
            str(r.n_samples),
            f"{r.accuracy * 100:.1f}%",
            r.finished_at.strftime("%Y-%m-%d %H:%M"),
        )
    console.print(table)


def print_compare(runs: list[RunSummary]) -> None:
    if not runs:
        console.print("[dim]Nothing to compare.[/]")
        return
    table = Table(title=f"Comparing {len(runs)} runs")
    table.add_column("Model")
    table.add_column("Task")
    table.add_column("N", justify="right")
    table.add_column("Accuracy", justify="right", style="bold")
    table.add_column("Mean latency", justify="right")
    table.add_column("Tokens (p/c)", justify="right")
    for r in runs:
        table.add_row(
            r.model,
            r.task,
            str(r.n_samples),
            f"{r.accuracy * 100:.1f}%",
            f"{r.mean_latency_ms:.0f}ms",
            f"{r.total_prompt_tokens:,} / {r.total_completion_tokens:,}",
        )
    console.print(table)


def print_task_list(tasks: list[tuple[str, str]]) -> None:
    table = Table(title="Available tasks")
    table.add_column("Name", style="cyan")
    table.add_column("Description")
    for name, desc in tasks:
        table.add_row(name, desc)
    console.print(table)


def print_sample_failures(results: list[SampleResult], limit: int = 5) -> None:
    failures = [r for r in results if not r.correct][:limit]
    if not failures:
        return
    console.print(f"[yellow]First {len(failures)} failing samples:[/]")
    for r in failures:
        console.print(f"  [dim]{r.sample_id}[/] target=[green]{r.target}[/]")
        snippet = (r.error or r.prediction[:200]).replace("\n", " ")
        console.print(f"    [dim]{snippet}[/]")
