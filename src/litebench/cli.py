from __future__ import annotations

import asyncio
import json as jsonlib
import sys
from pathlib import Path

import click
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn

from litebench import __version__
from litebench.config import DB_PATH, ensure_dirs, resolve_model
from litebench.core.models import RunSummary, SampleResult
from litebench.core.runner import Runner
from litebench.core.storage import Storage
from litebench.llm.client import LLMClient
from litebench.output.console import (
    console,
    print_compare,
    print_run_list,
    print_sample_failures,
    print_summary,
    print_task_list,
)
from litebench.tasks import get_task, list_tasks
from litebench.tasks.arc import ARCTask
from litebench.tasks.custom import CustomTask
from litebench.tasks.mmlu import MMLUTask


@click.group()
@click.version_option(__version__, prog_name="litebench")
def main() -> None:
    """LiteBench — a pip-installable benchmark runner for LLMs and agents."""


@main.command("list")
def list_cmd() -> None:
    """List built-in tasks."""
    tasks = []
    for name in list_tasks():
        task = get_task(name)
        tasks.append((name, task.description))
    print_task_list(tasks)


@main.command()
@click.argument("task_name")
@click.option("--model", "-m", required=True, help="Model id (e.g. gpt-5, deepseek/deepseek-chat, or shortcut: gpt-5/opus/kimi).")
@click.option("--samples", "-n", default=20, type=int, help="Number of samples to run (default: 20).")
@click.option("--concurrency", "-c", default=8, type=int, help="Parallel requests (default: 8).")
@click.option("--temperature", "-t", default=0.0, type=float, help="Sampling temperature (default: 0.0).")
@click.option("--max-tokens", default=1024, type=int, help="Max completion tokens (default: 1024).")
@click.option("--subject", default=None, help="MMLU only: restrict to one subject (e.g. 'computer_security').")
@click.option("--arc-easy", is_flag=True, help="ARC only: use ARC-Easy instead of ARC-Challenge.")
@click.option("--split", default="test", help="Dataset split (default: test).")
@click.option("--json-out", type=click.Path(path_type=Path), default=None, help="Write per-sample JSON to this path.")
@click.option("--no-save", is_flag=True, help="Don't persist the run to the local DB.")
def run(
    task_name: str,
    model: str,
    samples: int,
    concurrency: int,
    temperature: float,
    max_tokens: int,
    subject: str | None,
    arc_easy: bool,
    split: str,
    json_out: Path | None,
    no_save: bool,
) -> None:
    """Run a benchmark. Example: litebench run gsm8k -m deepseek/deepseek-chat -n 50

    TASK_NAME can be a built-in task (humaneval/gsm8k/mmlu/truthfulqa/math/arc)
    or a path to a YAML file describing a custom task.
    """
    ensure_dirs()
    resolved = resolve_model(model)

    task_path = Path(task_name)
    if task_path.exists() and task_path.suffix.lower() in {".yaml", ".yml"}:
        task = CustomTask(task_path)
    elif task_name.lower() == "mmlu" and subject:
        task = MMLUTask(subject=subject)
    elif task_name.lower() == "arc" and arc_easy:
        task = ARCTask(config="ARC-Easy")
    else:
        try:
            task = get_task(task_name)
        except ValueError as e:
            console.print(f"[red]{e}[/]")
            sys.exit(1)

    console.print(f"Loading [cyan]{task.name}[/] samples...")
    sample_list = list(task.load_samples(n=samples, split=split))
    if not sample_list:
        console.print("[red]No samples loaded.[/]")
        sys.exit(1)
    console.print(f"Loaded [bold]{len(sample_list)}[/] samples. Running on [cyan]{resolved}[/]...")

    client = LLMClient(
        model=resolved,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    async def _go():
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("acc={task.fields[acc]}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            pid = progress.add_task("Running", total=len(sample_list), acc="—")
            correct = 0

            def on_progress(done: int, total: int, result):
                nonlocal correct
                if result.correct:
                    correct += 1
                progress.update(pid, completed=done, acc=f"{correct / done * 100:.1f}%")

            runner = Runner(task=task, client=client, concurrency=concurrency, on_progress=on_progress)
            summary, results = await runner.run(sample_list)

        if not no_save:
            storage = Storage(DB_PATH)
            await storage.init()
            await storage.save_run(summary, results)

        print_summary(summary)
        print_sample_failures(results)

        if json_out:
            payload = {
                "summary": summary.model_dump(mode="json"),
                "results": [r.model_dump(mode="json") for r in results],
            }
            json_out.parent.mkdir(parents=True, exist_ok=True)
            json_out.write_text(
                jsonlib.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            console.print(f"[dim]Per-sample results → {json_out}[/]")

    asyncio.run(_go())


@main.command()
@click.option("--limit", "-l", default=20, type=int)
@click.option("--task", "-t", default=None)
def runs(limit: int, task: str | None) -> None:
    """Show recent runs from the local history."""
    ensure_dirs()

    async def _go():
        storage = Storage(DB_PATH)
        await storage.init()
        run_list = await storage.list_runs(limit=limit, task=task)
        print_run_list(run_list)

    asyncio.run(_go())


@main.command()
@click.option("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1).")
@click.option("--port", default=8600, type=int, help="Port (default: 8600).")
def serve(host: str, port: int) -> None:
    """Start the local web dashboard over the SQLite run history."""
    try:
        import uvicorn
    except ImportError:
        console.print("[red]Web UI requires the [web] extras. Install with:[/] [cyan]pip install 'litebench[web]'[/]")
        sys.exit(1)

    from litebench.server.app import create_app

    console.print(f"[cyan]LiteBench[/] dashboard on [bold]http://{host}:{port}[/]  (Ctrl+C to stop)")
    uvicorn.run(create_app(), host=host, port=port, log_level="warning")


@main.command()
@click.argument("run_ids", nargs=-1, required=True)
def compare(run_ids: tuple[str, ...]) -> None:
    """Compare N past runs side-by-side (pass the short 8-char run prefixes)."""
    ensure_dirs()

    async def _go():
        storage = Storage(DB_PATH)
        await storage.init()
        all_runs = await storage.list_runs(limit=500)
        wanted = []
        for rid in run_ids:
            match = next((r for r in all_runs if r.run_id.startswith(rid)), None)
            if not match:
                console.print(f"[red]Run not found: {rid}[/]")
                continue
            wanted.append(match)
        print_compare(wanted)

    asyncio.run(_go())


@main.command("export")
@click.argument("run_id")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["json", "jsonl"]),
    default="json",
    show_default=True,
    help="Export format.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output path. Defaults to litebench-<run>.json/jsonl.",
)
def export_cmd(run_id: str, fmt: str, output: Path | None) -> None:
    """Export one saved run, including per-sample results."""
    ensure_dirs()

    async def _go():
        storage = Storage(DB_PATH)
        await storage.init()
        runs = await storage.list_runs(limit=500)
        matches = [r for r in runs if r.run_id.startswith(run_id)]
        if not matches:
            console.print(f"[red]Run not found:[/] {run_id}")
            sys.exit(1)
        if len(matches) > 1:
            console.print(f"[red]Run prefix is ambiguous:[/] {run_id}")
            for r in matches[:10]:
                console.print(f"  {r.run_id[:12]}  {r.task} · {r.model}")
            sys.exit(1)

        summary = matches[0]
        samples = await storage.list_samples(summary.run_id)
        out_path = output or Path(f"litebench-{summary.run_id[:8]}.{fmt}")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if fmt == "json":
            out_path.write_text(
                jsonlib.dumps(_export_payload(summary, samples), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        else:
            lines = [
                jsonlib.dumps(
                    {
                        "run_id": summary.run_id,
                        "task": summary.task,
                        "model": summary.model,
                        "sample": sample.model_dump(mode="json"),
                    },
                    ensure_ascii=False,
                )
                for sample in samples
            ]
            out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        console.print(f"[green]Exported {len(samples)} samples:[/] {out_path}")

    asyncio.run(_go())


def _export_payload(summary: RunSummary, samples: list[SampleResult]) -> dict:
    return {
        "summary": summary.model_dump(mode="json"),
        "results": [sample.model_dump(mode="json") for sample in samples],
    }


if __name__ == "__main__":
    main()
