"""
CLI entrypoint for the instruction-tuning dataset pipeline.

Usage:
    python main.py run --config configs/config.yaml
    python main.py run --config configs/config.yaml --override configs/config.local.yaml
    python main.py validate-config --config configs/config.yaml
    python main.py run-stage collect --config configs/config.yaml
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from src.config.loader import ConfigError, load_config
from src.pipeline import DatasetPipeline
from src.utils.logging_setup import configure_logging, get_logger

app = typer.Typer(add_completion=False, help="AI Agent Engineering & Data Science dataset pipeline.")
console = Console()
logger = get_logger(__name__)


@app.command()
def run(
    config: Path = typer.Option(Path("configs/config.yaml"), "--config", "-c", help="Path to YAML config."),
    override: Optional[Path] = typer.Option(None, "--override", "-o", help="Optional override YAML config."),
) -> None:
    """Run the full 12-stage pipeline end-to-end."""
    pipeline_config = _load_and_configure(config, override)
    pipeline = DatasetPipeline(pipeline_config)

    stats = pipeline.run_full_pipeline()

    console.print("\n[bold green]Pipeline finished successfully.[/bold green]")
    _print_summary_table(stats)


@app.command()
def run_stage(
    stage: str = typer.Argument(
        ..., help="One of: collect, clean, extract-code, chunk, score, filter, generate, validate, export, stats"
    ),
    config: Path = typer.Option(Path("configs/config.yaml"), "--config", "-c"),
    override: Optional[Path] = typer.Option(None, "--override", "-o"),
) -> None:
    """Run a single pipeline stage. Useful for debugging or resuming a run.

    Reads from / writes to the intermediate JSONL files under `data/`, so
    stages can be re-run independently once earlier stages have produced
    their outputs at least once.
    """
    pipeline_config = _load_and_configure(config, override)
    pipeline = DatasetPipeline(pipeline_config)
    dirs = pipeline_config.directories

    if stage == "collect":
        pipeline.run_collection()
    elif stage == "clean":
        docs = DatasetPipeline.load_documents(dirs.raw / "collected.jsonl")
        pipeline.run_cleaning(docs)
    elif stage == "extract-code":
        docs = DatasetPipeline.load_documents(dirs.clean / "cleaned.jsonl")
        extracted = pipeline.run_code_extraction(docs)
        pipeline._persist_documents(extracted, dirs.clean / "cleaned.jsonl")
    elif stage == "chunk":
        docs = DatasetPipeline.load_documents(dirs.clean / "cleaned.jsonl")
        pipeline.run_chunking(docs)
    elif stage == "score":
        docs = DatasetPipeline.load_documents(dirs.clean / "cleaned.jsonl")
        scored = pipeline.run_quality_scoring(docs)
        pipeline._persist_documents(scored, dirs.clean / "cleaned.jsonl")
    elif stage == "filter":
        docs = DatasetPipeline.load_documents(dirs.clean / "cleaned.jsonl")
        pipeline.run_filtering(docs)
    else:
        console.print(
            f"[bold red]Stage '{stage}' cannot be run in isolation via this shortcut; "
            f"use `run` for the full pipeline, or extend `run_stage` in main.py.[/bold red]"
        )
        raise typer.Exit(code=1)

    console.print(f"[bold green]Stage '{stage}' complete.[/bold green]")


@app.command()
def validate_config(config: Path = typer.Option(Path("configs/config.yaml"), "--config", "-c")) -> None:
    """Validate a config file without running the pipeline."""
    try:
        pipeline_config = load_config(config)
    except ConfigError as exc:
        console.print(f"[bold red]Configuration is invalid:[/bold red]\n{exc}")
        raise typer.Exit(code=1)

    console.print(f"[bold green]Configuration is valid.[/bold green] Project: {pipeline_config.project_name}")
    total_sources = (
        len(pipeline_config.sources.github_repos)
        + len(pipeline_config.sources.documentation)
        + len(pipeline_config.sources.huggingface_datasets)
        + len(pipeline_config.sources.blogs)
        + len(pipeline_config.sources.papers)
    )
    console.print(f"Configured sources: {total_sources}")


def _load_and_configure(config_path: Path, override_path: Optional[Path]):
    try:
        pipeline_config = load_config(config_path, override_path)
    except ConfigError as exc:
        console.print(f"[bold red]Failed to load configuration:[/bold red]\n{exc}")
        raise typer.Exit(code=1)

    configure_logging(pipeline_config.logging, pipeline_config.directories.logs)
    return pipeline_config


def _print_summary_table(stats: dict) -> None:
    table = Table(title="Dataset Pipeline Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Documents collected", str(stats.get("documents_collected", 0)))
    table.add_row("Documents after filtering", str(stats.get("documents_after_filtering", 0)))
    table.add_row("Duplicates removed", str(stats.get("duplicates_removed", 0)))
    table.add_row("Average quality score", str(stats.get("average_quality_score", 0)))
    table.add_row(
        "Instruction examples generated",
        str(stats.get("instruction_dataset", {}).get("total_examples", 0)),
    )
    console.print(table)


if __name__ == "__main__":
    app()
