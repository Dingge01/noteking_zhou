"""NoteKing CLI: command-line interface for video-to-notes conversion."""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()


@click.group(invoke_without_command=True)
@click.pass_context
@click.version_option(version="0.1.0", prog_name="noteking")
def main(ctx):
    """NoteKing - The ultimate video/blog to learning notes tool.

    Supports 30+ platforms: Bilibili, YouTube, Douyin, Xiaohongshu, etc.
    13 output templates: detailed notes, mind maps, flashcards, quizzes, and more.
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@main.command()
@click.argument("url")
@click.option("-t", "--template", default="detailed",
              help="Output template: brief/detailed/mindmap/flashcard/quiz/timeline/exam/tutorial/news/podcast/xhs_note/latex_pdf/custom")
@click.option("-o", "--output", default=None, help="Output directory")
@click.option("--api-key", default=None, help="LLM API key")
@click.option("--base-url", default=None, help="LLM API base URL")
@click.option("--model", default=None, help="LLM model name")
@click.option("--proxy", default=None, help="Proxy URL (e.g., socks5://127.0.0.1:7890)")
@click.option("--custom-prompt", default="", help="Custom prompt for 'custom' template")
@click.option("--no-cache", is_flag=True, help="Disable result caching")
def run(url, template, output, api_key, base_url, model, proxy, custom_prompt, no_cache):
    """Process a video/blog URL and generate notes.

    Examples:

      noteking run "https://www.bilibili.com/video/BV1xx411c79H" -t detailed

      noteking run "https://youtu.be/dQw4w9WgXcQ" -t mindmap

      noteking run "./lecture.mp4" -t exam
    """
    from core.config import AppConfig
    from core import summarize

    config = AppConfig.load()

    if api_key:
        config.llm.api_key = api_key
    if base_url:
        config.llm.base_url = base_url
    if model:
        config.llm.model = model
    if proxy:
        config.proxy.enabled = True
        if proxy.startswith("socks"):
            config.proxy.socks5 = proxy
        else:
            config.proxy.https = proxy
            config.proxy.http = proxy

    console.print(Panel(
        f"[bold cyan]NoteKing[/bold cyan] - Video to Notes\n"
        f"URL: {url}\n"
        f"Template: {template}",
        title="Processing",
    ))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching video info and subtitles...", total=None)
        try:
            result = summarize(
                url=url,
                template=template,
                config=config,
                custom_prompt=custom_prompt,
                use_cache=not no_cache,
                output_dir=output,
            )
            progress.update(task, description="Done!")
        except Exception as e:
            progress.update(task, description=f"[red]Error: {e}[/red]")
            console.print(f"\n[red]Error:[/red] {e}")
            sys.exit(1)

    console.print(f"\n[green]Success![/green]")
    console.print(f"Title: {result.get('title', 'N/A')}")
    console.print(f"Platform: {result.get('platform', 'N/A')}")
    console.print(f"Source: {result.get('source', 'N/A')}")
    console.print(f"Output: {result.get('output_file', 'N/A')}")

    if result.get("failed"):
        console.print(f"\n[yellow]Failed entries:[/yellow]")
        for f in result["failed"]:
            console.print(f"  - {f}")


@main.command()
@click.argument("url")
@click.option("-o", "--output", default=None, help="Output directory")
@click.option("--proxy", default=None, help="Proxy URL")
def transcript(url, output, proxy):
    """Extract only the transcript/subtitles from a video."""
    from core.config import AppConfig
    from core import get_transcript

    config = AppConfig.load()
    if proxy:
        config.proxy.enabled = True
        config.proxy.https = proxy

    text = get_transcript(url, config)
    if output:
        Path(output).mkdir(parents=True, exist_ok=True)
        out_file = Path(output) / "transcript.txt"
        out_file.write_text(text, encoding="utf-8")
        console.print(f"[green]Saved to {out_file}[/green]")
    else:
        console.print(text)


@main.command()
def templates():
    """List all available output templates."""
    from core.templates import TEMPLATE_LIST

    table = Table(title="Available Templates")
    table.add_column("Name", style="cyan")
    table.add_column("Display Name", style="green")
    table.add_column("Description")

    for t in TEMPLATE_LIST:
        table.add_row(t["name"], t["display_name"], t["description"])

    console.print(table)


@main.command()
@click.option("--api-key", prompt="LLM API Key", help="Your LLM API key")
@click.option("--base-url", default="", help="LLM API base URL (for DeepSeek, Ollama, etc.)")
@click.option("--model", default="gpt-4o-mini", help="LLM model name")
@click.option("--proxy", default="", help="Proxy URL for YouTube access")
def setup(api_key, base_url, model, proxy):
    """Interactive setup wizard for first-time configuration."""
    from core.config import AppConfig

    config = AppConfig.load()
    config.llm.api_key = api_key
    if base_url:
        config.llm.base_url = base_url
    config.llm.model = model
    if proxy:
        config.proxy.enabled = True
        if proxy.startswith("socks"):
            config.proxy.socks5 = proxy
        else:
            config.proxy.https = proxy
    config.save()

    console.print("[green]Configuration saved![/green]")
    console.print(f"Config file: {config.save.__func__}")


@main.command()
def cache_clear():
    """Clear all cached results."""
    from core.config import AppConfig
    from core.cache import Cache

    config = AppConfig.load()
    c = Cache(config)
    count = c.clear()
    console.print(f"[green]Cleared {count} cached entries.[/green]")


if __name__ == "__main__":
    main()
