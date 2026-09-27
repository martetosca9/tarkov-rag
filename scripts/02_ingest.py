#!/usr/bin/env python3
"""
Ingestion script for Escape from Tarkov quests.
Fetches wikitext from the Escape from Tarkov MediaWiki API,
parses quest data into structured JSON, and saves to disk.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
from pathlib import Path
import sys
import time
from typing import List, Tuple

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import Quest
from src.parser import QuestParser
from src.scraper import WikiScraper

console = Console()
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest and parse Escape from Tarkov quests.")
    parser.add_argument(
        "--quest",
        type=str,
        default=None,
        help="Parse a single quest by exact title (e.g. --quest 'Debut').",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Limit parsing to the first N quests (e.g. --sample 10).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore local raw wikitext cache and re-fetch from the API.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of concurrent worker threads for API requests (default: 5).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/quests.json",
        help="Destination path for the parsed structured JSON file.",
    )
    return parser.parse_args()


def process_single_quest(
    scraper: WikiScraper, parser: QuestParser, title: str, use_cache: bool
) -> Tuple[str, Quest | None]:
    """Fetch and parse one quest."""
    wikitext = scraper.fetch_quest_wikitext(title, use_cache=use_cache)
    if not wikitext:
        return title, None
    quest = parser.parse(title, wikitext)
    return title, quest


def main():
    args = parse_args()
    scraper = WikiScraper(raw_cache_dir="data/raw")
    parser = QuestParser()

    console.rule("[bold cyan]Tarkov RAG - Quest Ingestion Pipeline[/bold cyan]")

    # 1. Determine list of quest titles to process
    if args.quest:
        titles = [args.quest]
        console.print(f"[green]Targeting single quest:[/green] [bold]{args.quest}[/bold]")
    else:
        with console.status("[bold green]Fetching quest catalog from MediaWiki Category:Quests..."):
            titles = scraper.list_quest_titles()

        if args.sample:
            titles = titles[: args.sample]
            console.print(f"[yellow]Sample mode enabled:[/yellow] processing first {len(titles)} quests.")
        else:
            console.print(f"[green]Found {len(titles)} total quests to process.[/green]")

    # 2. Fetch and parse quests with progress tracking
    parsed_quests: List[Quest] = []
    failed_titles: List[str] = []
    use_cache = not args.no_cache

    progress_columns = [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ]

    with Progress(*progress_columns, console=console) as progress:
        task = progress.add_task("[cyan]Ingesting quests...", total=len(titles))

        if args.workers > 1 and len(titles) > 1:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                future_to_title = {
                    executor.submit(process_single_quest, scraper, parser, t, use_cache): t
                    for t in titles
                }
                for future in as_completed(future_to_title):
                    title, quest = future.result()
                    if quest:
                        parsed_quests.append(quest)
                    else:
                        failed_titles.append(title)
                    progress.update(task, description=f"[cyan]Processed:[/] [dim]{title[:25]}[/dim]")
                    progress.advance(task)
        else:
            for title in titles:
                progress.update(task, description=f"[cyan]Processing:[/] [dim]{title[:25]}[/dim]")
                _, quest = process_single_quest(scraper, parser, title, use_cache)
                if quest:
                    parsed_quests.append(quest)
                else:
                    failed_titles.append(title)
                progress.advance(task)

    # Sort parsed quests alphabetically by name
    parsed_quests.sort(key=lambda q: q.name)

    # 3. Save processed quests to destination file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    quests_data = [q.model_dump() for q in parsed_quests]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(quests_data, f, ensure_ascii=False, indent=2)

    # 4. Display summary stats
    file_size_kb = output_path.stat().st_size / 1024

    console.print()
    table = Table(title="[bold green]Ingestion Summary[/bold green]", show_header=True)
    table.add_column("Metric", style="bold white")
    table.add_column("Value", style="cyan")

    table.add_row("Total targeted", str(len(titles)))
    table.add_row("Successfully parsed", str(len(parsed_quests)))
    table.add_row("Failed or skipped", str(len(failed_titles)))
    table.add_row("Output file", str(output_path))
    table.add_row("Output file size", f"{file_size_kb:.1f} KB")

    console.print(table)

    # Trader distribution summary
    trader_counts = {}
    for q in parsed_quests:
        t = q.trader or "Unknown"
        trader_counts[t] = trader_counts.get(t, 0) + 1

    if trader_counts:
        trader_table = Table(title="[bold blue]Quests by Trader[/bold blue]", show_header=True)
        trader_table.add_column("Trader", style="bold white")
        trader_table.add_column("Quest Count", justify="right", style="green")

        for trader, count in sorted(trader_counts.items(), key=lambda x: x[1], reverse=True):
            trader_table.add_row(trader, str(count))

        console.print(trader_table)

    if failed_titles:
        console.print(f"[yellow]Skipped/Failed titles ({len(failed_titles)}):[/yellow] {failed_titles[:10]}")


if __name__ == "__main__":
    main()
