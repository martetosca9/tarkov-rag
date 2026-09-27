#!/usr/bin/env python3
"""
CLI Query interface for the Tarkov RAG application.
Allows users to ask natural language questions about Escape from Tarkov quests.
"""

import argparse
from pathlib import Path
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag import TarkovRAG
from src.vectorstore import QuestVectorStore

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query Tarkov quests via natural language RAG.")
    parser.add_argument(
        "query",
        type=str,
        nargs="?",
        default=None,
        help="Optional single question to ask. If omitted, starts interactive chat mode.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of candidate quests to retrieve as context (default: 3).",
    )
    parser.add_argument(
        "--trader",
        type=str,
        default=None,
        help="Filter retrieval to quests given by a specific trader (e.g. Prapor, Ragman).",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Force offline mode (retrieve quest data and citations without calling external LLM).",
    )
    return parser.parse_args()


def display_result(result: dict):
    """Format and render the query response with rich visual elements."""
    console.print()
    answer_text = result["answer"]
    mode = result.get("mode", "offline")

    mode_badges = {
        "llm": "[bold green]LLM Answer (GPT-4o-mini)[/bold green]",
        "offline": "[bold yellow]Direct Vector Context (Offline Mode)[/bold yellow]",
        "offline_fallback": "[bold red]Offline Fallback[/bold red]",
    }
    title = mode_badges.get(mode, "[bold cyan]Response[/bold cyan]")

    console.print(Panel(Markdown(answer_text), title=title, border_style="cyan", padding=(1, 2)))

    # Sources summary
    sources = result.get("sources", [])
    if sources:
        console.print("[bold dim]Source Quests Cited:[/bold dim]")
        for s in sources:
            console.print(f"  • [bold]{s['name']}[/bold] ({s['trader']}) - [link={s['url']}]{s['url']}[/link]")
    console.print()


def run_interactive(rag: TarkovRAG, top_k: int, trader: str | None, force_offline: bool):
    """Run interactive question loop."""
    console.rule("[bold cyan]Tarkov RAG - Interactive Quest Assistant[/bold cyan]")
    console.print("[dim]Type your quest question below. Type 'exit' or 'quit' to end.[/dim]\n")

    if not rag.has_llm_client and not force_offline:
        console.print(
            "[yellow]Note:[/] OPENAI_API_KEY is not configured in .env. Running in structured context retrieval mode.\n"
        )

    while True:
        try:
            question = Prompt.ask("[bold green]Ask Tarkov RAG[/bold green]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Exiting...[/dim]")
            break

        if not question.strip():
            continue
        if question.strip().lower() in ("exit", "quit", "q"):
            console.print("[dim]Exiting...[/dim]")
            break

        with console.status("[bold cyan]Searching quest database & generating answer..."):
            result = rag.answer_question(
                question=question,
                n_results=top_k,
                trader_filter=trader,
                force_offline=force_offline,
            )

        display_result(result)


def main():
    args = parse_args()

    store = QuestVectorStore()
    rag = TarkovRAG(vector_store=store)

    if args.query:
        # Single query mode
        with console.status("[bold cyan]Searching quest database..."):
            result = rag.answer_question(
                question=args.query,
                n_results=args.top_k,
                trader_filter=args.trader,
                force_offline=args.offline,
            )
        display_result(result)
    else:
        # Interactive mode
        run_interactive(rag, top_k=args.top_k, trader=args.trader, force_offline=args.offline)


if __name__ == "__main__":
    main()
