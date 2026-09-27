#!/usr/bin/env python3
"""
Embedding script for Escape from Tarkov quests.
Loads parsed quest JSON, generates embeddings, and indexes them in ChromaDB.
"""

import argparse
import logging
from pathlib import Path
import sys

from rich.console import Console
from rich.table import Table

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.vectorstore import QuestVectorStore

console = Console()
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed and index Tarkov quests into ChromaDB.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/quests.json",
        help="Path to the processed quests JSON file (default: data/processed/quests.json).",
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default="chroma_db",
        help="Directory to persist ChromaDB files (default: chroma_db).",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="tarkov_quests",
        help="ChromaDB collection name (default: tarkov_quests).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing collection before re-indexing.",
    )
    parser.add_argument(
        "--test-query",
        type=str,
        default="what are the objectives for Debut?",
        help="Verification query to test semantic retrieval after indexing.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    console.rule("[bold cyan]Tarkov RAG - Vector Indexing Pipeline[/bold cyan]")

    store = QuestVectorStore(
        persist_dir=args.persist_dir,
        collection_name=args.collection,
    )

    if args.reset:
        console.print("[yellow]Resetting existing Chroma collection...[/yellow]")
        try:
            store.client.delete_collection(name=args.collection)
            store.collection = store.client.get_or_create_collection(
                name=args.collection,
                metadata={"hnsw:space": "cosine"},
            )
            console.print("[green]Collection reset complete.[/green]")
        except Exception as e:
            console.print(f"[dim]Note on reset: {e}[/dim]")

    # 1. Load processed quests
    with console.status(f"[bold green]Loading quests from {args.input}..."):
        quests = store.load_quests_from_json(args.input)

    console.print(f"[green]Loaded {len(quests)} quests from JSON.[/green]")

    # 2. Index into ChromaDB
    with console.status(f"[bold cyan]Embedding and indexing {len(quests)} quests into ChromaDB..."):
        indexed_count = store.index_quests(quests, batch_size=100)

    # 3. Print indexing summary
    table = Table(title="[bold green]Indexing Summary[/bold green]", show_header=True)
    table.add_column("Property", style="bold white")
    table.add_column("Value", style="cyan")

    table.add_row("Input File", args.input)
    table.add_row("Total Quests Indexed", str(indexed_count))
    table.add_row("Chroma Storage Path", args.persist_dir)
    table.add_row("Collection Name", args.collection)
    table.add_row("Distance Metric", "Cosine Similarity")

    console.print(table)

    # 4. Run test retrieval
    if args.test_query:
        console.print()
        console.rule("[bold magenta]Verification Retrieval Test[/bold magenta]")
        console.print(f"[bold yellow]Test Query:[/] [italic]{args.test_query}[/italic]\n")

        results = store.query(args.test_query, n_results=3)

        if not results:
            console.print("[red]No results returned.[/red]")
            return

        for idx, res in enumerate(results, 1):
            console.print(
                f"[bold cyan]#{idx} {res['name']}[/bold cyan] "
                f"(Trader: [green]{res['trader']}[/green] | "
                f"Similarity: [bold yellow]{res['similarity']:.3f}[/bold yellow])"
            )
            console.print(f"   [dim]Wiki URL:[/] {res['wiki_url']}")
            if res["prerequisite_chain"]:
                console.print(f"   [dim]Prerequisite Chain:[/] {' -> '.join(res['prerequisite_chain'])}")
            console.print()


if __name__ == "__main__":
    main()
