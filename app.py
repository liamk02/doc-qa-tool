"""Document Q&A CLI - ask questions against your own documents, answered by Claude.

Usage:
    python app.py ingest      # (re)build the search index from files in docs/
    python app.py ask         # interactive Q&A loop against that index
"""

import argparse
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from embed_store import load_index
from indexer import build_index
from qa import ask

BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "docs"
INDEX_PATH = BASE_DIR / "index.pkl"


def cmd_ingest(_args):
    if not any(DOCS_DIR.iterdir()):
        print(f"No files found in {DOCS_DIR}. Drop some .txt/.md/.pdf files in there first.")
        return

    print(f"Reading, chunking, and embedding documents from {DOCS_DIR}...")
    print("(first run downloads a small model, be patient)")
    count = build_index(DOCS_DIR, INDEX_PATH)
    if count == 0:
        print("No supported files found (.txt, .md, .pdf).")
        return

    print(f"Index built: {count} chunks.")
    print(f"Saved to {INDEX_PATH}")


def cmd_ask(_args):
    if not INDEX_PATH.exists():
        print("No index found. Run `python app.py ingest` first.")
        return

    records, embeddings = load_index(INDEX_PATH)
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env / .env

    # Plain-text history: {"role": ..., "content": <plain text, no document context>}.
    # See qa.py for why only the newest question gets document context injected.
    history: list[dict] = []

    print(f"Loaded index: {len(records)} chunks. Ask a question (or 'quit' to exit).\n")
    while True:
        question = input("> ").strip()
        if question.lower() in {"quit", "exit"}:
            break
        if not question:
            continue

        try:
            answer = ask(question, history, records, embeddings, client)
            print(f"\n{answer}\n")
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": answer})
        except anthropic.AuthenticationError:
            print("Invalid or missing API key. Set ANTHROPIC_API_KEY in your .env file.")
            break
        except anthropic.APIStatusError as e:
            print(f"API error: {e.message}")


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Document Q&A CLI powered by Claude.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest", help="Build the index from docs/").set_defaults(func=cmd_ingest)
    sub.add_parser("ask", help="Ask questions against the index").set_defaults(func=cmd_ask)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
