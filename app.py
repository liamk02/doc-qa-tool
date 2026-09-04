"""Document Q&A CLI - ask questions against your own documents, answered by Claude.

Usage:
    python app.py ingest      # (re)build the search index from files in docs/
    python app.py ask         # interactive Q&A loop against that index
"""

import argparse
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from chunking import load_documents
from embed_store import embed, load_index, save_index, top_k

BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "docs"
INDEX_PATH = BASE_DIR / "index.pkl"
MODEL = "claude-opus-5"

SYSTEM_PROMPT = (
    "You answer questions using ONLY the provided document excerpts. "
    "If the excerpts don't contain the answer, say so plainly instead of guessing. "
    "Cite which source file each part of your answer comes from. "
    "Earlier turns in this conversation may reference documents too - use that "
    "history to understand follow-up questions (e.g. 'what about X instead')."
)


def cmd_ingest(_args):
    if not any(DOCS_DIR.iterdir()):
        print(f"No files found in {DOCS_DIR}. Drop some .txt/.md/.pdf files in there first.")
        return

    print(f"Reading and chunking documents from {DOCS_DIR}...")
    records = load_documents(DOCS_DIR)
    if not records:
        print("No supported files found (.txt, .md, .pdf).")
        return

    print(f"Embedding {len(records)} chunks (first run downloads a small model, be patient)...")
    embeddings = embed([r["text"] for r in records])

    save_index(records, embeddings, INDEX_PATH)
    sources = {r["source"] for r in records}
    print(f"Index built: {len(records)} chunks from {len(sources)} file(s).")
    print(f"Saved to {INDEX_PATH}")


def build_user_turn(question: str, records: list[dict], embeddings, k: int = 4) -> str:
    """Retrieve the chunks most relevant to this question and format them as a turn.

    Retrieval runs fresh on every question (using just that question's text) -
    conversation memory comes from resending prior turns below, not from this step.
    """
    query_embedding = embed([question])[0]
    indices = top_k(query_embedding, embeddings, k=k)
    context_chunks = [records[i] for i in indices]

    context_text = "\n\n---\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in context_chunks
    )
    return f"Document excerpts:\n\n{context_text}\n\nQuestion: {question}"


def cmd_ask(_args):
    if not INDEX_PATH.exists():
        print("No index found. Run `python app.py ingest` first.")
        return

    records, embeddings = load_index(INDEX_PATH)
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env / .env

    # The API is stateless - conversation memory means resending this full
    # history on every request, so Claude can see prior questions/answers.
    messages: list[dict] = []

    print(f"Loaded index: {len(records)} chunks. Ask a question (or 'quit' to exit).\n")
    while True:
        question = input("> ").strip()
        if question.lower() in {"quit", "exit"}:
            break
        if not question:
            continue

        messages.append({"role": "user", "content": build_user_turn(question, records, embeddings)})

        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=messages,
            )
            answer = next((b.text for b in response.content if b.type == "text"), "")
            print(f"\n{answer}\n")
            messages.append({"role": "assistant", "content": answer})
        except anthropic.AuthenticationError:
            print("Invalid or missing API key. Set ANTHROPIC_API_KEY in your .env file.")
            messages.pop()  # don't leave a dangling unanswered turn in history
            break
        except anthropic.APIStatusError as e:
            print(f"API error: {e.message}")
            messages.pop()


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
