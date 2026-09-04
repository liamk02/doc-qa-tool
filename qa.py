"""Shared question-answering logic used by both the CLI (app.py) and the web app (web.py)."""

from embed_store import embed, top_k

MODEL = "claude-opus-5"

SYSTEM_PROMPT = (
    "You answer questions using ONLY the provided document excerpts. "
    "If the excerpts don't contain the answer, say so plainly instead of guessing. "
    "Cite which source file each part of your answer comes from. "
    "Earlier turns in this conversation may reference documents too - use that "
    "history to understand follow-up questions (e.g. 'what about X instead')."
)


def retrieve_context(question: str, records: list[dict], embeddings, k: int = 4) -> str:
    """Find the k most relevant chunks for a question and format them as context text."""
    query_embedding = embed([question], input_type="query")[0]
    indices = top_k(query_embedding, embeddings, k=k)
    context_chunks = [records[i] for i in indices]
    return "\n\n---\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in context_chunks
    )


def ask(
    question: str,
    history: list[dict],
    records: list[dict],
    embeddings,
    client,
    model: str = MODEL,
) -> str:
    """Answer a question and return just the answer text.

    `history` is a list of {"role": "user"|"assistant", "content": <plain text>}.
    Only the CURRENT question gets document context injected - past turns are
    kept as plain text. That's cheaper than resending retrieved chunks every
    turn, and Claude's own prior answers already carry the retrieved facts
    forward for follow-up questions to build on.
    """
    context_text = retrieve_context(question, records, embeddings)
    current_turn = {
        "role": "user",
        "content": f"Document excerpts:\n\n{context_text}\n\nQuestion: {question}",
    }

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=history + [current_turn],
    )
    return next((block.text for block in response.content if block.type == "text"), "")
