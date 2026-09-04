# Document Q&A Tool

Ask questions against your own documents, answered by Claude using only what's
actually in them (retrieval-augmented generation, aka RAG).

## How it works

1. `chunking.py` reads files from `docs/` and splits them into overlapping chunks
2. `embed_store.py` turns each chunk into a vector (local model, no API key needed)
   and does similarity search with plain numpy
3. `app.py` ties it together: `ingest` builds the index, `ask` retrieves the
   most relevant chunks for your question and asks Claude to answer from them

## Setup

```powershell
# From this folder:
.\venv\Scripts\Activate.ps1
copy .env.example .env
# then edit .env and paste in your Anthropic API key
```

## Usage

```powershell
# 1. Drop .txt / .md / .pdf files into docs/
# 2. Build the index
python app.py ingest

# 3. Ask questions
python app.py ask
```

## Notes

- Embeddings run locally (`sentence-transformers`, free, no account needed).
- Only the final answer-generation step calls the Claude API.
- `docs/` and `index.pkl` are for your own files - be mindful before committing
  real/sensitive documents to a public GitHub repo.
