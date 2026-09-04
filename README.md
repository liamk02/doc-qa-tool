# Document Q&A Tool

Ask questions against your own documents, answered by Claude using only what's
actually in them (retrieval-augmented generation, aka RAG).

## How it works

1. `chunking.py` reads files from `docs/` and splits them into overlapping chunks
2. `embed_store.py` turns each chunk into a vector and does similarity search
   with plain numpy. Two embedding backends, chosen automatically: local
   (`sentence-transformers`, free, no API key - the default for your own
   machine) or Voyage AI (hosted, used automatically when `VOYAGE_API_KEY` is
   set - what the public deployment uses, since it avoids PyTorch's memory
   footprint on a small server)
3. `app.py` ties it together: `ingest` builds the index, `ask` retrieves the
   most relevant chunks for your question and asks Claude to answer from them

## Setup (local)

```powershell
# From this folder:
.\venv\Scripts\Activate.ps1
pip install -r requirements-local.txt
copy .env.example .env
# then edit .env and paste in your Anthropic API key
```

## Usage

```powershell
# 1. Drop .txt / .md / .pdf files into docs/
# 2. Build the index
python app.py ingest

# 3a. Ask questions from the terminal
python app.py ask

# 3b. ...or launch the web interface instead
python web.py
# then open http://127.0.0.1:5000
```

`qa.py` holds the shared retrieval + answer logic that both `app.py` and
`web.py` call into - the only difference between the two is how the question
comes in and the answer gets displayed.

## Deploying the web app publicly

`web.py` is written to be safely exposed to anyone on the internet, not just
run locally:

- **Rate limited** - 8 questions per visitor per hour (`flask-limiter`)
- **Input validated** - oversized questions/history are rejected before they
  ever reach Claude, so a crafted request can't inflate token usage
- **Cheaper public model** - defaults to `claude-sonnet-5` instead of the
  CLI's `claude-opus-5` (override with the `PUBLIC_MODEL` env var)
- **Hosted embeddings (Voyage AI)** - set `VOYAGE_API_KEY` in the deploy
  environment. Local `sentence-transformers` needs PyTorch, which reliably
  runs a small free-tier server (e.g. Render's 512MB) out of memory; Voyage
  keeps the deployed process lightweight. `requirements.txt` deliberately
  omits `sentence-transformers`/PyTorch for this reason - local dev installs
  `requirements-local.txt` instead, which adds it back
- **Self-building index** - builds `index.pkl` from `docs/` automatically on
  first request if it's missing (or was built by a different embedding
  backend than the one currently active), so a fresh deploy needs no manual
  `ingest` step

None of that replaces a monthly spend cap - set one in the Anthropic Console
(Settings -> Billing) before deploying, regardless of the above. Same idea
for Voyage: add a payment method in their dashboard (Billing) even though the
first 200M tokens are free - without one, new accounts are throttled to 3
requests/minute, which a public demo will hit almost immediately.

### Deploy to Render (free tier)

1. Push this repo to GitHub (already done if you're reading this from here)
2. On [render.com](https://render.com), create a **New Web Service** from this repo
3. **Build command:** `pip install -r requirements.txt`
4. **Start command:** `gunicorn web:app` (or leave it - the included `Procfile` sets this)
5. Add environment variables:
   - `ANTHROPIC_API_KEY` = your Anthropic key
   - `VOYAGE_API_KEY` = your Voyage key
   - optionally `PUBLIC_MODEL` to override the default Claude model
6. Deploy - Render gives you a public URL once the build finishes

Remember: only files actually committed to this repo exist on the deployed
server. `docs/` is gitignored except the sample file, so the public deploy
only ever answers questions about that sample policy document - not any real
documents you've tested locally.

## Notes

- Only two things ever call an external API: the final answer-generation
  step (Claude) and, on the public deployment only, embeddings (Voyage AI).
- `docs/` and `index.pkl` are for your own files - be mindful before committing
  real/sensitive documents to a public GitHub repo.
