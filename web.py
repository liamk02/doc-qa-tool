"""Web interface for the document Q&A tool - safe to deploy publicly.

Local usage:
    python app.py ingest   # optional - web.py builds the index itself if missing
    python web.py
    (then open http://127.0.0.1:5000)

Public deployment (e.g. Render): set ANTHROPIC_API_KEY as a server-side
environment variable (never exposed to the browser) and run with a real WSGI
server, e.g.:
    gunicorn web:app

Because this endpoint can be hit by anyone on the internet once deployed, it
has three layers of cost protection on top of what app.py needs locally:
1. Rate limiting (flask-limiter) - caps requests per visitor.
2. Input validation - rejects oversized questions/history before they reach
   the API, so a crafted request can't inflate token usage.
3. A cheaper default model (PUBLIC_MODEL) than the local CLI uses.
You should ALSO set a monthly spend cap in the Anthropic Console - none of
the above replaces that; they just keep normal/abusive traffic cheap.
"""

import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from indexer import load_or_build_index
from qa import ask

BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "docs"
INDEX_PATH = BASE_DIR / "index.pkl"

# Cheaper than the CLI's default model - this endpoint can be hit by anyone,
# so the public-facing cost profile per question should stay low. Override
# with the PUBLIC_MODEL env var if you want a different tradeoff.
PUBLIC_MODEL = os.environ.get("PUBLIC_MODEL", "claude-sonnet-5")

MAX_QUESTION_CHARS = 500
MAX_HISTORY_ENTRIES = 12       # 6 question/answer pairs
MAX_HISTORY_ENTRY_CHARS = 2000

load_dotenv()
app = Flask(__name__, static_folder="static", static_url_path="")
client = anthropic.Anthropic()

limiter = Limiter(get_remote_address, app=app, default_limits=[])

# Loaded lazily on first request, then cached in memory for the life of the process.
_records = None
_embeddings = None


def get_index():
    global _records, _embeddings
    if _records is None:
        if not any(DOCS_DIR.iterdir()):
            raise FileNotFoundError("No documents found to index - add files to docs/.")
        # Builds automatically if missing, or if it was built by a different
        # embedding backend (vectors aren't compatible across backends) - see
        # the README on keeping real/sensitive documents out of a public deployment.
        _records, _embeddings = load_or_build_index(DOCS_DIR, INDEX_PATH)
    return _records, _embeddings


def validate_ask_request(data: dict) -> str | None:
    """Return an error message if the request is invalid, else None.

    Runs BEFORE anything reaches Claude - this is what stops a crafted
    request from inflating token usage on a public endpoint.
    """
    question = (data.get("question") or "").strip()
    if not question:
        return "Frågan är tom."
    if len(question) > MAX_QUESTION_CHARS:
        return f"Frågan är för lång (max {MAX_QUESTION_CHARS} tecken)."

    history = data.get("history")
    if history is None:
        history = []
    if not isinstance(history, list) or len(history) > MAX_HISTORY_ENTRIES:
        return "Konversationshistoriken är ogiltig eller för lång."
    for turn in history:
        if (
            not isinstance(turn, dict)
            or turn.get("role") not in ("user", "assistant")
            or not isinstance(turn.get("content"), str)
            or len(turn["content"]) > MAX_HISTORY_ENTRY_CHARS
        ):
            return "Konversationshistoriken är ogiltig eller för lång."
    return None


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/health")
def health():
    # Deliberately makes no Claude API call - a liveness check should never
    # cost money, however often (or however many visitors' browsers) it runs.
    response = jsonify({"status": "ok", "service": "doc-qa-tool"})
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response


@app.route("/api/ask", methods=["POST"])
@limiter.limit("8 per hour")
def api_ask():
    data = request.get_json(force=True, silent=True) or {}

    error = validate_ask_request(data)
    if error:
        return jsonify({"error": error}), 400

    question = data["question"].strip()
    history = data.get("history") or []

    try:
        records, embeddings = get_index()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 400

    try:
        answer = ask(question, history, records, embeddings, client, model=PUBLIC_MODEL)
    except anthropic.AuthenticationError:
        return jsonify({"error": "Servern är felkonfigurerad (ogiltig API-nyckel)."}), 500
    except anthropic.RateLimitError:
        return jsonify({"error": "Demon har mycket trafik just nu - försök igen om en stund."}), 429
    except anthropic.APIStatusError as e:
        return jsonify({"error": f"API-fel: {e.message}"}), 502

    return jsonify({"answer": answer})


@app.errorhandler(429)
def rate_limited(_e):
    return jsonify({"error": "Du har nått frågegränsen för den här demon - försök igen om en stund."}), 429


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(debug=debug, port=int(os.environ.get("PORT", 5000)))
