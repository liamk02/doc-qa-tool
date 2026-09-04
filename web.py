"""Minimal web interface for the document Q&A tool.

Usage:
    python web.py
    (then open http://127.0.0.1:5000 in a browser)

Requires an index built first: python app.py ingest
"""

from pathlib import Path

import anthropic
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory

from embed_store import load_index
from qa import ask

BASE_DIR = Path(__file__).parent
INDEX_PATH = BASE_DIR / "index.pkl"

load_dotenv()
app = Flask(__name__, static_folder="static", static_url_path="")
client = anthropic.Anthropic()

# Loaded lazily on first request, then cached in memory for the life of the process.
_records = None
_embeddings = None


def get_index():
    global _records, _embeddings
    if _records is None:
        if not INDEX_PATH.exists():
            raise FileNotFoundError("No index found. Run `python app.py ingest` first.")
        _records, _embeddings = load_index(INDEX_PATH)
    return _records, _embeddings


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/ask", methods=["POST"])
def api_ask():
    data = request.get_json(force=True, silent=True) or {}
    question = (data.get("question") or "").strip()
    history = data.get("history") or []

    if not question:
        return jsonify({"error": "Question is empty."}), 400

    try:
        records, embeddings = get_index()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 400

    try:
        answer = ask(question, history, records, embeddings, client)
    except anthropic.AuthenticationError:
        return jsonify({"error": "Invalid or missing API key. Check your .env file."}), 500
    except anthropic.APIStatusError as e:
        return jsonify({"error": f"API error: {e.message}"}), 502

    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
