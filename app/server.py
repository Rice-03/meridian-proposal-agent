"""The local web app.

    python -m app.server        then open http://127.0.0.1:8000

Routes:
    GET  /                  the page (static/index.html)
    GET  /generator/<file>  Old Mutual's generator, served unmodified from the same origin
                            so the page can call window.loadProposal() inside it
    GET  /api/status        is a key configured, which model, is the generator file present
    POST /api/propose       {"notes": "..."} -> {"proposal", "flags", "evidence", "questions", ...}

The server only listens on 127.0.0.1. The API key is read from the environment and is
never sent to the browser.
"""

import os
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from .extract import MAX_NOTES_CHARS, run
from .followup import HEADER as CLARIFICATION_HEADER
from .followup import ask as ask_questions
from .followup import clean_notes
from .llm import DEFAULT_MODEL, LLMError
from .whatsapp import register as register_whatsapp

ROOT = Path(__file__).resolve().parent.parent
GENERATOR_FILE = "challenge-generator.html"

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 256 * 1024


def _key_is_set():
    return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))


@app.get("/")
def index():
    return send_from_directory(ROOT / "static", "index.html")


@app.get("/generator/<path:name>")
def generator(name):
    return send_from_directory(ROOT / "generator", name)


@app.get("/api/status")
def status():
    return jsonify(
        llm_ready=_key_is_set(),
        model=os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL,
        generator_ready=(ROOT / "generator" / GENERATOR_FILE).exists(),
        max_notes_chars=MAX_NOTES_CHARS,
    )


@app.post("/api/propose")
def propose():
    body = request.get_json(silent=True)
    notes = body.get("notes", "") if isinstance(body, dict) else ""
    if not isinstance(notes, str):
        return jsonify(error="The notes must be text."), 400
    notes, skipped = clean_notes(notes)   # "skip" / "leave blank" answers never change the notes
    try:
        result = run(notes)
    except LLMError as exc:
        return jsonify(error=str(exc)), 400
    return jsonify(
        proposal=result.proposal,
        flags=result.flags,
        evidence=result.evidence,
        retried=result.retried,
        corrections=result.corrections,
        questions=ask_questions(result.flags, notes),
        clarification_header=CLARIFICATION_HEADER,
        skipped=skipped,
    )


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="These notes are too long."), 413


register_whatsapp(app)   # does nothing unless the Twilio settings are in the environment (see app/whatsapp.py)


def main():
    port = int(os.environ.get("PORT", "8000"))
    print(f"Open http://127.0.0.1:{port}  (Ctrl+C to stop)")
    if not _key_is_set():
        print("Note: GEMINI_API_KEY is not set. The page will open, but Generate will not work until it is.")
    app.run(host="127.0.0.1", port=port, debug=False)


if __name__ == "__main__":
    main()