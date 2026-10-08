"""WhatsApp front door (Twilio sandbox). Optional: nothing here runs unless it is switched on.

A Portfolio Manager (PM) sends notes, or a voice note, to the Twilio WhatsApp sandbox number.
This module runs the SAME pipeline as the web page and replies in the chat:

    text or voice note  ->  (voice: Gemini transcribes it, and the PM is shown what was heard)
                        ->  extract + verify (extract.run)  ->  follow-up questions (followup.ask)
                        ->  a short text summary with the gaps and the questions
                        ->  the proposal PDF, made by the real generator (pdf_export.py), once there is
                            no open question. While the agent is still asking, the PDF waits; the PM can
                            reply "pdf" to get the draft as it is.

The PM answers a question by replying "1: balanced". The answer is added to the notes under the
same PM CLARIFICATION heading the web page uses, and the pipeline runs again.

It is switched on by setting these environment variables (all three are needed):
    TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN   from the Twilio console
    PUBLIC_BASE_URL                         the public address of the tunnel, e.g. https://abc123.ngrok-free.app
Optional:
    TWILIO_WHATSAPP_FROM   the sandbox number, default whatsapp:+14155238886
    WHATSAPP_ALLOWED       comma separated phone numbers that may use it; empty means anyone who joined the sandbox
    PORT                   the port this app runs on, default 8000

Safety: a tunnel makes the whole app reachable from the internet, including /api/propose, which uses the
Gemini key. So while this is on, every request that arrives through a tunnel (it carries forwarded-for
headers) is refused unless it is for /whatsapp or for a PDF file. /whatsapp also checks Twilio's signature.
"""

import base64
import hashlib
import hmac
import os
import re
import secrets
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from urllib.parse import urlparse

from flask import Response, abort, request, send_file

from .extract import run
from .followup import HEADER, ask, clean_notes, split_clarification
from .llm import LLMError, transcribe_audio
from .pdf_export import PdfExportError, export_pdf

DEFAULT_SENDER = "whatsapp:+14155238886"
MAX_BODY = 1500            # Twilio refuses a WhatsApp message longer than 1600 characters
MAX_ANSWER_CHARS = 300     # a longer message while a question is open is treated as new notes
SESSION_SECONDS = 3600
MEDIA_KEEP_SECONDS = 3600
MEDIA_DIR = Path(tempfile.gettempdir()) / "meridian-whatsapp-pdfs"
_MEDIA_NAME = re.compile(r"^[0-9a-f]{32}\.pdf$")
_FORWARDED_HEADERS = ("X-Forwarded-For", "X-Forwarded-Host", "Forwarded", "Cf-Connecting-Ip")

MODEL_NAMES = {"balanced": "Balanced", "global-growth": "Global growth", "income": "Income"}

EXECUTOR = ThreadPoolExecutor(max_workers=1)   # one message at a time: the PDF step is heavy and Gemini has a daily limit
_SESSIONS = {}
_LOCK = threading.Lock()


class WhatsAppError(Exception):
    pass


# ------------------------------------------------------------ settings

@dataclass(frozen=True)
class Config:
    sid: str
    token: str
    sender: str
    public_url: str
    local_url: str
    allowed: tuple


def _whatsapp_number(text):
    t = text.strip().lower()
    return t if t.startswith("whatsapp:") else "whatsapp:" + t


def config_from_env(env=None):
    env = os.environ if env is None else env
    sid, token, public = (env.get(k, "").strip() for k in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "PUBLIC_BASE_URL"))
    if not (sid and token and public):
        return None
    allowed = tuple(_whatsapp_number(n) for n in env.get("WHATSAPP_ALLOWED", "").split(",") if n.strip())
    return Config(
        sid=sid, token=token,
        sender=env.get("TWILIO_WHATSAPP_FROM", "").strip() or DEFAULT_SENDER,
        public_url=public.rstrip("/"),
        local_url="http://127.0.0.1:" + (env.get("PORT", "").strip() or "8000"),
        allowed=allowed,
    )


# ------------------------------------------------------------ Twilio: signature, sending, downloading

def twilio_signature(token, url, params):
    """Twilio signs: the full URL followed by every POST parameter (name then value) sorted by name."""
    data = url + "".join(k + params[k] for k in sorted(params))
    return base64.b64encode(hmac.new(token.encode(), data.encode(), hashlib.sha1).digest()).decode()


def valid_signature(token, url, params, header):
    return bool(header) and hmac.compare_digest(twilio_signature(token, url, params), header)


def send_message(cfg, to, body=None, media_url=None):
    import httpx   # installed with google-genai
    data = {"From": cfg.sender, "To": to}
    if body:
        data["Body"] = body[:MAX_BODY]
    if media_url:
        data["MediaUrl"] = media_url
    r = httpx.post(f"https://api.twilio.com/2010-04-01/Accounts/{cfg.sid}/Messages.json",
                   data=data, auth=(cfg.sid, cfg.token), timeout=30)
    if r.status_code >= 400:
        raise WhatsAppError(f"Twilio refused the message ({r.status_code}): {r.text[:200]}")


def download_media(cfg, url):
    import httpx
    host = (urlparse(url).hostname or "").lower()
    if not (host == "twilio.com" or host.endswith(".twilio.com")):
        raise WhatsAppError("The voice note did not come from Twilio.")
    r = httpx.get(url, auth=(cfg.sid, cfg.token), follow_redirects=True, timeout=60)   # auth is not forwarded to other hosts
    r.raise_for_status()
    return r.content


# ------------------------------------------------------------ the conversation

def _session(sender):
    with _LOCK:
        s = _SESSIONS.get(sender)
        if s and time.time() - s["at"] > SESSION_SECONDS:
            _SESSIONS.pop(sender, None)
            s = None
        return s


def _store(sender, notes, questions, proposal):
    with _LOCK:
        _SESSIONS[sender] = {"notes": notes, "questions": questions, "proposal": proposal, "at": time.time()}


def _forget(sender):
    with _LOCK:
        _SESSIONS.pop(sender, None)


_NUMBERED = re.compile(r"^\s*(\d+)\s*[:.)\-]\s*(.+?)\s*$")


def parse_answers(text, questions):
    """'1: balanced' lines -> {0: 'balanced'}. With exactly one open question, a plain reply is its answer."""
    answers = {}
    for line in (text or "").splitlines():
        m = _NUMBERED.match(line)
        if m and 1 <= int(m.group(1)) <= len(questions):
            answers[int(m.group(1)) - 1] = m.group(2)
    if not answers and len(questions) == 1 and (text or "").strip():
        answers[0] = " ".join(text.split())
    return answers


def add_answers(notes, questions, answers):
    """Add answers to the notes the way the web page does: '- label: answer' under the clarification heading.
    An answer to a topic that was already answered replaces the older one."""
    main, clar = split_clarification(notes)
    lines = {}
    for line in clar[len(HEADER):].splitlines() if clar else []:
        m = re.match(r"^\s*-\s*([^:\n]+):\s*(.*)$", line)
        if m:
            lines[m.group(1).strip()] = m.group(2).strip()
    for i, answer in answers.items():
        lines[questions[i]["label"]] = " ".join(answer.split())
    return main.rstrip() + "\n\n" + HEADER + "\n" + "\n".join(f"- {k}: {v}" for k, v in lines.items())


def _money(value, currency):
    try:
        d = Decimal(str(value))
    except InvalidOperation:
        return str(value)
    text = f"{d:,.0f}" if d == d.to_integral() else f"{d:,.2f}"
    text = text.replace(",", " ")
    return ("R " if currency == "ZAR" else (currency + " " if currency else "")) + text


def summary_text(proposal, flags, questions, skipped=()):
    """The text reply. Only things the agent found are listed as found; everything else is a gap."""
    goal = proposal.get("goalAssessment", {})
    obj = proposal.get("objective", {})
    figures = proposal.get("strategy", {}).get("keyFigures", {})
    currency = goal.get("currency") or proposal.get("strategy", {}).get("currency")
    found = []

    def add(label, value):
        if value:
            found.append(f"{label}: {value}")

    add("Client type", goal.get("clientType"))
    add("Mandate", goal.get("mandateType"))
    add("Model", MODEL_NAMES.get(proposal.get("applyModel")))
    add("Amount", _money(figures["totalInvestment"], currency) if figures.get("totalInvestment") else None)
    add("Income per month", _money(figures["incomeValue"], currency) if figures.get("incomeValue") else None)
    add("Horizon", obj.get("investmentHorizon") or goal.get("investmentHorizon"))
    add("Target return", obj.get("targetReturn"))
    add("Benchmark", obj.get("benchmark"))
    fee = proposal.get("fees", {}).get("advisorFee")
    add("Adviser fee", f"{fee}%" if fee else None)

    asked = {m for q in questions for m in q.get("flags", [])}
    gaps = [f["message"] for f in flags if f["level"] == "missing" and f["message"] not in asked]
    check = [f["message"] for f in flags if f["level"] in ("unverified", "assumption") and f["message"] not in asked]

    out = [f"Draft proposal for {proposal.get('clientName') or 'the client'}"]
    out.append("\n".join(found) if found else "I could not find any figures in the notes.")
    if skipped:
        out.append("Skipped, no change made: " + ", ".join(skipped) + ".")
    if gaps:
        out.append("Not in the notes (left blank, not guessed):\n" + "\n".join("- " + g for g in gaps))
    if check:
        out.append("Please check:\n" + "\n".join("- " + c for c in check))
    if questions:
        out.append("Questions:\n" + "\n".join(f"{i}. {q['question']}" for i, q in enumerate(questions, 1))
                   + '\nReply with your answers, for example "1: balanced". Or send "new" and then fresh notes.')
    else:
        out.append('Send "new" and then fresh notes to start another proposal.')
    out.append('Answer them and I will send the PDF. Or reply "pdf" to get the draft as it is.' if questions
               else "The PDF follows in a moment.")
    text = "\n\n".join(out)
    return text if len(text) <= MAX_BODY else text[: MAX_BODY - 1].rstrip() + "…"


HELP = ("Send me the meeting notes as a message or a voice note and I will draft the proposal.\n"
        "I never guess a figure: anything the notes do not say is listed as a gap, and if I am unsure I ask.\n"
        'If I have questions, answer them and I will send the PDF. Reply "pdf" to get the draft without waiting.\n'
        'To start over, send "new".')


def _clean_old_pdfs():
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    for f in MEDIA_DIR.glob("*.pdf"):
        try:
            if time.time() - f.stat().st_mtime > MEDIA_KEEP_SECONDS:
                f.unlink()
        except OSError:
            pass


def process_message(cfg, sender, text, media_url=None, media_type=None):
    """Handle one incoming WhatsApp message. Runs in the worker thread. Never raises."""
    try:
        _process(cfg, sender, text, media_url, media_type)
    except (LLMError, WhatsAppError, PdfExportError) as exc:
        _try_send(cfg, sender, str(exc))
    except Exception:
        traceback.print_exc()
        _try_send(cfg, sender, "Sorry, something went wrong on my side. Please try again.")


def _try_send(cfg, to, body):
    try:
        send_message(cfg, to, body)
    except Exception:
        traceback.print_exc()


def _process(cfg, sender, text, media_url, media_type):
    text = (text or "").strip()

    if media_url and (media_type or "").lower().startswith("audio"):
        heard = transcribe_audio(download_media(cfg, media_url), media_type)
        send_message(cfg, sender, f'I heard: "{heard}"\n\nIf that is wrong, send the notes again as text.')
        text = heard

    if not text:
        send_message(cfg, sender, HELP)
        return

    words = text.split(None, 1)
    first = words[0].lower().strip('.,:;!"')
    if first in ("hi", "hello", "help", "?"):
        send_message(cfg, sender, HELP)
        return
    if text.lower().strip(' .!"') == "pdf":
        session = _session(sender)
        if not session:
            send_message(cfg, sender, "There is nothing to send yet. Send me the meeting notes first.")
        else:
            _send_pdf(cfg, sender, session["proposal"])
        return
    fresh = first == "new"
    if fresh:
        _forget(sender)
        text = words[1].lstrip(" :,-\n") if len(words) > 1 else ""
        if not text:
            send_message(cfg, sender, "Ok. Send the new notes as a message or a voice note.")
            return

    session = None if fresh else _session(sender)
    if session and session["questions"] and len(text) <= MAX_ANSWER_CHARS:
        answers = parse_answers(text, session["questions"])
        if not answers:
            send_message(cfg, sender, 'I did not see which question that answers. Reply like "1: balanced", or send "new" and then fresh notes.')
            return
        notes = add_answers(session["notes"], session["questions"], answers)
    else:
        notes = text

    notes, skipped = clean_notes(notes)
    send_message(cfg, sender, "Got it, reading the notes.")
    result = run(notes)
    questions = ask(result.flags, notes)
    _store(sender, notes, questions, result.proposal)
    send_message(cfg, sender, summary_text(result.proposal, result.flags, questions, skipped))
    if not questions:   # while the agent is still unsure, the PDF waits for the answers (or for the PM to say "pdf")
        _send_pdf(cfg, sender, result.proposal)


def _send_pdf(cfg, sender, proposal):
    _clean_old_pdfs()
    name = secrets.token_hex(16) + ".pdf"
    export_pdf(proposal, cfg.local_url, MEDIA_DIR / name)
    who = proposal.get("clientName") or "the client"
    send_message(cfg, sender, f"Draft proposal for {who}. Please check the points above before it goes to the client.",
                 media_url=f"{cfg.public_url}/media/{name}")


# ------------------------------------------------------------ the web routes

def register(app, env=None):
    """Add /whatsapp and the PDF route to the app, if WhatsApp is configured. Returns the Config or None."""
    cfg = config_from_env(env)
    if cfg is None:
        return None

    @app.before_request
    def refuse_everything_else_from_the_internet():
        if any(h in request.headers for h in _FORWARDED_HEADERS):
            if not (request.path == "/whatsapp" or request.path.startswith("/media/")):
                abort(404)

    def webhook():
        params = request.form.to_dict()
        if not valid_signature(cfg.token, cfg.public_url + "/whatsapp", params, request.headers.get("X-Twilio-Signature", "")):
            abort(403)
        sender = params.get("From", "")
        if cfg.allowed and sender.lower() not in cfg.allowed:
            return Response("<Response></Response>", mimetype="text/xml")
        has_media = (params.get("NumMedia") or "0") not in ("", "0")
        EXECUTOR.submit(process_message, cfg, sender, params.get("Body", ""),
                        params.get("MediaUrl0") if has_media else None, params.get("MediaContentType0") if has_media else None)
        return Response("<Response></Response>", mimetype="text/xml")   # the real reply is sent separately, so Twilio's 15 second limit never matters

    def media(name):
        path = MEDIA_DIR / name
        if not _MEDIA_NAME.match(name) or not path.is_file():
            abort(404)
        return send_file(path, mimetype="application/pdf")

    app.add_url_rule("/whatsapp", "whatsapp_webhook", webhook, methods=["POST"])
    app.add_url_rule("/media/<name>", "whatsapp_media", media)
    print(f"WhatsApp is ON. In the Twilio sandbox settings, set 'When a message comes in' to {cfg.public_url}/whatsapp (method POST).")
    return cfg