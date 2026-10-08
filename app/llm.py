"""The only file that talks to an LLM provider.

generate_json(system_prompt, user_text, schema) -> dict

To use a different provider (Claude, OpenAI, a local model), replace the body of
generate_json. Nothing else in the project needs to change: the rest of the code
only ever sees a Python dict.

Configuration (environment variables):
    GEMINI_API_KEY   your Google AI Studio key (GOOGLE_API_KEY also works)
    GEMINI_MODEL     optional, defaults to DEFAULT_MODEL below
    GEMINI_FALLBACK_MODEL  optional backup model used when the main one is overloaded or out of quota
                     (defaults to FALLBACK_MODEL below; set it to empty to turn the backup off)
"""

import json
import os
import re
import time

DEFAULT_MODEL = "gemini-3.8-flash"
FALLBACK_MODEL = "gemini-3.1-flash-lite"   # tried when the main model is overloaded or out of quota
REQUEST_TIMEOUT_MS = 40000                 # one request may not hang the page for longer than this
ATTEMPTS_PER_MODEL = 2


class LLMError(Exception):
    """Raised for any problem getting a usable answer from the model."""


def parse_json(text):
    """Turn the model's reply into a dict, tolerating ```json fences."""
    if text is None or not str(text).strip():
        raise LLMError("The model returned an empty answer.")
    cleaned = str(text).strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise LLMError(f"The model's answer was not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise LLMError("The model's answer was JSON, but not an object.")
    return data


def _is_temporary(error_text):
    t = error_text.lower()
    return any(s in t for s in ("429", "503", "unavailable", "overloaded", "resource_exhausted", "timed out", "timeout"))


def _is_daily_quota(error_text):
    t = error_text.lower()
    return "perday" in t or "per day" in t


def describe_failure(error_text, model_name):
    """A short, readable message instead of the provider's raw error."""
    t = error_text.lower()
    if "429" in t or "resource_exhausted" in t or "quota" in t:
        if _is_daily_quota(error_text):
            return (f"The daily request limit for the Gemini model '{model_name}' has been reached on this key. "
                    "Try again later, or set GEMINI_MODEL to a different model (python check_gemini.py lists the ones your key can use).")
        return ("Gemini is limiting the number of requests per minute right now. Wait a minute and try again.")
    return f"Gemini request failed (model '{model_name}'): {error_text}"


def _models_to_try(model=None):
    """The main model first, then one backup (Google's free models are sometimes overloaded)."""
    first = model or os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL
    backup = os.environ.get("GEMINI_FALLBACK_MODEL", FALLBACK_MODEL).strip()   # set it to empty to turn the backup off
    return [first] + ([backup] if backup and backup != first else [])


def _another_model_may_help(error_text):
    t = error_text.lower()
    return _is_temporary(error_text) or _is_daily_quota(error_text) or "404" in t or "not found" in t


def _setup():
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise LLMError("GEMINI_API_KEY is not set. Set it in the same window you start the app from.")
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise LLMError("The google-genai package is not installed. Run: pip install -r requirements.txt") from exc
    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS))
    return client, types


def _with_backup(model, request):
    """Run request(model_name) on the main model; if it is overloaded or out of quota, once on the backup."""
    names = _models_to_try(model)
    last_error, last_model = "", ""
    for model_name in names:
        last_model = model_name
        for attempt in range(ATTEMPTS_PER_MODEL):
            try:
                result = request(model_name)
                if model_name != names[0]:
                    print(f"Note: the main model was unavailable, so '{model_name}' answered this request.")
                return result
            except LLMError:
                raise
            except Exception as exc:  # network, quota, bad model name, blocked content...
                last_error = str(exc)
                if _is_daily_quota(last_error):
                    break  # waiting a few seconds will not help on this model
                if attempt < ATTEMPTS_PER_MODEL - 1 and _is_temporary(last_error):
                    time.sleep(2 * (attempt + 1))
                    continue
                break
        if not _another_model_may_help(last_error):
            break   # for example a bad key: a different model will not fix it
    raise LLMError(describe_failure(last_error, last_model))


def generate_json(system_prompt, user_text, schema, model=None):
    client, types = _setup()
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=0,
        response_mime_type="application/json",
        response_json_schema=schema,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),  # we use no tools; silences an SDK notice
    )
    return _with_backup(model, lambda name: parse_json(
        client.models.generate_content(model=name, contents=user_text, config=config).text))


TRANSCRIBE_PROMPT = (
    "Transcribe this voice note exactly as it is spoken. It is a Portfolio Manager talking about a client meeting; "
    "South African English and some Afrikaans or Zulu names are likely. "
    "Return only the transcript as plain text. Do not summarise, correct, translate or add anything, "
    "and do not guess words you cannot hear. Keep numbers and amounts exactly as said."
)


def transcribe_audio(audio_bytes, mime_type, model=None):
    """Voice note -> text, with the same backup-model behaviour as generate_json."""
    if not audio_bytes:
        raise LLMError("The voice note was empty.")
    client, types = _setup()
    mime = (mime_type or "audio/ogg").split(";")[0].strip().lower()
    config = types.GenerateContentConfig(temperature=0)
    part = types.Part.from_bytes(data=audio_bytes, mime_type=mime)

    def request(name):
        text = client.models.generate_content(model=name, contents=[TRANSCRIBE_PROMPT, part], config=config).text
        if text is None or not str(text).strip():
            raise LLMError("Nothing could be heard in the voice note.")
        return str(text).strip()

    return _with_backup(model, request)