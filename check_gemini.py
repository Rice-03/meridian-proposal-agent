"""Run this first to confirm your Gemini key works with this project.

    python check_gemini.py

It does three things: shows which model will be used, lists the Gemini models your
key can see, and runs one real extraction on an invented note to prove the whole
chain (key, model, schema, checks) works end to end.
"""

import os
import sys

from app.extract import EXTRACTION_SCHEMA, run
from app.llm import DEFAULT_MODEL, LLMError

NOTE = (
    "Invented test note. Client: Mr Sam Example, individual. Wants to invest about R2.5 million, "
    "no income needed, long horizon of 10+ years. Put him in the balanced portfolio. Adviser fee 0.75%."
)


def main():
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        print("GEMINI_API_KEY is not set in this window. In Command Prompt run:  set GEMINI_API_KEY=your_key_here")
        return 1
    model = os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL
    print(f"Model that will be used: {model}")

    try:
        from google import genai
        client = genai.Client(api_key=key)
        names = sorted(m.name.replace("models/", "") for m in client.models.list())
        flash = [n for n in names if "flash" in n]
        print(f"Models your key can see: {len(names)} (flash models: {', '.join(flash[:12]) or 'none'})")
        if model not in names:
            print(f"WARNING: '{model}' is not in that list. Set another one, e.g.  set GEMINI_MODEL=<one of the flash models above>")
    except Exception as exc:
        print(f"Could not list models: {exc}")

    print("\nRunning a real extraction on an invented note...")
    try:
        result = run(NOTE)
    except LLMError as exc:
        print(f"FAILED: {exc}")
        return 1

    print("\nProposal object that would be loaded:")
    import json
    print(json.dumps(result.proposal, indent=2))
    print("\nFlags for the PM:")
    for f in result.flags:
        print(f"  [{f['level']}] {f['message']}")
    print(f"\n(correction request used: {result.retried}; schema properties: {len(EXTRACTION_SCHEMA['properties'])})")
    print("\nIf you can read a proposal above with R2500000 and 0.75, everything works.")
    return 0


if __name__ == "__main__":
    sys.exit(main())