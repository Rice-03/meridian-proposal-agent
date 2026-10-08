"""The "never invent a figure" safeguard.

The LLM returns what it believes the notes say, plus, for every figure, the
exact words from the notes it relied on (its "evidence"). This module does not
trust any of that. For every figure it checks, in plain code, that:

  1. the evidence really is a quote from the notes, and
  2. the number the LLM reported really is a number in that quote.

Anything that fails is removed and reported to the PM instead of being sent
to the proposal. Anything not stated at all is reported as missing.

verify(extraction, notes) -> Report(clean, flags, evidence)
"""

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

from .followup import split_clarification
from .numparse import normalize, numbers_in

MODELS = ("balanced", "global-growth", "income")

# Asset-class keys the generator recognises (proposal-schema.md).
ALLOCATION_KEYS = ("sa-equity", "intl-equity", "fixed-interest", "bonds", "property", "cash", "other")

# Currency each shared model uses (from proposal-schema.md).
MODEL_CURRENCY = {"balanced": "ZAR", "global-growth": "USD", "income": "ZAR"}

CURRENCY_WORDS = {
    "ZAR": ("zar", "rand", "r"),
    "USD": ("usd", "us dollar", "dollar", "us$", "$"),
    "GBP": ("gbp", "pound", "sterling"),
    "EUR": ("eur", "euro"),
}

# Figures that must be backed by a quote from the notes.
#   (value key, evidence key, kind, label)
FIGURES = [
    ("investment_amount", "investment_amount_evidence", "amount", "Investment amount"),
    ("income_amount", "income_evidence", "amount", "Income amount"),
    ("adviser_fee_percent", "adviser_fee_evidence", "number", "Adviser fee"),
    ("target_return", "target_return_evidence", "return", "Target return"),
    ("horizon", "horizon_evidence", "horizon", "Investment horizon"),
]

_GENERIC_CAPITALISED = {"index", "composite", "global", "local", "equity", "the", "a"}
_RETURN_FILLER = {"plus", "per", "annum", "p", "a", "pa", "return", "target", "above", "over"}


@dataclass
class Flag:
    level: str      # "missing" | "unverified" | "assumption" | "info"
    field: str
    message: str

    def as_dict(self):
        return {"level": self.level, "field": self.field, "message": self.message}


@dataclass
class Report:
    clean: dict = field(default_factory=dict)
    flags: list = field(default_factory=list)
    evidence: list = field(default_factory=list)  # [{"field","value","quote"}]


# ---------------------------------------------------------------- helpers

def _blank(value):
    return value is None or (isinstance(value, str) and not value.strip())


def _to_decimal(value):
    try:
        return Decimal(str(value).replace(",", "").replace(" ", ""))
    except (InvalidOperation, ValueError):
        return None


def quote_is_in_notes(quote, notes):
    q = normalize(quote)
    return bool(q) and q in normalize(notes)


def _tokens(text):
    return set(normalize(text).split())


def _alpha_words(text):
    return {w for w in re.findall(r"[a-z]+", (text or "").lower())}


def _check_figure(kind, claimed, quote):
    """Return (ok, reason). `claimed` is what the LLM reported, `quote` its evidence."""
    quote_numbers = numbers_in(quote)

    if kind in ("amount", "number"):
        value = _to_decimal(claimed)
        if value is None:
            return False, "the reported value is not a number"
        if value not in quote_numbers:
            return False, "that number does not appear in the quoted words"
        return True, ""

    claimed_numbers = numbers_in(str(claimed))

    if kind == "return":
        if not claimed_numbers:
            return False, "the reported target return contains no number"
        if not all(n in quote_numbers for n in claimed_numbers):
            return False, "a number in it does not appear in the quoted words"
        # words such as "CPI" must also come from the quote
        needed = _alpha_words(str(claimed)) - _RETURN_FILLER
        if not needed <= _alpha_words(quote):
            return False, "a term in it (e.g. an index name) does not appear in the quoted words"
        return True, ""

    if kind == "horizon":
        if claimed_numbers and not all(n in quote_numbers for n in claimed_numbers):
            return False, "a number in it does not appear in the quoted words"
        return True, ""

    return False, "unknown kind"


# ----------------------------------------------------------------- verify

_HEDGED = re.compile(r"-ish\b|\bi think\b|\bmaybe\b|\bperhaps\b|\bprobably\b|\bsort of\b|\bkind of\b|\bsomething like\b", re.I)

_EDGE_QUOTES = " \t\r\n\"'\u201c\u201d\u2018\u2019`"


def _tidy(value):
    """Strip whitespace and stray quote marks the model sometimes leaves around text."""
    return value.strip(_EDGE_QUOTES) if isinstance(value, str) else value


_LEADING_ARTICLE = re.compile(r"^(?:a|an|the)\s+", re.I)


def _tidy_benchmark(text):
    """'a global equity index' -> 'Global equity index' (it is printed as a label)."""
    text = _LEADING_ARTICLE.sub("", text.strip())
    return text[:1].upper() + text[1:]


def verify(extraction, notes):
    ext = {k: _tidy(v) for k, v in dict(extraction or {}).items()}
    report = Report()
    clean = {}
    flags = report.flags
    notes_numbers = numbers_in(notes)
    notes_tokens = _tokens(notes)
    notes_norm = normalize(notes)

    # ---- who the proposal is for
    name = ext.get("client_name")
    if _blank(name):
        flags.append(Flag("missing", "client_name", "Client name not found in the notes."))
    elif not _tokens(name) <= notes_tokens:
        flags.append(Flag("unverified", "client_name",
                          f"The model reported the client as '{name}', but those words are not all in the notes. Left blank: please enter the name."))
    else:
        clean["client_name"] = name.strip()

    advisor = ext.get("financial_advisor_name")
    if not _blank(advisor):
        if _tokens(advisor) <= notes_tokens:
            clean["financial_advisor_name"] = advisor.strip()
        else:
            flags.append(Flag("unverified", "financial_advisor_name",
                              f"The model reported the financial advisor as '{advisor}', but those words are not all in the notes. Left blank."))

    addressed = ext.get("addressed_to")
    if addressed == "advisor":
        if "financial_advisor_name" in clean:
            clean["addressed_to"] = "advisor"
        else:
            flags.append(Flag("unverified", "addressed_to",
                              "Notes say the proposal goes to the advisor, but no advisor name could be confirmed. Addressed to the client for now."))
    elif addressed == "client":
        clean["addressed_to"] = "client"

    # ---- classification
    ctype = ext.get("client_type")
    if not _blank(ctype):
        clean["client_type"] = ctype
        if normalize(ctype) not in notes_norm:
            flags.append(Flag("assumption", "client_type",
                              f"Client type '{ctype}' is not stated in the notes; it was inferred. Please confirm."))

    mandate = ext.get("mandate_type")
    if not _blank(mandate):
        if normalize(mandate) in notes_norm:
            clean["mandate_type"] = mandate
        else:
            flags.append(Flag("missing", "mandate_type",
                              f"Mandate type '{mandate}' is not stated in the notes. Left blank: please confirm."))
    else:
        flags.append(Flag("missing", "mandate_type", "Mandate type (e.g. discretionary) not stated."))

    # ---- currency
    ccy = ext.get("currency")
    if not _blank(ccy):
        words = CURRENCY_WORDS.get(ccy)
        ok = False
        if words:
            for w in words:
                if w in ("r", "$", "us$"):
                    # symbols: look in the raw notes
                    if re.search(r"(?<![A-Za-z])" + re.escape(w) + r"\s?\d", notes, re.IGNORECASE):
                        ok = True
                elif " " in w:
                    if normalize(w) in notes_norm:
                        ok = True
                elif w in notes_tokens or (w + "s") in notes_tokens:
                    ok = True
        if ok:
            clean["currency"] = ccy
        else:
            flags.append(Flag("unverified", "currency",
                              f"The model reported currency {ccy}, but it is not stated in the notes. Left to the model portfolio's default."))

    # ---- model portfolio
    model = ext.get("model")
    model_quote = ext.get("model_evidence")
    if model in MODELS and not _blank(model_quote) and quote_is_in_notes(model_quote, notes):
        clean["model"] = model
        report.evidence.append({"field": "Model portfolio", "value": model, "quote": model_quote})
        _original, clarification = split_clarification(notes)
        said_by_pm_in_answer = quote_is_in_notes(model_quote, clarification) and not _HEDGED.search(model_quote)
        if said_by_pm_in_answer:
            pass  # the PM named the portfolio when the agent asked: that is the confirmation
        elif ext.get("model_is_tentative") is True or _HEDGED.search(model_quote):
            flags.append(Flag("assumption", "model",
                              f"The notes only hint at a portfolio (\"{model_quote}\"). '{model}' was applied tentatively: "
                              "please confirm this is what was meant."))
        else:
            flags.append(Flag("assumption", "model",
                              f"Model portfolio '{model}' was chosen from the words \"{model_quote}\". Please confirm it is the right one."))
    else:
        if model in MODELS:
            flags.append(Flag("unverified", "model",
                              f"The model chose portfolio '{model}' but gave no quote from the notes to support it. No portfolio applied."))
        else:
            flags.append(Flag("missing", "model", "No model portfolio could be chosen from the notes."))

    if "model" in clean and "currency" in clean and MODEL_CURRENCY[clean["model"]] != clean["currency"]:
        flags.append(Flag("assumption", "currency",
                          f"The notes use {clean['currency']} but the '{clean['model']}' portfolio is in {MODEL_CURRENCY[clean['model']]}. "
                          f"{clean['currency']} was applied: please check the portfolio choice."))

    # ---- figures that need evidence
    for value_key, evidence_key, kind, label in FIGURES:
        claimed = ext.get(value_key)
        quote = ext.get(evidence_key)
        if _blank(claimed):
            continue
        if _blank(quote) or not quote_is_in_notes(quote, notes):
            flags.append(Flag("unverified", value_key,
                              f"{label}: the model reported '{claimed}' but could not point to those words in the notes. Left blank: please confirm."))
            continue
        ok, reason = _check_figure(kind, claimed, quote)
        if not ok:
            flags.append(Flag("unverified", value_key,
                              f"{label}: the model reported '{claimed}' from \"{quote}\", but {reason}. Left blank: please confirm."))
            continue
        clean[value_key] = claimed
        report.evidence.append({"field": label, "value": str(claimed), "quote": quote})

    # ---- income needs a clear period; the tool field is "per month"
    if "income_amount" in clean:
        if ext.get("income_period") != "monthly":
            flags.append(Flag("unverified", "income_amount",
                              "Income was stated, but not clearly as a monthly amount (the tool's field is per month). Not converted and left blank."))
            del clean["income_amount"]
            report.evidence = [e for e in report.evidence if e["field"] != "Income amount"]

    # ---- bespoke allocation: only when the notes give the percentages themselves
    alloc = ext.get("allocation")
    if alloc:
        rows, problem = [], None
        for row in (alloc if isinstance(alloc, list) else []):
            row = row if isinstance(row, dict) else {}
            pct = _to_decimal(row.get("percent"))
            quote = row.get("evidence")
            if row.get("key") not in ALLOCATION_KEYS or _blank(row.get("class")):
                problem = "an asset class was not recognised"
            elif pct is None or pct < 0 or pct > 100:
                problem = "a percentage was not valid"
            elif _blank(quote) or not quote_is_in_notes(quote, notes):
                problem = f"no supporting words from the notes for {row.get('class')}"
            elif pct not in numbers_in(quote):
                problem = f"the {pct}% for {row.get('class')} is not in the quoted words"
            else:
                rows.append({"class": row["class"], "key": row["key"], "percent": pct, "quote": quote})
            if problem:
                break
        if not problem and (not rows or sum(r["percent"] for r in rows) != 100):
            problem = "the percentages do not add up to 100"
        if problem:
            flags.append(Flag("unverified", "allocation",
                              f"Custom asset allocation not used: {problem}. The model portfolio's own allocation applies."))
        else:
            clean["allocation"] = [{"class": r["class"], "key": r["key"], "percent": r["percent"]} for r in rows]
            for r in rows:
                report.evidence.append({"field": f"Allocation: {r['class']}", "value": f"{r['percent']}%", "quote": r["quote"]})
            flags.append(Flag("assumption", "allocation",
                              "A custom asset allocation from the notes replaces the model portfolio's allocation. "
                              "The portfolio composition text still comes from the model and may no longer match: please review it."))

    # ---- benchmark: free text, but no index may be named unless the notes name it
    bench = ext.get("benchmark")
    bench_quote = ext.get("benchmark_evidence")
    if not _blank(bench):
        proper = {w.lower() for w in re.findall(r"\b[A-Z][A-Za-z]+\b", bench)} - _GENERIC_CAPITALISED
        if _blank(bench_quote) or not quote_is_in_notes(bench_quote, notes):
            flags.append(Flag("unverified", "benchmark",
                              "Benchmark: no supporting quote from the notes. Left blank."))
        elif not proper <= notes_tokens:
            flags.append(Flag("unverified", "benchmark",
                              f"Benchmark: '{bench}' names an index that is not in the notes. Left blank: the notes only say \"{bench_quote}\"."))
        else:
            bench = _tidy_benchmark(bench)
            clean["benchmark"] = bench
            report.evidence.append({"field": "Benchmark", "value": bench, "quote": bench_quote})
            if not proper:  # a category ("a global equity index"), not a named index
                flags.append(Flag("missing", "benchmark",
                                  f"Benchmark: the notes say \"{bench_quote}\" without naming a specific index. "
                                  "Please name the index (for example MSCI World) before sending."))

    # ---- replacement
    if ext.get("is_replacement") is True:
        if "replac" in notes.lower():
            clean["is_replacement"] = True
            if not _blank(ext.get("replacement_details")):
                details = ext["replacement_details"]
                bad = [n for n in numbers_in(details) if n not in notes_numbers]
                if bad:
                    flags.append(Flag("unverified", "replacement_details",
                                      "Replacement details contain a number that is not in the notes. Details left blank."))
                else:
                    clean["replacement_details"] = details
        else:
            flags.append(Flag("unverified", "is_replacement",
                              "The model said this replaces an existing portfolio, but the notes do not say so. Not marked as a replacement."))

    # ---- narrative text: every number in it must come from the notes
    for key, label in (("client_situation", "Client situation"), ("risk_profile", "Risk profile text")):
        text = ext.get(key)
        if _blank(text):
            continue
        bad = [n for n in numbers_in(text) if n not in notes_numbers]
        if bad:
            shown = ", ".join(format(b.normalize(), "f") for b in bad)
            flags.append(Flag("unverified", key,
                              f"{label}: the written summary contains figure(s) not found in the notes ({shown}). Summary left blank."))
        else:
            clean[key] = text.strip()

    # ---- income requirement
    if ext.get("income_required") is False:
        clean["income_required"] = False
        flags.append(Flag("info", "income", "Notes say no income is required."))

    # ---- what is simply missing
    def missing(key, message):
        if key not in clean and not any(f.field == key for f in flags):
            flags.append(Flag("missing", key, message))

    missing("investment_amount", "Investment amount not stated.")
    missing("horizon", "Investment horizon not stated.")
    if ext.get("income_required") is not False:
        missing("income_amount", "Income requirement not stated.")
    missing("adviser_fee_percent", "Adviser fee not stated.")
    missing("target_return", "Target return not stated (left blank, not guessed).")

    report.clean = clean
    return report