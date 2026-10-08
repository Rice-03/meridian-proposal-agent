"""Find every number in a piece of text, written as digits or as words.

This is used to check that a figure the LLM reports really appears in the
adviser's notes. It understands, for example:

    "R6.2 million"                   -> 6200000
    "R45k"                           -> 45000
    "R28,000"                        -> 28000
    "6 200 000"                      -> 6200000
    "four and a half million"        -> 4500000
    "zero point seven five"          -> 0.75
    "CPI plus five"                  -> 5
    "0.60%"                          -> 0.60

It deliberately finds nothing in "a few million": that is not a number.
"""

import re
from decimal import Decimal

_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_SCALES = {
    "thousand": Decimal(1_000),
    "million": Decimal(1_000_000),
    "billion": Decimal(1_000_000_000),
}
_DIGIT_WORDS = {k: v for k, v in _UNITS.items() if v <= 9}

_SUFFIX = {
    "k": Decimal(1_000), "thousand": Decimal(1_000),
    "m": Decimal(1_000_000), "mil": Decimal(1_000_000), "million": Decimal(1_000_000),
    "bn": Decimal(1_000_000_000), "billion": Decimal(1_000_000_000),
}

# Either a digit number (optionally with a scale suffix such as m / k / million)
# or a plain word.  Order matters: digits are tried first.
_TOKEN = re.compile(
    r"""
    (?P<digits>
        (?<![\d.,])
        (?P<int>\d{1,3}(?:[ ,]\d{3})+|\d+)
        (?:\.(?P<frac>\d+))?
        (?:\s*(?P<suffix>million|thousand|billion|mil|bn|m|k)\b)?
    )
    |
    (?P<word>[A-Za-z]+)
    """,
    re.VERBOSE | re.IGNORECASE,
)


def _digit_value(match):
    integer = re.sub(r"[ ,]", "", match.group("int"))
    value = Decimal(integer + ("." + match.group("frac") if match.group("frac") else ""))
    suffix = match.group("suffix")
    if suffix:
        value *= _SUFFIX[suffix.lower()]
    return value


def numbers_in(text):
    """Return every number found in `text`, in order, as Decimals."""
    tokens = list(_TOKEN.finditer(text or ""))
    found = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.group("digits"):
            found.append(_digit_value(tok))
            i += 1
            continue

        word = tok.group("word").lower()
        if word not in _UNITS and word not in _TENS:
            i += 1
            continue

        # Build one spoken number: "four and a half million", "zero point seven five"
        total = Decimal(0)
        current = Decimal(0)
        used_scale = False
        used_point = False
        only_the_word_one = word == "one"
        j = i
        while j < len(tokens):
            t = tokens[j]
            if t.group("digits"):
                break
            w = t.group("word").lower()
            if w in _UNITS:
                current += _UNITS[w]
            elif w in _TENS:
                current += _TENS[w]
            elif w == "hundred":
                current = (current or Decimal(1)) * 100
                only_the_word_one = False
            elif w in _SCALES and (current or total):
                total += current * _SCALES[w]
                current = Decimal(0)
                used_scale = True
                only_the_word_one = False
            elif w == "point":
                digits = []
                k = j + 1
                while k < len(tokens) and tokens[k].group("word") and tokens[k].group("word").lower() in _DIGIT_WORDS:
                    digits.append(str(_DIGIT_WORDS[tokens[k].group("word").lower()]))
                    k += 1
                if not digits:
                    break
                current += Decimal("0." + "".join(digits))
                used_point = True
                only_the_word_one = False
                j = k - 1
            elif w == "and":
                # "and a half"  or  "hundred and fifty"
                nxt = tokens[j + 1].group("word").lower() if j + 1 < len(tokens) and tokens[j + 1].group("word") else None
                nxt2 = tokens[j + 2].group("word").lower() if j + 2 < len(tokens) and tokens[j + 2].group("word") else None
                if nxt == "a" and nxt2 == "half":
                    current += Decimal("0.5")
                    only_the_word_one = False
                    j += 2
                elif nxt in _UNITS or nxt in _TENS:
                    pass
                else:
                    break
            else:
                break
            j += 1

        value = total + current
        if not (only_the_word_one and not used_scale and not used_point):
            found.append(value)
        i = max(j, i + 1)
    return found


def normalize(text):
    """Lowercase and keep only letters and digits, so quotes can be compared fairly."""
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()