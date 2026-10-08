"""Run the four sample notes through the real model and compare the key fields
with Old Mutual's expected files.

    python run_samples.py          (all four)
    python run_samples.py 3        (just sample 3)

A difference is not automatically a mistake: the expected files contain some values the
notes never state (for example a target return on sample 3). Where we differ on purpose,
this script says so.
"""

import json
import sys
from pathlib import Path

from app.extract import run
from app.llm import LLMError

HERE = Path(__file__).parent / "samples"

FIELDS = [
    ("clientName", ("clientName",)),
    ("introGreetTo", ("introGreetTo",)),
    ("financialPlanner", ("financialPlanner",)),
    ("applyModel", ("applyModel",)),
    ("clientType", ("goalAssessment", "clientType")),
    ("mandateType", ("goalAssessment", "mandateType")),
    ("currency", ("goalAssessment", "currency")),
    ("totalInvestment", ("strategy", "keyFigures", "totalInvestment")),
    ("incomeValue", ("strategy", "keyFigures", "incomeValue")),
    ("targetReturn", ("objective", "targetReturn")),
    ("investmentHorizon", ("objective", "investmentHorizon")),
    ("benchmark", ("objective", "benchmark")),
    ("advisorFee", ("fees", "advisorFee")),
]


def dig(obj, path):
    for part in path:
        if not isinstance(obj, dict) or part not in obj:
            return None
        obj = obj[part]
    return obj


def expected_for(number, expected):
    """Expected files keep amounts under goalAssessment.keyFigures."""
    if number and dig(expected, ("strategy", "keyFigures")) is None:
        kf = dig(expected, ("goalAssessment", "keyFigures"))
        if kf:
            expected = dict(expected)
            expected["strategy"] = {**expected.get("strategy", {}), "keyFigures": kf}
    return expected


def same(a, b):
    if a is None or b is None:
        return a is None and b is None
    return str(a).strip().lower() == str(b).strip().lower()


def main(argv, llm=None):
    wanted = set(argv[1:])
    notes_files = sorted(HERE.glob("0*-*.txt"))
    for notes_path in notes_files:
        number = notes_path.name[:2]
        if wanted and number.lstrip("0") not in wanted and number not in wanted:
            continue
        expected_files = sorted(HERE.glob(f"{number}*expected*.json"))
        if not expected_files:
            print(f"Sample {number}: no expected file found in samples/, skipping.\n")
            continue
        expected = expected_for(number, json.loads(expected_files[0].read_text(encoding="utf-8")))
        print("=" * 78)
        print(f"Sample {number}: {notes_path.name}")
        print("=" * 78)
        try:
            result = run(notes_path.read_text(encoding="utf-8")) if llm is None else run(notes_path.read_text(encoding="utf-8"), llm=llm)
        except LLMError as exc:
            print(f"FAILED: {exc}\n")
            continue

        print(f"{'field':<20}{'ours':<34}{'expected file':<34}")
        for label, path in FIELDS:
            ours, theirs = dig(result.proposal, path), dig(expected, path)
            if ours is None and theirs is None:
                continue
            mark = "  " if same(ours, theirs) else "<>"
            print(f"{mark}{label:<18}{str(ours)[:32]:<34}{str(theirs)[:32]:<34}")

        print("\nFlags for the PM:")
        for f in result.flags:
            print(f"  [{f['level']}] {f['message']}")
        print(f"\n(correction request used: {result.retried})")
        for message in result.corrections:
            print(f"  first answer problem: {message}")
        print()


if __name__ == "__main__":
    main(sys.argv)