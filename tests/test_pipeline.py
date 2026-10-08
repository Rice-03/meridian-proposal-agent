"""Tests for the checks and the proposal builder, using the four supplied samples.

Each IDEAL_* dict is what a good LLM should return for that sample. The
adversarial tests then corrupt it the way a careless LLM might, and check the
safeguard catches it.
"""

import copy
import json
import os
import unittest

from app.checks import verify
from app.to_proposal import to_proposal

HERE = os.path.dirname(__file__)
SAMPLES = os.path.join(HERE, "..", "samples")


def notes(name):
    with open(os.path.join(SAMPLES, name), encoding="utf-8") as f:
        return f.read()


def expected(n):
    with open(os.path.join(SAMPLES, f"{n}-expected.json"), encoding="utf-8") as f:
        return json.load(f)


IDEAL_01 = {
    "client_name": "Mr George Whitfield", "client_type": "Individual", "mandate_type": "Discretionary",
    "addressed_to": "client", "currency": "ZAR",
    "model": "income", "model_evidence": "Put him in the Reg 28 bespoke model",
    "investment_amount": "6200000", "investment_amount_evidence": "Just took a lump sum of about R6.2 million",
    "income_required": True, "income_amount": "28000", "income_period": "monthly",
    "income_evidence": "draw an income of roughly R28,000 a month",
    "horizon": "15 years+", "horizon_evidence": "thinks 15+ years",
    "adviser_fee_percent": "0.60", "adviser_fee_evidence": "Adviser fee 0.60%",
    "client_situation": "Mr Whitfield, aged 63, has retired with a lump sum of approximately R6.2 million and needs about R28,000 a month. "
                        "Capital preservation matters most, with some growth to keep up with inflation, and a mostly local, income-oriented mandate.",
    "risk_profile": "Conservative to moderate. Capital preservation is the priority, with some growth to keep pace with inflation.",
}

IDEAL_02 = {
    "client_name": "The Sithole Family Trust", "client_type": "Trust", "addressed_to": "advisor",
    "financial_advisor_name": "Marcus Reid", "currency": "USD",
    "model": "global-growth", "model_evidence": "Use the global equity model",
    "investment_amount": "1200000", "investment_amount_evidence": "invest USD 1.2m offshore",
    "income_required": False,
    "horizon": "10 years+", "horizon_evidence": "Aggressive, 10-year+ view",
    "adviser_fee_percent": "0.50", "adviser_fee_evidence": "Adviser fee 0.50%",
    "benchmark": "Global equity index", "benchmark_evidence": "Benchmark against a global equity index",
    "is_replacement": True,
    "replacement_details": "Replaces an existing offshore unit-trust portfolio that the trust is unwinding.",
    "client_situation": "The trustees want to invest USD 1.2m offshore in a pure global equity mandate with no local exposure and no income, reinvesting for the next generation, with a 10-year-plus view.",
    "risk_profile": "Aggressive, with a ten-year-plus view.",
}

IDEAL_03 = {
    "client_name": "Mrs Lindiwe Petersen", "client_type": "Individual", "addressed_to": "client", "currency": "ZAR",
    "model": "balanced", "model_evidence": "Balanced-ish feel",
    "client_situation": "Mrs Petersen is a new prospect referred by an existing client. She wants to invest a few million rand, is not a gambler, but wants it to grow.",
    "risk_profile": "Balanced, pending confirmation. Not a gambler, but wants growth.",
}

IDEAL_04 = {
    "client_name": "Johannes Coetzee", "client_type": "Individual", "addressed_to": "client", "currency": "ZAR",
    "model": "balanced", "model_evidence": "Let's go with the flexible bespoke portfolio I think",
    "investment_amount": "4500000", "investment_amount_evidence": "call it four and a half million rand",
    "income_required": False,
    "horizon": "10 years+", "horizon_evidence": "easily ten years plus before they touch it",
    "target_return": "CPI + 5%", "target_return_evidence": "let's say CPI plus five",
    "adviser_fee_percent": "0.75", "adviser_fee_evidence": "Adviser fee the usual zero point seven five",
    "client_situation": "Johannes and Retha Coetzee, both around 50 and still working, have about R4.5 million of growth capital and need no income for now. They want a meaningful offshore allocation.",
    "risk_profile": "Moderate to aggressive. Experienced investors with a long horizon.",
}

SAMPLES_IDEAL = [
    ("01", "01-notes-retiree-income.txt", IDEAL_01),
    ("02", "02-notes-growth-offshore.txt", IDEAL_02),
    ("03", "03-notes-sparse.txt", IDEAL_03),
    ("04", "04-voice-note-transcript.txt", IDEAL_04),
]


def fields_flagged(report, level=None):
    return {f.field for f in report.flags if level is None or f.level == level}


class IdealExtractions(unittest.TestCase):
    def test_no_figure_is_rejected_for_a_good_extraction(self):
        for n, fname, ext in SAMPLES_IDEAL:
            with self.subTest(sample=n):
                report = verify(ext, notes(fname))
                self.assertEqual(fields_flagged(report, "unverified"), set(), [f.message for f in report.flags])

    def test_matches_the_expected_scalar_fields(self):
        for n, fname, ext in SAMPLES_IDEAL:
            with self.subTest(sample=n):
                got = to_proposal(verify(ext, notes(fname)).clean)
                exp = expected(n)
                for key in ("clientName", "introGreetTo", "applyModel", "financialPlanner"):
                    if key in exp:
                        self.assertEqual(got.get(key), exp[key], key)
                self.assertEqual(got.get("fees", {}).get("advisorFee"), exp.get("fees", {}).get("advisorFee"))
                self.assertEqual(got.get("objective", {}).get("targetReturn"), exp.get("objective", {}).get("targetReturn"))
                exp_kf = exp.get("goalAssessment", {}).get("keyFigures", {})
                got_kf = got.get("goalAssessment", {}).get("keyFigures", {})
                self.assertEqual(got_kf, exp_kf)
                # and the same amounts sit where the tool actually reads them
                self.assertEqual(got.get("strategy", {}).get("keyFigures", {}), exp_kf)
                self.assertEqual(got.get("replacements", {}).get("isReplacement"), exp.get("replacements", {}).get("isReplacement"))

    def test_sparse_sample_leaves_every_figure_out_and_flags_it(self):
        report = verify(IDEAL_03, notes("03-notes-sparse.txt"))
        got = to_proposal(report.clean)
        self.assertNotIn("keyFigures", got.get("goalAssessment", {}))
        self.assertNotIn("keyFigures", got.get("strategy", {}))
        self.assertNotIn("fees", got)
        self.assertNotIn("targetReturn", got.get("objective", {}))
        for f in ("investment_amount", "horizon", "income_amount", "adviser_fee_percent", "target_return"):
            self.assertIn(f, fields_flagged(report, "missing"), f)

    def test_no_income_needed_is_not_flagged_as_missing(self):
        report = verify(IDEAL_04, notes("04-voice-note-transcript.txt"))
        self.assertNotIn("income_amount", fields_flagged(report, "missing"))

    def test_target_return_left_out_when_not_stated(self):
        got = to_proposal(verify(IDEAL_01, notes("01-notes-retiree-income.txt")).clean)
        self.assertNotIn("targetReturn", got.get("objective", {}))

    def test_nothing_sent_that_is_unknown(self):
        got = to_proposal(verify(IDEAL_03, notes("03-notes-sparse.txt")).clean)

        def walk(x):
            if isinstance(x, dict):
                for v in x.values():
                    walk(v)
            else:
                self.assertNotIn(x, ("", None))
        walk(got)


class CarelessLLM(unittest.TestCase):
    def run_with(self, base, fname, **changes):
        ext = copy.deepcopy(base)
        ext.update(changes)
        return verify(ext, notes(fname))

    def test_invented_target_return_is_dropped(self):
        r = self.run_with(IDEAL_01, "01-notes-retiree-income.txt", target_return="CPI + 5%",
                          target_return_evidence="Says capital preservation matters most")
        self.assertIn("target_return", fields_flagged(r, "unverified"))
        self.assertNotIn("target_return", r.clean)

    def test_target_return_with_made_up_quote_is_dropped(self):
        r = self.run_with(IDEAL_01, "01-notes-retiree-income.txt", target_return="CPI + 5%",
                          target_return_evidence="target return of CPI plus five")
        self.assertIn("target_return", fields_flagged(r, "unverified"))

    def test_wrong_amount_is_dropped(self):
        r = self.run_with(IDEAL_01, "01-notes-retiree-income.txt", investment_amount="6500000")
        self.assertIn("investment_amount", fields_flagged(r, "unverified"))
        self.assertNotIn("investment_amount", r.clean)

    def test_amount_guessed_from_vague_words_is_dropped(self):
        r = self.run_with(IDEAL_03, "03-notes-sparse.txt", investment_amount="3000000",
                          investment_amount_evidence="wants to invest \"a few million, rand\"")
        self.assertIn("investment_amount", fields_flagged(r, "unverified"))
        self.assertNotIn("investment_amount", r.clean)

    def test_amount_with_no_evidence_is_dropped(self):
        r = self.run_with(IDEAL_01, "01-notes-retiree-income.txt", investment_amount_evidence=None)
        self.assertIn("investment_amount", fields_flagged(r, "unverified"))

    def test_horizon_invented_for_sparse_notes(self):
        r = self.run_with(IDEAL_03, "03-notes-sparse.txt", horizon="5 years+", horizon_evidence="we'll pin those down at the next meeting")
        self.assertIn("horizon", fields_flagged(r, "unverified"))

    def test_named_index_not_in_notes_is_dropped(self):
        r = self.run_with(IDEAL_02, "02-notes-growth-offshore.txt", benchmark="MSCI World Index (global equity)",
                          benchmark_evidence="Benchmark against a global equity index")
        self.assertIn("benchmark", fields_flagged(r, "unverified"))
        self.assertNotIn("benchmark", r.clean)

    def test_invented_number_in_summary_removes_the_summary(self):
        r = self.run_with(IDEAL_01, "01-notes-retiree-income.txt",
                          client_situation="Mr Whitfield has R7 million and wants R30,000 a month.")
        self.assertIn("client_situation", fields_flagged(r, "unverified"))
        self.assertNotIn("client_situation", r.clean)

    def test_wrong_client_name_is_dropped(self):
        r = self.run_with(IDEAL_01, "01-notes-retiree-income.txt", client_name="Mr John Smith")
        self.assertIn("client_name", fields_flagged(r, "unverified"))

    def test_invented_replacement_is_dropped(self):
        r = self.run_with(IDEAL_01, "01-notes-retiree-income.txt", is_replacement=True, replacement_details="Replaces an old portfolio.")
        self.assertIn("is_replacement", fields_flagged(r, "unverified"))
        self.assertNotIn("is_replacement", r.clean)

    def test_annual_income_is_not_converted(self):
        r = self.run_with(IDEAL_01, "01-notes-retiree-income.txt", income_period="annual")
        self.assertNotIn("income_amount", r.clean)
        self.assertIn("income_amount", fields_flagged(r, "unverified"))

    def test_model_without_a_quote_is_not_applied(self):
        r = self.run_with(IDEAL_01, "01-notes-retiree-income.txt", model_evidence="")
        self.assertNotIn("model", r.clean)

    def test_unknown_model_is_not_applied(self):
        r = self.run_with(IDEAL_01, "01-notes-retiree-income.txt", model="aggressive-plus")
        self.assertNotIn("model", r.clean)
        self.assertIn("model", fields_flagged(r, "missing"))

    def test_mandate_not_stated_is_left_out(self):
        r = self.run_with(IDEAL_02, "02-notes-growth-offshore.txt", mandate_type="Discretionary")
        self.assertNotIn("mandate_type", r.clean)
        self.assertIn("mandate_type", fields_flagged(r, "missing"))

    def test_advisor_greeting_needs_a_confirmed_name(self):
        r = self.run_with(IDEAL_02, "02-notes-growth-offshore.txt", financial_advisor_name="Jane Doe")
        self.assertNotIn("addressed_to", r.clean)
        got = to_proposal(r.clean)
        self.assertEqual(got["introGreetTo"], "client")

    def test_currency_conflict_with_model_is_flagged(self):
        r = self.run_with(IDEAL_02, "02-notes-growth-offshore.txt", model="balanced", model_evidence="Use the global equity model")
        self.assertIn("currency", fields_flagged(r, "assumption"))

    def test_garbage_input_does_not_crash(self):
        for ext in (None, {}, {"investment_amount": "abc", "investment_amount_evidence": "xyz"}, {"model": None}):
            r = verify(ext, "some notes")
            self.assertIsInstance(to_proposal(r.clean), dict)


if __name__ == "__main__":
    unittest.main()