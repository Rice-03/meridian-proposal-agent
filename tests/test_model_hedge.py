import unittest

from app.checks import verify

NOTES = "Mrs Petersen. Balanced-ish feel. Individual."
FIRM = "Mr Whitfield. Put him in the Reg 28 bespoke model."


def model_flag(report):
    return [f for f in report.flags if f.field == "model" and f.level == "assumption"]


class HedgedModelTests(unittest.TestCase):
    def test_hedged_wording_gets_the_stronger_flag_and_is_applied(self):
        r = verify({"client_name": "Mrs Petersen", "model": "balanced", "model_evidence": "Balanced-ish feel"}, NOTES)
        self.assertEqual(r.clean["model"], "balanced")
        self.assertIn("only hint", model_flag(r)[0].message)

    def test_model_can_mark_it_tentative_itself(self):
        r = verify({"client_name": "Mr Whitfield", "model": "income", "model_evidence": "Reg 28 bespoke model",
                    "model_is_tentative": True}, FIRM)
        self.assertIn("only hint", model_flag(r)[0].message)

    def test_firm_wording_gets_the_normal_flag(self):
        r = verify({"client_name": "Mr Whitfield", "model": "income", "model_evidence": "Reg 28 bespoke model",
                    "model_is_tentative": False}, FIRM)
        self.assertNotIn("only hint", model_flag(r)[0].message)
        self.assertIn("Please confirm", model_flag(r)[0].message)

    def test_quote_not_in_notes_still_rejected(self):
        r = verify({"client_name": "Mrs Petersen", "model": "balanced", "model_evidence": "balanced mandate please"}, NOTES)
        self.assertNotIn("model", r.clean)


if __name__ == "__main__":
    unittest.main()