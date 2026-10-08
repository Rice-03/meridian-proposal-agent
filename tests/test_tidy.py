import unittest

from app.checks import verify

NOTES = "Trust, USD 1.2m offshore. Aggressive, 10-year+ view. Use the global equity model. Adviser fee 0.50%."


class TidyTests(unittest.TestCase):
    def test_stray_quote_marks_are_removed(self):
        ext = {
            "client_name": "Trust",
            "horizon": '10-year+"',
            "horizon_evidence": "10-year+ view",
        }
        report = verify(ext, NOTES)
        self.assertEqual(report.clean["horizon"], "10-year+")

    def test_curly_quotes_and_spaces_removed(self):
        ext = {"client_name": " \u201cTrust\u201d ", "horizon": "10-year+", "horizon_evidence": "10-year+ view"}
        self.assertEqual(verify(ext, NOTES).clean["client_name"], "Trust")


if __name__ == "__main__":
    unittest.main()