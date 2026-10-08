import unittest

from app.checks import verify

GENERIC_NOTES = "The Sithole Family Trust. Benchmark against a global equity index. Use the global equity model."
NAMED_NOTES = "The Sithole Family Trust. Benchmark against the MSCI World Index. Use the global equity model."


def flagged(report, field, level):
    return [f for f in report.flags if f.field == field and f.level == level]


class BenchmarkFlagTests(unittest.TestCase):
    def test_generic_benchmark_is_kept_and_flagged(self):
        report = verify({"client_name": "The Sithole Family Trust", "benchmark": "Global equity index",
                         "benchmark_evidence": "Benchmark against a global equity index"}, GENERIC_NOTES)
        self.assertEqual(report.clean["benchmark"], "Global equity index")
        self.assertTrue(flagged(report, "benchmark", "missing"))

    def test_leading_article_is_dropped_and_first_letter_capitalised(self):
        report = verify({"client_name": "The Sithole Family Trust", "benchmark": "a global equity index",
                         "benchmark_evidence": "Benchmark against a global equity index"}, GENERIC_NOTES)
        self.assertEqual(report.clean["benchmark"], "Global equity index")
        named = verify({"client_name": "X", "benchmark": "the MSCI World Index",
                        "benchmark_evidence": "Benchmark against the MSCI World Index"}, NAMED_NOTES)
        self.assertEqual(named.clean["benchmark"], "MSCI World Index")

    def test_named_benchmark_in_notes_is_not_flagged(self):
        report = verify({"client_name": "The Sithole Family Trust", "benchmark": "MSCI World Index",
                         "benchmark_evidence": "Benchmark against the MSCI World Index"}, NAMED_NOTES)
        self.assertEqual(report.clean["benchmark"], "MSCI World Index")
        self.assertFalse([f for f in report.flags if f.field == "benchmark"])

    def test_invented_index_still_rejected(self):
        report = verify({"client_name": "The Sithole Family Trust", "benchmark": "MSCI World Index (global equity)",
                         "benchmark_evidence": "Benchmark against a global equity index"}, GENERIC_NOTES)
        self.assertNotIn("benchmark", report.clean)
        self.assertTrue(flagged(report, "benchmark", "unverified"))


if __name__ == "__main__":
    unittest.main()