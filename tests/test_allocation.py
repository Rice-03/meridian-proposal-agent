"""Bespoke allocation: only used when the notes give the percentages themselves."""

import copy
import unittest

from app.checks import verify
from app.to_proposal import to_proposal

NOTES = (
    "Notes: Ms Thandi Mokoena (invented client), Individual, discretionary. ZAR. "
    "Wants it split 60% international equity, 30% SA fixed interest and 10% cash. "
    "Adviser fee 0.70%."
)

BASE = {
    "client_name": "Ms Thandi Mokoena", "client_type": "Individual", "mandate_type": "discretionary",
    "currency": "ZAR", "addressed_to": "client",
    "adviser_fee_percent": "0.70", "adviser_fee_evidence": "Adviser fee 0.70%",
    "allocation": [
        {"class": "International Equity", "key": "intl-equity", "percent": 60, "evidence": "60% international equity"},
        {"class": "SA Fixed Interest", "key": "fixed-interest", "percent": 30, "evidence": "30% SA fixed interest"},
        {"class": "Cash", "key": "cash", "percent": 10, "evidence": "10% cash"},
    ],
}


def flagged(report, level):
    return {f.field for f in report.flags if f.level == level}


class BespokeAllocation(unittest.TestCase):
    def test_stated_allocation_is_used(self):
        r = verify(BASE, NOTES)
        self.assertEqual(flagged(r, "unverified"), set())
        got = to_proposal(r.clean)
        self.assertEqual(
            got["strategy"]["allocation"],
            [{"class": "International Equity", "key": "intl-equity", "percent": 60},
             {"class": "SA Fixed Interest", "key": "fixed-interest", "percent": 30},
             {"class": "Cash", "key": "cash", "percent": 10}],
        )

    def test_it_is_flagged_so_the_pm_reviews_the_composition(self):
        r = verify(BASE, NOTES)
        self.assertIn("allocation", flagged(r, "assumption"))

    def test_percentages_not_adding_to_100_are_rejected(self):
        ext = copy.deepcopy(BASE)
        ext["allocation"][2]["percent"] = 15
        ext["allocation"][2]["evidence"] = "10% cash"
        r = verify(ext, NOTES)
        self.assertNotIn("allocation", r.clean)
        self.assertIn("allocation", flagged(r, "unverified"))

    def test_a_percentage_not_in_its_quote_is_rejected(self):
        ext = copy.deepcopy(BASE)
        ext["allocation"][0]["percent"] = 65
        r = verify(ext, NOTES)
        self.assertNotIn("allocation", r.clean)

    def test_a_made_up_quote_is_rejected(self):
        ext = copy.deepcopy(BASE)
        ext["allocation"][1]["evidence"] = "30% SA bonds please"
        r = verify(ext, NOTES)
        self.assertNotIn("allocation", r.clean)

    def test_unknown_asset_class_key_is_rejected(self):
        ext = copy.deepcopy(BASE)
        ext["allocation"][0]["key"] = "crypto"
        r = verify(ext, NOTES)
        self.assertNotIn("allocation", r.clean)

    def test_allocation_invented_from_vague_notes_is_rejected(self):
        notes = "Wants a balanced mix of local and offshore. Risk-averse."
        ext = {"allocation": [
            {"class": "International Equity", "key": "intl-equity", "percent": 50, "evidence": "a balanced mix of local and offshore"},
            {"class": "SA Equity", "key": "sa-equity", "percent": 50, "evidence": "a balanced mix of local and offshore"}]}
        r = verify(ext, notes)
        self.assertNotIn("allocation", r.clean)
        self.assertNotIn("strategy", to_proposal(r.clean))

    def test_malformed_allocation_does_not_crash(self):
        for bad in ("60/40", [1, 2], [None], [{}], {"a": 1}):
            r = verify({"allocation": bad}, NOTES)
            self.assertNotIn("allocation", r.clean)
            to_proposal(r.clean)

    def test_no_allocation_means_no_strategy_allocation_key(self):
        ext = copy.deepcopy(BASE)
        del ext["allocation"]
        self.assertNotIn("allocation", to_proposal(verify(ext, NOTES).clean).get("strategy", {}))


if __name__ == "__main__":
    unittest.main()