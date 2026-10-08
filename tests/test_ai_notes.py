"""The model's own notes for the PM ("Noticed by the AI"): shown, but checked by code first."""

import copy
import unittest

from app import extract
from app.extract import MAX_AI_NOTES, clean_ai_notes
from tests.test_extract import FakeLLM
from tests.test_pipeline import IDEAL_01, notes

NOTES_01 = notes("01-notes-retiree-income.txt")      # mentions R6.2 million, R28,000, 15+ years, 0.60%
NOTES_03 = notes("03-notes-sparse.txt")              # "a few million", no figures


class CleanAiNotes(unittest.TestCase):
    def test_plain_notes_are_kept(self):
        raw = ["No investment amount is given, only 'a few million'.", "The time horizon is not stated."]
        self.assertEqual(clean_ai_notes(raw, NOTES_03), raw)

    def test_a_number_that_is_not_in_the_notes_drops_the_note(self):
        raw = ["The amount is probably around R3 million.", "No horizon is given."]
        self.assertEqual(clean_ai_notes(raw, NOTES_03), ["No horizon is given."])

    def test_a_number_that_is_in_the_notes_is_fine(self):
        raw = ["The notes give R28,000 a month but no start date."]
        self.assertEqual(clean_ai_notes(raw, NOTES_01), raw)

    def test_numbers_written_differently_still_match(self):
        raw = ["He wants about R6.2 million, which the notes call a lump sum."]
        self.assertEqual(clean_ai_notes(raw, NOTES_01), raw)

    def test_not_a_list_or_wrong_types_give_nothing(self):
        for raw in (None, "No amount", {"a": 1}, 5):
            self.assertEqual(clean_ai_notes(raw, NOTES_03), [])
        self.assertEqual(clean_ai_notes([None, 3, ["x"], "Fine."], NOTES_03), ["Fine."])

    def test_blank_long_and_repeated_notes_are_dropped(self):
        raw = ["", "   ", "x" * 500, "Horizon missing.", "horizon MISSING."]
        self.assertEqual(clean_ai_notes(raw, NOTES_03), ["Horizon missing."])

    def test_at_most_five(self):
        raw = [f"Observation number word {w}." for w in "abcdefgh"]
        self.assertEqual(len(clean_ai_notes(raw, NOTES_03)), MAX_AI_NOTES)

    def test_whitespace_is_tidied(self):
        self.assertEqual(clean_ai_notes(["No   amount\n given."], NOTES_03), ["No amount given."])


class InThePipeline(unittest.TestCase):
    def test_result_carries_cleaned_notes(self):
        answer = copy.deepcopy(IDEAL_01)
        answer["ai_notes"] = ["The notes do not say when the income starts.", "He may also want R9 million invested."]
        result = extract.run(NOTES_01, llm=FakeLLM(answer))
        self.assertEqual(result.ai_notes, ["The notes do not say when the income starts."])

    def test_missing_ai_notes_means_empty_list(self):
        result = extract.run(NOTES_01, llm=FakeLLM(IDEAL_01))
        self.assertEqual(result.ai_notes, [])

    def test_ai_notes_never_reach_the_proposal_or_the_flags(self):
        answer = copy.deepcopy(IDEAL_01)
        answer["ai_notes"] = ["Unusual wording about risk."]
        plain = extract.run(NOTES_01, llm=FakeLLM(IDEAL_01))
        with_notes = extract.run(NOTES_01, llm=FakeLLM(answer))
        self.assertEqual(plain.proposal, with_notes.proposal)
        self.assertEqual(plain.flags, with_notes.flags)

    def test_notes_come_from_the_answer_that_was_used_after_a_correction(self):
        bad = copy.deepcopy(IDEAL_01)
        bad["investment_amount"] = "9000000"            # not backed by the notes, so a correction is requested
        bad["ai_notes"] = ["From the first, rejected answer."]
        good = copy.deepcopy(IDEAL_01)
        good["ai_notes"] = ["From the corrected answer."]
        result = extract.run(NOTES_01, llm=FakeLLM(bad, good))
        self.assertTrue(result.retried)
        self.assertEqual(result.ai_notes, ["From the corrected answer."])

    def test_schema_and_prompt_ask_for_it(self):
        self.assertIn("ai_notes", extract.EXTRACTION_SCHEMA["properties"])
        self.assertIn("ai_notes", extract.SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()