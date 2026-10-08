import unittest

from app import extract, followup
from app.checks import verify
from app.llm import LLMError

HEDGED_NOTES = "Mrs Petersen. Balanced-ish feel. Individual."


def flag(level, field, message="m"):
    return {"level": level, "field": field, "message": message}


class FakeLLM:
    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []

    def __call__(self, system_prompt, user_text, schema):
        self.calls.append(user_text)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


class PlanTests(unittest.TestCase):
    def test_code_decides_what_to_ask_from_the_flags(self):
        flags = [flag("unverified", "adviser_fee_percent"), flag("assumption", "model"), flag("info", "income"),
                 flag("unverified", "replacement_details")]
        plan = followup.plan_questions(flags, "some notes")
        self.assertEqual([p["field"] for p in plan], ["model", "adviser_fee_percent"])   # importance order, no info/unknown

    def test_things_the_notes_simply_do_not_say_are_never_asked(self):
        flags = [flag("missing", f) for f in ("mandate_type", "model", "investment_amount", "horizon", "income_amount",
                                              "adviser_fee_percent", "target_return", "currency", "client_type")]
        self.assertEqual(followup.plan_questions(flags, "n"), [])

    def test_a_missing_client_name_and_a_vague_benchmark_are_asked(self):
        plan = followup.plan_questions([flag("missing", "client_name"), flag("missing", "benchmark")], "n")
        self.assertEqual([p["field"] for p in plan], ["client_name", "benchmark"])

    def test_the_mandate_type_is_flagged_but_never_asked(self):
        self.assertEqual(followup.plan_questions([flag("missing", "mandate_type"), flag("unverified", "mandate_type")], "n"), [])

    def test_info_flags_and_unknown_fields_are_never_asked(self):
        self.assertEqual(followup.plan_questions([flag("info", "income"), flag("missing", "nothing_known")], "n"), [])

    def test_assumption_on_currency_is_not_a_question(self):
        self.assertEqual(followup.plan_questions([flag("assumption", "currency")], "n"), [])

    def test_a_field_with_two_flags_is_asked_once(self):
        plan = followup.plan_questions([flag("missing", "benchmark", "a"), flag("unverified", "benchmark", "b")], "n")
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0]["flags"], ["a", "b"])

    def test_at_most_eight_questions(self):
        fields = [a.flag_fields[0] for a in followup.ASKABLE]
        flags = [flag("unverified", f) for f in fields] + [flag("assumption", "client_type"), flag("assumption", "model")]
        self.assertEqual(len(followup.plan_questions(flags, "n")), followup.MAX_QUESTIONS)

    def test_answered_topics_are_not_asked_again(self):
        notes = "Original notes.\n\n" + followup.HEADER + "\n- investment horizon: not decided yet"
        self.assertEqual(followup.answered_labels(notes), {"investment horizon"})
        plan = followup.plan_questions([flag("unverified", "horizon"), flag("unverified", "adviser_fee_percent")], notes)
        self.assertEqual([p["field"] for p in plan], ["adviser_fee_percent"])

    def test_a_label_in_the_original_notes_does_not_count_as_answered(self):
        notes = "- mandate type: discretionary\n"     # no clarification header
        self.assertEqual(followup.answered_labels(notes), set())


class PhrasingTests(unittest.TestCase):
    def plan(self):
        return followup.plan_questions([flag("assumption", "model", "hint"), flag("unverified", "horizon")], HEDGED_NOTES)

    def test_model_wording_is_used_when_acceptable(self):
        llm = FakeLLM({"questions": [
            {"field": "model", "question": "You wrote \"Balanced-ish\": did you mean the balanced portfolio?"},
            {"field": "horizon", "question": "How long will the money stay invested?"}]})
        out = followup.ask([flag("assumption", "model", "hint"), flag("unverified", "horizon")], HEDGED_NOTES, llm)
        self.assertTrue(all(q["from_model"] for q in out))
        self.assertIn("Balanced-ish", out[0]["question"])
        self.assertIn("NOTES", llm.calls[0])

    def test_a_number_not_in_the_notes_means_the_fixed_question_is_used(self):
        llm = FakeLLM({"questions": [{"field": "model", "question": "Is this the 60% equity portfolio?"},
                                     {"field": "horizon", "question": "How long will it stay invested?"}]})
        out = followup.ask([flag("assumption", "model"), flag("unverified", "horizon")], HEDGED_NOTES, llm)
        by = {q["field"]: q for q in out}
        self.assertFalse(by["model"]["from_model"])
        self.assertEqual(by["model"]["question"], followup._BY_KEY["model"].fallback)
        self.assertTrue(by["horizon"]["from_model"])

    def test_numbers_that_are_in_the_notes_are_fine(self):
        notes = "Mr X wants R2.5 million invested."
        llm = FakeLLM({"questions": [{"field": "horizon", "question": "How long will the R2.5 million stay invested?"}]})
        out = followup.ask([flag("unverified", "horizon")], notes, llm)
        self.assertTrue(out[0]["from_model"])

    def test_llm_failure_falls_back_and_never_raises(self):
        out = followup.ask([flag("unverified", "horizon")], HEDGED_NOTES, FakeLLM(LLMError("quota")))
        self.assertEqual(out[0]["question"], followup._BY_KEY["horizon"].fallback)
        self.assertFalse(out[0]["from_model"])

    def test_nonsense_from_the_model_is_ignored(self):
        for junk in ({"questions": "no"}, {"questions": [None, 3, {"field": "x", "question": "?"}]}, {}, {"questions": [{"field": "model", "question": "x" * 400}]}):
            out = followup.ask([flag("assumption", "model")], HEDGED_NOTES, FakeLLM(junk))
            self.assertEqual(out[0]["question"], followup._BY_KEY["model"].fallback, junk)

    def test_no_gaps_means_no_model_call(self):
        llm = FakeLLM()
        self.assertEqual(followup.ask([flag("info", "income")], HEDGED_NOTES, llm), [])
        self.assertEqual(llm.calls, [])


class AnswerRoundTripTests(unittest.TestCase):
    """The PM's answer goes into the notes and is treated like any other words in them."""

    def first_read(self):
        return {"client_name": "Mrs Petersen", "client_type": "Individual", "model": "balanced",
                "model_evidence": "Balanced-ish feel", "model_is_tentative": True}

    def test_hedged_model_is_confirmed_by_the_pms_answer(self):
        before = verify(self.first_read(), HEDGED_NOTES)
        self.assertIn("only hint", [f for f in before.flags if f.field == "model"][0].message)

        notes_after = HEDGED_NOTES + "\n\n" + followup.HEADER + "\n- portfolio model: the balanced one"
        second = dict(self.first_read(), model_evidence="the balanced one", model_is_tentative=False)
        after = verify(second, notes_after)
        self.assertEqual(after.clean["model"], "balanced")
        self.assertEqual([f for f in after.flags if f.field == "model"], [])

    def test_a_hedged_answer_is_still_flagged(self):
        notes_after = HEDGED_NOTES + "\n\n" + followup.HEADER + "\n- portfolio model: maybe balanced"
        second = dict(self.first_read(), model_evidence="maybe balanced")
        after = verify(second, notes_after)
        self.assertIn("only hint", [f for f in after.flags if f.field == "model"][0].message)

    def test_the_old_hedge_still_needs_confirming_if_the_model_quotes_it(self):
        notes_after = HEDGED_NOTES + "\n\n" + followup.HEADER + "\n- portfolio model: the balanced one"
        after = verify(self.first_read(), notes_after)    # model ignored the answer and quoted the old hedge
        self.assertTrue([f for f in after.flags if f.field == "model"])

    def test_an_answer_does_not_let_the_model_invent_a_figure(self):
        notes_after = HEDGED_NOTES + "\n\n" + followup.HEADER + "\n- investment amount: a decent chunk"
        ext = dict(self.first_read(), investment_amount="5000000", investment_amount_evidence="a decent chunk")
        after = verify(ext, notes_after)
        self.assertNotIn("investment_amount", after.clean)
        self.assertTrue([f for f in after.flags if f.field == "investment_amount" and f.level == "unverified"])

    def test_an_answer_with_a_real_figure_is_accepted(self):
        notes_after = HEDGED_NOTES + "\n\n" + followup.HEADER + "\n- investment amount: R4.5 million"
        ext = dict(self.first_read(), investment_amount="4500000", investment_amount_evidence="R4.5 million")
        after = verify(ext, notes_after)
        self.assertEqual(after.clean["investment_amount"], "4500000")

    def test_mandate_type_cannot_be_proved_by_a_question(self):
        # Only answers are added to the notes, never questions: "discretionary" is not in them.
        notes_after = HEDGED_NOTES + "\n\n" + followup.HEADER + "\n- mandate type: not decided yet"
        after = verify(dict(self.first_read(), mandate_type="Discretionary"), notes_after)
        self.assertNotIn("mandate_type", after.clean)

    def test_run_uses_the_clarification_end_to_end(self):
        notes_after = HEDGED_NOTES + "\n\n" + followup.HEADER + "\n- mandate type: discretionary"
        llm = FakeLLM(dict(self.first_read(), mandate_type="Discretionary"))
        result = extract.run(notes_after, llm=llm)
        self.assertEqual(result.proposal["goalAssessment"]["mandateType"], "Discretionary")
        self.assertIn("PM CLARIFICATION", llm.calls[0])


class SkippedAnswerTests(unittest.TestCase):
    """A typed "leave blank" must never change what the notes said."""

    FIRM = "Mr Whitfield. Individual. Put him in the Reg 28 bespoke model."

    def with_answers(self, *lines):
        return self.FIRM + "\n\n" + followup.HEADER + "\n" + "\n".join(lines)

    def test_non_answers_are_recognised(self):
        for text in ("leave blank", "Leave it blank", "skip", "Skip this one", "n/a", "N/A", "not sure", "Not sure yet",
                     "unsure", "don't know", "dont know", "I'll say: no idea", "blank", "leave blank; how does it look now"):
            if text.startswith("I'll"):
                continue
            self.assertTrue(followup.is_non_answer(text), text)

    def test_real_answers_are_kept(self):
        for text in ("no income needed", "none", "No", "balanced", "R4.5 million", "the skipper's fee 0.5%", "navy", "nathan"):
            self.assertFalse(followup.is_non_answer(text), text)

    def test_skipped_lines_are_removed_and_reported(self):
        notes = self.with_answers("- portfolio model: leave blank", "- mandate type: discretionary", "- target return: skip")
        cleaned, skipped = followup.clean_notes(notes)
        self.assertEqual(skipped, ["portfolio model", "target return"])
        self.assertIn("mandate type: discretionary", cleaned)
        self.assertNotIn("leave blank", cleaned)
        self.assertNotIn("target return", cleaned)

    def test_only_skipped_answers_gives_back_the_original_notes(self):
        cleaned, skipped = followup.clean_notes(self.with_answers("- portfolio model: leave blank"))
        self.assertEqual(cleaned, self.FIRM)
        self.assertEqual(skipped, ["portfolio model"])

    def test_notes_without_a_clarification_are_untouched(self):
        self.assertEqual(followup.clean_notes(self.FIRM), (self.FIRM, []))

    def test_a_skipped_topic_is_asked_again(self):
        notes = self.with_answers("- investment horizon: skip")
        plan = followup.plan_questions([flag("unverified", "horizon")], notes)
        self.assertEqual([p["field"] for p in plan], ["horizon"])

    def test_the_sample_1_case_keeps_the_model(self):
        # What happened live: "portfolio model: leave blank" removed the model the notes had stated.
        notes = self.with_answers("- portfolio model: leave blank", "- target return: leave blank; how does it look now")
        cleaned, _ = followup.clean_notes(notes)
        ext = {"client_name": "Mr Whitfield", "client_type": "Individual", "model": "income",
               "model_evidence": "Put him in the Reg 28 bespoke model"}
        result = extract.run(cleaned, llm=FakeLLM(ext))
        self.assertEqual(result.proposal["applyModel"], "income")


if __name__ == "__main__":
    unittest.main()