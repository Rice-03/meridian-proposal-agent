import os
import unittest
from unittest import mock

try:
    import flask  # noqa: F401
    HAVE_FLASK = True
except ImportError:  # the pipeline tests do not need Flask
    HAVE_FLASK = False

from app.extract import Result
from app.llm import LLMError


@unittest.skipUnless(HAVE_FLASK, "Flask is not installed (pip install -r requirements.txt)")
class ServerTests(unittest.TestCase):
    def setUp(self):
        from app import server
        self.server = server
        self.client = server.app.test_client()

    def test_status_never_contains_the_key(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "SECRET-KEY-VALUE"}):
            body = self.client.get("/api/status").get_data(as_text=True)
        self.assertIn('"llm_ready":true', body.replace(" ", ""))
        self.assertNotIn("SECRET-KEY-VALUE", body)

    def test_status_without_key(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            data = self.client.get("/api/status").get_json()
        self.assertFalse(data["llm_ready"])

    def test_propose_returns_proposal_flags_and_evidence(self):
        fake = Result(proposal={"clientName": "X"}, flags=[{"level": "missing", "field": "f", "message": "m"}],
                      evidence=[{"field": "a", "value": "1", "quote": "q"}], retried=True, corrections=["c"])
        with mock.patch.object(self.server, "run", return_value=fake) as run:
            res = self.client.post("/api/propose", json={"notes": "some notes"})
        run.assert_called_once_with("some notes")
        data = res.get_json()
        self.assertEqual(res.status_code, 200)
        self.assertEqual(data["proposal"], {"clientName": "X"})
        self.assertTrue(data["retried"])
        self.assertEqual(data["corrections"], ["c"])

    def test_propose_includes_the_ai_notes(self):
        fake = Result(proposal={}, ai_notes=["No amount is given."])
        with mock.patch.object(self.server, "run", return_value=fake):
            data = self.client.post("/api/propose", json={"notes": "n"}).get_json()
        self.assertEqual(data["ai_notes"], ["No amount is given."])

    def test_propose_includes_the_agents_questions_and_the_clarification_header(self):
        fake = Result(proposal={}, flags=[{"level": "unverified", "field": "horizon", "message": "Horizon could not be verified."}])
        asked = [{"field": "horizon", "label": "investment horizon", "question": "What horizon?", "flags": ["Horizon could not be verified."]}]
        with mock.patch.object(self.server, "run", return_value=fake), \
             mock.patch.object(self.server, "ask_questions", return_value=asked) as ask:
            data = self.client.post("/api/propose", json={"notes": "n"}).get_json()
        ask.assert_called_once_with(fake.flags, "n")
        self.assertEqual(data["questions"], asked)
        self.assertIn("PM CLARIFICATION", data["clarification_header"])

    def test_a_failed_question_wording_call_does_not_break_the_proposal(self):
        fake = Result(proposal={"clientName": "X"}, flags=[{"level": "unverified", "field": "horizon", "message": "m"}])
        with mock.patch.object(self.server, "run", return_value=fake), \
             mock.patch("app.followup.generate_json", side_effect=LLMError("quota")):
            res = self.client.post("/api/propose", json={"notes": "n"})
        self.assertEqual(res.status_code, 200)
        q = res.get_json()["questions"]
        self.assertEqual(q[0]["field"], "horizon")
        self.assertFalse(q[0]["from_model"])

    def test_skipped_answers_are_removed_before_the_pipeline_runs(self):
        from app.followup import HEADER
        fake = Result(proposal={}, flags=[])
        notes = "Firm notes.\n\n" + HEADER + "\n- portfolio model: leave blank"
        with mock.patch.object(self.server, "run", return_value=fake) as run, \
             mock.patch.object(self.server, "ask_questions", return_value=[]):
            data = self.client.post("/api/propose", json={"notes": notes}).get_json()
        run.assert_called_once_with("Firm notes.")
        self.assertEqual(data["skipped"], ["portfolio model"])

    def test_llm_errors_become_a_readable_message(self):
        with mock.patch.object(self.server, "run", side_effect=LLMError("GEMINI_API_KEY is not set.")):
            res = self.client.post("/api/propose", json={"notes": "x"})
        self.assertEqual(res.status_code, 400)
        self.assertIn("GEMINI_API_KEY", res.get_json()["error"])

    def test_bad_bodies_are_rejected_cleanly(self):
        for body in ({}, {"notes": 5}, {"notes": ["a"]}):
            res = self.client.post("/api/propose", json=body)
            self.assertEqual(res.status_code, 400, body)
        res = self.client.post("/api/propose", data="not json", content_type="text/plain")
        self.assertEqual(res.status_code, 400)

    def test_page_is_served(self):
        with self.client.get("/") as res:
            self.assertEqual(res.status_code, 200)

    def test_generator_is_served_when_present(self):
        if not (self.server.ROOT / "generator" / self.server.GENERATOR_FILE).exists():
            self.skipTest("generator/challenge-generator.html has not been copied in yet")
        with self.client.get("/generator/challenge-generator.html") as res:
            self.assertEqual(res.status_code, 200)

    def test_cannot_escape_the_generator_folder(self):
        with self.client.get("/generator/../app/server.py") as res:
            self.assertEqual(res.status_code, 404)


if __name__ == "__main__":
    unittest.main()