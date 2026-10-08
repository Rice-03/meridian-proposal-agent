import os
import unittest
from unittest import mock

from app import llm
from app.llm import LLMError, _is_daily_quota, describe_failure, generate_json

DAILY = ("429 RESOURCE_EXHAUSTED. You exceeded your current quota ... "
         "'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier' ... limit: 20, model: gemini-3.8-flash")
PER_MINUTE = "429 RESOURCE_EXHAUSTED. 'quotaId': 'GenerateRequestsPerMinutePerProjectPerModel-FreeTier'"


class DescribeFailureTests(unittest.TestCase):
    def test_daily_limit_is_recognised_and_explained(self):
        self.assertTrue(_is_daily_quota(DAILY))
        msg = describe_failure(DAILY, "gemini-3.8-flash")
        self.assertIn("daily request limit", msg)
        self.assertIn("GEMINI_MODEL", msg)
        self.assertNotIn("RESOURCE_EXHAUSTED", msg)

    def test_per_minute_limit_is_not_called_daily(self):
        self.assertFalse(_is_daily_quota(PER_MINUTE))
        self.assertIn("per minute", describe_failure(PER_MINUTE, "m"))

    def test_other_errors_keep_the_original_text(self):
        msg = describe_failure("404 model not found", "bad-model")
        self.assertIn("bad-model", msg)
        self.assertIn("404 model not found", msg)


OVERLOADED = "503 UNAVAILABLE. {'error': {'message': 'This model is currently experiencing high demand.'}}"


class _Reply:
    def __init__(self, text):
        self.text = text


class FakeClient:
    """Stands in for google.genai.Client. `plan` maps model name -> list of replies or errors."""
    def __init__(self, plan):
        self.plan, self.calls = plan, []
        self.models = self

    def generate_content(self, model, contents, config):
        self.calls.append(model)
        step = self.plan[model].pop(0)
        if isinstance(step, Exception):
            raise step
        return _Reply(step)


class FallbackTests(unittest.TestCase):
    def run_with(self, plan, env=None):
        client = FakeClient(plan)
        env = dict({"GEMINI_API_KEY": "k"}, **(env or {}))
        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch("google.genai.Client", return_value=client), \
             mock.patch.object(llm.time, "sleep"):
            try:
                return generate_json("s", "u", {}), client.calls
            except LLMError as exc:
                return exc, client.calls

    def test_an_overloaded_main_model_falls_back_to_the_backup(self):
        out, calls = self.run_with({"gemini-3.8-flash": [Exception(OVERLOADED)] * 2, "gemini-3.1-flash-lite": ['{"a": 1}']})
        self.assertEqual(out, {"a": 1})
        self.assertEqual(calls, ["gemini-3.8-flash", "gemini-3.8-flash", "gemini-3.1-flash-lite"])

    def test_a_timeout_counts_as_temporary(self):
        out, calls = self.run_with({"gemini-3.8-flash": [Exception("timed out"), '{"a": 2}']})
        self.assertEqual(out, {"a": 2})
        self.assertEqual(calls, ["gemini-3.8-flash"] * 2)

    def test_daily_quota_on_the_main_model_goes_straight_to_the_backup(self):
        out, calls = self.run_with({"gemini-3.8-flash": [Exception(DAILY)], "gemini-3.1-flash-lite": ['{"a": 3}']})
        self.assertEqual(out, {"a": 3})
        self.assertEqual(calls, ["gemini-3.8-flash", "gemini-3.1-flash-lite"])

    def test_a_bad_key_does_not_try_another_model(self):
        out, calls = self.run_with({"gemini-3.8-flash": [Exception("403 PERMISSION_DENIED API key not valid")]})
        self.assertIsInstance(out, LLMError)
        self.assertEqual(calls, ["gemini-3.8-flash"])

    def test_both_models_failing_gives_one_readable_error(self):
        out, calls = self.run_with({"gemini-3.8-flash": [Exception(OVERLOADED)] * 2, "gemini-3.1-flash-lite": [Exception(OVERLOADED)] * 2})
        self.assertIsInstance(out, LLMError)
        self.assertIn("gemini-3.1-flash-lite", str(out))
        self.assertEqual(len(calls), 4)

    def test_the_backup_can_be_turned_off(self):
        out, calls = self.run_with({"gemini-3.8-flash": [Exception(OVERLOADED)] * 2}, env={"GEMINI_FALLBACK_MODEL": ""})
        self.assertIsInstance(out, LLMError)
        self.assertEqual(calls, ["gemini-3.8-flash"] * 2)

    def test_the_main_model_is_not_repeated_as_its_own_backup(self):
        out, calls = self.run_with({"gemini-3.1-flash-lite": [Exception(OVERLOADED)] * 2}, env={"GEMINI_MODEL": "gemini-3.1-flash-lite"})
        self.assertIsInstance(out, LLMError)
        self.assertEqual(calls, ["gemini-3.1-flash-lite"] * 2)


if __name__ == "__main__":
    unittest.main()