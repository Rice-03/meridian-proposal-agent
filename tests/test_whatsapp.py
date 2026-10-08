import unittest
from pathlib import Path
from unittest import mock

from flask import Flask

from app import whatsapp
from app.extract import Result
from app.followup import HEADER
from app.llm import LLMError
from app.pdf_export import PdfExportError

ENV = {"TWILIO_ACCOUNT_SID": "AC123", "TWILIO_AUTH_TOKEN": "secret", "PUBLIC_BASE_URL": "https://abc.example.app/"}
SENDER = "whatsapp:+27821234567"
URL = "https://abc.example.app/whatsapp"


def flag(level, field, message):
    return {"level": level, "field": field, "message": message}


def question(label="portfolio model", text="Which model?", flags=("hint",)):
    return {"field": "model", "label": label, "question": text, "flags": list(flags)}


class ImmediateExecutor:
    def submit(self, fn, *args):
        fn(*args)


class ConfigTests(unittest.TestCase):
    def test_off_unless_all_three_settings_exist(self):
        self.assertIsNone(whatsapp.config_from_env({}))
        self.assertIsNone(whatsapp.config_from_env({"TWILIO_ACCOUNT_SID": "a", "TWILIO_AUTH_TOKEN": "b"}))

    def test_settings_are_read_and_tidied(self):
        cfg = whatsapp.config_from_env(dict(ENV, WHATSAPP_ALLOWED=" +27821234567, whatsapp:+27829999999 ", PORT="9000"))
        self.assertEqual(cfg.public_url, "https://abc.example.app")
        self.assertEqual(cfg.sender, whatsapp.DEFAULT_SENDER)
        self.assertEqual(cfg.local_url, "http://127.0.0.1:9000")
        self.assertEqual(cfg.allowed, ("whatsapp:+27821234567", "whatsapp:+27829999999"))

    def test_the_real_server_has_no_whatsapp_route_without_settings(self):
        from app import server
        rules = {r.rule for r in server.app.url_map.iter_rules()}
        self.assertNotIn("/whatsapp", rules)


class SignatureTests(unittest.TestCase):
    PARAMS = {"From": SENDER, "Body": "Mrs Petersen, R6.2 million", "NumMedia": "0"}

    def test_a_signature_checks_out_and_a_changed_message_does_not(self):
        sig = whatsapp.twilio_signature("secret", URL, self.PARAMS)
        self.assertTrue(whatsapp.valid_signature("secret", URL, self.PARAMS, sig))
        self.assertFalse(whatsapp.valid_signature("secret", URL, dict(self.PARAMS, Body="R9 million"), sig))
        self.assertFalse(whatsapp.valid_signature("other", URL, self.PARAMS, sig))
        self.assertFalse(whatsapp.valid_signature("secret", URL, self.PARAMS, ""))

    def test_matches_twilios_own_library_when_it_is_installed(self):
        try:
            from twilio.request_validator import RequestValidator
        except ImportError:
            self.skipTest("twilio package not installed (only used as a referee here)")
        params = {"From": SENDER, "Body": "a & b = c 100% ünï", "NumMedia": "1"}
        self.assertEqual(whatsapp.twilio_signature("secret", URL, params), RequestValidator("secret").compute_signature(URL, params))


class AnswerTests(unittest.TestCase):
    QS = [question("portfolio model"), question("benchmark", "Which index?")]

    def test_numbered_answers(self):
        self.assertEqual(whatsapp.parse_answers("1: balanced\n2) MSCI World", self.QS), {0: "balanced", 1: "MSCI World"})

    def test_a_plain_reply_answers_a_single_open_question(self):
        self.assertEqual(whatsapp.parse_answers("the balanced one", [question()]), {0: "the balanced one"})

    def test_a_plain_reply_with_several_questions_is_not_guessed(self):
        self.assertEqual(whatsapp.parse_answers("the balanced one", self.QS), {})

    def test_a_number_that_is_not_a_question_is_ignored(self):
        self.assertEqual(whatsapp.parse_answers("7: balanced", self.QS), {})

    def test_answers_go_under_the_same_heading_the_web_page_uses(self):
        notes = whatsapp.add_answers("Mrs P. Balanced-ish.", self.QS, {0: "balanced"})
        self.assertEqual(notes, "Mrs P. Balanced-ish.\n\n" + HEADER + "\n- portfolio model: balanced")

    def test_a_second_answer_replaces_the_first_and_the_heading_is_not_repeated(self):
        notes = whatsapp.add_answers("Notes.", self.QS, {0: "income"})
        notes = whatsapp.add_answers(notes, self.QS, {0: "balanced", 1: "MSCI World"})
        self.assertEqual(notes.count(HEADER), 1)
        self.assertEqual(notes.count("portfolio model"), 1)
        self.assertIn("- portfolio model: balanced", notes)
        self.assertIn("- benchmark: MSCI World", notes)


class SummaryTests(unittest.TestCase):
    PROPOSAL = {
        "clientName": "Mrs Lindiwe Petersen", "applyModel": "balanced",
        "goalAssessment": {"clientType": "Individual", "currency": "ZAR"},
        "strategy": {"currency": "ZAR", "keyFigures": {"totalInvestment": "6200000", "incomeValue": "28000"}},
        "objective": {"investmentHorizon": "15 years+"}, "fees": {"advisorFee": "0.60"},
    }

    def test_found_things_gaps_and_questions_are_all_there(self):
        flags = [flag("missing", "mandate_type", "Mandate type not stated."), flag("assumption", "model", "hint"),
                 flag("unverified", "benchmark", "Benchmark could not be verified.")]
        text = whatsapp.summary_text(self.PROPOSAL, flags, [question()])
        self.assertIn("Draft proposal for Mrs Lindiwe Petersen", text)
        self.assertIn("Model: Balanced", text)
        self.assertIn("Amount: R 6 200 000", text)
        self.assertIn("Income per month: R 28 000", text)
        self.assertIn("Adviser fee: 0.60%", text)
        self.assertIn("- Mandate type not stated.", text)
        self.assertIn("- Benchmark could not be verified.", text)
        self.assertIn("1. Which model?", text)
        self.assertNotIn("- hint", text)          # the flag that is being asked about is not listed twice

    def test_nothing_is_listed_as_found_that_the_agent_did_not_find(self):
        text = whatsapp.summary_text({"clientName": "Mrs P"}, [], [])
        self.assertIn("I could not find any figures", text)
        for label in ("Amount", "Model", "Horizon", "Adviser fee", "Target return"):
            self.assertNotIn(label + ":", text)

    def test_skipped_answers_are_reported(self):
        self.assertIn("Skipped, no change made: portfolio model.", whatsapp.summary_text(self.PROPOSAL, [], [], ["portfolio model"]))

    def test_it_never_exceeds_the_message_limit(self):
        flags = [flag("missing", "x", "gap " * 100) for _ in range(30)]
        self.assertLessEqual(len(whatsapp.summary_text(self.PROPOSAL, flags, [])), whatsapp.MAX_BODY)

    def test_other_currencies_keep_their_code(self):
        p = dict(self.PROPOSAL, goalAssessment={"currency": "USD"}, strategy={"keyFigures": {"totalInvestment": "1200000"}})
        self.assertIn("Amount: USD 1 200 000", whatsapp.summary_text(p, [], []))


class WebhookFlowTests(unittest.TestCase):
    """The whole conversation, with the model, Twilio and the PDF maker replaced by fakes."""

    def setUp(self):
        whatsapp._SESSIONS.clear()
        self.app = Flask(__name__)
        self.app.config["TESTING"] = True
        self.cfg = whatsapp.register(self.app, ENV)
        self.client = self.app.test_client()
        self.sent = []
        self.run_calls = []
        self.questions = [question()]
        result = Result(proposal={"clientName": "Mrs P", "applyModel": "balanced"}, flags=[flag("assumption", "model", "hint")])

        def fake_run(notes):
            self.run_calls.append(notes)
            return result

        def fake_pdf(proposal, base, dest):
            Path(dest).write_bytes(b"%PDF-1.4 fake")
            return dest

        self.patches = [
            mock.patch.object(whatsapp, "EXECUTOR", ImmediateExecutor()),
            mock.patch.object(whatsapp, "send_message", side_effect=lambda cfg, to, body=None, media_url=None: self.sent.append((to, body, media_url))),
            mock.patch.object(whatsapp, "run", side_effect=fake_run),
            mock.patch.object(whatsapp, "ask", side_effect=lambda flags, notes: list(self.questions)),
            mock.patch.object(whatsapp, "export_pdf", side_effect=fake_pdf),
        ]
        for p in self.patches:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in self.patches])

    def post(self, body="", sign=True, **extra):
        params = dict({"From": SENDER, "Body": body, "NumMedia": "0"}, **extra)
        sig = whatsapp.twilio_signature("secret", URL, params) if sign else "bad"
        return self.client.post("/whatsapp", data=params, headers={"X-Twilio-Signature": sig})

    def texts(self):
        return [b for _to, b, _m in self.sent if b]

    def test_a_forged_request_is_refused_and_nothing_runs(self):
        self.assertEqual(self.post("notes", sign=False).status_code, 403)
        self.assertEqual(self.run_calls, [])

    def test_notes_in_summary_and_questions_out_and_the_pdf_waits(self):
        res = self.post("Mrs P wants a balanced-ish portfolio")
        self.assertEqual(res.status_code, 200)
        self.assertIn("<Response>", res.get_data(as_text=True))
        self.assertEqual(self.run_calls, ["Mrs P wants a balanced-ish portfolio"])
        self.assertIn("reading the notes", self.texts()[0])
        self.assertIn("Draft proposal for Mrs P", self.texts()[1])
        self.assertIn("1. Which model?", self.texts()[1])
        self.assertIn('reply "pdf"', self.texts()[1])
        self.assertEqual(len(self.sent), 2)                      # no PDF while a question is open
        self.assertFalse(any(m for _t, _b, m in self.sent))

    def test_with_no_open_question_the_pdf_follows_the_summary(self):
        self.questions = []
        self.post("Mrs P, firm notes")
        self.assertIn("The PDF follows in a moment.", self.texts()[1])
        to, body, media = self.sent[-1]
        self.assertTrue(media.startswith("https://abc.example.app/media/") and media.endswith(".pdf"))

    def test_the_pdf_is_sent_once_the_answers_clear_the_questions(self):
        self.post("Mrs P wants a balanced-ish portfolio")
        self.questions = []                       # after the answer the agent is no longer unsure
        self.post("1: balanced")
        self.assertTrue(self.sent[-1][2].endswith(".pdf"))

    def test_replying_pdf_gives_the_draft_without_running_the_model_again(self):
        self.post("Mrs P wants a balanced-ish portfolio")
        runs = len(self.run_calls)
        self.post("PDF")
        self.assertEqual(len(self.run_calls), runs)
        self.assertTrue(self.sent[-1][2].endswith(".pdf"))

    def test_pdf_before_any_notes_says_there_is_nothing_to_send(self):
        self.post("pdf")
        self.assertIn("nothing to send yet", self.texts()[-1])
        self.assertFalse(any(m for _t, _b, m in self.sent))

    def test_the_pdf_link_serves_the_file_and_nothing_else(self):
        self.questions = []
        self.post("notes")
        name = self.sent[-1][2].rsplit("/", 1)[1]
        self.assertEqual(self.client.get("/media/" + name).data, b"%PDF-1.4 fake")
        self.assertEqual(self.client.get("/media/" + name).mimetype, "application/pdf")
        self.assertEqual(self.client.get("/media/../../etc/passwd").status_code, 404)
        self.assertEqual(self.client.get("/media/" + "0" * 32 + ".pdf").status_code, 404)

    def test_answering_a_question_adds_it_to_the_notes_and_runs_again(self):
        self.post("Mrs P wants a balanced-ish portfolio")
        self.post("1: balanced")
        self.assertEqual(len(self.run_calls), 2)
        self.assertEqual(self.run_calls[1], "Mrs P wants a balanced-ish portfolio\n\n" + HEADER + "\n- portfolio model: balanced")

    def test_a_skipped_answer_does_not_change_the_notes(self):
        self.post("Firm notes")
        self.post("1: leave blank")
        self.assertEqual(self.run_calls[1], "Firm notes")
        self.assertIn("Skipped, no change made: portfolio model.", self.texts()[-1])

    def test_a_short_reply_that_answers_nothing_is_not_run(self):
        with mock.patch.object(whatsapp, "ask", return_value=[question(), question("benchmark")]):
            self.post("Firm notes")
            runs = len(self.run_calls)
            self.post("hmm")
        self.assertEqual(len(self.run_calls), runs)
        self.assertIn("did not see which question", self.texts()[-1])

    def test_new_starts_over(self):
        self.post("First notes")
        self.post("new Second client notes")
        self.assertEqual(self.run_calls[-1], "Second client notes")
        self.post("new")
        self.assertIn("Send the new notes", self.texts()[-1])

    def test_long_messages_are_new_notes_even_while_a_question_is_open(self):
        self.post("First notes")
        long = "Another client. " * 30
        self.post(long)
        self.assertEqual(self.run_calls[-1], long.strip())

    def test_a_voice_note_is_transcribed_shown_back_and_used(self):
        with mock.patch.object(whatsapp, "download_media", return_value=b"audio") as dl, \
             mock.patch.object(whatsapp, "transcribe_audio", return_value="Mrs P wants R2 million") as tr:
            self.post("", NumMedia="1", MediaUrl0="https://api.twilio.com/media/1", MediaContentType0="audio/ogg")
        dl.assert_called_once()
        tr.assert_called_once_with(b"audio", "audio/ogg")
        self.assertIn('I heard: "Mrs P wants R2 million"', self.texts()[0])
        self.assertEqual(self.run_calls, ["Mrs P wants R2 million"])

    def test_a_pdf_failure_still_leaves_the_summary_and_says_so(self):
        self.questions = []
        with mock.patch.object(whatsapp, "export_pdf", side_effect=PdfExportError("Playwright is not installed.")):
            self.post("notes")
        self.assertIn("Draft proposal for Mrs P", self.texts()[1])
        self.assertIn("Playwright is not installed.", self.texts()[-1])

    def test_a_model_error_is_passed_on_in_plain_words(self):
        with mock.patch.object(whatsapp, "run", side_effect=LLMError("The daily request limit has been reached.")):
            self.post("notes")
        self.assertEqual(self.texts()[-1], "The daily request limit has been reached.")

    def test_a_surprise_error_never_crashes_the_worker_or_leaks_details(self):
        with mock.patch.object(whatsapp, "run", side_effect=ZeroDivisionError("secret detail")):
            self.post("notes")
        self.assertNotIn("secret detail", self.texts()[-1])
        self.assertIn("something went wrong", self.texts()[-1])

    def test_empty_messages_and_greetings_get_the_help_text(self):
        self.post("")
        self.post("Hi")
        self.assertTrue(all("Send me the meeting notes" in t for t in self.texts()))
        self.assertEqual(self.run_calls, [])

    def test_only_allowed_numbers_are_served_when_a_list_is_set(self):
        app = Flask(__name__)
        whatsapp.register(app, dict(ENV, WHATSAPP_ALLOWED="+27820000000"))
        params = {"From": SENDER, "Body": "notes", "NumMedia": "0"}
        res = app.test_client().post("/whatsapp", data=params, headers={"X-Twilio-Signature": whatsapp.twilio_signature("secret", URL, params)})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(self.run_calls, [])

    def test_a_tunnelled_request_can_only_reach_the_whatsapp_routes(self):
        @self.app.get("/api/private")
        def private():
            return "key-using endpoint"
        self.assertEqual(self.client.get("/api/private").status_code, 200)                                  # local: fine
        self.assertEqual(self.client.get("/api/private", headers={"X-Forwarded-For": "1.2.3.4"}).status_code, 404)
        self.assertEqual(self.client.get("/api/private", headers={"Cf-Connecting-Ip": "1.2.3.4"}).status_code, 404)
        self.assertEqual(self.client.get("/media/" + "0" * 32 + ".pdf", headers={"X-Forwarded-For": "1.2.3.4"}).status_code, 404)  # reaches the route (file missing)

    def test_media_is_only_fetched_from_twilio(self):
        with self.assertRaises(whatsapp.WhatsAppError):
            whatsapp.download_media(self.cfg, "https://evil.example.com/a.ogg")


if __name__ == "__main__":
    unittest.main()