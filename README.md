# Meridian Proposal Agent

An AI agent that turns a portfolio manager's rough notes (typed, or spoken into the microphone) into a proposal, and loads it into Old Mutual's offline proposal generator through `window.loadProposal(...)`.

Built for the Old Mutual AI analyst programmer stage 2 challenge. All data is synthetic.

**Model and framework:** Google Gemini (`gemini-3.8-flash`, with `gemini-3.1-flash-lite` as an automatic backup) through the `google-genai` Python SDK. The backend is Flask. There is no agent framework: the pipeline is a few plain Python functions, so every step can be read and tested. I built it with Claude Code and tested it myself (148 unit tests).

## What it does

1. The portfolio manager (PM) pastes or dictates notes into the page.
2. Gemini extracts the facts and, for every figure, quotes the exact words it came from.
3. Plain Python code checks each quote really is in the notes and that the number matches the quote. Anything it cannot verify is dropped and flagged. The model never writes the flags.
4. If the notes only hint at something, the agent asks a short follow-up question. Anything the notes simply do not say is listed under "Not in the notes" and is left blank.
5. The verified result is turned into a proposal object (see `proposal-schema.md` in the challenge pack) and loaded into the original generator, which is shown on the same page.

The page has three parts: the notes box, a review panel (questions and flags), and the Old Mutual generator, served unmodified inside the page.

## How to run

You need Python 3.10 or newer and a free Gemini API key from Google AI Studio.

```
python -m venv .venv
.venv\Scripts\activate            (Windows)   or   source .venv/bin/activate   (macOS, Linux)
pip install -r requirements.txt
```

`requirements.txt` includes Playwright, which is only used for the WhatsApp PDF. You do not need to run `python -m playwright install chromium` unless you try WhatsApp.

Set your key in the same window you start the app from:

```
set GEMINI_API_KEY=your_key_here             (Windows Command Prompt)
$env:GEMINI_API_KEY="your_key_here"          (Windows PowerShell)
export GEMINI_API_KEY=your_key_here          (macOS, Linux)
```

Optional check that the key works: `python check_gemini.py`.

Start the app and open http://127.0.0.1:8000:

```
python -m app.server
```

Try it: paste `samples/03-notes-sparse.txt` and click Generate. The other samples are in `samples/`. To run all four through the real model and compare with Old Mutual's expected files: `python run_samples.py`.

Run the tests (no key needed, the model is replaced by a fake): `python -m unittest discover -s tests -t .`

Notes:
- **Live microphone:** click "Record notes" (Chrome or Edge). It uses the browser's built-in speech recognition (language en-ZA), so it needs an internet connection. The same works for the answer boxes of the follow-up questions.
- **Free tier limit:** the free Gemini tier allows about 20 requests per model per day. If the main model is overloaded or out of quota, the agent switches to the backup model. You can change them with `GEMINI_MODEL` and `GEMINI_FALLBACK_MODEL`.
- **Generator proposals:** the generator saves a new proposal in the browser every time one is loaded, so its own proposal list grows with each run. Use a private window for a clean list.
- **The generator's PDF button** loads two libraries from the internet when clicked. Everything else works offline apart from the model call.
- The server only listens on 127.0.0.1. The API key is read from the environment, never sent to the browser, and never stored in the repo.

## Project layout

```
app/server.py        Flask app: the page, the generator, /api/propose
app/llm.py           the only file that talks to a model provider (Gemini). Timeouts, retries, backup model
app/extract.py       the prompt, the output schema, and the end-to-end run (with one correction pass)
app/checks.py        the deterministic verification and the flags
app/followup.py      decides which questions to ask; the model only phrases them
app/to_proposal.py   turns the verified result into the proposal object
app/numparse.py      number parsing helpers ("four and a half million", "R6.2m")
app/whatsapp.py      optional WhatsApp front end (see below)
app/pdf_export.py    makes the proposal PDF on the server, for WhatsApp
static/index.html    the page
generator/           Old Mutual's challenge-generator.html, unmodified
samples/             the four sample notes and expected outputs
tests/               148 unit tests
```

## Design choices and why

Old Mutual asked me to make assumptions and explain why. These are mine.

**1. The model extracts, code verifies.** The brief's rule is "never guess a number for a client". A model that checks its own work can be confidently wrong, so the model has to quote the notes for every figure, and plain code checks the quote is really there. Flags come from this code, not from the model, so the model cannot talk itself past a rule. The model still does real work: extraction, evidence quotes, marking hints as tentative, the client summary, phrasing the follow-up questions, and a correction pass when a check fails. I know a reviewer could prefer model-written flags. My answer is that for money, a repeatable check is safer, and the "what I would do next" section covers adding model-written advice in a separate place.

**2. Ask only when unsure, flag the rest.** The brief says to either ask a follow-up or flag the gaps. Asking about everything that is missing makes the agent annoying, and on a first call the PM often does not have the number yet. So the agent asks only when the notes contain something it had to read into (for example "balanced-ish", which is a hint and not a decision, or a figure it could not verify). Things that are simply not in the notes are listed under "Not in the notes" and left blank. The code decides what to ask, the model only words the question, and at most 8 questions are asked. Answers are added to the notes under a "PM CLARIFICATION" heading, and the model is told that heading is the final word on its topic. A skipped answer ("leave blank", "not sure") changes nothing.

**3. Hedged wording is used but flagged.** A hint such as "balanced-ish" selects the balanced portfolio (otherwise the proposal would be mostly empty), but it is flagged for the PM to confirm. A risk appetite alone ("not a gambler") never selects a portfolio and never becomes a target return.

**4. Nothing is defaulted.** If the notes do not give a value, the key is left out of the proposal object, so the generator keeps its own default. Amounts are stored as digits, income only if it is clearly monthly, and an index is never named unless the notes name it. The firm's own fees for a model (management, brokerage) come from the generator's model settings, not from the notes.

**5. The client's situation is written to two fields.** `proposal-schema.md` documents a `needs` field, but this generator no longer prints a needs page. It prints `goalAssessment.rationale` on the "Goal Assessment & Proposal" page. I fill both, so the text appears whichever one is read. All the code that knows the shape of the proposal is in `app/to_proposal.py`, so if the contract changes, that is the only file to edit.

**6. Missing information stays out of the client's text.** The summary of the client's situation is written about the client only. Gaps ("amount to be confirmed") are shown in the review panel for the PM, not written into text that may be printed for the client. The sample expected file for sample 3 puts a "TO CONFIRM" line inside the needs text. I chose the separate panel instead.

**7. The generator is not modified.** It is served as-is from the same origin, and the page calls `window.loadProposal`. The model choice (`applyModel`) loads the allocation, composition and the model's own objective settings, exactly as picking the model in the generator would. That is why the model shows in both of its dropdowns.

**8. One file for the provider.** Everything that talks to Gemini is in `app/llm.py`. To use another provider, replace `generate_json` there. Nothing else sees the provider, only Python dictionaries.

**9. The notes are data, not instructions.** Text inside the notes that tells the model to do something is ignored.

## Voice notes, WhatsApp, and ambiguity (the bonuses)

- **Voice notes:** live microphone dictation works in the page (Chrome or Edge), and goes through the same pipeline as typed notes. Audio file transcription is also built (`transcribe_audio` in `app/llm.py`, using Gemini) and is used for WhatsApp voice notes. I did not add an upload button to the page.
- **Ambiguity:** described above (design choice 2). `samples/03-notes-sparse.txt` is the test case.
- **WhatsApp:** the front end is built (`app/whatsapp.py`) on the Twilio WhatsApp sandbox. It runs the same pipeline, asks the follow-up questions in the chat, accepts voice notes (transcribed with Gemini, and echoed back so the PM can see what was heard), and replies with a summary and the proposal PDF. Inbound messages work: I confirmed Twilio delivers a message to the app through a tunnel. I could not demo the replies, because a Twilio free trial sender only sends pre-approved templates (error 21654), so the full conversation is covered by tests only, not shown live. It is off unless `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN` and `PUBLIC_BASE_URL` are all set. To try it, also run `python -m playwright install chromium` (used to make the PDF), expose the app with a tunnel (for example `cloudflared tunnel --url http://127.0.0.1:8000`), and set the sandbox's "When a message comes in" URL to `<PUBLIC_BASE_URL>/whatsapp` (POST). Requests are checked with Twilio's signature.

## What I would do next, and what I would change

With more time I would:

1. **Compare models properly, and try Claude.** I chose Gemini first because it has a free tier. The provider is isolated in one function, so trying Claude (and others) is a small change. But I would not switch on a hunch. I would build a bigger set of messy synthetic notes with the right answers, and measure how often each model invents a figure, misses a gap, or asks an unneeded question.
2. **Let the model add advice in a separate place.** Today only code creates flags. I would let the model also add advisory notes (for example "these two statements contradict each other"), shown separately and labelled as AI suggestions, so they are never confused with the checked flags.
3. **Add audio file upload** to the page (the transcription function already exists), and for real client calls use a controlled transcription service. Browser speech recognition sends audio to the browser vendor, which is fine for a demo and not for real client data.
4. **Finish WhatsApp** with an approved sender or message templates (or Meta's own API), so the whole conversation can run live.
5. **Stop the generator's proposal list growing.** Each load creates a new proposal. I would replace the previous one, but that means reaching into the generator's internals, so I chose to document it instead.
6. **Fill more of the proposal**: match the adviser's name against the generator's list of people and practices, and write the opening introduction text from the notes.
7. **Let the PM fix a flagged item in the review panel** before it loads, and keep a record of what was changed.
8. **Treat real data properly.** Before any real client notes are used, check the model provider's data terms, remove personal details before sending, and handle the data in line with POPIA.

What I would change about my approach: I would build the evaluation set first, before the prompts, so I could prove each change helps. I spent time on features (WhatsApp) before measuring the baseline properly. I would also check whether the code checks are too strict, because every false flag costs the PM time.