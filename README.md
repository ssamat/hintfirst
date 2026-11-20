# Hint First, Then Explain

A source-grounded AI tutor for one open textbook unit: OpenStax *Psychology 2e*, Chapter 6 "Learning" (CC BY 4.0). Built for CS6460 Educational Technology at Georgia Tech. Parts of this project were built with help from Claude.

The app has three roles:

1. **Ask the reading.** The learner's own questions get a short, direct answer from the chapter: at most three sentences, each cited `[6.x ¶N]`, ending with one question that sends them back to the text. Each paragraph has a 💬 button that scopes the chat to that paragraph and offers *Explain simply*, *Give an example*, and *Quiz me on this*. Every citation is a link: clicking it switches section if needed, highlights the paragraph, and scrolls to it. **📌 Save to notes** pins any answer under the paragraph it cites, highlighted in blue, and the 📌 button in the header lists them all. The chat remembers the session: a repeated question brings back the earlier answer (with **🔄 Explain it differently** for a new angle), new answers can link to earlier questions, and a new section opens with a one-line recap of what came before. Each new section also opens with a **quick review** question: an earlier practice question the learner missed or got right while unsure, with the options reshuffled.
2. **Practice questions.** The chapter's 16 Review Questions, opened per section with **Mark section done**. The learner answers first, rates their confidence (1–5), then sees the result. The help ladder (hint → bigger hint → worked answer) opens only after a first try or "I'm stuck". The first hint names the key distinction the question turns on (e.g., "Is something being added or taken away?"); the bigger hint says which sentence to look at. After the worked answer, **Explain it back** asks for one sentence on why the learner's wrong pick is wrong (or why the answer is right) and gives feedback. If a chat question copies, rewords, asks outright for, or lists the options of an open practice question, the chat sends the learner back to practice instead of answering. When a section's practice is done, a summary card appears right under the last question: questions worth another look (each linked to its paragraph), questions the learner got right while unsure, and, if confidence ran ahead of accuracy, one paragraph to reread.
3. **Assessments** (pre/post tests) stay separate and never touch the tutor.

**Layout.** The screen keeps to the reading, the chat, and practice. Everything else is one click away: ⚙️ holds "How this works" and toggles for the extras (quick review, explain it back, recap, notes in the reading, and a Sources box under answers, which is off by default because citations are links). Settings last for the session.

## Quick start

```bash
pip install -r requirements-dev.txt          # app + extraction + tests

# 1. Extract the chapter and its Review Questions from the PDF
python extract_unit.py --pdf Psychology2e_WEB.pdf        # writes data/unit.md (one chunk per paragraph)
python extract_practice.py --pdf Psychology2e_WEB.pdf    # writes data/practice.json

# 2. Add your Gemini key
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then edit

# 3. Run
streamlit run app.py
```

Without an API key the app still runs: chat answers quote the best-matching sentences of the top paragraph, practice hints point to the source paragraph, and worked answers quote it. `data/unit.md` and `data/practice.json` are included, so step 1 is only needed if you change the extraction.

## Practice questions

`extract_practice.py` parses the Review Questions from PDF pages 220–222 (book pages 208–209) and matches each one to its supporting paragraph with the tutor's retriever (one hand-checked override: rq7 → 6.2 ¶18). **This PDF has no answer key**, so the answers come from the chapter text and are stored with `"answer_verified": false`. Confirm them against the instructor answer key, then list the verified question numbers in `VERIFIED` in `extract_practice.py`.

Put the pre/post-test items in `data/assessments.json` (a JSON list of `{"id", "question", "options"}`). The extractor then drops any practice item that duplicates one (char n-gram cosine ≥ 0.8), and `tests/test_tutor.py::test_practice_does_not_duplicate_assessments` fails if a duplicate gets through. Until that file exists, the test is skipped.

## Admin view

Open `http://localhost:8501/?admin=1` for the Gemini toggle, safeguard details, relevance scores, per-section metrics, and the log download.
Set `ADMIN_MODE = "true"` in Secrets to make admin the default for a deployment. Learners get no sidebar.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI: consent, header (progress, 📌 notes, ⚙️ settings), reading with 💬 per paragraph, chat, practice with confidence, help ladder and explain-it-back, summary card, spaced review, admin view |
| `cite_router/index.html` | Hidden component that turns clicks on citation links into a jump to that paragraph |
| `audit.py` | Audits (answer key, citation support, hint leaks, prompt injection, privacy, contrast); writes `results/audit_report.md` |
| `tutor.py` | Retrieval, decline gate, practice redirect, prompts, citation and leak checks, fallbacks, Gemini client |
| `extract_unit.py` | Builds `data/unit.md` from the PDF (PyMuPDF; drops captions, tables, link boxes, objectives; keeps feature-box examples) |
| `extract_practice.py` | Builds `data/practice.json` from the Review Questions; drops duplicates of assessment items |
| `eval_gold_set.py` | Gold-set, redirect, and practice checks; writes `results/gold_results.csv` |
| `tests/` | `test_tutor.py` (logic) and `test_app.py` (end-to-end UI with a fake Gemini); no key needed |
| `data/gold_set.csv` | 30 gold chat questions (20 in-scope, 10 tricky) |
| `data/practice.json` | 16 practice items: `{id, section, question, options, answer_index, source_paragraph, distinction}`; `distinction` is the hand-written first hint (`KEY_DISTINCTIONS` in `extract_practice.py`) |
| `data/practice_overlap_set.csv` | Labeled questions for tuning the practice redirect threshold |
| `data/paraphrase_set.csv` | Looser rewordings of practice questions, tagged by `author` (`draft` = mine; add peers' rows with their own tag) |
| `data/cached_answers.json` | Demo fallback answers used when the API is unavailable |

`requirements.txt` holds only what the deployed app needs; `requirements-dev.txt` adds PyMuPDF and pytest.

## Evaluation

```bash
python eval_gold_set.py                                   # gates, retrieval, redirect, practice data; no key needed
python eval_gold_set.py --generate     # key from env or secrets.toml; also generates answers and practice hints
python eval_gold_set.py --generate --write-cache   # saves good answers as demo fallbacks
```

Results on the real unit:

| Check | Result |
|---|---|
| Chat behavior (answer / decline / clarify) | 30 / 30 |
| Right section in top 3 | 24 / 24 |
| Decline margin | lowest answerable 0.142 vs highest off-topic 0.123 (threshold 0.135) |
| Practice copies and close rewordings redirected | 31 / 32 |
| Ordinary questions wrongly redirected | 0 / 38 |
| Redirect margin | lowest copy 0.603 vs highest ordinary question 0.636 (threshold 0.65) |
| Looser rewordings redirected (`paraphrase_set.csv`, offline) | 3 / 16 (char match catches 1, "names 3+ options" catches 2) |
| *With `--generate` (gemini-3.8-flash):* | |
| Chat behavior | 29 / 30 (Gemini declined #12, an apply-it-to-a-phone question; after adding an "apply the definition to the learner's example" rule it was answered 3 / 3 times in a spot check) |
| Direct answers with every sentence cited | 23 / 23 |
| Within 3 sentences as written / ended with a connecting question | 23 / 23 / 23 / 23 |
| Citations only to retrieved paragraphs | 23 / 23 |
| Practice copies and close rewordings redirected (with the Gemini rewording check) | 32 / 32 |
| Ordinary questions wrongly redirected (with the Gemini check) | 1 / 38 ("Stop hinting and give me the definition of extinction" → rq7, whose answer *is* that definition; arguably right) |
| Looser rewordings redirected (with the Gemini check) | 14 / 16 (misses score 0.19 and 0.29, below the 0.3 floor, so Gemini never sees them) |
| Level-1 practice hints passing leak + names-the-answer checks | 16 / 16, each naming the item's key distinction |
| Level-2 practice hints passing the same checks | 15 / 16 (rq4: Gemini said out of scope, because its support is Table 6.1, which the extractor drops; it fell back to an extractive pointer). An earlier wording that asked for "the sentence that settles it" dropped this to 12 / 16 by quoting it. |

The decline gate uses the best character-level match (`Retriever.relevance`), threshold 0.135, plus a two-word-phrase match and, for scores between 0.08 and 0.135, a Gemini scope check. The redirect threshold (0.65) misses one loose rewording ("in which type of learning does the stimulus occur before the behavior…", 0.603) rather than redirecting the gold question "Is taking away a phone positive or negative punishment?" (0.636). It catches copies, not ideas: "What are the four steps of modeling?" scores 0.42 against "Which is the correct order of steps in the modeling process?" and is answered. Re-check both margins if you add questions.

Rewordings get two more layers (`Tutor.practice_redirect`): a question that names 3 or more of an item's options is redirected offline (2 is not enough: gold questions like "positive or negative punishment?" name 2), and with a key, any question scoring between 0.3 and 0.65 goes to a Gemini "same question, reworded?" check that is told to say no to questions about the topic in general. Live, it still answers "What is shaping?", "Why are slot machines so hard to stop playing?", and "What did the Little Albert study show?", but it now redirects two same-idea questions while their practice items are open: "What are the four steps of modeling?" (→ rq15) and "What did Bandura's Bobo doll study find?" (→ rq14). Both are labeled `either` in the overlap set; once the item is finished, they're answered.

## Evaluation plan (Milestone 2)

1. **Outside-written test items.** Two or three classmates each write five tricky chat questions and five rewordings of practice questions, without seeing the code. Add them to `data/gold_set.csv` and `data/paraphrase_set.csv` with their own `author` tag; `eval_gold_set.py` reports each author separately, so my own items and theirs can be compared.
2. **Automated checks** above, re-run on the combined set.
3. **Small pilot.** Three to five learners work through one section; logs (with consent) show hint use, repeats, saved notes, and calibration gaps, followed by a short survey on whether the summary card felt like guidance.

## Audits

```bash
python audit.py              # offline parts
python audit.py --generate   # also the Gemini parts (~130 calls)
```

Findings (live run, `results/audit_report.md`):

| Audit | Result | What changed |
|---|---|---|
| Answer key | 16 / 16 supported by their source paragraph, and Gemini, given only that paragraph, picks the same option for all 16 | Nothing. Still not the official key (the PDF has none), so `VERIFIED` stays empty until checked against the instructor key |
| Citation support | 47 / 56 cited sentences judged supported by the paragraph they cite (mean word overlap 91%) | Not fixed yet. Most misses are inference beyond the paragraph ("highly addictive") or a citation to a neighboring paragraph |
| Hint leaks | Level 1: 15 / 16 (rq2's hint said "result of experience" and fell back to the hand-written distinction). Level 2: 16 / 16. No sentence-counting references | The leak check now matches word forms ("suck" = "sucking") and catches hints with ⅔ of the answer's words. The bigger-hint prompt no longer counts sentences |
| Prompt injection | 6 / 6 | Asking outright for "the answer to the practice question…" is now redirected (offline it had leaked the answer) |
| Privacy | Emails and phone numbers scrubbed from logged questions and reflections; years like 1920-1921 kept. No personal details in the existing logs | Consent text says questions are stored word for word |
| Contrast | All text on the highlights passes AA in light and dark themes. The old link blue failed in dark mode (3.7:1) | Links keep the text color with a blue underline, plus an aria-label and focus outline |
| Data gaps | `data/assessments.json` missing; rq4's support is in a dropped table | Open |

## Tests

```bash
pytest -q
```

AppTest reads `.streamlit/secrets.toml`, so the UI tests set their own `GEMINI_API_KEY` and `ADMIN_MODE` and never use a real key.

## Design rationale (course grounding)

| Feature | Course principle |
|---|---|
| Direct, cited answers to the learner's own questions | Strong guidance for novices (Kirschner, Sweller, and Clark): a novice who asks gets the answer, not a puzzle. The worked-example effect (Sweller, cognitive load theory, Module 1): studying a worked solution beats unguided problem solving early on. The closing question keeps the learner in the text. |
| Practice: answer first, then hint → bigger hint → worked answer | The testing effect (Roediger and Karpicke, Module 5): retrieval practice strengthens memory more than rereading. Productive help-seeking (Bastani et al., Module 2; help-seeking, Module 1): help arrives after an attempt, at the practice stage, where struggling pays off, rather than in the chat. |
| Chat won't answer an open practice question | Keeps the testing effect intact: retrieval only helps if the learner does the retrieving. |
| 💬 per paragraph and quick actions next to it | Less split attention (Module 0) and less extraneous load (Sweller, Module 1): the question, the paragraph, and the answer stay in one place. |
| Progress strip | Make thinking visible: progress indicators (Module 1). |
| Confidence before the result, and the summary card | Accurate self-monitoring and self-regulated learning (Module 1): "You felt X/5 sure and got Y of Z right" shows the gap between feeling and knowing. The card names what to revisit *and* where the learner was underconfident, so it reads as guidance rather than a grade (Sadler, Module 5: feedback has to show how to close the gap). It appears right under the last question, so nobody has to scroll for it. |
| First hint names the key distinction | Help-seeking (Module 2): a targeted first hint makes the next step clear without taking it for the learner. |
| Save to notes, highlighted in the reading | Less split attention (Module 0): the learner's own notes sit next to the paragraph they came from. |
| Chat memory (repeat detection, recap, linked answers) | Building on prior knowledge: answers connect to what the learner already asked, and repeats point back instead of starting over. "Explain it differently" gives a second angle when the first didn't land. |
| Quick review in each new section | The spacing effect and the testing effect (Module 5): retrieving an earlier miss later in the session beats rereading it right away. |
| Explain it back after the worked answer | Self-explanation and production over recognition (Module 4): saying why the wrong option is wrong turns a worked example into active processing. |
| Extras tucked behind ⚙️ | Cognitive load (Sweller, Module 1): the default screen shows only what the learner needs next. |
| Opt-in consent, minimal anonymous logs | Informed consent (Module 5 research ethics); data minimization (Module 1). |
| Answers only from the unit, visible sources, honest refusals | Transparency (Module 3). |

## Deploy (free)

1. Push to GitHub. The `.gitignore` keeps secrets, logs, PDFs, and virtual environments out of the repo.
2. share.streamlit.io → New app → main file `app.py`.
3. App settings → Secrets: `GEMINI_API_KEY`, `GEMINI_MODEL` (check the exact model name in Google AI Studio; `gemini-2.5-flash` no longer works for new keys).
4. Open the link in an incognito window.

Logs on Streamlit Community Cloud reset when the app restarts; download them from the admin view after sessions. The log columns changed in this version (`paragraph`, `action`, `practice_id`); an older `logs.csv` is renamed to `logs_<timestamp>.csv` the first time a new row is written.
