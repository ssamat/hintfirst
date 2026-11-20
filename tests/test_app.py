"""End-to-end UI tests with a fake Gemini (no network, no key needed).

AppTest reads .streamlit/secrets.toml, so every test sets its own GEMINI_API_KEY and
ADMIN_MODE secrets instead of inheriting a real key or admin mode.

Run: pytest tests/test_app.py -v
"""
import json
import re
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import tutor as T

ROOT  = Path(__file__).resolve().parents[1]
APP   = str(ROOT / "app.py")
CALLS = []   # (system, prompt) for every fake Gemini call


def gen_calls():
    """Generation calls only, without the support check that follows each chat answer."""
    return [c for c in CALLS if c[0] != T.SUPPORT_CHECK_SYSTEM]


def fake_make_gemini(api_key, model="x"):
    def gen(system, prompt):
        CALLS.append((system, prompt))
        if system == T.SUPPORT_CHECK_SYSTEM:
            return "\n".join(f"{k} SUPPORTED" for k in range(1, prompt.count("SENTENCE ") + 1))
        cite = re.search(r"\[(\d+\.\d+ ¶\d+)\]", prompt).group(1)   # first retrieved paragraph
        if "self-check" in system:
            return json.dumps({"question": "Which behavior is a reflex?",
                               "options": ["Riding a bike", "Sucking at birth", "Reading"],
                               "answer_index": 1, "source": cite})
        if "You decide whether" in system:
            return "YES"
        if "You compare a learner's chat question" in system:
            return "YES" if "do babies do without being taught" in prompt else "NO"
        if "worked answer" in system:
            return f"1. The question asks which behavior is unlearned [{cite}].\n2. Babies suck at birth [{cite}]."
        if "second, more specific hint" in system:
            return f"Focus on which behavior nobody has to teach [{cite}]."
        if "You give HINTS" in system:
            return f"Which of these could a newborn already do? [{cite}]"
        if "wants an example" in system:
            return f"A puppy learning to sit for a treat. The treat follows the sit [{cite}]. Which part of [{cite}] matches?"
        return (f"Reinforcement increases a behavior [{cite}]. Punishment decreases it [{cite}]. "
                f"Which example in [{cite}] shows this?")
    return gen


@pytest.fixture
def restore_logs():
    logs = {p: p.read_bytes() for p in T.DATA.glob("logs*.csv")}
    yield
    for p in T.DATA.glob("logs*.csv"):
        if p not in logs:
            p.unlink()
    for p, data in logs.items():
        p.write_bytes(data)


def make_app(monkeypatch, key="fake-key", admin=False):
    monkeypatch.setattr(T, "make_gemini", fake_make_gemini)
    CALLS.clear()
    at = AppTest.from_file(APP, default_timeout=60)
    at.secrets["GEMINI_API_KEY"] = key
    at.secrets["ADMIN_MODE"] = "true" if admin else "false"
    at.run()
    return at


@pytest.fixture
def app(monkeypatch, restore_logs):
    return make_app(monkeypatch)


@pytest.fixture
def app_no_key(monkeypatch, restore_logs):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    return make_app(monkeypatch, key="")


def start(at, consent=False):
    if consent:
        at.checkbox(key="consent_check").check()
    at.button[0].click().run()
    assert not at.exception
    return at


def click(at, key):
    """Click the live copy of a button. AppTest keeps stale copies of elements that moved
    after a rerun (a browser drops them), so take the last one with this key."""
    [b for b in at.button if b.key == key][-1].click().run()
    assert not at.exception


def labels(at):
    return [b.label for b in at.button]


def plain(html_text):
    """Rendered text without the citation links' HTML (replies render through linkify)."""
    return re.sub(r"<[^>]+>", "", html_text).replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")


def texts(at):
    return [m.value for m in at.markdown] + [m.markdown[0].value for m in at.chat_message if m.markdown] \
        + [plain(m.value) for m in at.markdown] + [plain(m.markdown[0].value) for m in at.chat_message if m.markdown]


def last_reply(at):
    return plain(at.chat_message[-1].markdown[0].value)


# ── layout ────────────────────────────────────────────────────────────────────
def test_sidebar_collapsed_and_progress_strip(app):
    assert 'initial_sidebar_state="collapsed"' in Path(APP).read_text()
    at = start(app)
    assert len(at.sidebar.children) == 0                      # learners get no sidebar
    assert any("<b>6.1 ●</b> · 6.2 ○ · 6.3 ○ · 6.4 ○" in t for t in texts(at))
    click(at, "mark_done")
    assert any("<b>6.1 ✓</b> · 6.2 ○ · 6.3 ○ · 6.4 ○" in t for t in texts(at))
    assert at.radio(key="panel").value == "Practice"           # Mark section done opens practice


def test_admin_sidebar_has_metrics(monkeypatch, restore_logs):
    at = start(make_app(monkeypatch, admin=True))
    assert "Per-section metrics" in [s.value for s in at.sidebar.subheader]


# ── ask the reading ───────────────────────────────────────────────────────────
def test_direct_answer_with_citations(app):
    at = start(app)
    at.chat_input[0].set_value("What is the difference between negative reinforcement and punishment?").run()
    reply = last_reply(at)
    assert re.search(r"\[6\.\d ¶\d+\]", reply) and reply.rstrip().endswith("?")
    assert gen_calls()[-1][0].startswith(T.ANSWER_SYSTEM)
    assert CALLS[-1][0] == T.SUPPORT_CHECK_SYSTEM                # every direct answer is checked
    assert not {"Hint", "Bigger hint", "Worked answer"} & set(labels(at))   # no ladder in the chat


def test_declined_question(app):
    at = start(app)
    at.chat_input[0].set_value("What is the capital of France?").run()
    assert last_reply(at) == T.REFUSAL


def test_practice_question_redirected_once_open(app):
    at = start(app)
    rq1 = "Which of the following is an example of a reflex that occurs at some point in the development of a human being?"
    at.chat_input[0].set_value(rq1).run()
    assert last_reply(at) != T.PRACTICE_REDIRECT               # practice not open yet
    click(at, "mark_done")
    at.radio(key="panel").set_value("Ask the reading").run()
    at.chat_input[0].set_value(rq1).run()
    assert last_reply(at) == T.PRACTICE_REDIRECT


def test_paragraph_scoped_asking_is_logged(monkeypatch, restore_logs):
    at = start(make_app(monkeypatch), consent=True)
    click(at, "ask_6.1_2")
    assert {"Explain simply", "Give an example", "Quiz me on this"} <= set(labels(at))
    click(at, "act_simple")
    system, prompt = gen_calls()[-1]
    assert T.SIMPLE_RULE in system
    assert prompt.startswith("PASSAGES:\n[6.1 ¶2]")              # focus paragraph retrieved first
    assert "THE LEARNER IS ASKING ABOUT [6.1 ¶2]" in prompt
    assert "[6.1 ¶2]" in last_reply(at)

    at.chat_input[0].set_value("Why is this unlearned?").run()  # typed question stays scoped
    assert gen_calls()[-1][1].startswith("PASSAGES:\n[6.1 ¶2]")

    click(at, "act_quiz")
    assert any("Which behavior is a reflex?" in t for t in texts(at))
    rows = (T.DATA / "logs.csv").read_text()
    assert "focus" in rows and "simple" in rows and "6.1 ¶2" in rows


def test_paraphrased_practice_question_redirected(app):
    at = start(app)
    click(at, "mark_done")
    at.radio(key="panel").set_value("Ask the reading").run()
    at.chat_input[0].set_value("Which of these do babies do without being taught: riding a bike or walking?").run()
    assert last_reply(at) == T.PRACTICE_REDIRECT                # reworded, so Gemini judged it
    at.chat_input[0].set_value("What is a reflex?").run()
    assert last_reply(at) != T.PRACTICE_REDIRECT


def test_save_note_shows_under_paragraph(app):
    at = start(app)
    at.chat_input[0].set_value("What is a reflex?").run()
    click(at, "save_6.1_0")
    assert "Saved to notes" in labels(at)
    assert any("Your note" in t and "What is a reflex?" in t for t in texts(at))
    assert "note_go_6.1_0" in [b.key for b in at.button]         # listed in the 📌 notes popover
    click(at, "note_rm_6.1_0")
    assert not any("Your note" in t for t in texts(at))


def test_repeat_question_brings_back_earlier_answer(app):
    at = start(app)
    at.chat_input[0].set_value("What is a reflex?").run()
    first, n = last_reply(at), len(CALLS)
    at.chat_input[0].set_value("what is a reflex again").run()
    assert last_reply(at) == first and len(CALLS) == n          # no new generation
    assert any("You asked this earlier" in c.value for c in at.caption)
    at.chat_input[0].set_value("How do instincts differ from reflexes?").run()
    assert "EARLIER QUESTIONS:\n- What is a reflex?" in gen_calls()[-1][1]


def test_next_section_recaps_earlier_questions(app):
    at = start(app)
    at.chat_input[0].set_value("What is a reflex?").run()
    click(at, "save_6.1_0")
    [b for b in at.button if b.label == "Next →"][-1].click().run()
    [b for b in at.button if b.label == "Continue"][-1].click().run()
    assert any("So far you asked about" in c.value and "1 note" in c.value for c in at.caption)


# ── practice ──────────────────────────────────────────────────────────────────
def test_practice_flow(app):
    at = start(app)
    click(at, "mark_done")
    assert "Hint" not in labels(at)                            # no ladder before a first try
    assert "I'm stuck" in labels(at)

    rq1 = next(i for i in T.load_items(T.DATA / "practice.json") if i["id"] == "rq1")
    wrong = rq1["options"][(rq1["answer_index"] + 1) % 4]
    at.radio(key="p_rq1_choice").set_value(wrong).run()
    click(at, "p_rq1_check")
    assert not any("Not quite" in t for t in texts(at))         # confidence comes before the result
    at.slider(key="p_rq1_conf").set_value(5).run()
    click(at, "p_rq1_reveal")
    assert any("Not quite" in t for t in texts(at))

    click(at, "p_rq1_help1")                 # ladder unlocked by the first try
    assert any("newborn" in t for t in texts(at))
    click(at, "p_rq1_help2")
    assert any("nobody has to teach" in t for t in texts(at))
    click(at, "p_rq1_help3")
    assert any("Babies suck at birth" in t for t in texts(at))
    assert CALLS[-1][0] == T.WORKED_SYSTEM

    # "I'm stuck" opens the ladder before answering; then answer right and see calibration
    click(at, "p_rq2_stuck")
    assert "p_rq2_help1" in [b.key for b in at.button]
    rq2 = next(i for i in T.load_items(T.DATA / "practice.json") if i["id"] == "rq2")
    at.radio(key="p_rq2_choice").set_value(rq2["options"][rq2["answer_index"]]).run()
    click(at, "p_rq2_check")
    at.slider(key="p_rq2_conf").set_value(5).run()
    click(at, "p_rq2_reveal")
    assert any("You felt **5.0/5** sure and got **1 of 2** right." in t for t in texts(at))
    assert any(f"Rereading [{rq1['source_paragraph']}] first" in t for t in texts(at))
    assert any("Worth another look" in t for t in texts(at))
    assert "cal_goto_1" in [b.key for b in at.button]          # missed question 1 links to its paragraph
    assert not at.exception


def answer_wrong(at, item_id):
    item = next(i for i in T.load_items(T.DATA / "practice.json") if i["id"] == item_id)
    at.radio(key=f"p_{item_id}_choice").set_value(item["options"][(item["answer_index"] + 1) % 4]).run()
    click(at, f"p_{item_id}_check")
    click(at, f"p_{item_id}_reveal")
    return item


def test_practice_shows_one_new_question_at_a_time(app):
    at = start(app)
    click(at, "mark_done")
    keys = [r.key for r in at.radio]
    assert "p_rq1_choice" in keys and "p_rq2_choice" not in keys
    answer_wrong(at, "rq1")
    assert "p_rq2_choice" in [r.key for r in at.radio]


def test_pointer_fades_across_sections(app):
    # First practice section: a wrong answer highlights the source paragraph
    at = start(app)
    click(at, "mark_done")
    rq1 = answer_wrong(at, "rq1")
    para = rq1["source_paragraph"].split("¶")[1]
    assert any(f'class="reading hl"><b>¶{para}</b>' in t for t in texts(at))
    assert any("Reread the highlighted" in t for t in texts(at))

    # After two finished practice sections: only the section, until a missed retry
    at.session_state["cal_shown"] = {"6.2", "6.3"}
    at.session_state["highlight"] = None
    rq2 = answer_wrong(at, "rq2")
    assert any("The answer is in section 6.1." in t for t in texts(at))
    assert "p_rq2_goto" not in [b.key for b in at.button]
    assert not any('class="reading hl"' in t for t in texts(at))
    at.radio(key="p_rq2_retry").set_value(rq2["options"][(rq2["answer_index"] + 2) % 4]).run()
    click(at, "p_rq2_retrybtn")
    assert "p_rq2_goto" in [b.key for b in at.button]           # now it names the paragraph


def test_repeat_in_paragraph_mode(app):
    at = start(app)
    click(at, "ask_6.1_2")
    at.chat_input[0].set_value("Why is this unlearned?").run()
    first, n = last_reply(at), len(CALLS)
    at.chat_input[0].set_value("why is this unlearned again").run()
    assert last_reply(at) == first and len(CALLS) == n          # same paragraph: earlier answer
    click(at, "ask_6.1_3")
    at.chat_input[0].set_value("Why is this unlearned?").run()
    assert len(CALLS) > n                                       # another paragraph: a new answer


def test_vague_repeat_in_paragraph_mode(app):
    at = start(app)
    click(at, "ask_6.1_2")
    at.chat_input[0].set_value("what does this mean?").run()
    first, n = last_reply(at), len(CALLS)
    at.chat_input[0].set_value("explain this").run()
    assert last_reply(at) == first and len(CALLS) == n          # same paragraph: earlier answer
    assert any("You asked this earlier" in c.value for c in at.caption)


def test_session_ai_limit_falls_back_to_reading(monkeypatch, restore_logs):
    monkeypatch.setattr(T, "make_gemini", fake_make_gemini)
    CALLS.clear()
    at = AppTest.from_file(APP, default_timeout=60)
    at.secrets["GEMINI_API_KEY"], at.secrets["ADMIN_MODE"], at.secrets["AI_LIMIT"] = "fake-key", "false", "1"
    at.run()
    at = start(at)
    at.chat_input[0].set_value("What is a reflex?").run()
    n = len(CALLS)
    assert n > 0 and re.search(r"\[6\.\d ¶\d+\]", last_reply(at))   # the one AI answer
    at.chat_input[0].set_value("How do instincts differ from reflexes?").run()
    assert len(CALLS) == n                                       # past the limit: no API calls
    assert re.search(r"\[6\.\d ¶\d+\]", last_reply(at))           # still a cited answer
    assert "this session's 1 AI answers" in at.info[0].value


class FakeSheet:
    def __init__(self):
        self.rows = []

    def row_values(self, n):
        return self.rows[n - 1] if len(self.rows) >= n else []

    def append_row(self, values, value_input_option=None):
        self.rows.append(values)


def test_log_mirrored_to_google_sheet(monkeypatch, restore_logs):
    import sys
    import time
    import types
    sheet = FakeSheet()
    fake = types.ModuleType("gspread")
    fake.service_account_from_dict = lambda creds: types.SimpleNamespace(
        open_by_url=lambda url: types.SimpleNamespace(sheet1=sheet))
    monkeypatch.setitem(sys.modules, "gspread", fake)
    st.cache_resource.clear()                                    # log_sheet() was cached as None
    try:
        monkeypatch.setattr(T, "make_gemini", fake_make_gemini)
        at = AppTest.from_file(APP, default_timeout=60)
        at.secrets["GEMINI_API_KEY"], at.secrets["ADMIN_MODE"] = "fake-key", "false"
        at.secrets["LOG_SHEET_URL"] = "https://docs.google.com/spreadsheets/d/x"
        at.secrets["gcp_service_account"] = {"type": "service_account", "client_email": "x@y"}
        at.run()
        at = start(at, consent=True)
        at.chat_input[0].set_value("What is a reflex?").run()
        for _ in range(50):                                      # rows are appended in the background
            if len(sheet.rows) > 1:
                break
            time.sleep(0.05)
        assert sheet.rows[0][:3] == ["time_utc", "session", "section"]
        assert any("What is a reflex?" in r for r in sheet.rows[1:])
    finally:
        st.cache_resource.clear()


# ── no API key ────────────────────────────────────────────────────────────────
def test_runs_without_api_key(app_no_key):
    at = start(app_no_key)
    at.chat_input[0].set_value("What is negative reinforcement?").run()
    assert re.search(r"\[6\.\d ¶\d+\]", last_reply(at))
    click(at, "ask_6.1_2")
    click(at, "act_quiz")
    assert any("Quizzes need the live tutor" in t for t in texts(at))
    click(at, "mark_done")
    click(at, "p_rq1_stuck")
    click(at, "p_rq1_help1")
    assert any("Reread [6.1 ¶2]" in t for t in texts(at))
    assert CALLS == [] and not at.exception


def test_go_to_highlights_paragraph(app):
    at = start(app)
    at.toggle(key="pref_sources").set_value(True).run()         # Sources box is off by default
    at.chat_input[0].set_value("What is a reflex?").run()
    go = next(b for b in at.button if b.key and b.key.startswith("src_6.1_0_"))
    para = go.key.rsplit("_", 1)[1]
    go.click().run()
    assert not at.exception
    assert any(f'class="reading hl"><b>¶{para}</b>' in t for t in texts(at))
    assert at.session_state.scrolled_n == 1                     # the scroll script was sent


def test_citations_are_links(app):
    at = start(app)
    at.chat_input[0].set_value("What is a reflex?").run()
    raw = at.chat_message[-1].markdown[0].value
    assert re.search(r'<a class="cite" href="#" data-sec="6\.\d" data-para="\d+" '
                     r'aria-label="Go to section 6\.\d, paragraph \d+">6\.\d ¶\d+</a>', raw)


def test_spaced_review_in_next_section(app):
    at = start(app)
    click(at, "mark_done")
    rq1 = next(i for i in T.load_items(T.DATA / "practice.json") if i["id"] == "rq1")
    at.radio(key="p_rq1_choice").set_value(rq1["options"][(rq1["answer_index"] + 1) % 4]).run()
    click(at, "p_rq1_check")
    at.slider(key="p_rq1_conf").set_value(5).run()
    click(at, "p_rq1_reveal")
    [b for b in at.button if b.label == "Next →"][-1].click().run()
    assert any("Quick review from 6.1" in t for t in texts(at))
    assert at.radio(key="rv_6.2_choice").options and set(at.radio(key="rv_6.2_choice").options) == set(rq1["options"])
    at.radio(key="rv_6.2_choice").set_value(rq1["options"][rq1["answer_index"]]).run()
    click(at, "rv_6.2_check")
    click(at, "rv_6.2_reveal")
    assert any("Correct." in t for t in texts(at))
    click(at, "review_hide_6.2")
    assert not any("Quick review" in t for t in texts(at))


def test_self_explanation_after_worked_answer(app):
    at = start(app)
    click(at, "mark_done")
    click(at, "p_rq1_stuck")
    for lv in (1, 2, 3):
        click(at, f"p_rq1_help{lv}")
    assert any("Explain it back." in t and "right?" in t for t in texts(at))
    at.text_input(key="p_rq1_explain").set_value("Babies suck without being taught.").run()
    click(at, "p_rq1_explainbtn")
    assert CALLS[-1][0] == T.SELF_EXPLAIN_SYSTEM
    assert "LEARNER EXPLANATION: Babies suck without being taught." in CALLS[-1][1]
    assert any("You wrote:" in c.value for c in at.caption)


def test_explain_differently_on_repeat(app):
    at = start(app)
    at.chat_input[0].set_value("What is a reflex?").run()
    at.chat_input[0].set_value("what is a reflex again").run()
    click(at, "diff_6.1_1")
    system, prompt = gen_calls()[-1]
    assert T.DIFFERENT_RULE in system and "EARLIER ANSWER:" in prompt
    assert "diff_6.1_1" not in [b.key for b in at.button]      # offered once per repeat


def test_log_scrubs_contact_details(monkeypatch, restore_logs):
    at = start(make_app(monkeypatch), consent=True)
    at.chat_input[0].set_value("reflex question, email me at a.b@x.com or 404-555-1234").run()
    rows = (T.DATA / "logs.csv").read_text()
    assert "a.b@x.com" not in rows and "555-1234" not in rows and "[email]" in rows and "[phone]" in rows


def test_settings_hide_extras(app):
    at = start(app)
    at.chat_input[0].set_value("What is a reflex?").run()
    assert not any(b.key and b.key.startswith("src_") for b in at.button)    # sources hidden by default
    click(at, "save_6.1_0")
    assert any("Your note" in t for t in texts(at))
    at.toggle(key="pref_pinned").set_value(False).run()
    assert not any("Your note" in t for t in texts(at))
    assert at.session_state.prefs == {"pinned": False}


def test_ask_button_toggles_highlight(app):
    at = start(app)
    click(at, "ask_6.1_2")
    assert at.session_state.highlight == ("6.1", 2) and at.session_state.focus == ("6.1", 2)
    assert [b for b in at.button if b.key == "ask_6.1_2"][-1].label == "✕"
    click(at, "ask_6.1_2")                                       # second click turns it off
    assert at.session_state.highlight is None and at.session_state.focus is None
    click(at, "ask_6.1_3")
    click(at, "clear_focus")                                     # so does Clear in the chat
    assert at.session_state.highlight is None and at.session_state.focus is None
