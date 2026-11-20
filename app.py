"""Hint First, Then Explain — CS6460 prototype.

Three roles: "Ask the reading" (short direct answers with citations), practice
questions (one at a time; answer first; hints, a bigger hint, and a worked answer after
a first try or "I'm stuck"), and assessments, which stay separate and tutor-free. After
a wrong answer, the pointer to the source paragraph fades over the session.

Learners can save any chat answer as a note; it shows highlighted under the paragraph
it cites. The chat remembers earlier questions: a repeat brings back the earlier answer
(with "Explain it differently"), and new answers can link to what was asked before.
Each new section opens with a spaced-review question from an earlier one, and the worked
answer asks for a one-sentence self-explanation.

Run locally:   streamlit run app.py
Admin view:    http://localhost:8501/?admin=1
"""
import csv
import html
import os
import random
import re
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

import streamlit as st
import streamlit.components.v1 as components

import tutor as T

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hint First, Then Explain",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CITE_ROUTER = components.declare_component("cite_router", path=str(T.Path(__file__).parent / "cite_router"))
UNIT_PATH = T.DATA / "unit.md" if (T.DATA / "unit.md").exists() else T.DATA / "sample_unit.md"
IS_SAMPLE = UNIT_PATH.name == "sample_unit.md"
PRACTICE  = T.load_items(T.DATA / "practice.json")
LOG_PATH  = T.DATA / "logs.csv"
LOG_FIELDS = [
    "time_utc", "session", "section", "paragraph", "action", "practice_id", "question",
    "outcome", "top_score", "sections_retrieved", "latency_s", "note",
    "check_correct", "confidence", "reflection",
]

SECTIONS  = [
    ("6.1", "What Is Learning?"),
    ("6.2", "Classical Conditioning"),
    ("6.3", "Operant Conditioning"),
    ("6.4", "Observational Learning (Modeling)"),
]
SEC_IDS = [s[0] for s in SECTIONS]
PANELS  = ["Ask the reading", "Practice"]
LADDER  = ["Hint", "Bigger hint", "Worked answer"]
# After a wrong first answer, how far practice points to the source paragraph. The support fades
# as the learner finishes practice sections: highlight it, then name it, then only the section.
POINTER_FADE = ["highlight", "name", "section"]
ACTIONS = {"simple": "Explain simply", "example": "Give an example"}
# Learner settings (⚙️): the features stay, but extra panels can be turned off
PREFS = {
    "review":  ("Quick review question when a section opens", True),
    "explain": ("Ask me to explain the worked answer back", True),
    "recap":   ("Recap my earlier questions in a new section", True),
    "pinned":  ("Show my notes under the paragraphs", True),
    "sources": ("Show a Sources box under each answer", False),
}
NO_QUIZ = "Quizzes need the live tutor right now. Try this section's practice questions instead."

CSS = """<style>
.reading { max-width: 40rem; line-height: 1.65; margin-bottom: 0.15rem; }
.reading.hl { background: rgba(234, 179, 8, 0.28); border-left: 4px solid #eab308;
              padding: 6px 10px; border-radius: 4px; }
.reading.saved { background: rgba(59, 130, 246, 0.10); border-left: 3px solid #3b82f6;
                 padding: 6px 10px; border-radius: 4px; }
a.cite { color: inherit; text-decoration: underline; text-decoration-color: #3b82f6;
         text-decoration-thickness: 2px; text-underline-offset: 2px; cursor: pointer; white-space: nowrap; }
a.cite:hover, a.cite:focus-visible { background: rgba(59, 130, 246, 0.15); outline: 2px solid #3b82f6; }
.strip { font-size: 0.85rem; color: rgba(128, 128, 128, 0.95); letter-spacing: 0.02em; }
.strip b { color: inherit; }
.strip .done { color: #16a34a; }
.note { max-width: 40rem; font-size: 0.9rem; margin: 0.2rem 0 0.4rem 1rem; padding: 6px 10px;
        background: rgba(59, 130, 246, 0.06); border-radius: 4px; }
</style>"""


# ── Pure helpers (no Streamlit calls) ────────────────────────────────────────
def get_secret(name: str, default=None):
    try:
        return st.secrets.get(name, os.environ.get(name, default))
    except Exception:
        return os.environ.get(name, default)


def is_admin() -> bool:
    try:
        return (
            st.query_params.get("admin", "") == "1"
            or str(get_secret("ADMIN_MODE", "false")).lower() == "true"
        )
    except Exception:
        return False


def _empty_stats() -> dict:
    return dict(questions=0, paragraph_asks=0, quick_actions=0, practice_tried=0,
                practice_right=0, hints=0, worked=0, confidences=[])


def bump(section: str, field: str, n: int = 1):
    st.session_state.section_stats.setdefault(section, _empty_stats())[field] += n


def new_mc_state() -> dict:
    return dict(stage="answer", choice=None, confidence=None, first_correct=None,
                retry_correct=None, stuck=False, help=[], explain=None, pointer=None)


def pointer_level(practice_sections_done: int) -> str:
    return POINTER_FADE[min(practice_sections_done, len(POINTER_FADE) - 1)]


def finished(s: dict | None) -> bool:
    return bool(s) and bool(s["first_correct"] or s["retry_correct"] or len(s["help"]) >= 3)


def write_log(row: dict):
    if LOG_PATH.exists():
        with open(LOG_PATH, encoding="utf-8") as f:
            header = f.readline().strip().split(",")
        if header != LOG_FIELDS:   # older log format: keep it under another name
            LOG_PATH.rename(LOG_PATH.with_name(f"logs_{int(time.time())}.csv"))
    new_file = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS, extrasaction="ignore")
        if new_file:
            w.writeheader()
        w.writerow(row)
    ws = log_sheet()
    if ws is not None:   # in the background, so a slow Sheets call never holds up the learner
        threading.Thread(target=append_to_sheet, args=(ws, [str(row.get(f, "")) for f in LOG_FIELDS]),
                         daemon=True).start()


@st.cache_resource
def log_sheet():
    """The Google Sheet that mirrors logs.csv, or None. Streamlit Community Cloud wipes local
    files when the app restarts, so a deployed pilot needs a copy that lasts."""
    url, creds = get_secret("LOG_SHEET_URL"), get_secret("gcp_service_account")
    if not url or not creds:
        return None
    try:
        import gspread
        ws = gspread.service_account_from_dict(dict(creds)).open_by_url(url).sheet1
        if not ws.row_values(1):
            ws.append_row(LOG_FIELDS)
        return ws
    except Exception:
        return None


def append_to_sheet(ws, values: list[str]):
    try:
        ws.append_row(values, value_input_option="RAW")
    except Exception:
        pass   # logs.csv still has the row


def log(section: str, question: str, reply: T.Reply | None = None, latency: float = 0.0, **extra):
    """One log row, written only with the learner's consent; free text is scrubbed."""
    ss = st.session_state
    if not ss.consent:
        return
    question = T.scrub(question)
    if "reflection" in extra:
        extra["reflection"] = T.scrub(extra["reflection"])
    write_log({
        "time_utc":           datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "session":            ss.session,
        "section":            section,
        "question":           question,
        "outcome":            reply.outcome if reply else "",
        "top_score":          f"{reply.top_score:.3f}" if reply else "",
        "sections_retrieved": " ".join(c.section_id for c, _ in reply.sources) if reply else "",
        "latency_s":          f"{latency:.2f}",
        "note":               reply.note if reply else "",
        **extra,
    })


# ── Callbacks (run before the rerun, so they may set widget state) ───────────
def set_focus(sec: str, para: int):
    """💬 is a toggle: on asks about the paragraph and highlights it; on again turns both off."""
    ss = st.session_state
    if ss.focus == (sec, para):
        clear_focus()
        return
    ss.focus, ss.highlight, ss.panel = (sec, para), (sec, para), PANELS[0]
    log(sec, f"(ask about ¶{para})", action="focus", paragraph=f"{sec} ¶{para}")


def clear_focus():
    ss = st.session_state
    if ss.highlight == ss.focus:
        ss.highlight = None
    ss.focus = None


def mark_done(sec: str):
    st.session_state.done.add(sec)
    st.session_state.panel = PANELS[1]
    log(sec, "(mark section done)", action="mark_done")


def go_to(sec: str, para: int):
    ss = st.session_state
    ss.active_section, ss.highlight = sec, (sec, para)
    ss.scroll_n += 1   # tells the page to scroll the highlighted paragraph into view


def scroll_to_highlight():
    """Scroll the reading to the highlighted paragraph after a "Go to ¶" click.

    The script runs in a zero-height iframe; scroll_n changes the HTML so it reruns
    on every click, and it retries briefly while the new page is still drawing.
    """
    components.html(f"""<script>
      // go_to #{st.session_state.scroll_n}
      let tries = 0;
      const t = setInterval(() => {{
        const el = window.parent.document.querySelector(".reading.hl");
        if (el || ++tries > 20) {{
          clearInterval(t);
          if (el) el.scrollIntoView({{behavior: "smooth", block: "center"}});
        }}
      }}, 100);
    </script>""", height=0)


def save_note(turn: dict):
    """Pin a chat answer to the first paragraph it cites; it then shows under that paragraph."""
    ss = st.session_state
    r = turn["reply"]
    cites = [(s, n) for s, n in T.parse_citations(r.text) if n] or \
            [(c.section_id, c.para_num) for c, _ in r.sources[:1]]
    sec, para = cites[0]
    ss.notes.append(dict(key=turn["key"], section=sec, para=para, question=turn["question"], text=r.text))
    log(sec, turn["question"], action="save_note", paragraph=f"{sec} ¶{para}")


def remove_note(key: str):
    ss = st.session_state
    ss.notes = [n for n in ss.notes if n["key"] != key]


# ── Streamlit widget helpers ──────────────────────────────────────────────────
@contextmanager
def thinking(text: str = "Thinking…"):
    """Spinner around a tutor call; each one counts toward the session's AI limit."""
    st.session_state.ai_uses += 1
    with st.spinner(text):
        yield


def pref(name: str) -> bool:
    return st.session_state.prefs.get(name, PREFS[name][1])


def set_pref(name: str):
    st.session_state.prefs[name] = st.session_state[f"pref_{name}"]


def show_sources(reply: T.Reply, key_prefix: str):
    """Collapsible sources with 'Go to ¶' navigation buttons (off by default: citations are links)."""
    if not pref("sources") or reply.outcome in ("refused", "clarify", "redirect") or not reply.sources:
        return
    with st.expander("Sources"):
        for c, score in reply.sources:
            st.markdown(f"**{c.para_label}**" + (f" · relevance {score:.2f}" if admin else ""))
            st.caption(c.text[:300])
            st.button(f"Go to {c.para_label}", key=f"{key_prefix}_{c.section_id}_{c.para_num}",
                      on_click=go_to, args=(c.section_id, c.para_num))


def debug(reply: T.Reply):
    if admin and ss.show_debug:
        st.caption(f"{reply.outcome} · {reply.top_score:.3f} · {reply.note}")


def go_to_button(label: str, key: str):
    try:
        c = tutor.chunk_at(label)
    except (AttributeError, StopIteration):
        return
    st.button(f"Go to {c.para_label}", key=key, on_click=go_to, args=(c.section_id, c.para_num))


BRACKET_RE = re.compile(r"\[([^\[\]]*¶[^\[\]]*)\]")
PARA_RE    = re.compile(r"(?:(\d+\.\d+)\s+)?¶(\d+)(?:[-–]\d+)?")


def linkify(text: str) -> str:
    """Escape model text for HTML and turn citations ([6.3 ¶4], [6.2 ¶7, 6.3 ¶1]) into links;
    the cite_router component turns a click on one into a jump to that paragraph."""
    def bracket(m: re.Match) -> str:
        sec = None

        def one(p: re.Match) -> str:
            nonlocal sec
            sec = p.group(1) or sec
            if not sec:
                return p.group(0)
            return (f'<a class="cite" href="#" data-sec="{sec}" data-para="{p.group(2)}" '
                    f'aria-label="Go to section {sec}, paragraph {p.group(2)}">{p.group(0)}</a>')
        return "[" + PARA_RE.sub(one, m.group(1)) + "]"
    escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return BRACKET_RE.sub(bracket, escaped)


def md(text: str):
    st.markdown(linkify(text), unsafe_allow_html=True)


def render_mc(item: dict, s: dict, key: str, section: str, ladder: bool, action: str = ""):
    """Multiple choice: answer → confidence → result. With ladder=True (practice), the
    hint ladder opens after the first try or "I'm stuck". action names the log row."""
    opts = item["options"]
    st.markdown(f"**{item['question']}**")

    if s["stage"] == "answer":
        choice = st.radio("Your answer", opts, index=None, key=f"{key}_choice",
                          label_visibility="collapsed")
        c1, c2 = st.columns(2)
        if c1.button("Check answer", key=f"{key}_check", disabled=choice is None):
            s["choice"], s["stage"] = opts.index(choice), "confidence"
            st.rerun()
        if ladder and not s["stuck"] and c2.button("I'm stuck", key=f"{key}_stuck"):
            s["stuck"] = True
            log(section, item["question"], action="practice_stuck", practice_id=item["id"])
            st.rerun()

    elif s["stage"] == "confidence":
        st.markdown(f"Your answer: {opts[s['choice']]}")
        conf = st.slider("How sure are you? 1 = guessing, 5 = certain", 1, 5, 3, key=f"{key}_conf")
        if st.button("Show result", key=f"{key}_reveal"):
            correct = s["choice"] == item["answer_index"]
            s["confidence"], s["first_correct"], s["stage"] = conf, correct, "result"
            if ladder:
                bump(section, "practice_tried")
                bump(section, "practice_right", int(correct))
                ss.section_stats[section]["confidences"].append(conf)
                if not correct:
                    s["pointer"] = pointer_level(len(ss.cal_shown))
                    if s["pointer"] == "highlight":
                        src = tutor.chunk_at(item["source_paragraph"])
                        go_to(src.section_id, src.para_num)
            log(section, item["question"], action=action or ("practice_answer" if ladder else "quiz_answer"),
                practice_id=item["id"], check_correct="y" if correct else "n", confidence=conf,
                note=f"pointer: {s['pointer']}" if s["pointer"] else "")
            st.rerun()

    else:
        source = item.get("source_paragraph", "")
        if s["first_correct"]:
            md(f"**Correct.** This is in [{source}].")
            go_to_button(source, f"{key}_goto")
        elif not ladder:
            md(f"**Not quite.** Reread [{source}].")
            st.caption(f"Correct answer: {opts[item['answer_index']]}")
            go_to_button(source, f"{key}_goto")
        elif s["pointer"] == "section" and s["retry_correct"] is None:
            # Faded support: the learner finds the paragraph; a missed retry or a hint names it
            st.markdown(f"**Not quite.** The answer is in section {item['section']}. "
                        "Find the paragraph that settles it, or open a hint below.")
        else:
            lead = "Reread the highlighted" if s["pointer"] == "highlight" else "Reread"
            md(f"**Not quite.** {lead} [{source}], or open a hint below.")
            go_to_button(source, f"{key}_goto")
        if ladder and not finished(s):
            again = st.radio("Try again", opts, index=None, key=f"{key}_retry")
            if st.button("Try again", key=f"{key}_retrybtn", disabled=again is None):
                s["retry_correct"] = opts.index(again) == item["answer_index"]
                log(section, item["question"], action="practice_retry", practice_id=item["id"],
                    check_correct="y" if s["retry_correct"] else "n")
                st.rerun()
            if s["retry_correct"] is False:
                st.caption("Still not it. A hint might help.")
        elif s["retry_correct"]:
            st.markdown("**Correct on the second try.**")

    if ladder and (s["stuck"] or s["stage"] == "result"):
        render_ladder(item, s, key, section)


def render_ladder(item: dict, s: dict, key: str, section: str):
    for i, r in enumerate(s["help"]):
        st.markdown(f"**{LADDER[i]}**")
        md(r.text)
        debug(r)
    level = len(s["help"])
    if level < len(LADDER) and st.button(LADDER[level], key=f"{key}_help{level + 1}"):
        t0 = time.time()
        with thinking():
            r = tutor.practice_help(item, level + 1)
        s["help"].append(r)
        bump(section, "worked" if level + 1 == 3 else "hints")
        log(section, item["question"], r, time.time() - t0,
            action=f"practice_help{level + 1}", practice_id=item["id"])
        st.rerun()
    if level == len(LADDER) and pref("explain"):
        render_self_explanation(item, s, key, section)


def render_self_explanation(item: dict, s: dict, key: str, section: str):
    """After the worked answer: explain, in one sentence, why the learner's wrong pick is
    wrong (or why the answer is right if they never picked one)."""
    opts, ans = item["options"], item["answer_index"]
    explaining = s["choice"] if s["choice"] is not None and s["choice"] != ans else ans
    ask = (f'In your own words: why is "{opts[explaining]}" wrong?' if explaining != ans
           else f'In your own words: why is "{opts[ans]}" right?')
    with st.container(border=True):
        st.markdown(f"**Explain it back.** {ask}")
        if s["explain"]:
            st.caption(f"You wrote: {s['explain']['text']}")
            r = s["explain"]["reply"]
            if r.rating:
                st.markdown(f"*{r.rating.capitalize()}.*")
            md(r.text)
            debug(r)
            return
        text = st.text_input("One sentence", key=f"{key}_explain", label_visibility="collapsed",
                             placeholder="One sentence is enough")
        if st.button("Check my reason", key=f"{key}_explainbtn", disabled=not text.strip()):
            t0 = time.time()
            with thinking():
                r = tutor.self_explanation(item, explaining, text)
            s["explain"] = dict(text=text.strip(), reply=r)
            log(section, item["question"], r, time.time() - t0, action="self_explain",
                practice_id=item["id"], reflection=text.strip(), note=f"{r.rating or r.outcome}; {r.note}")
            st.rerun()


def active_practice() -> list[dict]:
    """Practice items the learner has opened but not finished; the chat won't answer these."""
    return [it for it in PRACTICE if it["section"] in ss.done and not finished(ss.practice.get(it["id"]))]


def focus_chunk(sec: str) -> T.Chunk | None:
    if ss.focus and ss.focus[0] == sec:
        return tutor.chunk_at(f"{ss.focus[0]} ¶{ss.focus[1]}")
    return None


def add_turn(sec: str, question: str, reply: T.Reply | None, focus: T.Chunk | None,
             action: str = "", quiz: dict | None = None, repeat_of: dict | None = None) -> dict:
    turns = ss.turns.setdefault(sec, [])
    turn = dict(question=question, key=f"{sec}_{len(turns)}", reply=reply, action=action,
                paragraph=f"{focus.section_id} ¶{focus.para_num}" if focus else "",
                quiz=quiz, quiz_state=new_mc_state(), repeat_of=repeat_of, section=sec)
    turns.append(turn)
    return turn


def asked_turns() -> list[dict]:
    """Typed questions the chat answered, every section, oldest first (the chat's memory)."""
    order = {sid: i for i, sid in enumerate(SEC_IDS)}
    return [t for sec in sorted(ss.turns, key=order.get) for t in ss.turns[sec]
            if not t["action"] and not t["repeat_of"] and t["reply"]
            and t["reply"].outcome in ("answered", "fallback")]


def explain_differently(sec: str, turn: dict):
    turn["diff_done"] = True
    st.session_state.ai_uses += 1   # a callback, so no spinner; it still counts toward the limit
    t0 = time.time()
    r = tutor.explain_differently(turn["question"], turn["reply"].text, current_section=sec)
    add_turn(sec, "Explain it differently", r, None, action="differently")
    log(sec, turn["question"], r, time.time() - t0, action="differently")


def review_candidates(sec: str) -> list[dict]:
    """Practice items from earlier sections worth retrieving again: misses (surest first),
    then right answers given with low confidence. Already-reviewed items are left out."""
    earlier = set(SEC_IDS[:SEC_IDS.index(sec)])
    out = []
    for it in PRACTICE:
        s = ss.practice.get(it["id"])
        if it["section"] not in earlier or not s or s["first_correct"] is None or it["id"] in ss.reviewed:
            continue
        if not s["first_correct"]:
            out.append((0, -s["confidence"], it))
        elif s["confidence"] <= T.UNDERCONFIDENT:
            out.append((1, s["confidence"], it))
    return [it for *_, it in sorted(out, key=lambda x: x[:2])]


def render_review(sec: str):
    """Spaced review: the first time a section opens, one earlier practice question comes back
    (options reshuffled), so retrieval is spread over the session instead of massed."""
    if not pref("review"):
        return
    if sec not in ss.reviews:
        picks = review_candidates(sec)
        if not picks:
            return
        it = picks[0]
        ss.reviewed.add(it["id"])
        order = random.Random(f"{ss.session}{it['id']}").sample(range(len(it["options"])), len(it["options"]))
        item = dict(it, id=f"review_{it['id']}", options=[it["options"][i] for i in order],
                    answer_index=order.index(it["answer_index"]))
        ss.reviews[sec] = dict(item=item, state=new_mc_state(), hidden=False, from_section=it["section"])
        log(sec, it["question"], action="review_shown", practice_id=it["id"])
    rv = ss.reviews[sec]
    if rv["hidden"]:
        return
    with st.container(border=True):
        c1, c2 = st.columns([4, 1])
        c1.markdown(f"**Quick review from {rv['from_section']}.** One question you found tricky earlier.")
        if c2.button("Hide", key=f"review_hide_{sec}"):
            rv["hidden"] = True
            st.rerun()
        render_mc(rv["item"], rv["state"], f"rv_{sec}", sec, ladder=False, action="review_answer")


def render_recap(sec: str):
    """On a fresh section, what the learner already covered, so they don't start from zero."""
    earlier = [t for t in asked_turns() if t["section"] != sec]
    notes = [n for n in ss.notes if n["section"] != sec]
    if not earlier and not notes:
        return
    bits = []
    if earlier:
        bits.append("you asked about " + "; ".join(f"\"{t['question']}\" ({t['section']})"
                                                     for t in earlier[-T.HISTORY_TURNS:]))
    if notes:
        bits.append(f"you saved {len(notes)} note{'s' if len(notes) > 1 else ''}")
    st.caption("So far " + " and ".join(bits) + ". Answers here can build on that.")


def render_notes():
    """Contents of the 📌 Notes popover in the header."""
    if not ss.notes:
        st.caption("No notes yet. Use 📌 Save under any answer to keep it here and next to its paragraph.")
        return
    for n in ss.notes:
        st.markdown(f"**[{n['section']} ¶{n['para']}]** · *{n['question']}*")
        st.caption(linkify(n["text"]), unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.button(f"Go to {n['section']} ¶{n['para']}", key=f"note_go_{n['key']}",
                  on_click=go_to, args=(n["section"], n["para"]))
        c2.button("Remove", key=f"note_rm_{n['key']}", on_click=remove_note, args=(n["key"],))


def render_settings():
    """Contents of the ⚙️ popover: how the tutor works, and which extras to show."""
    st.markdown("**How this works**")
    st.markdown("1. **Read** a section on the left. Click 💬 next to a paragraph to ask about it.\n"
                "2. **Ask** anything. Answers come only from the reading; click a citation like "
                "[6.3 ¶4] to jump to it.\n"
                "3. **Practice** after you mark the section done: answer first, then hints if you need them.")
    st.markdown("**How answers are checked**")
    if tutor.model_name:
        st.markdown(f"Answers are written by Google's {tutor.model_name}, using only the paragraphs the tutor "
                    "retrieved. Before you see an answer, a second check compares each sentence with the "
                    "paragraph it cites and drops any sentence that paragraph doesn't support. Off-topic "
                    "questions are declined before the AI is called. Click any citation to check it yourself.")
    else:
        st.markdown("The AI is off right now, so answers quote the best-matching sentences of the reading "
                    "directly. Click any citation to see the paragraph.")
    st.divider()
    st.markdown("**Show**")
    for name, (label, _) in PREFS.items():
        st.toggle(label, value=pref(name), key=f"pref_{name}", on_change=set_pref, args=(name,))
    st.divider()
    st.caption("Logging: " + ("on, anonymous (you agreed at the start)." if ss.consent else "off."))


# ── Panels ────────────────────────────────────────────────────────────────────
def render_chat(sec: str):
    focus = focus_chunk(sec)
    turns = ss.turns.setdefault(sec, [])
    if not turns and pref("recap"):
        render_recap(sec)

    for turn in turns:
        with st.chat_message("user"):
            st.write(turn["question"])
        r = turn["reply"]
        if r:
            with st.chat_message("assistant"):
                if turn["repeat_of"]:
                    st.caption(f"You asked this earlier in {turn['repeat_of']['section']}: "
                               f"\"{turn['repeat_of']['question']}\". Here's that answer again.")
                md(r.text)
                debug(r)
                show_sources(r, f"src_{turn['key']}")
                c1, c2 = st.columns(2)
                if r.outcome in ("answered", "fallback"):
                    saved = any(n["key"] == turn["key"] for n in ss.notes)
                    c1.button("Saved to notes" if saved else "Save to notes", icon="📌", key=f"save_{turn['key']}",
                              type="tertiary", disabled=saved, on_click=save_note, args=(turn,))
                if turn["repeat_of"] and not turn.get("diff_done"):
                    c2.button("Explain it differently", icon="🔄", key=f"diff_{turn['key']}", type="tertiary",
                              on_click=explain_differently, args=(sec, turn))
        if turn["action"] == "quiz":
            with st.chat_message("assistant"):
                if turn["quiz"]:
                    render_mc(turn["quiz"], turn["quiz_state"], f"q_{turn['key']}", sec, ladder=False)
                else:
                    st.markdown(NO_QUIZ)

    if focus:
        with st.container(border=True):
            c1, c2 = st.columns([4, 1])
            c1.markdown(f"Asking about **¶{focus.para_num}**")
            c2.button("Clear", key="clear_focus", on_click=clear_focus)
            cols = st.columns(3)
            for col, (action, label) in zip(cols, ACTIONS.items()):
                if col.button(label, key=f"act_{action}"):
                    t0 = time.time()
                    with thinking():
                        r = tutor.answer("", current_section=sec, focus=focus, action=action)
                    add_turn(sec, f"{label} (¶{focus.para_num})", r, focus, action)
                    bump(sec, "quick_actions")
                    log(sec, f"{label} (¶{focus.para_num})", r, time.time() - t0,
                        action=action, paragraph=f"{sec} ¶{focus.para_num}")
                    st.rerun()
            if cols[2].button("Quiz me on this", key="act_quiz"):
                with thinking():
                    data = tutor.selfcheck(f"Quiz me on {focus.para_label}", [focus], focus.text)
                quiz = None
                if data:
                    quiz = dict(id=f"quiz_{sec}_{len(turns)}", question=data["question"],
                                options=data["options"], answer_index=data["answer_index"],
                                source_paragraph=data["source"].strip("[]"))
                add_turn(sec, f"Quiz me on ¶{focus.para_num}", None, focus, "quiz", quiz)
                bump(sec, "quick_actions")
                log(sec, f"Quiz me on ¶{focus.para_num}", action="quiz",
                    paragraph=f"{sec} ¶{focus.para_num}", note="" if quiz else "no quiz generated")
                st.rerun()

    where = f"¶{focus.para_num}" if focus else f"{sec} {dict(SECTIONS)[sec]}"
    question = st.chat_input(f"Ask about {where}…")
    if question:
        t0 = time.time()
        earlier = asked_turns()
        # A repeat is the same question about the same thing: in 💬 mode, the same paragraph
        here = f"{focus.section_id} ¶{focus.para_num}" if focus else ""
        same = [t for t in earlier if t["paragraph"] == here]
        if focus and T.is_vague(question):   # "what does this mean?" twice about one paragraph
            i = next((k for k, t in enumerate(same) if T.is_vague(t["question"])), None)
        else:
            i = tutor.repeat_of(question, [t["question"] for t in same])
        # A repeat brings back the earlier answer, unless the question is now an open practice item
        if i is not None and not tutor.practice_redirect(question, active_practice())[0]:
            prev = same[i]
            add_turn(sec, question, prev["reply"], focus,
                     repeat_of=dict(section=prev["section"], question=prev["question"]))
            log(sec, question, prev["reply"], action="repeat", note=f"repeats: {prev['question']}")
            st.rerun()
        with thinking("Writing an answer and checking it against the reading…" if tutor.generate_fn else "Thinking…"):
            r = tutor.answer(question, current_section=sec, focus=focus, active_practice=active_practice(),
                             history=[t["question"] for t in earlier])
        add_turn(sec, question, r, focus)
        bump(sec, "questions")
        if focus:
            bump(sec, "paragraph_asks")
        log(sec, question, r, time.time() - t0, action="ask",
            paragraph=f"{sec} ¶{focus.para_num}" if focus else "")
        st.rerun()


def render_practice(sec: str):
    items = [it for it in PRACTICE if it["section"] == sec]
    if sec not in ss.done:
        st.markdown("Finish the reading, then mark the section done to open its practice questions.")
        st.button("Mark section done", key="mark_done_practice", on_click=mark_done, args=(sec,))
        return
    if not items:
        st.markdown("No practice questions for this section.")
        return
    st.caption("Answer first. Hints open after your first try, or if you're stuck.")
    # One new question at a time: the next opens once this one has a first answer
    for n, it in enumerate(items, 1):
        s = ss.practice.setdefault(it["id"], new_mc_state())
        with st.container(border=True):
            st.caption(f"Question {n} of {len(items)}")
            render_mc(it, s, f"p_{it['id']}", sec, ladder=True)
        if s["first_correct"] is None:
            break

    results = [dict(confidence=s["confidence"], correct=s["first_correct"], n=n,
                    source_paragraph=it["source_paragraph"])
               for n, it in enumerate(items, 1)
               if (s := ss.practice.get(it["id"])) and s["first_correct"] is not None]
    if len(results) < len(items):
        if results:
            st.caption(f"{len(results)} of {len(items)} answered.")
        return
    render_calibration(sec, T.calibration(results))


def render_calibration(sec: str, cal: dict):
    """Shown right under the last practice question, as soon as it is answered."""
    if sec not in ss.cal_shown:
        ss.cal_shown.add(sec)
        st.toast("Practice done. Your summary is right below.")
        log(sec, "(calibration)", action="calibration", note=f"{cal['right']}/{cal['n']} right, "
            f"mean confidence {cal['confidence']:.1f}, gap {cal['gap']:+.2f}")
    with st.container(border=True):
        st.markdown("#### How this practice went")
        st.markdown(f"You felt **{cal['confidence']:.1f}/5** sure and got **{cal['right']} of {cal['n']}** right.")
        if cal["revisit"]:
            st.markdown("**Worth another look**")
            for r in cal["revisit"]:
                c1, c2 = st.columns([3, 2])
                c1.markdown(f"Question {r['n']} · rated {r['confidence']}/5")
                with c2:
                    go_to_button(r["source_paragraph"], f"cal_goto_{r['n']}")
        if cal["underrated"]:
            qs = ", ".join(str(r["n"]) for r in cal["underrated"])
            st.markdown(f"**You knew more than you felt.** You got question{'s' if len(cal['underrated']) > 1 else ''} "
                        f"{qs} right while rating {'them' if len(cal['underrated']) > 1 else 'it'} "
                        f"{T.UNDERCONFIDENT}/5 or lower. You can trust that understanding.")
        if cal["reread"]:
            md(f"Your confidence ran a little ahead of your answers this time, which is common "
                        f"early on. Rereading [{cal['reread']}] first is the quickest win: it covers "
                        "the question you felt surest about.")
        elif not cal["revisit"]:
            st.markdown("Every answer right on the first try. Nice work.")
        elif not cal["underrated"]:
            st.markdown("Your confidence matched your answers well. That's a useful skill.")


@st.cache_resource
def load_tutor(live: bool):
    gen, model = None, ""
    key = get_secret("GEMINI_API_KEY")
    if live and key:
        model = get_secret("GEMINI_MODEL", "gemini-3.8-flash")
        gen   = T.make_gemini(key, model)
    return T.Tutor(UNIT_PATH, generate_fn=gen, model_name=model)


# ══════════════════════════════════════════════════════════════════════════════
#  APP STARTS HERE
# ══════════════════════════════════════════════════════════════════════════════

admin   = is_admin()
# AI answers per learner session, so a public link can't use up the API quota
AI_LIMIT = int(get_secret("AI_LIMIT", 40))
has_key = bool(get_secret("GEMINI_API_KEY"))
live    = has_key

# ── Session state defaults ────────────────────────────────────────────────────
ss = st.session_state
ss.setdefault("session",        uuid.uuid4().hex[:8])
ss.setdefault("active_section", "6.1")
ss.setdefault("turns",          {})
ss.setdefault("practice",       {})
ss.setdefault("done",           set())
ss.setdefault("focus",          None)
ss.setdefault("section_stats",  {})
ss.setdefault("consent",        None)
ss.setdefault("highlight",      None)
ss.setdefault("scroll_n",       0)
ss.setdefault("cite_n",         None)
ss.setdefault("scrolled_n",     0)
ss.setdefault("reflection",     {})
ss.setdefault("pending_next",   False)
ss.setdefault("pending_dir",    1)
ss.setdefault("show_debug",     False)
ss.setdefault("panel",          PANELS[0])
ss.setdefault("notes",          [])
ss.setdefault("cal_shown",      set())
ss.setdefault("reviews",        {})
ss.setdefault("reviewed",       set())
ss.setdefault("prefs",          {})
ss.setdefault("ai_uses",        0)

# ── Admin sidebar (learners get no sidebar) ───────────────────────────────────
if admin:
    with st.sidebar:
        st.header("Admin")
        live = st.toggle("Live Gemini", value=has_key, disabled=not has_key)
        ss.show_debug = st.checkbox("Show safeguard details", value=ss.show_debug)
        st.caption("live" if (live and has_key) else "fallback (no API calls)")
        if IS_SAMPLE:
            st.warning("Sample text — load data/unit.md")
        if LOG_PATH.exists():
            st.download_button("Download log", LOG_PATH.read_bytes(), "logs.csv", "text/csv")
        st.caption("Log copy: Google Sheet" if log_sheet() else "Log copy: none (logs.csv resets when the app restarts)")
        st.caption(f"AI uses this session: {ss.ai_uses} (learners are limited to {AI_LIMIT})")
        if st.button("Clear session"):
            ss.turns, ss.practice, ss.done, ss.section_stats, ss.focus = {}, {}, set(), {}, None
            ss.notes, ss.cal_shown, ss.reviews, ss.reviewed = [], set(), {}, set()
            ss.ai_uses = 0
            st.rerun()
        st.divider()
        st.subheader("Per-section metrics")
        rows = []
        for sid, _ in SECTIONS:
            s = ss.section_stats.get(sid, _empty_stats())
            conf = s["confidences"]
            rows.append({
                "section": sid, "questions": s["questions"], "¶ asks": s["paragraph_asks"],
                "quick actions": s["quick_actions"],
                "practice right": f"{s['practice_right']}/{s['practice_tried']}" if s["practice_tried"] else "—",
                "mean conf": f"{sum(conf) / len(conf):.1f}" if conf else "—",
                "hints": s["hints"], "worked": s["worked"],
            })
        st.dataframe(rows, hide_index=True)
        if ss.reflection:
            st.caption("**Reflections**")
            for sid, txt in ss.reflection.items():
                st.caption(f"*{sid}:* {txt}")

# Past the limit the tutor keeps working offline: answers and hints quote the reading.
over_limit = not admin and live and ss.ai_uses >= AI_LIMIT
tutor = load_tutor(live and not over_limit)

# ── Consent gate ──────────────────────────────────────────────────────────────
if ss.consent is None:
    with st.container():   # one top-level element, so nothing is left over once the app starts
        st.title("Hint First, Then Explain")
        st.markdown(f"A study partner for **{tutor.unit_title}**, one section at a time.")
        c1, c2, c3 = st.columns(3)
        c1.markdown("**📖 Read**  \nThe section sits on the left.")
        c2.markdown("**💬 Ask**  \nShort answers, straight from the reading.")
        c3.markdown("**✏️ Practice**  \nTry first, hints if you need them.")
        st.markdown("---")
        consent_val = st.checkbox(
            "Log my questions and answers anonymously to help improve this tutor. "
            "Questions are saved word for word (emails and phone numbers are removed), "
            "so please don't type personal details.",
            value=False, key="consent_check",
        )
        if st.button("Start", type="primary"):
            ss.consent = consent_val
            st.rerun()
    st.stop()

st.markdown(CSS, unsafe_allow_html=True)

# A click on a citation link anywhere on the page jumps to that paragraph
cite = CITE_ROUTER(key="cite_router", default=None)
if cite and cite.get("n") != ss.cite_n:
    ss.cite_n = cite["n"]
    if (cite["sec"], cite["para"]) in {(c.section_id, c.para_num) for c in tutor.chunks}:
        go_to(cite["sec"], cite["para"])
        log(cite["sec"], f"(citation link ¶{cite['para']})", action="cite_link",
            paragraph=f"{cite['sec']} ¶{cite['para']}")

sec_id    = ss.active_section
sec_idx   = SEC_IDS.index(sec_id)
sec_title = dict(SECTIONS)[sec_id]

# Reflection gate (shown when Prev/Next is clicked and the section has chat turns)
if ss.pending_next:
    with st.container():
        st.markdown("**Optional reflection: what helped most in this section?**")
        refl = st.text_input("Answers, practice hints, or rereading? (leave blank to skip)",
                             key="reflection_input", placeholder="Leave blank to skip")
        if st.button("Continue", type="primary"):
            if refl.strip():
                ss.reflection[sec_id] = refl.strip()
                log(sec_id, "(reflection)", action="reflection", reflection=refl.strip())
            ss.active_section = SEC_IDS[sec_idx + ss.pending_dir]
            ss.highlight, ss.focus, ss.pending_next = None, None, False
            st.rerun()
    st.stop()


def step(direction: int):
    if ss.turns.get(sec_id):
        ss.pending_next, ss.pending_dir = True, direction
    else:
        ss.active_section = SEC_IDS[sec_idx + direction]
        ss.highlight, ss.focus = None, None


# ── Header: progress strip + stepper ──────────────────────────────────────────
title_col, notes_col, gear_col, prev_col, next_col = st.columns([7, 1.3, 0.6, 0.9, 0.9],
                                                               vertical_alignment="bottom")
with title_col:
    strip = []
    for sid, _ in SECTIONS:
        if sid == sec_id:
            strip.append(f"<b>{sid} {'✓' if sid in ss.done else '●'}</b>")
        else:
            strip.append(f'<span class="done">{sid} ✓</span>' if sid in ss.done else f"{sid} ○")
    st.markdown(f'<div class="strip">{" · ".join(strip)}</div>', unsafe_allow_html=True)
    st.markdown(f"### {sec_id} · {sec_title}")
with notes_col:
    with st.popover(f"📌 {len(ss.notes)}" if ss.notes else "📌", help="My notes", use_container_width=True):
        render_notes()
with gear_col:
    with st.popover("⚙️", help="How this works and settings"):
        render_settings()
with prev_col:
    if sec_idx > 0:
        st.button("← Prev", on_click=step, args=(-1,), use_container_width=True)
with next_col:
    if sec_idx < len(SECTIONS) - 1:
        st.button("Next →", on_click=step, args=(1,), use_container_width=True)

col_read, col_tutor = st.columns([5, 4], gap="large")

# ━━━━ LEFT: reading ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with col_read:
    for c in [c for c in tutor.chunks if c.section_id == sec_id]:
        pinned = [n for n in ss.notes if (n["section"], n["para"]) == (sec_id, c.para_num)] if pref("pinned") else []
        cls = ("reading hl" if ss.highlight == (sec_id, c.para_num)
               else "reading saved" if pinned else "reading")
        text_col, ask_col = st.columns([20, 1])
        text_col.markdown(f'<div class="{cls}"><b>¶{c.para_num}</b>&nbsp; {html.escape(c.text)}</div>',
                          unsafe_allow_html=True)
        for n in pinned:
            text_col.markdown(f'<div class="note">📌 <b>Your note</b> · <i>{html.escape(n["question"])}</i><br>'
                              f'{linkify(n["text"])}</div>', unsafe_allow_html=True)
        asking = ss.focus == (sec_id, c.para_num)
        ask_col.button("✕" if asking else "💬", key=f"ask_{sec_id}_{c.para_num}", type="tertiary",
                       help=f"Stop asking about ¶{c.para_num}" if asking else f"Ask about ¶{c.para_num}",
                       on_click=set_focus, args=(sec_id, c.para_num))

    st.caption(
        "Source: [Psychology 2e, OpenStax]"
        "(https://openstax.org/details/books/psychology-2e), CC BY 4.0."
        if not IS_SAMPLE else "Sample text written for testing."
    )
    if sec_id in ss.done:
        st.caption("Section marked done.")
    else:
        st.button("Mark section done", key="mark_done", type="primary", on_click=mark_done, args=(sec_id,))

# ━━━━ RIGHT: ask the reading / practice ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with col_tutor:
    if over_limit:
        st.info(f"You've used this session's {AI_LIMIT} AI answers. The tutor still works: "
                "from here on, answers and hints quote the reading directly.")
    render_review(sec_id)
    panel = st.radio("Panel", PANELS, key="panel", horizontal=True, label_visibility="collapsed")
    if panel == PANELS[0]:
        render_chat(sec_id)
    else:
        render_practice(sec_id)

if ss.scroll_n != ss.scrolled_n:
    ss.scrolled_n = ss.scroll_n
    scroll_to_highlight()
