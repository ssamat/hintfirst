"""Core logic for the Hint First, Then Explain tutor.

Pipeline: load unit → chunk by section (with paragraph numbers) → retrieve
(local TF-IDF, boosted toward the current section, focus paragraph first) →
safeguards (relevance threshold, vague-question check, practice-question redirect)
→ generate with Gemini (grounding-only prompts) → post-checks (citation check,
support check on direct answers, leak check on hints) → fallbacks (cached or extractive replies) when the API is
unavailable.

Three roles: answer() is the chat (short direct answers with citations);
practice_help() is the hint → bigger hint → worked-answer ladder for practice
questions; selfcheck() writes "Quiz me" questions. Assessments never touch the tutor.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA = Path(__file__).parent / "data"

# ── Tunable safeguards ────────────────────────────────────────────────────────
RELEVANCE_THRESHOLD = 0.135  # below this the tutor declines (character-level match score, see Retriever.relevance)
SCOPE_FLOOR         = 0.08   # between this and the threshold, Gemini judges scope (if a key is set)
TOP_K               = 3      # passages given to the model
LEAK_NGRAM          = 10     # a hint that copies this many words in a row is rejected
MAX_CLAIMS          = 3      # direct answers: at most 3 cited sentences + 1 connecting question
PRACTICE_REDIRECT_THRESHOLD = 0.65  # chat question vs open practice item, char n-gram cosine (tuned on data/practice_overlap_set.csv)
DUPLICATE_THRESHOLD = 0.8    # practice item vs pre/post-test item (distinct practice items score up to 0.64)
OVERCONFIDENCE_GAP  = 0.3    # mean confidence/5 minus accuracy at or above this → suggest a reread
UNDERCONFIDENT      = 2      # a right answer rated at or below this is "you knew more than you felt"
PARAPHRASE_FLOOR    = 0.3    # between this and the redirect threshold, Gemini judges "same question, reworded?"
OPTIONS_LISTED      = 3      # a chat question naming this many of an item's options is a pasted practice question
REPEAT_THRESHOLD    = 0.8    # chat question vs an earlier one, char n-gram cosine (or same content words)
HISTORY_TURNS       = 3      # earlier questions passed to the answer prompt so answers can build on them
NAMES_SHARE         = 0.66   # a hint containing this share of the correct option's word stems names it
# Asking words that don't change what a question is about ("define shaping" repeats "what is shaping?")
REPEAT_FILLER       = {"define", "definition", "meaning", "describe", "whats", "term", "concept", "called"}


# ── Data model ────────────────────────────────────────────────────────────────
@dataclass
class Chunk:
    section_id:    str   # e.g. "6.2"
    section_title: str   # e.g. "Classical Conditioning"
    text:          str
    para_num:      int = 0   # 1-indexed within the section

    @property
    def label(self) -> str:
        return f"{self.section_id} {self.section_title}"

    @property
    def para_label(self) -> str:
        return f"[{self.section_id} ¶{self.para_num}]"


# ── Unit loading ──────────────────────────────────────────────────────────────
def load_unit(path: Path) -> tuple[str, list[Chunk]]:
    """Parse a markdown unit with paragraph numbering.

    Headings:
      # Title          → unit title
      ## 6.2 Title     → section; resets para counter
      ### Subsection   → heading is stripped; following text is regular chunks
    Blank lines separate paragraphs (chunks).
    """
    raw = path.read_text(encoding="utf-8")
    raw = re.sub(r"<!--.*?-->", "", raw, flags=re.S)

    title_match = re.search(r"^# (.+)$", raw, flags=re.M)
    unit_title = title_match.group(1).strip() if title_match else path.stem

    chunks: list[Chunk] = []
    current_id, current_title = "", ""
    para_count: dict[str, int] = {}

    for block in re.split(r"\n\s*\n", raw):
        block = block.strip()
        if not block or block.startswith("# "):
            continue

        first_line = block.splitlines()[0]

        # ## section heading
        heading = re.match(r"^##\s+(\d+(?:\.\d+)*)\s+(.+)$", first_line)
        if heading:
            current_id    = heading.group(1)
            current_title = heading.group(2).strip()
            para_count.setdefault(current_id, 0)
            block = "\n".join(block.splitlines()[1:]).strip()
            if not block:
                continue

        # ### subsection heading — strip the heading line, keep the body
        elif first_line.startswith("###"):
            block = "\n".join(block.splitlines()[1:]).strip()
            if not block:
                continue

        if current_id:
            para_count[current_id] = para_count.get(current_id, 0) + 1
            chunks.append(Chunk(
                current_id,
                current_title,
                " ".join(block.split()),
                para_count[current_id],
            ))

    if not chunks:
        raise ValueError("No sections found. Use '## 6.1 Section Title' headings.")
    return unit_title, chunks


# ── Retrieval ─────────────────────────────────────────────────────────────────
def fold_plurals(text: str) -> list[str]:
    """Word tokens with plain plurals folded ("reflexes" → "reflex", "instincts" → "instinct"),
    so a singular question matches a plural paragraph."""
    out = []
    for w in re.findall(r"(?u)\b\w\w+\b", text.lower()):
        if len(w) > 4 and re.search(r"(x|ch|sh|ss)es$", w):
            w = w[:-2]
        elif len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
            w = w[:-1]
        out.append(w)
    return out


FOLDED_STOP_WORDS = sorted({t for w in ENGLISH_STOP_WORDS for t in fold_plurals(w)} | set(ENGLISH_STOP_WORDS))


class Retriever:
    """TF-IDF retrieval: word n-grams for meaning, char n-grams for misspellings.

    boost_section multiplies raw scores by 1.15 for ranking only; threshold
    is always applied to the unmodified raw score so off-topic questions in the
    current section are still declined.
    """

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        docs = [f"{c.section_title}. {c.text}" for c in chunks]
        self.word_vec = TfidfVectorizer(stop_words=FOLDED_STOP_WORDS, ngram_range=(1, 2), sublinear_tf=True,
                                        tokenizer=fold_plurals, token_pattern=None)
        self.char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 5), sublinear_tf=True)
        self.word_m = self.word_vec.fit_transform(docs)
        self.char_m = self.char_vec.fit_transform(docs)
        # Adjacent content-word pairs in the unit ("knee jerk", "bobo doll") for has_phrase
        self.phrases: set[tuple[str, str]] = set()
        for c in chunks:
            words = re.findall(r"[a-z]+", c.text.lower())
            self.phrases |= {
                p for p in zip(words, words[1:])
                if content_words(" ".join(p)) == list(p)
            }

    def search(
        self,
        query: str,
        k: int = TOP_K,
        boost_section: str | None = None,
        focus: Chunk | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Top-k chunks; a focus paragraph ("Ask about ¶N") is always ranked first."""
        w = cosine_similarity(self.word_vec.transform([query]), self.word_m)[0]
        c = cosine_similarity(self.char_vec.transform([query]), self.char_m)[0]
        raw = 0.6 * w + 0.4 * c

        # Boost for ranking only — raw scores are returned for threshold checks
        ranked = raw.copy()
        if boost_section:
            for i, chunk in enumerate(self.chunks):
                if chunk.section_id == boost_section:
                    ranked[i] *= 1.15

        order = list(np.argsort(-ranked))
        if focus is not None:
            f = self.chunks.index(focus)
            order = [f] + [i for i in order if i != f]
        return [(self.chunks[i], float(raw[i])) for i in order[:k]]

    def relevance(self, query: str) -> float:
        """Gate score for 'is this about the unit at all?'.

        Uses the best character n-gram match, which separated in-scope (incl. misspelled)
        from off-topic questions on the gold set better than the blended ranking score.
        """
        return float(cosine_similarity(self.char_vec.transform([query]), self.char_m)[0].max())

    def has_phrase(self, query: str) -> bool:
        """True if the query contains a two-word phrase from the unit (e.g. "knee jerk").

        Catches specific in-scope questions whose char score is diluted by filler or typos;
        no off-topic gold-set question contains one.
        """
        words = re.findall(r"[a-z]+", query.lower())
        return any(p in self.phrases for p in zip(words, words[1:]))


# ── Safeguards ────────────────────────────────────────────────────────────────
FILLER = {
    "tell", "answer", "explain", "help", "get", "dont", "don", "understand",
    "part", "just", "please", "confused", "mean", "means", "know", "thing",
    "stuff", "really", "does", "did", "say", "says", "paragraph", "sentence", "section",
    "again", "remind",
}


def content_words(text: str) -> list[str]:
    words = re.findall(r"[a-z]+", text.lower())
    return [w for w in words if w not in ENGLISH_STOP_WORDS and w not in FILLER and len(w) > 2]


def is_vague(question: str) -> bool:
    return len(content_words(question)) < 1


ANALOGY_RE = re.compile(
    r"\b(analog(y|ies)|metaphor|compare (it|this|that) to|(real[- ]life|everyday|real[- ]world) example|eli5|like i'?m (5|five))\b",
    re.I,
)
ANALOGY_WORDS = {
    "analogy", "analogies", "metaphor", "compare", "example", "real", "life", "world",
    "everyday", "eli", "simple", "simpler", "terms", "five", "give", "make", "use", "topic",
}


def wants_analogy(question: str) -> bool:
    return bool(ANALOGY_RE.search(question))


def analogy_topic(question: str) -> list[str]:
    """Content words left once the analogy request itself is removed ("an analogy for X" → X)."""
    return [w for w in content_words(question) if w not in ANALOGY_WORDS]


def leaks_text(reply: str, passages: list[Chunk], n: int = LEAK_NGRAM) -> bool:
    """True if the reply copies n or more consecutive words from retrieved text."""
    def grams(t: str) -> set[tuple[str, ...]]:
        w = re.findall(r"[a-z']+", t.lower())
        return {tuple(w[i : i + n]) for i in range(len(w) - n + 1)}

    source = set().union(*(grams(p.text) for p in passages)) if passages else set()
    return bool(grams(reply) & source)


def cited_ids(reply: str) -> set[str]:
    """Extract section IDs from citations like [6.2] or [6.2 ¶3]."""
    return set(re.findall(r"\[(\d+(?:\.\d+)*)(?:\s+¶\d+(?:[-–]\d+)?)?\]", reply))


def parse_citations(text: str) -> list[tuple[str, int | None]]:
    """Return [(section_id, para_num|None)] for all citations in text."""
    results = []
    for m in re.finditer(r"\[(\d+\.\d+)(?:\s+¶(\d+)(?:[-–]\d+)?)?\]", text):
        results.append((m.group(1), int(m.group(2)) if m.group(2) else None))
    return results


# [6.3 ¶4], [6.3 ¶4-5], and lists like [6.2 ¶19, 6.2 ¶20] or [6.2 ¶19, ¶20]; groups = the first one
CITE_RE = re.compile(r"\[(\d+\.\d+)(?:\s+¶(\d+)(?:[-–]\d+)?)?(?:[,;]\s*(?:\d+\.\d+\s+)?¶\d+(?:[-–]\d+)?)*\]")


def split_sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\"'(])", " ".join(text.split())) if s]


def strip_bad_citations(text: str, passages: list[Chunk]) -> str:
    """Remove citations to paragraphs (or sections) that were not among the passages."""
    allowed = {(p.section_id, p.para_num) for p in passages}

    def keep(m: re.Match) -> str:
        sec, para = m.group(1), m.group(2)
        ok = (sec, int(para)) in allowed if para else sec in {s for s, _ in allowed}
        return m.group(0) if ok else ""

    cleaned = CITE_RE.sub(keep, text)
    return re.sub(r"[ \t]+([.?!,;])", r"\1", cleaned) if cleaned != text else text


def tidy_answer(text: str, passages: list[Chunk], strict: bool = True) -> tuple[str, list[str]]:
    """Post-check a direct answer: citations must point at retrieved paragraphs, at most
    MAX_CLAIMS claim sentences (strict mode), and it must end with a connecting question."""
    notes = []

    cleaned = strip_bad_citations(text, passages)
    if cleaned != text:
        notes.append("removed citations to paragraphs not retrieved")
    sents = split_sentences(cleaned)
    prompt = sents.pop() if sents and sents[-1].rstrip().endswith("?") else ""
    if strict and len(sents) > MAX_CLAIMS:
        sents = sents[:MAX_CLAIMS]
        notes.append(f"trimmed to {MAX_CLAIMS} sentences")
    if strict:
        uncited = sum(1 for s in sents if not CITE_RE.search(s))
        if uncited:
            notes.append(f"{uncited} sentence(s) without a citation")
    if sents and not any(CITE_RE.search(s) for s in sents):
        sents[-1] = f"{sents[-1]} {passages[0].para_label}"
        notes.append("added missing citation")
    if not prompt:
        prompt = f"Which example in {passages[0].para_label} shows this?"
        notes.append("added connecting question")
    return " ".join(sents + [prompt]), notes


def best_sentences(question: str, chunk: Chunk, retriever: "Retriever", n: int = 2) -> str:
    """The n sentences of a paragraph closest to the question, in reading order (offline answers)."""
    sents = split_sentences(chunk.text)
    if len(sents) <= n:
        return " ".join(sents)
    sim = (cosine_similarity(retriever.word_vec.transform([question]), retriever.word_vec.transform(sents))[0]
           + cosine_similarity(retriever.char_vec.transform([question]), retriever.char_vec.transform(sents))[0])
    pick = sorted(np.argsort(-sim)[:n])
    return " ".join(sents[i] for i in pick)


EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
# 10+ digits with separators (404-555-1234, +1 470 555 0100), or a local 555-1234; not years like 1920-1921
PHONE_RE = re.compile(r"(?<![\w-])\+?\(?\d(?:[\s().-]{0,2}\d){9,}|(?<![\d-])\d{3}[-.\s]\d{4}(?!\d)")


def scrub(text: str) -> str:
    """Drop emails and phone numbers from free text before it is logged."""
    return PHONE_RE.sub("[phone]", EMAIL_RE.sub("[email]", text))


# ── Practice items ────────────────────────────────────────────────────────────
def load_items(path: Path) -> list[dict]:
    """Practice or assessment items from a JSON list; [] if the file doesn't exist."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def item_text(item: dict) -> str:
    return " ".join([item["question"], *item.get("options", [])])


def duplicate_pairs(items: list[dict], others: list[dict],
                    threshold: float = DUPLICATE_THRESHOLD) -> list[tuple[dict, dict, float]]:
    """Pairs (item, other, similarity) whose question + options text is near-identical.

    Used to keep practice questions from duplicating pre/post-test items.
    """
    if not items or not others:
        return []
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    vec.fit([item_text(i) for i in items + others])
    sim = cosine_similarity(vec.transform([item_text(i) for i in items]),
                            vec.transform([item_text(o) for o in others]))
    return [(items[i], others[j], float(sim[i, j]))
            for i in range(len(items)) for j in range(len(others)) if sim[i, j] >= threshold]


def stem(word: str) -> str:
    """Crude suffix stripping so "sucking", "sucks", and "suck" compare equal (leak checks only)."""
    for suf in ("ing", "ed", "es", "s"):
        if len(word) > len(suf) + 3 and word.endswith(suf):
            return word[: -len(suf)]
    return word


def names_answer(text: str, item: dict) -> bool:
    """True if a hint gives away a correct option of 2+ content words: at least NAMES_SHARE of
    its word stems appear in the hint ("suck on a nipple" names "infant sucking on a nipple").

    Words the correct option shares with every other option ("stimulus" in rq7) don't count,
    since a hint can use them without pointing at one option.
    """
    opts = [{stem(w) for w in content_words(o)} for o in item["options"]]
    correct = opts[item["answer_index"]]
    others = [o for i, o in enumerate(opts) if i != item["answer_index"]]
    common = set.intersection(*others) & correct if others else set()
    words = correct - common if len(correct - common) >= 2 else correct
    if len(words) < 2:
        return False
    hint = {stem(w) for w in content_words(text)}
    return len(words & hint) / len(words) >= NAMES_SHARE


# "the answer to the practice question about…": asking for a practice answer outright
PRACTICE_ASK_RE = re.compile(r"\b(?:practice|quiz|review) questions?\b|\banswers? (?:to|for) (?:the|this|that|question|number)\b"
                             r"|\banswer key\b", re.I)


def options_listed(question: str, item: dict) -> int:
    """How many of an item's options appear in the question (every content word of the option)."""
    words = set(content_words(question))
    return sum(1 for o in item.get("options", []) if (w := set(content_words(o))) and w <= words)


def calibration(results: list[dict]) -> dict | None:
    """First-try results [{confidence 1-5, correct, source_paragraph, ...}] → calibration summary.

    Overconfident when mean confidence / 5 exceeds accuracy by OVERCONFIDENCE_GAP; then the
    missed item answered most confidently gives the paragraph to reread.
    revisit:     every missed item, surest first (what to reread).
    underrated:  right answers rated UNDERCONFIDENT or lower (where to trust yourself more).
    """
    if not results:
        return None
    n, right = len(results), sum(r["correct"] for r in results)
    conf = sum(r["confidence"] for r in results) / n
    gap = conf / 5 - right / n
    missed = sorted((r for r in results if not r["correct"]), key=lambda r: -r["confidence"])
    reread = missed[0]["source_paragraph"] if gap >= OVERCONFIDENCE_GAP and missed else None
    underrated = [r for r in results if r["correct"] and r["confidence"] <= UNDERCONFIDENT]
    return dict(n=n, right=right, confidence=conf, gap=gap, reread=reread,
                revisit=missed, underrated=underrated)


# ── Prompts ───────────────────────────────────────────────────────────────────
HINT_SYSTEM = """You are a tutor for ONE course unit. You give HINTS, never answers.
Rules:
- Use ONLY the passages provided. Do not use outside knowledge.
- Never state the definition, the answer, or the key fact the learner is asking for.
- Reply with at most 3 short sentences. First name the key distinction the question
  turns on, as a question, without saying which side is right (e.g., "Is something being
  added or taken away?" or "Does the stimulus come before or after the behavior?").
  If a KEY DISTINCTION is given, build the question on it. Then point to the exact
  paragraph with [section ¶N], e.g. [6.2 ¶3].
- If the learner says "just tell me", stay in hint mode and give a smaller, more
  specific guiding question instead.
- If the passages do not cover the question, reply with exactly: OUT_OF_SCOPE"""

HINT2_SYSTEM = """You are a tutor giving a second, more specific hint. Still do NOT give the answer.
Rules:
- Use ONLY the passages provided. Do not use outside knowledge.
- The first hint already named the key distinction. Now say WHERE in the paragraph to look,
  by what the sentence or example is about (e.g. "the example about the parent and the toy"),
  never by counting sentences, and ask which option fits it. Do not quote or retell what that
  part says about the answer, and do not use the words of the correct option.
- Point to the specific paragraph: [section ¶N], e.g. [6.3 ¶4].
- Reply with at most 3 sentences. Do not state the definition or the answer directly.
- If the passages do not cover the question, reply with exactly: OUT_OF_SCOPE"""

EXPLAIN_SYSTEM = """You are a tutor for ONE course unit. Give a clear, step-by-step explanation.
Rules:
- Use ONLY the passages provided. Do not use outside knowledge.
- Write 2 to 5 numbered steps. End every step with the paragraph it came from:
  [section ¶N], e.g. [6.3 ¶4].
- Keep the whole reply under 150 words.
- If the passages do not cover the question, reply with exactly: OUT_OF_SCOPE"""

ANSWER_SYSTEM = """You are a tutor for ONE course unit. Answer the learner's question directly.
Rules:
- Use ONLY the passages provided. Do not use outside knowledge.
- If the learner applies an idea from the passages to their own example (a phone, a pet),
  answer by applying the passages' definition to it; that counts as covered.
- Answer in at most 3 short sentences. End every sentence with the paragraph it came from:
  [section ¶N], e.g. [6.3 ¶4].
- Then add one final question that sends the learner back to the reading, for example
  "Which example in [6.3 ¶4] shows this?"
- Plain prose: no lists, no headings, no bold.
- EARLIER QUESTIONS, if given, are what the learner already asked. If the new question
  builds on one, link them in a few words inside your first sentence (e.g., "Like the
  negative reinforcement you asked about, ..."). Do not answer the earlier question again.
- If the passages do not cover the question, reply with exactly: OUT_OF_SCOPE"""

SIMPLE_RULE = """
- Explain it for a first-year student with no psychology background: everyday words,
  and define any term you have to use."""

EXAMPLE_SYSTEM = """You are a tutor for ONE course unit. The learner wants an example.
Rules:
- Take the idea ONLY from the passages provided. The example itself may be an everyday
  situation that is not in the passages, but add no facts about the idea.
- Give one concrete everyday example in 1-2 sentences, then one sentence saying which part
  of the example matches which part of the idea, ending with [section ¶N], e.g. [6.2 ¶3].
- End with one question that sends the learner back to the reading.
- Do not copy sentences from the passages. Keep the whole reply under 90 words.
- If the passages do not cover the idea, reply with exactly: OUT_OF_SCOPE"""

WORKED_SYSTEM = """You are a tutor showing the worked answer to a practice question.
Rules:
- Use ONLY the passages provided. Do not use outside knowledge.
- The correct answer is given to you. Write 2 to 4 numbered steps: what the question asks,
  what the reading says, why the correct option fits, and why the most tempting wrong
  option does not. End every step with [section ¶N], e.g. [6.3 ¶4].
- Keep the whole reply under 120 words."""

ANALOGY_SYSTEM = """You are a tutor for ONE course unit. The learner asked for an analogy.
Rules:
- Take the concept itself ONLY from the passages provided. The analogy may use everyday
  situations (that is the point), but do not add facts about the concept that the
  passages don't support.
- Give one vivid everyday analogy in 2-3 sentences, then one sentence mapping the parts of
  the analogy back to the concept, ending with [section ¶N], e.g. [6.2 ¶3].
- Do not copy sentences from the passages. Keep the whole reply under 120 words.
- If the passages do not cover the concept, reply with exactly: OUT_OF_SCOPE"""

SCOPE_SYSTEM = """You decide whether a learner's question is about the course passages provided.
Reply YES if the passages discuss the concept, term, or example the learner asks about
(even if the question is misspelled, loosely worded, or asks for an analogy or example).
Reply NO if the question is about something the passages do not discuss.
Reply with exactly one word: YES or NO."""

SAME_ITEM_SYSTEM = """You compare a learner's chat question with one practice question they have not finished.
Reply YES if the chat question is the practice question itself, reworded or summarized, so that
answering it would give away the practice answer.
Reply NO if the learner is asking about the topic in general, about a different detail, or
about a related idea, even when the practice question covers the same concept.
Reply with exactly one word: YES or NO."""

ATTEMPT_SYSTEM = """You are a tutor evaluating a learner's attempt to answer a question.
Rules:
- Use ONLY the passages provided. Do not use outside knowledge.
- In one sentence, describe what is on track in the attempt.
- Name ONE thing to reconsider, phrased as a guiding question, and cite [section ¶N].
- Never state the full correct answer.
- Maximum 3 sentences total.
- On a new line at the very end, write exactly one of: on track | partly there | not yet
- If the passages do not cover the question, reply with exactly: OUT_OF_SCOPE"""

DIFFERENT_RULE = """
- The learner already read EARLIER ANSWER and asked the same thing again, so it did not land.
  Explain it from a different angle: an everyday example, or a contrast with a related idea
  from the passages. Do not reuse sentences or phrasing from EARLIER ANSWER."""

SUPPORT_CHECK_SYSTEM = """You check whether each numbered sentence is supported by the textbook paragraph it cites.
A sentence is SUPPORTED if everything it claims is stated in or follows directly from its paragraph.
Applying the paragraph's definition to the learner's own example (a phone, a pet) counts as supported.
It is UNSUPPORTED if any claim is missing from its paragraph or contradicts it.
Reply with one line per sentence, in order: the number, then SUPPORTED or UNSUPPORTED (e.g., "1 SUPPORTED")."""

SELF_EXPLAIN_SYSTEM = """You are a tutor giving feedback on a learner's self-explanation of a practice question.
The learner has already seen the worked answer. You are given the question, the correct answer,
the option they are explaining, and their one-sentence explanation.
Rules:
- Use ONLY the passages provided. Do not use outside knowledge.
- In one sentence, say what their explanation gets right, specifically.
- In one sentence, name the one idea it is missing or gets wrong, citing [section ¶N].
  If nothing is missing, say so.
- Encouraging tone; at most 2 sentences.
- On a new line at the very end, write exactly one of: on track | partly there | not yet"""

SELFCHECK_SYSTEM = """You are a tutor creating a self-check question from the explanation passages.
Return ONLY valid JSON — no other text or markdown fences — in this exact format:
{"question": "...", "options": ["A", "B", "C"], "answer_index": 0, "source": "6.x ¶N"}
Rules:
- The question tests conceptual understanding, not just recall of a term.
- The correct answer must be directly supported by the passage.
- answer_index is 0, 1, or 2 (zero-indexed into the options array).
- source identifies the exact paragraph where the answer is found.
- All three options must be plausible; only one is correct."""


def build_prompt(question: str, passages: list[Chunk], focus: Chunk | None = None,
                 history: list[str] = ()) -> str:
    ctx = "\n\n".join(
        f"[{p.section_id} ¶{p.para_num}] ({p.section_title}) {p.text}"
        for p in passages
    )
    reading = f"\n\nTHE LEARNER IS ASKING ABOUT {focus.para_label}; answer about it first." if focus else ""
    earlier = ("\n\nEARLIER QUESTIONS:\n" + "\n".join(f"- {h}" for h in history[-HISTORY_TURNS:])
               if history else "")
    return f"PASSAGES:\n{ctx}{reading}{earlier}\n\nLEARNER QUESTION: {question}"


def practice_question(item: dict) -> str:
    opts = "\n".join(f"{'abcd'[i]}. {o}" for i, o in enumerate(item.get("options", [])))
    return f"{item['question']}\n{opts}".strip()


def build_attempt_prompt(question: str, attempt: str, passages: list[Chunk]) -> str:
    ctx = "\n\n".join(
        f"[{p.section_id} ¶{p.para_num}] ({p.section_title}) {p.text}"
        for p in passages
    )
    return f"PASSAGES:\n{ctx}\n\nLEARNER QUESTION: {question}\n\nLEARNER ATTEMPT: {attempt}"


# ── Reply ─────────────────────────────────────────────────────────────────────
@dataclass
class Reply:
    outcome:    str                         # answered | refused | clarify | fallback | redirect
    text:       str
    sources:    list[tuple[Chunk, float]] = field(default_factory=list)
    top_score:  float = 0.0
    note:       str = ""
    rating:     str = ""   # "on track" | "partly there" | "not yet" (attempt replies)


REFUSAL = (
    "That question doesn't seem to be covered in this unit, so I won't guess. "
    "Try asking about something from the reading, like a term or an example "
    "from one of the sections."
)
PRACTICE_REDIRECT = (
    "That's one of your practice questions. Try it there and use the hints if you get stuck."
)
CLARIFY = (
    "Which part is confusing? Tell me the term, example, or section you're stuck on "
    "(for example, \"the difference between extinction and spontaneous recovery\")."
)


# ── Tutor class ───────────────────────────────────────────────────────────────
class Tutor:
    def __init__(
        self,
        unit_path: Path,
        generate_fn=None,
        cache_path: Path | None = None,
        model_name: str = "",
    ):
        self.unit_title, self.chunks = load_unit(unit_path)
        self.retriever   = Retriever(self.chunks)
        self.generate_fn = generate_fn
        self.model_name  = model_name   # shown to learners in "How answers are checked"
        cache_path       = cache_path or DATA / "cached_answers.json"
        self.cache       = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    # ── answer (the chat) ─────────────────────────────────────────────────────
    def answer(
        self,
        question: str,
        current_section: str | None = None,
        focus: Chunk | None = None,
        action: str | None = None,
        active_practice: list[dict] = (),
        history: list[str] = (),
    ) -> Reply:
        """Chat reply: a short direct answer, every sentence cited, ending with one question
        that sends the learner back to the reading.

        focus:  the paragraph picked with "Ask about ¶N"; always retrieved first.
        action: None for a typed question, or a quick action on the focus paragraph:
                "simple" (explain simply) or "example" (give an example).
        active_practice: open practice items; a typed question that copies or rewords one
                is redirected.
        history: the learner's earlier questions this session, so the answer can build on them.
        """
        question = question.strip()
        if action is None:
            if is_vague(question) and focus is None:
                return Reply("clarify", CLARIFY, note="vague question: no content words")
            item, score, why = self.practice_redirect(question, active_practice)
            if item:
                return Reply("redirect", PRACTICE_REDIRECT, top_score=score,
                             note=f"matches practice item {item['id']} ({why})")

        if focus is not None and (action is not None or is_vague(question)):
            # Quick actions and "what does this mean?" are about the paragraph itself
            question = {
                "simple":  f"Explain {focus.para_label} simply.",
                "example": f"Give an everyday example of the idea in {focus.para_label}.",
            }.get(action, f"{question} (about {focus.para_label})")
            hits = self.retriever.search(focus.text, boost_section=focus.section_id, focus=focus)
            top, scope_note = 1.0, f"about {focus.para_label}"
        elif wants_analogy(question) and not analogy_topic(question) and current_section and focus is None:
            # "Give me an analogy" with no topic → the section the learner is reading
            hits = [(c, 1.0) for c in self.chunks if c.section_id == current_section][:TOP_K]
            top, scope_note = 1.0, "analogy for current section"
        else:
            hits = self.retriever.search(question, boost_section=current_section, focus=focus)
            top  = self.retriever.relevance(question)
            ok, scope_note = self._in_scope(question, [c for c, _ in hits], top)
            if not ok:
                return Reply("refused", REFUSAL, hits, top, note=scope_note)
        passages = [c for c, _ in hits]

        if action == "example":
            system, strict = EXAMPLE_SYSTEM, False
        elif wants_analogy(question):
            system, strict = ANALOGY_SYSTEM, False
        else:
            system, strict = ANSWER_SYSTEM + (SIMPLE_RULE if action == "simple" else ""), True

        if self.generate_fn is None:
            return self._answer_fallback(question, hits, top, "no API key")
        try:
            text = self.generate_fn(system, build_prompt(question, passages, focus, history)).strip()
        except Exception as e:
            return self._answer_fallback(question, hits, top, f"API error: {type(e).__name__}")

        if "OUT_OF_SCOPE" in text:
            return Reply("refused", REFUSAL, hits, top, note="model said passages don't cover it")
        text, notes = tidy_answer(text, passages, strict=strict)
        if strict:   # examples and analogies invent everyday situations on purpose, so only direct answers
            text, support_note = self.check_support(text)
            if text is None:
                return self._answer_fallback(question, hits, top, support_note)
            notes.append(support_note)
        notes = [n for n in [scope_note, *notes] if n]
        return Reply("answered", text, hits, top, "; ".join(notes) or "passed all checks")

    def check_support(self, text: str) -> tuple[str | None, str]:
        """Second pass against hallucination: Gemini judges each cited sentence against the
        paragraph it cites, and unsupported sentences are dropped. Returns (text, note); text
        is None when no claim survives. Fails open (keeps the answer) if the check can't run."""
        sents = split_sentences(text)
        claims, blocks = [], []
        for i, s in enumerate(sents):
            if s.rstrip().endswith("?"):
                continue
            try:
                paras = [self.chunk_at(f"{sec} ¶{n}") for sec, n in parse_citations(s) if n]
            except StopIteration:
                continue
            if paras:
                claims.append(i)
                claim = re.sub(r"\s+([.!,;])", r"\1", CITE_RE.sub("", s)).strip()
                blocks.append(f"SENTENCE {len(claims)}: {claim}\n"
                              f"PARAGRAPH {len(claims)}:\n" + "\n".join(p.text for p in paras))
        if not claims or self.generate_fn is None:
            return text, ""
        try:
            verdict = self.generate_fn(SUPPORT_CHECK_SYSTEM, "\n\n".join(blocks)).upper()
        except Exception as e:
            return text, f"support check skipped: {type(e).__name__}"
        judged = dict(re.findall(r"(?m)^\W*(\d+)\W+(UNSUPPORTED|SUPPORTED)\b", verdict))
        if any(str(k) not in judged for k in range(1, len(claims) + 1)):
            return text, "support check unreadable; kept answer"
        drop = {claims[int(k) - 1] for k, v in judged.items() if v == "UNSUPPORTED" and int(k) <= len(claims)}
        if not drop:
            return text, "support check passed"
        if len(drop) == len(claims):
            return None, f"support check: all {len(drop)} claim(s) unsupported"
        return (" ".join(s for i, s in enumerate(sents) if i not in drop),
                f"support check: removed {len(drop)} unsupported sentence(s)")

    def _answer_fallback(self, question: str, hits: list[tuple[Chunk, float]], top: float, why: str) -> Reply:
        cached = self.cache.get(question.lower().strip(" ?!."), {}).get("answer")
        if cached:
            return Reply("fallback", cached, hits, top, note=f"{why}; served cached answer")
        best = hits[0][0]
        text = (f"{best_sentences(question, best, self.retriever)} {best.para_label}\n\n"
                f"Which example in {best.para_label} shows this?")
        return Reply("fallback", text, hits, top, note=f"{why}; extractive answer")

    def explain_differently(self, question: str, earlier: str, current_section: str | None = None) -> Reply:
        """A second answer to a repeated question: simpler words and a new angle, not the
        earlier wording. Offline, quotes a different paragraph than the earlier answer led with."""
        hits = self.retriever.search(question, boost_section=current_section)
        passages = [c for c, _ in hits]
        if self.generate_fn is not None:
            prompt = build_prompt(question, passages) + f"\n\nEARLIER ANSWER:\n{earlier}"
            try:
                text = self.generate_fn(ANSWER_SYSTEM + SIMPLE_RULE + DIFFERENT_RULE, prompt).strip()
            except Exception as e:
                text, why = "", f"API error: {type(e).__name__}"
            else:
                why = "model said passages don't cover it" if "OUT_OF_SCOPE" in text else ""
            if text and not why:
                text, notes = tidy_answer(text, passages)
                text, support_note = self.check_support(text)
                if text is not None:
                    return Reply("answered", text, hits, 1.0,
                                 "; ".join(n for n in ["explained differently", *notes, support_note] if n))
                why = support_note
        else:
            why = "no API key"
        cited = {(s, n) for s, n in parse_citations(earlier)}
        other = next((c for c in passages if (c.section_id, c.para_num) not in cited), passages[0])
        text = (f"Another way to see it: {best_sentences(question, other, self.retriever)} {other.para_label}\n\n"
                f"How does {other.para_label} connect to the earlier answer?")
        return Reply("fallback", text, hits, 1.0, note=f"{why}; extractive answer from another paragraph")

    def self_explanation(self, item: dict, explaining: int, explanation: str) -> Reply:
        """Feedback on the learner's one-sentence reason why option `explaining` is wrong
        (or, when it is the correct option, why it is right). Asked after the worked answer."""
        source = self.chunk_at(item["source_paragraph"])
        hits = self.retriever.search(practice_question(item), boost_section=source.section_id, focus=source)
        passages = [c for c, _ in hits]
        correct = item["options"][item["answer_index"]]
        fallback = Reply("fallback", f"Compare your reason with {source.para_label}: does it name what "
                         f"makes \"{correct}\" fit that \"{item['options'][explaining]}\" doesn't?"
                         if explaining != item["answer_index"] else
                         f"Compare your reason with {source.para_label}: does it use the paragraph's own definition?",
                         hits, 1.0, note="extractive self-explanation prompt")
        if self.generate_fn is None or not explanation.strip():
            fallback.note = "no API key; " + fallback.note if self.generate_fn is None else fallback.note
            return fallback
        prompt = (build_prompt(practice_question(item), passages) + f"\n\nCORRECT ANSWER: {correct}"
                  f"\nOPTION EXPLAINED: {item['options'][explaining]}\nLEARNER EXPLANATION: {explanation}")
        try:
            text = self.generate_fn(SELF_EXPLAIN_SYSTEM, prompt).strip()
        except Exception as e:
            fallback.note = f"API error: {type(e).__name__}; " + fallback.note
            return fallback
        lines = text.splitlines()
        rating = ""
        if lines and lines[-1].strip().lower() in ("on track", "partly there", "not yet"):
            rating, text = lines[-1].strip().lower(), "\n".join(lines[:-1]).strip()
        text = strip_bad_citations(text, passages)
        r = Reply("answered", text, hits, 1.0, note="self-explanation feedback")
        r.rating = rating
        return r

    # ── practice ──────────────────────────────────────────────────────────────
    def practice_match(self, question: str, items: list[dict]) -> tuple[dict | None, float]:
        """Closest open practice item by char n-gram cosine (stem, or stem + options).

        The item is returned only at or above PRACTICE_REDIRECT_THRESHOLD; the score always is.
        """
        if not items:
            return None, 0.0
        v = self.retriever.char_vec
        q = v.transform([question])
        sim = np.maximum(
            cosine_similarity(q, v.transform([i["question"] for i in items]))[0],
            cosine_similarity(q, v.transform([item_text(i) for i in items]))[0],
        )
        best = int(sim.argmax())
        score = float(sim[best])
        return (items[best] if score >= PRACTICE_REDIRECT_THRESHOLD else None), score

    def practice_redirect(self, question: str, items: list[dict]) -> tuple[dict | None, float, str]:
        """Is this chat question an open practice question? Three layers, cheapest first:

        1. near-copy: char n-gram match at or above PRACTICE_REDIRECT_THRESHOLD;
        2. pasted options: the question names OPTIONS_LISTED or more of an item's options,
           or asks for a practice answer outright ("the answer to the practice question…");
        3. rewording: between PARAPHRASE_FLOOR and the threshold, Gemini judges whether it is
           the same question (only with a key; offline this layer is skipped).
        Returns (item or None, char score, reason).
        """
        item, score = self.practice_match(question, items)
        if item:
            return item, score, f"similarity {score:.2f}"
        for it in items:
            if (k := options_listed(question, it)) >= OPTIONS_LISTED:
                return it, score, f"names {k} of its options"
        if items and PRACTICE_ASK_RE.search(question):
            v = self.retriever.char_vec
            sims = cosine_similarity(v.transform([question]), v.transform([i["question"] for i in items]))[0]
            return items[int(sims.argmax())], score, "asks for a practice answer outright"
        if self.generate_fn is None or score < PARAPHRASE_FLOOR:
            return None, score, ""
        v = self.retriever.char_vec
        sims = cosine_similarity(v.transform([question]), v.transform([i["question"] for i in items]))[0]
        best = items[int(sims.argmax())]
        prompt = (f"PRACTICE QUESTION [{best['source_paragraph']}]:\n{practice_question(best)}"
                  f"\n\nLEARNER CHAT QUESTION: {question}")
        try:
            verdict = self.generate_fn(SAME_ITEM_SYSTEM, prompt).strip().upper()
        except Exception:
            return None, score, ""
        if verdict.startswith("YES"):
            return best, score, f"similarity {score:.2f}; Gemini: same question reworded"
        return None, score, ""

    def repeat_of(self, question: str, earlier: list[str]) -> int | None:
        """Index of an earlier question this one repeats (same content words, or a near-copy
        such as a misspelling), else None."""
        def key(q: str) -> set[str]:
            return {stem(w) for w in content_words(q) if w not in REPEAT_FILLER}

        words = key(question)
        if not earlier or not words:
            return None
        for i, e in enumerate(earlier):
            if key(e) == words:
                return i
        v = self.retriever.char_vec
        sim = cosine_similarity(v.transform([question]), v.transform(earlier))[0]
        best = int(sim.argmax())
        return best if sim[best] >= REPEAT_THRESHOLD else None

    def chunk_at(self, label: str) -> Chunk:
        """'6.3 ¶7' or '[6.3 ¶7]' → that paragraph."""
        m = re.search(r"(\d+\.\d+)\s+¶(\d+)", label)
        return next(c for c in self.chunks
                    if c.section_id == m.group(1) and c.para_num == int(m.group(2)))

    def practice_help(self, item: dict, level: int) -> Reply:
        """Help ladder for a practice question: 1 = hint, 2 = bigger hint, 3 = worked answer.

        Hints go through the leak check and must not name the correct option; a hint that
        fails either falls back to an extractive pointer at the source paragraph.
        """
        source   = self.chunk_at(item["source_paragraph"])
        question = practice_question(item)
        hits     = self.retriever.search(question, boost_section=source.section_id, focus=source)
        passages = [c for c, _ in hits]
        correct  = item["options"][item["answer_index"]]

        if level >= 3:
            fallback = (f"The answer is **{correct}**. "
                        f"{best_sentences(item['question'] + ' ' + correct, source, self.retriever)} "
                        f"{source.para_label}")
            if self.generate_fn is None:
                return Reply("fallback", fallback, hits, 1.0, note="no API key; extractive worked answer")
            try:
                text = self.generate_fn(
                    WORKED_SYSTEM, build_prompt(question, passages) + f"\n\nCORRECT ANSWER: {correct}"
                ).strip()
            except Exception as e:
                return Reply("fallback", fallback, hits, 1.0,
                             note=f"API error: {type(e).__name__}; extractive worked answer")
            cleaned = strip_bad_citations(text, passages)
            note = "removed citations to paragraphs not retrieved" if cleaned != text else "passed all checks"
            if not CITE_RE.search(cleaned):
                cleaned += f" {source.para_label}"
            return Reply("answered", cleaned, hits, 1.0, note)

        mode = "hint" if level == 1 else "hint2"
        distinction = item.get("distinction", "") if level == 1 else ""

        def fallback(why: str) -> Reply:
            if distinction:   # the hand-written key distinction beats a generic pointer
                return Reply("fallback", f"{distinction} Reread {source.para_label}.", hits, 1.0,
                             note=f"{why}; key distinction from practice.json")
            return self._fallback(question, mode, hits, 1.0, why)

        if self.generate_fn is None:
            return fallback("no API key")
        prompt = build_prompt(question, passages)
        if distinction:
            prompt += f"\n\nKEY DISTINCTION: {distinction}"
        try:
            text = self.generate_fn(HINT_SYSTEM if level == 1 else HINT2_SYSTEM, prompt).strip()
        except Exception as e:
            return fallback(f"API error: {type(e).__name__}")
        if "OUT_OF_SCOPE" in text:
            return fallback("model said OUT_OF_SCOPE")
        if leaks_text(text, passages):
            return fallback(f"leak check: level-{level} hint copied text")
        if names_answer(text, item):
            return fallback(f"level-{level} hint named the answer")
        if not cited_ids(text):
            text += f" Reread {source.para_label}."
        return Reply("answered", text, hits, 1.0, "passed all checks")

    # ── ask (hint ladder by level; kept for the gold-set eval) ──────────────
    def ask(
        self,
        question: str,
        mode: str = "hint",
        help_level: int = 1,
        current_section: str | None = None,
    ) -> Reply:
        """Return a hint (level 1 or 2) or explanation (level 3).

        Backward-compatible: mode="hint" → level 1, mode="explain" → level 3.
        mode="hint2" or help_level=2 → level-2 hint.
        """
        if mode == "explain":
            help_level = 3
        elif mode == "hint2":
            help_level = 2
        # mode="hint" keeps help_level as passed (default 1)

        question = question.strip()
        if is_vague(question):
            return Reply("clarify", CLARIFY, note="vague question: no content words")

        analogy = wants_analogy(question) and help_level in (1, 2)
        if analogy and not analogy_topic(question) and current_section:
            # "Give me an analogy" with no topic → the section the learner is reading
            hits = [(c, 1.0) for c in self.chunks if c.section_id == current_section][:TOP_K]
            top, scope_note = 1.0, "analogy for current section"
        else:
            hits = self.retriever.search(question, boost_section=current_section)
            top  = self.retriever.relevance(question)
            ok, scope_note = self._in_scope(question, [c for c, _ in hits], top)
            if not ok:
                return Reply("refused", REFUSAL, hits, top, note=scope_note)
        passages = [c for c, _ in hits]

        cache_mode = {1: "hint", 2: "hint2"}.get(help_level, "explain")
        if self.generate_fn is None:
            return self._fallback(question, cache_mode, hits, top, "no API key")

        system_map = {1: HINT_SYSTEM, 2: HINT2_SYSTEM, 3: EXPLAIN_SYSTEM}
        system = ANALOGY_SYSTEM if analogy else system_map.get(help_level, HINT_SYSTEM)

        try:
            text = self.generate_fn(system, build_prompt(question, passages)).strip()
        except Exception as e:
            return self._fallback(question, cache_mode, hits, top, f"API error: {type(e).__name__}")

        if "OUT_OF_SCOPE" in text:
            return Reply("refused", REFUSAL, hits, top, note="model said passages don't cover it")

        notes      = [scope_note] if scope_note else []
        valid_ids  = {p.section_id for p in passages}

        if help_level in (1, 2):
            if leaks_text(text, passages):
                return self._fallback(
                    question, cache_mode, hits, top,
                    f"leak check: level-{help_level} hint copied text",
                )
            if not cited_ids(text) & valid_ids:
                text += f" Reread [{passages[0].section_id} ¶{passages[0].para_num}]."
                notes.append("added missing section pointer")
        else:  # explain
            bad = cited_ids(text) - valid_ids
            if bad:
                notes.append(f"removed citations not in retrieved passages: {sorted(bad)}")
                for b in bad:
                    text = re.sub(rf"\[{re.escape(b)}(?:\s+¶\d+(?:[-–]\d+)?)?\]", "", text)

        return Reply("answered", text, hits, top, "; ".join(notes) or "passed all checks")

    # ── evaluate_attempt ──────────────────────────────────────────────────────
    def evaluate_attempt(
        self,
        question: str,
        attempt: str,
        current_section: str | None = None,
    ) -> Reply:
        """Evaluate a learner's self-produced answer against the passages."""
        question = question.strip()
        hits     = self.retriever.search(question, boost_section=current_section)
        top      = self.retriever.relevance(question)
        passages = [c for c, _ in hits]

        ok, scope_note = self._in_scope(question, passages, top)
        if not ok:
            return Reply("refused", REFUSAL, hits, top, note=f"attempt: {scope_note}")

        if self.generate_fn is None:
            r = Reply(
                "fallback",
                f"Good effort! Compare your answer with [{passages[0].section_id} "
                f"¶{passages[0].para_num}] — does it capture the key distinction?",
                hits, top, note="no API key; extractive feedback",
            )
            r.rating = "partly there"
            return r

        try:
            text = self.generate_fn(
                ATTEMPT_SYSTEM,
                build_attempt_prompt(question, attempt, passages),
            ).strip()
        except Exception as e:
            r = Reply(
                "fallback",
                f"Couldn't evaluate right now. Compare your answer with "
                f"[{passages[0].section_id} ¶{passages[0].para_num}].",
                hits, top, note=f"API error: {type(e).__name__}",
            )
            r.rating = ""
            return r

        if "OUT_OF_SCOPE" in text:
            return Reply("refused", REFUSAL, hits, top, note="attempt: out of scope")

        # Parse rating from last line
        rating = ""
        lines  = text.strip().splitlines()
        if lines:
            last = lines[-1].strip().lower()
            if last in ("on track", "partly there", "not yet"):
                rating = last
                text   = "\n".join(lines[:-1]).strip()

        if leaks_text(text, passages):
            text = (
                "You're on the right track! Check whether your answer captures "
                f"the key distinction. Compare with [{passages[0].section_id} "
                f"¶{passages[0].para_num}]."
            )
            rating = "partly there"

        r = Reply("answered", text, hits, top, note="attempt evaluated")
        r.rating = rating
        return r

    # ── selfcheck ─────────────────────────────────────────────────────────────
    def selfcheck(
        self,
        question: str,
        passages: list[Chunk],
        explanation: str,
    ) -> dict | None:
        """Generate a MC self-check question. Returns None if generation fails."""
        if self.generate_fn is None:
            return None
        prompt = (
            f"EXPLANATION JUST GIVEN:\n{explanation}\n\n"
            "PASSAGES USED:\n"
            + "\n\n".join(
                f"[{p.section_id} ¶{p.para_num}] {p.text}" for p in passages
            )
        )
        try:
            raw  = self.generate_fn(SELFCHECK_SYSTEM, prompt).strip()
            raw  = re.sub(r"^```(?:json)?\s*", "", raw)
            raw  = re.sub(r"\s*```$", "", raw)
            data = json.loads(raw)
            assert "question" in data and "options" in data
            assert "answer_index" in data and "source" in data
            assert len(data["options"]) == 3
            assert data["answer_index"] in (0, 1, 2)
            # Models tend to put the right answer first; shuffle so position gives nothing away.
            correct = data["options"][data["answer_index"]]
            random.shuffle(data["options"])
            data["answer_index"] = data["options"].index(correct)
            return data
        except Exception:
            return None   # bad JSON or assertion → skip quietly

    # ── scope gate ────────────────────────────────────────────────────────────
    def _in_scope(self, question: str, passages: list[Chunk], top: float) -> tuple[bool, str]:
        """Clear scores pass, unit phrases pass, and the gray zone goes to Gemini.

        Returns (ok, note). Without a key the gray zone is declined, as before.
        """
        if top >= RELEVANCE_THRESHOLD:
            return True, ""
        if self.retriever.has_phrase(question):
            return True, f"relevance {top:.2f} < threshold, but matched a unit phrase"
        below = f"relevance {top:.2f} < threshold {RELEVANCE_THRESHOLD}"
        if top < SCOPE_FLOOR or self.generate_fn is None:
            return False, below
        try:
            verdict = self.generate_fn(SCOPE_SYSTEM, build_prompt(question, passages)).strip().upper()
        except Exception as e:
            return False, f"{below}; scope check failed ({type(e).__name__})"
        if verdict.startswith("YES"):
            return True, f"{below}; Gemini scope check: yes"
        return False, f"{below}; Gemini scope check: no"

    # ── fallback ──────────────────────────────────────────────────────────────
    def _fallback(
        self,
        question: str,
        mode: str,
        hits: list[tuple[Chunk, float]],
        top: float,
        why: str,
    ) -> Reply:
        key    = question.lower().strip(" ?!.")
        cached = self.cache.get(key, {}).get(mode)
        if cached:
            return Reply("fallback", cached, hits, top, note=f"{why}; served cached answer")
        best = hits[0][0]
        if mode == "hint2":
            opening = " ".join(best.text.split()[:8])
            text = (
                f"Look closely at [{best.section_id} ¶{best.para_num}], the paragraph that starts "
                f"\"{opening}…\". Which key term in it matches what you're asking, and what does "
                "the paragraph say happens to the behavior?"
            )
        elif mode == "hint":
            text = (
                f"Reread [{best.section_id} ¶{best.para_num}] {best.section_title}. "
                "As you read, ask yourself: how does that paragraph's example "
                "connect to your question?"
            )
        else:
            text = (
                f"I can't generate a full explanation right now. The most relevant "
                f"passage is [{best.section_id} ¶{best.para_num}] {best.section_title}; "
                "open \"Show sources\" below to read it."
            )
        return Reply("fallback", text, hits, top, note=f"{why}; extractive fallback")


# ── Gemini client ─────────────────────────────────────────────────────────────
def make_gemini(api_key: str, model: str = "gemini-3.8-flash"):
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    def generate(system: str, prompt: str) -> str:
        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=0.3,
                # Thinking tokens count toward max_output_tokens; at the default level they
                # used ~380 of 400 and cut replies off mid-sentence.
                thinking_config=types.ThinkingConfig(thinking_level="low"),
                max_output_tokens=1024,
            ),
        )
        return resp.text or ""

    return generate
