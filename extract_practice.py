#!/usr/bin/env python3
"""Extract Chapter 6 Review Questions from Psychology2e_WEB.pdf into data/practice.json.

The 16 multiple-choice Review Questions sit on PDF pages 220-222 (book pages 208-209),
between the "Review Questions" and "Critical Thinking Questions" headings. Each item is
matched to the section and paragraph that support it with the tutor's own retriever
(query = question stem + correct option).

This PDF has no answer key (the official one is in OpenStax's instructor resources), so
answers come from PROPOSED_ANSWERS below. Each one is verified against the chapter: the
correct option is stated in its source paragraph (or, for rq4, in Table 6.1), and Gemini,
given only that paragraph, picks the same option (audit.py, section 1). Verified items are
written with "answer_verified": true.

Items that duplicate a pre/post-test item in data/assessments.json are dropped.

Usage:  python extract_practice.py [--pdf Psychology2e_WEB.pdf] [--out data/practice.json]
Needs:  pip install pymupdf   (extraction only; the app itself does not need it)
"""
import argparse
import json
import re
from pathlib import Path

import pymupdf

import tutor as T

ROOT = Path(__file__).parent
FIRST_PAGE, LAST_PAGE = 220, 222          # 1-indexed PDF pages holding the Review Questions
RUNNING = re.compile(r"^(\d+\s+6 • .*|6 • .*\s+\d+|Access for free at openstax\.org)$")

# Book question number -> correct option letter, read off the chapter text (see docstring).
PROPOSED_ANSWERS = {
    1: "c", 2: "b", 3: "a", 4: "d", 5: "b", 6: "d", 7: "a", 8: "b",
    9: "d", 10: "b", 11: "a", 12: "b", 13: "b", 14: "d", 15: "a", 16: "c",
}
# All 16 checked against the chapter text on 2026-11-20 (see docstring); 16/16 also agree with
# Gemini's paragraph-only pick. Re-check any item whose question, options, or source change.
VERIFIED = set(range(1, 17))
# Hand-checked fixes where the retriever's best paragraph doesn't hold the answer
SOURCE_OVERRIDES = {
    4: "6.2 ¶6",    # the answer is in Table 6.1 (dropped); ¶6: the neutral stimulus comes "immediately before"
    7: "6.2 ¶18",   # retriever picked the CS definition; ¶18 defines extinction
    13: "6.4 ¶1",   # ¶6 lists kinds of models; ¶1 says the people imitated "are called models"
}
# First hint for each question: the distinction it turns on, worded as a question, never the answer
KEY_DISTINCTIONS = {
    1: "Which of these does a person have to be taught, and which happens without any teaching?",
    2: "Is learning something you are born with, or something that comes from what happens to you?",
    3: "Which two kinds of learning work by connecting things: a stimulus with a stimulus, or a behavior with its consequence?",
    4: "Does the stimulus come before the behavior or after it?",
    5: "Before any conditioning, does this stimulus cause a response or not?",
    6: "Is Albert reacting to things that are similar, or telling different things apart?",
    7: "Which stimulus keeps showing up, and which one stops coming with it?",
    8: "Did the dogs salivate from an unlearned reflex, or from something they learned to expect?",
    9: "Is something being added or taken away, and is the goal to increase or decrease the behavior?",
    10: "Which of these meets a biological need on its own, and which matters only for what it can get you?",
    11: "Is the behavior rewarded all at once, or step by step as it gets closer to the goal?",
    12: "Is the payout based on the number of plays or on time, and is it predictable or not?",
    13: "What does the chapter call the person being watched, as opposed to the person watching?",
    14: "Did the children copy what they saw, or do something different from it?",
    15: "What has to happen first before you can remember, repeat, or want to repeat a behavior?",
    16: "Which researcher in this section studied learning by watching others, rather than by rewards or paired stimuli?",
}


def review_text(pdf_path: Path) -> str:
    doc = pymupdf.open(pdf_path)
    text = "\n".join(doc[p - 1].get_text() for p in range(FIRST_PAGE, LAST_PAGE + 1))
    start, end = text.index("Review Questions"), text.index("Critical Thinking Questions")
    lines = [l.strip() for l in text[start:end].splitlines()[1:]]
    return "\n".join(l for l in lines if l and not RUNNING.match(l))


def parse(text: str) -> list[dict]:
    items = []
    for m in re.finditer(r"^(\d+) \. (.+?)(?=^\d+ \. |\Z)", text, flags=re.M | re.S):
        num, body = int(m.group(1)), m.group(2)
        parts = re.split(r"^([a-d])\. ", body, flags=re.M)
        stem = " ".join(parts[0].split())
        options = [" ".join(p.split()) for p in parts[2::2]]
        if len(options) != 4:
            raise ValueError(f"Question {num}: expected 4 options, got {len(options)}")
        items.append({"num": num, "question": stem, "options": options})
    if len(items) != len(PROPOSED_ANSWERS):
        raise ValueError(f"Expected {len(PROPOSED_ANSWERS)} questions, parsed {len(items)}")
    return items


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf", default=str(ROOT / "Psychology2e_WEB.pdf"))
    ap.add_argument("--out", default=str(T.DATA / "practice.json"))
    args = ap.parse_args()

    tutor = T.Tutor(T.DATA / "unit.md")
    assessments = T.load_items(T.DATA / "assessments.json")
    practice = []
    for it in parse(review_text(Path(args.pdf))):
        answer_index = "abcd".index(PROPOSED_ANSWERS[it["num"]])
        query = it["question"].replace("________", " ") + " " + it["options"][answer_index]
        best = (tutor.chunk_at(SOURCE_OVERRIDES[it["num"]]) if it["num"] in SOURCE_OVERRIDES
                else tutor.retriever.search(query, k=1)[0][0])
        practice.append({
            "id": f"rq{it['num']}",
            "section": best.section_id,
            "question": it["question"],
            "options": it["options"],
            "answer_index": answer_index,
            "source_paragraph": f"{best.section_id} ¶{best.para_num}",
            "distinction": KEY_DISTINCTIONS.get(it["num"], ""),
            "origin": "book",
            "answer_verified": it["num"] in VERIFIED,
        })

    dupes = T.duplicate_pairs(practice, assessments)
    for p, a, score in dupes:
        print(f"DROP {p['id']}: duplicates assessment item {a.get('id', '?')} (similarity {score:.2f})")
    dropped = {p["id"] for p, _, _ in dupes}
    practice = [p for p in practice if p["id"] not in dropped]

    Path(args.out).write_text(json.dumps(practice, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for p in practice:
        print(f"{p['id']:5} {p['source_paragraph']:9} {p['options'][p['answer_index']][:28]:28} {p['question'][:60]}")
    unverified = [p["id"] for p in practice if not p["answer_verified"]]
    print(f"Wrote {len(practice)} items to {args.out}"
          + (f"; answers NOT verified against a key: {', '.join(unverified)}" if unverified else ""))


if __name__ == "__main__":
    main()
