"""Run the 30-question gold set and the practice checks; report fidelity numbers.

Usage:
  python eval_gold_set.py                       # safeguards + retrieval + practice checks (no key)
  python eval_gold_set.py --generate            # key from GEMINI_API_KEY or .streamlit/secrets.toml
  python eval_gold_set.py --generate --write-cache

Sections of the report:
  1. Chat (answer()): answer / decline / clarify behavior and retrieval on the gold set.
     With --generate, also direct-answer form: cited sentences, trimming, connecting question.
  2. Practice redirect: threshold check on data/practice_overlap_set.csv plus the gold set's
     answerable questions (which must not be redirected), then the looser rewordings in
     data/paraphrase_set.csv, reported by author ("draft" = written by the developer; add
     rows from other people with their own author tag). With --generate the Gemini
     "same question, reworded?" layer runs too.
  3. Practice items: count per section, unverified answers, duplicates of pre/post-test items.
     With --generate, also the hint ladder's leak / names-the-answer pass rates.
"""
import argparse
import csv
import json
import os
import time
import tomllib
from collections import Counter
from pathlib import Path

import tutor as T

ROOT = Path(__file__).parent


def setting(name: str, default=None):
    """Environment variable first, then .streamlit/secrets.toml (as the app does)."""
    secrets = ROOT / ".streamlit" / "secrets.toml"
    from_file = tomllib.loads(secrets.read_text()) if secrets.exists() else {}
    return os.environ.get(name) or from_file.get(name) or default


def chat_eval(tutor, gold, gen, args, cache):
    rows = []
    behavior_ok = section_ok = section_n = 0
    for g in gold:
        r = tutor.answer(g["question"])
        behavior = {"answered": "answer", "fallback": "answer", "refused": "refuse",
                    "clarify": "clarify"}.get(r.outcome, r.outcome)
        ok = behavior == g["expected_behavior"]
        behavior_ok += ok
        sec_hit = ""
        if g["expected_section"]:
            section_n += 1
            sec_hit = g["expected_section"] in {c.section_id for c, _ in r.sources[:T.TOP_K]}
            section_ok += bool(sec_hit)
        rows.append({
            "id": g["id"], "type": g["type"], "question": g["question"],
            "expected_behavior": g["expected_behavior"], "actual_outcome": r.outcome,
            "behavior_correct": ok, "expected_section": g["expected_section"],
            "top_section": r.sources[0][0].section_id if r.sources else "",
            "expected_section_retrieved": sec_hit, "top_score": f"{r.top_score:.3f}",
            "safeguard_note": r.note, "reply": r.text,
            "R1_correct": "", "R1_citation_supports": "",
        })
        if args.write_cache and r.outcome == "answered" and g["expected_behavior"] == "answer":
            cache.setdefault(g["question"].lower().strip(" ?!."), {})["answer"] = r.text
        if gen:
            time.sleep(args.sleep)

    print(f"\n1. CHAT  |  unit: {tutor.unit_title}  |  decline threshold {T.RELEVANCE_THRESHOLD}")
    print(f"Correct behavior (answer/decline/clarify): {behavior_ok}/{len(gold)}")
    print(f"Right section in top-{T.TOP_K}: {section_ok}/{section_n}")
    for t in ["in_scope", "misspelled", "extract_answer", "off_topic", "other_chapter", "vague"]:
        sub = [r for r in rows if r["type"] == t]
        if sub:
            print(f"  {t:20s} {sum(1 for r in sub if r['behavior_correct'])}/{len(sub)}")
    for r in rows:
        if not r["behavior_correct"]:
            print(f"  MISS #{r['id']} ({r['type']}): expected {r['expected_behavior']}, "
                  f"got {r['actual_outcome']} (score {r['top_score']})")
    ans = [float(r["top_score"]) for r in rows if r["expected_behavior"] == "answer"]
    dec = [float(r["top_score"]) for r in rows if r["expected_behavior"] == "refuse"]
    print(f"Decline margin: lowest answerable score {min(ans):.3f} vs highest off-topic "
          f"score {max(dec):.3f} (threshold {T.RELEVANCE_THRESHOLD})")

    generated = [r for r in rows if r["actual_outcome"] == "answered"]
    if generated:
        n = len(generated)
        print(f"Direct answers generated: {n}")
        print(f"  every sentence cited:            {sum('without a citation' not in r['safeguard_note'] for r in generated)}/{n}")
        print(f"  within {T.MAX_CLAIMS} sentences as written:     {sum('trimmed' not in r['safeguard_note'] for r in generated)}/{n}")
        print(f"  ended with a connecting question: {sum('added connecting question' not in r['safeguard_note'] for r in generated)}/{n}")
        print(f"  all citations to retrieved ¶:    {sum('removed citations' not in r['safeguard_note'] for r in generated)}/{n}")
    return rows


def redirect_eval(tutor, gold, practice):
    tuning = list(csv.DictReader(open(T.DATA / "practice_overlap_set.csv", encoding="utf-8")))
    seen = {r["question"] for r in tuning}
    tuning += [{"question": g["question"], "practice_id": "", "expected": "answer"}
               for g in gold if g["expected_behavior"] == "answer" and g["question"] not in seen]
    scored = [(r, *tutor.practice_redirect(r["question"], practice)[:2]) for r in tuning]
    pos = [(r, it, s) for r, it, s in scored if r["expected"] == "redirect"]
    neg = [(r, it, s) for r, it, s in scored if r["expected"] == "answer"]
    caught = [x for x in pos if x[1] and x[1]["id"] == x[0]["practice_id"]]
    false_pos = [x for x in neg if x[1]]

    print(f"\n2. PRACTICE REDIRECT  |  threshold {T.PRACTICE_REDIRECT_THRESHOLD} (all {len(practice)} items open)")
    print(f"Copies / close rewordings redirected to the right item: {len(caught)}/{len(pos)}")
    print(f"Ordinary questions wrongly redirected:                  {len(false_pos)}/{len(neg)}")
    print(f"Margin: lowest copy {min(s for _, _, s in pos):.3f} vs highest ordinary question "
          f"{max(s for _, _, s in neg):.3f}")
    for r, it, s in pos:
        if not (it and it["id"] == r["practice_id"]):
            print(f"  MISSED ({s:.3f}): {r['question'][:80]}")
    for r, it, s in false_pos:
        print(f"  FALSE REDIRECT ({s:.3f} → {it['id']}): {r['question'][:80]}")
    either = [(r, it, s) for r, it, s in scored if r["expected"] == "either"]
    if either:
        print("Same-idea questions (not scored; asked differently from the practice item):")
        for r, it, s in either:
            print(f"  {s:.3f} {'redirect' if it else 'answer  '} {r['question']}")

    path = T.DATA / "paraphrase_set.csv"
    if not path.exists():
        return
    para = list(csv.DictReader(open(path, encoding="utf-8")))
    layer = "char match + options + Gemini" if tutor.generate_fn else "char match + options (offline)"
    print(f"\nLooser rewordings (data/paraphrase_set.csv, {layer}):")
    for author in sorted({r["author"] for r in para}):
        rows = [r for r in para if r["author"] == author]
        hits = [(r, *tutor.practice_redirect(r["question"], practice)) for r in rows]
        ok = [h for h in hits if h[1] and h[1]["id"] == h[0]["practice_id"]]
        print(f"  {author:10s} redirected to the right item: {len(ok)}/{len(rows)}")
        for r, it, sc, why in hits:
            if not (it and it["id"] == r["practice_id"]):
                print(f"    MISSED ({sc:.3f}): {r['question'][:80]}")


def practice_eval(tutor, practice, gen, args):
    print(f"\n3. PRACTICE ITEMS  |  {len(practice)} items: "
          + ", ".join(f"{k} {v}" for k, v in sorted(Counter(i["section"] for i in practice).items())))
    unverified = [i["id"] for i in practice if not i.get("answer_verified")]
    if unverified:
        print(f"Answers not yet verified against the book's answer key: {len(unverified)}/{len(practice)}")
    assessments = T.load_items(T.DATA / "assessments.json")
    if assessments:
        dupes = T.duplicate_pairs(practice, assessments)
        print(f"Duplicates of pre/post-test items (threshold {T.DUPLICATE_THRESHOLD}): {len(dupes)}")
        for p, a, s in dupes:
            print(f"  {p['id']} ~ {a.get('id', '?')} ({s:.2f})")
    else:
        print("Duplicates of pre/post-test items: not checked (no data/assessments.json)")

    if not gen:
        return
    passed, total = Counter(), Counter()
    for item in practice:
        for level in (1, 2):
            r = tutor.practice_help(item, level)
            total[level] += 1
            passed[level] += r.outcome == "answered"
            if r.outcome != "answered":
                print(f"  L{level} {item['id']}: {r.note}")
            time.sleep(args.sleep)
    for level in (1, 2):
        print(f"Level-{level} hints passing leak + names-the-answer checks: {passed[level]}/{total[level]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--unit",    default=str(
        T.DATA / "unit.md" if (T.DATA / "unit.md").exists() else T.DATA / "sample_unit.md"
    ))
    ap.add_argument("--generate",    action="store_true")
    ap.add_argument("--write-cache", action="store_true")
    ap.add_argument("--model",       default=setting("GEMINI_MODEL", "gemini-3.8-flash"))
    ap.add_argument("--sleep",       type=float, default=4.0)
    args = ap.parse_args()

    gen = None
    if args.generate:
        key = setting("GEMINI_API_KEY")
        if not key:
            raise SystemExit("Set GEMINI_API_KEY (environment or .streamlit/secrets.toml) to use --generate")
        gen = T.make_gemini(key, args.model)

    tutor    = T.Tutor(Path(args.unit), generate_fn=gen)
    gold     = list(csv.DictReader(open(T.DATA / "gold_set.csv", encoding="utf-8")))
    practice = T.load_items(T.DATA / "practice.json")
    cache    = {}

    rows = chat_eval(tutor, gold, gen, args, cache)
    if practice:
        redirect_eval(tutor, gold, practice)
        practice_eval(tutor, practice, gen, args)

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    out_path = out / "gold_results.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    if args.write_cache and cache:
        old = json.loads((T.DATA / "cached_answers.json").read_text()) if (T.DATA / "cached_answers.json").exists() else {}
        for q, modes in cache.items():
            old.setdefault(q, {}).update(modes)
        (T.DATA / "cached_answers.json").write_text(json.dumps(old, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
