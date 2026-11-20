#!/usr/bin/env python3
"""Extract Chapter 6 ("Learning") from Psychology2e_WEB.pdf into data/unit.md.

Uses PyMuPDF text blocks and font information, so each chunk in unit.md is a
real paragraph from the book (not a whole page):

  body paragraphs      IBMPlexSerif-Regular 9pt       -> kept
  section headings     Archivo-SemiBold 13pt "6.x …"  -> "## 6.x Title"
  subsection headings  Archivo-SemiBold 9–10.8pt      -> "### Title" (tutor strips these)
  feature boxes        Archivo-Medium/Regular 10.8pt title + Sans body
                       (e.g. "Classical Conditioning at Stingray City") -> kept, prefixed "Example — Title:"
  figure captions, tables, LINK TO LEARNING boxes, learning objectives,
  chapter outline/intro, running headers and footers -> dropped
Paragraphs split across a page or column break are re-joined.

Usage:  python extract_unit.py [--pdf Psychology2e_WEB.pdf] [--out data/unit.md]
Needs:  pip install pymupdf   (extraction only; the app itself does not need it)
"""
import argparse
import re
from pathlib import Path

import pymupdf

ROOT = Path(__file__).parent
FIRST_PAGE, LAST_PAGE = 193, 218          # 1-indexed PDF pages of the chapter body
SECTION_RE = re.compile(r"^6\.([1-4])\s+(.+)$")
TITLES = {"6.1": "What Is Learning?", "6.2": "Classical Conditioning",
          "6.3": "Operant Conditioning", "6.4": "Observational Learning (Modeling)"}


def clean(text: str) -> str:
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)      # re-join words hyphenated at line ends
    text = text.replace("­", "")                    # soft hyphens
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?)])", r"\1", text)       # "reinforcement ," -> "reinforcement,"
    return re.sub(r"\(\s+", "(", text).strip()


def block_parts(block):
    """Return (font, size, text) for the block, using its first non-empty span for style."""
    spans = [s for line in block["lines"] for s in line["spans"] if s["text"].strip()]
    if not spans:
        return None
    lines = [" ".join(s["text"] for s in line["spans"]).strip() for line in block["lines"]]
    # join lines; keep a hyphen marker so clean() can re-join split words
    text = " ".join(l for l in lines if l)
    return spans[0]["font"], round(spans[0]["size"], 1), spans, text


def ends_sentence(text: str) -> bool:
    text = text.strip()
    if re.search(r"(\b[A-Z]\.|\bU\.S\.|\be\.g\.|\bi\.e\.|\bet al\.)$", text):   # initials / abbreviations
        return False
    return bool(re.search(r"[.?!:]['\"”’)]*$", text))


def continues_previous(text: str) -> bool:
    """True if a block clearly continues the previous paragraph (lowercase start, or closes an open parenthesis)."""
    if text[:1].islower():
        return True
    close, open_ = text.find(")"), text.find("(")
    return 0 <= close < 80 and (open_ == -1 or close < open_)


LINK_BOX = re.compile(r"openstax\.org/l/|^(Watch|View) (this|scenes)|^As you watch the video")


def extract(pdf_path: Path) -> dict[str, list[str]]:
    doc = pymupdf.open(pdf_path)
    sections: dict[str, list[str]] = {k: [] for k in TITLES}
    current = None            # None until the "6.1" heading (skips outline + intro)
    mode = None               # None = body, "feature" = keep Sans text, "skip" = drop Sans text

    def add(text: str, kind: str):
        if current is None or not text:
            return
        paras = sections[current]
        prev_kind = getattr(add, "last_kind", None)
        # re-join a paragraph that was split by a page/column break
        if paras and prev_kind == kind and not paras[-1].startswith("### ") and (
                not ends_sentence(paras[-1]) or continues_previous(text)):
            paras[-1] = clean(paras[-1] + " " + text)
        else:
            paras.append(clean(text))
        add.last_kind = kind

    for pno in range(FIRST_PAGE, LAST_PAGE + 1):
        blocks = [b for b in doc[pno - 1].get_text("dict")["blocks"] if b["type"] == 0]
        blocks.sort(key=lambda b: (round(b["bbox"][1]), b["bbox"][0]))
        for b in blocks:
            parts = block_parts(b)
            if not parts:
                continue
            font, size, spans, text = parts
            t = text.strip()

            if size <= 7.6:                                  # running header / footer
                continue
            if size >= 12.5 and "SemiBold" in font:          # section heading
                m = SECTION_RE.match(t)
                if m:
                    current, mode = f"6.{m.group(1)}", None
                    add.last_kind = None
                continue
            if size >= 14:                                    # chapter title, outline header
                continue
            if font.startswith("ArchivoBlack"):               # LINK TO LEARNING / EVERYDAY CONNECTION label
                mode = "skip" if "LINK" in t.upper() else "feature"
                continue
            if font.startswith("Archivo") and size >= 10.5 and ("Medium" in font or font.endswith("-Regular")):
                # feature box: title spans in Archivo, body spans after it
                title = " ".join(s["text"] for s in spans if s["font"] == font).strip()
                body = " ".join(s["text"] for s in spans if s["font"] != font).strip()
                mode = "feature"
                add.last_kind = None
                add(f"Example — {clean(title)}: {body}", "feature")
                continue
            if "SemiBold" in font:                            # subsection heading
                mode = None
                if current:
                    sections[current].append(f"### {clean(t)}")
                    add.last_kind = None
                continue
            if font.startswith("Archivo-Bold") or "Sans-Bold" in font:
                # FIGURE captions, LEARNING OBJECTIVES, TABLE titles -> skip what follows in Sans
                mode = "skip"
                continue
            if "Serif" in font:                               # body text
                if t.startswith("By the end of this section") or t.startswith("•"):
                    continue
                mode = None
                if LINK_BOX.search(t):                        # video-link boxes set in body font
                    add.last_kind = None
                    continue
                add(t, "body")
                continue
            if "Sans" in font:                                # feature body, caption tail, table cell, link box
                if mode == "feature" and not t.startswith("(credit"):
                    add(t, "feature")
                continue
    return sections


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf", default=str(ROOT / "Psychology2e_WEB.pdf"))
    ap.add_argument("--out", default=str(ROOT / "data" / "unit.md"))
    args = ap.parse_args()

    sections = extract(Path(args.pdf))
    lines = ["# Chapter 6: Learning", "",
             "<!-- Source: Psychology 2e, OpenStax (openstax.org/details/books/psychology-2e), CC BY 4.0. -->", ""]
    for sid, title in TITLES.items():
        lines += [f"## {sid} {title}", ""]
        for p in sections[sid]:
            lines += [p, ""]
        paras = [p for p in sections[sid] if not p.startswith("### ")]
        words = [len(p.split()) for p in paras]
        print(f"{sid} {title}: {len(paras)} paragraphs, {sum(words)} words "
              f"(avg {sum(words) // max(len(words), 1)}, max {max(words, default=0)})")
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
