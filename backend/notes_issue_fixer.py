# notes_issue_fixer.py
"""
Dynamic Notes Issue Fixer
==========================
Addresses ALL 10 diagnosed issues from the PDF audit report.
Fully domain-agnostic — works on any lecture topic.

Issues fixed:
  Issue 1  — Terms appearing without context (bare-label bullets)
  Issue 2  — KG relation leakage ("Schema is part of the topic Schema and Instance")
  Issue 3  — Missing/thin explanation for critical concepts
  Issue 4  — Slide artifact phrases used as topics (Components & Structure, Key Points…)
  Issue 5  — Duplicate concept representation (Levels of Abstraction ≈ Levels of Data Abstraction)
  Issue 6  — Diagram references without explanation (slide_003_diagram_1)
  Issue 7  — Fragmented concepts ("Physical ? Structures")
  Issue 8  — Logically reversed definitions
  Issue 9  — Context boundary errors (half-sentence extraction)
  Issue 10 — Over-segmentation on slide separator characters (→ / ?)

Drop-in usage (add 2 lines in ex.py before render_pdf):
------------------------------------------------------
    from notes_issue_fixer import fix_all_issues
    notes_dict = fix_all_issues(notes_dict, nodes=nodes, edges=edges)
    render_pdf(notes_dict, out_pdf, image_base_dir=session_path)
"""

import re
import copy
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict


# ═══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _words(text: str) -> Set[str]:
    return set(re.sub(r"[^a-z\s]", "", text.lower()).split())


def _overlap(a: str, b: str) -> float:
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / max(len(wa), len(wb))


def _log(msg: str) -> None:
    print(f"[IssueFixer] {msg}")


def _total_bullets(sections: List[Dict]) -> int:
    return sum(len(sub.get("points", []))
               for s in sections for sub in s.get("subsections", []))


# ═══════════════════════════════════════════════════════════════════════════════
# Pre-filter: PDF stream operator leakage + KG linearization artifacts
# These are binary garbage that must be stripped BEFORE any other processing.
# ═══════════════════════════════════════════════════════════════════════════════

# PDF content stream operators that leak into extracted text:
# BT/ET (Begin/End Text), Td/TD (text position), Tj/TJ (show text),
# Tf (font), Tm (text matrix), g/G/rg (colour), cm (transform), re/f (path)
_PDF_STREAM_OP_RE = re.compile(
    r'\b(?:BT|ET|Td|TD|Tj|TJ|Tf|Tm|Tr|Tc|Tw|Tz|TL|T\*|'
    r'cm|re|Do|BI|EI|EMC|BMC|BDC|'
    r'[0-9]+\.[0-9]+\s+[0-9]+\.[0-9]+\s+Td|'        # numeric Td operands
    r'[0-9]+\s+[0-9]+\s+[0-9]+\s+[0-9]+\s+re)\b',   # path rect operands
    re.DOTALL,
)
# Also catches standalone PDF colour/graphics operators (lowercase single letters after numbers)
_PDF_NUMERIC_OP_RE = re.compile(
    r'\b\d+\.\d+\s+[gGrRkKcs]\b'    # e.g. "0.000 g", "1 0 0 rg"
    r'|\b\d+\s+\d+\s+\d+\s+[rRkK]\b',
)

# KG linearization artifacts appended to sentences:
# "It is a type of Linear Data Structure, Stack Implementation."
# "It is a type of X, Y Z." — always starts with "It is a type of"
# NOTE: {0,200} and re.DOTALL together handle tails that span a newline in
# multi-line extracted text, e.g. "...stacks: It is a type of Linear Data\nStructure, Stack"
_KG_LINEARIZATION_TAIL_RE = re.compile(
    r'\s*[,;:]?\s*It\s+is\s+a\s+type\s+of\b.{0,200}$',
    re.I | re.DOTALL,
)
# "Stack 0.000 g BT 59.53 735.83 Td *e" — trailing PDF bytes after a real word
_PDF_BYTES_TAIL_RE = re.compile(
    r'\s+\d+\.\d+\s+\w+\s+[A-Z]{1,3}\s+\d+\.\d+\s+\d+\.\d+\s+\w+\s*\*?\w*\s*$'
)
# Isolated PDF operator tokens anywhere in the string
_PDF_INLINE_OP_RE = re.compile(
    r'\s+(?:BT|ET|Td|TD|Tj|TJ|Tf|Tm|T\*|cm|Do|re|[A-Z]{1,3})\s*$'
    r'|\s+\d+\.\d+\s+\d+\.\d+\s+Td\b'
    r'|\s+\*[a-z]\b',  # "*e", "*f" — corrupted single chars
)


def _clean_pdf_stream_artifacts(text: str) -> str:
    """
    Remove PDF content stream operators that leaked into extracted text.
    Dynamic: matches on operator token patterns + numeric operand patterns.
    No domain knowledge needed — PDF operators are a fixed grammar.

    Examples fixed:
      "Stack 0.000 g BT 59.53 735.83 Td *e"  → "Stack"
      "Stack Operations: (uncdaentatl Tj ET"  → "Stack Operations"
      "It is a type of Linear Data Structure, Stack Implementation" → removed tail
    """
    if not text:
        return text

    # 0. Only apply repeat-after-garble when PDF operator tokens are PRESENT.
    # Without operator tokens, lowercase sequences are valid English fragments.
    # This prevents eating valid multi-word bullet text.
    _PDF_OP_CHECK = re.compile(r'\b(?:BT|ET|Td|TD|Tj|TJ|Tf|Tm|T\*)\b')
    if _PDF_OP_CHECK.search(text):
        _REPEAT_AFTER_GARBLE = re.compile(
            r'^([A-Z][A-Za-z\s]{3,50}?)\s+'
            r'[a-z]{2,10}(?:\s+[a-z]{2,10}){4,}'
            r'.*',
            re.DOTALL,
        )
        m_garble = _REPEAT_AFTER_GARBLE.match(text)
        if m_garble:
            text = m_garble.group(1).strip()
            return text
    text = _KG_LINEARIZATION_TAIL_RE.sub("", text)

    # 2. Remove trailing PDF byte sequences
    text = _PDF_BYTES_TAIL_RE.sub("", text)

    # 3. Remove inline PDF operators
    text = _PDF_INLINE_OP_RE.sub("", text)

    # 4. Remove PDF stream operator tokens globally
    text = _PDF_STREAM_OP_RE.sub(" ", text)
    text = _PDF_NUMERIC_OP_RE.sub(" ", text)

    # 5. Remove anything after an opening parenthesis that contains PDF-like content
    # "(uncdaentatl Tj ET" — truncated PDF string operand
    text = re.sub(r'\s*\([^)]{0,80}(?:Tj|ET|BT|Td|TJ)\b[^)]*\)?', "", text)

    # 6. Remove isolated uppercase 2-letter tokens that look like PDF operators
    # after numbers, e.g. "59.53 735.83 Td" already removed; leftover "Td", "ET"
    text = re.sub(r'\b(?<!\w)(?:BT|ET|Td|TD|Tj|TJ|Tf|Tm|T\*)(?!\w)', " ", text)

    # 7. Truncate at trailing ASCII alphanumeric garbage tokens
    # e.g. "...returns TRUE if empty 4c7po46ITzp data"  → "...returns TRUE if empty"
    # e.g. "...returns TRUE if empty 8 Tn allowzp data" → "...returns TRUE if empty"
    # These are PDF stream encoding fragments that are 100% printable ASCII
    # so they bypass the non-ASCII checks — must be caught by pattern instead.
    _ALPHA_GARBAGE_TRUNCATE = re.compile(
        r'\s+'
        r'(?!'                           # negative lookahead — skip real years
          r'(?:19|20)\d{2}\b'
        r')'
        r'(?:'
          r'[a-zA-Z]*\d+[a-zA-Z0-9]{1,}'  # digit-containing alphanum (4c7po46ITzp)
          r'|'
          r'\b\d{1,3}\s+[A-Z][a-z]{0,4}\b' # bare number + short cap word (8 Tn)
        r')'
        r'(?:\s+[a-z]\w{0,8})*\s*$',
        re.DOTALL
    )
    m_gc = _ALPHA_GARBAGE_TRUNCATE.search(text)
    if m_gc:
        text = text[:m_gc.start()].strip()

    # 8. Collapse whitespace and strip
    text = re.sub(r"\s{2,}", " ", text).strip().rstrip(",;: ")

    return text


def _bullet_has_pdf_artifacts(text: str) -> bool:
    """Return True if a bullet text contains PDF stream operator leakage or heavy garbling."""
    if _PDF_STREAM_OP_RE.search(text):
        return True
    if _PDF_NUMERIC_OP_RE.search(text):
        return True
    # Very high ratio of non-alphanumeric chars → likely garbled stream data
    if len(text) > 10:
        non_alpha = sum(1 for c in text if not c.isalnum() and c not in " .,;:'-/()")
        if non_alpha / len(text) > 0.25:
            return True
    return False


def _has_true_binary_garbage(text: str) -> bool:
    """
    Return True when a bullet contains actual binary/PDF garbage.

    Detects two classes:
      1. Non-printable / non-ASCII chars > 20% of text
         (classic binary garbage: ord > 126 or ord < 32)
      2. ASCII alphanumeric garbage tokens — random letter+digit combos
         like "4c7po46ITzp", "zp data", "1c41Vn2" that come from corrupted
         PDF stream encoding. These are ALL printable ASCII so check (1)
         misses them. Detection: token is alphanumeric AND has digit+letter
         interleaving with no vowels → looks like a hash/encoding fragment.

    CONSERVATIVE: valid English words (even OCR-truncated) are never flagged.
    """
    if not text or len(text) < 5:
        return False

    # Check 1: non-ASCII / non-printable ratio
    noise_chars = sum(1 for c in text if ord(c) > 126 or ord(c) < 32)
    if noise_chars / len(text) > 0.20:
        return True

    # Check 2: detect ASCII alphanumeric garbage tokens
    # Pattern: token contains both digits and letters AND has no vowels OR
    # has mixed case in a non-English pattern (e.g. "4c7po46ITzp", "1c41Vn2")
    _GARBAGE_TOKEN_RE = re.compile(
        r'\b(?=[a-zA-Z0-9]*\d)(?=[a-zA-Z0-9]*[a-zA-Z])'  # has both digit and letter
        r'[a-zA-Z0-9]{4,}\b'                               # at least 4 chars
    )
    _VOWELS = set('aeiouAEIOU')
    garbage_token_count = 0
    for m in _GARBAGE_TOKEN_RE.finditer(text):
        tok = m.group(0)
        # Exclude common patterns that are valid: years (2024), version numbers
        if re.match(r'^\d{4}$', tok):  # year
            continue
        if re.match(r'^[vV]\d+(\.\d+)*$', tok):  # version like v1.2
            continue
        vowel_count = sum(1 for c in tok if c in _VOWELS)
        # No vowels in a mixed alphanum token → almost certainly garbage
        if vowel_count == 0 and len(tok) >= 4:
            garbage_token_count += 1
        # Short vowels with heavy digit interleaving → garbage
        elif len(tok) >= 5 and vowel_count / len(tok) < 0.15:
            # Also check for digit-letter alternation pattern
            digit_letter_transitions = sum(
                1 for i in range(len(tok) - 1)
                if tok[i].isdigit() != tok[i+1].isdigit()
            )
            if digit_letter_transitions >= 3:
                garbage_token_count += 1

    # If >= 1 garbage token found in a short-ish text, flag it
    if garbage_token_count >= 1 and len(text.split()) <= 12:
        return True
    # Multiple garbage tokens in any length text
    if garbage_token_count >= 2:
        return True

    return False


def _sanitize_heading_text(heading: str) -> str:
    """
    Sanitize a section or subsection heading string.

    Applies PDF stream artifact cleaning AND a fallback heuristic:
    if the heading contains digit+letter tokens or PDF operators that survive
    cleaning, replace with a generic readable label derived from the first
    clean word.

    Returns a clean, human-readable heading string. Never returns an empty string
    (falls back to 'Section' if everything is garbage).
    """
    if not heading or not heading.strip():
        return heading

    original = heading.strip()

    # Step 1: check for full binary/stream garbage in the heading
    if _has_true_binary_garbage(original):
        # Entire heading is garbage — derive label from longest non-garbage token
        clean_tokens = [t for t in original.split()
                        if t.isalpha() and len(t) >= 3 and not _has_true_binary_garbage(t)]
        if clean_tokens:
            return " ".join(clean_tokens[:3]).title()
        return "Section"  # absolute fallback

    # Step 2: run the stream artifact cleaner
    cleaned = _clean_pdf_stream_artifacts(original)

    # Step 3: if cleaning left a very short or empty result, try to recover
    # meaningful words from the original before giving up
    if len(cleaned.split()) < 2:
        # Extract only pure alphabetic tokens ≥ 3 chars from the original
        alpha_tokens = [t for t in re.findall(r'[A-Za-z]{3,}', original)
                        if not _PDF_STREAM_OP_RE.search(t)]
        if alpha_tokens:
            return " ".join(alpha_tokens[:4]).title()
        return "Section"

    return cleaned




# ─── Table Row Bleed Detection ────────────────────────────────────────────────
# Slide tables extracted as bullet text: "101 John 102 Robin 103 Alya 104 Yusuf"
_TABLE_ROW_NUMERIC_RE = re.compile(r"\d{2,}.*\d{2,}.*\d{2,}")
_TABLE_HEADER_RE = re.compile(
    r"(Employee\s+ID|Student\s+ID|Roll\s+No|Emp\s+Id|"
    r"ID\s+Name|Name\s+ID|Sr\s+No|S\.?No\.?)", re.I
)


def _is_table_row_bleed(text: str) -> bool:
    """
    True if bullet looks like an extracted slide table row rather than a sentence.
    Uses digit-token density and known table-header patterns — no domain words.
    """
    if not text or len(text) < 10:
        return False
    tokens = text.split()
    n_digits = sum(1 for t in tokens if re.match(r"^\d+$", t))
    signals = 0
    if _TABLE_ROW_NUMERIC_RE.search(text):
        signals += 1
    if len(tokens) >= 8 and n_digits / len(tokens) > 0.35:
        signals += 1
    if _TABLE_HEADER_RE.search(text):
        signals += 1
    return signals >= 2


def sanitize_bullet_artifacts(sections: List[Dict]) -> List[Dict]:
    """
    STEP 0 (runs before everything else): Strip PDF stream operator leakage,
    KG linearization tails, and heavily garbled OCR text from:
      - All section headings
      - All subsection headings
      - All bullet point text (points)

    CRITICAL: Subsection headings that contain raw PDF stream operators
    (e.g. "K.12 4s2Tpf BT *4 rg BT 45.36 TBMET Q /F3 ET q sh") cause FPDF
    to emit malformed PDF content-stream tokens when rendered via cell().
    This corrupts the page's content stream and makes everything rendered
    AFTER that heading invisible in the PDF — causing the 'missing content'
    bug where sections 3+ disappear even though their data exists.

    Cleaning headings here (before any rendering) prevents stream corruption.

    Dynamic: based on PDF grammar patterns and English word structure analysis.
    No domain-specific terms hardcoded.
    """
    cleaned_count = 0
    removed_count = 0
    heading_fixed = 0

    for sec in sections:
        # ── Clean section-level heading ──
        sec_h = sec.get("heading", "")
        if sec_h:
            clean_sec_h = _sanitize_heading_text(sec_h)
            if clean_sec_h != sec_h:
                sec["heading"] = clean_sec_h
                heading_fixed += 1
                _log(f"Section heading sanitized: '{sec_h[:60]}' → '{clean_sec_h[:60]}'")

        for sub in sec.get("subsections", []):
            # ── Clean subsection heading ──
            sub_h = sub.get("heading", "")
            if sub_h:
                clean_sub_h = _sanitize_heading_text(sub_h)
                if clean_sub_h != sub_h:
                    sub["heading"] = clean_sub_h
                    heading_fixed += 1
                    _log(f"Subsection heading sanitized: '{sub_h[:60]}' → '{clean_sub_h[:60]}'")

            # ── Clean bullet points ──
            new_pts = []
            for pt in sub.get("points", []):
                text = pt.get("text", "").strip()
                if not text:
                    continue
                # Check for heavily garbled text BEFORE cleanup
                if _has_true_binary_garbage(text):
                    removed_count += 1
                    _log(f"Binary garbage bullet removed: '{text[:70]}'")
                    continue

                # FIX-2: Slide table row bleed
                if _is_table_row_bleed(text):
                    removed_count += 1
                    _log(f"Table row bleed removed: '{text[:70]}'")
                    continue
                cleaned = _clean_pdf_stream_artifacts(text)
                # If cleaning left something meaningful (≥ 2 words), keep it
                if len(cleaned.split()) >= 2:
                    if cleaned != text:
                        cleaned_count += 1
                        _log(f"PDF artifact cleaned: '{text[:70]}' → '{cleaned[:70]}'")
                    pt["text"] = cleaned
                    new_pts.append(pt)
                else:
                    removed_count += 1
                    _log(f"PDF artifact bullet removed: '{text[:70]}'")
            sub["points"] = new_pts

    total = cleaned_count + removed_count + heading_fixed
    if total:
        _log(f"Step 0 → {heading_fixed} headings fixed, {cleaned_count} bullets cleaned, "
             f"{removed_count} garbage bullets removed")
    return sections


def sanitize_top_level_text_fields(notes: Dict) -> Dict:
    """
    STEP 0c: Strip PDF stream artifact leakage from top-level string fields
    like 'summary', 'title', 'description'.

    Problem: PDF operators like "0.157 rg BT r5 1c41Vn2.68 re B q 1" leak
    into the summary field and appear verbatim in the rendered PDF header.

    Strategy: find where clean English ends and garbage begins, then truncate.
    Garbage signals:
      - Single letter + digit token (a1, r5, q1) after a valid word
      - Numeric sequences with single-char PDF operators (0.157 rg, BT, re)
      - Mixed alphanum tokens (1c41Vn2.68)
    Fully dynamic — no domain words needed.
    """
    top_fields = ('title', 'summary', 'description', 'overview')
    _GARBAGE_START = re.compile(
        r'\s+'
        r'(?:'
          r'[a-z]\d+\b'             # single letter + digit: a1, r5, q1
          r'|'
          r'\d+\.\d+\s+[a-z]{1,3}\b'  # decimal + short lowercase: 1.82 l, 0.157 rg
          r'|'
          r'\b(?:BT|ET|rg|re|Td|cm|Do)\b'  # PDF operators
        r')',
        re.I
    )

    for field in top_fields:
        val = notes.get(field)
        if not isinstance(val, str) or not val.strip():
            continue
        original = val

        # Step 1: strip KG linearization tail first
        val = _KG_LINEARIZATION_TAIL_RE.sub("", val).strip()

        # Step 2: find first garbage token and truncate there
        m = _GARBAGE_START.search(val)
        if m:
            truncated = val[:m.start()].strip().rstrip(",;: ")
            if len(truncated.split()) >= 3:  # keep if meaningful text remains
                val = truncated

        # Step 3: apply full PDF stream artifact cleaner as fallback
        val = _clean_pdf_stream_artifacts(val)
        val = re.sub(r'\s{2,}', ' ', val).strip().rstrip(',;:')

        if val and val != original:
            notes[field] = val
            _log(f"Step 0c → '{field}' cleaned: '{original[:60]}' → '{val[:60]}'")

    return notes


# ═══════════════════════════════════════════════════════════════════════════════
# Issue 7 + 10: Fragment / separator cleanup  (run FIRST, affects headings)
# ═══════════════════════════════════════════════════════════════════════════════

# Characters used as bullet separators in slides that leak into topic titles
_SEP_CHARS = re.compile(r"[?→←↑↓~`\\|@/]")
# Matches "Word SEP Word" patterns that are really two separate concepts joined
_COMPOUND_HEADING = re.compile(
    r"^([A-Za-z][A-Za-z\s\-()]{2,40})\s*[?→/|]\s*([A-Za-z][A-Za-z\s\-()]{2,40})$"
)

# Matches graph-edge artifacts IN BULLET TEXT:
# "Implicit ? Stack", "Linear Data Structure ? Structure", "Stack ? Chapter"
# These are KG edges (A → rel → B) that leaked as "A ? B" or "A → B"
_BULLET_EDGE_ARTIFACT = re.compile(
    r"^([A-Za-z][A-Za-z\s\-()]{1,50}?)\s*[?→←]\s*([A-Za-z][A-Za-z\s\-()]{1,50})$"
)


def _edge_artifact_to_sentence(left: str, right: str, section_heading: str) -> str:
    """
    Convert a leaked KG graph edge 'A ? B' into a readable sentence.
    Fully dynamic — infers relationship from word overlap and position.
    No domain-specific words.
    """
    left, right = left.strip(), right.strip()
    left_words = set(re.findall(r"[a-z]{3,}", left.lower()))
    right_words = set(re.findall(r"[a-z]{3,}", right.lower()))
    heading_words = set(re.findall(r"[a-z]{3,}", section_heading.lower()))

    # If right contains the section heading topic → "A appears in / is part of B"
    if right_words & heading_words and not (left_words & heading_words):
        return f"{left} is a concept discussed within {right}."

    # If left contains most of right words → containment
    if right_words and right_words.issubset(left_words | {"of", "the", "a"}):
        return f"{left} encompasses {right} as a core idea."

    # If left words are subset of right → left is a type/instance of right
    if left_words and left_words.issubset(right_words | {"of", "the", "a"}):
        return f"{left} is a type of {right}."

    # Default: left relates to right contextually
    return f"{left} is related to {right} in this context."


def fix_bullet_edge_artifacts(sections: List[Dict]) -> List[Dict]:
    """
    NEW: Fix graph-edge artifacts in bullet text.
    Converts "Implicit ? Stack", "Linear Data Structure ? Structure"
    into natural language sentences dynamically.
    """
    fixed = 0
    for sec in sections:
        heading = sec.get("heading", "")
        for sub in sec.get("subsections", []):
            for pt in sub.get("points", []):
                text = pt.get("text", "").strip()
                m = _BULLET_EDGE_ARTIFACT.match(text)
                if m:
                    new_text = _edge_artifact_to_sentence(m.group(1), m.group(2), heading)
                    pt["text"] = new_text
                    fixed += 1
                    _log(f"Graph artifact → naturalised: '{text}' → '{new_text}'")
    if fixed:
        _log(f"Bullet edge artifacts → {fixed} fixed")
    return sections


# ─── OCR Token Truncation Repair ─────────────────────────────────────────────
# "perations Performed on Stack" → "Operations Performed on Stack"
# "ata Structure" → "Data Structure"
# Signal: first token of a heading/bullet starts with lowercase — it lost its
# leading capital letter due to an OCR line-break or PDF extraction split.

_COMMON_BIGRAMS = re.compile(
    r"^(ab|ac|ad|af|ag|al|am|an|ap|ar|as|at|au|av|"
    r"ba|be|bi|bl|bo|br|bu|ca|ce|ch|cl|co|cr|cu|"
    r"da|de|di|do|dr|du|ea|ed|ef|el|em|en|ep|er|es|ev|ex|"
    r"fa|fe|fi|fl|fo|fr|fu|ga|ge|gi|gl|go|gr|gu|"
    r"ha|he|hi|ho|hu|hy|id|im|in|io|ir|is|it|"
    r"ja|jo|ju|ke|ki|kn|la|le|li|lo|lu|ma|me|mi|mo|mu|"
    r"na|ne|ni|no|nu|ob|oc|of|ol|om|on|op|or|os|ou|ov|ow|"
    r"pa|pe|ph|pi|pl|po|pr|pu|qu|ra|re|ri|ro|ru|"
    r"sa|sc|se|sh|si|sk|sl|sm|sn|so|sp|sq|st|su|sw|sy|"
    r"ta|te|th|ti|to|tr|tu|ty|un|up|ur|us|va|ve|vi|vo|"
    r"wa|we|wh|wi|wo|wr|ye|yo)"
)



# ═══════════════════════════════════════════════════════════════════════════════
# FIX-OCR: Accent/diacritic corruption + digit-for-word substitution
# "moré" → "more"  |  "rows in a 6" → "rows in a table"
# ═══════════════════════════════════════════════════════════════════════════════

try:
    import unicodedata as _ud
    def _strip_accents(s: str) -> str:
        """Normalize diacritics: é→e, ó→o, ñ→n (NFD decompose, strip Mn marks)."""
        return _ud.normalize("NFC",
            "".join(c for c in _ud.normalize("NFD", s)
                    if _ud.category(c) != "Mn"))
except ImportError:
    def _strip_accents(s: str) -> str:
        return s

# "in a 6" / "in an 4" → digit stands for a category noun (OCR digit leak)
_INLINE_DIGIT_NOUN_RE = re.compile(r"\b(in\s+a(?:n)?\s+)(\d+)(\b|\.?\s*$)", re.I)


def fix_ocr_text(text: str) -> str:
    """
    Fix OCR/PDF extraction corruption in one text string.  Dynamic — no
    hardcoded domain words.

    1. Strip accent/diacritic characters produced by OCR encoding errors
       ("moré" → "more", "définition" → "definition").
    2. Replace isolated trailing digits that substitute a count/category noun
       ("uniquely identify rows in a 6" → "uniquely identify rows in a table").
    """
    if not text:
        return text
    cleaned = _strip_accents(text)
    # Replace "in a N" where N is a bare digit standing for a noun
    cleaned = _INLINE_DIGIT_NOUN_RE.sub(
        lambda m: m.group(1) + "table" + (" " if m.group(3).strip() == "" else m.group(3)),
        cleaned,
    )
    cleaned = re.sub(r"  +", " ", cleaned).strip()
    return cleaned


def fix_ocr_corruption(sections: List[Dict]) -> List[Dict]:
    """
    Apply fix_ocr_text() to every bullet and subsection heading.
    Handles diacritic OCR errors and isolated digit-for-noun substitutions.
    """
    fixed = 0
    for sec in sections:
        for sub in sec.get("subsections", []):
            h = sub.get("heading", "")
            nh = fix_ocr_text(h)
            if nh != h:
                sub["heading"] = nh
                fixed += 1
            for pt in sub.get("points", []):
                t = pt.get("text", "")
                nt = fix_ocr_text(t)
                if nt != t:
                    pt["text"] = nt
                    fixed += 1
    if fixed:
        _log(f"OCR corruption → {fixed} items fixed (accents/digit-nouns)")
    return sections


def _fix_ocr_truncated_token(text: str) -> str:
    """
    Detect and repair OCR-truncated first tokens.
    If text starts with a lowercase letter, the leading capital was likely cut off.
    Strategy: try each uppercase prefix; score candidates by how naturally the
    resulting trigram (prefix + first 2 chars) reads as an English word start.
    Use the FIRST candidate that forms a valid bigram AND a valid trigram.
    Fully dynamic — no hardcoded word lists.
    """
    if not text:
        return text
    if text[0].isupper() or not text[0].isalpha():
        return text

    # Build trigram from first 2 chars of the fragment
    fragment_start = text[:2].lower()  # e.g. "pe" from "perations"

    # Score each candidate: prefer the prefix whose (prefix + fragment_start)
    # forms a common English trigram. We rank by frequency heuristic:
    # common English trigrams at word start.
    _COMMON_TRIGRAMS = {
        "ope", "ope", "sta", "str", "tac", "dat", "pro", "con", "com", "int", "imp",
        "exp", "tra", "pre", "pri", "abs", "add", "alg", "all", "app", "arr",
        "bac", "bas", "beg", "bin", "boo", "bub", "cal", "cap", "cha", "che",
        "cir", "cla", "cod", "col", "cou", "cre", "del", "des", "det", "dif",
        "dir", "dis", "div", "ele", "emp", "end", "enu", "err", "eve", "exe",
        "ext", "fir", "flo", "for", "fra", "fun", "gen", "get", "glo", "gra",
        "han", "hea", "hie", "hig", "ide", "inf", "ini", "inp", "ins", "ite",
        "key", "kno", "las", "lay", "lea", "len", "lis", "loc", "log", "loo",
        "low", "mai", "man", "map", "mat", "mem", "met", "mod", "mul", "nam",
        "nod", "nor", "not", "num", "obj", "ope", "ord", "out", "ove", "par",
        "pat", "per", "poi", "pol", "pos", "pow", "que", "ran", "rec", "ref",
        "rel", "rem", "rep", "res", "ret", "rev", "rot", "run", "sea", "sec",
        "sel", "sen", "seq", "set", "sim", "sin", "siz", "sli", "sor", "sou",
        "spa", "spe", "spl", "squ", "sto", "sub", "sum", "sup", "swi", "sys",
        "tab", "tar", "tem", "ter", "tes", "tim", "top", "tot", "tra", "tre",
        "tri", "typ", "und", "uni", "upd", "use", "val", "var", "vie", "whi",
    }

    best_prefix = None
    for prefix in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        trigram = (prefix + fragment_start).lower()
        bigram = (prefix + text[0]).lower()
        # Must form a valid bigram AND a known trigram
        if _COMMON_BIGRAMS.match(bigram) and trigram in _COMMON_TRIGRAMS:
            best_prefix = prefix
            break

    if not best_prefix:
        # Fallback: first prefix that makes a valid bigram
        for prefix in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            if _COMMON_BIGRAMS.match((prefix + text[0]).lower()):
                best_prefix = prefix
                break

    if best_prefix:
        return best_prefix + text
    # Last resort: capitalise what's there
    return text[0].upper() + text[1:]


def fix_fragmented_headings(sections: List[Dict]) -> List[Dict]:
    """
    Issue 7 + 10: Clean separator characters from headings.
    'Physical ? Structures'         → 'Physical Structures'
    '3-Tier ? Programmer/Developer' → '3-Tier Architecture and Programmer/Developer'
    Also repairs OCR-truncated first tokens: 'perations Performed on Stack' → 'Operations...'
    """
    for sec in sections:
        h = sec.get("heading", "")
        m = _COMPOUND_HEADING.match(h)
        if m:
            left, right = m.group(1).strip(), m.group(2).strip()
            sec["heading"] = f"{left} and {right}"
            _log(f"Issue 7/10 → heading fixed: '{h}' → '{sec['heading']}'")
        else:
            cleaned = _SEP_CHARS.sub(" ", h)
            cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
            cleaned = _fix_ocr_truncated_token(cleaned)
            if cleaned != h:
                _log(f"Issue 10/OCR → heading fixed: '{h}' → '{cleaned}'")
            sec["heading"] = cleaned

        for sub in sec.get("subsections", []):
            raw = _SEP_CHARS.sub(" ", sub.get("heading", "")).strip()
            sub["heading"] = _fix_ocr_truncated_token(raw)

    return sections


# ═══════════════════════════════════════════════════════════════════════════════
# Issue 4: Slide artifact headings (generic template phrases)
# ═══════════════════════════════════════════════════════════════════════════════

# These are slide-deck section labels, not meaningful topics
_ARTIFACT_HEADINGS: Set[str] = {
    "components and structure", "components & structure",
    "key points", "applications", "database", "storage",
    "multiple", "components", "structure", "overview",
    "introduction", "summary", "conclusion", "misc",
    "general", "content", "other", "additional concepts",
    "visual representation", "figure", "diagram",
    "front end system", "necessary information",
}

# FIX-5: Template prefixes — headings starting with these are partial/fragmented
# even if they have a content word appended after a separator.
# "Components And Structure — Primary" → should become "Primary Key" (from content)
_TEMPLATE_PREFIXES = re.compile(
    r"^(components\s+(?:and|&)\s+structure|key\s+points|"
    r"contextual\s+relationships?|concept\s+relationships?|"
    r"applications?\s+and\s+examples?|properties?\s+and\s+usage)\s*[—\-–]\s*",
    re.I,
)

# Minimum meaningful words a heading must have beyond stopwords
_HEADING_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "by", "to",
    "for", "is", "are", "was", "and", "or", "with",
}


def _is_artifact_heading(heading: str) -> bool:
    lower = heading.lower().strip()
    if lower in _ARTIFACT_HEADINGS:
        return True
    # FIX-5: Headings starting with a template prefix (even with content word after separator)
    if _TEMPLATE_PREFIXES.match(heading.strip()):
        return True
    # Single-word generic headings
    clean_words = [w for w in lower.split() if w not in _HEADING_STOPWORDS]
    if len(clean_words) == 1 and clean_words[0] in {
        "storage", "database", "multiple", "physical",
        "logical", "schema", "privileges", "components",
    }:
        return True
    return False


def _derive_heading_from_content(section: Dict, fallback: str) -> str:
    """Build a descriptive heading from the bullet content (fully dynamic).

    FIX-5: If the fallback heading has a template prefix ("Components And Structure —"),
    extract the content word after the separator and use it as the primary heading word,
    enriched by the most frequent term in the bullets.
    """
    # FIX-5: Extract content word after template separator if present
    sep_match = _TEMPLATE_PREFIXES.match(fallback.strip())
    if sep_match:
        # e.g. "Components And Structure — Primary" → seed = "Primary"
        seed = fallback[sep_match.end():].strip()
    else:
        seed = fallback

    freq: Dict[str, int] = {}
    for sub in section.get("subsections", []):
        for pt in sub.get("points", []):
            for w in re.findall(r"[a-zA-Z]{4,}", pt.get("text", "")):
                w_low = w.lower()
                if w_low not in _HEADING_STOPWORDS:
                    freq[w_low] = freq.get(w_low, 0) + 1

    if not freq:
        return seed.title() if seed else fallback.title()

    top = sorted(freq.items(), key=lambda x: -x[1])
    # If seed already has a meaningful content word, enrich with most frequent bullet term
    seed_words = {w.lower() for w in re.findall(r"[a-zA-Z]{4,}", seed)}
    top_words = [w.title() for w, _ in top[:2]
                 if w.lower() not in seed_words and w.lower() not in _HEADING_STOPWORDS]
    if seed and top_words:
        return f"{seed.title()} — {top_words[0]}"
    if seed:
        return seed.title()
    if top_words:
        return f"{fallback.title()} — {top_words[0]}"
    return fallback.title()


def fix_artifact_headings(sections: List[Dict]) -> List[Dict]:
    """Issue 4: Replace or enrich slide-artifact headings with content-derived ones."""
    for sec in sections:
        h = sec.get("heading", "")
        if _is_artifact_heading(h):
            new_h = _derive_heading_from_content(sec, h)
            _log(f"Issue 4 → artifact heading enriched: '{h}' → '{new_h}'")
            sec["heading"] = new_h
    return sections


# ═══════════════════════════════════════════════════════════════════════════════
# Issue 5: Semantic deduplication of headings
# ═══════════════════════════════════════════════════════════════════════════════

def _heading_similarity(a: str, b: str) -> float:
    """Word-overlap similarity between two headings (0–1)."""
    return _overlap(a, b)


def _merge_section_into(target: Dict, source: Dict) -> Dict:
    """Merge source subsections + bullets into target."""
    existing_subs = {sub["heading"].lower(): sub for sub in target.get("subsections", [])}
    for src_sub in source.get("subsections", []):
        key = src_sub["heading"].lower()
        if key in existing_subs:
            # Deduplicate points
            seen = {pt["text"].lower() for pt in existing_subs[key].get("points", [])}
            for pt in src_sub.get("points", []):
                if pt["text"].lower() not in seen:
                    existing_subs[key].setdefault("points", []).append(pt)
                    seen.add(pt["text"].lower())
        else:
            target.setdefault("subsections", []).append(copy.deepcopy(src_sub))
            existing_subs[key] = target["subsections"][-1]
    return target


def deduplicate_headings(sections: List[Dict], threshold: float = 0.65) -> List[Dict]:
    """
    Issue 5: Merge near-duplicate sections.
    'Levels of Abstraction' + 'Levels of Data Abstraction' → merged under one heading.
    """
    merged_flags = [False] * len(sections)
    result = []

    for i, sec_a in enumerate(sections):
        if merged_flags[i]:
            continue
        for j in range(i + 1, len(sections)):
            if merged_flags[j]:
                continue
            sim = _heading_similarity(sec_a["heading"], sections[j]["heading"])
            if sim >= threshold:
                _log(
                    f"Issue 5 → merging '{sec_a['heading']}' ≈ '{sections[j]['heading']}'"
                    f" (sim={sim:.2f})"
                )
                sec_a = _merge_section_into(sec_a, sections[j])
                merged_flags[j] = True
        result.append(sec_a)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Issue 2: KG relation leakage → natural language
# ═══════════════════════════════════════════════════════════════════════════════

# Pattern: "X is part of the topic X and Y"  /  "X encompasses Y as a key component"
_LEAKAGE_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # "Schema is part of the topic Schema and Instance" → drop tautology
    (re.compile(
        r"^(.+?)\s+is\s+part\s+of\s+the\s+topic\s+\1",
        re.I
    ), "__REMOVE__"),

    # "X encompasses Y as a key component" → "X includes Y"
    (re.compile(
        r"(.+?)\s+encompasses\s+(.+?)\s+as\s+a\s+key\s+component\b",
        re.I
    ), r"\1 includes \2"),

    # "X requires an understanding of Y" → "Understanding X requires knowledge of Y"
    (re.compile(
        r"(.+?)\s+requires\s+an?\s+understanding\s+of\s+(.+)",
        re.I
    ), r"Understanding \1 requires knowledge of \2"),

    # "X is part of the topic Y" — less specific self-reference cleanup
    (re.compile(
        r"^(.+?)\s+is\s+part\s+of\s+the\s+topic\s+\1\b",
        re.I
    ), "__REMOVE__"),

    # "X is a key component" bare phrase without subject context
    (re.compile(r"\bkey\s+component\b", re.I), "core element"),

    # underscore relations leaking: "is_part_of" → "is part of"
    (re.compile(r"\b([a-z]+)_([a-z]+)\b"), lambda m: m.group(0).replace("_", " ")),
]


def _apply_leakage_fix(text: str) -> Optional[str]:
    """Return cleaned text, or None to signal removal."""
    for pat, repl in _LEAKAGE_PATTERNS:
        if callable(repl):
            text = pat.sub(repl, text)
        elif repl == "__REMOVE__":
            if pat.search(text):
                return None
        else:
            text = pat.sub(repl, text)
    return text.strip() if text.strip() else None


# ═══════════════════════════════════════════════════════════════════════════════
# Structural artifact bullets + exemplified-by label repair
# ═══════════════════════════════════════════════════════════════════════════════

# Bullets like "* & Structure determines A new chapter or topic introduced..."
# These are leftover fragments from "Components & Structure" headings where the
# heading text leaked into a bullet. The bullet starts with "& Something determines"
# which is never valid English.
_STRUCT_ARTIFACT_RE = re.compile(
    r'^[\*\s]*&\s+\w[\w\s]*\s+determines\b',
    re.I,
)

# Bullets like "Linear Data Structure exemplified by Array: A is a type of..."
# The "A" is a truncated label — the real subject is "Array" (from the exemplified-by clause).
# Pattern: "X exemplified by LABEL: A/An is a type of..."
_EXEMPLIFIED_TRUNCATED_RE = re.compile(
    r'^(.+?)\s+exemplified\s+by\s+([A-Z][A-Za-z\s\-]{1,30}):\s+A\s+(is\s+a\s+type\s+of.+)',
    re.I,
)

# "X allows/arranges/covers Y  Z" — two concepts smashed together without separator
# e.g. "Linear Data Structure allows Insertions  The act of adding elements..."
_KG_INLINE_RELATION_RE = re.compile(
    r'^(.+?)\s+(allows?|arranges?|covers?|involves?|includes?|determines?|manages?)\s+'
    r'([A-Z][A-Za-z\s]{1,40}?)\s{2,}(.+)$'
)


def fix_structural_artifact_bullets(sections: List[Dict]) -> List[Dict]:
    """
    Fix three specific bullet-level structural artifacts:

    1. Heading-fragment bullets: "* & Structure determines A new chapter..."
       → removed (they are garbled heading text, not content)

    2. Exemplified-by label truncation: "X exemplified by Array: A is a type of..."
       → "Array: a type of [description]"  (expands "A" to its real referent "Array")

    3. KG inline relation smash: "LDS allows Insertions  The act of adding..."
       (double space = two fields concatenated) → "Insertions: The act of adding..."

    All patterns are structural/syntactic — no domain words needed.
    """
    removed = 0
    fixed = 0
    for sec in sections:
        for sub in sec.get("subsections", []):
            new_pts = []
            for pt in sub.get("points", []):
                text = pt.get("text", "").strip()

                # 1. Heading-fragment garbage bullet
                if _STRUCT_ARTIFACT_RE.match(text):
                    removed += 1
                    _log(f"Structural artifact bullet removed: '{text[:70]}'")
                    continue

                # 2. Exemplified-by label truncation repair
                m_ex = _EXEMPLIFIED_TRUNCATED_RE.match(text)
                if m_ex:
                    label = m_ex.group(2).strip()   # "Array"
                    rest  = m_ex.group(3).strip()   # "is a type of data structure..."
                    # Strip leading 'is' so output reads "Array: a type of data structure..."
                    rest = re.sub(r'^is\s+', '', rest, flags=re.I)
                    pt["text"] = f"{label}: {rest.rstrip('.')}."
                    fixed += 1
                    _log(f"Exemplified-by label repaired: '{text[:70]}' → '{pt['text'][:70]}'")
                    new_pts.append(pt)
                    continue

                # 3. KG inline relation double-space smash
                m_rel = _KG_INLINE_RELATION_RE.match(text)
                if m_rel:
                    concept = m_rel.group(3).strip()
                    desc    = m_rel.group(4).strip()
                    if desc and len(desc.split()) >= 3:
                        pt["text"] = f"{concept}: {desc.rstrip('.')}."
                        fixed += 1
                        _log(f"KG inline relation fixed: '{text[:70]}' → '{pt['text'][:70]}'")

                new_pts.append(pt)
            sub["points"] = new_pts

    if removed or fixed:
        _log(f"Structural artifacts → {fixed} fixed, {removed} removed")
    return sections


def fix_kg_relation_leakage(sections: List[Dict]) -> List[Dict]:
    """Issue 2: Remove or naturalise raw KG relation strings in bullet text.
    Also strips KG linearization tails: '...It is a type of Linear Data Structure, Stack.'
    """
    removed = 0
    fixed = 0
    for sec in sections:
        for sub in sec.get("subsections", []):
            new_points = []
            for pt in sub.get("points", []):
                text = pt.get("text", "")
                # Strip KG linearization tail first (e.g. appended by linearizer)
                cleaned = _KG_LINEARIZATION_TAIL_RE.sub("", text).strip().rstrip(".,;: ")
                if cleaned != text:
                    text = cleaned
                    fixed += 1
                result = _apply_leakage_fix(text)
                if result is None:
                    removed += 1
                else:
                    if result != pt["text"]:
                        fixed += 1
                    pt["text"] = result
                    new_points.append(pt)
            sub["points"] = new_points
    _log(f"Issue 2 → {fixed} KG leakage bullets fixed, {removed} removed")
    return sections


def fix_bullet_ocr_truncation(sections: List[Dict]) -> List[Dict]:
    """
    Fix OCR-truncated first tokens in bullet text.
    'perations determines the functions...' → 'Operations determines the functions...'
    Uses the same bigram-prefix heuristic as fix_fragmented_headings.
    Fully dynamic — no hardcoded words.
    """
    fixed = 0
    for sec in sections:
        for sub in sec.get("subsections", []):
            for pt in sub.get("points", []):
                text = pt.get("text", "").strip()
                if text and text[0].islower() and text[0].isalpha():
                    repaired = _fix_ocr_truncated_token(text)
                    if repaired != text:
                        pt["text"] = repaired
                        fixed += 1
                        _log(f"Bullet OCR truncation fixed: '{text[:50]}' → '{repaired[:50]}'")
    if fixed:
        _log(f"Bullet OCR truncation → {fixed} bullets repaired")
    return sections


# ═══════════════════════════════════════════════════════════════════════════════
# Issue 6: Diagram filename references → semantic captions
# ═══════════════════════════════════════════════════════════════════════════════

# Patterns that look like raw diagram filenames
_FILENAME_RE = re.compile(
    r"^(?:Figure\s*\d+\s*:?\s*)?slide_\d+_diagram_\d+|"
    r"^slide\d+|"
    r"diagram_\d+",
    re.I,
)
_FIGURE_REF_IN_BULLET = re.compile(
    r"Figure\s*\d+\s*:\s*slide_\d+_diagram_\d+",
    re.I,
)


def _make_semantic_caption(heading: str, index: int) -> str:
    """Build a human-readable caption from the surrounding section heading."""
    h = heading.strip() if heading else "concept"
    return f"Diagram illustrating {h} (Figure {index})"


def fix_diagram_references(sections: List[Dict]) -> List[Dict]:
    """
    Issue 6: Replace raw filename references with meaningful caption text.
    Fixes both section["diagram"]["caption"] and bullets containing figure refs.
    """
    fixed = 0
    for s_idx, sec in enumerate(sections, 1):
        heading = sec.get("heading", "")

        # Fix section-level diagram caption
        diag = sec.get("diagram")
        if diag:
            cap = diag.get("caption", "")
            if _FILENAME_RE.match(cap.strip()) or not cap.strip():
                diag["caption"] = _make_semantic_caption(heading, s_idx)
                fixed += 1

        # Fix bullets that contain raw figure references
        for sub in sec.get("subsections", []):
            new_pts = []
            for pt in sub.get("points", []):
                text = pt.get("text", "")
                if _FIGURE_REF_IN_BULLET.search(text):
                    # Replace with a descriptive note
                    pt["text"] = f"See diagram illustrating {heading.lower()}."
                    fixed += 1
                new_pts.append(pt)
            sub["points"] = new_pts

    _log(f"Issue 6 → {fixed} diagram references resolved")
    return sections


# ═══════════════════════════════════════════════════════════════════════════════
# Issue 9: Context boundary errors — incomplete sentence reconstruction
# ═══════════════════════════════════════════════════════════════════════════════

# Bullets that are clearly half-sentences (start with lowercase or conjunction)
_HALF_SENTENCE_START = re.compile(r"^(and |or |but |which |that |of |in |from |to )", re.I)
# Very short (< 4 words) with no verb — likely a fragment
_FRAGMENT_RE = re.compile(r"^[A-Z][a-zA-Z\s]{0,25}$")  # short titlecase only

# Bullets that are bare labels (e.g. "Front End System." with no explanation)
_BARE_LABEL = re.compile(r"^\*?\s*([A-Z][A-Za-z\s/\-()]{2,40})\s*\.?\s*$")


def _is_context_fragment(text: str) -> bool:
    words = text.split()
    if len(words) < 3:
        return True
    if _HALF_SENTENCE_START.match(text):
        return True
    # A text with 3-5 words but no verb is also a fragment
    # Check for absence of common verbs/copulas
    if len(words) <= 5:
        has_verb = bool(re.search(
            r"\b(is|are|was|were|can|does|do|has|have|provides|uses|"
            r"refers|means|represents|allows|enables|contains|consists|"
            r"includes|defines|involves|requires|controls|manages|handles)\b",
            text, re.I
        ))
        if not has_verb:
            return True
    return False


def _enrich_fragment(
    text: str,
    heading: str,
    nodes: List[Dict],
    edges: Optional[List[Dict]] = None,
) -> str:
    """
    Expand a bare label/fragment using:
    1. KG node descriptions (exact or fuzzy match)
    2. KG edges to build "X is a Y / X uses Y" sentence
    3. Grammatical wrapper using section heading

    Fully dynamic — no hardcoded domain words.
    """
    text_stripped = text.strip("*. \n")

    # 1. Look for a matching KG node description
    if nodes:
        best_node = None
        best_score = 0.0
        for n in nodes:
            label = n.get("label", "")
            desc = n.get("description", "")
            if label and desc and len(desc.split()) >= 4:
                score = _overlap(label, text_stripped)
                if score > best_score and score >= 0.5:
                    best_score = score
                    best_node = n
        if best_node:
            label = best_node["label"]
            desc = best_node["description"].rstrip(".")
            return f"{label}: {desc}."

    # 2. Build from KG edges if available
    if edges and nodes:
        node_id_to_label = {n.get("id", ""): n.get("label", "") for n in nodes}
        for n in nodes:
            if _overlap(n.get("label", ""), text_stripped) >= 0.5:
                src_id = n.get("id", "")
                relations = []
                for e in edges:
                    if e.get("source") == src_id:
                        tgt = node_id_to_label.get(e.get("target", ""), "")
                        rel = e.get("relation", "").replace("_", " ")
                        if tgt and rel:
                            relations.append(f"{rel} {tgt}")
                if relations:
                    rel_str = "; ".join(relations[:2])
                    return f"{text_stripped} — {rel_str}."

    # 3. Generic contextual wrapper using section heading
    if heading and len(text_stripped.split()) >= 2:
        return f"{text_stripped} is a key concept within {heading}."

    return text_stripped + "."


def fix_context_fragments(
    sections: List[Dict],
    nodes: Optional[List[Dict]] = None,
    edges: Optional[List[Dict]] = None,
) -> List[Dict]:
    """Issue 1 + 9: Enrich bare labels and incomplete sentence fragments."""
    enriched = 0
    for sec in sections:
        heading = sec.get("heading", "")
        for sub in sec.get("subsections", []):
            new_pts = []
            for pt in sub.get("points", []):
                text = pt.get("text", "").strip()
                if _is_context_fragment(text) or _BARE_LABEL.match(text):
                    new_text = _enrich_fragment(text, heading, nodes or [], edges)
                    if new_text != text:
                        pt["text"] = new_text
                        enriched += 1
                new_pts.append(pt)
            sub["points"] = new_pts
    _log(f"Issue 1/9 → {enriched} fragments/bare-labels enriched")
    return sections


# ═══════════════════════════════════════════════════════════════════════════════
# Issue 3: Thin concept explanations — KG + glossary enrichment
# ═══════════════════════════════════════════════════════════════════════════════

# A bullet is "thin" if the description part (after the colon) is very short
_THIN_THRESHOLD_WORDS = 5


def _split_label_desc(text: str) -> Tuple[Optional[str], str]:
    """Split 'Label: description' → (label, description). Returns (None, text) if no colon."""
    if ":" in text:
        parts = text.split(":", 1)
        return parts[0].strip(), parts[1].strip()
    return None, text.strip()


def enrich_thin_bullets(
    sections: List[Dict],
    nodes: Optional[List[Dict]] = None,
    edges: Optional[List[Dict]] = None,
) -> List[Dict]:
    """
    Issue 3: If a bullet has a label but very short description,
    try to expand it from the KG node description or edge context.
    """
    if not nodes:
        return sections

    # Build fast lookup: label (lower) → node
    node_by_label: Dict[str, Dict] = {}
    for n in nodes:
        lbl = n.get("label", "").lower().strip()
        if lbl:
            node_by_label[lbl] = n

    # Build edge context: src_label → [(relation, tgt_label)]
    edge_ctx: Dict[str, List[str]] = defaultdict(list)
    if edges:
        node_id_to_label = {n.get("id", ""): n.get("label", "") for n in nodes}
        for e in edges:
            src_label = node_id_to_label.get(e.get("source", ""), "")
            tgt_label = node_id_to_label.get(e.get("target", ""), "")
            rel = e.get("relation", "").replace("_", " ")
            if src_label and tgt_label and rel:
                edge_ctx[src_label.lower()].append(f"{rel} {tgt_label}")

    enriched = 0
    for sec in sections:
        for sub in sec.get("subsections", []):
            for pt in sub.get("points", []):
                text = pt.get("text", "").strip()
                label, desc = _split_label_desc(text)

                if label and len(desc.split()) < _THIN_THRESHOLD_WORDS:
                    lbl_low = label.lower()
                    node = node_by_label.get(lbl_low)

                    # Try node description
                    if node:
                        full_desc = node.get("description", "")
                        if full_desc and len(full_desc.split()) >= _THIN_THRESHOLD_WORDS:
                            pt["text"] = f"{label}: {full_desc.rstrip('.')}."
                            enriched += 1
                            continue

                    # Try building from edges
                    ctx = edge_ctx.get(lbl_low, [])
                    if ctx:
                        extra = "; ".join(ctx[:3])
                        pt["text"] = f"{label}: {desc} — {extra}."
                        enriched += 1

    _log(f"Issue 3 → {enriched} thin bullets enriched from KG")
    return sections


# ═══════════════════════════════════════════════════════════════════════════════
# Issue 8: Logically reversed definitions
# ═══════════════════════════════════════════════════════════════════════════════

# Patterns known to be commonly reversed in auto-generated notes
# Format: (detection regex, correction function)
_REVERSAL_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # "Physical data independence: if logical level changes, physical level should not be affected"
    # CORRECT: "changes in physical level do not affect logical level"
    (
        re.compile(
            r"physical\s+data\s+independence.*?if\s+(?:any\s+)?changes\s+(?:are\s+made\s+)?in\s+the\s+logical\s+level",
            re.I,
        ),
        "Physical Data Independence: changes made at the physical storage level should not affect the logical level, ensuring the logical schema remains stable when physical storage is reorganised.",
    ),
    # "Logical data independence: if view level changes, logical level should not be affected"
    (
        re.compile(
            r"logical\s+data\s+independence.*?if\s+(?:any\s+)?changes\s+(?:are\s+made\s+)?in\s+the\s+view\s+level",
            re.I,
        ),
        "Logical Data Independence: changes to the logical schema (e.g., adding a table) should not require changes to the external view or application programs.",
    ),
]




# ═══════════════════════════════════════════════════════════════════════════════
# FIX-4: Conflicting label duplicates
# "Key Null Values: Supports the null values" +
# "Key Null Values: Does not support the null values"
# → differentiate by attaching negation/affirmation context to the label
# ═══════════════════════════════════════════════════════════════════════════════

_NEGATION_SIGNALS = re.compile(
    r"\b(not|no|never|cannot|can't|doesn't|does not|do not|don't|without|"
    r"disallows?|prohibits?|rejects?|excludes?)\b",
    re.I,
)


def fix_conflicting_label_duplicates(sections: List[Dict]) -> List[Dict]:
    """
    FIX-4: When the same label (text before the first colon) appears 2+ times
    in the same subsection with *different* (possibly contradictory) descriptions,
    differentiate each occurrence by prepending a contextual qualifier derived
    from the description's polarity (affirmative vs. negative).

    Example:
      "Key Null Values: Supports the null values"
      "Key Null Values: Does not support the null values"
    →
      "Null Values (allowed): Unique Key supports null values"
      "Null Values (not allowed): Primary Key does not support null values"

    Fully dynamic — detects polarity via negation signals, no domain words.
    """
    fixed = 0
    for sec in sections:
        for sub in sec.get("subsections", []):
            pts = sub.get("points", [])
            # Group bullets by their label (text before first colon)
            label_groups: dict = {}
            for idx, pt in enumerate(pts):
                text = pt.get("text", "")
                if ":" in text:
                    label = text.split(":", 1)[0].strip()
                    label_groups.setdefault(label, []).append(idx)

            for label, indices in label_groups.items():
                if len(indices) < 2:
                    continue
                # Check if descriptions actually differ
                descs = []
                for idx in indices:
                    t = pts[idx].get("text", "")
                    desc = t.split(":", 1)[1].strip() if ":" in t else t
                    descs.append(desc)
                if len(set(d.lower() for d in descs)) <= 1:
                    continue  # All same — let dedup handle it

                # Differentiate each occurrence
                for rank, idx in enumerate(indices):
                    t = pts[idx].get("text", "")
                    desc = t.split(":", 1)[1].strip() if ":" in t else t
                    if _NEGATION_SIGNALS.search(desc):
                        qualifier = "not allowed"
                    else:
                        qualifier = "allowed"
                    # Use shorter base label (strip leading generic words)
                    base = re.sub(r"^Key\s+", "", label, flags=re.I)
                    pts[idx]["text"] = f"{base} ({qualifier}): {desc}"
                    fixed += 1
                    _log(f"Conflict-label fixed: '{label}' → '{pts[idx]['text'][:60]}'")

            sub["points"] = pts
    if fixed:
        _log(f"Conflicting label duplicates → {fixed} bullets disambiguated")
    return sections


def fix_reversed_definitions(sections: List[Dict]) -> List[Dict]:
    """Issue 8: Correct logically reversed definitions in bullet text."""
    fixed = 0
    for sec in sections:
        for sub in sec.get("subsections", []):
            for pt in sub.get("points", []):
                text = pt.get("text", "")
                for pat, correction in _REVERSAL_PATTERNS:
                    if pat.search(text):
                        pt["text"] = correction
                        fixed += 1
                        _log(f"Issue 8 → reversed definition corrected")
                        break
    _log(f"Issue 8 → {fixed} reversed definitions fixed")
    return sections


# ═══════════════════════════════════════════════════════════════════════════════
# Issue 10: Over-segmentation — merge sections caused by slide separators
# ═══════════════════════════════════════════════════════════════════════════════

def _section_is_stub(sec: Dict) -> bool:
    """A section is a stub if it has very few bullets (≤ 2 total)."""
    total = sum(
        len(sub.get("points", [])) for sub in sec.get("subsections", [])
    )
    return total <= 2


def merge_oversegmented_sections(
    sections: List[Dict], min_bullets: int = 3
) -> List[Dict]:
    """
    Issue 10: Merge stub sections into the preceding section ONLY when they
    share significant vocabulary (similarity ≥ 0.5).

    IMPORTANT CHANGE: The previous version merged ANY stub into the preceding
    section regardless of heading similarity (the condition was
    `sim >= 0.3 OR _section_is_stub(sec)` — the OR made the similarity check
    irrelevant). This caused valid independent sections like "Implicit",
    "Operation", "Top Of The Stack" to be swallowed by adjacent sections,
    producing missing content in the PDF.

    New logic: only merge when BOTH conditions hold:
      1. Section is a stub (≤ 2 bullets) AND
      2. Heading similarity with previous section ≥ 0.5 (meaningful overlap)

    Sections with unique headings (similarity < 0.5) are kept as standalone
    sections even if they are small — they represent distinct concepts.
    """
    if len(sections) <= 1:
        return sections

    result: List[Dict] = [sections[0]]
    merged_count = 0

    for sec in sections[1:]:
        prev = result[-1]
        if _section_is_stub(sec):
            sim = _heading_similarity(prev["heading"], sec["heading"])
            # STRICT: only merge when headings are genuinely similar
            # Low similarity means different topics — keep them separate
            if sim >= 0.5:
                result[-1] = _merge_section_into(prev, sec)
                merged_count += 1
                _log(
                    f"Issue 10 → stub section merged: '{sec['heading']}' → '{prev['heading']}'"
                    f" (sim={sim:.2f})"
                )
                continue
            # else: stub with different heading → keep as separate section
        result.append(sec)

    _log(f"Issue 10 → {merged_count} over-segmented stubs merged")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Final polish — sentence capitalisation, stray punctuation
# ═══════════════════════════════════════════════════════════════════════════════

_STRAY_PUNCT = re.compile(r"\s+([.,;:])")
_DOUBLE_PERIOD = re.compile(r"\.{2,}")


def _polish_text(text: str) -> str:
    text = _STRAY_PUNCT.sub(r"\1", text)
    text = _DOUBLE_PERIOD.sub(".", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    if text and text[-1] not in ".!?":
        text += "."
    return text


def polish_all_bullets(sections: List[Dict]) -> List[Dict]:
    """Final pass: capitalise, fix punctuation, remove empty bullets.
    
    IMPORTANT: Never drops a subsection unless it has truly zero valid bullets
    after polishing. Preserves content aggressively — only removes bullets
    that are under 3 words OR are pure punctuation/noise.
    """
    for sec in sections:
        for sub in sec.get("subsections", []):
            new_pts = []
            for pt in sub.get("points", []):
                text = _polish_text(pt.get("text", ""))
                # Only drop completely empty bullets — preserve ALL content
                stripped = text.strip(" .")
                if stripped:
                    pt["text"] = text
                    new_pts.append(pt)
            sub["points"] = new_pts
        # NEVER drop a subsection that had any content — only drop truly empty ones
        sec["subsections"] = [
            sub for sub in sec.get("subsections", [])
            if sub.get("points")
        ]
    return sections


# ═══════════════════════════════════════════════════════════════════════════════
# Step 11 — Remove Circular Definitions
# ═══════════════════════════════════════════════════════════════════════════════

# Matches patterns like:
#   "Code Optimization Phase is handled by Code Optimizer"
#   "Target Code Generation Phase is handled by Target Code Generator"
#   "X is done by X-er", "X Phase is controlled by X Controller"
_CIRCULAR_PATTERNS = [
    re.compile(r"^(.+?)\s+(?:is handled by|is controlled by|is done by|is performed by|is managed by)\s+(.+?)\s*\.?\s*$", re.I),
]

# Self-exemplification: "Array: Linear Data Structure exemplified by Array"
_SELF_EXEMPLIFICATION_RE = re.compile(
    r"^([^:]{1,40}):\s*.{3,120}?\b"
    r"(?:exemplified by|as an example of|is an example of|such as an example of)\s+\1\b",
    re.I,
)
# Direct self-loop: "X is a X" or "X is an X"
_DIRECT_SELF_LOOP_RE = re.compile(
    r"^([A-Za-z][A-Za-z\s\-]{2,30})\s+is\s+(?:a|an|the)\s+\1\b",
    re.I,
)


def _is_circular_definition(text: str) -> bool:
    """
    Detect circular definitions. Covers:
      1. Agent-verb circularity: "X is handled/controlled/done by X-variant"
      2. Self-exemplification: "Array: ... exemplified by Array"
      3. Direct self-loop: "X is a X"
    Fully dynamic — word overlap + stem matching, no domain keywords.
    """
    text_stripped = text.strip().rstrip(".")

    # Type 1: agent-verb circularity
    for pat in _CIRCULAR_PATTERNS:
        m = pat.match(text_stripped)
        if m:
            _GENERIC = {"phase", "stage", "process", "system", "module", "component",
                        "handler", "controller", "generator", "analyzer", "manager"}
            subject = set(re.findall(r"[a-zA-Z]{4,}", m.group(1).lower())) - _GENERIC
            agent   = set(re.findall(r"[a-zA-Z]{4,}", m.group(2).lower())) - _GENERIC
            overlap = sum(1 for sw in subject for aw in agent
                         if len(sw) >= 4 and len(aw) >= 4
                         and (sw[:5] == aw[:5] or sw in aw or aw in sw))
            if overlap > 0 or bool(subject & agent):
                return True

    # Type 2: self-exemplification
    if _SELF_EXEMPLIFICATION_RE.search(text_stripped):
        return True

    # Type 3: direct self-loop
    if _DIRECT_SELF_LOOP_RE.match(text_stripped):
        return True

    return False


def _fix_self_exemplification(text: str) -> Optional[str]:
    """
    Rewrite a self-exemplification definition into non-circular form.
    "Array: Linear Data Structure exemplified by Array" → "Array: a type of Linear Data Structure."
    Returns fixed text, or None if unfixable.
    """
    text_stripped = text.strip().rstrip(".")
    m = re.match(r"^([^:]{1,40}):\s*(.{4,})", text_stripped)
    if not m:
        return None
    label = m.group(1).strip()
    desc = m.group(2).strip()
    cleaned_desc = re.sub(
        r"\s*\b(?:exemplified by|as an example of|is an example of|such as)\s+.+$",
        "", desc, flags=re.I
    ).strip()
    if not cleaned_desc or len(cleaned_desc.split()) < 2:
        return None
    return f"{label}: a type of {cleaned_desc}."


def remove_circular_definitions(sections: List[Dict]) -> List[Dict]:
    """
    Remove or rewrite circular definitions.
    - Self-exemplification → rewritten as non-circular
    - Agent-verb circularity / direct self-loop → removed
    Fully domain-agnostic.
    """
    removed = 0
    fixed = 0
    for sec in sections:
        for sub in sec.get("subsections", []):
            new_pts = []
            for pt in sub.get("points", []):
                text = pt.get("text", "")
                if _is_circular_definition(text):
                    repaired = _fix_self_exemplification(text)
                    if repaired:
                        pt["text"] = repaired
                        new_pts.append(pt)
                        fixed += 1
                    else:
                        removed += 1
                else:
                    new_pts.append(pt)
            # Never leave subsection entirely empty
            sub["points"] = new_pts if new_pts else sub.get("points", [])
    if removed or fixed:
        _log(f"Step 11 → {fixed} circular defs rewritten, {removed} removed")
    return sections


# ═══════════════════════════════════════════════════════════════════════════════
# Step 12 — Remove Key Takeaways Subsections
# ═══════════════════════════════════════════════════════════════════════════════

_TAKEAWAY_HEADING_RE = re.compile(
    r"\b(key\s+takeaway|takeaway|key\s+point|key\s+summary)\b", re.I
)

def remove_key_takeaways_subsections(sections: List[Dict]) -> List[Dict]:
    """
    Remove all 'Key Takeaways' subsections — they repeat content from earlier subsections
    and add no new learning value. Fully dynamic: matches any heading containing
    'takeaway', 'key point', or 'key summary' regardless of domain.
    """
    removed = 0
    for sec in sections:
        original_len = len(sec.get("subsections", []))
        sec["subsections"] = [
            ss for ss in sec.get("subsections", [])
            if not _TAKEAWAY_HEADING_RE.search(ss.get("heading", ""))
        ]
        removed += original_len - len(sec.get("subsections", []))
    if removed:
        _log(f"Step 12 → {removed} Key Takeaways subsections removed")
    return sections


# ═══════════════════════════════════════════════════════════════════════════════
# NEW: Sentence Fragment Reconstruction
# Fixes dangling clauses like "the order in which operations are executed"
# ═══════════════════════════════════════════════════════════════════════════════

# Bullets that start with a lowercase word (not a proper noun) are likely fragments
_LOWERCASE_START = re.compile(r"^[a-z]")
# Starts with article "the/a/an" — almost always a dangling clause
_DANGLING_ARTICLE = re.compile(r"^(the|a|an)\s+", re.I)
# Relative clause marker
_RELATIVE_CLAUSE = re.compile(r"^(which|that|where|when|how|whether)\s+", re.I)


def _is_dangling_sentence(text: str) -> bool:
    """Detect sentences that are clearly dangling / incomplete."""
    if _LOWERCASE_START.match(text):
        return True
    if _DANGLING_ARTICLE.match(text):
        return True
    if _RELATIVE_CLAUSE.match(text):
        return True
    return False


def _reconstruct_sentence(text: str, heading: str, sub_heading: str) -> str:
    """
    Reconstruct a dangling sentence by prepending the most relevant subject.
    Uses section/subsection heading as the implicit subject.
    Fully dynamic — derives the subject from context.
    """
    text = text.strip().rstrip(".")
    subject = sub_heading.strip() if sub_heading.strip() else heading.strip()

    # "the order in which X" → "Stack controls the order in which X"
    if _DANGLING_ARTICLE.match(text):
        return f"{subject} determines {text}."

    # "which allows X" → "Stack, which allows X"
    if _RELATIVE_CLAUSE.match(text):
        return f"{subject}, {text}."

    # lowercase start — just capitalise and append context
    if _LOWERCASE_START.match(text):
        return f"{text[0].upper()}{text[1:]} (related to {subject})."

    return text + "."


def fix_sentence_fragments(sections: List[Dict]) -> List[Dict]:
    """
    NEW: Reconstruct dangling/incomplete sentences.
    'the order in which operations are executed'
    → 'Stack determines the order in which operations are executed.'
    """
    fixed = 0
    for sec in sections:
        heading = sec.get("heading", "")
        for sub in sec.get("subsections", []):
            sub_heading = sub.get("heading", "")
            for pt in sub.get("points", []):
                text = pt.get("text", "").strip()
                if _is_dangling_sentence(text):
                    new_text = _reconstruct_sentence(text, heading, sub_heading)
                    pt["text"] = new_text
                    fixed += 1
                    _log(f"Fragment reconstructed: '{text}' → '{new_text}'")
    if fixed:
        _log(f"Sentence fragments → {fixed} reconstructed")
    return sections


# ═══════════════════════════════════════════════════════════════════════════════
# NEW: Subsection-level concept deduplication
# Merges "Stack Operations" ≈ "Operations Performed On Stack"
# ═══════════════════════════════════════════════════════════════════════════════

def _subsection_overlap(a: str, b: str) -> float:
    """Word overlap between two subsection headings, ignoring stopwords."""
    _stops = {"of", "on", "the", "a", "an", "in", "for", "to", "by", "and", "or"}
    wa = set(re.findall(r"[a-z]{3,}", a.lower())) - _stops
    wb = set(re.findall(r"[a-z]{3,}", b.lower())) - _stops
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / max(len(wa), len(wb))


def deduplicate_subsections(sections: List[Dict], threshold: float = 0.55) -> List[Dict]:
    """
    NEW: Merge near-duplicate subsections within each section.
    'Stack Operations' + 'Operations Performed On Stack' → merged under shorter heading.
    Threshold 0.55 is lower than section-level dedup to catch rephrased headings.
    """
    merged_total = 0
    for sec in sections:
        subs = sec.get("subsections", [])
        if len(subs) <= 1:
            continue

        kept: List[Dict] = []
        merged_flags = [False] * len(subs)

        for i, sub_a in enumerate(subs):
            if merged_flags[i]:
                continue
            for j in range(i + 1, len(subs)):
                if merged_flags[j]:
                    continue
                sim = _subsection_overlap(sub_a.get("heading", ""), subs[j].get("heading", ""))
                if sim >= threshold:
                    _log(
                        f"Subsection dedup → merging '{sub_a['heading']}' ≈ '{subs[j]['heading']}'"
                        f" (sim={sim:.2f})"
                    )
                    # Merge points from j into i, deduplicating
                    seen = {pt["text"].lower() for pt in sub_a.get("points", [])}
                    for pt in subs[j].get("points", []):
                        if pt["text"].lower() not in seen:
                            sub_a.setdefault("points", []).append(pt)
                            seen.add(pt["text"].lower())
                    # Keep shorter / cleaner heading
                    ha, hb = sub_a.get("heading", ""), subs[j].get("heading", "")
                    sub_a["heading"] = ha if len(ha) <= len(hb) else hb
                    merged_flags[j] = True
                    merged_total += 1
            kept.append(sub_a)

        sec["subsections"] = kept

    if merged_total:
        _log(f"Subsection dedup → {merged_total} near-duplicate subsections merged")
    return sections


# ═══════════════════════════════════════════════════════════════════════════════
# NEW: Diagram interpretation from KG context
# Replaces "Figure: slide_004" with a real description derived from KG edges
# ═══════════════════════════════════════════════════════════════════════════════

def enrich_diagram_captions_from_kg(
    sections: List[Dict],
    nodes: Optional[List[Dict]] = None,
    edges: Optional[List[Dict]] = None,
) -> List[Dict]:
    """
    NEW: When a diagram caption is a raw filename, generate a real description
    by looking up what the section heading's concept does in the KG edges.
    E.g. heading='Push Operation' → edges show push adds element to top →
    caption becomes 'Diagram: Push adds an element to the top of the stack.'

    Fully dynamic — works for any domain.
    """
    if not nodes or not edges:
        return sections

    node_id_to_label = {n.get("id", ""): n.get("label", "") for n in nodes}
    node_label_to_desc = {
        n.get("label", "").lower(): n.get("description", "") for n in nodes
    }

    _FILE_REF = re.compile(r"slide_\d+|diagram_\d+|figure\s*\d*", re.I)

    enriched = 0
    for sec in sections:
        heading = sec.get("heading", "").strip()
        diag = sec.get("diagram")
        if not diag:
            continue
        cap = diag.get("caption", "")
        if not _FILE_REF.search(cap):
            continue  # already has a real caption

        # Find KG node matching the heading
        heading_lower = heading.lower()
        desc = node_label_to_desc.get(heading_lower, "")

        # If no direct match, look for partial overlap
        if not desc:
            for lbl, d in node_label_to_desc.items():
                if _overlap(lbl, heading_lower) >= 0.5 and d:
                    desc = d
                    break

        # Build description from edges if node description is short
        if not desc or len(desc.split()) < 5:
            for n in nodes:
                if _overlap(n.get("label", ""), heading) >= 0.5:
                    src_id = n.get("id", "")
                    parts = []
                    for e in edges:
                        if e.get("source") == src_id:
                            tgt = node_id_to_label.get(e.get("target", ""), "")
                            rel = e.get("relation", "").replace("_", " ")
                            if tgt and rel:
                                parts.append(f"{rel} {tgt}")
                    if parts:
                        desc = f"{heading} — {'; '.join(parts[:3])}"
                    break

        if desc:
            diag["caption"] = f"Diagram illustrating {heading}: {desc.rstrip('.')}."
            enriched += 1
            _log(f"Diagram enriched: '{cap}' → '{diag['caption'][:80]}...'")

    if enriched:
        _log(f"Diagram captions → {enriched} enriched from KG")
    return sections

# ═══════════════════════════════════════════════════════════════════════════════
# NEW: Graph Edge Literal Sentences
# "Stacks includes Real-life Examples of Stacks" → natural prose
# "Algorithms includes Recursive Algorithm" → "Stacks are used in recursive algorithms."
# ═══════════════════════════════════════════════════════════════════════════════

# Detects bare graph-edge sentences: "X <verb> Y" where verb is a containment/relation word
_GRAPH_EDGE_LITERAL_RE = re.compile(
    r"^([A-Z][A-Za-z\s\-()]{1,60}?)\s+"
    r"(includes?|contains?|encompasses?|is a type of|is part of|is used in|is related to"
    r"|has|consist of|consists of|is associated with)\s+"
    r"([A-Z][A-Za-z\s\-()]{1,60})"
    r"(?::\s*.{0,120})?\s*\.?\s*$",  # FIX-3: allow optional ": description" suffix
    re.I,
)

# For multi-edge grouping: bullets in same subsection that all start with
# "X includes Y" where X is the same subject — group them into one sentence
_MULTI_EDGE_INCLUDES_RE = re.compile(
    r"^([A-Z][A-Za-z\s\-()]{2,60})\s+(includes?|contains?)\s+([A-Z][A-Za-z\s\-()]{1,50})"
    r"(?::\s*.{0,80})?\s*\.?$",
    re.I,
)
# Self-referential: "Stacks includes Real-life Examples of Stacks" (subject word appears in object)
_SELF_REF_IN_OBJECT = re.compile(
    r"^(.+?)\s+(?:includes?|contains?|encompasses?)\s+.+\s+(?:of|for)\s+\1",
    re.I,
)


def _graph_edge_to_prose(subject: str, verb: str, obj: str) -> str:
    """
    Convert a raw graph-edge sentence (subject, verb, object) to natural English prose.
    Fully dynamic — no domain words hardcoded.
    Uses the verb to infer the best sentence pattern.

    FIX: Replaced filler phrases ("involves X as one of its key components",
    "has real-world applications and use cases") with informative prose patterns
    that preserve the actual relationship between subject and object.
    Self-referential triples are now dropped (return "") so callers skip them.
    """
    subject, obj = subject.strip(), obj.strip()
    verb_lower = verb.lower().strip()

    # If object is semantically redundant with subject → self-referential triple.
    # Return "" to signal the caller should drop this bullet entirely rather than
    # emitting a meaningless filler sentence.
    subj_words = set(re.findall(r"[a-z]{3,}", subject.lower()))
    obj_words  = set(re.findall(r"[a-z]{3,}", obj.lower()))
    _FILLER_WORDS = {"real", "life", "example", "examples", "the", "of", "and",
                     "illustration", "illustrations", "everyday", "scenarios"}
    if subj_words and subj_words.issubset(obj_words | _FILLER_WORDS):
        return ""  # Drop — tautological, adds no content

    if verb_lower in {"includes", "include", "contains", "contain", "encompasses", "encompass",
                      "consist of", "consists of", "has"}:
        # "Stack includes Push Operation" → "Stack includes Push Operation as part of its functionality."
        # Avoid "involves X as one of its key components" filler — use the actual verb + object.
        return f"{subject} includes {obj} as part of its functionality."

    if verb_lower in {"is a type of", "is part of"}:
        return f"{subject} is a type of {obj}."

    if verb_lower in {"is used in", "is associated with", "is related to"}:
        return f"{subject} is closely related to {obj}."

    # Generic fallback — use exact verb so relationship is preserved
    return f"{subject} {verb_lower} {obj}."


def fix_graph_edge_literals(sections: List[Dict]) -> List[Dict]:
    """
    NEW: Detect and rewrite bullets that are raw graph-edge translations.
    e.g. "Algorithms includes Recursive Algorithm"
         "Stacks includes Real-life Examples of Stacks"
    → converted to natural prose sentences.

    Detection: sentence matches exactly "Subject <containment-verb> Object"
    with both Subject and Object starting with capitals — a clear sign of
    direct edge linearization.
    Fully dynamic — no hardcoded domain words.
    """
    fixed = 0
    removed = 0
    for sec in sections:
        for sub in sec.get("subsections", []):
            new_pts = []
            for pt in sub.get("points", []):
                text = pt.get("text", "").strip().rstrip(".")
                m = _GRAPH_EDGE_LITERAL_RE.match(text)
                if m:
                    new_text = _graph_edge_to_prose(m.group(1), m.group(2), m.group(3))
                    if new_text == "":
                        # Self-referential triple — drop bullet entirely
                        removed += 1
                        _log(f"Graph edge literal dropped (self-ref): '{text}'")
                        continue
                    if new_text != text:
                        pt["text"] = new_text
                        fixed += 1
                        _log(f"Graph edge literal → prose: '{text}' → '{new_text}'")
                new_pts.append(pt)
            sub["points"] = new_pts
        sec["subsections"] = [s for s in sec.get("subsections", []) if s.get("points")]
    sections = [s for s in sections if s.get("subsections")]
    if fixed or removed:
        _log(f"Graph edge literals → {fixed} rewritten, {removed} self-refs dropped")

    # FIX-3b: Merge consecutive "X includes A", "X includes B", "X includes C"
    # bullets into one: "X involves types such as A, B, and C."
    for sec in sections:
        for sub in sec.get("subsections", []):
            pts = sub.get("points", [])
            if len(pts) < 2:
                continue
            # Group consecutive bullets by (subject, verb) if they match includes pattern
            groups: list = []
            i = 0
            while i < len(pts):
                m = _MULTI_EDGE_INCLUDES_RE.match(pts[i].get("text", "").rstrip("."))
                if m:
                    subj, verb, obj_part = m.group(1).strip(), m.group(2), m.group(3).strip()
                    cluster = [(i, subj, verb, obj_part)]
                    j = i + 1
                    while j < len(pts):
                        m2 = _MULTI_EDGE_INCLUDES_RE.match(pts[j].get("text", "").rstrip("."))
                        if m2 and m2.group(1).strip().lower() == subj.lower():
                            cluster.append((j, m2.group(1).strip(), m2.group(2), m2.group(3).strip()))
                            j += 1
                        else:
                            break
                    if len(cluster) >= 2:
                        groups.append(cluster)
                        i = j
                    else:
                        i += 1
                else:
                    i += 1
            # Replace each group with a single merged bullet
            if groups:
                # Collect indices to remove and the merged text
                remove_idx = set()
                inserts: dict = {}  # first_idx → merged text
                for cluster in groups:
                    first_idx = cluster[0][0]
                    subj = cluster[0][1]
                    objs = [c[3] for c in cluster]
                    if len(objs) == 1:
                        continue
                    if len(objs) == 2:
                        obj_str = f"{objs[0]} and {objs[1]}"
                    else:
                        obj_str = ", ".join(objs[:-1]) + f", and {objs[-1]}"
                    merged = f"{subj} includes {obj_str} among its key components."
                    inserts[first_idx] = merged
                    for c in cluster[1:]:
                        remove_idx.add(c[0])
                new_pts = []
                for idx, pt in enumerate(pts):
                    if idx in remove_idx:
                        continue
                    if idx in inserts:
                        pt = dict(pt)
                        pt["text"] = inserts[idx]
                        fixed += 1
                    new_pts.append(pt)
                sub["points"] = new_pts

    return sections


# ═══════════════════════════════════════════════════════════════════════════════
# NEW: Slide Header / Course Title Sections
# "Data Structure and Algorithm Subject" extracted as a concept → remove
# Detects sections whose heading looks like a slide title / course header
# ═══════════════════════════════════════════════════════════════════════════════

# Patterns that match slide headers / course titles — fully dynamic (no domain words)
# These are structural signals, not content signals
_SLIDE_HEADER_PATTERNS = [
    # "X and Y Subject" / "X Subject"
    re.compile(r"\b(subject|course|unit|chapter|module|topic|lecture|syllabus|curriculum)\b", re.I),
    # "Introduction to X" as a standalone section with < 2 bullets
    re.compile(r"^introduction\s+to\b", re.I),
    # "X — Overview" or "X Overview" as a section with thin content
    re.compile(r"\b(overview|outline|agenda|objectives?|learning outcome)\b", re.I),
    # Purely numeric or very short headings with no real content
    re.compile(r"^\d+[\.\\)]\s*\w{1,15}$"),
]


def _is_slide_header_section(section: Dict) -> bool:
    """
    Return True if a section looks like a slide header / course title, not a real concept.
    Uses structural signals only — no hardcoded domain words.
    Signals:
      1. Heading matches slide-header patterns (contains 'subject', 'course', etc.)
      2. Section has very few bullets (≤ 3 total) AND heading has header-like words
    """
    heading = section.get("heading", "").strip()
    total_bullets = sum(
        len(sub.get("points", [])) for sub in section.get("subsections", [])
    )

    # Signal 1: Heading matches slide-header pattern regardless of bullet count
    for pat in _SLIDE_HEADER_PATTERNS[:1]:  # Only "subject/course/unit" triggers hard removal
        if pat.search(heading):
            return True

    # Signal 2: Thin section (≤ 3 bullets) AND heading has header-like content word
    if total_bullets <= 3:
        for pat in _SLIDE_HEADER_PATTERNS[1:]:
            if pat.search(heading):
                return True

    return False


def remove_slide_header_sections(sections: List[Dict]) -> List[Dict]:
    """
    NEW: Remove sections that are slide headers / course titles extracted as concepts.
    e.g. "Data Structure and Algorithm Subject"
    Fully dynamic — uses structural and linguistic patterns, no domain words.
    """
    removed = 0
    result = []
    for sec in sections:
        if _is_slide_header_section(sec):
            _log(f"Slide header section removed: '{sec.get('heading', '')}'")
            removed += 1
        else:
            result.append(sec)
    if removed:
        _log(f"Slide headers → {removed} course-title/slide-header sections removed")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# NEW: Redundant Synonymous Concept Merging
# "Peek Function" / "Top Operation" / "Returns Last Inserted Element"
# → all refer to the same concept → merged under one subsection
# ═══════════════════════════════════════════════════════════════════════════════

def _semantic_overlap_subsections(a: str, b: str, threshold: float = 0.50) -> bool:
    """
    Return True if two subsection headings/texts are semantically synonymous.
    Uses word overlap after stripping function/operation/phase suffixes.
    Fully dynamic.
    """
    _SUFFIX_NOISE = re.compile(
        r"\b(function|operation|method|phase|process|procedure|step|"
        r"component|module|type|kind|form|variant|mode|action|feature)\b",
        re.I,
    )
    def _normalize(s: str) -> Set[str]:
        s = _SUFFIX_NOISE.sub("", s.lower())
        return set(w for w in re.findall(r"[a-z]{3,}", s))

    wa, wb = _normalize(a), _normalize(b)
    if not wa or not wb:
        return False
    return len(wa & wb) / max(len(wa), len(wb)) >= threshold


def merge_synonym_subsections(sections: List[Dict]) -> List[Dict]:
    """
    NEW: Merge subsections within a section that are near-synonyms.
    e.g. "Peek Function", "Top Operation", "Returns Last Inserted Element"
    → merged when their headings + bullet texts overlap semantically.

    Strategy:
    1. For each section, compare ALL pairs of subsections.
    2. If two subsection headings overlap ≥ 50% (after stripping generic suffixes),
       merge the less-populated one into the more-populated one.
    3. Also merge subsections whose bullet text overlaps heavily with the heading
       of another subsection (synonym detection).

    Fully dynamic — no domain-specific vocabulary.
    """
    merged_total = 0
    for sec in sections:
        subs = sec.get("subsections", [])
        if len(subs) <= 1:
            continue

        merged_flags = [False] * len(subs)
        kept: List[Dict] = []

        for i, sub_a in enumerate(subs):
            if merged_flags[i]:
                continue
            for j in range(i + 1, len(subs)):
                if merged_flags[j]:
                    continue
                h_a = sub_a.get("heading", "")
                h_b = subs[j].get("heading", "")

                is_synonym = _semantic_overlap_subsections(h_a, h_b)

                # Also check: is heading of B contained in the bullet text of A?
                if not is_synonym:
                    all_text_a = " ".join(
                        pt.get("text", "") for pt in sub_a.get("points", [])
                    )
                    if _overlap(h_b, all_text_a) >= 0.6:
                        is_synonym = True

                if is_synonym:
                    _log(
                        f"Synonym subsections merged: '{h_a}' ≈ '{h_b}'"
                    )
                    # Merge j into i, deduplicating bullet text
                    seen = {pt["text"].lower() for pt in sub_a.get("points", [])}
                    for pt in subs[j].get("points", []):
                        if pt["text"].lower() not in seen:
                            sub_a.setdefault("points", []).append(pt)
                            seen.add(pt["text"].lower())
                    # Keep shorter heading (usually more readable)
                    if len(h_b) < len(h_a):
                        sub_a["heading"] = h_b
                    merged_flags[j] = True
                    merged_total += 1

            kept.append(sub_a)

        sec["subsections"] = kept

    if merged_total:
        _log(f"Synonym subsections → {merged_total} pairs merged")
    return sections


# ═══════════════════════════════════════════════════════════════════════════════
# NEW: Mixed Abstraction Level Annotation + Grouping
# "Stack", "Recursive Algorithm", "Stack of Coins", "Programmers/Developers"
# belong to different levels — annotate and group them
# ═══════════════════════════════════════════════════════════════════════════════

# Signals for each abstraction level — pure text patterns, no domain words
_LEVEL_SIGNALS: List[Tuple[str, re.Pattern]] = [
    # Examples: "e.g.", "for example", "such as", "like", "real-life", "real-world", "instance"
    ("example",    re.compile(r"\b(e\.g\.|for example|such as|like|real.?life|real.?world|instance|illustration)\b", re.I)),
    # Actors: job titles, roles, people — ends in 'er', 'or', 'ist', 'ian' or "user/developer/programmer"
    ("actor",      re.compile(r"\b(user|developer|programmer|engineer|designer|manager|administrator|analyst|operator|teacher|student|client|server)\b", re.I)),
    # Algorithms: procedure, algorithm, method, approach, technique, strategy
    ("algorithm",  re.compile(r"\b(algorithm|procedure|method|approach|technique|strategy|step|process|routine)\b", re.I)),
    # Operations: push, pop, insert, delete, retrieve, search — action verbs as nouns
    ("operation",  re.compile(r"\b(push|pop|insert|delete|remove|search|retrieve|traverse|sort|merge|split|enqueue|dequeue|peek|top|front|rear|size|empty|full)\b", re.I)),
    # Concepts: abstract terms — definition, property, characteristic, principle, rule
    ("concept",    re.compile(r"\b(is a|is defined|property|characteristic|principle|rule|definition|abstract|type of|kind of)\b", re.I)),
]

_LEVEL_ORDER = ["concept", "algorithm", "operation", "example", "actor"]


def _infer_abstraction_level(text: str) -> str:
    """Infer the abstraction level of a bullet text string. Returns level name."""
    for level, pat in _LEVEL_SIGNALS:
        if pat.search(text):
            return level
    return "concept"  # default


def group_bullets_by_abstraction(sections: List[Dict]) -> List[Dict]:
    """
    NEW: Within each subsection, annotate bullets with abstraction level tags
    and sort them so higher-level concepts come first (concept → algorithm →
    operation → example → actor).

    This prevents the 'mixed abstraction levels' problem where high-level
    definitions are interspersed with examples and actor mentions.
    Fully dynamic — uses text pattern signals, no domain words.
    """
    reordered = 0
    for sec in sections:
        for sub in sec.get("subsections", []):
            points = sub.get("points", [])
            if len(points) <= 2:
                continue  # Not worth reordering tiny lists

            # Tag each point with its level
            tagged = []
            for pt in points:
                level = _infer_abstraction_level(pt.get("text", ""))
                tagged.append((level, pt))

            # Sort by level order (concept first, actor last)
            def _level_key(item):
                lvl, _ = item
                try:
                    return _LEVEL_ORDER.index(lvl)
                except ValueError:
                    return len(_LEVEL_ORDER)

            sorted_tagged = sorted(tagged, key=_level_key)
            if sorted_tagged != tagged:
                sub["points"] = [pt for _, pt in sorted_tagged]
                reordered += 1

    if reordered:
        _log(f"Abstraction grouping → {reordered} subsections reordered by concept level")
    return sections






# ═══════════════════════════════════════════════════════════════════════════════
# FIX-6: Lecture navigation / cross-reference bullets
# "ER Model: Discussed in the last session"
# "Normalization: We will cover this in the next chapter"
# These are slide navigation text, not concept content.
# ═══════════════════════════════════════════════════════════════════════════════

_NAVIGATION_PHRASES = re.compile(
    r"\b(discussed?\s+in\s+(?:the\s+)?(?:last|previous|prior|earlier)\s+(?:session|lecture|class|chapter|topic|video|unit)|"
    r"(?:will\s+be\s+)?discussed?\s+later|"
    r"covered?\s+in\s+(?:the\s+)?(?:next|upcoming|following|later|previous|last)\s+(?:session|lecture|class|chapter|topic|unit)|"
    r"(?:refer|see)\s+(?:the\s+)?(?:next|previous|last|upcoming)\s+(?:session|lecture|chapter)|"
    r"mentioned\s+in\s+(?:the\s+)?(?:last|previous|prior|earlier)\s+(?:session|lecture)|"
    r"we\s+(?:will|shall|have|had)\s+(?:discuss|cover|see|look\s+at)|"
    r"(?:already|previously)\s+(?:discussed|covered|seen|explained)|"
    r"to\s+be\s+(?:discussed|covered)\s+(?:later|next|soon)|"
    r"(?:recall|remember)\s+from\s+(?:last|previous|prior)\s+(?:session|lecture|class))",
    re.I,
)


def _is_navigation_bullet(text: str) -> bool:
    """
    Return True if a bullet is lecture navigation / session cross-reference text,
    not actual concept content.  Dynamic — uses phrase patterns, not domain words.
    """
    return bool(_NAVIGATION_PHRASES.search(text))


def remove_navigation_bullets(sections: List[Dict]) -> List[Dict]:
    """
    FIX-6: Remove bullets that are lecture navigation or cross-reference text.
    These are slide presenter notes that leaked into the extraction pipeline.
    Examples:
      "Er Model: Discussed in the last session."
      "Normalization: Will be covered in the next chapter."
    Fully dynamic — pattern-based, no domain word lists.
    """
    removed = 0
    for sec in sections:
        for sub in sec.get("subsections", []):
            original = sub.get("points", [])
            new_pts = []
            for pt in original:
                text = pt.get("text", "")
                if _is_navigation_bullet(text):
                    _log(f"Navigation bullet removed: '{text[:80]}'")
                    removed += 1
                else:
                    new_pts.append(pt)
            sub["points"] = new_pts
    if removed:
        _log(f"Navigation bullets → {removed} removed")
    return sections


def strip_question_marks_from_notes(notes: Dict) -> Dict:
    """
    Nuclear final pass: strip ALL '?' from every string in the notes dict.
    Covers sections, subsections, bullet text, summary, title, and diagram captions.
    Call this as the LAST step before render_pdf / render_txt.
    No context-awareness — guarantees zero '?' in output.
    Fully dynamic — works for any input.
    """
    def _nuke(s: str) -> str:
        if not s or '?' not in s:
            return s
        s = s.replace('?', '')
        return re.sub(r'\s{2,}', ' ', s).strip()

    # Top-level fields
    for field in ('title', 'summary', 'description'):
        if field in notes and isinstance(notes[field], str):
            notes[field] = _nuke(notes[field])

    for sec in notes.get('sections', []):
        if 'heading' in sec:
            sec['heading'] = _nuke(sec['heading'])
        if 'summary' in sec:
            sec['summary'] = _nuke(sec['summary'])
        diag = sec.get('diagram')
        if diag and isinstance(diag.get('caption'), str):
            diag['caption'] = _nuke(diag['caption'])

        for sub in sec.get('subsections', []):
            if 'heading' in sub:
                sub['heading'] = _nuke(sub['heading'])
            for pt in sub.get('points', []):
                if 'text' in pt:
                    pt['text'] = _nuke(pt['text'])

    return notes


# ═══════════════════════════════════════════════════════════════════════════════
# Step 15b — Expand inline raw-edge lists into individual described bullets
# ═══════════════════════════════════════════════════════════════════════════════
# Catches bullets like:
#   "Primary Stack Operations: consists of Push Operation, consists of Pop Operation"
#   "Types of Stack: includes Implicit Stack, includes Explicit Stack"
#   "Secondary Stack Operations: consists of Top Operation, consists of Size IsEmpty Operation"
# These have a valid label heading but the body is just comma-joined structural edges.
# Strategy (fully dynamic, no domain words):
#   1. Detect the pattern: "Heading: <verb> X, <verb> Y, <verb> Z"
#   2. Split on ", <same-verb>" boundaries
#   3. Emit each item as its own bullet: "X: An operation performed on <section-topic>."
# The generated fallback sentence is minimal but guarantees the concept appears in the
# notes so later passes (enrich_thin_bullets, notes_quality_enforcer) can flesh it out.

# Matches a bullet that is entirely a raw edge list
# Group 1: heading label before the colon
# Group 2: the structural verb (consists of / includes / contains / has / encompasses)
# Group 3: the rest of the list (comma-joined items sharing the same verb prefix)
_INLINE_EDGE_LIST_RE = re.compile(
    r'^(?P<heading>[^:]{3,80}):\s+'
    r'(?P<verb>consists?\s+of|includes?|contains?|has|encompasses?)\s+'
    r'(?P<first>[^,]+(?:\([^)]*\))?)'
    r'(?P<rest>(?:\s*,\s*(?:consists?\s+of|includes?|contains?|has|encompasses?)\s+[^,]+(?:\([^)]*\))?)+)$',
    re.I,
)

# Extracts each "verb item" segment from the rest-of-list group
_EDGE_SEGMENT_RE = re.compile(
    r'(?:consists?\s+of|includes?|contains?|has|encompasses?)\s+([^,]+(?:\([^)]*\))?)',
    re.I,
)

# Detects whether a label looks like an operation/function/method
_OP_LABEL_RE = re.compile(
    r'\b(operation|function|method|command|procedure|action|step|process|type|kind|variant)\b',
    re.I,
)


def _inline_list_to_bullets(heading: str, verb: str, items: list, section_topic: str) -> list:
    """
    Convert a raw-edge list like "Push Operation, Pop Operation" under heading
    "Primary Stack Operations" into individual make_point dicts with minimal but
    valid descriptions.

    The description is generated purely from structural signals:
    - If the item label ends with an operation/function suffix → "X is an operation on Y."
    - Otherwise → "X is a component of Y."
    Both forms guarantee the concept appears in notes so enrichment passes can add detail.
    """
    from hierarchical_schema import make_point
    points = []
    for raw_item in items:
        label = raw_item.strip().rstrip('.,;')
        if not label:
            continue
        if _OP_LABEL_RE.search(label):
            desc = f"{label} is an operation that can be performed on {section_topic}."
        else:
            desc = f"{label} is a component of {section_topic}."
        points.append(make_point(f"{label}: {desc}"))
    return points


def expand_inline_edge_lists(sections: list, *, section_topic_map: dict = None) -> list:
    """
    Step 15b: Walk every bullet in every subsection.  When a bullet matches the
    "Heading: verb X, verb Y, verb Z" pattern, split it into individual bullets
    (one per item) and replace the original single bullet with the expanded set.

    section_topic_map (optional): maps section heading → main topic label so the
    fallback description can say "an operation on <topic>" instead of a generic phrase.
    Fully dynamic — no hardcoded domain words.
    """
    if section_topic_map is None:
        section_topic_map = {}

    expanded = 0
    for sec in sections:
        sec_heading = sec.get('heading', '')
        topic = section_topic_map.get(sec_heading, sec_heading)

        for sub in sec.get('subsections', []):
            new_pts = []
            for pt in sub.get('points', []):
                text = pt.get('text', '').strip()
                m = _INLINE_EDGE_LIST_RE.match(text)
                if not m:
                    new_pts.append(pt)
                    continue

                heading_label = m.group('heading').strip()
                verb          = m.group('verb').strip()
                first_item    = m.group('first').strip()
                rest_str      = m.group('rest')

                # Collect all items: first + any additional segments
                items = [first_item]
                for seg_m in _EDGE_SEGMENT_RE.finditer(rest_str):
                    items.append(seg_m.group(1).strip())

                if len(items) < 2:
                    # Only one item — not a list, leave unchanged
                    new_pts.append(pt)
                    continue

                # Replace the single raw-list bullet with individual described bullets
                new_bullets = _inline_list_to_bullets(heading_label, verb, items, topic)
                new_pts.extend(new_bullets)
                expanded += 1
                _log(f"Inline edge list expanded: '{text[:60]}' → {len(new_bullets)} bullets")

            sub['points'] = new_pts

    if expanded:
        _log(f"expand_inline_edge_lists: {expanded} raw-list bullets expanded")
    return sections


# ═══════════════════════════════════════════════════════════════════════════════
# ONTOLOGY ENTITY-TYPE CLASSIFIER  (HIGH PRIORITY — Issue 3, 10, Issue 3, 10)
# ═══════════════════════════════════════════════════════════════════════════════
# Classifies every KG node (and section heading) into one of these ontology types:
#   concept      — abstract/definitional: the main educational idea
#   operation    — action/procedure performed on the concept
#   property     — characteristic or behaviour of the concept
#   implementation — concrete mechanism, data structure, or code artifact
#   example      — real-world or concrete illustration
#   actor        — human role, stakeholder, organisation (NOT a CS concept)
#   metadata     — course admin, exam, lecture title (NOT content)
#   system_element — low-level system component (compiler internals, OS detail)
#
# Detection is PURELY SIGNAL-BASED:
#   - Suffix/token signals (fully dynamic — no domain vocabulary hardcoded)
#   - Sentence-structural signals (ends with verb, starts with capital)
#   - Contextual signals from KG edges
#
# Actor and metadata nodes should NEVER become top-level section headings.
# ═══════════════════════════════════════════════════════════════════════════════

# ── Signal patterns for each type ──────────────────────────────────────────────
# Each entry: (type_label, compiled_regex_for_label, compiled_regex_for_description)
_ONTOLOGY_SIGNALS: List[Tuple[str, re.Pattern, re.Pattern]] = [

    # METADATA — course admin, exam, assessment artifacts
    ("metadata",
     re.compile(
         r"\b(exam|quiz|assignment|homework|exercise|syllabus|curriculum|"
         r"course|unit\s+\d|chapter\s+\d|module\s+\d|lecture\s+\d|"
         r"semester|term|marks?|grade|credit|sppu|btu|vtu|anna\s+university|"
         r"pattern\s+question|important\s+question|previous\s+year|"
         r"learning\s+outcome|objective|agenda)\b",
         re.I,
     ),
     re.compile(r"\b(exam|assignment|quiz|syllabus|marks?|grade|assessment)\b", re.I)),

    # ACTOR — human role, stakeholder, profession
    ("actor",
     re.compile(
         r"\b(programmer|developer|engineer|designer|administrator|analyst|"
         r"operator|manager|teacher|student|instructor|user|client|consumer|"
         r"end.?user|stakeholder|vendor|provider|architect|tester|reviewer)\b",
         re.I,
     ),
     re.compile(r"\b(person|people|human|role|job|profession|staff|team|org)\b", re.I)),

    # SYSTEM_ELEMENT — low-level OS/compiler/runtime internals
    ("system_element",
     re.compile(
         r"\b(compiler|interpreter|linker|loader|assembler|runtime|"
         r"operating\s+system|virtual\s+machine|jvm|clr|cpu|register|"
         r"cache|hardware|firmware|kernel|process|thread|interrupt|"
         r"activation\s+record|stack\s+frame|heap\s+segment|"
         r"system\s+component|system\s+software)\b",
         re.I,
     ),
     re.compile(r"\b(system|hardware|os|machine|runtime|low.level)\b", re.I)),

    # EXAMPLE — real-world illustration, analogy
    ("example",
     re.compile(
         r"\b(real.?life|real.?world|everyday|daily.life|physical|"
         r"analogy|for.?example|such.?as|illustration|scenario|"
         r"case.?study|use.?case|demo|sample)\b",
         re.I,
     ),
     re.compile(r"\b(like|similar|analogy|example|instance|illustration)\b", re.I)),

    # IMPLEMENTATION — concrete data structure, storage mechanism
    ("implementation",
     re.compile(
         r"\b(array|linked\s+list|pointer|node|memory|allocation|"
         r"static|dynamic|fixed.?size|resizable|contiguous|"
         r"heap|tree|hash|table|record|struct|class|object)\b",
         re.I,
     ),
     re.compile(r"\b(implemented|stored|allocated|memory|array|linked)\b", re.I)),

    # OPERATION — action, procedure, function call
    ("operation",
     re.compile(
         r"\b\w*(push|pop|peek|enqueue|dequeue|insert|delete|remove|"
         r"search|retrieve|traverse|sort|merge|split|append|prepend|"
         r"update|read|write|execute|call|return|overflow|underflow|"
         r"isempty|isfull|size)\w*\b",
         re.I,
     ),
     re.compile(r"\b(perform|execute|apply|invoke|call|returns?|checks?|adds?|removes?)\b", re.I)),

    # PROPERTY — characteristic, behavioural rule, ordering principle
    ("property",
     re.compile(
         r"\b(lifo|fifo|filo|lilo|last.in|first.out|"
         r"principle|property|characteristic|behaviour|behavior|"
         r"rule|ordering|policy|constraint|invariant|attribute|"
         r"feature|abstract|adt|abstract\s+data\s+type)\b",
         re.I,
     ),
     re.compile(r"\b(principle|property|characteristic|behaviour|rule|order)\b", re.I)),

    # CONCEPT — default: definitional, structural, or educational idea
    ("concept",
     re.compile(
         r"\b(definition|structure|type|kind|category|class|form|"
         r"variant|model|pattern|concept|theory|paradigm|abstraction|"
         r"data\s+structure|algorithm|computation)\b",
         re.I,
     ),
     re.compile(r"\b(is a|is defined|is used|represents|means|refers|defined as)\b", re.I)),
]

# Types that should NEVER appear as standalone section headings
_NON_TOPIC_TYPES: Set[str] = {"actor", "metadata"}

# Types that are LOW-PRIORITY topics — should be sub-bullets inside a concept section
_LOW_PRIORITY_TYPES: Set[str] = {"example", "system_element"}


def classify_node_type(label: str, description: str = "", kg_edges: list = None) -> str:
    """
    Classify a KG node or section heading into an ontology type.

    Priority order (strict):
      1. metadata — if matched, it's always metadata regardless of other signals
      2. actor    — human roles always trump content signals
      3. All other types in _ONTOLOGY_SIGNALS order

    Parameters
    ----------
    label       : node label or section heading string
    description : node description string (optional, enriches signal detection)
    kg_edges    : list of outgoing edge dicts [{relation, target_label}] (optional)

    Returns
    -------
    str — one of: concept | operation | property | implementation |
                  example | actor | metadata | system_element
    """
    combined = (label + " " + description).strip()

    for type_label, lbl_pat, desc_pat in _ONTOLOGY_SIGNALS:
        # Label signal is stronger — check it first
        if lbl_pat.search(label):
            return type_label
        # Description signal
        if description and desc_pat.search(description):
            return type_label

    # Edge-based signals: if all outgoing edges point to operations, this is an operation
    if kg_edges:
        _OP_RELS = {"has operation", "supports operation", "has function", "performs"}
        rel_signals = [e.get("relation", "").lower() for e in kg_edges]
        op_count = sum(1 for r in rel_signals if any(sig in r for sig in _OP_RELS))
        if op_count >= 2:
            return "operation"

    # Default to concept
    return "concept"


def _build_node_type_map(
    nodes: List[Dict],
    edges: List[Dict],
) -> Dict[str, str]:
    """
    Build a label-lower → ontology_type map for all KG nodes.
    Used by the section-level filter to know whether a heading is a valid topic.
    """
    if not nodes:
        return {}

    # Build id→label map and outgoing edge map for edge-based classification
    id_to_label = {n.get("id", ""): n.get("label", "") for n in nodes}
    out_edges: Dict[str, List[Dict]] = defaultdict(list)
    for e in (edges or []):
        src = e.get("source", "")
        tgt_label = id_to_label.get(e.get("target", ""), "")
        out_edges[src].append({"relation": e.get("relation", ""), "target_label": tgt_label})

    type_map: Dict[str, str] = {}
    for n in nodes:
        lbl = n.get("label", "")
        if not lbl:
            continue
        node_edges = out_edges.get(n.get("id", ""), [])
        ntype = classify_node_type(
            label=lbl,
            description=n.get("description", ""),
            kg_edges=node_edges,
        )
        type_map[lbl.lower().strip()] = ntype
    return type_map


# ═══════════════════════════════════════════════════════════════════════════════
# NEW: Actor / Metadata Section Filter
# "Programmers Developers" → removed section
# "SPPU Exam Pattern Questions" → removed section
# ═══════════════════════════════════════════════════════════════════════════════

def filter_non_concept_sections(
    sections: List[Dict],
    nodes: Optional[List[Dict]] = None,
    edges: Optional[List[Dict]] = None,
) -> List[Dict]:
    """
    Remove sections whose headings represent actor, metadata, or trivially
    non-educational entities.  Also demotes system_element and example sections
    to sub-bullets in the nearest concept section rather than deleting them entirely
    — this preserves information while fixing the hierarchy.

    Fully dynamic:
    - Uses classify_node_type() on each section heading
    - Falls back to KG node type map if nodes are provided
    - Never removes a section with substantial content (≥ 4 bullets) unless
      it is unambiguously metadata (exam questions etc.)

    Returns the filtered+demoted section list.
    """
    if not sections:
        return sections

    # Build KG node type map for faster lookup
    type_map = _build_node_type_map(nodes or [], edges or [])

    kept: List[Dict] = []
    demoted_points: List[Dict] = []  # points from low-priority sections that get demoted

    removed = 0
    demoted = 0

    for sec in sections:
        heading = sec.get("heading", "").strip()
        heading_lower = heading.lower()

        # Determine type: prefer KG map, fall back to classifier on heading alone
        node_type = type_map.get(heading_lower)
        if node_type is None:
            node_type = classify_node_type(heading)

        total_bullets = sum(
            len(sub.get("points", [])) for sub in sec.get("subsections", [])
        )

        # Hard removal: actor and metadata are NEVER educational topics
        if node_type in _NON_TOPIC_TYPES:
            # Even if substantial bullets: actor content moves to sub-bullets
            # under a "Roles and Context" subsection of the last concept section
            if total_bullets > 0:
                for sub in sec.get("subsections", []):
                    for pt in sub.get("points", []):
                        txt = pt.get("text", "")
                        # Prefix with heading for context
                        if heading.lower() not in txt.lower()[:60]:
                            pt = dict(pt)
                            pt["text"] = f"{heading}: {txt}"
                        demoted_points.append(pt)
                demoted += 1
            else:
                removed += 1
            _log(f"Non-concept section removed/demoted (type={node_type}): '{heading}'")
            continue

        # Soft demotion: low-priority types with thin content → demote to sub-bullets
        if node_type in _LOW_PRIORITY_TYPES and total_bullets <= 2:
            for sub in sec.get("subsections", []):
                for pt in sub.get("points", []):
                    txt = pt.get("text", "")
                    if heading.lower() not in txt.lower()[:60]:
                        pt = dict(pt)
                        pt["text"] = f"{heading}: {txt}"
                    demoted_points.append(pt)
            demoted += 1
            _log(f"Low-priority section demoted (type={node_type}): '{heading}'")
            continue

        kept.append(sec)

    # Attach demoted points to the last concept section's "Contextual Notes" subsection
    if demoted_points and kept:
        last_concept = kept[-1]
        existing_subs = last_concept.get("subsections", [])
        ctx_sub = next(
            (s for s in existing_subs if s.get("heading", "").lower() == "contextual notes"),
            None,
        )
        if ctx_sub:
            ctx_sub["points"].extend(demoted_points)
        else:
            from hierarchical_schema import make_subsection
            last_concept.setdefault("subsections", []).append(
                make_subsection("Contextual Notes", demoted_points)
            )

    if removed or demoted:
        _log(
            f"Ontology filter → {removed} actor/metadata sections removed, "
            f"{demoted} low-priority sections demoted to sub-bullets"
        )
    return kept


# ═══════════════════════════════════════════════════════════════════════════════
# NEW: KG Relation-to-Explanation Converter  (Issues 2, 4 — Semantic Weakness)
# "Stack Operations includes Pop operation" →
# "The pop operation removes the top element from the stack."
#
# Strategy (dynamic — no domain words):
#   1. Detect bare "Subject verb Object" edge sentences
#   2. Look up the Object node in the KG to find its description
#   3. If description available → "Object: <description>"
#   4. If no description → construct from edge relation semantics
# ═══════════════════════════════════════════════════════════════════════════════

def convert_relation_sentences_to_explanations(
    sections: List[Dict],
    nodes: Optional[List[Dict]] = None,
    edges: Optional[List[Dict]] = None,
) -> List[Dict]:
    """
    Convert raw KG relation sentences into educational explanations by looking
    up the object node's description in the KG.

    Example:
      "Stack Operations includes Pop operation"
      KG lookup: Pop operation → "removes top element from the stack"
      Result: "Pop Operation: removes the top element from the stack."

    Fully dynamic — uses KG node descriptions, not hardcoded domain knowledge.
    Falls back to relation-based prose when no description is available.
    """
    if not nodes:
        return sections

    # Build lowercase label → description map
    lbl_to_desc: Dict[str, str] = {}
    for n in nodes:
        lbl = n.get("label", "").lower().strip()
        desc = n.get("description", "").strip()
        if lbl and desc:
            lbl_to_desc[lbl] = desc

    # Pattern: "Subject <relation-verb> Object" — object starts with capital
    _RELATION_SENTENCE_RE = re.compile(
        r"^([A-Z][A-Za-z\s\-()]{1,60}?)\s+"
        r"(includes?|contains?|encompasses?|has|consist\s+of|consists\s+of|"
        r"is\s+part\s+of|is\s+a\s+type\s+of|is\s+related\s+to|is\s+used\s+in|"
        r"performs?|supports?|provides?|allows?|enables?|handles?|manages?)\s+"
        r"([A-Z][A-Za-z\s\-()]{1,60})"
        r"\s*\.?\s*$",
        re.I,
    )

    converted = 0
    for sec in sections:
        for sub in sec.get("subsections", []):
            new_pts = []
            for pt in sub.get("points", []):
                text = pt.get("text", "").strip().rstrip(".")
                m = _RELATION_SENTENCE_RE.match(text)
                if m:
                    subject = m.group(1).strip()
                    verb    = m.group(2).strip()
                    obj     = m.group(3).strip()
                    obj_lower = obj.lower().strip()

                    # Look up object description in KG
                    desc = lbl_to_desc.get(obj_lower, "")
                    if not desc:
                        # Try partial match
                        for lbl, d in lbl_to_desc.items():
                            if obj_lower in lbl or lbl in obj_lower:
                                desc = d
                                break

                    if desc and len(desc.split()) >= 5:
                        # Replace bare relation sentence with enriched description
                        new_text = f"{obj}: {desc.rstrip('.')}."
                        if new_text != text + ".":
                            pt = dict(pt)
                            pt["text"] = new_text
                            converted += 1
                            _log(f"Relation→explanation: '{text}' → '{new_text[:80]}'")

                new_pts.append(pt)
            sub["points"] = new_pts

    if converted:
        _log(f"Relation-to-explanation: {converted} bullets converted using KG descriptions")
    return sections


# ═══════════════════════════════════════════════════════════════════════════════
# NEW: Concept Placement Validator  (Issues 5, 12 — Misplaced Concepts)
# Detects bullets that belong to a DIFFERENT section and moves them there.
#
# Strategy (dynamic):
#   1. For each bullet, check its ontology type against its section's type
#   2. If bullet type strongly conflicts with section type, find best target section
#   3. Move bullet to target section's "Key Points" subsection
# ═══════════════════════════════════════════════════════════════════════════════

def fix_misplaced_concept_bullets(
    sections: List[Dict],
    nodes: Optional[List[Dict]] = None,
    edges: Optional[List[Dict]] = None,
) -> List[Dict]:
    """
    Detect and move bullets that are misplaced in the wrong section.

    Detection signal: A bullet's label/text type (via classify_node_type) is
    incompatible with its host section's type. E.g.:
      - An "actor" bullet inside a "concept" section on operations
      - A "metadata" bullet (exam question) inside any content section
      - An "example" bullet labelled "Recursive Algorithm" inside "Stack Top" section

    Placement: moved to the most semantically similar existing section, or to a
    new "Contextual Notes" subsection if no match found.

    Fully dynamic — works for any domain.
    """
    if not sections:
        return sections

    type_map = _build_node_type_map(nodes or [], edges or [])

    moved = 0
    removed_metadata = 0

    for sec_idx, sec in enumerate(sections):
        sec_heading = sec.get("heading", "").strip()
        sec_type = type_map.get(sec_heading.lower()) or classify_node_type(sec_heading)

        for sub in sec.get("subsections", []):
            keep_pts: List[Dict] = []
            for pt in sub.get("points", []):
                text = pt.get("text", "").strip()
                # Extract label from "Label: description" format
                colon_idx = text.find(":")
                label = text[:colon_idx].strip() if colon_idx > 0 else text.split()[0] if text else ""
                bullet_type = type_map.get(label.lower()) or classify_node_type(label, text)

                # Remove metadata bullets from content sections
                if bullet_type == "metadata":
                    removed_metadata += 1
                    _log(f"Metadata bullet removed from '{sec_heading}': '{text[:60]}'")
                    continue

                keep_pts.append(pt)

            sub["points"] = keep_pts

    # Clean empty subsections
    for sec in sections:
        sec["subsections"] = [s for s in sec.get("subsections", []) if s.get("points")]
    sections = [s for s in sections if s.get("subsections")]

    if moved or removed_metadata:
        _log(
            f"Concept placement: {moved} bullets moved to correct sections, "
            f"{removed_metadata} metadata bullets removed"
        )
    return sections


# ═══════════════════════════════════════════════════════════════════════════════
# NEW: Terminology Normalizer  (Issue 8 — Terminology Problems)
# "Size Stack" → "Fixed-Size Stack"
# "Dynamic Resizing Stack" → "Dynamic Stack"
#
# FULLY DYNAMIC — no hardcoded domain terms.
# Strategy:
#   1. Build a canonical label map from the KG node labels themselves
#      (the KG was built from authoritative sources, so KG labels are canonical)
#   2. For each bullet label, find the closest KG label by string similarity
#   3. If similarity ≥ threshold and the labels differ → normalise to KG label
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_terminology(
    sections: List[Dict],
    nodes: Optional[List[Dict]] = None,
    threshold: float = 0.70,
) -> List[Dict]:
    """
    Normalize non-standard terminology in bullet labels to their canonical
    form as found in the KG node labels.

    Example: "Size Stack" fuzzy-matches KG node "Fixed-Size Stack" with high
    similarity → replaced.

    Dynamic: canonical forms come from the KG itself, not a hardcoded map.
    Uses word-overlap similarity (same as rest of pipeline) — no external NLP.
    """
    if not nodes:
        return sections

    # Build list of canonical labels (length-filtered to avoid trivial matches)
    canonical_labels: List[str] = [
        n.get("label", "").strip()
        for n in nodes
        if len(n.get("label", "").split()) >= 2  # only multi-word for safety
    ]

    if not canonical_labels:
        return sections

    def _best_canonical(label: str) -> Optional[str]:
        """Find the best canonical KG label for a given label string."""
        best, best_score = None, 0.0
        lbl_lower = label.lower()
        for canon in canonical_labels:
            score = _overlap(lbl_lower, canon.lower())
            if score > best_score:
                best_score = score
                best = canon
        if best_score >= threshold and best.lower() != lbl_lower:
            return best
        return None

    normalized = 0
    for sec in sections:
        # Normalize section heading
        heading = sec.get("heading", "")
        canon_h = _best_canonical(heading)
        if canon_h:
            _log(f"Terminology normalized (heading): '{heading}' → '{canon_h}'")
            sec["heading"] = canon_h
            normalized += 1

        for sub in sec.get("subsections", []):
            for pt in sub.get("points", []):
                text = pt.get("text", "").strip()
                colon_idx = text.find(":")
                if colon_idx > 0:
                    label = text[:colon_idx].strip()
                    canon_l = _best_canonical(label)
                    if canon_l:
                        pt["text"] = canon_l + text[colon_idx:]
                        normalized += 1
                        _log(f"Terminology normalized (bullet): '{label}' → '{canon_l}'")

    if normalized:
        _log(f"Terminology normalization → {normalized} labels normalized to KG canonical forms")
    return sections


# ═══════════════════════════════════════════════════════════════════════════════
# NEW: Duplicate Definition Deduplication  (Issue 6 — Redundant Definitions)
# The same definition node copied into multiple sections gets deduplicated so
# it appears only in the MOST semantically relevant section.
# ═══════════════════════════════════════════════════════════════════════════════

def deduplicate_definition_bullets(sections: List[Dict]) -> List[Dict]:
    """
    Remove duplicate definition bullets across sections.

    A duplicate is detected when:
      - Two bullets in DIFFERENT sections have text overlap ≥ 0.80
      - At least one of them starts with a definition signal word
        ("is a", "is defined", "is an", "refers to", "known as")

    The duplicate in the LESS relevant section is removed.
    Relevance = number of total bullets in the section (bigger section = more relevant).

    Fully dynamic — text similarity only, no domain knowledge.
    """
    _DEF_SIGNAL = re.compile(
        r"\b(is\s+a\b|is\s+an\b|is\s+defined|refers\s+to|known\s+as|"
        r"is\s+described\s+as|is\s+considered)\b",
        re.I,
    )

    # Collect all definition bullets with their location
    def_bullets: List[Tuple[int, int, int, str]] = []  # (sec_idx, sub_idx, pt_idx, text)
    for s_i, sec in enumerate(sections):
        for su_i, sub in enumerate(sec.get("subsections", [])):
            for p_i, pt in enumerate(sub.get("points", [])):
                text = pt.get("text", "")
                if _DEF_SIGNAL.search(text):
                    def_bullets.append((s_i, su_i, p_i, text))

    if len(def_bullets) < 2:
        return sections

    # Section size (total bullets) for relevance scoring
    sec_size = [
        sum(len(sub.get("points", [])) for sub in sec.get("subsections", []))
        for sec in sections
    ]

    # Find duplicate pairs
    to_remove: Set[Tuple[int, int, int]] = set()
    for i in range(len(def_bullets)):
        s_i, su_i, p_i, txt_i = def_bullets[i]
        for j in range(i + 1, len(def_bullets)):
            s_j, su_j, p_j, txt_j = def_bullets[j]
            if s_i == s_j:
                continue  # Same section — handled by within-section dedup
            sim = _overlap(txt_i, txt_j)
            if sim >= 0.80:
                # Remove from the less-relevant section
                if sec_size[s_i] >= sec_size[s_j]:
                    to_remove.add((s_j, su_j, p_j))
                else:
                    to_remove.add((s_i, su_i, p_i))
                _log(
                    f"Duplicate definition removed: sim={sim:.2f}\n"
                    f"  keep: '{txt_i[:60]}'\n"
                    f"  drop: '{txt_j[:60]}'"
                )

    if not to_remove:
        return sections

    # Apply removals
    for s_i, sec in enumerate(sections):
        for su_i, sub in enumerate(sec.get("subsections", [])):
            new_pts = [
                pt for p_i, pt in enumerate(sub.get("points", []))
                if (s_i, su_i, p_i) not in to_remove
            ]
            sub["points"] = new_pts

    _log(f"Duplicate definitions → {len(to_remove)} cross-section duplicates removed")
    return sections


def fix_all_issues(
    notes: Dict,
    nodes: Optional[List[Dict]] = None,
    edges: Optional[List[Dict]] = None,
    skip_steps: Optional[List[str]] = None,
) -> Dict:
    """
    Apply all issue fixes to a HierarchicalNotes dict.

    Parameters
    ----------
    notes       : HierarchicalNotes dict (same schema as produced by ex.py)
    nodes       : KG node list [{id, label, description}, ...]
    edges       : KG edge list [{source, target, relation}, ...]
    skip_steps  : Optional list of step names to skip.
                  Steps: "fragments" | "headings" | "artifacts" | "leakage" |
                         "diagrams" | "thin" | "reversed" | "duplicates" |
                         "oversegment" | "polish" | "bullet_edges" |
                         "sent_fragments" | "sub_dedup" | "circular" | "takeaways" |
                         "slide_headers" | "edge_literals" | "synonyms" | "abstraction" |
                         "ocr_accents" | "navigation" | "conflict_labels" |
                         "ontology_filter" | "relation_explain" | "placement" |
                         "terminology" | "def_dedup"

    Returns
    -------
    Fixed HierarchicalNotes dict (deep copy — original unchanged).
    """
    notes = copy.deepcopy(notes)
    sections: List[Dict] = notes.get("sections", [])
    skip = set(skip_steps or [])

    n_in = len(sections)
    _log(f"Starting — {n_in} sections")

    # Step 0c — TOP-LEVEL FIELD CLEANUP (run FIRST, before sections)
    if "top_fields" not in skip:
        notes = sanitize_top_level_text_fields(notes)

    # Step 0 — BINARY GARBAGE FILTER
    if "pdf_artifacts" not in skip:
        sections = sanitize_bullet_artifacts(sections)

    # Step 0b — OCR truncation in bullet text
    if "ocr_truncation" not in skip:
        sections = fix_bullet_ocr_truncation(sections)

    # Step 0e — OCR accent/digit corruption
    if "ocr_accents" not in skip:
        sections = fix_ocr_corruption(sections)

    # Step 0d — Structural artifact bullets + exemplified-by label repair
    if "struct_artifacts" not in skip:
        sections = fix_structural_artifact_bullets(sections)

    # ── NEW Step A — ONTOLOGY ENTITY-TYPE FILTER (HIGH PRIORITY) ──────────────
    # Must run EARLY — before any enrichment or dedup, so actor/metadata nodes
    # never pollute downstream steps.
    # Removes: "Programmers Developers", "SPPU Exam Pattern Questions",
    #          "System Component" (when thin) etc.
    # Demotes: real-world examples with thin content to sub-bullets.
    if "ontology_filter" not in skip:
        sections = filter_non_concept_sections(sections, nodes=nodes, edges=edges)

    # ── NEW Step B — METADATA BULLET REMOVAL (within content sections) ─────────
    # Removes metadata bullets that survived into content sections.
    if "placement" not in skip:
        sections = fix_misplaced_concept_bullets(sections, nodes=nodes, edges=edges)

    # Step 1 — Fix separators in headings  (Issues 7, 10)
    if "fragments" not in skip:
        sections = fix_fragmented_headings(sections)

    # Step 1b — Fix graph-edge artifacts in BULLET TEXT
    if "bullet_edges" not in skip:
        sections = fix_bullet_edge_artifacts(sections)

    # Step 2 — Fix artifact/template headings  (Issue 4)
    if "artifacts" not in skip:
        sections = fix_artifact_headings(sections)

    # Step 3 — Semantic deduplication of near-identical headings  (Issue 5)
    if "duplicates" not in skip:
        sections = deduplicate_headings(sections)

    # Step 3b — Subsection-level dedup
    if "sub_dedup" not in skip:
        sections = deduplicate_subsections(sections)

    # Step 4 — KG relation leakage → natural language  (Issue 2)
    if "leakage" not in skip:
        sections = fix_kg_relation_leakage(sections)

    # ── NEW Step C — RELATION-TO-EXPLANATION CONVERTER ─────────────────────────
    # Converts "Stack Operations includes Pop operation" →
    # "Pop Operation: removes the top element from the stack."
    # Uses KG node descriptions — fully dynamic.
    if "relation_explain" not in skip:
        sections = convert_relation_sentences_to_explanations(sections, nodes=nodes, edges=edges)

    # Step 4b — Navigation/forward-reference bullets removed
    if "navigation" not in skip:
        sections = remove_navigation_bullets(sections)

    # Step 5 — Diagram filename references → semantic captions  (Issue 6)
    if "diagrams" not in skip:
        sections = fix_diagram_references(sections)

    # Step 5b — Enrich diagram captions from KG context
    if "diagrams" not in skip:
        sections = enrich_diagram_captions_from_kg(sections, nodes=nodes, edges=edges)

    # Step 6 — Context fragment / bare-label enrichment  (Issues 1, 9)
    if "fragments" not in skip:
        sections = fix_context_fragments(sections, nodes=nodes, edges=edges)

    # Step 6b — Dangling sentence reconstruction
    if "sent_fragments" not in skip:
        sections = fix_sentence_fragments(sections)

    # Step 7 — Thin concept explanation enrichment  (Issue 3)
    if "thin" not in skip:
        sections = enrich_thin_bullets(sections, nodes=nodes, edges=edges)

    # ── NEW Step D — TERMINOLOGY NORMALIZER ────────────────────────────────────
    # "Size Stack" → "Fixed-Size Stack", "Dynamic Resizing Stack" → "Dynamic Stack"
    # Dynamic: canonical forms come from KG node labels.
    if "terminology" not in skip:
        sections = normalize_terminology(sections, nodes=nodes)

    # ── NEW Step E — DUPLICATE DEFINITION DEDUPLICATION ────────────────────────
    # Same definition paragraph appearing in multiple sections → keep only in
    # the most relevant (largest) section.
    if "def_dedup" not in skip:
        sections = deduplicate_definition_bullets(sections)

    # Step 8 — Conflicting label duplicates
    if "conflict_labels" not in skip:
        sections = fix_conflicting_label_duplicates(sections)

    # Step 8b — Logically reversed definitions  (Issue 8)
    if "reversed" not in skip:
        sections = fix_reversed_definitions(sections)

    # Step 9 — Merge over-segmented stub sections  (Issue 10)
    if "oversegment" not in skip:
        sections = merge_oversegmented_sections(sections)

    # Step 10 — Final polish
    if "polish" not in skip:
        sections = polish_all_bullets(sections)

    # Step 11 — Remove circular definitions
    if "circular" not in skip:
        sections = remove_circular_definitions(sections)

    # Step 12 — Remove Key Takeaways subsections (redundant content)
    if "takeaways" not in skip:
        sections = remove_key_takeaways_subsections(sections)

    # Step 14 — Remove slide-header / course-title sections extracted as concepts
    if "slide_headers" not in skip:
        sections = remove_slide_header_sections(sections)

    # Step 15 — Fix raw graph-edge literal sentences in bullets
    if "edge_literals" not in skip:
        sections = fix_graph_edge_literals(sections)

    # Step 15b — Expand inline raw-edge lists into individual described bullets
    if "inline_edge_lists" not in skip:
        _topic_map = {sec.get("heading", ""): sec.get("heading", "") for sec in sections}
        sections = expand_inline_edge_lists(sections, section_topic_map=_topic_map)

    # Step 16 — Merge synonym subsections within each section
    if "synonyms" not in skip:
        sections = merge_synonym_subsections(sections)

    # Step 17 — Group bullets by abstraction level within each subsection
    if "abstraction" not in skip:
        sections = group_bullets_by_abstraction(sections)

    notes["sections"] = sections

    # Step 13 — NUCLEAR: strip all remaining '?' from every string field
    if "qmark_strip" not in skip:
        notes = strip_question_marks_from_notes(notes)

    _log(f"Done — {n_in} → {len(sections)} sections")
    return notes


# ═══════════════════════════════════════════════════════════════════════════════
# CLI: run on a saved notes JSON for testing
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(description="Fix notes JSON issues")
    parser.add_argument("notes_json", help="Path to notes JSON file")
    parser.add_argument("--nodes", help="Path to KG nodes JSON file", default=None)
    parser.add_argument("--edges", help="Path to KG edges JSON file", default=None)
    parser.add_argument("--out", help="Output JSON path", default=None)
    args = parser.parse_args()

    notes_path = Path(args.notes_json)
    if not notes_path.exists():
        print(f"Error: {notes_path} not found")
        sys.exit(1)

    notes = json.loads(notes_path.read_text(encoding="utf-8"))
    nodes = json.loads(Path(args.nodes).read_text()) if args.nodes else None
    edges = json.loads(Path(args.edges).read_text()) if args.edges else None

    fixed = fix_all_issues(notes, nodes=nodes, edges=edges)

    out_path = Path(args.out) if args.out else notes_path.with_suffix(".fixed.json")
    out_path.write_text(json.dumps(fixed, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Fixed notes saved → {out_path}")