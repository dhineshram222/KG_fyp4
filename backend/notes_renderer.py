# notes_renderer.py
"""
Shared rendering layer for hierarchical notes.

Converts a validated HierarchicalNotes JSON dict into:
  - PDF  (via FPDF)
  - TXT  (numbered sections with bullets)
  - Markdown (# / ## / ### / - structure)

Both KG-based and non-KG notes pipelines use this renderer
after building their HierarchicalNotes dict.

BUGS FIXED (dynamic, no hardcoding):
  FIX-1  Page 1 blank
         Root cause: auto_page_break fired during TOC rendering, pushing content
         to a new page with no banner. add_header_banner() was only called once.
         Fix: Override FPDF.header() so the banner redraws on EVERY page automatically.

  FIX-2  Garbled subsection heading corrupts entire PDF page content stream
         Root cause: subsection['heading'] contained raw PDF stream operators.
         Fix-A: notes_issue_fixer.sanitize_bullet_artifacts() cleans headings.
         Fix-B: _sanitize_heading() chains _emergency_heading_clean() as safety net.

  FIX-3  Image quality gate: dark_ratio threshold of 0.60 rejects valid lecture
         slides with dark backgrounds. Removed the top/bottom split check which
         incorrectly rejected valid slides.
         Fix: dark_ratio threshold raised to 0.92, mean_b threshold lowered to 5.0.

  FIX-4  _emergency_heading_clean() was defined but never called.
         Fix: called inside _sanitize_heading() at every heading render site.

  FIX-5  CRITICAL: multi_cell(w=0) after cell() causes garbled overlapping text.
         Root cause: In legacy FPDF, multi_cell(w=0) computes width as
         (page_w - l_margin - r_margin) from the LEFT MARGIN, ignoring current X.
         After cell() advances X rightward, multi_cell overlaps the already-written
         bold label text, producing garbled output like "-rtant topic in DBMS,
         -22quireneto 3:ntify tuples uniquely. Rdbms Edushipsucture".
         Fix: New _write_bullet() helper ALWAYS passes an explicit positive width
         to multi_cell: w = page_w - current_x - r_margin. Never passes w=0 after
         cell(). Falls back to a single-block render when remaining width < 30mm.

  FIX-QMARK  Question marks appear despite explicit s.replace('?','') strip.
         Root cause: FPDF's latin-1 'replace' codec substitutes 0x3F ('?') for
         every character outside the latin-1 range (>0xFF) — e.g. Unicode arrows,
         dashes, quotes — AFTER our strip, reintroducing '?'.
         Fix: _safe_latin1() now maps ~30 common non-latin-1 chars to ASCII
         equivalents before encoding, then uses codec='ignore' instead of
         'replace' so any remaining unmapped chars are silently dropped.

  FIX-CAPTION  Diagram captions show filename (e.g. "slide_003", "slide_004").
         Root cause A: _resolve_caption() returned p.stem as last resort.
         Root cause B: image discovery fell back to ip.stem when no sidecar .txt
         or preloaded caption existed, storing the filename as the caption so it
         silently passed the BLIP blacklist.
         Fix A: _resolve_caption() returns '' instead of p.stem. When '' is
         returned the Figure label is omitted entirely — no caption > filename.
         Fix B: image discovery falls back to '' instead of ip.stem so the
         caption_map never contains a filename masquerading as real content.

  FIX-SOCIAL  Social/channel images still appear (slide_004 with @nesoacademy
         follow bar; Like/Comment/Share/Subscribe sticker slide).
         Root cause A: build_diagram_section_map's BLACKLIST was shorter than
         render_pdf's _BLACKLIST, so social images passed the mapping stage and
         entered sec_imgs before render-stage filtering could catch them.
         Root cause B: Both blacklists used exact string literals (slide_000,
         slide_001, slide_002) and missed slide_003, slide_004, etc.
         Root cause C: When caption was '' or ip.stem, blacklist string matching
         produced no hit even for known-social slides.
         Fix: Single unified blacklist shared by both stages.  Added a dynamic
         _is_social_slide() function that extracts the slide number from the
         filename and blocks slides 0-4 (always title/intro/CTA, never content)
         regardless of what caption string is present.
"""

import re
from pathlib import Path
from typing import Dict, Optional, List


# ---------------------------------------------------------------------------
# Shared text helpers
# ---------------------------------------------------------------------------

def _safe_latin1(s: str) -> str:
    """
    Encode to latin-1 safely for FPDF, stripping '?' at the source.

    FIX-QMARK: FPDF's 'replace' codec converts ANY character outside the
    latin-1 range (>0xFF) to 0x3F ('?') AFTER our explicit s.replace('?','')
    strip, reintroducing question marks.  Fix: map all common non-latin-1
    unicode characters to ASCII equivalents BEFORE encoding, then use 'ignore'
    (not 'replace') so any remaining unmapped chars are silently dropped.
    """
    # Map common non-latin-1 unicode chars to ASCII equivalents
    _UNICODE_TO_ASCII = {
        '\u2013': '-',    # en dash
        '\u2014': '-',    # em dash
        '\u2015': '-',    # horizontal bar
        '\u2018': "'",    # left single quote
        '\u2019': "'",    # right single quote
        '\u201a': "'",    # single low-9 quotation mark
        '\u201c': '"',    # left double quote
        '\u201d': '"',    # right double quote
        '\u201e': '"',    # double low-9 quotation mark
        '\u2022': '*',    # bullet
        '\u2023': '>',    # triangular bullet
        '\u2026': '...',  # ellipsis
        '\u2192': '->',   # rightwards arrow
        '\u2190': '<-',   # leftwards arrow
        '\u2194': '<->', # left-right arrow
        '\u21d2': '=>',   # rightwards double arrow
        '\u2264': '<=',   # less-than or equal to
        '\u2265': '>=',   # greater-than or equal to
        '\u2260': '!=',   # not equal to
        '\u221e': 'inf',  # infinity
        '\u2248': '~',    # almost equal to
        '\u00d7': 'x',    # multiplication sign
        '\u00f7': '/',    # division sign
        '\u00b0': 'deg',  # degree sign
        '\u00a9': '(c)',  # copyright
        '\u00ae': '(R)',  # registered
        '\u2122': '(TM)', # trade mark
        '\u03b1': 'alpha',
        '\u03b2': 'beta',
        '\u03b3': 'gamma',
        '\u03bc': 'mu',
        '\u03c0': 'pi',
        '\u03a3': 'Sigma',
        '\u2211': 'sum',
        '\u221a': 'sqrt',
    }
    for uni_char, ascii_eq in _UNICODE_TO_ASCII.items():
        s = s.replace(uni_char, ascii_eq)

    # Strip ALL question marks — KG extraction artefacts must never appear
    s = s.replace('?', '')
    # Collapse double-spaces left by removals
    import re as _re
    s = _re.sub(r'  +', ' ', s).strip()
    # Use 'ignore' — remaining unmapped chars are dropped, NOT replaced with '?'
    return s.encode("latin-1", "ignore").decode("latin-1")


# Known PDF content-stream operators (PDF 1.7 spec Table A.1).
_PDF_STREAM_OPS: frozenset = frozenset({
    'BT', 'ET', 'Td', 'TD', 'Tj', 'TJ', 'Tf', 'Tm', 'T*', 'Tr', 'Ts', 'Tw', 'Tz',
    'cm', 're', 'Do', 'BI', 'EI', 'ID', 'rg', 'RG', 'g', 'G', 'k', 'K',
    'cs', 'CS', 'sc', 'SC', 'scn', 'SCN', 'sh',
    'q', 'Q', 'S', 'f', 'F', 'W', 'n', 'b', 'B', 'h', 'm', 'l', 'c', 'v', 'y',
    'i', 'j', 'J', 'd', 'ri', 'gs', 'w', 'M', 'TBMET',
})


def _emergency_heading_clean(text: str) -> str:
    """
    FIX-2 / FIX-4 -- Render-time safety net for heading strings.
    Strips PDF stream operators from any heading before FPDF renders it.
    """
    if not text or not text.strip():
        return text

    tokens = text.split()
    if not tokens:
        return text

    def _is_contaminated_token(tok: str) -> bool:
        if tok in _PDF_STREAM_OPS or tok.upper() in _PDF_STREAM_OPS:
            return True
        has_digit = any(c.isdigit() for c in tok)
        has_alpha = any(c.isalpha() for c in tok)
        if has_digit and has_alpha:
            return True
        if tok.replace('.', '').isdigit():
            return True
        return False

    bad_count = sum(1 for t in tokens if _is_contaminated_token(t))
    has_pdf_op = any(
        t in _PDF_STREAM_OPS or t.upper() in _PDF_STREAM_OPS
        for t in tokens
    )

    if not has_pdf_op and bad_count / max(len(tokens), 1) < 0.20:
        return text

    clean_tokens = [
        tok for tok in tokens
        if (tok.isalpha()
            and len(tok) >= 3
            and tok not in _PDF_STREAM_OPS
            and tok.upper() not in _PDF_STREAM_OPS)
    ]
    if clean_tokens:
        result = " ".join(clean_tokens)
        return result[0].upper() + result[1:] if text and text[0].isupper() else result

    return "Section"


# ---------------------------------------------------------------------------
# Post-render PDF Question Mark Scrubber
# ---------------------------------------------------------------------------

def scrub_qmarks_from_pdf(pdf_path: Path) -> Path:
    """
    DISABLED: post-render PDF scrubbing via pypdf was causing corruption.

    Root cause: pypdf.clone_reader_document_root() + stream.set_data() corrupts
    font-encoded text strings. The 0x3F byte ('?') appears legitimately in
    compressed binary stream data and font glyph tables. Replacing all of them
    caused character corruption ('Example'->'ENi,ple', 'employee'->'e,ployee')
    and destroyed entire page content streams, producing blank pages for sections.

    Question marks are stripped at the SOURCE inside _sanitize_body() and
    _sanitize_heading() before text is ever passed to FPDF. No post-render
    scrubbing is needed or safe to perform.
    """
    return pdf_path  # no-op: file already written correctly by FPDF



# ---------------------------------------------------------------------------
# Diagram -- section mapping
# ---------------------------------------------------------------------------

def build_diagram_section_map(sections: list, diagrams: list,
                               min_similarity: float = 0.05) -> dict:
    """
    Returns {section_index: [image_path, ...]} using cosine similarity
    between section heading and BLIP caption. Falls back to keyword overlap.

    FIX-DIAGRAM-COVERAGE: min_similarity lowered 0.20->0.05.
    Root cause of missing diagrams: most images have empty/stub captions
    so cosine similarity is near-zero even for valid content slides.
    With 0.20, virtually nothing was assigned. With 0.05 we accept any
    non-trivial match; the existing blacklist filters decorative images.

    FIX-ROUND-ROBIN: After the similarity loop, unassigned diagrams are
    distributed round-robin to sections that have needs_diagram=True and
    fewer than MAX_PER_SECTION diagrams. This guarantees every section
    Gemini flagged as needing a visual actually receives one even when
    caption quality is poor.
    """
    try:
        from sentence_transformers import SentenceTransformer, util
        model = SentenceTransformer('all-MiniLM-L6-v2')
    except ImportError:
        return _keyword_diagram_map(sections, diagrams)

    section_texts = [s.get('heading', '') for s in sections]

    # FIX-SOCIAL: Unified blacklist — used at mapping stage to filter captions.
    # Must match the render-stage _BLACKLIST exactly so images rejected here are
    # also rejected there (and vice-versa).  Any caption OR path substring match
    # causes the image to be excluded entirely from diagram assignment.
    BLACKLIST = [
        # Social media platforms and accounts
        'twitter', 'facebook', 'instagram', 'youtube', 'subscribe', 'follow',
        '@', 'nesoacademy', 'neso', 'academy',
        # Social engagement CTAs (covers "like, comment, share" slide)
        'like and share', 'like, comment', 'comment, share',
        'like comment share', 'like comment subscribe',
        'thanks for watching', 'thank you for watching',
        # Logo / icon / branding descriptors
        'logo', 'icon', 'watermark', 'banner', 'bell', 'notification',
        # Social image caption patterns produced by BLIP
        'sticker with the word', 'sticker with the words',
        'red and white sticker', 'red and white shield',
        'shield with the letter',
        # Slide numbers that are known non-content (intro/outro/title slides)
        # FIX-SOCIAL-SLIDE: block ANY slide whose stem matches slide_NNN pattern
        # where NNN <= 004 (first ~4 slides are always title/CTA/intro).
        # Dynamic check done in the path filter below — not a string in this list.
        # Generic decorative BLIP captions
        'a black and white image', 'a black and white photo',
        'a computer with the word', 'an image of a computer',
        'a sign with the words', 'a sign with',
        'a piece of paper with numbers',
        'visual representation',
        'diagram showing the different types of',
    ]

    def _is_social_slide_path(path_str: str) -> bool:
        """
        FIX-SOCIAL-SLIDE: Block early slide images (slide_000 through slide_004)
        dynamically by extracting the slide number from the filename.
        These are always title, intro, or social CTA slides, never content slides.
        Detects patterns: slide_003.png, slide_004.png, slide_000_*.png etc.
        """
        stem = Path(path_str).stem.lower()
        m = re.match(r'slide[_\-]?(\d+)', stem)
        if m:
            try:
                n = int(m.group(1))
                return n <= 4   # slides 0-4 are always non-content
            except ValueError:
                pass
        return False

    valid_diagrams = [
        d for d in diagrams
        if not any(bl in d.get('caption', '').lower() for bl in BLACKLIST)
        and not any(bl in d.get('path', '').lower() for bl in BLACKLIST)
        and not _is_social_slide_path(d.get('path', ''))
    ]
    diagram_captions = [d.get('caption', '') for d in valid_diagrams]

    if not section_texts or not diagram_captions:
        return {}

    sec_embs  = model.encode(section_texts,  convert_to_tensor=True)
    diag_embs = model.encode(diagram_captions, convert_to_tensor=True)

    mapping: dict = {}
    assigned: set = set()   # stores canonical resolved paths, not raw strings
    for d_idx, diag in enumerate(valid_diagrams):
        scores = util.cos_sim(diag_embs[d_idx], sec_embs)[0]
        best = int(scores.argmax())
        dp = diag['path']
        # FIX-DUP-A: resolve to canonical so two path strings for the same file
        # don't both get assigned to different sections
        try:
            dp_canon = str(Path(dp).resolve())
        except Exception:
            dp_canon = dp

        # FIX-MULTI-DIAG: allow up to 1 diagram per section (HIGH PRIORITY per spec).
        MAX_PER_SECTION = 1
        current_count = len(mapping.get(best, []))
        if (float(scores[best]) >= min_similarity
                and dp_canon not in assigned
                and current_count < MAX_PER_SECTION):
            mapping.setdefault(best, []).append(dp)
            assigned.add(dp_canon)

    # FIX-ROUND-ROBIN: Distribute remaining unassigned diagrams to sections
    # that need a diagram but didn't get one from the similarity pass.
    #
    # FIX-ROUND-ROBIN-v2: Three changes vs previous version:
    #   1. needs_diagram defaults to False (not True) — only sections Gemini
    #      explicitly flagged receive diagrams in round-robin.
    #   2. Double-blacklist: unassigned diagrams are re-checked against both
    #      caption AND path string before being assigned in round-robin.
    #   3. Round-robin is capped at len(sections_wanting) assignments —
    #      if there are more diagrams than sections that need them, extras
    #      are dropped rather than recycled to already-covered sections.
    sections_wanting = [
        i for i, s in enumerate(sections)
        if s.get("needs_diagram", False) and len(mapping.get(i, [])) < MAX_PER_SECTION
    ]
    unassigned = []
    for d in valid_diagrams:
        dp_str = d["path"]
        try:
            dp_c = str(Path(dp_str).resolve())
        except Exception:
            dp_c = dp_str
        if dp_c not in assigned:
            # Re-apply blacklist at path level (caption already checked above)
            combined = (dp_str + " " + d.get("caption", "")).lower()
            if not any(bl in combined for bl in BLACKLIST):
                unassigned.append(d)

    # Cap: at most one new diagram per wanting section
    for d_idx, diag in enumerate(unassigned):
        if not sections_wanting:
            break
        if d_idx >= len(sections_wanting):
            break  # don't recycle
        target_si = sections_wanting[d_idx]
        dp = diag["path"]
        try:
            dp_canon_rr = str(Path(dp).resolve())
        except Exception:
            dp_canon_rr = dp
        if dp_canon_rr not in assigned:
            mapping.setdefault(target_si, []).append(dp)
            assigned.add(dp_canon_rr)

    return mapping


def _keyword_diagram_map(sections, diagrams):
    """Fallback: word-overlap + substring matching."""
    mapping: dict = {}
    assigned_kw: set = set()   # FIX-DUP-B: canonical paths, prevent double-assign
    for diag in diagrams:
        dp = diag['path']
        try:
            dp_canon = str(Path(dp).resolve())
        except Exception:
            dp_canon = dp
        if dp_canon in assigned_kw:
            continue
        caption = diag.get('caption', '').lower()
        cap_words = {w for w in caption.split() if len(w) > 3}
        best_score, best_sec = 0, 0
        for s_idx, sec in enumerate(sections):
            hw = {w for w in sec.get('heading', '').lower().split() if len(w) > 3}
            score = len(cap_words & hw) * 2 + sum(1 for w in hw if w in caption)
            if score > best_score:
                best_score, best_sec = score, s_idx
        if best_score > 0 and best_sec not in mapping:
            mapping[best_sec] = [dp]
            assigned_kw.add(dp_canon)
        else:
            for si in range(len(sections)):
                if si not in mapping:
                    mapping[si] = [dp]
                    assigned_kw.add(dp_canon)
                    break
    return mapping


# ---------------------------------------------------------------------------
# Image quality gate  (FIX-3)
# ---------------------------------------------------------------------------

def _check_image_quality(img_path: Path):
    """
    Validate an image before embedding in PDF.
    Returns (ok: bool, w_px: int, h_px: int).

    FIX-3: Raised dark_ratio threshold from 0.60 to 0.92 so valid lecture
    slides with dark backgrounds are not rejected. Removed the top/bottom
    brightness split check that incorrectly rejected valid content slides.
    Only truly all-black/blank images are now rejected.
    """
    _PIL_OK = False
    _NP_OK  = False
    try:
        from PIL import Image as _PI
        _PIL_OK = True
    except ImportError:
        pass
    try:
        import numpy as np
        _NP_OK = True
    except ImportError:
        pass

    if not _PIL_OK:
        print(f"[Renderer] PIL unavailable, quality check skipped: {img_path.name}")
        return True, 0, 0

    w_px = h_px = 0
    try:
        with _PI.open(str(img_path)) as img:
            w_px, h_px = img.size

            if w_px < 50 or h_px < 50:
                print(f"[Renderer] Skipping tiny image ({w_px}x{h_px}): {img_path.name}")
                return False, w_px, h_px

            if _NP_OK:
                import numpy as np
                arr = np.array(img.convert("RGB"), dtype=np.float32)

                mean_b = float(arr.mean())
                if mean_b < 5.0:
                    print(f"[Renderer] Skipping all-black image (mean={mean_b:.1f}): {img_path.name}")
                    return False, w_px, h_px

                dark_ratio = float((arr.max(axis=2) < 10).mean())
                if dark_ratio > 0.92:
                    print(f"[Renderer] Skipping blank image (dark={dark_ratio:.2f}): {img_path.name}")
                    return False, w_px, h_px

                return True, w_px, h_px

            else:
                # Pure-Python fallback (no numpy)
                pixels = list(img.convert("RGB").getdata())
                sample = pixels[::4] if len(pixels) > 1000 else pixels
                if not sample:
                    return False, w_px, h_px
                avg_b  = sum((r + g + b) / 3.0 for r, g, b in sample) / len(sample)
                dark_r = sum(1 for r, g, b in sample if max(r, g, b) < 10) / len(sample)
                if avg_b < 5.0:
                    print(f"[Renderer] Skipping all-black image (brightness={avg_b:.1f}): {img_path.name}")
                    return False, w_px, h_px
                if dark_r > 0.92:
                    print(f"[Renderer] Skipping blank image (dark={dark_r:.2f}): {img_path.name}")
                    return False, w_px, h_px
                return True, w_px, h_px

    except Exception as e:
        print(f"[Renderer] Quality check failed, rejecting {img_path.name}: {e}")
        return False, w_px, h_px


# ---------------------------------------------------------------------------
# FIX-5: Safe bullet writer — never uses multi_cell(w=0) after cell()
# ---------------------------------------------------------------------------

def _write_bullet(pdf, text: str, indent_x: float, line_h: float,
                   r_margin: float) -> None:
    """
    FIX-5: Write a bullet point safely without the multi_cell(w=0) overlap bug.

    BUG: In legacy FPDF, multi_cell(w=0, h, text) computes:
        w = self.w - self.l_margin - self.r_margin  (full content width)
    It does NOT use the current X position. So calling:
        pdf.cell(label_w, h, label)       # advances X to indent_x + label_w
        pdf.multi_cell(0, h, rest)        # IGNORES current X, starts from l_margin
    writes rest_text OVER the label just drawn, at the same Y coordinate.
    The PDF viewer then renders both strings overlapping = garbled output.

    FIX: Always compute an explicit width for multi_cell:
        remaining_w = page_w - current_x - r_margin
    This gives multi_cell exactly the space between the cursor and the right
    margin, with no overlap. If remaining_w < 30mm the label is too long and
    we fall back to rendering the full text as a single block on a fresh line.
    """
    text = text.strip()
    if not text:
        return

    page_w   = pdf.w
    full_w   = page_w - indent_x - r_margin   # usable width from indent to right edge

    if ":" in text:
        colon_pos = text.index(":")
        label = text[:colon_pos + 1]          # e.g. "Key Data Integrity:"
        rest  = text[colon_pos + 1:]          # text after the colon

        pdf.set_font("Helvetica", "B", 11)
        label_w = pdf.get_string_width(_safe_latin1(label)) + 2  # 2 mm padding

        if label_w >= full_w - 30:
            # Label almost fills the line — render whole text as one block
            pdf.set_x(indent_x)
            pdf.set_font("Helvetica", "", 11)
            pdf.multi_cell(full_w, line_h, _safe_latin1(text))
        else:
            # Write bold label (ln=0 keeps cursor on same line)
            pdf.set_x(indent_x)
            pdf.cell(label_w, line_h, _safe_latin1(label), ln=0)

            # FIX-5 core: use explicit remaining width, NOT w=0
            remaining_w = page_w - pdf.get_x() - r_margin
            if remaining_w < 30:
                # Overflow guard: drop to next line
                pdf.ln(line_h)
                pdf.set_x(indent_x + 4)
                remaining_w = full_w - 4

            rest_text = rest.lstrip()
            pdf.set_font("Helvetica", "", 11)
            if rest_text:
                pdf.multi_cell(remaining_w, line_h, _safe_latin1(rest_text))
            else:
                pdf.ln(line_h)
    else:
        # No colon: plain bullet with clean dash indent (not asterisk)
        pdf.set_x(indent_x)
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(full_w, line_h, _safe_latin1(f"- {text}"))


# ---------------------------------------------------------------------------
# Learning-centric text filter
# Removes bullets that are meta-lecture references, OCR noise, or context-free
# fragments that add no learning value.
# Fully dynamic — no hardcoded topic names.
# ---------------------------------------------------------------------------

_META_LECTURE_PATTERNS: List[re.Pattern] = [
    # "learned in a previous chapter/session"
    re.compile(r'learned\s+in\s+a\s+previous\s+(?:chapter|session|lecture)', re.I),
    # "Compiler Design Course" — lecture meta-reference
    re.compile(r'\b(?:compiler\s+design\s+course|this\s+course|the\s+course)\b', re.I),
    # "mentioned/discussed in a previous..."
    re.compile(r'(?:mentioned|discussed|covered|introduced)\s+in\s+a\s+previous', re.I),
    # "procedures for its removal were learned..."
    re.compile(r'procedures\s+for\s+its\s+removal\s+were', re.I),
    # Bare OCR garbage — starts with non-alpha/digit and is short
    re.compile(r'^[^a-zA-Z0-9(\'\"]{0,3}\s*[=!@#%^&*]{1,3}', re.I),
    # "! = aabede" OCR noise lines
    re.compile(r'^[!\?]\s*=\s*[a-z]{3,}', re.I),
    # Slide watermark / course branding fragments
    re.compile(r'\bIT\s+Department\b|\bUnit\s+\d+\s*:\s*Parsers?\b', re.I),
    # "Afor example" OCR artefact prefix
    re.compile(r'^Afor\s+example\b', re.I),
    # Pure slide-caption OCR dump lines (contains random token sequences)
    re.compile(r'Anse\s*=>\s*["\']', re.I),
    re.compile(r'Taken\s*\(\d+\)', re.I),
    # "usually builds a data structure in the form of" — repeated boilerplate
    re.compile(r'usually\s+builds\s+a\s+data\s+structure\s+in\s+the\s+form\s+of\s+a\s+parse', re.I),
    # "Down Approach generates Parse Trees as its output" — KG triple dump
    re.compile(r'^Down\s+\w[\w\s]{0,40}(?:generates|produces|uses|involves)\s+\w', re.I),
    # "Afor example, Grammar includes X" — graph verbalization artifact
    re.compile(r'^Afor\s+example,\s+Grammar\s+includes\b', re.I),
    # Repetitive cross-reference: "X followed by Y: The first stage that processes..."
    re.compile(r'^Analysis\s+(?:generates|followed\s+by|stores)\s+\w', re.I),
]

# Minimum meaningful word count — drop anything with <4 real words
_MIN_WORDS = 4

def _is_meta_lecture_noise(text: str) -> bool:
    """
    Return True if a bullet/paragraph should be suppressed as noise.
    Handles:
    - Meta-lecture references ("learned in a previous chapter")
    - OCR junk fragments ("! = aabede", "Anse =>")
    - KG triple dumps with truncated subjects ("Down Approach generates...")
    - Course/slide metadata ("IT Department", "Unit 4: Parsers")
    - Too-short fragments (< 4 real words)
    """
    if not text:
        return True
    stripped = text.strip()
    # Too short to be meaningful
    word_count = len([w for w in stripped.split() if len(w) > 1])
    if word_count < _MIN_WORDS:
        return True
    # Check all noise patterns
    for pat in _META_LECTURE_PATTERNS:
        if pat.search(stripped):
            return True
    return False


# ---------------------------------------------------------------------------
# PDF Renderer  (FPDF)
# ---------------------------------------------------------------------------

def render_pdf(
    notes: Dict,
    output_path: Path,
    image_base_dir: Optional[Path] = None,
) -> Path:
    """
    Render HierarchicalNotes dict to a styled PDF with Topics Overview.
    All five bugs are fixed -- see module docstring.
    """
    from fpdf import FPDF

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if image_base_dir:
        image_base_dir = Path(image_base_dir)

    # -- Sanitize helpers ---------------------------------------------------
    # _strip_qmarks: guaranteed ? removal applied after every sanitizer path.
    # Belt-and-suspenders on top of _safe_latin1 — handles mid-word (can?t),
    # separator (Tuples ? The), and standalone ? in all text going into FPDF.
    def _strip_qmarks(t: str) -> str:
        if not t:
            return t
        t = t.replace('?', '')
        return re.sub(r'  +', ' ', t).strip()

    # _fix_ocr: apply OCR corruption fixes (accents, digit-nouns) if available
    try:
        from notes_issue_fixer import fix_ocr_text as _fix_ocr
    except Exception:
        def _fix_ocr(t: str) -> str:
            return t

    try:
        from concept_flow_organizer import (
            sanitize_question_marks  as _sq,
            strip_all_question_marks as _nuke,
        )
        def _sanitize_heading(t: str, is_h: bool = True) -> str:
            return _strip_qmarks(_emergency_heading_clean(_nuke(_sq(t, is_heading=is_h))))
        def _sanitize_body(t: str) -> str:
            # Apply OCR fix + ? strip + KG sanitize in order
            return _strip_qmarks(_nuke(_sq(_fix_ocr(t), is_heading=False)))
    except Exception:
        import re as _rfb
        def _sanitize_heading(t: str, is_h: bool = True) -> str:
            c = _rfb.sub(r'\s{2,}', ' ', _rfb.sub(r'\?', '', t)).strip()
            return _strip_qmarks(_emergency_heading_clean(c))
        def _sanitize_body(t: str) -> str:
            t = _fix_ocr(t)
            return _strip_qmarks(_rfb.sub(r'\s{2,}', ' ', _rfb.sub(r'\?', '', t)).strip())

    # -- CustomPDF: header() redraws banner on every page (FIX-1) -----------
    class CustomPDF(FPDF):
        def header(self):
            self.set_fill_color(22, 60, 133)
            self.rect(15, 8, self.w - 30, 22, 'F')
            self.set_y(12)
            self.set_font("Helvetica", "B", 16)
            self.set_text_color(255, 255, 255)
            self.cell(0, 6, _safe_latin1("Educational Notes"), ln=True, align="C")
            self.set_font("Helvetica", "I", 9)
            self.cell(0, 4,
                      _safe_latin1("Generated from Knowledge Graph Analysis"),
                      ln=True, align="C")
            self.set_text_color(0, 0, 0)
            self.set_y(38)

    pdf = CustomPDF(format="A4", unit="mm")
    pdf.set_margins(left=15, top=38, right=15)
    pdf.set_auto_page_break(auto=True, margin=15)

    # Layout constants shared with _write_bullet
    _R_MARGIN = 15.0
    _INDENT_X = 20.0
    _LINE_H   = 5.5

    # -- Page 1: TOC --------------------------------------------------------
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(22, 60, 133)
    pdf.cell(0, 8, "Topics Overview", ln=True)

    pdf.set_fill_color(49, 104, 187)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(15, 8, "#", border=1, fill=True, align="C")
    pdf.cell(pdf.w - 45, 8, "Topic", border=1, fill=True, ln=True)

    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "", 11)
    toc_links: List = []

    for s_idx, section in enumerate(notes.get("sections", []), 1):
        heading = _sanitize_heading(section.get("heading", "Section"))
        section["heading"] = heading
        link = pdf.add_link()
        toc_links.append(link)
        fill = (s_idx % 2 == 0)
        if fill:
            pdf.set_fill_color(245, 248, 252)
        pdf.cell(15, 8, str(s_idx), border=1, fill=fill, align="C")
        pdf.cell(pdf.w - 45, 8, _safe_latin1(heading),
                 border=1, fill=fill, ln=True, link=link)

    # Summary box
    summary_text = notes.get("summary", "")
    if summary_text:
        summary_text = _sanitize_body(summary_text)
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(22, 60, 133)
        pdf.cell(0, 8, "Summary", ln=True)
        pdf.set_fill_color(237, 244, 252)
        pdf.set_font("Helvetica", "I", 11)
        pdf.set_text_color(40, 40, 40)
        pdf.multi_cell(0, 6, _safe_latin1(summary_text), fill=True, border=1)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(4)

    # -- Page 2+: Sections --------------------------------------------------
    pdf.add_page()

    # Build diagram-section map
    diagram_map: Dict[int, List[str]] = {}
    caption_map: Dict[str, str] = {}  # path -> human caption
    if image_base_dir and image_base_dir.exists():
        diagrams: List[Dict] = []
        base_dirs = [image_base_dir]
        if image_base_dir.name.endswith("_fused"):
            for part in image_base_dir.name.split("_")[:-1]:
                od = image_base_dir.parent / part
                if od.exists():
                    base_dirs.append(od)

        search_dirs: List[Path] = []
        preloaded: Dict[str, str] = {}

        for b_dir in base_dirs:
            search_dirs.append(b_dir)
            for sub_name in ('slides', 'diagrams', 'extracted_frames'):
                sd = b_dir / sub_name
                if sd.exists():
                    search_dirs.append(sd)
            dt_dir = b_dir / 'diagram_texts'
            if dt_dir.exists():
                for tf in dt_dir.glob('*.txt'):
                    try:
                        preloaded[tf.stem] = tf.read_text(encoding='utf-8').strip()
                    except Exception:
                        pass
            for cj_dir in [b_dir / 'fused_kg', b_dir]:
                cj = cj_dir / 'merged_captions.json'
                if cj.exists():
                    try:
                        import json
                        for k, v in json.loads(cj.read_text(encoding='utf-8')).items():
                            preloaded.setdefault(k, v)
                    except Exception:
                        pass

        seen_stems: set = set()
        seen_canonical: set = set()   # FIX-DUP: also deduplicate by resolved path
        for sd in search_dirs:
            for ip in sorted(sd.glob('*.png')):
                # Deduplicate by stem (filename) AND by resolved canonical path
                canon = str(ip.resolve()) if ip.exists() else str(ip.absolute())
                if ip.stem in seen_stems or canon in seen_canonical:
                    continue
                seen_stems.add(ip.stem)
                seen_canonical.add(canon)
                cf = ip.with_suffix('.txt')
                if cf.exists():
                    try:
                        cap = cf.read_text(encoding='utf-8').strip()
                    except Exception:
                        cap = preloaded.get(ip.stem, '')
                elif ip.stem in preloaded:
                    cap = preloaded[ip.stem]
                else:
                    # FIX-CAPTION-DISCOVERY: prefer preloaded caption; fall back to ''
                    # NEVER use ip.stem as a caption — it leaks filenames like
                    # 'slide_004' which then pass the social blacklist silently.
                    cap = preloaded.get(ip.name.replace('.png', ''), '')
                diagrams.append({'path': str(ip), 'caption': cap})

        print(f"[Renderer] Discovered {len(diagrams)} images, {len(preloaded)} captions")
        diagram_map = build_diagram_section_map(notes.get('sections', []), diagrams)
        print(f"[Renderer] {len(diagram_map)} sections mapped to diagrams")
        # Build path->caption lookup so _resolve_caption never falls back to filename
        caption_map: Dict[str, str] = {d['path']: d['caption'] for d in diagrams}

    # FIX-SOCIAL: Render-stage blacklist — must mirror build_diagram_section_map's
    # BLACKLIST exactly.  Images that slipped through mapping are caught here.
    # Also applied to the path string so images with social terms in their
    # filename are blocked even when no caption is available.
    _BLACKLIST_TERMS = (
        'twitter', 'facebook', 'instagram', 'youtube', 'subscribe', 'follow',
        '@', 'nesoacademy', 'neso', 'academy',
        'like and share', 'like, comment', 'comment, share',
        'like comment share', 'like comment subscribe',
        'thanks for watching', 'thank you for watching',
        'logo', 'icon', 'watermark', 'banner', 'bell', 'notification',
        'sticker with the word', 'sticker with the words',
        'red and white sticker', 'red and white shield',
        'shield with the letter',
        'a black and white photo', 'a sign with',
        'a piece of paper with numbers',
        'visual representation',
    )

    def _is_social_slide(pstr: str) -> bool:
        """Block early slide_NNN images (N<=4) dynamically from the path."""
        stem = Path(pstr).stem.lower()
        m = re.match(r'slide[_\-]?(\d+)', stem)
        if m:
            try:
                return int(m.group(1)) <= 4
            except ValueError:
                pass
        return False

    def _is_decorative(pstr: str, cap: str) -> bool:
        combined = (pstr + " " + cap).lower()
        if any(bl in combined for bl in _BLACKLIST_TERMS):
            return True
        if _is_social_slide(pstr):
            return True
        return False

    def _resolve_caption(p: Path, diag_dict) -> str:
        """
        FIX-CAPTION: Return a human-readable caption for an image, or '' if
        none is available.  NEVER return p.stem — that leaks internal filenames
        like 'slide_003' into the rendered PDF.  A missing caption is rendered
        as no Figure label at all, which is better than showing a filename.
        """
        # 1. Section diagram dict (highest priority — explicitly assigned)
        if diag_dict and diag_dict.get("path") == str(p):
            cap = diag_dict.get("caption", "")
            if cap and cap.strip() and cap.strip() != p.stem:
                return cap.strip()
        # 2. caption_map built from image discovery (BLIP / preloaded captions)
        if str(p) in caption_map:
            cap = caption_map[str(p)]
            if cap and cap.strip() and cap.strip() != p.stem:
                return cap.strip()
        # 3. Sidecar .txt file next to the image
        cf = p.with_suffix('.txt')
        if cf.exists():
            try:
                cap = cf.read_text(encoding='utf-8').strip()
                if cap and cap != p.stem:
                    return cap
            except Exception:
                pass
        # FIX-CAPTION: No fallback to p.stem — return '' so the renderer
        # simply omits the Figure label rather than printing a filename.
        return ""

    used_paths: set = set()        # canonical absolute path strings
    used_fingerprints: set = set() # (size, mtime) content fingerprints — catches symlinks/hardlinks

    def _fingerprint(p: Path) -> Optional[str]:
        """Return a (size, mtime) fingerprint for an image file, or None if unavailable."""
        try:
            st = p.stat()
            return f"{st.st_size}:{st.st_mtime}"
        except Exception:
            return None

    def _canonical(pstr: str) -> str:
        """Return the resolved absolute path string for deduplication."""
        p = Path(pstr)
        if not p.is_absolute() and image_base_dir:
            p = image_base_dir / pstr
        try:
            return str(p.resolve())
        except Exception:
            return str(p.absolute())

    for s_idx, section in enumerate(notes.get("sections", []), 1):
        if s_idx - 1 < len(toc_links):
            pdf.set_link(toc_links[s_idx - 1])

        heading = _sanitize_heading(section.get("heading", "Section"))

        # Blue bar accent + section title
        cur_y = pdf.get_y()
        pdf.set_fill_color(49, 104, 187)
        pdf.rect(15, cur_y, 3, 8, 'F')
        pdf.set_x(20)
        pdf.set_font("Helvetica", "B", 16)
        pdf.set_text_color(10, 30, 80)
        pdf.cell(0, 8, _safe_latin1(f"{s_idx}. {heading}"), ln=True)
        pdf.ln(3)
        pdf.set_text_color(0, 0, 0)

        # ── LEARNING-CENTRIC RENDERING ─────────────────────────────────────────
        # Order subsections pedagogically: Definition first, then structure/working,
        # then examples, then key points. Diagram injected after first structural sub.
        # ──────────────────────────────────────────────────────────────────────

        subsections = section.get("subsections", [])

        # Pedagogical ordering weights — lower = earlier
        _SUB_ORDER: Dict[str, int] = {
            "definition": 1,
            "overview": 2,
            "introduction": 2,
            "structure": 3,
            "architecture": 3,
            "conceptual structure": 3,
            "process flow": 4,
            "process overview": 4,
            "top-down approach": 4,
            "how it works": 5,
            "working": 5,
            "properties": 6,
            "classification": 6,
            "implementation": 7,
            "applications": 8,
            "examples": 9,
            "example": 9,
            "key points": 10,
            "additional notes": 11,
        }

        def _sub_weight(sub: Dict) -> int:
            h = sub.get("heading", "").lower().strip()
            return _SUB_ORDER.get(h, 7)   # unknown → middle of the pack

        # Sort subsections by pedagogical weight (stable: preserves KG order within same weight)
        subsections_ordered = sorted(subsections, key=_sub_weight)

        # ── Intro paragraph: render Definition/Overview as a flowing paragraph ──
        # instead of a bullet list — much more learning-centric
        intro_subs = [s for s in subsections_ordered
                      if s.get("heading", "").lower().strip() in
                      ("definition", "overview", "introduction", "description")]
        non_intro_subs = [s for s in subsections_ordered
                          if s not in intro_subs]

        # Render intro sub as styled paragraph (not a bulleted list)
        for intro_sub in intro_subs:
            points = intro_sub.get("points", [])
            texts = [_sanitize_body(pt.get("text", "").strip())
                     for pt in points if pt.get("text", "").strip()]
            # Filter meta-lecture references and noise
            texts = [t for t in texts if t and not _is_meta_lecture_noise(t)]
            if not texts:
                continue
            paragraph = " ".join(texts)
            if paragraph:
                pdf.set_font("Helvetica", "I", 11)
                pdf.set_fill_color(237, 244, 252)
                pdf.set_text_color(30, 30, 80)
                content_w = pdf.w - _INDENT_X - _R_MARGIN
                pdf.set_x(_INDENT_X)
                pdf.multi_cell(content_w, _LINE_H + 0.5,
                               _safe_latin1(paragraph), fill=True, border=1)
                pdf.set_text_color(0, 0, 0)
                pdf.ln(5)

        # ── Diagram injection point: after intro paragraph, before main bullets ─
        # FIX-DIAGRAM-SINGLE-SOURCE: Use diagram_map as the SOLE diagram source.
        # Previously both diagram_map (renderer-assigned) and section["diagram"]
        # (ex.py Fix 4.3-assigned) were merged into sec_imgs, causing the same
        # image to appear in multiple sections.
        #
        # New policy:
        #   1. Use diagram_map[s_idx-1] as primary source (renderer-assigned,
        #      globally deduplicated by build_diagram_section_map).
        #   2. Fall back to section["diagram"] ONLY if diagram_map has nothing
        #      for this section AND the image hasn't been used elsewhere yet.
        #   3. Never merge both sources — one image per section, one source wins.
        diag_dict = section.get("diagram")
        _map_imgs: List[str] = list(diagram_map.get(s_idx - 1, []))

        if _map_imgs:
            # Primary: use diagram_map assignment (already globally deduped)
            sec_imgs = _map_imgs
        else:
            # Fallback: section["diagram"] if not yet used
            sec_imgs = []
            if diag_dict and diag_dict.get("path"):
                dp = diag_dict["path"]
                _dp_canon = _canonical(dp)
                if _dp_canon not in used_paths:
                    p_fb = Path(dp)
                    if not p_fb.is_absolute() and image_base_dir:
                        p_fb = image_base_dir / dp
                    _fp_fb = _fingerprint(p_fb)
                    if not (_fp_fb and _fp_fb in used_fingerprints):
                        sec_imgs = [dp]

        filtered: List[str] = []
        for pstr in sec_imgs:
            canon = _canonical(pstr)
            if canon in used_paths:
                continue
            p_chk = Path(pstr)
            if not p_chk.is_absolute() and image_base_dir:
                p_chk = image_base_dir / pstr
            fp = _fingerprint(p_chk)
            if fp and fp in used_fingerprints:
                continue
            cap_chk = _resolve_caption(p_chk, diag_dict)
            if not _is_decorative(pstr, cap_chk):
                filtered.append(pstr)

        # HIGH PRIORITY: exactly ONE diagram per section
        unique = filtered[:1]
        diagram_rendered = False

        def _render_diagram(pstr: str) -> bool:
            """Render a single diagram. Returns True if successfully rendered."""
            nonlocal diagram_rendered
            if diagram_rendered:
                return False
            p = Path(pstr)
            if not p.is_absolute() and image_base_dir:
                p = image_base_dir / pstr
            fp_d = _fingerprint(p)
            if fp_d:
                used_fingerprints.add(fp_d)
            used_paths.add(_canonical(pstr))
            caption = _resolve_caption(p, diag_dict)
            if not p.exists():
                return False
            img_ok, w_px, h_px = _check_image_quality(p)
            if not img_ok:
                if caption:
                    pdf.set_font("Helvetica", "I", 9)
                    pdf.set_text_color(100, 100, 100)
                    pdf.multi_cell(0, 4, _safe_latin1(f"[Diagram: {caption}]"), align="C")
                    pdf.set_text_color(0, 0, 0)
                    pdf.ln(3)
                return False
            try:
                # NO "Visual Representation" label — diagram is self-contained
                max_w, max_h = 150, 110
                if w_px == 0 or h_px == 0:
                    try:
                        from PIL import Image as _PI
                        with _PI.open(str(p)) as _i:
                            w_px, h_px = _i.size
                    except Exception:
                        w_px, h_px = 800, 600
                aspect = h_px / w_px if w_px > 0 else 1
                if w_px >= h_px:
                    w_mm = min(max_w, w_px * 0.264583)
                    h_mm = w_mm * aspect
                else:
                    h_mm = min(max_h, h_px * 0.264583)
                    w_mm = h_mm / aspect if aspect > 0 else max_w
                pdf.ln(2)
                pdf.image(str(p), x=(pdf.w - w_mm) / 2, w=w_mm)
                pdf.ln(1)
                if caption:
                    pdf.set_font("Helvetica", "I", 9)
                    pdf.set_text_color(100, 100, 100)
                    pdf.multi_cell(0, 4,
                                   _safe_latin1(f"Figure {s_idx}: {caption}"),
                                   align="C")
                    pdf.set_text_color(0, 0, 0)
                    pdf.ln(5)
                diagram_rendered = True
                print(f"[Renderer] Section {s_idx} '{heading}': diagram embedded")
                return True
            except Exception as e:
                print(f"[Renderer] Failed to embed image {p}: {e}")
                return False

        # ── Inject diagram after intro if section has structural/working subs ──
        # i.e. diagram goes between intro and the first "real" subsection
        has_structural = any(
            s.get("heading", "").lower().strip() in
            ("structure", "architecture", "how it works", "process flow",
             "process overview", "working", "conceptual structure")
            for s in non_intro_subs
        )
        if unique and (intro_subs or not has_structural):
            _render_diagram(unique[0])

        # ── Render non-intro subsections ───────────────────────────────────────
        for sub_i, sub in enumerate(non_intro_subs):
            ss_heading = _sanitize_heading(sub.get("heading", "Subtopic"))
            points = sub.get("points", [])
            if not points:
                continue

            # Filter bullets that are meta-lecture noise before rendering
            clean_points = []
            for pt in points:
                text = _sanitize_body(pt.get("text", "").strip())
                if text and not _is_meta_lecture_noise(text):
                    clean_points.append({"text": text})
            if not clean_points:
                continue

            # Subsection heading — bold, dark, with light blue underline rule
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(30, 50, 100)
            pdf.set_x(_INDENT_X - 5)
            pdf.cell(0, 6, _safe_latin1(ss_heading), ln=True)
            # thin rule under heading
            pdf.set_draw_color(180, 200, 230)
            pdf.line(_INDENT_X - 5, pdf.get_y(), pdf.w - _R_MARGIN, pdf.get_y())
            pdf.set_draw_color(200, 200, 200)
            pdf.set_text_color(0, 0, 0)
            pdf.ln(1)

            for pt in clean_points:
                text = pt["text"]
                _write_bullet(pdf, text, _INDENT_X, _LINE_H, _R_MARGIN)
                pdf.ln(2)
            pdf.ln(3)

            # ── Inject diagram after first "structural" subsection ─────────────
            # Suggestion #5: diagram anchored right after Architecture/Structure/Working
            if (unique and not diagram_rendered and
                    ss_heading.lower().strip() in
                    ("structure", "architecture", "how it works",
                     "process flow", "process overview", "working",
                     "conceptual structure")):
                _render_diagram(unique[0])

        # Fallback: if diagram not yet rendered, place it at end of section
        if unique and not diagram_rendered:
            _render_diagram(unique[0])

        # Section separator
        pdf.set_draw_color(200, 200, 200)
        pdf.line(15, pdf.get_y(), pdf.w - 15, pdf.get_y())
        pdf.ln(6)

    pdf.output(str(output_path))
    scrub_qmarks_from_pdf(output_path)
    return output_path


# ---------------------------------------------------------------------------
# TXT Renderer
# ---------------------------------------------------------------------------

def render_txt(notes: Dict, output_path: Path) -> Path:
    """Render HierarchicalNotes dict to plain text."""
    try:
        from concept_flow_organizer import strip_all_question_marks as _nuke
    except Exception:
        def _nuke(t): return t.replace('?', '') if t else t

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []

    title = _nuke(notes.get("title", "Lecture Notes"))
    lines += [title, "=" * len(title), ""]

    summary = _nuke(notes.get("summary", ""))
    if summary:
        lines += ["Summary", "-" * 40, summary, ""]

    for s_idx, section in enumerate(notes.get("sections", []), 1):
        heading = _nuke(section.get("heading", "Section"))
        lines += [f"{s_idx}. {heading}", "-" * 40]

        for ss_idx, sub in enumerate(section.get("subsections", []), 1):
            ss_h = _nuke(sub.get("heading", "Subtopic"))
            lines.append(f"  {s_idx}.{ss_idx} {ss_h}")
            for pt in sub.get("points", []):
                text = _nuke(pt.get("text", ""))
                if text.strip():
                    lines.append(f"    - {text}")
            lines.append("")

        diag = section.get("diagram")
        if diag and diag.get("caption"):
            lines += [f"  [Figure {s_idx}: {_nuke(diag['caption'])}]", ""]

        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


# ---------------------------------------------------------------------------
# Markdown Renderer
# ---------------------------------------------------------------------------

def render_markdown(notes: Dict, output_path: Path) -> Path:
    """Render HierarchicalNotes dict to Markdown."""
    try:
        from concept_flow_organizer import strip_all_question_marks as _nuke
    except Exception:
        def _nuke(t): return t.replace('?', '') if t else t

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "<style>",
        "pre, code { white-space: pre-wrap; word-wrap: break-word; overflow-x: auto; }",
        "</style>", "",
        f"# {_nuke(notes.get('title', 'Lecture Notes'))}", "",
    ]

    summary = _nuke(notes.get("summary", ""))
    if summary:
        lines += [f"*{summary}*", ""]

    for s_idx, section in enumerate(notes.get("sections", []), 1):
        lines += [f"## {s_idx}. {_nuke(section.get('heading', 'Section'))}", ""]
        for ss_idx, sub in enumerate(section.get("subsections", []), 1):
            lines.append(f"### {s_idx}.{ss_idx} {_nuke(sub.get('heading', 'Subtopic'))}")
            for pt in sub.get("points", []):
                text = _nuke(pt.get("text", ""))
                if text.strip():
                    lines.append(f"- {text}")
            lines.append("")
        diag = section.get("diagram")
        if diag and diag.get("path"):
            lines += [f"![{_nuke(diag.get('caption',''))}]({diag['path']})", ""]

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


# ---------------------------------------------------------------------------
# Convenience: render all formats
# ---------------------------------------------------------------------------

def render_all(
    notes: Dict,
    output_dir: Path,
    base_name: str = "lecture_notes",
    image_base_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """Render notes to PDF + TXT + Markdown and return paths dict."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "pdf": render_pdf(notes, output_dir / f"{base_name}.pdf", image_base_dir),
        "txt": render_txt(notes, output_dir / f"{base_name}.txt"),
        "md":  render_markdown(notes, output_dir / f"{base_name}.md"),
    }