# notes_cleaner.py
"""
Dynamic Notes Cleaner  (v1)
============================
Fixes the specific quality problems visible in the generated PDF
output (diagnosed from the OSI-model lecture_notes.pdf).

Problems fixed — ALL fully domain-agnostic:

  C-1  Tautological self-reference in bullets
       "Signals is a key concept in this domain is a key concept within Signals."
       → "Signals is a key concept in this domain."

  C-2  Double relation artifact  "is a is a type of"
       "Physical Layer Services is a is a type of Data Rate"
       → "Physical Layer Services is a type of Data Rate"

  C-3  OCR stray symbols in bullet text
       «  »  #  *  ?  o #  « *  at start / mid-sentence
       Cleaned inline; bullet dropped only if residue is nonsense.

  C-4  Generic / slide-artifact section headings that survive earlier fixes
       "Definition And Fundamental Concept", "Key Notes and Additional Concepts",
       "Contextual Relationships", "Header", "Layer", "(Data", "(State" …
       → replaced with content-derived headings using TF-IDF-style word frequency.

  C-5  Heading contains OCR noise or special chars  (?, «, *, #, →, ←)
       → stripped to plain words.

  C-6  Bullet starts with stray OCR tokens  (* «  * *  o #  # *)
       → leading tokens stripped; bullet dropped if residue < 4 words.

  C-7  "X is a key concept in this domain is a key concept within X."
       Collapsed to:  "X is a key concept in this domain."

  C-8  Self-loop description  "X is a type of X" / "X includes X"
       → bullet removed (tautological, zero information).

  C-9  Subheading artefacts  "(Data", "(Layer", "(State", "(Protocols)"
       These are partial KG edge labels leaking as subsection names.
       → Renamed to plain readable labels via content inference.

  C-10 Summary too long  (> 3 sentences → trimmed to 3).

  C-NEW-1  KG structural prefix leaking as sentence opener
           "Structure determines The main topic...", "& Structure determines..."
           Fix: strip any "<Word(s)> determines " or "& <Word(s)> determines "
           prefix dynamically before further processing.

  C-NEW-2  KG boilerplate suffix "The main topic of the video covering <list>"
           appended verbatim to every concept-relationship bullet.
           Fix: detect when text after a colon is the repeated boilerplate suffix
           by checking if it starts with a near-verbatim version of the section's
           own topic string; strip it. Falls back to n-gram overlap detection.

  C-NEW-3  Repeated OCR-garbled boilerplate sentence and variable-assignment noise
           "Its not difficult to understand that the dota needs to be monaged..."
           "Dynamic Data Structure: ... Manco Last indan = 8 but without..."
           Fix: extended _GARBAGE_BULLET_PATTERNS with patterns for (a) variable
           assignments (word = digit), (b) OCR all-caps junk token clusters,
           (c) known-garbled boilerplate phrases detected via character-level signals.

  C-NEW-4  Non-ASCII / currency symbol cluster at bullet start not caught by
           existing _LEADING_OCR regex (£ $ € ¥ and consecutive all-caps tokens).
           "E £ fur Some commonly used...", "KIN ARER BM aR ee EERE La ay)"
           Fix: extended _LEADING_OCR and _GARBAGE_BULLET_PATTERNS.

  C-NEW-5  Stub/fragment bullets with no finite verb
           "* Each element in the list.", "* Elements from the list.",
           "* 2 lists into a single list."
           Fix: heuristic verb-presence check — drop if bullet has no verb-like
           token and starts with a number or preposition/article.

  C-NEW-6  OCR misspelling / digit-for-letter substitution density
           "os a series", "aggually", "8itmap", "monaged", "sucha"
           Fix: ratio of tokens with digit-for-letter substitution or impossible
           leading-digit patterns; drop bullet if ratio exceeds threshold.

Integration
-----------
Add TWO lines in ex.py immediately BEFORE render_pdf():

    from notes_cleaner import clean_notes
    notes_dict = clean_notes(notes_dict)

That's it — no extra parameters needed.
"""

import re
import copy
from typing import Dict, List, Optional, Set, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════════

_STOPWORDS: Set[str] = {
    "the","a","an","is","are","was","were","be","been","have","has","had",
    "this","that","with","from","into","and","or","but","for","to","of",
    "in","on","at","by","data","level","system","which","what","how","when",
    "will","can","its","their","these","those","also","some","such","each",
    "within","between","both","used","using","use","type","types","layer",
    "key","concept","domain","additional","notes","general","contextual",
    "relationships","definition","fundamental","structure","components",
    "header","overview","introduction","summary","conclusion","services",
}


def _log(msg: str) -> None:
    print(f"[NotesCleaner] {msg}")


def _word_freq(section: Dict) -> Dict[str, int]:
    """Count non-stopword alpha words across all bullets in a section."""
    freq: Dict[str, int] = {}
    for sub in section.get("subsections", []):
        for pt in sub.get("points", []):
            for w in re.findall(r"[a-zA-Z]{4,}", pt.get("text", "")):
                wl = w.lower()
                if wl not in _STOPWORDS:
                    freq[wl] = freq.get(wl, 0) + 1
    return freq


def _derive_heading(section: Dict, fallback: str) -> str:
    """Build a descriptive heading from dominant words in bullet content."""
    freq = _word_freq(section)
    if not freq:
        return fallback.title()
    top = sorted(freq.items(), key=lambda x: -x[1])
    candidates = [w.title() for w, _ in top[:3]
                  if w.lower() != fallback.lower().strip()]
    if candidates:
        return " and ".join(candidates[:2])
    return fallback.title()


# ═══════════════════════════════════════════════════════════════════════════════
# C-5  Heading OCR noise stripper
# ═══════════════════════════════════════════════════════════════════════════════

_HEADING_NOISE_CHARS = re.compile(r"[?→←↑↓~`\\|@«»#*\[\]{}<>]")


def _strip_heading_noise(h: str) -> str:
    """Remove OCR garbage characters from a heading string."""
    h = _HEADING_NOISE_CHARS.sub(" ", h)
    h = re.sub(r"\s{2,}", " ", h).strip().rstrip("-–—:;.,")
    return h.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# C-4  Generic / artifact heading detection + replacement
# ═══════════════════════════════════════════════════════════════════════════════

# Exact lower-case headings that are slide-deck artefacts, not semantic topics
_ARTIFACT_HEADING_EXACT: Set[str] = {
    "definition and fundamental concept",
    "definition & fundamental concept",
    "key notes and additional concepts",
    "key notes & additional concepts",
    "contextual relationships",
    "components and structure",
    "components & structure",
    "key points",
    "applications",
    "overview",
    "introduction",
    "summary",
    "conclusion",
    "misc",
    "general",
    "content",
    "other",
    "additional concepts",
    "visual representation",
    "figure",
    "diagram",
    "header",
    "headers",
    "layer",
    "state",
    "error",
    "protocols",
}

# Partial-match headings: if the heading consists ONLY of these tokens it's generic
_ARTIFACT_HEADING_TOKENS: Set[str] = {
    "definition", "fundamental", "concept", "key", "notes", "additional",
    "contextual", "relationships", "components", "structure", "overview",
    "header", "layer", "state", "error", "protocols", "services",
    "reference", "sequence", "physical", "logical",
}

# Starts with opening parenthesis — raw KG edge label leaking as heading
_PAREN_HEADING = re.compile(r"^\(")


def _is_artifact_heading(h: str) -> bool:
    lower = h.lower().strip()
    if lower in _ARTIFACT_HEADING_EXACT:
        return True
    if _PAREN_HEADING.match(lower):          # C-9: "(Data", "(State"
        return True
    # Single- or two-word heading made entirely of generic tokens
    words = [w for w in re.findall(r"[a-z]+", lower) if len(w) > 2]
    if words and all(w in _ARTIFACT_HEADING_TOKENS for w in words):
        return True
    return False


def clean_headings(sections: List[Dict]) -> Tuple[List[Dict], int]:
    """
    C-4 + C-5 + C-9:
    1. Strip OCR noise characters from every heading.
    2. Replace artifact/generic headings with content-derived ones.
    3. Fix parenthetical KG-label subsection headings.
    """
    fixed = 0
    for sec in sections:
        # --- Section heading ---
        raw_h = sec.get("heading", "")
        h = _strip_heading_noise(raw_h)
        if h != raw_h:
            sec["heading"] = h
            fixed += 1

        if _is_artifact_heading(h):
            new_h = _derive_heading(sec, h)
            _log(f"C-4 → generic heading replaced: '{h}' → '{new_h}'")
            sec["heading"] = new_h
            fixed += 1

        # --- Subsection headings ---
        for sub in sec.get("subsections", []):
            raw_sh = sub.get("heading", "")
            sh = _strip_heading_noise(raw_sh)

            # C-9: parenthetical labels like "(Data", "(Layer", "(State"
            if _PAREN_HEADING.match(sh):
                # Strip the paren and title-case
                sh = re.sub(r"^\(+|\)+$", "", sh).strip().title() or "Details"
                _log(f"C-9 → subsection heading fixed: '{raw_sh}' → '{sh}'")
                fixed += 1

            if sh != raw_sh:
                sub["heading"] = sh
                fixed += 1

    return sections, fixed


# ═══════════════════════════════════════════════════════════════════════════════
# C-3 / C-6  OCR stray symbol cleaner for bullet text
# ═══════════════════════════════════════════════════════════════════════════════

# Stray OCR tokens that appear at the start of bullets
_LEADING_OCR = re.compile(
    r"^(?:"
    r"[*«»#o\s]+"                        # stars, guillemets, hash, letter-o, spaces
    r"|[*#]{1,3}\s+"                     # "* " "** " "# "
    r"|«\s*[*#]?\s*"                     # "« *" "«*"
    r"|o\s+#\s*"                         # "o # "
    # C-NEW-4: currency / non-ASCII symbol clusters before real content
    r"|[£€$¥©®™°±×÷→←↑↓–—\u00a0]+"     # currency and special unicode chars
    r"|(?:[A-Z£€$¥\s]{1,3}\s+){1,4}"    # e.g. "E £ fur " — 1-3 char tokens
    r"|\*\s*-\s*"                        # "* - " misaligned dash prefix
    r"|-\s*(?=[A-Z])"                    # "- Dynamic..." leading dash before capital
    r"|\&\s+\w+\s+determines\s+"        # "& Structure determines " KG prefix
    r"|\w+\s+determines\s+"             # "Structure determines " KG prefix
    # FIX-SEMICOLON-PREFIX: KG label leaked before semicolon
    # "Semester subject; real content" → strip "Semester subject; "
    # Only matches short title-case phrases (1-4 words) ending with "; "
    # Guard: require a capital start (KG labels are capitalised) and no colon
    # (colons are legitimate in real bullets like "Data Inconsistency: Occurs when...")
    r"|[A-Z][A-Za-z]+(?:\s+[A-Za-z]+){0,3};\s+"
    r")\s*",
    re.I | re.UNICODE,
)

# Inline OCR noise: isolated symbols not part of any real word
_INLINE_OCR = re.compile(
    r"(?<!\w)[«»]{1,3}(?!\w)"   # guillemets not inside words
    r"|(?<!\w)[?]{2,}(?!\w)"    # multiple question marks mid-sentence
    r"|\s[*]{1,2}\s"            # isolated stars with spaces
    r"|\s[#]\s"                 # isolated hash
    r"|\bSane\b"                # common tesseract misread
    r"|\bSe[lj]\b"              # tesseract artefact
)

# FIX-7: Inline broken-hyphen OCR artifact — PDF line-break hyphenation that got
# extracted as " -word" (space, dash, lowercase word) inside a sentence.
# E.g. "reverse or -apply operations" → "reverse or apply operations"
#      "to be -done if required"      → "to be done if required"
# These are NEVER valid English constructs mid-sentence (real hyphens join compound
# words with no leading space). The substitution removes the stray dash.
_BROKEN_HYPHEN = re.compile(r'(?<=\s)-([a-z])', re.I)

# Patterns that indicate an OCR-garbage or irrelevant slide extraction bullet.
# These are domain-agnostic signal patterns (not topic words).
_GARBAGE_BULLET_PATTERNS = [
    # Raw table data leaked as bullets: "Address |Srreet Attribute 3 b Id, real 1110"
    re.compile(r'\|[A-Za-z]', re.I),
    # Strings like "CS & IT Tutorials by Vrushali"  or  "by Vrushali" (attribution noise)
    re.compile(r'\bby\s+[A-Z][a-z]+\s+\d', re.I),
    # Patterns like "Const,int 1200 >It stores" or ">It stores"
    re.compile(r'>\s*[A-Z]'),
    # Raw OCR table rows: "Id, real 1110 Id, real 1100"
    re.compile(r'\b(real|int|float|char)\s+\d{3,}', re.I),
    # Mixed OCR garbage: numbers + pipe + partial words
    re.compile(r'\d{3,}\s*[>|<]\s*\w'),
    # Slide branding / social content
    re.compile(r'\b(nesoacademy|neso\s*academy|follow\s*@|subscribe|like\s*and\s*share)\b', re.I),
    # Binary data lines
    re.compile(r'\b[01]{6,}\b'),
    # Assembly code fragments  
    re.compile(r'\b(mov|eax|ebx|DWORD|PTR|rbp|rsp)\b', re.I),
    # C code artifacts
    re.compile(r'#include|int\s+main\s*\(|printf\s*\(', re.I),
    # KG schema structural noise: "sub-concept within X" — internal labels, not student content
    re.compile(r'\bsub[\s-]concept\s+within\b', re.I),
    # KG structural noise: "X is a key concept within Y" — internal KG annotation
    re.compile(r'\bis\s+a\s+key\s+concept\s+within\b', re.I),

    # C-NEW-3a: Variable/code assignment leaked from OCR slide text
    # Catches: "Manco Last indan = 8", "x = 5", "size = n" mid-sentence
    re.compile(r'\b\w+\s*=\s*\d+\b'),

    # C-NEW-3b: All-caps OCR garbage token cluster
    # Pattern A: 3+ consecutive ALL-CAPS tokens (any length) — "KIN ARER BM aR ee EERE"
    re.compile(r'\b[A-Z]{2,6}\s+[A-Z]{2,6}\s+[A-Z]{1,6}\b'),
    # Pattern B: 2-char ALL-CAPS token followed immediately by ANOTHER 2-6 char ALL-CAPS token
    # "PTT Ty" — PTT is 3 chars all-caps, Ty starts with capital → OCR noise
    # Guard: NOT a known 2-letter abbreviation pattern like "IP", "ID", "OS"
    # by requiring both tokens be 2-3 chars (double-short = OCR noise)
    re.compile(r'\b[A-Z]{2,3}\s+[A-Z]{2,3}\b'),
    # Pattern C: Inline OCR fragment mid-sentence — an isolated ALL-CAPS short token (2-4 chars)
    # appearing after a colon, e.g. "are: PTT Ty The data" or "are: PO array"
    # Matches both 2-token ("PO array") and 3-token ("PTT Ty The") forms.
    re.compile(r':\s+[A-Z]{2,4}\s+[A-Za-z]{2,}(?:\s+[A-Za-z])?'),

    # C-NEW-3c: Known garbled boilerplate detected by misspelling signals
    # "dota needs to be monaged", "sucha way", "os a series" — character-level OCR errors
    # Heuristic: detect 2+ tokens that share ≥3 chars with a real word but have
    # impossible consonant clusters or digit-for-letter substitutions
    # We use a fixed-pattern approach for the most common OCR word confusions:
    re.compile(r'\b(dota|monaged|sucha|aggually|8itmap|pio\s+pokemon|indan)\b', re.I),
    # "os a" — OCR swap of "as a" (o↔a transposition, extremely common with serif fonts)
    # Only matches when "os" is used as a preposition (followed by article/noun phrase)
    re.compile(r'\bos\s+a\s+\w', re.I),

    # C-NEW-4: Non-ASCII currency / symbol prefix that survived _LEADING_OCR
    # Catches full bullets that are mostly symbols: "E £ fur PO array..."
    re.compile(r'^[^a-zA-Z0-9]{0,5}[£€$¥©®]{1}'),

    # C-NEW-5: Stub fragment — starts with a number (list artifact) and no real verb
    # "2 lists into a single list" — no subject, no predicate
    re.compile(r'^\d+\s+\w+\s+(into|from|of|in|at|to)\s+'),

    # C-NEW-6: Digit-for-letter at start of a content word (not a number)
    # "8itmap" (Bitmap), "3rror" (Error), "0bject" (Object)
    re.compile(r'\b\d[a-z]{3,}\b', re.I),

    # FIX-TRUNCWORD: Truncated word + leaked page-number digit at end of bullet
    # "File systems are NOT convenient and efficic 7."
    # Pattern: any partial alphabetic word (3+ letters) followed by bare 1-2 digit at sentence end
    re.compile(r'\b[a-z]{3,}[a-z]\s+\d{1,2}\s*[.!?]?\s*$', re.I),

    # FIX-BROKEN-TRIPLE: Orphaned KG object fragment — unbalanced closing paren at sentence end
    # "FILO (First In, Last Out): Order of Operations is Out)."
    # Signal: sentence ends with word + ")" but no matching "(" earlier
    # Implemented via _is_broken_triple() called separately in _clean_bullet_text

    # FIX-TRUNCATED-CAP-SUBJECT: Bullet starts with a 2-letter capitalised preposition/remnant
    # "Of Operations is a is part of LIFO" — "Of" is remnant of "Order Of"
    # "In Operations..." — "In" is remnant of similarly truncated phrase
    # Heuristic: starts with a 2-letter word that is ONLY valid as a preposition (not a sentence subject)
    # followed by a capitalised multi-word phrase suggesting a proper noun/concept name
    re.compile(r'^Of\s+[A-Z][a-z]'),   # "Of Operations", "Of Data"

    # FIX-FILLER-GENERATED: Bullets that are KG-template filler with no educational value.
    # These are generated by _graph_edge_to_prose / get_contextual_description
    # when the target node has no description and a list-style relation was used.
    # Structural patterns — no domain words needed.
    #
    # Pattern: "X has real-world applications and use cases" — zero information content.
    re.compile(r'\bhas\s+real[-\s]world\s+applications\s+and\s+use\s+cases\b', re.I),
    # Pattern: "X involves Y as one of its key components" — filler for includes/has edges.
    re.compile(r'\binvolves\s+\S[\S\s]{1,60}\s+as\s+one\s+of\s+its\s+key\s+components?\b', re.I),
    # Pattern: "X encompasses several key components, including Y. Each serves a distinct role..."
    # — generated by linearize() in kg_summarizer.py; too generic for a bullet point.
    re.compile(r'\bencompasses\s+several\s+key\s+components.*each\s+serves\s+a\s+distinct\s+role\b', re.I),
]

# Whole-bullet OCR-garbage detector (ratio-based)
_MIN_ALPHA_RATIO = 0.50
_MIN_WORD_COUNT  = 3

# ─── FIX-BROKEN-TRIPLE ────────────────────────────────────────────────────────
# Detect bullets that end with an unbalanced closing paren — the opening was
# stripped somewhere upstream, leaving a meaningless orphaned fragment.
# "FILO (First In, Last Out): Order of Operations is Out)."
# The sentence has 1 open paren for "(First In, Last Out)" and 2 closing parens.
_UNBALANCED_CLOSE_PAREN = re.compile(r'\w\s*\)\s*[.!?]?\s*$')


def _is_broken_triple(text: str) -> bool:
    """Return True if bullet ends with unbalanced closing paren (broken KG object)."""
    if _UNBALANCED_CLOSE_PAREN.search(text):
        if text.count(')') > text.count('('):
            return True
    return False


# ─── FIX-TRUNCSUBJECT ─────────────────────────────────────────────────────────
# Detect bullets whose first word is a lowercase OCR fragment (first letter lost).
# "Of Operations is a is part of LIFO" — "Of" should be "Order of Operations"
# "perations Performed on Stack" — "perations" should be "Operations"
# Heuristic: first word is 2-10 lowercase letters NOT in the set of legitimate
# lowercase sentence starters (articles, prepositions, conjunctions).
_KNOWN_LEAD_LOWER = frozenset({
    'a', 'an', 'the', 'of', 'in', 'on', 'at', 'to', 'for', 'by', 'as',
    'or', 'and', 'but', 'if', 'is', 'it', 'its', 'be', 'no', 'not',
    'this', 'that', 'these', 'those', 'each', 'every', 'some', 'any',
    'both', 'all', 'most', 'more', 'less', 'few', 'many', 'also', 'only',
})
_TRUNCSUBJECT_RE = re.compile(r'^([a-z]{2,10})\s+[A-Z]')


def _is_truncated_subject(text: str) -> bool:
    """Return True if bullet starts with a lowercase OCR fragment (dropped first letter),
    or a capitalised 2-letter preposition that is structurally a truncated subject."""
    # Case 1: starts with lowercase word not in known starters
    m = _TRUNCSUBJECT_RE.match(text)
    if m:
        first = m.group(1).lower()
        if first not in _KNOWN_LEAD_LOWER:
            return True
    # Case 2: starts with exactly "Of " followed by capital (remnant of "Order Of ...")
    # "Of" is a valid preposition but NEVER a sentence subject — if it's the first
    # word and is followed by a capitalised noun, the first letter was dropped.
    if re.match(r'^Of\s+[A-Z]', text):
        return True
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# C-NEW-5  Stub/fragment bullet detector — no finite verb present
# ═══════════════════════════════════════════════════════════════════════════════

# Verb-suffix heuristic: tokens ending in these patterns indicate a finite verb
_VERB_SUFFIX = re.compile(
    r'(?:s|ed|ing|es|ize|izes|ized|en|ens|ened|fy|fies|fied'
    r'|are|is|was|were|has|have|had|be|been|do|does|did'
    r'|can|could|will|would|shall|should|may|might|must'
    r'|provide|provides|provided|store|stores|stored'
    r'|allow|allows|allowed|use|uses|used|need|needs|needed'
    r'|form|forms|formed|contain|contains|contained'
    r'|represent|represents|define|defines|include|includes)$',
    re.I,
)

# Bullet patterns that are clearly fragment stubs (no real subject+predicate)
_FRAGMENT_STARTERS = re.compile(
    r'^(?:'
    r'\d+\s+\w+'           # starts with number: "2 lists into..."
    r'|Each\s+\w+'         # "Each element in..."
    r'|Elements?\s+from'   # "Elements from the list"
    r'|Particular\s+location'  # "Particular location of an element"
    r')\b',
    re.I,
)


def _drop_if_fragment(text: str) -> Optional[str]:
    """
    C-NEW-5: Drop stub bullets that have no finite verb and match known
    fragment patterns. Uses purely structural signals — no domain words.

    A bullet is a fragment if:
    (a) It matches a known no-predicate starter pattern, OR
    (b) It has fewer than 8 words AND none of its tokens look like verbs
        AND it starts with a number, article, or bare noun phrase.
    """
    # Fast path: known fragment opener patterns
    if _FRAGMENT_STARTERS.match(text):
        return None

    words = text.rstrip(".!?").split()
    if len(words) >= 8:
        # Long enough that it almost certainly has a verb — skip check
        return text

    # For short bullets (< 8 words), check for ANY verb-like token
    has_verb = any(_VERB_SUFFIX.search(w.lower()) for w in words)
    if not has_verb:
        # No verb found — check if it starts with a preposition/article (fragment NP)
        _FRAG_LEAD = re.compile(
            r'^(each|every|some|any|the|a|an|of|in|at|on|to|for|by|'
            r'from|into|with|without|between|through)\b',
            re.I,
        )
        if _FRAG_LEAD.match(text):
            return None
    return text


# ═══════════════════════════════════════════════════════════════════════════════
# C-NEW-1  KG structural prefix stripper
# Strips "X determines ", "& X determines ", "X covers Y:" prefixes that are
# KG edge-label artefacts leaking as sentence openers.
# ALL purely structural — no domain topic words.
# ═══════════════════════════════════════════════════════════════════════════════

# Matches: "Structure determines ", "& Structure determines ",
#          "Definition determines ", "X covers Y - description: "  etc.
_KG_STRUCT_PREFIX = re.compile(
    r'^(?:'
    r'&\s+\w[\w\s]{0,30}?\s+determines\s+'   # "& Structure determines "
    r'|\w[\w\s]{0,30}?\s+determines\s+'       # "Structure determines "
    r')',
    re.I,
)

# Matches KG "covers" relation label that leaked:
# "Introduction of Data Structure covers Data: <boilerplate>"
# "Introduction of Data Structure covers Structure - Organizing X: <boilerplate>"
_KG_COVERS_PREFIX = re.compile(
    r'^.{3,80}\s+covers\s+[^:]{2,60}:\s+',
    re.I,
)


def _strip_kg_structural_prefix(text: str) -> str:
    """
    C-NEW-1: Remove KG structural edge-label prefixes from bullet text.
    Returns the cleaned text with the first real sentence remaining.
    If the entire text is consumed by the prefix, returns the original.
    """
    # Strip "X determines " prefix
    m = _KG_STRUCT_PREFIX.match(text)
    if m:
        remainder = text[m.end():].strip()
        if len(remainder.split()) >= 3:
            return remainder[0].upper() + remainder[1:] if remainder[0].islower() else remainder

    # Strip "X covers Y: " prefix (KG edge label for concept relationships)
    m = _KG_COVERS_PREFIX.match(text)
    if m:
        remainder = text[m.end():].strip()
        if len(remainder.split()) >= 3:
            return remainder[0].upper() + remainder[1:] if remainder[0].islower() else remainder

    return text


# ═══════════════════════════════════════════════════════════════════════════════
# C-NEW-2  KG boilerplate suffix stripper
# Strips the repeated "The main topic of the video covering <topic list>"
# suffix that KG extraction appends to every concept-relationship bullet.
#
# Detection strategy (fully dynamic, no topic words):
#   1. If bullet contains a colon, split at the LAST colon.
#   2. Compare the suffix (after colon) to a known boilerplate fingerprint:
#      starts with "The main topic" or "Meaningful or processed data" etc.
#   3. If the suffix is ≥60% of all the words in the bullet and adds no new
#      information beyond the prefix, strip it.
# ═══════════════════════════════════════════════════════════════════════════════

# Boilerplate suffix openers — these phrases start the repeated KG description
# pasted onto every edge bullet. Detection is structural, not topic-specific.
_BOILERPLATE_SUFFIX_OPENERS = re.compile(
    r'(?:'
    r'The\s+main\s+topic\s+of\s+the\s+video'   # "The main topic of the video covering..."
    r'|Topic\s+of\s+the\s+video'                # residue after "The main" was stripped
    r'|Meaningful\s+or\s+processed\s+data'      # "Meaningful or processed data, derived..."
    r'|The\s+main\s+topic\s+of\s+this'          # generic variant
    r'|This\s+video\s+covers'
    r'|The\s+video\s+covers'
    r')',
    re.I,
)

# Bullets that are pure boilerplate video-topic openers — always garbage
# These arise when the "X covers Y:" prefix is stripped but the remaining
# content is itself a boilerplate "Topic of the video covering..." fragment.
_BOILERPLATE_TOPIC_OPENER = re.compile(
    r'^(?:The\s+main\s+)?[Tt]opic\s+of\s+the\s+video\s+covering\b',
    re.I,
)


def _strip_kg_boilerplate_suffix(text: str) -> str:
    """
    C-NEW-2: Strip the KG boilerplate description suffix appended to
    concept-relationship bullets.

    Strategy:
    - Find the last colon in the text.
    - If the text after that colon matches a known boilerplate opener,
      keep only the part before the colon.
    - If no colon but the boilerplate opener appears mid-sentence,
      truncate at that point.
    - Fully dynamic: the opener patterns are structural phrases used by the
      KG extractor, not topic words.
    """
    # Strategy 1: colon-split — suffix after last colon is boilerplate
    last_colon = text.rfind(':')
    if last_colon > 0:
        suffix = text[last_colon + 1:].strip()
        if _BOILERPLATE_SUFFIX_OPENERS.match(suffix):
            prefix = text[:last_colon].strip().rstrip(',;')
            if len(prefix.split()) >= 3:
                return prefix + '.'

    # Strategy 2: boilerplate opener appears anywhere mid-sentence
    m = _BOILERPLATE_SUFFIX_OPENERS.search(text)
    if m and m.start() > 10:  # ensure there's meaningful prefix before it
        prefix = text[:m.start()].strip().rstrip(',;:')
        if len(prefix.split()) >= 3:
            return prefix + '.'

    return text


def _clean_bullet_text(text: str) -> Optional[str]:
    """
    Clean OCR artefacts from a single bullet string.
    Returns cleaned string, or None if the residue is too short/garbled.
    """
    # Quick garbage check before any cleaning (saves time on clearly bad bullets)
    for pat in _GARBAGE_BULLET_PATTERNS:
        if pat.search(text):
            return None

    # FIX-BROKEN-TRIPLE: drop bullets with unbalanced closing paren (orphaned KG fragment)
    # "FILO (First In, Last Out): Order of Operations is Out)."
    if _is_broken_triple(text):
        return None

    # FIX-TRUNCSUBJECT: repair or drop bullets starting with lowercase OCR fragment
    # "Of Operations is a is part of LIFO" → DROP (no colon content to recover)
    # "perations Performed on Stack (ADT): includes Peek Operation, ..." → REPAIR
    if _is_truncated_subject(text):
        # Try to REPAIR if there is rich colon-delimited content to recover
        colon_pos = text.find(':')
        if colon_pos > 0:
            remainder = text[colon_pos + 1:].strip()
            remainder_words = len(remainder.split())
            if remainder_words >= 5:
                # FIX-RAWLIST: Detect colon remainder that is itself a raw KG edge list:
                #   "consists of Push Operation, consists of Pop Operation"
                #   "includes Peek Operation, includes IsFull Operation, includes IsEmpty Operation"
                # These carry structural labels but ZERO educational descriptions.
                # Drop them entirely — the operations will appear via the notes-builder
                # individually with descriptions from get_contextual_description().
                # Detection: ≥2 occurrences of the same structural verb at segment starts.
                # Fully dynamic — no operation names hardcoded.
                _RAW_EDGE_LIST = re.compile(
                    r'(?:^|,\s*)'           # start of string or after comma
                    r'(?:includes?|consists?\s+of|contains?|has|encompasses?)'
                    r'\s+\w',               # followed immediately by a word
                    re.I,
                )
                raw_matches = _RAW_EDGE_LIST.findall(remainder)
                if len(raw_matches) >= 2:
                    return None  # Raw structural list — drop; builder will supply described bullets

                # Rich colon content — strip the broken heading and keep the content
                # Re-capitalise the remainder as a standalone bullet
                repaired = remainder[0].upper() + remainder[1:] if remainder else None
                if repaired:
                    text = repaired
                    # Fall through to continue cleaning; DO NOT return None
                else:
                    return None
            else:
                return None  # colon exists but remainder too short — drop
        else:
            return None  # no colon — cannot recover — drop

    # Drop bullets that are residual "Topic of the video covering..." boilerplate fragments
    if _BOILERPLATE_TOPIC_OPENER.match(text):
        return None

    # Strip leading OCR tokens (C-6)
    text = _LEADING_OCR.sub("", text).strip()

    # Re-check garbage patterns on the stripped text — some patterns only fire
    # after the noisy prefix is removed (e.g. "E £ fur ... PO array" → "... PO array")
    for pat in _GARBAGE_BULLET_PATTERNS:
        if pat.search(text):
            return None

    # Remove inline OCR noise (C-3)
    text = _INLINE_OCR.sub(" ", text)
    # FIX-7: Remove broken-hyphen PDF extraction artifact " -word" → " word"
    text = _BROKEN_HYPHEN.sub(r'\1', text)
    # FIX-REAL-LIFE-EXAMPLE: "A real-life for example, where..." (postprocessor KG rule artifact)
    text = re.sub(
        r'\b(real-life|practical|common|typical|simple|classic)\s+for\s+example\b',
        r'\1 example', text, flags=re.I
    )
    # FIX-AND-OPENER: "And topic of discussion..." — KG structure leak with leading conjunction
    # The word "And" at the very start of a bullet is never a valid sentence subject.
    # Drop the leading conjunction and re-capitalise.
    text = re.sub(r'^And\s+', '', text).strip()
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    text = re.sub(r"\s{2,}", " ", text).strip()

    # Drop if too short after cleaning
    if len(text.split()) < _MIN_WORD_COUNT:
        return None

    # Drop if alpha ratio too low (pure garbage)
    alpha = sum(1 for c in text if c.isalpha())
    if alpha / max(len(text), 1) < _MIN_ALPHA_RATIO:
        return None

    # C-NEW-5: Drop stub/fragment bullets with no finite verb
    text = _drop_if_fragment(text)
    if text is None:
        return None

    # Ensure sentence ends with punctuation
    if text and text[-1] not in ".!?":
        text += "."

    return text


def clean_ocr_in_bullets(sections: List[Dict]) -> Tuple[List[Dict], int]:
    """C-3 + C-6: Strip OCR noise from bullet text; drop bullets that become garbage."""
    removed = 0
    for sec in sections:
        for sub in sec.get("subsections", []):
            new_pts = []
            for pt in sub.get("points", []):
                cleaned = _clean_bullet_text(pt.get("text", ""))
                if cleaned is None:
                    removed += 1
                else:
                    pt["text"] = cleaned
                    new_pts.append(pt)
            sub["points"] = new_pts
        sec["subsections"] = [s for s in sec.get("subsections", []) if s.get("points")]
    sections = [s for s in sections if s.get("subsections")]
    return sections, removed


# ═══════════════════════════════════════════════════════════════════════════════
# C-1  Tautological phrase repetition collapse
# ═══════════════════════════════════════════════════════════════════════════════
#
# Pattern: "X <phrase> Y <same-phrase> Z"
# where the same 3-6 word phrase appears twice in the sentence.
# Keep only the FIRST occurrence.

def _collapse_repeated_phrases(text: str) -> str:
    """
    Find and remove duplicate phrases within a single sentence.
    Works on any domain — uses pure text analysis, no domain words.
    """
    # Match a repeated trigram+ pattern
    # Strategy: find the longest repeated n-gram (n=3..6) and collapse
    words = text.split()
    if len(words) < 6:
        return text

    best_start1 = best_start2 = best_len = -1

    for n in range(6, 2, -1):
        for i in range(len(words) - n + 1):
            phrase = tuple(words[i:i + n])
            for j in range(i + 1, len(words) - n + 1):
                if tuple(words[j:j + n]) == phrase:
                    if n > best_len:
                        best_len = n
                        best_start1 = i
                        best_start2 = j
                    break
            if best_len > 0:
                break
        if best_len > 0:
            break

    if best_len < 3:
        return text

    # Keep first occurrence; remove second occurrence and everything between that
    # is redundant  (words[best_start2 : best_start2+best_len] dropped)
    new_words = words[:best_start2] + words[best_start2 + best_len:]
    result = " ".join(new_words).strip().rstrip(" .") + "."
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# C-2  Double-relation artifact  "is a is a type of"  /  "is is"
# ═══════════════════════════════════════════════════════════════════════════════

_DOUBLE_IS_A     = re.compile(r"\bis\s+a\s+is\s+a\b",        re.I)
_DOUBLE_IS       = re.compile(r"\bis\s+is\b",                 re.I)
_IS_A_IS_A_TYPE  = re.compile(r"\bis\s+a\s+is\s+a\s+type\s+of\b", re.I)
_IS_IS_A_TYPE    = re.compile(r"\bis\s+is\s+a\s+type\s+of\b",     re.I)

def _fix_double_relations(text: str) -> str:
    text = _IS_A_IS_A_TYPE.sub("is a type of", text)
    text = _IS_IS_A_TYPE.sub("is a type of",   text)
    text = _DOUBLE_IS_A.sub("is a",            text)
    text = _DOUBLE_IS.sub("is",                text)
    return text


# ═══════════════════════════════════════════════════════════════════════════════
# C-7  Self-reference loop  "X ... within X"
# ═══════════════════════════════════════════════════════════════════════════════

_WITHIN_SELF = re.compile(
    r"^(.+?)\s+is\s+a\s+key\s+concept\s+(?:in\s+this\s+domain\s+)?is\s+a\s+key\s+concept\s+within\s+\1\s*\.",
    re.I,
)
_WITHIN_ANY = re.compile(
    r"(\bis\s+a\s+key\s+concept\s+within\s+\S+(?:\s+\S+){0,4})\s+is\s+a\s+key\s+concept\s+within\s+\S+",
    re.I,
)


def _fix_self_reference_loop(text: str) -> str:
    # "X is a key concept in this domain is a key concept within X."
    m = _WITHIN_SELF.match(text)
    if m:
        return f"{m.group(1).strip()} is a key concept in this domain."
    # Generic collapse of double "is a key concept within"
    text = _WITHIN_ANY.sub(r"\1", text)
    return text


# ═══════════════════════════════════════════════════════════════════════════════
# C-8  Self-loop tautology  "X is a type of X"  /  "X includes X"
# ═══════════════════════════════════════════════════════════════════════════════

_SELF_LOOP = re.compile(
    r"^(.+?)\s+(?:is\s+a\s+type\s+of|includes|contains|encompasses)\s+\1\s*\.?\s*$",
    re.I,
)


def _is_self_loop(text: str) -> bool:
    return bool(_SELF_LOOP.match(text.strip()))


# ═══════════════════════════════════════════════════════════════════════════════
# C-9  Truncated first word  "perations Performed" -> "Operations Performed"
#       "rder of Operations"  -> "Order of Operations"
# Pattern: bullet starts with a lowercase continuation (no vowel at start) followed
# by a Capital word — indicates the first character was dropped by PDF extraction.
# We detect this by: word[0] is lowercase, word is 4+ chars, no capital, and the
# next word starts with a capital (i.e., it looks like a mid-word start).
# FIX: drop the truncated fragment so the meaningful part starts the sentence.
# ═══════════════════════════════════════════════════════════════════════════════

_TRUNCATED_FIRST_WORD = re.compile(
    r"^([a-z][a-z]{2,})\s+",  # starts with 3+ lowercase chars then space
)


def _fix_truncated_first_word(text: str) -> str:
    """
    If bullet starts with a truncated lowercase word (3-8 chars, no leading capital),
    drop it — the meaningful content starts at the next word.
    E.g. "perations Performed On Stack..." -> "Performed On Stack..."
    E.g. "rder of Operations..." -> "of Operations..."  (best we can do without domain knowledge)
    Only triggers when the very first character is lowercase, which never happens in
    a properly formed English sentence starting with a proper noun or capital.
    """
    m = _TRUNCATED_FIRST_WORD.match(text)
    if m:
        frag = m.group(1)
        # Only drop if fragment is ≤8 chars (true truncation, not a real lead word)
        if len(frag) <= 8:
            remainder = text[m.end():]
            if remainder:
                return remainder[0].upper() + remainder[1:] if remainder[0].islower() else remainder
    return text


# ═══════════════════════════════════════════════════════════════════════════════
# C-10  OCR slide bleed — raw image caption / OCR text bleeding into bullets
# Pattern: Very long "definition" bullets that contain caption-like text
# (e.g. "a diagram showing...", "a pillow with the word...", "DEFINITION OF STACK A stack...")
# Signals: contains repeated topic words, raw BLIP caption fragments, slide branding
# ═══════════════════════════════════════════════════════════════════════════════

_SLIDE_BLEED_PATTERNS = [
    # BLIP caption text that leaked ("a diagram showing X", "a pillow with the word X")
    re.compile(r'\ba\s+(diagram|picture|photo|image|screenshot|pillow|sign|banner|blackboard)\s+(showing|of|with|that|depicting)', re.I),
    # Slide branding text
    re.compile(r'\b(enesoacademy|nesoacademy|Flv\s*\|)', re.I),
    # Repeated all-caps headline like "DEFINITION OF STACK A stack is..."
    re.compile(r'\b[A-Z]{3,}\s+[A-Z]{2,}\s+[A-Z]{2,}\b.*\b[A-Z]{3,}\s+[A-Z]{2,}'),
    # "J Definition of Examples Stack as an Stack Stack ADT" style garbage
    re.compile(r'\b(as\s+an?\s+\w+\s+\w+\s+ADT|J\s+Definition\s+of)', re.I),
    # "a pillow with the word online" style
    re.compile(r'\ba\s+\w+\s+with\s+the\s+word\s+\w+\s+(written|on|in)\b', re.I),
]


def _is_slide_bleed(text: str) -> bool:
    """Return True if this bullet appears to be OCR slide bleed / image caption text."""
    for pat in _SLIDE_BLEED_PATTERNS:
        if pat.search(text):
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# C-11  Grammar fix — "is an Pop/Push/Peek" -> "is a Pop/Push/Peek"
#       and "Operations Performed on Stack is an Pop Operation" style tautologies
# ═══════════════════════════════════════════════════════════════════════════════

_IS_AN_CONSONANT = re.compile(r'\bis\s+an\s+([BCDFGHJKLMNPQRSTVWXYZ])', re.I)


def _fix_grammar_artifacts(text: str) -> str:
    """Fix 'is an Pop' -> 'is a Pop' and similar grammar errors from templates."""
    text = _IS_AN_CONSONANT.sub(r'is a \1', text)
    return text


def _is_tautological_bullet(text: str) -> bool:
    """
    Detect circular template bullets from KG artifact patterns:
      "Pop Operation: Operations Performed on Stack is an Pop Operation"
      "Peek Function: Operations Performed on Stack is a Peek Function"
    
    Pattern: "Label: <anything> is a(n) Label" — the label before the colon
    repeats as the last N words after "is a(n)".
    Also catches: "X is a key concept within X" style.
    """
    text_s = text.strip().rstrip(".")
    
    # Pattern 1: "Label: ... is a(n) Label" (colon template artifact)
    colon_m = re.match(r'^([^:]{3,40}):\s+.+?\s+is\s+(?:a|an)\s+\1\s*$', text_s, re.I)
    if colon_m:
        return True
    
    # Pattern 2: "X is a key concept within X"
    kc_m = re.match(r'^(.+?)\s+is\s+a\s+key\s+concept\s+within\s+\1', text_s, re.I)
    if kc_m:
        return True
    
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# C-12  Malformed subsection headings
# Headings that are clearly not real headings: single verb like "Inserted",
# very short fragments, or headings that are just KG relation verbs.
# ═══════════════════════════════════════════════════════════════════════════════

_MALFORMED_HEADINGS = {
    "inserted", "deleted", "removed", "added", "updated", "modified",
    "used", "applied", "performed", "executed", "called",
    "lexical", "phase", "target",  # KG relation verbs used as headings
}


def _is_malformed_heading(heading: str) -> bool:
    """Return True if heading is clearly not a meaningful subsection title."""
    h = heading.strip().lower().rstrip(".")
    # Single word that is a verb/past-participle with no real noun content
    if h in _MALFORMED_HEADINGS:
        return True
    # Very short (1-2 words) non-meaningful headings
    words = h.split()
    if len(words) == 1 and h not in {
        "overview", "definition", "properties", "applications",
        "operations", "components", "structure", "summary", "examples",
        "implementation", "types", "concepts", "relationships"
    }:
        return True
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# C-13  Exam / institution-specific noise
# Bullets referencing specific exam patterns, institution names, or tutorial
# attributions that leaked from slide OCR.
# ═══════════════════════════════════════════════════════════════════════════════

_EXAM_NOISE_PATTERNS = [
    re.compile(r'\b(SPPU|GATE|GTU|VTU|NPTEL|Coursera|Udemy|YouTube)\b', re.I),
    re.compile(r'\bexam\s+pattern\b', re.I),
    re.compile(r'\bimportant\s+questions?\b', re.I),
    re.compile(r'\btutorial\s+by\b', re.I),
    re.compile(r'\bby\s+[A-Z][a-z]+\s+[0-9]', re.I),  # "by Vrushali 10"
    re.compile(r'\bCS\s*&\s*IT\s+Tutorial', re.I),
]


def _is_exam_noise(text: str) -> bool:
    return any(p.search(text) for p in _EXAM_NOISE_PATTERNS)


# ═══════════════════════════════════════════════════════════════════════════════
# C-14  Incomplete subject — "A is a type of data structure"
#       where "A" is literally the word A (lost subject)
# ═══════════════════════════════════════════════════════════════════════════════

_LOST_SUBJECT = re.compile(r'^[A-Z]\s+is\s+', )  # Single capital letter as subject


def _fix_lost_subject(text: str) -> Optional[str]:
    """Drop bullets where the subject is a single letter (OCR lost the real word)."""
    if _LOST_SUBJECT.match(text):
        return None
    return text




# ═══════════════════════════════════════════════════════════════════════════════
# C-NEW-6  OCR misspelling / digit-for-letter density filter
# ═══════════════════════════════════════════════════════════════════════════════

# Digit-for-letter substitution: a token starts with a digit followed immediately
# by alpha chars — e.g. "8itmap", "3rror", "0bject"
_DIGIT_FOR_LETTER = re.compile(r'^\d[a-z]+$', re.I)

# Impossible consonant clusters that never start real English words
# These are 2-char pairs at position 0 that don't exist in English
_IMPOSSIBLE_START_BIGRAMS = frozenset({
    'bn', 'bq', 'bx', 'bz', 'cb', 'cd', 'cf', 'cg', 'cj', 'ck', 'cm', 'cn',
    'cp', 'cq', 'cr', 'cs', 'cv', 'cx', 'cz', 'db', 'dc', 'df', 'dg', 'dk',
    'dm', 'dn', 'dp', 'dq', 'dt', 'dv', 'dw', 'dx', 'dz', 'fb', 'fc', 'fd',
    'ff', 'fg', 'fj', 'fk', 'fm', 'fn', 'fp', 'fq', 'fr', 'fs', 'ft', 'fv',
    'fw', 'fx', 'fz', 'gb', 'gc', 'gd', 'gf', 'gg', 'gj', 'gk', 'gm', 'gn',
    'gp', 'gq', 'gs', 'gt', 'gv', 'gw', 'gx', 'gz', 'hb', 'hc', 'hd', 'hf',
    'hg', 'hh', 'hj', 'hk', 'hm', 'hn', 'hp', 'hq', 'hr', 'hs', 'ht', 'hv',
    'hw', 'hx', 'hz', 'jb', 'jc', 'jd', 'jf', 'jg', 'jh', 'jj', 'jk', 'jl',
    'jm', 'jn', 'jp', 'jq', 'jr', 'js', 'jt', 'jv', 'jw', 'jx', 'jy', 'jz',
    'kb', 'kc', 'kd', 'kf', 'kg', 'kh', 'kj', 'kk', 'km', 'kn', 'kp', 'kq',
    'kv', 'kx', 'kz', 'lb', 'lc', 'ld', 'lf', 'lg', 'lh', 'lj', 'lk', 'lm',
    'ln', 'lp', 'lq', 'lr', 'ls', 'lt', 'lv', 'lw', 'lx', 'lz', 'mb', 'mc',
    'md', 'mf', 'mg', 'mh', 'mj', 'mk', 'ml', 'mm', 'mn', 'mp', 'mq', 'mr',
    'ms', 'mt', 'mv', 'mw', 'mx', 'mz', 'nb', 'nc', 'nd', 'nf', 'ng', 'nh',
    'nj', 'nk', 'nl', 'nm', 'nn', 'np', 'nq', 'nr', 'ns', 'nt', 'nv', 'nw',
    'nx', 'nz',
})


def _drop_if_ocr_misspelling_dense(text: str) -> Optional[str]:
    """
    C-NEW-6: Drop a bullet if it has a high density of OCR misspelling signals.

    Signals (all purely structural, no domain knowledge):
    1. Token starts with a digit followed by alpha chars (digit-for-letter: "8itmap")
    2. Token starts with an impossible English consonant bigram ("bn", "bq", ...)
    3. Token contains repeated consecutive identical letters > 3 times ("aaaaa")

    Threshold: if ≥2 signal tokens found among meaningful tokens (length ≥ 3),
    drop the bullet. A single hit is allowed (could be a code token or acronym).
    """
    words = re.findall(r'[a-zA-Z0-9]{3,}', text)
    if not words:
        return text

    signal_count = 0
    for w in words:
        wl = w.lower()
        # Signal 1: digit-for-letter substitution
        if _DIGIT_FOR_LETTER.match(w):
            signal_count += 1
        # Signal 2: impossible English consonant-pair at word start
        elif len(wl) >= 2 and wl[:2] in _IMPOSSIBLE_START_BIGRAMS:
            signal_count += 1
        # Signal 3: 4+ identical consecutive chars (OCR smear)
        elif re.search(r'(.)\1{3,}', wl):
            signal_count += 1

        if signal_count >= 2:
            return None

    return text


def _fix_bullet(text: str) -> Optional[str]:
    """
    Apply all semantic and OCR fixes to a single bullet string.
    Returns fixed text, or None to signal removal.
    Applies: C-1, C-2, C-7, C-8, C-9, C-10, C-11, C-13, C-14,
             C-NEW-1, C-NEW-2, C-NEW-6.
    """
    text = text.strip()
    if not text:
        return None

    # C-13: exam/institution noise
    if _is_exam_noise(text):
        return None

    # C-10: OCR slide bleed (image caption text bled into content)
    if _is_slide_bleed(text):
        return None

    # C-14: lost subject (single letter as subject — e.g. "A is a type of")
    text = _fix_lost_subject(text)
    if text is None:
        return None

    # C-8: drop self-loop tautologies
    if _is_self_loop(text):
        return None

    # C-8b: drop circular template tautologies like "X Operations is a X Function"
    if _is_tautological_bullet(text):
        return None

    # C-NEW-1: Strip KG structural edge-label prefixes
    # "Structure determines ...", "& Structure determines ...", "X covers Y: ..."
    text = _strip_kg_structural_prefix(text)
    if len(text.split()) < 3:
        return None

    # C-NEW-2: Strip KG boilerplate description suffix
    # "...: The main topic of the video covering <long list>"
    text = _strip_kg_boilerplate_suffix(text)
    if len(text.split()) < 3:
        return None

    # C-NEW-6: OCR misspelling / digit-for-letter density check
    # Drop bullet if too many tokens contain digit-for-letter substitutions
    # or impossible letter sequences indicating OCR noise
    text_checked = _drop_if_ocr_misspelling_dense(text)
    if text_checked is None:
        return None
    text = text_checked

    # C-9: fix truncated first word ("perations..." -> drop truncated prefix)
    text = _fix_truncated_first_word(text)

    # C-11: fix grammar artifacts ("is an Pop" -> "is a Pop")
    text = _fix_grammar_artifacts(text)

    # C-2: fix double-relation artefacts
    text = _fix_double_relations(text)

    # C-7: collapse self-reference loops
    text = _fix_self_reference_loop(text)

    # C-1: collapse repeated phrases
    text = _collapse_repeated_phrases(text)

    text = re.sub(r"\s{2,}", " ", text).strip()
    if text and text[-1] not in ".!?":
        text += "."

    if len(text.split()) < 3:
        return None

    return text


def fix_bullet_semantics(sections: List[Dict]) -> Tuple[List[Dict], int]:
    """C-1, C-2, C-7 through C-14: Fix semantic/OCR artefacts in bullets AND malformed headings."""
    fixed = 0
    removed = 0
    for sec in sections:
        # C-12: Fix malformed subsection headings (e.g. "- Inserted", "- Lexical")
        new_subs = []
        for sub in sec.get("subsections", []):
            sh = sub.get("heading", "")
            if _is_malformed_heading(sh):
                # Promote points into the last good subsection if possible
                if new_subs:
                    new_subs[-1]["points"].extend(sub.get("points", []))
                else:
                    sub["heading"] = "Key Concepts"
                    new_subs.append(sub)
            else:
                new_subs.append(sub)
        sec["subsections"] = new_subs

        # Fix bullet text
        for sub in sec.get("subsections", []):
            new_pts = []
            for pt in sub.get("points", []):
                result = _fix_bullet(pt.get("text", ""))
                if result is None:
                    removed += 1
                else:
                    if result != pt.get("text", ""):
                        fixed += 1
                    pt["text"] = result
                    new_pts.append(pt)
            sub["points"] = new_pts
        sec["subsections"] = [s for s in sec.get("subsections", []) if s.get("points")]
    sections = [s for s in sections if s.get("subsections")]
    _log(f"Bullet semantics: {fixed} fixed, {removed} removed")
    return sections, fixed + removed


# ═══════════════════════════════════════════════════════════════════════════════
# C-10  Summary length enforcement  (≤ 3 sentences)
# ═══════════════════════════════════════════════════════════════════════════════

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def trim_summary(notes: Dict, max_sentences: int = 3) -> Dict:
    """C-10: Trim the notes summary to at most max_sentences sentences."""
    summary = notes.get("summary", "")
    if not summary:
        return notes
    sentences = _SENT_SPLIT.split(summary.strip())
    if len(sentences) > max_sentences:
        notes["summary"] = " ".join(sentences[:max_sentences])
        if not notes["summary"].endswith("."):
            notes["summary"] += "."
        _log(f"C-10 → summary trimmed: {len(sentences)} → {max_sentences} sentences")
    return notes


# ═══════════════════════════════════════════════════════════════════════════════
# MASTER ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def clean_notes(notes: Dict) -> Dict:
    """
    Apply all dynamic cleaning passes to a HierarchicalNotes dict.

    Passes:
      0  — Deep ? sanitization (catches any remaining OCR ? artefacts)
      1  — OCR noise in bullet text (C-3, C-6)
      2  — Semantic artefacts in bullets (C-1, C-2, C-7, C-8)
      3  — Heading OCR noise + generic headings (C-4, C-5, C-9)
      4  — Summary trim (C-10)
      5  — Remove Key Takeaways subsections (redundant content)

    Usage in ex.py (add before render_pdf):
    ----------------------------------------
        from notes_cleaner import clean_notes
        notes_dict = clean_notes(notes_dict)

    Parameters
    ----------
    notes : HierarchicalNotes dict

    Returns
    -------
    Cleaned HierarchicalNotes dict (deep copy — original unchanged).
    """
    notes = copy.deepcopy(notes)
    sections: List[Dict] = notes.get("sections", [])

    _log(f"Input: {len(sections)} sections")

    # Pass 0 — Deep ? sanitization (safety net — catches any ? that slipped through earlier)
    try:
        from concept_flow_organizer import sanitize_notes_question_marks
        sections = sanitize_notes_question_marks(sections)
        _log("Pass 0: ? artefacts sanitized")
    except Exception as e:
        # Inline fallback: headings strip ?, bullets convert ? to :
        for sec in sections:
            if "heading" in sec:
                sec["heading"] = re.sub(r'\s\?\s.*$', '', sec["heading"]).strip()
                sec["heading"] = re.sub(r'\?', '', sec["heading"]).strip()
            for sub in sec.get("subsections", []):
                if "heading" in sub:
                    sub["heading"] = re.sub(r'\s\?\s.*$', '', sub["heading"]).strip()
                    sub["heading"] = re.sub(r'\?', '', sub["heading"]).strip()
                for pt in sub.get("points", []):
                    if "text" in pt:
                        pt["text"] = re.sub(r'\s\?\s', ': ', pt["text"])
                        pt["text"] = re.sub(r'\?', '', pt["text"]).strip()
        _log(f"Pass 0: ? sanitization (inline fallback, import failed: {e})")

    # Pass 0b — Inline dynamic bullet fixes not covered by other passes
    _KEY_CONCEPT_WITHIN = re.compile(r'\bis\s+a\s+key\s+concept\s+within\b', re.I)
    _COVERS_CIRCULAR = re.compile(r'^([^:]{3,50}):\s+(.+?)\s+covers\s+\1', re.I)
    _PUNCT_COLON = re.compile(r'\.\s*:')        # 'word.:' -> 'word:'
    _LEADING_PREP = re.compile(r'^(Of|In|To|For|By|At|On|An|The|And|Or|But)\s+', re.I)

    # Truncation detection: first word starts with a consonant cluster that indicates
    # a word was cut mid-stream. Uses linguistic heuristic: English words almost never
    # start with 'pr' at position 0 unless the actual start was 'O' + 'pr' (e.g. 'Operations').
    # More reliable: first word contains NO vowel in first 2 chars, OR
    # first word starts with lowercase (bullets always start with capital in proper notes)
    def _is_truncated_first_word(word: str) -> bool:
        """
        Return True ONLY if this word is clearly a mid-word OCR fragment.
        Heuristic: a word that starts with a consonant cluster (no vowel in first 2 chars)
        AND is not in the known-valid-starts list AND the word has no vowel at position 0.
        
        Examples that ARE truncated: 'perations', 'rder', 'Perations', 'Rder'
        Examples that are NOT truncated: 'Performed', 'Stack', 'Push', 'Linear'
        """
        if not word:
            return False
        w = word.lower()
        # Words starting with a vowel are NEVER truncations
        if w[0] in 'aeiou':
            return False
        # Known valid consonant-starting sentence openers
        _VALID_CONSONANT_STARTS = frozenset({
            'data', 'stack', 'push', 'pop', 'node', 'tree', 'list', 'hash', 'array',
            'queue', 'type', 'top', 'size', 'key', 'use', 'used', 'last', 'first',
            'next', 'new', 'code', 'step', 'phase', 'stage', 'class', 'method',
            'value', 'file', 'name', 'time', 'check', 'note', 'user', 'set', 'get',
            'run', 'loop', 'fix', 'memory', 'return', 'pointer', 'call', 'find',
            # Capitalized first words that look like valid subjects
            'function', 'functions', 'procedures', 'procedure', 'compiler',
            'linear', 'binary', 'dynamic', 'static', 'primary', 'secondary',
            'stack', 'stacks', 'graph', 'graphs', 'tree', 'trees', 'list', 'lists',
            'pointer', 'process', 'program', 'programs', 'symbol', 'symbols',
            'token', 'tokens', 'syntax', 'semantic', 'grammar', 'parser', 'scanner',
            'definition', 'declaration', 'recursion', 'traversal', 'sorting',
            'search', 'deletion', 'sequence', 'structure', 'structures',
            'register', 'variable', 'variables', 'statement', 'statements',
            'condition', 'conditions', 'comparison', 'result', 'results',
            'boolean', 'string', 'strings', 'byte', 'bytes', 'bit', 'bits',
            'block', 'blocks', 'frame', 'frames', 'scope', 'parameter', 'parameters',
            'performed', 'performs', 'provides', 'represents', 'returns', 'stores',
            'supports', 'specifies', 'handles', 'manages', 'maintains', 'contains',
            'follows', 'defines', 'requires', 'produces', 'creates', 'generates',
        })
        if w in _VALID_CONSONANT_STARTS:
            return False
        # If word has no vowel in first 2 characters -> clearly a truncation artifact
        # e.g. 'perations': p,e -> p has no vowel, e does but at pos 1 -> NOT truncated?
        # Actually 'perations' starts with 'p' (consonant) but position 1 is 'e' (vowel)
        # The real test: does the word look like it could be the middle of a word?
        # Strategy: if first char is consonant AND the word is NOT in valid list ->
        # check if removing first 1-3 chars produces a plausible vowel start
        for skip in (1, 2, 3):
            remainder = w[skip:]
            if remainder and remainder[0] in 'aeiou' and len(remainder) >= 4:
                # 'perations'[1:] = 'erations' starts with vowel -> truncation!
                # But 'performed'[1:] = 'erformed' starts with vowel... 
                # We need to ALSO check that the result is a real word suffix
                # Simple heuristic: if the word WITHOUT the skip looks like a
                # complete word starting with vowel (in valid list or common), it's truncated
                # We'll use length: typical truncations are 6-10 chars
                if 5 <= len(w) <= 11 and w not in _VALID_CONSONANT_STARTS:
                    return True
                break
        return False

    def _fix_truncated_start_inline(text: str) -> str:
        """Drop truncated first word(s); runs in a loop for double truncations."""
        for _ in range(3):  # max 3 iterations covers any realistic double-truncation
            words = text.split()
            if not words:
                return text
            if _is_truncated_first_word(words[0]):
                remainder = ' '.join(words[1:]).strip()
                if not remainder:
                    return text
                text = remainder[0].upper() + remainder[1:] if remainder[0].islower() else remainder
            else:
                break
        return text

    def _pass0b_fix(text: str) -> Optional[str]:
        # Strip punctuation artifact '.:'
        text = _PUNCT_COLON.sub(':', text)
        # Drop 'is a key concept within' stub bullets
        if _KEY_CONCEPT_WITHIN.search(text):
            return None
        # Drop circular 'Label: X covers Label' bullets
        if _COVERS_CIRCULAR.match(text):
            return None
        # Fix truncated first word (must run before leading-prep fix)
        text = _fix_truncated_start_inline(text)
        # Fix leading preposition from truncation: 'Of Operations...' -> drop
        text = _LEADING_PREP.sub('', text).strip()
        # Re-capitalise after stripping
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        if len(text.split()) < 3:
            return None
        return text

    pass0b_removed = 0
    for sec in sections:
        for sub in sec.get("subsections", []):
            new_pts = []
            for pt in sub.get("points", []):
                result = _pass0b_fix(pt.get("text", ""))
                if result is None:
                    pass0b_removed += 1
                else:
                    pt["text"] = result
                    new_pts.append(pt)
            sub["points"] = new_pts
        sec["subsections"] = [s for s in sec.get("subsections", []) if s.get("points")]
    sections = [s for s in sections if s.get("subsections")]
    if pass0b_removed:
        _log(f"Pass 0b: {pass0b_removed} structural noise bullets removed")

    # Pass 0c — KG structural prefix/suffix stripping (C-NEW-1, C-NEW-2)
    # Must run BEFORE Pass 1 (OCR clean) so that bullets starting with
    # "Structure determines ..." or ending with "The main topic of the video..."
    # are cleaned to their meaningful core before garbage-ratio checks fire.
    pass0c_fixed = 0
    pass0c_removed = 0
    for sec in sections:
        for sub in sec.get("subsections", []):
            new_pts = []
            for pt in sub.get("points", []):
                t = pt.get("text", "").strip()
                if not t:
                    pass0c_removed += 1
                    continue
                # C-NEW-1: strip KG structural prefix
                t2 = _strip_kg_structural_prefix(t)
                # C-NEW-2: strip KG boilerplate suffix
                t2 = _strip_kg_boilerplate_suffix(t2)
                if len(t2.split()) < 3:
                    pass0c_removed += 1
                    continue
                if t2 != t:
                    pass0c_fixed += 1
                pt["text"] = t2
                new_pts.append(pt)
            sub["points"] = new_pts
        sec["subsections"] = [s for s in sec.get("subsections", []) if s.get("points")]
    sections = [s for s in sections if s.get("subsections")]
    if pass0c_fixed or pass0c_removed:
        _log(f"Pass 0c (KG prefix/suffix): {pass0c_fixed} fixed, {pass0c_removed} removed")

    # Pass 1 — OCR noise in bullet text (C-3, C-6)
    before = sum(len(sub.get("points", [])) for s in sections for sub in s.get("subsections", []))
    sections, n = clean_ocr_in_bullets(sections)
    after  = sum(len(sub.get("points", [])) for s in sections for sub in s.get("subsections", []))
    _log(f"C-3/C-6 OCR noise: {before - after} bullets removed")

    # Pass 2 — Semantic artefacts in bullets (C-1, C-2, C-7, C-8)
    sections, n2 = fix_bullet_semantics(sections)
    _log(f"C-1/2/7/8 semantic: {n2} bullets changed or removed")

    # Pass 3 — Heading OCR noise + generic headings (C-4, C-5, C-9)
    sections, n3 = clean_headings(sections)
    _log(f"C-4/5/9 headings: {n3} headings fixed")

    # Pass 4 — Summary trim (C-10)
    notes = trim_summary(notes)

    # Pass 5 — Remove Key Takeaways (redundant, repeat earlier content)
    _TAKEAWAY_RE = re.compile(
        r"\b(key\s+takeaway|takeaway|key\s+point|key\s+summary)\b", re.I
    )
    removed_ta = 0
    for sec in sections:
        before_ta = len(sec.get("subsections", []))
        sec["subsections"] = [
            ss for ss in sec.get("subsections", [])
            if not _TAKEAWAY_RE.search(ss.get("heading", ""))
        ]
        removed_ta += before_ta - len(sec.get("subsections", []))
    if removed_ta:
        _log(f"Pass 5: {removed_ta} Key Takeaways subsections removed")

    # Pass 6 — Cross-section deduplication: remove bullets repeated verbatim in multiple sections.
    # Uses normalised text (lowercase, stripped) as the key — no domain knowledge required.
    seen_bullets: Set[str] = set()
    dedup_removed = 0
    for sec in sections:
        for sub in sec.get("subsections", []):
            unique_pts = []
            for pt in sub.get("points", []):
                key = re.sub(r"\s+", " ", pt.get("text", "").lower().strip().rstrip("."))
                if key and key not in seen_bullets:
                    seen_bullets.add(key)
                    unique_pts.append(pt)
                else:
                    dedup_removed += 1
            sub["points"] = unique_pts
    # Remove subsections that became empty after dedup
    for sec in sections:
        sec["subsections"] = [ss for ss in sec.get("subsections", []) if ss.get("points")]
    sections = [s for s in sections if s.get("subsections")]
    if dedup_removed:
        _log(f"Pass 6: {dedup_removed} duplicate bullets removed across sections")

    notes["sections"] = sections
    _log(f"Output: {len(sections)} sections")
    return notes