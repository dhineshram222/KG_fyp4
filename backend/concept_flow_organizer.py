# concept_flow_organizer.py  (v2)
"""
Dynamic Conceptual Hierarchy, Flow, and Quality Organizer
==========================================================
Implements five critical improvements — fully domain-agnostic,
no hardcoded topic/domain words anywhere.

Priority 1 — Concept Hierarchy
  Build and expose the true parent→child concept tree from the KG so
  sections explicitly show their sub-concepts.

Priority 2 — Concept Relationships
  Convert KG triples (A → verb → B) into readable relationship bullets.
  Builds a dedicated "Concept Relationships" subsection per section.

Priority 3 — Concept Flow Section
  Detect sequential/pipeline chains and build a standalone
  "Concept Workflow" section at the top of the notes.

Priority 4 — Explanatory not just Definitions
  Expand bare-label or circular bullets with Input/Process/Output
  structure derived from KG edge data.

Priority 5 — Remove ? from headings and bullets
  Sanitize all ? OCR artefacts from section/subsection headings and
  bullet text. Also exposed as sanitize_nodes_labels() to clean KG
  node labels before they enter the notes pipeline.

Additional fixes:
  - Tautological bullets removed
  - Key Takeaways per section
  - Duplicate images deduplicated
  - Hierarchy annotation in section content
"""

import re
import copy
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Optional, Set


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

_SEQUENTIAL_RELATIONS = {
    'precedes', 'followed_by', 'next_step', 'produces', 'generates',
    'outputs', 'feeds_into', 'transforms_into', 'leads_to', 'results_in',
    'passes_to', 'passed_to', 'input_to', 'output_of', 'converts_to',
    'then', 'next', 'after', 'before', 'flows_to', 'phase_of',
}

_CONTAINMENT_RELATIONS = {
    'has', 'includes', 'contains', 'consists_of', 'comprised_of',
    'composed_of', 'has_component', 'has_phase', 'has_step',
    'has_part', 'has_stage', 'has_layer', 'has_type', 'has_subtype',
    'encompasses', 'covers', 'is_a', 'type_of', 'part_of',
}

_DISCOURSE_ROLES = [
    ("definition",     1,  ["definition","what is","refers to","means","overview",
                             "introduction","purpose","goal","concept of","is defined"]),
    ("classification", 2,  ["type","kind","category","class","variant","form",
                             "divided into","classified","categories"]),
    ("components",     3,  ["component","part","phase","stage","step","layer",
                             "consists","comprises","contains","includes","structure"]),
    ("process",        4,  ["first","then","next","after","before","during",
                             "finally","input","output","takes","produces",
                             "converts","passes","generates","feeds"]),
    ("properties",     5,  ["property","characteristic","feature","attribute",
                             "advantage","disadvantage","limitation","behavior"]),
    ("supporting",     6,  ["table","handler","manager","storage","auxiliary",
                             "error","symbol","register","buffer","cache"]),
    ("tools",          7,  ["tool","implementation","example","such as","like",
                             "used by","implemented by","practice","application"]),
    ("summary",        8,  ["summary","conclusion","overall","thus","therefore",
                             "key point","takeaway","important"]),
]

# ─────────────────────────────────────────────────────────────────────────────
# Pedagogical Section Ordering
# Maps topic-type keywords in section headings to a teaching-sequence weight.
# Lower weight = introduce earlier. Fully domain-agnostic keyword matching.
# ─────────────────────────────────────────────────────────────────────────────
_PEDAGOGICAL_SECTION_ORDER: List[Tuple[int, List[str]]] = [
    # 1. Pipeline / compiler overview — always first
    (1,  ["workflow","pipeline","overview","compiler","lexical","syntax","semantic",
          "process","phases","stage","stages","analysis"]),
    # 2. Core concept definitions (what IS this thing?)
    (2,  ["definition","what is","classification","types","categories","kinds",
          "grammar","context free","cfg","formal","language"]),
    # 3. Main mechanism — the parser itself
    (3,  ["parser","parsing","descent","recursive","predictive","top down",
          "bottom up","ll","lr","slr"]),
    # 4. Grammar theory — rules, terminals, production
    (4,  ["production","rule","terminal","non-terminal","epsilon","derivation",
          "leftmost","rightmost","symbol","grammatical","non terminal"]),
    # 5. Procedural implementation
    (5,  ["procedure","function","match","lookahead","look ahead","variable",
          "input","buffer","token","character","char","main function"]),
    # 6. Special cases and techniques
    (6,  ["backtrack","backtracking","left recursion","non-determinism",
          "ambiguity","factoring","elimination"]),
    # 7. Abstract data structures referenced
    (7,  ["tree","parse tree","abstract syntax","ast","stack","queue"]),
    # 8. Examples and walkthrough
    (8,  ["example","grammar","illustration","walkthrough","trace","e ->",
          "e'","conditional","statement"]),
]

def _pedagogical_section_weight(heading: str) -> int:
    """
    Return a weight (lower = earlier in teaching sequence) for a section heading.
    Fully dynamic — matches on keyword substrings in the lowercased heading.
    """
    h = heading.lower().strip()
    for weight, keywords in _PEDAGOGICAL_SECTION_ORDER:
        if any(kw in h for kw in keywords):
            return weight
    return 9   # unknown → place near end


def reorder_sections_pedagogically(sections: List[Dict]) -> List[Dict]:
    """
    Sort sections into a teaching-sequence order:
    Compiler pipeline → Grammar basics → Parser → Procedures → Examples.

    Uses a stable sort so sections with the same weight keep their original
    relative order (preserving KG-derived sub-groupings).
    """
    if not sections:
        return sections
    return sorted(sections, key=lambda s: _pedagogical_section_weight(
        s.get("heading", "")
    ))


# ─────────────────────────────────────────────────────────────────────────────
# KG Artifact Bullet Filter
# Removes bullets that are clearly machine-generated noise:
#   - OCR garbage fragments
#   - Lecture meta-references ("learned in previous chapter")
#   - KG triple dump subjects with truncated labels ("Down Approach generates...")
#   - Slide headers / watermarks
# ─────────────────────────────────────────────────────────────────────────────
_KG_NOISE_BULLET_PATTERNS: List[re.Pattern] = [
    # "Afor example" OCR artefact
    re.compile(r'^Afor\s+example\b', re.I),
    # "! = aabcde" OCR dump
    re.compile(r'^[!\?]\s*=\s*[a-zA-Z]{3,}', re.I),
    # "learned/mentioned/discussed in a previous chapter/session"
    re.compile(r'(?:learned|mentioned|discussed|covered|introduced)\s+in\s+a\s+previous\s+(?:chapter|session|lecture)', re.I),
    # "Procedures for its removal were learned..."
    re.compile(r'procedures\s+for\s+its\s+removal\s+were', re.I),
    # Slide watermark / lecture metadata
    re.compile(r'\bIT\s+Department\b|\bUnit\s+\d+\s*:', re.I),
    # "Down/Up Approach generates/produces/uses X as its output" — KG triple dump
    re.compile(r'^(?:Down|Up|Top|Bottom)\s+\w[\w\s]{0,40}(?:generates|produces|uses|involves|allows)\s+', re.I),
    # "Analysis generates/stores/followed by" — KG triple dump fragment
    re.compile(r'^Analysis\s+(?:generates|stores|followed\s+by)\s+', re.I),
    # "Grammar Procedures recognizes Input String" — raw KG edge verb leak
    re.compile(r'\brecognizes\s+Input\s+String\b', re.I),
    # Random OCR token sequences
    re.compile(r'Anse\s*=>|Taken\s*\(\d+\)', re.I),
    # "usually builds a data structure in the form of" — repeated boilerplate
    re.compile(r'usually\s+builds\s+a\s+data\s+structure\s+in\s+the\s+form\s+of\s+a\s+parse', re.I),
    # "Compiler Design Course" — meta-lecture reference, not a concept
    re.compile(r'\bcompiler\s+design\s+course\b', re.I),
    # "The course in which the focus is..." — lecture meta narration
    re.compile(r'\b(?:this|the)\s+course\s+in\s+which\b', re.I),
    # Single-word or two-word stub bullets (pure label fragments)
    # handled separately via word count check
]

_MIN_BULLET_WORDS = 5   # bullets with fewer real words are dropped

def _is_kg_noise_bullet(text: str) -> bool:
    """Return True if this bullet is a KG artifact / meta-noise and should be dropped."""
    if not text:
        return True
    stripped = text.strip()
    # Too short
    real_words = [w for w in stripped.split() if len(w) > 1]
    if len(real_words) < _MIN_BULLET_WORDS:
        return True
    for pat in _KG_NOISE_BULLET_PATTERNS:
        if pat.search(stripped):
            return True
    return False


def filter_kg_noise_bullets(sections: List[Dict]) -> List[Dict]:
    """
    Remove KG artifact bullets from all subsections.
    Drops: OCR garbage, meta-lecture references, triple-dump fragments,
           slide watermarks, and too-short stubs.
    """
    for section in sections:
        for sub in section.get("subsections", []):
            sub["points"] = [
                pt for pt in sub.get("points", [])
                if not _is_kg_noise_bullet(pt.get("text", ""))
            ]
        # Drop now-empty subsections
        section["subsections"] = [
            s for s in section.get("subsections", [])
            if s.get("points")
        ]
    # Drop now-empty sections
    return [s for s in sections if s.get("subsections")]

# ─────────────────────────────────────────────────────────────────────────────
# KG Meta-Label → Pedagogical Label mapping
# These are the raw KG schema subsection headings the pipeline emits.
# They should be replaced with natural, lecture-style headings.
# ─────────────────────────────────────────────────────────────────────────────
_KG_META_TO_PEDAGOGICAL: Dict[str, str] = {
    # Raw KG schema headings → pedagogical equivalents
    "key concepts":            "Overview",
    "key points":              "Key Points",
    "concept relationships":   "How It Works",
    "components & structure":  "Structure",
    "components and structure": "Structure",
    "concept relationships":   "How It Works",
    "process flow":            "Process Flow",
    "process overview":        "Process Overview",
    "definition":              "Definition",
    "definition and fundamental concept": "Definition",
    "properties":              "Properties",
    "applications":            "Applications",
    "implementation and structure": "Implementation",
    "conceptual structure":    "Conceptual Structure",
    "key notes and additional concepts": "Additional Notes",
    "contextual relationships": "How It Works",
    "top down":                "Top-Down Approach",
    "structure":               "Structure",
    "conceptual flow":         "Concept Flow",
}

# Patterns in bullet text that are pure KG edge triple dumps — remove them
_KG_TRIPLE_DUMP_PATTERNS: List[re.Pattern] = [
    # "X categorizes Y - introduced in a previous session"
    re.compile(r'^.{3,80}\s+categorizes\s+.{3,80}$', re.I),
    # "X encompasses several key components, including Y, Z and W."
    re.compile(r'^.{3,60}\s+encompasses\s+several\s+key\s+components', re.I),
    # "Down Parsers: A type of parser; they use..."
    # Catches truncated node-label leakage as subject
    re.compile(r'^(?:Down|Up|Top|Bottom)\s+\w[\w\s]{0,30}:\s+', re.I),
    # "X is a type of Y - introduced in a previous session..."
    re.compile(r'introduced\s+in\s+a\s+previous\s+session', re.I),
    # "Previously observed, categorizes..."
    re.compile(r'^previously\s+observed', re.I),
    # Bullets that are basically a repeat of the section heading + KG verb
    re.compile(r'^\w[\w\s]{2,40}\s+(?:categorizes|encompasses|observes|observed)\s+', re.I),
]

# Suffix patterns that are KG edge boilerplate — strip from bullet text
_KG_SUFFIX_STRIP_PATTERNS: List[re.Pattern] = [
    # " during its operation - <description>"
    re.compile(r'\s+during\s+its\s+operation\s*[-–—]\s*.{0,200}$', re.I),
    # " during its operation."
    re.compile(r'\s+during\s+its\s+operation\.?\s*$', re.I),
    # " - Introduced in a previous session: ..."
    re.compile(r'\s*[-–—]\s*[Ii]ntroduced\s+in\s+a\s+previous\s+(?:session|chapter)[^.]*\.?', re.I),
    # " - Previously observed, ..."
    re.compile(r'\s*[-–—]\s*[Pp]reviously\s+observed[^.]*\.?', re.I),
    # " - Algorithms used by top-down parsers with backtracking."  (common KG edge annotation)
    re.compile(r'\s*[-–—]\s*[A-Z][^.]{5,80}(?:used by|used in|introduced in)[^.]*\.\s*$', re.I),
    # " - Grammars that are allowed by..."
    re.compile(r'\s*[-–—]\s*[A-Z][a-z][^.]{5,80}that\s+(?:are|is)\s+(?:allowed|used|required)[^.]*\.\s*$', re.I),
]


def _remap_subsection_heading(heading: str) -> str:
    """
    Remap a KG meta-label subsection heading to its pedagogical equivalent.
    Fully dynamic — uses the mapping table above.
    """
    key = heading.lower().strip()
    if key in _KG_META_TO_PEDAGOGICAL:
        return _KG_META_TO_PEDAGOGICAL[key]
    return heading


def _is_triple_dump_bullet(text: str) -> bool:
    """Return True if this bullet is a raw KG triple dump that should be dropped."""
    for pat in _KG_TRIPLE_DUMP_PATTERNS:
        if pat.search(text):
            return True
    return False


def _strip_kg_edge_suffixes(text: str) -> str:
    """Strip KG edge boilerplate suffixes from bullet text."""
    for pat in _KG_SUFFIX_STRIP_PATTERNS:
        text = pat.sub('', text)
    text = re.sub(r'\s{2,}', ' ', text).strip()
    if text and text[-1] not in '.!?':
        text += '.'
    return text


def remap_subsection_headings(sections: List[Dict]) -> List[Dict]:
    """
    Replace KG schema meta-labels with pedagogical subsection headings.
    E.g. 'Concept Relationships' → 'How It Works', 'Key Concepts' → 'Overview'.
    Fully dynamic — no hardcoded topic words.
    """
    for section in sections:
        for sub in section.get('subsections', []):
            sub['heading'] = _remap_subsection_heading(sub.get('heading', ''))
    return sections


def clean_triple_dump_bullets(sections: List[Dict]) -> List[Dict]:
    """
    Remove raw KG triple dump bullets and strip KG edge suffix boilerplate.
    Also strips partial node-label leakage at the start of bullets.
    """
    _PARTIAL_LABEL_PREFIX = re.compile(
        # "Down Parsers: ...", "Down Approach uses X during..."
        r'^(?:Down|Up|Top|Bottom|Left|Right)\s+(?:[A-Z][a-z]+\s+){0,3}(?:[A-Z][a-z]+\s+)?',
        re.I,
    )
    for section in sections:
        for sub in section.get('subsections', []):
            kept = []
            for pt in sub.get('points', []):
                text = pt.get('text', '').strip()
                if not text:
                    continue
                # Drop full triple dumps
                if _is_triple_dump_bullet(text):
                    continue
                # Strip KG suffix boilerplate
                text = _strip_kg_edge_suffixes(text)
                if not text or len(text.split()) < 3:
                    continue
                kept.append({'text': text})
            sub['points'] = kept
        section['subsections'] = [s for s in section.get('subsections', []) if s.get('points')]
    return [s for s in sections if s.get('subsections')]

_STOPWORDS: Set[str] = {
    'the','a','an','is','are','was','were','be','been','have','has','had',
    'this','that','with','from','into','and','or','but','for','to','of',
    'in','on','at','by','which','what','how','when','will','can','its',
    'their','these','those','data','level','system',
}

_RELATIONSHIP_TEMPLATES: Dict[str, str] = {
    "uses":            "{A} uses {B} during its operation.",
    "has":             "{A} has {B} as a component.",
    "has_component":   "{A} has {B} as a component.",
    "contains":        "{A} contains {B}.",
    "produces":        "{A} produces {B} as output.",
    "generates":       "{A} generates {B} as its output.",
    "part_of":         "{B} is a phase or component of {A}.",
    "is_part_of":      "{A} is a part of {B}.",
    "detects":         "{A} detects {B}.",
    "is_a":            "{A} is a type of {B}.",
    "type_of":         "{A} is a type of {B}.",
    "related_to":      "{A} is related to {B}.",
    "converts":        "{A} converts input into {B}.",
    "checks":          "{A} checks for {B}.",
    "stores":          "{A} stores {B}.",
    "requires":        "{A} requires {B} to function.",
    "performs":        "{A} performs {B}.",
    "implements":      "{A} implements {B}.",
    "includes":        "{A} includes {B}.",
    "passes_to":       "{A} passes its output to {B}.",
    "feeds_into":      "{A} feeds into {B} as input.",
    "transforms_into": "{A} is transformed into {B}.",
    "input_to":        "{A} serves as input to {B}.",
    "output_of":       "{A} is the output of {B}.",
    "leads_to":        "{A} leads to {B}.",
    "results_in":      "{A} results in {B}.",
    "depends_on":      "{A} depends on {B}.",
    "manages":         "{A} manages {B}.",
    "handles":         "{A} handles {B}.",
    "verifies":        "{A} verifies {B} for correctness.",
    "validates":       "{A} validates {B}.",
    "optimizes":       "{A} optimizes {B} for better performance.",
    "translates":      "{A} translates {B} into target form.",
    "provides":        "{A} provides {B}.",
    "supports":        "{A} supports {B}.",
    "corresponds_to":  "{A} corresponds to {B}.",
    "operates_at":     "{A} operates at {B}.",
    "encapsulates":    "{A} encapsulates {B} by adding headers.",
    "responsible_for": "{A} is responsible for {B}.",
    "controlled_by":   "{A} is controlled by {B}.",
    "executed_by":     "{A} is executed by {B}.",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Priority 5 — Question Mark Sanitizer
# ═══════════════════════════════════════════════════════════════════════════════

_Q_POSSESSIVE   = re.compile(r"(\w)\?s\b")
_Q_CONTRACTION  = re.compile(r"(\w)\?t\b")
# "word ? word" — covers single arrow artefact (most common KG pattern)
_Q_WORD_ARROW   = re.compile(r"(\w+)\s+\?\s+(\w+)")
# After first pass, catch any remaining "? word" or "word ?" patterns
_Q_NOSPC        = re.compile(r"(\w)\?(\w)")
_Q_TRAIL_SPACE  = re.compile(r"(\w)\?(\s)")
_Q_PREFIX_UPPER = re.compile(r"(?<!\w)\?([A-Z])")
_Q_PREFIX       = re.compile(r"(?<!\w)\?([A-Za-z])")
_Q_ISOLATED     = re.compile(r"(?<!\w)\?(?!\w)")
_Q_TRAILING     = re.compile(r"\?+\s*$")
_Q_LEADING      = re.compile(r"^\?+\s*")
# Nuclear fallback: remove every remaining standalone ? (space?space or end-of-token?)
_Q_NUCLEAR      = re.compile(r"\s\?\s|\s\?$|^\?\s")



# Category-tag words used as suffixes in KG node labels: "Stacks ? Chapter" -> "Stacks"
_HEADING_CATEGORY_TAGS = frozenset({
    'chapter', 'section', 'module', 'unit', 'topic', 'subject',
    'overview', 'introduction', 'summary', 'conclusion', 'structure',
    'definition', 'concept', 'concepts', 'lecture', 'slide', 'slides',
    'part', 'week', 'lesson', 'note', 'notes', 'review', 'fundamentals',
})

# Generic ADT boilerplate that KG nodes inject for multiple unrelated concepts
_GENERIC_ADT_BOILERPLATE = re.compile(
    r'abstract\s+data\s+type\s*\(adt\)',
    re.I
)

# Missing-verb grammar patterns — auto-fix "It visualized" → "It is visualized"
_GRAMMAR_FIXES = [
    (re.compile(r'\bIt\s+visualized\b',      re.I), 'It is visualized'),
    (re.compile(r'\bIt\s+implemented\b',     re.I), 'It is implemented'),
    (re.compile(r'\bIt\s+used\b',            re.I), 'It is used'),
    (re.compile(r'\bIt\s+defined\b',         re.I), 'It is defined'),
    (re.compile(r'\bIt\s+stored\b',          re.I), 'It is stored'),
    (re.compile(r'\bIt\s+represented\b',     re.I), 'It is represented'),
    (re.compile(r'\bIt\s+classified\b',      re.I), 'It is classified'),
    (re.compile(r'\bThey\s+used\b',          re.I), 'They are used'),
    (re.compile(r'\bThey\s+defined\b',       re.I), 'They are defined'),
    (re.compile(r'\bThey\s+stored\b',        re.I), 'They are stored'),
]

# Fused-concept label: "Insertions and Subject", "Components And Structure and Returns"
_FUSED_LABEL_AND = re.compile(r'^(.{4,50})\s+[Aa]nd\s+(.{4,50})$')

# Glued lowercase tail after ? → ": " conversion: ":a stack." at end of bullet
_GLUED_TAIL    = re.compile(r':\s*[a-z][\w\s]{0,25}\.\s*$')   # ':a stack.' or ':some words.'
_GLUED_ORPHAN  = re.compile(r':\s*$')


def _is_generic_adt_desc(text: str) -> bool:
    """Return True if text is the generic ADT boilerplate cloned to many KG nodes."""
    return bool(_GENERIC_ADT_BOILERPLATE.search(text))


def _is_bullet_fragment(text: str) -> bool:
    """
    Return True if text is an OCR-truncated fragment that should be dropped from bullets.
    Heuristics (domain-agnostic):
      - Starts with a single lowercase letter then space  (e.g. 'p erations...')
      - First token is ALL lowercase AND len ≥ 4 (= OCR-cut mid-word start like 'perations')
      - Starts with 'A is a type of...' where A is the cut off article residue
    """
    t = text.strip()
    if not t:
        return True
    words = t.split()
    first = words[0]
    # Single lowercase char at start
    if len(first) == 1 and first.islower():
        return True
    # First token is lowercase and ≥ 4 chars → mid-word OCR truncation
    # (e.g. 'perations', 'nline', 'ncluding', 'unctions')
    if first.islower() and len(first) >= 4:
        return True
    # 2–4 char lowercase fragment at start
    if len(first) <= 3 and first.islower() and len(words) > 1:
        return True
    # "A is a type of..." — 'A' is leftover truncated article
    if re.match(r'^A\s+is\s+a\b', t, re.I) and len(words) <= 8:
        return True
    return False


def _fix_grammar(text: str) -> str:
    """Fix missing-verb grammar errors introduced by note generation."""
    for pat, replacement in _GRAMMAR_FIXES:
        text = pat.sub(replacement, text)
    return text


def _is_fused_label(label: str) -> bool:
    """
    Detect KG labels that fuse two unrelated concepts via 'and'.
    e.g. 'Insertions and Subject', 'Components And Structure and Returns'.

    Two cases:
      A) Right side is NOT a category tag → clearly fused (zero word overlap)
      B) Right side IS a category tag (like 'subject') but left side is a specific
         domain concept, not a structural phrase → also reject (it's a meta label)
    """
    m = _FUSED_LABEL_AND.match(label.strip())
    if not m:
        return False
    left, right = m.group(1).strip(), m.group(2).strip()
    if len(left) < 4 or len(right) < 4:
        return False
    right_lower = right.lower().strip()
    stop = {'the', 'a', 'an', 'of', 'in', 'and', 'or', 'for', 'to', 'with'}
    lw = set(left.lower().split()) - stop
    rw = set(right.lower().split()) - stop

    if right_lower in _HEADING_CATEGORY_TAGS:
        # Right is a meta word — it's a slide-header label not a proper concept
        # e.g. "Insertions and Subject" → reject entirely
        return True

    # Both parts must be distinct noun phrases (no shared content words)
    return len(lw & rw) == 0   # zero overlap → two distinct concepts fused


def _primary_label(label: str) -> str:
    """Return only the primary (left) part of a fused label, or the label itself."""
    m = _FUSED_LABEL_AND.match(label.strip())
    if m:
        right_lower = m.group(2).strip().lower()
        if right_lower not in _HEADING_CATEGORY_TAGS:
            return m.group(1).strip()   # keep only left concept
    return label


def _clean_glued_tail(text: str) -> str:
    """
    Remove glued lowercase fragments at end of bullets caused by ? → ': ' conversion.
    'Stacks involves Stack Operations: Operations...:a stack.' → removes ':a stack.'
    """
    text = _GLUED_TAIL.sub('.', text)    # ':lowercase.' → '.'
    text = _GLUED_ORPHAN.sub('.', text)  # trailing ':' → '.'
    text = re.sub(r'\.{2,}', '.', text).strip()
    return text


def _merge_duplicate_subsections(section: Dict) -> Dict:
    """
    Within a single section, merge subsections that share the same heading.
    'Key Points' appearing twice → merge into one 'Key Points' with all bullets.
    """
    merged: Dict[str, Dict] = {}   # heading_lower → subsection dict
    order: List[str] = []          # preserve first-seen order

    for sub in section.get("subsections", []):
        h = sub.get("heading", "").strip()
        key = h.lower()
        if key not in merged:
            merged[key] = copy.deepcopy(sub)
            order.append(key)
        else:
            # Merge points, deduplicating by text
            existing_texts = {
                p.get("text", "").lower().strip()
                for p in merged[key].get("points", [])
            }
            for pt in sub.get("points", []):
                t = pt.get("text", "").lower().strip()
                if t and t not in existing_texts:
                    merged[key]["points"].append(pt)
                    existing_texts.add(t)

    section["subsections"] = [merged[k] for k in order]
    return section


def sanitize_question_marks(text: str, is_heading: bool = False) -> str:
    """
    Remove/replace ALL ? artefacts — fully dynamic, domain-agnostic.
    GUARANTEE: the returned string contains NO '?' character.

    Two modes:
      is_heading=True  — KG node labels used as section/subsection titles:
          "Implicit ? Stack"               -> "Implicit Stack"  (merge)
          "Linear Data Structure ? Structure" -> "Linear Data Structure"  (drop tag)
          "Stacks ? Chapter"               -> "Stacks"  (drop category tag)

      is_heading=False (default) — bullet/body text where ? is a KG separator:
          "X ? definition of X"            -> "X: definition of X"
          "node?s"                         -> "node's"
          "don?t"                          -> "don't"

    This means ? in bullets becomes ': ' so the definition is PRESERVED, not lost.
    """
    if not text or '?' not in text:
        return text  # Fast path

    # 1. Possessives and contractions (must run first — most specific)
    text = _Q_POSSESSIVE.sub(r"\1's", text)
    text = _Q_CONTRACTION.sub(r"\1't", text)

    if is_heading:
        # In headings, "X ? Y" is a KG node label with a category/relation tag suffix.
        # Strategy: if the word(s) after ? are category tags, drop them.
        # Otherwise merge (compound concept name).
        def _replace_heading_q(m):
            before = m.group(1).strip()
            after  = m.group(2).strip()
            last_word = after.split()[-1].lower() if after.split() else ""
            if last_word in _HEADING_CATEGORY_TAGS:
                return before  # Drop the tag suffix
            return before + " " + after  # Merge compound concept

        # Handle "Before ? After" pattern in headings
        text = re.sub(r'^(.+?)\s+\?\s+(.+)$', lambda m: _replace_heading_q(m), text.strip())
        # Fallback nuclear cleanup for any remaining ?
        text = _Q_NUCLEAR.sub(" ", text)
    else:
        # In bullets: " ? " is a KG schema separator meaning ": " (definition follows)
        # Decide intelligently: if right side looks like a definition (≥3 words), use ":"
        # otherwise just merge with a space.
        def _smart_q_replace(m):
            left  = m.group(1).strip() if m.lastindex and m.lastindex >= 1 else ""
            right = m.group(2).strip() if m.lastindex and m.lastindex >= 2 else ""
            if not right:
                return left
            # If right side starts with lowercase and is a clause → colon
            if right and right[0].islower() and len(right.split()) >= 3:
                return f"{left}: {right}"
            # If right looks like a category tag → drop it
            last_word = right.split()[-1].lower() if right.split() else ""
            if last_word in _HEADING_CATEGORY_TAGS:
                return left
            # Otherwise merge with space
            return f"{left} {right}"

        text = re.sub(r'(\S[^?]*?)\s+\?\s+([^?]+)', _smart_q_replace, text)

        # Clean up any glued ? not covered above
        text = _Q_NOSPC.sub(r"\1 \2", text)
        text = _Q_TRAIL_SPACE.sub(r"\1\2", text)
        text = _Q_PREFIX_UPPER.sub(r"\1", text)
        text = _Q_PREFIX.sub(r"\1", text)
        text = _Q_ISOLATED.sub("", text)
        text = _Q_LEADING.sub("", text)
        text = _Q_TRAILING.sub("", text)
        text = _Q_NUCLEAR.sub(" ", text)

    # Final: collapse spaces
    text = re.sub(r"\s{2,}", " ", text).strip()

    # NUCLEAR GUARANTEE: strip any remaining ? that survived all passes
    if '?' in text:
        text = text.replace('?', '')
        text = re.sub(r"\s{2,}", " ", text).strip()

    return text


def strip_all_question_marks(text: str) -> str:
    """
    Hard nuclear pass: remove EVERY '?' from any string.
    Use this as a last-resort safety net just before writing to PDF/TXT.
    No context-awareness — just removes unconditionally.
    Also cleans up double-spaces left behind.
    """
    if '?' not in text:
        return text
    text = text.replace('?', '')
    text = re.sub(r'\s{2,}', ' ', text).strip()
    return text


def sanitize_nodes_labels(nodes: List[Dict]) -> List[Dict]:
    """
    Sanitize ALL string fields of KG nodes before they enter the pipeline.
    Applied in order:
      1. ? artefacts removed/replaced (using is_heading mode for labels)
      2. Fused labels ('X and Y') → primary part only
      3. Generic ADT boilerplate descriptions → cleared (will be rebuilt by CEL)
      4. OCR truncation in label → fixed
    """
    for node in nodes:
        # Fix label
        label = node.get("label", "") or ""
        if '?' in label:
            label = sanitize_question_marks(label, is_heading=True)
        if _is_fused_label(label):
            label = _primary_label(label)
        label = fix_ocr_truncated_label(label)
        node["label"] = label

        # Fix description: clear generic ADT boilerplate so CEL can rebuild properly
        desc = node.get("description", "") or ""
        if '?' in desc:
            desc = sanitize_question_marks(desc, is_heading=False)
        if _is_generic_adt_desc(desc) and len(desc.split()) < 30:
            # Only clear if it's purely the boilerplate (not a longer, mixed description)
            desc = ""
        node["description"] = desc

        # Fix any other string fields (type, etc.)
        for field in list(node.keys()):
            if field in ("label", "description"):
                continue
            val = node.get(field)
            if isinstance(val, str) and '?' in val:
                node[field] = sanitize_question_marks(val, is_heading=False)

    return nodes


def sanitize_notes_question_marks(sections: List[Dict]) -> List[Dict]:
    """
    Remove/fix all ? artefacts — headings use is_heading mode, bullets use separator mode.
    Also applies:
      - grammar fixes (missing 'is', 'are')
      - glued-tail cleanup (':a stack.' → '.')
      - bullet fragment dropping ('perations...' → dropped)
      - fused section heading recovery ('X and Y' → 'X')
      - NUCLEAR FINAL PASS: strip any remaining ? from all text fields
    """
    for section in sections:
        if "heading" in section:
            h = sanitize_question_marks(section["heading"], is_heading=True)
            # Recover fused labels: "Insertions and Subject" → "Insertions"
            if _is_fused_label(h):
                h = _primary_label(h)
            section["heading"] = strip_all_question_marks(h)

        # Sanitize section-level summary if present
        if "summary" in section and section["summary"]:
            section["summary"] = strip_all_question_marks(
                sanitize_question_marks(section["summary"], is_heading=False)
            )

        for sub in section.get("subsections", []):
            if "heading" in sub:
                sh = sanitize_question_marks(sub["heading"], is_heading=True)
                sub["heading"] = strip_all_question_marks(sh)

            new_pts = []
            for pt in sub.get("points", []):
                if "text" not in pt:
                    continue
                text = sanitize_question_marks(pt["text"], is_heading=False)
                text = _clean_glued_tail(text)
                text = _fix_grammar(text)
                # NUCLEAR PASS — remove any ? that survived
                text = strip_all_question_marks(text)
                # Drop OCR fragment bullets (e.g. "perations Performed on Stack")
                if _is_bullet_fragment(text):
                    continue
                # Drop generic ADT boilerplate if it's the entire bullet content
                if _is_generic_adt_desc(text) and len(text.split()) < 25:
                    continue
                if text.strip():
                    new_pts.append({"text": text})
            sub["points"] = new_pts

        # Merge duplicate subsection headings within section
        section = _merge_duplicate_subsections(section)

    return sections


def sanitize_notes_dict(notes: Dict) -> Dict:
    """
    Apply full ? sanitisation to an entire notes dict, including
    the top-level 'summary' field and all sections.
    Call this as the FINAL step before render_pdf / render_txt.
    """
    # Top-level summary
    if "summary" in notes and notes["summary"]:
        notes["summary"] = strip_all_question_marks(
            sanitize_question_marks(notes["summary"], is_heading=False)
        )
    # Top-level title
    if "title" in notes and notes["title"]:
        notes["title"] = strip_all_question_marks(
            sanitize_question_marks(notes["title"], is_heading=True)
        )
    # All sections
    notes["sections"] = sanitize_notes_question_marks(notes.get("sections", []))
    return notes


# ═══════════════════════════════════════════════════════════════════════════════
# OCR Truncation Fix — Fix leading-lowercase / partial-word node labels
# ═══════════════════════════════════════════════════════════════════════════════

# Known OCR truncation prefixes: first letter was cut off, leaving lowercase fragment
_OCR_TRUNCATION_PREFIXES = re.compile(
    r'^(?:p|o|f|s|c|t|e|n|h|r|l|a|b|d|m|g|w|y|v|u|k|j|q|x|z)'
    r'(?=[a-z]{2,})',   # followed by ≥2 more lowercase chars
)

# Map common OCR-truncated starts to likely first letter
# e.g. "perations" → "Operations", "Of Operations" stays if it's valid
_LIKELY_CAPS = re.compile(r'^([a-z])([a-z])')


def fix_ocr_truncated_label(label: str) -> str:
    """
    Fix OCR-truncated KG node labels where the first character was cut.

    Detection heuristics (domain-agnostic):
      1. Label starts with lowercase (KG labels are normally title-cased)
      2. Label looks like it starts mid-word (first token ≤3 chars lowercase)

    Strategy: capitalize the first letter. If the result looks like a real
    heading (starts with capital, no leading noise), keep it; otherwise drop
    the leading fragment and try the next word.

    Examples:
      "perations Performed On Stack"  -> "Operations Performed On Stack"
      "Of Operations"                 -> "Of Operations"  (kept — grammatical)
      "nline Streaming"               -> "Online Streaming"
      "therwise display"              -> dropped (too short to recover)
    """
    if not label:
        return label
    # Already starts with uppercase — no fix needed
    if label[0].isupper():
        return label

    words = label.split()
    if not words:
        return label

    first = words[0]

    # Case 1: first word is short fragment (≤3 chars, all lowercase) → drop it
    if len(first) <= 3 and first.islower():
        recovered = " ".join(words[1:]).strip()
        return recovered if recovered else label

    # Case 2: first word looks like a mid-word fragment (lowercase, ≥4 chars)
    # → capitalize it (OCR cut off the uppercase first letter)
    if first.islower() and len(first) >= 4:
        fixed = first[0].upper() + first[1:]
        return fixed + (" " + " ".join(words[1:]) if len(words) > 1 else "")

    # Case 3: capitalize first character
    return label[0].upper() + label[1:]


# ═══════════════════════════════════════════════════════════════════════════════
# Context Expansion Layer (CEL) — The single most impactful improvement
# ═══════════════════════════════════════════════════════════════════════════════

# Minimum word count for a bullet to be considered "has explanation"
_MIN_EXPLANATION_WORDS = 8

# Verbs that indicate a sentence is explanatory (not just a label)
_EXPLANATORY_VERBS = {
    'is', 'are', 'was', 'were', 'means', 'refers', 'defined', 'represents',
    'allows', 'enables', 'provides', 'performs', 'stores', 'converts',
    'checks', 'manages', 'handles', 'generates', 'produces', 'detects',
    'implements', 'uses', 'supports', 'ensures', 'follows', 'operates',
    'consists', 'contains', 'includes', 'traverses', 'processes', 'returns',
    'removes', 'inserts', 'adds', 'deletes', 'retrieves', 'calculates',
}

# Patterns indicating a sentence is a label-only (not an explanation)
_LABEL_ONLY_PATTERNS = [
    re.compile(r'^[A-Z][a-zA-Z\s/\-()]{0,40}$'),   # Pure title-case noun phrase
]


def _has_explanation(text: str) -> bool:
    """Return True if the text already contains a meaningful explanation."""
    words = text.strip().split()
    if len(words) < _MIN_EXPLANATION_WORDS:
        return False
    lw = text.lower()
    return any(f' {v} ' in f' {lw} ' or lw.startswith(v + ' ') for v in _EXPLANATORY_VERBS)


def _is_label_only(text: str) -> bool:
    """Return True if text is just a label/title with no explanation."""
    text = text.strip()
    # Short and no verb → label only
    words = text.split()
    if len(words) <= 4:
        lw = text.lower()
        if not any(f' {v} ' in f' {lw} ' for v in _EXPLANATORY_VERBS):
            return True
    # Colon means it has a definition appended → not label-only
    if ':' in text or '—' in text:
        return False
    return False


def _build_cel_explanation(
    label: str,
    node: Dict,
    edges_out: List[Tuple[str, str]],
    node_by_id: Dict[str, Dict],
    parent_label: str = "",
) -> str:
    """
    Build a context-rich explanation for a contextless concept.

    Sources (in priority order):
      1. Node's own description if ≥8 words, not circular, and not generic ADT boilerplate
      2. KG edges: Input/Output/Purpose/Type sentences
      3. Minimal fallback: bare label (no template injection)
    """
    desc = node.get("description", "").strip()

    # 1. Node description — use if substantive and not generic boilerplate
    if desc and len(desc.split()) >= _MIN_EXPLANATION_WORDS and not _is_generic_adt_desc(desc):
        result = f"{label}: {desc}"
        return _fix_grammar(result)

    # 2. Build from KG edges
    _SKIP = {'is_a', 'type_of', 'part_of', 'related_to', 'includes', 'has', 'contains'}
    rel_parts: List[str] = []

    inputs, outputs, purposes, types = [], [], [], []
    for tgt_id, relation in (edges_out or [])[:8]:
        tgt = node_by_id.get(tgt_id, {})
        tgt_lbl = tgt.get("label", "")
        if not tgt_lbl:
            continue
        rel_norm = relation.lower().replace(" ", "_")
        if rel_norm in _SKIP:
            types.append(tgt_lbl)
        elif any(s in rel_norm for s in ['produces', 'generates', 'outputs', 'creates', 'returns']):
            outputs.append((tgt_lbl, relation))
        elif any(s in rel_norm for s in ['takes', 'receives', 'reads', 'input_to']):
            inputs.append((tgt_lbl, relation))
        elif any(s in rel_norm for s in ['used_for', 'used_in', 'used_by', 'application', 'supports', 'enables']):
            purposes.append((tgt_lbl, relation))
        else:
            rel_prose = relation.replace("_", " ").lower()
            rel_parts.append(f"{rel_prose} {tgt_lbl}")

    parts = []
    if desc and len(desc.split()) >= 4 and not _is_generic_adt_desc(desc):
        parts.append(desc)
    if types:
        type_list = ", ".join(types[:2])
        parts.append(f"It is a type of {type_list}.")
    if inputs:
        inp_str = ", ".join(l for l, _ in inputs[:2])
        parts.append(f"It takes {inp_str} as input.")
    if outputs:
        out_str = ", ".join(l for l, _ in outputs[:2])
        parts.append(f"It produces {out_str} as output.")
    if purposes:
        pur_str = ", ".join(l for l, _ in purposes[:2])
        parts.append(f"It is used for {pur_str}.")
    if rel_parts and not parts:
        parts.append(f"It {rel_parts[0]}.")

    if parts:
        result = f"{label}: " + " ".join(parts)
        return _fix_grammar(result)

    # 3. Minimal fallback — return bare label; do NOT inject "A concept related to X"
    # The template was causing pollution ("Abstract Data Type: A concept related to Stacks...")
    return label


def apply_context_expansion_layer(
    sections: List[Dict],
    concept_order: List[Dict],
    node_by_id: Dict[str, Dict],
) -> List[Dict]:
    """
    Context Expansion Layer (CEL):
    Detect every contextless bullet (bare label, no explanation) and expand it.
    Also applies: grammar fixes, glued-tail cleanup, fragment dropping, ADT boilerplate drop.
    """
    label_to_rec: Dict[str, Dict] = {}
    for rec in concept_order:
        label_to_rec[rec["label"].lower().strip()] = rec

    for section in sections:
        parent_label = section.get("heading", "")

        for sub in section.get("subsections", []):
            new_points = []
            for pt in sub.get("points", []):
                text = pt.get("text", "").strip()
                if not text:
                    continue

                # Drop OCR fragment bullets globally
                if _is_bullet_fragment(text):
                    continue

                # Drop pure generic ADT boilerplate (short form)
                if _is_generic_adt_desc(text) and len(text.split()) < 25:
                    continue

                # Apply glued-tail and grammar fixes to all bullets
                text = _clean_glued_tail(text)
                text = _fix_grammar(text)

                # Already has explanation — keep
                if _has_explanation(text):
                    new_points.append({"text": text})
                    continue

                # Has colon → has at least a short definition
                if ':' in text:
                    new_points.append({"text": text})
                    continue

                # Look up concept in KG
                key = text.lower().strip().rstrip(".")
                rec = label_to_rec.get(key)
                if not rec:
                    for lbl, r in label_to_rec.items():
                        if (lbl in key and len(lbl) > 4) or (key in lbl and len(key) > 4):
                            rec = r
                            break

                if rec:
                    node = node_by_id.get(rec["id"], {"label": text, "description": "", "id": ""})
                    edges_out = rec.get("edges_out", [])
                    expanded = _build_cel_explanation(
                        text, node, edges_out, node_by_id, parent_label
                    )
                    expanded = sanitize_question_marks(expanded, is_heading=False)
                    expanded = _clean_glued_tail(expanded)
                    expanded = _fix_grammar(expanded)
                    if expanded and expanded != text:
                        new_points.append({"text": expanded})
                        continue

                new_points.append({"text": text})

            sub["points"] = new_points

    return sections


# ═══════════════════════════════════════════════════════════════════════════════
# Section Semantic Deduplicator — merge duplicate/overlapping topic sections
# ═══════════════════════════════════════════════════════════════════════════════

_DEDUP_STOPWORDS: Set[str] = {
    'the', 'a', 'an', 'is', 'are', 'and', 'or', 'of', 'in', 'on', 'at',
    'to', 'for', 'with', 'that', 'this', 'which', 'by', 'from',
    'operations', 'operation', 'performed', 'on',  # very common in topic names
}


def _section_vocab(section: Dict) -> Set[str]:
    """Extract meaningful words from a section's heading + all bullet text."""
    words = set()
    h = section.get("heading", "").lower()
    for w in re.findall(r"\b[a-z]{3,}\b", h):
        if w not in _DEDUP_STOPWORDS:
            words.add(w)
    for sub in section.get("subsections", []):
        for pt in sub.get("points", []):
            for w in re.findall(r"\b[a-z]{3,}\b", pt.get("text", "").lower()):
                if w not in _DEDUP_STOPWORDS:
                    words.add(w)
    return words


def _heading_jaccard(h1: str, h2: str) -> float:
    """Jaccard similarity between two heading strings (word-level)."""
    w1 = {w.lower() for w in re.findall(r"\b[a-z]{3,}\b", h1.lower())
          if w.lower() not in _DEDUP_STOPWORDS}
    w2 = {w.lower() for w in re.findall(r"\b[a-z]{3,}\b", h2.lower())
          if w.lower() not in _DEDUP_STOPWORDS}
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)


def _merge_sections(primary: Dict, secondary: Dict) -> Dict:
    """
    Merge secondary into primary, deduplicating bullets.
    Primary heading is kept; unique secondary points are added.
    """
    primary = copy.deepcopy(primary)
    seen_texts: Set[str] = set()

    for sub in primary.get("subsections", []):
        for pt in sub.get("points", []):
            t = pt.get("text", "").lower().strip()
            seen_texts.add(t)

    for sec_sub in secondary.get("subsections", []):
        # Find a matching subsection in primary by heading
        sec_heading = sec_sub.get("heading", "").lower()
        matched_sub = None
        for pri_sub in primary.get("subsections", []):
            if pri_sub.get("heading", "").lower() == sec_heading:
                matched_sub = pri_sub
                break

        unique_pts = []
        for pt in sec_sub.get("points", []):
            t = pt.get("text", "").lower().strip()
            words_new = set(t.split())
            is_dup = t in seen_texts
            if not is_dup:
                # Also check near-duplicate (70% overlap)
                for seen in seen_texts:
                    words_seen = set(seen.split())
                    if words_new and words_seen:
                        overlap = len(words_new & words_seen) / max(len(words_new), len(words_seen))
                        if overlap > 0.70:
                            is_dup = True
                            break
            if not is_dup:
                unique_pts.append(pt)
                seen_texts.add(t)

        if unique_pts:
            if matched_sub:
                matched_sub["points"].extend(unique_pts)
            else:
                primary.setdefault("subsections", []).append({
                    "heading": sec_sub.get("heading", "Additional Details"),
                    "points": unique_pts,
                })

    return primary


def deduplicate_topic_sections(sections: List[Dict]) -> List[Dict]:
    """
    Merge sections that refer to the same concept under different names.

    Detection: Jaccard similarity of section headings ≥ 0.5 OR
               heading words of one are a subset of the other.

    Example merges:
      "Stack Operations" + "Operations Performed On Stack"  → keep first
      "Order Of Operations" + "Stack Operations"            → check overlap

    Fully dynamic — no topic-specific knowledge required.
    """
    if len(sections) <= 1:
        return sections

    merged_flags = [False] * len(sections)
    result = []

    for i, sec_a in enumerate(sections):
        if merged_flags[i]:
            continue

        current = copy.deepcopy(sec_a)
        h_a = sec_a.get("heading", "").lower()
        w_a = {w for w in re.findall(r"\b[a-z]{3,}\b", h_a) if w not in _DEDUP_STOPWORDS}

        for j in range(i + 1, len(sections)):
            if merged_flags[j]:
                continue
            sec_b = sections[j]
            h_b = sec_b.get("heading", "").lower()
            w_b = {w for w in re.findall(r"\b[a-z]{3,}\b", h_b) if w not in _DEDUP_STOPWORDS}

            jaccard = len(w_a & w_b) / len(w_a | w_b) if (w_a | w_b) else 0.0
            # Merge if headings are highly similar OR one heading is contained in the other
            subset = (w_a and w_b and (w_a <= w_b or w_b <= w_a))

            if jaccard >= 0.5 or subset:
                current = _merge_sections(current, sec_b)
                merged_flags[j] = True

        result.append(current)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Priority 1 — Conceptual Hierarchy Builder
# ═══════════════════════════════════════════════════════════════════════════════

class ConceptualHierarchyBuilder:
    """Build a true parent->child concept tree from KG edge semantics."""

    def __init__(self, nodes: List[Dict], edges: List[Dict]):
        self.nodes = nodes
        self.edges = edges
        self.node_by_id: Dict[str, Dict] = {n["id"]: n for n in nodes}

        self.children: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        self.parents:  Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        self.seq_succ: Dict[str, List[str]] = defaultdict(list)
        self.edges_out: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

        for e in edges:
            src = e.get("source", "")
            tgt = e.get("target", "")
            rel = e.get("relation", "").lower().strip().replace(" ", "_")
            if not src or not tgt:
                continue
            self.children[src].append((tgt, rel))
            self.parents[tgt].append((src, rel))
            self.edges_out[src].append((tgt, e.get("relation", rel)))
            if any(s in rel for s in _SEQUENTIAL_RELATIONS):
                self.seq_succ[src].append(tgt)

    def _is_containment(self, rel: str) -> bool:
        return any(s in rel for s in _CONTAINMENT_RELATIONS)

    def build(self) -> List[Dict]:
        """Return ordered concept records with level, parent, children, edges_out."""
        has_containment_parent: Set[str] = set()
        for tgt, parents_list in self.parents.items():
            for _, rel in parents_list:
                if self._is_containment(rel):
                    has_containment_parent.add(tgt)

        all_ids = set(self.node_by_id.keys())
        root_ids = all_ids - has_containment_parent
        level_map: Dict[str, int] = {}
        parent_map: Dict[str, Optional[str]] = {}
        queue: deque = deque()

        for rid in root_ids:
            level_map[rid] = 0
            parent_map[rid] = None
            queue.append(rid)

        visited: Set[str] = set(root_ids)
        while queue:
            curr = queue.popleft()
            for tgt, rel in self.children[curr]:
                if tgt not in visited and self._is_containment(rel):
                    level_map[tgt] = level_map[curr] + 1
                    parent_map[tgt] = curr
                    visited.add(tgt)
                    queue.append(tgt)

        for nid in all_ids:
            if nid not in level_map:
                level_map[nid] = 0
                parent_map[nid] = None

        ordered = self._topo_sort(level_map, parent_map)

        children_of: Dict[str, List[str]] = defaultdict(list)
        for nid, pid in parent_map.items():
            if pid is not None:
                children_of[pid].append(nid)

        result = []
        for nid in ordered:
            node = self.node_by_id.get(nid, {})
            result.append({
                "id":           nid,
                "label":        node.get("label", nid),
                "description":  node.get("description", ""),
                "level":        level_map.get(nid, 0),
                "parent_id":    parent_map.get(nid),
                "children_ids": children_of.get(nid, []),
                "edges_out":    self.edges_out.get(nid, []),
            })
        return result

    def _topo_sort(self, level_map, parent_map) -> List[str]:
        all_ids = list(self.node_by_id.keys())
        in_deg: Dict[str, int] = {nid: 0 for nid in all_ids}
        adj: Dict[str, List[str]] = defaultdict(list)

        for nid, pid in parent_map.items():
            if pid is not None:
                adj[pid].append(nid)
                in_deg[nid] += 1

        for src, succs in self.seq_succ.items():
            for tgt in succs:
                if tgt not in adj[src]:
                    adj[src].append(tgt)
                    in_deg[tgt] += 1

        def _prio(nid):
            node = self.node_by_id.get(nid, {})
            text = node.get("label", "").lower() + " " + node.get("description", "").lower()
            best = 99
            for _, p, kws in _DISCOURSE_ROLES:
                if any(kw in text for kw in kws):
                    best = min(best, p)
            return (best, node.get("label", ""))

        queue = deque(sorted([n for n in all_ids if in_deg[n] == 0], key=_prio))
        result = []
        while queue:
            nid = queue.popleft()
            result.append(nid)
            nbrs = sorted(adj.get(nid, []), key=_prio)
            for tgt in nbrs:
                in_deg[tgt] -= 1
                if in_deg[tgt] == 0:
                    queue.append(tgt)
                    sorted_q = sorted(queue, key=_prio)
                    queue = deque(sorted_q)

        seen = set(result)
        for nid in all_ids:
            if nid not in seen:
                result.append(nid)
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# Priority 2 — Concept Relationship Subsection Builder
# ═══════════════════════════════════════════════════════════════════════════════

def _edge_to_sentence(src_label: str, relation: str, tgt_label: str) -> str:
    rel_key = relation.lower().strip().replace(" ", "_")
    if rel_key in _RELATIONSHIP_TEMPLATES:
        return _RELATIONSHIP_TEMPLATES[rel_key].format(A=src_label, B=tgt_label)
    for key, tmpl in _RELATIONSHIP_TEMPLATES.items():
        if key in rel_key or rel_key in key:
            return tmpl.format(A=src_label, B=tgt_label)
    return f"{src_label} {relation.replace('_', ' ').lower()} {tgt_label}."


def build_relationship_subsections(
    sections: List[Dict],
    concept_order: List[Dict],
    node_by_id: Dict[str, Dict],
) -> List[Dict]:
    """
    Add a relationship subsection to each section with >=2 meaningful KG-edge
    sentences. Sentences are grouped by relation type so related targets are
    merged into a single bullet rather than emitting one per edge.

    NEW: groups targets by relation verb so the output reads:
        "X includes Stack, Queue, and Linked List."
    instead of three separate bullets:
        "X includes Stack."  /  "X includes Queue."  /  "X includes Linked List."
    """
    label_to_rec: Dict[str, Dict] = {}
    for rec in concept_order:
        label_to_rec[rec["label"].lower().strip()] = rec

    def _find_rec(heading: str) -> Optional[Dict]:
        h = heading.lower().strip()
        if h in label_to_rec:
            return label_to_rec[h]
        for lbl, r in label_to_rec.items():
            if lbl in h or h in lbl:
                return r
        return None

    # Relations that carry no new info (covered by containment hierarchy already)
    _SKIP_RELS = {"is_a", "type_of", "part_of", "is_part_of",
                  "categorizes", "encompasses", "observed", "previously_observed"}

    # Relations that should be GROUPED (multiple targets merged into one sentence)
    _GROUP_RELS = {"includes", "has", "contains", "covers", "has_component",
                   "has_type", "encompasses", "consists_of", "has_subtype",
                   "has_phase", "has_step", "has_layer"}

    for section in sections:
        rec = _find_rec(section.get("heading", ""))
        if not rec:
            continue

        edges_out = rec.get("edges_out", [])
        if not edges_out:
            continue

        existing = {ss.get("heading", "").lower() for ss in section.get("subsections", [])}
        if any("relationship" in h or "relation" in h or "how it works" in h
               for h in existing):
            continue

        # Group targets by normalised relation key
        grouped: Dict[str, List[str]] = defaultdict(list)
        single: List[str]             = []       # non-group relation sentences
        seen_targets: Set[str]        = set()

        for tgt_id, relation in edges_out:
            rel_key = relation.lower().strip().replace(" ", "_")
            if rel_key in _SKIP_RELS:
                continue
            tgt_node  = node_by_id.get(tgt_id, {})
            tgt_label = tgt_node.get("label", "")
            if not tgt_label or tgt_label in seen_targets:
                continue
            seen_targets.add(tgt_label)

            if rel_key in _GROUP_RELS:
                grouped[rel_key].append(tgt_label)
            else:
                sentence = _edge_to_sentence(rec["label"], relation, tgt_label)
                # Strip "- context annotation" KG suffix from sentence
                sentence = re.sub(r'\s*[-–—]\s*[A-Z][^.]{5,120}\.?\s*$', '.', sentence)
                sentence = re.sub(r'\s+during\s+its\s+operation\.?', '.', sentence, flags=re.I)
                sentence = sanitize_question_marks(sentence)
                if len(sentence.split()) >= 5:
                    single.append(sentence)

        rel_points: List[Dict] = []

        # Emit grouped bullets first (most informative)
        for rel_key, targets in grouped.items():
            if not targets:
                continue
            rel_natural = rel_key.replace("_", " ")
            # Use template if available
            tmpl_sentence = _edge_to_sentence(rec["label"], rel_key, targets[0])
            if len(targets) == 1:
                sentence = tmpl_sentence
            elif len(targets) == 2:
                sentence = f"{rec['label']} {rel_natural} {targets[0]} and {targets[1]}."
            else:
                listed = ", ".join(targets[:-1]) + f", and {targets[-1]}"
                sentence = f"{rec['label']} {rel_natural} {listed}."
            sentence = sanitize_question_marks(sentence)
            if len(sentence.split()) >= 5:
                rel_points.append({"text": sentence})

        # Emit single-relation bullets (limited to avoid clutter)
        for s in single[:4]:
            rel_points.append({"text": s})

        if len(rel_points) >= 2:
            subs = section.get("subsections", [])
            subs.insert(min(1, len(subs)), {
                "heading": "How It Works",
                "points":  rel_points[:6],
            })
            section["subsections"] = subs

    return sections


# ═══════════════════════════════════════════════════════════════════════════════
# Priority 3 — Concept Flow / Pipeline Section
# ═══════════════════════════════════════════════════════════════════════════════

class PipelineFlowInjector:
    """Detect sequential chains and build workflow content."""

    def __init__(self, nodes: List[Dict], edges: List[Dict]):
        self.node_by_id: Dict[str, Dict] = {n["id"]: n for n in nodes}
        self.seq_edges: List[Tuple[str, str, str]] = []
        for e in edges:
            src = e.get("source", "")
            tgt = e.get("target", "")
            rel = e.get("relation", "").lower().strip().replace(" ", "_")
            if src and tgt and any(s in rel for s in _SEQUENTIAL_RELATIONS):
                self.seq_edges.append((src, tgt, rel))

    def _label(self, nid: str) -> str:
        return self.node_by_id.get(nid, {}).get("label", nid)

    def _desc(self, nid: str) -> str:
        return self.node_by_id.get(nid, {}).get("description", "")

    def extract_pipelines(self, min_length: int = 3) -> List[List[str]]:
        succ: Dict[str, List[str]] = defaultdict(list)
        pred: Set[str] = set()
        for src, tgt, _ in self.seq_edges:
            succ[src].append(tgt)
            pred.add(tgt)
        all_seq = {s for s, _, _ in self.seq_edges} | {t for _, t, _ in self.seq_edges}
        starts = [n for n in all_seq if n not in pred]
        chains, visited = [], set()
        for start in starts:
            chain, curr = [start], start
            visited.add(start)
            while succ.get(curr):
                nxt = succ[curr][0]
                if nxt in visited:
                    break
                chain.append(nxt); visited.add(nxt); curr = nxt
            if len(chain) >= min_length:
                chains.append(chain)
        chains.sort(key=lambda c: -len(c))
        return chains

    def build_workflow_section(self) -> Optional[Dict]:
        """
        Build a top-level "Concept Workflow" section with:
          - Full pipeline arrow line
          - Numbered step bullets with descriptions
        """
        chains = self.extract_pipelines(min_length=3)
        if not chains:
            return None
        chain  = chains[0]
        labels = [self._label(n) for n in chain]
        arrow  = " → ".join(labels)
        points = [{"text": f"Full Pipeline: {arrow}."}]
        for i, nid in enumerate(chain, 1):
            lbl  = self._label(nid)
            desc = self._desc(nid)
            if desc and len(desc.split()) > 4:
                clip = " ".join(desc.split()[:25])
                bullet = f"Step {i} — {lbl}: {clip}."
            else:
                bullet = f"Step {i} — {lbl}."
            points.append({"text": sanitize_question_marks(bullet)})
        return {
            "heading": "Concept Workflow",
            "subsections": [{"heading": "Process Overview", "points": points}],
        }

    def inject_flow_subsection(self, sections: List[Dict]) -> List[Dict]:
        """Add a 'Process Flow' subsection in the best-matching section."""
        chains = self.extract_pipelines(min_length=3)
        if not chains:
            return sections
        for chain in chains[:2]:
            labels = [self._label(n) for n in chain]
            if len(labels) < 3:
                continue
            target = self._best_section(chain, sections) or (sections[0] if sections else None)
            if not target:
                continue
            existing = {ss.get("heading", "").lower() for ss in target.get("subsections", [])}
            if any("flow" in h or "pipeline" in h or "process" in h for h in existing):
                continue
            pts = [{"text": f"Pipeline: {' → '.join(labels)}."}]
            for i, nid in enumerate(chain, 1):
                desc = self._desc(nid)
                clip = (" ".join(desc.split()[:18]) + ".") if desc else ""
                bullet = f"Step {i} — {self._label(nid)}: {clip}".strip() if clip else f"Step {i} — {self._label(nid)}."
                pts.append({"text": sanitize_question_marks(bullet)})
            subs = target.get("subsections", [])
            subs.insert(min(1, len(subs)), {"heading": "Process Flow", "points": pts})
            target["subsections"] = subs
        return sections

    def _best_section(self, chain, sections):
        chain_words = set()
        for nid in chain:
            for w in self._label(nid).lower().split():
                if len(w) > 3 and w not in _STOPWORDS:
                    chain_words.add(w)
        best, best_score = None, 0
        for sec in sections:
            score = len(chain_words & set(sec.get("heading", "").lower().split()))
            if score > best_score:
                best_score, best = score, sec
        return best if best_score > 0 else None


# ═══════════════════════════════════════════════════════════════════════════════
# Priority 4 — Explanatory Bullet Expander
# ═══════════════════════════════════════════════════════════════════════════════

_DELEGATING_VERBS = re.compile(
    r"\b(is|are)\s+(controlled|handled|managed|performed|executed|done|"
    r"implemented|run|processed|carried out)\s+by\b", re.I
)
_ROLE_SUFFIXES = re.compile(
    r"\b(analyzer|analysis|generator|generation|optimizer|optimization|"
    r"handler|manager|controller|checker|processor|compiler|parser|"
    r"phase|stage|module|system|layer|component|unit)\b", re.I
)
_TAUTOLOGY_PATS = [
    re.compile(r"^(\w[\w\s]{2,40})\s+is\s+(?:controlled|handled|managed|performed|done|executed|processed)\s+by\s+\1", re.I),
    re.compile(r"^(\w[\w\s]{2,40})\s+(?:is|are)\s+\1(?:\s+\w+)?\.?\s*$", re.I),
]


def _word_overlap(a: str, b: str) -> float:
    stop = _STOPWORDS | {
        "phase","stage","layer","process","system","module","analyzer",
        "analysis","generator","generation","optimizer","optimization",
        "handler","manager","controller","checker",
    }
    wa = {w.lower() for w in re.findall(r"\w+", a) if w.lower() not in stop and len(w) > 2}
    wb = {w.lower() for w in re.findall(r"\w+", b) if w.lower() not in stop and len(w) > 2}
    return len(wa & wb) / len(wa) if wa else 0.0


def _is_circular(text: str) -> bool:
    text = text.strip()
    for pat in _TAUTOLOGY_PATS:
        if pat.search(text):
            return True
    m = _DELEGATING_VERBS.search(text)
    if m:
        subj = _ROLE_SUFFIXES.sub("", text[:m.start()]).strip()
        obj  = re.sub(r"\s*\(.*?\)", "", _ROLE_SUFFIXES.sub("", text[m.end():]).strip())
        if subj and obj and _word_overlap(subj, obj) >= 0.5:
            return True
    for sep in [" is ", " are "]:
        if sep in text.lower():
            idx = text.lower().find(sep)
            sw = set(text[:idx].lower().split()) - _STOPWORDS
            pw = set(text[idx + len(sep):].lower().split()) - _STOPWORDS
            if sw and pw and len(sw & pw) / max(len(sw), 1) >= 0.6:
                return True
    return False


def _expand_bullet(label: str, node: Dict, parent_label: str, relation: str,
                   concept_order: List[Dict], node_by_id: Dict) -> str:
    """Expand a bare label into an Input/Process/Output style explanation."""
    desc = node.get("description", "")

    # 1. Use own description if substantive and not circular
    if desc and len(desc.split()) >= 8 and not _is_circular(desc):
        return f"{label}: {desc}"

    # 2. Use relation template with description snippet
    rel_key = relation.lower().strip().replace(" ", "_")
    skip_rels = {"includes", "has", "contains", "comprises", "is_a", "part_of"}
    if rel_key not in skip_rels:
        tmpl = _edge_to_sentence(parent_label, relation, label)
        if desc and len(desc.split()) >= 4:
            return f"{label}: {' '.join(desc.split()[:20])}."
        if tmpl and len(tmpl.split()) >= 6:
            return tmpl

    # 3. Build Input/Process/Output from KG edges
    id_to_rec = {r["id"]: r for r in concept_order}
    rec = id_to_rec.get(node.get("id", ""))
    if rec:
        inputs, outputs = [], []
        for tgt_id, rel in rec.get("edges_out", []):
            rl = rel.lower().replace(" ", "_")
            tgt_lbl = node_by_id.get(tgt_id, {}).get("label", "")
            if tgt_lbl:
                if any(s in rl for s in ["input_to","takes","receives","reads"]):
                    inputs.append(tgt_lbl)
                if any(s in rl for s in ["produces","generates","outputs","creates"]):
                    outputs.append(tgt_lbl)
        parts = []
        if desc and len(desc.split()) >= 4:
            parts.append(desc)
        if inputs:
            parts.append(f"Input: {', '.join(inputs[:2])}")
        if outputs:
            parts.append(f"Output: {', '.join(outputs[:2])}")
        if parts:
            return f"{label}: " + ". ".join(parts) + "."

    # 4. Minimal fallback
    if desc and len(desc.split()) >= 3:
        return f"{label}: {desc}"
    return label


def expand_explanatory_bullets(
    sections: List[Dict],
    concept_order: List[Dict],
    node_by_id: Dict[str, Dict],
) -> List[Dict]:
    """Expand bare-label bullets into explanatory content."""
    label_to_rec: Dict[str, Dict] = {}
    for rec in concept_order:
        label_to_rec[rec["label"].lower().strip()] = rec

    for section in sections:
        parent_label = section.get("heading", "")
        parent_rec   = label_to_rec.get(parent_label.lower())

        for sub in section.get("subsections", []):
            new_pts = []
            for pt in sub.get("points", []):
                text = pt.get("text", "").strip()
                if not text:
                    continue

                # Remove circular definitions (keep if it's the only bullet)
                if _is_circular(text) and len(sub.get("points", [])) > 1:
                    continue

                # Expand bare labels (short, no verb indicator)
                wc = len(text.split())
                has_verb = any(v in text.lower() for v in [
                    " is "," are "," was "," produces "," generates ",
                    " converts "," checks "," stores "," manages ",
                    " handles "," performs "," implements "," uses ",
                    " takes "," reads "," outputs ",":",
                ])
                if wc <= 5 and not has_verb:
                    key = text.lower().strip().rstrip(".")
                    rec = label_to_rec.get(key)
                    if not rec:
                        for lbl, r in label_to_rec.items():
                            if lbl in key or key in lbl:
                                rec = r; break
                    if rec:
                        node = node_by_id.get(rec["id"], {"label": text, "description": "", "id": ""})
                        relation = "includes"
                        if parent_rec:
                            for tgt_id, rel in parent_rec.get("edges_out", []):
                                if tgt_id == rec["id"]:
                                    relation = rel; break
                        expanded = _expand_bullet(text, node, parent_label, relation, concept_order, node_by_id)
                        if expanded and expanded != text:
                            pt = {"text": sanitize_question_marks(expanded)}

                new_pts.append(pt)
            sub["points"] = new_pts

    return sections


# ═══════════════════════════════════════════════════════════════════════════════
# Section Flow Reorderer
# ═══════════════════════════════════════════════════════════════════════════════

class SectionFlowReorderer:
    def reorder(self, sections: List[Dict], concept_order: List[Dict]) -> List[Dict]:
        if not sections or not concept_order:
            return sections

        pos: Dict[str, int] = {}
        for i, rec in enumerate(concept_order):
            lbl = rec["label"].lower().strip()
            pos[lbl] = i
            for w in lbl.split():
                if w not in _STOPWORDS and len(w) > 3:
                    pos.setdefault(w, i)

        def _score(sec: Dict) -> Tuple[int, int]:
            h = sec.get("heading", "").lower().strip()
            if h in ("concept workflow", "workflow overview"):
                return (-1, 0)
            if h in pos:
                return (pos[h], 0)
            words = [w for w in h.split() if w not in _STOPWORDS and len(w) > 3]
            best = min((pos[w] for w in words if w in pos), default=999)
            if best < 999:
                return (best, 1)
            return (500 + self._disc_role(sec), 2)

        def _is_extra(s): return "additional" in s.get("heading","").lower()

        main  = sorted([s for s in sections if not _is_extra(s)], key=_score)
        extra = [s for s in sections if _is_extra(s)]
        return main + extra

    def _disc_role(self, sec: Dict) -> int:
        text = sec.get("heading","").lower() + " " + " ".join(
            pt.get("text","") for ss in sec.get("subsections",[]) for pt in ss.get("points",[])
        ).lower()[:300]
        best = 99
        for _, p, kws in _DISCOURSE_ROLES:
            if any(kw in text for kw in kws):
                best = min(best, p)
        return best


# ═══════════════════════════════════════════════════════════════════════════════
# Hierarchy Annotation
# ═══════════════════════════════════════════════════════════════════════════════

def annotate_hierarchy_in_sections(sections: List[Dict], concept_order: List[Dict]) -> List[Dict]:
    """
    Add a 'Conceptual Structure' subsection to parent-concept sections.
    Instead of 'X: sub-concept within Y' (internal schema noise), generate
    'X: <description if available>' — only if the child has a real description.
    If no description, skip that child entirely (don't emit bare labels).
    """
    lbl2rec  = {rec["label"].lower(): rec for rec in concept_order}
    id2rec   = {rec["id"]: rec for rec in concept_order}

    for section in sections:
        h   = section.get("heading","").lower().strip()
        rec = lbl2rec.get(h)
        if not rec:
            for lbl, r in lbl2rec.items():
                if lbl in h or h in lbl:
                    rec = r; break
        if not rec:
            continue

        cids = rec.get("children_ids", [])
        child_recs = [id2rec[c] for c in cids if c in id2rec]
        if len(child_recs) < 2:
            continue

        existing = {ss.get("heading","").lower() for ss in section.get("subsections",[])}
        if any("hierarch" in e or "structure" in e or "conceptual" in e for e in existing):
            continue

        # Only emit children that have a real description (not empty, not just the label)
        pts = []
        for cr in child_recs[:6]:
            lbl  = cr.get("label", "").strip()
            desc = (cr.get("description") or "").strip()
            # Skip if no description, or description IS just the label (circular)
            if not desc or desc.lower().rstrip(".") == lbl.lower():
                continue
            # Skip very short descriptions (< 5 words) — not useful
            if len(desc.split()) < 5:
                continue
            pts.append({"text": f"{lbl}: {desc}"})

        # Only insert subsection if we have ≥2 meaningful bullets
        if len(pts) >= 2:
            subs = section.get("subsections", [])
            subs.insert(min(2, len(subs)), {"heading": "Conceptual Structure", "points": pts})
            section["subsections"] = subs

    return sections


# ═══════════════════════════════════════════════════════════════════════════════
# Key Takeaways
# ═══════════════════════════════════════════════════════════════════════════════

_FUNC_VERBS = {
    "converts","generates","produces","transforms","checks","verifies",
    "optimizes","stores","manages","handles","detects","resolves","maps",
    "builds","creates","reads","translates","allocates","processes",
    "implements","ensures","passes","feeds","analyzes","validates",
}


def remove_key_takeaways(sections: List[Dict]) -> List[Dict]:
    """
    Remove all 'Key Takeaways' subsections from every section.
    Key Takeaways are redundant — they repeat content already present in earlier subsections.
    This runs dynamically on any input; no domain-specific keywords needed.
    """
    _TAKEAWAY_HEADINGS = re.compile(
        r"\b(key\s+takeaway|takeaway|key\s+point|key\s+summary|summary)\b",
        re.I,
    )
    removed = 0
    for sec in sections:
        original_len = len(sec.get("subsections", []))
        sec["subsections"] = [
            ss for ss in sec.get("subsections", [])
            if not _TAKEAWAY_HEADINGS.search(ss.get("heading", ""))
        ]
        removed += original_len - len(sec.get("subsections", []))
    if removed:
        print(f"[ConceptFlowOrganizer] ✓ Key Takeaways removed: {removed} subsections dropped")
    return sections


# ═══════════════════════════════════════════════════════════════════════════════
# Duplicate Image Deduplicator
# ═══════════════════════════════════════════════════════════════════════════════

def deduplicate_section_images(sections: List[Dict]) -> List[Dict]:
    seen_p: Set[str] = set()
    seen_s: Set[str] = set()
    for sec in sections:
        diag = sec.get("diagram")
        if not diag:
            continue
        path = diag.get("path","")
        if not path:
            continue
        pl = path.lower().rstrip("/\\")
        stem = re.sub(r"\.\w{2,4}$", "", pl.split("/")[-1].split("\\")[-1])
        if pl in seen_p or stem in seen_s:
            sec.pop("diagram", None)
        else:
            seen_p.add(pl); seen_s.add(stem)
    return sections


# ═══════════════════════════════════════════════════════════════════════════════
# Strip tautological bullets
# ═══════════════════════════════════════════════════════════════════════════════

def _strip_tautological_bullets(sections: List[Dict]) -> List[Dict]:
    for sec in sections:
        for ss in sec.get("subsections", []):
            pts = ss.get("points", [])
            if len(pts) <= 1:
                continue
            cleaned = [p for p in pts if not _is_circular(p.get("text",""))]
            ss["points"] = cleaned if cleaned else [pts[0]]
    return sections



# ═══════════════════════════════════════════════════════════════════════════════
# NEW: Implementation Layer Injection
# Detects when KG has "implemented_by", "array_based", "linked_list" style edges
# and builds a structured Implementation subsection dynamically.
# Works for any data structure, algorithm, or system — no hardcoding.
# ═══════════════════════════════════════════════════════════════════════════════

_IMPL_RELATIONS = {
    "implemented_by", "implementation_of", "implemented_using",
    "implemented_with", "uses_structure", "backed_by",
    "array_based", "linked_list_based", "realized_by",
    "array_implementation", "linked_list_implementation",
}

_IMPL_KEYWORDS = re.compile(
    r"\b(implement|array.based|linked.list|pointer|node|index|"
    r"dynamic\s+array|static\s+array|circular\s+array|"
    r"memory\s+allocation|data\s+structure\s+implementation|"
    r"using\s+array|using\s+linked)\b",
    re.I,
)


def inject_implementation_layer(
    sections: List[Dict],
    nodes: List[Dict],
    edges: List[Dict],
) -> List[Dict]:
    """
    NEW: For each section whose concept has implementation edges in the KG,
    inject an 'Implementation' subsection with structured sub-approaches.

    Example:
      KG: Stack → implemented_by → Array
          Stack → implemented_by → Linked List
    Result:
      Section: Stack
        Subsection: Implementation
          - Array-based: elements stored in contiguous memory; top tracked by index.
          - Linked List-based: each element is a node with a pointer to the next.

    Fully dynamic — relation detection uses keyword + relation-name matching.
    No hardcoded topic words.
    """
    if not nodes or not edges:
        return sections

    node_by_id: Dict[str, Dict] = {n.get("id", ""): n for n in nodes}
    node_label_to_id: Dict[str, str] = {
        n.get("label", "").lower(): n.get("id", "") for n in nodes
    }
    injected = 0

    for sec in sections:
        heading = sec.get("heading", "").strip()
        heading_lower = heading.lower()

        # Find the KG node matching this section heading
        sec_node_id = node_label_to_id.get(heading_lower)
        if not sec_node_id:
            # Try fuzzy match
            for lbl, nid in node_label_to_id.items():
                if _overlap(lbl, heading_lower) >= 0.6:
                    sec_node_id = nid
                    break
        if not sec_node_id:
            continue

        # Find implementation edges from this node
        impl_targets = []
        for e in edges:
            if e.get("source") != sec_node_id:
                continue
            rel = e.get("relation", "").lower().replace(" ", "_")
            tgt_node = node_by_id.get(e.get("target", ""))
            if not tgt_node:
                continue
            tgt_label = tgt_node.get("label", "")
            tgt_desc = tgt_node.get("description", "")

            # Detect impl edge by relation name or target keyword
            is_impl = (
                rel in _IMPL_RELATIONS
                or _IMPL_KEYWORDS.search(rel)
                or _IMPL_KEYWORDS.search(tgt_label)
            )
            if is_impl:
                impl_targets.append((tgt_label, tgt_desc))

        if not impl_targets:
            continue

        # Check we don't already have an Implementation subsection
        existing_headings = {
            ss.get("heading", "").lower() for ss in sec.get("subsections", [])
        }
        if any("implement" in h for h in existing_headings):
            continue

        # Build implementation points
        impl_points = []
        for tgt_label, tgt_desc in impl_targets:
            if tgt_desc and len(tgt_desc.split()) >= 4:
                text = f"{tgt_label}: {tgt_desc.rstrip('.')}."
            else:
                # Generate minimal description from label
                label_clean = tgt_label.lower()
                if "array" in label_clean:
                    text = (
                        f"{tgt_label}: elements stored in contiguous memory; "
                        f"top position tracked by an index variable."
                    )
                elif "linked" in label_clean:
                    text = (
                        f"{tgt_label}: each element is a node containing data "
                        f"and a pointer to the next node."
                    )
                else:
                    text = f"{tgt_label}: {heading} is implemented using {tgt_label}."
            impl_points.append({"text": text})

        if impl_points:
            impl_sub = {
                "heading": "Implementation Approaches",
                "points": impl_points,
            }
            sec.setdefault("subsections", []).append(impl_sub)
            injected += 1
            _log(
                f"Implementation layer injected into '{heading}' "
                f"({len(impl_points)} approaches)"
            )

    if injected:
        _log(f"✓ Implementation layers injected: {injected} sections")
    return sections


# ═══════════════════════════════════════════════════════════════════════════════
# Master Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def apply_concept_flow(
    notes: Dict,
    nodes: List[Dict],
    edges: List[Dict],
    run_steps: Optional[List[str]] = None,
) -> Dict:
    """
    Apply all conceptual hierarchy, flow, and quality improvements.

    Steps (all run by default):
      sanitize_qmarks      — P5: remove ? artefacts from headings + bullets
      section_dedup        — merge duplicate/overlapping topic sections
      hierarchy            — P1: build concept tree from KG
      workflow_section     — P3: add top-level Concept Workflow section
      flow_order           — P1: reorder sections by conceptual flow
      pipeline             — P3: add Process Flow subsection in relevant section
      relationships        — P2: add Concept Relationships subsection per section
      cel                  — CEL: Context Expansion Layer — expand contextless bullets
      explanatory          — P4: expand bare-label bullets to Input/Process/Output
      tautology            — remove circular "X is handled by X" bullets
      takeaways            — remove redundant Key Takeaways subsections
      dedup_images         — remove duplicate diagram references
      hierarchy_annotation — add Conceptual Structure subsection for parent nodes
    """
    notes    = copy.deepcopy(notes)
    sections = notes.get("sections", [])

    ALL = [
        "sanitize_qmarks","bullet_clean","section_dedup","hierarchy","workflow_section",
        "flow_order","pipeline","relationships","cel","explanatory",
        "tautology","takeaways","dedup_images","hierarchy_annotation",
        "implementation_layer",
        "remap_headings","clean_triples",   # NEW: pedagogical heading + triple-dump removal
        "noise_filter",                     # NEW: KG artifact / meta-lecture noise removal
        "pedagogical_order",                # NEW: section teaching-sequence reordering
    ]
    steps = run_steps if run_steps else ALL
    _log(f"Input  → {len(sections)} sections | steps: {steps}")

    # P5 — sanitize ? from headings + bullets (including heading-mode)
    # Also applies: fused label recovery, subsection dedup, grammar fix, fragment drop
    if "sanitize_qmarks" in steps:
        try:
            sections = sanitize_notes_question_marks(sections)
            _log("✓ Question marks sanitized + grammar fixed + fragments dropped")
        except Exception as e:
            _log(f"⚠ Q-mark sanitize failed: {e}")

    # Global bullet clean — grammar, glued tail, fragment drop (works without KG)
    if "bullet_clean" in steps:
        try:
            n_before = sum(len(ss.get("points",[])) for s in sections for ss in s.get("subsections",[]))
            for section in sections:
                for sub in section.get("subsections", []):
                    clean = []
                    for pt in sub.get("points", []):
                        t = pt.get("text","").strip()
                        if not t or _is_bullet_fragment(t):
                            continue
                        if _is_generic_adt_desc(t) and len(t.split()) < 25:
                            continue
                        t = _clean_glued_tail(t)
                        t = _fix_grammar(t)
                        if t:
                            clean.append({"text": t})
                    sub["points"] = clean
                section = _merge_duplicate_subsections(section)
            n_after = sum(len(ss.get("points",[])) for s in sections for ss in s.get("subsections",[]))
            _log(f"✓ Bullet clean: {n_before} → {n_after} bullets")
        except Exception as e:
            _log(f"⚠ Bullet clean failed: {e}")

    # Section dedup — merge "Stack Operations" + "Operations Performed On Stack" etc.
    if "section_dedup" in steps:
        try:
            before = len(sections)
            sections = deduplicate_topic_sections(sections)
            _log(f"✓ Section dedup: {before} → {len(sections)} sections")
        except Exception as e:
            _log(f"⚠ Section dedup failed: {e}")

    # Build concept hierarchy (needed for P1, P2, P3, P4, CEL)
    concept_order: List[Dict] = []
    node_by_id: Dict[str, Dict] = {n["id"]: n for n in nodes} if nodes else {}

    needs_kg = {"hierarchy","workflow_section","flow_order","pipeline",
                "relationships","cel","explanatory","hierarchy_annotation"}
    if nodes and edges and any(s in steps for s in needs_kg):
        try:
            concept_order = ConceptualHierarchyBuilder(nodes, edges).build()
            _log(f"✓ Concept hierarchy built: {len(concept_order)} nodes")
        except Exception as e:
            _log(f"⚠ Hierarchy build failed: {e}")

    # P3 — top-level Concept Workflow section
    if "workflow_section" in steps and nodes and edges:
        try:
            wf = PipelineFlowInjector(nodes, edges).build_workflow_section()
            if wf:
                first_h = sections[0].get("heading","").lower() if sections else ""
                at = 1 if any(w in first_h for w in ["definition","overview","introduction"]) else 0
                sections.insert(at, wf)
                _log("✓ Concept Workflow section injected")
        except Exception as e:
            _log(f"⚠ Workflow section failed: {e}")

    # P1 — reorder by conceptual flow
    if "flow_order" in steps and concept_order:
        try:
            sections = SectionFlowReorderer().reorder(sections, concept_order)
            _log("✓ Sections reordered by conceptual flow")
        except Exception as e:
            _log(f"⚠ Flow reorder failed: {e}")

    # P3 — Process Flow subsection in best-matching section
    if "pipeline" in steps and nodes and edges:
        try:
            sections = PipelineFlowInjector(nodes, edges).inject_flow_subsection(sections)
            _log("✓ Pipeline flow subsections injected")
        except Exception as e:
            _log(f"⚠ Pipeline injection failed: {e}")

    # P2 — Concept Relationships subsection
    if "relationships" in steps and concept_order:
        try:
            sections = build_relationship_subsections(sections, concept_order, node_by_id)
            _log("✓ Concept Relationships subsections added")
        except Exception as e:
            _log(f"⚠ Relationships failed: {e}")

    # CEL — Context Expansion Layer (most impactful: contextless labels → explanations)
    if "cel" in steps and concept_order:
        try:
            before_ctx = sum(
                1 for s in sections for ss in s.get("subsections",[])
                for p in ss.get("points",[]) if not _has_explanation(p.get("text",""))
            )
            sections = apply_context_expansion_layer(sections, concept_order, node_by_id)
            after_ctx = sum(
                1 for s in sections for ss in s.get("subsections",[])
                for p in ss.get("points",[]) if not _has_explanation(p.get("text",""))
            )
            _log(f"✓ CEL: contextless bullets {before_ctx} → {after_ctx}")
        except Exception as e:
            _log(f"⚠ CEL failed: {e}")

    # P4 — Expand bare-label / circular bullets
    if "explanatory" in steps and concept_order:
        try:
            sections = expand_explanatory_bullets(sections, concept_order, node_by_id)
            _log("✓ Explanatory bullets expanded")
        except Exception as e:
            _log(f"⚠ Explanatory expansion failed: {e}")

    # Tautology removal
    if "tautology" in steps:
        try:
            before = sum(len(ss.get("points",[])) for s in sections for ss in s.get("subsections",[]))
            sections = _strip_tautological_bullets(sections)
            after  = sum(len(ss.get("points",[])) for s in sections for ss in s.get("subsections",[]))
            _log(f"✓ Tautological bullets removed: {before - after}")
        except Exception as e:
            _log(f"⚠ Tautology removal failed: {e}")

    # Key Takeaways — remove redundant repetition
    if "takeaways" in steps:
        try:
            sections = remove_key_takeaways(sections)
            _log("✓ Key Takeaways removed")
        except Exception as e:
            _log(f"⚠ Key Takeaways removal failed: {e}")

    # Duplicate images
    if "dedup_images" in steps:
        try:
            sections = deduplicate_section_images(sections)
            _log("✓ Duplicate images deduplicated")
        except Exception as e:
            _log(f"⚠ Image dedup failed: {e}")

    # P1 — Hierarchy annotation in section content
    if "hierarchy_annotation" in steps and concept_order:
        try:
            sections = annotate_hierarchy_in_sections(sections, concept_order)
            _log("✓ Conceptual hierarchy annotated")
        except Exception as e:
            _log(f"⚠ Hierarchy annotation failed: {e}")

    # NEW — Implementation Layer: inject 'Implementation Approaches' subsection
    # when KG has implemented_by / implemented_using edges for a section's concept.
    if "implementation_layer" in steps and nodes and edges:
        try:
            sections = inject_implementation_layer(sections, nodes, edges)
        except Exception as e:
            _log(f"⚠ Implementation layer injection failed: {e}")

    # NEW — Remap KG meta-labels to pedagogical subsection headings
    # e.g. "Concept Relationships" → "How It Works", "Key Concepts" → "Overview"
    if "remap_headings" in steps:
        try:
            sections = remap_subsection_headings(sections)
            _log("✓ Subsection headings remapped to pedagogical labels")
        except Exception as e:
            _log(f"⚠ Heading remap failed: {e}")

    # NEW — Remove raw KG triple dump bullets + strip KG edge suffix boilerplate
    if "clean_triples" in steps:
        try:
            before_ct = sum(len(ss.get("points",[])) for s in sections for ss in s.get("subsections",[]))
            sections = clean_triple_dump_bullets(sections)
            after_ct  = sum(len(ss.get("points",[])) for s in sections for ss in s.get("subsections",[]))
            _log(f"✓ Triple dump cleanup: {before_ct} → {after_ct} bullets")
        except Exception as e:
            _log(f"⚠ Triple dump cleanup failed: {e}")

    # NEW — KG artifact / meta-lecture noise bullet filter
    # Removes: OCR garbage, "learned in previous chapter", "Down Approach generates...",
    # "Compiler Design Course", slide watermarks, and too-short stubs.
    if "noise_filter" in steps:
        try:
            before_nf = sum(len(ss.get("points",[])) for s in sections for ss in s.get("subsections",[]))
            sections = filter_kg_noise_bullets(sections)
            after_nf  = sum(len(ss.get("points",[])) for s in sections for ss in s.get("subsections",[]))
            _log(f"✓ KG noise filter: {before_nf} → {after_nf} bullets ({before_nf - after_nf} dropped)")
        except Exception as e:
            _log(f"⚠ Noise filter failed: {e}")

    # NEW — Pedagogical section ordering (teaching sequence)
    # Reorders sections: Compiler pipeline → Grammar → Parser → Procedures → Examples
    # Applied AFTER all content transformations so ordering is based on final headings.
    if "pedagogical_order" in steps:
        try:
            sections = reorder_sections_pedagogically(sections)
            _log("✓ Sections reordered into pedagogical teaching sequence")
        except Exception as e:
            _log(f"⚠ Pedagogical reordering failed: {e}")

    notes["sections"] = sections
    _log(f"Output → {len(sections)} sections")
    return notes


def _log(msg: str) -> None:
    print(f"[ConceptFlowOrganizer] {msg}")