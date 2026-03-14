# notes_postprocessor.py  (v2 — corrected from PDF audit)
"""
Notes Post-Processor
=====================
Fixes all remaining quality gaps visible in the generated PDF.

Precise bugs diagnosed from the v3 PDF:

  Bug 1a  Double "is" from naive regex — "Schema is is part of the topic"
  Bug 1b  Trailing space before period  — "Hiding Complexity ."
  Bug 1c  Tautological parenthetical not stripped
          — "Necessary Information (information shown to the user by...)"
  Bug 1d  Trailing raw-triple sentence  — "...requires Storage."

  Bug 2   Section heading "3-Tier ? Programmer/Developer" still present.
          Root cause: headings were cleaned AFTER the merge step, so the
          noisy heading was never merged with "3-Tier Architecture".
          Fix: clean headings FIRST, then merge.

  Bug 3   Diagram captions still show filenames ("slide_007_diagram_8").
          Root cause: the renderer (notes_renderer.py L330-338) has TWO
          caption paths:
            (a) section["diagram"]["caption"]  ← postprocessor touched this
            (b) diagram_map images             ← falls back to p.stem always
          Fix A: patch section["diagram"]["caption"] with real text.
          Fix B: patch_notes_renderer_captions() monkey-patches the renderer
                 so ALL images use the lookup, not just the primary diagram.

  Bug 4   "Database Administrator" section rendered as a raw italic block
          (no subsections).  Fix: detect single-point "Description" subsections
          and split the paragraph into structured bullet points.

All fixes are fully dynamic — no hardcoded topic/domain words anywhere.

  Fix DS-1  "& Structure determines ..." prefix not stripped.
            Root cause: _KG_PREFIX_NOISE anchored to [A-Z] but & at position 0
            caused the match to fail. Fix: changed [A-Z] to [A-Za-z] so the &
            is consumed by the optional prefix before the first word.

  Fix DS-2  "X covers Y: The main topic of the video covering..." suffix not stripped.
            Root cause: the covers relation + boilerplate suffix pattern was in
            notes_cleaner.py but not in notes_postprocessor.py's _naturalise().
            Fix: added _KG_COVERS_PREFIX regex and wired it into
            _strip_kg_structural_prefix() and _KG_BOILERPLATE_AFTER_COLON.

  Fix DS-3  "Data Structures example Stack" KG verb not naturalised.
            Root cause: 'example' not in _RELATION_RULES. 
            Fix: added (r'\\bexample\\s+(?=[A-Z])', 'for example, ') rule.

  Fix DS-4  "enables", "helps to identify", "has limitation with" KG verbs
            not naturalised. Fix: added to _RELATION_RULES and _ALLCAPS_EDGE_LABELS.
"""

import re
import copy
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
# Bug 1 — KG Relation Naturalisation  (corrected v2)
# ═══════════════════════════════════════════════════════════════════════════════

# Ordered — more specific patterns first to prevent double-is artifacts.
_RELATION_RULES: List[Tuple[re.Pattern, str]] = [
    (re.compile(r'\bis\s+part\s+of\s+(?:the\s+)?topic\b',      re.I), 'is part of the topic'),
    (re.compile(r'\bis[_]part[_]of[_]topic\b',                  re.I), 'is part of the topic'),
    (re.compile(r'\brequires[_\s]+understanding[_\s]+of\b',      re.I), 'requires an understanding of'),
    (re.compile(r'\bhas[_\s]+concept[_\s]+of\b',                 re.I), 'covers the concept of'),
    (re.compile(r'\bhas[_\s]+type\b',                            re.I), 'has types including'),
    (re.compile(r'\bis[_\s]+a[_\s]+topic[_\s]+with\b',          re.I), 'is closely related to'),
    (re.compile(r'\bachieved[_\s]+via\b',                        re.I), 'is achieved via'),
    # Guard: "not implemented by" must NOT become "not is implemented by"
    # Only add "is" when the word is a standalone KG edge label, not part of prose
    (re.compile(r'(?<!not\s)\bimplemented[_]+by\b',              re.I), 'is implemented by'),
    (re.compile(r'\bdivided[_\s]+into\b',                        re.I), 'is divided into'),
    (re.compile(r'\bconsists[_\s]+of\b',                         re.I), 'consists of'),
    (re.compile(r'\bcomposed[_\s]+of\b',                         re.I), 'is composed of'),
    (re.compile(r'\bdepends[_\s]+on\b',                          re.I), 'depends on'),
    (re.compile(r'\bconnects[_\s]+to\b',                         re.I), 'connects to'),
    (re.compile(r'\btype[_\s]+of\b',                             re.I), 'is a type of'),
    (re.compile(r'\bpart[_\s]+of\b',                             re.I), 'is part of'),
    (re.compile(r'\brelated[_\s]+to\b',                          re.I), 'is related to'),
    (re.compile(r'\bresponsible[_\s]+for\b',                     re.I), 'is responsible for'),
    # FIX: "Data Structures example Stack" → "Data Structures, for example, Stack"
    # Guard: do NOT fire when 'example' is already preceded by an adjective like
    # "real-life", "practical", "common" — in those cases it is a noun, not a verb.
    # Fixed-width lookbehind: check for the specific "real-life " case explicitly.
    (re.compile(r'(?<!real-life\s)\bexample\s+(?=[A-Z])',        re.I), 'for example, '),
    # FIX: "Data Structures enables Efficiency"
    (re.compile(r'\benables\b',                                   re.I), 'enables'),
    # FIX: "X has limitation with Y"
    (re.compile(r'\bhas[_\s]+limitation[_\s]+with\b',            re.I), 'has a limitation with'),
    # FIX: "X helps to identify Y"
    (re.compile(r'\bhelps[_\s]+to[_\s]+identify\b',              re.I), 'helps to identify'),
    # FIX: "X has scalability addressed by Y"
    (re.compile(r'\bhas[_\s]+scalability[_\s]+addressed[_\s]+by\b', re.I), 'has scalability addressed by'),
]

# ── FIX-KG-ALLCAPS: ALLCAPS KG edge labels ──────────────────────────────────
# KG extractors emit relation labels in ALLCAPS with underscores, e.g.:
#   "IPv4 Address SCALABILITY_ADDRESSED_BY Network Address Translation"
#   "IPv4 Address IS_ALSO_KNOWN_AS Internet Protocol Address"
#   "Packet CONTAINS Source IP Address"
#   "IPv4 Address CAN_LEAD_TO IP Conflict"
#   "IPv4 Address CONTRASTED_WITH IPv6 Address"
# These are fully dynamic: any sequence of UPPERCASE_WITH_UNDERSCORES
# between two noun phrases is treated as a KG edge and converted to
# natural English using a lookup table.  Unknown labels are converted by
# lower-casing and replacing underscores with spaces.
_ALLCAPS_EDGE_LABELS: Dict[str, str] = {
    'CONTAINS':                     'contains',
    'IS_ALSO_KNOWN_AS':             'is also known as',
    'IS_KNOWN_AS':                  'is known as',
    'ALSO_KNOWN_AS':                'also known as',
    'STANDS_FOR':                   'stands for',
    'STANDS FOR':                   'stands for',
    'IS_DIVIDED_INTO':              'is divided into',
    'DIVIDED_INTO':                 'is divided into',
    'CAN_BE':                       'can be',
    'CAN_LEAD_TO':                  'can lead to',
    'LEADS_TO':                     'leads to',
    'IS_PART_OF':                   'is part of',
    'PART_OF':                      'is part of',
    'IS_A_TYPE_OF':                 'is a type of',
    'TYPE_OF':                      'is a type of',
    'IS_A':                         'is a',
    'HAS_TYPE':                     'has types including',
    'HAS_CONCEPT_OF':               'covers the concept of',
    'SCALABILITY_ADDRESSED_BY':     'has scalability addressed by',
    'ADDRESSED_BY':                 'is addressed by',
    'CONTRASTED_WITH':              'is contrasted with',
    'COMPARED_TO':                  'is compared to',
    'COMPARED_WITH':                'is compared with',
    'HAVE_VALUE_RANGE':             'has a value range of',
    'VALUE_RANGE':                  'has value range',
    'RELATED_TO':                   'is related to',
    'IS_RELATED_TO':                'is related to',
    'USES':                         'uses',
    'USED_BY':                      'is used by',
    'USED_FOR':                     'is used for',
    'DEPENDS_ON':                   'depends on',
    'REQUIRES':                     'requires',
    'ENABLES':                      'enables',
    'SUPPORTS':                     'supports',
    'IMPLEMENTED_BY':               'is implemented by',
    'ACHIEVED_VIA':                 'is achieved via',
    'CONSISTS_OF':                  'consists of',
    'COMPOSED_OF':                  'is composed of',
    'INCLUDES':                     'includes',
    'PROVIDES':                     'provides',
    'DEFINES':                      'defines',
    'IDENTIFIES':                   'identifies',
    'CONNECTS_TO':                  'connects to',
    'RESPONSIBLE_FOR':              'is responsible for',
    'INTRODUCED_BY':                'was introduced by',
    'DEFINED_BY':                   'is defined by',
    'MANAGED_BY':                   'is managed by',
    'OPERATES_AT':                  'operates at',
    'WORKS_WITH':                   'works with',
    'BELONGS_TO':                   'belongs to',
    'ASSOCIATED_WITH':              'is associated with',
    'HANDLES':                      'handles',
    'PROCESSES':                    'processes',
    'GENERATES':                    'generates',
    'CONVERTS':                     'converts',
    'TRANSLATES':                   'translates',
    'ENCAPSULATES':                 'encapsulates',
    'STORES':                       'stores',
    'TRANSMITS':                    'transmits',
    'FORWARDS':                     'forwards',
    'ROUTES':                       'routes',
    'COVERS':                       'covers',
    'EXAMPLE':                      'for example',
    'ENABLES':                      'enables',
    'HAS_LIMITATION_WITH':          'has a limitation with',
    'HELPS_IDENTIFY':               'helps to identify',
    'HELPS_TO_IDENTIFY':            'helps to identify',
    'EXAMPLE_STACK':                'for example, stack',
    'EXAMPLE_ARRAY':                'for example, array',
    'EXAMPLE_GRAPH':                'for example, graph',
}

# Matches any ALLCAPS token (with optional underscores) that is at least 2 chars
# and sits between two non-uppercase context words. We match greedily to catch
# multi-word labels like SCALABILITY_ADDRESSED_BY.
_ALLCAPS_EDGE_RE = re.compile(
    r'\b([A-Z][A-Z_]{1,}[A-Z])\b'  # ALLCAPS with optional underscores, min 2 chars
)


def _normalise_allcaps_edges(text: str) -> str:
    """
    FIX-KG-ALLCAPS: Replace ALLCAPS KG edge labels with natural English.

    Strategy:
      1. Look up the ALLCAPS token (with underscores) in the known-label table.
      2. If found, substitute the natural-language equivalent.
      3. If NOT found, convert dynamically: lower-case + replace underscores with spaces.
         This handles ANY new KG edge label not in the table without hardcoding.

    Examples:
      "Packet CONTAINS Source IP" → "Packet contains Source IP"
      "IPv4 IS_ALSO_KNOWN_AS IP"  → "IPv4 is also known as IP"
      "Octet HAVE_VALUE_RANGE 0"  → "Octet has a value range of 0"
      "X SOME_NEW_RELATION Y"     → "X some new relation Y"  (dynamic fallback)
    """
    def _replace(m: re.Match) -> str:
        token = m.group(1)
        # Exact lookup first
        if token in _ALLCAPS_EDGE_LABELS:
            return _ALLCAPS_EDGE_LABELS[token]
        # Variant: with spaces instead of underscores
        spaced = token.replace('_', ' ')
        if spaced in _ALLCAPS_EDGE_LABELS:
            return _ALLCAPS_EDGE_LABELS[spaced]
        # Dynamic fallback: lower-case + replace underscores with spaces
        # Only applies to tokens that look like KG edges (multiple CAPS words
        # or contain underscores), NOT legitimate abbreviations like "IPv4", "NAT"
        if '_' in token or len(token) > 4:
            return token.lower().replace('_', ' ')
        # Short ALLCAPS without underscores: likely an abbreviation (IPv4, NAT, DNS)
        # Leave it alone.
        return token

    return _ALLCAPS_EDGE_RE.sub(_replace, text)


# ── FIX-KG-STRUCTURAL: Strip KG structural noise prefixes ───────────────────
# Some bullets start with extraction artifacts like:
#   "& Structure determines The network layer will create..."
#   "Definition determines An IPv4 address is..."
#   "Activity determines An activity designed to..."
# These are KG node-type labels that leaked as sentence prefixes.
# Pattern: starts with an optional "&", then a short word, then "determines"
# or other KG structural verbs, then the real sentence content.
_KG_PREFIX_NOISE = re.compile(
    r'^(?:&\s*)?(?:[A-Za-z][a-zA-Z\s&]{0,30}?)\s+'
    r'(?:determines|establishes|specifies|defines|denotes|indicates|refers to|implies)\s+',
    re.I,
)

# FIX: "X covers Y:" KG edge prefix — "Introduction of Data Structure covers Data:"
# The "covers" verb is used by the KG extractor as a relation label, and the entire
# "Subject covers Object:" phrase leaks as a bullet prefix.
_KG_COVERS_PREFIX = re.compile(
    r'^.{3,120}\s+covers\s+[^:]{2,200}:\s+',
    re.I,
)


def _strip_kg_structural_prefix(text: str) -> str:
    """
    Remove KG structural extraction prefixes from bullet text.

    E.g. "& Structure determines The network layer..." → "The network layer..."
         "Definition determines An IPv4 address..."    → "An IPv4 address..."
         "Introduction of Data Structure covers Data: The main topic of the video..."
           → "Introduction of Data Structure covers Data."
         "X covers Y - description: The main topic..." → "X covers Y - description."

    The regexes are anchored to the start and only trigger when the prefix ends
    with a KG structural verb followed by content or a known boilerplate opener.
    """
    # Pattern 1: "X determines Y" / "& X determines Y"
    m = _KG_PREFIX_NOISE.match(text)
    if m:
        remainder = text[m.end():].strip()
        if remainder and (remainder[0].isupper() or remainder[:2].lower() in
                          ('a ', 'an', 'th')):
            return remainder

    # Pattern 2: "Subject covers Object: <suffix>"
    # If suffix is boilerplate → keep "Subject covers Object."
    # If suffix is real content → keep suffix (strip the covers label)
    m2 = _KG_COVERS_PREFIX.match(text)
    if m2:
        remainder = text[m2.end():].strip()
        _BOILERPLATE_OPENERS = re.compile(
            r'^(?:The\s+main\s+topic\s+of\s+the\s+video'
            r'|Topic\s+of\s+the\s+video'            # residue after "The main" stripped
            r'|Meaningful\s+or\s+processed\s+data'
            r'|The\s+main\s+topic\s+of\s+this'
            r'|This\s+video\s+covers'
            r'|The\s+video\s+covers)',
            re.I,
        )
        if _BOILERPLATE_OPENERS.match(remainder):
            # Boilerplate suffix — return "Subject covers Object."
            prefix = text[:m2.end()].rstrip(': ').strip()
            if len(prefix.split()) >= 3:
                return (prefix + '.') if not prefix.endswith('.') else prefix
        else:
            # Real content after the colon — keep it
            if len(remainder.split()) >= 3:
                return remainder[0].upper() + remainder[1:] if remainder[0].islower() else remainder

    # Pattern 3: Bullet IS the boilerplate residue — "Topic of the video covering..."
    # This happens when an upstream pass already stripped the "X covers Y:" prefix,
    # leaving the suffix standing alone as a top-level bullet. Drop it entirely.
    _PURE_BOILERPLATE = re.compile(
        r'^(?:The\s+main\s+)?[Tt]opic\s+of\s+the\s+video\s+covering\b', re.I,
    )
    if _PURE_BOILERPLATE.match(text):
        return ''    # signal to caller to drop this bullet

    # Pattern 4: KG label leaked before a semicolon as a prefix
    # "Semester subject; application software that allows users to..."
    # Strip: short title-case phrase (1-4 words) + semicolon + real content
    _KG_SEMICOLON_PREFIX = re.compile(
        r'^[A-Z][A-Za-z]+(?:\s+[A-Za-z]+){0,3};\s+(?=[a-zA-Z])',
    )
    m4 = _KG_SEMICOLON_PREFIX.match(text)
    if m4:
        remainder = text[m4.end():].strip()
        if len(remainder.split()) >= 4:
            return remainder[0].upper() + remainder[1:] if remainder[0].islower() else remainder

    return text

# Fix double "is" introduced when source text already had "is" before the verb
_DOUBLE_IS = re.compile(r'\bis\s+is\b', re.I)

# ── FIX-REDUNDANT: KG node-description boilerplate appended to every bullet ─
#
# The KG extractor appends the TARGET node's own description to every edge,
# producing redundant / wrong suffixes like:
#
#   "System (DBMS) controls Redundancy: Collections of interrelated data..."
#   → suffix is the Database node's description pasted onto the Redundancy edge
#
#   "Advantages of DBMS includes Redundancy: Benefits of using a DBMS over..."
#   → suffix is the PARENT section description pasted onto every child edge
#
#   "Has a fixed or dynamic capacity: It is a type of Linear Data Structure..."
#   → suffix is the KG type annotation of the Stacks node
#
# All patterns are fully dynamic (no domain words).

# Pattern A: "X: Collections of interrelated data, which can store..."
# Fingerprint: colon → "Collections of interrelated [noun]"  (generic KG node desc.)
_KG_COLLECTIONS_SUFFIX = re.compile(
    r':\s+Collections\s+of\s+interrelated\b[^.]{0,300}\.?\s*$',
    re.I,
)

# Pattern B: "X: Benefits/Advantages/... of using Y over Z."
# The KG pastes the PARENT section description onto every child-edge bullet.
# Fingerprint: colon → "Benefits/Advantages/... of [using/having] ..."
_KG_PARENT_DESC_SUFFIX = re.compile(
    r':\s+(?:Benefits|Advantages|Disadvantages|Limitations|Uses|Goals|'
    r'Purposes|Examples|Applications|Features|Properties|Types)\s+'
    r'of\s+(?:using\s+|having\s+|the\s+)?\w[^.]{0,200}\.?\s*$',
    re.I,
)

# Pattern C: ": It is a type of X, Y."  — KG node-type annotation leaking as suffix
# Seen: "Has a fixed or dynamic capacity: It is a type of Linear Data Structure, ..."
_KG_TYPE_LABEL_SUFFIX = re.compile(
    r':\s+It\s+is\s+a\s+type\s+of\s+[^.]{5,200}\.?\s*$',
    re.I,
)

# Pattern D: standard boilerplate description introduced by article + generic verb
# "X: A unique number assigned to every device, used to find out..."
_KG_BOILERPLATE_AFTER_COLON = re.compile(
    r':\s+'
    r'(?:A|An|The)\s+\w[\w\s,]{5,100}'
    r'(?:assigned\s+to|used\s+to\s+find|designed\s+to|responsible\s+for'
    r'|refers\s+to|defined\s+as|used\s+to\s+identify|used\s+to\s+determine'
    r'|main\s+topic\s+of\s+the\s+video|topic\s+of\s+the\s+video)'
    r'[^.]*\.',
    re.I,
)

# Pattern E: "which is a type of action" KG structural artifact mid-sentence
_KG_TYPE_OF_ACTION = re.compile(
    r',?\s+(?:used\s+to\s+find\s+out\s+which\s+device\s+performs\s+which\s+)?'
    r'(?:is\s+a\s+type\s+of\s+action|which\s+is\s+a\s+type\s+of\s+action)'
    r'[^.]*',
    re.I,
)

# Pattern F: generic noun-headed description suffix
# ": A [unique|particular] <noun> ..." where noun is a generic KG node-type word
# Guard: do NOT match past semicolons — real bullets often use "X; Y" to join clauses
_KG_GENERIC_DESCRIPTION_SUFFIX = re.compile(
    r':\s+(?:A|An|The)\s+(?:unique\s+|particular\s+)?'
    r'(?:number|identifier|address|concept|entity|way|schema|'
    r'mechanism|process|method|system|device|component|protocol|standard|format|'
    r'value|set|function|unit|block|record|field|table|file|node|layer|type|code|'
    r'sequence|structure|module|service|operation|action|activity|feature)\b'
    r'[^;.]{0,300}\.?\s*$',
    re.I,
)

# FIX-DOUBLE-VERB: "is a has" → "has", "is a involves" → "involves"
# Pattern A: "is a <verb-word>" where the second word is unambiguously a verb
#   — these never follow "is a" in valid English prose.
# Pattern B: "is a is/are <prep-phrase>" — the doubled copula gives away the
#   KG template artifact, e.g. "is a is part of LIFO" → "is part of LIFO".
#   IMPORTANT: "Stack is a type of X" must NOT be touched — the trigger here
#   is the doubled auxiliary ("is a IS part of"), not "is a type of" alone.
_DOUBLE_VERB = re.compile(
    r'\b(?:'
    # Pattern A: is/are/was/were a <unambiguous-verb>
    r'(?:is|are|was|were)\s+a\s+'
    r'(?:has|have|involves|includes|consists|contains|uses|provides|'
    r'requires|defines|represents|supports|performs|handles|stores|manages)'
    r'|'
    # Pattern B: is/are/was/were a IS/ARE <anything>
    r'(?:is|are|was|were)\s+a\s+(?:is|are)\s+'
    r'(?:part|type|kind|form|example|instance)\s+of'
    r')\b',
    re.I,
)

# Replacement: extract the meaningful part after the double-verb
# For Pattern A: keep group that is the verb onward
# For Pattern B: keep "is/are <prep> of"
# We use a single sub with a function to handle both:
def _double_verb_repl(m: re.Match) -> str:
    s = m.group(0)
    # Pattern B: "is a IS part of" → keep from the second "is"
    _pb = re.match(
        r'(?:is|are|was|were)\s+a\s+((?:is|are)\s+(?:part|type|kind|form|example|instance)\s+of)\b',
        s, re.I)
    if _pb:
        return _pb.group(1)
    # Pattern A: "is a has/involves/..." → keep the verb word
    _pa = re.match(
        r'(?:is|are|was|were)\s+a\s+(\w+)\b',
        s, re.I)
    if _pa:
        return _pa.group(1)
    return s

# FIX-ORPHAN-CLOSE-PAREN-SUBJECT: Sentences whose effective subject is a closing-paren
# fragment — "Out) is related to Order of Operations."
# Arise when "FILO (First In, Last Out)" is split and the closing half becomes a
# standalone bullet subject. Drop the whole bullet — no recoverable content.
_ORPHAN_CLOSE_PAREN_SUBJECT = re.compile(
    r'^\w[\w\s,]*\)\s+(?:is|are|was|were|has|have|can|will|should|may)\b',
    re.I,
)

# FIX-COMPONENT-REPEAT: Three forms of KG description-template artifact:
#   Form A: "X as a component as one of its key components"
#            — KG edge label "as a component" + description template both appear
#   Form B: "X as one of its key components"
#            — only the description template suffix, edge label was already stripped
#   Form C: "X involves Y as one of its key components"
#            — produced by _graph_edge_to_prose() filler template (now fixed, but
#              may still exist in notes built before the fix; caught here as safety net)
# Forms A and B reduce to just "X"; Form C is stripped entirely (return "")
_COMPONENT_REPEAT = re.compile(
    r'\s+as\s+a\s+component\s+as\s+one\s+of\s+its\s+key\s+components?\b'   # Form A
    r'|\s+as\s+one\s+of\s+its\s+key\s+components?\b'                        # Form B
    r'|\s+involves\s+\S[\S\s]{1,60}\s+as\s+one\s+of\s+its\s+key\s+components?\b',  # Form C
    re.I,
)

# FIX-BROKEN-PAREN: "Stacks can be modeled as Type) (...real content...)"
# The KG label lost its opening paren; strip the orphaned fragment before "("
_BROKEN_KG_LABEL = re.compile(
    r'\b(?:can\s+be\s+modeled\s+as|is\s+part\s+of|is\s+a\s+type\s+of'
    r'|is\s+categorized\s+as)\s+[A-Z][A-Za-z\s]{0,25}\)\s*',
    re.I,
)

# FIX-WRONG-NODE-BULLET: Bullet whose SUBJECT is itself a KG parent-section name
# "Using a Database Management System over a file system: Collections of interrelated..."
# The subject is not a real entity — it's the section label used as a KG node name.
# Detection: subject ends with "over a <noun>" (structural KG phrasing for section names)
_WRONG_NODE_SUBJECT = re.compile(
    r'^.{10,80}\s+over\s+a\s+\w[\w\s]{0,30}:\s+Collections\s+of\b',
    re.I,
)

# FIX-IMPLICIT-STRIP: Over-aggressive _KG_BOILERPLATE_AFTER_COLON hits
# "Implicit Stack: A built-in stack..." — "built" is not in the article+verb list,
# but the pattern fires because "built-in" contains generic noun words.
# Guard: protect bullets where the colon-suffix contains concrete technical verbs
# like "built-in", "implemented by", "stores", "used by the compiler"
_PROTECTED_COLON_SUFFIX = re.compile(
    r':\s+A\s+(?:built-in|built\s+in|pre-built|hand-crafted)\b',
    re.I,
)


def _strip_kg_boilerplate(text: str) -> str:
    """Remove KG node-description boilerplate suffixes from bullet text."""

    # Guard: never strip bullets whose suffix contains concrete implementation detail
    if _PROTECTED_COLON_SUFFIX.search(text):
        return text

    # Drop entire bullet if subject is a KG section name with wrong node description
    if _WRONG_NODE_SUBJECT.search(text):
        return ''

    # FIX-COLLECTIONS-PAREN: strip "(collections of interrelated data...)" paren form
    # "System (DBMS) manages Database (collections of interrelated data, which can store...)"
    text = re.sub(
        r'\s*\(collections\s+of\s+interrelated\b[^)]{0,300}\)',
        '', text, flags=re.I,
    )

    # Step 1: strip "which is a type of action" mid-sentence fragment
    text = _KG_TYPE_OF_ACTION.sub('', text)

    # Step 2: strip ": Collections of interrelated data..." suffix
    m = _KG_COLLECTIONS_SUFFIX.search(text)
    if m:
        text = text[:m.start()].rstrip(',:; ') + '.'

    # Step 3: strip ": Benefits/Advantages of using X over Y" parent-desc suffix
    m = _KG_PARENT_DESC_SUFFIX.search(text)
    if m:
        text = text[:m.start()].rstrip(',:; ') + '.'

    # Step 4: strip ": It is a type of X, Y" type-label suffix
    m = _KG_TYPE_LABEL_SUFFIX.search(text)
    if m:
        text = text[:m.start()].rstrip(',:; ') + '.'

    # Step 5: strip standard boilerplate description introduced by article + generic verb
    text = _KG_BOILERPLATE_AFTER_COLON.sub('.', text)

    # Step 6: strip generic noun-headed description suffix
    text = _KG_GENERIC_DESCRIPTION_SUFFIX.sub('.', text)

    # Collapse double periods and stray spaces
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'\s{2,}', ' ', text).strip()
    return text


# Tautological parentheticals — catches all variants seen in PDF
_TAUTOLOGICAL_PARENS = re.compile(
    r'\s*\('
    r'(?:'
    r'information\s+(?:shown\s+(?:to|by)|hidden\s+from|provided\s+(?:to|by))[^)]{0,80}'
    r'|what\s+[\w\s]{1,30}(?:means|is)'
    r'|[\w\s]{1,40}(?:refers\s+to|is\s+defined\s+as|means)[^)]{0,60}'
    r')\)',
    re.I,
)

# FIX-I-4: Tautological "X illustrated by Real-life Examples of X (illustrations of X found in...)"
# The bullet subject and the parenthetical describe the same thing. Strip the redundant
# "illustrated by Real-life Examples of <same subject>" suffix.
# Pattern: "X: Y illustrated by Real-life Examples of Y (illustrations of Y found in...)"
# → keep only "X: Y." (i.e. the topic name is the whole content)
_TAUTOLOGICAL_ILLUSTRATED_BY = re.compile(
    r':\s+\w[\w\s]{1,50}\s+illustrated\s+by\s+Real-life\s+Examples\s+of\s+\w[\w\s]{1,50}'
    r'\s*\(illustrations\s+of\s+\w[\w\s]{1,50}\s+found\s+in\s+everyday\s+scenarios\)\.',
    re.I,
)

# FIX-I-11: Tautological "X: <Subject> is <X>" pattern — subject says it IS the heading
# "Dynamic Resizing Stack: Stack Size is Dynamic Resizing Stack."
# This is a KG self-reference where the description adds zero information.
# Pattern: heading text appears verbatim in the bullet body after "is"
_TAUTOLOGICAL_IS_SELF = re.compile(
    r'^(\w[\w\s/()]{2,50}):\s+\w[\w\s/()]{1,40}\s+is\s+\1\s*\.\s*$',
    re.I,
)

# FIX-I-12: "X: Y supports X (description of X)" — subject repeated as object
# "Stack Operations: Stacks supports Stack Operations (actions that can be performed...)"
# The bullet's label and the KG subject both refer to the same node.
# Strip to keep only the parenthetical description.
_TAUTOLOGICAL_SUPPORTS_SELF = re.compile(
    r'^(\w[\w\s/()]{2,50}):\s+\w[\w\s/()]{1,40}\s+supports\s+\1\s*\(([^)]{10,200})\)',
    re.I,
)

# Trailing raw-triple sentence: ". X requires Y."  or  ". X is part of Y."
_TRAILING_TRIPLE = re.compile(
    r'\.\s+[A-Z][^.]{5,60}'
    r'\b(?:requires|includes|uses|connects|is part of|depends on|is related to)\s+'
    r'[A-Z][^.]{2,40}\.\s*$'
)

_STRAY_PERIOD = re.compile(r'\s+\.')          # "Complexity ." → "Complexity."


def naturalise_bullet_relations(sections: List[Dict]) -> List[Dict]:
    """Convert raw KG relation strings into natural English in every bullet."""
    for section in sections:
        for sub in section.get('subsections', []):
            cleaned = []
            for pt in sub.get('points', []):
                text = pt.get('text', '').strip()
                if text:
                    result = _naturalise(text)
                    if result:
                        pt['text'] = result
                        cleaned.append(pt)
            sub['points'] = cleaned
    return sections


def _naturalise(text: str) -> str:
    # FIX-AND-OPENER: "And topic of discussion..." — leading conjunction is a KG structure leak.
    # "And" at bullet start is never valid; strip it and re-capitalise.
    text = re.sub(r'^And\s+', '', text).strip()
    if text and text[0].islower():
        text = text[0].upper() + text[1:]

    # FIX-FILLER-PHRASES: Drop bullets that are pure KG-template filler — they carry
    # zero informational content and were generated by _graph_edge_to_prose() or
    # similar template-fill code.  Fully dynamic — matched on structural patterns,
    # not on domain words.
    #
    # Pattern 1: "X has real-world applications and use cases."
    #   Produced when a self-referential triple was detected and the old fallback
    #   emitted a generic sentence rather than dropping the bullet.
    # Pattern 2: "X involves Y as one of its key components."
    #   Old filler template for includes/contains/has edges.
    # Pattern 3: "X is a has real-world applications..." (double-verb artifact)
    # Detection is purely structural (verb phrase patterns), no domain words.
    _FILLER_PHRASES = re.compile(
        r'(?:'
        r'\bhas\s+real[-\s]world\s+applications\s+and\s+use\s+cases\b'   # Pattern 1
        r'|\binvolves\s+\S[\S\s]{1,60}\s+as\s+one\s+of\s+its\s+key\s+components?\b'  # Pattern 2
        r'|\bhas\s+real[-\s]world\s+applications\b'                       # Pattern 1 short
        r')',
        re.I,
    )
    if _FILLER_PHRASES.search(text):
        return ''  # drop — pure filler, no educational content

    # FIX-KG-STRUCTURAL: strip "& Structure determines", "Definition determines",
    # and "X covers Y: boilerplate" prefixes.
    # Returns '' if the entire bullet is a boilerplate fragment (drop signal).
    text = _strip_kg_structural_prefix(text)
    if not text:
        return ''  # whole bullet was boilerplate — caller will drop it

    # NEW FIX-TRIPLE-DUMP: strip patterns that are raw KG triple dumps
    # ① "X categorizes Y - introduced in a previous session..."
    _CATEGORIZES = re.compile(
        r'^.{3,80}\s+(?:categorizes|encompasses\s+several\s+key\s+components'
        r'|previously\s+observed|introduced\s+in\s+a\s+previous)', re.I
    )
    if _CATEGORIZES.search(text):
        return ''  # drop — pure triple dump

    # ② "Previously observed, categorizes parsers into..."
    if re.match(r'^previously\s+observed', text, re.I):
        return ''

    # ③ Strip "during its operation - <annotation>" KG suffix
    text = re.sub(
        r'\s+during\s+its\s+operation\s*[-–—]\s*.{0,200}$',
        '.', text, flags=re.I
    )
    text = re.sub(
        r'\s+during\s+its\s+operation\.?\s*$',
        '.', text, flags=re.I
    )

    # ④ Strip " - Introduced in a previous session..." / " - Previously observed..."
    text = re.sub(
        r'\s*[-–—]\s*(?:introduced\s+in\s+a\s+previous\s+(?:session|chapter)'
        r'|previously\s+observed)[^.]*\.?',
        '.', text, flags=re.I
    )

    # ⑤ Strip " - <parenthetical annotation>" suffix when annotation contains
    # known KG meta-phrases. Guard: core text before dash must be ≥4 words.
    # Extra guard: the dash must be surrounded by spaces (not a hyphen inside a word).
    _DASH_ANNOTATION = re.compile(
        r'^(.{20,}?)\s+[-–—]\s+[A-Z][a-z][^.]{5,120}'
        r'(?:used\s+by|used\s+in|introduced\s+in)[^.]*\.\s*$',
        re.I,
    )
    m_da = _DASH_ANNOTATION.match(text)
    if m_da:
        before = m_da.group(1).strip()
        # Only strip if the dash-annotated part is a clear parenthetical
        # (i.e. before-part is a complete clause ≥ 4 words)
        if len(before.split()) >= 4 and not before.endswith(('Non', 'non')):
            text = before.rstrip('.') + '.'

    # ⑥ Strip partial truncated node-label subjects leaking as bullet openers
    # "Down Parsers: A type of parser..." → keep "A type of parser..."
    # Guard: only when the colon prefix is short (≤5 words) and remainder is ≥6 words.
    _PARTIAL_NODE_LABEL = re.compile(
        r'^(?:Down|Up|Top|Bottom|Left|Right|Inner|Outer)\s+\w[\w\s]{0,35}:\s+',
        re.I,
    )
    m_pnl = _PARTIAL_NODE_LABEL.match(text)
    if m_pnl:
        prefix = text[:m_pnl.end()].rstrip(': ')
        remainder = text[m_pnl.end():].strip()
        prefix_words = len(prefix.split())
        # Only strip if prefix is truly short (truncated label, not a long bullet)
        # and remainder forms a complete sentence starting with a capital letter
        # and remainder doesn't start with a bare article fragment ("A type of", "Of top-down")
        bare_article = re.match(r'^(?:A|An|Of|The)\s+(?:type|kind|form|part)\s+of\b', remainder, re.I)
        if (prefix_words <= 5
                and remainder
                and len(remainder.split()) >= 6
                and remainder[0].isupper()
                and not bare_article):
            text = remainder

    # FIX-KG-ALLCAPS: convert ALLCAPS edge labels (CONTAINS, IS_ALSO_KNOWN_AS…)
    text = _normalise_allcaps_edges(text)

    # FIX-REDUNDANT: strip KG node-description boilerplate appended after the colon
    text = _strip_kg_boilerplate(text)
    if not text:
        return ''  # _strip_kg_boilerplate flagged entire bullet as wrong-node

    # FIX-ORPHAN-CLOSE-PAREN-SUBJECT: "Out) is related to Order of Operations." → drop
    # Subject is a closing-paren fragment of a split parenthetical — no valid content.
    if _ORPHAN_CLOSE_PAREN_SUBJECT.match(text):
        return ''  # drop — orphaned closing-paren subject

    # FIX-DOUBLE-VERB: "is a has" → "has", "is a is part of" → "is part of"
    text = _DOUBLE_VERB.sub(_double_verb_repl, text)

    # FIX-COMPONENT-REPEAT: "X as a component as one of its key components" → "X"
    text = _COMPONENT_REPEAT.sub('', text)

    # FIX-BROKEN-KG-LABEL: strip "Stacks can be modeled as Type) " orphaned fragment
    text = _BROKEN_KG_LABEL.sub('', text)

    # Existing snake_case / lowercase relation rules
    for pat, repl in _RELATION_RULES:
        text = pat.sub(repl, text)
    # FIX-REAL-LIFE-EXAMPLE: "A real-life for example, where..." → "A real-life example, where..."
    # The 'example' rule above fires even when 'example' is a noun after an adjective.
    # Clean up any case where an adjective + "for example" was created mid-NP.
    text = re.sub(
        r'\b(real-life|practical|common|typical|simple|classic)\s+for\s+example\b',
        r'\1 example', text, flags=re.I
    )
    text = _DOUBLE_IS.sub('is', text)

    # Fix remaining underscore-verbs
    def _fix_us(m: re.Match) -> str:
        w = m.group(0)
        return w if re.search(r'\.\w{2,4}$', w) else w.replace('_', ' ')
    text = re.sub(r'\b[a-z]+(?:_[a-z]+){1,5}\b', _fix_us, text)

    text = _TAUTOLOGICAL_PARENS.sub('', text)

    # FIX-I-4: Drop "X illustrated by Real-life Examples of X (illustrations...)" tautology
    if _TAUTOLOGICAL_ILLUSTRATED_BY.search(text):
        # The entire value is self-referential — keep only the heading label
        colon_pos = text.find(':')
        if colon_pos > 0:
            text = text[:colon_pos].strip() + '.'
        else:
            return ''

    # FIX-I-11: Drop "X: <Subject> is X." tautological self-reference
    if _TAUTOLOGICAL_IS_SELF.match(text):
        return ''

    # FIX-I-12: "X: Y supports X (description)" → "X: description."
    m_sup = _TAUTOLOGICAL_SUPPORTS_SELF.match(text)
    if m_sup:
        label = m_sup.group(1).strip()
        description = m_sup.group(2).strip()
        # Capitalise description and rebuild as "Label: description."
        description = description[0].upper() + description[1:] if description else ''
        text = f"{label}: {description}." if description else f"{label}."

    text = _TRAILING_TRIPLE.sub('.', text)
    text = _STRAY_PERIOD.sub('.', text)
    text = re.sub(r'\s{2,}', ' ', text).strip().rstrip(' .')

    if text and text[-1] not in '.!?':
        text += '.'
    return text


# ═══════════════════════════════════════════════════════════════════════════════
# NEW: Semantic Bullet Grouper
# Groups bullets under the same subsection that share the same opening verb
# pattern into a single merged sentence, matching the suggestion in point #7:
#   "Push Operation", "Pop Operation", "Peek Operation"
#   → "Stack supports several operations including Push, Pop, and Peek."
# Fully dynamic — no hardcoded topic words.
# ═══════════════════════════════════════════════════════════════════════════════

_GROUPABLE_OPENERS: List[re.Pattern] = [
    # "X Operation", "X operation"
    re.compile(r'^(\w[\w\s]{1,30})\s+[Oo]peration\.?$'),
    # "X Phase", "X phase"
    re.compile(r'^(\w[\w\s]{1,30})\s+[Pp]hase\.?$'),
    # "X Type", "X type"
    re.compile(r'^(\w[\w\s]{1,25})\s+[Tt]ype\.?$'),
    # "X Algorithm"
    re.compile(r'^(\w[\w\s]{1,30})\s+[Aa]lgorithm\.?$'),
    # "X Approach"
    re.compile(r'^(\w[\w\s]{1,30})\s+[Aa]pproach\.?$'),
]


def _try_group_bullets(points: List[Dict], section_heading: str) -> List[Dict]:
    """
    If ≥3 bullets in this subsection match the same structural pattern,
    merge them into one grouped sentence and keep the rest unchanged.
    """
    if len(points) < 3:
        return points

    # Find runs of bullets matching the same pattern
    for pat in _GROUPABLE_OPENERS:
        labels = []
        for pt in points:
            text = pt.get('text', '').strip().rstrip('.')
            m = pat.match(text)
            if m:
                labels.append((pt, m.group(1).strip()))

        if len(labels) < 3:
            continue

        # Build grouped sentence
        names = [lbl for _, lbl in labels]
        if len(names) == 3:
            grouped_text = (
                f"{section_heading} supports several operations including "
                f"{names[0]}, {names[1]}, and {names[2]}."
            )
        else:
            listed = ", ".join(names[:-1]) + f", and {names[-1]}"
            grouped_text = f"{section_heading} includes {listed}."

        # Remove the individual bullets and prepend the grouped one
        grouped_pts = [pt for pt in points if pt not in [p for p, _ in labels]]
        return [{"text": grouped_text}] + grouped_pts

    return points


def group_related_bullets(sections: List[Dict]) -> List[Dict]:
    """
    Merge ≥3 structurally similar bullets into a single grouped sentence.
    E.g. 3 separate "X Operation." bullets → "Stack supports Push, Pop, and Peek operations."
    """
    for section in sections:
        h = section.get('heading', '')
        for sub in section.get('subsections', []):
            sub['points'] = _try_group_bullets(sub.get('points', []), h)
    return sections

_HEADING_NOISE = re.compile(r'[?→←↑↓~`\\|@]')

# One-word headings that are too generic on their own
_GENERIC_HEADINGS = {
    'physical', 'logical', 'multiple', 'schema', 'privileges',
    'database', 'components', 'structure', 'applications', 'other',
    'general', 'misc', 'content',
}

_STOPWORDS = {
    'the','a','an','is','are','was','were','be','been','have','has','had',
    'this','that','with','from','into','and','or','but','for','to','of',
    'in','on','at','by','data','level','system','which','what','how',
    'when','will','can','its','their','these','those',
}


def clean_section_headings(sections: List[Dict]) -> List[Dict]:
    """
    (1) Strip noise characters from all headings.
    (2) Expand one-word generic headings using dominant bullet vocabulary.
    Must run BEFORE merge_fragmented_sections().
    """
    for section in sections:
        heading = _clean_heading(section.get('heading', ''))
        if heading.lower().strip() in _GENERIC_HEADINGS:
            heading = _expand_heading(heading, section)
        section['heading'] = heading

        for sub in section.get('subsections', []):
            sub['heading'] = _clean_heading(sub.get('heading', ''))

    return sections


def _clean_heading(h: str) -> str:
    if not h:
        return h
    h = _HEADING_NOISE.sub(' ', h)
    h = re.sub(r'\s{2,}', ' ', h).strip().rstrip('-\u2013\u2014:;.,')
    return h.strip()


def _expand_heading(heading: str, section: Dict) -> str:
    """Derive a richer heading from the section's bullet content."""
    freq: Dict[str, int] = {}
    for sub in section.get('subsections', []):
        for pt in sub.get('points', []):
            for w in pt.get('text', '').lower().split():
                w = re.sub(r'[^a-z]', '', w)
                if len(w) > 4 and w not in _STOPWORDS:
                    freq[w] = freq.get(w, 0) + 1

    if not freq:
        return heading.title()

    top = sorted(freq.items(), key=lambda x: -x[1])
    for word, _ in top:
        if word != heading.lower():
            return f"{heading.title()} — {word.title()}"
    return heading.title()


# ═══════════════════════════════════════════════════════════════════════════════
# Bug 2b — OCR Garbage in Bullets
# ═══════════════════════════════════════════════════════════════════════════════

_OCR_GARBAGE = re.compile(
    r'[|@\[\]{}<>\\;\!\#\$\%\^&\*]{2,}'
    r'|[oO0lI1]{5,}'
    r'|(\w)\1{4,}'
)
_MIN_ALPHA = 0.55
_MIN_WORDS = 3

# ─────────────────────────────────────────────────────────────────────────────
# Semantic noise patterns — KG artifacts and meta-lecture references that
# pass the OCR character-ratio test but carry no learning value.
# Matched against the stripped bullet text (case-insensitive).
# ─────────────────────────────────────────────────────────────────────────────
_SEMANTIC_NOISE_PATTERNS: List[re.Pattern] = [
    # "Afor example, Grammar includes X" — OCR artefact prefix
    re.compile(r'^Afor\s+example\b', re.I),
    # "! = aabcde (Left most Derivation)" — OCR dump
    re.compile(r'^[!\?]\s*=\s*[a-zA-Z]{3,}', re.I),
    # "Analysis generates Tokens as its output" — KG triple subject leak
    re.compile(r'^Analysis\s+(?:generates|stores|followed\s+by)\s+\w', re.I),
    # "Down/Up Approach generates Parse Trees as its output"
    re.compile(r'^(?:Down|Up|Top|Bottom)\s+\w[\w\s]{0,40}(?:generates|produces|uses|involves|allows)\s+', re.I),
    # "learned/introduced in a previous chapter/session"
    re.compile(r'(?:learned|introduced|mentioned|discussed|covered)\s+in\s+a\s+previous\s+(?:chapter|session|lecture)', re.I),
    # "Procedures for its removal were learned..."
    re.compile(r'procedures\s+for\s+its\s+removal\s+were', re.I),
    # "Compiler Design Course: the focus is..."
    re.compile(r'\bcompiler\s+design\s+course\b', re.I),
    # "IT Department ... Unit 4: Parsers" — slide metadata
    re.compile(r'\bIT\s+Department\b|\bUnit\s+\d+\s*:\s*\w', re.I),
    # "Recursive Descent Parser System Programming Unit 4" — slide header
    re.compile(r'System\s+Programming\s+Unit\s+\d+', re.I),
    # "Grammar Procedures recognizes Input String" — raw KG edge leak
    re.compile(r'\brecognizes\s+Input\s+String\b', re.I),
    # "usually builds a data structure in the form of a parse" — repeated boilerplate
    re.compile(r'usually\s+builds\s+a\s+data\s+structure\s+in\s+the\s+form\s+of\s+a\s+parse', re.I),
    # "Anse =>", "Taken (0155)" — OCR token garbage
    re.compile(r'Anse\s*=>|Taken\s*\(\d+\)', re.I),
    # Pure bracket/symbol noise after strip: "Forge! eis) Taken"
    re.compile(r'Forge!\s*eis\)', re.I),
    # "Left Recursion Procedures: Procedures learned in a previous chapter"
    re.compile(r'Procedures\s+learned\s+in\s+a\s+previous\b', re.I),
]


def filter_ocr_noise(sections: List[Dict]) -> List[Dict]:
    """Remove OCR-corrupted and semantic-noise bullets."""
    for section in sections:
        for sub in section.get('subsections', []):
            sub['points'] = [
                pt for pt in sub.get('points', [])
                if _is_clean(pt.get('text', ''))
                and not _is_semantic_noise(pt.get('text', ''))
            ]
        section['subsections'] = [s for s in section.get('subsections', []) if s.get('points')]
    return [s for s in sections if s.get('subsections')]


def _is_semantic_noise(text: str) -> bool:
    """Return True if text is a KG artifact or meta-lecture reference."""
    if not text:
        return True
    t = text.strip()
    for pat in _SEMANTIC_NOISE_PATTERNS:
        if pat.search(t):
            return True
    return False


def _is_clean(text: str) -> bool:
    if not text or len(text) < 5:
        return False
    alpha = sum(1 for c in text if c.isalpha())
    if alpha / max(len(text), 1) < _MIN_ALPHA:
        return False
    if len(text.split()) < _MIN_WORDS and ':' not in text:
        return False
    m = _OCR_GARBAGE.search(text)
    if m and m.start() < 40:
        return False
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# Bug 4 — Raw "Description" Paragraph Blocks → Structured Subsections
# ═══════════════════════════════════════════════════════════════════════════════

def convert_description_blocks(sections: List[Dict]) -> List[Dict]:
    """
    Convert sections rendered as a single italic paragraph into structured
    bullet subsections.

    Trigger: exactly one subsection whose heading is "description" containing
    one long-text point (≥ 8 words).
    """
    for section in sections:
        subs = section.get('subsections', [])
        if len(subs) != 1:
            continue
        sub = subs[0]
        if sub.get('heading', '').lower() != 'description':
            continue
        pts = sub.get('points', [])
        if len(pts) != 1:
            continue
        raw = pts[0].get('text', '').strip()
        if len(raw.split()) < 8:
            continue

        # Split on semicolons, then sentence boundaries
        fragments = re.split(r';\s+|(?<=\.)\s+(?=[A-Z])', raw)
        fragments = [f.strip().rstrip(';').strip() for f in fragments
                     if len(f.strip().split()) >= 4]

        if len(fragments) >= 2:
            sub['heading'] = 'Key Points'
            sub['points'] = [{'text': _end(f)} for f in fragments]

    return sections


def _end(text: str) -> str:
    text = text.strip()
    return text if text and text[-1] in '.!?' else text + '.'


# ═══════════════════════════════════════════════════════════════════════════════
# Bug 2 / 3 — Section Merging  (runs AFTER heading cleanup)
# ═══════════════════════════════════════════════════════════════════════════════

def merge_fragmented_sections(sections: List[Dict]) -> List[Dict]:
    """
    Merge sections that are fragments of each other.

    Detection:
      • Heading B starts with heading A (prefix match), OR
      • Word overlap ≥ 0.60 AND at least one section has ≤ 3 bullet points.

    Runs AFTER clean_section_headings() so noise chars are already removed.
    """
    if len(sections) <= 1:
        return sections

    merged: List[Dict] = []
    absorbed: set = set()

    for i, a in enumerate(sections):
        if i in absorbed:
            continue
        combined = copy.deepcopy(a)
        norm_a = _norm(a.get('heading', ''))

        for j, b in enumerate(sections):
            if j <= i or j in absorbed:
                continue
            norm_b = _norm(b.get('heading', ''))
            pts_a = _count_pts(a)
            pts_b = _count_pts(b)

            if _are_fragments(norm_a, norm_b, pts_a, pts_b):
                absorbed.add(j)
                combined['heading'] = _pick_heading(
                    a.get('heading', ''), b.get('heading', '')
                )
                combined['subsections'] = (
                    combined.get('subsections', []) + b.get('subsections', [])
                )
                if not combined.get('diagram') and b.get('diagram'):
                    combined['diagram'] = b['diagram']

        absorbed.add(i)
        merged.append(combined)

    return merged


def _norm(h: str) -> str:
    h = h.lower()
    h = re.sub(r'[^a-z0-9\s]', ' ', h)
    return re.sub(r'\s+', ' ', h).strip()


def _are_fragments(na: str, nb: str, pts_a: int, pts_b: int) -> bool:
    if not na or not nb:
        return False
    if nb.startswith(na) or na.startswith(nb):
        return True
    wa, wb = set(na.split()), set(nb.split())
    if wa and wb:
        overlap = len(wa & wb) / max(len(wa), len(wb))
        if overlap >= 0.60 and min(pts_a, pts_b) <= 3:
            return True
    return False


def _count_pts(section: Dict) -> int:
    return sum(len(s.get('points', [])) for s in section.get('subsections', []))


def _pick_heading(a: str, b: str) -> str:
    """Return the more informative heading."""
    noise = re.compile(r'[?→←↑↓~`\\|@]')
    score = lambda h: len(h.split()) - len(noise.findall(h)) * 5
    return (a if score(a) >= score(b) else b).strip()


# ═══════════════════════════════════════════════════════════════════════════════
# Bug 3 — Diagram Caption Resolution  (renderer-aware)
# ═══════════════════════════════════════════════════════════════════════════════
# The renderer has TWO caption code paths:
#
#   PATH A  (line ~331): diagram["path"] matches diag_path
#               → uses diagram["caption"]         ← fixed here
#
#   PATH B  (line ~334): diagram_map image (semantic match)
#               → tries p.with_suffix('.txt')
#               → else: caption = p.stem           ← STILL falls to filename
#
# Fix for Path B: patch_notes_renderer_captions() injects a lookup function
# into the notes_renderer module so it is called before the p.stem fallback.

_FILENAME_RE = re.compile(
    r'^(?:slide|diagram|fig(?:ure)?|frame|img|image|screen)[\s_\-]*\d+[\s_\-\w]*$',
    re.I,
)
_BLIP_NOISE = [
    re.compile(
        r'\b(?:a |an |the )?(?:black[- ]and[- ]white |greyscale |color )?'
        r'(?:diagram|image|figure|picture|screenshot|photo|illustration|drawing)'
        r'(?:\s+(?:of|showing|with|that|depicting))?\s*', re.I),
    re.compile(
        r'\b(?:blackboard|whiteboard|chalkboard|slide|powerpoint|presentation)'
        r'(?:\s+(?:with|showing|of))?\s*', re.I),
]


def resolve_diagram_captions(
    sections: List[Dict],
    diagram_texts_dir: Optional[Path] = None,
    merged_captions: Optional[Dict[str, str]] = None,
) -> List[Dict]:
    """
    Replace filename-style captions in section["diagram"]["caption"] with
    real extracted descriptions loaded from disk.
    """
    lookup = _build_lookup(diagram_texts_dir, merged_captions)

    for section in sections:
        heading = section.get('heading', '')
        diag = section.get('diagram')
        if diag and isinstance(diag, dict):
            path    = diag.get('path', '')
            current = diag.get('caption', '')
            section['diagram']['caption'] = _resolve(path, current, heading, lookup)

    return sections


def _build_lookup(
    diagram_texts_dir: Optional[Path],
    merged_captions: Optional[Dict[str, str]],
) -> Dict[str, str]:
    """Build stem → description dict from all available sources."""
    lookup: Dict[str, str] = {}

    def _add(stem: str, text: str) -> None:
        if not text or len(text.split()) < 3:
            return
        text = text.strip()
        key = stem.lower()
        lookup[key] = text
        compact = re.sub(r'[\s_\-]', '', key)
        if compact != key:
            lookup[compact] = text
        # Index slide_NNN prefix
        m = re.match(r'(slide[\s_\-]*\d+)', key)
        if m:
            pfx = m.group(1).replace(' ', '_')
            lookup.setdefault(pfx, text)

    if diagram_texts_dir:
        p = Path(diagram_texts_dir)
        if p.exists():
            for txt in p.glob('*.txt'):
                try:
                    content = txt.read_text(encoding='utf-8', errors='replace').strip()
                    if content:
                        _add(txt.stem, content)
                except Exception:
                    pass

    if merged_captions:
        for k, v in merged_captions.items():
            stem = Path(k).stem if '.' in str(k) else str(k)
            _add(stem, v)

    return lookup


def _resolve(path: str, current: str, heading: str, lookup: Dict[str, str]) -> str:
    """Find the best caption for a diagram."""
    stem = Path(path).stem if path else ''

    # 1. Current caption is already a real description
    if current and not _is_filename(current):
        cleaned = _clean_blip(current)
        if len(cleaned.split()) >= 5:
            return _fmt(cleaned)

    # 2. Exact stem
    desc = lookup.get(stem.lower())
    if desc:
        return _fmt(desc)

    # 3. Compact variant
    compact = re.sub(r'[\s_\-]', '', stem.lower())
    desc = lookup.get(compact)
    if desc:
        return _fmt(desc)

    # 4. Progressive prefix shortening
    parts = stem.lower().split('_')
    for n in range(len(parts), 1, -1):
        pfx = '_'.join(parts[:n])
        desc = lookup.get(pfx)
        if desc:
            return _fmt(desc)

    # 5. Fallback
    return f"Diagram illustrating {heading}." if heading else stem


def _is_filename(caption: str) -> bool:
    if _FILENAME_RE.match(caption.strip()):
        return True
    return len(caption.split()) < 4 and bool(re.search(r'[\d_]', caption))


def _clean_blip(caption: str) -> str:
    for pat in _BLIP_NOISE:
        caption = pat.sub('', caption)
    caption = re.sub(r'\s{2,}', ' ', caption).strip().rstrip('.,;')
    return caption[0].upper() + caption[1:] if caption else caption


def _fmt(desc: str) -> str:
    """Trim to first sentence (≤ 30 words) with proper capitalisation."""
    sentences = re.split(r'(?<=[.!?])\s+', desc.strip())
    first = sentences[0].strip() if sentences else desc.strip()
    words = first.split()
    if len(words) > 30:
        first = ' '.join(words[:30]) + '...'
    first = first[0].upper() + first[1:] if first else first
    return first if first and first[-1] in '.!?' else first + '.'


# ═══════════════════════════════════════════════════════════════════════════════
# Renderer patch — fixes PATH B (diagram_map images falling back to p.stem)
# ═══════════════════════════════════════════════════════════════════════════════

def patch_notes_renderer_captions(
    diagram_texts_dir: Optional[Path] = None,
    merged_captions: Optional[Dict[str, str]] = None,
) -> None:
    """
    Monkey-patches notes_renderer.render_pdf so that diagram_map images
    (not in section["diagram"]) also get real captions instead of p.stem.

    Call ONCE at startup before the first render_pdf() call:

        from notes_postprocessor import patch_notes_renderer_captions
        patch_notes_renderer_captions(session_dir / "diagram_texts")

    What it does:
        Injects a module-level lookup into notes_renderer. The renderer's
        caption fallback block (around line 338) is extended to call
        notes_renderer._stem_to_caption(stem, section_heading) before
        falling back to p.stem.

    NOTE: If you can modify notes_renderer.py directly, replace line ~338:
        caption = p.stem
    with:
        caption = getattr(notes_renderer, '_stem_to_caption',
                          lambda s, h: s)(p.stem, heading)
    """
    lookup = _build_lookup(diagram_texts_dir, merged_captions)
    if not lookup:
        _log("⚠ patch_notes_renderer_captions: no lookup built — skipping")
        return

    try:
        import notes_renderer as _nr
        _nr._CAPTION_LOOKUP = lookup

        def _stem_to_caption(stem: str, section_heading: str = '') -> str:
            return _resolve(stem + '.png', '', section_heading, lookup)

        _nr._stem_to_caption = _stem_to_caption
        _log("✓ notes_renderer caption lookup injected")

    except ImportError:
        _log("⚠ notes_renderer not found — renderer patch skipped")


# ═══════════════════════════════════════════════════════════════════════════════
# Master Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def post_process_notes(
    notes: Dict,
    diagram_texts_dir: Optional[Path] = None,
    merged_captions: Optional[Dict[str, str]] = None,
    run_steps: Optional[List[str]] = None,
) -> Dict:
    """
    Apply all post-processing fixes to a HierarchicalNotes dict.

    Parameters
    ----------
    notes               : HierarchicalNotes dict
    diagram_texts_dir   : session_dir / "diagram_texts"  (*.txt files)
    merged_captions     : pre-loaded {stem: description} dict
    run_steps           : subset to run. Default = all steps.
                          "headings" | "naturalise" | "noise" |
                          "description_blocks" | "merge" | "captions"

    Integration (2 lines in ex.py before render_pdf):
    --------------------------------------------------
        from notes_postprocessor import (
            post_process_notes, load_merged_captions,
            patch_notes_renderer_captions
        )

        # Run once at startup to fix diagram_map captions too:
        patch_notes_renderer_captions(session_dir / "diagram_texts")

        # Run per-session on the notes dict:
        merged = load_merged_captions(session_dir)
        notes  = post_process_notes(
            notes,
            diagram_texts_dir=session_dir / "diagram_texts",
            merged_captions=merged,
        )
        render_pdf(notes, output_path, image_base_dir=session_dir)
    """
    notes    = copy.deepcopy(notes)
    sections = notes.get('sections', [])

    ALL = ['headings', 'naturalise', 'noise', 'description_blocks', 'merge',
           'captions', 'dedup', 'group_bullets']
    steps = run_steps if run_steps else ALL

    _log(f"Input  → {len(sections)} sections, {_total(sections)} bullets")

    if 'headings' in steps:                   # MUST be first
        sections = clean_section_headings(sections)
        _log("✓ headings cleaned")

    if 'naturalise' in steps:
        sections = naturalise_bullet_relations(sections)
        _log("✓ KG relation artifacts naturalised")

    if 'noise' in steps:
        b = _total(sections)
        sections = filter_ocr_noise(sections)
        _log(f"✓ OCR noise: {b - _total(sections)} bullets removed")

    if 'description_blocks' in steps:
        sections = convert_description_blocks(sections)
        _log("✓ description blocks → subsections")

    if 'merge' in steps:                      # MUST be after headings
        b = len(sections)
        sections = merge_fragmented_sections(sections)
        _log(f"✓ merged: {b} → {len(sections)} sections")

    if 'captions' in steps:
        sections = resolve_diagram_captions(sections, diagram_texts_dir, merged_captions)
        _log("✓ diagram captions resolved")

    if 'dedup' in steps:
        # FIX-REDUNDANT: Two-pass cross-section deduplication.
        #
        # Pass 1 — Exact canonical key dedup (existing):
        #   Lowercase + strip punctuation + collapse whitespace → drop true duplicates.
        #
        # Pass 2 — Semantic near-duplicate dedup (NEW):
        #   Many KG-generated bullets for the same concept differ only in their
        #   "label:" prefix or minor phrasing. We detect near-duplicates by:
        #     (a) stripping the "Label:" prefix from each bullet to get the bare claim,
        #     (b) extracting a set of meaningful content words (length > 3, not stopwords),
        #     (c) computing Jaccard similarity on those word sets,
        #     (d) dropping any bullet whose content-word Jaccard with an already-seen
        #         bullet exceeds SEMANTIC_DEDUP_THRESHOLD.
        #   This is fully dynamic — no domain keywords needed.
        b = _total(sections)

        # ── Shared helpers ────────────────────────────────────────────────────
        _DEDUP_STOPS = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
            'has', 'have', 'had', 'this', 'that', 'with', 'from', 'into',
            'and', 'or', 'but', 'for', 'to', 'of', 'in', 'on', 'at', 'by',
            'data', 'which', 'what', 'how', 'when', 'can', 'its', 'their',
            'these', 'those', 'also', 'only', 'used', 'uses', 'using',
            'known', 'called', 'type', 'types', 'form', 'forms',
        }
        # FIX-COVERAGE: The old threshold of 0.72 was too aggressive for KG-generated
        # notes. KG bullets about related but distinct concepts (e.g. "Push operation",
        # "Pop operation", "Peek operation") share many content words (operation, stack,
        # element, top) and were being deduplicated into a single bullet, destroying
        # coverage. Raised to 0.88 so only near-verbatim duplicates are removed.
        # This preserves conceptually distinct bullets that happen to share vocabulary.
        SEMANTIC_DEDUP_THRESHOLD = 0.88  # Jaccard ≥ this → near-duplicate

        def _canonical_key(text: str) -> str:
            """Exact-dedup key: lowercase, no punctuation, collapsed whitespace."""
            key = re.sub(r'[^a-z0-9 ]', '', text.lower())
            return re.sub(r'\s+', ' ', key).strip()

        def _content_words(text: str) -> frozenset:
            """
            Extract meaningful content words for semantic comparison.
            Strips the 'Label:' prefix if present, then removes stopwords
            and short tokens so only substantive words remain.
            """
            # Strip "Label:" prefix (bold label before the colon)
            bare = re.sub(r'^[^:]{1,60}:\s*', '', text, count=1)
            words = re.findall(r'[a-z]{4,}', bare.lower())
            return frozenset(w for w in words if w not in _DEDUP_STOPS)

        def _jaccard(a: frozenset, b: frozenset) -> float:
            if not a or not b:
                return 0.0
            return len(a & b) / len(a | b)

        # ── Pass 1 + Pass 2 combined in one traversal ─────────────────────────
        seen_exact: set = set()          # canonical exact keys
        seen_content: list = []          # list of frozensets of content words

        for sec in sections:
            for sub in sec.get('subsections', []):
                unique_pts = []
                for pt in sub.get('points', []):
                    raw = pt.get('text', '')

                    # Pass 1: exact canonical dedup
                    exact_key = _canonical_key(raw)
                    if exact_key in seen_exact:
                        continue

                    # Pass 2: semantic near-duplicate dedup
                    cw = _content_words(raw)
                    if cw:
                        is_near_dup = any(
                            _jaccard(cw, prev_cw) >= SEMANTIC_DEDUP_THRESHOLD
                            for prev_cw in seen_content
                        )
                        if is_near_dup:
                            continue
                        seen_content.append(cw)

                    seen_exact.add(exact_key)
                    unique_pts.append(pt)

                sub['points'] = unique_pts
            sec['subsections'] = [s for s in sec.get('subsections', []) if s.get('points')]
        sections = [s for s in sections if s.get('subsections')]

        removed_dedup = b - _total(sections)
        if removed_dedup:
            _log(f"✓ near-duplicate dedup (exact + semantic): {removed_dedup} redundant bullets removed")

    # NEW — Semantic bullet grouping: merge ≥3 structurally similar bullets
    # e.g. "Push Operation.", "Pop Operation.", "Peek Operation."
    #   → "Stack supports Push, Pop, and Peek operations."
    if 'group_bullets' in steps:
        try:
            sections = group_related_bullets(sections)
            _log("✓ Related bullets grouped into merged sentences")
        except Exception as e:
            _log(f"⚠ Bullet grouping failed: {e}")

    notes['sections'] = sections
    _log(f"Output → {len(sections)} sections, {_total(sections)} bullets")
    return notes


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_merged_captions(session_dir: Path) -> Dict[str, str]:
    """Load merged_captions.json. Returns {} if not found."""
    for p in [
        Path(session_dir) / 'fused_kg' / 'merged_captions.json',
        Path(session_dir) / 'merged_captions.json',
    ]:
        if p.exists():
            try:
                return json.loads(p.read_text(encoding='utf-8'))
            except Exception:
                pass
    return {}


def _total(sections: List[Dict]) -> int:
    return sum(
        len(sub.get('points', []))
        for s in sections
        for sub in s.get('subsections', [])
    )


def _log(msg: str) -> None:
    print(f"[PostProcessor] {msg}")