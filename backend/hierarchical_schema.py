# hierarchical_schema.py
"""
Structural contract for hierarchical notes generation.

Defines the JSON schema for notes output and provides:
- validate_hierarchy(): Check structural correctness
- fix_hierarchy(): Auto-fix common structural problems
- build_notes_dict(): Helper to construct valid notes dicts

Both KG-based and non-KG notes pipelines produce this schema,
which is then rendered by notes_renderer.py.
"""

from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import re


# ─── Subsection Type Classification (Change 5 / P4) ─────────────────────────

class SubsectionType(Enum):
    """Classifies bullets by their KG edge type for typed subsection grouping."""
    DEFINITION = "Definition"
    PROTOCOL = "Protocols & Usage"
    COMPONENT = "Components & Structure"
    PROPERTY = "Properties"
    APPLICATION = "Applications"
    EXAMPLE = "Examples"
    GENERAL = "Key Points"


# Edge relation → SubsectionType mapping (Change 6)
_EDGE_TYPE_MAP = {
    # Definitional
    'is_a': SubsectionType.DEFINITION,
    'is a': SubsectionType.DEFINITION,
    'defines': SubsectionType.DEFINITION,
    'defined_by': SubsectionType.DEFINITION,
    'type_of': SubsectionType.DEFINITION,
    # Protocol / Functional
    'uses': SubsectionType.PROTOCOL,
    'implemented_by': SubsectionType.PROTOCOL,
    'achieves_via': SubsectionType.PROTOCOL,
    'responsible_for': SubsectionType.PROTOCOL,
    'provides': SubsectionType.PROTOCOL,
    'ensures': SubsectionType.PROTOCOL,
    'connects': SubsectionType.PROTOCOL,
    # Structural / Compositional
    'part_of': SubsectionType.COMPONENT,
    'comprises': SubsectionType.COMPONENT,
    'contains': SubsectionType.COMPONENT,
    'consists_of': SubsectionType.COMPONENT,
    # Properties
    'property': SubsectionType.PROPERTY,
    'has_property': SubsectionType.PROPERTY,
    'compares_to': SubsectionType.PROPERTY,
    # Causal / Process
    'causes': SubsectionType.APPLICATION,
    'results_in': SubsectionType.APPLICATION,
    'depends_on': SubsectionType.APPLICATION,
}


def classify_bullet(edge_relation: str) -> SubsectionType:
    """Map a KG edge relation string to a SubsectionType (Change 6 / P4).
    
    Args:
        edge_relation: The relation/verb from a KG edge
        
    Returns:
        SubsectionType enum value
    """
    if not edge_relation:
        return SubsectionType.GENERAL
    key = edge_relation.lower().strip().replace(' ', '_')
    # Try exact match first
    if key in _EDGE_TYPE_MAP:
        return _EDGE_TYPE_MAP[key]
    # Try with spaces
    key_spaces = edge_relation.lower().strip()
    if key_spaces in _EDGE_TYPE_MAP:
        return _EDGE_TYPE_MAP[key_spaces]
    # Partial match fallback
    for pattern, stype in _EDGE_TYPE_MAP.items():
        if pattern in key or key in pattern:
            return stype
    return SubsectionType.GENERAL


def generate_section_heading(topic_label: str) -> str:
    """Generate a clean, conceptual section heading from a KG node label (Change 7 / P1).

    Dynamically handles:
    - Fragment-style labels that are sentence clauses (start with articles/prepositions)
    - Single generic words that are not descriptive as headings
    - Raw KG graph artifact labels
    - Applies title-case and acronym fixes

    Args:
        topic_label: Raw KG node label or concept name

    Returns:
        Clean, conceptual, title-cased section heading
    """
    if not topic_label or not topic_label.strip():
        return "Key Concepts"

    heading = topic_label.strip()

    # Remove trailing punctuation
    heading = heading.rstrip('.,;:!?')

    # ── FIX 1: Strip transcript-fragment noise prefixes ───────────────────────
    noise_prefixes = [
        r'^(so|now|here|okay|ok|right|well|and|or|but|then)\s+',
        r'^(these|this|that|those)\s+(all\s+)?',
        r'^(let us|let\'s|we will|we\'ll)\s+',
        r'^(coming to|moving on to|as i said)\s+',
    ]
    for pattern in noise_prefixes:
        heading = re.sub(pattern, '', heading, flags=re.IGNORECASE).strip()

    # ── FIX 2: Rewrite fragment-style labels that are sentence clauses ────────
    # Detects labels starting with articles/determiners/prepositions that form
    # incomplete clauses rather than concept names.
    # e.g. "A Linear Data Structure In Which" → extract the core noun phrase
    # e.g. "Top Of The Stack" → "Stack Top"
    # e.g. "In Which Insertions Are Allowed" → drop the whole fragment clause
    _FRAGMENT_STARTERS = re.compile(
        r'^(a|an|the|in|of|at|by|for|with|on|to|into|from|about|'
        r'which|that|where|when|how|what|as|its|their)\s+',
        re.IGNORECASE,
    )

    # Sentence-clause patterns: "A <noun> In Which <clause>" → "<noun>"
    _CLAUSE_PATTERN = re.compile(
        r'^(?:a|an|the)\s+(.+?)\s+(?:in which|where|that|which)\b.*$',
        re.IGNORECASE,
    )
    clause_match = _CLAUSE_PATTERN.match(heading)
    if clause_match:
        # Extract the core noun phrase between the article and relative clause
        heading = clause_match.group(1).strip()

    # "Top Of The Stack" / "X Of The Y" → "Y X" (swap preposition phrase)
    _OF_THE_PATTERN = re.compile(
        r'^(\w[\w\s]*?)\s+[Oo]f\s+[Tt]he\s+(.+)$',
    )
    of_match = _OF_THE_PATTERN.match(heading)
    if of_match and not clause_match:
        head_part = of_match.group(1).strip()
        tail_part = of_match.group(2).strip()
        # Only rewrite if this looks like "property of concept" (e.g. "Top Of Stack")
        # Guard: don't rewrite if head_part is already a good standalone concept
        _STRUCTURAL_HEADS = {
            'top', 'bottom', 'end', 'start', 'head', 'tail', 'front', 'rear',
            'part', 'type', 'kind', 'form', 'example', 'use', 'case',
        }
        if head_part.lower() in _STRUCTURAL_HEADS:
            heading = f"{tail_part} {head_part}".strip()

    # If still starts with an article/preposition fragment, strip the leading word(s)
    while _FRAGMENT_STARTERS.match(heading) and len(heading.split()) > 1:
        heading = _FRAGMENT_STARTERS.sub('', heading).strip()

    # ── FIX 3: Rewrite overly generic single-word headings ────────────────────
    # Single-word headings that are not descriptive concept names get
    # a context-neutral suffix so they're not confusingly generic.
    _GENERIC_SINGLES = {
        'explicit': 'Explicit Stack',
        'implicit': 'Implicit Stack',
        'real': 'Real-Life Applications',
        'life': 'Real-Life Applications',
        'other': 'Additional Concepts',
        'details': 'Implementation Details',
        'overview': 'Topic Overview',
        'introduction': 'Introduction',
        'summary': 'Summary',
        'operations': 'Stack Operations',
        'structure': 'Data Structure',
        'types': 'Types and Variants',
        'examples': 'Illustrative Examples',
        'applications': 'Practical Applications',
        'concepts': 'Key Concepts',
        'notes': 'Additional Notes',
        'methods': 'Implementation Methods',
    }
    heading_lower = heading.lower().strip()
    if heading_lower in _GENERIC_SINGLES:
        heading = _GENERIC_SINGLES[heading_lower]

    # ── FIX 4: Rewrite composite "X And Y" patterns when X or Y is generic ───
    # "Real Life" → "Real-Life Applications"  (two-word but still generic)
    _GENERIC_COMPOSITES = {
        'real life': 'Real-Life Applications',
        'real-life': 'Real-Life Applications',
        'other details': 'Additional Details',
        'key points': 'Key Points',
        'how it works': 'How It Works',
        'primary operations': 'Primary Operations',
        'secondary operations': 'Secondary Operations',
    }
    if heading_lower in _GENERIC_COMPOSITES:
        heading = _GENERIC_COMPOSITES[heading_lower]

    # ── Title-case ────────────────────────────────────────────────────────────
    heading = heading.title()

    # ── FIX 5: Restore common acronyms broken by title-case ──────────────────
    acronym_fixes = {
        'Tcp': 'TCP', 'Udp': 'UDP', 'Ip': 'IP', 'Arp': 'ARP',
        'Icmp': 'ICMP', 'Http': 'HTTP', 'Smtp': 'SMTP', 'Ftp': 'FTP',
        'Dns': 'DNS', 'Ntp': 'NTP', 'Osi': 'OSI', 'Pdu': 'PDU',
        'Mac': 'MAC', 'Lan': 'LAN', 'Wan': 'WAN', 'Api': 'API',
        'Adt': 'ADT', 'Lifo': 'LIFO', 'Fifo': 'FIFO', 'Filo': 'FILO',
        'Dbms': 'DBMS', 'Rdbms': 'RDBMS', 'Sql': 'SQL',
    }
    for wrong, correct in acronym_fixes.items():
        heading = heading.replace(wrong, correct)

    # ── Ensure reasonable length ──────────────────────────────────────────────
    if len(heading) > 60:
        heading = heading[:57] + '...'

    return heading if heading.strip() else "Key Concepts"


# ─── Schema Types (plain dicts, no external deps) ────────────────────────────

def make_point(text: str) -> Dict:
    """Create a bullet point entry."""
    return {"text": text.strip()}


def make_subsection(heading: str, points: List[Dict]) -> Dict:
    """Create a subsection with heading and bullet points."""
    return {
        "heading": heading.strip(),
        "points": points
    }


def make_section(
    heading: str,
    subsections: List[Dict],
    diagram_path: Optional[str] = None,
    diagram_caption: Optional[str] = None
) -> Dict:
    """Create a top-level section."""
    section = {
        "heading": heading.strip(),
        "subsections": subsections
    }
    if diagram_path:
        section["diagram"] = {
            "path": diagram_path,
            "caption": diagram_caption or ""
        }
    return section


def make_notes(title: str, sections: List[Dict], summary: str = "") -> Dict:
    """Create the top-level notes structure.
    
    Args:
        title: The notes title
        sections: List of section dicts
        summary: Optional KG unified summary (academic prose, ≤7 sentences)
    """
    notes = {
        "title": title.strip(),
        "sections": sections
    }
    if summary and summary.strip():
        notes["summary"] = summary.strip()
    return notes


# ─── Validation ──────────────────────────────────────────────────────────────

def validate_hierarchy(notes: Dict) -> Tuple[bool, List[str]]:
    """
    Validate that a notes dict satisfies the hierarchical contract.

    Rules:
      1. Must have a non-empty title
      2. Must have ≥ 2 sections
      3. Each section must have a non-empty heading
      4. Each section must have ≥ 1 subsection
      5. Each subsection must have a non-empty heading
      6. Each subsection must have ≥ 1 point
      7. Each point must have non-empty text

    Returns:
        (is_valid, list_of_violation_messages)
    """
    violations = []

    # Title
    title = notes.get("title", "")
    if not title or not title.strip():
        violations.append("Missing or empty title")

    # Sections
    sections = notes.get("sections", [])
    if len(sections) < 2:
        violations.append(f"Need ≥ 2 sections, found {len(sections)}")

    for s_idx, section in enumerate(sections):
        s_prefix = f"Section {s_idx + 1}"

        # Section heading
        heading = section.get("heading", "")
        if not heading or not heading.strip():
            violations.append(f"{s_prefix}: empty heading")

        # Subsections
        subsections = section.get("subsections", [])
        if len(subsections) < 1:
            violations.append(f"{s_prefix} '{heading}': no subsections")

        for ss_idx, subsec in enumerate(subsections):
            ss_prefix = f"{s_prefix}, Subsection {ss_idx + 1}"

            ss_heading = subsec.get("heading", "")
            if not ss_heading or not ss_heading.strip():
                violations.append(f"{ss_prefix}: empty heading")

            points = subsec.get("points", [])
            if len(points) < 1:
                violations.append(f"{ss_prefix} '{ss_heading}': no points")

            for p_idx, pt in enumerate(points):
                text = pt.get("text", "")
                if not text or not text.strip():
                    violations.append(f"{ss_prefix}, Point {p_idx + 1}: empty text")

    return (len(violations) == 0, violations)


# ─── Auto-Fix ────────────────────────────────────────────────────────────────

def fix_hierarchy(notes: Dict) -> Dict:
    """
    Auto-fix common structural problems so the output passes validation.

    Fixes applied:
      1. Orphan points (section has points but no subsections) →
         wrap in a "Key Points" subsection
      2. Empty subsections → remove them
      3. Sections with 0 subsections after cleanup →
         merge content into previous section or create minimal subsection
      4. If < 2 sections → split largest section if possible
      5. Empty title → set generic title
    """
    import copy
    notes = copy.deepcopy(notes)

    # Fix 1: Title
    if not notes.get("title", "").strip():
        notes["title"] = "Lecture Notes"

    sections = notes.get("sections", [])

    # Fix 2: Promote orphan points into subsections
    for section in sections:
        subsections = section.get("subsections", [])

        # Check if section has a flat "points" key (legacy format)
        flat_points = section.pop("points", [])
        if flat_points and not subsections:
            section["subsections"] = [make_subsection("Key Points", flat_points)]
        elif flat_points and subsections:
            subsections.append(make_subsection("Additional Points", flat_points))
            section["subsections"] = subsections

    # Fix 3: Remove empty subsections, remove empty points
    for section in sections:
        cleaned_subs = []
        for sub in section.get("subsections", []):
            # Remove empty points
            sub["points"] = [p for p in sub.get("points", []) if p.get("text", "").strip()]
            if sub["points"]:  # Keep subsection only if it has content
                if not sub.get("heading", "").strip():
                    sub["heading"] = "Details"
                cleaned_subs.append(sub)
        section["subsections"] = cleaned_subs

    # Fix 4: Sections with 0 subsections → create minimal subsection from heading
    for section in sections:
        if not section.get("subsections"):
            heading = section.get("heading", "Topic")
            section["subsections"] = [
                make_subsection("Overview", [make_point(f"{heading} is a key concept in this domain.")])
            ]

    # Fix 5: If < 2 sections, try to split the largest one
    if len(sections) == 1 and len(sections[0].get("subsections", [])) >= 2:
        original = sections[0]
        subs = original["subsections"]
        mid = len(subs) // 2

        section_a = make_section(
            original["heading"],
            subs[:mid]
        )
        section_b = make_section(
            f"{original['heading']} (Continued)",
            subs[mid:]
        )
        sections = [section_a, section_b]
    elif len(sections) == 0:
        sections = [
            make_section("Introduction", [
                make_subsection("Overview", [make_point("Content overview.")])
            ]),
            make_section("Summary", [
                make_subsection("Key Takeaways", [make_point("Main concepts covered.")])
            ])
        ]

    notes["sections"] = sections
    return notes


# ─── Convenience: Flatten to text for debugging ─────────────────────────────

def notes_to_plain_text(notes: Dict) -> str:
    """Convert HierarchicalNotes dict to plain numbered text (for debugging)."""
    lines = [notes.get("title", "Notes"), "=" * 40, ""]

    for s_idx, section in enumerate(notes.get("sections", []), 1):
        lines.append(f"{s_idx}. {section.get('heading', 'Section')}")
        lines.append("-" * 30)

        for ss_idx, sub in enumerate(section.get("subsections", []), 1):
            lines.append(f"  {s_idx}.{ss_idx} {sub.get('heading', 'Subsection')}")

            for pt in sub.get("points", []):
                lines.append(f"    • {pt.get('text', '')}")

            lines.append("")
        lines.append("")

    return "\n".join(lines)


def promote_subsections(sections: list, min_points: int = 4) -> list:
    """
    Promote any subsection with >= min_points into its own top-level section.
    Fixes: 'Additional Concepts' containing HTTP, UDP, ARP, IPv4 all separately.
    """
    result = []
    for section in sections:
        # Keep subsections that are small
        keep_subs = []
        for sub in section.get('subsections', []):
            points = sub.get('points', [])
            if len(points) >= min_points:
                # Promote to top-level section
                result.append({
                    'heading': sub['heading'],
                    'summary': sub.get('summary', ''),
                    'subsections': [{
                        'heading': 'Overview',
                        'points': points,
                        'type': sub.get('type', 'Key Points')
                    }],
                    'images': sub.get('images', [])
                })
            else:
                keep_subs.append(sub)
        section['subsections'] = keep_subs
        result.append(section)
    return result