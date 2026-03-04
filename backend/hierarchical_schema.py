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


def make_notes(title: str, sections: List[Dict]) -> Dict:
    """Create the top-level notes structure."""
    return {
        "title": title.strip(),
        "sections": sections
    }


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
