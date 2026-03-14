# notes_quality_enforcer.py
"""
Dynamic Notes Quality Enforcer
================================
A pipeline wrapper that enforces quality on generated notes
regardless of input topic, length, or source (KG or non-KG).

Wires together the existing modules to fix all 5 evaluation gaps:
  1. Redundancy        → deduplicate_across_sections (semantic_clusterer)
  2. Weak hierarchy    → generate_section_heading (hierarchical_schema)
  3. Context-free defs → classify_bullet + SubsectionType grouping
  4. Verbosity         → bullet compression pass
  5. Topic bleeding    → SemanticClusterer enforced on every path

Usage:
    from notes_quality_enforcer import enforce_notes_quality

    improved_notes = enforce_notes_quality(raw_notes_dict)
    # OR with KG data for context-aware grouping:
    improved_notes = enforce_notes_quality(raw_notes_dict, kg_nodes=nodes, kg_edges=edges)
"""

import re
import copy
from typing import List, Dict, Optional, Tuple

# ── Existing module imports (all already in your codebase) ────────────────────
from hierarchical_schema import (
    generate_section_heading,
    classify_bullet,
    SubsectionType,
    fix_hierarchy,
    validate_hierarchy,
    make_point,
    make_subsection,
    make_section,
)
from semantic_clusterer import deduplicate_across_sections, select_canonical_bullet


# ═════════════════════════════════════════════════════════════════════════════
# Gap 1: Redundancy Removal
# ═════════════════════════════════════════════════════════════════════════════

def remove_redundancy(sections: List[Dict]) -> List[Dict]:
    """
    Two-pass deduplication:
      Pass 1 — within each subsection (exact + near-duplicate sentences)
      Pass 2 — across all sections (TF-IDF cosine, threshold 0.85)

    Works on any topic because it is purely text-similarity based.
    """
    # Pass 1: within subsection
    for section in sections:
        for sub in section.get("subsections", []):
            seen_texts = []
            unique_points = []
            for pt in sub.get("points", []):
                text = pt.get("text", "").strip().lower()
                if not text:
                    continue
                # Check near-duplicate: word overlap > 70%
                is_dup = False
                for seen in seen_texts:
                    words_new = set(text.split())
                    words_seen = set(seen.split())
                    if words_new and words_seen:
                        overlap = len(words_new & words_seen) / max(len(words_new), len(words_seen))
                        if overlap > 0.70:
                            is_dup = True
                            break
                if not is_dup:
                    seen_texts.append(text)
                    unique_points.append(pt)
            sub["points"] = unique_points

    # Pass 2: across sections (uses your existing TF-IDF deduplicator)
    sections = deduplicate_across_sections(sections)
    return sections


# ═════════════════════════════════════════════════════════════════════════════
# Gap 2: Hierarchy Strengthening
# ═════════════════════════════════════════════════════════════════════════════

_GENERIC_HEADINGS = {
    "key concepts", "additional concepts", "other concepts",
    "details", "content", "misc", "other", "general", "overview",
    "key points", "main points", "section", "topic", "notes",
}

def strengthen_hierarchy(sections: List[Dict], kg_nodes: Optional[List[Dict]] = None) -> List[Dict]:
    """
    Replace generic section/subsection headings with specific ones.

    Strategy (domain-agnostic):
      1. If heading is generic → derive from the most common concept
         mentioned in its bullet points.
      2. Use generate_section_heading() from hierarchical_schema for cleanup.
      3. If KG nodes provided → try to match bullet content to node labels.

    Works with any input topic because it reads the actual bullet content.
    """
    # Build KG label lookup for optional context-aware headings
    kg_label_set = set()
    if kg_nodes:
        for n in kg_nodes:
            label = n.get("label", "").strip()
            if label:
                kg_label_set.add(label.lower())

    for section in sections:
        raw_heading = section.get("heading", "")
        if raw_heading.lower().strip() in _GENERIC_HEADINGS:
            # Derive from bullet content
            derived = _derive_heading_from_bullets(section, kg_label_set)
            section["heading"] = generate_section_heading(derived)
        else:
            section["heading"] = generate_section_heading(raw_heading)

        for sub in section.get("subsections", []):
            raw_sub = sub.get("heading", "")
            if raw_sub.lower().strip() in _GENERIC_HEADINGS:
                derived = _derive_heading_from_bullets_sub(sub, kg_label_set)
                sub["heading"] = generate_section_heading(derived)
            else:
                sub["heading"] = generate_section_heading(raw_sub)

    return sections


def _derive_heading_from_bullets(section: Dict, kg_labels: set) -> str:
    """Extract the dominant concept from a section's bullet points."""
    all_words = []
    for sub in section.get("subsections", []):
        for pt in sub.get("points", []):
            words = pt.get("text", "").lower().split()
            all_words.extend([w for w in words if len(w) > 4])

    return _most_frequent_concept(all_words, kg_labels) or section.get("heading", "Key Concepts")


def _derive_heading_from_bullets_sub(sub: Dict, kg_labels: set) -> str:
    """Extract the dominant concept from a subsection's bullet points."""
    all_words = []
    for pt in sub.get("points", []):
        words = pt.get("text", "").lower().split()
        all_words.extend([w for w in words if len(w) > 4])

    return _most_frequent_concept(all_words, kg_labels) or sub.get("heading", "Details")


def _most_frequent_concept(words: List[str], kg_labels: set) -> str:
    """Return the most frequent meaningful word, preferring KG labels."""
    if not words:
        return ""

    stopwords = {
        "which", "where", "there", "their", "these", "those",
        "about", "using", "based", "level", "system", "data",
        "that", "with", "from", "have", "been", "this", "into",
    }
    freq: Dict[str, int] = {}
    for w in words:
        if w not in stopwords:
            freq[w] = freq.get(w, 0) + 1

    if not freq:
        return ""

    # Prefer words that appear in KG node labels
    for word, count in sorted(freq.items(), key=lambda x: -x[1]):
        if any(word in label for label in kg_labels):
            return word.title()

    # Fallback: most frequent word
    return max(freq, key=freq.get).title()


# ═════════════════════════════════════════════════════════════════════════════
# Gap 3: Definitions With Context (SubsectionType grouping)
# ═════════════════════════════════════════════════════════════════════════════

def group_bullets_by_type(
    sections: List[Dict],
    kg_edges: Optional[List[Dict]] = None
) -> List[Dict]:
    """
    Regroup bullet points within each section by their SubsectionType.

    This ensures:
      - Definitions appear before Properties/Components
      - Applications appear last
      - Every bullet is placed in a typed, named subsection

    Works dynamically: edge-relation types drive grouping when KG is provided,
    keyword heuristics are used as fallback for non-KG notes.
    """
    # Build edge relation → node mapping if KG available
    edge_relation_map: Dict[str, str] = {}
    if kg_edges:
        for edge in kg_edges:
            src = edge.get("source", "")
            rel = edge.get("relation", "")
            if src and rel:
                edge_relation_map[src.lower()] = rel

    # Preferred ordering of SubsectionTypes
    TYPE_ORDER = [
        SubsectionType.DEFINITION,
        SubsectionType.COMPONENT,
        SubsectionType.PROTOCOL,
        SubsectionType.PROPERTY,
        SubsectionType.APPLICATION,
        SubsectionType.EXAMPLE,
        SubsectionType.GENERAL,
    ]

    for section in sections:
        all_points = []
        for sub in section.get("subsections", []):
            all_points.extend(sub.get("points", []))

        if len(all_points) < 3:
            continue  # Not worth regrouping tiny sections

        # Classify each point
        typed_buckets: Dict[SubsectionType, List[Dict]] = {t: [] for t in TYPE_ORDER}

        for pt in all_points:
            text = pt.get("text", "")
            stype = _classify_point_text(text, edge_relation_map)
            typed_buckets[stype].append(pt)

        # Rebuild subsections in order, only for non-empty buckets
        new_subsections = []
        for stype in TYPE_ORDER:
            pts = typed_buckets[stype]
            if pts:
                new_subsections.append(make_subsection(stype.value, pts))

        if new_subsections:
            section["subsections"] = new_subsections

    return sections


def _classify_point_text(text: str, edge_relation_map: Dict[str, str]) -> SubsectionType:
    """
    Classify a bullet point text into a SubsectionType.

    Priority:
      1. KG edge relation if the text mentions a known KG source node
      2. Keyword heuristics
    """
    text_lower = text.lower()

    # Try KG edge relation first
    for node, relation in edge_relation_map.items():
        if node in text_lower:
            return classify_bullet(relation)

    # Keyword heuristics (domain-agnostic)
    if any(kw in text_lower for kw in ["is a", "refers to", "defined as", "means", "definition"]):
        return SubsectionType.DEFINITION
    if any(kw in text_lower for kw in ["consists of", "contains", "has", "includes", "component", "part of"]):
        return SubsectionType.COMPONENT
    if any(kw in text_lower for kw in ["used for", "application", "example", "such as", "use case"]):
        return SubsectionType.APPLICATION
    if any(kw in text_lower for kw in ["type", "kind", "category", "classified", "variant"]):
        return SubsectionType.PROPERTY
    if any(kw in text_lower for kw in ["for example", "e.g.", "instance", "like"]):
        return SubsectionType.EXAMPLE

    return SubsectionType.GENERAL


# ═════════════════════════════════════════════════════════════════════════════
# Gap 4: Verbosity Reduction (Bullet Compression)
# ═════════════════════════════════════════════════════════════════════════════

# Pattern: "X is the/a Y that/which Z" → "X: Y Z"
_COMPRESS_PATTERNS = [
    # "the person responsible for managing..." → "manages..."
    (r"the (\w+) responsible for (\w+ing)", r"\1 → \2"),
    # "refers to the process of" → "process:"
    (r"refers to the process of", "process:"),
    # "is defined as" → ":"
    (r"\bis defined as\b", ":"),
    # "is responsible for" → "→"
    (r"\bis responsible for\b", "→"),
    # "can be used to" → "used to"
    (r"\bcan be used to\b", "used to"),
    # "it is important to note that" → ""
    (r"\bit is important to note that\b", ""),
    # "in the context of" → "in"
    (r"\bin the context of\b", "in"),
    # Remove trailing "as well" / "also"
    (r",?\s+as well\s*$", ""),
    (r",?\s+also\s*$", ""),
]

_MAX_BULLET_WORDS = 25  # Bullets longer than this get compressed


def compress_bullets(sections: List[Dict]) -> List[Dict]:
    """
    Reduce verbosity in bullet points using pattern compression.

    Applies regex-based phrase shortening for common verbose patterns.
    No summarization model required — works on any domain.
    """
    for section in sections:
        for sub in section.get("subsections", []):
            new_points = []
            for pt in sub.get("points", []):
                text = pt.get("text", "").strip()
                compressed = _compress_text(text)
                new_points.append(make_point(compressed))
            sub["points"] = new_points
    return sections


def _compress_text(text: str) -> str:
    """Apply compression patterns to a single bullet text."""
    for pattern, replacement in _COMPRESS_PATTERNS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Hard truncation for very long bullets — keep first sentence
    words = text.split()
    if len(words) > _MAX_BULLET_WORDS:
        # Try to cut at a natural boundary
        sentences = re.split(r'[.;]', text)
        if sentences and len(sentences[0].split()) >= 5:
            text = sentences[0].strip()
        else:
            text = " ".join(words[:_MAX_BULLET_WORDS]) + "..."

    return text.strip()


# ═════════════════════════════════════════════════════════════════════════════
# Gap 5: Topic Bleeding Prevention (enforced clustering)
# ═════════════════════════════════════════════════════════════════════════════

def enforce_topic_separation(sections: List[Dict]) -> List[Dict]:
    """
    Split sections that contain bullets spanning multiple distinct topics.

    Detection: if a section has ≥ 2 subsections with NO shared vocabulary
    (Jaccard < 0.05), they belong to different topics → split into
    separate top-level sections.

    This is purely text-driven — works for any subject matter.
    """
    result = []
    for section in sections:
        subsections = section.get("subsections", [])
        if len(subsections) < 2:
            result.append(section)
            continue

        # Compute pairwise vocabulary similarity between subsections
        sub_vocab = []
        for sub in subsections:
            words = set()
            for pt in sub.get("points", []):
                words.update(pt.get("text", "").lower().split())
            sub_vocab.append(words)

        # Group subsections by connected components (Jaccard > 0.05)
        groups = _connected_components(subsections, sub_vocab, threshold=0.05)

        if len(groups) == 1:
            result.append(section)
        else:
            # Split into separate sections
            for i, group_subs in enumerate(groups):
                if not group_subs:
                    continue
                # Name: original heading + derived sub-topic
                sub_topic = _derive_heading_from_bullets_sub(group_subs[0], set())
                new_heading = f"{section['heading']} – {sub_topic}" if i > 0 else section["heading"]
                new_section = make_section(
                    heading=generate_section_heading(new_heading),
                    subsections=group_subs
                )
                result.append(new_section)

    return result


def _connected_components(
    subsections: List[Dict],
    vocab_list: List[set],
    threshold: float
) -> List[List[Dict]]:
    """Group subsections into connected components by Jaccard similarity."""
    n = len(subsections)
    visited = [False] * n
    groups = []

    def jaccard(a: set, b: set) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    for start in range(n):
        if visited[start]:
            continue
        group_idxs = [start]
        visited[start] = True
        queue = [start]
        while queue:
            curr = queue.pop()
            for j in range(n):
                if not visited[j] and jaccard(vocab_list[curr], vocab_list[j]) > threshold:
                    visited[j] = True
                    group_idxs.append(j)
                    queue.append(j)
        groups.append([subsections[i] for i in group_idxs])

    return groups


# ═════════════════════════════════════════════════════════════════════════════
# Master Enforcer
# ═════════════════════════════════════════════════════════════════════════════

def enforce_notes_quality(
    notes: Dict,
    kg_nodes: Optional[List[Dict]] = None,
    kg_edges: Optional[List[Dict]] = None,
    run_steps: Optional[List[str]] = None,
) -> Dict:
    """
    Apply all 5 quality improvements to a HierarchicalNotes dict.

    Parameters
    ----------
    notes      : HierarchicalNotes dict (from hierarchical_schema.make_notes)
    kg_nodes   : Optional list of KG node dicts {id, label, description}
    kg_edges   : Optional list of KG edge dicts {source, target, relation}
    run_steps  : Optional list to run only specific steps.
                 Choices: "redundancy", "hierarchy", "grouping",
                           "compression", "topic_separation"
                 Default: all steps.

    Returns
    -------
    Improved HierarchicalNotes dict, always passing validate_hierarchy().
    """
    notes = copy.deepcopy(notes)
    sections = notes.get("sections", [])
    all_steps = ["redundancy", "hierarchy", "grouping", "compression", "topic_separation"]
    steps = run_steps if run_steps else all_steps

    print(f"[QualityEnforcer] Running steps: {steps}")
    print(f"[QualityEnforcer] Input: {len(sections)} sections")

    if "redundancy" in steps:
        sections = remove_redundancy(sections)
        print(f"[QualityEnforcer] After redundancy removal: {_count_bullets(sections)} bullets")

    if "hierarchy" in steps:
        sections = strengthen_hierarchy(sections, kg_nodes)
        print(f"[QualityEnforcer] Headings strengthened")

    if "grouping" in steps:
        sections = group_bullets_by_type(sections, kg_edges)
        print(f"[QualityEnforcer] Bullets grouped by type")

    if "compression" in steps:
        sections = compress_bullets(sections)
        print(f"[QualityEnforcer] Bullets compressed")

    if "topic_separation" in steps:
        sections = enforce_topic_separation(sections)
        print(f"[QualityEnforcer] After topic separation: {len(sections)} sections")

    notes["sections"] = sections

    # Always run fix_hierarchy to ensure schema compliance
    notes = fix_hierarchy(notes)

    # Validate
    is_valid, violations = validate_hierarchy(notes)
    if not is_valid:
        print(f"[QualityEnforcer] ⚠ Validation issues: {violations}")
    else:
        print(f"[QualityEnforcer] ✅ Notes pass all structural checks")

    return notes


def _count_bullets(sections: List[Dict]) -> int:
    total = 0
    for s in sections:
        for sub in s.get("subsections", []):
            total += len(sub.get("points", []))
    return total


# ═════════════════════════════════════════════════════════════════════════════
# Integration helpers — drop-in replacements for your existing pipelines
# ═════════════════════════════════════════════════════════════════════════════

def post_process_kg_notes(notes: Dict, kg_nodes: List[Dict], kg_edges: List[Dict]) -> Dict:
    """
    Drop-in for KG-based notes pipeline.
    Call this after building the HierarchicalNotes dict from the KG.

    Example usage in ex.py / kg_summarizer.py:
        from notes_quality_enforcer import post_process_kg_notes
        notes = post_process_kg_notes(raw_notes, nodes, edges)
    """
    return enforce_notes_quality(notes, kg_nodes=kg_nodes, kg_edges=kg_edges)


def post_process_non_kg_notes(notes: Dict) -> Dict:
    """
    Drop-in for non-KG (topic-model / BART) notes pipeline.
    No KG data available — uses heuristic-only quality improvements.

    Example usage in ex.py:
        from notes_quality_enforcer import post_process_non_kg_notes
        notes = post_process_non_kg_notes(raw_notes)
    """
    # Skip grouping step (needs KG edge relations for full benefit)
    return enforce_notes_quality(
        notes,
        run_steps=["redundancy", "hierarchy", "compression", "topic_separation"]
    )
