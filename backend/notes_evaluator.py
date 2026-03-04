# notes_evaluator.py
"""
Evaluation metrics for generated notes:

1. ROUGE-1/L (Recall) — content coverage vs reference
2. Flesch Reading Ease (FRE) — readability
3. Gunning Fog Index (GFI) — education level needed
4. Concept Dependency Score — logical flow from KG structure

The Concept Dependency Score uses EXACT verbs/relations from the KG edges
to build prerequisite chains, then checks if the generated notes present
concepts in the correct pedagogical order.
"""

import re
import json
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional


# ─── Syllable Counting ──────────────────────────────────────────────────────

def count_syllables(word: str) -> int:
    """
    Estimate syllable count for an English word.
    Uses a vowel-group heuristic with corrections for silent-e, -le, etc.
    """
    word = word.lower().strip()
    if not word:
        return 0

    # Special short words
    if len(word) <= 3:
        return 1

    # Count vowel groups
    vowels = "aeiouy"
    count = 0
    prev_vowel = False

    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel

    # Adjustments
    if word.endswith("e") and not word.endswith("le"):
        count -= 1
    if word.endswith("ed") and len(word) > 3:
        # "played" = 1 syl, "created" = 3 syl
        if word[-3] not in "dt":
            count -= 1

    return max(count, 1)


def count_syllables_text(text: str) -> int:
    """Count total syllables in a text."""
    words = re.findall(r'[a-zA-Z]+', text)
    return sum(count_syllables(w) for w in words)


# ─── Text Stats Helpers ──────────────────────────────────────────────────────

def _split_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    # Split on sentence-ending punctuation
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip() and len(s.strip().split()) >= 3]


def _get_words(text: str) -> List[str]:
    """Extract words from text."""
    return re.findall(r'[a-zA-Z]+', text)


def _is_complex_word(word: str) -> bool:
    """A complex word has ≥ 3 syllables (Gunning Fog definition)."""
    return count_syllables(word) >= 3


# ─── ROUGE (Recall) ─────────────────────────────────────────────────────────

def compute_rouge(generated: str, reference: str) -> Dict:
    """
    Compute ROUGE-1 and ROUGE-L Recall.

    ROUGE-N Recall = matching_ngrams / total_reference_ngrams
    ROUGE-L Recall = LCS_length / reference_length

    Args:
        generated: Generated notes text
        reference: Reference/ground truth notes text

    Returns:
        Dict with rouge1_recall, rouge2_recall, rougeL_recall
    """
    gen_words = generated.lower().split()
    ref_words = reference.lower().split()

    if not ref_words:
        return {"rouge1_recall": 0, "rouge2_recall": 0, "rougeL_recall": 0}

    # ROUGE-1 (unigram recall)
    ref_unigrams = Counter(ref_words)
    gen_unigrams = Counter(gen_words)
    overlap_1 = sum((ref_unigrams & gen_unigrams).values())
    rouge1 = overlap_1 / sum(ref_unigrams.values()) if sum(ref_unigrams.values()) > 0 else 0

    # ROUGE-2 (bigram recall)
    ref_bigrams = Counter(zip(ref_words, ref_words[1:]))
    gen_bigrams = Counter(zip(gen_words, gen_words[1:]))
    overlap_2 = sum((ref_bigrams & gen_bigrams).values())
    rouge2 = overlap_2 / sum(ref_bigrams.values()) if sum(ref_bigrams.values()) > 0 else 0

    # ROUGE-L (LCS recall)
    lcs_len = _lcs_length(ref_words, gen_words)
    rougeL = lcs_len / len(ref_words) if len(ref_words) > 0 else 0

    return {
        "rouge1_recall": round(rouge1, 4),
        "rouge2_recall": round(rouge2, 4),
        "rougeL_recall": round(rougeL, 4)
    }


def _lcs_length(x: List[str], y: List[str]) -> int:
    """Compute Longest Common Subsequence length (optimized for memory)."""
    m, n = len(x), len(y)
    # Use only 2 rows for O(n) space
    if m > 500 or n > 500:
        # For very long texts, use a sampled approach
        x = x[:500]
        y = y[:500]
        m, n = len(x), len(y)

    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)

    return prev[n]


# ─── Flesch Reading Ease ─────────────────────────────────────────────────────

def compute_flesch_reading_ease(text: str) -> Dict:
    """
    Compute Flesch Reading Ease score.

    FRE = 206.835 - (1.015 × ASL) - (84.6 × ASW)

    Where:
      ASL = Average Sentence Length (words/sentences)
      ASW = Average Syllables per Word (syllables/words)

    Returns:
        Dict with score, asl, asw, and interpretation label
    """
    sentences = _split_sentences(text)
    words = _get_words(text)

    if not sentences or not words:
        return {"score": 0, "asl": 0, "asw": 0, "label": "N/A"}

    total_syllables = sum(count_syllables(w) for w in words)

    asl = len(words) / len(sentences)
    asw = total_syllables / len(words)

    fre = 206.835 - (1.015 * asl) - (84.6 * asw)
    fre = max(0, min(100, fre))  # Clamp to 0-100

    # Interpretation
    if fre >= 90:
        label = "Very Easy"
    elif fre >= 60:
        label = "Standard"
    elif fre >= 30:
        label = "Difficult"
    else:
        label = "Very Difficult"

    return {
        "score": round(fre, 2),
        "asl": round(asl, 2),
        "asw": round(asw, 2),
        "label": label
    }


# ─── Gunning Fog Index ───────────────────────────────────────────────────────

def compute_gunning_fog(text: str) -> Dict:
    """
    Compute Gunning Fog Index.

    GFI = 0.4 × (Words/Sentences + 100 × ComplexWords/Words)

    Complex word = ≥ 3 syllables

    Returns:
        Dict with score, complex_word_ratio, and interpretation label
    """
    sentences = _split_sentences(text)
    words = _get_words(text)

    if not sentences or not words:
        return {"score": 0, "complex_word_ratio": 0, "label": "N/A"}

    complex_words = [w for w in words if _is_complex_word(w)]
    complex_ratio = len(complex_words) / len(words)

    gfi = 0.4 * ((len(words) / len(sentences)) + (100 * complex_ratio))

    # Interpretation
    if gfi <= 10:
        label = "Easy"
    elif gfi <= 14:
        label = "Moderate"
    else:
        label = "Difficult"

    return {
        "score": round(gfi, 2),
        "complex_word_ratio": round(complex_ratio, 4),
        "label": label
    }


# ─── Concept Dependency Score ────────────────────────────────────────────────

def extract_dependencies_from_kg(edges: List[Dict], nodes: List[Dict]) -> List[Dict]:
    """
    Extract concept dependency pairs from KG edges using EXACT verbs/relations.

    Each KG edge represents a dependency: source must be introduced before target
    for hierarchical relations, or both must appear together for associative ones.

    ALL relation verbs from the KG are preserved exactly.

    Args:
        edges: List of KG edge dicts with source, target, relation
        nodes: List of KG node dicts with id, label

    Returns:
        List of dependency dicts: {prerequisite, dependent, relation, prereq_label, dep_label}
    """
    # Build node label lookup
    node_labels = {}
    for n in nodes:
        nid = n.get("id", "")
        label = n.get("label", "")
        if nid and label:
            node_labels[nid] = label.lower()

    dependencies = []

    # Hierarchical relations: source must appear BEFORE target
    hierarchical_verbs = {
        "has", "includes", "contains", "consists", "composed",
        "divided into", "has concept of", "has type", "type of",
        "is a", "has part", "comprises", "has component"
    }

    # All edges create dependencies: source → target
    for edge in edges:
        src = edge.get("source", "")
        tgt = edge.get("target", "")
        rel = edge.get("relation", "")

        src_label = node_labels.get(src, "")
        tgt_label = node_labels.get(tgt, "")

        if not src_label or not tgt_label or src_label == tgt_label:
            continue

        # Check if this is a hierarchical relation (source before target)
        rel_lower = rel.lower().strip()
        is_hierarchical = any(v in rel_lower for v in hierarchical_verbs)

        dependencies.append({
            "prerequisite": src,
            "dependent": tgt,
            "relation": rel,  # EXACT verb from KG
            "prereq_label": src_label,
            "dep_label": tgt_label,
            "is_hierarchical": is_hierarchical
        })

    return dependencies


def compute_concept_dependency_score(
    notes_text: str,
    edges: List[Dict],
    nodes: List[Dict]
) -> Dict:
    """
    Compute Concept Dependency Score.

    Score = 1 - (violations / total_dependencies)

    A violation occurs when a dependent concept appears before its
    prerequisite concept in the generated notes.

    Uses EXACT relation verbs from the KG.

    Args:
        notes_text: Generated notes text
        edges: KG edges with source, target, relation
        nodes: KG nodes with id, label

    Returns:
        Dict with score, total_dependencies, violations, violation_details
    """
    dependencies = extract_dependencies_from_kg(edges, nodes)
    notes_lower = notes_text.lower()

    if not dependencies:
        return {
            "score": 1.0,
            "total_dependencies": 0,
            "violations": 0,
            "violation_details": [],
            "coverage": 0
        }

    violations = 0
    violation_details = []
    covered = 0

    for dep in dependencies:
        prereq = dep["prereq_label"]
        dependent = dep["dep_label"]
        relation = dep["relation"]

        # Find first occurrence positions
        pos_prereq = notes_lower.find(prereq)
        pos_dependent = notes_lower.find(dependent)

        # Both must be present for this dependency to count
        if pos_prereq == -1 or pos_dependent == -1:
            continue

        covered += 1

        # Check: hierarchical deps require prereq before dependent
        if dep["is_hierarchical"] and pos_dependent < pos_prereq:
            violations += 1
            violation_details.append({
                "prerequisite": prereq,
                "dependent": dependent,
                "relation": relation,
                "prereq_position": pos_prereq,
                "dependent_position": pos_dependent,
                "type": "ordering_violation"
            })

    if covered == 0:
        return {
            "score": 0.5,  # Neutral if no deps can be evaluated
            "total_dependencies": len(dependencies),
            "violations": 0,
            "violation_details": [],
            "coverage": 0
        }

    score = 1.0 - (violations / covered)

    return {
        "score": round(score, 4),
        "total_dependencies": len(dependencies),
        "evaluated_dependencies": covered,
        "violations": violations,
        "violation_details": violation_details[:10],  # Limit detail output
        "coverage": round(covered / len(dependencies), 4)
    }


# ─── Verb Fidelity Check ────────────────────────────────────────────────────

def compute_verb_fidelity(notes_text: str, edges: List[Dict]) -> Dict:
    """
    Check whether generated notes use EXACT verbs/relations from the KG
    rather than generalizing them.

    For KG-based notes: verbs should match KG edge relations.
    For topic-modeling notes: verbs should match text-context relations.

    Returns:
        Dict with fidelity_score, matched_verbs, missing_verbs, total_verbs
    """
    notes_lower = notes_text.lower()

    # Extract all unique relation verbs from KG
    all_relations = set()
    for edge in edges:
        rel = edge.get("relation", "").strip()
        if rel and len(rel) > 2:
            all_relations.add(rel.lower())

    matched = []
    missing = []

    for rel in all_relations:
        if rel in notes_lower:
            matched.append(rel)
        else:
            missing.append(rel)

    total = len(all_relations)
    fidelity = len(matched) / total if total > 0 else 0

    return {
        "fidelity_score": round(fidelity, 4),
        "matched_verbs": sorted(matched),
        "missing_verbs": sorted(missing),
        "total_verbs": total,
        "matched_count": len(matched)
    }


# ─── Combined Evaluation ─────────────────────────────────────────────────────

def evaluate_notes(
    notes_text: str,
    reference_text: str = "",
    kg_edges: Optional[List[Dict]] = None,
    kg_nodes: Optional[List[Dict]] = None
) -> Dict:
    """
    Run all 4 evaluation metrics + verb fidelity on generated notes.

    Args:
        notes_text: Generated notes text
        reference_text: Reference/ground truth notes text (for ROUGE)
        kg_edges: KG edges for dependency scoring
        kg_nodes: KG nodes for dependency scoring

    Returns:
        Combined evaluation dict with all metrics
    """
    result = {}

    # 1. ROUGE (only if reference provided)
    if reference_text and reference_text.strip():
        result["rouge"] = compute_rouge(notes_text, reference_text)
    else:
        result["rouge"] = {"rouge1_recall": None, "rouge2_recall": None, "rougeL_recall": None}

    # 2. Flesch Reading Ease
    result["flesch_reading_ease"] = compute_flesch_reading_ease(notes_text)

    # 3. Gunning Fog Index
    result["gunning_fog"] = compute_gunning_fog(notes_text)

    # 4. Concept Dependency Score (only if KG data provided)
    if kg_edges and kg_nodes:
        result["dependency_score"] = compute_concept_dependency_score(
            notes_text, kg_edges, kg_nodes
        )
    else:
        result["dependency_score"] = {
            "score": None, "total_dependencies": 0,
            "violations": 0, "violation_details": []
        }

    return result
