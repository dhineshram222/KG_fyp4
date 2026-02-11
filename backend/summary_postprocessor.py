# summary_postprocessor.py
"""
Universal Summary Post-Processor

Provides clean_and_constrain_summary() used by ALL summary pipelines to:
1. Remove noise (OCR artifacts, image descriptions, hallucinations, meta-commentary)
2. Filter fragments (no verb, too short, incoherent)
3. Deduplicate near-identical sentences
4. Constrain to 7-10 context-rich sentences
"""

import re
from typing import List, Optional
from collections import OrderedDict


# ─── Noise Patterns ───────────────────────────────────────────────────────────

# Image-description artifacts from BLIP / OCR pipelines
_IMAGE_DESC_PATTERNS = [
    r"a\s+(black\s+and\s+white|color|close[- ]up|blurry)?\s*(photo|image|picture|screenshot|graph|diagram)\s+of\b",
    r"an?\s+image\s+of\s+a\b",
    r"(the\s+)?logo\s+of\s+(the\s+)?",
    r"a\s+red\s+square\s+with\b",
    r"images?\s+of\s+a\s+computer\s+screen\b",
    r"a\s+graph\s+is\s+shown\b",
    r"a\s+wave\b.*?flying\s+through\b",
]

# BART / CNN-DailyMail hallucination patterns
_HALLUCINATION_PATTERNS = [
    r"CNN\.com",
    r"iReporter",
    r"DailyMail",
    r"Click here for",
    r"travel snapshots",
    r"submit your best shots",
    r"Visit CNN",
    r"\.COM/Travel",
]

# Meta-commentary / filler (matched case-insensitively)
_META_COMMENTARY_PATTERNS = [
    r"^subscribe\b",
    r"^thanks?\s+(for)?\s*watching\b",
    r"^see\s+you\s+in\s+(the\s+)?next\b",
    r"^hope\s+you\s+understood\b",
    r"^(hit|press|click)\s+(the\s+)?like\b",
    r"^comment\s+(down\s+)?below\b",
    r"^share\s+(this\s+)?video\b",
    r"^follow\s+(us\s+)?on\b",
    r"^(join|check)\s+(our|the)\s+(whatsapp|telegram)\b",
    r"^i\s+have\s+mentioned\s+complete\s+playlist\b",
    r"^in\s+(below\s+)?description\s+box\b",
    r"^refer\b.*?playlist\b",
    r"in\s+this\s+channel\s+you\s+will\s+get\b",
    r"in\s+today'?s\s+session\s+we\s+will\s+discuss\b",
    r"in\s+your\s+exam\s+they\s+will\s+ask\b",
    r"^hello\s+everyone\b",
    r"^welcome\s+back\b",
    r"for\s+confidential\s+support\s+call\b",
    r"visit\s+a\s+local\s+samaritans\b",
    r"national\s+suicide\s+prevention\s+line\b",
    r"full\s+video\s+transcript\s+is\s+available\b",
    r"at\s+the\s+end\s+of\s+this\s+session\s+you\s+will\s+learn\b",
    r"the\s+following\s+points\s+are\s+covered\b",
    r"this\s+video\s+includes\b",
    r"in\s+this\s+video\s*,?\s*we\s+will\s+learn\b",
    r"our\s+next\s+topic\s+in\s+the\s+first\s+unit\s+is\b",
]

# Promotional / exam noise
_PROMO_PATTERNS = [
    r"playlist",
    r"subscribe.*channel",
    r"link.*description",
    r"helpline",
    r"whatsapp",
    r"telegram",
    r"contact.*number",
]

# Incoherent / garbage patterns (found in the user's examples)
_GARBAGE_PATTERNS = [
    r"^[A-Za-z]:[\s]*$",                    # single letter lines
    r"(?:feed|fede|feede)[:\s;,]+",          # OCR garbage
    r"feeding[:;,\s]+feeding",               # repeated 'feeding' pattern
    r"te\s+te\b",                            # OCR fragment
    r"^Porse\s+Tree\b",                      # OCR fragment
    r"^\w{1,2}\s*$",                         # 1-2 char lines
]


# Check for Dictionary Availability
try:
    from nltk.corpus import words
    _ENGLISH_VOCAB = set(words.words())
except:
    _ENGLISH_VOCAB = set()

def _is_noise_sentence(sent: str) -> bool:
    """Check if a sentence is noise that should be removed."""
    sent_clean = sent.strip()
    sent_lower = sent_clean.lower()

    # 0. Rhetorical Questions (Summaries should be declarative)
    if sent_clean.endswith("?"):
        return True

    # 1. Too short to be informative
    words_in_sent = sent_lower.split()
    if len(words_in_sent) < 4:
        return True

    # 2. Image description artifacts
    for p in _IMAGE_DESC_PATTERNS:
        if re.search(p, sent_lower):
            return True

    # 3. BART hallucinations
    for p in _HALLUCINATION_PATTERNS:
        if re.search(p, sent, re.IGNORECASE):
            return True

    # 4. Meta-commentary
    for p in _META_COMMENTARY_PATTERNS:
        if re.search(p, sent_lower):
            return True

    # 5. Promotional content (strict)
    promo_hits = sum(1 for p in _PROMO_PATTERNS if re.search(p, sent_lower))
    if promo_hits >= 1:
        return True
    
    # 6. Specific Ignore List (User reported)
    if "neso academy" in sent_lower or "quality education" in sent_lower:
        return True

    # 7. Garbage / OCR noise
    for p in _GARBAGE_PATTERNS:
        if re.search(p, sent_lower):
            return True

    # 8. Mostly non-alphabetic characters (OCR garbage)
    alpha_ratio = sum(c.isalpha() for c in sent) / max(len(sent), 1)
    if alpha_ratio < 0.6:  # Increased strictness
        return True

    # 9. Dictionary Check (Gibberish / Reversed Text Filter)
    # If >50% of substantial words are NOT in dictionary, it's likely OCR fail or reversed.
    if _ENGLISH_VOCAB:
        long_words = [w for w in words_in_sent if len(w) > 4 and w.isalpha()]
        if len(long_words) >= 3:
            valid_count = sum(1 for w in long_words if w in _ENGLISH_VOCAB)
            if valid_count / len(long_words) < 0.5:
                return True

    return False


def _is_fragment(sent: str) -> bool:
    """Check if sentence is an incomplete fragment."""
    words = sent.split()
    if len(words) < 4:
        return True

    # Starts with lowercase and is not a continuation (likely a fragment)
    if sent[0].islower() and not sent.startswith(("i ", "i'", "e.g", "i.e")):
        return True

    # Check for dangling prepositions at end (indicates incomplete thought)
    dangling_ends = (" by.", " of.", " in.", " on.", " to.", " for.", " with.", " as.")
    if sent.lower().endswith(dangling_ends):
        return True

    # No verb indicator heuristic: check for common verb patterns
    verb_indicators = [
        " is ", " are ", " was ", " were ", " has ", " have ", " had ",
        " does ", " do ", " did ", " will ", " can ", " could ", " would ",
        " should ", " may ", " might ", " shall ", " must ",
        " means ", " refers ", " includes ", " contains ", " produces ",
        " generates ", " processes ", " converts ", " takes ", " reads ",
        " uses ", " performs ", " involves ", " requires ", " provides ",
        " creates ", " identifies ", " allocate", " optimize", " select",
        " called ", " known ",
    ]
    sent_lower = " " + sent.lower() + " "
    has_verb = any(v in sent_lower for v in verb_indicators)

    # Allow longer sentences even without detected verb (they may use less common verbs)
    if not has_verb and len(words) < 8:
        return True

    return False


def _sentence_token_overlap(s1: str, s2: str) -> float:
    """Calculate token overlap ratio between two sentences."""
    tokens1 = set(s1.lower().split())
    tokens2 = set(s2.lower().split())
    if not tokens1 or not tokens2:
        return 0.0
    intersection = tokens1 & tokens2
    # Jaccard-like: overlap / smaller set
    return len(intersection) / min(len(tokens1), len(tokens2))


def _deduplicate_sentences(sentences: List[str], threshold: float = 0.75) -> List[str]:
    """Remove near-duplicate sentences using token overlap."""
    if not sentences:
        return sentences

    unique = []
    for sent in sentences:
        is_dup = False
        for existing in unique:
            if _sentence_token_overlap(sent, existing) > threshold:
                # Keep the longer (more informative) one
                if len(sent) > len(existing):
                    unique.remove(existing)
                    unique.append(sent)
                is_dup = True
                break
        if not is_dup:
            unique.append(sent)

    return unique


def _score_sentence_informativeness(sent: str) -> float:
    """
    Score a sentence by informativeness (higher = more informative).
    
    Heuristic based on:
    - Length (medium is best)
    - Contains technical/definitional language
    - Contains specific nouns/terms
    """
    words = sent.split()
    word_count = len(words)

    # Base score from length (prefer medium sentences)
    if word_count < 6:
        score = 0.3
    elif word_count < 10:
        score = 0.6
    elif word_count < 25:
        score = 1.0
    elif word_count < 35:
        score = 0.8
    else:
        score = 0.6  # very long sentences are less ideal

    # Bonus for definitional / explanatory content
    definitional_words = [
        "is defined", "refers to", "is responsible", "is called",
        "is known", "is used", "is a", "means", "involves",
        "consists of", "includes", "processes", "converts",
        "generates", "produces", "performs", "allocates",
    ]
    sent_lower = sent.lower()
    for d in definitional_words:
        if d in sent_lower:
            score += 0.3
            break

    # Bonus for containing capitalized technical terms (not at start)
    cap_terms = re.findall(r'(?<!\. )[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', sent[1:])
    score += min(len(cap_terms) * 0.1, 0.3)

    # Penalty for starting with vague words
    vague_starters = ["it ", "this ", "that ", "there ", "here "]
    for v in vague_starters:
        if sent_lower.startswith(v):
            score -= 0.15
            break

    return score


def clean_and_constrain_summary(
    summary_text: str,
    min_sentences: int = 7,
    max_sentences: int = 10,
    remove_noise: bool = True,
    remove_fragments: bool = True,
    deduplicate: bool = True,
) -> str:
    """
    Universal summary post-processor.
    
    Takes raw summary text and produces a clean, context-rich summary
    of 7-10 sentences (configurable).
    
    Pipeline:
    1. Split into sentences
    2. Remove noise (hallucinations, image descriptions, meta-commentary)
    3. Remove fragments (incomplete sentences)
    4. Deduplicate near-identical sentences
    5. Rank by informativeness and constrain to target range
    
    Args:
        summary_text: The raw summary text
        min_sentences: Minimum sentences in output (default: 7)
        max_sentences: Maximum sentences in output (default: 10)
        remove_noise: Whether to filter noise sentences
        remove_fragments: Whether to filter sentence fragments
        deduplicate: Whether to deduplicate near-identical sentences
    
    Returns:
        Cleaned and constrained summary text
    """
    if not summary_text or not summary_text.strip():
        return summary_text

    # Step 0: Pre-clean whitespace and obvious garbage
    text = re.sub(r'\s+', ' ', summary_text).strip()

    # Step 1: Sentence splitting
    # Use regex-based splitting to handle various punctuation
    # First try nltk
    try:
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(text)
    except Exception:
        # Fallback: split on period + space + capital
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

    if not sentences:
        return summary_text

    # Step 2: Remove noise
    if remove_noise:
        sentences = [s for s in sentences if not _is_noise_sentence(s)]

    # Step 3: Remove fragments
    if remove_fragments:
        sentences = [s for s in sentences if not _is_fragment(s)]

    # Step 4: Deduplicate
    if deduplicate:
        sentences = _deduplicate_sentences(sentences, threshold=0.70)

    if not sentences:
        return summary_text  # Fallback: return original if everything was filtered

    # Step 5: Rank by informativeness and constrain
    if len(sentences) > max_sentences:
        # Score and keep top sentences, maintaining original order
        scored = [(i, s, _score_sentence_informativeness(s)) for i, s in enumerate(sentences)]
        scored.sort(key=lambda x: x[2], reverse=True)
        top_indices = sorted([x[0] for x in scored[:max_sentences]])
        sentences = [sentences[i] for i in top_indices]

    # Ensure proper sentence endings
    cleaned = []
    for s in sentences:
        s = s.strip()
        if s and not s.endswith(('.', '!', '?')):
            s += '.'
        # Ensure first character is uppercase
        if s and s[0].islower():
            s = s[0].upper() + s[1:]
        cleaned.append(s)

    result = ' '.join(cleaned)

    # Final sanity: if result is very short, return original
    if len(result) < 50 and len(summary_text) > 100:
        return summary_text

    return result
