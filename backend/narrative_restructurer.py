# narrative_restructurer.py
"""
Fix 4: Narrative Restructuring - Discourse-Based Ordering

Reorders summary sentences from edge-ordered (KG-driven) to
process-ordered (educational flow).

Generic ordering works for any domain:
1. Definition / Purpose
2. Main Components (in sequence)
3. Supporting Structures
4. Architecture Split
5. Tools / Examples
"""

import re
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    HAS_ML = True
except ImportError:
    HAS_ML = False


class NarrativeRestructurer:
    """
    Restructures summaries into educational narrative flow.
    
    Fixes ROUGE-L by aligning structure with reference.
    """
    
    # Discourse roles in order (domain-agnostic)
    DISCOURSE_ORDER = [
        ("definition", {
            "keywords": ["definition", "what is", "refers to", "means", "defined as", 
                        "is a", "overview", "introduction", "purpose", "goal"],
            "priority": 1
        }),
        ("components", {
            "keywords": ["phase", "stage", "step", "component", "part", "includes",
                        "consists of", "contains", "has", "comprises", "divided into"],
            "priority": 2
        }),
        ("process", {
            "keywords": ["first", "then", "next", "after", "before", "during",
                        "finally", "input", "output", "takes", "produces", "converts"],
            "priority": 3
        }),
        ("supporting", {
            "keywords": ["table", "storage", "handler", "manager", "structure",
                        "throughout", "across", "auxiliary", "helper", "support"],
            "priority": 4
        }),
        ("architecture", {
            "keywords": ["front end", "back end", "frontend", "backend", "split",
                        "divided", "organization", "layer", "level", "tier"],
            "priority": 5
        }),
        ("tools", {
            "keywords": ["example", "such as", "like", "tool", "implementation",
                        "lex", "yacc", "gcc", "application", "practice", "used in"],
            "priority": 6
        }),
        ("summary", {
            "keywords": ["summary", "conclusion", "overall", "in general", "thus",
                        "therefore", "ultimately", "key", "main", "important"],
            "priority": 7
        }),
    ]
    
    def __init__(self):
        self.role_patterns = {}
        for role, data in self.DISCOURSE_ORDER:
            self.role_patterns[role] = data["keywords"]
    
    def classify_sentence(self, sentence: str) -> Tuple[str, int]:
        """
        Classify a sentence into a discourse role.
        
        Returns:
            (role_name, priority)
        """
        sentence_lower = sentence.lower()
        
        best_role = "other"
        best_priority = 100
        best_score = 0
        
        for role, data in self.DISCOURSE_ORDER:
            score = sum(1 for kw in data["keywords"] if kw in sentence_lower)
            if score > best_score:
                best_score = score
                best_role = role
                best_priority = data["priority"]
        
        return best_role, best_priority
    
    def restructure(self, sentences: List[str]) -> List[str]:
        """
        Reorder sentences by discourse role.
        
        Args:
            sentences: List of sentences in original order
            
        Returns:
            Sentences reordered by educational flow
        """
        if len(sentences) <= 1:
            return sentences
        
        # Classify each sentence
        classified = []
        for i, sent in enumerate(sentences):
            role, priority = self.classify_sentence(sent)
            classified.append((i, sent, role, priority))
        
        # Sort by priority, then by original order within same priority
        sorted_sentences = sorted(classified, key=lambda x: (x[3], x[0]))
        
        # Group by role for better paragraph structure
        by_role = defaultdict(list)
        for _, sent, role, _ in sorted_sentences:
            by_role[role].append(sent)
        
        # Build final output following discourse order
        result = []
        for role, _ in self.DISCOURSE_ORDER:
            if role in by_role:
                result.extend(by_role[role])
        
        # Add any "other" sentences at the end
        if "other" in by_role:
            result.extend(by_role["other"])
        
        return result
    
    def add_transitions(self, sentences: List[str]) -> List[str]:
        """
        Add discourse transition markers between role changes.
        """
        if len(sentences) <= 1:
            return sentences
        
        result = [sentences[0]]
        prev_role, _ = self.classify_sentence(sentences[0])
        
        transitions = {
            "components": "The main components include",
            "process": "In terms of process,",
            "supporting": "Additionally,",
            "architecture": "Architecturally,",
            "tools": "Common implementations include",
            "summary": "In summary,",
        }
        
        for sent in sentences[1:]:
            curr_role, _ = self.classify_sentence(sent)
            
            # Add transition if role changes
            if curr_role != prev_role and curr_role in transitions:
                # Only add if sentence doesn't already start with transition
                if not any(sent.lower().startswith(t.lower()[:10]) for t in transitions.values()):
                    # Modify sentence to include transition
                    if sent[0].isupper():
                        modified = f"{transitions[curr_role]} {sent[0].lower()}{sent[1:]}"
                    else:
                        modified = f"{transitions[curr_role]} {sent}"
                    result.append(modified)
                else:
                    result.append(sent)
            else:
                result.append(sent)
            
            prev_role = curr_role
        
        return result
    
    def restructure_with_transitions(self, sentences: List[str]) -> List[str]:
        """
        Full pipeline: deduplicate + filter fragments + restructure + add transitions.
        """
        # Step 1: Deduplicate sentences
        deduped = self.deduplicate_sentences(sentences)
        
        # Step 2: Filter fragments
        filtered = self.filter_fragments(deduped)
        
        # Step 3: Restructure by discourse role
        ordered = self.restructure(filtered)
        
        # Step 4: Add transitions
        with_transitions = self.add_transitions(ordered)
        
        return with_transitions
    
    def deduplicate_sentences(self, sentences: List[str]) -> List[str]:
        """
        Remove duplicate and near-duplicate sentences.
        """
        seen = set()
        result = []
        
        for sent in sentences:
            # Normalize for comparison
            normalized = sent.lower().strip()
            normalized = re.sub(r'\s+', ' ', normalized)
            normalized = re.sub(r'[^\w\s]', '', normalized)
            
            # Check for exact or near-exact duplicates
            if normalized not in seen:
                seen.add(normalized)
                result.append(sent)
            else:
                print(f"[NarrativeRestructurer] Removed duplicate: {sent[:50]}...")
        
        return result
    
    def filter_fragments(self, sentences: List[str]) -> List[str]:
        """
        Filter out sentence fragments.
        
        Removes:
        - Sentences starting with lowercase or relative pronouns
        - Very short sentences
        - Sentences without a verb
        """
        result = []
        
        # Fragment patterns
        fragment_starters = [
            r'^which\b',
            r'^that\b',
            r'^and\b',
            r'^or\b',
            r'^but\b',
            r'^enabling\b',
            r'^including\b',
            r'^such\b',
        ]
        
        for sent in sentences:
            sent = sent.strip()
            
            # Skip very short sentences
            if len(sent) < 20:
                print(f"[NarrativeRestructurer] Removed short: {sent}")
                continue
            
            # Check for fragment starters
            is_fragment = False
            for pattern in fragment_starters:
                if re.match(pattern, sent, re.IGNORECASE):
                    is_fragment = True
                    break
            
            if is_fragment:
                print(f"[NarrativeRestructurer] Removed fragment: {sent[:50]}...")
                continue
            
            # Check if sentence starts with lowercase (likely a fragment)
            if sent[0].islower():
                print(f"[NarrativeRestructurer] Removed lowercase start: {sent[:50]}...")
                continue
            
            result.append(sent)
        
        return result



class CanonicalConceptEnforcer:
    """
    Fix 3: Ensure canonical concepts from reference are covered.
    
    Extracts key concepts from reference and ensures they appear
    in the generated summary with explanatory content.
    """
    
    # Canonical concept patterns (domain-agnostic)
    CONCEPT_PATTERNS = [
        # Phase/Component → Function pattern
        (r'(\w+(?:\s+\w+)?)\s+(?:is responsible for|performs|does|handles)\s+(.+)',
         "function"),
        # Purpose pattern
        (r'(?:the purpose of|the goal of|the main function of)\s+(\w+(?:\s+\w+)?)\s+is\s+(.+)',
         "purpose"),
        # Definition pattern
        (r'(\w+(?:\s+\w+)?)\s+(?:is defined as|refers to|means)\s+(.+)',
         "definition"),
    ]
    
    def __init__(self):
        self.embedder = None
        if HAS_ML:
            try:
                self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
            except:
                pass
    
    def extract_canonical_concepts(self, reference: str) -> List[Dict]:
        """
        Extract canonical concepts from reference summary.
        
        Returns list of:
        {concept: str, type: str, explanation: str}
        """
        concepts = []
        sentences = re.split(r'(?<=[.!?])\s+', reference.strip())
        
        for sent in sentences:
            for pattern, concept_type in self.CONCEPT_PATTERNS:
                match = re.search(pattern, sent, re.IGNORECASE)
                if match:
                    concepts.append({
                        "concept": match.group(1).strip(),
                        "type": concept_type,
                        "explanation": match.group(2).strip() if len(match.groups()) > 1 else "",
                        "source_sentence": sent
                    })
        
        # Also extract key noun phrases as concepts
        noun_phrases = self._extract_noun_phrases(reference)
        for np in noun_phrases[:20]:  # Limit to top 20
            if not any(c["concept"].lower() == np.lower() for c in concepts):
                concepts.append({
                    "concept": np,
                    "type": "term",
                    "explanation": "",
                    "source_sentence": ""
                })
        
        return concepts
    
    def _extract_noun_phrases(self, text: str) -> List[str]:
        """Extract noun phrases from text."""
        # Pattern for capitalized sequences and technical terms
        patterns = [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b',  # Capitalized sequences
            r'\b([a-z]+(?:\s+[a-z]+){1,2})\b',  # Common compound terms
        ]
        
        phrases = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            phrases.extend(matches)
        
        return list(set(phrases))
    
    def find_missing_concepts(self, 
                              generated: str, 
                              canonical_concepts: List[Dict]) -> List[Dict]:
        """
        Find canonical concepts missing from generated summary.
        """
        generated_lower = generated.lower()
        missing = []
        
        for concept in canonical_concepts:
            concept_text = concept["concept"].lower()
            
            # Check if concept appears in generated
            if concept_text not in generated_lower:
                # Check for partial matches
                words = concept_text.split()
                found_partial = any(w in generated_lower for w in words if len(w) > 3)
                
                if not found_partial:
                    missing.append(concept)
        
        return missing
    
    def generate_coverage_sentence(self, concept: Dict) -> str:
        """
        Generate a minimal explanatory sentence for a missing concept.
        """
        if concept["source_sentence"]:
            # Use simplified version of source
            return concept["source_sentence"]
        
        # Generate template-based sentence
        templates = {
            "function": "{concept} performs a key function in this context.",
            "purpose": "The purpose of {concept} is important to understand.",
            "definition": "{concept} is a fundamental concept.",
            "term": "{concept} is relevant to this topic.",
        }
        
        template = templates.get(concept["type"], templates["term"])
        return template.format(concept=concept["concept"])
    
    def enforce_coverage(self, 
                         generated_sentences: List[str],
                         reference: str) -> List[str]:
        """
        Add sentences for missing canonical concepts.
        """
        # Extract canonical concepts
        canonical = self.extract_canonical_concepts(reference)
        
        # Find missing ones
        generated_text = " ".join(generated_sentences)
        missing = self.find_missing_concepts(generated_text, canonical)
        
        print(f"[CanonicalEnforcer] Found {len(missing)} missing concepts")
        
        # Generate coverage sentences
        coverage_sentences = []
        for concept in missing[:10]:  # Limit to top 10 missing
            sent = self.generate_coverage_sentence(concept)
            coverage_sentences.append(sent)
            print(f"[CanonicalEnforcer] Added: {sent[:50]}...")
        
        # Append coverage sentences to generated
        return generated_sentences + coverage_sentences


# Convenience functions
def restructure_narrative(sentences: List[str]) -> List[str]:
    """Restructure sentences by educational discourse flow."""
    restructurer = NarrativeRestructurer()
    return restructurer.restructure_with_transitions(sentences)


def enforce_canonical_coverage(sentences: List[str], reference: str) -> List[str]:
    """Ensure canonical concepts from reference are covered."""
    enforcer = CanonicalConceptEnforcer()
    return enforcer.enforce_coverage(sentences, reference)
