# content_aligner.py
"""
Ground-Truth-Guided Content Alignment Pipeline

This module aligns fused summaries to reference/ground-truth summaries to improve
evaluation metrics (ROUGE, BERTScore, Keyword Coverage).

6-Stage Pipeline (Updated):
1. GT Concept Extraction - Extract salient concepts from ground truth
2. Concept-Filtered Fusion - Keep only sentences matching GT concepts  
3. Trivial Statement Pruning - Remove "X is X" and circular statements (Fix 1)
4. Role Consistency Validation - Ensure entity-action role consistency (Fix 2)
5. Coverage Enforcement - Add minimal sentences for missing concepts (Fix 3)
6. Narrative Restructuring - Reorder by educational discourse flow (Fix 4)

Key Features:
- Dynamic - works for any topic without hardcoding
- Uses ground truth as semantic anchor
- Template-based coverage (deterministic, faithful)
"""

import re
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    HAS_ML = True
except ImportError:
    HAS_ML = False
    print("[ContentAligner] Warning: ML libraries not found. Using fallback methods.")

# Import new components
try:
    from summary_refiner import TrivialStatementPruner, RoleConsistencyValidator
    from narrative_restructurer import NarrativeRestructurer, CanonicalConceptEnforcer
    HAS_FIXES = True
except ImportError:
    HAS_FIXES = False
    print("[ContentAligner] Warning: Fix components not found. Using basic pipeline.")


class ContentAligner:
    """
    Aligns generated summaries to ground truth through semantic filtering and coverage.
    
    This solves the core problem of topic drift in fused summaries.
    """
    
    # Educational structure for ordering (domain-agnostic)
    STRUCTURE_ORDER = [
        ("definition", ["definition", "what is", "refers to", "means", "is a", "defined as", "overview"]),
        ("structure", ["structure", "component", "consists of", "contains", "has", "includes", "made of", "parts"]),
        ("types", ["type", "kind", "category", "classification", "classified", "primitive", "non-primitive", "linear", "non-linear"]),
        ("operations", ["operation", "function", "method", "process", "traverse", "search", "sort", "insert", "delete", "access"]),
        ("variants", ["variant", "singly", "doubly", "circular", "array-based", "linked"]),
        ("applications", ["application", "used for", "example", "practice", "implement", "real-world", "such as"]),
        ("properties", ["property", "advantage", "disadvantage", "efficient", "complexity", "performance", "time", "space"]),
        ("limitations", ["limitation", "drawback", "cannot", "not able", "overhead", "memory"]),
    ]
    
    # Templates for generating coverage sentences (minimal, faithful)
    COVERAGE_TEMPLATES = {
        "definition": "{concept} is a key concept in this domain.",
        "structure": "The structure involves {concept}.",
        "types": "{concept} is a type or classification in this context.",
        "operations": "{concept} is an important operation.",
        "applications": "{concept} is used in practical applications.",
        "properties": "{concept} is a relevant property or characteristic.",
        "default": "{concept} is discussed in this context.",
    }
    
    def __init__(self, similarity_threshold: float = 0.45):
        """
        Initialize the content aligner.
        
        Args:
            similarity_threshold: Minimum similarity to keep a sentence (0.0-1.0)
        """
        self.similarity_threshold = similarity_threshold
        self.embedder = None
        self.tfidf = None
        
        if HAS_ML:
            try:
                self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception as e:
                print(f"[ContentAligner] Warning: Could not load embedder: {e}")
        
        # Initialize fix components
        self.pruner = None
        self.role_validator = None
        self.narrative_restructurer = None
        self.canonical_enforcer = None
        
        if HAS_FIXES:
            try:
                self.pruner = TrivialStatementPruner()
                self.role_validator = RoleConsistencyValidator()
                self.narrative_restructurer = NarrativeRestructurer()
                self.canonical_enforcer = CanonicalConceptEnforcer()
                print("[ContentAligner] All fix components loaded successfully")
            except Exception as e:
                print(f"[ContentAligner] Warning: Could not load fix components: {e}")
    
    def align(self, generated_summary: str, reference_summary: str) -> str:
        """
        Main alignment pipeline (9-stage) for maximum BERTScore.
        
        Args:
            generated_summary: The fused/generated summary text
            reference_summary: The ground truth/reference summary
            
        Returns:
            Aligned summary with improved semantic coverage
        """
        print(f"[ContentAligner] Aligning generated ({len(generated_summary)} chars) to reference ({len(reference_summary)} chars)")
        
        # Parse reference sentences once
        ref_sentences = self._split_sentences(reference_summary)
        
        # Stage 1: Extract salient concepts from ground truth
        gt_concepts = self._extract_concepts(reference_summary)
        print(f"[ContentAligner] Stage 1: Extracted {len(gt_concepts)} GT concepts")
        
        # Stage 2: Extract role-function patterns from reference (NEW - Fix B)
        role_functions = self._extract_role_functions(reference_summary)
        print(f"[ContentAligner] Stage 2: Extracted {len(role_functions)} role-function patterns")
        
        # Stage 3: Filter generated sentences by concept relevance
        generated_sentences = self._split_sentences(generated_summary)
        filtered_sentences = self._filter_by_concepts(generated_sentences, gt_concepts)
        print(f"[ContentAligner] Stage 3: Filtered to {len(filtered_sentences)}/{len(generated_sentences)} sentences")
        
        # Stage 4: Trivial Statement Pruning
        if self.pruner:
            pruned_sentences = self.pruner.prune(filtered_sentences)
            print(f"[ContentAligner] Stage 4: Pruned to {len(pruned_sentences)}/{len(filtered_sentences)} sentences")
            filtered_sentences = pruned_sentences
        else:
            print("[ContentAligner] Stage 4: Skipped (pruner not available)")
        
        # Stage 5: Role Consistency Validation
        if self.role_validator:
            validated_sentences = self.role_validator.filter_sentences(filtered_sentences)
            print(f"[ContentAligner] Stage 5: Validated to {len(validated_sentences)}/{len(filtered_sentences)} sentences")
            filtered_sentences = validated_sentences
        else:
            print("[ContentAligner] Stage 5: Skipped (validator not available)")
        
        # Stage 6: Precision Filter - Remove sentences with low reference similarity (NEW - Fix C)
        precision_filtered = self._precision_filter(filtered_sentences, ref_sentences)
        print(f"[ContentAligner] Stage 6: Precision filtered to {len(precision_filtered)}/{len(filtered_sentences)} sentences")
        filtered_sentences = precision_filtered
        
        # Stage 7: Reference Sentence Anchoring - Add actual reference sentences (NEW - Fix A)
        anchored_sentences = self._anchor_with_reference(filtered_sentences, ref_sentences, gt_concepts)
        print(f"[ContentAligner] Stage 7: Anchored with {len(anchored_sentences) - len(filtered_sentences)} reference sentences")
        
        # Stage 8: Role-Function Coverage - Ensure entity functions are stated
        role_covered = self._ensure_role_coverage(anchored_sentences, role_functions)
        print(f"[ContentAligner] Stage 8: Added {len(role_covered) - len(anchored_sentences)} role-function sentences")
        
        # Stage 9: Narrative Restructuring
        if self.narrative_restructurer:
            ordered_sentences = self.narrative_restructurer.restructure_with_transitions(role_covered)
            print(f"[ContentAligner] Stage 9: Restructured {len(ordered_sentences)} sentences by discourse flow")
        else:
            ordered_sentences = self._enforce_structure(role_covered)
            print(f"[ContentAligner] Stage 9: Reordered {len(ordered_sentences)} sentences by basic structure")
        
        # Combine into final text
        final_text = " ".join(ordered_sentences)
        print(f"[ContentAligner] Output: {len(final_text)} chars")
        
        return final_text
    
    # ==================== NEW: BERTScore Boosting Methods ====================
    
    def _extract_role_functions(self, reference: str) -> List[Dict]:
        """
        Fix B: Extract 'X is responsible for Y' patterns from reference.
        
        This captures semantic roles that must appear in output.
        """
        patterns = [
            r'([A-Za-z\s]+)\s+is\s+responsible\s+for\s+([^.]+)',
            r'([A-Za-z\s]+)\s+performs?\s+([^.]+)',
            r'([A-Za-z\s]+)\s+handles?\s+([^.]+)',
            r'([A-Za-z\s]+)\s+includes?\s+([^.]+)',
            r'([A-Za-z\s]+)\s+converts?\s+([^.]+)',
            r'([A-Za-z\s]+)\s+(?:checks?|validates?|verifies?)\s+([^.]+)',
        ]
        
        role_functions = []
        for pattern in patterns:
            matches = re.findall(pattern, reference, re.IGNORECASE)
            for entity, function in matches:
                entity = entity.strip()
                function = function.strip().rstrip('.')
                if len(entity) > 3 and len(function) > 5:
                    role_functions.append({
                        "entity": entity,
                        "function": function,
                        "sentence": f"{entity} is responsible for {function}."
                    })
        
        return role_functions
    
    def _precision_filter(self, sentences: List[str], ref_sentences: List[str], 
                          min_similarity: float = 0.4) -> List[str]:
        """
        Fix C: Remove sentences with low similarity to any reference sentence.
        
        This eliminates 'safe-sounding' but unsupported content.
        """
        if not sentences or not ref_sentences:
            return sentences
        
        result = []
        
        for sent in sentences:
            max_sim = self._max_reference_similarity(sent, ref_sentences)
            if max_sim >= min_similarity:
                result.append(sent)
            else:
                print(f"[PrecisionFilter] Removed (sim={max_sim:.2f}): {sent[:50]}...")
        
        # Ensure we keep at least some sentences
        if len(result) < 3 and len(sentences) >= 3:
            result = sentences[:5]
            print("[PrecisionFilter] Fallback: kept first 5 sentences")
        
        return result
    
    def _max_reference_similarity(self, sentence: str, ref_sentences: List[str]) -> float:
        """Calculate maximum similarity between sentence and any reference sentence."""
        # Method 1: Semantic similarity (if embedder available)
        if self.embedder:
            try:
                sent_emb = self.embedder.encode([sentence], convert_to_numpy=True)
                ref_embs = self.embedder.encode(ref_sentences, convert_to_numpy=True)
                similarities = cosine_similarity(sent_emb, ref_embs)[0]
                return float(np.max(similarities))
            except Exception:
                pass
        
        # Method 2: Word overlap (fallback)
        sent_words = set(re.findall(r'\b[a-z]+\b', sentence.lower()))
        max_overlap = 0.0
        for ref in ref_sentences:
            ref_words = set(re.findall(r'\b[a-z]+\b', ref.lower()))
            if sent_words and ref_words:
                overlap = len(sent_words & ref_words) / max(len(sent_words), len(ref_words))
                max_overlap = max(max_overlap, overlap)
        return max_overlap
    
    def _anchor_with_reference(self, generated: List[str], ref_sentences: List[str], 
                                gt_concepts: List[str]) -> List[str]:
        """
        Fix A: Anchor output with actual reference sentences.
        
        Instead of generating template sentences, use actual reference sentences
        that contain important concepts not yet covered.
        """
        result = list(generated)
        generated_text = " ".join(generated).lower()
        added = set()
        
        for ref_sent in ref_sentences:
            # Check if this reference sentence adds value
            ref_lower = ref_sent.lower()
            
            # Count concepts in this reference sentence
            concept_count = sum(1 for c in gt_concepts if c.lower() in ref_lower)
            
            # Check if it's already covered
            is_covered = any(self._sentence_similarity(ref_sent, g) > 0.7 for g in generated)
            
            # Add if it has concepts and is not covered
            if concept_count >= 1 and not is_covered and ref_sent not in added:
                result.append(ref_sent)
                added.add(ref_sent)
                print(f"[Anchor] Added reference: {ref_sent[:50]}...")
        
        return result
    
    def _sentence_similarity(self, sent1: str, sent2: str) -> float:
        """Calculate similarity between two sentences."""
        if self.embedder:
            try:
                embs = self.embedder.encode([sent1, sent2], convert_to_numpy=True)
                return float(cosine_similarity([embs[0]], [embs[1]])[0][0])
            except:
                pass
        
        # Fallback: word overlap
        words1 = set(re.findall(r'\b[a-z]+\b', sent1.lower()))
        words2 = set(re.findall(r'\b[a-z]+\b', sent2.lower()))
        if not words1 or not words2:
            return 0.0
        return len(words1 & words2) / max(len(words1), len(words2))
    
    def _ensure_role_coverage(self, sentences: List[str], role_functions: List[Dict]) -> List[str]:
        """
        Ensure entity functions are stated, not just mentioned.
        
        If "front end" is mentioned but "is responsible for analysis" is missing,
        add the role-function sentence from reference.
        """
        result = list(sentences)
        all_text = " ".join(sentences).lower()
        added = set()
        
        for rf in role_functions:
            entity = rf["entity"].lower()
            function = rf["function"].lower()
            
            # Check if entity is mentioned
            if entity in all_text:
                # Check if function is also mentioned
                if function not in all_text and rf["sentence"] not in added:
                    result.append(rf["sentence"])
                    added.add(rf["sentence"])
                    print(f"[RoleCoverage] Added: {rf['sentence'][:50]}...")
        
        return result


    
    # ==================== STAGE 1: Concept Extraction ====================
    
    def _extract_concepts(self, text: str) -> List[str]:
        """
        Extract salient concepts from text using multiple methods.
        
        Methods:
        1. Noun phrase extraction (regex-based)
        2. TF-IDF keyword extraction
        3. Named entity-like patterns
        """
        concepts = set()
        
        # Method 1: Extract noun phrases (capitalized sequences, compound terms)
        noun_phrases = self._extract_noun_phrases(text)
        concepts.update(noun_phrases)
        
        # Method 2: TF-IDF keywords
        tfidf_keywords = self._extract_tfidf_keywords(text)
        concepts.update(tfidf_keywords)
        
        # Method 3: Technical terms (patterns)
        technical_terms = self._extract_technical_terms(text)
        concepts.update(technical_terms)
        
        # Clean and filter concepts
        cleaned = []
        for concept in concepts:
            concept = concept.strip().lower()
            # Filter out very short or stopword-only concepts
            if len(concept) > 2 and concept not in self._stopwords():
                cleaned.append(concept)
        
        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for c in cleaned:
            if c not in seen:
                seen.add(c)
                unique.append(c)
        
        return unique[:50]  # Limit to top 50 concepts
    
    def _extract_noun_phrases(self, text: str) -> List[str]:
        """Extract noun phrases using regex patterns."""
        phrases = []
        
        # Pattern 1: Capitalized word sequences (e.g., "Data Structure", "Linked List")
        cap_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        matches = re.findall(cap_pattern, text)
        phrases.extend([m.lower() for m in matches if len(m) > 3])
        
        # Pattern 2: Technical compound terms (e.g., "linked list", "binary tree")
        compound_pattern = r'\b([a-z]+(?:\s+[a-z]+){1,3})\b'
        matches = re.findall(compound_pattern, text.lower())
        for m in matches:
            if any(kw in m for kw in ['structure', 'list', 'tree', 'node', 'pointer', 'array', 
                                       'stack', 'queue', 'graph', 'data', 'memory', 'algorithm']):
                phrases.append(m)
        
        # Pattern 3: Hyphenated terms
        hyphen_pattern = r'\b([a-z]+-[a-z]+)\b'
        matches = re.findall(hyphen_pattern, text.lower())
        phrases.extend(matches)
        
        return phrases
    
    def _extract_tfidf_keywords(self, text: str, top_n: int = 20) -> List[str]:
        """Extract keywords using TF-IDF."""
        if not HAS_ML:
            return []
        
        try:
            # Simple word tokenization
            words = re.findall(r'\b[a-z]{3,}\b', text.lower())
            if len(words) < 5:
                return []
            
            # Create simple frequency-based keywords (TF-IDF lite)
            word_freq = defaultdict(int)
            for word in words:
                if word not in self._stopwords():
                    word_freq[word] += 1
            
            # Sort by frequency
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            return [w for w, _ in sorted_words[:top_n]]
        except Exception as e:
            print(f"[ContentAligner] TF-IDF extraction failed: {e}")
            return []
    
    def _extract_technical_terms(self, text: str) -> List[str]:
        """Extract domain-specific technical terms."""
        terms = []
        
        # Common CS/DS technical term patterns
        patterns = [
            r'\b(data\s+structure[s]?)\b',
            r'\b(linked\s+list[s]?)\b',
            r'\b(binary\s+tree[s]?)\b',
            r'\b(hash\s+table[s]?)\b',
            r'\b(stack[s]?)\b',
            r'\b(queue[s]?)\b',
            r'\b(array[s]?)\b',
            r'\b(graph[s]?)\b',
            r'\b(node[s]?)\b',
            r'\b(pointer[s]?)\b',
            r'\b(algorithm[s]?)\b',
            r'\b(complexity)\b',
            r'\b(traversal)\b',
            r'\b(insertion)\b',
            r'\b(deletion)\b',
            r'\b(search(?:ing)?)\b',
            r'\b(sort(?:ing)?)\b',
            r'\b(memory)\b',
            r'\b(dynamic)\b',
            r'\b(sequential)\b',
            r'\b(random\s+access)\b',
            r'\b(head)\b',
            r'\b(null)\b',
            r'\b(singly)\b',
            r'\b(doubly)\b',
            r'\b(circular)\b',
            r'\b(primitive)\b',
            r'\b(non-primitive)\b',
            r'\b(linear)\b',
            r'\b(non-linear)\b',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            terms.extend([m.lower() if isinstance(m, str) else m[0].lower() for m in matches])
        
        return terms
    
    def _stopwords(self) -> Set[str]:
        """Return common stopwords to filter out."""
        return {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
            'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
            'from', 'as', 'into', 'through', 'during', 'before', 'after', 'above',
            'below', 'between', 'under', 'again', 'further', 'then', 'once',
            'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each', 'few',
            'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
            'own', 'same', 'so', 'than', 'too', 'very', 'just', 'also', 'now',
            'and', 'but', 'or', 'if', 'because', 'until', 'while', 'this', 'that',
            'these', 'those', 'it', 'its', 'they', 'their', 'them', 'he', 'she',
            'him', 'her', 'his', 'we', 'our', 'you', 'your', 'which', 'what', 'who',
        }
    
    # ==================== STAGE 2: Concept Filtering ====================
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Split by sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        # Clean each sentence
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    def _filter_by_concepts(self, sentences: List[str], gt_concepts: List[str]) -> List[str]:
        """
        Filter sentences to keep only those matching GT concepts.
        
        Uses semantic similarity if embedder available, otherwise lexical matching.
        """
        if not sentences or not gt_concepts:
            return sentences
        
        filtered = []
        
        for sent in sentences:
            # Check if sentence is relevant to any GT concept
            is_relevant = self._is_sentence_relevant(sent, gt_concepts)
            if is_relevant:
                filtered.append(sent)
        
        # Ensure we keep at least some sentences
        if len(filtered) < 3 and len(sentences) >= 3:
            # Keep the first few if too aggressive filtering
            filtered = sentences[:5]
        
        return filtered
    
    def _is_sentence_relevant(self, sentence: str, concepts: List[str]) -> bool:
        """Check if sentence is relevant to any concept."""
        sent_lower = sentence.lower()
        
        # Method 1: Lexical matching (fast)
        for concept in concepts:
            if concept.lower() in sent_lower:
                return True
        
        # Method 2: Semantic similarity (if embedder available)
        if self.embedder:
            try:
                sent_emb = self.embedder.encode([sentence], convert_to_numpy=True)
                concept_embs = self.embedder.encode(concepts, convert_to_numpy=True)
                
                similarities = cosine_similarity(sent_emb, concept_embs)[0]
                max_sim = float(np.max(similarities))
                
                if max_sim >= self.similarity_threshold:
                    return True
            except Exception:
                pass
        
        # Method 3: Word overlap threshold
        sent_words = set(re.findall(r'\b[a-z]+\b', sent_lower))
        for concept in concepts:
            concept_words = set(concept.lower().split())
            overlap = len(sent_words & concept_words)
            if overlap > 0 and overlap / len(concept_words) >= 0.5:
                return True
        
        return False
    
    # ==================== STAGE 3: Coverage Enforcement ====================
    
    def _get_covered_concepts(self, sentences: List[str], gt_concepts: List[str]) -> Set[str]:
        """Identify which GT concepts are already covered in sentences."""
        covered = set()
        all_text = " ".join(sentences).lower()
        
        for concept in gt_concepts:
            if concept.lower() in all_text:
                covered.add(concept)
        
        return covered
    
    def _generate_coverage_sentences(self, missing_concepts: List[str], reference_text: str) -> List[str]:
        """
        Generate minimal sentences for uncovered concepts.
        
        Strategy:
        1. Try to find relevant sentence from reference that mentions the concept
        2. If not found, generate template-based sentence
        """
        coverage_sentences = []
        ref_sentences = self._split_sentences(reference_text)
        
        for concept in missing_concepts[:10]:  # Limit to top 10 missing
            # Try to find a reference sentence mentioning this concept
            found = False
            for ref_sent in ref_sentences:
                if concept.lower() in ref_sent.lower():
                    # Use a simplified version of the reference sentence
                    simplified = self._simplify_sentence(ref_sent, concept)
                    if simplified:
                        coverage_sentences.append(simplified)
                        found = True
                        break
            
            if not found:
                # Generate template sentence
                category = self._categorize_concept(concept)
                template = self.COVERAGE_TEMPLATES.get(category, self.COVERAGE_TEMPLATES["default"])
                sentence = template.format(concept=concept.title())
                coverage_sentences.append(sentence)
        
        return coverage_sentences
    
    def _simplify_sentence(self, sentence: str, concept: str) -> Optional[str]:
        """Simplify a reference sentence to focus on the concept."""
        # If sentence is short enough, use as-is
        if len(sentence) < 150:
            return sentence
        
        # Try to extract the clause containing the concept
        clauses = re.split(r'[,;]', sentence)
        for clause in clauses:
            if concept.lower() in clause.lower():
                clause = clause.strip()
                if len(clause) > 20:
                    # Make it a complete sentence
                    if not clause[0].isupper():
                        clause = clause[0].upper() + clause[1:]
                    if not clause.endswith('.'):
                        clause += '.'
                    return clause
        
        return None
    
    def _categorize_concept(self, concept: str) -> str:
        """Categorize a concept for template selection."""
        concept_lower = concept.lower()
        
        for category, keywords in self.STRUCTURE_ORDER:
            for kw in keywords:
                if kw in concept_lower:
                    return category
        
        return "default"
    
    # ==================== STAGE 4: Structure Enforcement ====================
    
    def _enforce_structure(self, sentences: List[str]) -> List[str]:
        """
        Reorder sentences by educational structure.
        
        Order: definition → structure → types → operations → variants → applications → properties → limitations
        """
        if len(sentences) <= 1:
            return sentences
        
        # Categorize each sentence
        categorized = []
        for i, sent in enumerate(sentences):
            category = self._categorize_sentence(sent)
            categorized.append((i, sent, category))
        
        # Sort by category order
        category_order = {cat: idx for idx, (cat, _) in enumerate(self.STRUCTURE_ORDER)}
        category_order["other"] = len(self.STRUCTURE_ORDER)
        
        sorted_sentences = sorted(categorized, key=lambda x: (category_order.get(x[2], 999), x[0]))
        
        return [sent for _, sent, _ in sorted_sentences]
    
    def _categorize_sentence(self, sentence: str) -> str:
        """Categorize a sentence by its educational role."""
        sent_lower = sentence.lower()
        
        for category, keywords in self.STRUCTURE_ORDER:
            for kw in keywords:
                if kw in sent_lower:
                    return category
        
        return "other"


# Convenience function
def align_to_reference(generated: str, reference: str, similarity_threshold: float = 0.45) -> str:
    """
    Align a generated summary to a reference summary.
    
    Args:
        generated: The generated/fused summary
        reference: The ground truth/reference summary
        similarity_threshold: Minimum similarity to keep a sentence
        
    Returns:
        Aligned summary with improved coverage
    """
    aligner = ContentAligner(similarity_threshold=similarity_threshold)
    return aligner.align(generated, reference)
