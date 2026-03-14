# semantic_clusterer.py
"""
Semantic Clustering for KG-based Notes Generation

Implements Fix 1 & 2:
- Fix 1: One Section = One Core Concept (semantic clustering)
- Fix 2: Section-wise Summarization (summarize each cluster separately)

This prevents topic bleeding and improves structural metrics (ROUGE-L).
"""

import re
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_ML = True
except ImportError:
    HAS_ML = False
    print("[SemanticClusterer] Warning: ML libraries not found.")


class SemanticClusterer:
    """
    Clusters KG nodes and triples by semantic similarity.
    
    Each cluster represents one coherent topic/section for notes.
    Prevents topic bleeding (e.g., mixing Linked List with Stack).
    """
    
    # Role-based section categories (domain-agnostic)
    SECTION_ROLES = [
        ("definition", ["definition", "what is", "refers to", "means", "defined as", "is a", "overview", "introduction"]),
        ("structure", ["structure", "component", "consists of", "contains", "has", "includes", "made of", "parts", "elements"]),
        ("types", ["type", "kind", "category", "classification", "variant", "singly", "doubly", "circular", "primitive", "non-primitive"]),
        ("operations", ["operation", "function", "method", "traverse", "search", "sort", "insert", "delete", "access", "push", "pop"]),
        ("advantages", ["advantage", "benefit", "efficient", "fast", "dynamic", "flexible", "better"]),
        ("disadvantages", ["disadvantage", "limitation", "drawback", "slow", "overhead", "memory", "cannot"]),
        ("applications", ["application", "used for", "example", "practice", "implement", "real-world", "such as", "use case"]),
    ]
    
    def __init__(self, similarity_threshold: float = 0.75, min_cluster_size: int = 2):
        """
        Initialize the clusterer.
        
        Args:
            similarity_threshold: Minimum similarity for same cluster (0.0-1.0)
            min_cluster_size: Minimum nodes per cluster
        """
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size
        self.embedder = None
        
        if HAS_ML:
            try:
                self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception as e:
                print(f"[SemanticClusterer] Could not load embedder: {e}")
    
    def cluster_triples(self, triples: List[Tuple[str, str, str]]) -> Dict[str, List[Tuple[str, str, str]]]:
        """
        Cluster triples by semantic similarity of their subjects.
        
        Args:
            triples: List of (subject, relation, object) tuples
            
        Returns:
            Dict mapping cluster_name -> list of triples
        """
        if not triples:
            return {}
        
        # Extract unique subjects
        subjects = list(set(t[0] for t in triples))
        
        if len(subjects) <= 1:
            return {"main": triples}
        
        # Cluster subjects
        clusters = self._cluster_texts(subjects)
        
        # Map triples to clusters
        clustered_triples = defaultdict(list)
        for triple in triples:
            subject = triple[0]
            for cluster_name, cluster_subjects in clusters.items():
                if subject in cluster_subjects:
                    clustered_triples[cluster_name].append(triple)
                    break
            else:
                # Assign to "other" if not matched
                clustered_triples["other"].append(triple)
        
        return dict(clustered_triples)
    
    def cluster_sentences(self, sentences: List[str]) -> Dict[str, List[str]]:
        """
        Cluster sentences by semantic similarity.
        
        Args:
            sentences: List of sentences
            
        Returns:
            Dict mapping cluster_name -> list of sentences
        """
        if not sentences:
            return {}
        
        if len(sentences) <= 2:
            return {"main": sentences}
        
        # Cluster sentences directly
        return self._cluster_texts(sentences, return_as_dict=True)
    
    def _cluster_texts(self, texts: List[str], return_as_dict: bool = False) -> Dict[str, List[str]]:
        """
        Internal method to cluster texts using embeddings or fallback.
        """
        if self.embedder and HAS_ML:
            return self._cluster_with_embeddings(texts, return_as_dict)
        else:
            return self._cluster_with_keywords(texts, return_as_dict)
    
    def _cluster_with_embeddings(self, texts: List[str], return_as_dict: bool = False) -> Dict[str, List[str]]:
        """
        Cluster texts using sentence embeddings.
        """
        try:
            # Get embeddings
            embeddings = self.embedder.encode(texts, convert_to_numpy=True)
            
            # Calculate distance matrix
            distance_threshold = 1 - self.similarity_threshold
            
            # Agglomerative clustering
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_threshold,
                metric='cosine',
                linkage='average'
            )
            labels = clustering.fit_predict(embeddings)
            
            # Group texts by cluster label
            clusters = defaultdict(list)
            for text, label in zip(texts, labels):
                cluster_name = self._generate_cluster_name(text, label)
                clusters[cluster_name].append(text)
            
            return dict(clusters)
            
        except Exception as e:
            print(f"[SemanticClusterer] Embedding clustering failed: {e}")
            return self._cluster_with_keywords(texts, return_as_dict)
    
    def _cluster_with_keywords(self, texts: List[str], return_as_dict: bool = False) -> Dict[str, List[str]]:
        """
        Fallback clustering using keyword matching.
        """
        clusters = defaultdict(list)
        
        # Common topic keywords for fallback clustering
        topic_keywords = {
            "linked_list": ["linked", "node", "pointer", "head", "next"],
            "stack": ["stack", "push", "pop", "lifo", "top"],
            "queue": ["queue", "enqueue", "dequeue", "fifo", "front", "rear"],
            "array": ["array", "index", "element", "contiguous", "random access"],
            "tree": ["tree", "root", "leaf", "parent", "child", "binary"],
            "graph": ["graph", "vertex", "edge", "adjacency", "path"],
            "data_structure": ["data structure", "structure", "organize", "store"],
            "definition": ["definition", "what is", "means", "refers"],
            "operations": ["operation", "insert", "delete", "search", "traverse"],
            "applications": ["application", "used", "example", "implement"],
        }
        
        for text in texts:
            text_lower = text.lower()
            matched = False
            
            for topic, keywords in topic_keywords.items():
                if any(kw in text_lower for kw in keywords):
                    clusters[topic].append(text)
                    matched = True
                    break
            
            if not matched:
                clusters["other"].append(text)
        
        return dict(clusters)
    
    def _generate_cluster_name(self, sample_text: str, label: int) -> str:
        """
        Generate a descriptive name for a cluster based on its content.
        """
        text_lower = sample_text.lower()
        
        # Check for known topics
        topic_keywords = {
            "Linked List": ["linked", "node", "pointer"],
            "Stack": ["stack", "push", "pop", "lifo"],
            "Queue": ["queue", "enqueue", "dequeue", "fifo"],
            "Array": ["array", "index", "element"],
            "Tree": ["tree", "root", "leaf", "binary"],
            "Graph": ["graph", "vertex", "edge"],
            "Definition": ["definition", "what is", "means"],
            "Operations": ["operation", "insert", "delete"],
            "Applications": ["application", "used", "example"],
        }
        
        for name, keywords in topic_keywords.items():
            if any(kw in text_lower for kw in keywords):
                return name
        
        return f"Topic_{label}"
    
    def summarize_clusters(
        self, 
        clusters: Dict[str, List[str]], 
        summarizer_fn=None
    ) -> Dict[str, str]:
        """
        Summarize each cluster separately (Fix 2).
        
        Args:
            clusters: Dict of cluster_name -> sentences
            summarizer_fn: Optional function to summarize text
            
        Returns:
            Dict of cluster_name -> summary
        """
        summaries = {}
        
        for cluster_name, sentences in clusters.items():
            if not sentences:
                continue
            
            # Combine sentences for this cluster
            cluster_text = " ".join(sentences)
            
            if summarizer_fn:
                # Use provided summarization function
                summaries[cluster_name] = summarizer_fn(cluster_text)
            else:
                # Simple fallback: take first few sentences
                summaries[cluster_name] = self._simple_summarize(sentences)
        
        return summaries
    
    def _simple_summarize(self, sentences: List[str], max_sentences: int = 5) -> str:
        """
        Simple summarization: keep most informative sentences.
        """
        if len(sentences) <= max_sentences:
            return " ".join(sentences)
        
        # Score sentences by length and keyword density
        scored = []
        for sent in sentences:
            score = len(sent.split())  # Length as proxy for information
            # Boost sentences with definition keywords
            if any(kw in sent.lower() for kw in ["is a", "refers to", "defined as", "means"]):
                score += 10
            scored.append((score, sent))
        
        # Take top sentences
        scored.sort(reverse=True)
        selected = [sent for _, sent in scored[:max_sentences]]
        
        return " ".join(selected)
    
    def assign_section_roles(self, cluster_summaries: Dict[str, str]) -> Dict[str, str]:
        """
        Assign role-based section categories to cluster summaries.
        
        Returns ordered dict with role names as keys.
        """
        role_assignments = {}
        
        for cluster_name, summary in cluster_summaries.items():
            summary_lower = summary.lower()
            
            # Find best matching role
            best_role = "Content"  # Default
            best_score = 0
            
            for role, keywords in self.SECTION_ROLES:
                score = sum(1 for kw in keywords if kw in summary_lower)
                if score > best_score:
                    best_score = score
                    best_role = role.title()
            
            # If cluster name itself indicates a topic, use it
            if cluster_name not in ["main", "other", "Content"]:
                role_assignments[f"{cluster_name}"] = summary
            else:
                role_assignments[best_role] = summary
        
        return role_assignments
    
    def generate_structured_notes(
        self, 
        sentences: List[str], 
        summarizer_fn=None
    ) -> str:
        """
        Full pipeline: Cluster → Summarize → Structure → Output.
        
        Args:
            sentences: List of sentences from linearized KG
            summarizer_fn: Optional summarization function
            
        Returns:
            Structured notes text
        """
        # Step 1: Cluster sentences
        clusters = self.cluster_sentences(sentences)
        print(f"[SemanticClusterer] Created {len(clusters)} clusters")
        
        # Step 2: Summarize each cluster
        summaries = self.summarize_clusters(clusters, summarizer_fn)
        
        # Step 3: Assign section roles
        sections = self.assign_section_roles(summaries)
        
        # Step 4: Format as structured notes
        output_parts = []
        for section_name, content in sections.items():
            if content.strip():
                output_parts.append(f"{content}")
        
        return " ".join(output_parts)


# Convenience function
def cluster_and_summarize(sentences: List[str], summarizer_fn=None) -> str:
    """
    Cluster sentences and generate structured notes.
    """
    clusterer = SemanticClusterer()
    return clusterer.generate_structured_notes(sentences, summarizer_fn)


# ─── Cross-Section Deduplication (Change 8 / P3) ────────────────────────────

def deduplicate_across_sections(sections: List[Dict]) -> List[Dict]:
    """Remove near-duplicate bullets across all sections using TF-IDF cosine similarity.
    
    Threshold: 0.85 — keeps the topologically earlier occurrence.
    
    Args:
        sections: List of section dicts with 'subsections' -> 'points' structure
        
    Returns:
        Sections with duplicates removed
    """
    if not sections:
        return sections
    
    # Collect all bullet texts with their location
    all_bullets = []  # (section_idx, sub_idx, point_idx, text)
    for s_idx, section in enumerate(sections):
        for ss_idx, sub in enumerate(section.get('subsections', [])):
            for p_idx, pt in enumerate(sub.get('points', [])):
                text = pt.get('text', '').strip()
                if text:
                    all_bullets.append((s_idx, ss_idx, p_idx, text))
    
    if len(all_bullets) < 2:
        return sections
    
    # Compute TF-IDF similarity
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim
        
        texts = [b[3] for b in all_bullets]
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(texts)
        sim_matrix = cos_sim(tfidf_matrix)
        
        # Mark later duplicates for removal
        to_remove = set()
        for i in range(len(all_bullets)):
            if i in to_remove:
                continue
            for j in range(i + 1, len(all_bullets)):
                if j in to_remove:
                    continue
                if sim_matrix[i, j] > 0.85:
                    # Remove the later occurrence
                    to_remove.add(j)
        
        if to_remove:
            print(f"[Deduplication] Removing {len(to_remove)} duplicate bullets across sections")
        
        # Build removal set as (s_idx, ss_idx, p_idx)
        remove_keys = set()
        for idx in to_remove:
            s_idx, ss_idx, p_idx, _ = all_bullets[idx]
            remove_keys.add((s_idx, ss_idx, p_idx))
        
        # Filter points
        import copy
        result = copy.deepcopy(sections)
        for s_idx, section in enumerate(result):
            for ss_idx, sub in enumerate(section.get('subsections', [])):
                sub['points'] = [
                    pt for p_idx, pt in enumerate(sub.get('points', []))
                    if (s_idx, ss_idx, p_idx) not in remove_keys
                ]
        
        # Remove empty subsections
        for section in result:
            section['subsections'] = [
                sub for sub in section.get('subsections', [])
                if sub.get('points')
            ]
        
        return result
        
    except ImportError:
        print("[Deduplication] sklearn not available, skipping cross-section deduplication")
        return sections


def select_canonical_bullet(cluster_sentences: List[str]) -> str:
    """Pick the most informative representative from a cluster (Change 9).
    
    Uses vocabulary coverage score: unique_words / total_words.
    Higher ratio = more diverse vocabulary = more informative.
    
    Args:
        cluster_sentences: List of sentences in a cluster
        
    Returns:
        The most informative sentence
    """
    if not cluster_sentences:
        return ""
    if len(cluster_sentences) == 1:
        return cluster_sentences[0]
    
    best_sent = cluster_sentences[0]
    best_score = 0.0
    
    for sent in cluster_sentences:
        words = sent.lower().split()
        if not words:
            continue
        unique_ratio = len(set(words)) / len(words)
        # Bonus for longer sentences (more content)
        length_bonus = min(len(words) / 20.0, 1.0)
        score = unique_ratio * 0.7 + length_bonus * 0.3
        if score > best_score:
            best_score = score
            best_sent = sent
    
    return best_sent

