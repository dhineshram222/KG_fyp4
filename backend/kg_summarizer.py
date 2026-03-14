import networkx as nx
import numpy as np
from collections import defaultdict, Counter
from transformers import pipeline, BartTokenizer, BartForConditionalGeneration
from typing import List, Dict, Any, Optional, Tuple
import torch
import warnings
import re
import math

# Import the summary refiner for structure-aware sentence reconstruction
try:
    from summary_refiner import FusedSummaryRefiner
    HAS_REFINER = True
except ImportError:
    HAS_REFINER = False
    print("[Warning] summary_refiner not found. Fused summary refinement disabled.")

# Suppress warnings
warnings.filterwarnings("ignore")

# ── Relation Expansion Templates (Change 2 / P2 + Fix 5.2) ──
# Maps edge relation types to richer sentence patterns.
# {S} = subject label, {O} = object label, {D} = subject node description
EXPANSION_TEMPLATES = {
    'provides':      '{S} provides {O}, {D}',
    'ensures':       '{S} ensures {O}, {D}',
    'is_a':          '{S} is a type of {O}. {D}',
    'is a':          '{S} is a type of {O}. {D}',
    'defines':       '{S} defines {O}. {D}',
    'uses':          '{S} uses {O}. {D}',
    'implemented_by':'{S} is implemented by {O}. {D}',
    'part_of':       '{O} includes {S} as a component. {D}',
    'causes':        '{S} causes {O}. {D}',
    'results_in':    '{S} results in {O}. {D}',
    'depends_on':    '{S} depends on {O} being established first. {D}',
    'property':      '{S} has the property of being {O}. {D}',
    'compares_to':   '{S} can be compared to {O}. {D}',
    'connects':      '{S} connects to {O}. {D}',
    'achieves_via':  '{S} achieves this via {O}. {D}',
    'responsible_for':'{S} is responsible for {O}. {D}',
    # Fix 5.2 — New relation types for TCP/IP and protocol-heavy lectures
    'merges_functionalities_of': '{S} consolidates the roles of {O} into a single unified layer. {D}',
    'uses_protocol':  '{S} operates using the {O} protocol for data exchange. {D}',
    'corresponds_to': '{S} is functionally equivalent to {O} in the OSI reference model. {D}',
    'contains_protocol': '{S} includes the {O} protocol. {D}',
    'encapsulates':   '{S} encapsulates data from {O} by adding headers. {D}',
    'transforms_into': '{S} is transformed into {O} during data transmission. {D}',
    'operates_at':    '{S} operates at the {O}. {D}',
    'developed_by':   '{S} was developed by {O}. {D}',
    'precedes':       '{S} was established before {O}. {D}',
    'has_layer':      '{S} includes {O} as one of its layers. {D}',
    'supports':       '{S} supports {O}. {D}',
    'replaces':       '{S} replaces or supersedes {O}. {D}',
    'is_associated_with': '{S} is closely associated with {O}. {D}',
    'stands_for':     '{S} stands for {O}. {D}',
}


def _truncate_at_boundary(text: str, max_chars: int = 150) -> str:
    """Truncate text at sentence or clause boundary, not mid-word (Fix 4.2)."""
    if len(text) <= max_chars:
        return text
    # Try sentence boundary first (., !, ?)
    for punct in ['. ', '! ', '? ']:
        idx = text.rfind(punct, 0, max_chars)
        if idx > max_chars * 0.5:  # At least halfway
            return text[:idx + 1].strip()
    # Fall back to clause boundary
    for punct in [', ', '; ']:
        idx = text.rfind(punct, 0, max_chars)
        if idx > max_chars * 0.4:
            return text[:idx].strip()
    # Last resort: word boundary
    return text[:max_chars].rsplit(' ', 1)[0].strip()


def snake_case_to_prose(text: str) -> str:
    """Convert snake_case relation tokens to readable prose (Fix 5.2 + 5.5).
    
    Replaces patterns like 'merges_functionalities_of' with 'merges functionalities of'.
    Also handles common relation verbs leaking into summaries.
    """
    # Replace snake_case tokens (2+ words joined by underscores)
    result = re.sub(r'\b([a-z]+(?:_[a-z]+)+)\b', lambda m: m.group(0).replace('_', ' '), text)
    # Clean up any double spaces
    result = re.sub(r'\s+', ' ', result).strip()
    return result


# ── Graph-to-Text Post-Processor ──
# Cleans ALL artefacts that can appear in linearized KG output.
# Fully dynamic — no domain-specific keywords.

# Patterns that are raw relation strings leaking into prose
_RAW_RELATION_RE = re.compile(
    r'\b('
    r'is_a|is_part_of|part_of|type_of|has_type|related_to|associated_with|'
    r'is_associated_with|belongs_to|is_related_to|is_used_in|used_in|'
    r'is_defined_as|defined_as|is_an_instance_of|instance_of|'
    r'has_property|has_attribute|is_subtype_of|subtype_of|'
    r'is_connected_to|connected_to|refers_to|is_referred_to_as'
    r')\b',
    re.I,
)

_RAW_RELATION_MAP = {
    'is_a': 'is a type of',
    'is_part_of': 'is part of',
    'part_of': 'is part of',
    'type_of': 'is a type of',
    'has_type': 'has types including',
    'related_to': 'is related to',
    'associated_with': 'is associated with',
    'is_associated_with': 'is associated with',
    'belongs_to': 'belongs to',
    'is_related_to': 'is related to',
    'is_used_in': 'is used in',
    'used_in': 'is used in',
    'is_defined_as': 'is defined as',
    'defined_as': 'is defined as',
    'is_an_instance_of': 'is an instance of',
    'instance_of': 'is an instance of',
    'has_property': 'has the property of',
    'has_attribute': 'has the attribute of',
    'is_subtype_of': 'is a subtype of',
    'subtype_of': 'is a subtype of',
    'is_connected_to': 'is connected to',
    'connected_to': 'is connected to',
    'refers_to': 'refers to',
    'is_referred_to_as': 'is referred to as',
}

# Arrow and separator symbols that leak from KG into text
_ARROW_SYMBOLS_RE = re.compile(r'[\u2192\u2190\u2194\u2196\u2197\u2198\u2199→←↔↗↘↙↖]')

# Detect "A ? B" or "A → B" pattern as a raw KG triple in text
_KG_TRIPLE_IN_TEXT = re.compile(
    r'([A-Za-z][A-Za-z\s\-]{1,40}?)\s*[?→←]\s*([A-Za-z][A-Za-z\s\-]{1,40})',
)


def _naturalise_triple(m: re.Match) -> str:
    """Convert 'A ? B' or 'A → B' in prose to 'A relates to B.'"""
    left  = m.group(1).strip().rstrip(',;:')
    right = m.group(2).strip().rstrip(',;:')
    if not left or not right:
        return left or right or ""
    # Infer verb from overlap
    lw = set(re.findall(r'[a-z]{3,}', left.lower()))
    rw = set(re.findall(r'[a-z]{3,}', right.lower()))
    if rw and rw.issubset(lw):
        return f"{left} encompasses {right}"
    if lw and lw.issubset(rw):
        return f"{left} is a type of {right}"
    return f"{left} relates to {right}"


def clean_graph_text(text: str) -> str:
    """
    Post-process linearized KG text to remove ALL graph artefacts:
      1. Snake_case relation tokens  → human-readable
      2. Raw relation strings (is_part_of etc.) → prose
      3. Arrow / separator symbols  → removed or replaced
      4. Standalone ? symbols       → removed
      5. 'A ? B' / 'A → B' triples → naturalised sentence
      6. Double spaces, stray punct → cleaned
    Fully dynamic — no domain-specific words anywhere.
    """
    if not text:
        return text

    # 1. snake_case → prose
    text = snake_case_to_prose(text)

    # 2. Raw relation tokens → readable verbs
    def _replace_raw_rel(m):
        token = m.group(1).lower()
        return _RAW_RELATION_MAP.get(token, token.replace('_', ' '))
    text = _RAW_RELATION_RE.sub(_replace_raw_rel, text)

    # 3. Arrow symbols → space (they were KG edge indicators)
    text = _ARROW_SYMBOLS_RE.sub(' ', text)

    # 4. Standalone ? symbols (KG separator artefacts)
    #    First handle "A ? B" triples
    text = _KG_TRIPLE_IN_TEXT.sub(_naturalise_triple, text)
    #    Then remove any remaining standalone ?
    text = re.sub(r'\s\?\s', ' ', text)      # "word ? word" → "word  word"
    text = re.sub(r'(?<!\w)\?(?!\w)', '', text)  # isolated ?
    text = re.sub(r'\?', '', text)            # nuclear: remove every remaining ?

    # 5. Fix "is is" double-is artefact
    text = re.sub(r'\bis\s+is\b', 'is', text, flags=re.I)

    # 6. Clean up whitespace and punctuation
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    text = re.sub(r'\.{2,}', '.', text)
    text = text.strip()

    # 7. Capitalise first character
    if text and text[0].islower():
        text = text[0].upper() + text[1:]

    return text


def expand_node_context(label: str, desc: str, edges_out: List[Tuple[str, str]]) -> str:
    """
    Build a context-rich explanatory sentence for a node that has a bare/thin description.
    Dynamically constructs sentence from:
      - label (the concept name)
      - desc  (node description, may be empty)
      - edges_out: list of (relation, target_label) tuples from this node's outgoing edges

    Returns a single explanatory sentence. No hardcoded domain words.
    """
    label = label.strip()
    if not label:
        return ""

    desc = (desc or "").strip().rstrip(".")

    # Group edges by relation type
    by_rel: Dict[str, List[str]] = {}
    for rel, tgt in edges_out:
        rel_clean = rel.replace('_', ' ').strip().lower()
        by_rel.setdefault(rel_clean, []).append(tgt.strip())

    parts: List[str] = []

    # Start with description if available
    if desc and len(desc.split()) >= 4:
        parts.append(desc + ".")

    # Add edge-derived clauses
    for rel, targets in by_rel.items():
        if not targets:
            continue
        tgt_str = (
            ", ".join(targets[:-1]) + " and " + targets[-1]
            if len(targets) > 1 else targets[0]
        )
        # Map relation to prose
        rel_prose = _RAW_RELATION_MAP.get(rel.replace(' ', '_'), rel)
        clause = f"{label} {rel_prose} {tgt_str}."
        parts.append(clause)

    if not parts:
        return f"{label} is a key concept in this topic."

    # Limit to 2 clauses max for conciseness
    return " ".join(parts[:2])


# ── Curriculum-Aware Ordering (Fix 5.1) ──
# Priority tiers for pedagogical section ordering.
# Lower number = appears earlier in notes.
CONCEPT_PRIORITY_MAP = {
    # Priority 1 — Historical / Definitional
    'arpanet': 1, 'history': 1, 'origin': 1, 'introduction': 1,
    'definition': 1, 'overview': 1,
    # Priority 2 — Architecture / Model
    'tcp/ip': 2, 'tcp ip': 2, 'osi': 2, 'model': 2, 'protocol suite': 2,
    'layer': 2, 'architecture': 2, 'reference model': 2,
    # Priority 3 — Structural Groups
    'types': 3, 'components': 3, 'elements': 3, 'phases': 3, 'stages': 3,
    # Priority 4 — Properties and Features
    'properties': 4, 'features': 4, 'characteristics': 4, 'advantages': 4,
    # Priority 5 — Cross-cutting
    'error control': 5, 'flow control': 5, 'encapsulation': 5,
    'components and structure': 5, 'details': 5,
    # Priority 6 — Catch-all
    'additional': 6, 'other': 6, 'miscellaneous': 6, 'concepts': 6,
}


def _get_concept_priority(label: str) -> int:
    """Get the curriculum priority for a topic label (Fix 5.1).
    
    Uses fuzzy substring matching against CONCEPT_PRIORITY_MAP.
    Lower priority number = appears earlier.
    """
    label_lower = label.lower().strip()
    
    # Exact match first
    if label_lower in CONCEPT_PRIORITY_MAP:
        return CONCEPT_PRIORITY_MAP[label_lower]
    
    # Substring match — find the best (lowest priority) match
    best_priority = 99  # Default: put at the very end
    for keyword, priority in CONCEPT_PRIORITY_MAP.items():
        if keyword in label_lower or label_lower in keyword:
            best_priority = min(best_priority, priority)
    
    return best_priority

# Data structures to match StructuralKGSummarizer expectations
class Node:
    def __init__(self, id, label, description=""):
        self.id = id
        self.label = label
        self.description = description

class Edge:
    def __init__(self, source, target, relation):
        self.source = source
        self.target = target
        self.relation = relation

class KnowledgeGraph:
    def __init__(self, nodes: List[Node], edges: List[Edge]):
        self.nodes = nodes
        self.edges = edges

class BARTSummarizer:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        self.device = 0 if torch.cuda.is_available() else -1
        try:
            self.summarizer = pipeline("summarization", model=model_name, device=self.device)
        except Exception as e:
            print(f"Error loading BART model: {e}")
            self.summarizer = None

    def summarize(self, text, max_length=400, min_length=150):
        if not text:
            return ""
        if not self.summarizer:
            return "Summarizer not initialized."
        
        # Truncate if too long (BART max is 1024 tokens ~ 4000 chars)
        if len(text) > 4000:
            text = text[:4000]
        
        # Dynamic length: don't request longer output than input
        input_words = len(text.split())
        effective_max = min(max_length, max(input_words, 50))
        effective_min = min(min_length, effective_max - 10)
            
        try:
            summary = self.summarizer(
                text, 
                max_length=effective_max, 
                min_length=max(effective_min, 30), 
                do_sample=False,
                no_repeat_ngram_size=3,  # Prevent repetitive phrases
                length_penalty=1.2,       # Encourage richer output
            )
            return summary[0]['summary_text']
        except Exception as e:
            print(f"BART summarization failed: {e}")
            return text  # Fallback to original text

def _strip_parenthetical_if_redundant(label: str) -> str:
    """Remove parenthetical if it duplicates the acronym already in label."""
    match = re.match(r'^([A-Z][A-Za-z0-9/]*(?:\s+[A-Z][A-Za-z0-9/]*)*)\s*\((.+?)\)$',
                     label.strip())
    if match:
        acronym_part = match.group(1).strip()
        full_name    = match.group(2).strip()
        initials = ''.join(w[0].upper() for w in full_name.split() if w)
        if initials == acronym_part.replace(' ','').upper():
            return acronym_part
    return label

GROUPABLE_VERBS = {
    'contains_protocol', 'contains protocol', 'includes',
    'includes the', 'part_of', 'part of', 'contains',
}

def _group_protocol_objects(src, by_verb, G_clean):
    """Collapse multiple protocol-list edges into one compound sentence."""
    grouped_objs = []
    grouped_verb = None
    other_verbs = {}

    for verb, objs in by_verb.items():
        v_lower = verb.lower().strip().replace('_',' ')
        if v_lower in GROUPABLE_VERBS:
            # Strip redundant protocol suffix: 'DNS (Domain Name System)' → 'DNS'
            cleaned = []
            for o in objs:
                # Remove '(Full Name)' if short acronym already present
                o_clean = re.sub(r'\s*\([^)]*\)', '', o).strip()
                if len(o_clean) > 0:
                    cleaned.append(o_clean)
                else:
                    cleaned.append(o)
            grouped_objs.extend(cleaned)
            grouped_verb = verb
        else:
            other_verbs[verb] = objs

    return grouped_objs, grouped_verb, other_verbs


# ====================== STRUCTURAL KG SUMMARIZATION PIPELINE ======================
class StructuralKGSummarizer:
    """Structural pipeline for KG summarization: 
       Centrality → Clustering → Salient Subgraph → Path Selection → Linearization → BART"""
    
    def __init__(self, use_refiner: bool = True):
        # Reuse the cached BART model from ex.py if available
        # Reuse the cached BART model from ex.py if available to ensure shared VRAM
        try:
            from ex import _get_bart_model, generate_global_summary
            tokenizer, model = _get_bart_model()
            self.bart_model = model
            self.bart_tokenizer = tokenizer
            self.bart_func = generate_global_summary
            device_id = model.device.index if model.device.index is not None else 0
            self.bart_device = device_id if torch.cuda.is_available() else -1
            self.bart = True
            print("[StructuralKGSummarizer] Successfully linked with shared BART model")
        except Exception as e:
            print(f"[StructuralKGSummarizer] Failed to load BART: {e}")
            self.bart = False
        self.use_refiner = use_refiner and HAS_REFINER
        if self.use_refiner:
            self.refiner = FusedSummaryRefiner()
            print("[StructuralKGSummarizer] Initialized with summary refiner")
        else:
            self.refiner = None
    
    def to_nx(self, kg):
        """Convert KnowledgeGraph to NetworkX graph."""
        G = nx.DiGraph()
        for node in kg.nodes:
            G.add_node(node.id, label=node.label, desc=node.description if node.description else "")
        for edge in kg.edges:
            G.add_edge(edge.source, edge.target, label=edge.relation, weight=1.0)
        return G
    
    def centrality(self, G, transcript_text: str = ""):
        """Compute centrality scores (PageRank + Degree + optional TF-IDF from transcript).
        
        Change 3 / P12: When transcript_text is provided, node labels are scored by
        TF-IDF relevance. Formula: 0.35*pagerank + 0.35*tfidf + 0.15*degree + 0.15*def_bonus
        """
        if len(G.nodes()) < 2:
            return {node: 1.0 for node in G.nodes()}
        
        try:
            pr = nx.pagerank(G)
            deg = nx.degree_centrality(G)
            
            # TF-IDF scoring of node labels against transcript (Change 3)
            tfidf_scores = {}
            if transcript_text and len(transcript_text) > 50:
                transcript_lower = transcript_text.lower()
                transcript_words = transcript_lower.split()
                total_words = len(transcript_words) if transcript_words else 1
                word_freq = Counter(transcript_words)
                
                for node in G.nodes():
                    label = G.nodes[node].get('label', '').lower().strip()
                    if not label:
                        tfidf_scores[node] = 0.0
                        continue
                    label_tokens = label.split()
                    tf = sum(word_freq.get(t, 0) for t in label_tokens) / total_words
                    # Simple IDF approximation: rarer terms in transcript get higher weight
                    idf = sum(1.0 / (1.0 + word_freq.get(t, 0)) for t in label_tokens) / max(len(label_tokens), 1)
                    tfidf_scores[node] = tf * idf * 100  # Scale up
                
                # Normalize TF-IDF to [0, 1]
                max_tfidf = max(tfidf_scores.values()) if tfidf_scores else 1.0
                if max_tfidf > 0:
                    tfidf_scores = {n: v / max_tfidf for n, v in tfidf_scores.items()}
            else:
                tfidf_scores = {node: 0.0 for node in G.nodes()}
            
            # Definitional node bonus: nodes with is_a / defines edges
            def_bonus = {node: 0.0 for node in G.nodes()}
            for u, v, data in G.edges(data=True):
                rel = data.get('label', '').lower().strip()
                if rel in ('is_a', 'is a', 'defines', 'defined_by'):
                    def_bonus[u] = 1.0
                    def_bonus[v] = max(def_bonus[v], 0.5)
            
            # Combined scoring
            scores = {}
            for node in G.nodes():
                scores[node] = (
                    0.35 * pr.get(node, 0) +
                    0.35 * tfidf_scores.get(node, 0) +
                    0.15 * deg.get(node, 0) +
                    0.15 * def_bonus.get(node, 0)
                )
            return scores
        except Exception as e:
            print(f"Warning in centrality computation: {e}")
            # Fallback: equal centrality
            return {node: 1.0 for node in G.nodes()}
    
    def cluster(self, G):
        """Cluster graph using connected components."""
        if len(G.nodes()) == 0:
            return []
        
        undirected = G.to_undirected()
        if nx.is_connected(undirected):
            # If the graph is connected, use communities
            try:
                import community as community_louvain # type: ignore
                partition = community_louvain.best_partition(undirected)
                clusters = {}
                for node, cluster_id in partition.items():
                    if cluster_id not in clusters:
                        clusters[cluster_id] = []
                    clusters[cluster_id].append(node)
                return list(clusters.values())
            except:
                # Fallback to connected components
                return [list(undirected.nodes())]
        else:
            # Use connected components
            return [list(component) for component in nx.connected_components(undirected)]
    
    def salient_subgraph(self, G, scores, top_k=10):
        """Extract salient subgraph from top-k central nodes with better connectivity."""
        if len(G.nodes()) == 0:
            return G
        
        # Sort nodes by centrality in descending order
        nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate total salience mass
        total_salience = sum(score for _, score in nodes)
        
        # Keep nodes until cumulative salience ≥ 0.85 (85% of total)
        cumulative_salience = 0.0
        selected_nodes = []
        salience_threshold = 0.85  # Keep 85% of salience mass
        
        for node_id, score in nodes:
            cumulative_salience += score
            if cumulative_salience / total_salience >= salience_threshold:
                break
            selected_nodes.append(node_id)
            
        
        # Ensure we have at least 3 nodes
        if len(selected_nodes) < 3 and len(nodes) >= 3:
            selected_nodes = [node[0] for node in nodes[:3]]
        
        # Also enforce maximum nodes for efficiency
        if len(selected_nodes) > top_k:
            selected_nodes = selected_nodes[:top_k]
        
        print(f"[Salience Mass] Selected {len(selected_nodes)} nodes with {cumulative_salience:.2%} salience mass")
        
        top_nodes = selected_nodes
            
        # Extract subgraph
        subgraph = G.subgraph(top_nodes).copy()
        
        # Ensure connectivity by adding important connecting nodes
        undirected = subgraph.to_undirected()
        
        if not nx.is_connected(undirected) and len(subgraph.nodes()) > 1:
            # Find bridging nodes that connect disconnected components
            components = list(nx.connected_components(undirected))
            bridges = set()
            
            # Look for shortest paths in original graph
            for comp1 in components:
                for comp2 in components:
                    if comp1 == comp2:
                        continue
                    
                    # Find shortest path between components
                    min_path_length = float('inf')
                    best_bridge = None
                    
                    for node1 in comp1:
                        for node2 in comp2:
                            try:
                                if nx.has_path(G, node1, node2):
                                    path = nx.shortest_path(G, node1, node2)
                                    if 1 < len(path) < min_path_length:
                                        min_path_length = len(path)
                                        best_bridge = path[1] if len(path) > 2 else None
                            except:
                                pass
                    
                    if best_bridge and best_bridge not in subgraph.nodes():
                        bridges.add(best_bridge)
            
            # Add bridging nodes to subgraph
            for bridge in bridges:
                subgraph.add_node(bridge, **G.nodes[bridge])
                # Add edges to/from bridge
                for neighbor in G.neighbors(bridge):
                    if neighbor in subgraph.nodes():
                        if G.has_edge(bridge, neighbor):
                            subgraph.add_edge(bridge, neighbor, **G.edges[bridge, neighbor])
                        elif G.has_edge(neighbor, bridge):
                            subgraph.add_edge(neighbor, bridge, **G.edges[neighbor, bridge])
        
        return subgraph
    
    def select_paths(self, G, max_paths=5):
        """Select important paths in the graph."""
        if len(G.nodes()) < 2:
            return []
        
        paths = []
        nodes = list(G.nodes())
        
        # Select paths between central nodes
        centrality = self.centrality(G)
        central_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        central_nodes = [node[0] for node in central_nodes[:min(5, len(central_nodes))]]
        
        # Find paths between central nodes
        for i in range(len(central_nodes)):
            for j in range(i + 1, len(central_nodes)):
                try:
                    if nx.has_path(G, central_nodes[i], central_nodes[j]):
                        path = nx.shortest_path(G, central_nodes[i], central_nodes[j])
                        if len(path) > 1:
                            paths.append(path)
                except:
                    pass
        
        # If not enough paths, add some random walks
        if len(paths) < max_paths:
            for _ in range(max_paths - len(paths)):
                if nodes:
                    start = np.random.choice(nodes)
                    walk = [start]
                    for _ in range(4):  # Walk length of 5
                        neighbors = list(G.neighbors(walk[-1]))
                        if neighbors:
                            next_node = np.random.choice(neighbors)
                            walk.append(next_node)
                        else:
                            break
                    if len(walk) > 1:
                        paths.append(walk)
        
        return paths[:max_paths]

    def walk_cross_layer_paths(self, G_clean, scores, max_paths: int = 8) -> List[str]:
        """
        Walk 2-hop paths between high-centrality nodes to generate
        contextual cross-layer relationship sentences (Fix 4.3).
        """
        sentences = []
        top_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:15]
        top_ids = {n for n, _ in top_nodes}
        
        for src, _ in top_nodes:
            for mid, rel1_data in G_clean[src].items():
                if mid not in top_ids: continue
                # Handle both DiGraph and MultiDiGraph gracefully
                r1_list = rel1_data.values() if 'label' not in rel1_data else [rel1_data]
                for r1 in r1_list:
                    rel1 = r1.get('label', 'relates to').replace('_', ' ')
                    for dst, rel2_data in G_clean[mid].items():
                        if dst == src or dst not in top_ids: continue
                        r2_list = rel2_data.values() if 'label' not in rel2_data else [rel2_data]
                        for r2 in r2_list:
                            src_lbl = G_clean.nodes[src].get('label', src)
                            mid_lbl = G_clean.nodes[mid].get('label', mid)
                            dst_lbl = G_clean.nodes[dst].get('label', dst)
                            rel2 = r2.get('label', 'relates to').replace('_', ' ')
                            sent = f"{src_lbl} {rel1} {mid_lbl}, which {rel2} {dst_lbl}."
                            sentences.append(sent)
        
        # Deduplicate and limit
        seen = set()
        unique_sents = []
        for s in sentences:
            if s not in seen:
                seen.add(s)
                unique_sents.append(s)
        return unique_sents[:max_paths]
    def linearize(self, G, paths=None):
        """
        Convert graph to Natural, Logically Floiwng Text (Advanced Strategy).
        
        1. Pre-process Graph:
           - Remove redundant "is associated with" edges if a specific verb exists between same nodes.
           - Remove self-loops and empty labels.
        2. Cluster by Subject:
           - Group all relations for a subject.
           - Sort subjects by Centrality (Global Importance).
        3. Generate Compound Sentences:
           - "The View Level provides multiple views and connects to the Logical Level."
           - Uses specific verbs from KG (priority) over generic ones.
        4. Fix Hallucinations:
           - Regex-based cleanup of common artifacts ("At Conceptually Level", "Level Levels").
        """
        if len(G.nodes()) == 0:
            return "Empty knowledge graph."

        # ── 1. Graph Cleaning & Deduplication ──
        # Create a working copy to modify edges without affecting original G
        # (Though G is usually a subgraph here, but better safe)
        if hasattr(G, 'copy'):
            G_clean = G.copy()
        else:
            G_clean = G

        # Helper to clean text
        def clean_text(t):
            if not t: return ""
            # Hallucination Fixes
            t = re.sub(r'(?i)at\s+conceptually\s+level', 'at the Conceptual Level', t)
            t = re.sub(r'(?i)level\s+levels', 'Level', t)
            t = re.sub(r'(?i)user-\s+system', 'User-system', t)
            t = re.sub(r'(?i)and\s+and', 'and', t)
            t = re.sub(r'(?i)^\s*(and|or)\s+', '', t).strip()
            return t

        # Deduplicate edges: If (A, specific, B) exists, drop (A, associated, B)
        # We need to iterate over all node pairs that have edges
        # G_clean.edges is (u, v, key) in MultiDiGraph or (u, v) in DiGraph
        # We assume DiGraph behavior for simplicity, or iterate edges carefully.
        
        # Build adjacency map: source -> target -> list of relations
        adj_map = defaultdict(lambda: defaultdict(list))
        for u, v, data in G_clean.edges(data=True):
            rel = clean_text(data.get('label', 'related to'))
            adj_map[u][v].append(rel)

        # ── 2. Subject Clustering & Ordering ──
        # Comput Centrality for logical ordering
        try:
            centrality = nx.degree_centrality(G_clean)
        except:
            centrality = {n: 1 for n in G_clean.nodes()}
            
        sorted_subjects = sorted(centrality.keys(), key=lambda n: centrality[n], reverse=True)
        
        sentences = []
        processed_pairs = set()

        for src in sorted_subjects:
            targets_map = adj_map.get(src)
            if not targets_map:
                continue
            
            src_label = clean_text(G_clean.nodes[src].get('label', src))
            if not src_label: continue

            # For this subject, collect best relations
            valid_relations = []
            tgt_labels_seen = []
            
            for tgt, rels in targets_map.items():
                if tgt == src: continue # No self-loops
                
                tgt_label = clean_text(G_clean.nodes[tgt].get('label', tgt))
                tgt_label = _strip_parenthetical_if_redundant(tgt_label)
                if not tgt_label or tgt_label.lower() == src_label.lower(): continue
                
                # Semantic Deduplication of targets (e.g., ADT vs Abstract Data Type)
                tgt_words = set(tgt_label.lower().split())
                is_duplicate = False
                for seen_label in tgt_labels_seen:
                    seen_words = set(seen_label.lower().split())
                    if not tgt_words or not seen_words: continue
                    overlap = len(tgt_words.intersection(seen_words)) / max(len(tgt_words), len(seen_words))
                    if overlap > 0.8:  # 80% word overlap means it's likely a semantic duplicate
                        is_duplicate = True
                        break
                if is_duplicate:
                    continue
                tgt_labels_seen.append(tgt_label)
                
                # Pick BEST relation
                # Priority: Specific Verb > "is a" > "connects" > "related/associated"
                best_rel = "is related to"
                specific_found = False
                
                # Clean rels
                clean_rels = []
                for r in rels:
                    r_clean = r.lower().strip()
                    # Normalize strict "associated" forms to lowest priority
                    if r_clean in ('associated', 'associated with', 'is associated', 'is associated with'):
                         r_clean = 'is related to'
                    # Strip generic prefixes like "is", "are"
                    if r_clean.startswith("is ") or r_clean.startswith("are "):
                         r_clean = " ".join(r_clean.split()[1:])
                    clean_rels.append((r, r_clean)) # Keep original case for output, use lower for check
                
                # Check for specific verbs
                for raw, lower in clean_rels:
                    if lower not in ('is related to', 'related to', 'associated'):
                        best_rel = raw # Use specific verb!
                        specific_found = True
                        break
                
                if not specific_found:
                     # If only generic exists, use "is related to"
                     best_rel = "is related to"

                valid_relations.append((best_rel, tgt_label))

            if not valid_relations:
                continue

            # ── 3. Concept Expansion NLG (Fix 1) ──
            by_verb = defaultdict(list)
            for r, t in valid_relations:
                by_verb[r].append(t)
            
            grouped_objs, grouped_verb, other_verbs = _group_protocol_objects(src, by_verb, G_clean)
            
            src_desc = clean_text(G_clean.nodes[src].get('desc', '') or
                                  G_clean.nodes[src].get('description', ''))
            
            if grouped_objs and len(grouped_objs) > 1:
                proto_str = ', '.join(grouped_objs[:-1]) + ' and ' + grouped_objs[-1]
                ctx = src_desc if src_desc else (
                    f'{src_label} employs multiple protocols for network communication.'
                )
                compound = (
                    f'{src_label} utilises several protocols including {proto_str}. {ctx}'
                ).strip()
                sentences.append(compound)
                by_verb = other_verbs
            elif grouped_objs and len(grouped_objs) == 1:
                src_desc_short = src_desc[:80] if src_desc else ''
                sentences.append(
                    f'{grouped_objs[0]} is a protocol associated with {src_label}. {src_desc_short}'.strip()
                )
                by_verb = other_verbs
            
            # Fix 2: Concept-level Edge Deduplication
            # Merge verbs that are semantically similar
            VERB_GROUPS = {
                'includes': ['includes', 'contains', 'contains_protocol', 'has',
                             'has_layer', 'comprises', 'consists of'],
                'is a': ['is a', 'is a type of', 'type of', 'defined as'],
                'uses': ['uses', 'uses_protocol', 'operates using', 'employs'],
                'provides': ['provides', 'ensures', 'offers', 'supports'],
            }
            
            # Merge verb groups
            merged_by_verb = defaultdict(list)
            for verb, objects in by_verb.items():
                verb_lower = verb.lower().strip()
                merged_key = verb  # default: keep original
                for canonical, variants in VERB_GROUPS.items():
                    if verb_lower in variants:
                        merged_key = canonical
                        break
                merged_by_verb[merged_key].extend(objects)
            
            # Deduplicate objects within each verb group
            for verb in merged_by_verb:
                seen = []
                unique = []
                for obj in merged_by_verb[verb]:
                    obj_lower = obj.lower().strip()
                    if obj_lower not in seen:
                        seen.append(obj_lower)
                        unique.append(obj)
                merged_by_verb[verb] = unique
            
            # ── Concept Expansion: Generate explanatory sentences ──
            expanded_sentences = []
            node_desc = G_clean.nodes[src].get('desc', '').strip()
            desc_snippet = _truncate_at_boundary(node_desc) if node_desc and len(node_desc) > 20 else ''
            
            for verb, objects in merged_by_verb.items():
                verb_key = verb.lower().strip()
                
                # Fix 4.1B: Protocol label deduplication
                cleaned_objects = []
                for obj in objects:
                    if 'protocol' in verb_key:
                        obj = re.sub(r'(?i)\bProtocol\b$', '', obj).strip()
                    if obj:
                        cleaned_objects.append(obj)
                
                if not cleaned_objects:
                    continue
                
                # Format object list
                if len(cleaned_objects) == 1:
                    obj_str = cleaned_objects[0]
                else:
                    obj_str = ", ".join(cleaned_objects[:-1]) + " and " + cleaned_objects[-1]
                
                # ── Concept Expansion NLG ──
                # For multi-object groups, generate EXPLANATORY sentences,
                # not graph statements
                if verb_key == 'includes' and len(cleaned_objects) >= 2:
                    # "X includes A, B, and C" → explanatory form
                    sentence = (
                        f"{src_label} encompasses several key components, "
                        f"including {obj_str}. "
                        f"Each serves a distinct role within {src_label.lower()}."
                    )
                    if desc_snippet:
                        sentence = f"{desc_snippet} {sentence}"
                    expanded_sentences.append(sentence)
                    
                elif verb_key == 'uses' and len(cleaned_objects) >= 2:
                    sentence = (
                        f"{src_label} relies on protocols such as {obj_str} "
                        f"for communication and data exchange."
                    )
                    if desc_snippet:
                        sentence = f"{desc_snippet} {sentence}"
                    expanded_sentences.append(sentence)
                    
                elif verb_key == 'provides' and len(cleaned_objects) >= 2:
                    sentence = (
                        f"{src_label} provides essential services including {obj_str}, "
                        f"which are fundamental to its operation."
                    )
                    expanded_sentences.append(sentence)
                    
                elif verb_key == 'is a' and len(cleaned_objects) == 1:
                    sentence = f"{src_label} is a type of {obj_str}."
                    if desc_snippet:
                        sentence += f" {desc_snippet}"
                    expanded_sentences.append(sentence)
                    
                else:
                    # Fall back to template expansion for other verbs
                    template = EXPANSION_TEMPLATES.get(verb_key)
                    if template and desc_snippet:
                        try:
                            sentence = template.format(S=src_label, O=obj_str, D=desc_snippet)
                            expanded_sentences.append(sentence)
                        except (KeyError, IndexError):
                            expanded_sentences.append(f"{src_label} {verb} {obj_str}.")
                    elif desc_snippet:
                        expanded_sentences.append(f"{src_label} {verb} {obj_str}. {desc_snippet}")
                    else:
                        expanded_sentences.append(f"{src_label} {verb} {obj_str}.")
            
            if not expanded_sentences:
                continue
            
            # Assemble full sentence from expanded parts
            full_sent = " ".join(expanded_sentences)
            
            # Fix 4.1C: Orphaned Subject Recovery
            ORPHAN_PATTERNS = [r'^[Aa]\s+layer\b', r'^[Ll]ayer\s+in\b', r'^[Ww]hich\b', r'^[Aa]nd\b', r'^[Ii]s\b', r'^[Aa]re\b']
            if any(re.match(p, full_sent) for p in ORPHAN_PATTERNS):
                full_sent = f"{src_label}: {full_sent}"
            
            # Final cleanup
            full_sent = re.sub(r'\s+', ' ', full_sent).strip()
            if full_sent and full_sent[0].islower():
                full_sent = full_sent[0].upper() + full_sent[1:]
                
            sentences.append(full_sent)

        # Limit to top N sentences to avoid overwhelming context
        if len(sentences) > 30:
            sentences = sentences[:30]
        
        raw_text = " ".join(sentences)
        # Post-process: remove ALL graph artefacts (?, →, snake_case, raw relations)
        return clean_graph_text(raw_text)


    def enrich_empty_descriptions(self, G, transcript_text: str = ""):
        """
        For nodes with empty/thin descriptions, enrich from:
        1. Transcript text (2-sentence window around first mention)
        2. KG edge context (relations + neighbour labels → explanatory sentence)

        Ensures {D} in EXPANSION_TEMPLATES always has meaningful content.
        Fully dynamic — no domain-specific keywords.
        """
        # Split transcript into sentences if available
        transcript_sentences = []
        if transcript_text and len(transcript_text) > 50:
            try:
                import nltk
                transcript_sentences = nltk.sent_tokenize(transcript_text)
            except Exception:
                transcript_sentences = [
                    s.strip() for s in re.split(r'(?<=[.!?])\s+', transcript_text) if s.strip()
                ]

        enriched_count = 0
        for node_id in G.nodes():
            node_data = G.nodes[node_id]
            desc = node_data.get('desc', '').strip()
            label = node_data.get('label', '').strip()

            if not label or len(label) < 2:
                continue

            # Already has a real description → skip
            if desc and len(desc.split()) >= 6:
                continue

            enriched = False

            # Strategy 1: transcript window
            if transcript_sentences:
                label_lower = label.lower()
                for i, sent in enumerate(transcript_sentences):
                    if label_lower in sent.lower():
                        window = sent
                        if i + 1 < len(transcript_sentences):
                            window += ' ' + transcript_sentences[i + 1]
                        window = re.sub(r'\s+', ' ', window).strip()
                        if len(window) > 200:
                            window = _truncate_at_boundary(window, 200)
                        G.nodes[node_id]['desc'] = window
                        enriched = True
                        enriched_count += 1
                        break

            # Strategy 2: build from outgoing edges if transcript failed
            if not enriched:
                edges_out = []
                for tgt_id in G.successors(node_id):
                    tgt_label = G.nodes[tgt_id].get('label', '').strip()
                    rel = G.edges[node_id, tgt_id].get('label', 'relates to')
                    if tgt_label and rel:
                        edges_out.append((rel, tgt_label))

                if edges_out:
                    ctx = expand_node_context(label, desc, edges_out)
                    if ctx and len(ctx.split()) >= 5:
                        G.nodes[node_id]['desc'] = ctx
                        enriched_count += 1

        if enriched_count:
            print(f"[ContextExpansion] Enriched descriptions for {enriched_count} nodes")

    def inject_sparse_cluster_anchors(self, G, clusters, structural_text: str, scores: dict) -> str:
        """Fix 4.2: For clusters with <4 triples and no definition node,
        prepend a parent-context sentence derived from the cluster's parent node.
        """
        anchor_sentences = []
        
        for cluster_nodes in clusters:
            if len(cluster_nodes) >= 4:
                continue
            
            # Check if any node has a definition-type edge
            has_definition = False
            parent_label = None
            
            for n in cluster_nodes:
                if n not in G: continue
                for tgt, edata in G[n].items():
                    r_list = edata.values() if 'label' not in edata else [edata]
                    for r in r_list:
                        rel = r.get('label', '').lower().replace('_', ' ')
                        if rel in ('is a', 'type of', 'part of', 'belongs to'):
                            parent_label = G.nodes[tgt].get('label', '')
                        if rel in ('defines', 'defined as', 'is a', 'definition'):
                            has_definition = True
            
            if has_definition:
                continue
            
            # Build anchor from highest-centrality node in the cluster
            best_node = max(cluster_nodes, key=lambda n: scores.get(n, 0))
            best_label = G.nodes[best_node].get('label', '')
            best_desc = G.nodes[best_node].get('desc', '')
            
            if parent_label:
                anchor = f"{best_label} is a concept within {parent_label} that serves an important functional role."
            elif best_desc:
                anchor = f"{best_label}: {_truncate_at_boundary(best_desc, 120)}"
            else:
                # Generate from connected nodes
                neighbors = [G.nodes[nb].get('label', '') for nb in G.neighbors(best_node) if G.nodes[nb].get('label', '')]
                if neighbors:
                    anchor = f"{best_label} is associated with {', '.join(neighbors[:3])}."
                else:
                    continue
            
            anchor_sentences.append(anchor)
        
        if anchor_sentences:
            print(f"[Fix 4.2] Injected {len(anchor_sentences)} context-anchor sentences for sparse clusters")
            structural_text = ' '.join(anchor_sentences) + ' ' + structural_text
        
        return structural_text

    def summarize(self, kg, refine: bool = True, transcript_text: str = ""):
        """Full pipeline with atomic generation, refinement, BART summarization,
        and post-processing for 7-10 context-rich, noise-free sentences.
        
        Args:
            kg: KnowledgeGraph object with nodes and edges
            refine: Whether to apply structure-aware sentence reconstruction
            transcript_text: Original transcript text for TF-IDF centrality weighting
            
        Returns:
            Tuple of (summary_text, topic_labels) where topic_labels is a dict
            mapping cluster_id -> representative node label for section headings.
        """
        # Convert to NetworkX
        G = self.to_nx(kg)
        
        # Step 1: Centrality (Change 3 — with TF-IDF)
        scores = self.centrality(G, transcript_text=transcript_text)
        
        # Step 2: Salient Subgraph
        # FIX-CONTENT: Increase top_k dynamically based on graph size to avoid
        # missing KG content. Use min(N*0.7, 60) so small graphs aren't over-expanded
        # and large graphs still capture the majority of meaningful nodes.
        n_total = G.number_of_nodes()
        dynamic_top_k = max(25, min(60, int(n_total * 0.70)))
        salient = self.salient_subgraph(G, scores, top_k=dynamic_top_k)
        print(f"[SalientSubgraph] top_k={dynamic_top_k} (graph has {n_total} nodes)")

        # Step 2b: Extract topic_labels (Change 4 + FIX-HEADINGS)
        # FIX-HEADINGS: Instead of using the raw node label of the most central node
        # (which is often a fragment like "A Linear Data Structure In Which" or a
        # single generic word like "Explicit"), derive a conceptual heading by:
        #   1. Collecting ALL node labels in the cluster
        #   2. Identifying the most informative label via:
        #      (a) Preferring labels that are noun-phrases (not clause fragments)
        #      (b) Preferring labels whose words appear most frequently across the cluster
        #      (c) Falling back to the highest-centrality node label only if no better
        #          label can be found
        # This is fully dynamic — no hardcoded domain words.

        # Helpers for conceptual label selection
        _FRAGMENT_ARTICLE_RE = re.compile(
            r'^(a|an|the|in|of|at|by|for|with|on|to|into|from|about|'
            r'which|that|where|when|how|what|as|its|their)\s+',
            re.IGNORECASE,
        )
        _CLAUSE_VERB_RE = re.compile(
            r'\b(in which|where|that|which|are|is|was|were|can|may|will|shall)\b',
            re.IGNORECASE,
        )

        def _label_is_fragment(lbl: str) -> bool:
            """Return True if the label looks like a sentence fragment, not a concept name."""
            if _FRAGMENT_ARTICLE_RE.match(lbl.strip()):
                return True
            if _CLAUSE_VERB_RE.search(lbl.strip()):
                return True
            if len(lbl.split()) > 7:  # Very long labels are usually clause fragments
                return True
            return False

        def _concept_score(lbl: str, cluster_word_freq: Counter) -> float:
            """
            Score a label by how well it represents the cluster as a concept name.
            Higher is better.
            Penalises fragment-style labels heavily.
            Rewards labels whose words appear frequently in the cluster.
            """
            if not lbl or not lbl.strip():
                return -1.0
            if _label_is_fragment(lbl):
                return 0.0
            words = [w.lower() for w in re.findall(r'[a-zA-Z]{3,}', lbl)]
            if not words:
                return 0.0
            freq_score = sum(cluster_word_freq.get(w, 0) for w in words) / len(words)
            # Prefer shorter, denser concept names (2–5 words)
            length_bonus = 1.0 if 2 <= len(lbl.split()) <= 5 else 0.5
            return freq_score * length_bonus + length_bonus

        topic_labels = {}
        clusters = self.cluster(salient)
        for idx, cluster_nodes in enumerate(clusters):
            if not cluster_nodes:
                continue

            # Collect all node labels in this cluster
            all_labels = []
            word_freq: Counter = Counter()
            for n in cluster_nodes:
                lbl = salient.nodes[n].get('label', '').strip()
                if lbl:
                    all_labels.append((n, lbl))
                    for w in re.findall(r'[a-zA-Z]{3,}', lbl.lower()):
                        word_freq[w] += 1

            if not all_labels:
                topic_labels[idx] = "Key Concepts"
                continue

            # Score each label and pick the best conceptual one
            scored = sorted(
                all_labels,
                key=lambda nl: (
                    _concept_score(nl[1], word_freq),
                    scores.get(nl[0], 0),
                ),
                reverse=True,
            )
            best_label = scored[0][1]

            # If the best label is still a fragment, fall back to the most central node
            if _label_is_fragment(best_label):
                best_node = max(cluster_nodes, key=lambda n: scores.get(n, 0))
                best_label = salient.nodes[best_node].get('label', str(best_node))

            topic_labels[idx] = best_label.strip().title()

        # Fix 4.5: Dynamic Context Header Injection based on shared parent graph types
        for idx, cluster_nodes in enumerate(clusters):
            if not cluster_nodes or idx not in topic_labels: continue
            
            parent_types = []
            for n in cluster_nodes:
                if n not in G: continue
                for target, edge_data in G[n].items():
                    r_list = edge_data.values() if 'label' not in edge_data else [edge_data]
                    for r in r_list:
                        lbl = r.get('label', '').lower()
                        if lbl in ('is a', 'is_a', 'type of', 'type_of'):
                            tgt_label = G.nodes[target].get('label', '')
                            if tgt_label: parent_types.append(tgt_label.title())
                            
            if parent_types:
                most_common, count = Counter(parent_types).most_common(1)[0]
                if count >= len(cluster_nodes) * 0.5 and count > 1:
                    # Only override if the parent type is itself a good concept name
                    if not _label_is_fragment(most_common):
                        topic_labels[idx] = f"{most_common} (Group)"
        
        # Fix 5.1: Curriculum-aware post-sort of topic_labels
        # Sort by pedagogical priority so foundational concepts appear first
        sorted_items = sorted(topic_labels.items(), key=lambda kv: _get_concept_priority(kv[1]))
        topic_labels = {i: label for i, (_, label) in enumerate(sorted_items)}
        
        print(f"[Topic Labels] Extracted {len(topic_labels)} section headings (curriculum-sorted): {list(topic_labels.values())}")
        
        # Fix 4.1: Enrich empty node descriptions from transcript
        self.enrich_empty_descriptions(salient, transcript_text)
        
        # Step 3: Linearization (Compound Clauses)
        structural_text = self.linearize(salient)
        
        # Fix 4.2: Inject context-anchor sentences for sparse clusters
        structural_text = self.inject_sparse_cluster_anchors(salient, clusters, structural_text, scores)
        
        # Fix 4.3: Build cross-layer paths and append to structural_text
        cross_layer_sents = self.walk_cross_layer_paths(salient, scores)
        if cross_layer_sents:
            structural_text += ' ' + ' '.join(cross_layer_sents)
            
        print(f"[Structural Text] {structural_text[:200]}...")
        
        # Step 4: Structure-aware sentence reconstruction
        if refine and self.use_refiner and self.refiner:
            print("[Refiner] Applying structure-aware sentence reconstruction...")
            node_labels = [salient.nodes[n].get('label', n) for n in salient.nodes()]
            edge_dicts = [
                {'source': u, 'target': v, 'relation': d.get('label', 'related_to')}
                for u, v, d in salient.edges(data=True)
            ]
            refined_text = self.refiner.refine(structural_text, node_labels, edge_dicts)
            
            if hasattr(self.refiner, 'humanize'):
                refined_text = self.refiner.humanize(refined_text)
                print(f"[Humanized] Applied academic prose conversion")
            
            print(f"[Refined Text] {refined_text[:200]}...")
        else:
            refined_text = structural_text
        
        # Step 4b: Pre-process — convert snake_case relation tokens to prose (Fix 5.5)
        # Also apply full graph-artefact cleaner (removes ?, →, raw relation tokens)
        refined_text = snake_case_to_prose(refined_text)
        refined_text = clean_graph_text(refined_text)
        
        # Fix 6: Context Anchor Injection for Sparse Clusters
        CONTEXT_ANCHORS = {
            'routing protocol': ('The Internet Layer uses routing protocols to determine the optimal',
                                 'forwarding path for packets. Each protocol applies a different algorithm.'),
            'application layer': ('The Application Layer is the topmost layer of the TCP/IP model.',
                                  'It merges the OSI Session, Presentation and Application layers.'),
            'transport layer': ('The Transport Layer provides end-to-end communication services.',
                                'It handles segmentation, flow control, and error recovery.'),
            'internet layer': ('The Internet Layer is responsible for logical addressing and routing.',
                               'It is functionally equivalent to the Network Layer in the OSI model.'),
            'network access': ('The Network Access Layer is the lowest layer of the TCP/IP model.',
                               'It combines the OSI Data Link and Physical layers.'),
        }

        def inject_context_anchor(text: str, topic_label: str) -> str:
            """Prepend a context sentence if the cluster appears sparse/decontextualised."""
            topic_lower = topic_label.lower()
            for key, (sent1, sent2) in CONTEXT_ANCHORS.items():
                if key in topic_lower:
                    if sent1[:20].lower() not in text.lower():  # avoid duplicating
                        return f"{sent1} {sent2} " + text
            return text

        # Call before BART:
        refined_text = inject_context_anchor(refined_text, topic_labels.get(0, ''))

        # Step 5: Gemini call — generates the academic prose summary.
        summary = self._generate_gemini_summary(refined_text, topic_labels)

        # Fallback to BART if Gemini returned the raw text unchanged
        if summary == refined_text and getattr(self, 'bart', False):
            print("[KG Summary] Gemini unavailable, falling back to BART...")
            core_topic = topic_labels.get(0, "the core subject") if topic_labels else "the core subject"
            triple_count = len([s for s in refined_text.split('.') if len(s.strip()) > 20])

            if triple_count < 6:
                prompt_wrapper = (
                    f"You are a professor writing one concise educational paragraph. "
                    f"Explain {core_topic}: its definition, its role, "
                    f"and which components it uses. "
                    f"Do not repeat any sentence. Write exactly one paragraph. Begin:\n"
                )
                max_len, min_len = 200, 80
            else:
                prompt_wrapper = (
                    f"You are a university professor writing lecture summary notes. "
                    f"Synthesize the following facts into exactly 3 cohesive academic paragraphs. "
                    f"Do not enumerate items. Write continuous academic prose. Begin:\n"
                )
                max_len, min_len = 450, 200

            final_input = prompt_wrapper + refined_text
            try:
                summary = self.bart_func(
                    text=final_input,
                    model=self.bart_model,
                    tokenizer=self.bart_tokenizer,
                    device=self.bart_device,
                    max_len=max_len,
                    min_len=min_len
                )
                print("[KG Summary] Generated via BART fallback.")
            except Exception as e:
                print(f"[KG Summary] BART fallback also failed: {e}")
                summary = refined_text

        # Step 6: Post-processing  (moved here; was before old Gemini call)
        from summary_postprocessor import clean_and_constrain_summary
        summary = clean_and_constrain_summary(
            summary,
            min_sentences=10,
            max_sentences=12,
        )
        print(f"[KG Summary] Post-processed to {len(summary.split('.'))} sentences")

        return summary, topic_labels

    # ──────────────────────────────────────────────────────────────────────────
    # GEMINI CALL: summary only
    # ──────────────────────────────────────────────────────────────────────────

    def _generate_gemini_summary(
        self, linearized_text: str, topic_labels: dict
    ) -> str:
        """
        Gemini call that generates an academic prose summary (≤7 sentences).

        Returns:
            summary_text: str
            On any failure, returns linearized_text so the BART fallback can proceed.
        """
        import google.generativeai as genai

        topics_str = ", ".join(topic_labels.values()) if topic_labels else "the lecture content"

        prompt = (
            "You are a university professor writing concise educational lecture notes.\n"
            "Write a highly professional academic prose summary (≤7 sentences, no bullets) "
            "for the following lecture content.\n"
            "Rules:\n"
            "- ONLY continuous prose paragraphs. No bullets, lists, or internal newlines.\n"
            "- Academic yet accessible tone.\n"
            "- Cover ALL key topics and their relationships without exception.\n"
            "- Do NOT repeat sentences.\n\n"
            f"Key topics covered: {topics_str}\n\n"
            "Lecture content:\n" + linearized_text
        )

        try:
            model = genai.GenerativeModel("models/gemini-2.5-flash")
            response = model.generate_content(prompt)
            if response and response.text:
                summary = response.text.strip()
                print(f"[Gemini Summary] ✅ Generated summary ({len(summary)} chars)")
                return summary
            else:
                print("[Gemini Summary] ⚠️  Empty response")
        except Exception as e:
            print(f"[Gemini Summary] ⚠️  Gemini call failed: {e}")

        return linearized_text