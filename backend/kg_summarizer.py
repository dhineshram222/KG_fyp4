import networkx as nx
import numpy as np
from collections import defaultdict
from transformers import pipeline, BartTokenizer, BartForConditionalGeneration
from typing import List, Dict, Any, Optional
import torch
import warnings
import re

# Import the summary refiner for structure-aware sentence reconstruction
try:
    from summary_refiner import FusedSummaryRefiner
    HAS_REFINER = True
except ImportError:
    HAS_REFINER = False
    print("[Warning] summary_refiner not found. Fused summary refinement disabled.")

# Suppress warnings
warnings.filterwarnings("ignore")

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

# ====================== STRUCTURAL KG SUMMARIZATION PIPELINE ======================
class StructuralKGSummarizer:
    """Structural pipeline for KG summarization: 
       Centrality → Clustering → Salient Subgraph → Path Selection → Linearization → BART"""
    
    def __init__(self, use_refiner: bool = True):
        # Reuse the cached BART model from ex.py if available
        try:
            from ex import _get_bart_model
            tokenizer, model = _get_bart_model()
            self.bart = BARTSummarizer.__new__(BARTSummarizer)
            self.bart.device = 0 if torch.cuda.is_available() else -1
            self.bart.summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=self.bart.device)
        except Exception:
            self.bart = BARTSummarizer()
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
    
    def centrality(self, G):
        """Compute centrality scores (PageRank + Degree centrality)."""
        if len(G.nodes()) < 2:
            return {node: 1.0 for node in G.nodes()}
        
        try:
            pr = nx.pagerank(G)
            deg = nx.degree_centrality(G)
            # Combine scores
            return {node: (pr.get(node, 0) + deg.get(node, 0)) / 2 for node in G.nodes()}
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
            
            for tgt, rels in targets_map.items():
                if tgt == src: continue # No self-loops
                
                tgt_label = clean_text(G_clean.nodes[tgt].get('label', tgt))
                if not tgt_label or tgt_label.lower() == src_label.lower(): continue
                
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
                    clean_rels.append((r, r_clean)) # Keep original case for output, use lower for check
                
                # Check for specific verbs
                for raw, lower in clean_rels:
                    if lower not in ('is related to', 'related to'):
                        best_rel = raw # Use specific verb!
                        specific_found = True
                        break
                
                if not specific_found:
                     # If only generic exists, use "is related to"
                     best_rel = "is related to"

                valid_relations.append((best_rel, tgt_label))

            if not valid_relations:
                continue

            # ── 3. Compound Sentence Construction ──
            # Group by Verb:  { "provides": ["Data", "Views"], "is related to": ["Server"] }
            by_verb = defaultdict(list)
            for r, t in valid_relations:
                by_verb[r].append(t)
            
            # Construct clauses
            clauses = []
            for verb, objects in by_verb.items():
                # Format objects: "A, B, and C"
                if len(objects) == 1:
                    obj_str = objects[0]
                else:
                    obj_str = ", ".join(objects[:-1]) + f" and {objects[-1]}"
                
                # Verb formatting
                # If verb starts with "is" or "are" or "has", it's likely an auxiliary
                # If verb is a simple verb "provides", ensure it flows
                clauses.append(f"{verb} {obj_str}")

            # Assemble full sentence
            # "Subject clause1, clause2, and clause3."
            if not clauses: continue
            
            if len(clauses) == 1:
                full_sent = f"{src_label} {clauses[0]}."
            else:
                joined_clauses = ", ".join(clauses[:-1]) + f", and {clauses[-1]}"
                full_sent = f"{src_label} {joined_clauses}."
            
            # Final cleanup
            full_sent = re.sub(r'\s+', ' ', full_sent).strip()
            # Uppercase first letter
            if full_sent and full_sent[0].islower():
                full_sent = full_sent[0].upper() + full_sent[1:]
                
            sentences.append(full_sent)

        # Limit to top N sentences to avoid overwhelming context
        # (Though we clustered, so we have fewer sentences but they are denser)
        if len(sentences) > 30:
            sentences = sentences[:30]
            
        return " ".join(sentences)


    def summarize(self, kg, refine: bool = True):
        """Full pipeline with atomic generation, refinement, BART summarization,
        and post-processing for 7-10 context-rich, noise-free sentences.
        
        Args:
            kg: KnowledgeGraph object with nodes and edges
            refine: Whether to apply structure-aware sentence reconstruction
            
        Returns:
            Coherent summary text (7-10 sentences)
        """
        from summary_postprocessor import clean_and_constrain_summary

        # Convert to NetworkX
        G = self.to_nx(kg)
        
        # Step 1: Centrality
        scores = self.centrality(G)
        
        # Step 2: Salient Subgraph
        salient = self.salient_subgraph(G, scores, top_k=25)
        
        # Step 3: Linearization (Compound Clauses)
        structural_text = self.linearize(salient)
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
        
        # Step 5: BART Summarization (Constrained Paraphrasing)
        if hasattr(self, 'bart') and self.bart:
            # Prepend a prompt to encourage BART to synthesize instead of list
            prompt_wrapper = (
                "Provide a comprehensive summary of the following knowledge graph facts, "
                "connecting them into a coherent narrative that captures the overall meaning: "
            )
            final_input = prompt_wrapper + refined_text
            
            summary = self.bart.summarize(
                final_input,
                max_length=500,  # Allow more room for narrative
                min_length=200   # Force substantial content
            )
        else:
            summary = refined_text
        
        # Step 6: Post-processing — clean noise and constrain to 7-10 sentences
        summary = clean_and_constrain_summary(
            summary,
            min_sentences=7,
            max_sentences=10,
        )
        print(f"[KG Summary] Post-processed to {len(summary.split('.'))} sentences")
            
        return summary

