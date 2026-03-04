# kg_fusion.py

import json
import os
from typing import List, Dict, Tuple, Optional, Any, Set
from collections import defaultdict, Counter
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception as e:
    raise ImportError("Install sentence-transformers and scikit-learn: pip install sentence-transformers scikit-learn") from e


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, (list, dict)):
        return json.dumps(x, ensure_ascii=False)
    return str(x)


class KGFusionEnhanced:
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        semantic_weight: float = 0.5,
        structural_weight: float = 0.35,
        lexical_weight: float = 0.15,
        threshold: float = 0.78,
        type_strict: bool = True,
        allow_many_to_one: bool = False,
        relation_unify: bool = True,
        debug: bool = False
    ):
        """
        embedding_model: sentence-transformers model name
        semantic_weight, structural_weight, lexical_weight: weights for combined score (sum should be 1.0)
        threshold: final score threshold to consider entities as aligned
        type_strict: require type/label 'type' match when available (helps disambiguation)
        allow_many_to_one: if False, resolves one-to-many by picking highest-confidence match
        relation_unify: if True, tries to canonicalize relations when duplicates point between same nodes
        debug: print diagnostic info
        """
        self.model = SentenceTransformer(embedding_model)
        self.semantic_weight = semantic_weight
        self.structural_weight = structural_weight
        self.lexical_weight = lexical_weight
        self.threshold = threshold
        self.type_strict = type_strict
        self.allow_many_to_one = allow_many_to_one
        self.relation_unify = relation_unify
        self.debug = debug

    # Loading utilities

    def load_graph(self, nodes_path: str, edges_path: str) -> Tuple[List[Dict], List[Dict]]:
        with open(nodes_path, "r", encoding="utf-8") as f:
            nodes = json.load(f)
        with open(edges_path, "r", encoding="utf-8") as f:
            edges = json.load(f)
        
        nodes = [self._normalize_node(n, idx) for idx, n in enumerate(nodes, start=1)]
        edges = [self._normalize_edge(e) for e in edges]
        return nodes, edges

    def _normalize_node(self, node: Dict, idx: int) -> Dict:
      
        nid = _safe_str(node.get("id") or node.get("ID") or f"N_{idx}")
        label = _safe_str(node.get("label") or node.get("name") or node.get("title") or nid)
        desc = _safe_str(node.get("description") or node.get("desc") or "")
        ntype = _safe_str(node.get("type") or node.get("category") or "")
        return {"id": nid, "label": label, "description": desc, "type": ntype, "meta": node}

    def _normalize_edge(self, edge: Dict) -> Dict:
        src = _safe_str(edge.get("source") or edge.get("from") or edge.get("subject", ""))
        tgt = _safe_str(edge.get("target") or edge.get("to") or edge.get("object", ""))
        rel = _safe_str(edge.get("relation") or edge.get("rel") or edge.get("label") or "")
        
        meta = dict(edge)
        return {"source": src, "target": tgt, "relation": rel, "meta": meta}

  
    # Embeddings
 
    def _node_text(self, node: Dict) -> str:
        return f"{node['label']}. {node.get('description','')}".strip()

    def embed_nodes(self, nodes: List[Dict]) -> np.ndarray:
        texts = [self._node_text(n) for n in nodes]
        embs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embs

    def structural_embeddings(self, nodes: List[Dict], edges: List[Dict], label_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute neighbor-averaged structural embeddings: average embeddings of direct neighbors' labels.
        This captures context/role in the KG.
        """
        id_to_index = {n["id"]: i for i, n in enumerate(nodes)}
        neigh_emb = np.zeros_like(label_embeddings)
        counts = np.zeros((len(nodes),), dtype=int)
        for e in edges:
            s, t = e["source"], e["target"]
            if s in id_to_index and t in id_to_index:
                si = id_to_index[s]
                ti = id_to_index[t]
                neigh_emb[si] += label_embeddings[ti]
                neigh_emb[ti] += label_embeddings[si]
                counts[si] += 1
                counts[ti] += 1
        
        for i in range(len(nodes)):
            if counts[i] > 0:
                neigh_emb[i] = neigh_emb[i] / counts[i]
            else:
                
                neigh_emb[i] = label_embeddings[i]
        return neigh_emb

    
    # Similarity metrics
    
    def lexical_similarity(self, a: str, b: str) -> float:
        
        ta = set([w.lower() for w in a.split() if w.isalnum()])
        tb = set([w.lower() for w in b.split() if w.isalnum()])
        if not ta or not tb:
            return 0.0
        inter = ta.intersection(tb)
        union = ta.union(tb)
        return len(inter) / len(union)

   
    # Signature & Disambiguation
    
    def build_signature(self, node: Dict, edges: List[Dict], neighbor_depth: int = 1) -> Dict:
        """A compact signature for disambiguation: type, neighbor labels, relation types"""
        nid = node["id"]
        neighbors = []
        rels = []
        for e in edges:
            if e["source"] == nid:
                neighbors.append(e["target"])
                rels.append(e["relation"])
            elif e["target"] == nid:
                neighbors.append(e["source"])
                rels.append(e["relation"])
        sig = {"type": node.get("type", ""), "neighbors": neighbors, "rels": sorted(set(rels))}
        return sig

    def signature_overlap(self, sig1: Dict, sig2: Dict, nodes1_map: Dict[str, Dict], nodes2_map: Dict[str, Dict]) -> float:
        # compare neighbor label overlap
        nb1 = { _safe_str(nodes1_map[n]["label"]).lower() for n in sig1.get("neighbors", []) if n in nodes1_map }
        nb2 = { _safe_str(nodes2_map[n]["label"]).lower() for n in sig2.get("neighbors", []) if n in nodes2_map }
        if not nb1 or not nb2:
            return 0.0
        inter = nb1.intersection(nb2)
        union = nb1.union(nb2)
        return len(inter) / len(union)

    
    # Alignment
    
    def align(self, nodes1: List[Dict], edges1: List[Dict], nodes2: List[Dict], edges2: List[Dict]) -> Dict[str, Optional[str]]:
        """
        Return mapping: node_id_from_1 -> node_id_from_2 (or None)
        """

        
        emb1 = self.embed_nodes(nodes1)
        emb2 = self.embed_nodes(nodes2)
        struct1 = self.structural_embeddings(nodes1, edges1, emb1)
        struct2 = self.structural_embeddings(nodes2, edges2, emb2)

       
        nodes1_map = {n["id"]: n for n in nodes1}
        nodes2_map = {n["id"]: n for n in nodes2}

        sem_sim = cosine_similarity(emb1, emb2)          
        struct_sim = cosine_similarity(struct1, struct2) 

        alignments: Dict[str, Optional[str]] = {}
        candidate_scores: Dict[Tuple[str,str], float] = {}

        for i, n1 in enumerate(nodes1):
            best_j = None
            best_score = -1.0
            for j, n2 in enumerate(nodes2):
                
                lex = self.lexical_similarity(n1["label"], n2["label"])
                
                score = (self.semantic_weight * float(sem_sim[i, j])
                         + self.structural_weight * float(struct_sim[i, j])
                         + self.lexical_weight * lex)
                # type check penalty / require match if strict
                if self.type_strict:
                    type1 = (n1.get("type") or "").strip().lower()
                    type2 = (n2.get("type") or "").strip().lower()
                    if type1 and type2 and type1 != type2:
                        
                        score -= 0.25  
                candidate_scores[(n1["id"], n2["id"])] = score

                if score > best_score:
                    best_score = score
                    best_j = j

            
            if best_j is not None:
                sig1 = self.build_signature(n1, edges1)
                sig2 = self.build_signature(nodes2[best_j], edges2)
                sig_overlap = self.signature_overlap(sig1, sig2, nodes1_map, nodes2_map)
                
                best_score = best_score + 0.2 * sig_overlap

            if self.debug:
                print(f"[align] n1={n1['id']} best_n2={(nodes2[best_j]['id'] if best_j is not None else None)} score={best_score:.3f}")

            if best_score >= self.threshold:
                alignments[n1["id"]] = nodes2[best_j]["id"]
            else:
                alignments[n1["id"]] = None

        if not self.allow_many_to_one:
            rev: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
            for (n1id, n2id), score in candidate_scores.items():
                if alignments.get(n1id) == n2id:
                    rev[n2id].append((n1id, score))
            for n2id, lst in rev.items():
                if len(lst) <= 1:
                    continue
                lst_sorted = sorted(lst, key=lambda x: x[1], reverse=True)
                keep = set([lst_sorted[0][0]])
                for (n1id, _) in lst_sorted[1:]:
                    if alignments.get(n1id) == n2id:
                        alignments[n1id] = None
                        if self.debug:
                            print(f"[resolve] Demoting {n1id} (conflict on {n2id})")
        return alignments


    # Merge nodes & edges
   
    def merge(self, nodes1: List[Dict], edges1: List[Dict], nodes2: List[Dict], edges2: List[Dict],
              alignments: Dict[str, Optional[str]]) -> Tuple[List[Dict], List[Dict]]:
        
        nodes2_map = {n["id"]: n for n in nodes2}
        mapped_ids_from_2: Set[str] = set([v for v in alignments.values() if v is not None])

        fused_nodes: List[Dict] = []
        id_map_2_to_fused: Dict[str, str] = {}  

       
        for n1 in nodes1:
            mapped = alignments.get(n1["id"])
            if mapped:
                
                n2 = nodes2_map.get(mapped, {})
                fused_id = n1["id"]  
                fused_label = n1.get("label") or n2.get("label")
                fused_description = (n1.get("description") or "").strip()
                if not fused_description:
                    fused_description = (n2.get("description") or "").strip()
                
                t1 = (n1.get("type") or "").strip()
                t2 = (n2.get("type") or "").strip()
                fused_type = t1 or t2
                fused_nodes.append({
                    "id": fused_id,
                    "label": fused_label,
                    "description": fused_description,
                    "type": fused_type,
                    "sources": ["kg1", "kg2"],
                    "meta": {"kg1": n1.get("meta", {}), "kg2": n2.get("meta", {})}
                })
                id_map_2_to_fused[mapped] = fused_id
            else:
                fused_nodes.append({
                    "id": n1["id"],
                    "label": n1.get("label"),
                    "description": n1.get("description", ""),
                    "type": n1.get("type", ""),
                    "sources": ["kg1"],
                    "meta": {"kg1": n1.get("meta", {})}
                })

        
        for n2 in nodes2:
            if n2["id"] in mapped_ids_from_2:
                continue
           
            fused_id = f"KG2_{n2['id']}"
            fused_nodes.append({
                "id": fused_id,
                "label": n2.get("label"),
                "description": n2.get("description", ""),
                "type": n2.get("type", ""),
                "sources": ["kg2"],
                "meta": {"kg2": n2.get("meta", {})}
            })
            id_map_2_to_fused[n2["id"]] = fused_id

       
        fused_edges_raw: List[Dict] = []
        
        for e in edges1:
            fused_edges_raw.append({
                "source": e["source"],
                "target": e["target"],
                "relation": e["relation"] or "",
                "sources": ["kg1"],
                "meta": e.get("meta", {})
            })
        
        for e in edges2:
            src = e["source"]
            tgt = e["target"]
            mapped_src = id_map_2_to_fused.get(src)
            mapped_tgt = id_map_2_to_fused.get(tgt)
            if not mapped_src:
                mapped_src = f"KG2_{src}"
            if not mapped_tgt:
                mapped_tgt = f"KG2_{tgt}"
            fused_edges_raw.append({
                "source": mapped_src,
                "target": mapped_tgt,
                "relation": e.get("relation") or "",
                "sources": ["kg2"],
                "meta": e.get("meta", {})
            })

        
        def _canonicalize_rel(r: str) -> str:
            r = r.strip().lower()
            if r in ['includes', 'contains', 'consists of', 'comprises', 'has part']:
                return 'includes'
            if r in ['is a', 'is an', 'type', 'is type of', 'kind of']:
                return 'is a'
            if r in ['uses', 'utilizes', 'employs']:
                return 'uses'
            if r in ['associated with', 'related to', 'connected to', 'linked to', 'covers']:
                return 'is related to'
            if r in ['produces', 'generates', 'creates']:
                return 'produces'
            if r in ['defines', 'describes', 'means', 'refers to']:
                return 'defines'
            return r

        def _norm_rel(r: str) -> str:
            return _canonicalize_rel(r)

        edge_bucket = {}  
        for e in fused_edges_raw:
            k = (e["source"], e["target"], _norm_rel(e["relation"]))
            if k not in edge_bucket:
                edge_bucket[k] = {
                    "source": e["source"],
                    "target": e["target"],
                    "relations": [e["relation"]] if e.get("relation") else [],
                    "sources": set(e.get("sources", [])),
                    "meta_list": [e.get("meta", {})]
                }
            else:
                edge_bucket[k]["relations"].append(e.get("relation"))
                edge_bucket[k]["sources"].update(e.get("sources", []))
                edge_bucket[k]["meta_list"].append(e.get("meta", {}))

        fused_edges: List[Dict] = []
        for k, info in edge_bucket.items():
            src, tgt, _ = k
            relations = [r for r in info["relations"] if r]
            if not relations:
                chosen_relation = ""
            else:
                # Pick the most common canonical relation
                freq = Counter([_canonicalize_rel(r) for r in relations if r])
                chosen_relation = freq.most_common(1)[0][0]
                
            fused_edges.append({
                "source": src,
                "target": tgt,
                "relation": chosen_relation,
                "sources": list(info["sources"]),
                "meta": info["meta_list"]
            })


        return fused_nodes, fused_edges

   
    def fuse_from_files(self, nodes1_path: str, edges1_path: str, nodes2_path: str, edges2_path: str,
                        output_dir: Optional[str] = None) -> Tuple[List[Dict], List[Dict]]:
        nodes1, edges1 = self.load_graph(nodes1_path, edges1_path)
        nodes2, edges2 = self.load_graph(nodes2_path, edges2_path)
        align = self.align(nodes1, edges1, nodes2, edges2)
        fused_nodes, fused_edges = self.merge(nodes1, edges1, nodes2, edges2, align)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            
            with open(os.path.join(output_dir, "fused_nodes.json"), "w", encoding="utf-8") as f:
                json.dump(fused_nodes, f, indent=2, ensure_ascii=False)
            with open(os.path.join(output_dir, "fused_edges.json"), "w", encoding="utf-8") as f:
                json.dump(fused_edges, f, indent=2, ensure_ascii=False)
            
           
            with open(os.path.join(output_dir, "fused_summary.txt"), "w", encoding="utf-8") as f:
                for n in fused_nodes:
                    f.write(f"{n['label']}: {n.get('description','')}\n")
            
            print(f"✅ Fused KG saved to {output_dir} with {len(fused_nodes)} nodes and {len(fused_edges)} edges")

        return fused_nodes, fused_edges

   
    def fused_text_from_graph(self, fused_nodes: List[Dict], fused_edges: List[Dict], max_nodes: int = 200) -> str:
        parts = []
        for n in fused_nodes[:max_nodes]:
            lbl = _safe_str(n.get("label", ""))
            desc = _safe_str(n.get("description", ""))
            parts.append(f"{lbl}. {desc}".strip())
        
        rels = []
        for e in fused_edges[:max_nodes]:
            rels.append(f"{e['source']} -[{e.get('relation','')}]-> {e['target']}")
        return "\n".join(parts + [""] + rels)


