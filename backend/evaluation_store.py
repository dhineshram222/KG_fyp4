# evaluation_store.py
"""
Centralized evaluation storage for statistical testing.

Auto-pulls ROUGE-1 scores from session evaluation files.
Per-video JSON files named by topic. Re-runs overwrite (dedup by video name).
"""

import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List

# Session outputs directory (where evaluation JSONs from notes are saved)
BASE_DIR = Path(__file__).parent.resolve()
OUTPUTS_DIR = BASE_DIR.parent.parent / "outputs"

# Evaluation storage (stable, independent of session outputs)
# This stores the per-video evaluation files that persist across sessions
EVAL_DIR = BASE_DIR / "outputs" / "evaluation"
PER_VIDEO_DIR = EVAL_DIR / "per_video"


# ─── Dataset Registry ────────────────────────────────────────────────────────

DATASET_REGISTRY = {
    "intro_to_dsa": "Intro to DSA",
    "intro_to_linked_list": "Intro to Linked List",
    "intro_to_stack": "Intro to Stack",
    "dbms_over_file_system": "DBMS over file system",
    "data_abstraction": "data abstraction",
    "tier_architecture": "tier architecture(three-tier)",
    "data_models": "data models",
    "relational_model": "relational model",
    "keys": "keys",
    "er_model": "ER model",
    "phases_of_compiler": "phases of compiler",
    "recursive_descent_parser": "recursive descent parser",
    "computer_networks": "computer networks",
    "network_topology": "network topology",
    "tcp_ip_protocol": "tcp/ip protocol",
    "osi_model": "osi model",
    "stop_wait_protocol": "stop/wait protocol",
    "go_back_n": "go-back-n",
    "pure_aloha": "pure aloha",
    "csma_ca_vs_cd": "csma ca vs cd",
    "ipv4": "ipv4",
}

# Reverse lookup: display name → file key
_DISPLAY_TO_KEY = {v.lower(): k for k, v in DATASET_REGISTRY.items()}


def normalize_video_name(name: str) -> str:
    """Convert a display name to a file key."""
    key = _DISPLAY_TO_KEY.get(name.lower().strip())
    if key:
        return key
    normalized = name.lower().strip()
    normalized = re.sub(r'[/\\()\-]', ' ', normalized)
    normalized = re.sub(r'\s+', '_', normalized)
    normalized = re.sub(r'[^a-z0-9_]', '', normalized)
    normalized = re.sub(r'_+', '_', normalized).strip('_')
    return normalized


# ─── Auto-Pull ROUGE-1 from Session Folder ───────────────────────────────────

def extract_rouge1_from_session(session_id: str) -> Dict:
    """
    Auto-pull ROUGE-1 recall scores from the notes evaluation files
    saved in outputs/<session_id>/evaluations/.
    
    Returns: {"kg_rouge1": float|None, "nonkg_rouge1": float|None}
    """
    session_dir = OUTPUTS_DIR / session_id / "evaluations"
    result = {"kg_rouge1": None, "nonkg_rouge1": None}
    
    kg_eval_path = session_dir / "kg_notes_evaluation.json"
    nonkg_eval_path = session_dir / "non_kg_notes_evaluation.json"
    
    if kg_eval_path.exists():
        try:
            data = json.loads(kg_eval_path.read_text(encoding="utf-8"))
            rouge = data.get("rouge", {})
            result["kg_rouge1"] = rouge.get("rouge1_recall")
            print(f"[EvalStore] KG ROUGE-1 from session: {result['kg_rouge1']}")
        except Exception as e:
            print(f"[EvalStore] Failed to read KG eval: {e}")
    
    if nonkg_eval_path.exists():
        try:
            data = json.loads(nonkg_eval_path.read_text(encoding="utf-8"))
            rouge = data.get("rouge", {})
            result["nonkg_rouge1"] = rouge.get("rouge1_recall")
            print(f"[EvalStore] NonKG ROUGE-1 from session: {result['nonkg_rouge1']}")
        except Exception as e:
            print(f"[EvalStore] Failed to read NonKG eval: {e}")
    
    return result


# ─── Save / Load ──────────────────────────────────────────────────────────────

def save_video_evaluation(
    video_name: str,
    session_id: str,
    human_kg: Optional[float] = None,
    human_nonkg: Optional[float] = None,
) -> Dict:
    """
    Save per-video evaluation. Auto-pulls ROUGE-1 from session evaluation files.
    Overwrites if already exists (dedup by video name).
    
    Args:
        video_name: Display name from dataset (e.g., "Intro to Stack")
        session_id: Session ID to pull ROUGE-1 scores from
        human_kg: Human evaluation score for KG notes (average quiz correctness)
        human_nonkg: Human evaluation score for non-KG notes
    
    Returns:
        Dict with saved data
    """
    PER_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    
    file_key = normalize_video_name(video_name)
    file_path = PER_VIDEO_DIR / f"{file_key}.json"
    
    # Auto-extract ROUGE-1 from session evaluations
    scores = extract_rouge1_from_session(session_id)
    
    data = {
        "video_name": video_name,
        "file_key": file_key,
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            "kg": {
                "rouge1": round(scores["kg_rouge1"], 4) if scores["kg_rouge1"] is not None else None,
            },
            "nonkg": {
                "rouge1": round(scores["nonkg_rouge1"], 4) if scores["nonkg_rouge1"] is not None else None,
            }
        },
        "human_evaluation": {
            "kg": round(human_kg, 2) if human_kg is not None else None,
            "nonkg": round(human_nonkg, 2) if human_nonkg is not None else None,
        }
    }
    
    file_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    kg_r = scores["kg_rouge1"]
    nonkg_r = scores["nonkg_rouge1"]
    print(f"[EvalStore] Saved {file_key}.json (KG={kg_r}, NonKG={nonkg_r})")
    
    return data


def list_evaluated_videos() -> List[Dict]:
    """List all evaluated videos with their data."""
    if not PER_VIDEO_DIR.exists():
        return []
    results = []
    for f in sorted(PER_VIDEO_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            results.append(data)
        except Exception:
            pass
    return results


def aggregate_evaluations() -> Dict:
    """
    Aggregate all per-video evaluations into paired arrays.
    Can be run with ANY number of videos.
    """
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    videos = list_evaluated_videos()
    
    if not videos:
        return {"dataset_size": 0, "video_names": [], "metrics": {}, "human_evaluation": {}}
    
    video_names = []
    kg_rouge1 = []
    nonkg_rouge1 = []
    human_kg = []
    human_nonkg = []
    
    for v in videos:
        video_names.append(v.get("file_key", "unknown"))
        metrics = v.get("metrics", {})
        kg_rouge1.append(metrics.get("kg", {}).get("rouge1"))
        nonkg_rouge1.append(metrics.get("nonkg", {}).get("rouge1"))
        human = v.get("human_evaluation", {})
        human_kg.append(human.get("kg"))
        human_nonkg.append(human.get("nonkg"))
    
    aggregated = {
        "dataset_size": len(videos),
        "video_names": video_names,
        "metrics": {
            "rouge1": {
                "kg": kg_rouge1,
                "nonkg": nonkg_rouge1,
            }
        },
        "human_evaluation": {
            "kg": human_kg,
            "nonkg": human_nonkg,
        }
    }
    
    out_path = EVAL_DIR / "aggregated_results.json"
    out_path.write_text(json.dumps(aggregated, indent=2), encoding="utf-8")
    print(f"[EvalStore] Aggregated {len(videos)} videos → {out_path}")
    return aggregated
