# app.py
import os
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse  # <-- ADD THIS IMPORT
from pathlib import Path
from ex import process_video_full as process_video, evaluate_summaries
import uuid
import json
import time  
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from ex import fuse_two_knowledge_graphs, OUTPUTS_DIR, run_notes_generation_wrapper, generate_non_kg_unified_summary, generate_non_kg_notes
from kg_summarizer import StructuralKGSummarizer, KnowledgeGraph, Node, Edge

app = FastAPI(title="Lecture Video Processing API")

# ------------------ CORS setup ------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ Paths ------------------
BASE_DIR = Path(__file__).parent.resolve()
OUTPUTS_DIR = BASE_DIR.parent.parent / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(OUTPUTS_DIR)), name="static")

# Serve all generated outputs (slides, audio, transcripts, OCR, formulas, diagrams, summaries, graphs)
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

# ------------------ Request/Response Models ------------------
class VideoRequest(BaseModel):
    youtube_url: str
    whisper_model: str = "base"  # allow selecting whisper model (default "base")

class EvaluationRequest(BaseModel):
    session_id: str
    reference_summary: str

class FuseRequest(BaseModel):
    session1_id: str
    session2_id: str
    generate_summary: bool = False
    generate_notes: bool = False

# ------------------ Endpoints ------------------
@app.get("/fused_graph/{fused_session_id}/{filename}")
async def serve_fused_graph(fused_session_id: str, filename: str):
    """
    Serve fused graph files (HTML, PNG, JSON) from the fused_kg directory.
    """
    fused_path = OUTPUTS_DIR / fused_session_id / "fused_kg" / filename
    
    if not fused_path.exists():
        raise HTTPException(status_code=404, detail=f"Fused graph file not found: {fused_path}")
    
    # Determine content type based on file extension
    if filename.endswith('.html'):
        media_type = 'text/html'
    elif filename.endswith('.png'):
        media_type = 'image/png'
    elif filename.endswith('.json'):
        media_type = 'application/json'
    else:
        media_type = 'text/plain'
    
    return FileResponse(str(fused_path), media_type=media_type)

@app.post("/fuse_graphs")
def fuse_graphs(req: dict):
    try:
        # Accept ANY payload safely
        session1_id = req.get("session1_id") or req.get("session_id_1")
        session2_id = req.get("session2_id") or req.get("session_id_2")
        
        if not session1_id or not session2_id:
            raise HTTPException(status_code=422, detail="Missing session IDs")

        # Validate sessions exist
        s1 = OUTPUTS_DIR / session1_id
        s2 = OUTPUTS_DIR / session2_id

        if not s1.exists() or not s2.exists():
            raise HTTPException(status_code=404, detail="One or both session folders not found.")

        # Create fused session ID and directory
        fused_session_id = f"{session1_id}_{session2_id}_fused"
        fused_dir = OUTPUTS_DIR / fused_session_id / "fused_kg"
        fused_dir.mkdir(parents=True, exist_ok=True)

        # Perform fusion
        fusion_result = fuse_two_knowledge_graphs(s1, s2, fused_dir, fused_session_id)

        return {
            "status": "success",
            "session_id": fused_session_id,
            "fused_graph_html": fusion_result["fused_graph_html"],
            "fused_graph_image": fusion_result["fused_graph_image"],
            "fused_nodes_file": fusion_result["fused_nodes_file"],
            "fused_edges_file": fusion_result["fused_edges_file"],
        }
        
    except Exception as e:
        print(f"Fusion error: {e}")
        raise HTTPException(status_code=500, detail=f"Fusion failed: {str(e)}")

@app.post("/generate_notes")
def generate_notes(req: dict):
    """
    Generate PDF notes from session contents (fused text/diagrams).
    """
    try:
        session_id = req.get("session_id")
        if not session_id:
            raise HTTPException(status_code=422, detail="Missing session_id")
            
        session_dir = OUTPUTS_DIR / session_id
        if not session_dir.exists():
            raise HTTPException(status_code=404, detail="Session not found")
            
        # Run generation
        # This wrapper handles everything including finding diagrams etc.
        pdf_path = run_notes_generation_wrapper(session_dir)
        
        if not pdf_path or not pdf_path.exists():
            raise HTTPException(status_code=500, detail="Notes generation failed (no PDF produced).")
            
        return {
            "status": "success",
            "pdf_url": f"/outputs/{session_id}/notes/{pdf_path.name}",
            "filename": pdf_path.name
        }
    except Exception as e:
        print(f"Notes generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fused_summary")
def generate_fused_summary(req: dict):
    """
    Generate a structural summary for a fused knowledge graph.
    """
    try:
        session_id = req.get("session_id")
        if not session_id:
            raise HTTPException(status_code=422, detail="Missing session_id")

        # Define paths
        fused_dir = OUTPUTS_DIR / session_id / "fused_kg"
        nodes_path = fused_dir / "fused_nodes.json"
        edges_path = fused_dir / "fused_edges.json"
        summary_path = fused_dir / "fused_summary.txt"

        # Check if files exist
        if not nodes_path.exists() or not edges_path.exists():
            raise HTTPException(status_code=404, detail="Fused nodes/edges not found. Run fusion first.")

        # Check cache (optional, maybe user wants to regenerate)
        # if summary_path.exists():
        #     return {"status": "success", "summary": summary_path.read_text(encoding="utf-8")}

        # Load JSON
        with open(nodes_path, "r", encoding="utf-8") as f:
            nodes_data = json.load(f)
        with open(edges_path, "r", encoding="utf-8") as f:
            edges_data = json.load(f)

        # Convert to Objects
        nodes = []
        for n in nodes_data:
            nodes.append(Node(n["id"], n.get("label", n["id"]), n.get("description", "")))
        
        edges = []
        for e in edges_data:
            edges.append(Edge(e["source"], e["target"], e.get("relation", "related_to")))

        kg = KnowledgeGraph(nodes, edges)

        # Run Summarizer
        print(f"[Fused Summary] Starting structural summarization for {session_id}...")
        summarizer = StructuralKGSummarizer()
        summary_text = summarizer.summarize(kg)

        # Save
        summary_path.write_text(summary_text, encoding="utf-8")
        print(f"[Fused Summary] Saved to {summary_path}")

        return {
            "status": "success",
            "session_id": session_id,
            "fused_summary": summary_text
        }

    except Exception as e:
        print(f"Fused summary error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Summary generation failed: {str(e)}")


@app.post("/evaluate_fused")
def evaluate_fused_summary_endpoint(req: dict):
    """
    Evaluate fused summary against a reference summary using comprehensive metrics.
    
    Evaluates the generated summary directly (no alignment/modification).
    
    Metrics:
    - ROUGE-1: Lexical content coverage
    - ROUGE-L: Structural similarity
    - Keyword Coverage: Salience proxy
    - BERTScore: Semantic equivalence (P/R/F1)
    - Sentence Embedding Cosine: Global meaning alignment
    """
    try:
        from ex import evaluate_summaries
        
        session_id = req.get("session_id")
        reference_summary = req.get("reference_summary")
        
        if not session_id:
            raise HTTPException(status_code=422, detail="Missing session_id")
        if not reference_summary or not reference_summary.strip():
            raise HTTPException(status_code=422, detail="Missing reference_summary")
        
        # Load fused summary
        fused_dir = OUTPUTS_DIR / session_id / "fused_kg"
        summary_path = fused_dir / "fused_summary.txt"
        
        if not summary_path.exists():
            raise HTTPException(status_code=404, detail="Fused summary not found. Generate it first.")
        
        fused_summary = summary_path.read_text(encoding="utf-8").strip()
        
        if not fused_summary:
            raise HTTPException(status_code=400, detail="Fused summary is empty")
        
        # Evaluate the original summary directly (no alignment)
        print(f"[Evaluation] Evaluating fused summary for {session_id}...")
        evaluation_results = evaluate_summaries(fused_summary, reference_summary)
        
        # Save evaluation results
        evaluations_dir = OUTPUTS_DIR / session_id / "evaluations"
        evaluations_dir.mkdir(parents=True, exist_ok=True)
        
        eval_data = {
            "original_summary": fused_summary,
            "aligned_summary": None,
            "reference_summary": reference_summary,
            "alignment_applied": False,
            "evaluation": evaluation_results
        }
        
        eval_file = evaluations_dir / "fused_evaluation.json"
        eval_file.write_text(json.dumps(eval_data, indent=2), encoding="utf-8")
        
        print(f"[Evaluation] Results saved to {eval_file}")
        
        return {
            "status": "success",
            "session_id": session_id,
            "alignment_applied": False,
            "evaluation": evaluation_results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Fused evaluation error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@app.post("/non_kg_summary")
def generate_non_kg_summary_endpoint(req: dict):
    """
    Generate a unified summary from combined raw text (non-KG based).
    """
    try:
        session_id = req.get("session_id")
        if not session_id:
            raise HTTPException(status_code=422, detail="Missing session_id")

        # Parse original session IDs from fused session id
        # Format: {session1_id}_{session2_id}_fused
        fused_dir = OUTPUTS_DIR / session_id
        if not fused_dir.exists():
            raise HTTPException(status_code=404, detail="Fused session not found")

        # Try to find original session directories
        parts = session_id.rsplit("_fused", 1)[0]
        # Split by underscore - each session_id is a UUID
        # UUID format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx (36 chars)
        if len(parts) > 73:  # two UUIDs + underscore
            s1_id = parts[:36]
            s2_id = parts[37:73]
        else:
            # Fallback: split at middle
            mid = len(parts) // 2
            s1_id = parts[:mid]
            s2_id = parts[mid+1:]

        session1_dir = OUTPUTS_DIR / s1_id
        session2_dir = OUTPUTS_DIR / s2_id

        if not session1_dir.exists() or not session2_dir.exists():
            raise HTTPException(status_code=404, detail=f"Original session directories not found (s1={s1_id}, s2={s2_id})")

        # Check cache
        cached_path = fused_dir / "non_kg_summary.txt"
        # Always regenerate for now

        print(f"[NonKG Summary] Generating for {session_id}...")
        summary_text = generate_non_kg_unified_summary(session1_dir, session2_dir, fused_dir)

        return {
            "status": "success",
            "session_id": session_id,
            "summary_text": summary_text
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Non-KG summary error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Non-KG summary generation failed: {str(e)}")


@app.post("/non_kg_notes")
def generate_non_kg_notes_endpoint(req: dict):
    """
    Generate structured notes using topic modeling (non-KG based).
    """
    try:
        session_id = req.get("session_id")
        if not session_id:
            raise HTTPException(status_code=422, detail="Missing session_id")

        fused_dir = OUTPUTS_DIR / session_id
        if not fused_dir.exists():
            raise HTTPException(status_code=404, detail="Fused session not found")

        # Parse original session IDs
        parts = session_id.rsplit("_fused", 1)[0]
        if len(parts) > 73:
            s1_id = parts[:36]
            s2_id = parts[37:73]
        else:
            mid = len(parts) // 2
            s1_id = parts[:mid]
            s2_id = parts[mid+1:]

        session1_dir = OUTPUTS_DIR / s1_id
        session2_dir = OUTPUTS_DIR / s2_id

        if not session1_dir.exists() or not session2_dir.exists():
            raise HTTPException(status_code=404, detail=f"Original session directories not found")

        print(f"[NonKG Notes] Generating for {session_id}...")
        result_path = generate_non_kg_notes(session1_dir, session2_dir, fused_dir)

        if not result_path or not Path(result_path).exists():
            raise HTTPException(status_code=500, detail="Notes generation failed")

        result_path = Path(result_path)
        # Determine relative URL
        rel_path = result_path.relative_to(OUTPUTS_DIR)
        url = f"/outputs/{str(rel_path).replace(chr(92), '/')}"

        return {
            "status": "success",
            "session_id": session_id,
            "pdf_url": url,
            "filename": result_path.name
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Non-KG notes error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Non-KG notes generation failed: {str(e)}")


@app.post("/evaluate_non_kg")
def evaluate_non_kg_summary_endpoint(req: dict):
    """
    Evaluate non-KG summary against a reference summary.
    """
    try:
        from ex import evaluate_summaries

        session_id = req.get("session_id")
        reference_summary = req.get("reference_summary")

        if not session_id:
            raise HTTPException(status_code=422, detail="Missing session_id")
        if not reference_summary or not reference_summary.strip():
            raise HTTPException(status_code=422, detail="Missing reference_summary")

        # Load non-KG summary
        summary_path = OUTPUTS_DIR / session_id / "non_kg_summary.txt"
        if not summary_path.exists():
            raise HTTPException(status_code=404, detail="Non-KG summary not found. Generate it first.")

        summary_text = summary_path.read_text(encoding="utf-8").strip()
        if not summary_text:
            raise HTTPException(status_code=400, detail="Non-KG summary is empty")

        print(f"[NonKG Eval] Evaluating for {session_id}...")
        evaluation_results = evaluate_summaries(summary_text, reference_summary)

        # Save results
        evaluations_dir = OUTPUTS_DIR / session_id / "evaluations"
        evaluations_dir.mkdir(parents=True, exist_ok=True)
        eval_data = {
            "summary": summary_text,
            "reference_summary": reference_summary,
            "evaluation": evaluation_results
        }
        eval_file = evaluations_dir / "non_kg_evaluation.json"
        eval_file.write_text(json.dumps(eval_data, indent=2), encoding="utf-8")

        return {
            "status": "success",
            "session_id": session_id,
            "evaluation": evaluation_results
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Non-KG eval error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@app.post("/evaluate_notes")
def evaluate_notes_endpoint(req: dict):
    """
    Evaluate generated notes using 4 metrics + verb fidelity:
    
    1. ROUGE-1/L Recall (vs reference notes)
    2. Flesch Reading Ease (readability)
    3. Gunning Fog Index (education level)
    4. Concept Dependency Score (logical flow from KG)
    5. Verb Fidelity (exact KG verb usage)
    
    Accepts:
        session_id: str
        notes_type: "kg" or "non_kg"
        reference_notes: str (optional — if empty, tries ground truth folder)
    """
    try:
        from notes_evaluator import evaluate_notes

        session_id = req.get("session_id")
        notes_type = req.get("notes_type", "kg")  # "kg" or "non_kg"
        reference_notes = req.get("reference_notes", "")

        if not session_id:
            raise HTTPException(status_code=422, detail="Missing session_id")

        # Locate generated notes text file
        if notes_type == "kg":
            notes_path = OUTPUTS_DIR / session_id / "notes" / "lecture_notes.txt"
        else:
            notes_path = OUTPUTS_DIR / session_id / "notes" / "non_kg_notes.txt"

        if not notes_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Notes file not found: {notes_path.name}. Generate notes first."
            )

        notes_text = notes_path.read_text(encoding="utf-8").strip()
        if not notes_text:
            raise HTTPException(status_code=400, detail="Generated notes file is empty")

        # Load KG data for dependency scoring
        kg_edges = []
        kg_nodes = []

        # Try fused KG first, then session-level KG
        edges_candidates = [
            OUTPUTS_DIR / session_id / "fused_kg" / "fused_edges.json",
            OUTPUTS_DIR / session_id / "graphs" / "kg_edges.json"
        ]
        nodes_candidates = [
            OUTPUTS_DIR / session_id / "fused_kg" / "fused_nodes.json",
            OUTPUTS_DIR / session_id / "graphs" / "kg_nodes.json"
        ]

        for p in edges_candidates:
            if p.exists():
                try:
                    kg_edges = json.loads(p.read_text(encoding="utf-8"))
                    break
                except:
                    pass
        for p in nodes_candidates:
            if p.exists():
                try:
                    kg_nodes = json.loads(p.read_text(encoding="utf-8"))
                    break
                except:
                    pass

        # If no reference provided, evaluation requires it
        if not reference_notes or not reference_notes.strip():
            print(f"[Notes Eval] No reference notes provided.")

        print(f"[Notes Eval] Evaluating {notes_type} notes for {session_id}...")
        print(f"[Notes Eval] Notes length: {len(notes_text)} chars")
        print(f"[Notes Eval] Reference: {'provided' if reference_notes else 'none'}")
        print(f"[Notes Eval] KG edges: {len(kg_edges)}, KG nodes: {len(kg_nodes)}")

        # Run evaluation
        evaluation = evaluate_notes(
            notes_text=notes_text,
            reference_text=reference_notes,
            kg_edges=kg_edges if kg_edges else None,
            kg_nodes=kg_nodes if kg_nodes else None
        )

        # Save results
        evaluations_dir = OUTPUTS_DIR / session_id / "evaluations"
        evaluations_dir.mkdir(parents=True, exist_ok=True)
        eval_file = evaluations_dir / f"{notes_type}_notes_evaluation.json"
        eval_file.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")

        print(f"[Notes Eval] ✅ Results saved to {eval_file}")

        return {
            "status": "success",
            "session_id": session_id,
            "notes_type": notes_type,
            "evaluation": evaluation
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Notes evaluation error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Notes evaluation failed: {str(e)}")


@app.post("/process")
async def process(req: VideoRequest):
    # Generate unique session ID
    session_id = str(uuid.uuid4())
    try:
        result = process_video(req.youtube_url, whisper_model=req.whisper_model, session_id=session_id)

        # --- Summary handling ---
        bart_summary_text = ""
        textrank_bart_summary_text = ""
        
        # Look for summary files in the summaries directory
        summaries_dir = OUTPUTS_DIR / session_id / "summaries"
        bart_summary_path = summaries_dir / "global_summary_bart.txt"
        textrank_bart_summary_path = summaries_dir / "global_summary_textrank_bart.txt"
        
        if bart_summary_path.exists():
            with open(bart_summary_path, "r", encoding="utf-8") as f:
                bart_summary_text = f.read()
        
        if textrank_bart_summary_path.exists():
            with open(textrank_bart_summary_path, "r", encoding="utf-8") as f:
                textrank_bart_summary_text = f.read()

        # --- Knowledge graph handling ---
        graph_img_url = None
        graph_html_url = None
        graphs_dir = OUTPUTS_DIR / session_id / "graphs"
        graph_img_path = graphs_dir / "kg_graph.png"
        graph_html_path = graphs_dir / "kg_graph.html"
        
        if graph_img_path.exists():
            graph_img_url = f"/outputs/{session_id}/graphs/kg_graph.png"
        if graph_html_path.exists():
            graph_html_url = f"/outputs/{session_id}/graphs/kg_graph.html"

        # --- Combined fused text handling ---
        combined_fused_url = None
        combined_fused_path = OUTPUTS_DIR / session_id / "combined" / "all_fused_text.txt"
        if combined_fused_path.exists():
            combined_fused_url = f"/outputs/{session_id}/combined/all_fused_text.txt"

        return {
            "status": "success",
            "session_id": session_id,
            "bart_summary_text": bart_summary_text,
            "textrank_bart_summary_text": textrank_bart_summary_text,
            "knowledge_graph_image": graph_img_url,
            "knowledge_graph_html": graph_html_url,
            "combined_fused_text": combined_fused_url,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/evaluate")
async def evaluate_summaries_endpoint(req: EvaluationRequest):
    """
    Evaluate both BART and TextRank+BART summaries against a reference summary.
    """
    try:
        print(f"📊 Evaluation request received for session: {req.session_id}")
        print(f"📝 Reference summary preview: {req.reference_summary[:100]}...")
        
        # Validate session_id
        if not req.session_id or not req.session_id.strip():
            raise HTTPException(status_code=400, detail="Session ID is required")
        
        # Validate reference summary
        if not req.reference_summary or not req.reference_summary.strip():
            raise HTTPException(status_code=400, detail="Reference summary is required")

        # Load the generated summaries
        session_dir = OUTPUTS_DIR / req.session_id
        if not session_dir.exists():
            raise HTTPException(status_code=404, detail=f"Session {req.session_id} not found")
        
        summaries_dir = session_dir / "summaries"
        evaluations_dir = session_dir / "evaluations"
        evaluations_dir.mkdir(parents=True, exist_ok=True)
        
        bart_summary_path = summaries_dir / "global_summary_bart.txt"
        textrank_bart_summary_path = summaries_dir / "global_summary_textrank_bart.txt"
        
        print(f"🔍 Looking for summaries in: {summaries_dir}")
        print(f"📄 BART summary exists: {bart_summary_path.exists()}")
        print(f"📄 TextRank+BART summary exists: {textrank_bart_summary_path.exists()}")
        
        if not bart_summary_path.exists():
            raise HTTPException(status_code=404, detail="BART summary not found")
        
        if not textrank_bart_summary_path.exists():
            raise HTTPException(status_code=404, detail="TextRank+BART summary not found")
        
        # Read the generated summaries
        with open(bart_summary_path, "r", encoding="utf-8") as f:
            bart_summary = f.read().strip()
        
        with open(textrank_bart_summary_path, "r", encoding="utf-8") as f:
            textrank_bart_summary = f.read().strip()

        print(f"📝 BART summary preview: {bart_summary[:100]}...")
        print(f"📝 TextRank+BART summary preview: {textrank_bart_summary[:100]}...")
        print(f"📝 Reference summary preview: {req.reference_summary[:100]}...")

        # Validate that summaries are not empty
        if not bart_summary:
            raise HTTPException(status_code=400, detail="BART summary is empty")
        
        if not textrank_bart_summary:
            raise HTTPException(status_code=400, detail="TextRank+BART summary is empty")

        # Evaluate both summaries
        print("🧮 Evaluating BART summary...")
        bart_evaluation = evaluate_summaries(bart_summary, req.reference_summary)
        
        print("🧮 Evaluating TextRank+BART summary...")
        textrank_bart_evaluation = evaluate_summaries(textrank_bart_summary, req.reference_summary)
        
        print("✅ Evaluation completed successfully")
        
        # Save evaluation results
        evaluation_results = {
            "bart_evaluation": bart_evaluation,
            "textrank_bart_evaluation": textrank_bart_evaluation,
            "reference_summary": req.reference_summary,
            "bart_summary": bart_summary,
            "textrank_bart_summary": textrank_bart_summary,
            "timestamp": time.time()
        }
        
        evaluation_file = evaluations_dir / "evaluation_results.json"
        with open(evaluation_file, "w", encoding="utf-8") as f:
            json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
        
        return {
            "status": "success",
            "session_id": req.session_id,
            "bart_evaluation": bart_evaluation,
            "textrank_bart_evaluation": textrank_bart_evaluation,
            "evaluation_saved": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Evaluation error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@app.get("/")
async def root():
    return {
        "status": "ok",
        "info": "POST /process with {'youtube_url': '<YouTube link>'} to process a lecture video."
    }


# ──────────────── Statistical Testing Endpoints ────────────────

@app.get("/list_dataset")
def list_dataset():
    """Return the dataset registry for frontend dropdowns."""
    from evaluation_store import DATASET_REGISTRY
    return {
        "status": "success",
        "dataset": [
            {"file_key": k, "display_name": v}
            for k, v in DATASET_REGISTRY.items()
        ]
    }


@app.post("/save_evaluation")
def save_evaluation_endpoint(req: dict):
    """
    Save per-video evaluation. Auto-pulls ROUGE-1 from session evaluation files.
    Overwrites if the same video_name already exists (deduplication).
    
    Required: video_name, session_id
    Optional: human_kg, human_nonkg (display-only, no statistical testing)
    """
    try:
        from evaluation_store import save_video_evaluation

        video_name = req.get("video_name")
        session_id = req.get("session_id")
        human_kg = req.get("human_kg")
        human_nonkg = req.get("human_nonkg")

        if not video_name:
            raise HTTPException(status_code=422, detail="Missing video_name")
        if not session_id:
            raise HTTPException(status_code=422, detail="Missing session_id")

        data = save_video_evaluation(
            video_name=video_name,
            session_id=session_id,
            human_kg=float(human_kg) if human_kg is not None else None,
            human_nonkg=float(human_nonkg) if human_nonkg is not None else None,
        )

        warning = None
        kg_score = data.get("metrics", {}).get("kg", {}).get("rouge1")
        nkg_score = data.get("metrics", {}).get("nonkg", {}).get("rouge1")
        if kg_score is None and nkg_score is None:
            warning = "No ROUGE-1 scores found in session. Ensure you have evaluated notes first."

        return {
            "status": "success",
            "video_name": video_name,
            "data": data,
            "warning": warning
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Save evaluation error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/run_statistics")
def run_statistics_endpoint():
    """
    Aggregate all available per-video evaluations and run statistical tests.
    Can be run at ANY time — works with however many videos are available.
    """
    try:
        from evaluation_store import aggregate_evaluations
        from stat_tests import run_statistical_analysis

        # Step 1: Aggregate whatever we have
        aggregated = aggregate_evaluations()

        if aggregated.get("dataset_size", 0) < 2:
            return {
                "status": "insufficient_data",
                "message": f"Only {aggregated.get('dataset_size', 0)} videos evaluated. Need ≥2 for statistics.",
                "aggregated": aggregated,
            }

        # Step 2: Run all statistical tests
        results = run_statistical_analysis()

        return {
            "status": "success",
            "results": results,
        }

    except Exception as e:
        print(f"Statistics error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/evaluation_status")
def evaluation_status_endpoint():
    """
    Return summary of all evaluated videos + latest statistical results.
    """
    try:
        from evaluation_store import list_evaluated_videos, EVAL_DIR

        videos = list_evaluated_videos()

        # Load latest statistical results if available
        stat_path = EVAL_DIR / "statistical_results.json"
        stat_results = None
        if stat_path.exists():
            stat_results = json.loads(stat_path.read_text(encoding="utf-8"))

        return {
            "status": "success",
            "evaluated_count": len(videos),
            "total_dataset": 21,
            "videos": videos,
            "statistical_results": stat_results,
        }

    except Exception as e:
        print(f"Evaluation status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ------------------ Optional: Background job version ------------------
@app.post("/process_async")
async def process_async(req: VideoRequest, background_tasks: BackgroundTasks):
    def job():
        session_id = str(uuid.uuid4())
        process_video(req.youtube_url, whisper_model=req.whisper_model, session_id=session_id)

    background_tasks.add_task(job)
    return {"status": "queued", "info": f"Processing started for {req.youtube_url}"}