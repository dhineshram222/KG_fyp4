# Automatic E-Notes Generation from Multiple Slide-Based Videos Using Knowledge Graph

## Vision

Build a research-grade system that transforms multiple lecture videos into **high-quality, in-depth, human-readable e-notes and summaries** using Knowledge Graph structuring — targeting university students who need noise-free, well-ordered study material, even as first-time learners.

## Research Hypothesis

> Knowledge-graph–based structuring improves the quality and reliability of automatically generated notes from educational videos, compared to non-KG summarization baselines.

**Evaluation Protocol:** Both KG-based and non-KG approaches are evaluated against the same ground-truth summaries using quantitative metrics.

---

## System Architecture (Current)

```mermaid
graph TD
    A[YouTube URL] --> B[Video Download]
    B --> C[Slide Extraction]
    B --> D[Audio Extraction]
    C --> E[OCR Text]
    C --> F[Diagram Detection & Captioning]
    C --> G[Formula Detection]
    D --> H[Whisper Transcription]
    E & F & G & H --> I[Fused Text per Slide]
    I --> J[Summary Construction via LLM]
    I --> K[KG Construction via LLM]
    K --> L[KG Visualization]
    K --> M[Multi-Video KG Fusion]
    M --> N[Structural KG Summarization]
    N --> O[Summary Post-Processing](final kg unified summary)
    K,I --> P[E-Notes(kg) PDF Generation]
    I --> R(combined of both videos fused text)
    R --> S[Non-KG Baseline Summary]
    R --> Q[E-Notes(non-kg) PDF Generation] (topic modeling)   
    O & S --> T[Evaluation Against Ground Truth]
```

| Component | Files | Status |
|-----------|-------|--------|
| Video Download | `ex.py` | ✅ Stable |
| Slide Extraction | `ex.py` | ✅ Stable |
| Audio Transcription | `ex.py` (Whisper) | ✅ Stable |
| Diagram Extraction & Captioning | `ex.py` (BLIP) | ✅ Stable |
| Formula Detection | `ex.py` | ⚠️ Needs Fix |
| OCR | `ex.py` (Tesseract) | ⚠️ Needs Improvement |
| KG Construction | `ex.py` (LLM) | ✅ Stable (out of scope) |
| KG Summarization | `kg_summarizer.py` | 🔴 Core Improvement Target |
| Summary Post-Processing | `summary_postprocessor.py` | 🔴 Core Improvement Target |
| Summary Refinement | `summary_refiner.py` | 🔴 Core Improvement Target |
| KG Fusion | `kg_fusion.py` | ✅ Stable (out of scope) |
| Content Alignment | `content_aligner.py` | ⚠️ Review needed |
| Narrative Restructuring | `narrative_restructurer.py` | ⚠️ Review needed |
| Semantic Clustering | `semantic_clusterer.py` | ⚠️ Review needed |
| Notes Generation | `ex.py` | 🔴 Quality improvement needed |
| Non-KG Baseline | `ex.py` | 🔴 Same quality issues |
| Evaluation | `ex.py` | ✅ Functional |
| Backend API | `app.py` (FastAPI) | ✅ Stable |
| Frontend UI | `App.jsx` (React/Vite) | ✅ Stable |

---

## Priority Goals

### P0 — Summary Quality (All Pipelines)

**Problem:** All summaries (KG-based fused, non kg fused , individual video summary ) are low quality — noisy, shallow, lacking in-depth knowledge, no logical flow or ordering of information.

**Desired state:**
- Summaries read like expert-written study material
- Clear logical ordering (concepts introduced before they are referenced)
- Deep, specific content — not generic surface-level rephrasing
- Zero noise (no OCR artifacts(like letters without any context awareness), meta-commentary, hallucinations)
- 7–10 dense, informative sentences per summary

**Key files:** `kg_summarizer.py`, `summary_postprocessor.py`, `summary_refiner.py`, `narrative_restructurer.py`, `semantic_clusterer.py`

---

### P1 — Evaluation Scores (KG-Based Unified Summary)

**Target metrics** (priority order):

| Metric | Current | Target |
|--------|---------|--------|
| Keyword Coverage | Low | 80–90% |
| Cosine Similarity | Low | 80–90% |
| BERTScore (F1) | Low | > 50–60% |
| ROUGE-1 | Low | > 50–60% |
| ROUGE-L | Low | > 40–50% |

> [!IMPORTANT]
> Improving P0 (summary quality) should naturally lift P1 scores. These are deeply coupled goals.

---

### P2 — Multi-Video Fusion Text Quality + OCR

**Problems:**
- Fused text across multiple videos contains too much noise
- OCR extracts individual **letters** instead of **words and sentences**

**Desired state:**
- Clean, coherent fused text that reads as a unified document
- OCR reliably extracts legible words/sentences from slides

---

### P3 — Processing Speed

**Problem:** Single video pipeline takes ~20 minutes up to KG construction.

**Desired state:** Measurably faster processing without sacrificing quality.

> [!NOTE]
> This is lowest priority. Tackle only after P0–P2 show clear improvement.

---

## Known Issues (Detail)

### Formula Detection
- Detects ordinary English words as mathematical formulas (false positives)
- Needs stricter validation — real formulas contain operators, symbols, or numeric patterns

### Notes Quality (KG-Based)
- Too many topics generated even for content that could be a single topic
- One-liner definitions treated as separate topics instead of being grouped
- Missing content ordering — topics appear in random order, not logical flow
- Lacks depth — definitions without context or explanation

### Summary Ordering
- No coherent narrative flow across sentences
- Concepts may appear before their prerequisites are established
- Related ideas scattered across the summary rather than grouped

---

## Out of Scope (Do Not Modify)

- Slide extraction pipeline
- Audio extraction and Whisper transcription
- Knowledge Graph construction (LLM-based triple extraction)
- Knowledge Graph fusion algorithm
- These modules are stable and must not be changed unless explicitly requested

---

## Constraints

| Constraint | Detail |
|------------|--------|
| **Environment** | Windows, fully local |
| **APIs** | No paid APIs or external cloud services |
| **Stability** | Existing pipelines must remain functional at all times |
| **Change style** | Incremental and reversible — no large refactors |
| **Reproducibility** | Results must be comparable across runs |
| **Stack** | Prefer current stack; new libs only if clearly justified and non-breaking |

---

## Tech Stack (Current)

**Backend:** Python 3, FastAPI, Uvicorn
**Frontend:** React (Vite), vanilla CSS
**ML/NLP:** Whisper (transcription), BART (summarization), BLIP (image captioning), spaCy, NLTK, Sentence-Transformers, Tesseract (OCR), NetworkX (graphs), PyVis (graph viz)
**Evaluation:** ROUGE, BERTScore, Cosine Similarity, Keyword Coverage

---

## Iterative Improvement Protocol

> [!TIP]
> Each improvement cycle should follow this pattern:

1. **Identify** — Pick one specific sub-problem from P0–P3
2. **Measure** — Run evaluation on current state, record baseline scores
3. **Implement** — Make targeted, incremental change in relevant module
4. **Verify** — Re-run evaluation, compare against baseline
5. **Commit** — If improvement confirmed, commit. If regression, revert.

All changes are measured against ground truth in `F:\FYP\ground truth\`.
