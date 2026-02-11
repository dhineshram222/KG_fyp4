// App.jsx
import React, { useState } from "react";

function App() {
  const [video1Url, setVideo1Url] = useState("");
  const [video2Url, setVideo2Url] = useState("");
  const [referenceSummary1, setReferenceSummary1] = useState("");
  const [referenceSummary2, setReferenceSummary2] = useState("");
  const [result1, setResult1] = useState(null);
  const [result2, setResult2] = useState(null);
  const [loading1, setLoading1] = useState(false);
  const [loading2, setLoading2] = useState(false);
  const [evaluating1, setEvaluating1] = useState(false);
  const [evaluating2, setEvaluating2] = useState(false);
  const [error1, setError1] = useState(null);
  const [error2, setError2] = useState(null);
  const [showSummary1, setShowSummary1] = useState(true);
  const [showSummary2, setShowSummary2] = useState(true);
  const [showGraph1, setShowGraph1] = useState(true);
  const [showGraph2, setShowGraph2] = useState(true);
  const [showCombinedText1, setShowCombinedText1] = useState(false);
  const [showCombinedText2, setShowCombinedText2] = useState(false);
  const [showEvaluation1, setShowEvaluation1] = useState(false);
  const [showEvaluation2, setShowEvaluation2] = useState(false);

  // Fused evaluation state
  const [fusedReferenceSummary, setFusedReferenceSummary] = useState("");
  const [showFusedEvaluation, setShowFusedEvaluation] = useState(false);
  const [evaluatingFused, setEvaluatingFused] = useState(false);
  const [fusedEvaluationResult, setFusedEvaluationResult] = useState(null);

  // Unified fusion state
  const [fusedResult, setFusedResult] = useState(null);
  const [loadingFuse, setLoadingFuse] = useState(false);
  const [errorFuse, setErrorFuse] = useState(null);
  const [showUnifiedSummary, setShowUnifiedSummary] = useState(true);
  const [showUnifiedNotes, setShowUnifiedNotes] = useState(false);
  const [showUnifiedGraph, setShowUnifiedGraph] = useState(true);

  // Fused notes generation state
  const [generatingFusedNotes, setGeneratingFusedNotes] = useState(false);
  const [fusedNotesUrl, setFusedNotesUrl] = useState(null);

  // Fused Summary State
  const [generatingFusedSummary, setGeneratingFusedSummary] = useState(false);
  const [showFusedSummary, setShowFusedSummary] = useState(true);

  // Non-KG Unified State
  const [nonKgSummary, setNonKgSummary] = useState("");
  const [generatingNonKgSummary, setGeneratingNonKgSummary] = useState(false);
  const [nonKgNotesUrl, setNonKgNotesUrl] = useState(null);
  const [generatingNonKgNotes, setGeneratingNonKgNotes] = useState(false);
  const [showNonKgSummary, setShowNonKgSummary] = useState(false);
  const [nonKgReferenceSummary, setNonKgReferenceSummary] = useState("");
  const [nonKgEvaluationResult, setNonKgEvaluationResult] = useState(null);
  const [evaluatingNonKg, setEvaluatingNonKg] = useState(false);
  const [showNonKgEvaluation, setShowNonKgEvaluation] = useState(false);

  // Generic backend root; change if your backend runs on a different host/port
  const SERVER_ORIGIN = "http://127.0.0.1:8000";

  // Helper to set loading state for a given video index
  const setLoadingForIndex = (index, value) => {
    if (index === 1) setLoading1(value);
    else setLoading2(value);
  };

  const setErrorForIndex = (index, value) => {
    if (index === 1) setError1(value);
    else setError2(value);
  };

  const handleSubmit = async (videoUrl, setResult, setLoading, setError) => {
    if (!videoUrl.trim()) {
      setError("Please enter a valid YouTube URL.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${SERVER_ORIGIN}/process`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ youtube_url: videoUrl, whisper_model: "base" }),
      });

      if (!response.ok) {
        let errorData;
        try {
          errorData = await response.json();
        } catch {
          errorData = null;
        }
        throw new Error((errorData && (errorData.detail || errorData.error)) || `Processing failed (${response.status})`);
      }

      const data = await response.json();
      // Some backends respond with { status: "success", ... } or directly with session_id
      if (data.status === "success" || data.session_id || data.sessionId || data.session) {
        setResult(data);
      } else {
        // store whatever came back so UI can inspect it
        setResult(data);
      }
    } catch (err) {
      setError(err.message || "Failed to connect to server. Please try again.");
      console.error("Submit error:", err);
    }
    setLoading(false);
  };

  // Fixed handleEvaluate functions for each video
  const handleEvaluate = async (which, referenceSummary) => {
    if (!referenceSummary.trim()) {
      alert("Please enter a reference summary for evaluation.");
      return;
    }

    const whichResult = which === 1 ? result1 : result2;
    if (!whichResult?.session_id && !whichResult?.sessionId && !whichResult?.session) {
      alert("Please process a video first before evaluating.");
      return;
    }

    if (which === 1) setEvaluating1(true);
    else setEvaluating2(true);

    try {
      const sessionId = whichResult.session_id || whichResult.sessionId || whichResult.session;

      const response = await fetch(`${SERVER_ORIGIN}/evaluate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          reference_summary: referenceSummary,
        }),
      });

      if (!response.ok) {
        let errData = null;
        try { errData = await response.json(); } catch { }
        throw new Error((errData && (errData.detail || errData.error)) || `Evaluation failed (${response.status})`);
      }

      const data = await response.json();
      if (data.status === "success") {
        if (which === 1) setResult1(prev => ({ ...prev, evaluation: data }));
        else setResult2(prev => ({ ...prev, evaluation: data }));
      } else {
        throw new Error(data.detail || data.error || "Evaluation failed");
      }
    } catch (err) {
      console.error("Evaluation error:", err);
      alert("❌ Evaluation failed: " + (err.message || err));
    } finally {
      if (which === 1) setEvaluating1(false);
      else setEvaluating2(false);
    }
  };

  const handleGenerateNotes = async (result, setResult) => {
    if (!result?.session_id) return;

    const sessionId = result.session_id || result.sessionId || result.session;

    try {
      // Optimistically show generating status if needed, but we'll use a local loading state in the button or just alert
      const response = await fetch(`${SERVER_ORIGIN}/generate_notes`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId }),
      });

      if (!response.ok) {
        throw new Error(`Notes generation failed (${response.status})`);
      }

      const data = await response.json();
      if (data.status === "success") {
        setResult(prev => ({ ...prev, notes_url: data.pdf_url, notes_filename: data.filename }));
      } else {
        throw new Error(data.detail || "Unknown error");
      }
    } catch (err) {
      console.error("Notes generation error:", err);
      alert("Failed to generate notes: " + err.message);
    }
  };

  // ---------- NEW: Fusion handler ----------
  // In App.jsx, update the handleFuse function to use the new response structure
  const handleFuse = async (options = { include_summary: true, include_notes: true }) => {
    setErrorFuse(null);
    setFusedResult(null);

    if (!result1?.session_id && !result1?.sessionId && !result1?.session) {
      setErrorFuse("Please process Video 1 before fusing graphs.");
      return;
    }
    if (!result2?.session_id && !result2?.sessionId && !result2?.session) {
      setErrorFuse("Please process Video 2 before fusing graphs.");
      return;
    }

    setLoadingFuse(true);
    try {
      const s1 = result1.session_id || result1.sessionId || result1.session;
      const s2 = result2.session_id || result2.sessionId || result2.session;

      const payload = {
        session1_id: s1,
        session2_id: s2,
        session_id_1: s1,
        session_id_2: s2,
        include_summary: !!options.include_summary,
        include_notes: !!options.include_notes
      };

      const response = await fetch(`${SERVER_ORIGIN}/fuse_graphs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        let err;
        try {
          err = await response.json();
        } catch {
          err = null;
        }
        throw new Error((err && (err.detail || err.error)) || `Graph fusion failed (${response.status})`);
      }

      const data = await response.json();

      // Use the direct response from the backend
      const normalized = {
        status: data.status || "success",
        session_id: data.session_id || data.fused_session_id || `${s1}_${s2}_fused`,
        fused_graph_image: data.fused_graph_image,
        fused_graph_html: data.fused_graph_html,
        fused_nodes_file: data.fused_nodes_file,
        fused_edges_file: data.fused_edges_file,
        raw: data
      };

      setFusedResult(normalized);
    } catch (err) {
      setErrorFuse(err.message || "Failed to fuse graphs");
      console.error("Fuse error:", err);
    }
    setLoadingFuse(false);
  };

  // Utility to build absolute URL to backend-served files (images/html, etc.)
  const buildUrl = (path) => {
    if (!path) return null;
    // If path already absolute (starts with http), return as-is
    if (path.startsWith("http://") || path.startsWith("https://")) return path;
    if (path.startsWith("/fused_graph/")) {
      return `${SERVER_ORIGIN}${path}`;
    }
    // Ensure path begins with a slash for concatenation
    const p = path.startsWith("/") ? path : `/${path}`;
    return `${SERVER_ORIGIN}${p}`;
  };

  // Render evaluation results helper
  const renderEvaluationScores = (evaluation) => {
    if (!evaluation) return null;

    const bartEval = evaluation.bart_evaluation || evaluation.bart || {};
    const trbEval = evaluation.textrank_bart_evaluation || evaluation.textrank_bart || {};

    const safeGet = (obj, path, fallback = 0) => {
      try {
        return path.split('.').reduce((acc, k) => acc[k], obj) ?? fallback;
      } catch { return fallback; }
    };

    return (
      <div className="evaluation-scores">
        <h4>Evaluation Results</h4>
        <div className="evaluation-note">
          <strong>Note:</strong> Evaluation compares generated summaries against your reference summary using ROUGE and BLEU metrics.
        </div>

        <div style={{ flex: 1 }}>
          <h5>BART Summary</h5>
          <div>ROUGE-1 F1: {(safeGet(bartEval, "rouge.rouge1.fmeasure", 0) * 100).toFixed(2)}%</div>
          <div>ROUGE-2 F1: {(safeGet(bartEval, "rouge.rouge2.fmeasure", 0) * 100).toFixed(2)}%</div>
          <div>ROUGE-L F1: {(safeGet(bartEval, "rouge.rougeL.fmeasure", 0) * 100).toFixed(2)}%</div>
          <div>BLEU: {(safeGet(bartEval, "bleu", 0) * 100).toFixed(2)}%</div>
        </div>
      </div>
    );
  };

  // VideoSection component (keeps all original behavior)
  const VideoSection = ({
    title,
    videoUrl,
    setVideoUrl,
    referenceSummary,
    setReferenceSummary,
    result,
    loading,
    evaluating,
    error,
    showSummary,
    setShowSummary,
    showGraph,
    setShowGraph,
    showEvaluation,
    setShowEvaluation,
    showCombinedText,
    setShowCombinedText,
    handleSubmit,
    handleEvaluate,
    handleGenerateNotes,
    videoNumber
  }) => {
    // URLs for various resources
    const graphImageUrl = result?.knowledge_graph_image
      ? buildUrl(result.knowledge_graph_image)
      : result?.graph_image
        ? buildUrl(result.graph_image)
        : result?.knowledge_graph?.image
          ? buildUrl(result.knowledge_graph.image)
          : null;

    const graphHtmlUrl = result?.knowledge_graph_html
      ? buildUrl(result.knowledge_graph_html)
      : result?.graph_html
        ? buildUrl(result.graph_html)
        : result?.knowledge_graph?.html
          ? buildUrl(result.knowledge_graph.html)
          : null;

    const combinedTextUrl = result?.combined_fused_text
      ? buildUrl(result.combined_fused_text)
      : result?.combined_text
        ? buildUrl(result.combined_text)
        : result?.combined_fused
          ? buildUrl(result.combined_fused)
          : null;

    const [combinedTextContent, setCombinedTextContent] = useState("");

    const loadCombinedText = async () => {
      if (!combinedTextUrl) return;
      try {
        const response = await fetch(combinedTextUrl);
        const text = await response.text();
        setCombinedTextContent(text);
      } catch (err) {
        console.error("Failed to load combined text:", err);
        setCombinedTextContent("Failed to load combined text");
      }
    };

    return (
      <div className="video-section" style={{ border: "1px solid #ddd", padding: "1rem", borderRadius: 8, marginBottom: "1rem" }}>
        <h2>{title}</h2>
        <input
          value={videoUrl}
          onChange={(e) => setVideoUrl(e.target.value)}
          placeholder="Enter YouTube URL"
          className="input"
          style={{ width: "70%", padding: "0.5rem", marginRight: "0.5rem" }}
        />
        <button
          onClick={() => handleSubmit(videoUrl, videoNumber === 1 ? setResult1 : setResult2, (val) => setLoadingForIndex(videoNumber, val), (msg) => setErrorForIndex(videoNumber, msg))}
          disabled={loading}
          className="button"
        >
          {loading ? "Processing..." : "Process Video"}
        </button>

        {error && <div className="error" style={{ color: "crimson", marginTop: "0.5rem" }}>{error}</div>}

        {result && (
          <div className="result-container" style={{ marginTop: "1rem" }}>
            <div className="processing-info">
              <div className="info-message" style={{ marginBottom: "0.5rem" }}>
                ✅ Video processing complete!
              </div>
            </div>

            <div className="toggle-buttons" style={{ display: "flex", gap: "0.5rem", marginBottom: "0.5rem", flexWrap: "wrap" }}>
              <button onClick={() => setShowSummary(!showSummary)} className={`button secondary ${showSummary ? "active" : ""}`}>{showSummary ? "Hide Summary" : "Show Summary"}</button>
              <button onClick={() => setShowGraph(!showGraph)} className={`button secondary ${showGraph ? "active" : ""}`}>{showGraph ? "Hide Knowledge Graph" : "Show Knowledge Graph"}</button>
            </div>



            {showSummary && (
              <div className="summary-section" style={{ marginBottom: "1rem" }}>
                <h3 className="section-title">Generated Summaries</h3>
                <div className="summary-method">
                  <h4>BART Summary</h4>
                  <div className="summary bart-summary" style={{ background: "#f7f7f7", padding: "0.5rem", borderRadius: 6 }}>
                    {result.bart_summary_text || result.bart_summary || "No summary generated"}
                  </div>
                </div>
              </div>
            )}

            {showGraph && graphImageUrl && (
              <div className="graph-section" style={{ marginBottom: "1rem" }}>
                <h3 className="section-title">Knowledge Graph</h3>
                <img src={graphImageUrl} alt="Knowledge Graph" className="graph-image" style={{ maxWidth: "100%", borderRadius: 6 }} />
                {graphHtmlUrl && (
                  <div style={{ marginTop: "0.5rem" }}>
                    <a href={graphHtmlUrl} target="_blank" rel="noreferrer" className="graph-link">Open Interactive Graph</a>
                  </div>
                )}
              </div>
            )}


          </div>
        )}
      </div>
    );
  };

  // ---------- NEW: Unified Fusion Section ----------
  const UnifiedSection = () => {
    const fusedImgUrl = buildUrl(fusedResult?.fused_graph_image);
    const fusedHtmlUrl = buildUrl(fusedResult?.fused_graph_html);

    // KG-Based
    const fusedSummaryText = fusedResult?.fused_summary || fusedResult?.fused_summary_text || fusedResult?.unified_summary || "";

    // Existing fuse handler
    const handleFuseClick = async () => {
      // Option A default: include both optional fields
      await handleFuse({ include_summary: true, include_notes: true });
    };

    // --- KG HANDLERS ---
    const handleGenerateFusedSummary = async () => {
      if (!fusedResult?.session_id) return;
      setGeneratingFusedSummary(true);
      try {
        const response = await fetch(`${SERVER_ORIGIN}/fused_summary`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: fusedResult.session_id }),
        });

        if (!response.ok) {
          const err = await response.json();
          throw new Error(err.detail || "Summary generation failed");
        }

        const data = await response.json();
        setFusedResult(prev => ({ ...prev, fused_summary: data.fused_summary }));
        setShowUnifiedSummary(true);

      } catch (err) {
        console.error("Fused summary error:", err);
        alert("Failed to generate summary: " + err.message);
      }
      setGeneratingFusedSummary(false);
    };

    const handleEvaluateFusedSummary = async () => {
      if (!fusedReferenceSummary.trim()) {
        alert("Please enter a reference summary for evaluation.");
        return;
      }
      if (!fusedResult?.session_id) {
        alert("Please generate a fused summary first.");
        return;
      }

      setEvaluatingFused(true);
      try {
        const response = await fetch(`${SERVER_ORIGIN}/evaluate_fused`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            session_id: fusedResult.session_id,
            reference_summary: fusedReferenceSummary
          })
        });

        if (!response.ok) {
          const err = await response.json();
          throw new Error(err.detail || "Evaluation failed");
        }

        const data = await response.json();
        setFusedEvaluationResult(data.evaluation);
        setShowFusedEvaluation(true);
      } catch (err) {
        console.error("Fused evaluation error:", err);
        alert("Evaluation failed: " + err.message);
      }
      setEvaluatingFused(false);
    };

    // --- NON-KG HANDLERS ---
    const handleGenerateNonKgSummary = async () => {
      if (!fusedResult?.session_id) return;
      setGeneratingNonKgSummary(true);
      try {
        const response = await fetch(`${SERVER_ORIGIN}/non_kg_summary`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: fusedResult.session_id }),
        });
        if (!response.ok) {
          const err = await response.json();
          throw new Error(err.detail || "Summary generation failed");
        }
        const data = await response.json();
        setNonKgSummary(data.summary_text);
        setShowNonKgSummary(true);
      } catch (err) {
        console.error(err);
        alert("Failed to generate non-KG summary: " + err.message);
      }
      setGeneratingNonKgSummary(false);
    };

    const handleGenerateNonKgNotes = async () => {
      if (!fusedResult?.session_id) return;
      setGeneratingNonKgNotes(true);
      try {
        const response = await fetch(`${SERVER_ORIGIN}/non_kg_notes`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: fusedResult.session_id }),
        });
        if (!response.ok) {
          const err = await response.json();
          throw new Error(err.detail || "Notes generation failed");
        }
        const data = await response.json();
        setNonKgNotesUrl(`${SERVER_ORIGIN}${data.pdf_url}`);
      } catch (err) {
        console.error(err);
        alert("Failed to generate notes: " + err.message);
      }
      setGeneratingNonKgNotes(false);
    };

    const handleEvaluateNonKg = async () => {
      if (!nonKgReferenceSummary.trim()) {
        alert("Please enter a reference summary.");
        return;
      }
      setEvaluatingNonKg(true);
      try {
        const response = await fetch(`${SERVER_ORIGIN}/evaluate_non_kg`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            session_id: fusedResult.session_id,
            reference_summary: nonKgReferenceSummary
          })
        });
        if (!response.ok) {
          const err = await response.json();
          throw new Error(err.detail || "Evaluation failed");
        }
        const data = await response.json();
        setNonKgEvaluationResult(data.evaluation);
        setShowNonKgEvaluation(true);
      } catch (err) {
        console.error(err);
        alert("Failed to evaluate: " + err.message);
      }
      setEvaluatingNonKg(false);
    };


    const downloadText = (text, filename = "unified_summary.txt") => {
      const blob = new Blob([text || ""], { type: "text/plain;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      a.click();
      URL.revokeObjectURL(url);
    };

    // Render comprehensive evaluation scores
    const renderEvaluationScores = (evaluation, title = "Evaluation Results") => {
      if (!evaluation) return null;

      const safeGet = (obj, path, fallback = 0) => {
        try {
          return path.split('.').reduce((acc, k) => acc[k], obj) ?? fallback;
        } catch { return fallback; }
      };

      return (
        <div className="evaluation-scores" style={{ background: "#f9f9f9", padding: "1rem", borderRadius: 8, marginTop: "1rem", fontSize: "0.9rem" }}>
          <h4>📊 {title}</h4>
          <table style={{ width: "100%", borderCollapse: "collapse", marginTop: "0.5rem" }}>
            <thead>
              <tr style={{ borderBottom: "2px solid #ddd" }}>
                <th style={{ textAlign: "left", padding: "0.5rem" }}>Metric</th>
                <th style={{ textAlign: "right", padding: "0.5rem" }}>Score</th>
              </tr>
            </thead>
            <tbody>
              {/* ROUGE-1 */}
              <tr style={{ borderBottom: "1px solid #eee" }}>
                <td style={{ padding: "0.5rem" }}>ROUGE-1 F1</td>
                <td style={{ padding: "0.5rem", textAlign: "right", fontWeight: "bold" }}>{(safeGet(evaluation, "rouge.rouge1.fmeasure", 0) * 100).toFixed(2)}%</td>
              </tr>
              {/* ROUGE-L */}
              <tr style={{ borderBottom: "1px solid #eee" }}>
                <td style={{ padding: "0.5rem" }}>ROUGE-L F1</td>
                <td style={{ padding: "0.5rem", textAlign: "right", fontWeight: "bold" }}>{(safeGet(evaluation, "rouge.rougeL.fmeasure", 0) * 100).toFixed(2)}%</td>
              </tr>
              {/* Keyword Coverage */}
              <tr style={{ borderBottom: "1px solid #eee" }}>
                <td style={{ padding: "0.5rem" }}>Keyword Coverage</td>
                <td style={{ padding: "0.5rem", textAlign: "right", fontWeight: "bold" }}>{(safeGet(evaluation, "keyword_coverage", 0) * 100).toFixed(2)}%</td>
              </tr>
              {/* BERTScore F1 */}
              <tr style={{ borderBottom: "1px solid #eee" }}>
                <td style={{ padding: "0.5rem" }}>BERTScore F1</td>
                <td style={{ padding: "0.5rem", textAlign: "right", fontWeight: "bold" }}>{(safeGet(evaluation, "bertscore.f1", 0) * 100).toFixed(2)}%</td>
              </tr>
              {/* Sentence Cosine */}
              <tr>
                <td style={{ padding: "0.5rem" }}>Sentence Cosine</td>
                <td style={{ padding: "0.5rem", textAlign: "right", fontWeight: "bold" }}>{(safeGet(evaluation, "sentence_cosine", 0) * 100).toFixed(2)}%</td>
              </tr>
            </tbody>
          </table>
        </div>
      );
    };

    return (
      <div className="unified-section" style={{ border: "2px dashed #aaa", padding: "1rem", borderRadius: 8, marginTop: "2rem" }}>
        <h2>Unified / Fused Outputs</h2>
        <p>Combine knowledge from both videos into a unified graph and summary.</p>

        <div style={{ display: "flex", gap: "0.5rem", marginBottom: "0.5rem" }}>
          <button onClick={handleFuseClick} disabled={loadingFuse} className="button primary">{loadingFuse ? "Fusing graphs..." : "Create Unified Knowledge Graph"}</button>
        </div>

        {errorFuse && <div style={{ color: "crimson", marginTop: "0.5rem" }}>{errorFuse}</div>}

        {fusedResult && (
          <div style={{ marginTop: "1rem" }}>
            <div style={{ marginBottom: "0.5rem" }}>
              <strong>Fused session:</strong> {fusedResult.session_id || "n/a"}
            </div>

            {/* FUSED GRAPH VISUALIZATION */}
            {fusedImgUrl && (
              <div style={{ marginBottom: "2rem", textAlign: 'center' }}>
                <h4>Fused Knowledge Graph</h4>
                <img src={fusedImgUrl} alt="Fused KG" style={{ maxWidth: "100%", maxHeight: "500px", borderRadius: 6, border: "1px solid #ddd" }} />
                {fusedHtmlUrl && <div style={{ marginTop: "0.5rem" }}><a href={fusedHtmlUrl} target="_blank" rel="noreferrer">Open interactive fused graph</a></div>}
              </div>
            )}

            <div style={{ display: "flex", gap: "2rem", flexWrap: "wrap" }}>

              {/* LEFT COLUMN: KG-BASED ANALYSIS */}
              <div style={{ flex: 1, minWidth: "300px", paddingRight: "1rem", borderRight: "1px solid #eee" }}>
                <h3 style={{ borderBottom: "2px solid #007bff", paddingBottom: "0.5rem", color: "#007bff" }}>📊 KG-Based Analysis</h3>
                <p style={{ fontSize: "0.85rem", color: "#666" }}>Derived strictly from the Knowledge Graph structure.</p>

                {/* KG Summary */}
                <div style={{ marginTop: "1rem" }}>
                  <h4>1. KG Unified Summary</h4>
                  <div style={{ display: "flex", gap: "10px", alignItems: "center", marginBottom: "10px" }}>
                    {!fusedSummaryText && (
                      <button
                        onClick={handleGenerateFusedSummary}
                        disabled={generatingFusedSummary}
                        className="button primary"
                        style={{ backgroundColor: "#007bff" }}
                      >
                        {generatingFusedSummary ? "Generating..." : "Generate KG Summary"}
                      </button>
                    )}

                    {fusedSummaryText && (
                      <button
                        onClick={() => setShowFusedSummary(!showFusedSummary)}
                        className="button secondary"
                      >
                        {showFusedSummary ? "Hide Summary" : "Show Summary"}
                      </button>
                    )}
                  </div>

                  {fusedSummaryText && showFusedSummary && (
                    <div>
                      <div style={{ background: "#eef6fc", padding: "0.5rem", borderRadius: 6, maxHeight: "200px", overflowY: "auto", fontSize: "0.9rem" }}>
                        {fusedSummaryText}
                      </div>
                      <div style={{ marginTop: "0.5rem" }}>
                        <button
                          className="button primary"
                          onClick={handleGenerateFusedSummary}
                          disabled={generatingFusedSummary}
                          style={{ backgroundColor: "#007bff", marginRight: "0.5rem", fontSize: "0.9rem", padding: "0.3rem 0.8rem" }}
                        >
                          {generatingFusedSummary ? "Regenerating..." : "Regenerate Summary"}
                        </button>
                        <button className="button secondary" onClick={() => downloadText(fusedSummaryText, "kg_summary.txt")}>Download</button>
                      </div>
                    </div>
                  )}
                </div>

                {/* KG Notes */}
                <div style={{ marginTop: "1.5rem" }}>
                  <h4>2. KG Structured Notes</h4>
                  <button
                    onClick={async () => {
                      if (!fusedResult?.session_id) return;
                      setGeneratingFusedNotes(true);
                      try {
                        const response = await fetch(`${SERVER_ORIGIN}/generate_notes`, {
                          method: "POST",
                          headers: { "Content-Type": "application/json" },
                          body: JSON.stringify({ session_id: fusedResult.session_id })
                        });
                        if (!response.ok) throw new Error("Notes generation failed");
                        const data = await response.json();
                        if (data.pdf_url) {
                          setFusedNotesUrl(`${SERVER_ORIGIN}${data.pdf_url}`);
                        }
                      } catch (err) {
                        alert("Failed: " + err.message);
                      }
                      setGeneratingFusedNotes(false);
                    }}
                    disabled={generatingFusedNotes}
                    className="button primary"
                    style={{ backgroundColor: "#007bff", marginRight: "0.5rem" }}
                  >
                    {generatingFusedNotes ? "Generating..." : "Generate KG Notes (PDF)"}
                  </button>
                  {fusedNotesUrl && (
                    <a href={fusedNotesUrl} target="_blank" rel="noreferrer" className="button secondary">Download PDF</a>
                  )}
                </div>

                {/* KG Evaluation */}
                <div style={{ marginTop: "1.5rem", borderTop: "1px dashed #ccc", paddingTop: "1rem" }}>
                  <h4>3. KG Evaluation</h4>
                  <textarea
                    value={fusedReferenceSummary}
                    onChange={(e) => setFusedReferenceSummary(e.target.value)}
                    placeholder="Reference summary..."
                    rows="3"
                    style={{ width: "100%", padding: "0.5rem", marginBottom: "0.5rem" }}
                    disabled={!fusedSummaryText}
                  />
                  <button
                    onClick={handleEvaluateFusedSummary}
                    disabled={evaluatingFused || !fusedSummaryText || !fusedReferenceSummary.trim()}
                    className="button primary"
                    style={{ backgroundColor: "#007bff" }}
                  >
                    {evaluatingFused ? "Evaluating..." : "Evaluate KG Summary"}
                  </button>
                  {showFusedEvaluation && fusedEvaluationResult && renderEvaluationScores(fusedEvaluationResult, "KG Results")}
                </div>

              </div>

              {/* RIGHT COLUMN: NON-KG BASED ANALYSIS */}
              <div style={{ flex: 1, minWidth: "300px", paddingLeft: "1rem" }}>
                <h3 style={{ borderBottom: "2px solid #28a745", paddingBottom: "0.5rem", color: "#28a745" }}>📝 Text-Based Analysis</h3>
                <p style={{ fontSize: "0.85rem", color: "#666" }}>Derived from combined transcripts (BART + Topic Modeling).</p>

                {/* Non-KG Summary */}
                <div style={{ marginTop: "1rem" }}>
                  <h4>1. Unified Text Summary</h4>
                  <div style={{ display: "flex", gap: "10px", alignItems: "center", marginBottom: "10px" }}>
                    {!nonKgSummary && (
                      <button
                        onClick={handleGenerateNonKgSummary}
                        disabled={generatingNonKgSummary}
                        className="button primary"
                        style={{ backgroundColor: "#28a745" }}
                      >
                        {generatingNonKgSummary ? "Generating..." : "Generate Text Summary"}
                      </button>
                    )}

                    {nonKgSummary && (
                      <button
                        onClick={() => setShowNonKgSummary(!showNonKgSummary)}
                        className="button secondary"
                      >
                        {showNonKgSummary ? "Hide Summary" : "Show Summary"}
                      </button>
                    )}
                  </div>

                  {nonKgSummary && showNonKgSummary && (
                    <div>
                      <div style={{ background: "#f0fff4", padding: "0.5rem", borderRadius: 6, maxHeight: "300px", overflowY: "auto", fontSize: "0.9rem", border: "1px solid #c3e6cb" }}>
                        {nonKgSummary}
                      </div>
                      <div style={{ marginTop: "0.5rem" }}>
                        <button
                          className="button primary"
                          onClick={handleGenerateNonKgSummary}
                          disabled={generatingNonKgSummary}
                          style={{ backgroundColor: "#28a745", marginRight: "0.5rem", fontSize: "0.9rem", padding: "0.3rem 0.8rem" }}
                        >
                          {generatingNonKgSummary ? "Regenerating..." : "Regenerate"}
                        </button>
                        <button className="button secondary" onClick={() => downloadText(nonKgSummary, "text_summary.txt")}>Download</button>
                      </div>
                    </div>
                  )}
                </div>

                {/* Non-KG Notes */}
                <div style={{ marginTop: "1.5rem" }}>
                  <h4>2. Topic-Modeled Notes</h4>
                  <button
                    onClick={handleGenerateNonKgNotes}
                    disabled={generatingNonKgNotes}
                    className="button primary"
                    style={{ backgroundColor: "#28a745", marginRight: "0.5rem" }}
                  >
                    {generatingNonKgNotes ? "Generating..." : "Generate Text Notes (PDF)"}
                  </button>
                  {nonKgNotesUrl && (
                    <a href={nonKgNotesUrl} target="_blank" rel="noreferrer" className="button secondary">Download PDF</a>
                  )}
                </div>

                {/* Non-KG Evaluation */}
                <div style={{ marginTop: "1.5rem", borderTop: "1px dashed #ccc", paddingTop: "1rem" }}>
                  <h4>3. Text Evaluation</h4>
                  <textarea
                    value={nonKgReferenceSummary}
                    onChange={(e) => setNonKgReferenceSummary(e.target.value)}
                    placeholder="Reference summary..."
                    rows="3"
                    style={{ width: "100%", padding: "0.5rem", marginBottom: "0.5rem" }}
                    disabled={!nonKgSummary}
                  />
                  <button
                    onClick={handleEvaluateNonKg}
                    disabled={evaluatingNonKg || !nonKgSummary || !nonKgReferenceSummary.trim()}
                    className="button primary"
                    style={{ backgroundColor: "#28a745" }}
                  >
                    {evaluatingNonKg ? "Evaluating..." : "Evaluate Text Summary"}
                  </button>
                  {showNonKgEvaluation && nonKgEvaluationResult && renderEvaluationScores(nonKgEvaluationResult, "Text Results")}
                </div>

              </div>
            </div>
          </div>
        )}

        {!fusedResult && !loadingFuse && <div style={{ marginTop: "0.5rem", color: "#666" }}>No fused graph yet. Process both videos and click "Create Unified Knowledge Graph".</div>}
      </div>
    );
  };

  return (
    <div className="container" style={{ maxWidth: 1100, margin: "1rem auto", fontFamily: "Arial, sans-serif" }}>
      <h1>YouTube Lecture Processor</h1>


      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem" }}>
        <div>
          <VideoSection
            title="Video 1"
            videoUrl={video1Url}
            setVideoUrl={setVideo1Url}
            referenceSummary={referenceSummary1}
            setReferenceSummary={setReferenceSummary1}
            result={result1}
            loading={loading1}
            evaluating={evaluating1}
            error={error1}
            showSummary={showSummary1}
            setShowSummary={setShowSummary1}
            showGraph={showGraph1}
            setShowGraph={setShowGraph1}
            showEvaluation={showEvaluation1}
            setShowEvaluation={setShowEvaluation1}
            showCombinedText={showCombinedText1}
            setShowCombinedText={setShowCombinedText1}
            handleSubmit={(url, setter, loadingSetter, errorSetter) => handleSubmit(url, setter, loadingSetter, errorSetter)}
            handleEvaluate={(ref) => handleEvaluate(1, ref)}
            handleGenerateNotes={handleGenerateNotes}
            videoNumber={1}
          />
        </div>

        <div>
          <VideoSection
            title="Video 2"
            videoUrl={video2Url}
            setVideoUrl={setVideo2Url}
            referenceSummary={referenceSummary2}
            setReferenceSummary={setReferenceSummary2}
            result={result2}
            loading={loading2}
            evaluating={evaluating2}
            error={error2}
            showSummary={showSummary2}
            setShowSummary={setShowSummary2}
            showGraph={showGraph2}
            setShowGraph={setShowGraph2}
            showEvaluation={showEvaluation2}
            setShowEvaluation={setShowEvaluation2}
            showCombinedText={showCombinedText2}
            setShowCombinedText={setShowCombinedText2}
            handleSubmit={(url, setter, loadingSetter, errorSetter) => handleSubmit(url, setter, loadingSetter, errorSetter)}
            handleEvaluate={(ref) => handleEvaluate(2, ref)}
            handleGenerateNotes={handleGenerateNotes}
            videoNumber={2}
          />
        </div>
      </div>

      <div style={{ marginTop: "1rem" }}>
        <UnifiedSection />
      </div>
    </div>
  );
}

export default App;
