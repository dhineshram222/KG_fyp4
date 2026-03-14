// App.jsx
import React, { useState, useEffect } from "react";
import useAppStore from './store';

function App() {
    const {
        video1Url, setVideo1Url,
        video2Url, setVideo2Url,
        referenceSummary1, setReferenceSummary1,
        referenceSummary2, setReferenceSummary2,
        result1, setResult1,
        result2, setResult2,
        loading1, setLoading1,
        loading2, setLoading2,
        evaluating1, setEvaluating1,
        evaluating2, setEvaluating2,
        error1, setError1,
        error2, setError2,
        showSummary1, setShowSummary1,
        showSummary2, setShowSummary2,
        showGraph1, setShowGraph1,
        showGraph2, setShowGraph2,
        showCombinedText1, setShowCombinedText1,
        showCombinedText2, setShowCombinedText2,
        showEvaluation1, setShowEvaluation1,
        showEvaluation2, setShowEvaluation2,

        // Fused evaluation state
        fusedReferenceSummary, setFusedReferenceSummary,
        showFusedEvaluation, setShowFusedEvaluation,
        evaluatingFused, setEvaluatingFused,
        fusedEvaluationResult, setFusedEvaluationResult,

        // Unified fusion state
        fusedResult, setFusedResult,
        loadingFuse, setLoadingFuse,
        errorFuse, setErrorFuse,
        showUnifiedSummary, setShowUnifiedSummary,
        showUnifiedNotes, setShowUnifiedNotes,
        showUnifiedGraph, setShowUnifiedGraph,

        // Fused notes generation state
        generatingFusedNotes, setGeneratingFusedNotes,
        fusedNotesUrl, setFusedNotesUrl,

        // Fused Summary State
        generatingFusedSummary, setGeneratingFusedSummary,
        showFusedSummary, setShowFusedSummary,

        // Non-KG Unified State
        nonKgSummary, setNonKgSummary,
        generatingNonKgSummary, setGeneratingNonKgSummary,
        nonKgNotesUrl, setNonKgNotesUrl,
        generatingNonKgNotes, setGeneratingNonKgNotes,
        showNonKgSummary, setShowNonKgSummary,
        nonKgReferenceSummary, setNonKgReferenceSummary,
        nonKgEvaluationResult, setNonKgEvaluationResult,
        evaluatingNonKg, setEvaluatingNonKg,
        showNonKgEvaluation, setShowNonKgEvaluation,

        // Notes Evaluation State
        kgNotesRefText, setKgNotesRefText,
        kgNotesEval, setKgNotesEval,
        evaluatingKgNotes, setEvaluatingKgNotes,
        nonKgNotesRefText, setNonKgNotesRefText,
        nonKgNotesEval, setNonKgNotesEval,
        evaluatingNonKgNotes, setEvaluatingNonKgNotes,

        // Statistical Testing State
        statDataset, setStatDataset,
        statVideoName, setStatVideoName,
        statKgRouge, setStatKgRouge,
        statNonKgRouge, setStatNonKgRouge,
        statHumanKg, setStatHumanKg,
        statHumanNonKg, setStatHumanNonKg,
        statSaving, setStatSaving,
        statRunning, setStatRunning,
        statVideos, setStatVideos,
        statResults, setStatResults,
        statMessage, setStatMessage,
        datasetLoading, setDatasetLoading,
        datasetError, setDatasetError
    } = useAppStore();

    // Generic backend root; change if your backend runs on a different host/port
    const SERVER_ORIGIN = "http://127.0.0.1:8000";

    // Load statistical testing data once on mount
    const loadEvalStatus = async () => {
        try {
            const r = await fetch(`${SERVER_ORIGIN}/evaluation_status`);
            const d = await r.json();
            if (d.videos) setStatVideos(d.videos);
            if (d.statistical_results) setStatResults(d.statistical_results);
        } catch (e) { console.error(e); }
    };

    useEffect(() => {
        setDatasetLoading(true);
        setDatasetError(null);
        fetch(`${SERVER_ORIGIN}/list_dataset`)
            .then(r => {
                if (!r.ok) throw new Error("Dataset fetch failed");
                return r.json();
            })
            .then(d => {
                console.log("[list_dataset] Response:", d);
                if (d.dataset && Array.isArray(d.dataset)) {
                    console.log("[list_dataset] Setting", d.dataset.length, "items");
                    setStatDataset(d.dataset);
                } else {
                    console.warn("[list_dataset] No valid dataset key in response:", d);
                    setDatasetError("Invalid dataset response");
                }
            })
            .catch((err) => {
                console.error("[list_dataset] Fetch failed:", err);
                setDatasetError(err.message);
            })
            .finally(() => {
                setDatasetLoading(false);
            });
        loadEvalStatus();
    }, []);

    // When statVideoName changes, load any previously saved human eval scores
    useEffect(() => {
        if (statVideoName) {
            const existing = statVideos.find(v => v.video_name === statVideoName);
            if (existing && existing.human_evaluation) {
                setStatHumanKg(existing.human_evaluation.kg ?? "");
                setStatHumanNonKg(existing.human_evaluation.nonkg ?? "");
            } else {
                setStatHumanKg("");
                setStatHumanNonKg("");
            }
        } else {
            setStatHumanKg("");
            setStatHumanNonKg("");
        }
    }, [statVideoName, statVideos]);

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

        const safeGet = (obj, path, fallback = 0) => {
            try {
                return path.split('.').reduce((acc, k) => acc[k], obj) ?? fallback;
            } catch { return fallback; }
        };

        const colorFor = (val, max = 1) => {
            if (val == null) return "#888";
            const ratio = val / max;
            if (ratio >= 0.7) return "#28a745";
            if (ratio >= 0.4) return "#ffc107";
            return "#dc3545";
        };

        const pct = (v) => v != null ? (v * 100).toFixed(1) + "%" : "N/A";

        return (
            <div style={{ background: "#f8f9fa", padding: "1rem", borderRadius: 8, marginTop: "0.75rem", fontSize: "0.88rem", border: "1px solid #dee2e6" }}>
                <h4 style={{ marginTop: 0, marginBottom: "0.75rem" }}>📊 Evaluation Results</h4>
                <div style={{ marginBottom: "0.75rem", color: "#666", fontSize: "0.82rem" }}>
                    <strong>Note:</strong> Compares generated summary against reference.
                </div>
                <table style={{ width: "100%", borderCollapse: "collapse" }}>
                    <thead>
                        <tr style={{ borderBottom: "2px solid #ccc" }}>
                            <th style={{ textAlign: "left", padding: "6px" }}>Metric</th>
                            <th style={{ textAlign: "right", padding: "6px" }}>Score</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr style={{ borderBottom: "1px solid #eee" }}>
                            <td style={{ padding: "6px" }}>ROUGE-1 F1</td>
                            <td style={{ padding: "6px", textAlign: "right", fontWeight: "bold", color: colorFor(safeGet(bartEval, "rouge.rouge1.fmeasure")) }}>{pct(safeGet(bartEval, "rouge.rouge1.fmeasure"))}</td>
                        </tr>
                        <tr style={{ borderBottom: "1px solid #eee" }}>
                            <td style={{ padding: "6px" }}>ROUGE-2 F1</td>
                            <td style={{ padding: "6px", textAlign: "right", fontWeight: "bold", color: colorFor(safeGet(bartEval, "rouge.rouge2.fmeasure")) }}>{pct(safeGet(bartEval, "rouge.rouge2.fmeasure"))}</td>
                        </tr>
                        <tr style={{ borderBottom: "1px solid #eee" }}>
                            <td style={{ padding: "6px" }}>ROUGE-L F1</td>
                            <td style={{ padding: "6px", textAlign: "right", fontWeight: "bold", color: colorFor(safeGet(bartEval, "rouge.rougeL.fmeasure")) }}>{pct(safeGet(bartEval, "rouge.rougeL.fmeasure"))}</td>
                        </tr>
                        <tr>
                            <td style={{ padding: "6px" }}>BLEU</td>
                            <td style={{ padding: "6px", textAlign: "right", fontWeight: "bold", color: colorFor(safeGet(bartEval, "bleu")) }}>{pct(safeGet(bartEval, "bleu"))}</td>
                        </tr>
                    </tbody>
                </table>
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
                setShowFusedSummary(true);

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

        // ─── NOTES EVALUATION HANDLERS ───
        const handleEvaluateKgNotes = async () => {
            if (!fusedResult?.session_id) { alert("Generate notes first."); return; }
            setEvaluatingKgNotes(true);
            try {
                const response = await fetch(`${SERVER_ORIGIN}/evaluate_notes`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        session_id: fusedResult.session_id,
                        notes_type: "kg",
                        reference_notes: kgNotesRefText
                    })
                });
                if (!response.ok) {
                    const err = await response.json();
                    throw new Error(err.detail || "Evaluation failed");
                }
                const data = await response.json();
                setKgNotesEval(data.evaluation);
            } catch (err) {
                console.error(err);
                alert("Notes evaluation failed: " + err.message);
            }
            setEvaluatingKgNotes(false);
        };

        const handleEvaluateNonKgNotes = async () => {
            if (!fusedResult?.session_id) { alert("Generate notes first."); return; }
            setEvaluatingNonKgNotes(true);
            try {
                const response = await fetch(`${SERVER_ORIGIN}/evaluate_notes`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        session_id: fusedResult.session_id,
                        notes_type: "non_kg",
                        reference_notes: nonKgNotesRefText
                    })
                });
                if (!response.ok) {
                    const err = await response.json();
                    throw new Error(err.detail || "Evaluation failed");
                }
                const data = await response.json();
                setNonKgNotesEval(data.evaluation);
            } catch (err) {
                console.error(err);
                alert("Notes evaluation failed: " + err.message);
            }
            setEvaluatingNonKgNotes(false);
        };

        // ─── NOTES EVALUATION SCORE RENDERER ───
        const renderNotesEvaluation = (evaluation) => {
            if (!evaluation) return null;
            const colorFor = (val, max = 1) => {
                if (val == null) return "#888";
                const ratio = val / max;
                if (ratio >= 0.7) return "#28a745";
                if (ratio >= 0.4) return "#ffc107";
                return "#dc3545";
            };
            const pct = (v) => v != null ? (v * 100).toFixed(1) + "%" : "N/A";
            const rouge = evaluation.rouge || {};
            const fre = evaluation.flesch_reading_ease || {};
            const gfi = evaluation.gunning_fog || {};
            const dep = evaluation.dependency_score || {};
            const verb = evaluation.verb_fidelity || {};

            return (
                <div style={{ background: "#f8f9fa", padding: "1rem", borderRadius: 8, marginTop: "0.75rem", fontSize: "0.88rem", border: "1px solid #dee2e6" }}>
                    <h4 style={{ marginTop: 0, marginBottom: "0.75rem" }}>📊 Notes Evaluation</h4>
                    <table style={{ width: "100%", borderCollapse: "collapse" }}>
                        <thead>
                            <tr style={{ borderBottom: "2px solid #ccc" }}>
                                <th style={{ textAlign: "left", padding: "6px" }}>Metric</th>
                                <th style={{ textAlign: "right", padding: "6px" }}>Score</th>
                                <th style={{ textAlign: "left", padding: "6px" }}>Interpretation</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr style={{ borderBottom: "1px solid #eee" }}>
                                <td style={{ padding: "6px" }}>ROUGE-1 Recall</td>
                                <td style={{ padding: "6px", textAlign: "right", fontWeight: "bold", color: colorFor(rouge.rouge1_recall) }}>{pct(rouge.rouge1_recall)}</td>
                                <td style={{ padding: "6px", color: "#666" }}>Content coverage</td>
                            </tr>
                            <tr style={{ borderBottom: "1px solid #eee" }}>
                                <td style={{ padding: "6px" }}>ROUGE-L Recall</td>
                                <td style={{ padding: "6px", textAlign: "right", fontWeight: "bold", color: colorFor(rouge.rougeL_recall) }}>{pct(rouge.rougeL_recall)}</td>
                                <td style={{ padding: "6px", color: "#666" }}>Structural overlap</td>
                            </tr>
                            <tr style={{ borderBottom: "1px solid #eee" }}>
                                <td style={{ padding: "6px" }}>Flesch Reading Ease</td>
                                <td style={{ padding: "6px", textAlign: "right", fontWeight: "bold" }}>{fre.score ?? "N/A"}</td>
                                <td style={{ padding: "6px", color: "#666" }}>{fre.label || ""}</td>
                            </tr>
                            <tr style={{ borderBottom: "1px solid #eee" }}>
                                <td style={{ padding: "6px" }}>Gunning Fog Index</td>
                                <td style={{ padding: "6px", textAlign: "right", fontWeight: "bold" }}>{gfi.score ?? "N/A"}</td>
                                <td style={{ padding: "6px", color: "#666" }}>{gfi.label || ""}</td>
                            </tr>
                            <tr style={{ borderBottom: "1px solid #eee", background: "#fffbe6" }}>
                                <td style={{ padding: "6px", fontWeight: "bold" }}>📌 Dependency Score</td>
                                <td style={{ padding: "6px", textAlign: "right", fontWeight: "bold", color: colorFor(dep.score), fontSize: "1.05em" }}>{pct(dep.score)}</td>
                                <td style={{ padding: "6px", color: "#666" }}>{dep.violations != null ? `${dep.violations} violations / ${dep.evaluated_dependencies || dep.total_dependencies} deps` : "No KG data"}</td>
                            </tr>
                            <tr>
                                <td style={{ padding: "6px" }}>Verb Fidelity</td>
                                <td style={{ padding: "6px", textAlign: "right", fontWeight: "bold", color: colorFor(verb.fidelity_score) }}>{pct(verb.fidelity_score)}</td>
                                <td style={{ padding: "6px", color: "#666" }}>{verb.matched_count != null ? `${verb.matched_count}/${verb.total_verbs} KG verbs used` : ""}</td>
                            </tr>
                        </tbody>
                    </table>
                    {verb.missing_verbs && verb.missing_verbs.length > 0 && (
                        <div style={{ marginTop: "0.5rem", fontSize: "0.82rem", color: "#999" }}>
                            <strong>Missing verbs:</strong> {verb.missing_verbs.slice(0, 8).join(", ")}{verb.missing_verbs.length > 8 ? "..." : ""}
                        </div>
                    )}
                </div>
            );
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

            const colorFor = (val, max = 1) => {
                if (val == null) return "#888";
                const ratio = val / max;
                if (ratio >= 0.7) return "#28a745";
                if (ratio >= 0.4) return "#ffc107";
                return "#dc3545";
            };

            const pct = (v) => v != null ? (v * 100).toFixed(1) + "%" : "N/A";

            return (
                <div style={{ background: "#f8f9fa", padding: "1rem", borderRadius: 8, marginTop: "0.75rem", fontSize: "0.88rem", border: "1px solid #dee2e6" }}>
                    <h4 style={{ marginTop: 0, marginBottom: "0.75rem" }}>📊 {title}</h4>
                    <table style={{ width: "100%", borderCollapse: "collapse" }}>
                        <thead>
                            <tr style={{ borderBottom: "2px solid #ccc" }}>
                                <th style={{ textAlign: "left", padding: "6px" }}>Metric</th>
                                <th style={{ textAlign: "right", padding: "6px" }}>Score</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr style={{ borderBottom: "1px solid #eee" }}>
                                <td style={{ padding: "6px" }}>ROUGE-1 F1</td>
                                <td style={{ padding: "6px", textAlign: "right", fontWeight: "bold", color: colorFor(safeGet(evaluation, "rouge.rouge1.fmeasure")) }}>{pct(safeGet(evaluation, "rouge.rouge1.fmeasure"))}</td>
                            </tr>
                            <tr style={{ borderBottom: "1px solid #eee" }}>
                                <td style={{ padding: "6px" }}>ROUGE-L F1</td>
                                <td style={{ padding: "6px", textAlign: "right", fontWeight: "bold", color: colorFor(safeGet(evaluation, "rouge.rougeL.fmeasure")) }}>{pct(safeGet(evaluation, "rouge.rougeL.fmeasure"))}</td>
                            </tr>
                            <tr style={{ borderBottom: "1px solid #eee" }}>
                                <td style={{ padding: "6px" }}>Keyword Coverage</td>
                                <td style={{ padding: "6px", textAlign: "right", fontWeight: "bold", color: colorFor(safeGet(evaluation, "keyword_coverage")) }}>{pct(safeGet(evaluation, "keyword_coverage"))}</td>
                            </tr>
                            <tr style={{ borderBottom: "1px solid #eee" }}>
                                <td style={{ padding: "6px" }}>BERTScore F1</td>
                                <td style={{ padding: "6px", textAlign: "right", fontWeight: "bold", color: colorFor(safeGet(evaluation, "bertscore.f1")) }}>{pct(safeGet(evaluation, "bertscore.f1"))}</td>
                            </tr>
                            <tr>
                                <td style={{ padding: "6px" }}>Sentence Cosine</td>
                                <td style={{ padding: "6px", textAlign: "right", fontWeight: "bold", color: colorFor(safeGet(evaluation, "sentence_cosine")) }}>{pct(safeGet(evaluation, "sentence_cosine"))}</td>
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

                                    {/* KG Notes Evaluation */}
                                    {fusedNotesUrl && (
                                        <div style={{ marginTop: "1rem", paddingTop: "0.75rem", borderTop: "1px dashed #ccc" }}>
                                            <h5 style={{ marginBottom: "0.5rem" }}>Evaluate KG Notes</h5>
                                            <textarea
                                                value={kgNotesRefText}
                                                onChange={(e) => setKgNotesRefText(e.target.value)}
                                                placeholder="Paste reference/ground truth notes (optional — will use ground truth folder if empty)..."
                                                rows="3"
                                                style={{ width: "100%", padding: "0.5rem", marginBottom: "0.5rem", fontSize: "0.85rem" }}
                                            />
                                            <div style={{ display: "flex", gap: "0.5rem" }}>
                                                <button
                                                    onClick={handleEvaluateKgNotes}
                                                    disabled={evaluatingKgNotes}
                                                    className="button primary"
                                                    style={{ backgroundColor: "#007bff", fontSize: "0.85rem", padding: "0.3rem 0.8rem" }}
                                                >
                                                    {evaluatingKgNotes ? "Evaluating..." : "Evaluate KG Notes"}
                                                </button>
                                                {kgNotesEval && (
                                                    <button
                                                        onClick={() => { setKgNotesEval(null); setKgNotesRefText(""); }}
                                                        className="button secondary"
                                                        style={{ fontSize: "0.85rem", padding: "0.3rem 0.8rem" }}
                                                    >
                                                        Clear
                                                    </button>
                                                )}
                                            </div>
                                            {kgNotesEval && renderNotesEvaluation(kgNotesEval)}
                                        </div>
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

                                    {/* Non-KG Notes Evaluation */}
                                    {nonKgNotesUrl && (
                                        <div style={{ marginTop: "1rem", paddingTop: "0.75rem", borderTop: "1px dashed #ccc" }}>
                                            <h5 style={{ marginBottom: "0.5rem" }}>Evaluate Text Notes</h5>
                                            <textarea
                                                value={nonKgNotesRefText}
                                                onChange={(e) => setNonKgNotesRefText(e.target.value)}
                                                placeholder="Paste reference/ground truth notes (optional — will use ground truth folder if empty)..."
                                                rows="3"
                                                style={{ width: "100%", padding: "0.5rem", marginBottom: "0.5rem", fontSize: "0.85rem" }}
                                            />
                                            <div style={{ display: "flex", gap: "0.5rem" }}>
                                                <button
                                                    onClick={handleEvaluateNonKgNotes}
                                                    disabled={evaluatingNonKgNotes}
                                                    className="button primary"
                                                    style={{ backgroundColor: "#28a745", fontSize: "0.85rem", padding: "0.3rem 0.8rem" }}
                                                >
                                                    {evaluatingNonKgNotes ? "Evaluating..." : "Evaluate Text Notes"}
                                                </button>
                                                {nonKgNotesEval && (
                                                    <button
                                                        onClick={() => { setNonKgNotesEval(null); setNonKgNotesRefText(""); }}
                                                        className="button secondary"
                                                        style={{ fontSize: "0.85rem", padding: "0.3rem 0.8rem" }}
                                                    >
                                                        Clear
                                                    </button>
                                                )}
                                            </div>
                                            {nonKgNotesEval && renderNotesEvaluation(nonKgNotesEval)}
                                        </div>
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

    // ──────────── Statistical Testing Section ────────────
    const renderStatisticalTestingSection = () => {

        const handleSaveEval = async (selectedVideoName) => {
            if (statSaving) return;
            const vName = selectedVideoName || statVideoName;
            if (!vName) {
                setStatMessage("⚠️ Please select a video name first.");
                return;
            }

            const currentSessionId = fusedResult?.session_id;
            if (!currentSessionId) {
                setStatMessage("⚠️ Process and fuse videos first to get a session ID.");
                return;
            }

            setStatSaving(true);
            setStatMessage("");
            try {
                const body = { video_name: vName, session_id: currentSessionId };

                // Use the current input values if we're triggered by the button,
                // or whatever is in state (which might be the loaded values if triggered by select)
                if (statHumanKg) body.human_kg = parseFloat(statHumanKg);
                if (statHumanNonKg) body.human_nonkg = parseFloat(statHumanNonKg);

                const r = await fetch(`${SERVER_ORIGIN}/save_evaluation`, {
                    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
                });
                const d = await r.json();
                if (d.status === "success") {
                    const kg_r = d.data?.metrics?.kg?.rouge1;
                    const nkg_r = d.data?.metrics?.nonkg?.rouge1;
                    if (kg_r == null && nkg_r == null) {
                        setStatMessage(`⚠️ Saved: ${vName}, but no ROUGE-1 scores found! Did you generate & evaluate notes first?`);
                    } else {
                        setStatMessage(`✅ Saved: ${vName} (KG=${kg_r ?? "—"}, NonKG=${nkg_r ?? "—"})`);
                    }
                    await loadEvalStatus();
                } else { setStatMessage("❌ " + (d.detail || "Save failed")); }
            } catch (e) { setStatMessage("❌ " + e.message); }
            setStatSaving(false);
        };

        const handleRunStats = async () => {
            if (statRunning) return;
            setStatRunning(true); setStatMessage("");
            try {
                const r = await fetch(`${SERVER_ORIGIN}/run_statistics`, { method: "POST" });
                const d = await r.json();
                if (d.status === "success") { setStatResults(d.results); setStatMessage("✅ Statistical tests completed."); }
                else if (d.status === "insufficient_data") { setStatMessage("⚠️ " + d.message); }
                else { setStatMessage("❌ " + (d.detail || "Failed")); }
            } catch (e) { setStatMessage("❌ " + e.message); }
            setStatRunning(false);
        };

        const sigColor = (sig) => sig === true ? "#28a745" : sig === false ? "#dc3545" : "#888";
        const sigLabel = (sig) => sig === true ? "✅ Significant" : sig === false ? "❌ Not significant" : "—";

        return (
            <div style={{ border: "2px solid #6f42c1", padding: "1.5rem", borderRadius: 10, marginTop: "2rem", background: "#faf8ff" }}>
                <h2 style={{ color: "#6f42c1", marginTop: 0 }}>📈 Statistical Testing (Notes Evaluation)</h2>

                {/* ── Save Evaluation Form ── */}
                <div style={{ background: "#fff", padding: "1rem", borderRadius: 8, border: "1px solid #dee2e6", marginBottom: "1.5rem" }}>
                    <h4 style={{ marginTop: 0, marginBottom: "0.75rem" }}>Save Video Evaluation</h4>
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.75rem", marginBottom: "0.75rem" }}>
                        <div>
                            <label style={{ fontSize: "0.82rem", fontWeight: "bold", display: "block", marginBottom: 4 }}>Video Name</label>
                            <select value={statVideoName} onChange={e => setStatVideoName(e.target.value)} disabled={datasetLoading || datasetError} style={{ width: "100%", padding: "0.4rem", fontSize: "0.9rem", borderRadius: 4, border: "1px solid #ccc" }}>
                                <option value="">{datasetLoading ? "— Loading dataset... —" : datasetError ? "— Error loading dataset —" : "— Select from dataset —"}</option>
                                {statDataset.map(v => (<option key={v.file_key} value={v.display_name}>{v.display_name}</option>))}
                            </select>
                            {datasetError && <div style={{ color: "crimson", fontSize: "0.8rem", marginTop: "2px" }}>Failed to load. Is backend running?</div>}
                        </div>
                        <div style={{ display: "flex", alignItems: "flex-end" }}>
                            <span style={{ fontSize: "0.82rem", color: "#666" }}>Session: <strong>{fusedResult?.session_id ? fusedResult.session_id.slice(0, 8) + "..." : "—"}</strong><br />ROUGE-1 auto-pulled from evaluated notes</span>
                        </div>
                        <div>
                            <label style={{ fontSize: "0.82rem", fontWeight: "bold", display: "block", marginBottom: 4 }}>Human Eval — KG Quiz Score</label>
                            <input type="number" step="0.1" min="0" max="10" value={statHumanKg} onChange={e => setStatHumanKg(e.target.value)} placeholder="e.g. 7.5 (optional)" style={{ width: "100%", padding: "0.4rem", fontSize: "0.9rem", borderRadius: 4, border: "1px solid #ccc" }} />
                        </div>
                        <div>
                            <label style={{ fontSize: "0.82rem", fontWeight: "bold", display: "block", marginBottom: 4 }}>Human Eval — Non-KG Quiz Score</label>
                            <input type="number" step="0.1" min="0" max="10" value={statHumanNonKg} onChange={e => setStatHumanNonKg(e.target.value)} placeholder="e.g. 6.0 (optional)" style={{ width: "100%", padding: "0.4rem", fontSize: "0.9rem", borderRadius: 4, border: "1px solid #ccc" }} />
                        </div>
                    </div>
                    <button onClick={() => handleSaveEval()} disabled={statSaving} className="button primary" style={{ backgroundColor: "#6f42c1" }}>{statSaving ? "Saving..." : "Save Evaluation"}</button>
                    {statMessage && <span style={{ marginLeft: "1rem", fontSize: "0.9rem" }}>{statMessage}</span>}
                </div>

                {/* ── Evaluated Videos Table ── */}
                {statVideos.length > 0 && (
                    <div style={{ background: "#fff", padding: "1rem", borderRadius: 8, border: "1px solid #dee2e6", marginBottom: "1.5rem" }}>
                        <h4 style={{ marginTop: 0 }}>Evaluated Videos ({statVideos.length}/21)</h4>
                        <div style={{ maxHeight: 250, overflowY: "auto" }}>
                            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.85rem" }}>
                                <thead>
                                    <tr style={{ borderBottom: "2px solid #ccc", background: "#f1f3f5" }}>
                                        <th style={{ padding: "6px", textAlign: "left" }}>#</th>
                                        <th style={{ padding: "6px", textAlign: "left" }}>Video</th>
                                        <th style={{ padding: "6px", textAlign: "right" }}>KG ROUGE-1</th>
                                        <th style={{ padding: "6px", textAlign: "right" }}>Non-KG ROUGE-1</th>
                                        <th style={{ padding: "6px", textAlign: "right" }}>Human KG</th>
                                        <th style={{ padding: "6px", textAlign: "right" }}>Human Non-KG</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {statVideos.map((v, i) => (
                                        <tr key={v.file_key} style={{ borderBottom: "1px solid #eee" }}>
                                            <td style={{ padding: "6px" }}>{i + 1}</td>
                                            <td style={{ padding: "6px" }}>{v.video_name}</td>
                                            <td style={{ padding: "6px", textAlign: "right", fontWeight: "bold", color: "#007bff" }}>{v.metrics?.kg?.rouge1?.toFixed(4) ?? "—"}</td>
                                            <td style={{ padding: "6px", textAlign: "right", fontWeight: "bold", color: "#28a745" }}>{v.metrics?.nonkg?.rouge1?.toFixed(4) ?? "—"}</td>
                                            <td style={{ padding: "6px", textAlign: "right" }}>{v.human_evaluation?.kg ?? "—"}</td>
                                            <td style={{ padding: "6px", textAlign: "right" }}>{v.human_evaluation?.nonkg ?? "—"}</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                )}

                {/* ── Run Statistics ── */}
                <div style={{ display: "flex", gap: "1rem", alignItems: "center", marginBottom: "1.5rem" }}>
                    <button onClick={handleRunStats} disabled={statRunning} className="button primary" style={{ backgroundColor: "#6f42c1", fontSize: "1rem", padding: "0.5rem 1.5rem" }}>{statRunning ? "Running Tests..." : "🧪 Run Statistical Tests"}</button>
                </div>

                {/* ── Statistical Results ── */}
                {statResults && statResults.results && (
                    <div style={{ background: "#fff", padding: "1rem", borderRadius: 8, border: "1px solid #dee2e6" }}>
                        <h4 style={{ marginTop: 0 }}>📊 Notes ROUGE-1 — Statistical Results (N={statResults.dataset_size}, α=0.05)</h4>
                        {Object.entries(statResults.results).map(([metricName, data]) => (
                            <div key={metricName} style={{ marginBottom: "1.5rem", padding: "1rem", background: "#f8f9fa", borderRadius: 6, border: "1px solid #e9ecef" }}>
                                {data.error ? (<p style={{ color: "#dc3545" }}>{data.error}</p>) : (<>
                                    {data.verdict && (
                                        <div style={{ background: data.mean_difference > 0 ? "#d4edda" : "#f8d7da", border: `1px solid ${data.mean_difference > 0 ? "#c3e6cb" : "#f5c6cb"}`, padding: "0.75rem 1rem", borderRadius: 6, marginBottom: "1rem", fontSize: "1.05rem", fontWeight: "bold", color: data.mean_difference > 0 ? "#155724" : "#721c24" }}>
                                            🏆 {data.verdict} (Δ = {data.mean_difference > 0 ? "+" : ""}{data.mean_difference?.toFixed(4)})
                                        </div>
                                    )}
                                    <div style={{ display: "flex", gap: "2rem", marginBottom: "0.75rem", flexWrap: "wrap" }}>
                                        <div><strong>KG Mean:</strong> {data.kg_mean?.toFixed(4)} ± {data.kg_std?.toFixed(4)}</div>
                                        <div><strong>Non-KG Mean:</strong> {data.nonkg_mean?.toFixed(4)} ± {data.nonkg_std?.toFixed(4)}</div>
                                        <div><strong>N pairs:</strong> {data.n_pairs}</div>
                                    </div>
                                    <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.85rem" }}>
                                        <thead><tr style={{ borderBottom: "2px solid #ccc" }}><th style={{ textAlign: "left", padding: "6px" }}>Test</th><th style={{ textAlign: "right", padding: "6px" }}>Statistic</th><th style={{ textAlign: "right", padding: "6px" }}>p-value</th><th style={{ textAlign: "center", padding: "6px" }}>Result</th></tr></thead>
                                        <tbody>
                                            <tr style={{ borderBottom: "1px solid #eee", background: "#fffbe6" }}><td style={{ padding: "6px" }}>Shapiro-Wilk (normality)</td><td style={{ padding: "6px", textAlign: "right" }}>{data.shapiro?.W?.toFixed(4) ?? "—"}</td><td style={{ padding: "6px", textAlign: "right" }}>{data.shapiro?.p_value?.toFixed(6) ?? "—"}</td><td style={{ padding: "6px", textAlign: "center" }}>{data.shapiro?.is_normal === true ? "Normal ✅" : data.shapiro?.is_normal === false ? "Not normal ⚠️" : data.shapiro?.note || "—"}</td></tr>
                                            <tr style={{ borderBottom: "1px solid #eee" }}><td style={{ padding: "6px" }}>Paired t-test</td><td style={{ padding: "6px", textAlign: "right" }}>{data.paired_t?.t_stat?.toFixed(4) ?? "—"}</td><td style={{ padding: "6px", textAlign: "right", fontWeight: "bold" }}>{data.paired_t?.p_value?.toFixed(6) ?? "—"}</td><td style={{ padding: "6px", textAlign: "center", color: sigColor(data.paired_t?.significant) }}>{data.paired_t?.note || sigLabel(data.paired_t?.significant)}</td></tr>
                                            <tr style={{ borderBottom: "1px solid #eee" }}><td style={{ padding: "6px" }}>Wilcoxon signed-rank</td><td style={{ padding: "6px", textAlign: "right" }}>{data.wilcoxon?.w_stat?.toFixed(4) ?? "—"}</td><td style={{ padding: "6px", textAlign: "right", fontWeight: "bold" }}>{data.wilcoxon?.p_value?.toFixed(6) ?? "—"}</td><td style={{ padding: "6px", textAlign: "center", color: sigColor(data.wilcoxon?.significant) }}>{data.wilcoxon?.note || sigLabel(data.wilcoxon?.significant)}</td></tr>
                                            <tr><td style={{ padding: "6px" }}>Bootstrap (10k, 95% CI)</td><td style={{ padding: "6px", textAlign: "right" }}>{data.bootstrap?.mean_diff?.toFixed(4) ?? "—"}</td><td style={{ padding: "6px", textAlign: "right" }}>[{data.bootstrap?.ci_95?.[0]?.toFixed(4)}, {data.bootstrap?.ci_95?.[1]?.toFixed(4)}]</td><td style={{ padding: "6px", textAlign: "center", color: sigColor(data.bootstrap?.significant) }}>{sigLabel(data.bootstrap?.significant)}</td></tr>
                                        </tbody>
                                    </table>
                                </>)}
                            </div>
                        ))}

                        {/* Human Evaluation Display (no statistical testing) */}
                        {statResults.human_evaluation && (
                            <div style={{ padding: "1rem", background: "#fff3cd", borderRadius: 6, border: "1px solid #ffc107", marginTop: "1rem" }}>
                                <h5 style={{ marginTop: 0, color: "#856404" }}>👤 Human Evaluation (Quiz Scores — Display Only)</h5>
                                <div style={{ display: "flex", gap: "2rem" }}>
                                    <div><strong>KG Notes Avg:</strong> {statResults.human_evaluation.kg_avg ?? "—"}</div>
                                    <div><strong>Non-KG Notes Avg:</strong> {statResults.human_evaluation.nonkg_avg ?? "—"}</div>
                                    <div><strong>N responses:</strong> {statResults.human_evaluation.n_responses ?? 0}</div>
                                </div>
                            </div>
                        )}
                    </div>
                )}
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
                {UnifiedSection()}
            </div>

            {renderStatisticalTestingSection()}
        </div>
    );
}

export default App;
