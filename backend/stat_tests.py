# stat_tests.py
"""
Statistical testing for KG vs Non-KG notes evaluation.

Only tests ROUGE-1 (notes evaluation). Human evaluation is display-only.

Tests:
  1. Shapiro-Wilk normality test (on differences)
  2. Paired t-test (mean difference)
  3. Wilcoxon signed-rank (median difference)
  4. Bootstrap resampling (95% CI of mean difference)
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


def shapiro_normality(kg: List[float], nonkg: List[float]) -> Dict:
    """Test if paired differences are normally distributed."""
    from scipy.stats import shapiro
    diffs = np.array(kg) - np.array(nonkg)
    if len(diffs) < 3:
        return {"W": None, "p_value": None, "is_normal": None, "note": "Need ≥3 samples"}
    W, p = shapiro(diffs)
    return {"W": round(float(W), 4), "p_value": round(float(p), 6), "is_normal": bool(p >= 0.05)}


def paired_t_test(kg: List[float], nonkg: List[float]) -> Dict:
    """Paired t-test: H0 = mean difference is zero."""
    from scipy.stats import ttest_rel
    if len(kg) < 2:
        return {"t_stat": None, "p_value": None, "significant": None, "note": "Need ≥2 samples"}
    t_stat, p_value = ttest_rel(kg, nonkg)
    return {"t_stat": round(float(t_stat), 4), "p_value": round(float(p_value), 6), "significant": bool(p_value < 0.05)}


def wilcoxon_test(kg: List[float], nonkg: List[float]) -> Dict:
    """Wilcoxon signed-rank: non-parametric alternative."""
    from scipy.stats import wilcoxon
    diffs = np.array(kg) - np.array(nonkg)
    nonzero = diffs[diffs != 0]
    if len(nonzero) < 6:
        return {"w_stat": None, "p_value": None, "significant": None, "note": "Need ≥6 non-zero differences"}
    try:
        w_stat, p_value = wilcoxon(kg, nonkg)
        return {"w_stat": round(float(w_stat), 4), "p_value": round(float(p_value), 6), "significant": bool(p_value < 0.05)}
    except Exception as e:
        return {"w_stat": None, "p_value": None, "significant": None, "note": str(e)}


def bootstrap_ci(kg: List[float], nonkg: List[float], n: int = 10000, seed: int = 42) -> Dict:
    """Bootstrap resampling: estimate 95% CI of mean difference."""
    rng = np.random.RandomState(seed)
    kg_arr, nonkg_arr = np.array(kg), np.array(nonkg)
    if len(kg_arr) < 2:
        return {"mean_diff": None, "ci_95": None, "significant": None, "note": "Need ≥2 samples"}
    diffs = [float(np.mean(kg_arr[idx := rng.choice(len(kg_arr), len(kg_arr), replace=True)] - nonkg_arr[idx])) for _ in range(n)]
    lower, upper = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    return {
        "mean_diff": round(float(np.mean(diffs)), 6),
        "ci_95": [round(lower, 6), round(upper, 6)],
        "significant": bool(lower > 0 or upper < 0),
    }


def run_all_tests(kg: List[float], nonkg: List[float]) -> Dict:
    """Run all 4 statistical tests on ROUGE-1 paired arrays."""
    kg_clean = [k for k, n in zip(kg, nonkg) if k is not None and n is not None]
    nonkg_clean = [n for k, n in zip(kg, nonkg) if k is not None and n is not None]
    
    if len(kg_clean) < 2:
        return {"error": f"Only {len(kg_clean)} valid pairs, need ≥2"}
    
    kg_mean = float(np.mean(kg_clean))
    nonkg_mean = float(np.mean(nonkg_clean))
    diff = kg_mean - nonkg_mean
    
    # Determine verdict
    if diff > 0:
        verdict = "KG Notes are better"
    elif diff < 0:
        verdict = "Non-KG Notes are better"
    else:
        verdict = "No difference"
    
    return {
        "n_pairs": len(kg_clean),
        "kg_mean": round(kg_mean, 4),
        "nonkg_mean": round(nonkg_mean, 4),
        "kg_std": round(float(np.std(kg_clean, ddof=1)), 4) if len(kg_clean) > 1 else 0,
        "nonkg_std": round(float(np.std(nonkg_clean, ddof=1)), 4) if len(nonkg_clean) > 1 else 0,
        "mean_difference": round(diff, 4),
        "verdict": verdict,
        "shapiro": shapiro_normality(kg_clean, nonkg_clean),
        "paired_t": paired_t_test(kg_clean, nonkg_clean),
        "wilcoxon": wilcoxon_test(kg_clean, nonkg_clean),
        "bootstrap": bootstrap_ci(kg_clean, nonkg_clean),
    }


def run_statistical_analysis(aggregated_path: Optional[str] = None) -> Dict:
    """
    Full pipeline: load aggregated data, run statistical tests on ROUGE-1 only.
    Human evaluation is returned as-is (no statistical testing).
    """
    BASE_DIR = Path(__file__).parent.resolve()
    EVAL_DIR = BASE_DIR / "outputs" / "evaluation"
    
    if aggregated_path:
        agg_path = Path(aggregated_path)
    else:
        agg_path = EVAL_DIR / "aggregated_results.json"
    
    if not agg_path.exists():
        from evaluation_store import aggregate_evaluations
        aggregate_evaluations()
    
    if not agg_path.exists():
        return {"error": "No aggregated results found. Evaluate some videos first."}
    
    agg_data = json.loads(agg_path.read_text(encoding="utf-8"))
    dataset_size = agg_data.get("dataset_size", 0)
    
    if dataset_size < 2:
        return {"error": f"Only {dataset_size} videos evaluated. Need ≥2 for statistics."}
    
    # Run tests only on ROUGE-1 (notes evaluation)
    metrics = agg_data.get("metrics", {})
    rouge1_data = metrics.get("rouge1", {})
    kg = rouge1_data.get("kg", [])
    nonkg = rouge1_data.get("nonkg", [])
    
    stat_results = {}
    if kg and nonkg:
        stat_results["rouge1"] = run_all_tests(kg, nonkg)
    
    # Human evaluation — return as-is, no statistical testing
    human = agg_data.get("human_evaluation", {})
    human_kg_scores = [s for s in human.get("kg", []) if s is not None]
    human_nonkg_scores = [s for s in human.get("nonkg", []) if s is not None]
    
    human_summary = None
    if human_kg_scores or human_nonkg_scores:
        human_summary = {
            "kg_avg": round(float(np.mean(human_kg_scores)), 2) if human_kg_scores else None,
            "nonkg_avg": round(float(np.mean(human_nonkg_scores)), 2) if human_nonkg_scores else None,
            "kg_scores": human_kg_scores,
            "nonkg_scores": human_nonkg_scores,
            "n_responses": max(len(human_kg_scores), len(human_nonkg_scores)),
        }
    
    results = {
        "dataset_size": dataset_size,
        "video_names": agg_data.get("video_names", []),
        "alpha": 0.05,
        "results": stat_results,
        "human_evaluation": human_summary,
    }
    
    out_path = EVAL_DIR / "statistical_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[StatTests] Results saved to {out_path}")
    return results
