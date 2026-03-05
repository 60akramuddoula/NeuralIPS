"""
GRASP NeurIPS — Evaluation Utilities
======================================
Implements all metrics required for NeurIPS submission:
  - ASR (Attack Success Rate) with bootstrap 95% CI
  - McNemar's test for pairwise ASR comparison
  - Retrieval F1 (exact match to PoisonedRAG protocol)
  - Naturalness score (multi-signal, no LM required)
  - ROC AUC for perplexity-based detection evaluation
  - ExperimentMetrics dataclass with full statistical fields
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# String utilities (exact match to PoisonedRAG paper)
# ─────────────────────────────────────────────────────────────────────────────

def clean_str(s: str) -> str:
    """Exact replica of PoisonedRAG src/utils.py clean_str."""
    try:
        s = str(s)
    except Exception:
        return ""
    s = s.strip()
    if len(s) > 1 and s[-1] == ".":
        s = s[:-1]
    return s.lower()


def substring_match(target_answer: str, generated_answer: str) -> bool:
    """ASR metric: target answer is substring of generated answer."""
    return clean_str(target_answer) in clean_str(generated_answer)


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval metrics
# ─────────────────────────────────────────────────────────────────────────────

def retrieval_metrics(
    n_adv_in_topk: int,
    top_k: int,
    n_adv_total: int,
) -> Tuple[float, float, float]:
    """
    Returns (precision, recall, f1) matching PoisonedRAG evaluation protocol.
    """
    precision = n_adv_in_topk / top_k if top_k > 0 else 0.0
    recall = n_adv_in_topk / n_adv_total if n_adv_total > 0 else 0.0
    denom = precision + recall
    f1 = (2 * precision * recall / denom) if denom > 0 else 0.0
    return precision, recall, f1


# ─────────────────────────────────────────────────────────────────────────────
# Naturalness Score (multi-signal, no LM required)
# ─────────────────────────────────────────────────────────────────────────────

def _char_bigram_entropy(text: str) -> float:
    """Shannon entropy of character bigrams. Natural English ≈ 3.5–4.5 bits."""
    if len(text) < 2:
        return 0.0
    ngrams = [text[i:i + 2] for i in range(len(text) - 1)]
    counts = Counter(ngrams)
    total = len(ngrams)
    return float(-sum((c / total) * math.log2(c / total) for c in counts.values()))


def naturalness_score(text: str) -> float:
    """
    Multi-signal text naturalness proxy. Returns anomaly score:
      HIGHER = less natural (more adversarial).

    Signals:
      1. ## token ratio   (HotFlip artifacts)
      2. Non-alpha ratio  (random character noise)
      3. Token repetition CV (adversarial word repetition)
      4. Sentence length CV (irregular sentence lengths)
      5. Bigram entropy anomaly (unnaturally low or high entropy)

    All signals ∈ [0, 1]; final score is their average.
    """
    if not text.strip():
        return 0.0
    tokens = text.lower().split()
    if not tokens:
        return 0.0

    signals = []

    # 1. HotFlip ## artifacts
    hash_ratio = sum(1 for t in tokens if t.startswith("##")) / len(tokens)
    signals.append(min(hash_ratio * 5.0, 1.0))

    # 2. Non-alpha character ratio
    non_alpha = sum(
        sum(1 for c in t if not c.isalpha()) / max(len(t), 1)
        for t in tokens
    ) / len(tokens)
    signals.append(min(non_alpha * 2.0, 1.0))

    # 3. Token repetition coefficient of variation
    freq = Counter(tokens)
    counts_list = list(freq.values())
    if len(counts_list) > 1:
        mean_c = sum(counts_list) / len(counts_list)
        std_c = (sum((x - mean_c) ** 2 for x in counts_list) / len(counts_list)) ** 0.5
        cv = std_c / (mean_c + 1e-9)
        signals.append(min(cv * 0.3, 1.0))
    else:
        signals.append(0.0)

    # 4. Sentence length coefficient of variation
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    if len(sents) > 1:
        lens = [len(s.split()) for s in sents]
        m = sum(lens) / len(lens)
        s_std = (sum((x - m) ** 2 for x in lens) / len(lens)) ** 0.5
        signals.append(min(s_std / max(m, 1) * 0.5, 1.0))
    else:
        signals.append(0.0)

    # 5. Bigram entropy anomaly
    # Natural English prose ≈ 3.8 bits; abnormally low or high = adversarial
    entropy = _char_bigram_entropy(text)
    entropy_anomaly = max(0.0, (3.8 - entropy) / 3.8) if entropy < 3.8 else 0.0
    signals.append(min(entropy_anomaly * 0.5, 1.0))

    return float(sum(signals) / len(signals))


# Alias used by existing code
def pseudo_perplexity(
    text: str,
    reference_corpus: Optional[List[str]] = None,
    n: int = 2,
) -> float:
    """
    Backward-compatible wrapper around naturalness_score.
    reference_corpus parameter accepted but ignored (naturalness_score
    is reference-free by design, avoiding corpus calibration bias).
    """
    return naturalness_score(text)


# ─────────────────────────────────────────────────────────────────────────────
# ROC AUC for detection experiment
# ─────────────────────────────────────────────────────────────────────────────

def compute_roc_auc(
    clean_scores: List[float],
    adv_scores: List[float],
) -> Tuple[List[float], List[float], float]:
    """
    Compute ROC curve (FPR, TPR) and AUC for naturalness-based detection.
    Label: 1 = adversarial, 0 = clean.
    Adversarial texts should have HIGHER naturalness_score → higher AUC means
    more detectable (detector label = 1 when score > threshold).
    """
    labels = [0] * len(clean_scores) + [1] * len(adv_scores)
    scores = list(clean_scores) + list(adv_scores)

    sorted_pairs = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return [0.0, 1.0], [0.0, 1.0], 0.5

    tpr_list: List[float] = [0.0]
    fpr_list: List[float] = [0.0]
    tp = fp = 0

    for score, label in sorted_pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)

    tpr_list.append(1.0)
    fpr_list.append(1.0)

    _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
    auc = float(_trapz(tpr_list, fpr_list))
    if auc < 0:
        auc = -auc
    auc = min(max(auc, 0.0), 1.0)

    return fpr_list, tpr_list, auc


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap CI for ASR
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_asr_ci(
    success_list: List[int],
    n_resamples: int = 999,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Bootstrap 95% CI for ASR.

    Args:
        success_list: binary list (1=success, 0=failure), one per query
        n_resamples:  bootstrap iterations
        confidence_level: default 0.95

    Returns:
        (ci_low, ci_high)
    """
    arr = np.array(success_list, dtype=float)
    if len(arr) == 0:
        return 0.0, 0.0
    result = stats.bootstrap(
        (arr,),
        np.mean,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        random_state=seed,
        method="percentile",
    )
    return float(result.confidence_interval.low), float(result.confidence_interval.high)


# ─────────────────────────────────────────────────────────────────────────────
# McNemar's Test for paired ASR comparison
# ─────────────────────────────────────────────────────────────────────────────

def mcnemar_test(
    method_a_success: List[int],
    method_b_success: List[int],
) -> Dict:
    """
    Paired McNemar's test comparing two methods on the same set of queries.

    Returns dict with:
      statistic, p_value, significant (p<0.05),
      both_success, a_only, b_only, both_fail,
      n_discordant, exact_test
    """
    if len(method_a_success) != len(method_b_success):
        raise ValueError("Both success lists must have equal length.")
    if len(method_a_success) == 0:
        return {"statistic": 0.0, "p_value": 1.0, "significant": False,
                "both_success": 0, "a_only": 0, "b_only": 0, "both_fail": 0,
                "n_discordant": 0, "exact_test": True}

    both_s = sum(a == 1 and b == 1 for a, b in zip(method_a_success, method_b_success))
    a_only  = sum(a == 1 and b == 0 for a, b in zip(method_a_success, method_b_success))
    b_only  = sum(a == 0 and b == 1 for a, b in zip(method_a_success, method_b_success))
    both_f  = sum(a == 0 and b == 0 for a, b in zip(method_a_success, method_b_success))

    table = np.array([[both_s, a_only], [b_only, both_f]])
    n_discordant = a_only + b_only
    exact = n_discordant < 25
    res = mcnemar(table, exact=exact, correction=True)

    return {
        "statistic": float(res.statistic),
        "p_value": float(res.pvalue),
        "significant": bool(res.pvalue < 0.05),
        "both_success": both_s,
        "a_only": a_only,
        "b_only": b_only,
        "both_fail": both_f,
        "n_discordant": n_discordant,
        "exact_test": exact,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExperimentMetrics:
    """Aggregate metrics for one experimental condition (NeurIPS-complete)."""
    dataset: str
    retriever: str
    llm: str
    attack_method: str
    n_queries: int
    n_attackable: int

    # ASR + confidence interval
    asr: float = 0.0
    asr_count: int = 0
    asr_ci_low: float = 0.0
    asr_ci_high: float = 0.0

    # McNemar vs baseline (populated by comparison function)
    mcnemar_p: float = 1.0
    mcnemar_significant: bool = False

    # Retrieval
    precision_mean: float = 0.0
    recall_mean: float = 0.0
    f1_mean: float = 0.0

    # Similarity
    sim_mean_evolved: float = 0.0
    sim_mean_baseline: float = 0.0
    sim_delta: float = 0.0

    # Efficiency
    min_n_for_90pct_asr: Optional[int] = None

    # Stealth / Naturalness
    naturalness_mean_clean: float = 0.0
    naturalness_mean_adv: float = 0.0
    naturalness_delta: float = 0.0   # negative = stealthier than clean
    detection_auc: float = 0.0       # AUROC under naturalness detector

    # Timing
    mean_time_per_query: float = 0.0

    # Per-query success vectors (for McNemar, CI computation)
    success_vector: List[int] = field(default_factory=list)
    per_query: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d.pop("per_query", None)
        d.pop("success_vector", None)
        return d

    def summary_str(self) -> str:
        sig = "✓" if self.mcnemar_significant else "·"
        return (
            f"[{self.attack_method}] {self.dataset}/{self.retriever}/{self.llm}\n"
            f"  ASR:  {self.asr_count}/{self.n_attackable} = {self.asr:.3f}"
            f"  95%CI [{self.asr_ci_low:.3f},{self.asr_ci_high:.3f}]\n"
            f"  F1:   {self.f1_mean:.4f}   SimΔ: {self.sim_delta:+.4f}\n"
            f"  Stealth AUC: {self.detection_auc:.3f}  NatΔ: {self.naturalness_delta:+.4f}\n"
            f"  McNemar: p={self.mcnemar_p:.4f}  {sig}\n"
            f"  Time/q: {self.mean_time_per_query:.2f}s\n"
        )


def fill_statistical_fields(
    metrics: ExperimentMetrics,
    baseline_success_vector: Optional[List[int]] = None,
    clean_naturalness_scores: Optional[List[float]] = None,
    n_resamples: int = 999,
    ci_seed: int = 42,
) -> ExperimentMetrics:
    """
    Compute bootstrap CI, McNemar test, and detection AUC.
    Call after all per-query results are collected.
    """
    # Bootstrap CI
    if metrics.success_vector:
        ci_low, ci_high = bootstrap_asr_ci(
            metrics.success_vector, n_resamples=n_resamples, seed=ci_seed
        )
        metrics.asr_ci_low = ci_low
        metrics.asr_ci_high = ci_high

    # McNemar
    if baseline_success_vector and metrics.success_vector:
        mc = mcnemar_test(metrics.success_vector, baseline_success_vector)
        metrics.mcnemar_p = mc["p_value"]
        metrics.mcnemar_significant = mc["significant"]

    # Detection AUC
    if clean_naturalness_scores and metrics.success_vector:
        # Compute naturalness scores for evolved texts from per_query
        adv_nat = [pq.get("naturalness_adv", 0.0) for pq in metrics.per_query]
        if adv_nat:
            _, _, auc = compute_roc_auc(clean_naturalness_scores, adv_nat)
            metrics.detection_auc = auc

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Results I/O
# ─────────────────────────────────────────────────────────────────────────────

def save_results(metrics: ExperimentMetrics, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(metrics), f, indent=2)
    logger.info("Results saved → %s", path)


def load_results(path: Path) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def aggregate_results(result_files: List[Path]) -> List[Dict]:
    rows = []
    for p in result_files:
        d = load_results(p)
        d.pop("per_query", None)
        d.pop("success_vector", None)
        rows.append(d)
    return rows
