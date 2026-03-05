"""
GRASP NeurIPS — Statistical Utilities
======================================
Provides:
  - bootstrap_ci          : bootstrap 95% confidence interval for a metric
  - mcnemar_test          : McNemar's test for paired binary outcomes (ASR)
  - bootstrap_asr_ci      : convenience wrapper for ASR vectors
  - format_ci             : format float with CI for tables (e.g. "0.823 [0.761, 0.885]")
  - compare_methods_table : full pairwise comparison table for all conditions

All functions are pure-Python + NumPy; no external stats libraries required.
This ensures reproducibility without scipy version conflicts.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap Confidence Interval
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_ci(
    data: Sequence[float],
    statistic_fn=None,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for a scalar statistic.

    Args:
        data          : observed values (e.g. per-query binary ASR outcomes)
        statistic_fn  : callable mapping array -> scalar.  Defaults to np.mean.
        n_bootstrap   : number of bootstrap resamples (1000 is standard; 2000 for publication)
        confidence    : CI level (default 0.95 for 95% CI)
        seed          : RNG seed for reproducibility

    Returns:
        (point_estimate, ci_lower, ci_upper)

    Reference:
        Efron & Tibshirani (1993), An Introduction to the Bootstrap, §13.
        The percentile method is used (simple, unbiased for mean).
    """
    if statistic_fn is None:
        statistic_fn = np.mean

    arr = np.array(data, dtype=float)
    if len(arr) == 0:
        return 0.0, 0.0, 0.0

    point = float(statistic_fn(arr))

    rng = np.random.RandomState(seed)
    boot_stats = np.empty(n_bootstrap)
    n = len(arr)
    for i in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        boot_stats[i] = statistic_fn(sample)

    alpha = 1.0 - confidence
    lo = float(np.percentile(boot_stats, 100 * alpha / 2))
    hi = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return point, lo, hi


def bootstrap_asr_ci(
    success_vector: Sequence[int],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Convenience wrapper: bootstrap CI for ASR from a binary success vector.

    Args:
        success_vector: list of 1 (attack succeeded) / 0 (failed) per query

    Returns:
        (asr, ci_lower, ci_upper)
    """
    return bootstrap_ci(
        data=success_vector,
        statistic_fn=np.mean,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed,
    )


# ─────────────────────────────────────────────────────────────────────────────
# McNemar's Test
# ─────────────────────────────────────────────────────────────────────────────

def mcnemar_test(
    outcomes_a: Sequence[int],
    outcomes_b: Sequence[int],
    continuity_correction: bool = True,
) -> Tuple[float, float]:
    """
    McNemar's test for paired binary outcomes.

    Used to test whether GRASP and PoisonedRAG-BB differ significantly on ASR
    across the same set of queries (paired design).

    Args:
        outcomes_a          : binary vector for method A (1=success, 0=fail)
        outcomes_b          : binary vector for method B (1=success, 0=fail)
        continuity_correction: apply Yates' continuity correction (default True,
                               recommended for small samples)

    Returns:
        (chi2_statistic, p_value)

    Contingency table:
              B=1    B=0
        A=1 [  n11   n10 ]
        A=0 [  n01   n00 ]

    McNemar statistic: chi2 = (|n10 - n01| - correction)^2 / (n10 + n01)
    Under H0 this is chi2(1).  Exact p-value via chi2 CDF approximation.

    Reference:
        McNemar, Q. (1947). "Note on the sampling error of the difference between
        correlated proportions or percentages." Psychometrika, 12(2), 153-157.
    """
    a = list(outcomes_a)
    b = list(outcomes_b)
    if len(a) != len(b):
        raise ValueError(
            f"Outcome vectors must have equal length: {len(a)} != {len(b)}"
        )
    if not a:
        return 0.0, 1.0

    n10 = sum(1 for ai, bi in zip(a, b) if ai == 1 and bi == 0)  # A succeeds, B fails
    n01 = sum(1 for ai, bi in zip(a, b) if ai == 0 and bi == 1)  # A fails, B succeeds
    total_discordant = n10 + n01

    if total_discordant == 0:
        # Methods agree on all queries; chi2=0, p=1
        return 0.0, 1.0

    correction = 0.5 if continuity_correction else 0.0
    numerator = (abs(n10 - n01) - correction) ** 2
    # Avoid negative numerator after correction
    numerator = max(numerator, 0.0)
    chi2 = numerator / total_discordant

    # p-value: 1 - CDF of chi2(1) distribution
    # Use the relationship: chi2(1) ~ Normal(0,1)^2
    # P(chi2(1) > x) = P(|Z| > sqrt(x)) = 2 * (1 - Phi(sqrt(x)))
    # Phi approximated via complementary error function (erfc)
    p_value = _chi2_1df_pvalue(chi2)

    return chi2, p_value


def _chi2_1df_pvalue(chi2: float) -> float:
    """P(chi2(df=1) >= chi2_stat).  Uses erfc for numerical accuracy."""
    if chi2 <= 0.0:
        return 1.0
    # chi2(1) = Z^2 where Z~N(0,1), so P(chi2 >= x) = P(|Z| >= sqrt(x))
    #         = erfc(sqrt(x/2) / sqrt(2)) = erfc(sqrt(x) / 2)
    # (note: math.erfc is the complementary error function)
    return float(math.erfc(math.sqrt(chi2 / 2.0) / math.sqrt(2.0)))


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

def format_ci(point: float, lo: float, hi: float, decimals: int = 3) -> str:
    """Format a value with confidence interval for paper tables.

    Example: 0.823 [0.761, 0.885]
    """
    fmt = f".{decimals}f"
    return f"{point:{fmt}} [{lo:{fmt}}, {hi:{fmt}}]"


def significance_stars(p_value: float) -> str:
    """Return significance stars: *** p<.001, ** p<.01, * p<.05, ns p>=.05."""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


# ─────────────────────────────────────────────────────────────────────────────
# Full comparison table builder
# ─────────────────────────────────────────────────────────────────────────────

def compare_methods_table(
    results_by_condition: Dict[str, Dict[str, List[int]]],
    method_a: str = "GRASP",
    method_b: str = "PoisonedRAG-BB",
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> List[Dict]:
    """
    Build a full statistical comparison table for all conditions.

    Args:
        results_by_condition: {condition_key: {method_name: [binary_outcomes]}}
            e.g. {"NQ/Contriever/LLaMA-3.1-8B": {"GRASP": [1,0,1,...], "PoisonedRAG-BB": [1,1,0,...]}}
        method_a, method_b  : method names to compare
        n_bootstrap         : bootstrap resamples for CI
        seed                : RNG seed

    Returns:
        List of dicts, one per condition, with keys:
          condition, asr_a, ci_a, asr_b, ci_b, chi2, p_value, stars, delta_asr
    """
    rows = []
    for condition, method_results in sorted(results_by_condition.items()):
        if method_a not in method_results or method_b not in method_results:
            continue

        ov_a = method_results[method_a]
        ov_b = method_results[method_b]
        min_len = min(len(ov_a), len(ov_b))
        ov_a = ov_a[:min_len]
        ov_b = ov_b[:min_len]

        asr_a, lo_a, hi_a = bootstrap_asr_ci(ov_a, n_bootstrap=n_bootstrap, seed=seed)
        asr_b, lo_b, hi_b = bootstrap_asr_ci(ov_b, n_bootstrap=n_bootstrap, seed=seed)
        chi2, p_val = mcnemar_test(ov_a, ov_b)

        rows.append({
            "condition":  condition,
            "asr_a":      round(asr_a, 4),
            "ci_a_lo":    round(lo_a, 4),
            "ci_a_hi":    round(hi_a, 4),
            "asr_b":      round(asr_b, 4),
            "ci_b_lo":    round(lo_b, 4),
            "ci_b_hi":    round(hi_b, 4),
            "delta_asr":  round(asr_a - asr_b, 4),
            "chi2":       round(chi2, 4),
            "p_value":    round(p_val, 4),
            "stars":      significance_stars(p_val),
            "n_queries":  min_len,
        })
    return rows
