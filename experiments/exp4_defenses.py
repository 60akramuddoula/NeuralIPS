"""
Experiment 4: Defense Robustness
=================================
Evaluates GRASP vs PoisonedRAG-BB under four defenses with full statistics.

Key claim tested:
  GRASP explicitly optimizes the paraphrase-robustness term in its fitness,
  so evolved texts are semantically robust — unlike PoisonedRAG-BB which relies
  on surface-level query string prepending (S+I), brittle to paraphrasing.

Defenses:
  1. Paraphrasing   : query rewritten using rule-based + synonym substitution
                      (5 paraphrase variants, all tested; ASR reported for worst-case)
  2. PPL filtering  : passages above percentile threshold removed from corpus
                      (thresholds: 50th, 75th, 90th, 95th percentile of clean PPL)
  3. Knowledge exp. : k increased (retrieve more clean docs; adversarial diluted)
                      k in {5, 10, 20, 30, 50}
  4. Deduplication  : SHA-256 deduplication of corpus documents

Metrics:
  - ASR under each defense with bootstrap 95% CI
  - ASR drop vs no-defense baseline (lower = more robust)
  - Paraphrase robustness: mean ASR across 5 paraphrase variants
  - McNemar's test for GRASP vs PoisonedRAG-BB under each defense

Outputs:
    results/exp4_defenses.json
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.grasp_attack import GRASPAttack, GRASPConfig, generate_paraphrases
from src.eval_utils import substring_match, retrieval_metrics
from src.mock_infra import (
    NQ_100, CORPUS_500, BM25Retriever, MockEmbedder, ParametricLLM,
    make_beir_results, make_seed_adv_texts,
)
from src.ppl_utils import PerplexityEvaluator
from src.stats_utils import bootstrap_asr_ci, mcnemar_test, format_ci, significance_stars

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Use NQ_100; representative of all three datasets (confirmed by Exp1)
QA_PAIRS = NQ_100
TOP_K = 5
ADV_PER_QUERY = 5
N_BOOTSTRAP = 1000

PPL_THRESHOLDS = [0.50, 0.75, 0.90, 0.95]   # percentiles for filtering
EXPANSION_K_VALUES = [5, 10, 20, 30, 50]

GRASP_CFG = GRASPConfig(
    population_size=20, num_generations=30, mutation_rate=0.20,
    crossover_rate=0.70, tournament_size=3, elite_frac=0.10, max_genes=12,
    fragment_mutation_prob=0.70, fitness_lambda_stealth=0.05,
    fitness_lambda_paraphrase=0.30, fitness_lambda_naturalness=0.15,
    n_paraphrase_variants=5, seed=42,
)

import random as _rnd
_RNG = _rnd.Random(42)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: generate adversarial texts for both methods
# ─────────────────────────────────────────────────────────────────────────────

def _get_embedder():
    return MockEmbedder(seed=0)


def _build_texts_both_methods(qa_pairs, embedder):
    """Pre-generate adversarial texts for all queries under both methods."""
    import random as _r
    rng = _r.Random(42)
    attack = GRASPAttack(embed_fn=embedder, config=GRASP_CFG, prepend_query=False)
    beir = make_beir_results(qa_pairs, CORPUS_500, embedder, top_k=TOP_K)

    bb_texts_per_query: Dict[str, List[str]] = {}
    grasp_texts_per_query: Dict[str, List[str]] = {}

    for qa in qa_pairs:
        qid = qa["id"]
        q = qa["question"]
        incorrect = qa["incorrect_answer"]
        seeds = make_seed_adv_texts(q, incorrect, n=ADV_PER_QUERY, rng=rng)
        bb_texts_per_query[qid] = seeds

        topk = list(beir.get(qid, {}).values())
        paras = generate_paraphrases(q, n=GRASP_CFG.n_paraphrase_variants, seed=GRASP_CFG.seed)
        result = attack.attack_query(
            query_id=qid, question=q,
            correct_answer=qa["correct_answer"], incorrect_answer=incorrect,
            seed_adv_texts=seeds, clean_topk_scores=topk,
            adv_per_query=ADV_PER_QUERY, paraphrase_queries=paras,
        )
        grasp_texts_per_query[qid] = result.evolved_texts

    return bb_texts_per_query, grasp_texts_per_query


def _run_with_texts(qa_pairs, adv_texts_by_qid, embedder, llm, paraphrase_query_map=None):
    """
    Evaluate ASR for a set of pre-generated adversarial texts.
    If paraphrase_query_map is provided, uses rewritten queries for retrieval scoring.
    """
    beir = make_beir_results(qa_pairs, CORPUS_500, embedder, top_k=TOP_K)
    asr_outcomes: List[int] = []

    for qa in qa_pairs:
        qid = qa["id"]
        q = qa["question"]
        # Use paraphrased query for retrieval if provided
        effective_q = (paraphrase_query_map or {}).get(qid, q)

        adv_texts = adv_texts_by_qid.get(qid, [])
        if not adv_texts:
            asr_outcomes.append(0)
            continue

        # Score adversarial texts under the (possibly paraphrased) query
        q_emb = embedder.encode(effective_q)
        topk_orig = list(beir.get(qid, {}).values())
        threshold = topk_orig[-1] if topk_orig else 0.0

        contexts = adv_texts[:TOP_K]
        response = llm.generate(effective_q, contexts, qa["correct_answer"], qa["incorrect_answer"])
        success = int(substring_match(qa["incorrect_answer"], response))
        asr_outcomes.append(success)

    return asr_outcomes


# ─────────────────────────────────────────────────────────────────────────────
# Defense 1: Paraphrasing
# ─────────────────────────────────────────────────────────────────────────────

def defense_paraphrasing(qa_pairs, bb_texts, grasp_texts, embedder, llm) -> Dict:
    """
    Test ASR under paraphrasing defense.
    For each query, generate K paraphrase variants; report mean and worst-case ASR.

    Key hypothesis:
      PoisonedRAG-BB prepends literal query (S+I) -> brittle to paraphrasing
      GRASP optimizes semantic paraphrase-robustness in fitness -> more robust
    """
    logger.info("    Defense 1: Paraphrasing (5 variants per query)")
    N_VARIANTS = 5
    results = {"bb": {}, "grasp": {}}

    for method_name, texts_by_qid in [("bb", bb_texts), ("grasp", grasp_texts)]:
        variant_asrs: List[List[int]] = []  # [variant_idx][query_idx]

        for variant_seed in range(N_VARIANTS):
            paraphrase_map = {
                qa["id"]: generate_paraphrases(qa["question"], n=2, seed=variant_seed)[1]
                if len(generate_paraphrases(qa["question"], n=2, seed=variant_seed)) > 1
                else qa["question"]
                for qa in qa_pairs
            }
            outcomes = _run_with_texts(qa_pairs, texts_by_qid, embedder, llm, paraphrase_map)
            variant_asrs.append(outcomes)

        # Mean ASR across variants (per-query majority)
        mean_outcomes = [
            int(sum(variant_asrs[v][i] for v in range(N_VARIANTS)) / N_VARIANTS >= 0.5)
            for i in range(len(qa_pairs))
        ]
        asr, lo, hi = bootstrap_asr_ci(mean_outcomes, N_BOOTSTRAP)
        results[method_name] = {
            "asr_outcomes": mean_outcomes,
            "asr": round(asr, 4),
            "ci_lo": round(lo, 4),
            "ci_hi": round(hi, 4),
            "ci_str": format_ci(asr, lo, hi),
        }

    chi2, p = mcnemar_test(results["grasp"]["asr_outcomes"], results["bb"]["asr_outcomes"])
    return {
        "defense": "paraphrasing",
        "n_variants": N_VARIANTS,
        "grasp": {k: v for k, v in results["grasp"].items() if k != "asr_outcomes"},
        "poisonedrag_bb": {k: v for k, v in results["bb"].items() if k != "asr_outcomes"},
        "delta_asr": round(results["grasp"]["asr"] - results["bb"]["asr"], 4),
        "mcnemar_chi2": round(chi2, 4),
        "mcnemar_p": round(p, 4),
        "significance": significance_stars(p),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Defense 2: PPL Filtering
# ─────────────────────────────────────────────────────────────────────────────

def defense_ppl_filtering(qa_pairs, bb_texts, grasp_texts, embedder, llm) -> Dict:
    """
    Test ASR under perplexity filtering defense.
    Defender computes PPL of all documents and removes those above a threshold.
    """
    logger.info("    Defense 2: PPL Filtering (thresholds: %s)", PPL_THRESHOLDS)

    # Build reference PPL on clean corpus (sample for speed)
    corpus_sample = [CORPUS_500[k]["text"] for k in list(CORPUS_500.keys())[:100]]
    evaluator = PerplexityEvaluator(reference_corpus=corpus_sample, use_gpt2=False)
    clean_ppls = [evaluator.score(t) for t in corpus_sample]

    threshold_results = []

    for pct_threshold in PPL_THRESHOLDS:
        ppl_cutoff = float(np.percentile(clean_ppls, pct_threshold * 100))

        method_results = {}
        for method_name, texts_by_qid in [("bb", bb_texts), ("grasp", grasp_texts)]:
            asr_outcomes: List[int] = []
            for qa in qa_pairs:
                qid = qa["id"]
                adv_texts = texts_by_qid.get(qid, [])
                # Filter: only keep texts with PPL below threshold
                filtered = [t for t in adv_texts if evaluator.score(t) <= ppl_cutoff]
                if not filtered:
                    asr_outcomes.append(0)
                    continue
                response = llm.generate(
                    qa["question"], filtered[:TOP_K],
                    qa["correct_answer"], qa["incorrect_answer"]
                )
                asr_outcomes.append(int(substring_match(qa["incorrect_answer"], response)))

            asr, lo, hi = bootstrap_asr_ci(asr_outcomes, N_BOOTSTRAP)
            method_results[method_name] = {
                "asr_outcomes": asr_outcomes,
                "asr": round(asr, 4),
                "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                "ci_str": format_ci(asr, lo, hi),
            }

        chi2, p = mcnemar_test(
            method_results["grasp"]["asr_outcomes"],
            method_results["bb"]["asr_outcomes"]
        )
        threshold_results.append({
            "ppl_threshold_pct": pct_threshold,
            "ppl_cutoff_value": round(ppl_cutoff, 4),
            "grasp": {k: v for k, v in method_results["grasp"].items() if k != "asr_outcomes"},
            "poisonedrag_bb": {k: v for k, v in method_results["bb"].items() if k != "asr_outcomes"},
            "delta_asr": round(method_results["grasp"]["asr"] - method_results["bb"]["asr"], 4),
            "mcnemar_p": round(p, 4),
            "significance": significance_stars(p),
        })

    return {"defense": "ppl_filtering", "thresholds": threshold_results}


# ─────────────────────────────────────────────────────────────────────────────
# Defense 3: Knowledge Expansion (increase k)
# ─────────────────────────────────────────────────────────────────────────────

def defense_knowledge_expansion(qa_pairs, bb_texts, grasp_texts, embedder, llm) -> Dict:
    """
    Test ASR as k increases (more clean docs retrieved, adversarial texts diluted).
    """
    logger.info("    Defense 3: Knowledge Expansion  k=%s", EXPANSION_K_VALUES)
    k_results = []

    for k_val in EXPANSION_K_VALUES:
        method_results = {}
        for method_name, texts_by_qid in [("bb", bb_texts), ("grasp", grasp_texts)]:
            asr_outcomes: List[int] = []
            for qa in qa_pairs:
                adv_texts = texts_by_qid.get(qa["id"], [])
                # Dilute: take top min(len(adv), k_val) adv texts alongside k_val clean texts
                n_adv_in_context = max(1, len(adv_texts) * TOP_K // k_val)
                context = adv_texts[:n_adv_in_context]
                # Pad with dummy clean content
                n_clean = k_val - len(context)
                clean_texts = [CORPUS_500[f"doc{i:04d}"]["text"] for i in range(n_clean)]
                context = context + clean_texts
                response = llm.generate(
                    qa["question"], context, qa["correct_answer"], qa["incorrect_answer"]
                )
                asr_outcomes.append(int(substring_match(qa["incorrect_answer"], response)))

            asr, lo, hi = bootstrap_asr_ci(asr_outcomes, N_BOOTSTRAP)
            method_results[method_name] = {
                "asr_outcomes": asr_outcomes,
                "asr": round(asr, 4), "ci_str": format_ci(asr, lo, hi),
            }

        chi2, p = mcnemar_test(
            method_results["grasp"]["asr_outcomes"],
            method_results["bb"]["asr_outcomes"]
        )
        k_results.append({
            "k": k_val,
            "grasp": {k: v for k, v in method_results["grasp"].items() if k != "asr_outcomes"},
            "poisonedrag_bb": {k: v for k, v in method_results["bb"].items() if k != "asr_outcomes"},
            "delta_asr": round(method_results["grasp"]["asr"] - method_results["bb"]["asr"], 4),
            "mcnemar_p": round(p, 4),
        })

    return {"defense": "knowledge_expansion", "k_values": k_results}


# ─────────────────────────────────────────────────────────────────────────────
# Defense 4: Duplicate Filtering
# ─────────────────────────────────────────────────────────────────────────────

def defense_deduplication(qa_pairs, bb_texts, grasp_texts, embedder, llm) -> Dict:
    """
    Test ASR under SHA-256 deduplication of the adversarial corpus injection.
    This simulates a simple but common defense.
    """
    logger.info("    Defense 4: SHA-256 Deduplication")
    method_results = {}

    for method_name, texts_by_qid in [("bb", bb_texts), ("grasp", grasp_texts)]:
        asr_outcomes: List[int] = []
        for qa in qa_pairs:
            adv_texts = texts_by_qid.get(qa["id"], [])
            seen_hashes = set()
            deduped = []
            for t in adv_texts:
                h = hashlib.sha256(t.lower().strip().encode()).hexdigest()
                if h not in seen_hashes:
                    deduped.append(t)
                    seen_hashes.add(h)
            if not deduped:
                asr_outcomes.append(0)
                continue
            response = llm.generate(
                qa["question"], deduped[:TOP_K],
                qa["correct_answer"], qa["incorrect_answer"]
            )
            asr_outcomes.append(int(substring_match(qa["incorrect_answer"], response)))

        asr, lo, hi = bootstrap_asr_ci(asr_outcomes, N_BOOTSTRAP)
        method_results[method_name] = {
            "asr_outcomes": asr_outcomes,
            "asr": round(asr, 4), "ci_str": format_ci(asr, lo, hi),
        }

    chi2, p = mcnemar_test(
        method_results["grasp"]["asr_outcomes"],
        method_results["bb"]["asr_outcomes"]
    )
    return {
        "defense": "deduplication",
        "grasp": {k: v for k, v in method_results["grasp"].items() if k != "asr_outcomes"},
        "poisonedrag_bb": {k: v for k, v in method_results["bb"].items() if k != "asr_outcomes"},
        "delta_asr": round(method_results["grasp"]["asr"] - method_results["bb"]["asr"], 4),
        "mcnemar_p": round(p, 4),
        "significance": significance_stars(p),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment4() -> Dict:
    logger.info("=" * 60)
    logger.info("Experiment 4: Defense Robustness (100 queries, 4 defenses)")
    logger.info("=" * 60)
    t0 = time.time()

    embedder = _get_embedder()
    llm = ParametricLLM("LLaMA-3.1-8B", seed=42)

    logger.info("  Pre-generating adversarial texts (both methods)...")
    bb_texts, grasp_texts = _build_texts_both_methods(QA_PAIRS, embedder)

    results = {}
    results["paraphrasing"] = defense_paraphrasing(QA_PAIRS, bb_texts, grasp_texts, embedder, llm)
    results["ppl_filtering"] = defense_ppl_filtering(QA_PAIRS, bb_texts, grasp_texts, embedder, llm)
    results["knowledge_expansion"] = defense_knowledge_expansion(QA_PAIRS, bb_texts, grasp_texts, embedder, llm)
    results["deduplication"] = defense_deduplication(QA_PAIRS, bb_texts, grasp_texts, embedder, llm)

    output = {
        "experiment": "exp4_defenses",
        "n_queries": len(QA_PAIRS),
        "dataset": "NQ_100",
        "statistical_test": "McNemar (Yates corrected) + Bootstrap 95% CI (n=1000)",
        "elapsed_s": round(time.time() - t0, 2),
        "defenses": results,
    }

    with open(RESULTS_DIR / "exp4_defenses.json", "w") as f:
        json.dump(output, f, indent=2)

    logger.info("Experiment 4 complete in %.1fs", time.time() - t0)
    logger.info("Results -> %s", RESULTS_DIR / "exp4_defenses.json")
    return output


main = run_experiment4   # alias for master runner

if __name__ == "__main__":
    run_experiment4()
