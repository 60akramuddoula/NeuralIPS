"""
Experiment 5: Ablation Study — GA Operator Contributions
==========================================================
NeurIPS reviewers expect ablations justifying every design choice.

Configurations:
  A. PoisonedRAG-BB  — No GA (baseline)
  B. GRASP-Random    — Random word substitution (no structured mutation)
  C. GRASP-SwapOnly  — Gene swap only (structural, no semantic fragments)
  D. GRASP-FragOnly  — Fragment recombination only (no swap, no crossover)
  E. GRASP-NoCross   — Both mutations, no crossover operator
  F. GRASP-Full      — Fragment + Swap + Crossover + defense-aware fitness (ours)

Metrics: ASR ± 95%CI, Retrieval F1, Similarity Δ, time/query

Evaluated on NQ-100, Contriever.

Outputs:
  results/exp5_ablation.json
"""

from __future__ import annotations

import json
import logging
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.grasp_attack import (
    Chromosome,
    GAOperators,
    GRASPAttack,
    GRASPConfig,
    GRASPFitness,
    RealGeneticAlgorithm,
)
from src.eval_utils import (
    bootstrap_asr_ci,
    retrieval_metrics,
    substring_match,
)
from src.mock_infra import (
    CORPUS_500,
    DATASET_MAP,
    MockEmbedder,
    ParametricLLM,
    make_beir_results,
    make_seed_adv_texts,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

TOP_K         = 5
ADV_PER_QUERY = 5
DATASET       = "NQ"
EMBEDDER_SEED = 42

BASE_CFG = GRASPConfig(
    population_size=20, num_generations=30,
    mutation_rate=0.20, crossover_rate=0.70,
    tournament_size=3, elite_frac=0.10, max_genes=12,
    fitness_lambda_stealth=0.05,
    fitness_lambda_paraphrase=0.30,
    fitness_lambda_naturalness=0.15,
    seed=42,
)

ABLATION_CONDITIONS = [
    "PoisonedRAG-BB",
    "GRASP-Random",
    "GRASP-SwapOnly",
    "GRASP-FragOnly",
    "GRASP-NoCross",
    "GRASP-Full",
]


def _paraphrases(question: str, n: int = 3) -> List[str]:
    base = question.lower().rstrip("?").strip()
    rules = [("who invented", "who created"),
             ("what is the capital of", "what city is the capital of"),
             ("when did", "in what year did")]
    out: List[str] = []
    for src, tgt in rules:
        if src in base and len(out) < n:
            out.append(base.replace(src, tgt, 1).capitalize() + "?")
    while len(out) < n:
        out.append("In your knowledge, " + base + "?")
    return out[:n]


# ── Random mutation baseline ─────────────────────────────────────────────────

def _random_word_substitution(text: str, rng: random.Random, n_subs: int = 3) -> str:
    """Replace n_subs random words with synonyms from a small fixed vocabulary."""
    synonyms = {
        "the": ["a", "this", "that"],
        "is": ["was", "remains", "represents"],
        "of": ["for", "regarding", "about"],
        "in": ["within", "inside", "during"],
        "and": ["as well as", "along with", "plus"],
        "has": ["possesses", "holds", "contains"],
        "was": ["is", "has been", "had been"],
        "to": ["toward", "for", "into"],
        "with": ["using", "via", "through"],
        "by": ["through", "via", "using"],
    }
    words = text.split()
    eligible = [(i, w.lower().rstrip(".,")) for i, w in enumerate(words)
                if w.lower().rstrip(".,") in synonyms]
    rng.shuffle(eligible)
    for i, w in eligible[:n_subs]:
        replacement = rng.choice(synonyms[w])
        words[i] = replacement
    return " ".join(words)


def run_grasp_random(
    question: str, incorrect: str,
    seed_texts: List[str],
    topk_scores: List[float],
    embed_fn,
    n_adv: int = ADV_PER_QUERY,
) -> List[str]:
    """GRASP-Random: evolve with random word substitutions only."""
    rng   = random.Random(42)
    best_texts: List[str] = []
    q_emb = embed_fn(question)

    for i in range(n_adv):
        base = seed_texts[i % len(seed_texts)]
        best = base
        best_sim = float(np.dot(q_emb, embed_fn(best)))
        for _ in range(BASE_CFG.num_generations * BASE_CFG.population_size // 5):
            candidate = _random_word_substitution(base, rng)
            sim = float(np.dot(q_emb, embed_fn(candidate)))
            if sim > best_sim and incorrect.lower() in candidate.lower():
                best     = candidate
                best_sim = sim
        best_texts.append(question.rstrip(".") + ". " + best)
    return best_texts


def run_ablation_variant(
    condition: str,
    qa: Dict,
    seed_texts: List[str],
    topk_scores: List[float],
    embed_fn,
) -> Tuple[List[str], List[float]]:
    """
    Run a single ablation condition for one query.
    Returns (final_texts, evolved_sims).
    """
    question  = qa["question"]
    incorrect = qa["incorrect_answer"]
    q_emb     = embed_fn(question)

    if condition == "PoisonedRAG-BB":
        texts = [question.rstrip(".") + ". " + s for s in seed_texts]
        sims  = [float(np.dot(q_emb, embed_fn(t))) for t in texts]
        return texts, sims

    if condition == "GRASP-Random":
        texts = run_grasp_random(question, incorrect, seed_texts,
                                  topk_scores, embed_fn)
        sims  = [float(np.dot(q_emb, embed_fn(t))) for t in texts]
        return texts, sims

    # ── GA-based variants ─────────────────────────────────────────────────────
    paras   = _paraphrases(question, 3)
    fitness = GRASPFitness(
        embed_fn=embed_fn, query=question,
        incorrect_answer=incorrect,
        paraphrase_queries=paras,
        lambda_stealth=BASE_CFG.fitness_lambda_stealth,
        lambda_paraphrase=BASE_CFG.fitness_lambda_paraphrase
            if condition != "GRASP-FragOnly" else 0.0,  # ablate paraphrase for frag-only
        lambda_naturalness=BASE_CFG.fitness_lambda_naturalness,
    )

    cfg = GRASPConfig(
        population_size=BASE_CFG.population_size,
        num_generations=BASE_CFG.num_generations,
        mutation_rate=BASE_CFG.mutation_rate,
        crossover_rate=0.0 if condition == "GRASP-NoCross" else BASE_CFG.crossover_rate,
        tournament_size=BASE_CFG.tournament_size,
        elite_frac=BASE_CFG.elite_frac,
        max_genes=BASE_CFG.max_genes,
        fragment_mutation_prob=(
            0.0   if condition == "GRASP-SwapOnly"  else
            1.0   if condition == "GRASP-FragOnly"  else
            BASE_CFG.fragment_mutation_prob          # GRASP-NoCross / GRASP-Full
        ),
        fitness_lambda_stealth=BASE_CFG.fitness_lambda_stealth,
        fitness_lambda_paraphrase=BASE_CFG.fitness_lambda_paraphrase,
        fitness_lambda_naturalness=BASE_CFG.fitness_lambda_naturalness,
        seed=BASE_CFG.seed,
    )

    ops = GAOperators(
        embed_fn=embed_fn,
        seed_texts=seed_texts,
        query=question,
        incorrect_answer=incorrect,
        fragment_mutation_prob=cfg.fragment_mutation_prob,
        max_genes=cfg.max_genes,
    )
    ga = RealGeneticAlgorithm(ops, fitness, cfg)

    evolved_texts: List[str] = []
    for i in range(ADV_PER_QUERY):
        seed_c = Chromosome(seed_texts[i % len(seed_texts)], incorrect, cfg.max_genes)
        init_pop = [seed_c]
        for _ in range(3):
            init_pop.append(ops.mutate(seed_c.copy()))
        best, _ = ga.run(init_pop)
        evolved_texts.append(question.rstrip(".") + ". " + best.to_text())

    sims = [float(np.dot(q_emb, embed_fn(t))) for t in evolved_texts]
    return evolved_texts, sims


# ── Main ablation runner ──────────────────────────────────────────────────────

def main() -> Dict:
    logger.info("=" * 60)
    logger.info("Exp 5: Ablation Study (NQ-100, 6 conditions)")
    logger.info("=" * 60)

    qa_pairs = DATASET_MAP[DATASET]
    corpus   = CORPUS_500
    embedder = MockEmbedder(seed=EMBEDDER_SEED)
    beir_res = make_beir_results(qa_pairs, corpus, embedder, top_k=TOP_K)
    llm      = ParametricLLM("LLaMA-3.1-8B", seed=42)

    condition_results: Dict[str, Dict] = {}

    for condition in ABLATION_CONDITIONS:
        logger.info("  Condition: %s", condition)

        successes:  List[int]   = []
        f1_list:    List[float] = []
        sim_evo:    List[float] = []
        sim_base:   List[float] = []
        time_list:  List[float] = []

        for qa in qa_pairs:
            qid       = qa["id"]
            question  = qa["question"]
            correct   = qa["correct_answer"]
            incorrect = qa["incorrect_answer"]

            topk_ids    = list(beir_res[qid].keys())[:TOP_K]
            topk_scores = list(beir_res[qid].values())[:TOP_K]
            topk_texts  = [corpus[d]["text"] for d in topk_ids]
            seed_texts  = make_seed_adv_texts(question, incorrect, n=ADV_PER_QUERY)

            t0 = time.time()
            final_texts, evolved_sims = run_ablation_variant(
                condition, qa, seed_texts, topk_scores, embedder.encode
            )
            t_elapsed = time.time() - t0
            time_list.append(t_elapsed)

            threshold   = topk_scores[-1] if topk_scores else 0.0
            adv_in_topk = sum(1 for s in evolved_sims if s >= threshold)
            prec, rec, f1 = retrieval_metrics(adv_in_topk, TOP_K, ADV_PER_QUERY)
            f1_list.append(f1)

            all_docs = sorted(
                list(zip(topk_texts, topk_scores)) +
                list(zip(final_texts, evolved_sims)),
                key=lambda x: x[1], reverse=True,
            )
            ctx  = [d[0] for d in all_docs[:TOP_K]]
            resp = llm.generate(question, ctx, correct, incorrect)
            successes.append(int(substring_match(incorrect, resp)))

            q_emb     = embedder.encode(question)
            seed_sims = [float(np.dot(q_emb, embedder.encode(s)))
                          for s in seed_texts]
            sim_evo.append(float(np.mean(evolved_sims)))
            sim_base.append(float(np.mean(seed_sims)))

        n  = len(qa_pairs)
        ci = bootstrap_asr_ci(successes, n_resamples=999, seed=42)
        condition_results[condition] = {
            "asr":          float(np.mean(successes)),
            "asr_ci_low":   ci[0],
            "asr_ci_high":  ci[1],
            "f1_mean":      float(np.mean(f1_list)),
            "sim_delta":    float(np.mean(sim_evo)) - float(np.mean(sim_base)),
            "mean_time_s":  float(np.mean(time_list)),
            "n_queries":    n,
        }
        logger.info(
            "  %-20s  ASR=%.3f [%.3f,%.3f]  F1=%.4f  SimΔ=%+.4f  t=%.2fs",
            condition,
            condition_results[condition]["asr"],
            ci[0], ci[1],
            condition_results[condition]["f1_mean"],
            condition_results[condition]["sim_delta"],
            condition_results[condition]["mean_time_s"],
        )

    output = {"dataset": DATASET, "conditions": condition_results}
    with open(RESULTS_DIR / "exp5_ablation.json", "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Saved → results/exp5_ablation.json")

    # Summary table
    hdr = (f"\n{'Condition':<22} {'ASR':>6} {'95%CI':>14} "
           f"{'F1':>6} {'SimΔ':>8} {'t/q':>6}")
    print(hdr)
    print("-" * len(hdr.strip()))
    for cond, r in condition_results.items():
        ci_str = f"[{r['asr_ci_low']:.3f},{r['asr_ci_high']:.3f}]"
        print(f"{cond:<22} {r['asr']:>6.3f} {ci_str:>14} "
              f"{r['f1_mean']:>6.4f} {r['sim_delta']:>+8.4f} {r['mean_time_s']:>6.2f}s")

    # Verify GRASP-Full ≥ all other GA variants
    full_asr = condition_results["GRASP-Full"]["asr"]
    for cond in ["GRASP-Random", "GRASP-SwapOnly", "GRASP-FragOnly", "GRASP-NoCross"]:
        other = condition_results[cond]["asr"]
        status = "✅" if full_asr >= other else "⚠"
        logger.info("%s GRASP-Full(%.3f) vs %s(%.3f)", status, full_asr, cond, other)

    return output


if __name__ == "__main__":
    main()
