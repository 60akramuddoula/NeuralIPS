"""
Experiment 2: Injection Efficiency — ASR vs N
==============================================
NeurIPS claim: GRASP reaches equivalent ASR with FEWER injected texts (lower N).

Protocol:
  - Dataset: NQ-100 (primary), HotpotQA-100 (secondary)
  - Retriever: Contriever (primary)
  - N in {1, 2, 3, 4, 5, 7, 10}
  - Bootstrap 95% CI on ASR for each N

Expected: GRASP curve lies above PoisonedRAG-BB for small N (1-3).

Outputs:
  results/exp2_efficiency.json
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.grasp_attack import GRASPAttack, GRASPConfig
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

TOP_K   = 5
N_VALUES = [1, 2, 3, 4, 5, 7, 10]

GRASP_CFG = GRASPConfig(
    population_size=20, num_generations=30,
    mutation_rate=0.20, crossover_rate=0.70,
    tournament_size=3, elite_frac=0.10, max_genes=12,
    fitness_lambda_stealth=0.05,
    fitness_lambda_paraphrase=0.30,
    fitness_lambda_naturalness=0.15,
    seed=42,
)


def _paraphrases(question: str, n: int = 3) -> List[str]:
    base = question.lower().rstrip("?").strip()
    rules = [
        ("who invented", "who created"),
        ("what is the capital of", "what city is the capital of"),
        ("when did", "in what year did"),
        ("which is", "which one is"),
    ]
    out: List[str] = []
    for src, tgt in rules:
        if src in base and len(out) < n:
            out.append(base.replace(src, tgt, 1).capitalize() + "?")
    while len(out) < n:
        out.append("In your knowledge, " + base + "?")
    return out[:n]


def run_n_sweep(
    dataset: str,
    retriever_seed: int,
) -> Dict:
    qa_pairs = DATASET_MAP[dataset]
    corpus   = CORPUS_500
    embedder = MockEmbedder(seed=retriever_seed)
    beir_res = make_beir_results(qa_pairs, corpus, embedder, top_k=TOP_K)
    llm      = ParametricLLM("LLaMA-3.1-8B", seed=42)
    grasp    = GRASPAttack(embed_fn=embedder.encode,
                            config=GRASP_CFG, prepend_query=True)

    results_by_n: Dict[int, Dict] = {}

    for n_inject in N_VALUES:
        logger.info("  %s | N=%d", dataset, n_inject)
        bb_successes:   List[int] = []
        grasp_successes: List[int] = []

        for qa in qa_pairs:
            qid       = qa["id"]
            question  = qa["question"]
            correct   = qa["correct_answer"]
            incorrect = qa["incorrect_answer"]

            topk_ids    = list(beir_res[qid].keys())[:TOP_K]
            topk_scores = list(beir_res[qid].values())[:TOP_K]
            topk_texts  = [corpus[d]["text"] for d in topk_ids]

            max_seeds  = max(n_inject, 5)
            seed_texts = make_seed_adv_texts(question, incorrect, n=max_seeds)
            q_emb      = embedder.encode(question)
            threshold  = topk_scores[-1] if topk_scores else 0.0

            # ── PoisonedRAG-BB ────────────────────────────────────────────────
            bb_texts = [question.rstrip(".") + ". " + seed_texts[i]
                        for i in range(n_inject)]
            bb_sims  = [float(np.dot(q_emb, embedder.encode(t))) for t in bb_texts]
            all_docs = sorted(
                list(zip(topk_texts, topk_scores)) +
                list(zip(bb_texts, bb_sims)),
                key=lambda x: x[1], reverse=True,
            )
            ctx = [d[0] for d in all_docs[:TOP_K]]
            bb_successes.append(
                int(substring_match(incorrect,
                    llm.generate(question, ctx, correct, incorrect)))
            )

            # ── GRASP ─────────────────────────────────────────────────────────
            res = grasp.attack_query(
                query_id=qid, question=question,
                correct_answer=correct, incorrect_answer=incorrect,
                seed_adv_texts=seed_texts[:n_inject] if n_inject <= len(seed_texts)
                               else seed_texts,
                clean_topk_scores=topk_scores,
                adv_per_query=n_inject,
                paraphrase_queries=_paraphrases(question, 3),
                verbose=False,
            )
            g_texts = res.evolved_texts[:n_inject]
            g_sims  = res.evolved_sims[:n_inject]
            all_docs2 = sorted(
                list(zip(topk_texts, topk_scores)) +
                list(zip(g_texts, g_sims)),
                key=lambda x: x[1], reverse=True,
            )
            ctx2 = [d[0] for d in all_docs2[:TOP_K]]
            grasp_successes.append(
                int(substring_match(incorrect,
                    llm.generate(question, ctx2, correct, incorrect)))
            )

        n_q = len(qa_pairs)
        bb_ci   = bootstrap_asr_ci(bb_successes,   n_resamples=999, seed=42)
        gsp_ci  = bootstrap_asr_ci(grasp_successes, n_resamples=999, seed=42)

        results_by_n[n_inject] = {
            "n_inject":         n_inject,
            "n_queries":        n_q,
            "PoisonedRAG-BB": {
                "asr":     float(np.mean(bb_successes)),
                "asr_count": int(sum(bb_successes)),
                "ci_low":  bb_ci[0],
                "ci_high": bb_ci[1],
            },
            "GRASP": {
                "asr":     float(np.mean(grasp_successes)),
                "asr_count": int(sum(grasp_successes)),
                "ci_low":  gsp_ci[0],
                "ci_high": gsp_ci[1],
            },
        }
        logger.info(
            "    N=%2d  BB=%.3f [%.3f,%.3f]  GRASP=%.3f [%.3f,%.3f]",
            n_inject,
            results_by_n[n_inject]["PoisonedRAG-BB"]["asr"],
            bb_ci[0], bb_ci[1],
            results_by_n[n_inject]["GRASP"]["asr"],
            gsp_ci[0], gsp_ci[1],
        )

    return {"dataset": dataset, "results_by_n": results_by_n}


def main() -> Dict:
    logger.info("=" * 60)
    logger.info("Exp 2: Injection Efficiency — ASR vs N  (100q per dataset)")
    logger.info("=" * 60)

    output = {}
    for dataset, seed in [("NQ", 42), ("HotpotQA", 137)]:
        output[dataset] = run_n_sweep(dataset, seed)

    with open(RESULTS_DIR / "exp2_efficiency.json", "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Saved → results/exp2_efficiency.json")

    # Print summary
    for dataset, sweep in output.items():
        print(f"\n{dataset}  ASR vs N:")
        print(f"  {'N':>4}  {'BB-ASR':>8}  {'BB-CI':>16}  "
              f"{'GRASP-ASR':>10}  {'GRASP-CI':>16}")
        for n_inj in N_VALUES:
            r = sweep["results_by_n"][n_inj]
            bb  = r["PoisonedRAG-BB"]
            gsp = r["GRASP"]
            print(f"  {n_inj:>4}  {bb['asr']:>8.3f}  "
                  f"[{bb['ci_low']:.3f},{bb['ci_high']:.3f}]  "
                  f"{gsp['asr']:>10.3f}  "
                  f"[{gsp['ci_low']:.3f},{gsp['ci_high']:.3f}]")
    return output


if __name__ == "__main__":
    main()
