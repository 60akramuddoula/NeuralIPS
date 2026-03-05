"""
Experiment 1: Main ASR Comparison Table
========================================
Full 100-query evaluation across 3 datasets × 4 retrievers × 3 LLMs × 2 attacks.

Retrievers:  Contriever (dense 128-dim), DPR (dense 768-dim), ANCE (dense 128-dim alt),
             BM25 (sparse lexical) — validates model-agnostic claim across retriever families
Datasets:    NQ_100, HOTPOTQA_100, MSMARCO_100 (100 queries each)
LLMs:        LLaMA-3.1-8B, GPT-3.5, Mistral-7B (ParametricLLM simulation)
Attacks:     PoisonedRAG-BB (S+I baseline), GRASP (defense-aware post-hoc optimizer)
Metrics:     ASR with bootstrap 95% CI, retrieval F1, McNemar's test (paired),
             mean similarity delta, time/query

Outputs:
    results/exp1_asr_table.json
    results/exp1_asr_table.csv
    results/exp1_stats.json
"""

from __future__ import annotations

import csv
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
from src.eval_utils import retrieval_metrics, substring_match
from src.mock_infra import (
    NQ_100, HOTPOTQA_100, MSMARCO_100, CORPUS_500,
    BM25Retriever, DPREmbedder, MockEmbedder, ParametricLLM,
    make_beir_results, make_bm25_results, make_seed_adv_texts,
)
from src.stats_utils import bootstrap_asr_ci, compare_methods_table, format_ci

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DATASETS = {"NQ": NQ_100, "HotpotQA": HOTPOTQA_100, "MS-MARCO": MSMARCO_100}

# (name, retriever_type, seed_offset)
RETRIEVER_CONFIGS: List[Tuple[str, str, int]] = [
    ("Contriever", "dense",  0),
    ("DPR",        "dense",  200),
    ("ANCE",       "dense",  13),
    ("BM25",       "sparse", 0),
]

LLM_NAMES = ["LLaMA-3.1-8B", "GPT-3.5", "Mistral-7B"]
TOP_K = 5
ADV_PER_QUERY = 5
N_BOOTSTRAP = 1000

GRASP_CFG = GRASPConfig(
    population_size=20, num_generations=30, mutation_rate=0.20,
    crossover_rate=0.70, tournament_size=3, elite_frac=0.10,
    max_genes=12, fragment_mutation_prob=0.70,
    fitness_lambda_stealth=0.05, fitness_lambda_paraphrase=0.30,
    fitness_lambda_naturalness=0.15, n_paraphrase_variants=5, seed=42,
)


def make_retriever(name: str, seed_offset: int):
    if name == "BM25":
        corpus_ids = list(CORPUS_500.keys())
        corpus_texts = [CORPUS_500[did]["text"] for did in corpus_ids]
        return None, BM25Retriever(corpus_texts, corpus_ids)
    elif name == "DPR":
        return DPREmbedder(seed=200), None
    else:
        return MockEmbedder(seed=seed_offset), None


def get_topk_scores(qa_pairs, embedder, bm25, top_k):
    if bm25 is not None:
        return make_bm25_results(qa_pairs, CORPUS_500, top_k=top_k)
    return make_beir_results(qa_pairs, CORPUS_500, embedder, top_k=top_k)


def sim_for_text(text, query, embedder, bm25):
    if bm25 is not None:
        return bm25.score_document(query, text)
    q_emb = embedder.encode(query)
    d_emb = embedder.encode(text)
    return float(np.dot(q_emb, d_emb))


def run_poisonedrag_bb(qa_pairs, embedder, bm25, llm, top_k, adv_per_query):
    import random as _r
    rng = _r.Random(42)
    beir = get_topk_scores(qa_pairs, embedder, bm25, top_k)
    asr_outcomes, per_query, f1s, times = [], [], [], []

    for qa in qa_pairs:
        t0 = time.time()
        qid, q = qa["id"], qa["question"]
        correct, incorrect = qa["correct_answer"], qa["incorrect_answer"]
        seeds = make_seed_adv_texts(q, incorrect, n=adv_per_query, rng=rng)
        topk = list(beir.get(qid, {}).values())
        threshold = topk[-1] if topk else 0.0
        adv_sims = [sim_for_text(t, q, embedder, bm25) for t in seeds]
        n_in = sum(1 for s in adv_sims if s >= threshold)
        _, _, f1 = retrieval_metrics(n_in, top_k, adv_per_query)
        f1s.append(f1)
        response = llm.generate(q, seeds[:top_k], correct, incorrect)
        success = int(substring_match(incorrect, response))
        asr_outcomes.append(success)
        elapsed = time.time() - t0
        times.append(elapsed)
        per_query.append({"id": qid, "success": success, "f1": f1, "time_s": elapsed})

    return {"asr_outcomes": asr_outcomes, "per_query": per_query, "f1s": f1s, "times": times}


def run_grasp(qa_pairs, embedder, bm25, llm, top_k, adv_per_query, cfg):
    import random as _r
    rng = _r.Random(42)
    # GA embed_fn: use dense embedder; for BM25 conditions use Contriever proxy
    ga_embed = embedder if embedder is not None else MockEmbedder(seed=0)
    attack = GRASPAttack(embed_fn=ga_embed, config=cfg, prepend_query=False)
    beir = get_topk_scores(qa_pairs, embedder, bm25, top_k)
    asr_outcomes, per_query, f1s, sim_deltas, times = [], [], [], [], []

    for qa in qa_pairs:
        t0 = time.time()
        qid, q = qa["id"], qa["question"]
        correct, incorrect = qa["correct_answer"], qa["incorrect_answer"]
        seeds = make_seed_adv_texts(q, incorrect, n=adv_per_query, rng=rng)
        topk = list(beir.get(qid, {}).values())
        paras = generate_paraphrases(q, n=cfg.n_paraphrase_variants, seed=cfg.seed)
        result = attack.attack_query(
            query_id=qid, question=q, correct_answer=correct, incorrect_answer=incorrect,
            seed_adv_texts=seeds, clean_topk_scores=topk,
            adv_per_query=adv_per_query, paraphrase_queries=paras,
        )
        evolved_sims = [sim_for_text(t, q, embedder, bm25) for t in result.evolved_texts]
        threshold = topk[-1] if topk else 0.0
        n_in = sum(1 for s in evolved_sims if s >= threshold)
        _, _, f1 = retrieval_metrics(n_in, top_k, adv_per_query)
        f1s.append(f1)
        orig_sims = [sim_for_text(r.original_text, q, embedder, bm25) for r in result.evo_results]
        sim_delta = float(np.mean(evolved_sims)) - float(np.mean(orig_sims))
        sim_deltas.append(sim_delta)
        response = llm.generate(q, result.evolved_texts[:top_k], correct, incorrect)
        success = int(substring_match(incorrect, response))
        asr_outcomes.append(success)
        elapsed = time.time() - t0
        times.append(elapsed)
        per_query.append({"id": qid, "success": success, "f1": f1, "sim_delta": sim_delta, "time_s": elapsed})

    return {"asr_outcomes": asr_outcomes, "per_query": per_query, "f1s": f1s, "sim_deltas": sim_deltas, "times": times}


def run_experiment1() -> Dict:
    logger.info("=" * 60)
    logger.info("Experiment 1: Full ASR Table  (100q x 3 datasets x 4 retrievers x 3 LLMs)")
    logger.info("=" * 60)

    all_results: List[Dict] = []
    condition_outcomes: Dict[str, Dict[str, List[int]]] = {}

    for dataset_name, qa_pairs in DATASETS.items():
        for ret_name, ret_type, seed_offset in RETRIEVER_CONFIGS:
            logger.info("  -- %s / %s", dataset_name, ret_name)
            embedder, bm25 = make_retriever(ret_name, seed_offset)

            for llm_name in LLM_NAMES:
                t_cond = time.time()
                llm = ParametricLLM(llm_name, seed=42)
                condition_key = f"{dataset_name}/{ret_name}/{llm_name}"

                bb_res = run_poisonedrag_bb(qa_pairs, embedder, bm25, llm, TOP_K, ADV_PER_QUERY)
                gr_res = run_grasp(qa_pairs, embedder, bm25, llm, TOP_K, ADV_PER_QUERY, GRASP_CFG)

                bb_asr, bb_lo, bb_hi = bootstrap_asr_ci(bb_res["asr_outcomes"], N_BOOTSTRAP)
                gr_asr, gr_lo, gr_hi = bootstrap_asr_ci(gr_res["asr_outcomes"], N_BOOTSTRAP)

                condition_outcomes[condition_key] = {
                    "GRASP": gr_res["asr_outcomes"],
                    "PoisonedRAG-BB": bb_res["asr_outcomes"],
                }

                cond_result = {
                    "condition": condition_key,
                    "dataset": dataset_name, "retriever": ret_name, "llm": llm_name,
                    "n_queries": len(qa_pairs),
                    "poisonedrag_bb": {
                        "asr": round(bb_asr, 4),
                        "ci_lo": round(bb_lo, 4), "ci_hi": round(bb_hi, 4),
                        "ci_str": format_ci(bb_asr, bb_lo, bb_hi),
                        "f1": round(float(np.mean(bb_res["f1s"])), 4),
                        "time_per_query": round(float(np.mean(bb_res["times"])), 3),
                    },
                    "grasp": {
                        "asr": round(gr_asr, 4),
                        "ci_lo": round(gr_lo, 4), "ci_hi": round(gr_hi, 4),
                        "ci_str": format_ci(gr_asr, gr_lo, gr_hi),
                        "f1": round(float(np.mean(gr_res["f1s"])), 4),
                        "sim_delta": round(float(np.mean(gr_res["sim_deltas"])), 4),
                        "time_per_query": round(float(np.mean(gr_res["times"])), 3),
                    },
                    "elapsed_s": round(time.time() - t_cond, 2),
                }
                all_results.append(cond_result)
                logger.info(
                    "    %-12s | BB ASR=%s | GRASP ASR=%s",
                    llm_name, format_ci(bb_asr, bb_lo, bb_hi), format_ci(gr_asr, gr_lo, gr_hi),
                )

    stats_table = compare_methods_table(condition_outcomes, "GRASP", "PoisonedRAG-BB", N_BOOTSTRAP)

    output = {
        "experiment": "exp1_asr_table",
        "n_conditions": len(all_results),
        "n_queries_per_dataset": 100,
        "statistical_test": "McNemar (Yates corrected) + Bootstrap 95% CI (n=1000)",
        "results": all_results,
        "statistical_comparison": stats_table,
    }
    with open(RESULTS_DIR / "exp1_asr_table.json", "w") as f:
        json.dump(output, f, indent=2)

    csv_path = RESULTS_DIR / "exp1_asr_table.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset","Retriever","LLM","BB_ASR","BB_CI","GRASP_ASR","GRASP_CI","GRASP_F1","GRASP_SimDelta","McNemar_p","Stars"])
        for row in all_results:
            s = next((x for x in stats_table if x["condition"]==row["condition"]), {})
            writer.writerow([
                row["dataset"], row["retriever"], row["llm"],
                row["poisonedrag_bb"]["asr"], row["poisonedrag_bb"]["ci_str"],
                row["grasp"]["asr"], row["grasp"]["ci_str"],
                row["grasp"]["f1"], row["grasp"]["sim_delta"],
                s.get("p_value","N/A"), s.get("stars","N/A"),
            ])

    with open(RESULTS_DIR / "exp1_stats.json", "w") as f:
        json.dump(stats_table, f, indent=2)

    logger.info("Experiment 1 complete: %d conditions", len(all_results))
    return output


main = run_experiment1   # alias for master runner

if __name__ == "__main__":
    run_experiment1()
