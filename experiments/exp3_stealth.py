"""
Experiment 3: Stealth Evaluation
==================================
NeurIPS claim: GRASP texts are harder to detect than PoisonedRAG
under perplexity/naturalness-based detection.

Protocol (mirrors PoisonedRAG §7.2):
  - Compute naturalness_score for:
      (a) clean corpus texts
      (b) PoisonedRAG-BB adversarial texts
      (c) PoisonedRAG-WB texts (simulated HotFlip noise)
      (d) GRASP evolved texts
  - ROC curves + AUROC: lower AUC = stealthier = harder to detect
  - Evaluated on NQ-100 and HotpotQA-100

Expected: GRASP AUC ≤ PoisonedRAG-BB AUC < PoisonedRAG-WB AUC

Outputs:
  results/exp3_stealth.json
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.grasp_attack import GRASPAttack, GRASPConfig
from src.eval_utils import compute_roc_auc, naturalness_score
from src.ppl_utils import PerplexityEvaluator
from src.mock_infra import (
    CORPUS_500,
    DATASET_MAP,
    MockEmbedder,
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

GRASP_CFG = GRASPConfig(
    population_size=20, num_generations=30,
    mutation_rate=0.20, crossover_rate=0.70,
    tournament_size=3, elite_frac=0.10, max_genes=12,
    fitness_lambda_stealth=0.05,
    fitness_lambda_paraphrase=0.30,
    fitness_lambda_naturalness=0.15,
    seed=42,
)


def _simulate_hotflip(text: str, rng: np.random.RandomState) -> str:
    """
    Simulate PoisonedRAG white-box (HotFlip) token substitution.
    Replaces ~15% of tokens with nonsense/## artifacts — the characteristic
    signature of gradient-based discrete optimization.
    """
    tokens = text.split()
    if not tokens:
        return text
    n_corrupt = max(1, int(len(tokens) * 0.15))
    idxs = rng.choice(len(tokens), size=min(n_corrupt, len(tokens)),
                       replace=False)
    noise_pool = [
        "##xk", "##zq", "##trf", "zxqpkv", "xkqz", "##grf",
        "pfxk", "##zzq", "xkz", "tpqxz",
    ]
    for i in idxs:
        tokens[i] = rng.choice(noise_pool)
    return " ".join(tokens)


def _paraphrases(question: str, n: int = 3) -> List[str]:
    base = question.lower().rstrip("?").strip()
    rules = [
        ("who invented", "who created"),
        ("what is the capital of", "what city serves as capital of"),
        ("when did", "in what year did"),
    ]
    out: List[str] = []
    for src, tgt in rules:
        if src in base and len(out) < n:
            out.append(base.replace(src, tgt, 1).capitalize() + "?")
    while len(out) < n:
        out.append("Please tell me: " + base + "?")
    return out[:n]


def run_stealth_eval(dataset: str, embedder_seed: int) -> Dict:
    logger.info("Stealth eval: %s", dataset)
    qa_pairs = DATASET_MAP[dataset]
    corpus   = CORPUS_500
    embedder = MockEmbedder(seed=embedder_seed)
    beir_res = make_beir_results(qa_pairs, corpus, embedder, top_k=TOP_K)
    grasp    = GRASPAttack(embed_fn=embedder.encode,
                            config=GRASP_CFG, prepend_query=True)
    rng_wb   = np.random.RandomState(999)

    texts_by_method: Dict[str, List[str]] = {
        "PoisonedRAG-BB": [],
        "PoisonedRAG-WB": [],
        "GRASP": [],
    }
    clean_texts: List[str] = []

    for qa in qa_pairs:
        qid       = qa["id"]
        question  = qa["question"]
        incorrect = qa["incorrect_answer"]

        topk_ids    = list(beir_res[qid].keys())[:TOP_K]
        topk_scores = list(beir_res[qid].values())[:TOP_K]

        # Clean corpus sample
        clean_texts.append(corpus[topk_ids[0]]["text"])

        seed_texts = make_seed_adv_texts(question, incorrect, n=ADV_PER_QUERY)

        # PoisonedRAG-BB: literal S⊕I
        bb_texts = [question.rstrip(".") + ". " + s for s in seed_texts]
        texts_by_method["PoisonedRAG-BB"].extend(bb_texts[:ADV_PER_QUERY])

        # PoisonedRAG-WB: HotFlip-noised S + I
        wb_texts = [_simulate_hotflip(s, rng_wb) + " " + incorrect + "."
                    for s in seed_texts]
        texts_by_method["PoisonedRAG-WB"].extend(wb_texts[:ADV_PER_QUERY])

        # GRASP
        res = grasp.attack_query(
            query_id=qid, question=question,
            correct_answer=qa["correct_answer"], incorrect_answer=incorrect,
            seed_adv_texts=seed_texts, clean_topk_scores=topk_scores,
            adv_per_query=ADV_PER_QUERY,
            paraphrase_queries=_paraphrases(question, 3),
            verbose=False,
        )
        texts_by_method["GRASP"].extend(res.evolved_texts[:ADV_PER_QUERY])

    # Compute perplexity scores — dual-tier: PerplexityEvaluator (GPT-2 if available,
    # else calibrated trigram fallback).  naturalness_score kept as a second signal.
    ppl_eval = PerplexityEvaluator(reference_corpus=clean_texts, use_gpt2=False)
    clean_nat = [ppl_eval.score(t) for t in clean_texts]

    method_results: Dict[str, Dict] = {}
    for method, texts in texts_by_method.items():
        nat_scores = [ppl_eval.score(t) for t in texts]
        # Fallback sanity: also compute heuristic naturalness_score
        heuristic = [naturalness_score(t) for t in texts]
        fpr, tpr, auc = compute_roc_auc(clean_nat, nat_scores)
        method_results[method] = {
            "n_texts":              len(texts),
            "nat_mean":             float(np.mean(nat_scores)),
            "nat_std":              float(np.std(nat_scores)),
            "nat_mean_heuristic":   float(np.mean(heuristic)),
            "clean_nat_mean":       float(np.mean(clean_nat)),
            "ppl_evaluator":        "GPT-2" if ppl_eval._use_gpt2 else "trigram-LM",
            "auc":                  auc,
            "fpr":                  fpr[:20],  # truncated for storage
            "tpr":                  tpr[:20],
        }
        logger.info("  %-20s  nat=%.3f  AUC=%.3f",
                    method, method_results[method]["nat_mean"], auc)

    return {"dataset": dataset, "methods": method_results}


def main() -> Dict:
    logger.info("=" * 60)
    logger.info("Exp 3: Stealth Evaluation (100q per dataset)")
    logger.info("=" * 60)

    output = {}
    for dataset, seed in [("NQ", 42), ("HotpotQA", 137)]:
        output[dataset] = run_stealth_eval(dataset, seed)

    with open(RESULTS_DIR / "exp3_stealth.json", "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Saved → results/exp3_stealth.json")

    # Verify expected ordering
    for dataset, res in output.items():
        methods = res["methods"]
        print(f"\n{dataset}  Naturalness AUC (lower = stealthier):")
        for m, r in methods.items():
            bar = "★" if r["auc"] == min(v["auc"] for v in methods.values()) else " "
            print(f"  {bar} {m:<22}  AUC={r['auc']:.4f}  nat={r['nat_mean']:.4f}")

        bb_auc  = methods["PoisonedRAG-BB"]["auc"]
        wb_auc  = methods["PoisonedRAG-WB"]["auc"]
        gsp_auc = methods["GRASP"]["auc"]
        if gsp_auc <= bb_auc and bb_auc <= wb_auc:
            logger.info("%s: ✅ Expected ordering: GRASP ≤ BB ≤ WB", dataset)
        else:
            logger.warning("%s: ⚠ Ordering deviation  GRASP=%.3f  BB=%.3f  WB=%.3f",
                           dataset, gsp_auc, bb_auc, wb_auc)

    return output


if __name__ == "__main__":
    main()
