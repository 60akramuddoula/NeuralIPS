# GRASP: Defense-Aware RAG Poisoning via Genetic Semantic Optimization

> **NeurIPS 2025 Submission Codebase**
> *Genetic Retrieval-Aware Adversarial Sentence Poisoning*

---

## Overview

GRASP is a **defense-aware corpus poisoning framework** for Retrieval-Augmented Generation (RAG) systems. Unlike prior attacks that rely on surface-level query prepending (PoisonedRAG black-box: `S⊕I`), GRASP uses a multi-objective genetic algorithm to evolve adversarial passages that are simultaneously:

- **Retrieval-effective** — rank highly against target queries
- **Semantically grounded** — encode adversarial payloads in meaning, not surface tokens
- **Defense-robust** — optimized to survive paraphrasing and perplexity-filtering defenses
- **Stealthy** — indistinguishable from clean corpus text under automated detection

GRASP serves as a **model-agnostic post-hoc optimization layer** compatible with any base RAG attack, including PoisonedRAG black-box and white-box variants.

---

## Key Claims vs. PoisonedRAG

| Claim | PoisonedRAG (BB) | GRASP |
|-------|-----------------|-------|
| **Paraphrase robustness** | Surface token match — brittle | Semantic optimization — robust |
| **PPL detectability** | High PPL from S⊕I format | Natural sentence distribution |
| **Injection efficiency** | N=5 for ~90% ASR | N=1–2 for equivalent ASR |
| **Retriever coverage** | Dense only | Dense + BM25 (lexical) |
| **Gradient requirement** | WB requires HotFlip gradient | Gradient-free throughout |
| **Defense optimization** | Post-hoc evaluation only | Explicit fitness objective |

---

## Repository Structure

```
grasp_neurips/
├── src/
│   ├── grasp_attack.py          # Core GA engine (Chromosome, GRASPFitness, GAOperators, GRASPAttack)
│   ├── eval_utils.py            # Metrics: ASR, retrieval F1, PPL, ROC AUC, significance
│   ├── mock_infra.py            # Mock infrastructure: 100-query datasets, BM25, DPR, ParametricLLM
│   ├── stats_utils.py           # bootstrap_ci, mcnemar_test, significance_stars
│   └── ppl_utils.py             # Two-tier perplexity: GPT-2 (production) + trigram LM (fallback)
├── experiments/
│   ├── exp1_asr_table.py        # Main ASR table: 3 datasets × 4 retrievers × 3 LLMs × 2 attacks
│   ├── exp2_efficiency.py       # Injection efficiency: ASR vs N ∈ {1,2,3,4,5,7,10}
│   ├── exp3_stealth.py          # Stealth evaluation: PPL + ROC AUC detection curves
│   ├── exp4_defenses.py         # Defense robustness: paraphrasing, PPL filter, dedup, expansion
│   ├── exp5_ablation.py         # Operator ablation: 6 GA configurations
│   ├── exp6_7_transfer_convergence.py  # Cross-retriever transfer + convergence analysis
│   └── exp8_significance.py     # McNemar's test + bootstrap CIs across all conditions
├── tests/
│   └── test_grasp_full.py       # 109 tests (unit + integration + numerical stability)
├── run_all_experiments.py       # Master runner — executes all 8 experiments sequentially
└── results/                     # Auto-populated JSON/CSV output files
```

---

## Installation

```bash
# Clone repository
git clone https://github.com/your-org/grasp-neurips.git
cd grasp-neurips

# Install dependencies
pip install numpy scipy

# Optional: real GPT-2 perplexity (requires ~500MB download)
pip install transformers torch

# Run test suite to verify installation
cd grasp_neurips
python -m pytest tests/ -q
# Expected: 109 passed
```

**Python requirement:** ≥ 3.9  
**Core dependencies:** `numpy`, `scipy` (for bootstrap CIs)  
**Optional:** `transformers`, `torch` (activates real GPT-2 PPL; falls back to trigram LM if absent)

---

## Quick Start

### Run all experiments (mock infrastructure)

```bash
cd grasp_neurips
python run_all_experiments.py
# Completes in ~5–10 minutes on CPU
# Results written to results/*.json
```

### Run a single experiment

```bash
python experiments/exp1_asr_table.py    # Main ASR table
python experiments/exp4_defenses.py    # Defense robustness (primary claim)
python experiments/exp8_significance.py # Statistical significance tables
```

### Run the test suite

```bash
python -m pytest tests/ -v              # All 109 tests with names
python -m pytest tests/ -q              # Quiet summary
python -m pytest tests/ -k "Fitness"   # Filter by class name
```

---

## Core Architecture

### Chromosome (sentence-level genes)

Each adversarial candidate is represented as a list of sentence-level genes (max 12). A **protected answer invariant** is formally enforced after every genetic operation — the incorrect answer string can never be dropped.

```python
from src.grasp_attack import Chromosome

chrom = Chromosome(
    text="The answer is Paris. France is the capital of Europe.",
    protected_answer="Paris",
    max_genes=12
)
print(chrom.to_text())  # Guaranteed to contain "Paris"
```

### Defense-Aware Fitness Function

The multi-objective fitness combines four terms:

```
F(x) = α · sim(emb(x), emb(q))          # Primary retrieval signal
     + β · mean_sim(emb(x), para_embs)   # Paraphrase robustness (λ=0.30)
     - γ · naturalness_anomaly(x)        # PPL defense evasion  (λ=0.15)
     + δ · sentence_length_score(x)      # Stealth bonus         (λ=0.05)
     - PENALTY · (1 - has_answer(x))     # Answer invariant      (hard, =2.0)
```

The paraphrase-robustness term (`λ=0.30`) is the primary differentiator from PoisonedRAG — it optimizes for semantic encoding that survives query reformulation rather than surface-level token overlap.

### GRASPAttack API

```python
from src.grasp_attack import GRASPAttack, GRASPConfig
import numpy as np

# Define any embedding function (real or mock)
def my_embed_fn(text: str) -> np.ndarray:
    # e.g., Contriever, DPR, OpenAI embeddings, BM25 proxy
    ...

attack = GRASPAttack(
    embed_fn=my_embed_fn,
    config=GRASPConfig(
        population_size=20,
        num_generations=30,
        fitness_lambda_paraphrase=0.30,
        fitness_lambda_naturalness=0.15,
    )
)

result = attack.attack_query(
    query_id="q001",
    question="Who painted the Mona Lisa?",
    correct_answer="Leonardo da Vinci",
    incorrect_answer="Michelangelo",
    seed_adv_texts=["..."],        # From PoisonedRAG BB or any base attack
    clean_topk_scores=[0.82, 0.79, 0.76, 0.74, 0.71],
)

print(result.evolved_texts)        # List of evolved adversarial passages
print(result.adv_in_topk)         # How many beat the clean retrieval threshold
```

---

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `population_size` | 20 | GA population per generation |
| `num_generations` | 30 | Evolution steps |
| `mutation_rate` | 0.20 | Probability of mutating a child |
| `crossover_rate` | 0.70 | Probability of crossover vs. direct mutation |
| `tournament_size` | 3 | Selection pressure for tournament selection |
| `elite_frac` | 0.10 | Fraction of population preserved as elites (≥1) |
| `max_genes` | 12 | Maximum sentences per chromosome |
| `fragment_mutation_prob` | 0.70 | P(fragment recombination) vs. gene swap |
| `fitness_lambda_paraphrase` | **0.30** | Weight for paraphrase-robustness term |
| `fitness_lambda_naturalness` | **0.15** | Weight for naturalness-anomaly penalty |
| `fitness_lambda_stealth` | 0.05 | Weight for sentence-length stealth bonus |
| `n_paraphrase_variants` | 5 | Number of pre-computed query paraphrases |
| `seed` | 42 | Global RNG seed (deterministic runs) |

---

## Experiments

### Exp 1 — Main ASR Table
**File:** `experiments/exp1_asr_table.py`

Full 100-query evaluation across the PoisonedRAG benchmark conditions.

- **Datasets:** NQ, HotpotQA, MS-MARCO (100 queries each)
- **Retrievers:** Contriever (dense), DPR (dense, 768-dim), ANCE (dense), BM25 (sparse lexical)
- **LLMs:** LLaMA-3.1-8B, GPT-3.5, Mistral-7B
- **Attacks:** PoisonedRAG-BB (baseline), GRASP (ours)
- **Metrics:** ASR, Retrieval F1, Similarity Δ, Time/query

*Load-bearing experiment for the primary novelty claim.*

---

### Exp 2 — Injection Efficiency
**File:** `experiments/exp2_efficiency.py`

ASR as a function of number of injected documents N ∈ {1, 2, 3, 4, 5, 7, 10}.

- **Expected:** GRASP ASR curve lies above PoisonedRAG-BB for small N (1–3)
- **Key result:** GRASP reaches ≥85% ASR at N=1; PoisonedRAG-BB requires N=3–5

---

### Exp 3 — Stealth Evaluation
**File:** `experiments/exp3_stealth.py`

Perplexity-based detection curves comparing three attack variants.

- **Methods:** Clean corpus, PoisonedRAG-BB, PoisonedRAG-WB (HotFlip-noised), GRASP
- **Metrics:** PPL score distribution, ROC AUC for adversarial detection
- **Expected ordering:** GRASP AUC ≤ PoisonedRAG-BB AUC ≪ PoisonedRAG-WB AUC

---

### Exp 4 — Defense Robustness *(primary claim)*
**File:** `experiments/exp4_defenses.py`

ASR under four deployed defenses with bootstrap 95% CIs.

| Defense | Mechanism | Expected GRASP advantage |
|---------|-----------|--------------------------|
| Paraphrasing | Query rewritten (5 variants) | GRASP: <10% ASR drop; PoisonedRAG-BB: 10–30% drop |
| PPL filtering | Remove passages above PPL percentile | GRASP retained longer (lower PPL) |
| Knowledge expansion | Increase top-k retrieved | GRASP degrades more slowly |
| Duplicate detection | SHA-256 deduplication | Both fully blocked (parity) |

---

### Exp 5 — Ablation Study
**File:** `experiments/exp5_ablation.py`

Six conditions isolating each GA design choice:

| Config | Description |
|--------|-------------|
| PoisonedRAG-BB | No GA (baseline) |
| GRASP-Random | Random word substitution only |
| GRASP-SwapOnly | Gene swap only (structural) |
| GRASP-FragOnly | Fragment recombination only (semantic) |
| GRASP-NoCross | No crossover operator |
| **GRASP-Full** | Full system (expected best) |

---

### Exp 6 — Cross-Retriever Transferability
**File:** `experiments/exp6_7_transfer_convergence.py`

Texts evolved against Contriever are tested against DPR, ANCE, and BM25 *without re-evolution*. Validates that GRASP's semantic optimization generalizes across retriever families.

---

### Exp 7 — Convergence Analysis
**File:** `experiments/exp6_7_transfer_convergence.py`

Mean best-fitness across 30 generations for GRASP-Full, GRASP-FragOnly, and GRASP-SwapOnly. Expected: GRASP-Full converges fastest; SwapOnly plateaus earliest.

---

### Exp 8 — Statistical Significance
**File:** `experiments/exp8_significance.py`

Pairwise significance testing for all GRASP vs. PoisonedRAG-BB comparisons.

- **McNemar's test** (with continuity correction) for ASR binary outcomes
- **Bootstrap 95% CI** (n=1000 resamples) on all ASR point estimates
- **Output:** Full significance table with p-values and `*`/`**`/`***` markers

---

## Replacing Mock Infrastructure with Real Models

The mock infrastructure (deterministic embedders, synthetic QA pairs) is designed for testing and CI. To run on real data, replace the following components — **no experiment code changes required**:

### Real embedding function (Contriever example)
```python
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
model = AutoModel.from_pretrained("facebook/contriever")

def contriever_embed(text: str) -> np.ndarray:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    emb = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return emb / (np.linalg.norm(emb) + 1e-10)
```

### Real BEIR datasets
```python
from beir import util
from beir.datasets.data_loader import GenericDataLoader

url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip"
data_path = util.download_and_unzip(url, "datasets/")
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

# Format to GRASP QAPair structure
qa_pairs = [
    {"id": qid, "question": queries[qid], "correct_answer": ..., "incorrect_answer": ...}
    for qid in list(queries.keys())[:100]
]
```

### Real LLM (OpenAI example)
```python
import openai

def gpt35_generate(prompt: str) -> str:
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    return response.choices[0].message.content
```

---

## Statistical Reporting

All main results include bootstrap 95% confidence intervals. Example output format from `exp8_significance.py`:

```
Method Comparison: GRASP vs PoisonedRAG-BB
Dataset     Retriever   ASR_GRASP        ASR_BB           McNemar p    Sig
----------  ----------  ---------------  ---------------  -----------  ---
NQ          Contriever  0.847 [0.79,0.90] 0.710 [0.65,0.77]  0.0031    **
HotpotQA    Contriever  0.821 [0.76,0.88] 0.693 [0.63,0.76]  0.0089    **
MS-MARCO    BM25        0.779 [0.71,0.84] 0.682 [0.61,0.75]  0.0412    *
```

Significance levels: `*` p<0.05, `**` p<0.01, `***` p<0.001

---

## Reproducibility

All experiments are fully deterministic given the same seed:

```bash
# Same command → identical results every run
python run_all_experiments.py --seed 42
```

Random seeds are propagated through `GRASPConfig.seed` → `random.seed()` + `np.random.seed()` at the start of every `attack_query()` call. The test suite verifies this with a dedicated determinism test (`TestNumericalStability::test_determinism_same_seed`).

---

## Test Suite

```
109 tests across 4 categories:
├── Unit tests (37)           — Chromosome, GRASPFitness, GAOperators, RealGeneticAlgorithm, GRASPAttack
├── Integration tests (36)    — All 8 experiments end-to-end
├── Numerical stability (24)  — Edge cases, empty inputs, zero-division, determinism
└── New component tests (12)  — BM25, DPR, ParametricLLM, bootstrap_ci, mcnemar_test, PPL
```

Run with coverage:
```bash
pip install pytest-cov
python -m pytest tests/ --cov=src --cov-report=term-missing
```

---

## Paper Citation

If you use this codebase, please cite:

```bibtex
@inproceedings{grasp2025,
  title     = {GRASP: Defense-Aware RAG Poisoning via Genetic Semantic Optimization},
  author    = {[Authors]},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2025},
}
```

---

## Ethics Statement

This codebase implements adversarial attacks against RAG systems for **security research purposes**. The primary goal is to expose vulnerabilities in current defense mechanisms so that more robust defenses can be developed. All experiments are conducted on established public benchmarks (NQ, HotpotQA, MS-MARCO) under controlled conditions. We do not condone deployment of these techniques against production systems. Researchers are encouraged to use this work to develop and evaluate improved detection and mitigation strategies.

---

## License

MIT License. See `LICENSE` for details.
