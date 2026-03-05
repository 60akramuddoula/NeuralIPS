"""
GRASP: Genetic Retrieval-Aware Adversarial Sentence Poisoning
=============================================================
NeurIPS submission — core attack implementation.

Architecture:
  Chromosome       : sentence-level genes with formally enforced answer invariant
  GRASPFitness     : DEFENSE-AWARE multi-objective fitness (retrieval + stealth
                     + paraphrase-robustness + naturalness penalty)
  GAOperators      : fragment recombination + gene swap + crossover
  RealGeneticAlgorithm : elitist tournament GA
  GRASPAttack      : model-agnostic orchestrator; works as post-hoc optimizer
                     over ANY base attack's seed texts

Primary novelty claim (defense-aware semantic poisoning):
  F(x) = sim(x, q)                           [retrieval objective]
        + λ_p * mean_k sim(x, paraphrase_k)  [paraphrase-robustness term]
        - λ_n * naturalness_anomaly(x)        [naturalness penalty]
        - PENALTY * (1 - has_answer(x))       [answer hard constraint]

  The paraphrase-robustness term is the key differentiator from PoisonedRAG-BB,
  which prepends the literal query string (S⊕I) and thus relies on surface-form
  lexical overlap — brittle against paraphrasing defenses.  GRASP instead evolves
  texts that score well across MULTIPLE semantic variants of the query.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import random
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GRASPConfig:
    """Hyperparameters for the GRASP genetic algorithm."""
    population_size:       int   = 20
    num_generations:       int   = 30
    mutation_rate:         float = 0.20
    crossover_rate:        float = 0.70
    tournament_size:       int   = 3
    elite_frac:            float = 0.10
    max_genes:             int   = 12
    fragment_mutation_prob:float = 0.70

    # Fitness weights
    fitness_lambda_stealth:  float = 0.05   # sentence-length naturalness reward
    fitness_lambda_paraphrase: float = 0.30  # paraphrase-robustness term weight
    fitness_lambda_naturalness:float = 0.15  # naturalness-anomaly penalty weight

    # Paraphrase variants used in fitness (pre-computed deterministically)
    n_paraphrase_variants: int = 5

    seed: int = 42


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvoResult:
    """Result for a single evolved adversarial text."""
    original_text:       str
    evolved_text:        str
    original_sim:        float
    evolved_sim:         float
    delta_sim:           float
    paraphrase_sim_mean: float
    has_answer:          bool
    generation_converged:int
    fitness_history:     List[float] = field(default_factory=list)
    time_seconds:        float = 0.0


@dataclass
class AttackResult:
    """Full attack result for one target query."""
    query_id:             str
    question:             str
    correct_answer:       str
    incorrect_answer:     str
    evolved_texts:        List[str]
    evo_results:          List[EvoResult]
    clean_threshold:      float
    evolved_sims:         List[float]
    adv_in_topk:          int
    attack_success:       Optional[bool] = None
    llm_response_clean:   Optional[str] = None
    llm_response_poisoned:Optional[str] = None
    total_time_seconds:   float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Paraphrase Generator (deterministic, no external API)
# ─────────────────────────────────────────────────────────────────────────────

_PARAPHRASE_RULES: List[Tuple[str, str]] = [
    (r"\bwho\b",          "which person"),
    (r"\bwhen\b",         "at what time"),
    (r"\bwhat is\b",      "what represents"),
    (r"\bwhat was\b",     "what had been"),
    (r"\bwhere\b",        "in what location"),
    (r"\bhow\b",          "in what way"),
    (r"\binvented\b",     "created"),
    (r"\bdiscovered\b",   "found"),
    (r"\bwrote\b",        "authored"),
    (r"\bborn\b",         "originally from"),
    (r"\bfounded\b",      "established"),
    (r"\bthe capital of\b","the governmental seat of"),
    (r"\blargest\b",      "biggest"),
    (r"\bsmallest\b",     "tiniest"),
    (r"\bfirst\b",        "initial"),
    (r"\bmain\b",         "primary"),
    (r"\bused\b",         "employed"),
    (r"\bknown as\b",     "referred to as"),
    (r"\bcalled\b",       "named"),
    (r"\bbuilt\b",        "constructed"),
]


def generate_paraphrases(question: str, n: int = 5, seed: int = 42) -> List[str]:
    """
    Generate n deterministic paraphrase variants of a question.
    Uses rule-based substitutions at different positions.
    Returns original + (n-1) paraphrased variants.
    """
    variants: List[str] = [question]
    rng = random.Random(seed)
    q_lower = question.lower()

    # Shuffle rule order per seed for diversity across queries
    rules = list(_PARAPHRASE_RULES)
    rng.shuffle(rules)

    for i, (pattern, replacement) in enumerate(rules):
        if len(variants) >= n:
            break
        if re.search(pattern, q_lower):
            new_q = re.sub(pattern, replacement, question, flags=re.IGNORECASE, count=1)
            if new_q != question and new_q not in variants:
                variants.append(new_q)

    # Fill remaining slots with word-order variants
    words = question.split()
    while len(variants) < n and len(words) >= 4:
        # Move a random non-first word to end (syntactic reshuffling)
        idx = rng.randint(1, len(words) - 1)
        reordered = " ".join(words[:idx] + words[idx + 1:] + [words[idx]])
        if reordered not in variants:
            variants.append(reordered)
        break  # avoid infinite loop on small questions

    # Pad with original if still short
    while len(variants) < n:
        variants.append(question)

    return variants[:n]


# ─────────────────────────────────────────────────────────────────────────────
# Chromosome (sentence-level genes)
# ─────────────────────────────────────────────────────────────────────────────

class Chromosome:
    """
    Represents one adversarial candidate text as a list of sentence-level genes.

    Invariant: the protected_answer string must appear (case-insensitive) in
    at least one gene at all times.  Every mutation operator enforces this.
    """

    def __init__(self, text: str, protected_answer: str, max_genes: int = 12):
        self.protected_answer = protected_answer.lower().rstrip(".")
        self.max_genes = max_genes
        self.genes: List[str] = self._split(text)
        self._fitness: Optional[float] = None

    def _split(self, text: str) -> List[str]:
        raw = re.split(r"(?<=[.!?])\s+", text.strip())
        genes = [s.strip() for s in raw if s.strip()]
        return genes[: self.max_genes] if genes else [text.strip()]

    def _has_answer(self) -> bool:
        return self.protected_answer in self.to_text().lower()

    def _ensure_answer(self, fallback_sentence: str = "") -> None:
        """If answer dropped, append sentinel sentence containing it."""
        if not self._has_answer():
            sentinel = f"Specifically, the answer is {self.protected_answer.capitalize()}."
            if len(self.genes) >= self.max_genes:
                self.genes[-1] = sentinel
            else:
                self.genes.append(sentinel)

    def to_text(self) -> str:
        return " ".join(self.genes)

    def copy(self) -> "Chromosome":
        c = Chromosome.__new__(Chromosome)
        c.protected_answer = self.protected_answer
        c.max_genes = self.max_genes
        c.genes = list(self.genes)
        c._fitness = None
        return c

    def invalidate_cache(self) -> None:
        self._fitness = None

    def __len__(self) -> int:
        return len(self.genes)

    def __repr__(self) -> str:
        return f"Chromosome({len(self.genes)} genes, answer={self._has_answer()})"


# ─────────────────────────────────────────────────────────────────────────────
# Defense-Aware Fitness Function
# ─────────────────────────────────────────────────────────────────────────────

class GRASPFitness:
    """
    Defense-aware multi-objective fitness function.

    F(x) = sim(emb(x), emb(q))                          [retrieval objective]
          + λ_p * (1/K) Σ_k sim(emb(x), emb(pq_k))     [paraphrase-robustness]
          - λ_n * naturalness_anomaly(x)                  [naturalness penalty]
          + λ_s * sentence_length_score(x)               [stealth bonus]
          - PENALTY * (1 − has_answer(x))                 [hard constraint]

    Paraphrase-robustness term (λ_p):
        PoisonedRAG-BB prepends the literal query string (S⊕I), achieving high
        retrieval similarity by token overlap.  This term drives GRASP to produce
        texts that score well across MULTIPLE semantic variants of the query,
        not just the original surface form.  Under a paraphrasing defense (which
        rewrites the query before retrieval), GRASP-optimized texts degrade far
        less than PoisonedRAG-BB texts.

    Naturalness penalty (λ_n):
        Penalizes texts with high anomaly scores (HotFlip ## tokens, repetition,
        irregular sentence lengths), incentivizing human-like prose that evades
        perplexity-based filters.
    """

    ANSWER_PENALTY = 2.0

    def __init__(
        self,
        embed_fn: Callable[[str], np.ndarray],
        query: str,
        incorrect_answer: str,
        paraphrase_queries: Optional[List[str]] = None,
        lambda_stealth: float = 0.05,
        lambda_paraphrase: float = 0.30,
        lambda_naturalness: float = 0.15,
    ):
        self.embed_fn = embed_fn
        self.incorrect_answer = incorrect_answer.lower().rstrip(".")
        self.lambda_stealth = lambda_stealth
        self.lambda_paraphrase = lambda_paraphrase
        self.lambda_naturalness = lambda_naturalness

        # Pre-compute query embedding
        self.query_emb: np.ndarray = embed_fn(query)

        # Pre-compute paraphrase embeddings (defense-robustness term)
        pqs = paraphrase_queries or [query]
        self.paraphrase_embs: List[np.ndarray] = [embed_fn(pq) for pq in pqs]

    def _sentence_length_score(self, text: str) -> float:
        """Reward texts with mean sentence length near natural prose (18 words)."""
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        word_counts = [len(s.split()) for s in sentences if s.strip()]
        if not word_counts:
            return 0.0
        mean_wc = float(np.mean(word_counts))
        return float(math.exp(-((mean_wc - 18) ** 2) / (2 * 10 ** 2)))

    def _naturalness_anomaly(self, text: str) -> float:
        """
        Multi-signal naturalness anomaly score (higher = less natural).
        Inline implementation avoids circular import with eval_utils.
        """
        if not text.strip():
            return 0.0
        tokens = text.lower().split()
        if not tokens:
            return 0.0

        signals: List[float] = []

        # HotFlip ## artifacts
        hash_ratio = sum(1 for t in tokens if t.startswith("##")) / len(tokens)
        signals.append(min(hash_ratio * 5.0, 1.0))

        # Non-alpha character ratio
        non_alpha = sum(
            sum(1 for c in t if not c.isalpha()) / max(len(t), 1)
            for t in tokens
        ) / len(tokens)
        signals.append(min(non_alpha * 2.0, 1.0))

        # Token repetition CV
        from collections import Counter
        freq = Counter(tokens)
        counts_list = list(freq.values())
        if len(counts_list) > 1:
            mean_c = sum(counts_list) / len(counts_list)
            std_c = (sum((x - mean_c) ** 2 for x in counts_list) / len(counts_list)) ** 0.5
            cv = std_c / (mean_c + 1e-9)
            signals.append(min(cv * 0.3, 1.0))
        else:
            signals.append(0.0)

        # Sentence length CV
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
        if len(sents) > 1:
            lens = [len(s.split()) for s in sents]
            m = sum(lens) / len(lens)
            s_std = (sum((x - m) ** 2 for x in lens) / len(lens)) ** 0.5
            signals.append(min(s_std / max(m, 1) * 0.5, 1.0))
        else:
            signals.append(0.0)

        return float(sum(signals) / len(signals))

    def __call__(self, chromosome: Chromosome) -> float:
        if chromosome._fitness is not None:
            return chromosome._fitness

        text = chromosome.to_text()
        emb = self.embed_fn(text)

        # 1. Primary retrieval similarity
        primary_sim = float(np.dot(self.query_emb, emb))

        # 2. Paraphrase-robustness: mean sim across paraphrase variants
        if self.paraphrase_embs:
            para_sims = [float(np.dot(p_emb, emb)) for p_emb in self.paraphrase_embs]
            para_mean = float(np.mean(para_sims))
        else:
            para_mean = primary_sim

        # 3. Naturalness penalty
        nat_anomaly = self._naturalness_anomaly(text)

        # 4. Sentence-length stealth bonus
        stealth = self._sentence_length_score(text)

        # 5. Hard answer constraint
        has_answer = self.incorrect_answer in text.lower()
        penalty = 0.0 if has_answer else self.ANSWER_PENALTY

        score = (
            primary_sim
            + self.lambda_paraphrase * para_mean
            - self.lambda_naturalness * nat_anomaly
            + self.lambda_stealth * stealth
            - penalty
        )
        chromosome._fitness = score
        return score

    def retrieval_sim(self, text: str) -> float:
        emb = self.embed_fn(text)
        return float(np.dot(self.query_emb, emb))

    def paraphrase_sim_mean(self, text: str) -> float:
        emb = self.embed_fn(text)
        if not self.paraphrase_embs:
            return self.retrieval_sim(text)
        return float(np.mean([np.dot(p_emb, emb) for p_emb in self.paraphrase_embs]))


# ─────────────────────────────────────────────────────────────────────────────
# Genetic Operators
# ─────────────────────────────────────────────────────────────────────────────

class GAOperators:
    """
    Two complementary mutation operators:

    1. Fragment Recombination (70% probability):
       Build a pool of sentence fragments from ALL seed adversarial texts.
       Score each fragment by TF contribution to query tokens + embedding sim.
       Replace a random non-answer gene with the highest-scoring unused fragment.

    2. Gene Swap (30%):
       Randomly permute two genes (sentence reordering for coherence).

    Crossover: single-point at gene level.
    """

    def __init__(
        self,
        embed_fn: Callable[[str], np.ndarray],
        seed_texts: List[str],
        query: str,
        incorrect_answer: str,
        fragment_mutation_prob: float = 0.70,
        max_genes: int = 12,
    ):
        self.embed_fn = embed_fn
        self.query = query
        self.incorrect_answer = incorrect_answer.lower().rstrip(".")
        self.fragment_mutation_prob = fragment_mutation_prob
        self.max_genes = max_genes

        # Pre-compute query embedding BEFORE scoring fragments
        self.query_emb: np.ndarray = embed_fn(query)
        self.query_tokens: set = set(query.lower().split())

        # Build and score fragment pool
        self.fragment_pool: List[str] = self._build_fragment_pool(seed_texts)
        self.fragment_scores: List[float] = [
            self._score_fragment(f) for f in self.fragment_pool
        ]
        logger.debug("GAOperators: %d fragments in pool", len(self.fragment_pool))

    def _build_fragment_pool(self, texts: List[str]) -> List[str]:
        pool: set = set()
        for text in texts:
            for s in re.split(r"(?<=[.!?])\s+", text.strip()):
                s = s.strip()
                if s and len(s.split()) >= 3:
                    pool.add(s)
        return list(pool)

    def _score_fragment(self, fragment: str) -> float:
        tokens = set(fragment.lower().split())
        overlap = len(tokens & self.query_tokens)
        emb = self.embed_fn(fragment)
        sim = float(np.dot(self.query_emb, emb))
        return overlap * 0.3 + sim

    def _best_fragment(self, existing_genes: List[str]) -> str:
        existing_set = {g.lower() for g in existing_genes}
        best_score = -999.0
        best_frag = self.fragment_pool[0] if self.fragment_pool else ""
        for frag, score in zip(self.fragment_pool, self.fragment_scores):
            if frag.lower() not in existing_set and score > best_score:
                best_score = score
                best_frag = frag
        return best_frag

    # ── mutation ─────────────────────────────────────────────────────────────

    def mutate(self, chromo: Chromosome) -> Chromosome:
        chromo.invalidate_cache()
        if random.random() < self.fragment_mutation_prob:
            return self.mutate_fragment_recombine(chromo)
        return self.mutate_gene_swap(chromo)

    def mutate_fragment_recombine(self, chromo: Chromosome) -> Chromosome:
        if not chromo.genes or not self.fragment_pool:
            return chromo
        replaceable = [
            i for i, g in enumerate(chromo.genes)
            if self.incorrect_answer not in g.lower()
        ]
        if not replaceable:
            if len(chromo.genes) < chromo.max_genes:
                chromo.genes.append(self._best_fragment(chromo.genes))
        else:
            idx = random.choice(replaceable)
            chromo.genes[idx] = self._best_fragment(chromo.genes)
            chromo._ensure_answer()
        chromo.invalidate_cache()
        return chromo

    def mutate_gene_swap(self, chromo: Chromosome) -> Chromosome:
        if len(chromo.genes) < 2:
            return chromo
        i, j = random.sample(range(len(chromo.genes)), 2)
        chromo.genes[i], chromo.genes[j] = chromo.genes[j], chromo.genes[i]
        chromo.invalidate_cache()
        return chromo

    # ── crossover ────────────────────────────────────────────────────────────

    def crossover(
        self, parent_a: Chromosome, parent_b: Chromosome
    ) -> Tuple[Chromosome, Chromosome]:
        genes_a, genes_b = parent_a.genes, parent_b.genes
        if len(genes_a) < 2 or len(genes_b) < 2:
            return parent_a.copy(), parent_b.copy()

        pt_a = random.randint(1, len(genes_a) - 1)
        pt_b = random.randint(1, len(genes_b) - 1)

        child1 = Chromosome.__new__(Chromosome)
        child1.protected_answer = parent_a.protected_answer
        child1.max_genes = parent_a.max_genes
        child1.genes = (genes_a[:pt_a] + genes_b[pt_b:])[:parent_a.max_genes]
        child1._fitness = None
        child1._ensure_answer()

        child2 = Chromosome.__new__(Chromosome)
        child2.protected_answer = parent_b.protected_answer
        child2.max_genes = parent_b.max_genes
        child2.genes = (genes_b[:pt_b] + genes_a[pt_a:])[:parent_b.max_genes]
        child2._fitness = None
        child2._ensure_answer()

        return child1, child2

    # ── selection ────────────────────────────────────────────────────────────

    def tournament_select(
        self,
        population: List[Chromosome],
        fitness_fn: GRASPFitness,
        k: int,
    ) -> Chromosome:
        contestants = random.sample(population, min(k, len(population)))
        return max(contestants, key=fitness_fn).copy()


# ─────────────────────────────────────────────────────────────────────────────
# Core Genetic Algorithm
# ─────────────────────────────────────────────────────────────────────────────

class RealGeneticAlgorithm:
    """Elitist tournament GA for one adversarial text slot."""

    def __init__(
        self,
        operators: GAOperators,
        fitness_fn: GRASPFitness,
        config: GRASPConfig,
    ):
        self.ops = operators
        self.fitness = fitness_fn
        self.cfg = config

    def run(
        self,
        initial_population: List[Chromosome],
        verbose: bool = False,
    ) -> Tuple[Chromosome, List[float]]:
        pop = [c.copy() for c in initial_population]
        while len(pop) < self.cfg.population_size:
            seed = random.choice(initial_population).copy()
            pop.append(self.ops.mutate(seed))

        best_ever: Chromosome = max(pop, key=self.fitness).copy()
        best_fitness_history: List[float] = []
        n_elite = max(1, int(self.cfg.elite_frac * self.cfg.population_size))

        for gen in range(self.cfg.num_generations):
            scored = sorted(
                [(self.fitness(c), c) for c in pop],
                key=lambda x: x[0], reverse=True
            )
            gen_best = scored[0][0]
            best_fitness_history.append(gen_best)

            if gen_best > self.fitness(best_ever):
                best_ever = scored[0][1].copy()

            if verbose and gen % 5 == 0:
                avg = float(np.mean([s for s, _ in scored]))
                logger.info(
                    "  Gen %2d/%d │ best=%.4f │ avg=%.4f │ overall=%.4f",
                    gen, self.cfg.num_generations - 1,
                    gen_best, avg, self.fitness(best_ever),
                )

            elites = [c.copy() for _, c in scored[:n_elite]]
            new_pop: List[Chromosome] = list(elites)

            while len(new_pop) < self.cfg.population_size:
                if random.random() < self.cfg.crossover_rate:
                    p1 = self.ops.tournament_select(pop, self.fitness, self.cfg.tournament_size)
                    p2 = self.ops.tournament_select(pop, self.fitness, self.cfg.tournament_size)
                    c1, c2 = self.ops.crossover(p1, p2)
                    if random.random() < self.cfg.mutation_rate:
                        c1 = self.ops.mutate(c1)
                    if random.random() < self.cfg.mutation_rate:
                        c2 = self.ops.mutate(c2)
                    new_pop.extend([c1, c2])
                else:
                    parent = self.ops.tournament_select(pop, self.fitness, self.cfg.tournament_size)
                    new_pop.append(self.ops.mutate(parent))

            pop = new_pop[: self.cfg.population_size]

        return best_ever, best_fitness_history


# ─────────────────────────────────────────────────────────────────────────────
# GRASP Attack Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class GRASPAttack:
    """
    Full GRASP pipeline — model-agnostic post-hoc optimizer.

    Works as a general refinement layer over any base attack's seed texts.
    The defense-aware fitness ensures evolved texts are robust to:
      (1) Paraphrasing defense: query rewritten before retrieval
      (2) PPL/naturalness filtering: anomaly-based passage removal
      (3) Deduplication: semantic diversity from fragment pool

    Usage:
        attack = GRASPAttack(embed_fn=my_embed, config=GRASPConfig())
        result = attack.attack_query(
            query_id="q001",
            question="Who invented the telephone?",
            correct_answer="Alexander Graham Bell",
            incorrect_answer="Thomas Edison",
            seed_adv_texts=[...],   # from PoisonedRAG-BB or any base attack
            clean_topk_scores=[...],
        )
    """

    def __init__(
        self,
        embed_fn: Callable[[str], np.ndarray],
        config: Optional[GRASPConfig] = None,
        prepend_query: bool = False,
    ):
        self.embed_fn = embed_fn
        self.cfg = config or GRASPConfig()
        # prepend_query=False by default: GRASP produces semantically evolved
        # texts WITHOUT literal query prepending (unlike PoisonedRAG-BB S⊕I)
        self.prepend_query = prepend_query

    def _make_final_text(self, question: str, evolved_text: str) -> str:
        if self.prepend_query:
            return question.rstrip(".") + ". " + evolved_text
        return evolved_text

    def _compute_sim(self, text: str, query_emb: np.ndarray) -> float:
        return float(np.dot(query_emb, self.embed_fn(text)))

    def attack_query(
        self,
        query_id: str,
        question: str,
        correct_answer: str,
        incorrect_answer: str,
        seed_adv_texts: List[str],
        clean_topk_scores: List[float],
        adv_per_query: Optional[int] = None,
        paraphrase_queries: Optional[List[str]] = None,
        verbose: bool = False,
    ) -> AttackResult:
        """
        Run GRASP for one target question. Returns AttackResult.

        Args:
            paraphrase_queries: Optional caller-supplied paraphrase variants.
                If None, auto-generates via generate_paraphrases().
                Supplying externally allows experiments to control the
                paraphrase distribution (e.g. rule-based vs model-based).
        """
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

        t0 = time.time()
        n_adv = min(adv_per_query or len(seed_adv_texts), len(seed_adv_texts))
        query_emb = self.embed_fn(question)
        clean_threshold = clean_topk_scores[-1] if clean_topk_scores else 0.0

        # Use caller-supplied paraphrases or auto-generate
        if paraphrase_queries is None:
            paraphrase_queries = generate_paraphrases(
                question, n=self.cfg.n_paraphrase_variants, seed=self.cfg.seed
            )

        fitness_fn = GRASPFitness(
            embed_fn=self.embed_fn,
            query=question,
            incorrect_answer=incorrect_answer,
            paraphrase_queries=paraphrase_queries,
            lambda_stealth=self.cfg.fitness_lambda_stealth,
            lambda_paraphrase=self.cfg.fitness_lambda_paraphrase,
            lambda_naturalness=self.cfg.fitness_lambda_naturalness,
        )
        operators = GAOperators(
            embed_fn=self.embed_fn,
            seed_texts=seed_adv_texts,
            query=question,
            incorrect_answer=incorrect_answer,
            fragment_mutation_prob=self.cfg.fragment_mutation_prob,
            max_genes=self.cfg.max_genes,
        )
        ga = RealGeneticAlgorithm(operators, fitness_fn, self.cfg)

        evo_results: List[EvoResult] = []
        evolved_texts: List[str] = []

        for adv_idx in range(n_adv):
            t_adv = time.time()
            original = seed_adv_texts[adv_idx]
            orig_sim = self._compute_sim(original, query_emb)

            seed_chromo = Chromosome(original, incorrect_answer, self.cfg.max_genes)
            initial_pop: List[Chromosome] = [seed_chromo]
            for _ in range(4):
                initial_pop.append(operators.mutate(seed_chromo.copy()))
            other_idx = (adv_idx + 1) % len(seed_adv_texts)
            other_chromo = Chromosome(
                seed_adv_texts[other_idx], incorrect_answer, self.cfg.max_genes
            )
            c1, c2 = operators.crossover(seed_chromo, other_chromo)
            initial_pop.extend([c1, c2])
            initial_pop.append(operators.mutate_fragment_recombine(seed_chromo.copy()))

            best_chromo, fitness_history = ga.run(initial_pop, verbose=verbose)

            evolved_text = best_chromo.to_text()
            evolved_sim = fitness_fn.retrieval_sim(evolved_text)
            para_sim = fitness_fn.paraphrase_sim_mean(evolved_text)
            converged_gen = next(
                (i for i, f in enumerate(fitness_history) if f >= max(fitness_history) - 1e-6),
                len(fitness_history) - 1,
            )

            evo_results.append(EvoResult(
                original_text=original,
                evolved_text=evolved_text,
                original_sim=orig_sim,
                evolved_sim=evolved_sim,
                delta_sim=evolved_sim - orig_sim,
                paraphrase_sim_mean=para_sim,
                has_answer=incorrect_answer.lower() in evolved_text.lower(),
                generation_converged=converged_gen,
                fitness_history=fitness_history,
                time_seconds=time.time() - t_adv,
            ))

            evolved_texts.append(self._make_final_text(question, evolved_text))

            if verbose:
                logger.info(
                    "  GA[%d]: sim %.4f→%.4f (Δ+%.4f)  para=%.4f  %s",
                    adv_idx, orig_sim, evolved_sim, evolved_sim - orig_sim,
                    para_sim, "✅" if evo_results[-1].has_answer else "⚠️",
                )

        evolved_sims = [self._compute_sim(t, query_emb) for t in evolved_texts]
        adv_in_topk = sum(1 for s in evolved_sims if s >= clean_threshold)

        return AttackResult(
            query_id=query_id,
            question=question,
            correct_answer=correct_answer,
            incorrect_answer=incorrect_answer,
            evolved_texts=evolved_texts,
            evo_results=evo_results,
            clean_threshold=clean_threshold,
            evolved_sims=evolved_sims,
            adv_in_topk=adv_in_topk,
            total_time_seconds=time.time() - t0,
        )
