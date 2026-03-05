"""
GRASP NeurIPS — Master Experiment Runner
=========================================
Runs all experiments sequentially and prints a consolidated summary.
"""

import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def run_all_experiments():
    results = {}
    t_total = time.time()

    experiments = [
        ("Exp1: ASR Comparison Table",   "experiments.exp1_asr_table"),
        ("Exp2: Injection Efficiency",   "experiments.exp2_efficiency"),
        ("Exp3: Stealth Evaluation",     "experiments.exp3_stealth"),
        ("Exp4: Defense Robustness",     "experiments.exp4_defenses"),
        ("Exp5: Operator Ablation",      "experiments.exp5_ablation"),
        ("Exp6+7: Transfer+Convergence", "experiments.exp6_7_transfer_convergence"),
        ("Exp8: Statistical Sig Tables",  "experiments.exp8_significance"),
    ]

    for name, module_path in experiments:
        logger.info("\n" + "=" * 70)
        logger.info("Running: %s", name)
        logger.info("=" * 70)
        t0 = time.time()
        try:
            import importlib
            mod = importlib.import_module(module_path)
            res = mod.main()
            results[name] = {"status": "OK", "time_s": time.time() - t0}
            logger.info("✅ %s completed in %.1fs", name, time.time() - t0)
        except Exception as e:
            results[name] = {"status": "FAILED", "error": str(e), "time_s": time.time() - t0}
            logger.error("❌ %s FAILED: %s", name, e)
            raise

    t_elapsed = time.time() - t_total

    print("\n" + "=" * 70)
    print("EXPERIMENT RUN SUMMARY")
    print("=" * 70)
    for name, info in results.items():
        status = "✅" if info["status"] == "OK" else "❌"
        print(f"  {status} {name:<40} {info['time_s']:>6.1f}s")
    print(f"\n  Total time: {t_elapsed:.1f}s")
    print("=" * 70)

    out = ROOT / "results" / "master_summary.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Master summary → %s", out)

    return results


if __name__ == "__main__":
    run_all_experiments()
