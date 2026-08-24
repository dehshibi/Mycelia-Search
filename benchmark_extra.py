"""
benchmark_extra.py
==================
Run CEC 2022 EXTRA benchmark functions on Myco variants.

This complements benchmark.py (which runs the core subset F1, F2, F4, F6, F9)
by running: F3, F5, F7, F8, F10, F11, F12.

Outputs:
- Per-run convergence histories as .npy
- Summary CSV
- Seeds logged
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cec_functions import CEC_EXTRA, CEC_INFO
from algorithms.myco_plasticity import MycelialSearchPlasticity
from algorithms.myco_louvain import MycelialSearchLouvain

# ---------------- Configuration ----------------

ALGORITHMS = {
    "Myco (Plasticity)": MycelialSearchPlasticity,
    "Myco (Louvain)": MycelialSearchLouvain,
}

DIM = 10  # CEC 2022 INPUT DATA only supports {10, 20}
BOUNDS = np.array([[-100] * DIM, [100] * DIM])
N_RUNS = 30

if DIM == 10:
    MAX_EVALS = 20000 * DIM
elif DIM == 20:
    MAX_EVALS = 50000 * DIM
else:
    raise ValueError("INPUT DATA only supports DIM 10 and 20.")

SEEDS = list(range(N_RUNS))

# Output directories (separate from main benchmark)
RESULTS_DIR = Path("results") / "extra"
MODELS_DIR = RESULTS_DIR / "models"
PLOTS_DIR = RESULTS_DIR / "plots"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def run_single_benchmark(algo_class, func, func_name, run_id, seed):
    """
    Run one optimisation and save the convergence history.
    """
    np.random.seed(seed)

    # Initialise algorithm; ROI can be True/False for the ablation study
    algo = algo_class(func, BOUNDS, use_roi=True)

    # Run and track history
    _, best_fitness = algo.optimize(MAX_EVALS, verbose=False)
    history = algo.history  # convergence curve (list of best-so-far)

    # Save history as .npy
    algo_name = algo.__class__.__name__
    history_file = MODELS_DIR / f"{func_name}_{algo_name}_run{run_id}.npy"
    np.save(history_file, np.array(history))

    return best_fitness


def main():
    """
    Run extra CEC benchmark functions on Myco variants.
    """
    results = []

    print("=" * 80)
    print("CEC 2022 Extra Benchmark: Myco Variants")
    print("=" * 80)

    for func_name, func in CEC_EXTRA.items():
        info_key = func_name  # "F3", "F5", etc.
        print(f"\n Testing {func_name} ({CEC_INFO[info_key]['name']})")

        for algo_name, algo_class in ALGORITHMS.items():
            fitnesses = []

            for run_id in range(N_RUNS):
                seed = SEEDS[run_id]
                fitness = run_single_benchmark(algo_class, func, func_name, run_id, seed)
                fitnesses.append(fitness)
                print(f"  {algo_name} run {run_id + 1}/{N_RUNS}: {fitness:.4e}", end="\r")

            fitnesses = np.array(fitnesses, dtype=float)
            mean_fit = float(np.mean(fitnesses))
            median_fit = float(np.median(fitnesses))
            std_fit = float(np.std(fitnesses))
            Q1 = float(np.percentile(fitnesses, 25))
            Q3 = float(np.percentile(fitnesses, 75))
            IQR = Q3 - Q1
            best_fit = float(np.min(fitnesses))

            results.append({
                "Function": func_name,
                "Algorithm": algo_name,
                "Mean": mean_fit,
                "Std": std_fit,
                "Median": median_fit,
                "IQR": IQR,
                "Best": best_fit,
            })

            print(f"  {algo_name}: {mean_fit:.4e} ± {std_fit:.4e} -- {median_fit:.4e} ± {IQR:.4e} ")

    # Save summary CSV
    df = pd.DataFrame(results)
    summary_file = MODELS_DIR / "cec_extra_summary.csv"
    df.to_csv(summary_file, index=False)

    # Save seeds for reproducibility
    seeds_file = RESULTS_DIR / "seeds_extra.json"
    with open(seeds_file, "w") as f:
        json.dump({"seeds": SEEDS, "n_runs": N_RUNS}, f, indent=2)

    print(f"\n Results saved to {summary_file}")
    print(f" Seeds logged to {seeds_file}")


if __name__ == "__main__":
    main()
