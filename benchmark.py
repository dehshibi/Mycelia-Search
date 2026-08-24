"""
benchmark.py
============
Run the CEC 2022 benchmark on Myco variants.

Outputs:
1. Convergence history (per run) -> results/models/
2. Final statistics -> results/NPY Files/summary.csv
3. Seeds' log files for reproducibility
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

from cec_functions import CEC_CORE, CEC_INFO
from algorithms.myco_plasticity import MycelialSearchPlasticity
from algorithms.myco_louvain import MycelialSearchLouvain

# Configuration
ALGORITHMS = {
    "Myco (Plasticity)": MycelialSearchPlasticity,
    "Myco (Louvain)": MycelialSearchLouvain,
}

DIM = 10  # CEC 2022 INPUT DATA supports {10, 20}
BOUNDS = np.array([[-100] * DIM, [100] * DIM])
N_RUNS = 30

if DIM == 10:
    MAX_EVALS = 20000 * DIM
elif DIM == 20:
    MAX_EVALS = 50000 * DIM
else:
    raise ValueError("INPUT DATA only supports DIM 10 and 20.")

SEEDS = list(range(N_RUNS))

# Output directories
RESULTS_DIR = Path("results") / f"Core Functions D{DIM}"
MODELS_DIR = RESULTS_DIR / "NPY Files"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def run_single_benchmark(algo_class, func, func_name, run_id, seed):
    """Run one optimisation and save convergence history."""
    np.random.seed(seed)

    # Initialise algorithm; ROI can be True/False for the ablation study
    algo = algo_class(func, BOUNDS, use_roi=True)

    # Run and track history
    _, best_fitness = algo.optimize(MAX_EVALS, verbose=False)
    history = algo.history  # Convergence curve

    # Save history
    algo_name = algo.__class__.__name__
    history_file = MODELS_DIR / f"{func_name}_{algo_name}_run{run_id}.npy"
    np.save(history_file, np.array(history))

    return best_fitness


def main():
    """Run full CEC benchmark."""
    results = []

    print("=" * 80)
    print("CEC 2022 Benchmark: Myco Variants")
    print("=" * 80)

    for func_name, func in CEC_CORE.items():
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
            q1 = float(np.percentile(fitnesses, 25))
            q3 = float(np.percentile(fitnesses, 75))
            iqr = q3 - q1
            best_fit = float(np.min(fitnesses))

            results.append({
                "Function": func_name,
                "Algorithm": algo_name,
                "Mean": mean_fit,
                "Std": std_fit,
                "Median": median_fit,
                "IQR": iqr,
                "Best": best_fit,
            })

            print(f"  {algo_name} => MEAN/STD: {mean_fit:.4e} ± {std_fit:.4e} -- MEDIAN/IQR: {median_fit:.4e} ± {iqr:.4e} ")

    # Save summary
    df = pd.DataFrame(results)
    df.to_excel(MODELS_DIR / "CEC Benchmark Summary.xlsx", index=False)

    # Save seeds
    with open(RESULTS_DIR / "seeds.json", "w") as f:
        json.dump({"seeds": SEEDS, "n_runs": N_RUNS}, f, indent=2)

    print(f"\n NPY files saved to {MODELS_DIR}/")
    print(f" Seeds logged to {RESULTS_DIR}/seeds.json")


if __name__ == "__main__":
    main()
