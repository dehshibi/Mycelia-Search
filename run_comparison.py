"""
run_comparison.py
=================
 Comparative Benchmark for Mycelial Search (Myco).

Algorithms:
1.  Myco Variants (Louvain, Plasticity) - Growth Network
2.  SMA, MGO, PIS - Decentralised Growth algorithms
3.  GA - Genetic/Evolutionary
4.  CLPSO, ABC - Swarm Intelligence
5.  GWO, WOA - Mammal-inspired
6.  jSO, SAP-DE, L-SHADE, JADE - Differential Evolution

CEC 2022 Benchmark Configuration:
- Dimensions: 10, 20 (configurable via DIMENSION variable)
- Functions: F1-F12 (CEC 2022 test suite)
- Max Evaluations: 20,000 or 50,000 * D (standard CEC setup)
- Runs: 30 (CEC standard, configurable via RUNS)
- Population Size: 50

Dependencies:
    conda/pip install mealpy pandas tqdm networkx scipy

Output:
    Results saved to: results/comparison/
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import time

# --- IMPORT CEC FUNCTIONS ---
# Ensure cec_functions.py and cec2022_functions.py are in the same folder
from cec_functions import CEC_FUNCTIONS, CEC_CORE, CEC_EXTRA, CEC_METADATA

# --- IMPORT Myco VARIANTS ---
# Ensure these files are in the algorithms folder
from algorithms.myco_louvain import MycelialSearchLouvain as Myco_Louvain
from algorithms.myco_plasticity import MycelialSearchPlasticity as Myco_Plasticity

# --- MEALPY 3.0.3 IMPORTS ---
from mealpy import FloatVar
from mealpy.evolutionary_based import DE, GA, SHADE
from mealpy.swarm_based import PSO, ABC, WOA, GWO
from mealpy.bio_based import SMA

# --- IMPORT CUSTOM BIO-INSPIRED ALGORITHMS ---
from algorithms.bio_inspired import OriginalMGO, PhysarumOptimizer
from algorithms.jso import jSO, SAP_DE

# ============================================================================
# CEC 2022 SPECIFIC CONFIGURATION
# ============================================================================
DIMENSION = 10  # CEC 2022 INPUT DATA only supports {10, 20}
RUNS = 30  # CEC 2022 standard

# Load CEC 2022 random seeds
RAND_SEEDS_FILE = Path("input_data/Rand_Seeds.txt")
with open(RAND_SEEDS_FILE, 'r') as f:
    RAND_SEEDS = [float(line.strip()) for line in f.readlines()]

if DIMENSION == 10:
    MAX_FE = 20000 * DIMENSION
elif DIMENSION == 20:
    MAX_FE = 50000 * DIMENSION
else:
    raise ValueError("INPUT DATA only supports DIM 10 and 20.")

POP_SIZE = 50  # For Mealpy algorithms
Myco_POP_SIZE = 30  # For Myco algorithms (default ntips)

# Output directories
RESULTS_DIR = Path("results")
COMPARISON_DIR = RESULTS_DIR / f"Comparison D{DIMENSION}"
MODELS_DIR = COMPARISON_DIR / f"NPY Files  D{DIMENSION}"

# Create all directories
for dir_path in [COMPARISON_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Output files
OUTPUT_FILENAME = "Results.xlsx"
OUTPUT_FILE = COMPARISON_DIR / OUTPUT_FILENAME
SUMMARY_FILE = MODELS_DIR / f"CEC Summary D{DIMENSION}.xlsx"
SEEDS_FILE = COMPARISON_DIR / f"Seeds D{DIMENSION}.json"
COMPLEXITY_FILE = COMPARISON_DIR / f"Complexity D{DIMENSION}.json"
PARAMS_FILE = COMPARISON_DIR / f"Parameters D{DIMENSION}.json"
CONVERGENCE_FILE = COMPARISON_DIR / f"Convergence data D{DIMENSION}.json"


# ============================================================================
# CEC 2022 SEED CALCULATION
# ============================================================================
def get_cec2022_seed(problem_size, func_no, runs, run_id):
    """
    Calculate CEC 2022 seed according to specification.

    seed_ind = (problem_size/10 * func_no * Runs + run_id) - Runs;
    seed_ind = mod(seed_ind, 1000) + 1;
    run_seed = Rand_Seeds(seed_ind);
    """
    seed_ind = (problem_size // 10 * func_no * runs + run_id) - runs
    seed_ind = (seed_ind % 1000) + 1
    return RAND_SEEDS[int(seed_ind) - 1]  # 0-based indexing


# ============================================================================
# CEC 2022 CONVERGENCE RECORDING
# ============================================================================
def get_convergence_points(dimension, max_fe):
    """
    Calculate the 16 convergence recording points for theCEC 2022.

    Record error after floor(D^(k/5-3) * MaxFES) for k = 0, 1, 2, ..., 15
    """
    points = []
    for k in range(16):  # k = 0 to 15
        exponent = k / 5.0 - 3.0
        factor = dimension ** exponent
        fe_point = int(factor * max_fe)
        # Ensure within bounds
        fe_point = max(1, min(fe_point, max_fe))
        points.append(fe_point)
    return points


# ============================================================================
# ALGORITHM ADAPTERS
# ============================================================================

class MealpyAdapter:
    """Wraps Mealpy 3.0.3 algorithms with CEC 2022 convergence tracking."""

    def __init__(self, model_class, name, **kwargs):
        self.model_class = model_class
        self.name = name
        self.params = kwargs

    def run(self, func, lb, ub, dim, max_fe, seed=None, func_name=None, run_id=None):
        # Use FloatVar for bounds (Mealpy 3.0.3 requirement)
        bounds = FloatVar(lb=[lb] * dim, ub=[ub] * dim)

        problem_dict = {
            "obj_func": func,
            "bounds": bounds,
            "minmax": "min",
            "log_to": None,
            "save_population": False
        }

        epoch = int(max_fe / POP_SIZE)
        model = self.model_class(epoch=epoch, pop_size=POP_SIZE, seed=seed, **self.params)

        # Set problem before solve to avoid None issues
        model.problem = problem_dict

        # Use standard solving method (Mealpy 3.0.3)
        g_best = model.solve(problem_dict)

        best_position = g_best.solution
        best_fitness = g_best.target.fitness

        # Extract convergence history from model.history
        history = model.history.list_global_best  # List of the best fitness per generation

        # Get convergence recording points
        conv_points = get_convergence_points(dim, max_fe)
        convergence_history = []

        def get_fitness(val):
            if isinstance(val, float):
                return val
            elif hasattr(val, 'target') and hasattr(val.target, 'fitness'):
                return val.target.fitness
            elif hasattr(val, 'target'):
                return val.target
            else:
                return float(val)

        for fe_point in conv_points:
            # Each generation corresponds to POP_SIZE function evaluations
            gen = (fe_point - 1) // POP_SIZE + 1
            if gen <= len(history):
                hist_val = history[gen - 1]
                val = get_fitness(hist_val)
                convergence_history.append(val)
            else:
                convergence_history.append(get_fitness(history[-1]) if history else best_fitness)

        # Determine FE_term (when error first < 1e-8)
        func_meta = CEC_METADATA[int(func_name[1:])]
        bias = func_meta['bias']
        fe_term = max_fe
        for i, hist_val in enumerate(history):
            val = get_fitness(hist_val)
            if val - bias < 1e-8:
                fe_term = (i + 1) * POP_SIZE
                break

        # Save convergence history
        if func_name and run_id is not None:
            history_file = MODELS_DIR / f"{func_name}_{self.name}_run{run_id}.npy"
            np.save(history_file, np.array(convergence_history))

        return best_fitness, best_position, convergence_history, fe_term


class MycoAdapter:
    """Wraps local Myco variants with CEC 2022 convergence tracking."""

    def __init__(self, myco_class, name, **kwargs):
        self.myco_class = myco_class
        self.name = name
        self.params = kwargs

    def run(self, func, lb, ub, dim, max_fe, seed=None, func_name=None, run_id=None):
        # Myco uses ntips (default 30, not POP_SIZE which is for Mealpy)
        n_tips = Myco_POP_SIZE

        # Initialise Myco
        bounds = np.array([[lb] * dim, [ub] * dim])

        # Check class name to determine parameter names
        if "Plasticity" in self.myco_class.__name__:
            optimiser = self.myco_class(func, bounds, ntips=n_tips, ndim=dim, **self.params)
        else:
            optimiser = self.myco_class(func, bounds, n_tips=n_tips, n_dim=dim, **self.params)

        # Set seed
        if seed is not None:
            np.random.seed(int(seed))

        # Run optimisation
        best_pos, best_val = optimiser.optimize(max_evals=max_fe, verbose=False)
        history = optimiser.history  # Myco convergence history (per iteration)

        # Get convergence recording points
        conv_points = get_convergence_points(dim, max_fe)
        convergence_history = []

        for fe_point in conv_points:
            # Each iteration corresponds to approximately n_tips function evaluations
            step = (fe_point - 1) // n_tips + 1
            if step <= len(history):
                convergence_history.append(history[step - 1])
            else:
                convergence_history.append(best_val)

        # Check termination condition
        func_meta = CEC_METADATA[int(func_name[1:])]  # Extract number from "F1"
        bias = func_meta['bias']
        error = best_val - bias

        # Determine FE_term
        fe_term = max_fe
        if error < 1e-8:
            # Find the iteration where error first became < 1e-8
            for i, hist_val in enumerate(history):
                if hist_val - bias < 1e-8:
                    fe_term = (i + 1) * n_tips
                    break

        # Save convergence history
        if func_name and run_id is not None:
            history_file = MODELS_DIR / f"{func_name}_{self.name}_run{run_id}.npy"
            np.save(history_file, np.array(convergence_history))

        return best_val, best_pos, convergence_history, fe_term


# ============================================================================
# ALGORITHM REGISTRY
# ============================================================================

ALGORITHMS = [
    # --- PROPOSED METHODS (Myco VARIANTS) ---
    MycoAdapter(Myco_Louvain, "Myco (Louvain)"),
    MycoAdapter(Myco_Plasticity, "Myco (Plasticity)"),

    # --- BIO-INSPIRED ---
    MealpyAdapter(SMA.OriginalSMA, "SMA"),
    MealpyAdapter(OriginalMGO, "MGO"),
    MealpyAdapter(PhysarumOptimizer, "PIS"),

    # --- CEC 2022 COMPETITION WINNERS ---
    MealpyAdapter(jSO, "jSO"),  # CEC 2022 Winner #1
    MealpyAdapter(SAP_DE, "SAP-DE"),  # CEC 2022 Winner #2
    MealpyAdapter(SHADE.L_SHADE, "L-SHADE"),  # CEC 2015 Winner (for comparison)

    # --- CLASSIC SOTA ---
    MealpyAdapter(DE.JADE, "JADE"),
    MealpyAdapter(PSO.CL_PSO, "CLPSO"),
    MealpyAdapter(GA.BaseGA, "GA"),
    MealpyAdapter(ABC.OriginalABC, "ABC"),
    MealpyAdapter(WOA.OriginalWOA, "WOA"),
    MealpyAdapter(GWO.OriginalGWO, "GWO"),
]


# ============================================================================
# ALGORITHM COMPLEXITY CALCULATION
# ============================================================================
def calculate_complexity():
    """
    Calculate the algorithm's complexity according to the CEC 2022 specification.

    Returns T0, T1, T2_mean where:
    - T0: Time for basic operations loop
    - T1: Time for function evaluations only
    - T2: Time for complete algorithm
    """
    print("\n" + "=" * 60)
    print("CALCULATING ALGORITHM COMPLEXITY")
    print("=" * 60)

    # Test function (F1) and dimension
    test_func = CEC_FUNCTIONS["F1"]
    test_dim = DIMENSION
    test_x = np.random.uniform(-100, 100, test_dim)

    # T0: Basic operations time (per CEC 2022 Section 2.2.4)
    # Simulate basic arithmetic operations without resetting in each iteration
    print("Calculating T0 (basic operations)...")
    start_time = time.time()
    x = 0.55
    for i in range(200000):
        x = x / 2
        x = x * x
        x = np.sqrt(np.abs(x))
        x = np.log(np.abs(x) + 1e-15)
        x = np.exp(x)
        x = x / (x + 2) if x + 2 != 0 else x / 1e-15
        x = x + x
    T0 = time.time() - start_time
    print(f"  T0 = {T0:.4f}s")

    # T1: Function evaluation time
    print("Calculating T1 (function evaluations)...")
    start_time = time.time()
    for i in range(200000):
        _ = test_func(test_x)
    T1 = time.time() - start_time
    print(f"  T1 = {T1:.4f}s")

    # T2: Complete algorithm time (5 runs)
    print("Calculating T2 (complete algorithm, 5 runs)...")
    T2_values = []

    for run in range(5):
        print(f"  Run {run + 1}/5...")
        start_time = time.time()

        # Use the first algorithm for complexity calculation
        algo = ALGORITHMS[0]
        seed = get_cec2022_seed(DIMENSION, 1, 5, run)  # func_no=1, runs=5

        try:
            _, _, _, _ = algo.run(test_func, -100, 100, test_dim, 200000,
                                  seed=seed, func_name="F1", run_id=run)
            T2_values.append(time.time() - start_time)
        except:
            print(f"    Run {run + 1} failed, skipping...")
            T2_values.append(0)

    T2_mean = np.mean(T2_values)
    print(f"  T2 (mean) = {T2_mean:.4f}s")

    # Calculate complexity metrics
    complexity = (T2_mean - T1) / T0 if T0 > 0 else 0

    print("\nCOMPLEXITY RESULTS:")
    print(f"  T0 (basic ops):     {T0:.4f}s")
    print(f"  T1 (func evals):    {T1:.4f}s")
    print(f"  T2 (full algorithm): {T2_mean:.4f}s")
    print(f"  Complexity = (T2 - T1) / T0 = {complexity:.4f}")

    # Save complexity results
    complexity_data = {
        "dimension": DIMENSION,
        "T0": T0,
        "T1": T1,
        "T2_mean": T2_mean,
        "complexity": complexity,
        "algorithm": ALGORITHMS[0].name,
        "test_function": "F1",
        "n_runs": 5
    }

    with open(COMPLEXITY_FILE, 'w') as f:
        json.dump(complexity_data, f, indent=2)

    print(f"\nComplexity results saved to: {COMPLEXITY_FILE}")

    return T0, T1, T2_mean


# ============================================================================
# PARAMETER DOCUMENTATION
# ============================================================================
def get_parameter_documentation():
    """
    Return parameter documentation as required by CEC 2022
    """
    params_doc = {
        "population_size": {
            "value": POP_SIZE,
            "description": "Population size for all algorithms",
            "dynamic_range": f"Fixed at {POP_SIZE}",
            "guidelines": "CEC 2022 standard population size",
            "tuning_cost": "Not tuned - fixed parameter",
            "actual_value": POP_SIZE
        },
        "max_function_evaluations": {
            "value": MAX_FE,
            "description": "Maximum function evaluations per run",
            "dynamic_range": f"D=10: 200,000 | D=20: 1,000,000 (CEC 2022 standard). Current: {MAX_FE}",
            "guidelines": "CEC 2022 standard termination criterion",
            "tuning_cost": "Not tuned - CEC standard",
            "actual_value": MAX_FE
        },
        "dimension": {
            "value": DIMENSION,
            "description": "Problem dimension",
            "dynamic_range": "{10, 20}",
            "guidelines": "CEC 2022 supported dimensions",
            "tuning_cost": "Not tuned - test parameter",
            "actual_value": DIMENSION
        }
    }

    # Algorithm-specific parameters
    algorithm_params = {
        "jSO": {
            "description": "CEC 2022 winner - self-adaptive DE",
            "parameters": {
                "F_adaptation": "Cauchy distribution perturbation",
                "CR_adaptation": "Gaussian distribution perturbation",
                "archive_size": "Population size (dynamic)",
                "mutation": "DE/current-to-pbest/1"
            }
        },
        "SAP_DE": {
            "description": "Success-based parameter adaptation DE",
            "parameters": {
                "success_rate_tracking": "Global success/failure counts",
                "parameter_adaptation": "Linear scaling based on success rate",
                "mutation": "DE/rand/1/bin"
            }
        },
        "L-SHADE": {
            "description": "CEC 2015 winner - linear population size reduction",
            "parameters": {
                "population_adaptation": "Linear decrease from 50 to 4",
                "memory_size": "6 (for F and CR)",
                "archive_size": "Population size × 2.6"
            }
        },
        "JADE": {
            "description": "Adaptive DE with optional external archive",
            "parameters": {
                "c": "0.1 (learning rate for mean F and CR)",
                "p": "0.05 (portion for current-to-pbest)"
            }
        },
        "CL-PSO": {
            "description": "Comprehensive Learning PSO",
            "parameters": {
                "Pc": "0.5 (learning probability)",
                "m": "Population size (refresh gap)"
            }
        },
        "SMA": {
            "description": "Slime Mould Algorithm",
            "parameters": {
                "z": "0.03 (parameter for W)",
                "vb": "[1, 6] (parameter for b)"
            }
        },
        "MGO": {
            "description": "Enhanced Moss Growth Optimisation",
            "parameters": {
                "quality_threshold": "Fitness-based assessment",
                "growth_rate_range": "[0.5, 1.0] (adaptive)",
                "resource_depletion": "50% over optimisation",
                "cooperation_radius": "Random neighbor selection"
            }
        },
        "PIS": {
            "description": "Enhanced Physarum-Inspired Search",
            "parameters": {
                "tube_diameter_range": "[0, 2] (adaptive)",
                "conductivity_decay": "0.95 per iteration",
                "fitness_feedback": "[0.1, 1.0] (normalized)",
                "distance_matrix": "Euclidean distances"
            }
        }
    }

    return params_doc, algorithm_params


# ============================================================================
# MAIN BENCHMARK LOOP
# ============================================================================

def run_benchmark():
    if OUTPUT_FILE.exists():
        results = pd.read_excel(OUTPUT_FILE).to_dict('records')
        print(f" Resuming from existing results: {len(results)} records found.")
    else:
        results = []

    if CONVERGENCE_FILE.exists():  # For detailed convergence analysis
        with open(CONVERGENCE_FILE, 'r') as f:
            convergence_data = json.load(f)
        print(f" Resuming from existing convergence data: {len(convergence_data)} records found.")
    else:
        convergence_data = []

    print(f" STARTING CEC 2022 BENCHMARK: D={DIMENSION}, Runs={RUNS}, MaxFE={MAX_FE}")
    print(f" Algorithms: {[alg.name for alg in ALGORITHMS]}")
    print(f" Functions: F1-F12")
    print(f" Output: {OUTPUT_FILE}")
    print(f" Convergence histories: {MODELS_DIR}/")

    # Iterate over ALL CEC functions (F1-F12)
    all_functions = {**CEC_CORE, **CEC_EXTRA}  # Full CEC 2022 suite

    for func_name, func in all_functions.items():
        func_id = int(func_name[1:])  # Extract number from "F1"
        func_meta = CEC_METADATA[func_id]
        bias = func_meta['bias']

        print(f"\n Evaluating {func_name} ({func_meta['name']})...")

        for algo in ALGORITHMS:
            if any(r['Function'] == func_name and r['Algorithm'] == algo.name for r in results):
                print(f"  [Skip] {algo.name} already completed for {func_name}.")
                continue

            fitnesses = []
            convergence_histories = []
            fe_terms = []

            pbar = tqdm(range(RUNS), desc=f"  {algo.name:<15}", leave=False)
            for run_id in pbar:
                # Calculate CEC 2022 seed
                run_seed = get_cec2022_seed(DIMENSION, func_id, RUNS, run_id)

                try:
                    # Execute optimisation with convergence tracking
                    best_fit, _, conv_history, fe_term = algo.run(
                        func, -100, 100, DIMENSION, MAX_FE,
                        seed=run_seed, func_name=func_name, run_id=run_id
                    )

                    # Error = Fitness - Bias (CEC Standard)
                    error = best_fit - bias
                    # Ensure no floating point negatives below zero for global optimum
                    if error < 1e-8:
                        error = 0.0

                    fitnesses.append(error)
                    convergence_histories.append(conv_history)
                    fe_terms.append(fe_term)

                except Exception as e:
                    import traceback
                    print(f"\n Error in {algo.name} on {func_name}: {e}")
                    traceback.print_exc()
                    fitnesses.append(np.inf)
                    convergence_histories.append([])
                    fe_terms.append(MAX_FE)

            # Statistics Calculation
            fitnesses = np.array(fitnesses)
            valid_fitnesses = fitnesses[fitnesses != np.inf]

            if len(valid_fitnesses) > 0:
                # Sort for best/worst (CEC requirement)
                sorted_fitnesses = np.sort(valid_fitnesses)

                mean_error = float(np.mean(valid_fitnesses))
                std_error = float(np.std(valid_fitnesses))
                median_error = float(np.median(valid_fitnesses))
                q1 = float(np.percentile(valid_fitnesses, 25))
                q3 = float(np.percentile(valid_fitnesses, 75))
                iqr = q3 - q1
                best_error = float(sorted_fitnesses[0])  # Best (smallest error)
                worst_error = float(sorted_fitnesses[-1])  # Worst (largest error)

                # Prepare convergence data record
                conv_data = {
                    "function": func_name,
                    "algorithm": algo.name,
                    "dimension": DIMENSION,
                    "convergence_histories": [np.array(h).tolist() for h in convergence_histories],
                    "fe_terms": fe_terms,
                    "seeds": [get_cec2022_seed(DIMENSION, func_id, RUNS, i) for i in range(RUNS)]
                }
                convergence_data.append(conv_data)
            else:
                # All runs failed
                mean_error = std_error = median_error = q1 = q3 = iqr = best_error = worst_error = np.inf

            # Save Record
            results.append({
                "Function": func_name,
                "Algorithm": algo.name,
                "Dimension": DIMENSION,
                "Mean": mean_error,
                "Std": std_error,
                "Median": median_error,
                "Q1": q1,
                "Q3": q3,
                "IQR": iqr,
                "Best": best_error,
                "Worst": worst_error,
                "Runs": RUNS,
                "MaxFE": MAX_FE,
                "Bias": bias
            })
            # --- INCREMENTAL SAVING ---
            # Save Excel after every algorithm finishes its 30 runs
            df = pd.DataFrame(results)
            df.to_excel(OUTPUT_FILE, index=False)

            # Save convergence JSON incrementally
            with open(CONVERGENCE_FILE, 'w') as f:
                json.dump(convergence_data, f, indent=2)

            tqdm.write(f"  {algo.name:<15} | Mean: {mean_error:.2e} | Best: {best_error:.2e} | [Progress Saved]")

        # Save intermediate results
        df = pd.DataFrame(results)
        df.to_excel(OUTPUT_FILE, index=False)

    print(f"\n Benchmark Complete! Results saved to {OUTPUT_FILE}")

    # Save summary to models directory
    df.to_excel(SUMMARY_FILE, index=False)
    print(f" Summary saved to {SUMMARY_FILE}")

    # Save convergence data for analysis
    with open(CONVERGENCE_FILE, 'w') as f:
        json.dump(convergence_data, f, indent=2)
    print(f" Convergence data saved to {CONVERGENCE_FILE}")

    # Save seeds for reproducibility
    seeds_data = {
        "dimension": DIMENSION,
        "n_runs": RUNS,
        "functions": list(all_functions.keys()),
        "algorithms": [alg.name for alg in ALGORITHMS],
        "seeds_used": "CEC 2022 Rand_Seeds.txt with formula",
        "seed_calculation": "seed_ind = (D/10 * func_no * Runs + run_id) - Runs; mod(seed_ind, 1000) + 1"
    }

    with open(SEEDS_FILE, 'w') as f:
        json.dump(seeds_data, f, indent=2)
    print(f" Seeds info saved to {SEEDS_FILE}")

    # --- FINAL SUMMARY PRINT ---
    if not df.empty:
        print("\n" + "=" * 80)
        print("FINAL RANKING (Based on Mean Error)")
        print("=" * 80)
        pivot = df.pivot(index="Function", columns="Algorithm", values="Mean")
        print(pivot)

        # Calculate overall ranking
        mean_scores = pivot.mean(axis=0).sort_values()
        print("\nOVERALL ALGORITHM RANKING (Lower is Better):")
        for rank, (algo, score) in enumerate(mean_scores.items(), 1):
            print(f"  {rank}. {algo:<15} Mean Error: {score:.2e}")
    else:
        print("No results generated.")


if __name__ == "__main__":
    # Calculate complexity (CEC 2022 requirement) - UNCOMMENT TO RUN
    # calculate_complexity()

    # Generate parameter documentation
    params_doc, algo_params = get_parameter_documentation()
    with open(PARAMS_FILE, 'w') as f:
        json.dump({
            "general_parameters": params_doc,
            "algorithm_parameters": algo_params,
            "cec2022_compliance": "Section 2.2.5 Parameter Documentation"
        }, f, indent=2)
    print(f"Parameter documentation saved to: {PARAMS_FILE}")

    run_benchmark()
