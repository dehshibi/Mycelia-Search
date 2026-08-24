"""
Run CEC 2022 tests for the Myco with different community detection backends.

To quickly validate 
> ONE function, use:
    .\.conda\python.exe .\beckend_test.py `
      --backend all `
      --functions F1 `
      --quick-runs 1 `
      --max-fe 1000 `
      --no-diag

> ALL functions, use:
    .\.conda\python.exe .\beckend_test.py `
      --backend all `
      --quick-runs 1 `
      --max-fe 1000 `
      --no-diag
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial import cKDTree
from tqdm import tqdm

from cec_functions import CEC_CORE, CEC_EXTRA, CEC_METADATA


class MycoOptimizerLeiden:
    """Myco cord-plasticity optimiser with selectable community detection."""

    def __init__(self, objfunc, bounds, ntips=30, ndim=None, use_roi=True):
        self.objfunc = objfunc
        self.bounds = np.asarray(bounds, dtype=float)
        if self.bounds.shape[0] != 2 and self.bounds.shape[1] == 2:
            self.bounds = self.bounds.T
        self.ntips = ntips
        self.ndim = ndim or self.bounds.shape[1]

        scale = np.linalg.norm(self.bounds[1] - self.bounds[0])
        self.rfuse, self.intraweight, self.interweight = 0.1 * scale, 1.0, 0.3
        self.w, self.cexp, self.cflow = 0.73, 1.49, 1.49
        self.k_sigmoid_max, self.k_sigmoid_min = 20.0, 3.0
        self.plasticity_rate, self.decay_rate = 0.05, 0.02
        self.cord_conductance = {}
        self.min_comm_size, self.min_comm_density = 3, 0.30
        self.last_n_communities_raw = 0
        self.last_n_communities_filtered = 0
        self.last_n_weak_communities = 0
        self.community_detection_algorithm = "leiden"
        self.use_roi = use_roi
        self.kmaxanchors, self.roialpha = 10, 0.1
        self.roispawnthresholdrel = 0.05
        self.fe_count, self.max_evals = 0, None
        self.tips = np.random.uniform(
            self.bounds[0], self.bounds[1], (self.ntips, self.ndim)
        )
        self.velocities = np.zeros_like(self.tips)
        self.anchors = np.empty((0, self.ndim))
        self.anchorfitness = np.empty(0)
        self.gbestpos, self.gbestval = None, np.inf
        self.history = []

    def buildgraph(self, nodes):
        graph = nx.Graph()
        graph.add_nodes_from(range(len(nodes)))
        tree = cKDTree(nodes)
        for i in range(len(nodes)):
            for j in tree.query_ball_point(nodes[i], float(self.rfuse)):
                if i != j:
                    key = tuple(sorted((i, j)))
                    graph.add_edge(
                        i,
                        j,
                        weight=self.cord_conductance.get(key, 1.0)
                        / (np.linalg.norm(nodes[i] - nodes[j]) + 1e-6),
                    )
        for u, v in graph.edges():
            self.cord_conductance.setdefault(tuple(sorted((u, v))), 1.0)
        return graph

    def detectcommunities(self, graph):
        if len(graph) < 2:
            return {node: 0 for node in graph}
        try:
            algorithm = self.community_detection_algorithm
            if algorithm == "leiden":
                import cdlib

                if graph.number_of_edges() == 0:
                    return {node: node for node in graph}
                communities = cdlib.algorithms.leiden(graph).communities
            else:
                community = nx.algorithms.community
                methods = {
                    "louvain": lambda: community.louvain_communities(graph, seed=42),
                    "label_propagation": lambda: community.label_propagation_communities(
                        graph
                    ),
                    "greedy_modularity": lambda: community.greedy_modularity_communities(
                        graph
                    ),
                    "asyn_lpa": lambda: community.asyn_lpa_communities(
                        graph, seed=42
                    ),
                    "connected_components": lambda: nx.connected_components(graph),
                }
                communities = methods.get(
                    algorithm, methods["louvain"]
                )()
            return {
                node: community_id
                for community_id, nodes in enumerate(communities)
                for node in nodes
            }
        except Exception:
            return {node: 0 for node in graph}

    def filter_weak_communities(self, graph, assignments):
        groups = {}
        for node, community_id in assignments.items():
            groups.setdefault(community_id, []).append(node)
        self.last_n_communities_raw = len(groups)
        self.last_n_weak_communities = 0
        filtered = assignments.copy()
        next_singleton = max(assignments.values(), default=-1) + 1
        for nodes in groups.values():
            size = len(nodes)
            density = nx.density(graph.subgraph(nodes)) if size > 1 else 0.0
            if size < self.min_comm_size or density < self.min_comm_density:
                self.last_n_weak_communities += 1
                for node in nodes:
                    filtered[node] = next_singleton
                    next_singleton += 1
        self.last_n_communities_filtered = len(set(filtered.values()))
        return filtered

    @staticmethod
    def getspikingpotential(fitness):
        low, high = np.min(fitness), np.max(fitness)
        return (
            np.ones_like(fitness)
            if low == high
            else (high - fitness) / (high - low + 1e-10)
        )

    def _eval(self, point):
        self.fe_count += 1
        return self.objfunc(point)

    def _update_cord_plasticity(self, tip, flow, graph):
        neighbours = [node for node in graph.neighbors(tip) if node < self.ntips]
        norm = np.linalg.norm(flow)
        for neighbour in neighbours:
            key = tuple(sorted((tip, neighbour)))
            conductance = self.cord_conductance[key]
            if norm >= 1e-10:
                direction = self.tips[neighbour] - self.tips[tip]
                alignment = np.dot(direction / (np.linalg.norm(direction) + 1e-10), flow / norm)
                if alignment > 0.3:
                    conductance = min(
                        conductance * (1 + self.plasticity_rate * alignment), 5.0
                    )
            self.cord_conductance[key] = max(conductance * (1 - self.decay_rate), 0.1)

    def computecommunityflow(self, graph, assignments, nodes, potentials):
        flow = np.zeros((self.ntips, self.ndim))
        for i in range(self.ntips):
            neighbours = list(graph.neighbors(i))
            total, weight = np.zeros(self.ndim), 0.0
            for j in neighbours:
                community_weight = (
                    self.intraweight
                    if assignments.get(i, 0) == assignments.get(j, 0)
                    else self.interweight
                )
                signal = 1 / (1 + np.exp(-self.ksigmoid * (potentials[j] - potentials[i])))
                message = community_weight * graph[i][j]["weight"] * signal
                total += message * (nodes[j] - nodes[i])
                weight += message
            if weight > 0:
                flow[i] = total / weight
        return flow

    def numericalgradient(self, point):
        value = self._eval(point)
        epsilon = 1e-5 * max(abs(value), 1.0)
        gradient = np.zeros_like(point)
        for index in range(len(point)):
            plus, minus = point.copy(), point.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            gradient[index] = (self._eval(plus) - self._eval(minus)) / (2 * epsilon)
        return gradient

    def roicorrect(self, midpoint, anchor_a, anchor_b, gradient):
        anchor_vector = anchor_b - anchor_a
        anchor_norm, gradient_norm = np.linalg.norm(anchor_vector), np.linalg.norm(gradient)
        if anchor_norm < 1e-9 or gradient_norm < 1e-9:
            return midpoint
        unit_anchor = anchor_vector / anchor_norm
        unit_gradient = gradient / gradient_norm
        return midpoint - self.roialpha * (
            unit_gradient - np.dot(unit_gradient, unit_anchor) * unit_anchor
        )

    def step(self):
        progress = min(self.fe_count / self.max_evals, 1.0) if self.max_evals else 0.0
        self.ksigmoid = self.k_sigmoid_max - (
            self.k_sigmoid_max - self.k_sigmoid_min
        ) * progress
        fitness = np.array([self._eval(tip) for tip in self.tips])
        best = np.argmin(fitness)
        if fitness[best] < self.gbestval:
            self.gbestval, self.gbestpos = fitness[best], self.tips[best].copy()
        if len(self.anchors) == 0 or np.min(
            np.linalg.norm(self.anchors - self.gbestpos, axis=1)
        ) > 1e-6:
            self.anchors = np.vstack((self.anchors, self.gbestpos))
            self.anchorfitness = np.append(self.anchorfitness, self.gbestval)
        if len(self.anchors) > self.kmaxanchors:
            keep = np.argsort(self.anchorfitness)[: self.kmaxanchors]
            self.anchors, self.anchorfitness = self.anchors[keep], self.anchorfitness[keep]

        all_nodes = np.vstack((self.tips, self.anchors))
        anchor_fitness = np.array([self._eval(anchor) for anchor in self.anchors])
        potentials = self.getspikingpotential(np.concatenate((fitness, anchor_fitness)))
        graph = self.buildgraph(all_nodes)
        assignments = self.filter_weak_communities(graph, self.detectcommunities(graph))
        flow = self.computecommunityflow(graph, assignments, all_nodes, potentials)
        delta = 0.05 * self.rfuse
        for i in range(self.ntips):
            if graph.degree(i) <= 1:
                self.velocities[i] += np.random.uniform(-delta, delta, self.ndim)
            self.velocities[i] = (
                self.w * self.velocities[i]
                + self.cexp * np.random.rand(self.ndim) * (self.gbestpos - self.tips[i])
                + self.cflow * flow[i]
            )
            self._update_cord_plasticity(i, flow[i], graph)
        self.tips = np.clip(self.tips + self.velocities, self.bounds[0], self.bounds[1])
        if self.use_roi and len(self.anchors) >= 2:
            a, b = np.random.choice(len(self.anchors), 2, replace=False)
            gap = abs(self.anchorfitness[a] - self.anchorfitness[b])
            relative_gap = gap / (max(self.anchorfitness) - min(self.anchorfitness) + 1e-10)
            if relative_gap > self.roispawnthresholdrel:
                midpoint = (self.anchors[a] + self.anchors[b]) / 2
                spawn = np.clip(
                    self.roicorrect(midpoint, self.anchors[a], self.anchors[b], self.numericalgradient(midpoint)),
                    self.bounds[0],
                    self.bounds[1],
                )
                if self._eval(spawn) < np.max(fitness):
                    self.tips[np.argmax(fitness)] = spawn
        self.history.append(self.gbestval)
        return self.gbestval

    def optimize(self, max_evals=100000, verbose=True, progress_desc=None):
        self.fe_count, self.max_evals = 0, max_evals
        progress = tqdm(
            total=max_evals,
            desc=progress_desc or "Optimization",
            unit="FE",
            leave=False,
            disable=not verbose,
        )
        try:
            while self.fe_count < max_evals:
                previous_fe = self.fe_count
                self.step()
                progress.update(self.fe_count - previous_fe)
        finally:
            progress.close()
        return self.gbestpos.copy(), self.gbestval


BACKENDS = [
    "leiden",
    "louvain",
    "greedy_modularity",
    "label_propagation",
    "asyn_lpa",
    "connected_components",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="CEC 2022 benchmark for Myco community-detection backends"
    )
    parser.add_argument(
        "--backend",
        choices=BACKENDS + ["all"],
        default="greedy_modularity",
        help="Backend to test, or 'all' to test every supported backend",
    )
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument(
        "--max-fe",
        type=int,
        default=None,
        help="Override the standard CEC function-evaluation budget",
    )
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--ntips", type=int, default=30)
    parser.add_argument(
        "--functions",
        nargs="+",
        default=None,
        help="CEC function keys (e.g., F1, F2, F6)",
    )
    parser.add_argument(
        "--quick-runs",
        type=int,
        default=None,
        help="Override --runs for a short spot-check",
    )
    parser.add_argument("--roi", action="store_true", help="Enable ROI spawning")
    parser.add_argument(
        "--no-diag",
        action="store_true",
        help="Disable community diagnostics in the output",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-function and per-run progress output",
    )
    return parser.parse_args()


def get_max_fe(dimension):
    if dimension == 10:
        return 20000 * dimension
    if dimension == 20:
        return 50000 * dimension
    else:
        raise ValueError("INPUT DATA only supports DIM 10 and 20.")


def get_convergence_points(dimension, max_fe):
    return [
        max(1, min(int(dimension ** (k / 5.0 - 3.0) * max_fe), max_fe))
        for k in range(16)
    ]


def load_seeds():
    seed_file = Path(__file__).resolve().parent / "input_data" / "Rand_Seeds.txt"
    with seed_file.open(encoding="utf-8") as handle:
        return [float(line.strip()) for line in handle if line.strip()]


def get_cec2022_seed(seeds, dimension, function_number, runs, run_id):
    seed_index = (dimension // 10 * function_number * runs + run_id) - runs
    seed_index = (seed_index % 1000) + 1
    return seeds[seed_index - 1]


def run_optimizer(backend, function, dimension, max_fe, ntips, use_roi, seed, quiet):
    np.random.seed(int(seed))
    bounds = np.array([[-100.0] * dimension, [100.0] * dimension])
    optimizer = MycoOptimizerLeiden(
        function, bounds, ntips=ntips, ndim=dimension, use_roi=use_roi
    )
    optimizer.community_detection_algorithm = backend
    best_pos, best_value = optimizer.optimize(
        max_evals=max_fe,
        verbose=not quiet,
        progress_desc=f"Myco ({backend})",
    )

    convergence = []
    for fe_point in get_convergence_points(dimension, max_fe):
        step = (fe_point - 1) // ntips + 1
        convergence.append(
            optimizer.history[step - 1]
            if step <= len(optimizer.history)
            else best_value
        )

    diagnostics = (
        optimizer.last_n_communities_raw,
        optimizer.last_n_communities_filtered,
        optimizer.last_n_weak_communities,
    )
    return best_pos, best_value, convergence, diagnostics


def benchmark_backend(
    backend,
    functions,
    seeds,
    dimension,
    runs,
    ntips,
    use_roi,
    diagnostics_enabled,
    output_dir,
    quiet,
    max_fe_override,
):
    max_fe = max_fe_override if max_fe_override is not None else get_max_fe(dimension)
    backend_slug = backend.replace("_", "-")
    result_dir = output_dir / f"Myco ({backend_slug})"
    models_dir = result_dir / "NPY Files"
    result_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    output_file = result_dir / f"Results D{dimension}.xlsx"
    convergence_file = result_dir / f"Convergence data D{dimension}.json"
    seeds_file = result_dir / f"Seeds D{dimension}.json"

    results = []
    convergence_data = []

    for function_name, function in functions.items():
        metadata = CEC_METADATA[int(function_name[1:])]
        bias = metadata["bias"]
        if not quiet:
            print(f"\nEvaluating {function_name} ({metadata['name']}) with {backend}...")

        fitnesses = []
        histories = []
        raw_counts, filtered_counts, weak_counts = [], [], []
        progress = tqdm(
            range(runs),
            desc=f"  Myco ({backend})",
            leave=False,
            disable=quiet,
        )
        for run_id in progress:
            seed = get_cec2022_seed(
                seeds, dimension, int(function_name[1:]), runs, run_id
            )
            try:
                _, best_value, convergence, comm_diags = run_optimizer(
                    backend,
                    function,
                    dimension,
                    max_fe,
                    ntips,
                    use_roi,
                    seed,
                    quiet,
                )
                error = best_value - bias
                fitnesses.append(0.0 if error < 1e-8 else error)
                histories.append(convergence)
                raw_counts.append(comm_diags[0])
                filtered_counts.append(comm_diags[1])
                weak_counts.append(comm_diags[2])
                progress.set_postfix_str(f"Best: {best_value:.2e}")
            except Exception as error:
                print(
                    f"\nError in {backend} on {function_name} run {run_id}: {error}"
                )
                fitnesses.append(np.inf)
                histories.append([])
                raw_counts.append(0)
                filtered_counts.append(0)
                weak_counts.append(0)

        valid = np.asarray(fitnesses)
        valid = valid[np.isfinite(valid)]
        if len(valid):
            result = {
                "Function": function_name,
                "Algorithm": f"Myco (-{backend_slug})",
                "Dimension": dimension,
                "Mean": float(np.mean(valid)),
                "Std": float(np.std(valid)),
                "Median": float(np.median(valid)),
                "Q1": float(np.percentile(valid, 25)),
                "Q3": float(np.percentile(valid, 75)),
                "IQR": float(np.percentile(valid, 75) - np.percentile(valid, 25)),
                "Best": float(np.min(valid)),
                "Worst": float(np.max(valid)),
                "Runs": runs,
                "MaxFE": max_fe,
                "Bias": bias,
            }
            convergence_data.append(
                {
                    "function": function_name,
                    "algorithm": f"Myco (-{backend_slug})",
                    "dimension": dimension,
                    "convergence_histories": histories,
                    "seeds": [
                        get_cec2022_seed(
                            seeds, dimension, int(function_name[1:]), runs, run_id
                        )
                        for run_id in range(runs)
                    ],
                }
            )
        else:
            result = {
                "Function": function_name,
                "Algorithm": f"Myco (-{backend_slug})",
                "Dimension": dimension,
                "Mean": np.inf,
                "Std": np.inf,
                "Median": np.inf,
                "Q1": np.inf,
                "Q3": np.inf,
                "IQR": np.inf,
                "Best": np.inf,
                "Worst": np.inf,
                "Runs": runs,
                "MaxFE": max_fe,
                "Bias": bias,
            }

        if diagnostics_enabled:
            result.update(
                {
                    "CommRawMean": float(np.mean(raw_counts)),
                    "CommRawMedian": float(np.median(raw_counts)),
                    "CommFilteredMean": float(np.mean(filtered_counts)),
                    "CommFilteredMedian": float(np.median(filtered_counts)),
                    "WeakCommMean": float(np.mean(weak_counts)),
                    "WeakCommMedian": float(np.median(weak_counts)),
                }
            )
        results.append(result)

    pd.DataFrame(results).to_excel(output_file, index=False)
    convergence_file.write_text(json.dumps(convergence_data, indent=2), encoding="utf-8")
    seeds_file.write_text(
        json.dumps(
            {
                "dimension": dimension,
                "n_runs": runs,
                "functions": list(functions),
                "backend": backend,
                "seed_source": "input_data/Rand_Seeds.txt",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Results saved to {output_file}")


def main():
    args = parse_args()
    runs = args.quick_runs if args.quick_runs is not None else args.runs
    all_functions = {**CEC_CORE, **CEC_EXTRA}
    if args.functions:
        unknown = [name for name in args.functions if name not in all_functions]
        if unknown:
            raise ValueError(f"Unknown CEC function(s): {', '.join(unknown)}")
        functions = {name: all_functions[name] for name in args.functions}
    else:
        functions = all_functions

    backends = BACKENDS if args.backend == "all" else [args.backend]
    seeds = load_seeds()
    for backend in backends:
        benchmark_backend(
            backend,
            functions,
            seeds,
            args.dim,
            runs,
            args.ntips,
            args.roi,
            not args.no_diag,
            args.output_dir,
            args.quiet,
            args.max_fe,
        )


if __name__ == "__main__":
    main()
