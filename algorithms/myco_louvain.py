"""
Myco (Louvain): Louvain Community Detection
Network = Graph with communities (mycelial aggregates)
Flow: Strong intra-community, weak inter-community
"""

import numpy as np
from scipy.spatial import cKDTree
import networkx as nx


class MycelialSearchLouvain:
    """
    True mycelial network: Louvain communities on spatial graph
    Intra-community: Strong coordination (shared cytoplasm)
    Inter-community: Weak long-range signalling
    """
    # ROI can be True/False for the ablation study
    def __init__(self, obj_func, bounds, n_tips=30, n_dim=None, use_roi=True):
        self.obj_func = obj_func
        self.bounds = np.array(bounds)
        if self.bounds.shape[0] == 2:
            pass
        elif self.bounds.shape[1] == 2:
            self.bounds = self.bounds.T

        self.n_tips = n_tips
        self.n_dim = n_dim or self.bounds.shape[1]

        # Network parameters
        self.r_fuse = 0.1 * np.linalg.norm(self.bounds[1] - self.bounds[0])
        self.intra_weight = 1.0
        self.inter_weight = 0.3

        # Core parameters
        self.w = 0.73
        self.c_exp = 1.49
        self.c_flow = 1.49
        self.k_sigmoid = 10

        self.k_max_anchors = 10
        self.roi_alpha = 0.1
        self.roi_spawn_threshold_rel = 0.05

        # ROI control
        self.use_roi = use_roi

        # FE accounting
        self.fe_count = 0
        self.max_evals = None

        # Initialise
        self.tips = np.random.uniform(self.bounds[0], self.bounds[1], (self.n_tips, self.n_dim))
        self.velocities = np.zeros_like(self.tips)
        self.anchors = np.empty((0, self.n_dim))
        self.anchor_fitness = np.empty(0)
        self.g_best_pos = None
        self.g_best_val = np.inf
        self.history = []
        self.iteration = 0

        print(f"Myco (Louvain) initialised: n_tips={self.n_tips}, n_dim={self.n_dim}")

    def _build_graph(self, nodes):
        """Build spatial graph: edges within r_fuse."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(nodes)))

        tree = cKDTree(nodes)
        for i in range(len(nodes)):
            indices = tree.query_ball_point(nodes[i], float(self.r_fuse))
            for j in indices:
                if i < j:
                    dist = np.linalg.norm(nodes[i] - nodes[j])
                    graph.add_edge(i, j, weight=1.0 / (dist + 1e-6))

        return graph

    def _detect_communities(self, graph):
        """Louvain community detection."""
        if len(graph.nodes()) < 2:
            return {0: 0}

        try:
            from networkx.algorithms import community
            communities_gen = community.louvain_communities(graph, seed=42)

            node_to_comm = {}
            for comm_id, comm_nodes in enumerate(communities_gen):
                for node in comm_nodes:
                    node_to_comm[node] = comm_id

            return node_to_comm
        except ImportError:
            return {i: 0 for i in range(len(graph.nodes()))}

    @staticmethod
    def _get_spiking_potential(fitness_vals):
        """Compute spiking potentials from fitness values."""
        f_min, f_max = np.min(fitness_vals), np.max(fitness_vals)
        if f_max == f_min:
            return np.ones_like(fitness_vals)
        return (f_max - fitness_vals) / (f_max - f_min + 1e-10)

    def _eval(self, x):
        """Objective evaluation with FE counting."""
        val = self.obj_func(x)
        self.fe_count += 1
        return val

    def _compute_community_flow(self, graph, node_to_comm, all_nodes, all_potentials):
        """Message passing with community awareness."""
        flow = np.zeros((len(self.tips), self.n_dim))

        for i in range(len(self.tips)):
            if i not in graph.nodes():
                continue

            comm_i = node_to_comm.get(i, 0)
            neighbors = list(graph.neighbors(i))

            if len(neighbors) == 0:
                continue

            total_flow = np.zeros(self.n_dim)
            total_weight = 0.0

            for j in neighbors:
                comm_j = node_to_comm.get(j, 0)
                w_comm = self.intra_weight if comm_i == comm_j else self.inter_weight

                delta_s = all_potentials[j] - all_potentials[i]
                sigmoid_val = 1.0 / (1.0 + np.exp(-self.k_sigmoid * delta_s))
                edge_weight = graph[i][j]['weight']

                msg_weight = w_comm * edge_weight * sigmoid_val
                direction = all_nodes[j] - all_nodes[i]
                total_flow += msg_weight * direction
                total_weight += msg_weight

            if total_weight > 0:
                flow[i] = total_flow / total_weight

        return flow

    def _numerical_gradient(self, x, eps=None):
        """Compute numerical gradient."""
        if eps is None:
            f_x = self._eval(x)
            eps = 1e-5 * max(abs(f_x), 1.0)
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus, x_minus = x.copy(), x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            f_plus = self._eval(x_plus)
            f_minus = self._eval(x_minus)
            grad[i] = (f_plus - f_minus) / (2 * eps)
        return grad

    def _roi_correct(self, m_point, anchor_a, anchor_b, grad):
        """Ridge-oriented correction."""
        ab_vec = anchor_b - anchor_a
        ab_norm = ab_vec / (np.linalg.norm(ab_vec) + 1e-9)
        grad_norm = grad / (np.linalg.norm(grad) + 1e-9)
        grad_parallel = np.dot(grad_norm, ab_norm) * ab_norm
        grad_perp = grad_norm - grad_parallel
        return m_point - self.roi_alpha * grad_perp

    def step(self):
        """Single optimisation step."""
        tips_fitness = np.empty(self.n_tips)
        for i in range(self.n_tips):
            tips_fitness[i] = self._eval(self.tips[i])

        for i in range(self.n_tips):
            if tips_fitness[i] < self.g_best_val:
                self.g_best_val = tips_fitness[i]
                self.g_best_pos = self.tips[i].copy()

        if self.g_best_pos is not None:
            if len(self.anchors) == 0 or np.min(np.linalg.norm(self.anchors - self.g_best_pos, axis=1)) > 1e-6:
                self.anchors = np.vstack([self.anchors, self.g_best_pos])
                self.anchor_fitness = np.append(self.anchor_fitness, self.g_best_val)

        if len(self.anchors) > self.k_max_anchors:
            top_idx = np.argsort(self.anchor_fitness)[:self.k_max_anchors]
            self.anchors = self.anchors[top_idx]
            self.anchor_fitness = self.anchor_fitness[top_idx]

        all_nodes = np.vstack([self.tips, self.anchors])
        if len(self.anchors) > 0:
            anchors_fitness = np.empty(len(self.anchors))
            for k in range(len(self.anchors)):
                anchors_fitness[k] = self._eval(self.anchors[k])
        else:
            anchors_fitness = np.array([])
        all_fitness = np.concatenate([tips_fitness, anchors_fitness])
        all_potentials = self._get_spiking_potential(all_fitness)

        graph = self._build_graph(all_nodes)
        node_to_comm = self._detect_communities(graph)
        flow_vectors = self._compute_community_flow(graph, node_to_comm, all_nodes, all_potentials)

        new_tips = self.tips.copy()
        for i in range(self.n_tips):
            r1 = np.random.rand(self.n_dim)
            self.velocities[i] = (
                self.w * self.velocities[i] +
                self.c_exp * r1 * (self.g_best_pos - self.tips[i]) +
                self.c_flow * flow_vectors[i]
            )
            new_tips[i] = self.tips[i] + self.velocities[i]

        self.tips = np.clip(new_tips, self.bounds[0], self.bounds[1])

        # ROI (Disabled when use_roi=False)
        if self.use_roi and len(self.anchors) >= 2:
            idx_a, idx_b = np.random.choice(len(self.anchors), 2, replace=False)
            fitness_gap = abs(self.anchor_fitness[idx_a] - self.anchor_fitness[idx_b])
            fitness_range = (max(self.anchor_fitness) - min(self.anchor_fitness) + 1e-10)
            relative_gap = fitness_gap / fitness_range

            if relative_gap < self.roi_spawn_threshold_rel:
                m_point = (self.anchors[idx_a] + self.anchors[idx_b]) / 2
                grad = self._numerical_gradient(m_point)
                x_spawn = self._roi_correct(m_point, self.anchors[idx_a], self.anchors[idx_b], grad)
                x_spawn = np.clip(x_spawn, self.bounds[0], self.bounds[1])
                f_spawn = self._eval(x_spawn)
                worst_idx = np.argmax(tips_fitness)
                if f_spawn < tips_fitness[worst_idx]:
                    self.tips[worst_idx] = x_spawn

        self.iteration += 1
        self.history.append(self.g_best_val)
        return self.g_best_val

    def optimize(self, max_evals=100000, verbose=True):
        """Run optimisation up to a maximum number of function evaluations."""
        self.fe_count = 0
        self.max_evals = max_evals
        it = 0

        while self.fe_count < self.max_evals:
            self.step()
            it += 1
            if verbose and it % 100 == 0:
                print(f"Iter {it:4d}: Best = {self.g_best_val:.2e}")

        return self.g_best_pos.copy(), self.g_best_val
