"""
Mycelial Search (Myco) + Cord Plasticity
=================================================
Extends Louvain Myco with cord plasticity for ridges.
"""

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree


class MycelialSearchPlasticity:
    """
    Myco with cord plasticity reinforces hyphal highways along resource gradients.
    API identical to MycelialSearchLouvain for ablation compatibility.
    """

    def __init__(self, objfunc, bounds, ntips=30, ndim=None, use_roi=True):
        self.objfunc = objfunc
        self.bounds = np.array(bounds)
        if self.bounds.shape[0] == 2:
            pass
        elif self.bounds.shape[1] == 2:
            self.bounds = self.bounds.T

        self.ntips = ntips
        self.ndim = ndim or self.bounds.shape[1]

        # Network parameters (same as Louvain baseline)
        scale = np.linalg.norm(self.bounds[1] - self.bounds[0])
        self.rfuse = 0.1 * scale
        self.intraweight = 1.0
        self.interweight = 0.3

        # Core parameters (same as Louvain baseline)
        self.w = 0.73
        self.cexp = 1.49
        self.cflow = 1.49
        self.ksigmoid = 10

        # PLASTICITY PARAMETERS
        self.plasticity_rate = 0.05  # 5% reinforcement per step
        self.decay_rate = 0.02  # 2% pruning per step
        self.cord_conductance = {}  # Dynamic cord weights

        # ROI control
        self.use_roi = use_roi

        # FE accounting
        self.fe_count = 0
        self.max_evals = None

        # ROI parameters (same as Louvain baseline)
        self.kmaxanchors = 10
        self.roialpha = 0.1
        self.roispawnthresholdrel = 0.05

        # Initialise
        self.tips = np.random.uniform(self.bounds[0], self.bounds[1], (self.ntips, self.ndim))
        self.velocities = np.zeros_like(self.tips)
        self.anchors = np.empty((0, self.ndim))
        self.anchorfitness = np.empty(0)
        self.gbestpos = None
        self.gbestval = np.inf
        self.history = []
        self.iteration = 0

        print(f"Myco (Plasticity) initialised: ntips={self.ntips}, ndim={self.ndim}")

    def buildgraph(self, nodes):
        """Build spatial graph with plasticity-aware weights."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(nodes)))
        tree = cKDTree(nodes)

        for i in range(len(nodes)):
            indices = tree.query_ball_point(nodes[i], float(self.rfuse))
            for j in indices:
                if i != j:
                    dist = np.linalg.norm(nodes[i] - nodes[j])
                    edge_key = tuple(sorted((i, j)))
                    # Use existing conductance if present (plasticity), else 1.0
                    g = self.cord_conductance.get(edge_key, 1.0)
                    weight = g / (dist + 1e-6)
                    graph.add_edge(i, j, weight=weight)

        # Ensure every edge has an entry in cord_conductance
        for u, v in graph.edges():
            edge_key = tuple(sorted((u, v)))
            if edge_key not in self.cord_conductance:
                self.cord_conductance[edge_key] = 1.0

        return graph

    def detectcommunities(self, graph):
        """Louvain communities (identical to baseline, no external package)."""
        if len(graph.nodes) < 2:
            return {i: 0 for i in range(len(graph.nodes))}

        try:
            from networkx.algorithms import community
            communities_gen = community.louvain_communities(graph, seed=42)
            nodetocomm = {}
            for comm_id, comm_nodes in enumerate(communities_gen):
                for node in comm_nodes:
                    nodetocomm[node] = comm_id
            return nodetocomm
        except ImportError:
            return {i: 0 for i in range(len(graph.nodes))}

    @staticmethod
    def getspikingpotential(fitnessvals):
        """Spiking potentials (identical to baseline)."""
        fmin, fmax = np.min(fitnessvals), np.max(fitnessvals)
        if fmax == fmin:
            return np.ones_like(fitnessvals)
        return (fmax - fitnessvals) / (fmax - fmin + 1e-10)

    def _eval(self, x):
        """Objective evaluation with FE counting."""
        val = self.objfunc(x)
        self.fe_count += 1
        return val

    def _update_cord_plasticity(self, tip_idx, flow_direction, graph):
        """Cord plasticity: only TIP-TIP edges."""
        neighbors = list(graph.neighbors(tip_idx))
        if not neighbors:
            return

        flow_norm = np.linalg.norm(flow_direction)
        if flow_norm < 1e-10:
            # Global decay for tip-tip edges only
            for n_idx in neighbors:
                if n_idx < self.ntips:  # Skip anchors
                    edge_key = tuple(sorted((tip_idx, n_idx)))
                    self.cord_conductance[edge_key] *= (1 - self.decay_rate)
                    self.cord_conductance[edge_key] = max(self.cord_conductance[edge_key], 0.1)
            return

        flow_unit = flow_direction / flow_norm

        for neighbor_idx in neighbors:
            if neighbor_idx >= self.ntips:  # Skip anchors
                continue

            edge_key = tuple(sorted((tip_idx, neighbor_idx)))

            cord_vector = self.tips[neighbor_idx] - self.tips[tip_idx]
            cord_norm = np.linalg.norm(cord_vector)
            if cord_norm < 1e-10:
                continue

            cord_direction = cord_vector / cord_norm
            alignment = np.dot(cord_direction, flow_unit)

            if alignment > 0.3:
                self.cord_conductance[edge_key] *= (1 + self.plasticity_rate * alignment)
                self.cord_conductance[edge_key] = min(self.cord_conductance[edge_key], 5.0)

            self.cord_conductance[edge_key] *= (1 - self.decay_rate)
            self.cord_conductance[edge_key] = max(self.cord_conductance[edge_key], 0.1)

    def computecommunityflow(self, graph, nodetocomm, allnodes, allpotentials):
        """Community flow with cord plasticity weights (extends baseline)."""
        flow = np.zeros((len(self.tips), self.ndim))

        for i in range(len(self.tips)):
            if i not in graph.nodes:
                continue

            comm_i = nodetocomm.get(i, 0)
            neighbors = list(graph.neighbors(i))
            if len(neighbors) == 0:
                continue

            total_flow = np.zeros(self.ndim)
            total_weight = 0.0

            for j in neighbors:
                comm_j = nodetocomm.get(j, 0)
                wcomm = self.intraweight if comm_i == comm_j else self.interweight

                delta_s = allpotentials[j] - allpotentials[i]
                sigmoid_val = 1.0 / (1.0 + np.exp(-self.ksigmoid * delta_s))

                edge_weight = graph[i][j]['weight']
                msg_weight = wcomm * edge_weight * sigmoid_val
                direction = allnodes[j] - allnodes[i]

                total_flow += msg_weight * direction
                total_weight += msg_weight

            if total_weight > 0:
                flow[i] = total_flow / total_weight

        return flow

    def numericalgradient(self, x, eps=None):
        """Numerical gradient (identical to baseline)."""
        if eps is None:
            fx = self._eval(x)
            eps = 1e-5 * max(abs(fx), 1.0)
        grad = np.zeros_like(x)
        for i in range(len(x)):
            xplus = x.copy()
            xplus[i] += eps
            xminus = x.copy()
            xminus[i] -= eps
            f_plus = self._eval(xplus)
            f_minus = self._eval(xminus)
            grad[i] = (f_plus - f_minus) / (2 * eps)
        return grad

    def roicorrect(self, mpoint, anchora, anchorb, grad):
        """ROI correction (identical to baseline)."""
        abvec = anchorb - anchora
        ab_norm = np.linalg.norm(abvec)
        if ab_norm < 1e-9:
            return mpoint

        ab_unit = abvec / ab_norm
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-9:
            return mpoint

        grad_unit = grad / grad_norm
        grad_parallel = np.dot(grad_unit, ab_unit) * ab_unit
        grad_perp = grad_unit - grad_parallel
        return mpoint - self.roialpha * grad_perp

    def step(self):
        """ Single step with plasticity (fixed indexing)."""
        tipsfitness = np.empty(self.ntips)
        for i in range(self.ntips):
            tipsfitness[i] = self._eval(self.tips[i])

        for i in range(self.ntips):
            if tipsfitness[i] < self.gbestval:
                self.gbestval = tipsfitness[i]
                self.gbestpos = self.tips[i].copy()

        # Anchor management
        if self.gbestpos is not None:
            if len(self.anchors) == 0 or np.min(np.linalg.norm(self.anchors - self.gbestpos, axis=1)) > 1e-6:
                self.anchors = np.vstack([self.anchors, self.gbestpos])
                self.anchorfitness = np.append(self.anchorfitness, self.gbestval)

            if len(self.anchors) > self.kmaxanchors:
                top_idx = np.argsort(self.anchorfitness)[:self.kmaxanchors]
                self.anchors = self.anchors[top_idx]
                self.anchorfitness = self.anchorfitness[top_idx]

        # Network flow (tips + anchors)
        allnodes = np.vstack([self.tips, self.anchors])
        if len(self.anchors) > 0:
            anchorsfitness = np.empty(len(self.anchors))
            for k in range(len(self.anchors)):
                anchorsfitness[k] = self._eval(self.anchors[k])
        else:
            anchorsfitness = np.array([])

        allfitness = np.concatenate([tipsfitness, anchorsfitness])
        allpotentials = self.getspikingpotential(allfitness)

        graph = self.buildgraph(allnodes)
        nodetocomm = self.detectcommunities(graph)
        flowvectors = self.computecommunityflow(graph, nodetocomm, allnodes, allpotentials)

        # velocities + placticity
        newtips = self.tips.copy()
        for i in range(self.ntips):
            r1 = np.random.rand(self.ndim)
            self.velocities[i] = (self.w * self.velocities[i] +
                                  self.cexp * r1 * (self.gbestpos - self.tips[i]) +
                                  self.cflow * flowvectors[i])

            # Only update plasticity for TIP-TIP edges (ignore anchor neighbours)
            tip_flow = flowvectors[i]
            self._update_cord_plasticity(i, tip_flow, graph)

            newtips[i] = self.tips[i] + self.velocities[i]

        self.tips = np.clip(newtips, self.bounds[0], self.bounds[1])

        # ROI spawning (disabled in main benchmarks when use_roi=False)
        if self.use_roi and len(self.anchors) >= 2:
            idxa, idxb = np.random.choice(len(self.anchors), 2, replace=False)
            fitnessgap = abs(self.anchorfitness[idxa] - self.anchorfitness[idxb])
            fitnessrange = max(self.anchorfitness) - min(self.anchorfitness) + 1e-10
            relativegap = fitnessgap / fitnessrange

            if relativegap > self.roispawnthresholdrel:
                mpoint = (self.anchors[idxa] + self.anchors[idxb]) / 2
                grad = self.numericalgradient(mpoint)
                xspawn = self.roicorrect(mpoint, self.anchors[idxa], self.anchors[idxb], grad)
                xspawn = np.clip(xspawn, self.bounds[0], self.bounds[1])
                fspawn = self._eval(xspawn)

                worstidx = np.argmax(tipsfitness)
                if fspawn < tipsfitness[worstidx]:
                    self.tips[worstidx] = xspawn

        self.iteration += 1
        self.history.append(self.gbestval)
        return self.gbestval

    def optimize(self, max_evals=100000, verbose=True):
        """Run optimisation up to a maximum number of function evaluations."""
        self.fe_count = 0
        self.max_evals = max_evals
        it = 0

        while self.fe_count < self.max_evals:
            self.step()
            it += 1
            if verbose and it % 100 == 0:
                print(f"Iter {it:4d}: Best = {self.gbestval:.2e}, FEs = {self.fe_count}")

        return self.gbestpos.copy(), self.gbestval

