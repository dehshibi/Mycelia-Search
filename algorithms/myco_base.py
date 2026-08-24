import numpy as np
from scipy.spatial import cKDTree


class MycelialSearch(object):
    """
    Mycelial Search V1.0
    use_flow: False = PSO-like
              True  = Myco with network flow
    """

    def __init__(self, obj_func, bounds, n_tips=30, n_dim=None, use_flow=False):
        """

        """
        self.obj_func = obj_func
        self.bounds = np.array(bounds)
        # Normalise bounds shape
        if self.bounds.shape[0] == 2:
            pass  # already (2, D)
        elif self.bounds.shape[1] == 2:
            self.bounds = self.bounds.T  # (D, 2) → (2, D)
        else:
            raise ValueError(f"bounds must be (2, D) or (D, 2), got {self.bounds.shape}")

        self.n_tips = n_tips
        self.n_dim = n_dim or self.bounds.shape[1]
        self.use_flow = use_flow  # Network flow toggle

        # Hyperparameters
        self.w = 0.73
        self.c_exp = 1.49
        self.c_flow = 1.49  # only used if use_flow=True
        self.k_sigmoid = 10
        self.k_neighbors = 5
        self.k_max_anchors = 10
        self.roi_alpha = 0.1
        self.r_fuse = 0.1 * np.linalg.norm(self.bounds[1] - self.bounds[0])
        self.roi_spawn_threshold_rel = 0.05

        # Initialise 
        self.tips = np.random.uniform(self.bounds[0], self.bounds[1], (self.n_tips, self.n_dim))
        self.velocities = np.zeros_like(self.tips)
        self.anchors = np.empty((0, self.n_dim))
        self.anchor_fitness = np.empty(0)
        self.g_best_pos = None
        self.g_best_val = np.inf
        self.history = []
        self.iteration = 0

    def _get_spiking_potential(self, fitness_vals):
        f_min = np.min(fitness_vals)
        f_max = np.max(fitness_vals)
        if f_max == f_min:
            return np.ones_like(fitness_vals, dtype=float)
        return (f_max - fitness_vals) / (f_max - f_min + 1e-10)

    def _numerical_gradient(self, x, eps=None):
        if eps is None:
            f_x = self.obj_func(x)
            eps = 1e-5 * max(abs(f_x), 1.0)
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            grad[i] = (self.obj_func(x_plus) - self.obj_func(x_minus)) / (2.0 * eps)
        return grad

    def _roi_correct(self, m_point, anchor_a, anchor_b, grad, alpha=0.1):
        """
        Ridge-oriented correction perpendicular to the anchor pair line.
        """
        AB = anchor_b - anchor_a
        AB_norm = AB / (np.linalg.norm(AB) + 1e-9)
        grad_norm = grad / (np.linalg.norm(grad) + 1e-9)
        grad_parallel = np.dot(grad_norm, AB_norm) * AB_norm
        grad_perp = grad_norm - grad_parallel
        x_spawn = m_point - alpha * grad_perp
        return x_spawn

    def step(self):
        # Evaluate tips
        tips_fitness = np.array([self.obj_func(t) for t in self.tips])

        # Update global best and create anchors
        for i in range(self.n_tips):
            if tips_fitness[i] < self.g_best_val:
                self.g_best_val = tips_fitness[i]
                self.g_best_pos = self.tips[i].copy()
                self.anchors = np.vstack([self.anchors, self.tips[i]])
                self.anchor_fitness = np.append(self.anchor_fitness, tips_fitness[i])

        # Prune anchors
        if len(self.anchors) > self.k_max_anchors:
            top_indices = np.argsort(self.anchor_fitness)[:self.k_max_anchors]
            self.anchors = self.anchors[top_indices]
            self.anchor_fitness = self.anchor_fitness[top_indices]

        # Re-evaluate anchors for current potentials
        anchors_fitness = np.array([self.obj_func(a) for a in self.anchors]) if len(self.anchors) > 0 else np.array([])
        all_fitness = np.concatenate([tips_fitness, anchors_fitness])
        all_potentials = self._get_spiking_potential(all_fitness)

        # Build KDTree
        all_nodes = np.vstack([self.tips, self.anchors])
        tree = cKDTree(all_nodes)

        new_tips = self.tips.copy()

        # Velocity update for each tip
        for i in range(self.n_tips):
            # Anastomosis: find neighbors
            k_query = min(self.k_neighbors + 1, len(all_nodes))
            dists, idxs = tree.query(
                self.tips[i], k=k_query, distance_upper_bound=self.r_fuse
            )

            if np.isscalar(idxs):
                idxs = np.array([idxs])
                dists = np.array([dists])

            # Filter valid neighbors
            valid_mask = (
                    (idxs >= 0) & (idxs < len(all_nodes)) &
                    (idxs != i) & (dists < self.r_fuse)
            )
            neighbors_idx = idxs[valid_mask]

            # Limit neighbors
            if len(neighbors_idx) > self.k_neighbors:
                neighbors_idx = np.random.choice(
                    neighbors_idx, self.k_neighbors, replace=False
                )

            # Network flow (DISABLED BY DEFAULT)
            flow_vector = np.zeros(self.n_dim)
            if self.use_flow:
                for j in neighbors_idx:
                    delta_s = all_potentials[j] - all_potentials[i]
                    sigmoid_val = 1.0 / (1.0 + np.exp(-self.k_sigmoid * delta_s))
                    r2 = np.random.rand(self.n_dim)
                    flow_vector += r2 * sigmoid_val * (all_nodes[j] - self.tips[i])

            # Core velocity update
            r1 = np.random.rand(self.n_dim)
            self.velocities[i] = (
                    self.w * self.velocities[i] +
                    self.c_exp * r1 * (self.g_best_pos - self.tips[i]) +
                    self.c_flow * flow_vector  # zero if use_flow=False
            )

            new_tips[i] = self.tips[i] + self.velocities[i]

        # Boundary enforcement
        self.tips = np.clip(new_tips, self.bounds[0], self.bounds[1])

        # ROI 
        if len(self.anchors) >= 2:
            idx_a, idx_b = np.random.choice(len(self.anchors), 2, replace=False)
            fitness_gap = abs(self.anchor_fitness[idx_a] - self.anchor_fitness[idx_b])
            fitness_range = (max(self.anchor_fitness) - min(self.anchor_fitness) + 1e-10)
            relative_gap = fitness_gap / fitness_range

            if relative_gap < self.roi_spawn_threshold_rel:
                m_point = (self.anchors[idx_a] + self.anchors[idx_b]) / 2.0
                grad = self._numerical_gradient(m_point)
                x_spawn = self._roi_correct(m_point, self.anchors[idx_a], self.anchors[idx_b], grad)
                x_spawn = np.clip(x_spawn, self.bounds[0], self.bounds[1])

                f_spawn = self.obj_func(x_spawn)
                worst_idx = np.argmax(tips_fitness)
                if f_spawn < tips_fitness[worst_idx]:
                    self.tips[worst_idx] = x_spawn

        # Record history
        self.iteration += 1
        self.history.append(self.g_best_val)

        return self.g_best_val

    def optimize(self, n_iterations=1000, verbose=True):
        """Run optimisation and return best position and value."""
        for it in range(n_iterations):
            self.step()
            if verbose and (it + 1) % 100 == 0:
                print(f"Iter {it + 1:4d}: Best = {self.g_best_val:.2e}")
        return self.g_best_pos.copy(), self.g_best_val

    def __repr__(self):
        status = "use_flow=False (network disabled)" if not self.use_flow else "use_flow=True (full Myco)"
        return f"Myco(n_dim={self.n_dim}, n_tips={self.n_tips}, status='{status}', best={self.g_best_val:.2e})"


# ===== EXAMPLE USAGE & TESTING =====
if __name__ == "__main__":
    import matplotlib.pyplot as plt


    # Define test functions
    def sphere(x):
        """Simple unimodal function: sum of squares."""
        return np.sum(x ** 2)


    def ackley(x):
        """Plateau-like function with many local minima."""
        n = len(x)
        a = 20
        b = 0.2
        c = 2 * np.pi

        sum1 = np.sum(x ** 2)
        sum2 = np.sum(np.cos(c * x))

        return -a * np.exp(-b * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + a + np.e


    def rosenbrock(x):
        """Ridge function: curved valley."""
        return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


    def rastrigin(x):
        """Highly multimodal function."""
        A = 10
        return A * len(x) + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))


    # Test on Sphere (simplest case)
    print("=" * 60)
    print("Testing Myco on Sphere Function (D=30)")
    print("=" * 60)

    bounds = np.array([[-5.12] * 30, [5.12] * 30])
    myco = MycelialSearch(sphere, bounds, n_tips=30, n_dim=30)

    best_pos, best_val = myco.optimize(n_iterations=1000, verbose=True)

    print(f"\nFinal Result:")
    print(f"  Best Fitness: {best_val:.6e}")
    print(f"  Iterations: {len(myco.history)}")
    print(f"  Convergence: {myco.history[0]:.6e} -> {myco.history[-1]:.6e}")

    # Plot convergence
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.semilogy(myco.history, linewidth=2)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Best Fitness (log scale)', fontsize=12)
    plt.title('Myco Convergence on Sphere', fontsize=12)
    plt.grid(True, alpha=0.3)

    # Test on Rosenbrock (ridge function)
    print("\n" + "=" * 60)
    print("Testing Myco on Rosenbrock Function (D=30)")
    print("=" * 60)

    bounds_ros = np.array([[-2.048] * 30, [2.048] * 30])
    myco_ros = MycelialSearch(rosenbrock, bounds_ros, n_tips=30, n_dim=30)

    best_pos_ros, best_val_ros = myco_ros.optimize(n_iterations=1000, verbose=False)

    print(f"Final Result:")
    print(f"  Best Fitness: {best_val_ros:.6e}")
    print(f"  Convergence: {myco_ros.history[0]:.6e} -> {myco_ros.history[-1]:.6e}")

    plt.subplot(1, 2, 2)
    plt.semilogy(myco_ros.history, linewidth=2, color='orange')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Best Fitness (log scale)', fontsize=12)
    plt.title('Myco Convergence on Rosenbrock', fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('myco_convergence.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n All tests completed successfully!")
    print(" Convergence plots saved to: myco_convergence.png")
