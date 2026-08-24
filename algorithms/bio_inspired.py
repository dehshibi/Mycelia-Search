"""
bio_inspired.py
===============
Custom bio-inspired algorithms for the CEC 2022 benchmark.
Compatible with Mealpy 3.0.3 Agent-based API.

Algorithms:
- MGO (Moss Growth Optimisation) - 2024
- PIS (Physarum-Inspired Search) - Custom
"""

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalMGO(Optimizer):
    """
    Moss Growth Optimisation (MGO).

    Reference: Zheng et al., "Moss Growth Optimisation",
    J. Computational Design and Engineering, 2024.
    
    Mechanisms:
    - Moss quality assessment: fitness-based survival probability
    - Heterogeneous growth rates: different growth speeds per moss
    - Environmental resource depletion: energy model
    - Cooperative propagation: moss interaction
    """

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False
        self.history_pop = []
        self.moss_quality = None  # Fitness-based quality scores
        self.growth_rates = None   # Heterogeneous growth rates
        self.resources = None      # Environmental resources per agent

    def initialization(self):
        super().initialization()
        # Initialise history population for selection from historical moss
        self.history_pop = [self.pop.copy() for _ in range(10)]
        
        # Initialise moss quality assessment (normalise fitness)
        fitness_vals = np.array([agent.target.fitness for agent in self.pop])
        self.moss_quality = self._assess_moss_quality(fitness_vals)
        
        # Initialise heterogeneous growth rates (inversely proportional to fitness)
        self.growth_rates = 0.5 + 0.5 * self.moss_quality  # Range [0.5, 1.0]
        
        # Initialise environmental resources (total energy budget)
        self.resources = np.ones(self.pop_size)

    def _assess_moss_quality(self, fitness_vals):
        """Assess moss quality based on fitness (better = higher quality)."""
        # Normalise to [0, 1] where 1 = best, 0 = worst
        min_fit = np.min(fitness_vals)
        max_fit = np.max(fitness_vals)
        if max_fit - min_fit < 1e-10:
            return np.ones(len(fitness_vals)) * 0.5
        quality = 1.0 - (fitness_vals - min_fit) / (max_fit - min_fit)
        return quality

    def evolve(self, epoch):
        """Main evolution step for MGO with all mechanisms.
        
        Implements:
        1. Moss quality assessment (survival probability)
        2. Heterogeneous growth rates (different update speeds)
        3. Environmental resource depletion (energy model)
        4. Cooperative propagation (neighbour influence)
        """
        if len(self.history_pop) == 0:
            self.history_pop = [self.pop.copy() for _ in range(10)]

        pop_new = []
        
        # Update resource availability (depletion over time)
        resource_depletion = 1.0 - (epoch / self.epoch) * 0.5  # Resources deplete to 50%
        current_resources = self.resources * resource_depletion

        for idx in range(self.pop_size):
            # Current moss properties
            pos_current = self.pop[idx].solution
            pos_best = self.g_best.solution
            dim = self.problem.n_dims
            
            # Moss quality affects exploration-exploitation balance
            quality = self.moss_quality[idx]
            growth_rate = self.growth_rates[idx]
            resource = current_resources[idx]
            
            # Quality-based survival probability (fitness landscape assessment)
            survival_prob = 0.3 + 0.7 * quality  # Range [0.3, 1.0]
            
            # Exploration vs Exploitation
            exploration_prob = 1.0 - (epoch / self.epoch)
            
            # Cooperative propagation: find neighbor moss
            neighbor_idx = np.random.randint(0, self.pop_size)
            pos_neighbor = self.pop[neighbor_idx].solution
            
            if np.random.rand() < exploration_prob:
                # SPORE DISPERSAL (Exploration)
                # Distance proportional to quality and growth rate
                dispersion_strength = (1.0 - quality) * growth_rate
                dispersion = np.random.uniform(-1, 1, dim) * (pos_best - pos_current) * dispersion_strength
                pos_new = pos_current + dispersion
                
            else:
                # PROPAGATION (Exploitation)
                if np.random.rand() < 0.5:
                    # Rhizoid propagation (toward best solution)
                    step = np.random.uniform(0, growth_rate, dim)
                    pos_new = pos_current + step * (pos_best - pos_current)
                else:
                    # Cooperative propagation (toward neighbour moss)
                    cooperation_strength = 0.5 * quality  # Better moss cooperate more
                    pos_new = pos_current + cooperation_strength * (pos_neighbor - pos_current)
            
            # RESOURCE-BASED UPDATE (energy model)
            # Utilise available resources for growth
            if resource > 0.5:
                # High resources: aggressive growth
                adaptive_scale = 1.5 * resource
            elif resource > 0.2:
                # Medium resources: normal growth
                adaptive_scale = resource
            else:
                # Low resources: conservative growth
                adaptive_scale = 0.3 * resource
            
            pos_new = pos_current + adaptive_scale * (pos_new - pos_current)
            
            # HISTORICAL POPULATION TRACKING (select from best historical moss)
            if np.random.rand() < 0.2 and len(self.history_pop) > 0:
                # 20% chance to incorporate historical solution
                hist_pop = self.history_pop[np.random.randint(0, len(self.history_pop))]
                if len(hist_pop) > 0:
                    hist_agent = hist_pop[np.random.randint(0, len(hist_pop))]
                    blend_factor = np.random.rand()
                    pos_new = blend_factor * pos_new + (1 - blend_factor) * hist_agent.solution
            
            # Boundary correction
            pos_new = self.correct_solution(pos_new)
            agent_new = self.generate_agent(pos_new)
            
            # QUALITY ASSESSMENT: Accept if quality is maintained or improved
            if self.compare_target(agent_new.target, self.pop[idx].target, self.problem.minmax):
                pop_new.append(agent_new)
                # Update quality assessment
                new_quality = self._assess_moss_quality(np.array([agent_new.target.fitness]))
                self.moss_quality[idx] = new_quality[0]
                # Growth rate increases with success
                self.growth_rates[idx] = min(1.0, self.growth_rates[idx] * 1.1)
                # Resources restore on success
                self.resources[idx] = min(1.0, self.resources[idx] + 0.1)
            else:
                pop_new.append(self.pop[idx])
                # Growth rate decreases with failure
                self.growth_rates[idx] = max(0.5, self.growth_rates[idx] * 0.9)
                # Resources deplete on failure
                self.resources[idx] = max(0.1, self.resources[idx] - 0.05)

        # Update population
        self.pop = self.update_target_for_population(pop_new)
        
        # Update history population (keep best 10 generations)
        self.history_pop.append([agent.copy() for agent in self.pop])
        if len(self.history_pop) > 10:
            self.history_pop.pop(0)
        
        # Reassess moss quality for next iteration
        fitness_vals = np.array([agent.target.fitness for agent in self.pop])
        self.moss_quality = self._assess_moss_quality(fitness_vals)


class PhysarumOptimizer(Optimizer):
    """
    Physarum-Inspired Search (PIS).

    Simulates adaptive tube-resizing (conductivity) of Physarum polycephalum based on fitness feedback. Inspired by slime mold network optimisation.
    
    Key Features:
    - Tube diameter dynamics: tubes connecting good solutions grow
    - Fitness-based conductivity: better paths have higher conductivity
    - Adaptive flow: solution quality guides the flow of changes
    - Network contraction: poor tubes shrink over time
    """

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.tube_diameters = None      # Conductivity between agents
        self.distance_matrix = None     # Distance between solutions
        self.fitness_feedback = None    # Fitness-based tube adaptation

    def initialization(self):
        super().initialization()
        # Initialise tube diameters (conductivity) with uniform values
        self.tube_diameters = np.ones((self.pop_size, self.pop_size))
        np.fill_diagonal(self.tube_diameters, 0)
        
        # Initialise distance matrix
        self.distance_matrix = np.zeros((self.pop_size, self.pop_size))
        for i in range(self.pop_size):
            for j in range(i+1, self.pop_size):
                dist = np.linalg.norm(self.pop[i].solution - self.pop[j].solution)
                self.distance_matrix[i, j] = dist
                self.distance_matrix[j, i] = dist
        
        # Initialise fitness feedback (normalised fitness values)
        fitness_vals = np.array([agent.target.fitness for agent in self.pop])
        self.fitness_feedback = self._compute_fitness_feedback(fitness_vals)

    def _compute_fitness_feedback(self, fitness_vals):
        """Compute normalised fitness feedback [0, 1] for tube adaptation."""
        min_fit = np.min(fitness_vals)
        max_fit = np.max(fitness_vals)
        if max_fit - min_fit < 1e-10:
            return np.ones(len(fitness_vals)) * 0.5
        # Better fitness = higher feedback (for minimisation, invert)
        feedback = 1.0 - (fitness_vals - min_fit) / (max_fit - min_fit)
        return np.clip(feedback, 0.1, 1.0)

    def _update_tube_dynamics(self):
        """Update tube diameters based on fitness-weighted flow."""
        # Tube shrinkage: all tubes decay slightly
        decay_rate = 0.95
        self.tube_diameters *= decay_rate
        
        # Tube growth: tubes connecting good solutions grow
        for i in range(self.pop_size):
            for j in range(i+1, self.pop_size):
                # Growth proportional to combined fitness of both ends
                combined_fitness = (self.fitness_feedback[i] + self.fitness_feedback[j]) / 2
                growth = 0.1 * combined_fitness  # Max growth per iteration
                
                self.tube_diameters[i, j] += growth
                self.tube_diameters[j, i] = self.tube_diameters[i, j]
        
        # Ensure tubes stay in a reasonable range [0, 2]
        self.tube_diameters = np.clip(self.tube_diameters, 0, 2)
        np.fill_diagonal(self.tube_diameters, 0)

    def evolve(self, epoch):
        """Simulates Physarum tube contraction/expansion dynamics."""
        pop_new = []
        
        # Update tube dynamics based on current fitness landscape
        self._update_tube_dynamics()

        for idx in range(self.pop_size):
            pos_best = self.g_best.solution
            pos_current = self.pop[idx].solution

            # Select two indices based on tube conductivity (weighted)
            tube_weights = self.tube_diameters[idx, :].copy()
            tube_weights[idx] = 0  # Can't select self
            
            if np.sum(tube_weights) > 0:
                # Weighted selection: higher conductivity = higher probability
                probs = tube_weights / np.sum(tube_weights)
                idx_list = np.random.choice(self.pop_size, 2, replace=False, p=probs)
            else:
                # Fallback to random selection if no tubes
                idx_list = np.random.choice(self.pop_size, 2, replace=False)
            
            pos_r1 = self.pop[idx_list[0]].solution
            pos_r2 = self.pop[idx_list[1]].solution

            # Tube-based movement: flow proportional to diameter
            tube_conductivity = self.tube_diameters[idx, idx_list[0]]
            
            # Adaptive weight based on tube conductivity and fitness
            weight = tube_conductivity * np.random.exponential(0.5)
            
            # Fitness-based noise (better solutions add less noise)
            noise_magnitude = (1.0 - self.fitness_feedback[idx]) * 0.5
            noise = (np.random.rand(pos_current.shape[0]) - 0.5) * noise_magnitude
            
            # Direction influenced by tube dynamics
            direction = weight * (pos_r1 - pos_r2)
            
            # Movement combines best direction with tube-guided flow
            pos_new = pos_best + direction + noise
            
            # Boundary handling
            pos_new = self.correct_solution(pos_new)
            agent_new = self.generate_agent(pos_new)
            
            # Selection: keep better solution
            if self.compare_target(agent_new.target, self.pop[idx].target, self.problem.minmax):
                pop_new.append(agent_new)
            else:
                pop_new.append(self.pop[idx])

        # Update population
        self.pop = self.update_target_for_population(pop_new)
        
        # Update distance matrix
        for i in range(self.pop_size):
            for j in range(i+1, self.pop_size):
                dist = np.linalg.norm(self.pop[i].solution - self.pop[j].solution)
                self.distance_matrix[i, j] = dist
                self.distance_matrix[j, i] = dist
        
        # Update fitness feedback for next iteration
        fitness_vals = np.array([agent.target.fitness for agent in self.pop])
        self.fitness_feedback = self._compute_fitness_feedback(fitness_vals)
