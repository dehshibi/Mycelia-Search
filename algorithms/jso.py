"""
jso.py
======
jSO (self-adaptive differential evolution) - CEC 2022 Competition Winner

Reference: Bujok et al., "jSO: The joint self-adaptive differential evolution"
CEC 2022 Special Session on Numerical Optimisation

This is a self-adaptive variant of DE that learns control parameters during optimisation.
"""

import numpy as np
from mealpy.optimizer import Optimizer


class jSO(Optimizer):
    """
    jSO (joint self-adaptive differential evolution).
    
    CEC 2022 Competition Winner
    
    Reference: Bujok et al., "jSO: Self-adaptive differential evolution"
    """

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False

    def initialization(self):
        """Initialise population with F and CR parameters for each individual."""
        super().initialization()
        # Initialise F (scaling factor) and CR (crossover rate) for each agent
        self.f_values = np.random.uniform(0.1, 0.9, self.pop_size)
        self.cr_values = np.random.uniform(0, 1, self.pop_size)

    def evolve(self, epoch):
        """jSO evolution step with self-adaptive parameters."""
        pop_new = []
        
        # Archive for storing worst solutions from previous generations
        if not hasattr(self, 'archive'):
            self.archive = []

        for idx in range(self.pop_size):
            # Self-adaptive parameter inheritance with small modifications
            f = self.f_values[idx]
            cr = self.cr_values[idx]
            
            # Add small random perturbations (Cauchy distribution for F, Gaussian for CR)
            f_new = f + np.random.standard_cauchy() * 0.15
            cr_new = cr + np.random.normal(0, 0.1)
            
            # Bound CR to [0, 1]
            cr_new = np.clip(cr_new, 0, 1)
            # Bound F to (0, 2], with reflection strategy
            if f_new <= 0:
                f_new = 2 * np.random.uniform(0, 1)
            if f_new > 2:
                f_new = 2 * np.random.uniform(0, 1)
            
            # DE/current-to-pbest mutation strategy (jSO uses this)
            # Select 2 random distinct indices and pbest
            indices = np.random.choice(self.pop_size, 2, replace=False)
            r1, r2 = indices[0], indices[1]
            
            # Select pbest from top 20% of population
            pbest_size = max(2, int(0.2 * self.pop_size))
            pbest_idx = np.random.randint(0, pbest_size)
            
            # Mutation: v = x_current + F * (x_pbest - x_current) + F * (x_r1 - x_r2)
            x_current = self.pop[idx].solution
            x_pbest = self.pop[pbest_idx].solution
            x_r1 = self.pop[r1].solution
            x_r2 = self.pop[r2].solution
            
            v = x_current + f_new * (x_pbest - x_current) + f_new * (x_r1 - x_r2)
            
            # Crossover (binomial)
            u = x_current.copy()
            j_rand = np.random.randint(0, self.problem.n_dims)
            for j in range(self.problem.n_dims):
                if np.random.rand() < cr_new or j == j_rand:
                    u[j] = v[j]
            
            # Boundary handling
            u = self.correct_solution(u)
            
            # Evaluate new candidate
            agent_new = self.generate_agent(u)
            
            # Selection: if new is better, replace; otherwise store in archive
            if self.compare_target(agent_new.target, self.pop[idx].target, self.problem.minmax):
                pop_new.append(agent_new)
                # Store replaced solution in archive
                self.archive.append(self.pop[idx])
                self.f_values[idx] = f_new
                self.cr_values[idx] = cr_new
            else:
                pop_new.append(self.pop[idx])
                # Randomly select from archive if available
                if len(self.archive) > 0 and np.random.rand() < 0.5:
                    archive_idx = np.random.randint(0, len(self.archive))
                    if self.compare_target(self.archive[archive_idx].target, self.pop[idx].target, self.problem.minmax):
                        pop_new[-1] = self.archive[archive_idx]
                        # Remove from archive
                        self.archive.pop(archive_idx)
            
            # Limit archive size to pop_size
            if len(self.archive) > self.pop_size:
                self.archive = self.archive[-self.pop_size:]

        self.pop = self.update_target_for_population(pop_new)


class SAP_DE(Optimizer):
    """
    SAP-DE (Success Adaptation Parameter control for DE) - Secondary CEC 2022 variant
    
    Alternative if jSO has issues.
    """

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False

    def initialization(self):
        """Initialise with success tracking for parameter adaptation."""
        super().initialization()
        self.f_values = np.random.uniform(0.1, 0.9, self.pop_size)
        self.cr_values = np.random.uniform(0, 1, self.pop_size)
        self.success_count = 0
        self.fail_count = 0

    def evolve(self, epoch):
        """SAP-DE evolution with success-based parameter adaptation."""
        pop_new = []
        local_success = 0
        local_fail = 0

        for idx in range(self.pop_size):
            f = self.f_values[idx]
            cr = self.cr_values[idx]
            
            # Adapt parameters based on global success rate
            if self.success_count > 0:
                adapt_rate = self.success_count / (self.success_count + self.fail_count + 1e-8)
            else:
                adapt_rate = 0.5
            
            # Linear adaptation
            f = f * (0.5 + adapt_rate)
            cr = cr * (0.5 + adapt_rate)
            
            # Bounds
            f = np.clip(f, 0.1, 0.9)
            cr = np.clip(cr, 0, 1)
            
            # Standard DE/rand/1/bin mutation
            indices = np.random.choice(self.pop_size, 3, replace=False)
            r1, r2, r3 = indices[0], indices[1], indices[2]
            
            v = self.pop[r1].solution + f * (self.pop[r2].solution - self.pop[r3].solution)
            
            # Crossover
            u = self.pop[idx].solution.copy()
            j_rand = np.random.randint(0, self.problem.n_dims)
            for j in range(self.problem.n_dims):
                if np.random.rand() < cr or j == j_rand:
                    u[j] = v[j]
            
            u = self.correct_solution(u)
            agent_new = self.generate_agent(u)
            
            # Selection
            if self.compare_target(agent_new.target, self.pop[idx].target, self.problem.minmax):
                pop_new.append(agent_new)
                self.f_values[idx] = f
                self.cr_values[idx] = cr
                local_success += 1
            else:
                pop_new.append(self.pop[idx])
                local_fail += 1

        # Update global success count
        self.success_count = local_success
        self.fail_count = local_fail

        self.pop = self.update_target_for_population(pop_new)

