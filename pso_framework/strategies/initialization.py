import numpy as np
from typing import Callable

class InitializationStrategy:
    def initialize(self, swarm, lb: np.ndarray, ub: np.ndarray, max_velocity: np.ndarray, fitness_func: Callable = None):
        raise NotImplementedError

class RandomUniformInit(InitializationStrategy):
    def initialize(self, swarm, lb: np.ndarray, ub: np.ndarray, max_velocity: np.ndarray, fitness_func: Callable = None):
        """
        Standard random uniform initialization within bounds.
        """
        swarm.X = np.random.uniform(lb, ub, size=(swarm.N, swarm.D))
        # Initialize velocity between -max_velocity and +max_velocity
        swarm.V = np.random.uniform(-max_velocity, max_velocity, size=(swarm.N, swarm.D))

class LatinHypercubeInit(InitializationStrategy):
    def initialize(self, swarm, lb: np.ndarray, ub: np.ndarray, max_velocity: np.ndarray, fitness_func: Callable = None):
        """
        Latin Hypercube Sampling for better coverage of the search space.
        """
        try:
            from scipy.stats import qmc
        except ImportError:
            raise ImportError("scipy is required for LatinHypercubeInit")
            
        sampler = qmc.LatinHypercube(d=swarm.D)
        sample = sampler.random(n=swarm.N)  # N x D in [0, 1]
        
        # Scale to bounds
        swarm.X = qmc.scale(sample, lb, ub)
        swarm.V = np.random.uniform(-max_velocity, max_velocity, size=(swarm.N, swarm.D))

class OppositionBasedInit(InitializationStrategy):
    def initialize(self, swarm, lb: np.ndarray, ub: np.ndarray, max_velocity: np.ndarray, fitness_func: Callable = None):
        """
        Opposition-Based Learning (OBL) Initialization.
        Generates normal points and their opposites, evaluates all 2*N points,
        and keeps the best N points. Requires fitness_func.
        """
        if fitness_func is None:
            raise ValueError("OppositionBasedInit requires a fitness_func to evaluate candidates.")
            
        # 1. Generate N normal points
        X_normal = np.random.uniform(lb, ub, size=(swarm.N, swarm.D))
        
        # 2. Generate N opposite points
        X_opposite = lb + ub - X_normal
        
        # 3. Evaluate all
        F_normal = np.array([fitness_func(x) for x in X_normal])
        F_opposite = np.array([fitness_func(x) for x in X_opposite])
        
        # 4. Select the best N out of 2N
        X_all = np.vstack((X_normal, X_opposite))
        F_all = np.concatenate((F_normal, F_opposite))
        
        best_indices = np.argsort(F_all)[:swarm.N]
        
        swarm.X = X_all[best_indices].copy()
        swarm.V = np.random.uniform(-max_velocity, max_velocity, size=(swarm.N, swarm.D))
