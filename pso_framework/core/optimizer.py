import numpy as np
import time
from typing import Callable, Dict, Any

from .swarm import Swarm
from pso_framework.strategies.initialization import InitializationStrategy, RandomUniformInit
from pso_framework.strategies.inertia import InertiaStrategy, LinearDecreasingInertia
from pso_framework.strategies.topology import TopologyStrategy, GlobalBestTopology
from pso_framework.strategies.boundary import BoundaryStrategy, ReflectingBoundary
from pso_framework.strategies.updater import UpdaterStrategy, StandardUpdater

class PSOExecutor:
    """
    The orchestrator for the PSO algorithm.
    Integrates all configured modular strategies.
    """
    def __init__(self,
                 fitness_func: Callable[[np.ndarray], float],
                 N: int,
                 D: int,
                 lb: np.ndarray,
                 ub: np.ndarray,
                 T_max: int,
                 c1: float = 2.0,
                 c2: float = 2.0,
                 max_velocity: np.ndarray = None,
                 init_strategy: InitializationStrategy = None,
                 inertia_strategy: InertiaStrategy = None,
                 topology_strategy: TopologyStrategy = None,
                 boundary_strategy: BoundaryStrategy = None,
                 updater_strategy: UpdaterStrategy = None,
                 seed: int = None):
        
        self.fitness_func = fitness_func
        self.N = N
        self.D = D
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.T_max = T_max
        self.c1 = c1
        self.c2 = c2
        
        if max_velocity is None:
            self.max_velocity = (self.ub - self.lb) / 2.0
        else:
            self.max_velocity = np.array(max_velocity)
            
        # Assign strategies with defaults
        self.init_strategy = init_strategy or RandomUniformInit()
        self.inertia_strategy = inertia_strategy or LinearDecreasingInertia()
        self.topology_strategy = topology_strategy or GlobalBestTopology()
        self.boundary_strategy = boundary_strategy or ReflectingBoundary()
        self.updater_strategy = updater_strategy or StandardUpdater()
        
        if seed is not None:
            np.random.seed(seed)
            
        self.swarm = Swarm(self.N, self.D)
        self.convergence_history = []
        
    def _evaluate_all(self):
        """evaluate fitness(obj function) for all particles."""
        F = np.array([self.fitness_func(self.swarm.X[i]) for i in range(self.N)])
        return F
        
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        executes the optimization loop.
        """
        start_time = time.time()
        
        # initialization
        self.init_strategy.initialize(self.swarm, self.lb, self.ub, self.max_velocity, self.fitness_func)
        
        # initial evaluation
        F_initial = self._evaluate_all()
        self.swarm.update_bests(F_initial)
        
        self.convergence_history.append(self.swarm.F_G)
        
        # Main Loop
        for t in range(1, self.T_max + 1):
            # QPSO needs beta. Linearly decrease from 1.0 to 0.5.
            beta = 1.0 - 0.5 * (t / self.T_max)
            kwargs['beta'] = beta
            
            # Get Context
            omega = self.inertia_strategy.get_inertia(self.swarm, t, self.T_max)
            G_local = self.topology_strategy.get_attractors(self.swarm)
            
            # update velocity and position
            self.updater_strategy.update(self.swarm, omega, G_local, 
                                         self.c1, self.c2, self.max_velocity, **kwargs)
                                         
            # Apply Boundaries
            self.boundary_strategy.apply(self.swarm, self.lb, self.ub)
            
            # Evaluate new positions
            F_new = self._evaluate_all()
            
            # Application Layer Injection (e.g. 2-opt)
            # If the user provides a custom callback to modify swarm based on external logic
            if 'callback' in kwargs:
                F_new = kwargs['callback'](self.swarm, F_new, t)
                
            # Update Bests
            self.swarm.update_bests(F_new)
            
            self.convergence_history.append(self.swarm.F_G)
            
        execution_time = time.time() - start_time
        return {
            'best_position': self.swarm.G.copy(),
            'best_fitness': self.swarm.F_G,
            'convergence': self.convergence_history,
            'execution_time': execution_time
        }
