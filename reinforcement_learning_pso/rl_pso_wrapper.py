import numpy as np
from .rl_agent import QLearningAgent, ACTIONS
from .state_encoder import swarm_diversity, discretize_state
from pso_framework.core.optimizer import PSOExecutor

class RLPSO_SPV(PSOExecutor):
    """
    Drop-in replacement for PSOExecutor that uses the RL agent to adapt parameters.
    """

    def __init__(self,
                 fitness_func, N, D, lb, ub, T_max,
                 q_table_path: str = 'saved/q_table.npy',
                 thresholds_path: str = 'saved/div_thresholds.npy',
                 **kwargs):
        super().__init__(fitness_func, N, D, lb, ub, T_max, **kwargs)
        self.agent = QLearningAgent(epsilon=0.0)
        self.agent.load(q_table_path)
        self.div_low, self.div_high = np.load(thresholds_path)
        self._stag = 0
        self._f_prev = None

    def run(self, **kwargs):
        import time
        start_time = time.time()
        
        self.init_strategy.initialize(self.swarm, self.lb, self.ub, self.max_velocity, self.fitness_func)
        
        F_initial = self._evaluate_all()
        self.swarm.update_bests(F_initial)
        
        self.convergence_history.append(self.swarm.F_G)
        self._f_prev = self.swarm.F_G
        self._stag = 0
        
        for t in range(1, self.T_max + 1):
            sigma = swarm_diversity(self.swarm.X)
            s = discretize_state(sigma, self._stag, t, self.T_max,
                                  self.div_low, self.div_high)
            omega_scalar, c1, c2 = self.agent.best_action(s)
            
            omega_arr = np.full((self.N, self.D), omega_scalar)
            G_local = self.topology_strategy.get_attractors(self.swarm)
            
            self.updater_strategy.update(self.swarm, omega_arr, G_local, 
                                         c1, c2, self.max_velocity, **kwargs)
                                         
            self.boundary_strategy.apply(self.swarm, self.lb, self.ub)
            
            F_new = self._evaluate_all()
            
            if 'callback' in kwargs:
                F_new = kwargs['callback'](self.swarm, F_new, t)
                
            self.swarm.update_bests(F_new)
            
            f_cur = self.swarm.F_G
            if f_cur < self._f_prev:
                self._stag = 0
            else:
                self._stag += 1
            self._f_prev = f_cur
            
            self.convergence_history.append(self.swarm.F_G)
            
        execution_time = time.time() - start_time
        return {
            'best_position': self.swarm.G.copy(),
            'best_fitness': self.swarm.F_G,
            'convergence': self.convergence_history,
            'execution_time': execution_time
        }
