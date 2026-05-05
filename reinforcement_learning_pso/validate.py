import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import random
from .state_encoder import swarm_diversity, discretize_state
from .rl_agent import QLearningAgent, ACTIONS
from .pso_spv import init_swarm, pso_step, decode_spv, tour_length
from pso_framework.applications.tsp import parse_tsplib

def load_instance(path: str) -> np.ndarray:
    """Load a TSPLIB .tsp file and return integer distance matrix."""
    parsed = parse_tsplib(path)
    return parsed['dist_matrix']

def run_rl_pso(D, agent, div_low, div_high,
               n_particles=30, T_max=200, seed=None):
    if seed is not None:
        np.random.seed(seed)
    pos, vel, pb, gb = init_swarm(D, n_particles)
    stag = 0
    f_prev = tour_length(decode_spv(gb), D)
    for t in range(T_max):
        sigma = swarm_diversity(pos)
        s     = discretize_state(sigma, stag, t, T_max, div_low, div_high)
        a     = int(np.argmax(agent.Q[s]))   # greedy: eps=0
        omega, c1, c2 = ACTIONS[a]
        pos, vel, pb, gb = pso_step(pos, vel, pb, gb, D, omega, c1, c2)
        f_cur = tour_length(decode_spv(gb), D)
        stag  = 0 if f_cur < f_prev else stag + 1
        f_prev = f_cur
    return f_cur

def run_vanilla_pso(D, omega=0.7, c1=1.5, c2=1.5,
                    n_particles=30, T_max=200, seed=None):
    if seed is not None:
        np.random.seed(seed)
    pos, vel, pb, gb = init_swarm(D, n_particles)
    for t in range(T_max):
        pos, vel, pb, gb = pso_step(pos, vel, pb, gb, D, omega, c1, c2)
    return tour_length(decode_spv(gb), D)

# ── Load artifacts ──────────────────────────────────────────────
agent = QLearningAgent(epsilon=0.0)
agent.load('saved/q_table.npy')
div_low, div_high = np.load('saved/div_thresholds.npy')

# ── Run comparison ──────────────────────────────────────────────
VALIDATION_PATHS = ['data/tsp225.tsp', 'data/a280.tsp']
val_matrices   = [load_instance(p) for p in VALIDATION_PATHS]

for path, D in zip(['tsp225', 'a280'], val_matrices):
    rl_results      = []
    vanilla_results = []

    for seed in range(15):
        rl_results.append(run_rl_pso(D, agent, div_low, div_high, seed=seed))
        vanilla_results.append(run_vanilla_pso(D, seed=seed))

    rl_mean  = np.mean(rl_results)
    v_mean   = np.mean(vanilla_results)
    improve  = (1 - rl_mean / v_mean) * 100

    print(f"\n{path}")
    print(f"  RL-PSO  : mean={rl_mean:.1f}  std={np.std(rl_results):.1f}")
    print(f"  Vanilla : mean={v_mean:.1f}  std={np.std(vanilla_results):.1f}")
    print(f"  Improvement: {improve:.1f}%")
