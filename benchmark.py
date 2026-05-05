import sys
import os
import time
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from reinforcement_learning_pso.validate import run_rl_pso, run_vanilla_pso, load_instance
from reinforcement_learning_pso.rl_agent import QLearningAgent

INSTANCES = {
    'eil51': 426,
    'berlin52': 7542,
    'eil76': 538,
    'kroA100': 21282,
    'ch130': 6110,
    'ch150': 6528
}

N_RUNS = 10  # number of independent runs to compute mean/std
N_PARTICLES = 30
T_MAX = 200

# Load RL Agent & Thresholds
agent = QLearningAgent(epsilon=0.0)
agent.load('saved/q_table.npy')
div_low, div_high = np.load('saved/div_thresholds.npy')

print(f"{'Instance':<10} | {'Algorithm':<10} | {'BKS':<6} | {'Best':<8} | {'Mean':<8} | {'Std':<6} | {'ER_best%':<8} | {'ER_mean%':<8} | {'CPU(s)':<6}")
print("-" * 105)

for name, bks in INSTANCES.items():
    path = f'data/{name}.tsp'
    D = load_instance(path)
    
    # --- VANILLA PSO ---
    v_results = []
    v_start_time = time.time()
    for seed in range(N_RUNS):
        res = run_vanilla_pso(D, omega=0.7, c1=1.5, c2=1.5, n_particles=N_PARTICLES, T_max=T_MAX, seed=seed)
        v_results.append(res)
    v_cpu = (time.time() - v_start_time) / N_RUNS
    
    v_best = np.min(v_results)
    v_mean = np.mean(v_results)
    v_std = np.std(v_results)
    v_er_best = ((v_best - bks) / bks) * 100
    v_er_mean = ((v_mean - bks) / bks) * 100
    
    print(f"{name:<10} | {'Vanilla':<10} | {bks:<6} | {v_best:<8.1f} | {v_mean:<8.1f} | {v_std:<6.1f} | {v_er_best:<8.2f} | {v_er_mean:<8.2f} | {v_cpu:<6.2f}")

    # --- RL PSO ---
    rl_results = []
    rl_start_time = time.time()
    for seed in range(N_RUNS):
        res = run_rl_pso(D, agent, div_low, div_high, n_particles=N_PARTICLES, T_max=T_MAX, seed=seed)
        rl_results.append(res)
    rl_cpu = (time.time() - rl_start_time) / N_RUNS
    
    rl_best = np.min(rl_results)
    rl_mean = np.mean(rl_results)
    rl_std = np.std(rl_results)
    rl_er_best = ((rl_best - bks) / bks) * 100
    rl_er_mean = ((rl_mean - bks) / bks) * 100

    print(f"{name:<10} | {'RL-PSO':<10} | {bks:<6} | {rl_best:<8.1f} | {rl_mean:<8.1f} | {rl_std:<6.1f} | {rl_er_best:<8.2f} | {rl_er_mean:<8.2f} | {rl_cpu:<6.2f}")
    print("-" * 105)
