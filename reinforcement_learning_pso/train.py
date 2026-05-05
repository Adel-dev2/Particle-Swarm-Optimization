import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import numpy as np
import random
from .state_encoder import swarm_diversity, discretize_state, calibrate_thresholds
from .rl_agent import QLearningAgent, ACTIONS
from .pso_spv import init_swarm, pso_step, decode_spv, tour_length
from pso_framework.applications.tsp import parse_tsplib

def load_instance(path: str) -> np.ndarray:
    """Load a TSPLIB .tsp file and return integer distance matrix."""
    parsed = parse_tsplib(path)
    return parsed['dist_matrix']

# Ensure directories exist
os.makedirs('saved', exist_ok=True)

# ── Load instances ──────────────────────────────────────────────
TRAINING_PATHS = [
    'data/eil51.tsp', 'data/berlin52.tsp', 'data/eil76.tsp',
    'data/kroA100.tsp', 'data/kroB100.tsp',
    'data/ch130.tsp', 'data/ch150.tsp', 'data/kroA200.tsp',
]

train_matrices = [load_instance(p) for p in TRAINING_PATHS]

# Weight smaller instances 3× more in sampling
sample_pool = train_matrices[:5] * 3 + train_matrices[5:]

# ── Phase 0: Calibration ────────────────────────────────────────
print("Running calibration...")
div_low, div_high = calibrate_thresholds(
    [train_matrices[0], train_matrices[3]],   # eil51, kroA100
    n_particles=30, T_max=200, n_runs=20
)
np.save('saved/div_thresholds.npy', [div_low, div_high])
print(f"  div_low={div_low:.3f}  div_high={div_high:.3f}")

# ── Phase 1: Training ───────────────────────────────────────────
N_PARTICLES   = 30
T_MAX         = 200
N_EPISODES    = 400

agent = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.5)

for episode in range(N_EPISODES):
    D   = random.choice(sample_pool)
    pos, vel, pb, gb = init_swarm(D, N_PARTICLES)
    f0  = f_prev = tour_length(decode_spv(gb), D)
    stag = 0

    for t in range(T_MAX):
        # 1. Observe state
        sigma = swarm_diversity(pos)
        s = discretize_state(sigma, stag, t, T_MAX, div_low, div_high)

        # 2. Choose action (ε-greedy)
        a = agent.choose_action(s)
        omega, c1, c2 = ACTIONS[a]

        # 3. Apply action — run one PSO step
        pos, vel, pb, gb = pso_step(pos, vel, pb, gb, D, omega, c1, c2)
        f_cur = tour_length(decode_spv(gb), D)

        # 4. Update stagnation counter
        stag = 0 if f_cur < f_prev else stag + 1

        # 5. Compute normalized reward
        reward = (f_prev - f_cur) / f0

        # 6. Observe next state
        sigma_next = swarm_diversity(pos)
        s_next = discretize_state(sigma_next, stag, t + 1, T_MAX,
                                   div_low, div_high)

        # 7. Q-table update (Bellman)
        agent.update(s, a, reward, s_next)

        f_prev = f_cur

    # Decay exploration after each episode
    agent.decay_epsilon(decay=0.001, min_epsilon=0.05)

    if episode % 50 == 0:
        print(f"Episode {episode:4d} | eps={agent.epsilon:.3f} | final_tour={f_cur:.1f}")

# ── Save ────────────────────────────────────────────────────────
agent.save('saved/q_table.npy')
print("Training complete.")
