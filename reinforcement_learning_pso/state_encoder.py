import numpy as np
import random
from itertools import product as iproduct
# Note: init_swarm and pso_step will be imported from our adapter in train.py

def swarm_diversity(positions: np.ndarray) -> float:
    """
    Compute mean distance of all particles from swarm centroid.
    
    Args:
        positions: shape (n_particles, n_cities) — SPV position matrix
    Returns:
        float — diversity scalar σ_t
    """
    centroid = positions.mean(axis=0)
    return float(np.mean(np.linalg.norm(positions - centroid, axis=1)))

def discretize_state(sigma: float, stag: int, t: int, T_max: int,
                     div_low: float, div_high: float) -> int:
    """
    Map continuous state variables to a single integer index in [0, 26].
    
    Bins:
        d_div:  0=low (σ < div_low), 1=medium, 2=high (σ > div_high)
        d_stag: 0=[0–5], 1=[6–15], 2=[>15]
        d_tau:  0=early [0,0.33), 1=mid [0.33,0.66), 2=late [0.66,1.0]
    
    Returns:
        int in [0, 26]
    """
    d_div  = 0 if sigma < div_low else (2 if sigma > div_high else 1)
    d_stag = 0 if stag <= 5 else (2 if stag > 15 else 1)
    tau    = t / T_max
    d_tau  = 0 if tau < 0.33 else (2 if tau >= 0.66 else 1)
    return d_div * 9 + d_stag * 3 + d_tau

def calibrate_thresholds(instance_matrices: list, n_particles: int = 30,
                          T_max: int = 200, n_runs: int = 20) -> tuple:
    """
    Run random PSO episodes to collect diversity samples.
    Compute 33rd and 66th percentiles as thresholds.
    
    Returns:
        (div_low, div_high) — float tuple
    """
    from .pso_spv import init_swarm, pso_step  # Local import to avoid circular dependency
    ACTIONS = list(iproduct([0.4, 0.7, 0.9], [1.0, 1.5, 2.0], [1.5, 2.0]))
    
    div_samples = []
    for _ in range(n_runs):
        D = random.choice(instance_matrices)
        pos, vel, pb, gb = init_swarm(D, n_particles)
        for t in range(T_max):
            omega, c1, c2 = random.choice(ACTIONS)
            pos, vel, pb, gb = pso_step(pos, vel, pb, gb, D, omega, c1, c2)
            div_samples.append(swarm_diversity(pos))
    
    div_low  = float(np.percentile(div_samples, 33))
    div_high = float(np.percentile(div_samples, 66))
    return div_low, div_high
