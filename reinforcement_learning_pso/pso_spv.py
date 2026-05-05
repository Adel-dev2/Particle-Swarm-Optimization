import numpy as np
from pso_framework.core.swarm import Swarm
from pso_framework.strategies.initialization import RandomUniformInit
from pso_framework.strategies.updater import StandardUpdater
from pso_framework.strategies.boundary import ReflectingBoundary
from pso_framework.applications.tsp import spv_decode as decode_spv, tour_length, create_tsp_fitness

def init_swarm(D: np.ndarray, n_particles: int = 30):
    n_cities = D.shape[0]
    swarm = Swarm(n_particles, n_cities)
    init_strategy = RandomUniformInit()
    fitness_func = create_tsp_fitness(D)
    
    lb = np.zeros(n_cities)
    ub = np.full(n_cities, n_cities - 1.0)
    max_velocity = (ub - lb) / 2.0
    
    init_strategy.initialize(swarm, lb, ub, max_velocity, fitness_func)
    
    F_initial = np.array([fitness_func(swarm.X[i]) for i in range(n_particles)])
    swarm.update_bests(F_initial)
    
    return swarm.X.copy(), swarm.V.copy(), swarm.P.copy(), swarm.G.copy()

def pso_step(pos, vel, pb, gb, D, omega, c1, c2):
    n_particles, n_cities = pos.shape
    swarm = Swarm(n_particles, n_cities)
    swarm.X = pos.copy()
    swarm.V = vel.copy()
    swarm.P = pb.copy()
    swarm.G = gb.copy()
    
    fitness_func = create_tsp_fitness(D)
    
    # We must explicitly set current fitnesses so update_bests works properly
    swarm.F_P = np.array([fitness_func(swarm.P[i]) for i in range(n_particles)])
    swarm.F_G = tour_length(decode_spv(swarm.G), D)
    
    updater = StandardUpdater()
    G_local = np.tile(swarm.G, (n_particles, 1))
    
    omega_arr = np.full((n_particles, n_cities), omega)
    
    lb = np.zeros(n_cities)
    ub = np.full(n_cities, n_cities - 1.0)
    max_velocity = (ub - lb) / 2.0
    
    updater.update(swarm, omega_arr, G_local, c1, c2, max_velocity)
    
    boundary = ReflectingBoundary()
    boundary.apply(swarm, lb, ub)
    
    F_new = np.array([fitness_func(swarm.X[i]) for i in range(n_particles)])
    swarm.update_bests(F_new)
    
    return swarm.X.copy(), swarm.V.copy(), swarm.P.copy(), swarm.G.copy()
