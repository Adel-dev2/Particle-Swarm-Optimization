import sys
import os

# Ensure the parent directory of pso_framework is in the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from pso_framework.core.optimizer import PSOExecutor
from pso_framework.applications.tsp import generate_random_tsp, create_tsp_fitness, TSPLoocalSearchCallback, spv_decode

from pso_framework.strategies.initialization import OppositionBasedInit, RandomUniformInit
from pso_framework.strategies.inertia import LinearDecreasingInertia, AdaptiveInertia
from pso_framework.strategies.topology import GlobalBestTopology, RingTopology
from pso_framework.strategies.boundary import ReflectingBoundary
from pso_framework.strategies.updater import StandardUpdater, ConstrictionUpdater

def run_experiment():
    print("Generating a 51-city TSP instance...")
    dist_matrix, _ = generate_random_tsp(n=51, seed=123)
    
    fitness_func = create_tsp_fitness(dist_matrix)
    
    print("Configuring PSO Framework...")
    # Trying out an advanced Configuration
    optimizer = PSOExecutor(
        fitness_func=fitness_func,
        N=50,
        D=51, # dimensionality is number of cities
        lb=[0.0] * 51,
        ub=[51.0] * 51,  # Position bounds
        T_max=300,
        
        # Modular Strategies
        init_strategy=OppositionBasedInit(), 
        inertia_strategy=LinearDecreasingInertia(omega_max=0.9, omega_min=0.4),
        topology_strategy=RingTopology(K=3), # Lbest topology for better diversity
        boundary_strategy=ReflectingBoundary(),
        updater_strategy=ConstrictionUpdater(), # Uses Clerc & Kennedy's Constriction Factor
        
        # PSO parameters config: Since ConstrictionUpdater requires sum > 4
        c1=2.05,
        c2=2.05,
        
        seed=1337
    )
    
    # Provide the 2-opt callback via kwargs
    local_search = TSPLoocalSearchCallback(dist_matrix, freq=20)
    print(local_search.dist)
    print("Starting PSO Optimization...")
    result = optimizer.run(callback=local_search)
    
    print(f"\nOptimization Complete in {result['execution_time']:.2f} seconds!")
    print(f"Best Fitness (Tour Length): {result['best_fitness']}")
    
    tour = spv_decode(result['best_position'])
    print(f"Recovered Route: {tour}")
    print(f"Convergence Start: {result['convergence'][0]}  -> End: {result['convergence'][-1]}")

if __name__ == '__main__':
    run_experiment()
