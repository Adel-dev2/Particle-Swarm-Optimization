import sys
import os
import argparse
import glob
import json
import csv
import numpy as np

# Ensure the parent directory of pso_framework is in the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from pso_framework.core.optimizer import PSOExecutor
from pso_framework.applications.tsp import parse_tsplib, create_tsp_fitness, TSPLoocalSearchCallback, spv_decode

from pso_framework.strategies.initialization import OppositionBasedInit, RandomUniformInit, LatinHypercubeInit
from pso_framework.strategies.inertia import LinearDecreasingInertia, AdaptiveInertia, ConstantInertia, ChaoticInertia
from pso_framework.strategies.topology import GlobalBestTopology, RingTopology, VonNeumannTopology
from pso_framework.strategies.boundary import ReflectingBoundary, AbsorbingBoundary, DampingBoundary, InvisibleBoundary
from pso_framework.strategies.updater import StandardUpdater, ConstrictionUpdater, BareBonesUpdater, QPSOUpdater

def get_strategy_mappings():
    return {
        'init': {
            'random': RandomUniformInit,
            'lhs': LatinHypercubeInit,
            'obl': OppositionBasedInit
        },
        'inertia': {
            'constant': ConstantInertia,
            'linear': LinearDecreasingInertia,
            'adaptive': AdaptiveInertia,
            'chaotic': ChaoticInertia
        },
        'topology': {
            'gbest': GlobalBestTopology,
            'ring': RingTopology,
            'von_neumann': VonNeumannTopology
        },
        'boundary': {
            'absorbing': AbsorbingBoundary,
            'reflecting': ReflectingBoundary,
            'damping': DampingBoundary,
            'invisible': InvisibleBoundary
        },
        'updater': {
            'standard': StandardUpdater,
            'constriction': ConstrictionUpdater,
            'barebones': BareBonesUpdater,
            'qpso': QPSOUpdater
        }
    }

def main():
    parser = argparse.ArgumentParser(description="Batch Executor for PSO on TSPLIB datasets")
    parser.add_argument('--data_dir', type=str, required=True, help="Directory containing .tsp or .vrp files")
    parser.add_argument('--runs', type=int, default=30, help="Number of independent runs per instance")
    parser.add_argument('--iterations', type=int, default=300, help="Max iterations per run")
    parser.add_argument('--swarm_size', type=int, default=50, help="Number of particles")
    
    # Strategy arguments
    parser.add_argument('--init', type=str, default='random', choices=['random', 'lhs', 'obl'])
    parser.add_argument('--inertia', type=str, default='linear', choices=['constant', 'linear', 'adaptive', 'chaotic'])
    parser.add_argument('--topology', type=str, default='gbest', choices=['gbest', 'ring', 'von_neumann'])
    parser.add_argument('--boundary', type=str, default='reflecting', choices=['absorbing', 'reflecting', 'damping', 'invisible'])
    parser.add_argument('--updater', type=str, default='standard', choices=['standard', 'constriction', 'barebones', 'qpso'])
    
    args = parser.parse_args()
    
    mappings = get_strategy_mappings()
    
    # Instantiate strategies (some may need specific params, we use defaults or simple ones)
    init_strat = mappings['init'][args.init]()
    inertia_strat = mappings['inertia'][args.inertia]()
    topology_strat = mappings['topology'][args.topology]()
    boundary_strat = mappings['boundary'][args.boundary]()
    updater_strat = mappings['updater'][args.updater]()
    
    # Collect files
    files = glob.glob(os.path.join(args.data_dir, '*.tsp')) + glob.glob(os.path.join(args.data_dir, '*.vrp'))
    if not files:
        print(f"No .tsp or .vrp files found in {args.data_dir}")
        return
        
    print(f"Found {len(files)} instances. Beginning batch execution...")
    
    # Create results dir
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    csv_file = os.path.join(results_dir, 'results.csv')
    json_file = os.path.join(results_dir, 'results.json')
    
    all_results = []
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Instance', 'N_Cities', 'Best', 'Mean', 'Worst', 'StdDev', 'Avg_Time_Sec'])
        
        for file_path in files:
            print(f"\nProcessing {os.path.basename(file_path)}...")
            data = parse_tsplib(file_path)
            dist_matrix = data['dist_matrix']
            n_cities = data['dimension']
            instance_name = data['name']
            
            fitness_func = create_tsp_fitness(dist_matrix)
            local_search = TSPLoocalSearchCallback(dist_matrix, freq=20)
            
            run_lengths = []
            run_times = []
            best_length_overall = float('inf')
            best_tour_overall = None
            best_convergence = []
            
            for r in range(args.runs):
                optimizer = PSOExecutor(
                    fitness_func=fitness_func,
                    N=args.swarm_size,
                    D=n_cities,
                    lb=[0.0] * n_cities,
                    ub=[float(n_cities)] * n_cities,
                    T_max=args.iterations,
                    init_strategy=init_strat,
                    inertia_strategy=inertia_strat,
                    topology_strategy=topology_strat,
                    boundary_strategy=boundary_strat,
                    updater_strategy=updater_strat,
                    c1=2.05 if args.updater == 'constriction' else 2.0,
                    c2=2.05 if args.updater == 'constriction' else 2.0,
                    seed=r # deterministic per run
                )
                
                res = optimizer.run(callback=local_search)
                run_lengths.append(res['best_fitness'])
                run_times.append(res['execution_time'])
                
                if res['best_fitness'] < best_length_overall:
                    best_length_overall = res['best_fitness']
                    best_tour_overall = spv_decode(res['best_position']).tolist()
                    best_convergence = res['convergence']
                    
                print(f"  Run {r+1}/{args.runs} - Best: {res['best_fitness']} (Time: {res['execution_time']:.2f}s)")
                
            # Compute stats
            run_lengths = np.array(run_lengths)
            best_val = run_lengths.min()
            mean_val = run_lengths.mean()
            worst_val = run_lengths.max()
            std_val = run_lengths.std()
            avg_time = np.mean(run_times)
            
            writer.writerow([instance_name, n_cities, best_val, mean_val, worst_val, std_val, avg_time])
            f.flush()
            
            all_results.append({
                'instance': instance_name,
                'n_cities': n_cities,
                'runs': args.runs,
                'stats': {
                    'best': best_val,
                    'mean': mean_val,
                    'worst': worst_val,
                    'std': std_val,
                    'avg_time': avg_time
                },
                'best_tour': best_tour_overall,
                'best_convergence': best_convergence,
                'all_lengths': run_lengths.tolist()
            })
            
    with open(json_file, 'w') as f:
        json.dump(all_results, f, indent=4)
        
    print(f"\nBatch complete. Results saved to {results_dir}")

if __name__ == '__main__':
    main()
