import numpy as np

def spv_decode(position: np.ndarray) -> np.ndarray:
    """
    Smallest Position Value (SPV) mapping rule.
    Converts continuous position vector into a discrete parameter permutation.
    """
    return np.argsort(position)

def tour_length(tour: np.ndarray, dist: np.ndarray) -> float:
    """
    Compute full round-trip TSP distance given a tour.
    """
    # Sum of distances between consecutive cities including wrap-around back to start
    return float(dist[tour, np.roll(tour, -1)].sum())

def generate_random_tsp(n: int, seed: int = 42):
    """
    Generate a random set of 2D coordinates for an n-city TSP.
    Returns the distance matrix and coordinates.
    """
    np.random.seed(seed)
    coords = np.random.rand(n, 2) * 100
    
    # Euclidean distance matrix (rounded to nearest integer as per TSPLIB standard)
    dist = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if i != j:
                d = np.sqrt(np.sum((coords[i] - coords[j])**2))
                dist[i, j] = int(np.round(d))
                
    return dist, coords

def two_opt(tour: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """
    Standard 2-opt implementation to improve a TSP tour.
    Iteratively swaps edges to find local improvements.
    """
    n = len(tour)
    tour = tour.copy()
    improved = True
    
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue # Skip wrap-around edge
                    
                a, b = tour[i], tour[i + 1]
                c, d = tour[j], tour[(j + 1) % n]
                
                # If swapping improves layout
                delta = (dist[a, c] + dist[b, d] - dist[a, b] - dist[c, d])
                if delta < 0:
                    tour[i + 1: j + 1] = tour[i + 1: j + 1][::-1]
                    improved = True
                    
    return tour

def create_tsp_fitness(dist: np.ndarray):
    """
    Returns a fitness function for continuous vectors representing a TSP path.
    """
    def fitness(x: np.ndarray) -> float:
        tour = spv_decode(x)
        return tour_length(tour, dist)
    return fitness

class TSPLoocalSearchCallback:
    """
    A callback hook passed to PSO.
    Executes 2-opt on the global best position occasionally.
    """
    def __init__(self, dist: np.ndarray, freq: int = 10):
        self.dist = dist
        self.freq = freq
        self.n = dist.shape[0]
        
    def __call__(self, swarm, F_new: np.ndarray, t: int) -> np.ndarray:
        if t % self.freq == 0:
            # Decode current global best back to tour
            current_best_tour = spv_decode(swarm.G)
            
            # Try applying 2-opt
            improved_tour = two_opt(current_best_tour, self.dist)
            improved_length = tour_length(improved_tour, self.dist)
            
            if improved_length < swarm.F_G:
                # Modifying swarm G implicitly modifies algorithm state
                swarm.F_G = improved_length
                
                # Reconstruct a consistent position for G so it decodes identically
                G_new = np.zeros(self.n)
                for rank, city in enumerate(improved_tour):
                    G_new[city] = float(rank)
                swarm.G = G_new
                
        return F_new

import os

def parse_tsplib(filepath: str):
    """
    Parses a TSPLIB file (.tsp or .vrp) with EUC_2D coordinates.
    Returns a dictionary containing the distance matrix, coordinates, and metadata.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    metadata = {}
    coords = []
    reading_coords = False
    
    for line in lines:
        line = line.strip()
        if not line or line == "EOF":
            continue
            
        if reading_coords:
            parts = line.split()
            if len(parts) >= 3:
                # TSPLIB coordinates are typically: ID X Y
                coords.append([float(parts[1]), float(parts[2])])
            continue
            
        if ":" in line:
            key, val = line.split(":", 1)
            metadata[key.strip()] = val.strip()
        elif "NODE_COORD_SECTION" in line:
            reading_coords = True
            
    coords = np.array(coords)
    n = len(coords)
    
    # Compute Euclidean distance matrix rounded to nearest integer
    dist = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if i != j:
                d = np.sqrt(np.sum((coords[i] - coords[j])**2))
                dist[i, j] = int(np.round(d))
                
    metadata['dimension'] = n
    metadata['name'] = metadata.get('NAME', os.path.basename(filepath))
    
    return {
        'name': metadata['name'],
        'dimension': n,
        'dist_matrix': dist,
        'coords': coords,
        'metadata': metadata
    }
