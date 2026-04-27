import numpy as np

class TopologyStrategy:
    def get_attractors(self, swarm) -> np.ndarray:
        """
        Given the swarm, return the social attractor (e.g. global or local best) 
        for each particle.
        Returns:
            G_local: np.ndarray of shape (N, D), the social attractor for each particle
        """
        raise NotImplementedError

class GlobalBestTopology(TopologyStrategy):
    """
    Standard Star Topology (Fully connected).
    All particles learn from the single absolute best in the swarm.
    """
    def get_attractors(self, swarm) -> np.ndarray:
        # Replicate global best G for all N particles
        return np.tile(swarm.G, (swarm.N, 1))

class RingTopology(TopologyStrategy):
    """
    Lbest (Ring) Topology.
    Each particle considers a neighborhood of K adjacent particles.
    Default K=3: particle i-1, i, i+1 (with wrap-around).
    """
    def __init__(self, K: int = 3):
        self.K = K
        
    def get_attractors(self, swarm) -> np.ndarray:
        N = swarm.N
        G_local = np.zeros_like(swarm.P)
        
        # Half window size
        hw = self.K // 2
        
        for i in range(N):
            # Compute indices of neighbors with wrap-around
            neighbors = [(i + j) % N for j in range(-hw, hw + 1)]
            
            # Find the best fitness among neighbors
            best_neighbor_idx = neighbors[np.argmin(swarm.F_P[neighbors])]
            G_local[i] = swarm.P[best_neighbor_idx].copy()
            
        return G_local

class VonNeumannTopology(TopologyStrategy):
    """
    Grid topology. Particles are arranged in a 2D grid.
    Neighborhood = itself + 4 cardinal neighbors (N, S, E, W).
    """
    def __init__(self, rows: int = None, cols: int = None):
        self.rows = rows
        self.cols = cols
        
    def get_attractors(self, swarm) -> np.ndarray:
        N = swarm.N
        
        if self.rows is None or self.cols is None:
            # Approximate square grid
            self.rows = int(np.floor(np.sqrt(N)))
            self.cols = int(np.ceil(N / self.rows))
            if self.rows * self.cols < N:
                self.cols += 1
                
        G_local = np.zeros_like(swarm.P)
        
        for i in range(N):
            r = i // self.cols
            c = i % self.cols
            
            # Cardinal neighbors
            up = ((r - 1) % self.rows) * self.cols + c
            down = ((r + 1) % self.rows) * self.cols + c
            left = r * self.cols + ((c - 1) % self.cols)
            right = r * self.cols + ((c + 1) % self.cols)
            
            # Keep within valid bounds (since rows*cols might be slightly > N)
            neighbors = [i]
            for n in [up, down, left, right]:
                if n < N:
                    neighbors.append(n)
                    
            best_neighbor_idx = neighbors[np.argmin(swarm.F_P[neighbors])]
            G_local[i] = swarm.P[best_neighbor_idx].copy()
            
        return G_local
