import numpy as np

class Swarm:
    """
    Holds the state of the swarm in a vectorized format.
    N particles in D dimensions.
    """
    def __init__(self, N: int, D: int):
        self.N = N
        self.D = D
        
        # Position and Velocity matrices (N x D)
        self.X = np.zeros((N, D))
        self.V = np.zeros((N, D))
        
        # Personal best positions (N x D)
        self.P = np.zeros((N, D))
        
        # Current fitness values (N,)
        self.F = np.full(N, np.inf)
        
        # Personal best fitness values (N,)
        self.F_P = np.full(N, np.inf)
        
        # Global best position (D,)
        self.G = np.zeros(D)
        self.F_G = np.inf
        
    def update_bests(self, F_new: np.ndarray):
        """
        updates personal and global bests
        """
        self.F = F_new.copy()
        
        # update personal bests(mise a jour)
        improved = self.F < self.F_P
        self.P[improved] = self.X[improved].copy()
        self.F_P[improved] = self.F[improved]
        
        # update global best
        best_idx = np.argmin(self.F_P)
        if self.F_P[best_idx] < self.F_G:
            self.G = self.P[best_idx].copy()
            self.F_G = self.F_P[best_idx]
