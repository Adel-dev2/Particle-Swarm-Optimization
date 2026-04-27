import numpy as np

class BoundaryStrategy:
    def apply(self, swarm, lb: np.ndarray, ub: np.ndarray):
        """
        Applies boundary constraints to the swarm.
        Modifies swarm.X and/or swarm.V in-place.
        """
        raise NotImplementedError

class AbsorbingBoundary(BoundaryStrategy):
    """
    If a particle hits the boundary, it stops and its velocity is reset to 0.
    """
    def apply(self, swarm, lb: np.ndarray, ub: np.ndarray):
        for d in range(swarm.D):
            over = swarm.X[:, d] > ub[d]
            under = swarm.X[:, d] < lb[d]
            
            swarm.X[over, d] = ub[d]
            swarm.V[over, d] = 0.0
            
            swarm.X[under, d] = lb[d]
            swarm.V[under, d] = 0.0

class ReflectingBoundary(BoundaryStrategy):
    """
    If a particle hits the boundary, it bounces back like a billiard ball, reversing its velocity.
    """
    def apply(self, swarm, lb: np.ndarray, ub: np.ndarray):
        for d in range(swarm.D):
            over = swarm.X[:, d] > ub[d]
            under = swarm.X[:, d] < lb[d]
            
            # Position bounce
            swarm.X[over, d] = 2.0 * ub[d] - swarm.X[over, d]
            swarm.V[over, d] = -swarm.V[over, d]
            
            swarm.X[under, d] = 2.0 * lb[d] - swarm.X[under, d]
            swarm.V[under, d] = -swarm.V[under, d]

class DampingBoundary(BoundaryStrategy):
    """
    Reflects the velocity but dynamically dampens it.
    """
    def apply(self, swarm, lb: np.ndarray, ub: np.ndarray):
        for d in range(swarm.D):
            over = swarm.X[:, d] > ub[d]
            under = swarm.X[:, d] < lb[d]
            
            swarm.X[over, d] = ub[d]
            # Randomly dampen velocity
            swarm.V[over, d] = -np.random.uniform(0, 1, size=over.sum()) * swarm.V[over, d]
            
            swarm.X[under, d] = lb[d]
            swarm.V[under, d] = -np.random.uniform(0, 1, size=under.sum()) * swarm.V[under, d]

class InvisibleBoundary(BoundaryStrategy):
    """
    Positions outside bounds are allowed but fitness is set to infinity.
    This effectively "invisibles" the particles from being tracked as pbest/gbest.
    """
    def apply(self, swarm, lb: np.ndarray, ub: np.ndarray):
        # We don't change X or V, but we need to track who's out of bounds
        # We handle this by setting fitness to inf BEFORE the actual evaluation
        # Or let the optimizer know. A simple way:
        pass # Actual enforcement must be done during or before evaluation.
             # Alternatively, if evaluate handles it, nothing to do here.
