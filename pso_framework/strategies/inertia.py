import numpy as np

class InertiaStrategy:
    def get_inertia(self, swarm, t: int, t_max: int) -> np.ndarray:
        """
        Returns the inertia weight for the current iteration.
        Returns a scalar or an array of shape (N, 1) or (N, D).
        """
        raise NotImplementedError

class ConstantInertia(InertiaStrategy):
    def __init__(self, omega_0: float = 0.729):
        self.omega_0 = omega_0
        
    def get_inertia(self, swarm, t: int, t_max: int) -> np.ndarray:
        return np.full((swarm.N, 1), self.omega_0)

class LinearDecreasingInertia(InertiaStrategy):
    def __init__(self, omega_max: float = 0.9, omega_min: float = 0.4):
        self.omega_max = omega_max
        self.omega_min = omega_min
        
    def get_inertia(self, swarm, t: int, t_max: int) -> np.ndarray:
        # Linear decay based on the current iteration
        omega = self.omega_max - (self.omega_max - self.omega_min) * (t / t_max)
        return np.full((swarm.N, 1), omega)

class AdaptiveInertia(InertiaStrategy):
    """
    Adaptive inertia weight based on fitness.
    Particles with poor fitness get high inertia (explore).
    Particles with good fitness get low inertia (exploit).
    """
    def __init__(self, omega_max: float = 0.9, omega_min: float = 0.4):
        self.omega_max = omega_max
        self.omega_min = omega_min
        
    def get_inertia(self, swarm, t: int, t_max: int) -> np.ndarray:
        f_min = swarm.F.min()
        f_max = swarm.F.max()
        
        if f_max == f_min:
            # Avoid division by zero if all fitnesses are identical
            return np.full((swarm.N, 1), self.omega_max)
            
        # Linear interpolation based on fitness
        # omega_i = omega_min + (omega_max - omega_min) * (f_i - f_min) / (f_max - f_min)
        omega = self.omega_min + (self.omega_max - self.omega_min) * ((swarm.F - f_min) / (f_max - f_min))
        
        return omega.reshape(-1, 1)

class ChaoticInertia(InertiaStrategy):
    """
    Chaotic inertia weight using the logistic map.
    """
    def __init__(self, mu: float = 4.0):
        self.mu = mu
        # Initial value should be in (0, 1) but avoid 0.25, 0.5, 0.75
        self.current_z = 0.381 
        
    def get_inertia(self, swarm, t: int, t_max: int) -> np.ndarray:
        # Update logistic map
        self.current_z = self.mu * self.current_z * (1.0 - self.current_z)
        # Using z directly as inertia weight
        return np.full((swarm.N, 1), self.current_z)
