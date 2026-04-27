import numpy as np

class UpdaterStrategy:
    def update(self, swarm, omega: np.ndarray, G_local: np.ndarray, 
               c1: float, c2: float, V_max: np.ndarray = None, **kwargs):
        """
        Updates swarm.V and swarm.X based on the specific PSO variant.
        """
        raise NotImplementedError

class StandardUpdater(UpdaterStrategy):
    def update(self, swarm, omega: np.ndarray, G_local: np.ndarray, 
               c1: float, c2: float, V_max: np.ndarray = None, **kwargs):
        
        R1 = np.random.uniform(0, 1, size=(swarm.N, swarm.D))
        R2 = np.random.uniform(0, 1, size=(swarm.N, swarm.D))
        
        # Velocity update
        swarm.V = (omega * swarm.V 
                   + c1 * R1 * (swarm.P - swarm.X) 
                   + c2 * R2 * (G_local - swarm.X))
                   
        # Clamping
        if V_max is not None:
            swarm.V = np.clip(swarm.V, -V_max, V_max)
            
        # Position update
        swarm.X = swarm.X + swarm.V

class ConstrictionUpdater(UpdaterStrategy):
    """
    Constriction Factor PSO (Clerc & Kennedy 2002).
    Ignores omega and calculates constriction chi internally.
    Requires c1 + c2 > 4.
    """
    def update(self, swarm, omega: np.ndarray, G_local: np.ndarray, 
               c1: float, c2: float, V_max: np.ndarray = None, **kwargs):
        
        phi = c1 + c2
        if phi <= 4:
            raise ValueError("ConstrictionUpdater requires c1 + c2 > 4")
            
        chi = 2.0 / np.abs(2 - phi - np.sqrt(phi**2 - 4*phi))
        
        R1 = np.random.uniform(0, 1, size=(swarm.N, swarm.D))
        R2 = np.random.uniform(0, 1, size=(swarm.N, swarm.D))
        
        # Constricted Velocity Update
        swarm.V = chi * (swarm.V 
                         + c1 * R1 * (swarm.P - swarm.X) 
                         + c2 * R2 * (G_local - swarm.X))
                         
        if V_max is not None:
            swarm.V = np.clip(swarm.V, -V_max, V_max)
            
        swarm.X = swarm.X + swarm.V

class BareBonesUpdater(UpdaterStrategy):
    """
    Bare-Bones PSO (Kennedy 2003).
    Ignores velocities. Samples next position from Gaussian centered between P and G.
    """
    def update(self, swarm, omega: np.ndarray, G_local: np.ndarray, 
               c1: float, c2: float, V_max: np.ndarray = None, **kwargs):
        
        # Mu is midpoint
        mu = (swarm.P + G_local) / 2.0
        # Sigma is distance
        sigma = np.abs(swarm.P - G_local)
        
        # Sample normal distribution
        swarm.X = np.random.normal(loc=mu, scale=sigma)
        # V is implicitly 0, but we can set it to the actual step taken if needed
        # Or just keep it as 0.
        swarm.V[:] = 0.0

class QPSOUpdater(UpdaterStrategy):
    """
    Quantum-Behaved PSO.
    Uses contraction-expansion coefficient beta.
    No velocity.
    """
    def update(self, swarm, omega: np.ndarray, G_local: np.ndarray, 
               c1: float, c2: float, V_max: np.ndarray = None, **kwargs):
               
        # We can extract beta from kwargs or dynamically from iteration
        # Typically beta decreases from 1.0 to 0.5. Let's assume beta is passed.
        beta = kwargs.get('beta', 0.75)
        
        # Mean best
        mbest = np.mean(swarm.P, axis=0)
        
        # Attractor
        R1 = np.random.uniform(0, 1, size=(swarm.N, swarm.D))
        R2 = np.random.uniform(0, 1, size=(swarm.N, swarm.D))
        phi1 = c1 * R1
        phi2 = c2 * R2
        P_att = (phi1 * swarm.P + phi2 * G_local) / (phi1 + phi2)
        
        u = np.random.uniform(0, 1, size=(swarm.N, swarm.D))
        L = np.abs(mbest - swarm.X)
        
        # Plus/minus randomly
        sign = np.where(np.random.rand(swarm.N, swarm.D) > 0.5, 1, -1)
        
        swarm.X = P_att + sign * beta * L * np.log(1.0 / u)
        swarm.V[:] = 0.0
