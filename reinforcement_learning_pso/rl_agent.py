import numpy as np
from itertools import product as iproduct

OMEGAS  = [0.4, 0.7, 0.9]
C1S     = [1.0, 1.5, 2.0]
C2S     = [1.5, 2.0]
ACTIONS = list(iproduct(OMEGAS, C1S, C2S))  # 18 actions

N_STATES  = 27
N_ACTIONS = 18

class QLearningAgent:
    """
    Q-learning agent for PSO parameter adaptation.
    
    Q-table shape: (27 states, 18 actions)
    State:  (diversity_bin, stagnation_bin, phase_bin) → int [0, 26]
    Action: index into ACTIONS list → (omega, c1, c2) tuple
    Reward: normalized tour length improvement
    """

    def __init__(self, alpha: float = 0.1, gamma: float = 0.9,
                 epsilon: float = 0.5):
        self.Q       = np.zeros((N_STATES, N_ACTIONS))
        self.alpha   = alpha
        self.gamma   = gamma
        self.epsilon = epsilon
        self.actions = ACTIONS

    def choose_action(self, state_idx: int) -> int:
        """ε-greedy action selection."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(N_ACTIONS)    # explore
        return int(np.argmax(self.Q[state_idx]))   # exploit

    def update(self, s: int, a: int, reward: float, s_next: int) -> None:
        """Apply Bellman update to Q[s, a]."""
        td_target = reward + self.gamma * np.max(self.Q[s_next])
        td_error  = td_target - self.Q[s, a]
        self.Q[s, a] += self.alpha * td_error

    def decay_epsilon(self, decay: float = 0.001,
                      min_epsilon: float = 0.05) -> None:
        self.epsilon = max(min_epsilon, self.epsilon - decay)

    def save(self, path: str) -> None:
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, self.Q)
        print(f"Q-table saved to {path}  shape={self.Q.shape}")

    def load(self, path: str) -> None:
        self.Q = np.load(path)
        print(f"Q-table loaded from {path}")

    def best_action(self, state_idx: int) -> tuple:
        """Return (omega, c1, c2) for greedy action in given state."""
        a = int(np.argmax(self.Q[state_idx]))
        return self.actions[a]
