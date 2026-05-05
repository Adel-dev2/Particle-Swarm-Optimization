# RL-PSO Trained Policy (Q-Table)

This table shows the **greedy policy** learned by the agent after 400 episodes. For each of the 27 possible swarm states (combinations of Diversity, Stagnation, and Phase), the agent selected the parameter set ($\omega$, $c_1$, $c_2$) that yields the highest expected reward (Q-Value).

| Diversity | Stagnation | Phase | Best Action (omega, c1, c2) | Q-Value |
|---|---|---|---|---|
| Low | Low (0-5) | Early | (0.4, 2.0, 2.0) | 0.0340 |
| Low | Low (0-5) | Mid | (0.9, 2.0, 2.0) | 0.0159 |
| Low | Low (0-5) | Late | (0.7, 1.0, 2.0) | 0.0092 |
| Low | Medium (6-15) | Early | (0.4, 1.0, 1.5) | 0.0325 |
| Low | Medium (6-15) | Mid | (0.4, 1.5, 1.5) | 0.0169 |
| Low | Medium (6-15) | Late | (0.4, 2.0, 1.5) | 0.0086 |
| Low | High (>15) | Early | (0.7, 2.0, 2.0) | 0.0248 |
| Low | High (>15) | Mid | (0.4, 1.0, 1.5) | 0.0184 |
| Low | High (>15) | Late | (0.4, 1.0, 1.5) | 0.0058 |
| Medium | Low (0-5) | Early | (0.4, 1.0, 1.5) | 0.0309 |
| Medium | Low (0-5) | Mid | (0.7, 1.0, 1.5) | 0.0159 |
| Medium | Low (0-5) | Late | (0.4, 2.0, 1.5) | 0.0113 |
| Medium | Medium (6-15) | Early | (0.4, 1.0, 1.5) | 0.0264 |
| Medium | Medium (6-15) | Mid | (0.4, 2.0, 2.0) | 0.0139 |
| Medium | Medium (6-15) | Late | (0.7, 1.5, 2.0) | 0.0078 |
| Medium | High (>15) | Early | (0.4, 1.5, 1.5) | 0.0229 |
| Medium | High (>15) | Mid | (0.9, 1.5, 1.5) | 0.0098 |
| Medium | High (>15) | Late | (0.4, 1.0, 2.0) | 0.0065 |
| High | Low (0-5) | Early | (0.7, 1.5, 1.5) | 0.0353 |
| High | Low (0-5) | Mid | (0.4, 1.0, 1.5) | 0.0150 |
| High | Low (0-5) | Late | (0.4, 1.0, 1.5) | 0.0105 |
| High | Medium (6-15) | Early | (0.4, 1.0, 2.0) | 0.0278 |
| High | Medium (6-15) | Mid | (0.4, 1.0, 2.0) | 0.0123 |
| High | Medium (6-15) | Late | (0.4, 1.0, 1.5) | 0.0066 |
| High | High (>15) | Early | (0.4, 1.0, 1.5) | 0.0245 |
| High | High (>15) | Mid | (0.7, 2.0, 2.0) | 0.0101 |
| High | High (>15) | Late | (0.7, 2.0, 2.0) | 0.0052 |

### Policy Analysis

Notice the emerging patterns from the training:
* **In Late Phase + High Stagnation**: The agent almost exclusively chooses `omega = 0.4` or `0.7` combined with varying cognitive/social pulls. It learns that high inertia (0.9) is detrimental late in the run when you need particles to refine and exploit the current best areas.
* **Early Phase**: The Q-Values are significantly higher ($\sim 0.03$) because initial random positioning gives way to massive initial improvements, while late-phase improvements (Q-Values $\sim 0.005$) are much harder to achieve.
