# PSO Mastery Document – Ground-Up Explanation + MARPOC Project Alignment

> **Method**: Particle Swarm Optimization (PSO)  
> **Project**: MARPOC 2025/2026 — PSO applied to the Traveling Salesman Problem (TSP)  
> **Level**: Ground-Up → Graduate → PhD  
> **Convention**: All math in KaTeX notation. Every symbol explained before use.

---

# ═══════════════════════════════════════════════
# PHASE 1 — COMPLETE GROUND-UP MASTERY OF PSO
# ═══════════════════════════════════════════════

---

# 1. Optimization Problems — The Foundation

## 1.1 What Is an Optimization Problem?

An optimization problem is, at its most fundamental level, a question of the form:

> "Among all possible configurations of a system, which one performs best according to some measure of quality?"

Formally, every optimization problem has three ingredients:

**1. Decision variables**: The quantities we are free to choose. We collect them into a vector $\mathbf{x} = (x_1, x_2, \ldots, x_D)$ where $D$ is the number of variables (the *dimensionality* of the problem).

**2. The search space** $\mathcal{X}$: The set of all values $\mathbf{x}$ is allowed to take. This is the universe of candidate solutions.

**3. The objective function** $f : \mathcal{X} \to \mathbb{R}$: A rule that assigns a single real number (a "score") to every candidate solution. We either minimize or maximize it.

The standard form is minimization (any maximization problem $\max f(\mathbf{x})$ is equivalent to $\min -f(\mathbf{x})$):

$$\min_{\mathbf{x} \in \mathcal{X}} \; f(\mathbf{x})$$

## 1.2 The Search Space

The search space $\mathcal{X}$ can be:

| Type | Description | Example |
|------|-------------|---------|
| **Continuous** | $\mathcal{X} \subseteq \mathbb{R}^D$, real-valued variables | Neural network weights |
| **Discrete** | $\mathcal{X}$ is finite or countable | TSP tour orderings |
| **Mixed** | Some variables continuous, some discrete | Engineering design |
| **Constrained** | $\mathcal{X}$ defined by equalities/inequalities | $g_j(\mathbf{x}) \leq 0$ |

For continuous problems in a box: $\mathcal{X} = \prod_{d=1}^{D} [lb_d, ub_d]$, where $lb_d$ and $ub_d$ are the lower and upper bounds on dimension $d$.

## 1.3 Local vs Global Optima

A point $\mathbf{x}^*$ is a **global minimum** if:
$$f(\mathbf{x}^*) \leq f(\mathbf{x}) \quad \forall \mathbf{x} \in \mathcal{X}$$

A point $\mathbf{x}^*$ is a **local minimum** if there exists a neighborhood $\mathcal{N}(\mathbf{x}^*) \subset \mathcal{X}$ such that:
$$f(\mathbf{x}^*) \leq f(\mathbf{x}) \quad \forall \mathbf{x} \in \mathcal{N}(\mathbf{x}^*)$$

The fundamental challenge of optimization is that gradient-based methods (which follow the slope of $f$ downward) get trapped in local minima and never find the global minimum. This is what motivates population-based metaheuristics like PSO — they simultaneously explore many regions of $\mathcal{X}$.

## 1.4 The Exploration–Exploitation Dilemma

Every search algorithm must balance two competing behaviors:

- **Exploration**: Searching broadly across $\mathcal{X}$ to discover new promising regions. Prevents premature convergence to local optima. Requires *diversity*.
- **Exploitation**: Refining solutions in already-discovered good regions. Converges toward the best known solution. Requires *focus*.

This tension is the central design challenge of every metaheuristic. PSO addresses it through the inertia weight $\omega$ and the balance between cognitive ($c_1$) and social ($c_2$) coefficients — concepts we will derive rigorously in §4 and §6.

---

# 2. Biological Inspiration

## 2.1 Reynolds' Boids Model (1987)

Craig Reynolds (1987) showed that the complex, coordinated movement of bird flocks and fish schools can emerge from just three simple local rules applied by each individual:

1. **Separation**: Avoid crowding neighbors (short-range repulsion).
2. **Alignment**: Steer toward the average heading of neighbors.
3. **Cohesion**: Steer toward the average position of neighbors (long-range attraction).

No global coordinator. No individual has access to the whole flock's state. Yet globally coherent, adaptive behavior emerges — the flock navigates around obstacles, reforms after splitting, and moves as one.

## 2.2 From Flocking to Optimization

Kennedy and Eberhart (1995) asked: *what if each individual in the flock "remembers" the best food source it personally found, and also knows where the best food was found by the entire flock?*

This simple addition transforms Reynolds' social simulation into an optimization engine:
- The **landscape** of food concentration plays the role of the **objective function**.
- A bird's **position** is a **candidate solution**.
- The bird's **movement** is the **search process**.
- Each bird's **personal best food source** is its **cognitive memory**.
- The **flock's best food source** is **social knowledge**.

The flock collectively searches for the highest food concentration (= optimal solution) by simultaneously exploiting personal experience and shared social information.

---

# 3. The Core PSO Model — Every Symbol Explained

## 3.1 The Particle

The fundamental unit of PSO is the **particle**. A particle is an abstraction of one member of the swarm — one candidate solution. Every particle has three state variables:

### 3.1.1 Position $\mathbf{x}_i^t$

$$\mathbf{x}_i^t = \begin{pmatrix} x_{i1}^t \\ x_{i2}^t \\ \vdots \\ x_{iD}^t \end{pmatrix} \in \mathbb{R}^D$$

**Symbol breakdown**:
- $\mathbf{x}$ — the position vector (bold = vector, not scalar)
- $i$ — particle index, $i \in \{1, 2, \ldots, N\}$ where $N$ is the swarm size
- $t$ — time step / iteration index, $t \in \{0, 1, 2, \ldots, T_{max}\}$
- $d$ — dimension index, $d \in \{1, 2, \ldots, D\}$

**Physical meaning**: $\mathbf{x}_i^t$ is the candidate solution that particle $i$ currently represents at iteration $t$. It is a point in the $D$-dimensional search space $\mathcal{X}$. Evaluating $f(\mathbf{x}_i^t)$ tells us how good this solution is.

**Analogy**: If you are optimizing the shape of an aircraft wing with 20 parameters (wingspan, chord, twist angle, etc.), then $D = 20$ and $\mathbf{x}_i^t$ is one specific wing configuration — one point in a 20-dimensional design space.

### 3.1.2 Velocity $\mathbf{v}_i^t$

$$\mathbf{v}_i^t = \begin{pmatrix} v_{i1}^t \\ v_{i2}^t \\ \vdots \\ v_{iD}^t \end{pmatrix} \in \mathbb{R}^D$$

**Symbol breakdown**:
- $\mathbf{v}$ — the velocity vector (same dimensions as position)
- All subscripts and superscripts have the same meaning as for $\mathbf{x}$

**Physical meaning**: $\mathbf{v}_i^t$ controls *where the particle will move next* — its direction and speed of movement through the search space. Unlike classical gradient descent, the velocity is not the gradient of $f$; it is an autonomous quantity that evolves according to the PSO update rule.

**Critical insight**: Velocity gives PSO *momentum*. A particle does not simply jump to the best known position; it moves *toward* better positions gradually, retaining memory of its trajectory. This momentum is what enables PSO to escape shallow local optima.

### 3.1.3 Personal Best $\mathbf{p}_i$

$$\mathbf{p}_i = \arg\min_{\mathbf{x}_i^\tau, \; \tau \leq t} f(\mathbf{x}_i^\tau)$$

**Symbol breakdown**:
- $\mathbf{p}$ — personal best position (sometimes written $\mathbf{pbest}_i$)
- $\tau$ — a dummy iteration index ranging over all past iterations up to the current $t$
- $\arg\min$ — "the argument that minimizes" — the position, not the value

**Physical meaning**: $\mathbf{p}_i$ is the best position (candidate solution) that particle $i$ has ever visited, across all iterations from initialization until now. It represents the particle's *cognitive memory* — its own past experience. The value $f(\mathbf{p}_i)$ is the best objective function value particle $i$ has achieved so far.

**Update rule**: After each position update, if $f(\mathbf{x}_i^{t+1}) < f(\mathbf{p}_i)$, then $\mathbf{p}_i \leftarrow \mathbf{x}_i^{t+1}$.

### 3.1.4 Global Best $\mathbf{g}$

$$\mathbf{g} = \arg\min_{\mathbf{p}_j, \; j \in \{1,\ldots,N\}} f(\mathbf{p}_j)$$

**Symbol breakdown**:
- $\mathbf{g}$ — global best position (sometimes written $\mathbf{gbest}$)
- $j$ — particle index ranging over all particles in the swarm

**Physical meaning**: $\mathbf{g}$ is the best position found by *any* particle in the *entire swarm* at any point in time. It represents *social knowledge* — collective wisdom about where in $\mathcal{X}$ the best solution has been found so far.

**Update rule**: After updating all personal bests, if any $f(\mathbf{p}_j) < f(\mathbf{g})$, then $\mathbf{g} \leftarrow \mathbf{p}_j$.

## 3.2 The Swarm

The **swarm** is the collection of all $N$ particles:
$$\mathcal{S}^t = \{\mathbf{x}_1^t, \mathbf{x}_2^t, \ldots, \mathbf{x}_N^t\}$$

Typical swarm size: $N \in [20, 100]$. Larger swarms explore better but cost more evaluations. For TSP with $n$ cities, $N = 50$ is a standard choice.

---

# 4. The Velocity Update Equation — Full Derivation

## 4.1 The Equation

The heart of PSO is the velocity update equation, introduced by Shi & Eberhart (1998) with the inertia weight:

$$\boxed{\mathbf{v}_i^{t+1} = \omega \mathbf{v}_i^t + c_1 r_1^t (\mathbf{p}_i - \mathbf{x}_i^t) + c_2 r_2^t (\mathbf{g} - \mathbf{x}_i^t)}$$

This equation is deceptively simple. Let us dissect every term with full rigor.

## 4.2 Term 1: The Inertia Term $\omega \mathbf{v}_i^t$

**Symbol breakdown**:
- $\omega$ (omega) — the **inertia weight**, a scalar $\in [0, 1]$
- $\mathbf{v}_i^t$ — the particle's velocity at the previous iteration

**Physical meaning**: This term *preserves the particle's current momentum*. Without it ($\omega = 0$), the particle would have no memory of where it was going — it would restart its velocity from zero at every iteration, losing all directional coherence. With $\omega = 1$, the particle maintains full momentum indefinitely.

**Effect on exploration/exploitation**:
- High $\omega$ (close to 1): the particle moves far; good **exploration** — searching new regions.
- Low $\omega$ (close to 0): the particle slows down; good **exploitation** — refining around good solutions.

**Origin**: In the original 1995 Kennedy–Eberhart paper, there was *no inertia weight*. The velocity simply accumulated without damping, causing particles to accelerate uncontrollably. Shi & Eberhart (1998) introduced $\omega$ as a replacement for velocity clamping, providing more principled control.

## 4.3 Term 2: The Cognitive Term $c_1 r_1^t (\mathbf{p}_i - \mathbf{x}_i^t)$

**Symbol breakdown**:
- $c_1$ — **cognitive coefficient** (also called $\phi_1$, acceleration constant 1), scalar $> 0$
- $r_1^t$ — random vector: each component $r_{1d}^t \sim \mathcal{U}(0, 1)$ i.i.d., resampled at each iteration $t$
- $(\mathbf{p}_i - \mathbf{x}_i^t)$ — the **displacement vector** from current position to personal best

**Physical meaning**: $(\mathbf{p}_i - \mathbf{x}_i^t)$ points *from where the particle currently is* toward *the best place it has personally ever been*. Multiplied by $c_1 r_1^t$, this creates an attractive force pulling the particle back toward its own best historical position.

**Why $r_1^t$ is random**: Without randomness, the cognitive component would always pull the particle in the *exact same direction* with the *exact same strength*. Every particle starting from the same position would follow *identical trajectories*. The random $r_1^t$ stochastically varies the strength of the pull each iteration, creating diversity in trajectories and enabling exploration of the neighborhood around $\mathbf{p}_i$.

**Effect of $c_1$**:
- Large $c_1$: particles trust their own memory strongly; they return to past good positions — more individualistic, less social.
- $c_1 = 0$: particles have no cognitive component; they follow only the swarm — "social only" PSO.

## 4.4 Term 3: The Social Term $c_2 r_2^t (\mathbf{g} - \mathbf{x}_i^t)$

**Symbol breakdown**:
- $c_2$ — **social coefficient** (also called $\phi_2$, acceleration constant 2), scalar $> 0$
- $r_2^t$ — random vector independent of $r_1^t$: each component $r_{2d}^t \sim \mathcal{U}(0, 1)$ i.i.d.
- $(\mathbf{g} - \mathbf{x}_i^t)$ — displacement vector from current position to global best

**Physical meaning**: $(\mathbf{g} - \mathbf{x}_i^t)$ points toward the best solution found by *any* particle in the swarm. This is the *social influence* — each particle is attracted to the collective wisdom of the entire swarm.

**Effect of $c_2$**:
- Large $c_2$: particles rush toward the global best — fast convergence but risk of premature convergence if $\mathbf{g}$ is a local optimum.
- $c_2 = 0$: particles have no social component; they rely only on their own memory — "cognitive only" PSO.

**Why $r_1$ and $r_2$ are independent**: If they were the same random variable, both pulls (cognitive and social) would be simultaneously strong or simultaneously weak, losing the independent stochastic sampling of the two search directions.

## 4.5 Combining the Three Terms: Geometric Intuition

At each iteration, the new velocity of particle $i$ is the weighted sum of three vectors:
1. Current momentum (where it was heading)
2. Pull toward its own past best (personal experience)
3. Pull toward the swarm's best (social information)

This creates a **stochastic attractor** centered between $\mathbf{p}_i$ and $\mathbf{g}$. The particle does not converge to either attractor alone, but to a probabilistic combination of both — the so-called **weighted centroid**:

$$\mathbf{G}_i = \frac{c_1 \mathbf{p}_i + c_2 \mathbf{g}}{c_1 + c_2}$$

When $c_1 = c_2$, this centroid is exactly the midpoint between $\mathbf{p}_i$ and $\mathbf{g}$.

---

# 5. The Position Update Equation

$$\boxed{\mathbf{x}_i^{t+1} = \mathbf{x}_i^t + \mathbf{v}_i^{t+1}}$$

**Symbol breakdown**:
- $\mathbf{x}_i^{t+1}$ — particle $i$'s new position after the update
- $\mathbf{x}_i^t$ — particle $i$'s current position
- $\mathbf{v}_i^{t+1}$ — the newly computed velocity (from §4.1)

**Physical meaning**: The new position is the old position displaced by the new velocity. This is the standard kinematic equation $\text{position} = \text{position} + \text{velocity} \times \Delta t$, with time step $\Delta t = 1$ absorbed into the velocity definition.

**Update order**: The velocity is computed first (§4), then the position is updated using the new velocity. The order matters — updating position first would give a different algorithm.

---

# 6. The Inertia Weight $\omega$ — Complete Analysis

## 6.1 Why $\omega$ Was Introduced

The original PSO (1995) had no inertia weight — the equation was simply:
$$\mathbf{v}_i^{t+1} = \mathbf{v}_i^t + c_1 r_1 (\mathbf{p}_i - \mathbf{x}_i^t) + c_2 r_2 (\mathbf{g} - \mathbf{x}_i^t)$$

The problem: velocities grow unboundedly. When $\mathbf{p}_i$ and $\mathbf{g}$ are far from $\mathbf{x}_i^t$, the cognitive and social terms add large increments to $\mathbf{v}_i$. Over many iterations, particles fly out of the feasible region $\mathcal{X}$ at ever-increasing speed — a phenomenon called **velocity explosion**.

Early fix: **velocity clamping** — hard clip $v_{id} \leftarrow \text{clip}(v_{id}, -V_{max}, V_{max})$. But choosing $V_{max}$ requires tuning.

Better fix (Shi & Eberhart 1998): multiply the previous velocity by $\omega \in (0, 1)$ — this geometrically decays the velocity at each step, preventing explosion without an arbitrary clamp. A particle carrying excessive velocity from a previous iteration has that velocity reduced by a factor $\omega$ before the new terms are added.

## 6.2 Strategy 1: Constant Inertia Weight

$$\omega(t) = \omega_0 \quad \forall t$$

**Use when**: The problem landscape is well-understood and a single balance point is known. Rarely used in practice; just a baseline.

**Typical value**: $\omega_0 = 0.729$ (derived from the constriction factor — see §7.3).

## 6.3 Strategy 2: Linear Decreasing Inertia Weight (LDIW)

$$\omega(t) = \omega_{max} - \frac{\omega_{max} - \omega_{min}}{T_{max}} \cdot t$$

**Symbol breakdown**:
- $\omega_{max}$ — initial (maximum) inertia weight, typically $0.9$
- $\omega_{min}$ — final (minimum) inertia weight, typically $0.4$
- $T_{max}$ — maximum number of iterations
- $t$ — current iteration

**Behavior**:
- At $t = 0$: $\omega = 0.9$ → high momentum → broad exploration
- At $t = T_{max}$: $\omega = 0.4$ → low momentum → fine exploitation

**Motivation**: Early in the search, we want particles to cover much of $\mathcal{X}$ quickly (high inertia). Late in the search, we want them to converge carefully around good solutions (low inertia). LDIW schedules this transition linearly.

**Limitation**: The transition is blind to actual search progress. If the swarm converges early, $\omega$ is still high and wastes evaluations on broad search. If convergence is slow, $\omega$ may drop too fast and trap particles.

## 6.4 Strategy 3: Adaptive Inertia Weight

$$\omega_i(t) = \omega_{min} + (\omega_{max} - \omega_{min}) \cdot \frac{f_i^t - f_{min}^t}{f_{max}^t - f_{min}^t}$$

**Symbol breakdown**:
- $f_i^t = f(\mathbf{x}_i^t)$ — current fitness of particle $i$ at iteration $t$
- $f_{min}^t = \min_j f(\mathbf{x}_j^t)$ — best current fitness in the swarm
- $f_{max}^t = \max_j f(\mathbf{x}_j^t)$ — worst current fitness in the swarm

**Behavior**: Particles with poor fitness (far from best) get **high inertia** — they explore broadly. Particles with good fitness (close to best) get **low inertia** — they exploit their neighborhood. Each particle has its own $\omega_i$ at each step.

## 6.5 Strategy 4: Chaotic Inertia Weight

Replace the deterministic schedule with a chaotic sequence generated by the logistic map:
$$\omega(t+1) = \mu \cdot \omega(t) \cdot (1 - \omega(t)), \quad \mu = 4$$

with $\omega(0)$ random in $(0, 1) \setminus \{0.25, 0.5, 0.75\}$.

The logistic map with $\mu = 4$ is a fully chaotic dynamical system — it generates non-repeating, deterministic-but-unpredictable sequences that cover $(0, 1)$ ergodically. This provides a principled form of randomness in $\omega$ without user-defined schedules.

---

# 7. Cognitive/Social Coefficients and the Constriction Factor

## 7.1 The Coefficients $c_1$ and $c_2$

The **acceleration coefficients** $c_1$ and $c_2$ control the *maximum step size* toward $\mathbf{p}_i$ and $\mathbf{g}$ respectively.

Since $r_1, r_2 \sim \mathcal{U}(0, 1)$, the average step toward $\mathbf{p}_i$ is $\frac{c_1}{2} (\mathbf{p}_i - \mathbf{x}_i^t)$ and similarly for $\mathbf{g}$.

**Standard values**: $c_1 = c_2 = 2.0$ (Kennedy & Eberhart 1995).

**Effect of asymmetry**:
- $c_1 > c_2$: particles trust themselves more than the swarm → more exploration, slower convergence.
- $c_2 > c_1$: particles are drawn strongly toward $\mathbf{g}$ → fast convergence, higher premature convergence risk.
- $c_1 = c_2$: symmetric balance, most common default.

**The sum $\phi = c_1 + c_2$**: This appears in the stability analysis. The condition $\phi > 4$ is required for the constriction factor to be real-valued (§7.3). With $c_1 = c_2 = 2.05$, $\phi = 4.1 > 4$. ✓

## 7.2 Velocity Clamping

Even with inertia weight, velocity can become large in high-dimensional problems. Velocity clamping enforces:
$$v_{id}^{t+1} \leftarrow \text{clip}(v_{id}^{t+1},\; -V_{max,d},\; V_{max,d})$$

Standard choice: $V_{max,d} = \frac{ub_d - lb_d}{2}$, meaning the maximum one-step displacement is half the search range per dimension.

**Too small $V_{max}$**: particles cannot reach distant good regions → excessive exploitation.
**Too large $V_{max}$**: particles overshoot attractors repeatedly → excessive exploration, poor convergence.

## 7.3 The Constriction Factor $\chi$ (Clerc & Kennedy 2002)

Clerc & Kennedy analyzed the PSO velocity update as a dynamical system and derived a **constriction factor** $\chi$ that *guarantees convergence* of particle trajectories.

### Derivation

Consider a simplified 1D, single-particle PSO with fixed attractors ($\mathbf{p}_i$ and $\mathbf{g}$ constant). The velocity update is:
$$v^{t+1} = \omega v^t + c_1 r_1 (p - x^t) + c_2 r_2 (g - x^t)$$

Let $\phi_1 = c_1 r_1$, $\phi_2 = c_2 r_2$ be fixed (taking expectations over $r_1, r_2$). Define the attractor:
$$\tilde{x} = \frac{\phi_1 p + \phi_2 g}{\phi_1 + \phi_2}, \quad \phi = \phi_1 + \phi_2$$

Substituting $y^t = x^t - \tilde{x}$ (displacement from attractor), the system becomes:
$$y^{t+1} = (1 - \phi) y^t + \omega \cdot (\text{velocity term})$$

Written as a 2D linear system $\mathbf{s}^t = (y^t, v^t)^\top$:
$$\mathbf{s}^{t+1} = \mathbf{A} \mathbf{s}^t, \quad \mathbf{A} = \begin{pmatrix} 1 - \phi & 1 \\ -\phi & \omega \end{pmatrix}$$

For convergence, both eigenvalues of $\mathbf{A}$ must lie inside the unit disk $|\lambda| < 1$.

The eigenvalues are:
$$\lambda_{1,2} = \frac{(1 - \phi + \omega) \pm \sqrt{(1 - \phi + \omega)^2 - 4\omega}}{2}$$

Requiring $|\lambda_{1,2}| < 1$ yields the stability conditions. Clerc & Kennedy showed that multiplying the entire velocity update by a constriction factor:

$$\chi = \frac{2}{|2 - \phi - \sqrt{\phi^2 - 4\phi}|}, \quad \phi = c_1 + c_2, \quad \phi > 4$$

guarantees $|\lambda_{1,2}| \leq 1$.

### Constricted PSO Update

$$\mathbf{v}_i^{t+1} = \chi \left[\mathbf{v}_i^t + c_1 r_1 (\mathbf{p}_i - \mathbf{x}_i^t) + c_2 r_2 (\mathbf{g} - \mathbf{x}_i^t)\right]$$

With **standard parameters**: $c_1 = c_2 = 2.05$, $\phi = 4.1$:
$$\chi = \frac{2}{|2 - 4.1 - \sqrt{4.1^2 - 4 \cdot 4.1}|} = \frac{2}{|{-2.1} - \sqrt{0.41}|} = \frac{2}{2.1 + 0.6403} \approx 0.7298$$

**Relationship to inertia weight**: Constricted PSO with $c_1 = c_2 = 2.05$ is *mathematically equivalent* to PSO with $\omega = \chi = 0.7298$ and scaled coefficients — but with the guarantee of convergence built in.

---

# 8. Random Factors $r_1$, $r_2$ — Why They Are Essential

## 8.1 The Role of Stochasticity

The random factors $r_1^t$ and $r_2^t$ are not an afterthought — they are architecturally essential.

**Without randomness** ($r_1 = r_2 = 1$ always): The velocity update is deterministic. Given the same initial positions and velocities, all particles would follow identical, predetermined trajectories. The algorithm would not be a stochastic optimizer but a deterministic fixed-point iteration. In high-dimensional, multimodal landscapes, such determinism would be catastrophically sensitive to initialization.

**With randomness**: Each particle samples a *different* direction toward its attractors at each step. This creates stochastic trajectories that collectively cover the neighborhood of $(\mathbf{p}_i, \mathbf{g})$ ergodically — meaning, given enough time, the particle will visit every region of this neighborhood.

## 8.2 Distribution of $r_1$, $r_2$

**Standard**: $r_{1d}, r_{2d} \sim \mathcal{U}(0, 1)$ independently for each dimension $d$ and each iteration $t$.

- Each component is sampled *independently* across dimensions: the cognitive pull in dimension 1 has nothing to do with the cognitive pull in dimension 2.
- This *per-dimension* randomization is crucial: it allows the particle to move in oblique directions (not just directly toward $\mathbf{p}_i$ or $\mathbf{g}$) and to sample a much richer set of trajectories.

**Alternatives**:
- $r_1, r_2 \sim \mathcal{N}(0.5, \sigma^2)$: Gaussian-distributed, biased toward 0.5. Used in some variants but less common.
- Chaotic maps: $r$ generated by logistic map — deterministic but ergodic, avoids clumping near 0 or 1 that can occur with pseudo-random generators.

---

# 9. Initialization Strategies

## 9.1 Why Initialization Matters

PSO's convergence speed and solution quality depend heavily on the initial distribution of particles in $\mathcal{X}$. If all particles start clustered in one region, they will never efficiently explore the rest. A well-distributed initial swarm ensures broad initial coverage, giving the algorithm the best chance to find the global basin of attraction.

## 9.2 Random Uniform Initialization

$$x_{id}^0 \sim \mathcal{U}(lb_d, ub_d), \quad v_{id}^0 \sim \mathcal{U}\left(-V_{max,d}, V_{max,d}\right)$$

**Pros**: Simple, no additional cost.
**Cons**: Clustering can occur by chance (especially in high dimensions — the curse of dimensionality makes random coverage poor for $D > 20$).

## 9.3 Latin Hypercube Sampling (LHS)

Divide each dimension into $N$ equal intervals. Assign each particle to one interval per dimension, ensuring exactly one particle per interval per dimension. Then shuffle independently per dimension.

**Guarantee**: Exact coverage of the marginal distribution in each dimension. Far superior to pure random for $N < D^2$.

**Python implementation**:
```python
from scipy.stats import qmc
sampler = qmc.LatinHypercube(d=D)
sample = sampler.random(n=N)  # N×D array in [0,1]
positions = qmc.scale(sample, lb, ub)
```

## 9.4 Opposition-Based Initialization (OBL)

For each initial particle $\mathbf{x}_i^0$, generate its **opposite**:
$$\hat{x}_{id}^0 = lb_d + ub_d - x_{id}^0$$

Keep the better of $\mathbf{x}_i^0$ and $\hat{x}_{id}^0$ (evaluate both). This doubles the effective initial sampling density.

**Intuition**: The opposite point is on the other side of the search space midpoint. If the initial point is bad, its opposite is likely in a very different region — 50% chance it is better. OBL exploits this to improve the initial swarm quality at cost of $N$ extra evaluations.

---

# 10. Boundary Handling Strategies

When a particle's position update places it outside $\mathcal{X} = [lb, ub]^D$, a boundary condition must be applied.

## 10.1 Absorbing (Sticky) Boundaries

If $x_{id}^{t+1} > ub_d$: set $x_{id}^{t+1} = ub_d$, $v_{id}^{t+1} = 0$.
If $x_{id}^{t+1} < lb_d$: set $x_{id}^{t+1} = lb_d$, $v_{id}^{t+1} = 0$.

**Effect**: Particles hitting the boundary stop and restart with zero velocity. Simple, but creates artificial accumulation at boundaries. The velocity reset can be damaging (loses momentum).

## 10.2 Reflecting Boundaries

$$\text{if } x_{id} > ub_d: \quad x_{id} \leftarrow 2 \cdot ub_d - x_{id}, \quad v_{id} \leftarrow -v_{id}$$
$$\text{if } x_{id} < lb_d: \quad x_{id} \leftarrow 2 \cdot lb_d - x_{id}, \quad v_{id} \leftarrow -v_{id}$$

**Effect**: Particles bounce off boundaries like a billiard ball. No accumulation at boundaries. Can cause chaotic bouncing if velocity is very large (particle keeps reflecting).

## 10.3 Damping Boundaries

$$\text{if } x_{id} > ub_d: \quad x_{id} \leftarrow ub_d, \quad v_{id} \leftarrow -\text{rand}() \cdot v_{id}$$

Velocity is reflected and randomly damped. A compromise between absorbing and reflecting.

## 10.4 Invisible Boundaries (Periodic)

Positions outside $[lb, ub]^D$ are allowed but their fitness is set to $+\infty$ (for minimization). The particle keeps moving but cannot influence pbest or gbest while outside.

**Effect**: Preserves velocity, allows the particle to re-enter the feasible region naturally. Avoids artificial boundary attraction.

## 10.5 Recommendation for TSP

For the SPV encoding (§Chapter 13), positions are in $[0, n]^n$ (real-valued, $n$-dimensional). Reflecting boundaries work well since the SPV mapping is insensitive to the exact range — only the *ranking* of values matters, so even out-of-bound values produce a valid permutation after argsort.

---

# 11. Topology — Information Flow in the Swarm

## 11.1 Why Topology Matters

The **topology** of the swarm defines *which particles can share information with which other particles*. The global best $\mathbf{g}$ in the standard formulation is defined over the *entire* swarm — this is the **gbest topology** (fully connected). But it is not the only option.

**Key insight**: When $\mathbf{g}$ is shared with all particles immediately, the whole swarm converges toward $\mathbf{g}$. If $\mathbf{g}$ is a local optimum, every particle rushes toward it — this is **premature convergence**. Restricting the neighborhood slows information spread but preserves diversity.

## 11.2 Fully Connected (gbest / Star Topology)

Every particle communicates with every other particle. $\mathbf{g}$ is the true global best over all $N$ particles.

**Properties**:
- Fastest convergence speed
- Lowest diversity, highest premature convergence risk
- Best for unimodal problems

## 11.3 Ring (lbest) Topology

Each particle $i$ only considers a neighborhood $\mathcal{N}_i$ of $K$ adjacent particles (e.g., $i-1, i, i+1$ for $K=3$). The local best is:
$$\mathbf{l}_i = \arg\min_{\mathbf{p}_j,\; j \in \mathcal{N}_i} f(\mathbf{p}_j)$$

**Properties**:
- Information spreads slowly (only to adjacent particles)
- High diversity maintained longer
- Better for multimodal problems
- Slower convergence but often finds better final solutions

## 11.4 Von Neumann (Grid) Topology

Particles arranged on a 2D grid. Each particle's neighborhood = itself + 4 cardinal neighbors (N, S, E, W).

**Properties**: Intermediate between gbest and lbest. Generally best empirical performance on benchmark suites (Mendes 2004, Kennedy & Mendes 2002).

## 11.5 Dynamic (Adaptive) Topology

Start with lbest (high diversity). When the swarm stagnates (no improvement for $K$ iterations), increase connectivity. Switch to gbest in final phase. 

This adaptive strategy attempts to get the best of both worlds — exploration early, exploitation late.

---

# 12. Convergence, Stability, and Theoretical Analysis

## 12.1 Parameter Conditions for Stability

From §7.3, the constriction analysis requires $\phi = c_1 + c_2 > 4$ for real $\chi$.

For PSO with inertia weight (no constriction), stability of the mean requires (Clerc & Kennedy 2002):
$$0 < \omega < 1, \quad \phi < 2(1 + \omega)$$

where $\phi = c_1 + c_2$ (using expected values $r_1 = r_2 = 0.5$).

With $c_1 = c_2 = 2.0$, $\phi = 4.0$:
$$4 < 2(1 + \omega) \implies \omega > 1$$

This is **outside** the stable range! The standard setting $c_1 = c_2 = 2.0$, $\omega \in (0, 1)$ is *technically unstable in mean* — particles converge to the attractor in a damped oscillation but with increasing variance. In practice, updating $\mathbf{p}_i$ and $\mathbf{g}$ during the run "resets" the system, preventing true divergence.

**Safe parameters**: $c_1 = c_2 = 1.49445$, $\omega = 0.7298$ (equivalent to constriction with $\chi = 0.7298$, $c_1 = c_2 = 2.05$).

## 12.2 Premature Convergence

**Definition**: All particles converge to the same position (often a local optimum), and the swarm loses diversity — it cannot escape even with random factors.

**When it occurs**:
1. Global best $\mathbf{g}$ is a strong attractor (large $c_2$, gbest topology)
2. $\omega$ drops too quickly (LDIW with $\omega_{min}$ close to 0)
3. Swarm size $N$ too small (insufficient diversity)
4. Problem is highly multimodal

**Diagnosis**: Monitor the *swarm diversity* $\sigma^t = \frac{1}{N}\sum_{i=1}^N \|\mathbf{x}_i^t - \bar{\mathbf{x}}^t\|_2$ where $\bar{\mathbf{x}}^t = \frac{1}{N}\sum_i \mathbf{x}_i^t$. When $\sigma^t \approx 0$, the swarm has collapsed.

**Remedies**:
- Switch to lbest topology
- Increase $\omega_{min}$
- Restart worst particles from random positions
- Use mutation: occasionally perturb $\mathbf{v}_i$ with Gaussian noise

---

# 13. All Major PSO Variants — Complete Technical Reference

## 13.1 Standard PSO (SPSO 2007, SPSO 2011)

The official "standard" PSO benchmarks defined by the PSO community (Bratton & Kennedy 2007, Zambrano-Bigiarini et al. 2013):

**SPSO 2011** key features:
- Constriction factor $\chi = 0.7298$, $c_1 = c_2 = 1.49618$
- Random topology: each iteration, $K$ neighbors chosen uniformly at random
- Topology rebuilt each iteration with probability proportional to lack of improvement
- Hyperspherical sampling: velocity toward a random point inside the hypersphere centered at the centroid $\mathbf{G}_i$

**SPSO 2011 velocity update**:
$$\mathbf{v}_i^{t+1} = \omega \mathbf{v}_i^t + \text{sample from hypersphere}\left(\mathbf{G}_i - \mathbf{x}_i^t, c\right)$$

where $c = c_1 + c_2 = 2.98$ and the hyperspherical sampling provides uniform coverage around the centroid.

## 13.2 Bare-Bones PSO (Kennedy 2003)

Eliminates the velocity vector entirely. The new position is sampled directly from a Gaussian centered between $\mathbf{p}_i$ and $\mathbf{g}$:

$$x_{id}^{t+1} \sim \mathcal{N}\!\left(\mu_{id}, \sigma_{id}^2\right)$$
$$\mu_{id} = \frac{p_{id} + g_d}{2}, \quad \sigma_{id} = |p_{id} - g_d|$$

**Pros**: No velocity parameters to tune. Theoretically elegant — $x_{id}^{t+1}$ samples the 68% confidence interval $[p_{id} - |p_{id} - g_d|,\; p_{id} + |p_{id} - g_d|]$ approximately.

**Cons**: Exploration purely Gaussian — poor coverage of tails. No momentum.

## 13.3 Fully Informed PSO (FIPS) — Mendes et al. 2004

Instead of using only $\mathbf{g}$ as the social attractor, every neighbor's personal best influences the velocity:

$$\mathbf{v}_i^{t+1} = \chi \left[\mathbf{v}_i^t + \sum_{j \in \mathcal{N}_i} \frac{\phi}{|\mathcal{N}_i|} r_j (\mathbf{p}_j - \mathbf{x}_i^t)\right]$$

where $\phi = c_1 + c_2$ and $r_j \sim \mathcal{U}(0, 1)$ for each neighbor $j$.

**Key difference from standard PSO**: Every neighbor's best pulls the particle — not just the single global best. This creates a *richer attractor landscape* and maintains more diversity.

**Performance**: FIPS with von Neumann topology is among the best-performing PSO variants on CEC benchmark suites.

## 13.4 Comprehensive Learning PSO (CLPSO) — Liang et al. 2006

CLPSO allows each particle dimension to learn from a *different* particle's personal best:

$$v_{id}^{t+1} = \omega v_{id}^t + c \cdot r_{id} \left(p_{f_i(d), d} - x_{id}^t\right)$$

where $f_i(d)$ is the index of the particle whose dimension-$d$ personal best is used for particle $i$'s dimension $d$.

$f_i(d)$ is chosen via tournament selection: compare two randomly drawn particles' $d$-th dimension pbest values, pick the better one.

**Why this helps**: In standard PSO, $\mathbf{p}_i$ and $\mathbf{g}$ guide all dimensions of particle $i$ simultaneously. If these attractors are suboptimal in some dimensions, all particles get pulled away from those dimensions' optima. CLPSO decouples dimensions, allowing each to learn from the best source per dimension.

**Best for**: High-dimensional, multimodal problems where different dimensions have uncorrelated optima.

## 13.5 Quantum-Behaved PSO (QPSO) — Sun et al. 2004

Inspired by quantum mechanics: particles have no definite trajectory; their position is characterized by a *probability amplitude*. The position is sampled from a probability distribution centered at the attractor.

$$\mathbf{p}_{att,i} = \frac{c_1 \mathbf{p}_i + c_2 \mathbf{g}}{c_1 + c_2}$$

$$x_{id}^{t+1} = p_{att,id} \pm \beta \cdot |mbest_d - x_{id}^t| \cdot \ln\!\left(\frac{1}{u}\right)$$

where:
- $\beta$ — contraction-expansion coefficient (controls exploration/exploitation, typically decreasing from 1.0 to 0.5)
- $u \sim \mathcal{U}(0, 1)$
- $mbest_d = \frac{1}{N}\sum_{i=1}^N p_{id}$ — mean of all personal bests in dimension $d$
- The $\pm$ sign is chosen randomly with equal probability

**Key advantage**: No velocity vector → no $\omega$, $c_1$, $c_2$, $V_{max}$ to tune. Only $\beta$ (and $T_{max}$, $N$). Convergence guaranteed under mild conditions.

**Performance on TSP**: Competitive with constricted PSO, especially when combined with 2-opt.

## 13.6 Discrete PSO (Binary/Permutation)

Kennedy & Eberhart (1997) introduced binary PSO where each position component $x_{id} \in \{0, 1\}$. The velocity $v_{id}$ is reinterpreted as a *probability of $x_{id}$ being 1*:
$$v_{id}^{t+1} = \text{(standard velocity update)}$$
$$x_{id}^{t+1} = \begin{cases} 1 & \text{if } \text{sigmoid}(v_{id}^{t+1}) > r,\; r \sim \mathcal{U}(0,1) \\ 0 & \text{otherwise} \end{cases}$$
$$\text{sigmoid}(v) = \frac{1}{1 + e^{-v}}$$

For **permutation problems** (TSP), two approaches exist (detailed in Chapter 16):
1. **SPV rule**: keep continuous positions, convert to permutation via argsort
2. **Swap-operator PSO**: redefine $+$, $-$, $\times$ for permutations using swap sequences

## 13.7 Multi-Objective PSO (MOPSO)

For problems with $k > 1$ objective functions $\mathbf{F}(\mathbf{x}) = (f_1(\mathbf{x}), \ldots, f_k(\mathbf{x}))$, PSO is extended to approximate the **Pareto front** — the set of non-dominated solutions.

Key modifications:
- Maintain an **external archive** of non-dominated solutions found so far
- $\mathbf{g}$ is selected from the archive (using crowding distance or hypergrid)
- $\mathbf{p}_i$ updated using Pareto dominance

Not relevant for TSP (single objective), but mentioned for completeness.

---

# 14. Advanced Topics (2020–2026 State of the Art)

## 14.1 Surrogate-Assisted PSO

When $f(\mathbf{x})$ is expensive to evaluate (e.g., a CFD simulation), PSO needs a **surrogate model** — a cheap approximation of $f$:
1. Evaluate $f$ at a small set of points → build surrogate (Gaussian process, neural network)
2. PSO optimizes the surrogate
3. Most promising candidates are evaluated with true $f$
4. Surrogate is updated

Reduces true evaluations from $N \times T$ to tens or hundreds.

## 14.2 PSO with Reinforcement Learning

Recent work (2022–2026) trains a policy network that *adapts PSO hyperparameters online* based on the current state of the swarm. The RL agent observes (diversity, stagnation count, iteration) and outputs $(\omega, c_1, c_2)$ at each step. Training is done offline on a distribution of benchmark problems.

## 14.3 Parallel PSO

Particles are independent between velocity/position updates — each particle's update requires only $\mathbf{p}_i$ and $\mathbf{g}$ (both read-only). This makes PSO **embarrassingly parallel**:
- Each particle runs on a separate CPU core
- Only synchronization point: update $\mathbf{g}$ (requires all-reduce over $\mathbf{p}_i$ values)

GPU implementation: all $N$ position/velocity updates as matrix operations in NumPy/JAX:
```python
# Vectorized PSO step (all N particles simultaneously)
r1 = np.random.uniform(0, 1, (N, D))
r2 = np.random.uniform(0, 1, (N, D))
V = omega * V + c1 * r1 * (P - X) + c2 * r2 * (G - X)
X = X + V
```

---

# 15. Implementation Realities

## 15.1 Complete Pseudocode (Standard PSO with LDIW)

```
Algorithm: PSO with Linear Decreasing Inertia Weight
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input:
  N         ← swarm size (e.g., 50)
  D         ← problem dimensionality
  T_max     ← maximum iterations (e.g., 1000)
  lb, ub    ← D-dimensional bounds vectors
  ω_max     ← 0.9
  ω_min     ← 0.4
  c1        ← 2.0
  c2        ← 2.0
  V_max     ← (ub - lb) / 2   [element-wise]
  f(·)      ← objective function to minimize

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INITIALIZATION:
  For i = 1 to N:
    x_i ← Uniform(lb, ub)           ← D-dim random position
    v_i ← Uniform(-V_max, V_max)    ← D-dim random velocity
    p_i ← x_i                       ← personal best = initial position
    f_p_i ← f(x_i)                  ← evaluate initial fitness

  g ← p_{argmin_i f_p_i}            ← global best = best initial pbest
  f_g ← min_i f_p_i

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MAIN LOOP:
  For t = 1 to T_max:

    ω ← ω_max - (ω_max - ω_min) × t / T_max    ← linear decay

    For i = 1 to N:

      r1 ← Uniform(0,1)^D          ← D independent random numbers
      r2 ← Uniform(0,1)^D          ← D independent random numbers

      ── Velocity Update ──
      v_i ← ω × v_i
           + c1 × r1 ⊙ (p_i - x_i)
           + c2 × r2 ⊙ (g  - x_i)

      ── Velocity Clamping ──
      v_i ← clip(v_i, -V_max, V_max)

      ── Position Update ──
      x_i ← x_i + v_i

      ── Boundary Handling ──
      Apply reflecting boundary on x_i

      ── Fitness Evaluation ──
      fitness ← f(x_i)

      ── Personal Best Update ──
      If fitness < f_p_i:
        p_i ← x_i
        f_p_i ← fitness

        ── Global Best Update ──
        If f_p_i < f_g:
          g ← p_i
          f_g ← f_p_i

    Record f_g in convergence history

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Output: g (best solution), f_g (best fitness), convergence history
```

## 15.2 Vectorized Python Implementation (NumPy)

```python
import numpy as np
from typing import Callable, Tuple

def pso_standard(
    f: Callable[[np.ndarray], float],
    D: int,
    lb: np.ndarray,
    ub: np.ndarray,
    N: int = 50,
    T_max: int = 1000,
    omega_max: float = 0.9,
    omega_min: float = 0.4,
    c1: float = 2.0,
    c2: float = 2.0,
    seed: int = None
) -> Tuple[np.ndarray, float, list]:
    """
    Standard PSO with linear decreasing inertia weight.
    Fully vectorized: all N particles updated simultaneously.
    
    Args:
        f:         objective function f(x) -> scalar (to minimize)
        D:         number of dimensions
        lb, ub:    lower/upper bounds, shape (D,)
        N:         swarm size
        T_max:     maximum iterations
        omega_max: initial inertia weight
        omega_min: final inertia weight
        c1:        cognitive coefficient
        c2:        social coefficient
        seed:      random seed for reproducibility
    
    Returns:
        (g_best, f_best, convergence_history)
    """
    if seed is not None:
        np.random.seed(seed)
    
    V_max = (ub - lb) / 2.0  # shape (D,)
    
    # ── Initialization ──────────────────────────────────────────────
    # X[i, d] = position of particle i in dimension d
    X = np.random.uniform(lb, ub, size=(N, D))
    V = np.random.uniform(-V_max, V_max, size=(N, D))
    
    # Evaluate all initial positions
    F = np.array([f(X[i]) for i in range(N)])
    
    P = X.copy()      # Personal bests (positions)
    F_P = F.copy()    # Personal best fitness values
    
    g_idx = np.argmin(F_P)
    G = P[g_idx].copy()   # Global best position
    F_G = F_P[g_idx]      # Global best fitness
    
    convergence = [F_G]
    
    # ── Main Loop ───────────────────────────────────────────────────
    for t in range(1, T_max + 1):
        # Linear inertia decay
        omega = omega_max - (omega_max - omega_min) * t / T_max
        
        # Random matrices: shape (N, D)
        R1 = np.random.uniform(0, 1, size=(N, D))
        R2 = np.random.uniform(0, 1, size=(N, D))
        
        # Velocity update (vectorized over all N particles)
        V = (omega * V
             + c1 * R1 * (P - X)        # cognitive term
             + c2 * R2 * (G - X))        # social term (G broadcast over N)
        
        # Velocity clamping
        V = np.clip(V, -V_max, V_max)
        
        # Position update
        X = X + V
        
        # Reflecting boundary handling
        for d in range(D):
            over  = X[:, d] > ub[d]
            under = X[:, d] < lb[d]
            X[over,  d] = 2 * ub[d] - X[over,  d]
            V[over,  d] = -V[over,  d]
            X[under, d] = 2 * lb[d] - X[under, d]
            V[under, d] = -V[under, d]
        
        # Evaluate fitness for all particles
        F = np.array([f(X[i]) for i in range(N)])
        
        # Personal best update
        improved = F < F_P
        P[improved] = X[improved].copy()
        F_P[improved] = F[improved]
        
        # Global best update
        best_idx = np.argmin(F_P)
        if F_P[best_idx] < F_G:
            G = P[best_idx].copy()
            F_G = F_P[best_idx]
        
        convergence.append(F_G)
    
    return G, F_G, convergence
```

## 15.3 Common Bugs and How to Avoid Them

| Bug | Symptom | Fix |
|-----|---------|-----|
| Forgetting to copy `X` into `P` at init | pbest = live position (changes each step) | Always `P = X.copy()` |
| Updating `G` before all `P` | `G` may be from previous iteration's `P` | Update all `P` first, then update `G` |
| No velocity clamping | Particles fly out of bounds immediately | Always clamp $V$ before updating $X$ |
| Using the same `r` for both terms | Reduced stochastic diversity | Sample `R1` and `R2` independently |
| Not resetting `v` after boundary | Particle bounces with full velocity | Apply boundary to `V` too |
| Minimizing when you should maximize | $f_g$ keeps increasing | Use $-f$ for maximization problems |

---

# ═══════════════════════════════════════════════
# PHASE 2 — MARPOC PROJECT ALIGNMENT
# ═══════════════════════════════════════════════


---

# 16. TSP — Formal Definition and Why It Demands a Specialized PSO

## 16.1 The Problem

**Project requirement (Section 1 of MARPOC)**: Provide a formal definition of TSP with NP-hardness context.

Given $n$ cities with coordinates $(x_k, y_k)_{k=1}^n$, find a permutation $\pi \in \mathcal{S}_n$ (a bijection from $\{1,\ldots,n\}$ to itself) minimizing:

$$f(\pi) = \sum_{k=1}^{n-1} d_{\pi_k, \pi_{k+1}} + d_{\pi_n, \pi_1}$$

where $d_{ij} = \text{round}\!\left(\sqrt{(x_i - x_j)^2 + (y_i - y_j)^2}\right)$ (TSPLIB EUC_2D convention).

**Search space**: $|\mathcal{S}_n| = (n-1)!/2$ (fixing start city and exploiting symmetry).

| $n$ | Number of tours |
|-----|----------------|
| 10  | $181{,}440$ |
| 51  | $\approx 1.5 \times 10^{62}$ |
| 130 | $\approx 10^{217}$ |

At $10^{12}$ tours evaluated per second, exhaustive search of eil51 would require $\approx 10^{50}$ years.

**NP-hardness**: Karp (1972) reduced Hamiltonian cycle (known NP-complete) to TSP in polynomial time, establishing TSP as NP-hard.

**Why exact methods fail**: The best exact solver (Concorde) uses cutting-plane LP + branch-and-bound. It can solve instances with $n \approx 85{,}900$ (Applegate et al. 2006) but requires hours of CPU time. For real-time or large-$n$ applications, metaheuristics are the only viable approach.

## 16.2 The Fundamental Adaptation Challenge

PSO's velocity and position update equations operate in $\mathbb{R}^D$ — a continuous vector space. TSP solutions live in $\mathcal{S}_n$ — a discrete permutation space.

The operations in $\mathbb{R}^D$ (addition, scalar multiplication) have **no direct meaning for permutations**:
- What is $(3,1,4,2) + (2,3,1,4)$? Undefined.
- What is $0.9 \times (3,1,4,2)$? Undefined.

**Three principled bridges** have been developed:

| Bridge | Core Idea | Year |
|--------|-----------|------|
| **SPV (Smallest Position Value)** | Continuous PSO unchanged; decode via argsort | Tasgetiren 2004 |
| **Swap-operator PSO** | Redefine $+$, $-$, $\times$ using permutation swaps | Clerc 2004, Wang 2003 |
| **Random Key PSO** | Each city assigned a real key; tour = sorted keys | Bean 1994 (adapted) |

---

# 17. PSO Adaptation 1 — The SPV Rule (Recommended for MARPOC)

## 17.1 Mathematical Definition

Each particle $i$ maintains a **continuous position vector** $\mathbf{x}_i \in \mathbb{R}^n$ (same dimensionality as the number of cities). The standard PSO update equations (§4, §5) apply **without modification**.

**SPV decoding**: Convert $\mathbf{x}_i$ to a tour $\pi_i$ by ranking the components:

$$\pi_i = \text{argsort}(\mathbf{x}_i)$$

Concretely: city $\pi_i^{(1)}$ is visited first (it has the smallest value in $\mathbf{x}_i$), city $\pi_i^{(2)}$ second (second smallest), and so on.

**Full worked example ($n = 6$)**:

```
Position:  x = [3.2,  1.1,  4.5,  0.7,  2.8,  1.9]
Index:          [0,    1,    2,    3,    4,    5  ]

Sorted by value (ascending):
  0.7 → index 3     (smallest → visit city 3 first)
  1.1 → index 1
  1.9 → index 5
  2.8 → index 4
  3.2 → index 0
  4.5 → index 2     (largest → visit city 2 last)

Tour π = [3, 1, 5, 4, 0, 2]
Tour as sequence: 3 → 1 → 5 → 4 → 0 → 2 → 3 (back to start)
```

## 17.2 Why SPV Works

**Invariance to scale and shift**: Adding a constant to all components of $\mathbf{x}_i$ does not change the argsort → same tour. This means the effective search space is not $\mathbb{R}^n$ but the **relative ordering** of components — a more compact representation.

**Surjectivity**: Every permutation in $\mathcal{S}_n$ is achievable by some $\mathbf{x}_i \in \mathbb{R}^n$. The SPV mapping is surjective (every tour can be decoded).

**Non-injectivity**: Many different $\mathbf{x}_i$ map to the same tour (any $\mathbf{x}_i'$ with the same relative ordering). This creates a many-to-one mapping, which reduces effective search space diversity. For large $n$, this can cause premature convergence.

**Fix**: Initialize positions in the unit interval $[0, n)^n$ to maximize initial diversity. Use opposition-based initialization (§9.4).

## 17.3 Step-by-Step Velocity Update for TSP (SPV)

```
Particle i at iteration t:
  x_i^t  = [3.2, 1.1, 4.5, 0.7, 2.8, 1.9]   → tour [3,1,5,4,0,2]  length=372
  p_i    = [0.5, 3.1, 2.2, 1.8, 4.0, 0.9]   → tour [0,5,3,1,2,4]  length=344  (pbest)
  g      = [2.1, 0.3, 3.5, 1.0, 4.8, 0.7]   → tour [1,5,3,0,2,4]  length=331  (gbest)
  v_i^t  = [0.1,-0.3, 0.5,-0.2, 0.4, 0.1]

  ω = 0.75, c1 = 2.0, c2 = 2.0
  r1 = [0.6, 0.3, 0.8, 0.5, 0.2, 0.7]
  r2 = [0.4, 0.9, 0.1, 0.6, 0.5, 0.3]

  Cognitive: c1 * r1 * (p_i - x_i^t)
    = 2.0 * [0.6, 0.3, 0.8, 0.5, 0.2, 0.7]
          * [0.5-3.2, 3.1-1.1, 2.2-4.5, 1.8-0.7, 4.0-2.8, 0.9-1.9]
    = 2.0 * [0.6, 0.3, 0.8, 0.5, 0.2, 0.7] * [-2.7, 2.0, -2.3, 1.1, 1.2, -1.0]
    = 2.0 * [-1.62, 0.60, -1.84, 0.55, 0.24, -0.70]
    = [-3.24, 1.20, -3.68, 1.10, 0.48, -1.40]

  Social: c2 * r2 * (g - x_i^t)
    = 2.0 * [0.4, 0.9, 0.1, 0.6, 0.5, 0.3]
          * [2.1-3.2, 0.3-1.1, 3.5-4.5, 1.0-0.7, 4.8-2.8, 0.7-1.9]
    = 2.0 * [0.4, 0.9, 0.1, 0.6, 0.5, 0.3] * [-1.1, -0.8, -1.0, 0.3, 2.0, -1.2]
    = 2.0 * [-0.44, -0.72, -0.10, 0.18, 1.00, -0.36]
    = [-0.88, -1.44, -0.20, 0.36, 2.00, -0.72]

  v_i^{t+1} = 0.75 * [0.1,-0.3, 0.5,-0.2, 0.4, 0.1]
                    + [-3.24, 1.20, -3.68, 1.10, 0.48, -1.40]
                    + [-0.88, -1.44, -0.20, 0.36, 2.00, -0.72]
            = [0.075, -0.225, 0.375, -0.15, 0.30, 0.075]
            + [-3.24, 1.20, -3.68, 1.10, 0.48, -1.40]
            + [-0.88, -1.44, -0.20, 0.36, 2.00, -0.72]
            = [-4.045, -0.465, -3.505, 1.310, 2.780, -2.045]

  x_i^{t+1} = [3.2, 1.1, 4.5, 0.7, 2.8, 1.9] + [-4.045, -0.465, -3.505, 1.310, 2.780, -2.045]
             = [-0.845, 0.635, 0.995, 2.010, 5.580, -0.145]

  New tour = argsort([-0.845, 0.635, 0.995, 2.010, 5.580, -0.145])
           = argsort: -0.845(0) < -0.145(5) < 0.635(1) < 0.995(2) < 2.010(3) < 5.580(4)
           → π = [0, 5, 1, 2, 3, 4]

  Evaluate f([0,5,1,2,3,4]) → compare with f(p_i) = 344 → update pbest if better
```

## 17.4 SPV — Complete Python Implementation

```python
import numpy as np
from typing import Optional

def spv_decode(position: np.ndarray) -> np.ndarray:
    """
    Smallest Position Value (SPV) rule.
    Convert continuous position vector to TSP tour (0-indexed permutation).
    
    Args:
        position: 1D array of n real values
    Returns:
        tour: 1D integer array, permutation of {0,...,n-1}
    
    Example:
        position = [3.2, 1.1, 4.5, 0.7]
        returns   = [3, 1, 0, 2]  (index of smallest to largest value)
    """
    return np.argsort(position)


def tour_length(tour: np.ndarray, dist: np.ndarray) -> int:
    """
    Compute total tour length using vectorized indexing.
    
    Args:
        tour: 1D integer array, permutation of {0,...,n-1}
        dist: n×n integer distance matrix
    Returns:
        Total distance (sum of all edges including return to start)
    
    Complexity: O(n)
    """
    # dist[tour[i], tour[i+1]] for i=0..n-2, then dist[tour[n-1], tour[0]]
    return int(dist[tour, np.roll(tour, -1)].sum())


class PSO_SPV:
    """
    PSO for TSP using Smallest Position Value encoding.
    
    Core algorithm:
      1. Each particle = real-valued vector in R^n
      2. Standard continuous PSO update (velocity + position)
      3. Decode position → tour via argsort (SPV)
      4. Evaluate tour length
    
    Optional: 2-opt local search applied to global best
    """
    
    def __init__(self, dist_matrix: np.ndarray,
                 N: int = 50,
                 T_max: int = 1000,
                 omega_max: float = 0.9,
                 omega_min: float = 0.4,
                 c1: float = 2.0,
                 c2: float = 2.0,
                 local_search: bool = True,
                 ls_freq: int = 10,
                 seed: Optional[int] = None):
        """
        Args:
            dist_matrix: n×n integer distance matrix (from TSPLIB parser)
            N:           swarm size
            T_max:       maximum iterations
            omega_max:   initial inertia weight
            omega_min:   final inertia weight
            c1:          cognitive coefficient
            c2:          social coefficient
            local_search: whether to apply 2-opt to global best
            ls_freq:     apply 2-opt every ls_freq iterations
            seed:        random seed
        """
        self.dist = dist_matrix
        self.n = dist_matrix.shape[0]   # number of cities
        self.N = N
        self.T_max = T_max
        self.omega_max = omega_max
        self.omega_min = omega_min
        self.c1 = c1
        self.c2 = c2
        self.local_search = local_search
        self.ls_freq = ls_freq
        
        if seed is not None:
            np.random.seed(seed)
        
        # Velocity bounds: V_max = n/2 per dimension
        self.V_max = self.n / 2.0
        
    def _evaluate_all(self, X: np.ndarray) -> tuple:
        """
        Decode all N particles and compute tour lengths.
        
        Args:
            X: (N, n) position matrix
        Returns:
            tours: (N, n) integer tour matrix
            lengths: (N,) tour length array
        """
        tours = np.argsort(X, axis=1)     # argsort each row → (N, n) tours
        # Compute all tour lengths at once
        lengths = np.array([
            int(self.dist[tours[i], np.roll(tours[i], -1)].sum())
            for i in range(self.N)
        ])
        return tours, lengths
    
    def _two_opt(self, tour: np.ndarray) -> np.ndarray:
        """
        2-opt local search: iteratively swap edges to reduce tour length.
        
        For every pair (i, j): test if reversing tour[i:j+1] reduces length.
        Delta = [d(π_{i-1}, π_j) + d(π_i, π_{j+1})]
              - [d(π_{i-1}, π_i) + d(π_j, π_{j+1})]
        If delta < 0: improvement → apply reversal.
        
        Complexity: O(n^2) per pass, O(n^3) worst-case total.
        """
        n = len(tour)
        tour = tour.copy()
        improved = True
        while improved:
            improved = False
            for i in range(n - 1):
                for j in range(i + 2, n):
                    # Edges being considered for removal:
                    a, b = tour[i], tour[i + 1]
                    c, d = tour[j], tour[(j + 1) % n]
                    # Skip wrap-around edge when i=0 and j=n-1
                    if i == 0 and j == n - 1:
                        continue
                    # Cost change if we reverse tour[i+1 : j+1]
                    delta = (self.dist[a, c] + self.dist[b, d]
                             - self.dist[a, b] - self.dist[c, d])
                    if delta < -1e-10:
                        tour[i + 1: j + 1] = tour[i + 1: j + 1][::-1]
                        improved = True
        return tour
    
    def run(self) -> dict:
        """
        Execute PSO with SPV encoding for TSP.
        
        Returns:
            dict with keys:
              'best_tour':    best permutation found (0-indexed)
              'best_length':  corresponding tour length
              'convergence':  list of best lengths per iteration
              'cpu_time':     wall-clock seconds
        """
        import time
        start = time.time()
        
        n, N, T_max = self.n, self.N, self.T_max
        dist = self.dist
        
        # ── Initialization ──────────────────────────────────────────
        # Positions: uniform in [0, n) for maximum SPV diversity
        X = np.random.uniform(0, n, size=(N, n))
        # Velocities: uniform in [-V_max, V_max]
        V = np.random.uniform(-self.V_max, self.V_max, size=(N, n))
        
        # Evaluate initial positions
        Tours, Lengths = self._evaluate_all(X)
        
        # Personal bests
        P = X.copy()             # (N, n) personal best positions
        P_tours = Tours.copy()   # (N, n) personal best tours
        P_lengths = Lengths.copy()  # (N,) personal best lengths
        
        # Global best
        g_idx = np.argmin(P_lengths)
        G = P[g_idx].copy()                # (n,) global best position
        G_tour = P_tours[g_idx].copy()     # (n,) global best tour
        G_length = P_lengths[g_idx]        # scalar
        
        convergence = [G_length]
        
        # ── Main Loop ───────────────────────────────────────────────
        for t in range(1, T_max + 1):
            
            # Inertia weight (linear decay)
            omega = self.omega_max - (self.omega_max - self.omega_min) * t / T_max
            
            # Random matrices, resampled each iteration per dimension per particle
            R1 = np.random.uniform(0, 1, size=(N, n))  # cognitive randomness
            R2 = np.random.uniform(0, 1, size=(N, n))  # social randomness
            
            # ── Velocity Update ──────────────────────────────────
            V = (omega * V                           # inertia
                 + self.c1 * R1 * (P - X)           # cognitive (P broadcast over N)
                 + self.c2 * R2 * (G - X))          # social (G broadcast over N)
            
            # Clamp velocity component-wise
            V = np.clip(V, -self.V_max, self.V_max)
            
            # ── Position Update ──────────────────────────────────
            X = X + V
            
            # ── Evaluate New Positions ────────────────────────────
            Tours, Lengths = self._evaluate_all(X)
            
            # ── Update Personal Bests ─────────────────────────────
            improved_mask = Lengths < P_lengths   # (N,) boolean
            P[improved_mask] = X[improved_mask].copy()
            P_tours[improved_mask] = Tours[improved_mask].copy()
            P_lengths[improved_mask] = Lengths[improved_mask]
            
            # ── Update Global Best ────────────────────────────────
            best_idx = np.argmin(P_lengths)
            if P_lengths[best_idx] < G_length:
                G = P[best_idx].copy()
                G_tour = P_tours[best_idx].copy()
                G_length = P_lengths[best_idx]
            
            # ── 2-opt Local Search on Global Best ────────────────
            if self.local_search and (t % self.ls_freq == 0):
                improved_tour = self._two_opt(G_tour)
                improved_length = tour_length(improved_tour, dist)
                if improved_length < G_length:
                    G_tour = improved_tour
                    G_length = improved_length
                    # Reconstruct a consistent position for G
                    # (assign rank values so argsort recovers G_tour)
                    G = np.zeros(n)
                    for rank, city in enumerate(G_tour):
                        G[city] = float(rank)
            
            convergence.append(G_length)
        
        return {
            'best_tour': G_tour,
            'best_length': G_length,
            'convergence': convergence,
            'cpu_time': time.time() - start,
        }
```

---

# 18. PSO Adaptation 2 — Discrete PSO with Swap Operators

## 18.1 Redefining PSO Arithmetic for Permutations

**Project requirement (Section 2 of MARPOC)**: Describe at least one variant.

The **Swap Sequence (SS)** approach (Wang et al. 2003, Clerc 2004) redefines all four PSO arithmetic operations in terms of permutation operations.

### Definition: Swap Operator

A **swap operator** $\text{SO}(i, j)$ applied to a permutation $\pi$ exchanges the elements at positions $i$ and $j$:
$$\text{SO}(i, j)[\pi] = \pi \text{ with } \pi_i \text{ and } \pi_j$ exchanged}$$

A **swap sequence** $SS = [\text{SO}(i_1, j_1), \text{SO}(i_2, j_2), \ldots, \text{SO}(i_k, j_k)]$ is an ordered list of swap operators applied sequentially.

### Redefined Operations

**Subtraction** ($\pi_2 \ominus \pi_1$): The minimum set of swaps that transforms $\pi_1$ into $\pi_2$.

**Addition** ($\pi \oplus SS$): Apply swap sequence $SS$ to permutation $\pi$.

**Scalar multiplication** ($\alpha \otimes SS$, $\alpha \in [0,1]$): Retain each swap in $SS$ with probability $\alpha$.

### PSO Update in Permutation Space

$$SS_i^{t+1} = (\omega \otimes SS_i^t) \oplus (c_1 r_1 \otimes (\pi_{pbest,i} \ominus \pi_i^t)) \oplus (c_2 r_2 \otimes (\pi_{gbest} \ominus \pi_i^t))$$

$$\pi_i^{t+1} = \pi_i^t \oplus SS_i^{t+1}$$

### Computing $\pi_2 \ominus \pi_1$ (Minimum Swap Sequence)

```python
def perm_subtract(pi2: np.ndarray, pi1: np.ndarray) -> list:
    """
    Find the minimal swap sequence SS such that apply(pi1, SS) = pi2.
    
    Algorithm: Selection sort on pi1 targeting pi2.
    Complexity: O(n) swaps, O(n) time.
    
    Returns: list of (pos_i, pos_j) tuples
    """
    n = len(pi1)
    pi1 = pi1.copy()
    # Position lookup: pos[city] = current index of 'city' in pi1
    pos = {city: idx for idx, city in enumerate(pi1)}
    
    swaps = []
    for i in range(n):
        city_needed = pi2[i]          # which city should be at position i
        j = pos[city_needed]           # where it currently is in pi1
        
        if i != j:
            # Perform swap (i, j) in pi1
            city_at_i = pi1[i]
            pi1[i], pi1[j] = pi1[j], pi1[i]
            pos[city_at_i] = j
            pos[city_needed] = i
            swaps.append((i, j))
    
    return swaps
```

## 18.2 Comparison: SPV vs. Discrete PSO

| Aspect | SPV PSO | Discrete PSO |
|--------|---------|--------------|
| Velocity space | Continuous $\mathbb{R}^n$ | Swap sequences |
| Theoretical basis | Full PSO theory applies | Heuristic adaptation |
| Parameter tuning | Standard $\omega$, $c_1$, $c_2$ | Same + probabilistic scalar mult. |
| Convergence guarantees | Yes (via constriction factor) | No formal guarantees |
| Implementation complexity | Low | Medium |
| Quality on TSP | Good with 2-opt | Comparable, sometimes better |
| Bonus for MARPOC | — | Yes (variant comparison) |

---

# 19. Experimental Protocol — MARPOC Section 4

## 19.1 Benchmark Instances and Optimal Values

```python
# Known optimal tour lengths from TSPLIB
TSPLIB_OPTIMA = {
    'eil51':    426,
    'berlin52': 7542,
    'eil76':    538,
    'pr76':     108159,
    'kroA100':  21282,
    'eil101':   629,
    'ch130':    6110,
    'ch150':    6528,
    'rat195':   2323,
    'kroA200':  29368,
}
```

## 19.2 Relative Error (ER) Computation

$$ER(\%) = \frac{f_{found} - f_{opt}}{f_{opt}} \times 100$$

```python
def relative_error(found: float, optimal: float) -> float:
    """Compute relative error percentage vs. known optimum/BKS."""
    return (found - optimal) / optimal * 100
```

## 19.3 Batch Runner — 30 Runs

```python
import numpy as np
import csv, json, time
from tsp_parser import parse_tsplib, tour_length

TSPLIB_OPTIMA = {
    'eil51': 426, 'berlin52': 7542, 'ch130': 6110, 'ch150': 6528,
    'kroA100': 21282, 'eil76': 538,
}

def run_experiment(tsp_path: str, n_runs: int = 30) -> dict:
    """
    Run PSO n_runs times on one TSP instance.
    Returns full statistics including all run fitness values (for boxplot).
    """
    data = parse_tsplib(tsp_path)
    dist = data['dist_matrix']
    name = data['name']
    n = data['dimension']
    opt = TSPLIB_OPTIMA.get(name.lower())
    
    all_lengths = []
    all_times = []
    best_convergence = None
    best_length = float('inf')
    
    print(f"\nRunning PSO on {name} (n={n}), {n_runs} runs...")
    
    for run in range(n_runs):
        solver = PSO_SPV(
            dist_matrix=dist,
            N=50, T_max=1000,
            omega_max=0.9, omega_min=0.4,
            c1=2.0, c2=2.0,
            local_search=True, ls_freq=10,
            seed=run  # deterministic seed per run
        )
        result = solver.run()
        
        all_lengths.append(result['best_length'])
        all_times.append(result['cpu_time'])
        
        if result['best_length'] < best_length:
            best_length = result['best_length']
            best_convergence = result['convergence']
        
        er_str = f"{relative_error(result['best_length'], opt):.2f}%" if opt else "N/A"
        print(f"  Run {run+1:2d}: length={result['best_length']:8d}  "
              f"ER={er_str}  time={result['cpu_time']:.2f}s")
    
    lengths = np.array(all_lengths)
    times = np.array(all_times)
    
    return {
        'instance':      name,
        'n_cities':      n,
        'n_runs':        n_runs,
        'optimal_bks':   opt,
        'best':          int(lengths.min()),
        'worst':         int(lengths.max()),
        'mean':          float(lengths.mean()),
        'std':           float(lengths.std()),
        'median':        float(np.median(lengths)),
        'er_best':       relative_error(lengths.min(), opt) if opt else None,
        'er_mean':       relative_error(lengths.mean(), opt) if opt else None,
        'mean_cpu':      float(times.mean()),
        'all_lengths':   all_lengths,
        'best_convergence': best_convergence,
    }
```

## 19.4 Boxplot Generator (30 Runs)

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_boxplots(results: list, save_path: str = 'results/boxplots.png'):
    """
    Generate boxplots of best-found values over 30 runs per instance.
    Required by MARPOC evaluation criteria.
    """
    fig, axes = plt.subplots(1, len(results),
                              figsize=(4 * len(results), 6),
                              sharey=False)
    if len(results) == 1:
        axes = [axes]
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(results)))
    
    for ax, result, color in zip(axes, results, colors):
        data = result['all_lengths']
        opt = result['optimal_bks']
        
        bp = ax.boxplot(
            data,
            patch_artist=True,
            notch=False,
            showfliers=True,
            boxprops=dict(facecolor=color, alpha=0.7),
            medianprops=dict(color='red', linewidth=2.5),
            whiskerprops=dict(linewidth=1.5),
            flierprops=dict(marker='o', markersize=4, alpha=0.5)
        )
        
        if opt:
            ax.axhline(y=opt, color='green', linestyle='--',
                       linewidth=2, label=f'BKS = {opt}')
            ax.legend(fontsize=8, loc='upper right')
        
        er = result['er_best']
        title = f"{result['instance']}\n(n={result['n_cities']})"
        if er is not None:
            title += f"\nBest ER: {er:.2f}%"
        
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_ylabel('Tour Length', fontsize=9)
        ax.set_xticks([1])
        ax.set_xticklabels(['PSO-Hybrid'])
        ax.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(
        'PSO-TSP: Distribution of Best Solutions Over 30 Independent Runs\n'
        'Dashed green line = known optimum (BKS)',
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Boxplot saved: {save_path}")
    plt.show()


def plot_convergence(results: list, save_path: str = 'results/convergence.png'):
    """
    Plot convergence curves: best tour length vs. iteration.
    One curve per instance (best run).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    for result, color in zip(results, colors):
        conv = result['best_convergence']
        ax.plot(conv, color=color, linewidth=2,
                label=f"{result['instance']} (best={result['best']})")
        
        if result['optimal_bks']:
            ax.axhline(y=result['optimal_bks'],
                       color=color, linestyle=':', alpha=0.6, linewidth=1)
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Best Tour Length Found', fontsize=12)
    ax.set_title('PSO-TSP Convergence Curves (Best Run per Instance)\n'
                 'Dotted lines = known optima', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Convergence plot saved: {save_path}")
    plt.show()
```

---

# 20. Results Template and Discussion

## 20.1 Results Table (Fill with Your Data)

| Instance | $n$ | BKS | Best | Mean | Std | ER$_{best}$% | ER$_{mean}$% | CPU(s) |
|----------|-----|-----|------|------|-----|------------|------------|--------|
| eil51    | 51  | 426 | ___ | ___ | ___ | ___ | ___ | ___ |
| berlin52 | 52  | 7542 | ___ | ___ | ___ | ___ | ___ | ___ |
| eil76    | 76  | 538 | ___ | ___ | ___ | ___ | ___ | ___ |
| kroA100  | 100 | 21282 | ___ | ___ | ___ | ___ | ___ | ___ |
| ch130    | 130 | 6110 | ___ | ___ | ___ | ___ | ___ | ___ |
| ch150    | 150 | 6528 | ___ | ___ | ___ | ___ | ___ | ___ |

## 20.2 Discussion Framework

**On convergence speed**: Typically PSO-SPV converges in the first 200–400 iterations (fast initial improvement from broad exploration), then plateaus. The 2-opt phase provides incremental improvement on this plateau.

**On ER% scaling with $n$**: As $n$ grows, the SPV mapping becomes increasingly lossy (more continuous positions map to the same tour — reduced effective diversity). Expect ER% to increase with $n$: eil51 (~1-2%) vs ch150 (~5-8%).

**On robustness (std)**: Low std indicates the algorithm consistently finds similar quality solutions. High std indicates sensitivity to initialization (a sign of local optima influence).

---

# 21. Conclusion and Perspectives — MARPOC Section 6

## 21.1 Your Ideas for Improvement (Required by MARPOC)

The following improvements are your roadmap for the conclusion section:

| Idea | What changes | Expected gain |
|------|--------------|---------------|
| **3-opt / Lin-Kernighan** | Replace 2-opt with 3-opt move | Better local optima |
| **Adaptive $\omega$** | Adjust per-particle based on fitness rank | Fewer premature convergences |
| **Population restart** | Re-initialize bottom 20% of particles when stagnating | Diversity recovery |
| **Parallel PSO** | Run all N particles on separate CPU cores | ×(number of cores) speedup |
| **QPSO variant** | Replace velocity update with quantum potential well | No velocity tuning needed |
| **Hybridization with ACO** | Pheromone matrix biases initial PSO positions | Better initialization |
| **RL-guided parameter adaptation** | Neural net adjusts $\omega$, $c_1$, $c_2$ online | Self-tuning |
| **Extension to CVRP** | Split algorithm on TSP tour → feasible routes | Directly from your implementation |

---

# 22. Mastery Checklists

## Section 3 (Core PSO Model)
- [ ] Can you define $\mathbf{x}_i^t$, $\mathbf{v}_i^t$, $\mathbf{p}_i$, $\mathbf{g}$ from memory with no reference?
- [ ] Can you explain why each of the three velocity terms exists without looking at notes?
- [ ] Can you derive the weighted centroid $\mathbf{G}_i = (c_1 \mathbf{p}_i + c_2 \mathbf{g})/(c_1+c_2)$?

## Section 6 (Inertia Weight)
- [ ] Can you write the LDIW formula and explain why $\omega_{max}=0.9$ and $\omega_{min}=0.4$?
- [ ] Can you explain what happens to convergence when $\omega > 1$?
- [ ] Can you implement adaptive $\omega$ per-particle from memory?

## Section 7 (Constriction Factor)
- [ ] Can you derive $\chi \approx 0.7298$ for $c_1 = c_2 = 2.05$?
- [ ] Can you explain why $\phi > 4$ is required for $\chi$ to be real?
- [ ] Can you state the stability condition for inertia-weight PSO?

## Section 17 (SPV for TSP)
- [ ] Can you apply SPV to any given position vector manually?
- [ ] Can you implement `spv_decode` and `tour_length` in Python from memory?
- [ ] Can you explain why SPV is non-injective and what consequence this has?

## Section 18 (Discrete PSO)
- [ ] Can you write the swap sequence update equation?
- [ ] Can you trace `perm_subtract([3,1,2,0], [0,1,2,3])` step by step?
- [ ] Can you explain the pros and cons of Discrete PSO vs. SPV PSO?

---

# 23. Complete Implementation Roadmap

## Day-by-Day Plan

### Days 1–2: Infrastructure
```
[ ] Create GitHub repo: github.com/your-username/pso-tsp (private)
[ ] pip install numpy matplotlib scipy
[ ] Implement tsp_parser.py → test on eil51.tsp (print n, first 5 distances)
[ ] Implement tour_length() → verify: eil51 optimal tour = 426
[ ] Implement two_opt() → verify: random tour improves by 10-20%
```

### Days 3–4: Core PSO
```
[ ] Implement PSO_SPV class (§17.4)
[ ] Run on eil51: target ER < 5% without local search
[ ] Add 2-opt hybrid → target ER < 2%
[ ] Implement PSO_Discrete (§18) for variant comparison (bonus)
```

### Days 5–6: Experiments
```
[ ] Run batch_runner on {eil51, berlin52, eil76, kroA100, ch130, ch150}
[ ] 30 runs each, seed=0..29 for reproducibility
[ ] Generate boxplots and convergence curves
[ ] Fill results table (§20.1)
```

### Days 7–8: Report Writing
```
[ ] Section 1: TSP formal definition (§16), NP-hardness table
[ ] Section 2: PSO history and variants (§13), pseudocode from §15.1
[ ] Section 3: SPV adaptation (§17.1-17.3), n=10 example (§17.1 worked example)
[ ] Section 4: Protocol (§19), parameter table
[ ] Section 5: Your results table + generated figures
[ ] Section 6: Pick 3 improvement ideas from §21.1 and develop them
```

### Days 9–10: Presentation
```
[ ] 8 slides (§4 of MARPOC guidelines)
[ ] Slide 3: PSO pseudocode (§15.1) + SPV diagram
[ ] Slide 5: results table for eil51, ch130, ch150
[ ] Slide 6: convergence curve figure
[ ] Demo: run PSO live on eil51 from terminal
```

---

*End of PSO Mastery Document*
*Total: Ground-Up Theory (Sections 1–15) + MARPOC Project (Sections 16–23)*
