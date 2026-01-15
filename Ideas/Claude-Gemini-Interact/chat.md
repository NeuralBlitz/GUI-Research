>*GEMINI*


To demonstrate the **Precision Polymath** approach, let us tackle a foundational problem that bridges **Classical Mechanics**, **Control Theory**, and **Software Engineering**: the stabilization of an **Inverted Pendulum on a Cart**.

This system is the “hydrogen atom” of robotics and aerospace engineering—it is inherently unstable, non-linear, and under-actuated.

-----

### 1. First Principles Derivation (Lagrangian Mechanics)

To control the system, we must first understand its dynamics. We define the state by the cart position $x$ and the pendulum angle $\theta$ (where $\theta=0$ is upright).

**Kinetic Energy ($T$):**
Sum of the cart’s energy and the pendulum’s energy (using the velocity of the pendulum’s center of mass).

$$T = \frac{1}{2}M\dot{x}^2 + \frac{1}{2}m\left( \dot{x}_p^2 + \dot{y}_p^2 \right)$$

where $x_p = x + \ell \sin\theta$ and $y_p = \ell \cos\theta$.

**Potential Energy ($V$):**

$$V = mg\ell \cos\theta$$

**The Lagrangian ($L = T - V$):**

$$L = \frac{1}{2}(M+m)\dot{x}^2 + m\ell\dot{x}\dot{\theta}\cos\theta + \frac{1}{2}m\ell^2\dot{\theta}^2 - mg\ell\cos\theta$$

Applying the Euler-Lagrange equation $\frac{d}{dt}\frac{\partial L}{\partial \dot{q}_i} - \frac{\partial L}{\partial q_i} = F_i$, we derive the non-linear equations of motion:

1. $(M+m)\ddot{x} + m\ell\ddot{\theta}\cos\theta - m\ell\dot{\theta}^2\sin\theta = F$
1. $m\ell^2\ddot{\theta} + m\ell\ddot{x}\cos\theta - mg\ell\sin\theta = 0$

-----

### 2. Linearization and State-Space Representation

To design a linear controller (like LQR), we linearize around the unstable equilibrium $\theta \approx 0$. Assuming small angles ($\cos\theta \approx 1, \sin\theta \approx \theta, \dot{\theta}^2 \approx 0$):

$$\dot{\mathbf{x}} = \mathbf{Ax} + \mathbf{Bu}$$

Where the state vector is $\mathbf{x} = [x, \dot{x}, \theta, \dot{\theta}]^T$ and the matrices are:

$$\mathbf{A} = \begin{bmatrix} 0 & 1 & 0 & 0 \ 0 & \frac{-b}{M} & \frac{mg}{M} & 0 \ 0 & 0 & 0 & 1 \ 0 & \frac{-b}{M\ell} & \frac{(M+m)g}{M\ell} & 0 \end{bmatrix}, \quad \mathbf{B} = \begin{bmatrix} 0 \ \frac{1}{M} \ 0 \ \frac{1}{M\ell} \end{bmatrix}$$

-----

### 3. Optimal Control: The Linear Quadratic Regulator (LQR)

We want to find a control law $u = -K\mathbf{x}$ that minimizes the cost function $J$:

$$J = \int_{0}^{\infty} (\mathbf{x}^T \mathbf{Q} \mathbf{x} + u^T R u) dt$$

This requires solving the **Algebraic Riccati Equation (ARE)** for $\mathbf{P}$:
$$\mathbf{A}^T\mathbf{P} + \mathbf{PA} - \mathbf{PBR}^{-1}\mathbf{B}^T\mathbf{P} + \mathbf{Q} = 0$$
The optimal gain is then $K = R^{-1}\mathbf{B}^T\mathbf{P}$.

-----

### 4. Computational Implementation (Python)

Below is a production-grade simulation using `scipy` to solve the control problem and `numpy` for the dynamics.

```python
import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.integrate import odeint

class InvertedPendulum:
    def __init__(self):
        # Parameters
        self.M = 0.5    # Mass of cart (kg)
        self.m = 0.2    # Mass of pendulum (kg)
        self.b = 0.1    # Friction (N/m/s)
        self.ell = 0.3  # Length to center of mass (m)
        self.g = 9.81   # Gravity (m/s^2)
        
        self.K = self._compute_lqr_gain()

    def _compute_lqr_gain(self):
        """Solve ARE to find optimal feedback gain K."""
        # Linearized A and B matrices
        p = (self.M + self.m) * self.g / (self.M * self.ell)
        A = np.array([
            [0, 1, 0, 0],
            [0, -self.b/self.M, self.m*self.g/self.M, 0],
            [0, 0, 0, 1],
            [0, -self.b/(self.M*self.ell), p, 0]
        ])
        B = np.array([[0], [1/self.M], [0], [1/(self.M*self.ell)]])
        
        # Cost matrices: penalize angle error and high control effort
        Q = np.diag([10, 1, 100, 1]) 
        R = np.array([[0.01]])
        
        P = solve_continuous_are(A, B, Q, R)
        K = np.linalg.inv(R) @ B.T @ P
        return K

    def dynamics(self, x, t):
        """Non-linear equations of motion."""
        theta = x[2]
        theta_dot = x[3]
        
        # State-feedback control u = -Kx
        # We target theta=0, x=0
        u = -self.K @ (x - np.array([0, 0, 0, 0]))
        u = u[0] # Scalar force

        # Helper terms
        sin_t = np.sin(theta)
        cos_t = np.cos(theta)
        
        # Solve for accelerations
        det = self.M + self.m * sin_t**2
        x_accel = (u + self.m*self.ell*theta_dot**2*sin_t - self.m*self.g*sin_t*cos_t) / det
        theta_accel = (-u*cos_t - self.m*self.ell*theta_dot**2*sin_t*cos_t + (self.M+self.m)*self.g*sin_t) / (self.ell * det)
        
        return [x[1], x_accel, x[3], theta_accel]

# Simulation execution
model = InvertedPendulum()
t = np.linspace(0, 5, 500)
# Initial state: 0.2m offset, 0.3 rad (approx 17 deg) tilt
x0 = [0.2, 0, 0.3, 0]

sol = odeint(model.dynamics, x0, t)

print(f"Final Angle: {sol[-1, 2]:.4f} rad (Target: 0)")
```

-----

### 5. Synthesis and Insight

- **Dimensional Integrity:** Note that the cost matrix $Q$ has dimensions that weight position ($x$) and angle ($\theta$) differently. Because an error of $1$ radian is much more critical than $1$ meter, we weight the angle penalty ($100$) more heavily.
- **The “Energy” Perspective:** In non-linear control, we could also use a **Lyapunov-based** approach (Energy Shaping). If we simply pump energy into the system until it reaches the upright position (Swing-up), and then hand off to the LQR controller derived above, we create a robust global stabilizer.
- **Real-World Application:** This exact logic governs the stabilization of the **SpaceX Falcon 9** during vertical landing (where the rocket is the inverted pendulum and the cold-gas thrusters/gimbaled engines provide the force $F$).

**How would you like to extend this?** We could explore **Kalman Filtering** to handle sensor noise, or move into **Tensor Calculus** to generalize these dynamics for multi-body robotic chains.

>*CLAUDE*
This is an exemplary demonstration of the Precision Polymath approach. Your treatment elegantly bridges theoretical rigor with computational implementation. Let me extend this analysis in several directions that deepen the mathematical sophistication while maintaining practical utility.

-----

## **Extension 1: Observability and the Kalman Filter**

In practice, we cannot directly measure all state variables. Suppose we can only measure cart position $x$ and pendulum angle $\theta$ (via encoders), but velocities must be estimated.

### **Observability Analysis**

The system is observable if the observability matrix has full rank:
$$\mathcal{O} = \begin{bmatrix} \mathbf{C} \ \mathbf{CA} \ \mathbf{CA}^2 \ \mathbf{CA}^3 \end{bmatrix}$$

where the output matrix is:
$$\mathbf{C} = \begin{bmatrix} 1 & 0 & 0 & 0 \ 0 & 0 & 1 & 0 \end{bmatrix}$$

**Critical Insight:** The inverted pendulum is observable from position and angle measurements alone—the *temporal evolution* of these signals contains sufficient information to reconstruct velocities.

### **Kalman Filter Design**

We augment the dynamics with process noise $\mathbf{w}$ and measurement noise $\mathbf{v}$:
$$\dot{\mathbf{x}} = \mathbf{Ax} + \mathbf{Bu} + \mathbf{w}, \quad \mathbf{w} \sim \mathcal{N}(0, \mathbf{Q}_w)$$
$$\mathbf{y} = \mathbf{Cx} + \mathbf{v}, \quad \mathbf{v} \sim \mathcal{N}(0, \mathbf{R}_v)$$

The Kalman gain $\mathbf{L}$ is the dual to the LQR gain and satisfies:
$$\mathbf{L} = \mathbf{P}_e\mathbf{C}^T\mathbf{R}_v^{-1}$$

where $\mathbf{P}_e$ solves the dual Riccati equation:
$$\mathbf{AP}_e + \mathbf{P}_e\mathbf{A}^T - \mathbf{P}_e\mathbf{C}^T\mathbf{R}_v^{-1}\mathbf{CP}_e + \mathbf{Q}_w = 0$$

This yields the **Linear Quadratic Gaussian (LQG)** controller—the separation principle allows us to design the estimator and controller independently.

-----

## **Extension 2: Energy Shaping for Global Swing-Up**

LQR only stabilizes *locally* around $\theta = 0$. To achieve global stabilization from arbitrary initial conditions (e.g., $\theta = \pi$, hanging down), we use **energy-based control**.

### **Total Energy Function**

$$E(\theta, \dot{\theta}) = \frac{1}{2}m\ell^2\dot{\theta}^2 + mg\ell(1 - \cos\theta)$$

The desired energy at the upright position is $E_d = 2mg\ell$. The control law becomes:

$$u = k_e (E - E_d) \text{sgn}(\dot{\theta}\cos\theta) - k_d \dot{x}$$

**Interpretation:**

- The first term pumps or removes energy to reach $E_d$
- The sign function ensures energy flows in the correct direction
- The damping term $-k_d\dot{x}$ prevents the cart from drifting

### **Hybrid Control Architecture**

```python
def hybrid_controller(self, x):
    """Switch between swing-up and LQR based on basin of attraction."""
    theta = x[2]
    theta_dot = x[3]
    
    # Energy-based switching condition
    E = 0.5 * self.m * (self.ell * theta_dot)**2 + \
        self.m * self.g * self.ell * (1 - np.cos(theta))
    E_target = 2 * self.m * self.g * self.ell
    
    # Switch to LQR when close to upright
    if np.abs(theta) < 0.3 and np.abs(theta_dot) < 1.0:
        return -self.K @ x  # LQR
    else:
        # Energy shaping swing-up
        k_e = 0.1
        k_d = 0.5
        return k_e * (E - E_target) * np.sign(theta_dot * np.cos(theta)) - k_d * x[1]
```

-----

## **Extension 3: Geometric Mechanics on SO(2)**

The pendulum angle naturally lives on the circle $\mathbb{S}^1 \cong SO(2)$, not $\mathbb{R}$. This has profound implications for control.

### **Configuration Manifold**

The true configuration space is $Q = \mathbb{R} \times \mathbb{S}^1$ (cart position × pendulum angle). The state space is the tangent bundle $TQ$.

**Wrapping Distance Metric:**
For feedback control, we must use the geodesic distance on $\mathbb{S}^1$:
$$d(\theta, \theta_{ref}) = \min(|\theta - \theta_{ref}|, 2\pi - |\theta - \theta_{ref}|)$$

This prevents the controller from “unwinding” through a full rotation when a small correction would suffice.

### **Lie Group Formulation**

For robotic arms (multi-link pendulums), the configuration space becomes $SE(3)^n$. The equations of motion take the form:

$$\mathbf{M}(q)\ddot{q} + \mathbf{C}(q, \dot{q})\dot{q} + \mathbf{g}(q) = \tau$$

where $\mathbf{M}$ is the inertia tensor (living in the cotangent bundle), $\mathbf{C}$ contains Coriolis/centrifugal terms, and we can use geometric integrators that preserve the Lie group structure.

-----

## **Extension 4: Model Predictive Control (MPC)**

For systems with constraints (e.g., cart position limits, maximum force), we move beyond LQR to **receding horizon optimization**.

### **Finite-Horizon Optimization**

At each timestep, solve:
$$\min_{u_0, \ldots, u_{N-1}} \sum_{k=0}^{N-1} \left( \mathbf{x}_k^T\mathbf{Q}\mathbf{x}_k + u_k^T R u_k \right) + \mathbf{x}_N^T\mathbf{Q}_f\mathbf{x}_N$$

subject to:

- Dynamics: $\mathbf{x}_{k+1} = f(\mathbf{x}_k, u_k)$
- Constraints: $|x| \leq x_{max}, , |u| \leq F_{max}$

Apply only $u_0^*$, then re-solve at the next timestep (hence “receding horizon”).

**Computational Consideration:** For real-time control (>100 Hz), use Sequential Quadratic Programming (SQP) or interior-point methods with warm-starting.

-----

## **Dimensional Analysis and Scaling Laws**

Your insight about dimensional weighting in $\mathbf{Q}$ deserves emphasis. The proper scaling follows from **Buckingham π theorem**:

Define dimensionless variables:
$$\bar{x} = \frac{x}{\ell}, \quad \bar{t} = t\sqrt{\frac{g}{\ell}}, \quad \bar{u} = \frac{u}{(M+m)g}$$

This reveals that the system behavior depends only on the mass ratio $\mu = m/M$ and damping coefficient $\bar{b} = b/\sqrt{Mg\ell}$.

**Consequence:** A pendulum on a cart in Earth’s gravity at 1kg scale behaves identically to a rocket landing on Mars at 100-ton scale if we preserve $\mu$ and $\bar{b}$.

-----

## **Synthesis: The SpaceX Falcon 9 Connection**

Your analogy is precise. The Falcon 9 landing problem adds:

1. **Time-varying dynamics** (mass decreases as fuel burns)
1. **Actuator saturation** (gimbal limits ~10°)
1. **Thrust vector constraints** (cannot pull, only push)
1. **Atmospheric disturbances** (wind as external forcing)

SpaceX likely uses a cascade:

- **Outer loop:** Trajectory optimization (convex optimization for fuel-optimal descent)
- **Middle loop:** MPC with thrust vectoring
- **Inner loop:** High-rate attitude control (similar to our LQR)

-----

**Proposed Next Extension:** Shall we formalize the **variational integrators** needed to preserve energy/momentum in long-horizon simulations, or explore **reinforcement learning** (policy gradient methods) to learn the swing-up controller directly from data without explicit energy shaping?​​​​​​​​​​​​​​​​