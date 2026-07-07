"""
Single Rigid Body (SRB) Dynamics for MPC
=========================================
MIT-style simplified dynamics model for hopping robot.

State: x = [p_x, p_y, p_z, v_x, v_y, v_z, roll, pitch, omega_x, omega_y]  (10D)
Input: u = [f_x, f_y, f_z, T]  (contact force + total thrust, 4D)

Dynamics (continuous):
    dp/dt = v
    dv/dt = (f + T*z_body) / m + g
    dθ/dt = ω  (small angle approx for roll/pitch)
    dω/dt = I^{-1} * (r × f + τ_prop)

References:
- MIT Cheetah 3: "Highly Dynamic Quadruped Locomotion via Whole-Body Impulse Control"
- MIT Mini Cheetah: "MIT Cheetah 3: Design and Control of a Robust, Dynamic Quadruped Robot"
"""

from dataclasses import dataclass
import numpy as np
from typing import Tuple


@dataclass
class SRBDynamics:
    """
    Single Rigid Body dynamics for hopping robot.
    
    Simplified assumptions:
    - Small roll/pitch angles (linearized attitude)
    - Yaw is uncontrolled (tri-rotor)
    - Contact point at foot location
    - Thrust along body z-axis
    """
    
    mass: float = 2.73       # Robot mass (kg)
    gravity: float = 9.81    # Gravity (m/s^2)
    I_body: np.ndarray = None  # Body inertia diagonal [Ixx, Iyy, Izz]
    
    def __post_init__(self):
        if self.I_body is None:
            # Default inertia for ~3kg hopper
            self.I_body = np.array([0.02, 0.02, 0.01])
    
    @property
    def nx(self) -> int:
        """State dimension"""
        return 10  # [px, py, pz, vx, vy, vz, roll, pitch, wx, wy]
    
    @property
    def nu(self) -> int:
        """Input dimension"""
        return 4  # [fx, fy, fz, T]
    
    def continuous_dynamics(
        self,
        x: np.ndarray,
        u: np.ndarray,
        r_foot: np.ndarray,
        in_contact: bool = True,
    ) -> np.ndarray:
        """
        Compute state derivative dx/dt.
        
        Args:
            x: State [px, py, pz, vx, vy, vz, roll, pitch, wx, wy]
            u: Input [fx, fy, fz, T] - contact force (world) + thrust
            r_foot: Foot position relative to CoM (world frame)
            in_contact: Whether foot is in contact
            
        Returns:
            dx: State derivative
        """
        # Unpack state
        px, py, pz = x[0], x[1], x[2]
        vx, vy, vz = x[3], x[4], x[5]
        roll, pitch = x[6], x[7]
        wx, wy = x[8], x[9]
        
        # Unpack input
        fx, fy, fz = u[0], u[1], u[2]
        T = u[3]
        
        # Body z-axis in world (small angle approx)
        z_body = np.array([
            -np.sin(pitch),
            np.sin(roll) * np.cos(pitch),
            np.cos(roll) * np.cos(pitch),
        ])
        
        # Gravity
        g_vec = np.array([0.0, 0.0, -self.gravity])
        
        # Linear acceleration
        f_contact = np.array([fx, fy, fz]) if in_contact else np.zeros(3)
        a = (f_contact + T * z_body) / self.mass + g_vec
        
        # Angular acceleration (moment from contact force)
        if in_contact:
            r = np.asarray(r_foot).flatten()
            tau_contact = np.cross(r, f_contact)
        else:
            tau_contact = np.zeros(3)
        
        # Simplified: ignore propeller torque contribution (small for tri-rotor)
        alpha_x = tau_contact[0] / self.I_body[0]
        alpha_y = tau_contact[1] / self.I_body[1]
        
        # State derivative
        dx = np.array([
            vx, vy, vz,           # dp/dt = v
            a[0], a[1], a[2],     # dv/dt = a
            wx, wy,               # dθ/dt = ω
            alpha_x, alpha_y,     # dω/dt = I^{-1}τ
        ])
        
        return dx
    
    def discretize(
        self,
        x: np.ndarray,
        u: np.ndarray,
        r_foot: np.ndarray,
        dt: float,
        in_contact: bool = True,
    ) -> np.ndarray:
        """
        Euler discretization: x_{k+1} = x_k + dt * f(x_k, u_k)
        """
        dx = self.continuous_dynamics(x, u, r_foot, in_contact)
        return x + dt * dx
    
    def linearize_A_B(
        self,
        x: np.ndarray,
        u: np.ndarray,
        r_foot: np.ndarray,
        dt: float,
        in_contact: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Linearize dynamics and return discrete A, B matrices.
        
        x_{k+1} ≈ A @ x_k + B @ u_k + c
        
        For MPC, we use the Jacobian around the current operating point.
        """
        nx, nu = self.nx, self.nu
        A = np.eye(nx)
        B = np.zeros((nx, nu))
        
        # Position derivatives
        A[0, 3] = dt  # px <- vx
        A[1, 4] = dt  # py <- vy
        A[2, 5] = dt  # pz <- vz
        
        # Velocity derivatives (from contact force)
        if in_contact:
            B[3, 0] = dt / self.mass  # vx <- fx
            B[4, 1] = dt / self.mass  # vy <- fy
            B[5, 2] = dt / self.mass  # vz <- fz
        
        # Velocity from thrust (small angle approx)
        roll, pitch = x[6], x[7]
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        
        B[3, 3] = -sp * dt / self.mass              # vx <- T
        B[4, 3] = sr * cp * dt / self.mass          # vy <- T
        B[5, 3] = cr * cp * dt / self.mass          # vz <- T
        
        # Attitude derivatives
        A[6, 8] = dt  # roll <- wx
        A[7, 9] = dt  # pitch <- wy
        
        # Angular velocity from contact torque
        if in_contact:
            r = np.asarray(r_foot).flatten()
            # τ = r × f, so ∂τ/∂f = skew(r)
            # For [fx, fy, fz] -> [τx, τy, τz]:
            #   τx = ry*fz - rz*fy
            #   τy = rz*fx - rx*fz
            # αx = τx/Ixx, αy = τy/Iyy
            B[8, 1] = -r[2] * dt / self.I_body[0]  # wx <- fy (via τx)
            B[8, 2] = r[1] * dt / self.I_body[0]   # wx <- fz (via τx)
            B[9, 0] = r[2] * dt / self.I_body[1]   # wy <- fx (via τy)
            B[9, 2] = -r[0] * dt / self.I_body[1]  # wy <- fz (via τy)
        
        return A, B
    
    def get_gravity_term(self, dt: float) -> np.ndarray:
        """Get gravity contribution to state update (constant term)"""
        c = np.zeros(self.nx)
        c[5] = -self.gravity * dt  # vz += -g*dt
        return c

