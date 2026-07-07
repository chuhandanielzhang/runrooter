"""
MIT-style Model Predictive Control for Hopping Robot
=====================================================
QP-based MPC using Single Rigid Body dynamics.

Key Features:
- Short horizon planning (N~10-20 steps, dt~20ms)
- Contact force + thrust optimization
- Friction cone constraints
- Attitude stabilization

References:
- MIT Cheetah 3 MPC: "Highly Dynamic Quadruped Locomotion via Whole-Body Impulse Control"
- Convex MPC: "Dynamic Locomotion in the MIT Cheetah 3 Through Convex Model-Predictive Control"
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np

try:
    import osqp
    from scipy import sparse
    HAS_OSQP = True
except ImportError:
    HAS_OSQP = False

from .srb_dynamics import SRBDynamics


@dataclass
class MITMPCConfig:
    """Configuration for MIT-style MPC"""
    
    # Horizon
    dt: float = 0.02       # MPC timestep (s)
    N: int = 15            # Horizon length
    
    # Physical constraints
    mu: float = 1.0        # Friction coefficient
    fz_min: float = 5.0    # Min normal force in contact (N)
    fz_max: float = 200.0  # Max normal force (N)
    T_min: float = 0.0     # Min total thrust (N)
    T_max: float = 15.0    # Max total thrust (N)
    
    # State weights (for tracking error)
    w_pz: float = 50.0     # Height tracking
    w_vx: float = 10.0     # Velocity x
    w_vy: float = 10.0     # Velocity y
    w_vz: float = 20.0     # Velocity z
    w_roll: float = 100.0  # Roll angle
    w_pitch: float = 100.0 # Pitch angle
    w_wx: float = 1.0      # Angular velocity x
    w_wy: float = 1.0      # Angular velocity y
    
    # Input weights
    w_f: float = 1e-4      # Contact force regularization
    w_T: float = 1e-3      # Thrust regularization
    
    # Solver settings
    solver_max_iter: int = 500
    solver_eps: float = 1e-3


class MITMPC:
    """
    MIT-style Model Predictive Controller
    
    Uses sparse QP formulation with stacked state/input vectors.
    
    Decision variables: z = [x_1, ..., x_N, u_0, ..., u_{N-1}]
    
    min  Σ ||x_k - x_ref||²_Q + ||u_k||²_R
    s.t. x_{k+1} = A_k x_k + B_k u_k + c_k  (dynamics)
         friction cone, force bounds (contact only)
         thrust bounds
    """
    
    def __init__(self, cfg: MITMPCConfig = None):
        if not HAS_OSQP:
            raise ImportError("OSQP required for MPC. Install with: pip install osqp")
        
        self.cfg = cfg or MITMPCConfig()
        self.dynamics = SRBDynamics()
        
        self._solver = None
        self._initialized = False
        self._u_prev = None
    
    @property
    def nx(self) -> int:
        return self.dynamics.nx
    
    @property 
    def nu(self) -> int:
        return self.dynamics.nu
    
    def _build_qp(
        self,
        x0: np.ndarray,
        x_ref: np.ndarray,
        r_foot: np.ndarray,
        contact_schedule: np.ndarray,
    ):
        """Build QP matrices for the MPC problem"""
        cfg = self.cfg
        N = cfg.N
        nx, nu = self.nx, self.nu
        
        # Decision variables: z = [x_1, ..., x_N, u_0, ..., u_{N-1}]
        n_vars = N * nx + N * nu
        
        # --- Build P (quadratic cost) ---
        Q = np.diag([
            0.0, 0.0, cfg.w_pz,           # px, py, pz
            cfg.w_vx, cfg.w_vy, cfg.w_vz, # vx, vy, vz
            cfg.w_roll, cfg.w_pitch,      # roll, pitch
            cfg.w_wx, cfg.w_wy,           # wx, wy
        ])
        R = np.diag([cfg.w_f, cfg.w_f, cfg.w_f, cfg.w_T])
        
        P = np.zeros((n_vars, n_vars))
        # State costs
        for k in range(N):
            idx = k * nx
            P[idx:idx+nx, idx:idx+nx] = Q
        # Input costs
        for k in range(N):
            idx = N * nx + k * nu
            P[idx:idx+nu, idx:idx+nu] = R
        
        # --- Build q (linear cost) ---
        q = np.zeros(n_vars)
        # Reference tracking: -2 * Q * x_ref (from expanding ||x - x_ref||²_Q)
        for k in range(N):
            idx = k * nx
            q[idx:idx+nx] = -Q @ x_ref
        
        # --- Build equality constraints (dynamics) ---
        # A_eq @ z = b_eq
        # For each k: x_{k+1} = A_k x_k + B_k u_k + c_k
        # Rewrite: x_{k+1} - A_k x_k - B_k u_k = c_k
        # For k=0: x_1 - B_0 u_0 = A_0 x_0 + c_0
        
        n_eq = N * nx
        A_eq = np.zeros((n_eq, n_vars))
        b_eq = np.zeros(n_eq)
        
        # Linearize dynamics at each step
        x_lin = x0.copy()
        for k in range(N):
            in_contact = bool(contact_schedule[k])
            A_k, B_k = self.dynamics.linearize_A_B(x_lin, np.zeros(nu), r_foot, cfg.dt, in_contact)
            c_k = self.dynamics.get_gravity_term(cfg.dt)
            
            row = k * nx
            
            # x_{k+1} coefficient: I
            A_eq[row:row+nx, k*nx:(k+1)*nx] = np.eye(nx)
            
            # x_k coefficient: -A_k (for k > 0)
            if k > 0:
                A_eq[row:row+nx, (k-1)*nx:k*nx] = -A_k
            
            # u_k coefficient: -B_k
            u_idx = N * nx + k * nu
            A_eq[row:row+nx, u_idx:u_idx+nu] = -B_k
            
            # RHS
            if k == 0:
                b_eq[row:row+nx] = A_k @ x0 + c_k
            else:
                b_eq[row:row+nx] = c_k
            
            # Update linearization point
            x_lin = A_k @ x_lin + c_k
        
        # --- Build inequality constraints ---
        # For each timestep:
        #   - Friction cone (contact): fx ± μfz <= 0, fy ± μfz <= 0
        #   - Force bounds (contact): fz_min <= fz <= fz_max
        #   - Force zero (flight): fx=fy=fz=0
        #   - Thrust bounds: T_min <= T <= T_max
        
        ineq_rows = []
        l_vals = []
        u_vals = []
        
        for k in range(N):
            u_idx = N * nx + k * nu
            in_contact = bool(contact_schedule[k])
            
            if in_contact:
                # Friction cone (4 constraints)
                # fx - μfz <= 0
                row = np.zeros(n_vars)
                row[u_idx + 0] = 1.0
                row[u_idx + 2] = -cfg.mu
                ineq_rows.append(row)
                l_vals.append(-np.inf)
                u_vals.append(0.0)
                
                # -fx - μfz <= 0
                row = np.zeros(n_vars)
                row[u_idx + 0] = -1.0
                row[u_idx + 2] = -cfg.mu
                ineq_rows.append(row)
                l_vals.append(-np.inf)
                u_vals.append(0.0)
                
                # fy - μfz <= 0
                row = np.zeros(n_vars)
                row[u_idx + 1] = 1.0
                row[u_idx + 2] = -cfg.mu
                ineq_rows.append(row)
                l_vals.append(-np.inf)
                u_vals.append(0.0)
                
                # -fy - μfz <= 0
                row = np.zeros(n_vars)
                row[u_idx + 1] = -1.0
                row[u_idx + 2] = -cfg.mu
                ineq_rows.append(row)
                l_vals.append(-np.inf)
                u_vals.append(0.0)
                
                # fz bounds
                row = np.zeros(n_vars)
                row[u_idx + 2] = 1.0
                ineq_rows.append(row)
                l_vals.append(cfg.fz_min)
                u_vals.append(cfg.fz_max)
            else:
                # Flight: f = 0
                for i in range(3):
                    row = np.zeros(n_vars)
                    row[u_idx + i] = 1.0
                    ineq_rows.append(row)
                    l_vals.append(0.0)
                    u_vals.append(0.0)
            
            # Thrust bounds (always)
            row = np.zeros(n_vars)
            row[u_idx + 3] = 1.0
            ineq_rows.append(row)
            l_vals.append(cfg.T_min)
            u_vals.append(cfg.T_max)
        
        A_ineq = np.array(ineq_rows) if ineq_rows else np.zeros((0, n_vars))
        l_ineq = np.array(l_vals) if l_vals else np.array([])
        u_ineq = np.array(u_vals) if u_vals else np.array([])
        
        # --- Combine equality and inequality constraints ---
        # OSQP uses: l <= A @ z <= u
        # For equality: l = u = b
        A_full = np.vstack([A_eq, A_ineq])
        l_full = np.concatenate([b_eq, l_ineq])
        u_full = np.concatenate([b_eq, u_ineq])
        
        return P, q, A_full, l_full, u_full
    
    def solve(
        self,
        x0: np.ndarray,
        x_ref: np.ndarray,
        r_foot: np.ndarray,
        contact_schedule: np.ndarray,
        u_prev: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Solve MPC problem.
        
        Args:
            x0: Current state (10D)
            x_ref: Reference state (10D) - constant over horizon
            r_foot: Foot position relative to CoM (3D, world frame)
            contact_schedule: Contact flags for horizon (N,)
            u_prev: Previous input for warm-start (unused currently)
            
        Returns:
            Dictionary with solution
        """
        cfg = self.cfg
        N = cfg.N
        nx, nu = self.nx, self.nu
        
        x0 = np.asarray(x0).flatten()
        x_ref = np.asarray(x_ref).flatten()
        r_foot = np.asarray(r_foot).flatten()
        contact_schedule = np.asarray(contact_schedule).flatten()
        
        # Build QP
        P, q, A, l, u = self._build_qp(x0, x_ref, r_foot, contact_schedule)
        
        P_sparse = sparse.csc_matrix(P)
        A_sparse = sparse.csc_matrix(A)
        
        # Solve
        solver = osqp.OSQP()
        solver.setup(
            P=P_sparse,
            q=q,
            A=A_sparse,
            l=l,
            u=u,
            eps_abs=cfg.solver_eps,
            eps_rel=cfg.solver_eps,
            max_iter=cfg.solver_max_iter,
            verbose=False,
            warm_start=True,
        )
        
        result = solver.solve()
        
        if result.info.status in ['solved', 'solved inaccurate', 'solved_inaccurate']:
            z = result.x
            
            # Extract states and inputs
            x_pred = np.zeros((N + 1, nx))
            x_pred[0] = x0
            x_pred[1:] = z[:N*nx].reshape(N, nx)
            
            u_opt = z[N*nx:].reshape(N, nu)
            
            self._u_prev = u_opt[0].copy()
            
            return {
                'u_opt': u_opt,
                'x_pred': x_pred,
                'status': result.info.status,
                'cost': result.info.obj_val,
                'solve_time': result.info.solve_time,
            }
        else:
            return {
                'u_opt': np.zeros((N, nu)),
                'x_pred': np.zeros((N + 1, nx)),
                'status': result.info.status,
                'cost': np.inf,
                'solve_time': result.info.solve_time,
            }
    
    def get_first_input(self, sol: Dict[str, Any]) -> np.ndarray:
        """Extract first optimal input [fx, fy, fz, T]"""
        return sol['u_opt'][0].copy()
    
    def get_contact_force(self, sol: Dict[str, Any]) -> np.ndarray:
        """Extract first contact force [fx, fy, fz] (world frame)"""
        return sol['u_opt'][0, :3].copy()
    
    def get_thrust(self, sol: Dict[str, Any]) -> float:
        """Extract first total thrust"""
        return float(sol['u_opt'][0, 3])
