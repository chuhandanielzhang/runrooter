"""
MIT lecture-style condensed MPC (minimal export).

We intentionally keep only the GRF-only condensed-QP MPC here to match the
paper/lecture formulation and avoid multiple parallel MPC implementations.
"""

from .mit_mpc_condensed_grf import MITCondensedGRFMPC, MITCondensedGRFMPCConfig
from .mit_mpc_condensed_wrench import MITCondensedWrenchMPC, MITCondensedWrenchMPCConfig

__all__ = [
    "MITCondensedGRFMPC",
    "MITCondensedGRFMPCConfig",
    "MITCondensedWrenchMPC",
    "MITCondensedWrenchMPCConfig",
]

