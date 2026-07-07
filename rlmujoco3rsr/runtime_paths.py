"""Paths for rlmujoco3rsr inside robot_runtime (override with CASE_REPO env)."""
import os

_PKG = os.path.dirname(os.path.abspath(__file__))
_RUNTIME = os.path.dirname(_PKG)  # .../robot_runtime

# Default: sibling upper_controller_pc (same layout as hopperHFAcase2026)
CASE_ROOT = os.environ.get("CASE_REPO", os.path.join(_RUNTIME, "upper_controller_pc"))
CONTROLLER_DIR = os.path.join(CASE_ROOT, "hopper_controller")
LCM_TYPES_DIR = os.path.join(CASE_ROOT, "hopper_lcm_types", "lcm_types")
POLICIES_DIR = os.path.join(_PKG, "policies")
