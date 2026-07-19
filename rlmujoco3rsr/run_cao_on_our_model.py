"""Launch the current Mode1 controller on the 3RSR MuJoCo plant."""
import os
import sys
import time
import threading

from runtime_paths import CONTROLLER_DIR
sys.path.insert(0, CONTROLLER_DIR)
from modee.lcm_controller import ModeELCMController, ModeELCMConfig
from modee.core import ModeEConfig

cfg = ModeEConfig()
# MuJoCo plant mass differs from the measured hardware mass.
cfg.mass_kg = float(os.environ.get("CAO_MASS", "2.73"))
cfg.leg_l0_m = float(os.environ.get("CAO_L0", "0.42"))
cfg.hop_height_m = float(os.environ.get("CAO_HOP_H", "0.20"))
cfg.tau_cmd_max_nm = (float(os.environ.get("CAO_TAU", "9.0")),) * 3
cfg.mode_1d = os.environ.get("CAO_1D", "0") == "1"        # real robot runs 3D (mode_1d=False)
cfg.prop_base_thrust_ratio = 0.10
cfg.stance_use_props = True
# CAO_PURE=1 selects leg-only operation.
_pure_leg = os.environ.get("CAO_PURE", "0") == "1"
if _pure_leg:
    cfg.stance_use_props = False
    cfg.prop_base_thrust_ratio = 0.0
cfg.stance_kp_z = float(os.environ.get("CAO_KZ", "1100"))
cfg.stance_kd_z = float(os.environ.get("CAO_KDZ", "20"))
# optional attitude-loop overrides for the sim plant (defaults = core.py values)
if os.environ.get("CAO_ST_KR"):
    cfg.stance_kpp = float(os.environ["CAO_ST_KR"])
if os.environ.get("CAO_ST_KW"):
    cfg.stance_kpd = float(os.environ["CAO_ST_KW"])
# SLIP-style stance allocation A/B (1=axial/side split, 0=legacy z/xy+lever).
if os.environ.get("CAO_LEG_ALLOC"):
    cfg.stance_leg_frame_alloc = bool(int(os.environ["CAO_LEG_ALLOC"]))
if os.environ.get("CAO_FL_KR"):
    cfg.flight_kR = float(os.environ["CAO_FL_KR"])
if os.environ.get("CAO_FL_KW"):
    cfg.flight_kW = float(os.environ["CAO_FL_KW"])
if os.environ.get("CAO_PROP_BASE"):
    cfg.prop_base_thrust_ratio = float(os.environ["CAO_PROP_BASE"])
# Friction-cone modulation: total prop DOWNforce (N) in stance; leg fz raised
# by the same amount (CoM dynamics unchanged, contact normal force +F_dn).
if os.environ.get("CAO_DOWNFORCE"):
    cfg.stance_downforce_n = float(os.environ["CAO_DOWNFORCE"])
# Downforce window after touchdown (s); <=0 = whole stance. Default 0.06.
if os.environ.get("CAO_DOWNFORCE_TD"):
    cfg.stance_downforce_td_s = float(os.environ["CAO_DOWNFORCE_TD"])
# Controller-side friction coefficient (should match the plant floor mu).
if os.environ.get("CAO_MU"):
    cfg.stance_mu = float(os.environ["CAO_MU"])
# swing (flight foot tracking) gains — the closed-chain sim leg has more inertia
# and joint damping than the real delta leg, so it may need stiffer swing PD.
if os.environ.get("CAO_SW_KP"):
    cfg.swing_kp_xy = float(os.environ["CAO_SW_KP"])
if os.environ.get("CAO_SW_KD"):
    cfg.swing_kd_xy = float(os.environ["CAO_SW_KD"])

lcm_cfg = ModeELCMConfig()
# SIM ISOLATION (2026-07-09): NEVER use the default lcm_url here -- it is the
# REAL robot bus (port 7667, ttl=255 -> leaves this host and reaches the
# Jetson driver; running the sim actuated the real robot twice today).
# Port 7669 + ttl=0 stays on loopback and cannot collide with run_modee.py
# (7667) even on the same machine. LCM_DEFAULT_URL (sweep scripts) wins.
lcm_cfg.lcm_url = os.environ.get("LCM_DEFAULT_URL", "udpm://239.255.76.67:7669?ttl=0")
lcm_cfg.print_hz = 2.0
lcm_cfg.tau_out_max_nm = float(os.environ.get("CAO_TAU", "9.0"))
# our 3RSR retracts further than their hardware (q_lcm up to ~1.60 at the sim
# joint limit); their default safe_q_max=1.38 SAFE-latches mid-flight otherwise
lcm_cfg.safe_q_max = 1.62

print(
    f"ModeE on OUR model: m={cfg.mass_kg} l0={cfg.leg_l0_m} "
    f"hop={cfg.hop_height_m} mode=1 1d={cfg.mode_1d}"
)
ctl = ModeELCMController(modee_cfg=cfg, lcm_cfg=lcm_cfg)
t1 = threading.Thread(target=ctl.run_lcm_handler, daemon=False)
t2 = threading.Thread(target=ctl.run_controller, daemon=False)
t1.start(); t2.start()
try:
    while True:
        time.sleep(0.5)
except KeyboardInterrupt:
    pass
finally:
    ctl.running = False
    t2.join(timeout=2.0); t1.join(timeout=2.0)
