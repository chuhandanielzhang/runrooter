"""Launch Cao's ModeE controller configured for OUR 3RSR_package_2 plant geometry.
Pair with cao_fake_robot.py (which simulates our model over LCM)."""
import os
import sys
import time
import threading

from runtime_paths import CONTROLLER_DIR
sys.path.insert(0, CONTROLLER_DIR)
from modee.lcm_controller import ModeELCMController, ModeELCMConfig
from modee.core import ModeEConfig

import os
cfg = ModeEConfig()
# nominal leg length: our home is 0.369, but the mechanism reaches ~0.54;
# give the controller push-off stroke like the real robot (l0 0.464)
cfg.leg_l0_m = float(os.environ.get("CAO_L0", "0.42"))
cfg.hop_height_m = float(os.environ.get("CAO_HOP_H", "0.20"))
cfg.control_mode = int(os.environ.get("CAO_MODE", "3"))   # 1=pure leg, 2=decouple, 3=unified QP (real default)
cfg.tau_cmd_max_nm = (float(os.environ.get("CAO_TAU", "9.0")),) * 3
cfg.mode_1d = os.environ.get("CAO_1D", "0") == "1"        # real robot runs 3D (mode_1d=False)
cfg.prop_base_thrust_ratio = 0.10 if cfg.control_mode >= 2 else 0.0
cfg.stance_use_props = cfg.control_mode >= 2
# stance virtual-spring stiffness: default 1100 N/m is stiff (little visible
# compression). Lower it (e.g. 500) for SLIP-style touchdown buffering.
cfg.stance_kp_z = float(os.environ.get("CAO_KZ", "1100"))
cfg.stance_kd_z = float(os.environ.get("CAO_KDZ", "20"))
# optional attitude-loop overrides for the sim plant (defaults = core.py values)
if os.environ.get("CAO_ST_KR"):
    cfg.stance_kpp_x = cfg.stance_kpp_y = float(os.environ["CAO_ST_KR"])
if os.environ.get("CAO_ST_KW"):
    cfg.stance_kpd_x = cfg.stance_kpd_y = float(os.environ["CAO_ST_KW"])
if os.environ.get("CAO_FL_KR"):
    cfg.flight_kR_roll = cfg.flight_kR_pitch = float(os.environ["CAO_FL_KR"])
if os.environ.get("CAO_FL_KW"):
    cfg.flight_kW_roll = cfg.flight_kW_pitch = float(os.environ["CAO_FL_KW"])
if os.environ.get("CAO_PROP_BASE"):
    cfg.prop_base_thrust_ratio = float(os.environ["CAO_PROP_BASE"])

lcm_cfg = ModeELCMConfig()
lcm_cfg.print_hz = 2.0
lcm_cfg.tau_out_max_nm = float(os.environ.get("CAO_TAU", "9.0"))
# our 3RSR retracts further than their hardware (q_lcm up to ~1.60 at the sim
# joint limit); their default safe_q_max=1.38 SAFE-latches mid-flight otherwise
lcm_cfg.safe_q_max = 1.62

print(f"ModeE on OUR model: l0={cfg.leg_l0_m} mode={cfg.control_mode} 1d={cfg.mode_1d}")
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
