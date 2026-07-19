#!/usr/bin/env python3

import argparse
import threading
import time

from modee.lcm_controller import ModeELCMController, ModeELCMConfig
from modee.core import ModeEConfig


def main() -> None:
    ap = argparse.ArgumentParser(
        description="ModeE controller for real robot (PC-side, via LCM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default (internal solver uses nominal torque limits; output torque is safely capped)
  python3 run_modee.py

  # First-time bring-up (recommended): keep QP internal limits normal, but cap OUTPUT torque
  python3 run_modee.py --tau-out-max 1 --pwm-max 1100 --thrust-ratio 0.03 --thrust-max-each 4

  # Custom LCM URL (if not using default multicast)
  python3 run_modee.py --lcm-url "udpm://239.255.76.67:7667?ttl=1"

  # Print every loop (no throttling)
  python3 run_modee.py --print-hz 0

  # Verbose printing (5 Hz)
  python3 run_modee.py --print-hz 5
        """,
    )

    # Safety parameters (most important for bring-up)
    ap.add_argument(
        "--tau-max",
        type=float,
        default=None,
        help="Torque limit (Nm) for BOTH internal solver clip and final motor output "
        "(sets tau_cmd_max_nm and tau_out_max_nm unless --tau-out-max is also given). "
        "Default internal: 40 Nm, default output: 30 Nm.",
    )
    ap.add_argument(
        "--tau-sign",
        type=float,
        default=None,
        help="Motor torque wiring sign (+1 or -1). Default: +1.0. Set -1 if your motors push the wrong way.",
    )
    ap.add_argument(
        "--pwm-max",
        type=float,
        default=None,
        help="Max PWM pulse width (us). Default: 2000 (bidir full-forward).",
    )
    ap.add_argument(
        "--pwm-min",
        type=float,
        default=None,
        help="Min PWM pulse width (us). Default: 1000 (stop / bidir center).",
    )
    ap.add_argument(
        "--thrust-ratio",
        type=float,
        default=None,
        help="Flight baseline prop thrust ratio (0.0-1.0). Default: 0.01.",
    )
    ap.add_argument(
        "--stance-thrust-ratio",
        type=float,
        default=None,
        help="Stance-phase prop idle / collective ratio (0.0-1.0). Default: 0.12.",
    )
    ap.add_argument(
        "--thrust-max-each",
        type=float,
        default=None,
        help="Calibrated per-arm maximum thrust (N). Default: 10.0.",
    )
    ap.add_argument(
        "--prop-bidir",
        dest="prop_bidir",
        action="store_true",
        default=None,
        help="Enable bidirectional (3D) prop thrust (pwm<1000 = reverse). Default: off.",
    )
    ap.add_argument(
        "--no-prop-bidir",
        dest="prop_bidir",
        action="store_false",
        help="Disable bidirectional prop thrust (forward-only).",
    )
    ap.add_argument(
        "--thrust-min-each",
        type=float,
        default=None,
        help="Per-arm minimum forward thrust (N). Default: 0.1.",
    )
    # LCM settings
    ap.add_argument(
        "--lcm-url",
        type=str,
        default=None,
        help='LCM URL. Default: "udpm://239.255.76.67:7667?ttl=255"',
    )

    # Control tuning
    ap.add_argument(
        "--max-cmd-vel",
        type=float,
        default=None,
        help="Max commanded velocity from gamepad stick (m/s). Default: 0.15.",
    )
    ap.add_argument(
        "--tau-out-max",
        type=float,
        default=None,
        help="Output torque limit (Nm) applied in Python right before publishing (does NOT affect ModeECore/QP).",
    )
    ap.add_argument(
        "--tau-out-scale",
        type=float,
        default=None,
        help="Output torque scale applied in Python right before publishing (e.g. 0.1 for bring-up). Default: 1.0",
    )
    ap.add_argument(
        "--swing-kp-xy",
        type=float,
        default=None,
        help="Flight swing perpendicular position gain kp (N/m). Default: 60.0.",
    )
    ap.add_argument(
        "--swing-kd-xy",
        type=float,
        default=None,
        help="Flight swing perpendicular damping gain kd (N/(m/s)). Default: 1.0.",
    )
    ap.add_argument(
        "--swing-kp-z",
        type=float,
        default=None,
        help="Flight swing axial position gain kp (N/m). Default: 1000.0.",
    )
    ap.add_argument(
        "--swing-kd-z",
        type=float,
        default=None,
        help="Flight swing axial damping gain kd (N/(m/s)). Default: 20.0.",
    )
    ap.add_argument(
        "--print-hz",
        type=float,
        default=None,
        help="Print frequency (Hz). <=0 means every loop. Default: 5.",
    )
    ap.add_argument(
        "--demo-negx",
        action="store_true",
        help="Enable demo mode: override desired velocity to a constant (-X) value (smooth ramp).",
    )
    ap.add_argument(
        "--demo-vx",
        type=float,
        default=None,
        help="Demo desired vx (m/s). Used only with --demo-negx. Default: -0.20",
    )
    ap.add_argument(
        "--demo-vy",
        type=float,
        default=None,
        help="Demo desired vy (m/s). Used only with --demo-negx. Default: 0.0",
    )
    ap.add_argument(
        "--cmd-dv-max",
        type=float,
        default=None,
        help="Rate limit on commanded desired velocity (m/s^2). Default: 0 (disabled).",
    )
    ap.add_argument(
        "--force-arm",
        action="store_true",
        help="Force ARM on startup (for testing without gamepad). WARNING: Robot will start sending torques immediately!",
    )
    ap.add_argument(
        "--leg-model",
        type=str,
        default=None,
        help='Leg kinematics backend: "delta" (real robot) or "serial" (MuJoCo hopper_serial.xml).',
    )
    ap.add_argument(
        "--use-hopper4-pwm",
        action="store_true",
        help="Use Hopper4-style k_thrust square-root PWM mapping instead of MotorTableModel lookup table.",
    )
    ap.add_argument(
        "--prop-k-thrust",
        type=float,
        default=None,
        help="Calibrated thrust coefficient in N/(PWM delta)^2. Default: 1.24e-5.",
    )

    # Flight phase attitude control gains (shared roll/pitch)
    ap.add_argument(
        "--flight-kR",
        type=float,
        default=None,
        help="Flight phase attitude error gain (roll and pitch). Default: 40.0.",
    )
    ap.add_argument(
        "--flight-kW",
        type=float,
        default=None,
        help="Flight phase angular-rate damping gain (roll and pitch). Default: 6.0.",
    )
    ap.add_argument(
        "--flight-tau-rp-max",
        type=float,
        default=None,
        help="Flight phase maximum roll/pitch torque limit (Nm). Default: 100.0.",
    )

    mode_dim_group = ap.add_mutually_exclusive_group()
    mode_dim_group.add_argument(
        "--1d-mode",
        action="store_true",
        dest="mode_1d",
        help="Enable 1D vertical hopping mode (no horizontal movement, foot stays directly below). "
             "Use this to verify height convergence before adding horizontal control.",
    )
    mode_dim_group.add_argument(
        "--2d-mode",
        action="store_true",
        dest="mode_2d",
        help="Enable 2D hopping mode by forcing mode_1d=False (keeps your tuned 1D defaults unchanged in core.py).",
    )
    ap.add_argument(
        "--no-energy-comp",
        action="store_true",
        help="Disable Hopper4-style energy compensation in PUSH phase (default: enabled).",
    )
    ap.add_argument(
        "--hop-height",
        type=float,
        default=None,
        help="Target hop height above l0 for energy calculation (m). Default: 0.07.",
    )

    args = ap.parse_args()

    # Build ModeEConfig with overrides
    modee_cfg = ModeEConfig()
    if args.tau_max is not None:
        modee_cfg.tau_cmd_max_nm = (float(args.tau_max),) * 3
    if args.tau_sign is not None:
        modee_cfg.tau_cmd_sign = (float(args.tau_sign),) * 3
    if args.pwm_max is not None:
        modee_cfg.pwm_max_us = float(args.pwm_max)
    if args.pwm_min is not None:
        modee_cfg.pwm_min_us = float(args.pwm_min)
    if args.thrust_ratio is not None:
        modee_cfg.prop_base_thrust_ratio = float(args.thrust_ratio)
    if getattr(args, "stance_thrust_ratio", None) is not None:
        modee_cfg.prop_stance_base_thrust_ratio = float(args.stance_thrust_ratio)
    if args.thrust_max_each is not None:
        modee_cfg.thrust_max_each_n = float(args.thrust_max_each)
    if args.thrust_min_each is not None:
        modee_cfg.wbc_thrust_min_each_n = float(args.thrust_min_each)
    if getattr(args, "prop_bidir", None) is not None:
        modee_cfg.prop_bidir = bool(args.prop_bidir)
        if bool(args.prop_bidir) and str(modee_cfg.prop_flight_reverse) == "fwd":
            modee_cfg.prop_flight_reverse = "auto"
    if args.swing_kp_xy is not None:
        modee_cfg.swing_kp_xy = float(args.swing_kp_xy)
    if args.swing_kd_xy is not None:
        modee_cfg.swing_kd_xy = float(args.swing_kd_xy)
    if args.swing_kp_z is not None:
        modee_cfg.swing_kp_z = float(args.swing_kp_z)
    if args.swing_kd_z is not None:
        modee_cfg.swing_kd_z = float(args.swing_kd_z)

    if args.leg_model is not None:
        modee_cfg.leg_model = str(args.leg_model).strip().lower()

    if bool(getattr(args, "mode_2d", False)):
        modee_cfg.mode_1d = False
    elif bool(getattr(args, "mode_1d", False)):
        modee_cfg.mode_1d = True
    if bool(getattr(args, "no_energy_comp", False)):
        modee_cfg.use_energy_compensation = False
    if args.hop_height is not None:
        modee_cfg.hop_height_m = float(args.hop_height)

    # Velocity estimator behavior:
    # - ModeE core defaults to Hopper4-like XY behavior (stance: leg kinematics; flight: hold XY).
    # If you later want runtime switches again, re-introduce CLI flags here.
    
    # Flight phase attitude control gains (shared roll/pitch)
    if args.flight_kR is not None:
        modee_cfg.flight_kR = float(args.flight_kR)
    if args.flight_kW is not None:
        modee_cfg.flight_kW = float(args.flight_kW)
    if args.flight_tau_rp_max is not None:
        modee_cfg.flight_tau_rp_max = float(args.flight_tau_rp_max)
    
    # Propeller PWM mapping method
    if args.use_hopper4_pwm:
        modee_cfg.use_hopper4_pwm_mapping = True
    if args.prop_k_thrust is not None:
        modee_cfg.prop_k_thrust = float(args.prop_k_thrust)
    
    # Build ModeELCMConfig with overrides
    lcm_cfg = ModeELCMConfig()
    if args.lcm_url is not None:
        lcm_cfg.lcm_url = str(args.lcm_url)
    if args.max_cmd_vel is not None:
        lcm_cfg.max_cmd_vel = float(args.max_cmd_vel)
    if args.print_hz is not None:
        lcm_cfg.print_hz = float(args.print_hz)
    if args.tau_out_max is not None:
        lcm_cfg.tau_out_max_nm = float(args.tau_out_max)
    elif args.tau_max is not None:
        # --tau-max also caps final motor output unless --tau-out-max overrides it.
        lcm_cfg.tau_out_max_nm = float(args.tau_max)
    if args.tau_out_scale is not None:
        lcm_cfg.tau_out_scale = float(args.tau_out_scale)

    if args.cmd_dv_max is not None:
        lcm_cfg.cmd_dv_max_mps2 = float(args.cmd_dv_max)

    if bool(getattr(args, "demo_negx", False)):
        lcm_cfg.demo_enable = True
        if args.demo_vx is not None:
            lcm_cfg.demo_vx_mps = float(args.demo_vx)
        if args.demo_vy is not None:
            lcm_cfg.demo_vy_mps = float(args.demo_vy)

    ctl = ModeELCMController(modee_cfg=modee_cfg, lcm_cfg=lcm_cfg)
    
    # Note: --force-arm flag is deprecated (always sends commands now)
    if args.force_arm:
        pass

    # Non-daemon threads so we can shutdown cleanly (close log file, flush buffers, etc.)
    lcm_thread = threading.Thread(target=ctl.run_lcm_handler, daemon=False)
    ctrl_thread = threading.Thread(target=ctl.run_controller, daemon=False)
    lcm_thread.start()
    ctrl_thread.start()

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        ctl.running = False
        # Join to allow ModeELCMController.run_controller() to close the CSV log cleanly.
        try:
            ctrl_thread.join(timeout=2.0)
        except Exception:
            pass
        try:
            lcm_thread.join(timeout=2.0)
        except Exception:
            pass


if __name__ == "__main__":
    main()



