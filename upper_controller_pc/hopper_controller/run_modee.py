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
        "Default internal: 20 Nm, default output: 30 Nm.",
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
        help="Max PWM pulse width (us). Default: 1300. For first bring-up, use 1100-1200.",
    )
    ap.add_argument(
        "--pwm-min",
        type=float,
        default=None,
        help="Min PWM pulse width (us). Default: 1000 (disarmed/idle).",
    )
    ap.add_argument(
        "--thrust-ratio",
        type=float,
        default=None,
        help="Baseline prop thrust ratio (0.0-1.0). Default: 0.10. For first bring-up, use 0.0-0.05.",
    )
    ap.add_argument(
        "--thrust-max-each",
        type=float,
        default=None,
        help="Per-arm max thrust cap passed to WBC (N). Default: 10.0. For first bring-up, use ~2-6.",
    )
    ap.add_argument(
        "--thrust-min-each",
        type=float,
        default=None,
        help="Per-arm MIN thrust lower bound in WBC-QP (N). Helps avoid props hitting pwm_min (stop/start) which causes wobble. Default: 0.0",
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
        help="Max commanded velocity from gamepad stick (m/s). Default: 0.8",
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
        help="Flight swing PERPENDICULAR (⊥ leg axis) position gain kp (N/m). Default: 300.0.",
    )
    ap.add_argument(
        "--swing-kd-xy",
        type=float,
        default=None,
        help="Flight swing PERPENDICULAR (⊥ leg axis) damping gain kd (N/(m/s)). Default: 0.0.",
    )
    ap.add_argument(
        "--swing-kp-z",
        type=float,
        default=None,
        help="Flight swing AXIAL (along leg axis) position gain kp (N/m). Default: 300.0.",
    )
    ap.add_argument(
        "--swing-kd-z",
        type=float,
        default=None,
        help="Flight swing AXIAL (along leg axis) damping gain kd (N/(m/s)). Default: 0.0.",
    )
    ap.add_argument(
        "--print-hz",
        type=float,
        default=None,
        help="Print frequency (Hz). <=0 means print every loop. Default: 0 (every loop).",
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
        help="Rate limit on commanded desired velocity (m/s^2). Helps keep hopping smooth. Default: 0.6",
    )
    ap.add_argument(
        "--force-arm",
        action="store_true",
        help="Force ARM on startup (for testing without gamepad). WARNING: Robot will start sending torques immediately!",
    )
    ap.add_argument(
        "--control-mode",
        type=int,
        default=None,
        choices=[2, 3],
        help="Control mode: 2=decouple (lstsq prop, stance+flight), "
        "3=mode2 + HLIP S2S foot placement (gain derived from measured Ts/z0 each hop, "
        "deadbeat-family; falls back to Raibert until the first stance is measured). "
        "Pure leg = mode 2 without pressing A (props never armed). Default: 2",
    )
    ap.add_argument(
        "--stance-use-props",
        action="store_true",
        help="Enable propellers in STANCE (helpful for balance in simulation). Default: disabled (leg-only stance).",
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
        help="Hopper4 thrust coefficient k_thrust (N per (pwm_delta)^2). Default: 1.47e-4. Only used when --use-hopper4-pwm is set.",
    )

    # Flight phase attitude control gains (separate roll/pitch)
    ap.add_argument(
        "--flight-kR-roll",
        type=float,
        default=None,
        help="Flight phase roll axis attitude error gain (proportional). Default: 20.0",
    )
    ap.add_argument(
        "--flight-kW-roll",
        type=float,
        default=None,
        help="Flight phase roll axis angular velocity damping gain (derivative). Default: 10.0",
    )
    ap.add_argument(
        "--flight-kR-pitch",
        type=float,
        default=None,
        help="Flight phase pitch axis attitude error gain (proportional). Default: 20.0",
    )
    ap.add_argument(
        "--flight-kW-pitch",
        type=float,
        default=None,
        help="Flight phase pitch axis angular velocity damping gain (derivative). Default: 10.0",
    )
    ap.add_argument(
        "--flight-tau-rp-max",
        type=float,
        default=None,
        help="Flight phase maximum roll/pitch torque limit (Nm). Default: 30.0",
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
        "--energy-kp",
        type=float,
        default=None,
        help="Energy compensation gain (Hopper4 Kp). Default: 7.0",
    )
    ap.add_argument(
        "--hop-height",
        type=float,
        default=None,
        help="Target hop height above l0 for energy calculation (m). Default: 0.15",
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
    if args.thrust_max_each is not None:
        modee_cfg.thrust_max_each_n = float(args.thrust_max_each)
    if args.thrust_min_each is not None:
        modee_cfg.wbc_thrust_min_each_n = float(args.thrust_min_each)
    if args.swing_kp_xy is not None:
        modee_cfg.swing_kp_xy = float(args.swing_kp_xy)
    if args.swing_kd_xy is not None:
        modee_cfg.swing_kd_xy = float(args.swing_kd_xy)
    if args.swing_kp_z is not None:
        modee_cfg.swing_kp_z = float(args.swing_kp_z)
    if args.swing_kd_z is not None:
        modee_cfg.swing_kd_z = float(args.swing_kd_z)

    if args.control_mode is not None:
        modee_cfg.control_mode = int(args.control_mode)

    if bool(getattr(args, "stance_use_props", False)):
        modee_cfg.stance_use_props = True

    if args.leg_model is not None:
        modee_cfg.leg_model = str(args.leg_model).strip().lower()

    if bool(getattr(args, "mode_2d", False)):
        modee_cfg.mode_1d = False
    elif bool(getattr(args, "mode_1d", False)):
        modee_cfg.mode_1d = True
    if bool(getattr(args, "no_energy_comp", False)):
        modee_cfg.use_energy_compensation = False
    if args.energy_kp is not None:
        modee_cfg.energy_comp_kp = float(args.energy_kp)
    if args.hop_height is not None:
        modee_cfg.hop_height_m = float(args.hop_height)

    # Velocity estimator behavior:
    # - ModeE core defaults to Hopper4-like XY behavior (stance: leg kinematics; flight: hold XY).
    # If you later want runtime switches again, re-introduce CLI flags here.
    
    # Flight phase attitude control gains (separate roll/pitch)
    if args.flight_kR_roll is not None:
        modee_cfg.flight_kR_roll = float(args.flight_kR_roll)
    if args.flight_kW_roll is not None:
        modee_cfg.flight_kW_roll = float(args.flight_kW_roll)
    if args.flight_kR_pitch is not None:
        modee_cfg.flight_kR_pitch = float(args.flight_kR_pitch)
    if args.flight_kW_pitch is not None:
        modee_cfg.flight_kW_pitch = float(args.flight_kW_pitch)
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

    _mode_names = {2: "DECOUPLE (lstsq prop; pure leg = don't press A)",
                   3: "MODE 3: DECOUPLE + HLIP S2S foot placement (measured Ts/z0, pole beta)"}
    print("=" * 58)
    print("HOPPER-AERO  |  Right stick → v_des  |  Y → log")
    print(f"  mode={modee_cfg.control_mode}: {_mode_names.get(modee_cfg.control_mode, '???')}")
    print(f"  max_cmd_vel={lcm_cfg.max_cmd_vel} m/s  kv={modee_cfg.flight_kv}  kr={modee_cfg.flight_kr}")
    print(f"  tau_cmd_max={modee_cfg.tau_cmd_max_nm} Nm  tau_out_max={lcm_cfg.tau_out_max_nm} Nm")
    print("=" * 58)

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



