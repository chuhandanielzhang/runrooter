#!/usr/bin/env python3
"""Realtime DaMiao leg-motor monitor (READ-ONLY, never enables the motors).

Polls M1/M2/M3 on can0 with the DaMiao status-request frame
(arb 0x7FF, data [idL,idH,0xCC,...]) and prints position / velocity /
torque (~current) / temperatures / status at a fixed rate.

The 0xCC request only asks the driver board to send one feedback frame;
it does NOT enable the motor and produces no torque.

Usage (on the Jetson):
  python3 damiao_monitor.py               # can0, IDs 1,2,3, 10 Hz
  python3 damiao_monitor.py --hz 50
  python3 damiao_monitor.py --ids 1,2,3 --channel can0

NOTE: stop hopper-driver first if it is running, or its 500 Hz command
stream will interleave with this poller (harmless, but readings jump).
"""
import argparse
import socket
import struct
import sys
import time

# DM4310 register limits (read from the actual units 2026-07-18):
P_MAX, V_MAX, T_MAX = 12.5, 45.0, 10.0

STATUS = {
    0x0: "DISABLED",
    0x1: "ENABLED",
    0x8: "OVERVOLT",
    0x9: "UNDERVOLT",
    0xA: "OVERCURR",
    0xB: "MOS_OVERTEMP",
    0xC: "COIL_OVERTEMP",
    0xD: "LOST_COMM",
    0xE: "OVERLOAD",
}

CAN_FRAME_FMT = "=IB3x8s"


def u2f(x, lo, hi, bits):
    return x * (hi - lo) / ((1 << bits) - 1) + lo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", default="can0")
    ap.add_argument("--ids", default="1,2,3")
    ap.add_argument("--hz", type=float, default=10.0)
    args = ap.parse_args()

    ids = [int(x) for x in args.ids.split(",") if x.strip()]
    s = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    s.bind((args.channel,))
    s.settimeout(0.02)

    latest = {}  # motor_id -> (pos, vel, torq, t_mos, t_rotor, status, rx_time)
    period = 1.0 / max(1.0, args.hz)
    print(f"DaMiao monitor on {args.channel}, IDs {ids}, {args.hz:.0f} Hz "
          f"(read-only 0xCC poll; Ctrl+C to quit)")
    try:
        while True:
            t0 = time.time()
            for mid in ids:
                req = bytes([mid & 0xFF, (mid >> 8) & 0xFF, 0xCC, 0, 0, 0, 0, 0])
                s.send(struct.pack(CAN_FRAME_FMT, 0x7FF, 8, req))
            # drain replies until next tick
            while time.time() - t0 < period:
                try:
                    frame = s.recv(16)
                except socket.timeout:
                    continue
                can_id, dlc, data = struct.unpack(CAN_FRAME_FMT, frame)
                if dlc != 8:
                    continue
                mid = data[0] & 0x0F
                if mid not in ids:
                    continue
                st = data[0] >> 4
                p = u2f((data[1] << 8) | data[2], -P_MAX, P_MAX, 16)
                v = u2f((data[3] << 4) | (data[4] >> 4), -V_MAX, V_MAX, 12)
                tq = u2f(((data[4] & 0xF) << 8) | data[5], -T_MAX, T_MAX, 12)
                latest[mid] = (p, v, tq, data[6], data[7], st, time.time())

            line = []
            now = time.time()
            for mid in ids:
                if mid in latest and now - latest[mid][6] < 1.0:
                    p, v, tq, tm, tr, st, _ = latest[mid]
                    line.append(
                        f"M{mid} {STATUS.get(st, hex(st)):8s} "
                        f"q={p:+7.3f} qd={v:+6.2f} tau={tq:+6.2f}Nm "
                        f"Tmos={tm:3.0f}C Trot={tr:3.0f}C")
                else:
                    line.append(f"M{mid} --no-reply--"
                                + " " * 41)
            sys.stdout.write("\r" + " | ".join(line))
            sys.stdout.flush()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
