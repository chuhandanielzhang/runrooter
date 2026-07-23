#!/usr/bin/env python3
"""Minimal IGMPv2 general querier.

Why: the PC <-> Jetson LCM link runs over UDP multicast (239.255.76.67).
The switch between them does IGMP snooping but nothing on the network acts
as an IGMP querier, so group memberships age out after a few minutes and
the switch silently stops forwarding PC -> Jetson multicast (motor_pwm_lcmt
etc.). Sending a periodic general query makes every host re-announce its
memberships, which keeps the snooping table fresh in both directions.

Must run as root (raw socket). Usage: igmp_querier.py [iface] [period_s]
"""
import socket
import struct
import sys
import time

IFACE = sys.argv[1] if len(sys.argv) > 1 else "enP8p1s0"
PERIOD_S = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
IPPROTO_IGMP = 2
SO_BINDTODEVICE = 25


def _checksum(data: bytes) -> int:
    if len(data) % 2:
        data += b"\x00"
    s = 0
    for i in range(0, len(data), 2):
        s += (data[i] << 8) + data[i + 1]
    s = (s >> 16) + (s & 0xFFFF)
    s += s >> 16
    return ~s & 0xFFFF


def build_general_query() -> bytes:
    # IGMPv2 membership query: type 0x11, max response time 10s (in 0.1s units)
    body = struct.pack("!BBH4s", 0x11, 100, 0, socket.inet_aton("0.0.0.0"))
    csum = _checksum(body)
    return struct.pack("!BBH4s", 0x11, 100, csum, socket.inet_aton("0.0.0.0"))


def main() -> None:
    pkt = build_general_query()
    sock = socket.socket(socket.AF_INET, socket.SOCK_RAW, IPPROTO_IGMP)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 1)
    sock.setsockopt(socket.SOL_SOCKET, SO_BINDTODEVICE, IFACE.encode() + b"\x00")
    print(f">>> IGMP querier on {IFACE}, general query every {PERIOD_S:.0f}s", flush=True)
    while True:
        try:
            sock.sendto(pkt, ("224.0.0.1", 0))
        except OSError as e:
            print(f"send failed: {e}", flush=True)
        time.sleep(PERIOD_S)


if __name__ == "__main__":
    main()
