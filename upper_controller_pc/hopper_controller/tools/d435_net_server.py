#!/usr/bin/env python3
"""D435 network frame server -- run ON the Jetson (camera host).

Streams aligned depth (zlib'd uint16) + color (JPEG) frames over TCP to
the terrain-gate viewer running on the PC:

  jetson$ python3 d435_net_server.py            # serves on :5556
  pc$     python3 tools/terrain_gate_live.py --net 192.168.1.100

Protocol (little endian):
  on connect : u32 header_len, JSON header {w,h,fx,fy,cx,cy,scale}
  per frame  : u32 depth_len, u32 jpeg_len, zlib(depth u16), jpeg(rgb)

Deps on the Jetson: pyrealsense2, numpy, cv2 (all stock / pip).
"""
import json
import socket
import struct
import threading
import zlib

import cv2
import numpy as np
import pyrealsense2 as rs

# Tried in order; USB2 (type 2.1) can't do 848x480@15, so fall back.
PROFILES = [(848, 480, 15), (640, 480, 15), (640, 480, 6)]
PORT = 5556


def main():
    pipe = rs.pipeline()
    prof = None
    for W, H, FPS in PROFILES:
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
        cfg.enable_stream(rs.stream.color, W, H, rs.format.rgb8, FPS)
        try:
            prof = pipe.start(cfg)
            print(f"[d435-net] streaming {W}x{H} @ {FPS} fps", flush=True)
            break
        except RuntimeError as e:
            print(f"[d435-net] {W}x{H}@{FPS} unavailable ({e})", flush=True)
    if prof is None:
        raise SystemExit("no workable depth+color profile")
    align = rs.align(rs.stream.depth)   # color -> depth frame
    scale = float(prof.get_device().first_depth_sensor().get_depth_scale())
    intr = (prof.get_stream(rs.stream.depth)
            .as_video_stream_profile().get_intrinsics())
    hdr = json.dumps(dict(w=W, h=H, fx=intr.fx, fy=intr.fy,
                          cx=intr.ppx, cy=intr.ppy, scale=scale)).encode()

    # Capture thread: keeps only the LATEST encoded frame; client threads
    # send at their own pace and can connect/disconnect freely.
    cond = threading.Condition()
    latest = {"seq": 0, "payload": None}

    def capture():
        while True:
            try:
                frames = align.process(
                    pipe.wait_for_frames(timeout_ms=2000))
            except RuntimeError as e:      # USB hiccup: keep trying
                print(f"[d435-net] frame timeout ({e})", flush=True)
                continue
            d = np.asanyarray(frames.get_depth_frame().get_data())
            c = frames.get_color_frame()
            rgb = (np.asanyarray(c.get_data()) if c
                   else np.zeros((H, W, 3), np.uint8))
            zd = zlib.compress(np.ascontiguousarray(d).tobytes(), 1)
            ok, jc = cv2.imencode(
                ".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 80])
            jb = jc.tobytes() if ok else b""
            payload = struct.pack("<II", len(zd), len(jb)) + zd + jb
            with cond:
                latest["seq"] += 1
                latest["payload"] = payload
                cond.notify_all()

    def serve(conn, addr):
        print(f"[d435-net] client {addr}", flush=True)
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        conn.settimeout(10)
        last = 0
        try:
            conn.sendall(struct.pack("<I", len(hdr)) + hdr)
            while True:
                with cond:
                    cond.wait_for(lambda: latest["seq"] > last, timeout=5)
                    if latest["seq"] <= last:
                        continue
                    last, payload = latest["seq"], latest["payload"]
                conn.sendall(payload)
        except (BrokenPipeError, ConnectionResetError, socket.timeout,
                TimeoutError, OSError):
            print(f"[d435-net] client {addr} disconnected", flush=True)
        finally:
            conn.close()

    threading.Thread(target=capture, daemon=True).start()
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", PORT))
    srv.listen(4)
    print(f"[d435-net] D435 up (scale {scale}); serving on :{PORT}",
          flush=True)
    while True:
        conn, addr = srv.accept()
        threading.Thread(target=serve, args=(conn, addr),
                         daemon=True).start()


if __name__ == "__main__":
    main()
