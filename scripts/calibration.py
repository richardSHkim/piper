#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Estimate alignment parameters (R, s) for PIKA Sense -> Robot base teleop.

Goal:
  dp_base ≈ s * R * dp_pika_world

How it works:
  - Ask user to move PIKA Sense along robot-base directions:
      +X, -X, +Y, -Y, +Z, -Z  (robot base frame)
  - Record start/end PIKA positions (world/base_link from get_pose())
  - Build unit direction vectors u_i in PIKA world
  - Pair them with desired robot unit vectors v_i
  - Solve R (SO(3)) using SVD (Orthogonal Procrustes/Wahba)
  - Solve s using median ratio of desired_robot_move / observed_hand_move
  - Save to calib.json

Dependencies:
  pip install numpy
"""

import argparse
import json
import time
from dataclasses import dataclass, field
from typing import Tuple, List

import numpy as np


# -----------------------------
# 0) YOU implement this adapter
# -----------------------------
class PikaAdapter:
    """
    Wrap your pika_sdk access here.

    Must return:
      pos: np.ndarray shape (3,), in meters, world(base_link) frame
      quat: np.ndarray shape (4,), quaternion (unused in this calibration, but kept for consistency)
    """
    def __init__(self, device_key: str = "WM0", port: str = "/dev/ttyUSB0", pos_unit: str = "m"):
        self.device_key = device_key
        self.port = port
        self.pos_scale = 1.0 if pos_unit == "m" else 0.001
        self._sense = None
        self.dev = None
        self._connect()

    def _connect(self) -> None:
        try:
            from pika import sense
        except ImportError as exc:
            raise RuntimeError(
                "Failed to import pika SDK (`from pika import sense`). "
                "Install/activate pika_sdk first."
            ) from exc

        self._sense = sense
        self.dev = self._sense(self.port)
        ok = self.dev.connect()
        if not ok:
            raise RuntimeError(f"Failed to connect PIKA Sense on port: {self.port}")

    def close(self) -> None:
        if self.dev is None:
            return
        try:
            self.dev.disconnect()
        except Exception:
            pass
        self.dev = None

    def get_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (pos_m, quat_xyzw)
        - pos unit must be meters (convert if needed)
        - quat order ideally [x,y,z,w] (not needed for R,s here)
        """
        if self.dev is None:
            raise RuntimeError("PIKA device is not connected.")

        pose_obj = self.dev.get_pose(self.device_key)
        if pose_obj is None:
            all_poses = self.dev.get_pose()
            if isinstance(all_poses, dict):
                pose_obj = all_poses.get(self.device_key)

        if pose_obj is None:
            raise RuntimeError(f"No pose available for device_key={self.device_key}")

        pos = np.asarray(pose_obj.position, dtype=np.float64) * self.pos_scale
        quat = np.asarray(pose_obj.rotation, dtype=np.float64)
        if pos.shape != (3,):
            raise RuntimeError(f"Unexpected position shape: {pos.shape}")
        if quat.shape != (4,):
            raise RuntimeError(f"Unexpected quaternion shape: {quat.shape}")
        return pos, quat


# -----------------------------
# 1) math utilities
# -----------------------------
def normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v)
    return v / n


def solve_rotation_procrustes(U_dirs: np.ndarray, V_dirs: np.ndarray) -> np.ndarray:
    """
    Solve R in SO(3) minimizing || V - R U ||_F
    U_dirs: (3,N) unit vectors in PIKA(world) frame
    V_dirs: (3,N) unit vectors in ROBOT(base) frame
    """
    H = V_dirs @ U_dirs.T
    Q, _, Pt = np.linalg.svd(H)
    Rm = Q @ Pt
    if np.linalg.det(Rm) < 0:
        Q[:, -1] *= -1
        Rm = Q @ Pt
    return Rm


def average_pose(tracker: PikaAdapter, secs: float = 0.25, hz: float = 120.0) -> Tuple[np.ndarray, np.ndarray]:
    """Noise-reduced pose by averaging positions and quaternion (rough)."""
    dt = 1.0 / hz
    n = max(2, int(secs * hz))
    ps = []
    qs = []
    for _ in range(n):
        p, q = tracker.get_pose()
        ps.append(p)
        qs.append(q)
        time.sleep(dt)
    p_mean = np.mean(np.stack(ps, axis=0), axis=0)

    # quaternion average (not used for R,s; keep simple & robust)
    qs = np.stack(qs, axis=0)
    q0 = qs[0]
    for i in range(len(qs)):
        if np.dot(q0, qs[i]) < 0:
            qs[i] *= -1
    q_mean = qs.mean(axis=0)
    q_mean = q_mean / max(1e-12, np.linalg.norm(q_mean))
    return p_mean, q_mean


@dataclass
class CalibConfig:
    # Define what robot base directions mean (default: +X right, +Y forward, +Z up)
    # If your robot uses different convention, change these vectors.
    right_vec: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=np.float64))
    forward_vec: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0], dtype=np.float64))
    up_vec: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0], dtype=np.float64))

    # Collection params
    avg_secs: float = 0.25
    min_move_m: float = 0.02      # ignore steps smaller than 2cm
    desired_robot_move_m: float = 0.05  # used to compute initial s (median)


def build_steps(cfg: CalibConfig):
    # 6 direction steps in robot base frame
    return [
        ("+X (Right)",  cfg.right_vec),
        ("-X (Left)",  -cfg.right_vec),
        ("+Y (Forward)", cfg.forward_vec),
        ("-Y (Backward)", -cfg.forward_vec),
        ("+Z (Up)",     cfg.up_vec),
        ("-Z (Down)",  -cfg.up_vec),
    ]


def validate_and_orthonormalize_axes(cfg: CalibConfig) -> CalibConfig:
    right = normalize(np.array(cfg.right_vec, dtype=np.float64))
    forward_raw = normalize(np.array(cfg.forward_vec, dtype=np.float64))
    up_hint = normalize(np.array(cfg.up_vec, dtype=np.float64))

    if np.linalg.norm(right) < 1e-9 or np.linalg.norm(forward_raw) < 1e-9 or np.linalg.norm(up_hint) < 1e-9:
        raise ValueError("Axis vectors must be non-zero.")

    # Gram-Schmidt on forward against right, then build right-handed up.
    forward = forward_raw - np.dot(forward_raw, right) * right
    forward = normalize(forward)
    if np.linalg.norm(forward) < 1e-9:
        raise ValueError("`right` and `forward` are colinear; cannot build a valid basis.")

    up = normalize(np.cross(right, forward))
    if np.linalg.norm(up) < 1e-9:
        raise ValueError("Failed to derive `up` from `right x forward`.")

    # Keep up direction close to user input when possible.
    if np.dot(up, up_hint) < 0:
        up *= -1.0
        forward *= -1.0

    return CalibConfig(
        right_vec=right,
        forward_vec=forward,
        up_vec=up,
        avg_secs=cfg.avg_secs,
        min_move_m=cfg.min_move_m,
        desired_robot_move_m=cfg.desired_robot_move_m,
    )


def calibrate_R_s(tracker: PikaAdapter, out_path: str, cfg: CalibConfig):
    print("\n=== Alignment Calibration (Robot-base reference) ===")
    print("You will do 6 short moves aligned to ROBOT BASE axes.")
    print("For each step:")
    print("  1) Hold still → press Enter to capture START")
    print("  2) Move straight ~5-15cm in the instructed direction")
    print("  3) Hold still → press Enter to capture END")
    print("\nIMPORTANT: move in ROBOT BASE directions, NOT in your hand-local directions.\n")

    steps = build_steps(cfg)

    U_list: List[np.ndarray] = []  # PIKA(world) unit direction vectors
    V_list: List[np.ndarray] = []  # ROBOT(base) unit direction vectors
    d_hand: List[float] = []

    for name, v_des in steps:
        while True:
            input(f"[{name}] Press Enter to capture START (hold still)")
            p0, _ = average_pose(tracker, secs=cfg.avg_secs)

            input(f"[{name}] Move now, then press Enter to capture END (hold still)")
            p1, _ = average_pose(tracker, secs=cfg.avg_secs)

            dp = p1 - p0
            dist = float(np.linalg.norm(dp))
            if dist < cfg.min_move_m:
                print(f"  !! Move too small: {dist:.4f} m. Please redo this step.")
                continue

            u = normalize(dp)
            U_list.append(u)
            V_list.append(normalize(v_des))
            d_hand.append(dist)
            print(f"  recorded dist={dist:.4f} m, u(pika_world)={u}")
            break

    if len(U_list) < 3:
        raise RuntimeError("Not enough valid steps. Need >=3 (recommend 6). Re-run and make larger moves.")

    U = np.stack(U_list, axis=1)  # 3xN
    V = np.stack(V_list, axis=1)  # 3xN

    R_map = solve_rotation_procrustes(U, V)

    # s initial estimate: median(desired_robot_move / hand_move)
    ratios = [cfg.desired_robot_move_m / d for d in d_hand if d > 1e-9]
    s = float(np.median(np.array(ratios, dtype=np.float64))) if ratios else 1.0

    # Save
    calib = {
        "version": 1,
        "timestamp_unix": time.time(),
        "device_key": tracker.device_key,
        "R_map_rowmajor_3x3": R_map.reshape(-1).tolist(),
        "s": s,
        "robot_base_convention": {
            "right": cfg.right_vec.tolist(),
            "forward": cfg.forward_vec.tolist(),
            "up": cfg.up_vec.tolist(),
        },
        "collection": {
            "avg_secs": cfg.avg_secs,
            "min_move_m": cfg.min_move_m,
            "desired_robot_move_m": cfg.desired_robot_move_m,
            "valid_steps": len(U_list),
        },
        "definition": "dp_base = s * R_map * dp_pika_world (incremental teleop recommended)"
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(calib, f, indent=2, ensure_ascii=False)

    print("\n=== Result ===")
    print("R_map (robot_base <- pika_world):\n", R_map)
    print("det(R_map) =", float(np.linalg.det(R_map)))
    print("s =", s)
    print("Saved:", out_path)
    print("\nNext: use this R_map and s inside teleop loop:\n"
          "  dp_base = s * R_map @ dp_pika_world\n")

    return calib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device_key", type=str, default="WM0")
    ap.add_argument("--port", type=str, default="/dev/ttyUSB0")
    ap.add_argument(
        "--pos_unit",
        choices=["m", "mm"],
        default="m",
        help="Position unit returned by SDK get_pose(): m or mm",
    )
    ap.add_argument("--out", type=str, default="calib.json")
    ap.add_argument("--avg_secs", type=float, default=0.25)
    ap.add_argument("--min_move_m", type=float, default=0.02)
    ap.add_argument("--desired_robot_move_m", type=float, default=0.05)

    # Robot base axis convention override (optional)
    ap.add_argument("--right", nargs=3, type=float, default=[1,0,0], help="Robot base RIGHT unit vector (default +X)")
    ap.add_argument("--forward", nargs=3, type=float, default=[0,1,0], help="Robot base FORWARD unit vector (default +Y)")
    ap.add_argument("--up", nargs=3, type=float, default=[0,0,1], help="Robot base UP unit vector (default +Z)")

    args = ap.parse_args()

    cfg = CalibConfig(
        right_vec=np.array(args.right, dtype=np.float64),
        forward_vec=np.array(args.forward, dtype=np.float64),
        up_vec=np.array(args.up, dtype=np.float64),
        avg_secs=args.avg_secs,
        min_move_m=args.min_move_m,
        desired_robot_move_m=args.desired_robot_move_m,
    )
    cfg = validate_and_orthonormalize_axes(cfg)

    tracker = PikaAdapter(device_key=args.device_key, port=args.port, pos_unit=args.pos_unit)
    try:
        calibrate_R_s(tracker, args.out, cfg)
    finally:
        tracker.close()


if __name__ == "__main__":
    main()
