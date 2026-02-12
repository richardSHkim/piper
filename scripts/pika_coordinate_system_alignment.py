#!/usr/bin/env python3

"""
Coordinate System Alignment utility for PIKA Sense.

Implements the alignment step described in:
https://www.hackster.io/agilexrobotics/method-for-teleoperating-any-robotic-arm-via-pika-fa6292

Core idea:
1) Convert PIKA pose to homogeneous transform T_pika.
2) Build adjustment transform T_adjust from axis mapping.
3) Align frame by right-multiplication: T_aligned = T_pika @ T_adjust.

This script only handles coordinate alignment (not incremental teleoperation control).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Pose:
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float


_AXIS_TO_VEC = {
    "+x": np.array([1.0, 0.0, 0.0], dtype=np.float64),
    "-x": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
    "+y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
    "-y": np.array([0.0, -1.0, 0.0], dtype=np.float64),
    "+z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
    "-z": np.array([0.0, 0.0, -1.0], dtype=np.float64),
}


def _normalize_quat(qx: float, qy: float, qz: float, qw: float) -> tuple[float, float, float, float]:
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-12:
        return 0.0, 0.0, 0.0, 1.0
    return qx / n, qy / n, qz / n, qw / n


def quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    qx, qy, qz, qw = _normalize_quat(qx, qy, qz, qw)
    return np.array(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def rot_to_quat(rot: np.ndarray) -> tuple[float, float, float, float]:
    tr = float(np.trace(rot))
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rot[2, 1] - rot[1, 2]) / s
        qy = (rot[0, 2] - rot[2, 0]) / s
        qz = (rot[1, 0] - rot[0, 1]) / s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        qw = (rot[2, 1] - rot[1, 2]) / s
        qx = 0.25 * s
        qy = (rot[0, 1] + rot[1, 0]) / s
        qz = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        qw = (rot[0, 2] - rot[2, 0]) / s
        qx = (rot[0, 1] + rot[1, 0]) / s
        qy = 0.25 * s
        qz = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
        qw = (rot[1, 0] - rot[0, 1]) / s
        qx = (rot[0, 2] + rot[2, 0]) / s
        qy = (rot[1, 2] + rot[2, 1]) / s
        qz = 0.25 * s
    return _normalize_quat(qx, qy, qz, qw)


def rot_to_rpy(rot: np.ndarray) -> tuple[float, float, float]:
    sy = float(np.clip(-rot[2, 0], -1.0, 1.0))
    pitch = float(math.asin(sy))
    cp = float(math.cos(pitch))
    if abs(cp) > 1e-8:
        roll = float(math.atan2(rot[2, 1], rot[2, 2]))
        yaw = float(math.atan2(rot[1, 0], rot[0, 0]))
    else:
        roll = 0.0
        yaw = float(math.atan2(-rot[0, 1], rot[1, 1]))
    return roll, pitch, yaw


def pose_to_mat(pose: Pose) -> np.ndarray:
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = quat_to_rot(pose.qx, pose.qy, pose.qz, pose.qw)
    t[0, 3] = pose.x
    t[1, 3] = pose.y
    t[2, 3] = pose.z
    return t


def mat_to_pose(t: np.ndarray) -> Pose:
    qx, qy, qz, qw = rot_to_quat(t[:3, :3])
    return Pose(
        x=float(t[0, 3]),
        y=float(t[1, 3]),
        z=float(t[2, 3]),
        qx=qx,
        qy=qy,
        qz=qz,
        qw=qw,
    )


def build_adjustment_rotation(tool_x_in_pika: str, tool_y_in_pika: str, tool_z_in_pika: str) -> np.ndarray:
    try:
        x_col = _AXIS_TO_VEC[tool_x_in_pika]
        y_col = _AXIS_TO_VEC[tool_y_in_pika]
        z_col = _AXIS_TO_VEC[tool_z_in_pika]
    except KeyError as exc:
        valid = ", ".join(sorted(_AXIS_TO_VEC.keys()))
        raise ValueError(f"Invalid axis token: {exc}. Valid: {valid}") from exc

    rot = np.column_stack([x_col, y_col, z_col])
    if not np.allclose(rot.T @ rot, np.eye(3), atol=1e-8):
        raise ValueError("Provided tool axes are not orthonormal.")
    det = float(np.linalg.det(rot))
    if det < 0.0:
        raise ValueError("Provided tool axes are left-handed (det<0). Use a right-handed axis set.")
    return rot


def align_pose_with_adjustment(pose: Pose, rot_adjust: np.ndarray) -> Pose:
    t_pika = pose_to_mat(pose)
    t_adjust = np.eye(4, dtype=np.float64)
    t_adjust[:3, :3] = rot_adjust
    # Coordinate-system alignment: multiply by adjustment matrix.
    t_aligned = t_pika @ t_adjust
    return mat_to_pose(t_aligned)


def format_pose_line(prefix: str, pose: Pose) -> str:
    r, p, y = rot_to_rpy(quat_to_rot(pose.qx, pose.qy, pose.qz, pose.qw))
    return (
        f"{prefix} "
        f"xyz=({pose.x:+.4f},{pose.y:+.4f},{pose.z:+.4f}) "
        f"quat=({pose.qx:+.4f},{pose.qy:+.4f},{pose.qz:+.4f},{pose.qw:+.4f}) "
        f"rpy=({r:+.4f},{p:+.4f},{y:+.4f})"
    )


def _select_pose_data(sense_dev, tracker_device: str | None):
    pose = sense_dev.get_pose(tracker_device) if tracker_device else sense_dev.get_pose()
    if pose is None:
        return None
    if isinstance(pose, dict):
        if not pose:
            return None
        if tracker_device and tracker_device in pose:
            return pose[tracker_device]
        wm_keys = sorted([k for k in pose if k.startswith("WM")])
        if wm_keys:
            return pose[wm_keys[0]]
        return pose[sorted(pose.keys())[0]]
    return pose


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PIKA coordinate-system alignment utility (separate step from incremental control)."
    )
    parser.add_argument("--port", default="/dev/ttyUSB0", help="PIKA Sense serial port.")
    parser.add_argument("--tracker-device", default=None, help="Tracker device id (e.g., WM0).")
    parser.add_argument("--hz", type=float, default=30.0, help="Loop rate.")
    parser.add_argument(
        "--tool-x-in-pika",
        default="-z",
        choices=sorted(_AXIS_TO_VEC.keys()),
        help="Robot tool X-axis direction expressed in PIKA frame.",
    )
    parser.add_argument(
        "--tool-y-in-pika",
        default="+y",
        choices=sorted(_AXIS_TO_VEC.keys()),
        help="Robot tool Y-axis direction expressed in PIKA frame.",
    )
    parser.add_argument(
        "--tool-z-in-pika",
        default="+x",
        choices=sorted(_AXIS_TO_VEC.keys()),
        help="Robot tool Z-axis direction expressed in PIKA frame.",
    )
    parser.add_argument(
        "--save-adjustment-json",
        default=None,
        help="Optional path to save adjustment matrix JSON.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Read and print one aligned pose then exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rot_adjust = build_adjustment_rotation(args.tool_x_in_pika, args.tool_y_in_pika, args.tool_z_in_pika)
    print("R_tool_from_pika:")
    print(np.array2string(rot_adjust, precision=6, suppress_small=False))

    if args.save_adjustment_json:
        out_path = Path(args.save_adjustment_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "R_tool_from_pika": rot_adjust.tolist(),
                    "tool_x_in_pika": args.tool_x_in_pika,
                    "tool_y_in_pika": args.tool_y_in_pika,
                    "tool_z_in_pika": args.tool_z_in_pika,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"saved: {out_path}")

    from pika import sense

    dev = sense(args.port)
    ok = dev.connect()
    print(f"connect[{args.port}]:", ok)
    if not ok:
        raise SystemExit(1)

    dt = 1.0 / max(args.hz, 1e-6)
    try:
        while True:
            pose_data = _select_pose_data(dev, args.tracker_device)
            if pose_data is None:
                print("pose: none")
            else:
                raw = Pose(
                    x=float(pose_data.position[0]),
                    y=float(pose_data.position[1]),
                    z=float(pose_data.position[2]),
                    qx=float(pose_data.rotation[0]),
                    qy=float(pose_data.rotation[1]),
                    qz=float(pose_data.rotation[2]),
                    qw=float(pose_data.rotation[3]),
                )
                aligned = align_pose_with_adjustment(raw, rot_adjust)
                print(format_pose_line("raw   ", raw))
                print(format_pose_line("aligned", aligned))
            if args.once:
                return
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\nCtrl+C received, shutting down safely...")
    finally:
        dev.disconnect()
        print("disconnect: done")


if __name__ == "__main__":
    main()
