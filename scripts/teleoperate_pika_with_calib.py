#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PIKA -> PiPER teleoperation using calibration result from scripts/calibration.py.

Mapping:
  dp_base = s * R_map @ dp_pika_world

This script performs incremental cartesian teleop:
  1) Read current PiPER end pose as target seed.
  2) Repeatedly read PIKA pose increments.
  3) Transform increments with calibration.
  4) Send updated target with EndPoseCtrl.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from piper_sdk import C_PiperInterface_V2

from calibration import PikaAdapter

M_TO_001MM = 1_000_000.0
_001MM_TO_M = 1.0 / 1_000_000.0
RAD_TO_001DEG = 180000.0 / math.pi
_001DEG_TO_RAD = math.pi / 180000.0
ZERO_SPEED_RATIO_DEFAULT = 30
ZERO_SETTLE_S_DEFAULT = 3.0


def wait_enable(arm: C_PiperInterface_V2, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if arm.EnablePiper():
            return
        time.sleep(0.05)
    raise RuntimeError(f"PiPER enable timeout after {timeout_s:.1f}s")


def move_piper_to_zero(arm: C_PiperInterface_V2, speed_ratio: int, gripper_effort: int = 1000) -> None:
    # Follow SDK demo sequence: joint mode -> zero joints -> gripper open(0).
    arm.ModeCtrl(0x01, 0x01, int(speed_ratio), 0x00)
    arm.JointCtrl(0, 0, 0, 0, 0, 0)
    arm.GripperCtrl(0, int(gripper_effort), 0x01, 0x00)


def load_calib(path: str) -> tuple[np.ndarray, float]:
    calib = json.loads(Path(path).read_text(encoding="utf-8"))
    if "R_map_rowmajor_3x3" not in calib:
        raise KeyError("calib.json missing key: R_map_rowmajor_3x3")
    if "s" not in calib:
        raise KeyError("calib.json missing key: s")

    R_map = np.asarray(calib["R_map_rowmajor_3x3"], dtype=np.float64).reshape(3, 3)
    s = float(calib["s"])
    return R_map, s


def read_endpose_m_rad(arm: C_PiperInterface_V2) -> tuple[float, float, float, float, float, float]:
    end_pose = arm.GetArmEndPoseMsgs().end_pose
    return (
        float(end_pose.X_axis) * _001MM_TO_M,
        float(end_pose.Y_axis) * _001MM_TO_M,
        float(end_pose.Z_axis) * _001MM_TO_M,
        float(end_pose.RX_axis) * _001DEG_TO_RAD,
        float(end_pose.RY_axis) * _001DEG_TO_RAD,
        float(end_pose.RZ_axis) * _001DEG_TO_RAD,
    )


def clamp_vec(v: np.ndarray, max_norm: float) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= max_norm:
        return v
    return v * (max_norm / max(n, 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Teleoperate PiPER with PIKA pose increments using calibration json."
    )
    parser.add_argument("--calib", type=str, default="calib.json")
    parser.add_argument("--can", type=str, default="can_follower")
    parser.add_argument("--speed-ratio", type=int, default=70, help="PiPER MotionCtrl_2 speed ratio [0..100]")
    parser.add_argument("--period", type=float, default=0.02, help="Loop period in seconds")
    parser.add_argument("--enable-timeout", type=float, default=5.0)
    parser.add_argument("--disable-on-exit", action="store_true")
    parser.add_argument("--installation-pos", type=int, default=0, choices=[0, 1, 2, 3])

    parser.add_argument("--pika-device-key", type=str, default="WM0")
    parser.add_argument("--pika-port", type=str, default="/dev/ttyUSB0")
    parser.add_argument("--pika-pos-unit", choices=["m", "mm"], default="m")
    parser.add_argument("--pika-startup-timeout-s", type=float, default=5.0)
    parser.add_argument("--pika-poll-hz", type=float, default=120.0)

    parser.add_argument("--max-pika-step-m", type=float, default=0.05, help="Reject/clip sudden tracker jumps")
    parser.add_argument("--max-base-step-m", type=float, default=0.03, help="Clip cartesian increment per cycle")
    parser.add_argument(
        "--workspace",
        nargs=6,
        type=float,
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX", "Z_MIN", "Z_MAX"),
        default=None,
        help="Optional workspace clamp in meters",
    )
    args = parser.parse_args()

    if args.period <= 0:
        raise ValueError("period must be > 0")
    if not (0 <= args.speed_ratio <= 100):
        raise ValueError("speed-ratio must be in [0, 100]")
    if args.max_pika_step_m <= 0 or args.max_base_step_m <= 0:
        raise ValueError("max step values must be > 0")

    R_map, s = load_calib(args.calib)

    arm = C_PiperInterface_V2(args.can)
    tracker = None
    try:
        arm.ConnectPort()
        time.sleep(0.2)
        wait_enable(arm, timeout_s=args.enable_timeout)
        print(f"[init] moving piper to joint zero (speed_ratio={ZERO_SPEED_RATIO_DEFAULT})")
        move_piper_to_zero(arm, speed_ratio=ZERO_SPEED_RATIO_DEFAULT)
        if ZERO_SETTLE_S_DEFAULT > 0:
            time.sleep(ZERO_SETTLE_S_DEFAULT)

        # Switch to EndPose control mode after go-zero.
        arm.MotionCtrl_2(0x01, 0x00, int(args.speed_ratio), 0x00, 0, int(args.installation_pos))

        tracker = PikaAdapter(
            device_key=args.pika_device_key,
            port=args.pika_port,
            pos_unit=args.pika_pos_unit,
            startup_timeout_s=args.pika_startup_timeout_s,
            poll_hz=args.pika_poll_hz,
        )

        target_x, target_y, target_z, target_roll, target_pitch, target_yaw = read_endpose_m_rad(arm)
        p_prev, _ = tracker.get_pose()

        print(
            "[start] "
            f"can={args.can}, pika={args.pika_device_key}@{args.pika_port}, "
            f"s={s:.6f}, period={args.period:.3f}s"
        )
        print(
            "[seed] "
            f"target=({target_x:.4f}, {target_y:.4f}, {target_z:.4f}) m, "
            f"rpy=({target_roll:.3f}, {target_pitch:.3f}, {target_yaw:.3f}) rad"
        )

        last_log = time.time()
        count = 0
        while True:
            p_cur, _ = tracker.get_pose()
            dp_pika = p_cur - p_prev
            p_prev = p_cur

            dp_pika = clamp_vec(dp_pika, args.max_pika_step_m)
            dp_base = s * (R_map @ dp_pika)
            dp_base = clamp_vec(dp_base, args.max_base_step_m)

            target_x += float(dp_base[0])
            target_y += float(dp_base[1])
            target_z += float(dp_base[2])

            if args.workspace is not None:
                x_min, x_max, y_min, y_max, z_min, z_max = args.workspace
                target_x = min(max(target_x, x_min), x_max)
                target_y = min(max(target_y, y_min), y_max)
                target_z = min(max(target_z, z_min), z_max)

            arm.EndPoseCtrl(
                int(round(target_x * M_TO_001MM)),
                int(round(target_y * M_TO_001MM)),
                int(round(target_z * M_TO_001MM)),
                int(round(target_roll * RAD_TO_001DEG)),
                int(round(target_pitch * RAD_TO_001DEG)),
                int(round(target_yaw * RAD_TO_001DEG)),
            )

            count += 1
            now = time.time()
            if now - last_log >= 1.0:
                hz = count / max(now - last_log, 1e-6)
                print(
                    f"[run] loop_hz={hz:.1f}, "
                    f"dp_pika={np.linalg.norm(dp_pika):.4f}m, "
                    f"dp_base={np.linalg.norm(dp_base):.4f}m, "
                    f"target=({target_x:.3f}, {target_y:.3f}, {target_z:.3f})"
                )
                last_log = now
                count = 0

            time.sleep(args.period)
    except KeyboardInterrupt:
        print("\n[stop] keyboard interrupt")
    finally:
        if tracker is not None:
            tracker.close()
        if args.disable_on_exit:
            arm.DisableArm(7)
        arm.DisconnectPort()


if __name__ == "__main__":
    main()
