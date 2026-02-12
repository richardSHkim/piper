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
import sys
import time
from pathlib import Path

import numpy as np
from piper_sdk import C_PiperInterface_V2

from calibration import PikaAdapter
try:
    import termios
except Exception:
    termios = None

M_TO_001MM = 1_000_000.0
_001MM_TO_M = 1.0 / 1_000_000.0
RAD_TO_001DEG = 180000.0 / math.pi
_001DEG_TO_RAD = math.pi / 180000.0
ZERO_SPEED_RATIO_DEFAULT = 30
ZERO_SETTLE_S_DEFAULT = 3.0
TRANSLATION_GAIN_DEFAULT = 3.0
END_POSE_INIT_M_RAD = (
    0.057,  # X: 57.0 mm
    0.0,    # Y: 0.0 mm
    0.260,  # Z: 260.0 mm
    0.0,    # Roll: 0.0 deg
    math.radians(85.0),  # Pitch: 85.0 deg
    0.0,    # Yaw: 0.0 deg
)


def wait_enable(arm: C_PiperInterface_V2, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if arm.EnablePiper():
            return
        time.sleep(0.05)
    raise RuntimeError(f"PiPER enable timeout after {timeout_s:.1f}s")


def move_piper_to_zero_and_open_gripper(
    arm: C_PiperInterface_V2,
    speed_ratio: int,
    gripper_effort: int,
    piper_gripper_opening_m: float,
) -> None:
    # Follow SDK demo sequence: joint mode -> zero joints -> gripper fully open.
    arm.ModeCtrl(0x01, 0x01, int(speed_ratio), 0x00)
    arm.JointCtrl(0, 0, 0, 0, 0, 0)
    open_stroke_001mm = int(round(float(piper_gripper_opening_m) * M_TO_001MM))
    arm.GripperCtrl(abs(open_stroke_001mm), int(gripper_effort), 0x01, 0x00)


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


def clamp_scalar(x: float, lo: float, hi: float) -> float:
    return min(max(x, lo), hi)


def wrap_to_pi(angle_rad: float) -> float:
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def normalize_quat_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n <= 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / n


def quat_conjugate_xyzw(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def quat_mul_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], dtype=np.float64)


def quat_to_rpy_xyzw(q: np.ndarray) -> tuple[float, float, float]:
    x, y, z, w = q
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def parse_axis_order(axis_order: str) -> tuple[int, int, int]:
    token = axis_order.strip().lower()
    if len(token) != 3 or set(token) != {"x", "y", "z"}:
        raise ValueError("pika-axis-order must be a permutation of xyz (e.g. xyz, xzy, yxz)")
    idx = {"x": 0, "y": 1, "z": 2}
    return (idx[token[0]], idx[token[1]], idx[token[2]])


def parse_axis_sign(axis_sign: str) -> np.ndarray:
    token = axis_sign.strip()
    if len(token) != 3 or any(c not in "+-" for c in token):
        raise ValueError("pika-axis-sign must be 3 chars using +/- (e.g. +++, +-+, --+)")
    return np.array([1.0 if c == "+" else -1.0 for c in token], dtype=np.float64)


def flush_stdin_buffer() -> None:
    # Drop any Enter presses made before readiness to prevent accidental start.
    if termios is None or (not sys.stdin.isatty()):
        return
    try:
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except Exception:
        pass


def wait_pika_pose_stable(
    tracker: PikaAdapter,
    stable_secs: float,
    poll_hz: float,
    max_step_m: float,
    timeout_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    dt = 1.0 / max(1.0, float(poll_hz))
    need_count = max(1, int(stable_secs * poll_hz))
    deadline = time.time() + timeout_s
    last_log = time.time()

    p_prev, q_prev = tracker.get_pose()
    stable_count = 0
    latest_step = 0.0
    while time.time() < deadline:
        time.sleep(dt)
        p_cur, q_cur = tracker.get_pose()
        latest_step = float(np.linalg.norm(p_cur - p_prev))
        p_prev = p_cur
        q_prev = q_cur

        if latest_step <= max_step_m:
            stable_count += 1
        else:
            stable_count = 0

        if stable_count >= need_count:
            return p_cur, q_prev

        now = time.time()
        if now - last_log >= 0.5:
            print(
                f"[ready-check] waiting pika stable... step={latest_step:.5f}m, "
                f"stable={(stable_count / max(need_count, 1)) * 100.0:.0f}%"
            )
            last_log = now

    raise RuntimeError(
        f"PIKA pose did not become stable within {timeout_s:.1f}s "
        f"(last step={latest_step:.5f}m, threshold={max_step_m:.5f}m)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Teleoperate PiPER with PIKA pose increments using calibration json."
    )
    parser.add_argument("--calib", type=str, default="calib.json")
    parser.add_argument("--can", type=str, default="can_follower")
    parser.add_argument("--speed-ratio", type=int, default=70, help="PiPER MotionCtrl_2 speed ratio [0..100]")
    parser.add_argument(
        "--translation-gain",
        type=float,
        default=TRANSLATION_GAIN_DEFAULT,
        help="Extra translation gain on top of calibration scale s",
    )
    parser.add_argument("--period", type=float, default=0.02, help="Loop period in seconds")
    parser.add_argument("--enable-timeout", type=float, default=5.0)
    parser.add_argument("--disable-on-exit", action="store_true")
    parser.add_argument("--installation-pos", type=int, default=0, choices=[0, 1, 2, 3])

    parser.add_argument("--pika-device-key", type=str, default="WM0")
    parser.add_argument("--pika-port", type=str, default="/dev/ttyUSB0")
    parser.add_argument("--pika-pos-unit", choices=["m", "mm"], default="m")
    parser.add_argument("--pika-startup-timeout-s", type=float, default=5.0)
    parser.add_argument("--pika-poll-hz", type=float, default=120.0)
    parser.add_argument("--ready-timeout-s", type=float, default=20.0, help="Timeout for readiness checks")
    parser.add_argument("--ready-stable-secs", type=float, default=1.0, help="Required continuous stable duration")
    parser.add_argument(
        "--ready-max-step-m",
        type=float,
        default=0.0015,
        help="Max per-sample PIKA motion to consider stable",
    )

    parser.add_argument("--max-pika-step-m", type=float, default=0.05, help="Reject/clip sudden tracker jumps")
    parser.add_argument("--max-base-step-m", type=float, default=0.03, help="Clip cartesian increment per cycle")
    parser.add_argument(
        "--pika-axis-order",
        type=str,
        default="xyz",
        help="Axis order remap for raw PIKA delta before calibration (permutation of xyz)",
    )
    parser.add_argument(
        "--pika-axis-sign",
        type=str,
        default="+++",
        help="Axis sign remap for raw PIKA delta before calibration (3 chars of +/-)",
    )
    parser.add_argument(
        "--dp-ema-alpha",
        type=float,
        default=0.35,
        help="EMA alpha for dp_pika smoothing in (0, 1]; lower = smoother",
    )
    parser.add_argument(
        "--jitter-deadband-m",
        type=float,
        default=0.0008,
        help="Set small dp_pika norms below this threshold to zero",
    )
    parser.add_argument("--disable-jitter-filter", action="store_true", help="Disable EMA+deadband jitter filter")
    parser.add_argument("--gripper-effort", type=int, default=1000, help="PiPER gripper effort [0..5000]")
    parser.add_argument(
        "--piper-gripper-opening-m",
        type=float,
        default=0.07,
        help="PiPER full gripper opening in meters for ratio->stroke mapping",
    )
    parser.add_argument("--pika-gripper-min-mm", type=float, default=0.0, help="PIKA gripper mm at fully closed")
    parser.add_argument("--pika-gripper-max-mm", type=float, default=90.0, help="PIKA gripper mm at fully open")
    parser.add_argument("--disable-gripper-sync", action="store_true", help="Disable PIKA->PiPER gripper mapping")
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
    if args.translation_gain <= 0:
        raise ValueError("translation-gain must be > 0")
    if args.max_pika_step_m <= 0 or args.max_base_step_m <= 0:
        raise ValueError("max step values must be > 0")
    if not (0 < args.dp_ema_alpha <= 1.0):
        raise ValueError("dp-ema-alpha must be in (0, 1]")
    if args.jitter_deadband_m < 0:
        raise ValueError("jitter-deadband-m must be >= 0")
    if not (0 <= args.gripper_effort <= 5000):
        raise ValueError("gripper-effort must be in [0, 5000]")
    if args.piper_gripper_opening_m <= 0:
        raise ValueError("piper-gripper-opening-m must be > 0")
    if args.pika_gripper_max_mm <= args.pika_gripper_min_mm:
        raise ValueError("pika-gripper-max-mm must be greater than pika-gripper-min-mm")
    if args.ready_timeout_s <= 0:
        raise ValueError("ready-timeout-s must be > 0")
    if args.ready_stable_secs <= 0:
        raise ValueError("ready-stable-secs must be > 0")
    if args.ready_max_step_m <= 0:
        raise ValueError("ready-max-step-m must be > 0")

    pika_axis_idx = parse_axis_order(args.pika_axis_order)
    pika_axis_sign = parse_axis_sign(args.pika_axis_sign)
    R_map, s = load_calib(args.calib)

    arm = C_PiperInterface_V2(args.can)
    tracker = None
    try:
        arm.ConnectPort()
        time.sleep(0.2)
        wait_enable(arm, timeout_s=args.enable_timeout)
        print(
            f"[init] moving piper to joint zero + gripper open "
            f"(speed_ratio={ZERO_SPEED_RATIO_DEFAULT})"
        )
        move_piper_to_zero_and_open_gripper(
            arm,
            speed_ratio=ZERO_SPEED_RATIO_DEFAULT,
            gripper_effort=args.gripper_effort,
            piper_gripper_opening_m=args.piper_gripper_opening_m,
        )
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

        print("[ready-check] waiting for stable PIKA pose...")
        p_prev, q_prev = wait_pika_pose_stable(
            tracker=tracker,
            stable_secs=args.ready_stable_secs,
            poll_hz=args.pika_poll_hz,
            max_step_m=args.ready_max_step_m,
            timeout_s=args.ready_timeout_s,
        )
        q_prev = normalize_quat_xyzw(q_prev)

        target_x, target_y, target_z, target_roll, target_pitch, target_yaw = END_POSE_INIT_M_RAD
        print(
            "[ready] piper initialized (zero + gripper open), pika pose stable. "
            "Press Enter to start teleop."
        )
        flush_stdin_buffer()
        input()

        print(
            "[start] "
            f"can={args.can}, pika={args.pika_device_key}@{args.pika_port}, "
            f"s={s:.6f}, gain={args.translation_gain:.2f}, period={args.period:.3f}s"
        )
        print(
            "[map] "
            f"pika_axis_order={args.pika_axis_order}, pika_axis_sign={args.pika_axis_sign}"
        )
        if args.disable_jitter_filter:
            print("[filter] jitter filter disabled")
        else:
            print(
                "[filter] "
                f"dp_ema_alpha={args.dp_ema_alpha:.2f}, "
                f"jitter_deadband_m={args.jitter_deadband_m:.4f}"
            )
        print(
            "[seed] "
            f"target=({target_x:.4f}, {target_y:.4f}, {target_z:.4f}) m, "
            f"rpy=({target_roll:.3f}, {target_pitch:.3f}, {target_yaw:.3f}) rad"
        )

        last_log = time.time()
        count = 0
        dp_ema = np.zeros(3, dtype=np.float64)
        while True:
            p_cur, q_cur = tracker.get_pose()
            dp_pika = p_cur - p_prev
            p_prev = p_cur
            dp_pika = dp_pika[list(pika_axis_idx)] * pika_axis_sign
            q_cur = normalize_quat_xyzw(q_cur)
            if float(np.dot(q_prev, q_cur)) < 0.0:
                q_cur = -q_cur
            q_delta = quat_mul_xyzw(q_cur, quat_conjugate_xyzw(q_prev))
            q_delta = normalize_quat_xyzw(q_delta)
            d_roll, d_pitch, d_yaw = quat_to_rpy_xyzw(q_delta)
            d_rpy_pika = np.array([d_roll, d_pitch, d_yaw], dtype=np.float64)
            d_rpy_base = R_map @ d_rpy_pika
            q_prev = q_cur

            dp_pika = clamp_vec(dp_pika, args.max_pika_step_m)
            if not args.disable_jitter_filter:
                dp_ema = (args.dp_ema_alpha * dp_pika) + ((1.0 - args.dp_ema_alpha) * dp_ema)
                dp_pika = dp_ema
                if float(np.linalg.norm(dp_pika)) < args.jitter_deadband_m:
                    dp_pika = np.zeros(3, dtype=np.float64)
            dp_base = (s * args.translation_gain) * (R_map @ dp_pika)
            dp_base = clamp_vec(dp_base, args.max_base_step_m)

            target_x += float(dp_base[0])
            target_y += float(dp_base[1])
            target_z += float(dp_base[2])
            target_roll = wrap_to_pi(target_roll + float(d_rpy_base[0]))
            target_pitch = wrap_to_pi(target_pitch + float(d_rpy_base[1]))
            target_yaw = wrap_to_pi(target_yaw + float(d_rpy_base[2]))

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

            if not args.disable_gripper_sync:
                gripper_mm = tracker.get_gripper_distance_mm()
                if gripper_mm is not None:
                    gripper_ratio = (gripper_mm - args.pika_gripper_min_mm) / (
                        args.pika_gripper_max_mm - args.pika_gripper_min_mm
                    )
                    gripper_ratio = clamp_scalar(gripper_ratio, 0.0, 1.0)
                    stroke_001mm = int(round(gripper_ratio * args.piper_gripper_opening_m * M_TO_001MM))
                    arm.GripperCtrl(abs(stroke_001mm), int(args.gripper_effort), 0x01, 0x00)

            count += 1
            now = time.time()
            if now - last_log >= 1.0:
                hz = count / max(now - last_log, 1e-6)
                print(
                    f"[run] loop_hz={hz:.1f}, "
                    f"target=({target_x:.3f}, {target_y:.3f}, {target_z:.3f}), "
                    f"rpy=({target_roll:.3f}, {target_pitch:.3f}, {target_yaw:.3f})"
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
