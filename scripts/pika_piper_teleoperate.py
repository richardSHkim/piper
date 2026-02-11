#!/usr/bin/env python3

import argparse
import time
from collections import deque
import math

from lerobot.processor import RobotProcessorPipeline
from lerobot.processor.converters import robot_action_observation_to_transition, transition_to_robot_action
from lerobot_teleoperator_pika import MapPikaActionToPiperJoints, PikaTeleoperator, PikaTeleoperatorConfig
from lerobot_robot_piper_follower import PiperFollower, PiperFollowerConfig


def make_teleop_action_processor(args: argparse.Namespace) -> tuple[RobotProcessorPipeline, MapPikaActionToPiperJoints]:
    step = MapPikaActionToPiperJoints(
        linear_scale=args.linear_scale,
        angular_scale=args.angular_scale,
        max_delta_pos_m=args.max_delta_pos_m,
        max_delta_rot_rad=args.max_delta_rot_rad,
        damping=args.ik_damping,
        max_iterations=args.ik_max_iterations,
        max_delta_q=args.ik_max_delta_q,
        pos_tol=args.ik_pos_tol,
        rot_tol=args.ik_rot_tol,
        solve_ik=(args.command_mode == "joint"),
    )
    pipeline = RobotProcessorPipeline(
        steps=[step],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    return pipeline, step


def _make_zero_action(gripper: float = 1.0) -> dict[str, float]:
    act = {f"joint_{i}.pos": 0.0 for i in range(1, 7)}
    act["gripper.pos"] = max(0.0, min(1.0, gripper))
    return act


def _format_pika_pose(raw_action: dict) -> str:
    return (
        f"pos=({raw_action.get('pika.pos.x', 0.0):.4f}, {raw_action.get('pika.pos.y', 0.0):.4f}, "
        f"{raw_action.get('pika.pos.z', 0.0):.4f}), "
        f"quat=({raw_action.get('pika.rot.x', 0.0):.4f}, {raw_action.get('pika.rot.y', 0.0):.4f}, "
        f"{raw_action.get('pika.rot.z', 0.0):.4f}, {raw_action.get('pika.rot.w', 1.0):.4f}), "
        f"gripper={raw_action.get('pika.gripper.pos', 0.0):.3f}, "
        f"pose_valid={raw_action.get('pika.pose.valid', 0.0):.0f}"
    )


def _format_joint_action(robot_action: dict) -> str:
    joints = ", ".join([f"j{i}={robot_action.get(f'joint_{i}.pos', 0.0):.4f}" for i in range(1, 7)])
    return f"{joints}, gripper={robot_action.get('gripper.pos', 0.0):.3f}"


def _rotation_matrix_to_rpy(rot) -> tuple[float, float, float]:
    # XYZ roll-pitch-yaw
    sy = -rot[2, 0]
    sy = max(min(sy, 1.0), -1.0)
    pitch = math.asin(sy)
    cp = math.cos(pitch)

    if abs(cp) > 1e-6:
        roll = math.atan2(rot[2, 1], rot[2, 2])
        yaw = math.atan2(rot[1, 0], rot[0, 0])
    else:
        # Gimbal lock fallback
        roll = 0.0
        yaw = math.atan2(-rot[0, 1], rot[1, 1])
    return roll, pitch, yaw


def main() -> None:
    parser = argparse.ArgumentParser(description="PIKA Sense -> PiPER follower teleoperation loop")
    parser.add_argument("--teleop-port", default="/dev/ttyUSB0", help="PIKA Sense serial port")
    parser.add_argument("--tracker-device", default=None, help="Optional tracker device name (e.g. WM0)")
    parser.add_argument("--robot-can-name", default="can_follower", help="PiPER follower CAN interface")
    parser.add_argument("--fps", type=int, default=50, help="Control loop rate")
    parser.add_argument("--speed-ratio", type=int, default=60, help="PiPER speed ratio [0..100]")
    parser.add_argument("--gripper-effort", type=int, default=1000, help="PiPER gripper effort [0..5000]")
    parser.add_argument(
        "--command-mode",
        choices=["endpose", "joint"],
        default="endpose",
        help="Send EndPoseCtrl(cartesian) or JointCtrl(joint) to PiPER",
    )
    parser.add_argument("--linear-scale", type=float, default=1.0, help="Scale for tracker translation delta")
    parser.add_argument("--angular-scale", type=float, default=1.0, help="Scale for tracker rotation delta")
    parser.add_argument("--max-delta-pos-m", type=float, default=0.03, help="Max translation delta per frame")
    parser.add_argument("--max-delta-rot-rad", type=float, default=0.30, help="Max rotation delta per frame")
    parser.add_argument("--ik-damping", type=float, default=0.08, help="IK damping factor")
    parser.add_argument("--ik-max-iterations", type=int, default=50, help="IK max iterations")
    parser.add_argument("--ik-max-delta-q", type=float, default=0.08, help="Per-joint max delta per frame")
    parser.add_argument("--ik-pos-tol", type=float, default=2e-3, help="IK position tolerance")
    parser.add_argument("--ik-rot-tol", type=float, default=2e-2, help="IK orientation tolerance")
    parser.add_argument("--print-interval", type=float, default=1.0, help="Status log period (s)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not send actions to PiPER; only print transformed action values",
    )
    parser.add_argument("--startup-move-zero", action="store_true", help="Move PiPER to zero pose before arming")
    parser.add_argument("--startup-zero-settle-s", type=float, default=1.0, help="Wait time after zero move")
    parser.add_argument(
        "--startup-sync-gripper",
        action="store_true",
        help="When moving to zero, set PiPER gripper to current PIKA gripper ratio",
    )
    parser.add_argument(
        "--arm-gesture-double-close",
        action="store_true",
        help="Require double-close gesture on PIKA gripper to start teleoperation",
    )
    parser.add_argument("--gesture-open-threshold", type=float, default=0.75, help="Open threshold for gesture")
    parser.add_argument("--gesture-close-threshold", type=float, default=0.25, help="Close threshold for gesture")
    parser.add_argument("--gesture-window-s", type=float, default=1.0, help="Double-close window in seconds")
    parser.add_argument("--gesture-debounce-s", type=float, default=0.15, help="Debounce per close event")
    parser.add_argument("--require-pose", action="store_true", help="Fail if tracker pose is missing")
    args = parser.parse_args()

    teleop_cfg = PikaTeleoperatorConfig(
        id="pika_teleop_01",
        port=args.teleop_port,
        tracker_device=args.tracker_device,
        require_pose=args.require_pose,
    )
    robot_cfg = PiperFollowerConfig(
        id="piper_follower_01",
        can_name=args.robot_can_name,
        speed_ratio=args.speed_ratio,
        gripper_effort=args.gripper_effort,
    )

    teleop = PikaTeleoperator(teleop_cfg)
    robot = PiperFollower(robot_cfg)
    teleop_action_processor, map_step = make_teleop_action_processor(args)

    teleop.connect()
    robot.connect()
    print(
        f"[start] teleop_port={args.teleop_port}, tracker={args.tracker_device}, "
        f"robot_can={args.robot_can_name}, fps={args.fps}"
    )

    # Warmup read for gripper and pose validity.
    startup_action = teleop.get_action()

    if args.startup_move_zero:
        zero_gripper = float(startup_action.get("pika.gripper.pos", 1.0)) if args.startup_sync_gripper else 1.0
        zero_action = _make_zero_action(zero_gripper)
        if args.dry_run:
            print(f"[dry-run] startup zero action={zero_action}")
        else:
            robot.send_action(zero_action)
        print(
            f"[startup] moved to zero pose (gripper={zero_gripper:.3f}), "
            f"settling {args.startup_zero_settle_s:.1f}s"
        )
        time.sleep(max(args.startup_zero_settle_s, 0.0))

    if args.arm_gesture_double_close:
        print(
            "[startup] align PIKA with PiPER, then close gripper quickly twice to arm teleoperation"
        )
        close_times: deque[float] = deque()
        prev_gripper = float(startup_action.get("pika.gripper.pos", 1.0))
        last_close_time = -1e9
        while True:
            raw = teleop.get_action()
            g = float(raw.get("pika.gripper.pos", 1.0))
            now = time.perf_counter()

            crossed_close = prev_gripper > args.gesture_open_threshold and g < args.gesture_close_threshold
            if crossed_close and (now - last_close_time) > args.gesture_debounce_s:
                close_times.append(now)
                last_close_time = now
                print(f"[startup] close event {len(close_times)}")

            while close_times and (now - close_times[0]) > args.gesture_window_s:
                close_times.popleft()

            if len(close_times) >= 2:
                obs = robot.get_observation()
                map_step.rebase_reference(raw, obs)
                print("[startup] teleoperation armed; reference pose captured")
                break

            prev_gripper = g
            time.sleep(0.01)
    else:
        # Capture current absolute PIKA pose as reference now.
        obs = robot.get_observation()
        map_step.rebase_reference(startup_action, obs)
        print("[startup] teleoperation armed immediately; reference pose captured")

    last_log = time.perf_counter()
    loop_count = 0
    period = 1.0 / max(args.fps, 1)

    try:
        while True:
            loop_start = time.perf_counter()
            obs = robot.get_observation()
            raw_action = teleop.get_action()
            robot_action = teleop_action_processor((raw_action, obs))
            if args.dry_run:
                tgt = map_step.get_target_pose()
                if tgt is not None:
                    roll, pitch, yaw = _rotation_matrix_to_rpy(tgt[:3, :3])
                    target_str = (
                        f"target_xyz_m=({tgt[0,3]:.4f}, {tgt[1,3]:.4f}, {tgt[2,3]:.4f}), "
                        f"target_rpy_rad=({roll:.4f}, {pitch:.4f}, {yaw:.4f})"
                    )
                else:
                    target_str = "target_pose=none"
                print(
                    f"[dry-run] pika_pose: {_format_pika_pose(raw_action)} | "
                    f"{target_str} | mapped_action: {_format_joint_action(robot_action)}"
                )
            else:
                if args.command_mode == "joint":
                    robot.send_action(robot_action)
                else:
                    tgt = map_step.get_target_pose()
                    if tgt is not None:
                        roll, pitch, yaw = _rotation_matrix_to_rpy(tgt[:3, :3])
                        x = int(round(float(tgt[0, 3]) * 1_000_000.0))  # m -> 0.001mm
                        y = int(round(float(tgt[1, 3]) * 1_000_000.0))
                        z = int(round(float(tgt[2, 3]) * 1_000_000.0))
                        rx = int(round(math.degrees(roll) * 1000.0))     # rad -> 0.001deg
                        ry = int(round(math.degrees(pitch) * 1000.0))
                        rz = int(round(math.degrees(yaw) * 1000.0))
                        robot._arm.MotionCtrl_2(0x01, 0x00, int(args.speed_ratio), 0x00)
                        robot._arm.EndPoseCtrl(x, y, z, rx, ry, rz)
                        if "gripper.pos" in robot_action:
                            stroke_001mm = int(
                                round(
                                    float(robot_action["gripper.pos"])
                                    * float(robot.config.gripper_opening_m)
                                    * 1_000_000.0
                                )
                            )
                            robot._arm.GripperCtrl(
                                abs(stroke_001mm),
                                int(args.gripper_effort),
                                0x01,
                                0x00,
                            )

            loop_count += 1
            now = time.perf_counter()
            if now - last_log >= args.print_interval:
                elapsed = now - last_log
                if args.command_mode == "endpose":
                    tgt = map_step.get_target_pose()
                    if tgt is not None:
                        roll, pitch, yaw = _rotation_matrix_to_rpy(tgt[:3, :3])
                        target_str = (
                            f"target_xyz=({tgt[0,3]:.3f},{tgt[1,3]:.3f},{tgt[2,3]:.3f})m "
                            f"target_rpy=({roll:.3f},{pitch:.3f},{yaw:.3f})rad"
                        )
                    else:
                        target_str = "target_pose=none"
                    print(
                        f"[run] loop_hz={loop_count / max(elapsed, 1e-6):.1f}, "
                        f"mode=endpose, {target_str}, "
                        f"obs_j1={obs.get('joint_1.pos', 0.0):.3f}, "
                        f"gripper={robot_action.get('gripper.pos', 0.0):.3f}, "
                        f"pose_valid={raw_action.get('pika.pose.valid', 0.0):.0f}"
                    )
                else:
                    print(
                        f"[run] loop_hz={loop_count / max(elapsed, 1e-6):.1f}, "
                        f"j1={robot_action.get('joint_1.pos', 0.0):.3f}, "
                        f"gripper={robot_action.get('gripper.pos', 0.0):.3f}, "
                        f"pose_valid={raw_action.get('pika.pose.valid', 0.0):.0f}"
                    )
                last_log = now
                loop_count = 0

            dt = time.perf_counter() - loop_start
            if dt < period:
                time.sleep(period - dt)
    except KeyboardInterrupt:
        print("\n[stop] keyboard interrupt")
    finally:
        teleop.disconnect()
        robot.disconnect()


if __name__ == "__main__":
    main()
