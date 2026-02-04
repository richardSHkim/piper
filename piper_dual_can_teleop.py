#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Dual-CAN teleoperation:
- Leader arm is operated by hand.
- Follower arm mirrors leader joint and gripper motion.
"""

import argparse
import time

from piper_sdk import C_PiperInterface_V2


def wait_enable(arm: C_PiperInterface_V2, timeout_s: float, name: str) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if arm.EnablePiper():
            return
        time.sleep(0.05)
    raise RuntimeError(f"{name}: failed to enable within {timeout_s:.1f}s")


def read_source_state(leader: C_PiperInterface_V2, source_mode: str):
    if source_mode == "feedback":
        joints_msg = leader.GetArmJointMsgs()
        grip_msg = leader.GetArmGripperMsgs()
        return joints_msg.Hz, joints_msg.joint_state, grip_msg.gripper_state

    joints_msg = leader.GetArmJointCtrl()
    grip_msg = leader.GetArmGripperCtrl()
    return joints_msg.Hz, joints_msg.joint_ctrl, grip_msg.gripper_ctrl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Teleoperate two Piper arms on different CAN interfaces."
    )
    parser.add_argument("--leader-can", default="can0", help="Leader CAN name")
    parser.add_argument("--follower-can", default="can1", help="Follower CAN name")
    parser.add_argument(
        "--source-mode",
        choices=["feedback", "control"],
        default="feedback",
        help="Use leader joint feedback or leader control frames as the source",
    )
    parser.add_argument(
        "--period",
        type=float,
        default=0.01,
        help="Loop period in seconds (default: 0.01 = 100 Hz)",
    )
    parser.add_argument(
        "--speed-ratio",
        type=int,
        default=100,
        help="Follower MotionCtrl_2 speed ratio [0..100]",
    )
    parser.add_argument(
        "--gripper-effort",
        type=int,
        default=1000,
        help="Follower gripper effort when source-mode=feedback [0..5000]",
    )
    parser.add_argument(
        "--disable-leader-motor",
        action="store_true",
        help="Disable leader motors so it can be moved by hand more easily",
    )
    parser.add_argument(
        "--disable-follower-on-exit",
        action="store_true",
        help="Disable follower motors when exiting",
    )
    args = parser.parse_args()

    if args.leader_can == args.follower_can:
        raise ValueError("leader-can and follower-can must be different")
    if not (0 <= args.speed_ratio <= 100):
        raise ValueError("speed-ratio must be in [0, 100]")
    if not (0 <= args.gripper_effort <= 5000):
        raise ValueError("gripper-effort must be in [0, 5000]")
    if args.period <= 0:
        raise ValueError("period must be > 0")

    leader = C_PiperInterface_V2(args.leader_can)
    follower = C_PiperInterface_V2(args.follower_can)

    leader.ConnectPort()
    follower.ConnectPort()
    time.sleep(0.2)

    if args.disable_leader_motor:
        leader.DisableArm(7)

    # Follower must be in CAN joint control mode.
    follower.MotionCtrl_2(0x01, 0x01, args.speed_ratio, 0x00)
    wait_enable(follower, timeout_s=5.0, name="follower")
    follower.GripperCtrl(0, args.gripper_effort, 0x01, 0x00)

    print(
        f"[start] leader={args.leader_can}, follower={args.follower_can}, "
        f"source={args.source_mode}, period={args.period:.4f}s"
    )
    last_log = time.time()
    loop_count = 0

    try:
        while True:
            hz, joints, gripper = read_source_state(leader, args.source_mode)

            follower.MotionCtrl_2(0x01, 0x01, args.speed_ratio, 0x00)
            follower.JointCtrl(
                int(joints.joint_1),
                int(joints.joint_2),
                int(joints.joint_3),
                int(joints.joint_4),
                int(joints.joint_5),
                int(joints.joint_6),
            )

            if args.source_mode == "control":
                gripper_effort = int(gripper.grippers_effort)
                gripper_code = int(gripper.status_code)
            else:
                gripper_effort = args.gripper_effort
                gripper_code = 0x01

            follower.GripperCtrl(
                abs(int(gripper.grippers_angle)),
                gripper_effort,
                gripper_code,
                0x00,
            )

            now = time.time()
            loop_count += 1
            if now - last_log >= 1.0:
                elapsed = now - last_log
                print(
                    f"[run] src_hz={hz:.1f}, loop_hz={loop_count / max(elapsed, 1e-6):.1f}, "
                    f"j1={joints.joint_1}, grip={gripper.grippers_angle}"
                )
                last_log = now
                loop_count = 0

            time.sleep(args.period)
    except KeyboardInterrupt:
        print("\n[stop] keyboard interrupt")
    finally:
        if args.disable_follower_on_exit:
            follower.DisableArm(7)
        leader.DisconnectPort()
        follower.DisconnectPort()


if __name__ == "__main__":
    main()
