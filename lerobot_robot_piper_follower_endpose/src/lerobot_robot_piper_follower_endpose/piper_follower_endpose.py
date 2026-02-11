#!/usr/bin/env python

import logging
import math
import time
from functools import cached_property

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.processor import RobotAction, RobotObservation
from lerobot.robots.robot import Robot
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from .config_piper_follower_endpose import PiperFollowerEndPoseConfig

logger = logging.getLogger(__name__)

RAD_TO_001DEG = 180000.0 / math.pi
_001DEG_TO_RAD = math.pi / 180000.0
M_TO_001MM = 1_000_000.0
_001MM_TO_M = 1.0 / 1_000_000.0


def _clamp(x: float, lo: float, hi: float) -> float:
    return min(max(x, lo), hi)


class PiperFollowerEndPose(Robot):
    """LeRobot Robot plugin for PiPER follower arm using EndPoseCtrl."""

    config_class = PiperFollowerEndPoseConfig
    name = "piper_follower_endpose"

    def __init__(self, config: PiperFollowerEndPoseConfig):
        super().__init__(config)
        self.config = config
        self._arm = None
        self._connected = False
        self._endpose_mode_active = False
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {
            "joint_1.pos": float,
            "joint_2.pos": float,
            "joint_3.pos": float,
            "joint_4.pos": float,
            "joint_5.pos": float,
            "joint_6.pos": float,
            "gripper.pos": float,
            "endpose.x": float,
            "endpose.y": float,
            "endpose.z": float,
            "endpose.roll": float,
            "endpose.pitch": float,
            "endpose.yaw": float,
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {
            "target_x": float,
            "target_y": float,
            "target_z": float,
            "target_roll": float,
            "target_pitch": float,
            "target_yaw": float,
            "gripper.pos": float,
        }

    @property
    def is_connected(self) -> bool:
        return self._connected and all(cam.is_connected for cam in self.cameras.values())

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        from piper_sdk import C_PiperInterface_V2

        self._arm = C_PiperInterface_V2(self.config.can_name, self.config.judge_flag)
        self._arm.ConnectPort()
        self._connected = True
        for cam in self.cameras.values():
            cam.connect()
        self.configure()
        logger.info("%s connected on %s", self.name, self.config.can_name)

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return

    def _wait_enable(self) -> None:
        deadline = time.time() + self.config.startup_enable_timeout_s
        while time.time() < deadline:
            if self._arm.EnablePiper():
                return
            time.sleep(0.05)
        raise RuntimeError(
            f"PiPER follower enable timeout after {self.config.startup_enable_timeout_s:.1f}s"
        )

    @check_if_not_connected
    def configure(self) -> None:
        # Set a sane default mode; will switch to EndPose mode in send_action.
        self._arm.MotionCtrl_2(0x01, 0x01, int(self.config.speed_ratio), 0x00)
        self._wait_enable()
        self._arm.GripperCtrl(0, int(self.config.gripper_effort), 0x01, 0x00)

    def _read_joint_rad(self) -> list[float]:
        joints = self._arm.GetArmJointMsgs().joint_state
        return [
            float(joints.joint_1) * _001DEG_TO_RAD,
            float(joints.joint_2) * _001DEG_TO_RAD,
            float(joints.joint_3) * _001DEG_TO_RAD,
            float(joints.joint_4) * _001DEG_TO_RAD,
            float(joints.joint_5) * _001DEG_TO_RAD,
            float(joints.joint_6) * _001DEG_TO_RAD,
        ]

    def _read_gripper_ratio(self) -> float:
        raw = float(self._arm.GetArmGripperMsgs().gripper_state.grippers_angle) * _001MM_TO_M
        return _clamp(raw / self.config.gripper_opening_m, 0.0, 1.0)

    def _read_endpose(self) -> tuple[float, float, float, float, float, float]:
        end_pose = self._arm.GetArmEndPoseMsgs().end_pose
        return (
            float(end_pose.X_axis) * _001MM_TO_M,
            float(end_pose.Y_axis) * _001MM_TO_M,
            float(end_pose.Z_axis) * _001MM_TO_M,
            float(end_pose.RX_axis) * _001DEG_TO_RAD,
            float(end_pose.RY_axis) * _001DEG_TO_RAD,
            float(end_pose.RZ_axis) * _001DEG_TO_RAD,
        )

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        joints = self._read_joint_rad()
        gripper = self._read_gripper_ratio()
        x, y, z, roll, pitch, yaw = self._read_endpose()
        obs_dict: RobotObservation = {
            "joint_1.pos": joints[0],
            "joint_2.pos": joints[1],
            "joint_3.pos": joints[2],
            "joint_4.pos": joints[3],
            "joint_5.pos": joints[4],
            "joint_6.pos": joints[5],
            "gripper.pos": gripper,
            "endpose.x": x,
            "endpose.y": y,
            "endpose.z": z,
            "endpose.roll": roll,
            "endpose.pitch": pitch,
            "endpose.yaw": yaw,
        }
        for cam_key, cam in self.cameras.items():
            obs_dict[cam_key] = cam.async_read()
        return obs_dict

    def _set_gripper(self, ratio: float) -> float:
        ratio = _clamp(float(ratio), 0.0, 1.0)
        stroke_001mm = int(round(ratio * self.config.gripper_opening_m * M_TO_001MM))
        self._arm.GripperCtrl(abs(stroke_001mm), int(self.config.gripper_effort), 0x01, 0x00)
        return ratio

    def _send_endpose_action(self, action: RobotAction) -> RobotAction:
        missing = [
            k
            for k in ("target_x", "target_y", "target_z", "target_roll", "target_pitch", "target_yaw")
            if k not in action
        ]
        if missing:
            raise ValueError(
                f"Missing EndPose action keys: {missing}. "
                "Expected target_x,target_y,target_z,target_roll,target_pitch,target_yaw"
            )

        x = float(action["target_x"])
        y = float(action["target_y"])
        z = float(action["target_z"])
        roll = float(action["target_roll"])
        pitch = float(action["target_pitch"])
        yaw = float(action["target_yaw"])

        # Switch to EndPose control mode once, then keep sending Cartesian targets.
        if not self._endpose_mode_active:
            self._arm.MotionCtrl_2(0x01, 0x00, int(self.config.speed_ratio), 0x00)
            self._endpose_mode_active = True
        self._arm.EndPoseCtrl(
            int(round(x * M_TO_001MM)),
            int(round(y * M_TO_001MM)),
            int(round(z * M_TO_001MM)),
            int(round(roll * RAD_TO_001DEG)),
            int(round(pitch * RAD_TO_001DEG)),
            int(round(yaw * RAD_TO_001DEG)),
        )

        sent: RobotAction = {
            "target_x": x,
            "target_y": y,
            "target_z": z,
            "target_roll": roll,
            "target_pitch": pitch,
            "target_yaw": yaw,
        }
        if "gripper.pos" in action:
            sent["gripper.pos"] = self._set_gripper(float(action["gripper.pos"]))
        return sent

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        if any(
            k in action
            for k in ("target_x", "target_y", "target_z", "target_roll", "target_pitch", "target_yaw")
        ):
            return self._send_endpose_action(action)

        raise ValueError(
            "Unsupported action schema for piper_follower_endpose. "
            "Use target_x,target_y,target_z,target_roll,target_pitch,target_yaw (+ optional gripper.pos)."
        )

    @check_if_not_connected
    def disconnect(self) -> None:
        try:
            if self.config.disable_on_disconnect:
                self._arm.DisableArm(7)
        finally:
            for cam in self.cameras.values():
                cam.disconnect()
            if self._arm is not None:
                self._arm.DisconnectPort()
            self._endpose_mode_active = False
            self._connected = False
            logger.info("%s disconnected", self.name)
