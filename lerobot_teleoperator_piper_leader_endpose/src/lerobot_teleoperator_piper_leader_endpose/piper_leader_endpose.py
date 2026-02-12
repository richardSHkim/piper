#!/usr/bin/env python

import logging
import math
from functools import cached_property
from typing import Any

import numpy as np

from lerobot.processor import RobotAction
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from .config_piper_leader_endpose import PiperLeaderEndPoseConfig

logger = logging.getLogger(__name__)
_001DEG_TO_RAD = math.pi / 180000.0
_001MM_TO_M = 1.0 / 1_000_000.0


def _clamp(x: float, lo: float, hi: float) -> float:
    return min(max(x, lo), hi)


class _PiperFK:
    """Simple Modified-DH FK used as robust pose source for hand-guiding mode."""

    def __init__(self) -> None:
        self.dh = [
            (0.0, 0.0, 0.123, 0.0),
            (-math.pi / 2.0, 0.0, 0.0, -172.22 / 180.0 * math.pi),
            (0.0, 0.28503, 0.0, -102.78 / 180.0 * math.pi),
            (math.pi / 2.0, -0.021984, 0.25075, 0.0),
            (-math.pi / 2.0, 0.0, 0.0, 0.0),
            (math.pi / 2.0, 0.0, 0.211, 0.0),
        ]

    @staticmethod
    def _modified_dh(alpha: float, a: float, d: float, theta: float) -> np.ndarray:
        ct, st = math.cos(theta), math.sin(theta)
        ca, sa = math.cos(alpha), math.sin(alpha)
        return np.array(
            [
                [ct, -st, 0.0, a],
                [st * ca, ct * ca, -sa, -sa * d],
                [st * sa, ct * sa, ca, ca * d],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _matrix_to_rpy_xyz(rot: np.ndarray) -> tuple[float, float, float]:
        sy = math.sqrt(rot[0, 0] * rot[0, 0] + rot[1, 0] * rot[1, 0])
        singular = sy < 1e-9
        if not singular:
            roll = math.atan2(rot[2, 1], rot[2, 2])
            pitch = math.atan2(-rot[2, 0], sy)
            yaw = math.atan2(rot[1, 0], rot[0, 0])
        else:
            roll = math.atan2(-rot[1, 2], rot[1, 1])
            pitch = math.atan2(-rot[2, 0], sy)
            yaw = 0.0
        return roll, pitch, yaw

    def pose_from_joints(self, joints_rad: list[float]) -> tuple[float, float, float, float, float, float]:
        t = np.eye(4, dtype=np.float64)
        for i, (alpha, a, d, theta_off) in enumerate(self.dh):
            t = t @ self._modified_dh(alpha, a, d, joints_rad[i] + theta_off)
        roll, pitch, yaw = self._matrix_to_rpy_xyz(t[:3, :3])
        return float(t[0, 3]), float(t[1, 3]), float(t[2, 3]), roll, pitch, yaw


class PiperLeaderEndPose(Teleoperator):
    """LeRobot Teleoperator plugin for PiPER leader arm with EndPose output."""

    config_class = PiperLeaderEndPoseConfig
    name = "piper_leader_endpose"

    def __init__(self, config: PiperLeaderEndPoseConfig):
        super().__init__(config)
        self.config = config
        self._arm = None
        self._connected = False
        self._warned_control_pose = False
        self._fk = _PiperFK()

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

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        from piper_sdk import C_PiperInterface_V2

        self._arm = C_PiperInterface_V2(self.config.can_name, self.config.judge_flag)
        self._arm.ConnectPort()
        self._connected = True
        self.configure()
        logger.info("%s connected on %s", self.name, self.config.can_name)

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return

    @check_if_not_connected
    def configure(self) -> None:
        if self.config.pose_source not in {"joint_fk", "endpose_feedback"}:
            raise ValueError(
                f"Unsupported pose_source={self.config.pose_source!r}. "
                "Use 'joint_fk' or 'endpose_feedback'."
            )
        if self.config.hand_guiding:
            self._arm.DisableArm(7)

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        if self.config.source_mode == "control" and not self._warned_control_pose:
            logger.warning(
                "source_mode=control selected, but PiPER SDK exposes pose from feedback frames. "
                "Using GetArmEndPoseMsgs for pose and control frames only for gripper."
            )
            self._warned_control_pose = True

        if self.config.pose_source == "endpose_feedback":
            end_pose = self._arm.GetArmEndPoseMsgs().end_pose
            target_x = float(end_pose.X_axis) * _001MM_TO_M
            target_y = float(end_pose.Y_axis) * _001MM_TO_M
            target_z = float(end_pose.Z_axis) * _001MM_TO_M
            target_roll = float(end_pose.RX_axis) * _001DEG_TO_RAD
            target_pitch = float(end_pose.RY_axis) * _001DEG_TO_RAD
            target_yaw = float(end_pose.RZ_axis) * _001DEG_TO_RAD
        else:
            joints = self._arm.GetArmJointMsgs().joint_state
            joints_rad = [
                float(joints.joint_1) * _001DEG_TO_RAD,
                float(joints.joint_2) * _001DEG_TO_RAD,
                float(joints.joint_3) * _001DEG_TO_RAD,
                float(joints.joint_4) * _001DEG_TO_RAD,
                float(joints.joint_5) * _001DEG_TO_RAD,
                float(joints.joint_6) * _001DEG_TO_RAD,
            ]
            target_x, target_y, target_z, target_roll, target_pitch, target_yaw = self._fk.pose_from_joints(
                joints_rad
            )

        if self.config.source_mode == "control":
            gripper = self._arm.GetArmGripperCtrl().gripper_ctrl
            gripper_raw = float(gripper.grippers_angle)
        else:
            gripper = self._arm.GetArmGripperMsgs().gripper_state
            gripper_raw = float(gripper.grippers_angle)

        return {
            "target_x": target_x,
            "target_y": target_y,
            "target_z": target_z,
            "target_roll": target_roll,
            "target_pitch": target_pitch,
            "target_yaw": target_yaw,
            "gripper.pos": _clamp((gripper_raw * _001MM_TO_M) / self.config.gripper_opening_m, 0.0, 1.0),
        }

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        del feedback
        return

    @check_if_not_connected
    def disconnect(self) -> None:
        self._arm.DisconnectPort()
        self._connected = False
        logger.info("%s disconnected", self.name)
