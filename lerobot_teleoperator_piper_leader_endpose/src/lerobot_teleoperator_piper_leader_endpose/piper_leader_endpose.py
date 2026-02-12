#!/usr/bin/env python

import logging
import math
from functools import cached_property
from typing import Any

from lerobot.processor import RobotAction
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from .config_piper_leader_endpose import PiperLeaderEndPoseConfig

logger = logging.getLogger(__name__)
_001DEG_TO_RAD = math.pi / 180000.0
_001MM_TO_M = 1.0 / 1_000_000.0


def _clamp(x: float, lo: float, hi: float) -> float:
    return min(max(x, lo), hi)


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

        end_pose = self._arm.GetArmEndPoseMsgs().end_pose

        if self.config.source_mode == "control":
            gripper = self._arm.GetArmGripperCtrl().gripper_ctrl
            gripper_raw = float(gripper.grippers_angle)
        else:
            gripper = self._arm.GetArmGripperMsgs().gripper_state
            gripper_raw = float(gripper.grippers_angle)

        return {
            "target_x": float(end_pose.X_axis) * _001MM_TO_M,
            "target_y": float(end_pose.Y_axis) * _001MM_TO_M,
            "target_z": float(end_pose.Z_axis) * _001MM_TO_M,
            "target_roll": float(end_pose.RX_axis) * _001DEG_TO_RAD,
            "target_pitch": float(end_pose.RY_axis) * _001DEG_TO_RAD,
            "target_yaw": float(end_pose.RZ_axis) * _001DEG_TO_RAD,
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
