#!/usr/bin/env python

import logging
from functools import cached_property
from typing import Any

from lerobot.processor import RobotAction
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from .config_piper_leader import PiperLeaderConfig

logger = logging.getLogger(__name__)
_001DEG_TO_RAD = 3.141592653589793 / 180000.0
_001MM_TO_M = 1.0 / 1_000_000.0


class PiperLeader(Teleoperator):
    """LeRobot Teleoperator plugin for PiPER leader arm."""

    config_class = PiperLeaderConfig
    name = "piper_leader"

    def __init__(self, config: PiperLeaderConfig):
        super().__init__(config)
        self.config = config
        self._arm = None
        self._connected = False

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {
            "joint_1.pos": float,
            "joint_2.pos": float,
            "joint_3.pos": float,
            "joint_4.pos": float,
            "joint_5.pos": float,
            "joint_6.pos": float,
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
        if self.config.source_mode == "control":
            joints = self._arm.GetArmJointCtrl().joint_ctrl
            gripper = self._arm.GetArmGripperCtrl().gripper_ctrl
            gripper_raw = float(gripper.grippers_angle)
        else:
            joints = self._arm.GetArmJointMsgs().joint_state
            gripper = self._arm.GetArmGripperMsgs().gripper_state
            gripper_raw = float(gripper.grippers_angle)

        return {
            "joint_1.pos": float(joints.joint_1) * _001DEG_TO_RAD,
            "joint_2.pos": float(joints.joint_2) * _001DEG_TO_RAD,
            "joint_3.pos": float(joints.joint_3) * _001DEG_TO_RAD,
            "joint_4.pos": float(joints.joint_4) * _001DEG_TO_RAD,
            "joint_5.pos": float(joints.joint_5) * _001DEG_TO_RAD,
            "joint_6.pos": float(joints.joint_6) * _001DEG_TO_RAD,
            "gripper.pos": min(max((gripper_raw * _001MM_TO_M) / self.config.gripper_opening_m, 0.0), 1.0),
        }

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        del feedback
        return

    @check_if_not_connected
    def disconnect(self) -> None:
        self._arm.DisconnectPort()
        self._connected = False
        logger.info("%s disconnected", self.name)
