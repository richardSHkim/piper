#!/usr/bin/env python

import logging
import math
from functools import cached_property
from typing import Any

from lerobot.processor import RobotAction
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from .config_piper_leader import PiperLeaderConfig

logger = logging.getLogger(__name__)

_001DEG_TO_RAD = math.pi / 180000.0
_001MM_TO_M = 1.0 / 1_000_000.0


def _clamp(x: float, lo: float, hi: float) -> float:
    return min(max(x, lo), hi)


class PiperLeader(Teleoperator):
    """LeRobot Teleoperator plugin for PiPER leader arm."""

    config_class = PiperLeaderConfig
    name = "piper_leader"

    def __init__(self, config: PiperLeaderConfig):
        super().__init__(config)
        self.config = config
        self._arm = None
        self._connected = False
        self._last_ee: tuple[float, float, float, float, float, float] | None = None

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {
            "delta_x": float,
            "delta_y": float,
            "delta_z": float,
            "delta_rx": float,
            "delta_ry": float,
            "delta_rz": float,
            "gripper": float,
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
        self._last_ee = self._read_ee_pose()

    def _read_ee_pose(self) -> tuple[float, float, float, float, float, float]:
        ee = self._arm.GetArmEndPoseMsgs().end_pose
        return (
            float(ee.X_axis) * _001MM_TO_M,
            float(ee.Y_axis) * _001MM_TO_M,
            float(ee.Z_axis) * _001MM_TO_M,
            float(ee.RX_axis) * _001DEG_TO_RAD,
            float(ee.RY_axis) * _001DEG_TO_RAD,
            float(ee.RZ_axis) * _001DEG_TO_RAD,
        )

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        ee_now = self._read_ee_pose()
        if self._last_ee is None:
            self._last_ee = ee_now

        delta = [ee_now[i] - self._last_ee[i] for i in range(6)]
        self._last_ee = ee_now

        delta[0] = _clamp(delta[0], -self.config.max_delta_translation_m, self.config.max_delta_translation_m)
        delta[1] = _clamp(delta[1], -self.config.max_delta_translation_m, self.config.max_delta_translation_m)
        delta[2] = _clamp(delta[2], -self.config.max_delta_translation_m, self.config.max_delta_translation_m)
        delta[3] = _clamp(delta[3], -self.config.max_delta_rotation_rad, self.config.max_delta_rotation_rad)
        delta[4] = _clamp(delta[4], -self.config.max_delta_rotation_rad, self.config.max_delta_rotation_rad)
        delta[5] = _clamp(delta[5], -self.config.max_delta_rotation_rad, self.config.max_delta_rotation_rad)

        if self.config.source_mode == "control":
            gripper = self._arm.GetArmGripperCtrl().gripper_ctrl
            gripper_raw = float(gripper.grippers_angle)
        else:
            gripper = self._arm.GetArmGripperMsgs().gripper_state
            gripper_raw = float(gripper.grippers_angle)

        return {
            "delta_x": delta[0],
            "delta_y": delta[1],
            "delta_z": delta[2],
            "delta_rx": delta[3],
            "delta_ry": delta[4],
            "delta_rz": delta[5],
            "gripper": _clamp((gripper_raw * _001MM_TO_M) / self.config.gripper_opening_m, 0.0, 1.0),
        }

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        del feedback
        return

    @check_if_not_connected
    def disconnect(self) -> None:
        self._arm.DisconnectPort()
        self._connected = False
        logger.info("%s disconnected", self.name)
