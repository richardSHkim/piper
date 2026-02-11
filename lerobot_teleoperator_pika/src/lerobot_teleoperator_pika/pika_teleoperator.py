#!/usr/bin/env python

import logging
from functools import cached_property
from typing import Any

from lerobot.processor import RobotAction
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from .config_pika_teleoperator import PikaTeleoperatorConfig

logger = logging.getLogger(__name__)


def _clamp(x: float, lo: float, hi: float) -> float:
    return min(max(x, lo), hi)


class PikaTeleoperator(Teleoperator):
    """LeRobot Teleoperator plugin for AgileX PIKA Sense."""

    config_class = PikaTeleoperatorConfig
    name = "pika_teleoperator"

    def __init__(self, config: PikaTeleoperatorConfig):
        super().__init__(config)
        self.config = config
        self._sense = None
        self._connected = False
        self._last_pose = None

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {
            "pika.pos.x": float,
            "pika.pos.y": float,
            "pika.pos.z": float,
            "pika.rot.x": float,
            "pika.rot.y": float,
            "pika.rot.z": float,
            "pika.rot.w": float,
            "pika.gripper.pos": float,
            "pika.pose.valid": float,
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
        from pika import sense

        self._sense = sense(self.config.port)
        if self.config.tracker_config_path or self.config.tracker_lh_config or self.config.tracker_args:
            self._sense.set_vive_tracker_config(
                config_path=self.config.tracker_config_path,
                lh_config=self.config.tracker_lh_config,
                args=self.config.tracker_args or None,
            )

        if not self._sense.connect():
            raise RuntimeError(f"Failed to connect PIKA Sense on {self.config.port}")

        self._connected = True
        logger.info("%s connected on %s", self.name, self.config.port)

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return

    @check_if_not_connected
    def configure(self) -> None:
        return

    def _select_pose_data(self):
        pose = self._sense.get_pose(self.config.tracker_device) if self.config.tracker_device else self._sense.get_pose()
        if pose is None:
            return None
        if isinstance(pose, dict):
            if not pose:
                return None
            if self.config.tracker_device and self.config.tracker_device in pose:
                return pose[self.config.tracker_device]
            wm_keys = sorted([k for k in pose if k.startswith("WM")])
            if wm_keys:
                return pose[wm_keys[0]]
            first_key = sorted(pose.keys())[0]
            return pose[first_key]
        return pose

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        pose_data = self._select_pose_data()
        pose_valid = 1.0

        if pose_data is None:
            pose_valid = 0.0
            if self.config.require_pose and self._last_pose is None:
                raise RuntimeError("No tracker pose available from PIKA Sense.")
            if self._last_pose is None:
                pos = [0.0, 0.0, 0.0]
                rot = [0.0, 0.0, 0.0, 1.0]
            else:
                pos, rot = self._last_pose
        else:
            pos = [float(v) for v in pose_data.position]
            rot = [float(v) for v in pose_data.rotation]
            self._last_pose = (pos, rot)

        gripper_mm = float(self._sense.get_gripper_distance())
        denom = max(self.config.gripper_max_mm - self.config.gripper_min_mm, 1e-6)
        gripper_pos = _clamp((gripper_mm - self.config.gripper_min_mm) / denom, 0.0, 1.0)

        return {
            "pika.pos.x": pos[0],
            "pika.pos.y": pos[1],
            "pika.pos.z": pos[2],
            "pika.rot.x": rot[0],
            "pika.rot.y": rot[1],
            "pika.rot.z": rot[2],
            "pika.rot.w": rot[3],
            "pika.gripper.pos": gripper_pos,
            "pika.pose.valid": pose_valid,
        }

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        del feedback
        return

    @check_if_not_connected
    def disconnect(self) -> None:
        self._sense.disconnect()
        self._connected = False
        logger.info("%s disconnected", self.name)
