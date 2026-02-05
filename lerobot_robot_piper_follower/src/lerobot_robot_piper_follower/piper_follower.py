#!/usr/bin/env python

import logging
import math
import time
from functools import cached_property

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.processor import RobotAction, RobotObservation
from lerobot.robots.robot import Robot
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from .config_piper_follower import PiperFollowerConfig

logger = logging.getLogger(__name__)

RAD_TO_001DEG = 180000.0 / math.pi
_001DEG_TO_RAD = math.pi / 180000.0
M_TO_001MM = 1_000_000.0
_001MM_TO_M = 1.0 / 1_000_000.0


def _clamp(x: float, lo: float, hi: float) -> float:
    return min(max(x, lo), hi)


class PiperFollower(Robot):
    """LeRobot Robot plugin for PiPER follower arm."""

    config_class = PiperFollowerConfig
    name = "piper_follower"

    def __init__(self, config: PiperFollowerConfig):
        super().__init__(config)
        self.config = config
        self._arm = None
        self._connected = False
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
        return self._motors_ft

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

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        joints = self._read_joint_rad()
        gripper = self._read_gripper_ratio()
        obs_dict: RobotObservation = {
            "joint_1.pos": joints[0],
            "joint_2.pos": joints[1],
            "joint_3.pos": joints[2],
            "joint_4.pos": joints[3],
            "joint_5.pos": joints[4],
            "joint_6.pos": joints[5],
            "gripper.pos": gripper,
        }
        for cam_key, cam in self.cameras.items():
            obs_dict[cam_key] = cam.async_read()

        return obs_dict

    def _set_gripper(self, ratio: float) -> float:
        ratio = _clamp(float(ratio), 0.0, 1.0)
        stroke_001mm = int(round(ratio * self.config.gripper_opening_m * M_TO_001MM))
        self._arm.GripperCtrl(abs(stroke_001mm), int(self.config.gripper_effort), 0x01, 0x00)
        return ratio

    def _send_joint_action(self, action: RobotAction) -> RobotAction:
        curr = self._read_joint_rad()
        targets = [
            float(action.get("joint_1.pos", curr[0])),
            float(action.get("joint_2.pos", curr[1])),
            float(action.get("joint_3.pos", curr[2])),
            float(action.get("joint_4.pos", curr[3])),
            float(action.get("joint_5.pos", curr[4])),
            float(action.get("joint_6.pos", curr[5])),
        ]

        self._arm.MotionCtrl_2(0x01, 0x01, int(self.config.speed_ratio), 0x00)
        self._arm.JointCtrl(
            int(round(targets[0] * RAD_TO_001DEG)),
            int(round(targets[1] * RAD_TO_001DEG)),
            int(round(targets[2] * RAD_TO_001DEG)),
            int(round(targets[3] * RAD_TO_001DEG)),
            int(round(targets[4] * RAD_TO_001DEG)),
            int(round(targets[5] * RAD_TO_001DEG)),
        )

        sent: RobotAction = {f"joint_{i + 1}.pos": targets[i] for i in range(6)}
        if "gripper.pos" in action:
            sent["gripper.pos"] = self._set_gripper(float(action["gripper.pos"]))
        return sent

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        if any(
            k in action
            for k in (
                "joint_1.pos",
                "joint_2.pos",
                "joint_3.pos",
                "joint_4.pos",
                "joint_5.pos",
                "joint_6.pos",
            )
        ):
            return self._send_joint_action(action)

        raise ValueError(
            "Unsupported action schema for piper_follower. "
            "Use joint_1.pos..joint_6.pos (+ optional gripper.pos)."
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
            self._connected = False
            logger.info("%s disconnected", self.name)
