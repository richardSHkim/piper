#!/usr/bin/env python

import logging
import math
import time
from functools import cached_property

from lerobot.processor import RobotAction, RobotObservation
from lerobot.robots.robot import Robot
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from .config_piper_follower import PiperFollowerConfig

logger = logging.getLogger(__name__)

RAD_TO_001DEG = 180000.0 / math.pi
_001DEG_TO_RAD = math.pi / 180000.0
M_TO_001MM = 1_000_000.0
_001MM_TO_M = 1.0 / 1_000_000.0
RAD_TO_001DEG_EE = RAD_TO_001DEG
_001DEG_TO_RAD_EE = _001DEG_TO_RAD


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
        self.cameras = {}

    @cached_property
    def observation_features(self) -> dict[str, type]:
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

    def _read_ee_pose(self) -> tuple[float, float, float, float, float, float]:
        ee = self._arm.GetArmEndPoseMsgs().end_pose
        return (
            float(ee.X_axis) * _001MM_TO_M,
            float(ee.Y_axis) * _001MM_TO_M,
            float(ee.Z_axis) * _001MM_TO_M,
            float(ee.RX_axis) * _001DEG_TO_RAD_EE,
            float(ee.RY_axis) * _001DEG_TO_RAD_EE,
            float(ee.RZ_axis) * _001DEG_TO_RAD_EE,
        )

    def _read_gripper_ratio(self) -> float:
        raw = float(self._arm.GetArmGripperMsgs().gripper_state.grippers_angle) * _001MM_TO_M
        return _clamp(raw / self.config.gripper_opening_m, 0.0, 1.0)

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        joints = self._read_joint_rad()
        gripper = self._read_gripper_ratio()
        return {
            "joint_1.pos": joints[0],
            "joint_2.pos": joints[1],
            "joint_3.pos": joints[2],
            "joint_4.pos": joints[3],
            "joint_5.pos": joints[4],
            "joint_6.pos": joints[5],
            "gripper.pos": gripper,
        }

    def _set_gripper(self, ratio: float) -> float:
        ratio = _clamp(float(ratio), 0.0, 1.0)
        stroke_001mm = int(round(ratio * self.config.gripper_opening_m * M_TO_001MM))
        self._arm.GripperCtrl(abs(stroke_001mm), int(self.config.gripper_effort), 0x01, 0x00)
        return ratio

    def _send_delta_ee(self, action: RobotAction) -> RobotAction:
        x, y, z, rx, ry, rz = self._read_ee_pose()

        dx = _clamp(
            float(action.get("delta_x", action.get("target_x", 0.0))),
            -self.config.max_delta_translation_m,
            self.config.max_delta_translation_m,
        )
        dy = _clamp(
            float(action.get("delta_y", action.get("target_y", 0.0))),
            -self.config.max_delta_translation_m,
            self.config.max_delta_translation_m,
        )
        dz = _clamp(
            float(action.get("delta_z", action.get("target_z", 0.0))),
            -self.config.max_delta_translation_m,
            self.config.max_delta_translation_m,
        )
        drx = _clamp(
            float(action.get("delta_rx", action.get("target_wx", 0.0))),
            -self.config.max_delta_rotation_rad,
            self.config.max_delta_rotation_rad,
        )
        dry = _clamp(
            float(action.get("delta_ry", action.get("target_wy", 0.0))),
            -self.config.max_delta_rotation_rad,
            self.config.max_delta_rotation_rad,
        )
        drz = _clamp(
            float(action.get("delta_rz", action.get("target_wz", 0.0))),
            -self.config.max_delta_rotation_rad,
            self.config.max_delta_rotation_rad,
        )

        tx = x + dx
        ty = y + dy
        tz = z + dz
        trx = rx + drx
        try_ = ry + dry
        trz = rz + drz

        self._arm.MotionCtrl_2(0x01, 0x00, int(self.config.speed_ratio), 0x00)
        self._arm.EndPoseCtrl(
            int(round(tx * M_TO_001MM)),
            int(round(ty * M_TO_001MM)),
            int(round(tz * M_TO_001MM)),
            int(round(trx * RAD_TO_001DEG_EE)),
            int(round(try_ * RAD_TO_001DEG_EE)),
            int(round(trz * RAD_TO_001DEG_EE)),
        )

        sent: RobotAction = {
            "ee.x": tx,
            "ee.y": ty,
            "ee.z": tz,
            "ee.rx": trx,
            "ee.ry": try_,
            "ee.rz": trz,
        }
        if "gripper" in action:
            sent["gripper.pos"] = self._set_gripper(float(action["gripper"]))
        return sent

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        if any(k in action for k in ("delta_x", "delta_y", "delta_z", "target_x", "target_y", "target_z")):
            return self._send_delta_ee(action)

        raise ValueError(
            "Unsupported action schema for piper_follower. "
            "Use delta_(x,y,z,rx,ry,rz) (+ optional gripper)."
        )

    @check_if_not_connected
    def disconnect(self) -> None:
        try:
            if self.config.disable_on_disconnect:
                self._arm.DisableArm(7)
        finally:
            self._arm.DisconnectPort()
            self._connected = False
            logger.info("%s disconnected", self.name)
