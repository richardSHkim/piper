#!/usr/bin/env python

from dataclasses import dataclass, field

import numpy as np

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import ProcessorStepRegistry, RobotAction, RobotActionProcessorStep, TransitionKey

from .pika_to_piper_processor import (
    PiperNumericalIK,
    _matrix_to_rotvec,
    _quat_to_matrix,
    _rotvec_to_matrix,
)


@ProcessorStepRegistry.register("map_pika_action_to_piper_endpose")
@dataclass
class MapPikaActionToPiperEndPose(RobotActionProcessorStep):
    """
    Maps PIKA absolute tracker stream to PiPER EndPose action keys.

    Output schema:
      target_x, target_y, target_z (m)
      target_roll, target_pitch, target_yaw (rad)
      gripper.pos (0..1)
    """

    linear_scale: float = 1.0
    angular_scale: float = 1.0
    ignore_invalid_pose: bool = True
    max_delta_pos_m: float = 0.03
    max_delta_rot_rad: float = 0.30
    workspace_min_xyz: tuple[float, float, float] = (-0.45, -0.45, 0.02)
    workspace_max_xyz: tuple[float, float, float] = (0.70, 0.45, 0.75)
    _ik: PiperNumericalIK = field(init=False, repr=False)
    _is_initialized: bool = field(default=False, init=False, repr=False)
    _last_input_pos: np.ndarray | None = field(default=None, init=False, repr=False)
    _last_input_rot: np.ndarray | None = field(default=None, init=False, repr=False)
    _target_pose: np.ndarray | None = field(default=None, init=False, repr=False)
    _q: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._ik = PiperNumericalIK()

    def _read_observation_joints(self) -> np.ndarray:
        obs = self.transition.get(TransitionKey.OBSERVATION)
        if not isinstance(obs, dict):
            return np.zeros(6, dtype=np.float64)
        keys = [f"joint_{i}.pos" for i in range(1, 7)]
        if not all(k in obs for k in keys):
            return np.zeros(6, dtype=np.float64)
        return np.array([float(obs[k]) for k in keys], dtype=np.float64)

    def _rpy_to_matrix(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
        ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
        rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        return rz @ ry @ rx

    def _read_observation_endpose(self) -> np.ndarray | None:
        obs = self.transition.get(TransitionKey.OBSERVATION)
        if not isinstance(obs, dict):
            return None
        keys = (
            "endpose.x",
            "endpose.y",
            "endpose.z",
            "endpose.roll",
            "endpose.pitch",
            "endpose.yaw",
        )
        if not all(k in obs for k in keys):
            return None

        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = float(obs["endpose.x"])
        pose[1, 3] = float(obs["endpose.y"])
        pose[2, 3] = float(obs["endpose.z"])
        pose[:3, :3] = self._rpy_to_matrix(
            float(obs["endpose.roll"]),
            float(obs["endpose.pitch"]),
            float(obs["endpose.yaw"]),
        )
        return pose

    def _parse_pose(self, action: RobotAction) -> tuple[np.ndarray, np.ndarray]:
        pos = np.array(
            [
                float(action["pika.pos.x"]),
                float(action["pika.pos.y"]),
                float(action["pika.pos.z"]),
            ],
            dtype=np.float64,
        )
        rot = _quat_to_matrix(
            float(action["pika.rot.x"]),
            float(action["pika.rot.y"]),
            float(action["pika.rot.z"]),
            float(action["pika.rot.w"]),
        )
        return pos, rot

    def _map_delta_to_robot_frame(
        self, dp_tracker: np.ndarray, drot_tracker: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        dp = np.array([dp_tracker[0], -dp_tracker[1], dp_tracker[2]], dtype=np.float64)
        drot = np.array([drot_tracker[0], -drot_tracker[1], drot_tracker[2]], dtype=np.float64)
        return dp, drot

    def _rotation_matrix_to_rpy(self, rot: np.ndarray) -> tuple[float, float, float]:
        sy = float(np.clip(-rot[2, 0], -1.0, 1.0))
        pitch = float(np.arcsin(sy))
        cp = float(np.cos(pitch))
        if abs(cp) > 1e-6:
            roll = float(np.arctan2(rot[2, 1], rot[2, 2]))
            yaw = float(np.arctan2(rot[1, 0], rot[0, 0]))
        else:
            roll = 0.0
            yaw = float(np.arctan2(-rot[0, 1], rot[1, 1]))
        return roll, pitch, yaw

    def action(self, action: RobotAction) -> RobotAction:
        pose_valid = float(action.get("pika.pose.valid", 1.0)) >= 0.5
        pos, rot = self._parse_pose(action)
        gripper = float(action.get("pika.gripper.pos", 0.0))

        if self.ignore_invalid_pose and not pose_valid:
            if not self._is_initialized:
                self._q = self._read_observation_joints()
                self._target_pose = self._read_observation_endpose()
                if self._target_pose is None:
                    self._target_pose = self._ik.fk(self._q)
                self._is_initialized = True
            assert self._target_pose is not None
            roll, pitch, yaw = self._rotation_matrix_to_rpy(self._target_pose[:3, :3])
            return {
                "target_x": float(self._target_pose[0, 3]),
                "target_y": float(self._target_pose[1, 3]),
                "target_z": float(self._target_pose[2, 3]),
                "target_roll": roll,
                "target_pitch": pitch,
                "target_yaw": yaw,
                "gripper.pos": max(0.0, min(1.0, gripper)),
            }

        if not self._is_initialized:
            self._q = self._read_observation_joints()
            self._target_pose = self._read_observation_endpose()
            if self._target_pose is None:
                self._target_pose = self._ik.fk(self._q)
            self._last_input_pos = pos
            self._last_input_rot = rot
            self._is_initialized = True
        else:
            assert self._target_pose is not None
            assert self._last_input_pos is not None
            assert self._last_input_rot is not None
            dp_tracker = pos - self._last_input_pos
            drot_tracker = _matrix_to_rotvec(self._last_input_rot.T @ rot)
            dp_tracker = np.clip(dp_tracker, -self.max_delta_pos_m, self.max_delta_pos_m)
            drot_tracker = np.clip(drot_tracker, -self.max_delta_rot_rad, self.max_delta_rot_rad)
            dp_robot, drot_robot = self._map_delta_to_robot_frame(dp_tracker, drot_tracker)

            self._target_pose[:3, 3] += self.linear_scale * dp_robot
            self._target_pose[:3, 3] = np.clip(
                self._target_pose[:3, 3],
                np.array(self.workspace_min_xyz, dtype=np.float64),
                np.array(self.workspace_max_xyz, dtype=np.float64),
            )
            self._target_pose[:3, :3] = self._target_pose[:3, :3] @ _rotvec_to_matrix(
                self.angular_scale * drot_robot
            )

            self._last_input_pos = pos
            self._last_input_rot = rot

        assert self._target_pose is not None
        roll, pitch, yaw = self._rotation_matrix_to_rpy(self._target_pose[:3, :3])
        return {
            "target_x": float(self._target_pose[0, 3]),
            "target_y": float(self._target_pose[1, 3]),
            "target_z": float(self._target_pose[2, 3]),
            "target_roll": roll,
            "target_pitch": pitch,
            "target_yaw": yaw,
            "gripper.pos": max(0.0, min(1.0, gripper)),
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        out = {k: v.copy() for k, v in features.items()}
        action_ft = out[PipelineFeatureType.ACTION]
        for key in [
            "pika.pos.x",
            "pika.pos.y",
            "pika.pos.z",
            "pika.rot.x",
            "pika.rot.y",
            "pika.rot.z",
            "pika.rot.w",
            "pika.gripper.pos",
            "pika.pose.valid",
        ]:
            action_ft.pop(key, None)

        for key in ["target_x", "target_y", "target_z", "target_roll", "target_pitch", "target_yaw", "gripper.pos"]:
            action_ft[key] = PolicyFeature(type=FeatureType.ACTION, shape=(1,))
        return out
