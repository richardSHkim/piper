#!/usr/bin/env python

from dataclasses import dataclass, field

import numpy as np

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import ProcessorStepRegistry, RobotAction, RobotActionProcessorStep, TransitionKey


def _clamp(x: float, lo: float, hi: float) -> float:
    return min(max(x, lo), hi)


def _normalize_angle(a: float) -> float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=np.float64,
    )


def _rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(rotvec)
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = rotvec / theta
    k = _skew(axis)
    return np.eye(3, dtype=np.float64) + np.sin(theta) * k + (1.0 - np.cos(theta)) * (k @ k)


def _matrix_to_rotvec(rot: np.ndarray) -> np.ndarray:
    tr = np.trace(rot)
    c = _clamp((tr - 1.0) * 0.5, -1.0, 1.0)
    theta = np.arccos(c)
    if theta < 1e-12:
        return np.zeros(3, dtype=np.float64)
    w = np.array(
        [
            rot[2, 1] - rot[1, 2],
            rot[0, 2] - rot[2, 0],
            rot[1, 0] - rot[0, 1],
        ],
        dtype=np.float64,
    )
    return (theta / (2.0 * np.sin(theta))) * w


def _quat_to_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    q = q / n
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


class PiperNumericalIK:
    """Lightweight FK + numerical Jacobian IK for PiPER 6-DoF arm."""

    def __init__(self):
        # Modified DH parameters: [alpha, a, d, theta_offset]
        self.dh = [
            (0.0, 0.0, 0.123, 0.0),
            (-np.pi / 2.0, 0.0, 0.0, -172.22 / 180.0 * np.pi),
            (0.0, 0.28503, 0.0, -102.78 / 180.0 * np.pi),
            (np.pi / 2.0, -0.021984, 0.25075, 0.0),
            (-np.pi / 2.0, 0.0, 0.0, 0.0),
            (np.pi / 2.0, 0.0, 0.211, 0.0),
        ]
        self.joint_limits = np.array(
            [
                [-2.618, 2.618],
                [0.0, np.pi],
                [-np.pi, 0.0],
                [-2.967, 2.967],
                [-1.2, 1.2],
                [-1.22, 1.22],
            ],
            dtype=np.float64,
        )

    def _modified_dh(self, alpha: float, a: float, d: float, theta: float) -> np.ndarray:
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        return np.array(
            [
                [ct, -st, 0.0, a],
                [st * ca, ct * ca, -sa, -sa * d],
                [st * sa, ct * sa, ca, ca * d],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def fk(self, joints: np.ndarray) -> np.ndarray:
        t = np.eye(4, dtype=np.float64)
        for i, (alpha, a, d, theta_off) in enumerate(self.dh):
            t = t @ self._modified_dh(alpha, a, d, joints[i] + theta_off)
        return t

    def _pose_error(self, current: np.ndarray, target: np.ndarray) -> np.ndarray:
        e = np.zeros(6, dtype=np.float64)
        e[:3] = target[:3, 3] - current[:3, 3]
        re = target[:3, :3] @ current[:3, :3].T
        e[3:] = _matrix_to_rotvec(re)
        return e

    def _numerical_jacobian(self, joints: np.ndarray, delta: float = 1e-6) -> np.ndarray:
        j = np.zeros((6, 6), dtype=np.float64)
        t0 = self.fk(joints)
        p0 = t0[:3, 3].copy()
        r0 = t0[:3, :3].copy()
        for i in range(6):
            q = joints.copy()
            q[i] += delta
            ti = self.fk(q)
            j[:3, i] = (ti[:3, 3] - p0) / delta
            dr = ti[:3, :3] @ r0.T
            j[3:, i] = _matrix_to_rotvec(dr) / delta
        return j

    def solve(
        self,
        initial_q: np.ndarray,
        target_pose: np.ndarray,
        max_iterations: int = 50,
        damping: float = 0.08,
        max_delta_q: float = 0.08,
        pos_tol: float = 2e-3,
        rot_tol: float = 2e-2,
    ) -> np.ndarray:
        q = initial_q.copy().astype(np.float64)
        for _ in range(max_iterations):
            curr = self.fk(q)
            e = self._pose_error(curr, target_pose)
            if np.linalg.norm(e[:3]) < pos_tol and np.linalg.norm(e[3:]) < rot_tol:
                return q

            jac = self._numerical_jacobian(q)
            jj_t = jac @ jac.T
            jj_t += (damping * damping) * np.eye(6, dtype=np.float64)
            dq = jac.T @ np.linalg.solve(jj_t, e)
            dq = np.clip(dq, -max_delta_q, max_delta_q)
            q = q + dq
            for i in range(6):
                q[i] = _normalize_angle(q[i])
                q[i] = np.clip(q[i], self.joint_limits[i, 0], self.joint_limits[i, 1])

        raise RuntimeError("IK did not converge")


@ProcessorStepRegistry.register("map_pika_action_to_piper_joints")
@dataclass
class MapPikaActionToPiperJoints(RobotActionProcessorStep):
    """
    Maps PIKA absolute tracker pose stream to PiPER joint targets by integrating
    delta EE pose and solving IK.
    """

    linear_scale: float = 1.0
    angular_scale: float = 1.0
    max_delta_pos_m: float = 0.03
    max_delta_rot_rad: float = 0.30
    damping: float = 0.08
    max_iterations: int = 50
    max_delta_q: float = 0.08
    pos_tol: float = 2e-3
    rot_tol: float = 2e-2
    solve_ik: bool = True
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

    def _read_observation_joints_from_obs(self, obs) -> np.ndarray:
        if not isinstance(obs, dict):
            return np.zeros(6, dtype=np.float64)
        keys = [f"joint_{i}.pos" for i in range(1, 7)]
        if not all(k in obs for k in keys):
            return np.zeros(6, dtype=np.float64)
        return np.array([float(obs[k]) for k in keys], dtype=np.float64)

    def _read_observation_joints(self) -> np.ndarray:
        obs = self.transition.get(TransitionKey.OBSERVATION)
        return self._read_observation_joints_from_obs(obs)

    def _build_action(self, q: np.ndarray, gripper: float) -> RobotAction:
        out: RobotAction = {f"joint_{i + 1}.pos": float(q[i]) for i in range(6)}
        out["gripper.pos"] = _clamp(float(gripper), 0.0, 1.0)
        return out

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
        # Tracker-to-robot frame remapping from empirical setup:
        #   tracker +Y corresponds to robot -Y.
        dp = np.array([dp_tracker[0], -dp_tracker[1], dp_tracker[2]], dtype=np.float64)
        drot = np.array([drot_tracker[0], -drot_tracker[1], drot_tracker[2]], dtype=np.float64)
        return dp, drot

    def rebase_reference(self, raw_action: RobotAction, observation: dict | None = None) -> None:
        """
        Set the current PIKA absolute pose as new teleop reference and align robot target pose
        to the current robot observation. This avoids jumps when teleoperation is armed.
        """
        pos, rot = self._parse_pose(raw_action)
        if observation is None:
            q = self._q.copy() if self._q is not None else np.zeros(6, dtype=np.float64)
        else:
            q = self._read_observation_joints_from_obs(observation)

        self._q = q
        self._target_pose = self._ik.fk(self._q)
        self._last_input_pos = pos
        self._last_input_rot = rot
        self._is_initialized = True

    def get_target_pose(self) -> np.ndarray | None:
        if self._target_pose is None:
            return None
        return self._target_pose.copy()

    def action(self, action: RobotAction) -> RobotAction:
        pos, rot = self._parse_pose(action)
        gripper = float(action.get("pika.gripper.pos", 0.0))

        if not self._is_initialized:
            self._q = self._read_observation_joints()
            self._target_pose = self._ik.fk(self._q)
            self._last_input_pos = pos
            self._last_input_rot = rot
            self._is_initialized = True
            return self._build_action(self._q, gripper)

        assert self._q is not None
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

        if self.solve_ik:
            try:
                self._q = self._ik.solve(
                    initial_q=self._q,
                    target_pose=self._target_pose,
                    max_iterations=self.max_iterations,
                    damping=self.damping,
                    max_delta_q=self.max_delta_q,
                    pos_tol=self.pos_tol,
                    rot_tol=self.rot_tol,
                )
            except RuntimeError:
                # Keep last valid q if IK fails for this frame.
                pass

        self._last_input_pos = pos
        self._last_input_rot = rot
        return self._build_action(self._q, gripper)

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

        for i in range(1, 7):
            action_ft[f"joint_{i}.pos"] = PolicyFeature(type=FeatureType.ACTION, shape=(1,))
        action_ft["gripper.pos"] = PolicyFeature(type=FeatureType.ACTION, shape=(1,))
        return out
