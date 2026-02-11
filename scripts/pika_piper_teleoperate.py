#!/usr/bin/env python3

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat

import numpy as np
import rerun as rr

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.robots import Robot, RobotConfig, make_robot_from_config
from lerobot.teleoperators import Teleoperator, TeleoperatorConfig, make_teleoperator_from_config
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


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
    def __init__(self):
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
        max_iterations: int,
        damping: float,
        max_delta_q: float,
        pos_tol: float,
        rot_tol: float,
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


@dataclass
class MapperConfig:
    linear_scale: float = 1.0
    angular_scale: float = 1.0
    max_delta_pos_m: float = 0.03
    max_delta_rot_rad: float = 0.30
    damping: float = 0.08
    max_iterations: int = 50
    max_delta_q: float = 0.08
    pos_tol: float = 2e-3
    rot_tol: float = 2e-2
    workspace_min_xyz: tuple[float, float, float] = (-0.45, -0.45, 0.02)
    workspace_max_xyz: tuple[float, float, float] = (0.70, 0.45, 0.75)
    pos_axis_map: tuple[int, int, int] = (0, 1, 2)
    pos_axis_sign: tuple[float, float, float] = (1.0, -1.0, 1.0)
    rot_axis_map: tuple[int, int, int] = (0, 1, 2)
    rot_axis_sign: tuple[float, float, float] = (1.0, -1.0, 1.0)


class PikaToPiperJointMapper:
    def __init__(self, cfg: MapperConfig):
        self.cfg = cfg
        self.ik = PiperNumericalIK()
        self._is_initialized = False
        self._last_input_pos: np.ndarray | None = None
        self._last_input_rot: np.ndarray | None = None
        self._target_pose: np.ndarray | None = None
        self._q: np.ndarray | None = None
        self._last_ik_time_ms: float = 0.0
        self._last_ik_success: bool = False
        self._last_ik_error_pos_m: float = 0.0
        self._last_ik_error_rot_rad: float = 0.0
        self._last_ik_fail_reason: str = ""
        self._last_target_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._last_joint_limit_margin_rad: float = 0.0
        self._last_dp_tracker: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._last_dp_robot: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._last_drot_tracker: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._last_drot_robot: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def _read_observation_joints(self, obs: dict | None) -> np.ndarray:
        if not isinstance(obs, dict):
            return np.zeros(6, dtype=np.float64)
        keys = [f"joint_{i}.pos" for i in range(1, 7)]
        if not all(k in obs for k in keys):
            return np.zeros(6, dtype=np.float64)
        return np.array([float(obs[k]) for k in keys], dtype=np.float64)

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

    def _build_action(self, q: np.ndarray, gripper: float) -> RobotAction:
        out: RobotAction = {f"joint_{i + 1}.pos": float(q[i]) for i in range(6)}
        out["gripper.pos"] = _clamp(float(gripper), 0.0, 1.0)
        return out

    def _map_delta_to_robot_frame(self, dp_tracker: np.ndarray, drot_tracker: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dp = np.array(
            [
                self.cfg.pos_axis_sign[0] * dp_tracker[self.cfg.pos_axis_map[0]],
                self.cfg.pos_axis_sign[1] * dp_tracker[self.cfg.pos_axis_map[1]],
                self.cfg.pos_axis_sign[2] * dp_tracker[self.cfg.pos_axis_map[2]],
            ],
            dtype=np.float64,
        )
        drot = np.array(
            [
                self.cfg.rot_axis_sign[0] * drot_tracker[self.cfg.rot_axis_map[0]],
                self.cfg.rot_axis_sign[1] * drot_tracker[self.cfg.rot_axis_map[1]],
                self.cfg.rot_axis_sign[2] * drot_tracker[self.cfg.rot_axis_map[2]],
            ],
            dtype=np.float64,
        )
        return dp, drot

    def rebase_reference(self, raw_action: RobotAction, observation: dict | None = None) -> None:
        pos, rot = self._parse_pose(raw_action)
        self._q = self._read_observation_joints(observation)
        self._target_pose = self.ik.fk(self._q)
        self._last_input_pos = pos
        self._last_input_rot = rot
        self._is_initialized = True

    def map_action(self, raw_action: RobotAction, observation: dict | None) -> RobotAction:
        pose_valid = float(raw_action.get("pika.pose.valid", 1.0)) >= 0.5
        gripper = float(raw_action.get("pika.gripper.pos", 0.0))

        if not self._is_initialized:
            if pose_valid:
                self.rebase_reference(raw_action, observation)
            else:
                self._q = self._read_observation_joints(observation)
                self._target_pose = self.ik.fk(self._q)
                self._is_initialized = True
            assert self._q is not None
            return self._build_action(self._q, gripper)

        assert self._q is not None
        assert self._target_pose is not None

        if not pose_valid:
            return self._build_action(self._q, gripper)

        pos, rot = self._parse_pose(raw_action)
        assert self._last_input_pos is not None
        assert self._last_input_rot is not None

        dp_tracker = pos - self._last_input_pos
        drot_tracker = _matrix_to_rotvec(self._last_input_rot.T @ rot)

        dp_tracker = np.clip(dp_tracker, -self.cfg.max_delta_pos_m, self.cfg.max_delta_pos_m)
        drot_tracker = np.clip(drot_tracker, -self.cfg.max_delta_rot_rad, self.cfg.max_delta_rot_rad)
        dp_robot, drot_robot = self._map_delta_to_robot_frame(dp_tracker, drot_tracker)
        self._last_dp_tracker = (float(dp_tracker[0]), float(dp_tracker[1]), float(dp_tracker[2]))
        self._last_dp_robot = (float(dp_robot[0]), float(dp_robot[1]), float(dp_robot[2]))
        self._last_drot_tracker = (
            float(drot_tracker[0]),
            float(drot_tracker[1]),
            float(drot_tracker[2]),
        )
        self._last_drot_robot = (float(drot_robot[0]), float(drot_robot[1]), float(drot_robot[2]))

        self._target_pose[:3, 3] += self.cfg.linear_scale * dp_robot
        self._target_pose[:3, 3] = np.clip(
            self._target_pose[:3, 3],
            np.array(self.cfg.workspace_min_xyz, dtype=np.float64),
            np.array(self.cfg.workspace_max_xyz, dtype=np.float64),
        )
        self._target_pose[:3, :3] = self._target_pose[:3, :3] @ _rotvec_to_matrix(
            self.cfg.angular_scale * drot_robot
        )
        self._last_target_xyz = (
            float(self._target_pose[0, 3]),
            float(self._target_pose[1, 3]),
            float(self._target_pose[2, 3]),
        )

        ik_start = time.perf_counter()
        try:
            self._q = self.ik.solve(
                initial_q=self._q,
                target_pose=self._target_pose,
                max_iterations=self.cfg.max_iterations,
                damping=self.cfg.damping,
                max_delta_q=self.cfg.max_delta_q,
                pos_tol=self.cfg.pos_tol,
                rot_tol=self.cfg.rot_tol,
            )
            self._last_ik_success = True
            self._last_ik_fail_reason = ""
        except RuntimeError:
            self._last_ik_success = False
            curr = self.ik.fk(self._q)
            e = self.ik._pose_error(curr, self._target_pose)
            self._last_ik_error_pos_m = float(np.linalg.norm(e[:3]))
            self._last_ik_error_rot_rad = float(np.linalg.norm(e[3:]))
            margins = np.minimum(self.ik.joint_limits[:, 1] - self._q, self._q - self.ik.joint_limits[:, 0])
            self._last_joint_limit_margin_rad = float(np.min(margins))
            reasons = []
            if self._last_ik_error_pos_m > self.cfg.pos_tol:
                reasons.append(f"pos_err={self._last_ik_error_pos_m:.4f}m")
            if self._last_ik_error_rot_rad > self.cfg.rot_tol:
                reasons.append(f"rot_err={self._last_ik_error_rot_rad:.4f}rad")
            if self._last_joint_limit_margin_rad < 0.05:
                reasons.append(f"near_joint_limit={self._last_joint_limit_margin_rad:.4f}rad")
            tx, ty, tz = self._last_target_xyz
            if (
                tx <= self.cfg.workspace_min_xyz[0]
                or tx >= self.cfg.workspace_max_xyz[0]
                or ty <= self.cfg.workspace_min_xyz[1]
                or ty >= self.cfg.workspace_max_xyz[1]
                or tz <= self.cfg.workspace_min_xyz[2]
                or tz >= self.cfg.workspace_max_xyz[2]
            ):
                reasons.append("workspace_edge")
            self._last_ik_fail_reason = ", ".join(reasons) if reasons else "ik_not_converged"
        self._last_ik_time_ms = (time.perf_counter() - ik_start) * 1e3

        self._last_input_pos = pos
        self._last_input_rot = rot
        return self._build_action(self._q, gripper)

    @property
    def last_ik_time_ms(self) -> float:
        return self._last_ik_time_ms

    @property
    def last_ik_success(self) -> bool:
        return self._last_ik_success

    @property
    def last_ik_fail_reason(self) -> str:
        return self._last_ik_fail_reason

    @property
    def last_dp_tracker(self) -> tuple[float, float, float]:
        return self._last_dp_tracker

    @property
    def last_dp_robot(self) -> tuple[float, float, float]:
        return self._last_dp_robot

    @property
    def last_drot_tracker(self) -> tuple[float, float, float]:
        return self._last_drot_tracker

    @property
    def last_drot_robot(self) -> tuple[float, float, float]:
        return self._last_drot_robot


def _quat_angle_rad(q1: tuple[float, float, float, float], q2: tuple[float, float, float, float]) -> float:
    n1 = np.linalg.norm(np.array(q1, dtype=np.float64))
    n2 = np.linalg.norm(np.array(q2, dtype=np.float64))
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    a1 = np.array(q1, dtype=np.float64) / n1
    a2 = np.array(q2, dtype=np.float64) / n2
    dot = abs(float(np.clip(np.dot(a1, a2), -1.0, 1.0)))
    return 2.0 * float(np.arccos(dot))


def wait_for_stable_pose(
    teleop: Teleoperator,
    fps: int,
    stable_frames: int,
    max_pos_delta_m: float,
    max_rot_delta_rad: float,
    timeout_s: float,
) -> RobotAction:
    stable = 0
    prev_pos = None
    prev_rot = None
    started = time.perf_counter()
    latest: RobotAction | None = None

    while stable < stable_frames:
        t0 = time.perf_counter()
        action = teleop.get_action()
        latest = action
        pose_valid = float(action.get("pika.pose.valid", 0.0)) >= 0.5

        if not pose_valid:
            stable = 0
            prev_pos = None
            prev_rot = None
        else:
            pos = (
                float(action["pika.pos.x"]),
                float(action["pika.pos.y"]),
                float(action["pika.pos.z"]),
            )
            rot = (
                float(action["pika.rot.x"]),
                float(action["pika.rot.y"]),
                float(action["pika.rot.z"]),
                float(action["pika.rot.w"]),
            )
            if prev_pos is None or prev_rot is None:
                stable = 1
            else:
                dp = float(np.linalg.norm(np.array(pos) - np.array(prev_pos)))
                drot = _quat_angle_rad(rot, prev_rot)
                stable = stable + 1 if (dp <= max_pos_delta_m and drot <= max_rot_delta_rad) else 1
            prev_pos = pos
            prev_rot = rot

        if timeout_s > 0 and (time.perf_counter() - started) >= timeout_s:
            raise TimeoutError("PIKA pose did not stabilize before timeout")

        dt = time.perf_counter() - t0
        precise_sleep(max(1 / fps - dt, 0.0))
        print(f"Stabilizing pose... {stable}/{stable_frames} valid frames", end="\r", flush=True)

    print()
    assert latest is not None
    return latest


@dataclass
class TeleoperatePikaPiperConfig:
    teleop: TeleoperatorConfig
    robot: RobotConfig
    fps: int = 60
    teleop_time_s: float | None = None
    display_data: bool = False
    display_ip: str | None = None
    display_port: int | None = None
    display_compressed_images: bool = False
    linear_scale: float = 1.0
    angular_scale: float = 1.0
    max_delta_pos_m: float = 0.03
    max_delta_rot_rad: float = 0.30
    ik_damping: float = 0.08
    ik_max_iterations: int = 50
    ik_max_delta_q: float = 0.08
    ik_pos_tol: float = 2e-3
    ik_rot_tol: float = 2e-2
    startup_wait_for_pose: bool = True
    startup_stable_frames: int = 30
    startup_max_pos_delta_m: float = 0.002
    startup_max_rot_delta_rad: float = 0.03
    startup_timeout_s: float = 15.0
    debug_mapping: bool = False
    debug_mapping_every_n: int = 20
    # Mapping strings: comma-separated tracker axis indices (0:x,1:y,2:z)
    # and signs (+1/-1) applied to mapped axes.
    pos_axis_map: str = "0,1,2"
    pos_axis_sign: str = "1,-1,1"
    rot_axis_map: str = "0,1,2"
    rot_axis_sign: str = "1,-1,1"


def _parse_axis_map(text: str) -> tuple[int, int, int]:
    vals = [int(x.strip()) for x in text.split(",")]
    if len(vals) != 3 or sorted(vals) != [0, 1, 2]:
        raise ValueError(f"Invalid axis map '{text}'. Use a permutation of 0,1,2 (e.g. 0,2,1).")
    return vals[0], vals[1], vals[2]


def _parse_axis_sign(text: str) -> tuple[float, float, float]:
    vals = [float(x.strip()) for x in text.split(",")]
    if len(vals) != 3 or any(v not in (-1.0, 1.0) for v in vals):
        raise ValueError(f"Invalid axis sign '{text}'. Use three values from -1 or 1 (e.g. 1,-1,1).")
    return vals[0], vals[1], vals[2]


def teleop_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    mapper: PikaToPiperJointMapper,
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    display_data: bool = False,
    duration: float | None = None,
    display_compressed_images: bool = False,
    debug_mapping: bool = False,
    debug_mapping_every_n: int = 20,
):
    start = time.perf_counter()
    iter_idx = 0

    while True:
        iter_idx += 1
        loop_start = time.perf_counter()
        obs = robot.get_observation()
        raw_action = teleop.get_action()
        mapped_action = mapper.map_action(raw_action, obs)
        robot_action_to_send = robot_action_processor((mapped_action, obs))
        _ = robot.send_action(robot_action_to_send)

        if display_data:
            obs_transition = robot_observation_processor(obs)
            log_rerun_data(
                observation=obs_transition,
                action=mapped_action,
                compress_images=display_compressed_images,
            )
            action_str = ", ".join(
                [f"{k}={float(v):.4f}" for k, v in robot_action_to_send.items() if isinstance(v, (int, float))]
            )
            print(f"[action] iter={iter_idx} {action_str}")

        dt_s = time.perf_counter() - loop_start
        precise_sleep(max(1 / fps - dt_s, 0.0))
        loop_s = time.perf_counter() - loop_start
        if mapper.last_ik_success:
            print(
                f"[loop] iter={iter_idx} loop_ms={loop_s * 1e3:.2f} hz={1 / loop_s:.1f} "
                f"ik_ms={mapper.last_ik_time_ms:.2f} ik_ok=1 pose_valid={int(float(raw_action.get('pika.pose.valid', 0.0)) >= 0.5)}"
            )
        else:
            print(
                f"[loop] iter={iter_idx} loop_ms={loop_s * 1e3:.2f} hz={1 / loop_s:.1f} "
                f"ik_ms={mapper.last_ik_time_ms:.2f} ik_ok=0 pose_valid={int(float(raw_action.get('pika.pose.valid', 0.0)) >= 0.5)} "
                f"reason={mapper.last_ik_fail_reason}"
            )

        if debug_mapping and iter_idx % max(1, debug_mapping_every_n) == 0:
            dpt = mapper.last_dp_tracker
            dpr = mapper.last_dp_robot
            drt = mapper.last_drot_tracker
            drr = mapper.last_drot_robot
            print(
                "[map] "
                f"dp_tracker=({dpt[0]:+.4f},{dpt[1]:+.4f},{dpt[2]:+.4f}) -> "
                f"dp_robot=({dpr[0]:+.4f},{dpr[1]:+.4f},{dpr[2]:+.4f}), "
                f"drot_tracker=({drt[0]:+.4f},{drt[1]:+.4f},{drt[2]:+.4f}) -> "
                f"drot_robot=({drr[0]:+.4f},{drr[1]:+.4f},{drr[2]:+.4f})"
            )

        if duration is not None and time.perf_counter() - start >= duration:
            return


@parser.wrap()
def teleoperate(cfg: TeleoperatePikaPiperConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    if cfg.display_data:
        init_rerun(session_name="teleoperation", ip=cfg.display_ip, port=cfg.display_port)
    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    teleop = make_teleoperator_from_config(cfg.teleop)
    robot = make_robot_from_config(cfg.robot)
    _, robot_action_processor, robot_observation_processor = make_default_processors()

    pos_axis_map = _parse_axis_map(cfg.pos_axis_map)
    pos_axis_sign = _parse_axis_sign(cfg.pos_axis_sign)
    rot_axis_map = _parse_axis_map(cfg.rot_axis_map)
    rot_axis_sign = _parse_axis_sign(cfg.rot_axis_sign)
    logging.info(
        "Axis mapping: pos_map=%s pos_sign=%s rot_map=%s rot_sign=%s",
        pos_axis_map,
        pos_axis_sign,
        rot_axis_map,
        rot_axis_sign,
    )

    mapper = PikaToPiperJointMapper(
        MapperConfig(
            linear_scale=cfg.linear_scale,
            angular_scale=cfg.angular_scale,
            max_delta_pos_m=cfg.max_delta_pos_m,
            max_delta_rot_rad=cfg.max_delta_rot_rad,
            damping=cfg.ik_damping,
            max_iterations=cfg.ik_max_iterations,
            max_delta_q=cfg.ik_max_delta_q,
            pos_tol=cfg.ik_pos_tol,
            rot_tol=cfg.ik_rot_tol,
            pos_axis_map=pos_axis_map,
            pos_axis_sign=pos_axis_sign,
            rot_axis_map=rot_axis_map,
            rot_axis_sign=rot_axis_sign,
        )
    )

    teleop.connect()
    robot.connect()

    try:
        if cfg.startup_wait_for_pose:
            startup_action = wait_for_stable_pose(
                teleop=teleop,
                fps=cfg.fps,
                stable_frames=cfg.startup_stable_frames,
                max_pos_delta_m=cfg.startup_max_pos_delta_m,
                max_rot_delta_rad=cfg.startup_max_rot_delta_rad,
                timeout_s=cfg.startup_timeout_s,
            )
        else:
            startup_action = teleop.get_action()

        startup_obs = robot.get_observation()
        mapper.rebase_reference(startup_action, startup_obs)

        teleop_loop(
            teleop=teleop,
            robot=robot,
            fps=cfg.fps,
            mapper=mapper,
            display_data=cfg.display_data,
            duration=cfg.teleop_time_s,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            display_compressed_images=display_compressed_images,
            debug_mapping=cfg.debug_mapping,
            debug_mapping_every_n=cfg.debug_mapping_every_n,
        )
    except KeyboardInterrupt:
        pass
    finally:
        if cfg.display_data:
            rr.rerun_shutdown()
        teleop.disconnect()
        robot.disconnect()


def main():
    register_third_party_plugins()
    teleoperate()


if __name__ == "__main__":
    main()
