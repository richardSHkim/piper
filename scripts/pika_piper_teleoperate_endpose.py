#!/usr/bin/env python3

import logging
import math
import time
from dataclasses import asdict, dataclass
from pprint import pformat

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
from lerobot.processor.converters import robot_action_observation_to_transition, transition_to_robot_action
from lerobot.robots import Robot, RobotConfig, make_robot_from_config
from lerobot.teleoperators import Teleoperator, TeleoperatorConfig, make_teleoperator_from_config
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, move_cursor_up
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
from lerobot_teleoperator_pika import MapPikaActionToPiperEndPose


@dataclass
class TeleoperateEndPoseConfig:
    teleop: TeleoperatorConfig
    robot: RobotConfig
    fps: int = 60
    teleop_time_s: float | None = None
    display_data: bool = False
    display_ip: str | None = None
    display_port: int | None = None
    display_compressed_images: bool = False
    # Custom processor params
    linear_scale: float = 1.0
    angular_scale: float = 1.0
    max_delta_pos_m: float = 0.03
    max_delta_rot_rad: float = 0.30
    startup_wait_for_pose: bool = True
    startup_stable_frames: int = 30
    startup_max_pos_delta_m: float = 0.002
    startup_max_rot_delta_rad: float = 0.03
    startup_timeout_s: float = 15.0


def _clamp(x: float, lo: float, hi: float) -> float:
    return min(max(x, lo), hi)


def _quat_angle_rad(q1: tuple[float, float, float, float], q2: tuple[float, float, float, float]) -> float:
    n1 = math.sqrt(sum(v * v for v in q1))
    n2 = math.sqrt(sum(v * v for v in q2))
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    a1 = tuple(v / n1 for v in q1)
    a2 = tuple(v / n2 for v in q2)
    dot = abs(sum(x * y for x, y in zip(a1, a2, strict=True)))
    dot = _clamp(dot, -1.0, 1.0)
    return 2.0 * math.acos(dot)


def wait_for_stable_pose(teleop: Teleoperator, fps: int, cfg: TeleoperateEndPoseConfig) -> None:
    logging.info(
        "Waiting for stable PIKA pose: valid=%d consecutive frames, max dp=%.4fm, max drot=%.4frad",
        cfg.startup_stable_frames,
        cfg.startup_max_pos_delta_m,
        cfg.startup_max_rot_delta_rad,
    )
    stable = 0
    prev_pos = None
    prev_rot = None
    started = time.perf_counter()

    while stable < cfg.startup_stable_frames:
        loop_start = time.perf_counter()
        action = teleop.get_action()
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
                dp = math.sqrt(sum((a - b) * (a - b) for a, b in zip(pos, prev_pos, strict=True)))
                drot = _quat_angle_rad(rot, prev_rot)
                if dp <= cfg.startup_max_pos_delta_m and drot <= cfg.startup_max_rot_delta_rad:
                    stable += 1
                else:
                    stable = 1
            prev_pos = pos
            prev_rot = rot

        if cfg.startup_timeout_s > 0 and (time.perf_counter() - started) >= cfg.startup_timeout_s:
            raise TimeoutError(
                "PIKA pose did not stabilize before timeout. "
                "Check base station tracking/visibility and tracker selection."
            )

        dt = time.perf_counter() - loop_start
        precise_sleep(max(1 / fps - dt, 0.0))
        print(
            f"Stabilizing pose... {stable}/{cfg.startup_stable_frames} valid frames",
            end="\r",
            flush=True,
        )
    print()
    logging.info("PIKA pose stabilized. Starting teleoperation.")


def make_pika_endpose_teleop_action_processor(
    cfg: TeleoperateEndPoseConfig,
) -> RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction]:
    step = MapPikaActionToPiperEndPose(
        linear_scale=cfg.linear_scale,
        angular_scale=cfg.angular_scale,
        max_delta_pos_m=cfg.max_delta_pos_m,
        max_delta_rot_rad=cfg.max_delta_rot_rad,
    )
    return RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[step],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )


def teleop_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    display_data: bool = False,
    duration: float | None = None,
    display_compressed_images: bool = False,
):
    display_len = max(len(key) for key in robot.action_features)
    start = time.perf_counter()

    while True:
        loop_start = time.perf_counter()
        obs = robot.get_observation()
        raw_action = teleop.get_action()
        teleop_action = teleop_action_processor((raw_action, obs))
        robot_action_to_send = robot_action_processor((teleop_action, obs))
        _ = robot.send_action(robot_action_to_send)

        if display_data:
            obs_transition = robot_observation_processor(obs)
            log_rerun_data(
                observation=obs_transition,
                action=teleop_action,
                compress_images=display_compressed_images,
            )

            print("\n" + "-" * (display_len + 10))
            print(f"{'NAME':<{display_len}} | {'NORM':>7}")
            for motor, value in robot_action_to_send.items():
                print(f"{motor:<{display_len}} | {value:>7.2f}")
            move_cursor_up(len(robot_action_to_send) + 3)

        dt_s = time.perf_counter() - loop_start
        precise_sleep(max(1 / fps - dt_s, 0.0))
        loop_s = time.perf_counter() - loop_start
        print(f"Teleop loop time: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")
        move_cursor_up(1)

        if duration is not None and time.perf_counter() - start >= duration:
            return


@parser.wrap()
def teleoperate_endpose(cfg: TeleoperateEndPoseConfig):
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
    teleop_action_processor = make_pika_endpose_teleop_action_processor(cfg)

    teleop.connect()
    robot.connect()

    try:
        if cfg.startup_wait_for_pose:
            wait_for_stable_pose(teleop=teleop, fps=cfg.fps, cfg=cfg)
        teleop_loop(
            teleop=teleop,
            robot=robot,
            fps=cfg.fps,
            display_data=cfg.display_data,
            duration=cfg.teleop_time_s,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            display_compressed_images=display_compressed_images,
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
    teleoperate_endpose()


if __name__ == "__main__":
    main()
