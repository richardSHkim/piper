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

RAD_TO_001DEG = 180000.0 / math.pi
_001DEG_TO_RAD = math.pi / 180000.0
M_TO_001MM = 1_000_000.0
_001MM_TO_M = 1.0 / 1_000_000.0


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
    debug_loopback: bool = False
    debug_every_n: int = 10
    dry_run_visualize: bool = False
    visualize_width: int = 1000
    visualize_height: int = 720
    visualize_target_axis_len_m: float = 0.10


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


def _rpy_to_quat(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return qx, qy, qz, qw


class RGBFrameVisualizer:
    def __init__(self, width: int, height: int):
        try:
            import pygame  # type: ignore
        except ImportError as exc:
            raise RuntimeError("pygame is required for --dry_run_visualize") from exc
        self.pygame = pygame
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("PIKA -> PiPER EndPose Dry-Run Visualization")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
        self.width = width
        self.height = height
        self.running = True
        self.camera_distance = 2.2
        self.scale_factor = 900.0
        self.zoom_step = 60.0

    def close(self) -> None:
        if self.running:
            self.pygame.quit()
        self.running = False

    def _project(self, p: tuple[float, float, float]) -> tuple[int, int]:
        x, y, z = p
        depth = z + self.camera_distance
        if depth <= 0.05:
            depth = 0.05
        factor = self.scale_factor / depth
        return int(self.width / 2 + x * factor), int(self.height / 2 - y * factor)

    def _quat_to_rot(self, qx: float, qy: float, qz: float, qw: float):
        import numpy as np

        return np.array(
            [
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
            ]
        )

    def _draw_frame(
        self,
        position: tuple[float, float, float],
        quat: tuple[float, float, float, float],
        axis_len_m: float,
        name: str,
        origin_color: tuple[int, int, int],
    ) -> None:
        import numpy as np

        rot = self._quat_to_rot(*quat)
        pos = np.array(position)
        x_end = tuple((pos + rot[:, 0] * axis_len_m).tolist())
        y_end = tuple((pos + rot[:, 1] * axis_len_m).tolist())
        z_end = tuple((pos + rot[:, 2] * axis_len_m).tolist())
        p0 = self._project(position)
        px = self._project(x_end)
        py = self._project(y_end)
        pz = self._project(z_end)
        self.pygame.draw.line(self.screen, (255, 0, 0), p0, px, 3)
        self.pygame.draw.line(self.screen, (0, 255, 0), p0, py, 3)
        self.pygame.draw.line(self.screen, (0, 0, 255), p0, pz, 3)
        self.screen.blit(self.font.render("X", True, (180, 20, 20)), (px[0] + 2, px[1] + 2))
        self.screen.blit(self.font.render("Y", True, (20, 130, 20)), (py[0] + 2, py[1] + 2))
        self.screen.blit(self.font.render("Z", True, (20, 20, 180)), (pz[0] + 2, pz[1] + 2))
        self.pygame.draw.circle(self.screen, origin_color, p0, 5)
        self.screen.blit(self.font.render(name, True, origin_color), (p0[0] + 8, p0[1] - 8))

    def update(
        self,
        target_pos: tuple[float, float, float],
        target_quat: tuple[float, float, float, float],
        target_axis_len_m: float,
        cmd_ints: tuple[int, int, int, int, int, int],
    ) -> bool:
        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT:
                self.running = False
                return False
            if event.type == self.pygame.KEYDOWN:
                if event.key in (self.pygame.K_w, self.pygame.K_UP):
                    self.scale_factor += self.zoom_step
                if event.key in (self.pygame.K_s, self.pygame.K_DOWN):
                    self.scale_factor = max(100.0, self.scale_factor - self.zoom_step)

        # Bright background + ground grid for better depth perception.
        self.screen.fill((235, 240, 248))
        self.pygame.draw.rect(self.screen, (220, 228, 240), (0, 0, self.width, self.height // 2))
        self.pygame.draw.rect(
            self.screen, (242, 246, 252), (0, self.height // 2, self.width, self.height - self.height // 2)
        )
        center = self._project((0.0, 0.0, 0.0))
        self.pygame.draw.line(self.screen, (180, 188, 202), (0, center[1]), (self.width, center[1]), 1)
        self.pygame.draw.line(self.screen, (180, 188, 202), (center[0], 0), (center[0], self.height), 1)
        for g in range(-6, 7):
            y0 = self._project((-0.6, g * 0.1, 0.0))
            y1 = self._project((0.6, g * 0.1, 0.0))
            x0 = self._project((g * 0.1, -0.6, 0.0))
            x1 = self._project((g * 0.1, 0.6, 0.0))
            self.pygame.draw.line(self.screen, (206, 214, 228), y0, y1, 1)
            self.pygame.draw.line(self.screen, (206, 214, 228), x0, x1, 1)
        # Depth cue: z-level rectangles (farther = lighter/smaller by projection)
        for z, c in ((0.10, (188, 198, 216)), (0.25, (170, 182, 202)), (0.40, (154, 168, 190))):
            corners = [(-0.45, -0.45, z), (0.45, -0.45, z), (0.45, 0.45, z), (-0.45, 0.45, z)]
            p = [self._project(v) for v in corners]
            self.pygame.draw.lines(self.screen, c, True, p, 1)
        # Base Z ruler for clearer vertical depth orientation.
        bz0 = self._project((0.0, 0.0, 0.0))
        bz1 = self._project((0.0, 0.0, 0.5))
        self.pygame.draw.line(self.screen, (60, 90, 180), bz0, bz1, 2)
        for z in (0.1, 0.2, 0.3, 0.4, 0.5):
            tick = self._project((0.0, 0.0, z))
            self.pygame.draw.line(self.screen, (60, 90, 180), (tick[0] - 4, tick[1]), (tick[0] + 4, tick[1]), 1)
            self.screen.blit(self.font.render(f"{z:.1f}", True, (60, 90, 180)), (tick[0] + 6, tick[1] - 8))
        self._draw_frame((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 0.12, "base", (120, 120, 220))
        self._draw_frame(target_pos, target_quat, target_axis_len_m, "cmd(target)", (255, 240, 120))
        info = self.font.render("W/S or Up/Down: zoom, close window: exit", True, (40, 46, 58))
        self.screen.blit(info, (10, 10))
        cmd_text = self.font.render(
            f"EndPoseCtrl int: X={cmd_ints[0]} Y={cmd_ints[1]} Z={cmd_ints[2]} RX={cmd_ints[3]} RY={cmd_ints[4]} RZ={cmd_ints[5]}",
            True,
            (70, 52, 22),
        )
        self.screen.blit(cmd_text, (10, 38))
        self.pygame.display.flip()
        self.clock.tick(60)
        return True


def _quantize_as_endpose_ctrl(action: RobotAction) -> tuple[RobotAction, tuple[int, int, int, int, int, int]]:
    x = float(action.get("target_x", 0.0))
    y = float(action.get("target_y", 0.0))
    z = float(action.get("target_z", 0.0))
    roll = float(action.get("target_roll", 0.0))
    pitch = float(action.get("target_pitch", 0.0))
    yaw = float(action.get("target_yaw", 0.0))

    x_i = int(round(x * M_TO_001MM))
    y_i = int(round(y * M_TO_001MM))
    z_i = int(round(z * M_TO_001MM))
    roll_i = int(round(roll * RAD_TO_001DEG))
    pitch_i = int(round(pitch * RAD_TO_001DEG))
    yaw_i = int(round(yaw * RAD_TO_001DEG))

    quantized: RobotAction = dict(action)
    quantized["target_x"] = x_i * _001MM_TO_M
    quantized["target_y"] = y_i * _001MM_TO_M
    quantized["target_z"] = z_i * _001MM_TO_M
    quantized["target_roll"] = roll_i * _001DEG_TO_RAD
    quantized["target_pitch"] = pitch_i * _001DEG_TO_RAD
    quantized["target_yaw"] = yaw_i * _001DEG_TO_RAD
    return quantized, (x_i, y_i, z_i, roll_i, pitch_i, yaw_i)


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
    debug_loopback: bool = False,
    debug_every_n: int = 10,
    dry_run_visualize: bool = False,
    visualizer: RGBFrameVisualizer | None = None,
    visualize_target_axis_len_m: float = 0.10,
):
    display_len = max(len(key) for key in robot.action_features)
    start = time.perf_counter()
    loop_idx = 0

    while True:
        loop_start = time.perf_counter()
        obs = robot.get_observation()
        raw_action = teleop.get_action()
        teleop_action = teleop_action_processor((raw_action, obs))
        robot_action_to_send = robot_action_processor((teleop_action, obs))
        vis_action, cmd_ints = _quantize_as_endpose_ctrl(robot_action_to_send)
        if not dry_run_visualize:
            _ = robot.send_action(robot_action_to_send)

        if dry_run_visualize and visualizer is not None:
            target_pos = (
                float(vis_action.get("target_x", 0.0)),
                float(vis_action.get("target_y", 0.0)),
                float(vis_action.get("target_z", 0.0)),
            )
            target_quat = _rpy_to_quat(
                float(vis_action.get("target_roll", 0.0)),
                float(vis_action.get("target_pitch", 0.0)),
                float(vis_action.get("target_yaw", 0.0)),
            )

            if not visualizer.update(
                target_pos=target_pos,
                target_quat=target_quat,
                target_axis_len_m=visualize_target_axis_len_m,
                cmd_ints=cmd_ints,
            ):
                return
        loop_idx += 1

        if debug_loopback and loop_idx % max(1, debug_every_n) == 0:
            logging.info(
                (
                    "raw pose valid=%.0f pos=(%.4f, %.4f, %.4f) target=(%.4f, %.4f, %.4f, %.4f, %.4f, %.4f) "
                    "obs_endpose=(%.4f, %.4f, %.4f, %.4f, %.4f, %.4f)"
                ),
                float(raw_action.get("pika.pose.valid", 0.0)),
                float(raw_action.get("pika.pos.x", 0.0)),
                float(raw_action.get("pika.pos.y", 0.0)),
                float(raw_action.get("pika.pos.z", 0.0)),
                float(robot_action_to_send.get("target_x", 0.0)),
                float(robot_action_to_send.get("target_y", 0.0)),
                float(robot_action_to_send.get("target_z", 0.0)),
                float(robot_action_to_send.get("target_roll", 0.0)),
                float(robot_action_to_send.get("target_pitch", 0.0)),
                float(robot_action_to_send.get("target_yaw", 0.0)),
                float(obs.get("endpose.x", 0.0)),
                float(obs.get("endpose.y", 0.0)),
                float(obs.get("endpose.z", 0.0)),
                float(obs.get("endpose.roll", 0.0)),
                float(obs.get("endpose.pitch", 0.0)),
                float(obs.get("endpose.yaw", 0.0)),
            )

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
    visualizer = None
    if cfg.dry_run_visualize:
        visualizer = RGBFrameVisualizer(width=cfg.visualize_width, height=cfg.visualize_height)

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
            debug_loopback=cfg.debug_loopback,
            debug_every_n=cfg.debug_every_n,
            dry_run_visualize=cfg.dry_run_visualize,
            visualizer=visualizer,
            visualize_target_axis_len_m=cfg.visualize_target_axis_len_m,
        )
    except KeyboardInterrupt:
        pass
    finally:
        if visualizer is not None:
            visualizer.close()
        if cfg.display_data:
            rr.rerun_shutdown()
        teleop.disconnect()
        robot.disconnect()


def main():
    register_third_party_plugins()
    teleoperate_endpose()


if __name__ == "__main__":
    main()
