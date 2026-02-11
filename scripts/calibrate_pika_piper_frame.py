#!/usr/bin/env python3

import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from pprint import pformat

import numpy as np

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.teleoperators import TeleoperatorConfig, make_teleoperator_from_config
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging


@dataclass
class CalibratePikaPiperFrameConfig:
    teleop: TeleoperatorConfig
    robot: RobotConfig
    output_path: str = "outputs/pika_piper_frame_calibration.json"
    num_samples: int = 20
    settle_s: float = 0.3
    average_window: int = 15
    average_interval_s: float = 0.01
    interactive: bool = True
    auto_interval_s: float = 1.0
    min_pose_spread_m: float = 0.10


def _read_single_pair(teleop, robot) -> tuple[np.ndarray, np.ndarray] | None:
    action = teleop.get_action()
    if float(action.get("pika.pose.valid", 0.0)) < 0.5:
        return None

    obs = robot.get_observation()
    required = ("endpose.x", "endpose.y", "endpose.z")
    if not all(k in obs for k in required):
        raise RuntimeError(
            "Robot observation does not include endpose.x/y/z. "
            "Use piper_follower_endpose for calibration."
        )

    p = np.array(
        [
            float(action["pika.pos.x"]),
            float(action["pika.pos.y"]),
            float(action["pika.pos.z"]),
        ],
        dtype=np.float64,
    )
    q = np.array(
        [
            float(obs["endpose.x"]),
            float(obs["endpose.y"]),
            float(obs["endpose.z"]),
        ],
        dtype=np.float64,
    )
    return p, q


def _read_averaged_pair(teleop, robot, window: int, interval_s: float) -> tuple[np.ndarray, np.ndarray]:
    p_list: list[np.ndarray] = []
    q_list: list[np.ndarray] = []
    retries = 0
    while len(p_list) < window:
        pair = _read_single_pair(teleop, robot)
        if pair is None:
            retries += 1
            if retries > window * 10:
                raise RuntimeError("Failed to read valid PIKA pose repeatedly during averaging.")
            time.sleep(interval_s)
            continue
        p, q = pair
        p_list.append(p)
        q_list.append(q)
        time.sleep(interval_s)
    return np.mean(np.stack(p_list, axis=0), axis=0), np.mean(np.stack(q_list, axis=0), axis=0)


def _estimate_r_t_kabsch(p_points: np.ndarray, q_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if p_points.shape != q_points.shape or p_points.ndim != 2 or p_points.shape[1] != 3:
        raise ValueError("Point arrays must both have shape (N, 3).")
    if p_points.shape[0] < 3:
        raise ValueError("At least 3 non-collinear points are required.")

    mu_p = p_points.mean(axis=0)
    mu_q = q_points.mean(axis=0)
    p_centered = p_points - mu_p
    q_centered = q_points - mu_q
    h = p_centered.T @ q_centered
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    t = mu_q - r @ mu_p
    return r, t


def _compute_rmse(p_points: np.ndarray, q_points: np.ndarray, r: np.ndarray, t: np.ndarray) -> float:
    q_pred = (r @ p_points.T).T + t
    err = q_points - q_pred
    return float(np.sqrt(np.mean(np.sum(err * err, axis=1))))


def _compute_spread(points: np.ndarray) -> float:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    return float(np.linalg.norm(maxs - mins))


@parser.wrap()
def calibrate(cfg: CalibratePikaPiperFrameConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    teleop = make_teleoperator_from_config(cfg.teleop)
    robot = make_robot_from_config(cfg.robot)

    p_samples: list[np.ndarray] = []
    q_samples: list[np.ndarray] = []

    teleop.connect()
    robot.connect()
    try:
        print("")
        print("Calibration started.")
        print("Move PIKA and Piper to diverse corresponding poses.")
        print(
            f"Collecting {cfg.num_samples} samples "
            f"(interactive={cfg.interactive}, settle={cfg.settle_s:.2f}s, avg_window={cfg.average_window})"
        )

        for i in range(cfg.num_samples):
            if cfg.interactive:
                input(f"[{i + 1}/{cfg.num_samples}] Move to a new pose, then press Enter to sample...")
            else:
                print(f"[{i + 1}/{cfg.num_samples}] Sampling in {cfg.auto_interval_s:.2f}s...")
                time.sleep(max(cfg.auto_interval_s, 0.0))

            time.sleep(max(cfg.settle_s, 0.0))
            p, q = _read_averaged_pair(
                teleop=teleop,
                robot=robot,
                window=max(1, cfg.average_window),
                interval_s=max(0.0, cfg.average_interval_s),
            )
            p_samples.append(p)
            q_samples.append(q)
            print(
                f"  captured pika=({p[0]:+.4f},{p[1]:+.4f},{p[2]:+.4f}) "
                f"piper=({q[0]:+.4f},{q[1]:+.4f},{q[2]:+.4f})"
            )

        p_points = np.stack(p_samples, axis=0)
        q_points = np.stack(q_samples, axis=0)
        spread = _compute_spread(p_points)
        if spread < cfg.min_pose_spread_m:
            raise RuntimeError(
                f"Sample spread too small ({spread:.4f}m < {cfg.min_pose_spread_m:.4f}m). "
                "Collect more diverse poses."
            )

        r, t = _estimate_r_t_kabsch(p_points, q_points)
        rmse = _compute_rmse(p_points, q_points, r, t)

        out = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "num_samples": int(cfg.num_samples),
            "rmse_m": rmse,
            "sample_spread_m": spread,
            "R_piper_from_pika": r.tolist(),
            "t_piper_from_pika_m": t.tolist(),
            "config": asdict(cfg),
        }

        out_path = Path(cfg.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

        print("")
        print("Calibration complete.")
        print(f"RMSE: {rmse:.6f} m")
        print(f"Sample spread: {spread:.4f} m")
        print(f"Saved: {out_path}")
        print("R (piper <- pika):")
        print(np.array2string(r, precision=6, suppress_small=False))
        print("t (m):")
        print(np.array2string(t, precision=6, suppress_small=False))
    finally:
        teleop.disconnect()
        robot.disconnect()


def main():
    register_third_party_plugins()
    calibrate()


if __name__ == "__main__":
    main()
