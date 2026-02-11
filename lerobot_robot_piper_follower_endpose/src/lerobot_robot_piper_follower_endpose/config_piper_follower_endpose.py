#!/usr/bin/env python

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("piper_follower_endpose")
@dataclass
class PiperFollowerEndPoseConfig(RobotConfig):
    """Configuration for PiPER follower with EndPoseCtrl cartesian command mode."""

    can_name: str = "can_follower"
    judge_flag: bool = False
    speed_ratio: int = 60
    gripper_effort: int = 1000
    gripper_opening_m: float = 0.07
    startup_enable_timeout_s: float = 5.0
    disable_on_disconnect: bool = True
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
