#!/usr/bin/env python

from dataclasses import dataclass

from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("piper_follower")
@dataclass
class PiperFollowerConfig(RobotConfig):
    """Configuration for a PiPER follower arm controlled over CAN."""

    can_name: str = "can_follower"
    judge_flag: bool = False
    speed_ratio: int = 60
    gripper_effort: int = 1000
    gripper_opening_m: float = 0.07
    startup_enable_timeout_s: float = 5.0
    disable_on_disconnect: bool = True
    max_delta_translation_m: float = 0.01
    max_delta_rotation_rad: float = 0.08726646259971647  # 5 deg
