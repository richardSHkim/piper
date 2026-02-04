#!/usr/bin/env python

from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("piper_leader")
@dataclass
class PiperLeaderConfig(TeleoperatorConfig):
    """Configuration for a PiPER leader teleoperator over CAN."""

    can_name: str = "can_leader"
    judge_flag: bool = False
    source_mode: str = "feedback"  # "feedback" | "control"
    hand_guiding: bool = True
    gripper_opening_m: float = 0.07
    max_delta_translation_m: float = 0.01
    max_delta_rotation_rad: float = 0.08726646259971647  # 5 deg
