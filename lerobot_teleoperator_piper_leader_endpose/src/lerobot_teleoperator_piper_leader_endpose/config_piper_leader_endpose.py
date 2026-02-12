#!/usr/bin/env python

from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("piper_leader_endpose")
@dataclass
class PiperLeaderEndPoseConfig(TeleoperatorConfig):
    """Configuration for a PiPER leader teleoperator with EndPose action output."""

    can_name: str = "can_leader"
    judge_flag: bool = False
    source_mode: str = "feedback"  # "feedback" | "control" (gripper source only)
    hand_guiding: bool = True
    gripper_opening_m: float = 0.07
