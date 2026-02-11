#!/usr/bin/env python

from dataclasses import dataclass, field

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("pika_teleoperator")
@dataclass
class PikaTeleoperatorConfig(TeleoperatorConfig):
    """Configuration for AgileX PIKA Sense teleoperator over serial."""

    port: str = "/dev/ttyUSB0"
    tracker_device: str | None = None
    require_pose: bool = False
    gripper_min_mm: float = 0.0
    gripper_max_mm: float = 70.0
    tracker_config_path: str | None = None
    tracker_lh_config: str | None = None
    tracker_args: list[str] = field(default_factory=list)
