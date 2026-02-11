#!/usr/bin/env python

from .config_pika_teleoperator import PikaTeleoperatorConfig
from .pika_teleoperator import PikaTeleoperator
from .pika_to_piper_endpose_processor import MapPikaActionToPiperEndPose
from .pika_to_piper_processor import MapPikaActionToPiperJoints

__all__ = [
    "PikaTeleoperatorConfig",
    "PikaTeleoperator",
    "MapPikaActionToPiperJoints",
    "MapPikaActionToPiperEndPose",
]


def main() -> None:
    print("lerobot_teleoperator_pika plugin is installed.")
