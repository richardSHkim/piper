#!/usr/bin/env python

from .config_piper_leader_endpose import PiperLeaderEndPoseConfig
from .piper_leader_endpose import PiperLeaderEndPose

__all__ = ["PiperLeaderEndPoseConfig", "PiperLeaderEndPose"]


def main() -> None:
    print("lerobot_teleoperator_piper_leader_endpose plugin is installed.")
