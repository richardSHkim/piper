#!/usr/bin/env python

from .config_piper_follower_endpose import PiperFollowerEndPoseConfig
from .piper_follower_endpose import PiperFollowerEndPose

__all__ = ["PiperFollowerEndPoseConfig", "PiperFollowerEndPose"]


def main() -> None:
    print("lerobot_robot_piper_follower_endpose plugin is installed.")
