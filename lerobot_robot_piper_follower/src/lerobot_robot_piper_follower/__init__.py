#!/usr/bin/env python

from .config_piper_follower import PiperFollowerConfig
from .piper_follower import PiperFollower

__all__ = ["PiperFollowerConfig", "PiperFollower"]


def main() -> None:
    print("lerobot_robot_piper_follower plugin is installed.")
