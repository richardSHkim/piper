#!/usr/bin/env python

from .config_piper_leader import PiperLeaderConfig
from .piper_leader import PiperLeader

__all__ = ["PiperLeaderConfig", "PiperLeader"]


def main() -> None:
    print("lerobot_teleoperator_piper_leader plugin is installed.")
