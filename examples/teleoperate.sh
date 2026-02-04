#!/bin/bash


lerobot-teleoperate \
  --robot.type=piper_follower \
  --robot.can_name=can_follower \
  --robot.id=piper_follower_01 \
  --teleop.type=piper_leader \
  --teleop.can_name=can_leader \
  --teleop.id=piper_leader_01 \
  --fps=50 \
  --display_data=true

