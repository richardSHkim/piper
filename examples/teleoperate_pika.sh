#!/bin/bash

# python scripts/pika_piper_teleoperate.py \
#   --teleop-port /dev/ttyUSB0 \
#   --tracker-device WM0 \
#   --robot-can-name can_follower \
#   --fps 50 \
#   --startup-move-zero \
#   --startup-sync-gripper \
#   --arm-gesture-double-close

python scripts/pika_piper_teleoperate_endpose.py \
  --teleop.type=pika_teleoperator \
  --teleop.port=/dev/ttyUSB0 \
  --teleop.tracker_device=WM0 \
  --robot.type=piper_follower_endpose \
  --robot.can_name=can0 \
  --robot.id=piper_follower_endpose_01 \
  --fps=50
