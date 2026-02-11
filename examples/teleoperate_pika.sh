#!/bin/bash

python scripts/pika_piper_teleoperate.py \
  --teleop-port /dev/ttyUSB0 \
  --tracker-device WM0 \
  --robot-can-name can_follower \
  --fps 50 \
  --startup-move-zero \
  --startup-sync-gripper \
  --arm-gesture-double-close
