#!/bin/bash

python scripts/pika_piper_teleoperate.py \
  --teleop.type=pika_teleoperator \
  --teleop.port=/dev/ttyUSB0 \
  --teleop.tracker_device=WM0 \
  --robot.type=piper_follower \
  --robot.can_name=can0 \
  --robot.id=piper_follower_01 \
  --fps=50 \
  --linear_scale=0.3 \
  --angular_scale=0.3 \
  --max_delta_pos_m=0.005 \
  --max_delta_rot_rad=0.05 \
  --startup_wait_for_pose=true \
  --startup_stable_frames=30 \
  --startup_max_pos_delta_m=0.002 \
  --startup_max_rot_delta_rad=0.03 \
  --startup_timeout_s=15.0
