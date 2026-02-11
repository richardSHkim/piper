#!/bin/bash

python scripts/pika_piper_teleoperate_endpose.py \
  --teleop.type=pika_teleoperator \
  --teleop.port=/dev/ttyUSB0 \
  --teleop.tracker_device=WM0 \
  --robot.type=piper_follower_endpose \
  --robot.can_name=can0 \
  --robot.id=piper_follower_endpose_01 \
  --fps=30 \
  --linear_scale=0.15 \
  --angular_scale=0.10 \
  --max_delta_pos_m=0.002 \
  --max_delta_rot_rad=0.02 \
  --startup_wait_for_pose=true \
  --startup_stable_frames=50 \
  --startup_max_pos_delta_m=0.001 \
  --startup_max_rot_delta_rad=0.02 \
  --startup_timeout_s=15.0 \
  --debug_loopback=true \
  --debug_every_n=5
