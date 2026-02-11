#!/bin/bash

python scripts/calibrate_pika_piper_frame.py \
  --teleop.type=pika_teleoperator \
  --teleop.port=/dev/ttyUSB0 \
  --teleop.tracker_device=WM0 \
  --robot.type=piper_follower_endpose \
  --robot.can_name=can0 \
  --robot.id=piper_follower_endpose_01 \
  --num_samples=20 \
  --interactive=true \
  --settle_s=0.3 \
  --average_window=15 \
  --average_interval_s=0.01 \
  --min_pose_spread_m=0.10 \
  --output_path=outputs/pika_piper_frame_calibration.json
