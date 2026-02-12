#!/bin/bash

lerobot-teleoperate \
  --robot.type=piper_follower_endpose \
  --robot.can_name=can_follower \
  --robot.id=piper_follower_endpose_01 \
  --teleop.type=piper_leader_endpose \
  --teleop.can_name=can_leader \
  --teleop.pose_source=joint_fk \
  --teleop.id=piper_leader_endpose_01 \
  --fps=50 \
  --display_data=true
