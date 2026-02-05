#!/bin/bash


lerobot-record \
    --robot.type=piper_follower \
    --robot.can_name=can_follower \
    --robot.id=piper_follower_01 \
    --robot.cameras="{ rgb: {type: realsense, serial_number_or_name: '323622272575', width: 848, height: 480, fps: 30}, fisheye: {type: opencv, index_or_path: 8, width: 640, height: 480, fps: 30}}" \
    --teleop.type=piper_leader \
    --teleop.can_name=can_leader \
    --teleop.id=piper_leader_01 \
    --dataset.repo_id=richardshkim/piper_dataset \
    --dataset.single_task="Do something" \
    --dataset.num_episodes=20 \
    --dataset.episode_time_s=30 \
    --dataset.reset_time_s=10 \
    --dataset.fps=50 \
    --display_data=true
