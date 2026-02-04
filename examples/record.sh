#!/bin/bash


lerobot-record \
    --robot.type=piper_follower \
    --robot.can_name=can_follower \
    --robot.id=piper_follower_01 \
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
