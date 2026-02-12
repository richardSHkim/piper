## lerobot_teleoperator_piper_leader_endpose

LeRobot teleoperator plugin for AgileX PiPER leader arm (CAN based), outputting EndPose targets.

### Output action schema

- `target_x`, `target_y`, `target_z` (m)
- `target_roll`, `target_pitch`, `target_yaw` (rad)
- `gripper.pos` (normalized 0..1)

This schema is directly compatible with `lerobot_robot_piper_follower_endpose`.

### Minimal config example

```yaml
teleop:
  type: piper_leader_endpose
  can_name: can_leader
  pose_source: joint_fk
  id: piper_leader_endpose_01
  hand_guiding: true
```

`pose_source` options:
- `joint_fk` (default): read leader joint feedback and compute end pose via FK. Recommended when hand-guiding.
- `endpose_feedback`: use `GetArmEndPoseMsgs()` directly.
