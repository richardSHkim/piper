## lerobot_robot_piper_follower

LeRobot robot plugin for AgileX PiPER follower arm (CAN based).

### Action schema

- Joint target: `joint_1.pos ... joint_6.pos` (rad)
- Gripper: `gripper.pos` (0..1, optional)

### Observation schema

- Joint state: `joint_1.pos ... joint_6.pos` (rad)
- Gripper: `gripper.pos` (0..1)

### Minimal config example

```yaml
robot:
  type: piper_follower
  can_name: can_follower
  id: piper_follower_01
```

Default CAN names expected for this project are `can_leader` and `can_follower`.
