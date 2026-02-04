## lerobot_robot_piper_follower

LeRobot robot plugin for AgileX PiPER follower arm (CAN based).

### Action schema

- End-effector delta: `delta_x delta_y delta_z delta_rx delta_ry delta_rz` (+ optional `gripper`), unit: m/rad.

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
