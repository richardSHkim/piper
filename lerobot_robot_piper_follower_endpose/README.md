## lerobot_robot_piper_follower_endpose

LeRobot robot plugin for AgileX PiPER follower arm using EndPoseCtrl (cartesian control).

### Action schema

- `target_x` (m)
- `target_y` (m)
- `target_z` (m)
- `target_roll` (rad)
- `target_pitch` (rad)
- `target_yaw` (rad)
- `gripper.pos` (0..1, optional)

### Observation schema

- `joint_1.pos ... joint_6.pos` (rad)
- `gripper.pos` (0..1)

### Minimal config example

```yaml
robot:
  type: piper_follower_endpose
  can_name: can_follower
  id: piper_follower_endpose_01
```
