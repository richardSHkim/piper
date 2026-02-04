## lerobot_teleoperator_piper_leader

LeRobot teleoperator plugin for AgileX PiPER leader arm (CAN based).

### Output action schema

- `delta_x delta_y delta_z delta_rx delta_ry delta_rz` (m/rad)
- `gripper` (normalized 0..1)

### Minimal config example

```yaml
teleop:
  type: piper_leader
  can_name: can_leader
  id: piper_leader_01
  source_mode: feedback
```

`source_mode=feedback` is recommended for hand-guided teleoperation.
