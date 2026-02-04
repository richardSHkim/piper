## lerobot_teleoperator_piper_leader

LeRobot teleoperator plugin for AgileX PiPER leader arm (CAN based).

### Output action schema

- `joint_1.pos` ... `joint_6.pos` (rad)
- `gripper.pos` (normalized 0..1)

### Minimal config example

```yaml
teleop:
  type: piper_leader
  can_name: can_leader
  id: piper_leader_01
  source_mode: feedback
```

`source_mode=feedback` is recommended for hand-guided teleoperation.
