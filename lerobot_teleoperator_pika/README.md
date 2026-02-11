## lerobot_teleoperator_pika

LeRobot teleoperator plugin for AgileX PIKA Sense (serial based).

### Teleoperator output schema

- `pika.pos.x, pika.pos.y, pika.pos.z` (m, tracker frame)
- `pika.rot.x, pika.rot.y, pika.rot.z, pika.rot.w` (quaternion)
- `pika.gripper.pos` (0..1)
- `pika.pose.valid` (1.0 or 0.0)

### Included processor

`MapPikaActionToPiperJoints` maps PIKA pose stream to PiPER joint targets:

- Integrates delta EE pose from consecutive tracker poses
- Solves IK with damped least squares
- Outputs `joint_1.pos ... joint_6.pos` and `gripper.pos`
