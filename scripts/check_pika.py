import time
import math
from pika import sense

PORT = "/dev/ttyUSB0"  # 실제 포트로 변경
dev = sense(PORT)


def quaternion_to_rpy(x: float, y: float, z: float, w: float) -> tuple[float, float, float]:
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def format_pose_6d(pose_obj) -> str:
    if pose_obj is None:
        return "none"

    if isinstance(pose_obj, dict):
        if not pose_obj:
            return "{}"
        parts = []
        for name, p in pose_obj.items():
            if p is None:
                parts.append(f"{name}: none")
                continue

            x, y, z = p.position
            qx, qy, qz, qw = p.rotation
            roll, pitch, yaw = quaternion_to_rpy(qx, qy, qz, qw)
            parts.append(
                f"{name}: x={x:.4f}, y={y:.4f}, z={z:.4f}, "
                f"roll={roll:.4f}, pitch={pitch:.4f}, yaw={yaw:.4f}"
            )
        return " | ".join(parts)

    x, y, z = pose_obj.position
    qx, qy, qz, qw = pose_obj.rotation
    roll, pitch, yaw = quaternion_to_rpy(qx, qy, qz, qw)
    return (
        f"x={x:.4f}, y={y:.4f}, z={z:.4f}, "
        f"roll={roll:.4f}, pitch={pitch:.4f}, yaw={yaw:.4f}"
    )


ok = dev.connect()
print("connect:", ok)
if not ok:
    raise SystemExit(1)

try:
    devices = dev.get_tracker_devices()
    print("tracker devices:", devices)

    i = 0
    while True:
        enc = dev.get_encoder_data()
        cmd = dev.get_command_state()
        dist = dev.get_gripper_distance()
        try:
            pose = format_pose_6d(dev.get_pose())
        except Exception as e:
            pose = f"pose_error={e}"

        print(
            f"rad={enc['rad']:.4f}, angle={enc['angle']:.2f}, "
            f"cmd={cmd}, dist_mm={dist:.2f}, pose={pose}"
        )
        i += 1
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\nCtrl+C received, shutting down safely...")
finally:
    dev.disconnect()
    print("disconnect: done")
