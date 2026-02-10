import argparse
import math
import time

from pika import sense


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


def read_one(tag: str, dev) -> str:
    enc = dev.get_encoder_data()
    cmd = dev.get_command_state()
    dist = dev.get_gripper_distance()
    pose = format_pose_6d(dev.get_pose())
    return (
        f"{tag} rad={enc['rad']:.4f}, angle={enc['angle']:.2f}, "
        f"cmd={cmd}, dist_mm={dist:.2f}, pose={pose}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Read two PIKA Sense devices continuously.")
    parser.add_argument("--port-a", default="/dev/ttyUSB0", help="Serial port for first device")
    parser.add_argument("--port-b", default="/dev/ttyUSB1", help="Serial port for second device")
    parser.add_argument("--sleep", type=float, default=0.1, help="Loop sleep seconds")
    args = parser.parse_args()

    dev_a = sense(args.port_a)
    dev_b = sense(args.port_b)

    ok_a = dev_a.connect()
    ok_b = dev_b.connect()
    print(f"connect[senseA={args.port_a}]:", ok_a)
    print(f"connect[senseB={args.port_b}]:", ok_b)
    if not ok_a or not ok_b:
        if ok_a:
            dev_a.disconnect()
        if ok_b:
            dev_b.disconnect()
        raise SystemExit(1)

    try:
        print("tracker devices A:", dev_a.get_tracker_devices())
        print("tracker devices B:", dev_b.get_tracker_devices())

        while True:
            try:
                print(read_one("senseA", dev_a))
            except Exception as e:
                print(f"senseA error: {e}")

            try:
                print(read_one("senseB", dev_b))
            except Exception as e:
                print(f"senseB error: {e}")

            time.sleep(args.sleep)
    except KeyboardInterrupt:
        print("\nCtrl+C received, shutting down safely...")
    finally:
        dev_a.disconnect()
        dev_b.disconnect()
        print("disconnect: done")


if __name__ == "__main__":
    main()
