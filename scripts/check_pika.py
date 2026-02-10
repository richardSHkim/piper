import time
from pika import sense

PORT = "/dev/ttyUSB0"  # 실제 포트로 변경
dev = sense(PORT)

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
            pose = dev.get_pose()
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
