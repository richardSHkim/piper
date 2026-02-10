import time
from pika import sense

PORT = "/dev/ttyUSB0"  # 실제 포트로 변경
dev = sense(PORT)

ok = dev.connect()
print("connect:", ok)
if not ok:
    raise SystemExit(1)

try:
    for i in range(10):
        enc = dev.get_encoder_data()
        cmd = dev.get_command_state()
        dist = dev.get_gripper_distance()
        print(f"[{i}] rad={enc['rad']:.4f}, angle={enc['angle']:.2f}, cmd={cmd}, dist_mm={dist:.2f}")
        time.sleep(0.1)
finally:
    dev.disconnect()
    print("disconnect: done")