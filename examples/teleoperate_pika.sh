#!/bin/bash

python scripts/teleoperate_pika_with_calib.py \
  --calib calib.json \
  --can can0 \
  --pika-port /dev/ttyUSB0 \
  --pika-device-key WM0 \
  --speed-ratio 70 \
  --period 0.02
