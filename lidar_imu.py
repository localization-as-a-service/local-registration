import argparse
import os

import cv2
import numpy as np
import pandas as pd
import pyrealsense2 as rs


def extract_motion_data(timestamp, data):
    return np.array([timestamp, data.x, data.y, data.z])


def start_imu_stream(alpha=0.8):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 100)
    config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 100)
    pipeline.start(config)

    accel_data = []
    gyro_data = []

    gravity = [0, 0, 0]
    last_t = -1

    displacement = [0, 0, 0]
    velocity = [0, 0, 0]

    reset_counter = 0

    while True:
        try:
            # Wait for a coherent pair of frames: depth and color
            frames: rs.composite_frame = pipeline.wait_for_frames()
            current_t = frames.get_frame_metadata(rs.frame_metadata_value.backend_timestamp)

            if last_t == -1:
                last_t = current_t

            accel_frame = frames[0].as_motion_frame()
            gyro_frame = frames[1].as_motion_frame()

            if reset_counter == 1600:
                displacement = [0, 0, 0]
                velocity = [0, 0, 0]
                reset_counter = 2000
            else:
                reset_counter += 1

            if not accel_frame and not gyro_frame:
                continue

            dt = (current_t - last_t) / 1e3
            dt = 1 / 100 if dt == 0 else dt

            acceleration = extract_motion_data(current_t, accel_frame.get_motion_data())

            for i in range(3):
                gravity[i] = alpha * gravity[i] + (1 - alpha) * acceleration[1 + i]
                acceleration[i + 1] = acceleration[i + 1] - gravity[i]

                velocity[i] = velocity[i] + acceleration[i + 1] * dt
                # Update the displacement
                displacement[i] = displacement[i] + (velocity[i] * dt) + (0.5 * acceleration[i + 1] * dt * dt)

            gyro_motion = extract_motion_data(current_t, gyro_frame.get_motion_data())

            accel_data.append(acceleration)
            gyro_data.append(gyro_motion)

            if current_t - last_t == 0:
                continue

            display = f"Acceleration: {acceleration[1]:+.3f} {acceleration[2]:+.3f} {acceleration[3]:+.3f}\t"
            display += f"Displacement: {displacement[0]:+.3f} {displacement[1]:+.3f} {displacement[2]:+.3f}\t"
            display += f"dt={dt:.3f}s"

            print(display, end="\r")
            last_t = current_t

        except KeyboardInterrupt:
            break
    pipeline.stop()


if __name__ == '__main__':
    start_imu_stream()
