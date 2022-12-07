import argparse
import os

import cv2
import numpy as np
import pandas as pd
import pyrealsense2 as rs


def extract_motion_data(timestamp, data):
    return np.array([timestamp, data.x, data.y, data.z])


def start_imu_stream():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 200)
    config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)
    pipeline.start(config)

    accel_data = []
    gyro_data = []

    gravity = [0, 0, 0]
    alpha = 0.8

    while True:
        try:
            # Wait for a coherent pair of frames: depth and color
            frames: rs.composite_frame = pipeline.wait_for_frames()
            timestamp = frames.get_frame_metadata(rs.frame_metadata_value.time_of_arrival)

            accel_frame = frames[0].as_motion_frame()
            gyro_frame = frames[1].as_motion_frame()

            if not accel_frame and not gyro_frame:
                continue

            accel_motion = extract_motion_data(timestamp, accel_frame.get_motion_data())

            for i in range(3):
                gravity[i] = alpha * gravity[i] + (1 - alpha) * accel_motion[1 + i]
                accel_motion[1 + i] = accel_motion[1 + i] - gravity[i]

            gyro_motion = extract_motion_data(timestamp, gyro_frame.get_motion_data())

            accel_data.append(accel_motion)
            gyro_data.append(gyro_motion)

            print(f"Acceleration: {accel_motion[1]:+.3f} {accel_motion[2]:+.3f} {accel_motion[3]:+.3f}", end="\r")
        except KeyboardInterrupt:
            break
    pipeline.stop()


if __name__ == '__main__':
    start_imu_stream()
