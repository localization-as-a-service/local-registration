import os
from time import time_ns
from multiprocessing import Process
import numpy as np
import pandas as pd
import pyrealsense2 as rs
import argparse
import cv2


def extract_motion_data(data):
    return np.array([data.x, data.y, data.z])


def start_imu_stream():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 200)
    pipeline.start(config)

    while True:
        try:
            # Wait for a coherent pair of frames: depth and color
            frames: rs.composite_frame = pipeline.wait_for_frames()

            motion_frame = frames[0].as_motion_frame()
            if not motion_frame:
                continue
            motion = extract_motion_data(motion_frame.get_motion_data())
            print(motion)
        except KeyboardInterrupt:
            break
    pipeline.stop()


def start_depth_camera():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)
    align = rs.align(rs.stream.color)

    while True:
        frames = pipeline.wait_for_frames()

        aligned_frames = align.process(frames)

        aligned_depth_frame = aligned_frames.get_depth_frame()
        aligned_color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not aligned_color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(aligned_color_frame.get_data())

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        images = np.hstack((color_image, depth_colormap))

        cv2.namedWindow("Secondary View", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("Secondary View", images)

        # To get key presses
        key = cv2.waitKey(1)

        if key in (27, ord("q")) or cv2.getWindowProperty("Secondary View", cv2.WND_PROP_AUTOSIZE) < 0:
            break

    pipeline.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Intel Lidar L515")
    parser.add_argument("--mode", choices=["depth", "imu"], default="depth")
    args = parser.parse_args()

    if args.mode == "depth":
        start_depth_camera()
    else:
        start_imu_stream()
