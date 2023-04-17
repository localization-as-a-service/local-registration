import os
from time import time_ns

import numpy as np
import pyrealsense2 as rs

from utils.depth_camera import DepthCamera

device_id = 0
out_dir = "data/raw_data/exp_11/metadata"

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

width, height = 640, 480

pipeline = rs.pipeline()

config = rs.config()
config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)

profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()

print(f"Depth scale: {np.round(1 / depth_scale)}")
print(f"Depth intrinsics: {depth_intrinsics}")

DepthCamera.save_intrinsics(depth_intrinsics, width, height, depth_scale, f"{out_dir}/device-{device_id}.json")

align = rs.align(rs.stream.color)

pc = rs.pointcloud()

time_t = time_ns()

try:
    while True:
        frames = pipeline.wait_for_frames()

        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        aligned_color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not aligned_color_frame:
            continue

        depth_intrinsics = aligned_frames.get_profile().as_video_stream_profile().get_intrinsics()
        DepthCamera.save_intrinsics(depth_intrinsics, width, height, depth_scale,
                                    f"{out_dir}/device-{device_id}-aligned.json")
        print(f"Depth intrinsics (Aligned): {depth_intrinsics}")
        break

finally:
    pipeline.stop()
