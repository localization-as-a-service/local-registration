from copy import deepcopy
import os
from threading import Thread
from time import time_ns
import cv2
import numpy as np
import pandas as pd
import pyrealsense2 as rs
import argparse


class LiDARCamera(Thread):
    def __init__(self, out_dir, sequence, subject_id, resolution):
        super(LiDARCamera, self).__init__()

        self.out_dir = os.path.join(out_dir, f"secondary/subject-{subject_id}/{sequence:02d}/frames")

        if not os.path.exists(self.out_dir): os.makedirs(self.out_dir)

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, resolution[0], resolution[1], rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, resolution[0], resolution[1], rs.format.bgr8, 30)

        # Start streaming
        self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)
        self.duration = 20


    def run(self):
        start_t = time_ns()
        previous_t = time_ns()
        try:
            while True:
                frames = self.pipeline.wait_for_frames()

                aligned_frames = self.align.process(frames)
                
                aligned_depth_frame = aligned_frames.get_depth_frame() 
                aligned_color_frame = aligned_frames.get_color_frame()

                if not aligned_depth_frame or not aligned_color_frame:
                    continue

                depth_image = np.asanyarray(aligned_depth_frame.get_data())
                color_image = np.asanyarray(aligned_color_frame.get_data())
                
                images = deepcopy(color_image)
                
                # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                
                # images = np.hstack((color_image, depth_colormap))
                
                current_t = time_ns()
                elapsed_t = (current_t - start_t) / 1e9
                fps = int(1 / (current_t - previous_t) * 1e9)
                previous_t = current_t
                
                # putting the FPS count on the frame
                cv2.putText(images, f"{fps:2d} FPS (Elapsed Time: {elapsed_t:.1f}s)", (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3, cv2.LINE_AA)

                cv2.namedWindow("Secondary View", cv2.WINDOW_AUTOSIZE)
                cv2.imshow("Secondary View", images)
                  
                cv2.imwrite(f"{self.out_dir}/frame-{current_t}.color.png", color_image)
                cv2.imwrite(f"{self.out_dir}/frame-{current_t}.depth.png", depth_image)
                                
                # To get key presses
                key = cv2.waitKey(1)

                if key in (27, ord("q")) or cv2.getWindowProperty("Secondary View", cv2.WND_PROP_AUTOSIZE) < 0:
                    break
                
                if elapsed_t > self.duration:
                    break

        finally:
            # Stop streaming
            self.pipeline.stop()
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Lidar Capture for Intel RealSense L515')
    parser.add_argument('-r', '--resolution', type=int, default=640, help='Resolution of the image')
    parser.add_argument('-s', '--subject', type=int, default=0, help='Subject ID')
    parser.add_argument('-o', '--output', type=str, default='data/default', help='Output directory')
    parser.add_argument('-q', '--sequence', type=int, default=0, help='Sequence ID')

    args = parser.parse_args()
    
    if args.resolution == 640:
        resolution = (640, 480)
    elif args.resolution == 1280:
        resolution = (1024, 768)
    else:
        assert "Invalid resolution! The resolution must be either 640 or 1280"
    
    LiDARCamera(out_dir=args.output,
                sequence=args.sequence,
                resolution=resolution,
                subject_id=args.subject).start()