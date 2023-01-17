from ast import arg
import pyrealsense2 as rs
import multiprocessing as mp
import numpy as np
import cv2
import os
import argparse

from time import time_ns

# create a class to save image with a process
class ImageSaver(mp.Process):
    def __init__(self, queue: mp.Queue):
        super(ImageSaver, self).__init__()
        self.queue = queue

    # save image
    def save_image(self, image: np.array, file_name: str):
        cv2.imwrite(file_name, image)
        
    # listen to the queue and save image
    def run(self) -> None:
        while True:
            try:
                image, file_name = self.queue.get()
                self.save_image(image, file_name)
            except KeyboardInterrupt:
                break
        self.join()


def main(args, queue: mp.Queue):
    out_dir = os.path.join(args.output, "global", f"device-{args.device}")

    if not os.path.exists(out_dir): os.makedirs(out_dir)
    
    if args.resolution == 640:
        width, height = 640, 480
    elif args.resolution == 1280:
        width, height = 1280, 768
    else:
        print("Invalid resolution! The resolution must be either 640 or 1280")

    pipeline = rs.pipeline()
    
    config = rs.config()
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)

    pipeline.start(config)
    align = rs.align(rs.stream.color)

    try:
        previous_t = time_ns()

        while True:
            frames = pipeline.wait_for_frames()

            aligned_frames = align.process(frames)
            # aligned_depth_frame is a 640x480 depth image
            aligned_depth_frame = aligned_frames.get_depth_frame() 
            # aligned_color_frame is a 640x480 depth image
            aligned_color_frame = aligned_frames.get_color_frame()

            if not aligned_depth_frame or not aligned_color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(aligned_color_frame.get_data())

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            # Concatenate the two images
            # images = np.hstack((color_image, depth_colormap))

            # current_t = time_ns()
            current_t = frames.get_frame_metadata(rs.frame_metadata_value.time_of_arrival)
            
            # if current_t - previous_t < 1e6: continue
            
            fps = int(1 / (current_t - previous_t + 1) * 1e3)
            previous_t = current_t

            # cv2.imwrite(f"{out_dir}/frame-{current_t}.color.png", color_image)
            # cv2.imwrite(f"{out_dir}/frame-{current_t}.depth.png", depth_image)
            queue.put_nowait((color_image, f"{out_dir}/frame-{current_t}.color.png"))
            queue.put_nowait((depth_image, f"{out_dir}/frame-{current_t}.depth.png"))

            # putting the FPS count on the frame
            cv2.putText(color_image, f"{fps:2d} FPS", (7, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3, cv2.LINE_AA)
            cv2.namedWindow("Aligned RGB & Depth", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("Aligned RGB & Depth", color_image)

            key = cv2.waitKey(1)

            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break

    finally:
        pipeline.stop()
        
    
    
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Lidar Capture for Intel RealSense L515')
    parser.add_argument('-r', '--resolution', type=int, default=640, help='Resolution of the image')
    parser.add_argument('-d', '--device', type=int, default=0, help='Device ID')
    parser.add_argument('-o', '--output', type=str, default='data/default', help='Output directory')
    parser.add_argument('-t', '--time', type=int, default=100000000, help='Time between captures')

    args = parser.parse_args()
    
    queue = mp.Queue()
    
    image_saver = ImageSaver(queue)
    image_saver.start()

    main(args, queue)
    
    image_saver.join()



