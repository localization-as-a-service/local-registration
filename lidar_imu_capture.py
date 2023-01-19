import argparse
import os
import time
import cv2
import numpy as np
import pandas as pd
import pyrealsense2 as rs
import multiprocessing as mp


def extract_motion_data(timestamp, data):
    return np.array([timestamp, data.x, data.y, data.z])


def save_data(data, output_file):
    df = pd.DataFrame(data, columns=["timestamp", "x", "y", "z"])
    df.to_csv(output_file, index=False)
    
    
def main(out_dir):
    camera_pipe = rs.pipeline()
    imu_pipe = rs.pipeline()
    
    camera_config = rs.config()
    imu_config = rs.config()
    
    camera_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    camera_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    imu_config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 400)
    imu_config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 400)
    
    align = rs.align(rs.stream.color)
    
    camera_pipe.start(camera_config)
    imu_pipe.start(imu_config)
    
    previous_t = [time.time() * 1000 for _ in range(2)]
    
    accel_data = []
    gyro_data = []
    
    start_t = time.time()
    elapsed_t = 0
    
    while True:
        try:
            elapsed_t = time.time() - start_t
            
            if elapsed_t > 20:
                break
            
            camera_frames: rs.composite_frame = camera_pipe.poll_for_frames()
            if camera_frames:
                current_t = camera_frames.get_frame_metadata(rs.frame_metadata_value.time_of_arrival)
                aligned_frames = align.process(camera_frames)

                aligned_depth_frame = aligned_frames.get_depth_frame()
                aligned_color_frame = aligned_frames.get_color_frame()

                if not aligned_depth_frame or not aligned_color_frame:
                    continue

                depth_image = np.asanyarray(aligned_depth_frame.get_data())
                color_image = np.asanyarray(aligned_color_frame.get_data())
                
                fps = int(1 / (current_t - previous_t[0] + 1) * 1e3)
                previous_t[0] = current_t

                # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

                # images = np.hstack((color_image, depth_colormap))

                cv2.imwrite(os.path.join(out_dir, "frames", f"frame-{current_t}.color.png"), color_image)
                cv2.imwrite(os.path.join(out_dir, "frames", f"frame-{current_t}.depth.png"), depth_image)
                
                message = f"FPS: {fps:2d} | Elapsed Time: {elapsed_t:03.2f}s | " + ("Calibration" if elapsed_t < 5 else "Move")

                cv2.putText(color_image, message, (7, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 76, 231), 2, cv2.LINE_AA)
                cv2.namedWindow("Secondary View", cv2.WINDOW_AUTOSIZE)
                cv2.imshow("Secondary View", color_image)

                # To get key presses
                key = cv2.waitKey(1)

                if key in (27, ord("q")) or cv2.getWindowProperty("Secondary View", cv2.WND_PROP_AUTOSIZE) < 0:
                    cv2.destroyAllWindows()
                    break
                
            motion_frames: rs.composite_frame = imu_pipe.poll_for_frames()
            if motion_frames:
                current_t = motion_frames.get_frame_metadata(rs.frame_metadata_value.time_of_arrival)
                
                accel_frame = motion_frames[0].as_motion_frame()
                gyro_frame = motion_frames[1].as_motion_frame()
                
                if (not accel_frame and not gyro_frame) or (current_t - previous_t[1] < 1e3 / 400):
                    continue
                
                accel_data.append(extract_motion_data(current_t, accel_frame.get_motion_data()))
                gyro_data.append(extract_motion_data(current_t, gyro_frame.get_motion_data()))
                
                fps = int(1 / (current_t - previous_t[1] + 1) * 1e3)
                previous_t[1] = current_t
                
                print("Number of accel samples: ", len(accel_data), end="\r")
                
        except KeyboardInterrupt:
            break
        
    save_data(accel_data, os.path.join(out_dir, "motion", "accel.csv"))
    save_data(gyro_data, os.path.join(out_dir, "motion", "gyro.csv"))
        
    camera_pipe.stop()
    imu_pipe.stop()
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Intel Lidar L515")
    parser.add_argument("--experiment", default=0, type=int)
    parser.add_argument("--trial", default=0, type=int)
    parser.add_argument("--subject", default=0, type=int)
    parser.add_argument("--sequence", default=0, type=int)
    args = parser.parse_args()

    out_dir = f"data/raw_data/exp_{args.experiment}/trial_{args.trial}/subject-{args.subject}/{args.sequence:02d}"

    if not os.path.exists(out_dir):
        os.makedirs(os.path.join(out_dir, "frames"))
        os.makedirs(os.path.join(out_dir, "motion"))

    # if args.mode == "depth":
    #     start_depth_camera(os.path.join(output_path, "frames"))
    # else:
    #     start_imu_stream(os.path.join(output_path, "motion"))
    
    main(out_dir)
