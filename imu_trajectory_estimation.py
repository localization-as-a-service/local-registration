import numpy as np
import pandas as pd
import open3d
import os
import tqdm
import matplotlib.pyplot as plt
import utils.helpers as helpers
import utils.fread as fread
import utils.registration as registration
import argparse

from utils.depth_camera import DepthCamera
from utils.config import Config


def main(config: Config, args: argparse.Namespace):
    motion_dir = config.get_motion_dir()
    # load imu data
    accel_df = pd.read_csv(os.path.join(motion_dir, "accelerometer"))
    gyro_df = pd.read_csv(os.path.join(motion_dir, "gyroscope"))
    # remove duplicates
    accel_df.drop_duplicates("timestamp", inplace=True)
    gyro_df.drop_duplicates("timestamp", inplace=True)
    # combine accel and gyro data
    imu_df = pd.merge(accel_df, gyro_df, on="timestamp", suffixes=("a", "g"))
    # remove gravity
    accel_mavg = imu_df[["xa", "ya", "za"]].rolling(window=args.window).mean()
    imu_df[["xa", "ya", "za"]] = imu_df[["xa", "ya", "za"]] - accel_mavg
    # remove nan values due to rolling mean
    imu_df.dropna(inplace=True)
    # compute dt in seconds
    imu_df.loc[:, "dt"] = np.concatenate([[0], (imu_df.timestamp.values[1:] - imu_df.timestamp.values[:-1]) / 1000])
    # remove first row as the dt is 0
    imu_df = imu_df.iloc[1:]
    # reset index in pandas data frame
    imu_df.reset_index(drop=True, inplace=True)
    
    # Fill 0 for displacement, angles, and coordinates
    imu_df.loc[:, "x"] = np.zeros(len(imu_df))
    imu_df.loc[:, "y"] = np.zeros(len(imu_df))
    imu_df.loc[:, "z"] = np.zeros(len(imu_df))
    
    # calculate displacement and rotation
    rotation_matrix = np.identity(4)
    
    velocity = [0, 0, 0]
    
    for i in tqdm.tqdm(range(1, len(imu_df))):
        v = imu_df.iloc[i].values
        dt = v[7]
        # current displacement and rotation
        da = np.degrees([v[j + 4] * dt for j in range(3)])
        dd = [(velocity[j] * dt) + (0.5 * v[j + 1] * dt * dt) for j in range(3)]
        
        d = np.dot(rotation_matrix, np.array([*dd, 1]))
        
        imu_df.iloc[i, 8] = imu_df.iloc[i - 1, 8] + d[0]
        imu_df.iloc[i, 9] = imu_df.iloc[i - 1, 9] + d[1]
        imu_df.iloc[i, 10] = imu_df.iloc[i - 1, 10] + d[2]
        
        rotation_matrix = helpers.rotate_transformation_matrix(rotation_matrix, da[0], da[1], da[2])
        velocity = [velocity[j] + v[j + 1] * dt for j in range(3)]
        
    # create trajectory as a pcd
    xyz = imu_df.loc[:, ["x", "y", "z"]].values
    pcd = helpers.make_pcd(xyz)
    pcd.paint_uniform_color([1, 0, 0])
    
    # load ground truth trajectory
    pose_file = os.path.join(config.get_groundtruth_dir(), f"{config.get_file_name()}.pose.npz")
    trajectory_t = np.load(pose_file)["trajectory_t"]
    pcd_gt = helpers.make_pcd(trajectory_t[:, :3, 3])
    pcd_gt.paint_uniform_color([0, 1, 0])
    
    open3d.visualization.draw_geometries([pcd_gt, pcd])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=400)
    parser.add_argument("--sequence", type=int, default=2)
    
    args = parser.parse_args()
    
    config = Config(
        feature_dir="data/features",
        sequence_dir="data/raw_data",
        experiment="exp_7",
        trial="trial_1",
        subject="subject-1",
        sequence=f"{args.sequence:02d}",
        groundtruth_dir="data/trajectories/groundtruth",
        output_dir="data/trajectories/estimated",
    )
    
    main(config, args)
