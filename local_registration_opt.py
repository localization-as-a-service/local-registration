import numpy as np
import pandas as pd
import open3d
import os
import tqdm
import matplotlib.pyplot as plt
import utils.helpers as helpers
import utils.fread as fread
import utils.registration as registration
import copy

from utils.depth_camera import DepthCamera
from utils.config import Config

import utils.registration as registration
import utils.fread as fread
import utils.FCGF as FCGF
import utils.helpers as helpers

from scipy import signal
from scipy.ndimage import gaussian_filter1d


def m1(config: Config):
    pose_file = os.path.join(config.get_groundtruth_dir(), f"{config.get_file_name()}.pose.npz")
    motion_dir = config.get_motion_dir(include_secondary=False)

    accel_df = pd.read_csv(os.path.join(motion_dir, "accel.csv"))
    gyro_df = pd.read_csv(os.path.join(motion_dir, "gyro.csv"))

    accel_df.drop_duplicates("timestamp", inplace=True)
    gyro_df.drop_duplicates("timestamp", inplace=True)
    imu_df = pd.merge(accel_df, gyro_df, on="timestamp", suffixes=("a", "g"))

    frame_rate = accel_df.shape[0] / (accel_df.timestamp.values[-1] - accel_df.timestamp.values[0]) * 1000
    win_len = int(frame_rate * 4) # 4 seconds window

    gravity = imu_df.iloc[:win_len, [1, 2, 3]].mean().values
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

    # apply gaussian filter to smooth acceleration and gyro data
    imu_df.loc[:, "xa"] = gaussian_filter1d(imu_df.xa.values, sigma=10)
    imu_df.loc[:, "ya"] = gaussian_filter1d(imu_df.ya.values, sigma=10)
    imu_df.loc[:, "za"] = gaussian_filter1d(imu_df.za.values, sigma=10)
    imu_df.loc[:, "xg"] = gaussian_filter1d(imu_df.xg.values, sigma=10)
    imu_df.loc[:, "yg"] = gaussian_filter1d(imu_df.yg.values, sigma=10)
    imu_df.loc[:, "zg"] = gaussian_filter1d(imu_df.zg.values, sigma=10)
    
    # calculate displacement and rotation
    rotation_matrix = np.identity(4)
    velocity = [0, 0, 0]

    for i in range(1, len(imu_df)):
        v = imu_df.iloc[i].values
        da = np.degrees([v[j + 4] * v[7] for j in range(3)])
        
        acceleration = imu_df.iloc[i, [1, 2, 3]].values
        gravity_rotated = np.dot(rotation_matrix, np.array([*gravity, 1]))
        acceleration = acceleration - gravity_rotated[:3]
        
        imu_df.iloc[i, 1] = acceleration[0]
        imu_df.iloc[i, 2] = acceleration[1]
        imu_df.iloc[i, 3] = acceleration[2]
        
    accel_mavg = imu_df[["xa", "ya", "za"]].rolling(window=win_len).mean()
    accel_mavg.fillna(0, inplace=True)
    imu_df[["xa", "ya", "za"]] = imu_df[["xa", "ya", "za"]] - accel_mavg
    
    imu_df = imu_df.iloc[win_len:].copy()

    # load ground truth trajectory
    sequence_ts = fread.get_timstamps_from_images(config.get_sequence_dir(include_secondary=False), ext=".depth.png")
    start_t = helpers.nearest(sequence_ts, imu_df.timestamp.values[0])
    start_index = np.where(sequence_ts == start_t)[0][0]

    poses = np.load(pose_file)
    trajectory_t = poses["trajectory_t"]

    depth_camera = DepthCamera("secondary", os.path.join(config.sequence_dir, config.experiment, "metadata", "device-0-aligned.json"))

    local_pcds = []

    for t in range(len(sequence_ts)):
        depth_img_file = os.path.join(config.get_sequence_dir(include_secondary=False), f"frame-{sequence_ts[t]}.depth.png")
        pcd = depth_camera.depth_to_point_cloud(depth_img_file)
        pcd = pcd.voxel_down_sample(voxel_size=config.voxel_size)
        local_pcds.append(pcd)
        
    start_t = start_index
    end_t = start_index + 1

    gt_transform = np.dot(helpers.inv_transform(trajectory_t[start_t]), trajectory_t[end_t])
    velocity = gt_transform[:3, 3] * 1e3 / (sequence_ts[end_t] - sequence_ts[start_t])

    local_t = [np.identity(4), gt_transform]

    for t in range(start_index + 1, len(sequence_ts) - 1):
        start_t, end_t = t, t + 1
        
        imu_slice_df = imu_df[(imu_df.timestamp >= sequence_ts[start_t]) & (imu_df.timestamp <= sequence_ts[end_t])]
        
        # calculate displacement and rotation
        rotation_matrix = np.identity(4)
        translation = np.zeros(3)

        for i in range(len(imu_slice_df)):
            v = imu_slice_df.iloc[i].values
            
            dt = v[7]
            
            # current displacement and rotation
            da = np.degrees([v[j + 4] * dt for j in range(3)])
            
            acceleration = imu_slice_df.iloc[i, [1, 2, 3]].values

            d = [(velocity[j] * dt) + (0.5 * acceleration[j] * dt * dt) for j in range(3)]
            d = np.dot(rotation_matrix, np.array([*d, 1]))
            
            translation = translation + d[:3]
            velocity = [velocity[j] + acceleration[j] * dt for j in range(3)]
            
            rotation_matrix = helpers.rotate_transformation_matrix(rotation_matrix, da[0], da[1], da[2])
            
        trans_mat = np.identity(4)
        trans_mat[:3, 3] = translation
        trans_mat[:3, :3] = rotation_matrix[:3, :3]
        
        source = copy.deepcopy(local_pcds[end_t])
        target = copy.deepcopy(local_pcds[start_t])
        
        refined_transform = registration.icp_refinement(source, target, 0.05, trans_init=trans_mat, max_iteration=50, p2p=True).transformation
        
        velocity = refined_transform[:3, 3] * 1e3 / (sequence_ts[end_t] - sequence_ts[start_t])
        
        local_t.append(refined_transform)

    local_t = np.array(local_t)
    
    trajectory_estimated_t = [np.identity(4)]

    for t in range(1, len(local_t)):
        trajectory_estimated_t.append(np.dot(trajectory_estimated_t[t - 1], local_t[t]))
        
    trajectory_estimated_t = np.array(trajectory_estimated_t)

    # trajectory_pcd = []

    # for t in range(start_index, len(sequence_ts)):
    #     pcd = copy.deepcopy(local_pcds[t])
    #     pcd.transform(trajectory_estimated_t[t - start_index])
    #     trajectory_pcd.append(pcd)
        
    # trajectory_pcd = helpers.merge_pcds(trajectory_pcd, config.voxel_size)
    
    # open3d.visualization.draw_geometries([trajectory_pcd])
    
    estimated_pcd = helpers.make_pcd(trajectory_estimated_t[:, :3, 3])
    groundtruth_pcd = helpers.make_pcd(trajectory_t[start_index:, :3, 3])
    
    avg_distance_error = np.mean(np.linalg.norm(trajectory_estimated_t[:, :3, 3] - trajectory_t[start_index:, :3, 3], axis=1))
    
    # registration.view(estimated_pcd, groundtruth_pcd, np.identity(4))
    
    print(f"RMSE: {avg_distance_error}")
    
def m2(config):
    pose_file = os.path.join(config.get_groundtruth_dir(), f"{config.get_file_name()}.pose.npz")
    motion_dir = config.get_motion_dir(include_secondary=False)
    
    accel_df = pd.read_csv(os.path.join(motion_dir, "accel.csv"))
    gyro_df = pd.read_csv(os.path.join(motion_dir, "gyro.csv"))

    accel_df.drop_duplicates("timestamp", inplace=True)
    gyro_df.drop_duplicates("timestamp", inplace=True)
    imu_df = pd.merge(accel_df, gyro_df, on="timestamp", suffixes=("a", "g"))

    frame_rate = accel_df.shape[0] / (accel_df.timestamp.values[-1] - accel_df.timestamp.values[0]) * 1000

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
    
    win_len = int(frame_rate * 4) # 4 seconds window
    gravity = imu_df.iloc[:win_len, [1, 2, 3]].mean().values

    imu_df[["xa", "ya", "za"]] = imu_df[["xa", "ya", "za"]] - gravity
        
    accel_mavg = imu_df[["xa", "ya", "za"]].rolling(window=win_len).mean()
    accel_mavg.fillna(0, inplace=True)

    imu_df[["xa", "ya", "za"]] = imu_df[["xa", "ya", "za"]] - accel_mavg

    imu_df = imu_df.iloc[win_len:]

    sequence_ts = fread.get_timstamps_from_images(config.get_sequence_dir(include_secondary=False), ext=".depth.png")

    poses = np.load(pose_file)
    trajectory_t = poses["trajectory_t"]

    start_ind = 30 * 5
    sequence_ts = sequence_ts[start_ind:]
    trajectory_t = trajectory_t[start_ind:]

    elapsed_time = np.array((sequence_ts - sequence_ts[0]) // 1e3, dtype=np.int32)
    pcd_reg_time = 1 # seconds
    imu_reg_time = 5 # seconds

    duration = np.ceil(elapsed_time[-1] / (pcd_reg_time + imu_reg_time)).astype(np.int16) + 1
    pcd_start_ids = np.arange(duration) * (pcd_reg_time + imu_reg_time)
    imu_start_ids = pcd_start_ids + pcd_reg_time

    pcd_inds = np.concatenate([np.expand_dims(pcd_start_ids, axis=1), np.expand_dims(imu_start_ids, axis=1)], axis=1)
    imu_inds = np.concatenate([np.expand_dims(imu_start_ids[:-1], axis=1), np.expand_dims(pcd_start_ids[1:], axis=1)], axis=1)

    inds = np.concatenate([pcd_inds[:-1], imu_inds], axis=1)

    # find first indices of each second
    elapsed_time_inds = np.argwhere(np.diff(elapsed_time)).flatten() + 1
    # add zero to the beginning
    elapsed_time_inds = np.concatenate([[0], elapsed_time_inds])

    trajectory_complete = open3d.geometry.PointCloud()
    prev_transformation_mat = np.identity(4)
    trajectory_df = pd.DataFrame(columns=["timestamp", "x", "y", "z"])

    for i in range(len(inds) - 1):

        # first iteration calulcate the time shift
        start_pcd_t, end_pcd_t, start_imu_t, end_imu_t = inds[i]
        calibration_ts = np.arange(elapsed_time_inds[start_pcd_t], elapsed_time_inds[end_pcd_t])
        start_t = calibration_ts[-2] 
        end_t = calibration_ts[-1]

        gt_transform = np.dot(helpers.inv_transform(trajectory_t[start_t]), trajectory_t[end_t])

        velocity = gt_transform[:3, 3] * 1e3 / (sequence_ts[end_t] - sequence_ts[start_t])

        # calculate displacement and rotation
        rotation_matrix = np.identity(4)
        translation = np.zeros(3)

        start_t = start_t + 1
        end_t = elapsed_time_inds[end_imu_t]

        imu_slice_df = imu_df[(imu_df.timestamp >= sequence_ts[start_t]) & (imu_df.timestamp <= sequence_ts[end_t])].copy()

        for i in range(1, len(imu_slice_df)):
            v = imu_slice_df.iloc[i].values
            
            dt = v[7]
            
            # current displacement and rotation
            da = np.degrees([v[j + 4]* dt for j in range(3)])
            
            acceleration = imu_slice_df.iloc[i, [1, 2, 3]].values

            d = [(velocity[j] * dt) + (0.5 * acceleration[j] * dt * dt) for j in range(3)]
            d = np.dot(rotation_matrix, np.array([*d, 1]))
            
            translation = translation + d[:3]
            
            imu_slice_df.iloc[i, 8:] = translation
            
            velocity = [velocity[j] + acceleration[j] * dt for j in range(3)]
            
            rotation_matrix = helpers.rotate_transformation_matrix(rotation_matrix, da[0], da[1], da[2])
            
        imu_slice = helpers.make_pcd(imu_slice_df[["x", "y", "z"]].values)
        gt_slice = helpers.make_pcd(trajectory_t[elapsed_time_inds[start_pcd_t]:elapsed_time_inds[start_imu_t], :3, 3])

        imu_slice.paint_uniform_color([1, 0.706, 0])
        gt_slice.paint_uniform_color([0, 0.651, 0.929])

        imu_slice.transform(trajectory_t[calibration_ts[-1]])
        # for next iteration
        rotation_matrix[:3, 3] = translation
        rotation_matrix = np.dot(trajectory_t[calibration_ts[-1]], rotation_matrix)
        rotation_matrix = np.dot(helpers.inv_transform(trajectory_t[calibration_ts[0]]), rotation_matrix)
        
        trajectory = gt_slice + imu_slice
        trajectory.transform(helpers.inv_transform(trajectory_t[calibration_ts[0]]))
        
        trajectory.transform(prev_transformation_mat)
        
        timestamps = np.concatenate((sequence_ts[elapsed_time_inds[start_pcd_t]:elapsed_time_inds[start_imu_t]], imu_slice_df.timestamp.values))
        data = np.concatenate((timestamps.reshape(-1, 1), np.asarray(trajectory.points)), axis=1)

        trajectory_df = pd.concat([trajectory_df, pd.DataFrame(data, columns=["timestamp", "x", "y", "z"])])
        
        prev_transformation_mat = np.dot(prev_transformation_mat, rotation_matrix)
            
        trajectory_complete += trajectory
        
    gt_trajectory = helpers.make_pcd(trajectory_t[elapsed_time_inds[0]:elapsed_time_inds[end_imu_t], :3, 3])
    gt_trajectory.transform(helpers.inv_transform(trajectory_t[elapsed_time_inds[0]]))
    
    open3d.visualization.draw_geometries([trajectory_complete, gt_trajectory])

    timestamps = sequence_ts[elapsed_time_inds[0]:elapsed_time_inds[end_imu_t]]
    xyz = np.asarray(gt_trajectory.points)

    gt_trajectory_df = pd.DataFrame(np.concatenate((timestamps.reshape(-1, 1), xyz), axis=1), columns=["timestamp", "x", "y", "z"])
    
    closest_timestamps = np.array([helpers.nearest(trajectory_df.timestamp.values, t) for t in gt_trajectory_df.timestamp.values], dtype=np.int64)
    trajectory_df = trajectory_df[trajectory_df.timestamp.isin(closest_timestamps)].drop_duplicates()
    average_distance_error = np.mean(np.linalg.norm(trajectory_df[["x", "y", "z"]].values - gt_trajectory_df[["x", "y", "z"]].values, axis=1))

    print(f"Average distance error: {average_distance_error}")

    
if __name__ == "__main__":
    config = Config(
        sequence_dir="data/raw_data",
        feature_dir="data/features",
        output_dir="data/trajectories/estimated_imu+depth",
        experiment="exp_8",
        trial="trial_2",
        subject="subject-1",
        sequence="01",
        groundtruth_dir="data/trajectories/groundtruth",
        voxel_size=0.03
    )
    
    for i in range(1, 5):
        config.sequence = f"{i:02d}"
        m1(config)