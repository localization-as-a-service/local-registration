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


def load_files(config: Config):
    accel_df = pd.read_csv(os.path.join(config.get_motion_dir(), "accel.csv"))
    gyro_df = pd.read_csv(os.path.join(config.get_motion_dir(), "gyro.csv"))

    accel_df.drop_duplicates("timestamp", inplace=True)
    gyro_df.drop_duplicates("timestamp", inplace=True)

    imu_df = pd.merge(accel_df, gyro_df, on="timestamp", suffixes=("a", "g"))
    frame_rate = accel_df.shape[0] / (accel_df.timestamp.values[-1] - accel_df.timestamp.values[0]) * 1000
    
    return imu_df, frame_rate


def calc_error(imu_df: pd.DataFrame, config: Config):
    # load ground truth trajectory
    sequence_ts = fread.get_timstamps_from_images(config.get_sequence_dir(), ext=".depth.png")
    start_t = helpers.nearest(sequence_ts, imu_df.timestamp.values[0])
    start_index = np.where(sequence_ts == start_t)[0][0]

    pose_file = os.path.join(config.get_groundtruth_dir(), f"{config.get_file_name()}.pose.npz")
    trajectory_t = np.load(pose_file)["trajectory_t"]
        
    groundtruth_df = pd.DataFrame(
        np.concatenate((
            np.expand_dims(sequence_ts[start_index:], axis=1),
            trajectory_t[start_index:, :3, 3]
        ), axis=1),
        columns=["timestamp", "x", "y", "z"]
    )

    groundtruth_df.loc[:, "imu_t"] =groundtruth_df.apply(lambda x: helpers.nearest(imu_df.timestamp.values, x.timestamp), axis=1)

    result = pd.merge(groundtruth_df, imu_df.loc[:, ["timestamp", "x", "y", "z"]], left_on="imu_t", right_on="timestamp", suffixes=("_gt", "_imu"))
    result.drop(columns=["imu_t", "timestamp_imu"], inplace=True)

    result.loc[:, "error"] = result.apply(lambda x: np.linalg.norm(np.asarray(x[1:4]) - np.asarray(x[4:7])), axis=1)

    return result



def low_pass_filter_based_estimation(config: Config, alpha: float = 0.8, skip_first_seconds: int = 4):
    imu_df, frame_rate = load_files(config)
    
    # Low pass filter
    gravity = [0, 0, 0]
    
    for i in range(imu_df.shape[0]):
        for j in range(3):
            gravity[j] = alpha * gravity[j] + (1 - alpha) * imu_df.iloc[i, 1 + j]
            imu_df.iloc[i, 1 + j] = imu_df.iloc[i, 1 + j] - gravity[j]
    
    # remove nan values due to rolling mean
    imu_df = imu_df.iloc[int(frame_rate * skip_first_seconds):]
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
        
    result = calc_error(imu_df, config)
    result.to_csv(os.path.join(config.get_output_file(f"{config.get_file_name()}.csv")), index=False)
    
    
def moving_average_based_estimation(config: Config, window_size: int = 4):
    imu_df, frame_rate = load_files(config)
    win_len = int(frame_rate * window_size)

    accel_mavg = imu_df[["xa", "ya", "za"]].rolling(window=win_len).mean()
    imu_df[["xa", "ya", "za"]] = imu_df[["xa", "ya", "za"]] - accel_mavg
    
    # remove nan values due to rolling mean
    imu_df = imu_df.iloc[win_len:]
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
    
    result = calc_error(imu_df, config)
    result.to_csv(os.path.join(config.get_output_file(f"{config.get_file_name()}.csv")), index=False)


def orientation_tracking_based_approach(config: Config, window_size: int = 4):
    imu_df, frame_rate = load_files(config)
    win_len = int(frame_rate * window_size) # 3 seconds window

    gravity = imu_df.iloc[:win_len, [1, 2, 3]].mean().values
    
    imu_df = imu_df.iloc[win_len:]
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
        
        acceleration = imu_df.iloc[i, [1, 2, 3]].values
        acceleration = np.dot(rotation_matrix, np.array([*acceleration, 1]))
        acceleration = acceleration[:3] - gravity
        
        d = [(velocity[j] * dt) + (0.5 * acceleration[j] * dt * dt) for j in range(3)]
        
        # d = np.dot(rotation_matrix, np.array([*dd, 1]))
        
        imu_df.iloc[i, 8] = imu_df.iloc[i - 1, 8] + d[0]
        imu_df.iloc[i, 9] = imu_df.iloc[i - 1, 9] + d[1]
        imu_df.iloc[i, 10] = imu_df.iloc[i - 1, 10] + d[2]
        
        rotation_matrix = helpers.rotate_transformation_matrix(rotation_matrix, da[0], da[1], da[2])
        
        velocity = [velocity[j] + acceleration[j] * dt for j in range(3)]
        
    result = calc_error(imu_df, config)
    result.to_csv(os.path.join(config.get_output_file(f"{config.get_file_name()}.csv")), index=False)
    
    
def orientation_tracking_with_filter_based_estimation(config: Config, window_size: int = 4):
    imu_df, frame_rate = load_files(config)
    win_len = int(frame_rate * window_size)
    
    gravity = imu_df.iloc[:win_len, [1, 2, 3]].mean().values
    
    # compute dt in seconds
    imu_df.loc[:, "dt"] = np.concatenate([[0], (imu_df.timestamp.values[1:] - imu_df.timestamp.values[:-1]) / 1000])
    # remove first row as the dt is 0
    imu_df = imu_df.iloc[1:]
    # reset index in pandas data frame
    imu_df.reset_index(drop=True, inplace=True)

    # filter out gravity
    rotation_matrix = np.identity(4)

    velocity = [0, 0, 0]

    for i in tqdm.tqdm(range(1, len(imu_df))):
        v = imu_df.iloc[i].values
        da = np.degrees([v[j + 4] * v[7] for j in range(3)])
        
        acceleration = imu_df.iloc[i, [1, 2, 3]].values
        gravity_rotated = np.dot(rotation_matrix, np.array([*gravity, 1]))
        acceleration = acceleration - gravity_rotated[:3]
        
        imu_df.iloc[i, 1] = acceleration[0]
        imu_df.iloc[i, 2] = acceleration[1]
        imu_df.iloc[i, 3] = acceleration[2]

    # moving average filter
    accel_mavg = imu_df[["xa", "ya", "za"]].rolling(window=win_len).mean()
    accel_mavg.fillna(0, inplace=True)
    imu_df[["xa", "ya", "za"]] = imu_df[["xa", "ya", "za"]] - accel_mavg
    
    # remove first second's data
    imu_df = imu_df.iloc[400:]

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
    
    result = calc_error(imu_df, config)
    result.to_csv(os.path.join(config.get_output_file(f"{config.get_file_name()}.csv")), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=float, default=4)
    
    args = parser.parse_args()
    
    config = Config(
        feature_dir="data/features",
        sequence_dir="data/raw_data",
        experiment="exp_8",
        trial="trial_2",
        subject="subject-1",
        sequence="00",
        groundtruth_dir="data/trajectories/groundtruth",
        output_dir="data/results/orientation_tracking_and_moving_average_filter",
    )
    
    for trial in os.listdir(os.path.join(config.sequence_dir, config.experiment)):
        
        if not trial.startswith("trial") or trial == "trial_1":
            continue
        
        for subject in os.listdir(os.path.join(config.sequence_dir, config.experiment, trial)):
            for sequence in os.listdir(os.path.join(config.sequence_dir, config.experiment, trial, subject)):
                print(f"Processing: {config.sequence_dir} >> {trial} >> {subject} >> {sequence}")
                
                config.trial = trial
                config.sequence = sequence
                
                # low_pass_filter_based_estimation(config, alpha=0.9, skip_first_seconds=4)
                # moving_average_based_estimation(config, window_size=args.window)
                orientation_tracking_with_filter_based_estimation(config, window_size=args.window)
    