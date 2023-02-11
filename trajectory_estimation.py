import open3d
import numpy as np
import pandas as pd
import os
import glob
import tqdm
import copy

from time import sleep
from scipy.signal import argrelmin, argrelmax
from PIL import Image
from utils.config import Config

import utils.registration as registration
import utils.fread as fread
import utils.FCGF as FCGF
import utils.helpers as helpers
import utils.grid_search as grid_search


def estimate_trajectory(config: Config):
    sequence_dir = config.get_sequence_dir(include_secondary=True)
    feature_dir = config.get_feature_dir()

    sequence_ts = fread.get_timstamps(feature_dir, ext=".secondary.npz")
    sequence_ts = helpers.sample_timestamps(sequence_ts, config.target_fps)
    num_frames = len(sequence_ts)
    
    if os.path.exists(config.get_output_file(config.get_file_name() + ".npz")):
        return
    
    print("     :: Calculating standard deviation of frames.")
    
    std_values = []

    for t in tqdm.trange(len(sequence_ts)):
        depth_img = Image.open(os.path.join(sequence_dir, f"frame-{sequence_ts[t]}.depth.png")).convert("I")
        depth_img = np.array(depth_img) / 4000
        std_values.append(np.std(depth_img))
        
    std_values = np.array(std_values)
    
    print("     :: Caching local PCDs and features.")
    
    local_pcds = []

    for t in tqdm.trange(num_frames):
        feature_file = os.path.join(feature_dir, f"{sequence_ts[t]}.secondary.npz")
        pcd = FCGF.get_features(feature_file, pcd_only=True)
        local_pcds.append(pcd)
    
    cache_path = f"data/trajectories/cache/{config.experiment}/{config.get_file_name()}.local.npz"
    
    assert os.path.exists(cache_path), "Local cache not found!"
    
    local_t = np.load(cache_path)["local_t"] 
        
    device_0_ts = fread.get_timstamps_from_images(os.path.join("data/raw_data/", config.experiment, config.trial, "global", "device-0"), ext=".depth.png")
    device_1_ts = fread.get_timstamps_from_images(os.path.join("data/raw_data/", config.experiment, config.trial, "global", "device-1"), ext=".depth.png")
    device_2_ts = fread.get_timstamps_from_images(os.path.join("data/raw_data/", config.experiment, config.trial, "global", "device-2"), ext=".depth.png")

    y = [[], [], []]

    for i in range(num_frames):
        y[0].append(helpers.nearest(device_0_ts, sequence_ts[i]))
        y[1].append(helpers.nearest(device_1_ts, sequence_ts[i]))
        y[2].append(helpers.nearest(device_2_ts, sequence_ts[i]))
        
        y[0][i] = np.abs(y[0][i] - sequence_ts[i])
        y[1][i] = np.abs(y[1][i] - sequence_ts[i])
        y[2][i] = np.abs(y[2][i] - sequence_ts[i])
        
    print("     :: Finding optimal global positions for registration")

    global_frame_delays = np.array(y)
    global_frame_delays_inds = np.ones(global_frame_delays.shape, dtype=np.int8)

    for r, c in np.argwhere(global_frame_delays > 100):
        global_frame_delays_inds[r, c] = 0
        
    global_frame_delays_inds = np.sum(global_frame_delays_inds, axis=0)
    global_frame_delays_inds = np.where(global_frame_delays_inds == 3, 1, 0)
        
    global_pos = [0]
    for t in tqdm.trange(num_frames):
        if global_frame_delays_inds[t] == 0: continue
        
        if t - global_pos[-1] >= config.target_fps * 0.8: 
            global_pos.append(t)
            continue
        
        if (np.abs(std_values[t] - std_values[global_pos[-1]]) > config.delta) and (t - global_pos[-1] > config.target_fps * 0.5):
            global_pos.append(t)

    global_pos = np.array(global_pos)
    
    print("     :: Finding cutoffs for local registration")
    
    cutoffs = registration.find_cutoffs(std_values, config.target_fps, config.min_std)
    cutoffs = np.concatenate([[0], cutoffs, [num_frames - 1]])

    cutoffs = [[cutoffs[i] + config.cutoff_margin, cutoffs[i + 1] - config.cutoff_margin] for i in range(len(cutoffs) - 1)]
    # check if the first frame in the last cutoff is the last frame in the sequence 
    cutoffs[-1][0] = min(num_frames - 1, cutoffs[-1][0])

    cutoffs = [c for c in cutoffs if c[0] < c[1]]
    
    print("     :: Global registration & verification.")
    
    global_t = [np.identity(4) for _ in range(num_frames)]

    for start_t, end_t in cutoffs:
        global_inds = global_pos[np.logical_and(global_pos >= start_t, global_pos <= end_t)]
        
        global_target_t = []
        found_correct_global = False
        found_correct_global_at = -1

        for t in tqdm.trange(len(global_inds)):
            if found_correct_global:
                break
            else:
                source_feature_file = os.path.join(feature_dir, f"{sequence_ts[global_inds[t]]}.secondary.npz")
                target_feature_file = os.path.join(feature_dir, f"{sequence_ts[global_inds[t]]}.global.npz")
                _, _, reg_result = grid_search.global_registration(source_feature_file, target_feature_file, config.voxel_size, cell_size=2, refine_enabled=True)
                global_target_t.append(reg_result.transformation if reg_result else np.identity(4))
                
            if t > 1 and not found_correct_global:
                total = 0
                for i in range(t, t - 3, -1):
                    if np.sum(global_target_t[i]) == 4:
                        total += 1
                        
                if total > 1: continue
                
                try:
                    global_target_t[t - 2], global_target_t[t - 1], global_target_t[t] = grid_search.validate(
                        global_target_t[t - 2], global_target_t[t - 1], global_target_t[t], 
                        helpers.merge_transformations(global_inds[t - 2], global_inds[t - 1], local_t),
                        helpers.merge_transformations(global_inds[t - 1], global_inds[t], local_t),
                        max_rot=2, max_dist=0.1
                    )
                    found_correct_global = True
                    found_correct_global_at = t
                except Exception as e:
                    continue
                
        if found_correct_global:
            global_t[global_inds[found_correct_global_at]] = global_target_t[found_correct_global_at]

            for t in range(global_inds[found_correct_global_at] + 1, end_t):
                global_t[t] = np.dot(global_t[t - 1], local_t[t])
                
            for t in range(global_inds[found_correct_global_at] - 1, start_t - 1, -1):
                global_t[t] = np.dot(global_t[t + 1], helpers.inv_transform(local_t[t + 1]))
                
        else:
            print("     :: Finding correct point failed. Registering every frame globally.")
            for t in tqdm.tqdm(range(start_t, end_t, 5)):
                source_feature_file = os.path.join(feature_dir, f"{sequence_ts[t]}.secondary.npz")
                target_feature_file = os.path.join(feature_dir, f"{sequence_ts[t]}.global.npz")
                _, _, reg_result = grid_search.global_registration(source_feature_file, target_feature_file, config.voxel_size, cell_size=2, refine_enabled=True)
                global_t[t] = reg_result.transformation if reg_result else np.identity(4)
    
    print("     :: Saving results.")
         
    np.savez_compressed(config.get_output_file(config.get_file_name() + ".npz"), 
                        global_t=global_t, 
                        local_t=local_t,
                        info = [found_correct_global, found_correct_global_at]
                        )


if __name__ == "__main__":
    config = Config(
        sequence_dir="data/raw_data",
        feature_dir="data/features",
        output_dir="data/trajectories/estimated_baseline",
        experiment="exp_10",
        trial="trial_1",
        subject="subject-1",
        sequence="01",
        groundtruth_dir="data/trajectories/groundtruth",
    )
    
    config.voxel_size=0.03
    config.target_fps=20
    config.min_std=0.5
    
    for trial in os.listdir(os.path.join(config.feature_dir, config.experiment)):
        config.trial = trial
        for subject in os.listdir(os.path.join(config.feature_dir, config.experiment, config.trial, str(config.voxel_size))):
            config.subject = subject    
            for sequence in os.listdir(os.path.join(config.feature_dir, config.experiment, config.trial, str(config.voxel_size), config.subject)):
                config.sequence = sequence
                print(f"Processing: {config.experiment} >> {config.trial} >> {config.subject} >> {config.sequence}")
                estimate_trajectory(config)


    """_summary_
    v7 - Improved alignment + stats + global cam availability + std=0.6
    v8 - Improved alignment + stats + global cam availability + std=0.5
    v9 - Old alignment + stats + global cam availability + std=0.5
    """
    