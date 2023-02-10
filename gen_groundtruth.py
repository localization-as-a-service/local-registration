import open3d
import numpy as np
import pandas as pd
import os
import tqdm
import copy
import glob

from scipy.signal import argrelmin
from PIL import Image

import utils.registration as registration
import utils.fread as fread
import utils.FCGF as FCGF
import utils.helpers as helpers


def register_frames(dataset_dir, experiment, trial, subject, sequence, voxel_size, out_dir):
    sequence_dir = os.path.join(dataset_dir, experiment, trial, str(voxel_size), subject, sequence)
    sequence_ts = fread.get_timstamps(sequence_dir, ext=".secondary.npz")
    num_frames = len(sequence_ts)
    
    file_name = f"{experiment}__{trial}__{subject}__{sequence}"
    
    if os.path.exists(os.path.join(out_dir, f"{file_name}.pose.npz")):
        return
    
    print("     :: Number of frames: {}".format(num_frames))
    
    local_pcds = []
    local_feat = []
    
    print("     :: Caching local PCDs and features.")

    for t in tqdm.trange(num_frames):
        feature_file = os.path.join(sequence_dir, f"{sequence_ts[t]}.secondary.npz")
        pcd, features, _ = FCGF.get_features(feature_file)
        local_pcds.append(pcd)
        local_feat.append(features)
    
    print("     :: Registering local PCDs.")
    
    local_t = [np.identity(4)]

    for t in tqdm.trange(num_frames - 1):
        source, source_feat = copy.deepcopy(local_pcds[t + 1]), local_feat[t + 1]
        target, target_feat = copy.deepcopy(local_pcds[t]), local_feat[t]

        ransac_reg = registration.ransac_feature_matching(source, target, source_feat, target_feat, n_ransac=3, threshold=0.05, p2p=False)
        icp_reg = registration.icp_refinement(source, target, threshold=0.05, trans_init=ransac_reg.transformation, max_iteration=200, p2p=False)
        
        local_t.append(icp_reg.transformation)
        
    print("     :: Calculating transformations.")
    
    trajectory_t = [np.identity(4)]

    for t in tqdm.trange(1, num_frames):
        trajectory_t.append(np.dot(trajectory_t[t - 1], local_t[t]))

    print("     :: Saving pose information.")
    
    np.savez(os.path.join(out_dir, f"{file_name}.pose.npz"), local_t=local_t, trajectory_t=trajectory_t)
    
    print("     :: Done.")
    

def visualize(dataset_dir, experiment, trial, subject, sequence, voxel_size, out_dir):
    sequence_dir = os.path.join(dataset_dir, experiment, trial, str(voxel_size), subject, sequence)
    sequence_ts = fread.get_timstamps(sequence_dir, ext=".secondary.npz")
    num_frames = len(sequence_ts)
    
    file_name = f"{experiment}__{trial}__{subject}__{sequence}"
    
    if not os.path.exists(os.path.join(out_dir, f"{file_name}.pose.npz")):
        print("     :: No pose information found.")
        return
    
    pose = np.load(os.path.join(out_dir, f"{file_name}.pose.npz"))
    trajectory_t = pose["trajectory_t"]
    
    print("     :: Number of frames: {}".format(num_frames))
    
    local_pcds = []
    
    for t in tqdm.trange(num_frames):
        feature_file = os.path.join(sequence_dir, f"{sequence_ts[t]}.secondary.npz")
        pcd = FCGF.get_features(feature_file, pcd_only=True)
        pcd.transform(trajectory_t[t])
        local_pcds.append(pcd)
        
    open3d.visualization.draw_geometries(local_pcds)

if __name__ == "__main__":
    VOXEL_SIZE = 0.03
    ROOT_DIR = "data/features"
    EXPERIMENT = "exp_10"
    OUT_DIR = "data/trajectories/groundtruth/exp_10"

    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)
    
    data = []
    
    for trial in os.listdir(os.path.join(ROOT_DIR, EXPERIMENT)):
        for subject in os.listdir(os.path.join(ROOT_DIR, EXPERIMENT, trial, str(VOXEL_SIZE))):
            for sequence in os.listdir(os.path.join(ROOT_DIR, EXPERIMENT, trial, str(VOXEL_SIZE), subject)):
                print(f"Processing: {EXPERIMENT} >> {trial} >> {subject} >> {sequence}")
                visualize(ROOT_DIR, EXPERIMENT, trial, subject, sequence, VOXEL_SIZE, OUT_DIR)    