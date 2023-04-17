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
import utils.grid_search as grid_search


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
        pcd = helpers.remove_statistical_outliers(pcd)
        pcd.transform(trajectory_t[t])
        local_pcds.append(pcd)
        
    trajectory = helpers.merge_pcds(local_pcds, voxel_size)
        
    open3d.visualization.draw_geometries([trajectory])
    
    
def create_fragments(dataset_dir, experiment, trial, subject, sequence, voxel_size, out_dir):
    min_std = 0.5
    threshold = 0.5
    target_fps = 20
    cutoff_margin = 5 # frames
    
    sequence_dir = f"data/raw_data/{experiment}/{trial}/secondary/{subject}/{sequence}/frames"
    feature_dir = os.path.join("data/features", experiment, trial, str(voxel_size), subject, sequence)

    sequence_ts = fread.get_timstamps(feature_dir, ext=".secondary.npz")
    num_frames = len(sequence_ts)

    file_name = f"{experiment}__{trial}__{subject}__{sequence}"
    
    sequence_ts = fread.get_timstamps_from_images(sequence_dir, ext=".depth.png")
    sequence_ts = helpers.sample_timestamps(sequence_ts, target_fps)
    num_frames = len(sequence_ts)
    
    print("     :: Caclulating Std. of frames.")
    
    std_values = []

    for t in range(len(sequence_ts)):
        depth_img = Image.open(os.path.join(sequence_dir, f"frame-{sequence_ts[t]}.depth.png")).convert("I")
        depth_img = np.array(depth_img) / 4000
        std_values.append(np.std(depth_img))
        
    std_values = np.array(std_values)

    print("     :: Caclulating cut-off frames.")
    
    cutoffs = registration.find_cutoffs(std_values, target_fps, min_std, threshold)
    cutoffs = np.concatenate([[0], cutoffs, [num_frames - 1]])
    cutoffs = list(zip(cutoffs[:-1] + cutoff_margin, cutoffs[1:] - cutoff_margin))
    
    if not os.path.exists(os.path.join(f"data/trajectories/groundtruth/{experiment}", f"{file_name}.pose.npz")):
        print("File not found!")
        return
        
    local_t = np.load(os.path.join(f"data/trajectories/groundtruth/{experiment}", f"{file_name}.pose.npz"))["local_t"]
    
    print("     :: Caching local PCDs.")
    
    local_pcds = []

    for t in tqdm.trange(num_frames):
        feature_file = os.path.join(feature_dir, f"{sequence_ts[t]}.secondary.npz")
        pcd, _, _ = FCGF.get_features(feature_file)
        local_pcds.append(pcd)
        
    print("     :: Making fragments.")
        
    fragments = []
    for start_t, end_t in tqdm.tqdm(cutoffs):
        trajectory_t = [np.identity(4)]

        for t in range(start_t + 1, end_t):
            trajectory_t.append(np.dot(trajectory_t[t - start_t - 1], local_t[t]))
        
        fragment = []
        for t in range(start_t, end_t):
            local_temp = copy.deepcopy(local_pcds[t])
            local_temp.transform(trajectory_t[t - start_t])
            fragment.append(local_temp)
            
        fragment = helpers.merge_pcds(fragment, 0.03)
        fragments.append(fragment)
    
    print("     :: Saving fragments.")
    
    fragments_dir = "data/fragments"
    
    if not os.path.exists(os.path.join(fragments_dir, experiment)):
        os.makedirs(os.path.join(fragments_dir, experiment))
        
    for i, fragment in enumerate(fragments):
        open3d.io.write_point_cloud(os.path.join(fragments_dir, experiment, f"{file_name}__{i:02d}.pcd"), fragment)
        
        
def register_fragments(dataset_dir, experiment, trial, subject, sequence, voxel_size, out_dir):
    file_name = f"{experiment}__{trial}__{subject}__{sequence}"
    
    fragment_files = glob.glob(os.path.join(f"data/fragments/{experiment}", f"{file_name}__*.npz"))
    fragment_files = sorted(fragment_files, key=lambda f: int(os.path.basename(f).split(".")[0].split("__")[-1]))
    
    fragments = []
    fragment_t = []
    
    for i in tqdm.trange(len(fragment_files)):
        fragment_pcd, global_pcd, refine_reg = grid_search.global_registration(
            src_feature_file=fragment_files[i],
            tgt_feature_file="data/reference/larc_kitchen_v3.npz",
            cell_size=3,
            voxel_size=0.03,
            refine_enabled=True
        )
        fragments.append(fragment_pcd)
        fragment_t.append(refine_reg.transformation)
    
    np.savez(os.path.join(out_dir, f"{file_name}.fragment.pose.npz"), fragment_t=fragment_t)


def seperate_poses(dataset_dir, experiment, trial, subject, sequence, voxel_size, out_dir):
    file_name = f"{experiment}__{trial}__{subject}__{sequence}"
    
    fragment_t = np.load(os.path.join(f"data/trajectories/groundtruth/{experiment}", f"{file_name}.fragment.pose.npz"))["fragment_t"]
    fragment_files = glob.glob(os.path.join(f"data/fragments/{experiment}", f"{file_name}__*.npz"))
    fragment_files = sorted(fragment_files, key=lambda f: int(os.path.basename(f).split(".")[0].split("__")[-1]))

    for i in range(len(fragment_t)):
        np.savetxt(os.path.normpath(fragment_files[i].replace("npz", "txt")), fragment_t[i])


if __name__ == "__main__":
    VOXEL_SIZE = 0.03
    ROOT_DIR = "data/features"
    EXPERIMENT = "exp_11"
    OUT_DIR = "data/trajectories/groundtruth/exp_11"

    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)
    
    for trial in os.listdir(os.path.join(ROOT_DIR, EXPERIMENT)):
        for subject in os.listdir(os.path.join(ROOT_DIR, EXPERIMENT, trial, str(VOXEL_SIZE))):
            for sequence in os.listdir(os.path.join(ROOT_DIR, EXPERIMENT, trial, str(VOXEL_SIZE), subject)):
                print(f"Processing: {EXPERIMENT} >> {trial} >> {subject} >> {sequence}")
                register_frames(ROOT_DIR, EXPERIMENT, trial, subject, sequence, VOXEL_SIZE, OUT_DIR)    
