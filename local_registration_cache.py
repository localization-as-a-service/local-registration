import open3d
import numpy as np
import tqdm
import os
import time

import utils.fread as fread
import utils.helpers as helpers
import utils.FCGF as FCGF
import utils.registration as registration

from utils.config import Config


def main(config: Config):
    output_path = config.get_output_file(f"{config.get_file_name()}.local.npz")
    
    if os.path.exists(output_path):
        print(f"-> Skipping {config.get_file_name()}.local.npz")
        return
    
    features_dir = config.get_feature_dir()
    sequence_ts = fread.get_timstamps(features_dir, ext=".secondary.npz")
    
    elapsed_times = (sequence_ts - sequence_ts[0]) / 1e3
    sequence_ts = sequence_ts[np.where(elapsed_times > 4, 1, 0) == 1]

    num_frames = len(sequence_ts)

    print(f"-> Loading {num_frames} point clouds...")
    
    local_pcds = []
    fpfh_feats = []

    for t in tqdm.trange(num_frames):
        feature_file = os.path.join(features_dir, f"{sequence_ts[t]}.secondary.npz")
        pcd = FCGF.get_features(feature_file, pcd_only=True)
        fpfh = registration.compute_fpfh(pcd, config.voxel_size)
        local_pcds.append(pcd)
        fpfh_feats.append(fpfh)
        
    print("-> Registering point clouds...")
        
    local_t = [np.identity(4)]

    for t in tqdm.trange(num_frames - 1):
        ransac_reg = registration.ransac_feature_matching(local_pcds[t + 1], local_pcds[t], fpfh_feats[t + 1], fpfh_feats[t], n_ransac=4, threshold=0.05)
        icp_reg = registration.icp_refinement(local_pcds[t + 1], local_pcds[t], trans_init=ransac_reg.transformation, max_iteration=200, threshold=0.05, p2p=False)
        local_t.append(icp_reg.transformation)

    np.savez_compressed(output_path, local_t=local_t, sequence_ts=sequence_ts)
    
    
if __name__ == "__main__":
    sequence = 1
    config = Config(
        feature_dir="data/features",
        sequence_dir="data/raw_data",
        experiment="exp_10",
        trial="trial_1",
        subject="subject-1",
        sequence=f"{sequence:02d}",
        groundtruth_dir="data/trajectories/groundtruth",
        output_dir="data/trajectories/cache",
    )
    
    for trial in os.listdir(os.path.join(config.feature_dir, config.experiment)):
        config.trial = trial
        for subject in os.listdir(os.path.join(config.feature_dir, config.experiment, config.trial, str(config.voxel_size))):
            config.subject = subject    
            for sequence in os.listdir(os.path.join(config.feature_dir, config.experiment, config.trial, str(config.voxel_size), config.subject)):
                config.sequence = sequence
                print(f"Processing: {config.experiment} >> {config.trial} >> {config.subject} >> {config.sequence}")
                main(config)
    
    main(config)
    
    
