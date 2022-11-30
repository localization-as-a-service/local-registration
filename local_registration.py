import open3d
import numpy as np
import tqdm
import os
import time

import utils.fread as fread
import utils.helpers as helpers
import utils.registration as registration


def main():
    exec_t = [[] for _ in range(4)] # [loading pcds, downsampling, feature extraction, ransac]
    
    sequence_dir = "../localization-data/data/point_clouds/exp_5/trial_1/subject-1/01"
    sequence_ts = fread.get_timstamps(sequence_dir, ext=".secondary.pcd")
    
    num_frames = len(sequence_ts)
    
    print("-> Loading point clouds...")
        
    local_pcds = []

    for t in tqdm.trange(num_frames):
        start_t = time.time_ns()
        pcd = helpers.read_pcd(os.path.join(sequence_dir, f"{sequence_ts[t]}.secondary.pcd"))
        end_t = time.time_ns()
        exec_t[0].append(end_t - start_t)
        
        start_t = time.time_ns()
        pcd = helpers.downsample(pcd, 0.05)
        end_t = time.time_ns()
        exec_t[1].append(end_t - start_t)
        
        local_pcds.append(pcd)
        
    print("-> Registering point clouds...")
        
    fpfh_feats = [registration.compute_fpfh(local_pcds[0], 0.05)]
    local_t = [np.identity(4)]

    for t in tqdm.trange(num_frames - 1):
        start_t = time.time_ns()
        source_fpfh = registration.compute_fpfh(local_pcds[t + 1], 0.05)
        end_t = time.time_ns()
        exec_t[2].append(end_t - start_t)
        
        start_t = time.time_ns()
        ransac_reg = registration.ransac_feature_matching(local_pcds[t + 1], local_pcds[t], source_fpfh, fpfh_feats[t], n_ransac=4, threshold=0.05)
        end_t = time.time_ns()
        exec_t[3].append(end_t - start_t)
            
        local_t.append(ransac_reg.transformation)
        fpfh_feats.append(source_fpfh)
        
    print("-> Refining trajectory transformations...")
        
    trajectory_t = [np.identity(4)]

    for t in tqdm.trange(num_frames):
        if t > 0:
            trajectory_t.append(np.dot(trajectory_t[t - 1], local_t[t]))
        local_pcds[t].transform(trajectory_t[t])
        
    print("-> Saving trajectory results...")
    
    trajectory = helpers.merge_pcds(local_pcds, 0.05)
    open3d.io.write_point_cloud("output/trajectory.pcd", trajectory)
    
    del exec_t[0][0]
    del exec_t[1][0]
    
    exec_t = np.array(exec_t)
    exec_t = np.mean(exec_t, axis=0)
    print(f"-> Execution times:")
    print(f"\t{'Loading pcds':<20}: {exec_t[0] / 1e6} ms")
    print(f"\t{'Downsampling':<20}: {exec_t[1] / 1e6} ms")
    print(f"\t{'Feature extraction':<20}: {exec_t[2] / 1e6} ms")
    print(f"\t{'RANSAC':<20}: {exec_t[3] / 1e6} ms")
    
    
if __name__ == "__main__":
    main()
