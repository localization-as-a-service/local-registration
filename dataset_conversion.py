import os
import shutil
import open3d
import numpy as np
import tqdm

import utils.registration as registration
import utils.helpers as helpers
import utils.fread as fread
from utils.depth_camera import DepthCamera


def get_timstamps(dir_path):
    seq_ts = [int(f.split(".")[0].split("-")[1]) for f in os.listdir(dir_path) if f.endswith(".depth.png")]
    return np.array(sorted(seq_ts))


def nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))


def convert_to_point_clouds(dataset_dir, subject_id=1, device_id=3):
    """
    Go through the dataset structure and convert all the depth images to point clouds

    Args:
        dataset_dir (str): the directory contains the raw captures
    """
    
    out_dir = dataset_dir.replace("raw_data", "point_clouds")

    if not os.path.exists(out_dir): os.makedirs(out_dir)

    # Load pre-computed camera extrinsic parameters
    pose_device_0 = np.loadtxt(os.path.join(dataset_dir, "global/transformations/device-0.txt"))
    pose_device_1 = np.loadtxt(os.path.join(dataset_dir, "global/transformations/device-1.txt"))
    pose_device_2 = np.loadtxt(os.path.join(dataset_dir, "global/transformations/device-2.txt"))

    # Load pre-computed camera intrinsic parameters
    # Global cameras
    device_0 = DepthCamera("device-0", os.path.join(dataset_dir, "../metadata/device-0-aligned.json"))
    device_1 = DepthCamera("device-1", os.path.join(dataset_dir, "../metadata/device-1-aligned.json"))
    device_2 = DepthCamera("device-2", os.path.join(dataset_dir, "../metadata/device-2-aligned.json"))
    # Secondary camera
    device_3 = DepthCamera("device-3", os.path.join(dataset_dir, f"../metadata/device-{device_id}-aligned.json"))

    # Iterate through the secondary directory
    subject = f"subject-{subject_id}"
    for seq_id in os.listdir(os.path.join(dataset_dir, "secondary", subject)):
        
        seq_dir = os.path.join(dataset_dir, "secondary", subject, seq_id, "frames")
        seq_out_dir = os.path.join(out_dir, subject, seq_id)

        if not os.path.exists(seq_out_dir): os.makedirs(seq_out_dir)

        seq_ts = get_timstamps(seq_dir)
        seq_ts = helpers.sample_timestamps(seq_ts, 20)

        device_0_ts = get_timstamps(os.path.join(dataset_dir, "global", "device-0"))
        device_1_ts = get_timstamps(os.path.join(dataset_dir, "global", "device-1"))
        device_2_ts = get_timstamps(os.path.join(dataset_dir, "global", "device-2"))
        
        for seq_t in tqdm.tqdm(seq_ts):
            if os.path.exists(os.path.join(seq_out_dir, f"{seq_t}.global.pcd")) and os.path.exists(os.path.join(seq_out_dir, f"{seq_t}.secondary.pcd")):
                continue
            
            pcd_g0 = device_0.depth_to_point_cloud(os.path.join(dataset_dir, "global", "device-0", f"frame-{nearest(device_0_ts, seq_t)}.depth.png"))
            pcd_g1 = device_1.depth_to_point_cloud(os.path.join(dataset_dir, "global", "device-1", f"frame-{nearest(device_1_ts, seq_t)}.depth.png"))
            pcd_g2 = device_2.depth_to_point_cloud(os.path.join(dataset_dir, "global", "device-2", f"frame-{nearest(device_2_ts, seq_t)}.depth.png"))
            
            pcd_g0.transform(pose_device_0)
            pcd_g1.transform(pose_device_1)
            pcd_g2.transform(pose_device_2)
            
            global_pcd = pcd_g0 + pcd_g1 + pcd_g2
            
            secondary_pcd = device_3.depth_to_point_cloud(os.path.join(seq_dir, f"frame-{seq_t}.depth.png"))
            
            open3d.io.write_point_cloud(os.path.join(seq_out_dir, f"{seq_t}.global.pcd"), global_pcd)
            open3d.io.write_point_cloud(os.path.join(seq_out_dir, f"{seq_t}.secondary.pcd"), secondary_pcd)


def convert_to_point_clouds_only_secondary(dataset_dir, subject_id=1, device_id=0):
    """
    Convert only secondary sequences

    Args:
        dataset_dir (str): the directory contains the raw captures
    """
    
    out_dir = dataset_dir.replace("raw_data", "point_clouds")
    
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    
    device = DepthCamera("device", os.path.join(dataset_dir, f"metadata/device-{device_id}-aligned.json"))
    
    for trial in os.listdir(dataset_dir):
        # check the directory name starts with trial
        if not trial.startswith("trial"):
            continue
        
        subject = f"subject-{subject_id}"
        
        for seq_id in os.listdir(os.path.join(dataset_dir, trial, subject)):
            
            seq_dir = os.path.join(dataset_dir, trial, subject, seq_id, "frames")
            seq_out_dir = os.path.join(out_dir, trial, subject, seq_id)

            if not os.path.exists(seq_out_dir): os.makedirs(seq_out_dir)

            seq_ts = get_timstamps(seq_dir)
            
            for seq_t in tqdm.tqdm(seq_ts):
                pcd = device.depth_to_point_cloud(os.path.join(seq_dir, f"frame-{seq_t}.depth.png"))
                open3d.io.write_point_cloud(os.path.join(seq_out_dir, f"{seq_t}.secondary.pcd"), pcd)
                    
            
            
def create_rotating_sequences(dataset_dir, experiment, trial, subject_id=1, device_id=3):
    device = DepthCamera("device-0", f"{dataset_dir}/{experiment}/metadata/device-{device_id}-aligned.json")
    
    out_dir = os.path.join("data/point_clouds", experiment, trial)
    # Iterate through the secondary directory
    for seq_id in os.listdir(os.path.join(dataset_dir, experiment, trial, "secondary", f"subject-{subject_id}")):
        
        seq_dir = os.path.join(dataset_dir, experiment, trial, "secondary", f"subject-{subject_id}", seq_id, "frames")
        seq_out_dir = os.path.join(out_dir, f"subject-{subject_id}", seq_id)

        if not os.path.exists(seq_out_dir): os.makedirs(seq_out_dir)

        seq_ts = get_timstamps(seq_dir)
        
        global_seq_ts = fread.get_timstamps(os.path.join(out_dir, "global"), ext=".pcd")
        
        for seq_t in tqdm.tqdm(seq_ts):
            secondary_pcd = device.depth_to_point_cloud(os.path.join(seq_dir, f"frame-{seq_t}.depth.png"))
            global_pcd = os.path.join(out_dir, "global", f"{nearest(global_seq_ts, seq_t)}.pcd")
            
            shutil.copy(global_pcd, os.path.join(seq_out_dir, f"{seq_t}.global.pcd"))
            open3d.io.write_point_cloud(os.path.join(seq_out_dir, f"{seq_t}.secondary.pcd"), secondary_pcd)
            
            
def make_rotating_global_pcds(dataset_dir, experiment, trial, device_id=0):
    global_transformation = np.loadtxt("temp/exp_2_trial_1.txt")
    device = DepthCamera("device-0", f"{dataset_dir}/{experiment}/metadata/device-{device_id}.json")
    voxel_size = 0.03
    
    sequence_dir = f"{dataset_dir}/{experiment}/{trial}/global/device-{device_id}"
    sequence_ts = get_timstamps(sequence_dir)
    
    trans_init = None
    
    sequence_t = [np.identity(4)]
    sequence_pcds = []

    for t in tqdm.trange(7):
        src_pcd_file = os.path.join(sequence_dir, f"frame-{sequence_ts[t + 1]}.depth.png")
        tgt_pcd_file = os.path.join(sequence_dir, f"frame-{sequence_ts[t]}.depth.png")

        source = device.depth_to_point_cloud(src_pcd_file)
        target = device.depth_to_point_cloud(tgt_pcd_file)
        
        source = helpers.preprocess(source, voxel_size)
        target = helpers.preprocess(target, voxel_size)
        
        source, source_fpfh = registration.compute_fpff(source, voxel_size, down_sample=False)
        target, target_fpfh = registration.compute_fpff(target, voxel_size, down_sample=False)
        
        if trans_init is None:
            ransac_reg = registration.exec_global_reg(source, target, source_fpfh, target_fpfh, n_ransac=4, threshold=0.05)
            icp_reg = registration.exec_local_reg(source, target, threshold=0.03, trans_init=ransac_reg.transformation, max_iteration=200, p2p=False)
            trans_init = icp_reg.transformation
        else:
            icp_reg = registration.exec_local_reg(source, target, threshold=0.03, trans_init=trans_init, max_iteration=200, p2p=False)
        
        source.paint_uniform_color(helpers.random_color())
        target.paint_uniform_color(helpers.random_color())
        
        transformation = np.dot(sequence_t[t], icp_reg.transformation)
        sequence_t.append(transformation)
        
        source.transform(transformation)
        target.transform(sequence_t[t])
        if t == 0:
            sequence_pcds.append(target)
        
        sequence_pcds.append(source)
    
    rotation_pcd = helpers.merge_pcds(sequence_pcds, voxel_size)
    # open3d.visualization.draw_geometries([rotation_pcd])
    
    out_dir = os.path.join("data/point_clouds", experiment, trial, "global")
    
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    
    for t in tqdm.trange(8, len(sequence_ts)):
        src_pcd_file = os.path.join(sequence_dir, f"frame-{sequence_ts[t]}.depth.png")
        
        source = device.depth_to_point_cloud(src_pcd_file)
        target = sequence_pcds[t % 8]
        
        icp_reg = registration.exec_local_reg(source, target, threshold=0.03, trans_init=sequence_t[t % 8], max_iteration=50)
        
        target = helpers.merge_pcds(np.delete(sequence_pcds, t % 8), voxel_size)
        source.transform(icp_reg.transformation)
        
        sequence_pcds[t % 8] = source
        
        current_pcd = helpers.merge_pcds(sequence_pcds, voxel_size)
        current_pcd.paint_uniform_color(helpers.random_color())
        current_pcd.transform(global_transformation)
        
        open3d.io.write_point_cloud(os.path.join(out_dir, f"{sequence_ts[t]}.pcd"), current_pcd)
    
        
if __name__ == "__main__":
    # convert_to_point_clouds("data/raw_data/exp_5/trial_1", subject_id=1, device_id=3)
    # convert_to_point_clouds("data/raw_data/exp_5/trial_2", subject_id=1, device_id=3)
    # convert_to_point_clouds("data/raw_data/exp_5/trial_3", subject_id=1, device_id=3)
    # convert_to_point_clouds("data/raw_data/exp_5/trial_3", subject_id=2, device_id=4)
    # convert_to_point_clouds("data/raw_data/exp_5/trial_4", subject_id=1, device_id=3)
    convert_to_point_clouds_only_secondary("data/raw_data/exp_7", subject_id=1, device_id=0)
    # make_rotating_global_pcds("data/raw_data", "exp_2", "trial_1")
    # create_rotating_sequences("data/raw_data", "exp_2", "trial_1", subject_id=1, device_id=3)
    
    