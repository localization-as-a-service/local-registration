import open3d
import numpy as np
import os


def downsample(pcd, voxel_size):
    return pcd.voxel_down_sample(voxel_size)


def read_pcd(filename):
    return open3d.io.read_point_cloud(filename)


def merge_pcds(pcds, voxel_size):
    global_pcd = open3d.geometry.PointCloud()

    for local_pcd in pcds:
        global_pcd += local_pcd
    
    return global_pcd.voxel_down_sample(voxel_size)