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


def rotate_transformation_matrix(t, rx, ry, rz):
    # Convert degrees to radians
    rx, ry, rz = np.radians(rx), np.radians(ry), np.radians(rz)

    RX = np.array([
        [1, 0, 0, 0],
        [0, np.cos(rx), -np.sin(rx), 0],
        [0, np.sin(rx), np.cos(rx), 0],
        [0, 0, 0, 1]
    ])

    RY = np.array([
        [np.cos(ry), 0, np.sin(ry), 0],
        [0, 1, 0, 0],
        [-np.sin(ry), 0, np.cos(ry), 0],
        [0, 0, 0, 1]
    ])

    RZ = np.array([
        [np.cos(rz), -np.sin(rz), 0, 0],
        [np.sin(rz), np.cos(rz), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    return np.dot(np.dot(np.dot(t, RZ), RY), RX)