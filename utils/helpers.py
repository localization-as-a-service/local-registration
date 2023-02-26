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


def merge_transformations(start_t: int, end_t: int, local_t: np.array):
    local_ts = np.identity(4)

    for t in range(start_t, end_t):
        local_ts = np.dot(local_t[t + 1], local_ts)
        
    return local_ts


def make_pcd(xyz, colors=None):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(xyz)
    if colors is not None:
        pcd.colors = open3d.utility.Vector3dVector(colors)
    return pcd


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


def nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))


def sample_timestamps(sequence_ts, frame_rates):
    sequence_ets = (sequence_ts - sequence_ts[0]) / 1e3
    frame_ids = sequence_ets * frame_rates
    frame_ids = frame_ids.astype(np.int)
    to_drop = np.where(frame_ids[1:] - frame_ids[:-1] == 0)[0]
    return np.delete(sequence_ts, to_drop)


def remove_statistical_outliers(point_cloud, nb_neighbors=20, std_ratio=1.0):
    _, ind = point_cloud.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return point_cloud.select_by_index(ind)


def inv_transform(T):
    T_inv = np.identity(4)
    T_inv[:3, :3] = T[:3, :3].T
    T_inv[:3, 3] = -np.dot(T[:3, :3].T, T[:3, 3])
    return T_inv


def rotation_matrix_to_euler_angles(R):
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2,1], R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else:
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])