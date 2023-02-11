import open3d
import copy
from scipy.signal import argrelmin
import numpy as np


def describe(source, target, result, end="\n"):
    print(f"Keypts: [{len(source.points)}, {len(target.points)}]", end="\t")
    print(f"No of matches: {len(result.correspondence_set)}", end="\t")
    print(f"Fitness: {result.fitness}", end="\t")
    print(f"Inlier RMSE: {result.inlier_rmse:.4f}", end=end)


def compute_fpfh(pcd, voxel_size):
    radius_normal = voxel_size * 2
    radius_feature = voxel_size * 5
    pcd.estimate_normals(open3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    pcd_fpfh = open3d.pipelines.registration.compute_fpfh_feature(pcd, open3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_fpfh


def ransac_feature_matching(source, target, source_feat, target_feat, n_ransac, threshold, p2p=True):
    if p2p:
        estimation_method = open3d.pipelines.registration.TransformationEstimationPointToPoint(False)
    else:   
        estimation_method = open3d.pipelines.registration.TransformationEstimationPointToPlane()
    
    return open3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source, target, source_feat, target_feat, True,
        threshold, estimation_method, n_ransac,
        [
            open3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            open3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(threshold)
        ],
        open3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    

def icp_refinement(source, target, threshold, trans_init, max_iteration=30, p2p=True):
    if p2p:
        estimation_method = open3d.pipelines.registration.TransformationEstimationPointToPoint(False)
    else:
        estimation_method = open3d.pipelines.registration.TransformationEstimationPointToPlane()
        
    return open3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init, estimation_method,
        open3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration)
    )
    
    
def view(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    open3d.visualization.draw_geometries([source_temp, target_temp])
    
    
def find_cutoffs(std_values, target_fps, min_std, threshold=0.5):
    cutoffs = argrelmin(std_values, order=target_fps // 2)[0]
    return cutoffs[np.where(np.abs(std_values[cutoffs] - min_std) < threshold)[0]]