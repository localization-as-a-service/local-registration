import open3d
import numpy as np
import os


def exec_local_reg(source, target, threshold, trans_init, max_iteration=30, p2p=True):
    if p2p:
        estimation_method = open3d.registration.TransformationEstimationPointToPoint(False)
    else:
        estimation_method = open3d.registration.TransformationEstimationPointToPlane()
        
    return open3d.registration.registration_icp(
        source, target, threshold, trans_init, estimation_method,
        open3d.registration.ICPConvergenceCriteria(max_iteration=max_iteration)
    )


def exec_global_reg(source, target, source_feat, target_feat, n_ransac, threshold, p2p=True):
    if p2p:
        estimation_method = open3d.registration.TransformationEstimationPointToPoint(False)
    else:
        estimation_method = open3d.registration.TransformationEstimationPointToPlane()
        
    result = open3d.registration.registration_ransac_based_on_feature_matching(
        source, target, source_feat, target_feat, threshold,
        estimation_method, n_ransac,
        [
            open3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            open3d.registration.CorrespondenceCheckerBasedOnDistance(threshold)
        ],
        open3d.registration.RANSACConvergenceCriteria(4000000, 600))
    return result


def compute_fpff(pcd, voxel_size, down_sample=True):
    if down_sample:
        pcd = open3d.voxel_down_sample(pcd, voxel_size)

    radius_normal = voxel_size * 2
    open3d.geometry.estimate_normals(
        pcd, open3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = open3d.registration.compute_fpfh_feature(
        pcd, open3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd, pcd_fpfh


def describe(source, target, result, end="\n"):
    print(f"Keypts: [{len(source.points)}, {len(target.points)}]", end="\t")
    print(f"No of matches: {len(result.correspondence_set)}", end="\t")
    print(f"Fitness: {result.fitness}", end="\t")
    print(f"Inlier RMSE: {result.inlier_rmse:.4f}", end=end)


def register_fragments_local(sequence_dir, sequence_ts, t, voxel_size):
    src_feature_file = os.path.join(sequence_dir, f"{sequence_ts[t]}.secondary.npz")
    tgt_feature_file = os.path.join(sequence_dir, f"{sequence_ts[t + 1]}.secondary.npz")

    source = FCGF.get_features(src_feature_file, pcd_only=True)
    target = FCGF.get_features(tgt_feature_file, pcd_only=True)
    
    source, source_fpfh = compute_fpff(source, voxel_size, down_sample=False)
    target, target_fpfh = compute_fpff(target, voxel_size, down_sample=False)
    
    source.paint_uniform_color(np.random.random(3).tolist())
    
    global_reg = exec_global_reg(source, target, source_fpfh, target_fpfh, n_ransac=4, threshold=0.05)
    local_reg = exec_local_reg(source, target, threshold=0.05, trans_init=global_reg.transformation, max_iteration=30)
    
    return source, target, local_reg