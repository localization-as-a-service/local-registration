import open3d
import numpy as np


'''
    Reads the output of the FCGF algorithm returns the keypoints and the features.
    pcd_only: if True, only the keypoints are returned
'''
def get_features(feature_file, pcd_only=False):
    data = np.load(feature_file)
    keypts = open3d.geometry.PointCloud()
    keypts.points = open3d.utility.Vector3dVector(data["keypts"])
    
    keypts.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    if pcd_only:
        return keypts
    
    scores = data["scores"]
    features = open3d.pipelines.registration.Feature()
    features.data = data["features"].T
    return keypts, features, scores