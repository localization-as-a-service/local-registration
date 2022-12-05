import json
import open3d
import numpy as np

from PIL import Image

class DepthCamera:
    def __init__(self, name, metadata) -> None:
        self.name = name
        with open(metadata, "r") as f:
            metadata = json.load(f)
            self.depth_scale = metadata["depth_scale"]
            self.width = metadata["width"]
            self.height = metadata["height"]
            self.fx = metadata["fx"]
            self.fy = metadata["fy"]
            self.px = metadata["px"]
            self.py = metadata["py"]
            
        self.intrinsics = open3d.camera.PinholeCameraIntrinsic(self.width, self.height, self.fx, self.fy, self.px, self.py)
        
    @staticmethod
    def save_intrinsics(intrinsics, width, height, depth_scale, filename):
        camera_properties = {
            "depth_scale": np.round(1 / depth_scale),
            "width": width,
            "height": height,
            "fx": intrinsics.fx,
            "fy": intrinsics.fy,
            "px": intrinsics.ppx,
            "py": intrinsics.ppy
        }

        json.dump(camera_properties, open(filename, "w"))
            
    def read_depth_image(self, depth_file):
        depth_image = Image.open(depth_file).convert("I")
        return depth_image
    
    def depth_to_point_cloud(self, depth_file):
        fx, fy = self.intrinsics.get_focal_length()
        cx, cy = self.intrinsics.get_principal_point()
        
        depth_image = self.read_depth_image(depth_file)
        
        points = []
        for v in range(0, depth_image.height):
            for u in range(0, depth_image.width):
                z = depth_image.getpixel((u,v)) / self.depth_scale
                if z == 0: 
                    continue
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                points.append([x, y, z])
                
        xpcd = open3d.geometry.PointCloud()
        xpcd.points = open3d.utility.Vector3dVector(points)
        
        return xpcd