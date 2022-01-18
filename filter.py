try:
    # This has to come before the open3d import for some reason?
    # https://github.com/isl-org/Open3D/issues/1428
    import gpd_python as gpd
except ImportError:
    print("Couldn't find gpd_python library.")
import numpy as np
import time
import open3d as o3d
import klampt
from klampt.io import numpy_convert
from klampt import vis

detector = gpd.GraspDetector("cfg/eigen_params.cfg")
pcd = o3d.io.read_point_cloud("orig_full.pcd")
cl, _ = pcd.remove_statistical_outlier(nb_neighbors=30,
                                                std_ratio=1.0)
print(len(pcd.points), len(cl.points))
o3d.visualization.draw_geometries([pcd])
o3d.visualization.draw_geometries([cl])
# pcd = cl
# pts = np.asarray(pcd.points)
# pts = pts[np.logical_not(np.isnan(pts[:, 0])), :]
# # tf = np.array([
# #     [0, 0, -1, 1],
# #     [0, 1,  0, 0],
# #     [1, 0,  0, 0],
# #     [0, 0,  0, 1]
# # ])
# tf = np.eye(4)
# pts = (tf @ (np.hstack(( pts, np.ones((len(pts), 1)) ))).T).T[:, :3]
# offset = np.mean(pts, axis=0, keepdims=True)
# grasps = gpd.detectGrasps(detector, pts - offset, np.zeros(3))
# print(grasps)
