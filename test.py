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
pcd = o3d.io.read_point_cloud("tutorials/krylon.pcd")
pts = np.asarray(pcd.points)
pts = pts[np.logical_not(np.isnan(pts[:, 0])), :]
# tf = np.array([
#     [0, 0, -1, 1],
#     [0, 1,  0, 0],
#     [1, 0,  0, 0],
#     [0, 0,  0, 1]
# ])
tf = np.eye(4)
pts = (tf @ (np.hstack(( pts, np.ones((len(pts), 1)) ))).T).T[:, :3]
offset = np.mean(pts, axis=0, keepdims=True)
grasps = gpd.detectGrasps(detector, pts - offset, np.ones((1, len(pts))), np.zeros((3, 1)))
print(grasps)
k_pcd = klampt.PointCloud()
for i in range(len(pts)):
    k_pcd.addPoint(pts[i, :])
world = klampt.WorldModel()
vis.add("pcd", k_pcd)
for i in range(len(grasps)):
    g: np.ndarray = grasps[i, :]
    pos = g[:3] + offset   # Undo mean subtraction
    rot = g[3:12].reshape(3, 3)
    # # Flip GPD convention to our convention
    # rot = rot @ np.array([
    #     [-1,  0, 0],
    #     [ 0, -1, 0],
    #     [ 0,  0, 1]
    # ])
    k_tf = (rot.T.flatten().tolist(), pos.reshape(-1).tolist())
    vis.add("grasp_{}".format(i), k_tf)
vis.show()
while vis.shown():
    time.sleep(0.05)
