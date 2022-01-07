
import open3d as o3d
try:
    # This has to come before the open3d import for some reason?
    # https://github.com/isl-org/Open3D/issues/1428
    import gpd_python as gpd
except ImportError:
    print("Couldn't find gpd_python library.")
import numpy as np

detector = gpd.GraspDetector("cfg/eigen_params.cfg")
pcd = o3d.io.read_point_cloud("tutorials/krylon.pcd")
pts = np.asarray(pcd.points)
tf = np.array([
    [0, 0, -1, 1],
    [0, 1,  0, 0],
    [1, 0,  0, 0],
    [0, 0,  0, 1]
])
pts = (tf @ (np.hstack(( pts, np.ones((len(pts), 1)) ))).T).T[:, :3]
grasps = gpd.detectGrasps(detector, pts - np.mean(pts, axis=0, keepdims=True))
print(grasps)
