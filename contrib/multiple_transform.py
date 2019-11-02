# -*- coding: utf-8 -*-
"""
Created on Wed Jul  25 16:22:08 2018
@author: haoshu fang
"""
import os
import numpy as np
import h5py as h5
from scipy.misc import imread
import IPython
import math
from open3d import *
from tqdm import tqdm
import sys

def getTransform(calibration, viewpoint_camera, referenceCamera, transformation):
    CamFromRefKey = "H_{0}_from_{1}".format(viewpoint_camera, referenceCamera)
    print CamFromRefKey
    CamFromRef = calibration[CamFromRefKey][:]

    TableFromRefKey = "H_table_from_reference_camera"
    TableFromRef = transformation[TableFromRefKey][:]
    print TableFromRef
    sys.exit()
    return np.dot(TableFromRef, np.linalg.inv(CamFromRef))


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print "Align points cloud with mesh and store one fused point cloud per viewpoint"
        print "Usage: python multiple_transform.py DATA_DIRECTORY OBJECTS_LIST_FILE"
        sys.exit(-1)

    data_directory = sys.argv[1]

    file = open(sys.argv[2])
    objects = [line.strip() for line in file]
    viewpoint_cameras = ["NP1",  "NP2", "NP3",  "NP4",  "NP5"]      # Camera which the viewpoint will be generated.
    viewpoint_angles = [str(x) for x in range(0,360,3)]         # Relative angle of the object w.r.t the camera (angle of the turntable).

    for object_ in objects:
      referenceCamera = "NP5"
      output_directory = os.path.join(data_directory, object_, "clouds_aligned")
      if not os.path.exists(output_directory):
        os.makedirs(output_directory)
      for viewpoint_camera in tqdm(viewpoint_cameras):
        points_concat = []
        for viewpoint_angle in viewpoint_angles:
          basename = "{0}_{1}".format(viewpoint_camera, viewpoint_angle)
          cloudFilename = os.path.join(data_directory, object_, "clouds", basename + ".pcd")

          calibrationFilename = os.path.join(data_directory, object_, "calibration.h5")
          calibration = h5.File(calibrationFilename)

          transformationFilename = os.path.join(data_directory, object_, "poses", referenceCamera+"_"+viewpoint_angle+"_pose.h5")
          transformation = h5.File(transformationFilename)

          H_table_from_cam = getTransform(calibration, viewpoint_camera, referenceCamera, transformation)

          pcd = read_point_cloud(cloudFilename)
          points = np.asarray(pcd.points)
          ones = np.ones(len(points))[:,np.newaxis]

          points_ = np.concatenate((points,ones),axis=1)
          new_points_ = np.dot(H_table_from_cam, points_.T)
          new_points = new_points_.T[:,:3]

          points_concat.append(new_points)

        new_points = np.zeros((0,3))
        for p in points_concat:
            new_points = np.concatenate((new_points, p))
        ply = PointCloud()
        ply.points = Vector3dVector(new_points)
        ply.colors = Vector3dVector(pcd.colors)
        resFilename = os.path.join(output_directory, str(viewpoint_camera) + "_merged.pcd")
        write_point_cloud(resFilename, ply)
