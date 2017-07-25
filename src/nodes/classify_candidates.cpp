/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2016, Andreas ten Pas
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */


// ROS
#include <ros/ros.h>

// Grasp Candidates Generator
#include <gpg/cloud_camera.h>

// Custom
#include "../../include/gpd/grasp_detector.h"
#include "../../include/gpd/sequential_importance_sampling.h"


int main(int argc, char* argv[]) 
{
  // initialize ROS
  ros::init(argc, argv, "classify_grasp_candidates");
  ros::NodeHandle node("~");
  
  // Set the position from which the camera sees the point cloud.
  Eigen::Matrix3Xd view_points(3,1);
  view_points << 0.0, 0.0, 0.0;
  std::vector<double> camera_position;
  node.getParam("camera_position", camera_position);
  view_points << camera_position[0], camera_position[1], camera_position[2];

  // Load point cloud from file.
  std::string filename;
  node.param("cloud_file_name", filename, std::string(""));
  CloudCamera cloud_cam(filename, view_points);
  if (cloud_cam.getCloudOriginal()->size() == 0)
  {
    std::cout << "Input point cloud is empty or does not exist!\n";
    return (-1);
  }

  // Detect grasp poses.
  bool use_importance_sampling;
  node.param("use_importance_sampling", use_importance_sampling, false);

  if (use_importance_sampling)
  {
    SequentialImportanceSampling detector(node);

    // Preprocess the point cloud (voxelize, workspace, etc.).
    detector.preprocessPointCloud(cloud_cam);

    // Detect grasps in the point cloud.
    std::vector<Grasp> grasps = detector.detectGrasps(cloud_cam);
  }
  else
  {
    GraspDetector detector(node);

    // Preprocess the point cloud (voxelize, workspace, etc.).
    detector.preprocessPointCloud(cloud_cam);

    // Detect grasps in the point cloud.
    std::vector<Grasp> grasps = detector.detectGrasps(cloud_cam);
  }

  return 0;
}
