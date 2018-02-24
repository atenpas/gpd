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


#ifndef GRASP_DETECTION_SERVER_H_
#define GRASP_DETECTION_SERVER_H_


// ROS
#include <eigen_conversions/eigen_msg.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

// GPG
#include <gpg/cloud_camera.h>

// this project (services)
#include <gpd/detect_grasps.h>

// this project (headers)
#include "../gpd/grasp_detector.h"
#include "../gpd/grasp_plotter.h"


class GraspDetectionServer
{
public:

  /**
   * \brief Constructor.
   * \param node the ROS node
  */
  GraspDetectionServer(ros::NodeHandle& node);

  /**
   * \brief Destructor.
  */
  ~GraspDetectionServer()
  {
    delete cloud_camera_;
    delete grasp_detector_;
    delete rviz_plotter_;
  }

  /**
   * \brief Service callback for detecting grasps.
   * \param req the service request
   * \param res the service response
   */
  bool detectGrasps(gpd::detect_grasps::Request& req, gpd::detect_grasps::Response& res);


private:

  /**
   * \brief Create a ROS message that contains a list of grasp poses from a list of handles.
   * \param hands the list of grasps
   * \return the ROS message that contains the grasp poses
  */
  gpd::GraspConfigList createGraspListMsg(const std::vector<Grasp>& hands);

  gpd::GraspConfig convertToGraspMsg(const Grasp& hand);

  ros::Publisher grasps_pub_; ///< ROS publisher for grasp list messages

  std_msgs::Header cloud_camera_header_; ///< stores header of the point cloud
  std::string frame_; ///< point cloud frame

  GraspDetector* grasp_detector_; ///< used to run the grasp pose detection
  CloudCamera* cloud_camera_; ///< stores point cloud with (optional) camera information and surface normals
  GraspPlotter* rviz_plotter_; ///< used to plot detected grasps in rviz

  bool use_rviz_; ///< if rviz is used for visualization instead of PCL
  std::vector<double> workspace_; ///< workspace limits
  Eigen::Vector3d view_point_; ///< (input) view point of the camera onto the point cloud
};

#endif /* GRASP_DETECTION_SERVER_H_ */
