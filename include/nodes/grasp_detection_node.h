/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2015, Andreas ten Pas
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

#ifndef GRASP_DETECTION_NODE_H_
#define GRASP_DETECTION_NODE_H_

// system
#include <algorithm>
#include <vector>

// ROS
#include <eigen_conversions/eigen_msg.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

// PCL
#include <pcl/common/common.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// GPG
#include <gpg/cloud_camera.h>

// this project (messages)
#include <gpd/CloudIndexed.h>
#include <gpd/CloudSamples.h>
#include <gpd/CloudSources.h>
#include <gpd/GraspConfig.h>
#include <gpd/GraspConfigList.h>
#include <gpd/SamplesMsg.h>

// this project (headers)
#include "../gpd/grasp_detector.h"
#include "../gpd/grasp_plotter.h"
#include "../gpd/sequential_importance_sampling.h"


typedef pcl::PointCloud<pcl::PointXYZRGBA> PointCloudRGBA;
typedef pcl::PointCloud<pcl::PointNormal> PointCloudPointNormal;


/** GraspDetectionNode class
 *
 * \brief A ROS node that can detect grasp poses in a point cloud.
 *
 * This class is a ROS node that handles all the ROS topics.
 *
*/
class GraspDetectionNode
{
public:
  
  /**
   * \brief Constructor.
   * \param node the ROS node
  */
  GraspDetectionNode(ros::NodeHandle& node);

  /**
   * \brief Destructor.
  */
  ~GraspDetectionNode()
  {
    delete cloud_camera_;
    delete importance_sampling_;
    delete grasp_detector_;
    delete rviz_plotter_;
  }

  /**
   * \brief Run the ROS node. Loops while waiting for incoming ROS messages.
  */
  void run();

  /**
   * \brief Detect grasp poses in a point cloud received from a ROS topic.
   * \return the list of grasp poses
  */
  std::vector<Grasp> detectGraspPosesInTopic();


private:

  /**
   * \brief Find the indices of the points within a ball around a given point in the cloud.
   * \param cloud the point cloud
   * \param centroid the centroid of the ball
   * \param radius the radius of the ball
   * \return the indices of the points in the point cloud that lie within the ball
  */
  std::vector<int> getSamplesInBall(const PointCloudRGBA::Ptr& cloud, const pcl::PointXYZRGBA& centroid, float radius);

  /**
   * \brief Callback function for the ROS topic that contains the input point cloud.
   * \param msg the incoming ROS message
  */
  void cloud_callback(const sensor_msgs::PointCloud2& msg);

  /**
   * \brief Callback function for the ROS topic that contains the input point cloud and a list of indices.
   * \param msg the incoming ROS message
  */
  void cloud_indexed_callback(const gpd::CloudIndexed& msg);
  
  /**
   * \brief Callback function for the ROS topic that contains the input point cloud and a list of (x,y,z) samples.
   * \param msg the incoming ROS message
  */
  void cloud_samples_callback(const gpd::CloudSamples& msg);

  /**
   * \brief Initialize the <cloud_camera> object given a <cloud_sources> message.
   * \param msg the <cloud_sources> message
   */
  void initCloudCamera(const gpd::CloudSources& msg);

  /**
   * \brief Callback function for the ROS topic that contains the input samples.
   * \param msg the incoming ROS message
  */
  void samples_callback(const gpd::SamplesMsg& msg);

  /**
   * \brief Create a ROS message that contains a list of grasp poses from a list of handles.
   * \param hands the list of grasps
   * \return the ROS message that contains the grasp poses
  */
  gpd::GraspConfigList createGraspListMsg(const std::vector<Grasp>& hands);

  gpd::GraspConfig convertToGraspMsg(const Grasp& hand);

  Eigen::Matrix3Xd fillMatrixFromFile(const std::string& filename, int num_normals);

  Eigen::Vector3d view_point_; ///< (input) view point of the camera onto the point cloud

  CloudCamera* cloud_camera_; ///< stores point cloud with (optional) camera information and surface normals
  std_msgs::Header cloud_camera_header_; ///< stores header of the point cloud

  int size_left_cloud_; ///< (input) size of the left point cloud (when using two point clouds as input)
  bool has_cloud_, has_normals_, has_samples_; ///< status variables for received (input) messages
  std::string frame_; ///< point cloud frame
  ros::Subscriber cloud_sub_; ///< ROS subscriber for point cloud messages
  ros::Subscriber samples_sub_; ///< ROS subscriber for samples messages
  ros::Publisher grasps_pub_; ///< ROS publisher for grasp list messages
  ros::Publisher grasps_rviz_pub_; ///< ROS publisher for grasps in rviz (visualization)

  bool use_importance_sampling_; ///< if importance sampling is used
  bool use_rviz_; ///< if rviz is used for visualization instead of PCL
  std::vector<double> workspace_; ///< workspace limits

  GraspDetector* grasp_detector_; ///< used to run the GPD algorithm
  SequentialImportanceSampling* importance_sampling_; ///< sequential importance sampling variation of GPD algorithm
  GraspPlotter* rviz_plotter_; ///< used to plot detected grasps in rviz

  /** constants for input point cloud types */
  static const int POINT_CLOUD_2; ///< sensor_msgs/PointCloud2
  static const int CLOUD_INDEXED; ///< gpd/CloudIndexed
  static const int CLOUD_SAMPLES; ///< gpd/CloudSamples
};

#endif /* GRASP_DETECTION_NODE_H_ */
