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


#ifndef GRASP_PLOTTER_H_
#define GRASP_PLOTTER_H_

// ROS
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

// GPG
#include <gpg/grasp.h>
#include <gpg/hand_search.h>


/** GraspPlotter class
 *
 * \brief Draw grasps in rviz.
 *
 * This class provides functions to draw grasps in rviz.
 *
 */
class GraspPlotter
{
public:

  /**
   * \brief Constructor.
   * \param node the ROS node
  */
  GraspPlotter(ros::NodeHandle& node, const HandSearch::Parameters& params);

  /**
   * \brief Visualize grasps in rviz.
   * \param hands the grasps to be visualized
   * \param frame the frame that the grasps are in
   */
  void drawGrasps(const std::vector<Grasp>& hands, const std::string& frame);

  /**
   * \brief Convert a list of grasps to a ROS message that can be published to rviz.
   * \param hands list of grasps
   * \param frame_id the name of the frame that the grasp is in
   */
  visualization_msgs::MarkerArray convertToVisualGraspMsg(const std::vector<Grasp>& hands,
    const std::string& frame_id);

  /**
   * \brief Convert a list of grasps to a ROS message that can be published to rviz.
   * \param hands list of grasps
   * \param center the center of the finger
   * \param rot the orientation of the hand
   * \param length the length of the finger
   * \param width the width of the finger
   * \param height the height of the finger
   * \param frame_id the name of the frame that the grasp is in
   */
  visualization_msgs::Marker createFingerMarker(const Eigen::Vector3d& center, const Eigen::Matrix3d& rot,
    const Eigen::Vector3d& lwh, int id, const std::string& frame_id);

  /**
   * \brief Convert a list of grasps to a ROS message that can be published to rviz.
   * \param hands list of grasps
   * \param start
   * \param frame_id the name of the frame that the grasp is in
   */
  visualization_msgs::Marker createHandBaseMarker(const Eigen::Vector3d& start, const Eigen::Vector3d& end,
    const Eigen::Matrix3d& frame, double length, double height, int id, const std::string& frame_id);


private:

  ros::Publisher rviz_pub_; ///< ROS publisher for grasps in rviz (visualization)

  double outer_diameter_;
  double hand_depth_;
  double finger_width_;
  double hand_height_;
};

#endif /* GRASP_PLOTTER_H_ */
