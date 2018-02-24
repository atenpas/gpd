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

#ifndef GRASP_DETECTOR_H_
#define GRASP_DETECTOR_H_


// System
#include <algorithm>
#include <vector>

// PCL
#include <pcl/common/common.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// ROS
#include <ros/ros.h>

// Grasp Candidates Generator
#include <gpg/cloud_camera.h>
#include <gpg/candidates_generator.h>
#include <gpg/grasp.h>
#include <gpg/grasp_set.h>
#include <gpg/plot.h>

// Custom
#include <gpd/clustering.h>
#include <gpd/learning.h>
#include <gpd/lenet.h>


/** GraspDetector class
 *
 * \brief Detect grasp poses in point clouds.
 *
 * This class detects grasps in a point cloud by first creating a large set of grasp hypotheses, and then
 * classifying each of them as a grasp or not. It also contains a function to preprocess the point cloud.
 *
*/
class GraspDetector
{
public:

  /**
   * \brief Constructor.
   * \param node ROS node handle
   */
  GraspDetector(ros::NodeHandle& node);

  /**
   * \brief Destructor.
   */
  ~GraspDetector()
  {
    delete candidates_generator_;
    delete learning_;
    delete clustering_;
    delete classifier_;
  }
  
  /**
   * \brief Preprocess the point cloud.
   * \param cloud_cam the point cloud
   */
  void preprocessPointCloud(CloudCamera& cloud_cam);

  /**
   * \brief Detect grasps in a given point cloud.
   * \param cloud_cam the point cloud
   * \return the list of detected grasps
   */
  std::vector<Grasp> detectGrasps(const CloudCamera& cloud_cam);

  /**
   * \brief Generate grasp candidates.
   * \param cloud_cam the point cloud
   * \return the list of grasp candidates
   */
  std::vector<GraspSet> generateGraspCandidates(const CloudCamera& cloud_cam);

  /**
   * \brief Filter grasps that are outside of the robot's workspace.
   * \param hand_set_list the list of grasp candidate sets
   * \param workspace the robot's workspace
   * \return the grasps that are inside the robot's workspace
   */
  std::vector<GraspSet> filterGraspsWorkspace(const std::vector<GraspSet>& hand_set_list,
    const std::vector<double>& workspace);
  
  /**
   * \brief Filter side grasps that are close to the table.
   * \param hand_set_list list of grasp candidate sets
   */
  std::vector<GraspSet> filterSideGraspsCloseToTable(const std::vector<GraspSet>& hand_set_list);

  /**
   * \brief Filter grasps that are half-antipodal.
   * \param hand_set_list the list of grasp candidate sets
   * \return the grasps that are not half-antipodal
   */
  std::vector<GraspSet> filterHalfAntipodal(const std::vector<GraspSet>& hand_set_list);

  /**
   * \brief Extract the valid grasps from a given list of grasp candidate sets.
   * \param hand_set_list the list of grasp candidate sets
   * \return the valid grasps
   */
  std::vector<Grasp> extractHypotheses(const std::vector<GraspSet>& hand_set_list);

  /**
   * \brief Match grasps with their corresponding grasp images.
   * \param[in] hand_set_list list of grasp candidate sets
   * \param[in] images list of grasp images
   * \param[out] grasps_out the grasps corresponding to the images
   * \param[out] images_out the images corresponding to the grasps
   */
  void extractGraspsAndImages(const std::vector<GraspSet>& hand_set_list, const std::vector<cv::Mat>& images,
    std::vector<Grasp>& grasps_out, std::vector<cv::Mat>& images_out);

  /**
   * \brief Classify grasp candidates as viable grasps or not.
   * \param cloud_cam the point cloud
   * \param candidates the grasp candidates to be classified
   * \return the classified grasps
   */
  std::vector<Grasp> classifyGraspCandidates(const CloudCamera& cloud_cam, std::vector<GraspSet>& candidates);

  /**
   * \brief Find clusters of grasps that are geometrically aligned.
   * \param grasps the grasps for which to search clusters
   * \return the grasps that are in clusters
   */
  std::vector<Grasp> findClusters(const std::vector<Grasp>& grasps);

  /**
   * \brief Compare if the score of a grasp is larger than the score of another grasp.
   * \param hypothesis1 a grasp
   * \param hypothesis2 another grasp
   * \return true if it is larger, false otherwise
   */
  static bool isScoreGreater(const Grasp& hypothesis1, const Grasp& hypothesis2)
  {
    return hypothesis1.getScore() > hypothesis2.getScore();
  }

  const HandSearch::Parameters& getHandSearchParameters()
  {
    return candidates_generator_->getHandSearchParams();
  }


private:

  CandidatesGenerator* candidates_generator_; ///< pointer to object for grasp candidate generation
  Learning* learning_; ///< pointer to object for grasp image creation
  Clustering* clustering_; ///< pointer to object for clustering geometrically aligned grasps
  Lenet* classifier_; ///< pointer to object for classification of candidates

  Learning::ImageParameters image_params_; // grasp image parameters

  // classification parameters
  double min_score_diff_; ///< minimum classifier confidence score
  bool create_image_batches_; ///< if images are created in batches (reduces memory usage)

  // plotting parameters
  bool plot_normals_; ///< if normals are plotted
  bool plot_samples_; ///< if samples/indices are plotted
  bool plot_filtered_grasps_; ///< if filtered grasps are plotted
  bool plot_valid_grasps_; ///< if positive grasp instances are plotted
  bool plot_clusters_; ///< if grasp clusters are plotted
  bool plot_selected_grasps_; ///< if selected grasps are plotted

  // filtering parameters
  bool filter_grasps_; ///< if grasps are filtered based on the robot's workspace and the robot hand width
  bool filter_table_side_grasps_; ///< if side grasps close to the table are filtered
  bool filter_half_antipodal_; ///< if grasps are filtered based on being half-antipodal
  bool cluster_grasps_; ///< if grasps are clustered
  double outer_diameter_; ///< the outer diameter of the robot hand
  double min_aperture_; ///< the minimum opening width of the robot hand
  double max_aperture_; ///< the maximum opening width of the robot hand
  std::vector<double> workspace_; ///< the workspace of the robot
  std::vector<double> vert_axis_; ///< vertical axis used for filtering side grasps that are close to the table
  double table_height_; ///< height of table (along vertical axis)
  double table_thresh_; ///< distance threshold below which side grasps are considered to be too close to the table
  double angle_thresh_; ///< angle threshold below which grasps are considered to be side grasps

  // selection parameters
  int num_selected_; ///< the number of selected grasps
};

#endif /* GRASP_DETECTOR_H_ */
