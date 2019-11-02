/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2018, Andreas ten Pas
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
#include <memory>
#include <vector>

// PCL
#include <pcl/common/common.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <gpd/candidate/candidates_generator.h>
#include <gpd/candidate/hand_geometry.h>
#include <gpd/candidate/hand_set.h>
#include <gpd/clustering.h>
#include <gpd/descriptor/image_generator.h>
#include <gpd/net/classifier.h>
#include <gpd/util/config_file.h>
#include <gpd/util/plot.h>

namespace gpd {

/**
 *
 * \brief Detect grasp poses in point clouds.
 *
 * This class detects grasp poses in a point clouds by first creating a large
 * set of grasp candidates, and then classifying each of them as a grasp or not.
 *
 */
class GraspDetector {
 public:
  /**
   * \brief Constructor.
   * \param node ROS node handle
   */
  GraspDetector(const std::string &config_filename);

  /**
   * \brief Detect grasps in a point cloud.
   * \param cloud_cam the point cloud
   * \return list of grasps
   */
  std::vector<std::unique_ptr<candidate::Hand>> detectGrasps(
      const util::Cloud &cloud);

  /**
   * \brief Preprocess the point cloud.
   * \param cloud_cam the point cloud
   */
  void preprocessPointCloud(util::Cloud &cloud);

  /**
   * Filter grasps based on the robot's workspace.
   * \param hand_set_list list of grasp candidate sets
   * \param workspace the robot's workspace as a 3D cube, centered at the origin
   * \param thresh_rad the angle in radians above which grasps are filtered
   * \return list of grasps after filtering
   */
  std::vector<std::unique_ptr<candidate::HandSet>> filterGraspsWorkspace(
      std::vector<std::unique_ptr<candidate::HandSet>> &hand_set_list,
      const std::vector<double> &workspace) const;

  /**
   * Filter grasps based on their approach direction.
   * \param hand_set_list list of grasp candidate sets
   * \param direction the direction used for filtering
   * \param thresh_rad the angle in radians above which grasps are filtered
   * \return list of grasps after filtering
   */
  std::vector<std::unique_ptr<candidate::HandSet>> filterGraspsDirection(
      std::vector<std::unique_ptr<candidate::HandSet>> &hand_set_list,
      const Eigen::Vector3d &direction, const double thresh_rad);

  /**
   * \brief Generate grasp candidates.
   * \param cloud the point cloud
   * \return the list of grasp candidates
   */
  std::vector<std::unique_ptr<candidate::HandSet>> generateGraspCandidates(
      const util::Cloud &cloud);

  /**
   * \brief Create grasp images and candidates for a given point cloud.
   * \param cloud the point cloud
   * \param[out] hands_out the grasp candidates
   * \param[out] images_out the grasp images
   * \return `false` if no grasp candidates are found, `true` otherwise
   */
  bool createGraspImages(
      util::Cloud &cloud,
      std::vector<std::unique_ptr<candidate::Hand>> &hands_out,
      std::vector<std::unique_ptr<cv::Mat>> &images_out);

  /**
   * \brief Evaluate the ground truth for a given list of grasps.
   * \param cloud_gt the point cloud (typically a mesh)
   * \param hands the grasps
   * \return the ground truth label for each grasp
   */
  std::vector<int> evalGroundTruth(
      const util::Cloud &cloud_gt,
      std::vector<std::unique_ptr<candidate::Hand>> &hands);

  /**
   * \brief Creates grasp images and prunes grasps below a given score.
   * \param cloud the point cloud
   * \param hand_set_list the grasps
   * \param min_score the score below which grasps are pruned
   * \return the grasps above the score
   */
  std::vector<std::unique_ptr<candidate::Hand>> pruneGraspCandidates(
      const util::Cloud &cloud,
      const std::vector<std::unique_ptr<candidate::HandSet>> &hand_set_list,
      double min_score);

  /**
   * \brief Select the k highest scoring grasps.
   * \param hands the grasps
   * \return the k highest scoring grasps
   */
  std::vector<std::unique_ptr<candidate::Hand>> selectGrasps(
      std::vector<std::unique_ptr<candidate::Hand>> &hands) const;

  /**
   * \brief Compare the scores of two given grasps.
   * \param hand1 the first grasp to be compared
   * \param hand1 the second grasp to be compared
   * \return `true` if \param hand1 has a larger score than \param hand2,
   * `false` otherwise
   */
  static bool isScoreGreater(const std::unique_ptr<candidate::Hand> &hand1,
                             const std::unique_ptr<candidate::Hand> &hand2) {
    return hand1->getScore() > hand2->getScore();
  }

  /**
   * \brief Return the hand search parameters.
   * \return the hand search parameters
   */
  const candidate::HandSearch::Parameters &getHandSearchParameters() {
    return candidates_generator_->getHandSearchParams();
  }

  /**
   * \brief Return the image geometry parameters.
   * \return the image geometry parameters
   */
  const descriptor::ImageGeometry &getImageGeometry() const {
    return image_generator_->getImageGeometry();
  }

 private:
  void printStdVector(const std::vector<int> &v, const std::string &name) const;

  void printStdVector(const std::vector<double> &v,
                      const std::string &name) const;

  std::unique_ptr<candidate::CandidatesGenerator> candidates_generator_;
  std::unique_ptr<descriptor::ImageGenerator> image_generator_;
  std::unique_ptr<Clustering> clustering_;
  std::unique_ptr<util::Plot> plotter_;
  std::shared_ptr<net::Classifier> classifier_;

  // classification parameters
  double min_score_;           ///< minimum classifier confidence score
  bool create_image_batches_;  ///< if images are created in batches (reduces
                               /// memory usage)

  // plotting parameters
  bool plot_normals_;              ///< if normals are plotted
  bool plot_samples_;              ///< if samples/indices are plotted
  bool plot_candidates_;           ///< if grasp candidates are plotted
  bool plot_filtered_candidates_;  ///< if filtered grasp candidates are plotted
  bool plot_valid_grasps_;         ///< if valid grasps are plotted
  bool plot_clustered_grasps_;     ///< if clustered grasps are plotted
  bool plot_selected_grasps_;      ///< if selected grasps are plotted

  // filtering parameters
  bool cluster_grasps_;  ///< if grasps are clustered
  double min_aperture_;  ///< the minimum opening width of the robot hand
  double max_aperture_;  ///< the maximum opening width of the robot hand
  std::vector<double> workspace_grasps_;  ///< the workspace of the robot with
                                          /// respect to hand poses
  bool filter_approach_direction_;
  Eigen::Vector3d direction_;
  double thresh_rad_;

  // selection parameters
  int num_selected_;  ///< the number of selected grasps
};

}  // namespace gpd

#endif /* GRASP_DETECTOR_H_ */
