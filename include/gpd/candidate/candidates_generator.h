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

#ifndef GRASP_CANDIDATES_GENERATOR_H
#define GRASP_CANDIDATES_GENERATOR_H

// System
#include <memory>
#include <vector>

// PCL
#include <pcl/common/common.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// Custom
#include <gpd/candidate/hand.h>
#include <gpd/candidate/hand_geometry.h>
#include <gpd/candidate/hand_search.h>
#include <gpd/candidate/hand_set.h>
#include <gpd/util/config_file.h>

namespace gpd {
namespace candidate {

/**
 *
 * \brief Generate grasp candidates.
 *
 * This class generates grasp candidates by searching for feasible robot hand
 * placements in a point cloud.
 *
 */
class CandidatesGenerator {
 public:
  /**
   * \brief Parameters for the candidates generator.
   */
  struct Parameters {
    bool remove_statistical_outliers_;  ///< if statistical outliers are removed
                                        /// from the point cloud
    bool sample_above_plane_;  ///< if samples are drawn above the support plane
    bool voxelize_;            ///< if the point cloud gets voxelized
    double voxel_size_;        ///< voxel size
    double normals_radius_;    ///< neighborhood search radius used for normal
                               ///< estimation
    int refine_normals_k_;  ///< If 0, do not refine. If > 0, this is the number
                            ///< of neighbors used for refinement.
    int num_samples_;       ///< the number of samples to be used in the search
    int num_threads_;  ///< the number of CPU threads to be used in the search
    std::vector<double> workspace_;  ///< the robot's workspace
  };

  /**
   * \brief Constructor.
   * \param params the parameters to be used for the candidate generation
   * \param hand_search_params the parameters to be used for the hand search
   */
  CandidatesGenerator(const Parameters &params,
                      const HandSearch::Parameters &hand_search_params);

  /**
   * \brief Preprocess the point cloud.
   * \param cloud_cam the point cloud
   */
  void preprocessPointCloud(util::Cloud &cloud);

  /**
   * \brief Generate grasp candidates given a point cloud.
   * \param cloud_cam the point cloud
   * \return list of grasp candidates
   */
  std::vector<std::unique_ptr<Hand>> generateGraspCandidates(
      const util::Cloud &cloud_cam);

  /**
   * \brief Generate grasp candidate sets given a point cloud.
   * \param cloud_cam the point cloud
   * \return lust of grasp candidate sets
   */
  std::vector<std::unique_ptr<HandSet>> generateGraspCandidateSets(
      const util::Cloud &cloud_cam);

  /**
   * \brief Reevaluate grasp candidates on a given point cloud.
   * \param cloud the point cloud
   * \param grasps the grasps to evaluate
   */
  std::vector<int> reevaluateHypotheses(
      const util::Cloud &cloud, std::vector<std::unique_ptr<Hand>> &grasps);

  /**
   * \brief Set the number of samples.
   * \param num_samples the number of samples
   */
  void setNumSamples(int num_samples) { params_.num_samples_ = num_samples; }

  /**
   * \brief Return the hand search parameters.
   * \return the hand search parameters
   */
  const HandSearch::Parameters &getHandSearchParams() const {
    return hand_search_->getParams();
  }

 private:
  std::unique_ptr<candidate::HandSearch> hand_search_;

  Parameters params_;
};

}  // namespace candidate
}  // namespace gpd

#endif /* GRASP_CANDIDATES_GENERATOR_H */
