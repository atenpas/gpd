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

#ifndef SEQUENTIAL_IMPORTANCE_SAMPLING_H
#define SEQUENTIAL_IMPORTANCE_SAMPLING_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <memory>
#include <set>

#include <gpd/grasp_detector.h>
#include <gpd/util/plot.h>

namespace gpd {

typedef pcl::PointCloud<pcl::PointXYZRGBA> PointCloudRGB;

/**
 *
 * \brief Grasp pose detection with the Cross Entropy Method.
 *
 * This class uses the Cross Entropy Method to focus the grasp candidate
 * generation on areas of the point cloud where more grasp candidates are
 * expected to be found.
 *
 */
class SequentialImportanceSampling {
 public:
  /**
   * \brief Constructor.
   * \param node ROS node handle
   */
  SequentialImportanceSampling(const std::string &config_filename);

  /**
   * \brief Detect grasps.
   * \param cloud_in the point cloud
   * \return the list of grasps
   */
  std::vector<std::unique_ptr<candidate::Hand>> detectGrasps(
      util::Cloud &cloud);

  /**
   * \brief Compare if two grasps are equal based on their position.
   * \param h1 the first grasp
   * \param h2 the second grasp
   */
  struct compareGraspPositions {
    bool operator()(const candidate::Hand &h1, const candidate::Hand &h2) {
      double position_thresh = 0.001;
      double approach_thresh = 0.99;
      return (h1.getApproach().dot(h2.getApproach()) < approach_thresh) &&
             ((h1.getPosition() - h2.getPosition()).norm() > position_thresh);
    }
  };

 private:
  /**
   * \brief Draw (x,y,z) grasp samples from sum of Gaussians.
   * \param hands the list of grasp candidate sets
   * \param sigma standard deviation of the Gaussian
   * \param num_gauss_samples number of samples to be drawn
   * \return the samples drawn from the sum of Gaussians
   */
  void drawSamplesFromSumOfGaussians(
      const std::vector<std::unique_ptr<candidate::HandSet>> &hand_sets,
      double sigma, int num_gauss_samples, Eigen::Matrix3Xd &samples_out);

  /**
   * \brief Draw (x,y,z) grasp samples from max of Gaussians.
   * \param hands the list of grasp candidate sets
   * \param sigma standard deviation of the Gaussian
   * \param num_gauss_samples number of samples to be drawn
   * \return the samples drawn from the sum of Gaussians
   */
  void drawSamplesFromMaxOfGaussians(
      const std::vector<std::unique_ptr<candidate::HandSet>> &hands,
      double sigma, int num_gauss_samples, Eigen::Matrix3Xd &samples_out,
      double term);

  void drawUniformSamples(const util::Cloud &cloud, int num_samples,
                          int start_idx, Eigen::Matrix3Xd &samples);

  std::unique_ptr<GraspDetector> grasp_detector_;
  std::unique_ptr<Clustering> clustering_;

  // sequential importance sampling parameters
  int num_iterations_;        ///< number of iterations of CEM
  int num_samples_;           ///< number of samples to use in each iteration
  int num_init_samples_;      ///< number of initial samples
  double prob_rand_samples_;  ///< probability of random samples
  double radius_;             ///< standard deviation of Gaussian distribution
  int sampling_method_;  ///< what sampling method is used (sum, max, weighted)
  double min_score_;     ///< minimum score to consider a candidate as a grasp

  // visualization parameters
  bool visualize_rounds_;  ///< if all iterations are visualized
  bool visualize_steps_;   ///< if all grasp candidates and all valid grasps are
                           /// visualized
  bool visualize_results_;  ///< if the final results are visualized

  // grasp filtering parameters
  std::vector<double> workspace_;         ///< the robot's workspace
  std::vector<double> workspace_grasps_;  ///< the robot's workspace

  bool filter_approach_direction_;
  Eigen::Vector3d direction_;
  double thresh_rad_;

  int num_threads_;  ///< number of CPU threads used in grasp detection
};

}  // namespace gpd

#endif /* SEQUENTIAL_IMPORTANCE_SAMPLING_H */
