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


#ifndef SEQUENTIAL_IMPORTANCE_SAMPLING_H
#define SEQUENTIAL_IMPORTANCE_SAMPLING_H


#include <ros/ros.h>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <set>

#include <gpg/cloud_camera.h>
#include <gpg/plot.h>

#include "../gpd/grasp_detector.h"


typedef pcl::PointCloud<pcl::PointXYZRGBA> PointCloudRGB;

typedef boost::variate_generator<boost::mt19937, boost::normal_distribution<> > Gaussian;


/** SequentialImportanceSampling class
 *
 * \brief Use sequential importance sampling to focus the sampling of grasp candidates.
 *
 * This class uses sequential importance sampling to focus the grasp candidate generation on areas of the point cloud
 * where more grasp candidates are expected to be found.
 *
 */
class SequentialImportanceSampling
{
public:

  /**
   * \brief Constructor.
   * \param node ROS node handle
   */
  SequentialImportanceSampling(ros::NodeHandle& node);

  /**
   * \brief Destructor.
   */
  ~SequentialImportanceSampling()
  {
    delete grasp_detector_;
  }

  /**
   * \brief Detect grasps.
   * \param cloud_cam_in the point cloud
   * \return the list of grasps
   */
  std::vector<Grasp> detectGrasps(const CloudCamera& cloud_cam_in);

  /**
   * Preprocess the point cloud (workspace filtering, voxelization, surface normals).
   * \param cloud_cam the point cloud
   */
  void preprocessPointCloud(CloudCamera& cloud_cam);

  /**
   * \brief Compare if two grasps are equal based on their position.
   * \param h1 the first grasp
   * \param h2 the second grasp
   */
  struct compareGraspPositions
  {
    bool operator()(const Grasp& h1, const Grasp& h2)
    {
      double position_thresh = 0.001;
      double approach_thresh = 0.99;
      return (h1.getApproach().dot(h2.getApproach()) < approach_thresh)
              && ((h1.getGraspBottom() - h2.getGraspBottom()).norm() > position_thresh);
    }
  };


private:

  /**
   * \brief Draw (x,y,z) grasp samples from sum of Gaussians.
   * \param hands the list of grasp candidate sets
   * \param generator Gaussian random number generator
   * \param sigma standard deviation of the Gaussian
   * \param num_gauss_samples number of samples to be drawn
   * \return the samples drawn from the sum of Gaussians
   */
  void drawSamplesFromSumOfGaussians(const std::vector<GraspSet>& hands, Gaussian& generator, double sigma,
    int num_gauss_samples, Eigen::Matrix3Xd& samples_out);

  /**
   * \brief Draw (x,y,z) grasp samples from max of Gaussians.
   * \param hands the list of grasp candidate sets
   * \param generator Gaussian random number generator
   * \param sigma standard deviation of the Gaussian
   * \param num_gauss_samples number of samples to be drawn
   * \return the samples drawn from the sum of Gaussians
   */
  void drawSamplesFromMaxOfGaussians(const std::vector<GraspSet>& hands, Gaussian& generator, double sigma,
    int num_gauss_samples, Eigen::Matrix3Xd& samples_out, double term);

  /**
   * \brief Draw weighted (x,y,z) grasp samples from set of Gaussians.
   * \param hands the list of grasp candidate sets
   * \param generator Gaussian random number generator
   * \param sigma standard deviation of the Gaussian
   * \param num_gauss_samples number of samples to be drawn
   * \param[out] samples_out the samples drawn from the set of Gaussians
   */
  void drawWeightedSamples(const std::vector<Grasp>& hands, Gaussian& generator, double sigma, int num_gauss_samples,
    Eigen::Matrix3Xd& samples_out);

  GraspDetector* grasp_detector_; ///< pointer to object for grasp detection

  // sequential importance sampling parameters
  int num_iterations_; ///< number of iterations of Sequential Importance Sampling
  int num_samples_; ///< number of samples to use in each iteration
  int num_init_samples_; ///< number of initial samples
  double prob_rand_samples_; ///< probability of random samples
  double radius_; ///< radius
  int sampling_method_; ///< what sampling method is used (sum, max, weighted)

  // visualization parameters
  bool visualize_rounds_; ///< if all iterations are visualized
  bool visualize_steps_; ///< if all grasp candidates and all valid grasps are visualized
  bool visualize_results_; ///< if the final results are visualized

  // grasp filtering parameters
  bool filter_grasps_; ///< if grasps are filtered based on the robot's workspace
  std::vector<double> workspace_; ///< the robot's workspace
  std::vector<double> workspace_grasps_; ///< the robot's workspace

  int num_threads_; ///< number of CPU threads used in grasp detection

  // standard parameters
  static const int NUM_ITERATIONS;
  static const int NUM_SAMPLES;
  static const int NUM_INIT_SAMPLES;
  static const double PROB_RAND_SAMPLES;
  static const double RADIUS;
  static const bool VISUALIZE_STEPS;
  static const bool VISUALIZE_RESULTS;
  static const int SAMPLING_METHOD;
};

#endif /* SEQUENTIAL_IMPORTANCE_SAMPLING_H */
