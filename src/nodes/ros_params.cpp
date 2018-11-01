/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2018 Intel Corporation
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

#include "nodes/ros_params.h"

void ROSParameters::getDetectionParams(ros::NodeHandle& node, GraspDetector::GraspDetectionParameters& param)
{
  // Read hand geometry parameters.
  node.param("finger_width", param.hand_search_params.finger_width_, 0.01);
  node.param("hand_outer_diameter", param.hand_search_params.hand_outer_diameter_, 0.09);
  node.param("hand_depth", param.hand_search_params.hand_depth_, 0.06);
  node.param("hand_height", param.hand_search_params.hand_height_, 0.02);
  node.param("init_bite", param.hand_search_params.init_bite_, 0.015);

  // Read local hand search parameters.
  node.param("nn_radius", param.hand_search_params.nn_radius_frames_, 0.01);
  node.param("num_orientations", param.hand_search_params.num_orientations_, 8);
  node.param("num_samples", param.hand_search_params.num_samples_, 500);
  node.param("num_threads", param.hand_search_params.num_threads_, 1);
  node.param("rotation_axis", param.hand_search_params.rotation_axis_, 2); // cannot be changed

  // Read plotting parameters.
  node.param("plot_samples", param.plot_samples_, false);
  node.param("plot_normals", param.plot_normals_, false);
  param.generator_params.plot_normals_ = param.plot_normals_;
  node.param("plot_filtered_grasps", param.plot_filtered_grasps_, false);
  node.param("plot_valid_grasps", param.plot_valid_grasps_, false);
  node.param("plot_clusters", param.plot_clusters_, false);
  node.param("plot_selected_grasps", param.plot_selected_grasps_, false);

  // Read general parameters.
  param.generator_params.num_samples_ = param.hand_search_params.num_samples_;
  param.generator_params.num_threads_ = param.hand_search_params.num_threads_;
  node.param("plot_candidates", param.generator_params.plot_grasps_, false);

  // Read preprocessing parameters.
  node.param("remove_outliers", param.generator_params.remove_statistical_outliers_, true);
  node.param("voxelize", param.generator_params.voxelize_, true);
  node.getParam("workspace", param.generator_params.workspace_);
  node.getParam("workspace_grasps", param.workspace_);

  // Read classification parameters and create classifier.
  node.param("model_file", param.model_file_, std::string(""));
  node.param("trained_file", param.weights_file_, std::string(""));
  node.param("min_score_diff", param.min_score_diff_, 500.0);
  node.param("create_image_batches", param.create_image_batches_, true);
  node.param("device", param.device_, 0);

  // Read grasp image parameters.
  node.param("image_outer_diameter", param.image_params.outer_diameter_, param.hand_search_params.hand_outer_diameter_);
  node.param("image_depth", param.image_params.depth_, param.hand_search_params.hand_depth_);
  node.param("image_height", param.image_params.height_, param.hand_search_params.hand_height_);
  node.param("image_size", param.image_params.size_, 60);
  node.param("image_num_channels", param.image_params.num_channels_, 15);

  // Read learning parameters.
  node.param("remove_plane_before_image_calculation", param.remove_plane_, false);

  // Read grasp filtering parameters
  node.param("filter_grasps", param.filter_grasps_, false);
  node.param("filter_half_antipodal", param.filter_half_antipodal_, false);
  param.gripper_width_range_.push_back(0.03);
  param.gripper_width_range_.push_back(0.07);
  node.getParam("gripper_width_range", param.gripper_width_range_);

  // Read clustering parameters
  node.param("min_inliers", param.min_inliers_, 0);

  // Read grasp selection parameters
  node.param("num_selected", param.num_selected_, 100);
}

void ROSParameters::getSamplingParams(ros::NodeHandle& node, SequentialImportanceSampling::SISamplingParameters& param)
{
  node.param("num_init_samples", param.num_init_samples_, SequentialImportanceSampling::NUM_INIT_SAMPLES);
  node.param("num_iterations", param.num_iterations_, SequentialImportanceSampling::NUM_ITERATIONS);
  node.param("num_samples_per_iteration", param.num_samples_, SequentialImportanceSampling::NUM_SAMPLES);
  node.param("prob_rand_samples", param.prob_rand_samples_, SequentialImportanceSampling::PROB_RAND_SAMPLES);
  node.param("std", param.radius_, SequentialImportanceSampling::RADIUS);
  node.param("sampling_method", param.sampling_method_, SequentialImportanceSampling::SAMPLING_METHOD);
  node.param("visualize_rounds", param.visualize_rounds_, false);
  node.param("visualize_steps", param.visualize_steps_, SequentialImportanceSampling::VISUALIZE_STEPS);
  node.param("visualize_results", param.visualize_results_, SequentialImportanceSampling::VISUALIZE_RESULTS);
  node.param("filter_grasps", param.filter_grasps_, false);
  node.param("num_threads", param.num_threads_, 1);

  node.getParam("workspace", param.workspace_);
  node.getParam("workspace_grasps", param.workspace_grasps_);
}

#ifdef USE_CAFFE
void ROSParameters::getGeneratorParams(ros::NodeHandle& node, DataGenerator::DataGenerationParameters& param)
{
  // Read hand geometry parameters
  node.param("finger_width", param.hand_search_params.finger_width_, 0.01);
  node.param("hand_outer_diameter", param.hand_search_params.hand_outer_diameter_, 0.09);
  node.param("hand_depth", param.hand_search_params.hand_depth_, 0.06);
  node.param("hand_height", param.hand_search_params.hand_height_, 0.02);
  node.param("init_bite", param.hand_search_params.init_bite_, 0.015);

  // Read local hand search parameters
  node.param("nn_radius", param.hand_search_params.nn_radius_frames_, 0.01);
  node.param("num_orientations", param.hand_search_params.num_orientations_, 8);
  node.param("num_samples", param.hand_search_params.num_samples_, 500);
  node.param("num_threads", param.hand_search_params.num_threads_, 1);
  node.param("rotation_axis", param.hand_search_params.rotation_axis_, 2);

  // Read general parameters
  param.generator_params.num_samples_ = param.hand_search_params.num_samples_;
  param.generator_params.num_threads_ = param.hand_search_params.num_threads_;
  node.param("plot_candidates", param.generator_params.plot_grasps_, false);

  // Read preprocessing parameters
  node.param("remove_outliers", param.generator_params.remove_statistical_outliers_, true);
  node.param("voxelize", param.generator_params.voxelize_, true);
  node.getParam("workspace", param.generator_params.workspace_);

  // Read plotting parameters.
  param.generator_params.plot_grasps_ = false;
  node.param("plot_normals", param.generator_params.plot_normals_, false);

  node.param("image_outer_diameter", param.image_params.outer_diameter_, 0.09);
  node.param("image_depth", param.image_params.depth_, 0.06);
  node.param("image_height", param.image_params.height_, 0.02);
  node.param("image_size", param.image_params.size_, 60);
  node.param("image_num_channels", param.image_params.num_channels_, 15);

  // Read learning parameters.
  node.param("remove_plane_before_image_calculation", param.remove_plane, false);
  node.param("num_orientations", param.num_orientations, 8);
  node.param("num_threads", param.num_threads, 1);

  // Set the position from which the camera sees the point cloud.
  std::vector<double> camera_position;
  node.getParam("camera_position", camera_position);
  param.view_points << camera_position[0], camera_position[1], camera_position[2];
  // Load the point cloud from the file.
  node.param("cloud_file_name", param.filename, std::string(""));

  node.param("data_root", param.data_root, std::string(""));
  node.param("objects_file", param.objects_file_location, std::string(""));
  node.param("output_root", param.output_root, std::string(""));
  node.param("plot_grasps", param.plot_grasps, false);
  node.param("num_views", param.num_views, 20);
}
#endif
