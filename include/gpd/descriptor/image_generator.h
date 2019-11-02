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

#ifndef IMAGE_GENERATOR_H_
#define IMAGE_GENERATOR_H_

#include <sys/stat.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/StdVector>

#include <opencv2/core/core.hpp>

#include <pcl/filters/extract_indices.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/cloud_viewer.h>

#include <gpd/candidate/hand_set.h>
#include <gpd/descriptor/image_strategy.h>
#include <gpd/util/cloud.h>
#include <gpd/util/eigen_utils.h>

typedef std::pair<Eigen::Matrix3Xd, Eigen::Matrix3Xd> Matrix3XdPair;
typedef pcl::PointCloud<pcl::PointXYZRGBA> PointCloudRGBA;

namespace gpd {
namespace descriptor {

/**
 *
 * \brief Create grasp images for classification.
 *
 * Creates images for the input layer of a convolutional neural network. Each
 * image represents a grasp candidate. We call these "grasp images".
 *
 */
class ImageGenerator {
 public:
  /**
   * \brief Constructor.
   * \param params parameters for grasp images
   * \param num_threads number of CPU threads to be used
   * \param is_plotting if images are visualized
   * \param remove_plane if the support/table plane is removed before
   * calculating images
   */
  ImageGenerator(const descriptor::ImageGeometry &image_geometry,
                 int num_threads, int num_orientations, bool is_plotting,
                 bool remove_plane);

  /**
   * \brief Create a list of grasp images for a given list of grasp candidates.
   * \param cloud_cam the point cloud
   * \param hand_set_list the list of grasp candidates
   * \return the list of grasp images
   */
  void createImages(
      const util::Cloud &cloud_cam,
      const std::vector<std::unique_ptr<candidate::HandSet>> &hand_set_list,
      std::vector<std::unique_ptr<cv::Mat>> &images_out,
      std::vector<std::unique_ptr<candidate::Hand>> &hands_out) const;

  /**
   * \brief Return the parameters of the grasp image.
   * \return the grasp image parameters
   */
  const descriptor::ImageGeometry &getImageGeometry() const {
    return image_params_;
  }

 private:
  /**
   * \brief Remove the plane from the point cloud. Sets <point_list> to all
   * non-planar points if the plane is found, otherwise <point_list> has the
   * same points as <cloud>.
   * \param cloud the cloud
   * \param point_list the list of points corresponding to the cloud
   */
  void removePlane(const util::Cloud &cloud_cam,
                   util::PointList &point_list) const;

  void createImageList(
      const std::vector<std::unique_ptr<candidate::HandSet>> &hand_set_list,
      const std::vector<util::PointList> &nn_points_list,
      std::vector<std::unique_ptr<cv::Mat>> &images_out,
      std::vector<std::unique_ptr<candidate::Hand>> &hands_out) const;

  int num_threads_;
  int num_orientations_;
  descriptor::ImageGeometry image_params_;
  std::unique_ptr<descriptor::ImageStrategy> image_strategy_;
  bool is_plotting_;
  bool remove_plane_;
};

}  // namespace descriptor
}  // namespace gpd

#endif /* IMAGE_GENERATOR_H_ */
