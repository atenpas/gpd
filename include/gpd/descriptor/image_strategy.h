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

#ifndef IMAGE_STRATEGY_H
#define IMAGE_STRATEGY_H

#include <vector>

#include <Eigen/Dense>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>

#include <gpd/candidate/hand_set.h>
#include <gpd/descriptor/image_geometry.h>

typedef std::pair<Eigen::Matrix3Xd, Eigen::Matrix3Xd> Matrix3XdPair;

namespace gpd {
namespace descriptor {

/**
 *
 * \brief Abstract base class for calculating grasp images/descriptors.
 *
 * Also offers methods to calculate various images.
 *
 */
class ImageStrategy {
 public:
  /**
   * \brief Create a strategy for calculating grasp images.
   * \param image_params the grasp image parameters
   * \param num_threads the number of CPU threads to be used
   * \param num_orientations the number of robot hand orientations
   * \param is_plotting if the images are visualized
   * \return the strategy for calculating grasp images
   */
  static std::unique_ptr<ImageStrategy> makeImageStrategy(
      const ImageGeometry &image_params, int num_threads, int num_orientations,
      bool is_plotting);

  /**
   * \brief Constructor.
   * \param image_params the grasp image parameters
   * \param num_threads the number of CPU threads to be used
   * \param num_orientations the number of robot hand orientations
   * \param is_plotting if the images are visualized
   * \return the strategy for calculating grasp images
   */
  ImageStrategy(const ImageGeometry &image_params, int num_threads,
                int num_orientations, bool is_plotting)
      : image_params_(image_params),
        num_orientations_(num_orientations),
        num_threads_(num_threads),
        is_plotting_(is_plotting) {}

  virtual ~ImageStrategy() {}

  /**
   * \brief Create grasp images given a list of grasp candidates.
   * \param hand_set the grasp candidates
   * \param nn_points the point neighborhoods used to calculate the images
   * \return the grasp images
   */
  virtual std::vector<std::unique_ptr<cv::Mat>> createImages(
      const candidate::HandSet &hand_set,
      const util::PointList &nn_points) const = 0;

  /**
   * \brief Return the grasp image parameters.
   * \return the grasp image parameters
   */
  const ImageGeometry &getImageParameters() const { return image_params_; }

 protected:
  /**
   * \brief Transform a given list of points to the unit image.
   * \param point_list the list of points
   * \param hand the grasp
   * \return the transformed points and their surface normals
   */
  Matrix3XdPair transformToUnitImage(const util::PointList &point_list,
                                     const candidate::Hand &hand) const;

  /**
   * \brief Find points that lie in the closing region of the robot hand.
   * \param hand the grasp
   * \param points the points to be checked
   * \return the indices of the points that lie inside the closing region
   */
  std::vector<int> findPointsInUnitImage(const candidate::Hand &hand,
                                         const Eigen::Matrix3Xd &points) const;

  /**
   * \brief Transform points to the unit image.
   * \param hand the grasp
   * \param points the points
   * \param indices the indices of the points to be transformed
   * \return the transformed points
   */
  Eigen::Matrix3Xd transformPointsToUnitImage(
      const candidate::Hand &hand, const Eigen::Matrix3Xd &points,
      const std::vector<int> &indices) const;

  /**
   * \brief Find the indices of the pixels that are occupied by a given list of
   * points.
   * \param points the points
   * \return the indices occupied by the points
   */
  Eigen::VectorXi findCellIndices(const Eigen::Matrix3Xd &points) const;

  /**
   * \brief Create a binary image based on which pixels are occupied by the
   * points.
   * \param cell_indices the indices of the points in the image
   * \return the image
   */
  cv::Mat createBinaryImage(const Eigen::VectorXi &cell_indices) const;

  /**
   * \brief Create an RGB image based on the surface normals of the points.
   * \param normals the surface normals
   * \param cell_indices the indices of the points in the image
   * \return the image
   */
  cv::Mat createNormalsImage(const Eigen::Matrix3Xd &normals,
                             const Eigen::VectorXi &cell_indices) const;

  /**
   * \brief Create a grey value image based on the depth value of the points.
   * \param points the points
   * \param cell_indices the indices of the points in the image
   * \return the image
   */
  cv::Mat createDepthImage(const Eigen::Matrix3Xd &points,
                           const Eigen::VectorXi &cell_indices) const;

  /**
   * \brief Create a grey value image based on the depth of the shadow points.
   * \param points the shadow points
   * \param cell_indices the indices of the shadow points in the image
   * \return the image
   */
  cv::Mat createShadowImage(const Eigen::Matrix3Xd &points,
                            const Eigen::VectorXi &cell_indices) const;

  ImageGeometry image_params_;  ///< grasp image parameters
  int num_orientations_;        ///< number of hand orientations
  int num_threads_;             ///< number of CPU threads to be used
  bool is_plotting_;            ///< if the grasp images are visualized

 private:
  /**
   * \brief Round a floating point vector to the closest, smaller integers.
   * \param a the floating point vector
   * \return the vector containing the integers
   */
  Eigen::VectorXi floorVector(const Eigen::VectorXd &a) const;
};

}  // namespace descriptor
}  // namespace gpd

#endif /* IMAGE_STRATEGY_H */
