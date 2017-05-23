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


#ifndef GRASP_IMAGE_15_CHANNELS_H_
#define GRASP_IMAGE_15_CHANNELS_H_


#include "../gpd/grasp_image.h"


/**
 * \brief Set of images for projection of points onto plane orthogonal to one of the robot hand axes.
 */
struct Projection
{
  cv::Mat normals_image_; ///< image based on surface normals (3 channels)
  cv::Mat depth_image_; ///< image based on depth (1 channel)
  cv::Mat shadow_image_; ///< image based on shadow (1 channel)
};


/** GraspImage15Channels class
 *
 * \brief Create grasp image with 15 channels.
 *
 * This class creates the 15 channels grasp image. The image is composed of three projections of points onto a plane
 * orthogonal to one of the robot hand axes. Each of these projections contains three images of which one has
 * three channels, and the other two have one channel each.
 *
*/
class GraspImage15Channels: public GraspImage
{
public:

  /**
   * \brief Constructor.
   * \param image_size the image size
   * \param is_plotting if the image is visualized
   * \param points the points on which the image is based
   * \param shadow the shadow on which the image is based
   */
  GraspImage15Channels(int image_size, bool is_plotting, Eigen::Matrix3Xd* points, Eigen::Matrix3Xd* normals,
    Eigen::Matrix3Xd* shadow);

  /**
   * \brief Virtual destructor.
   */
  virtual ~GraspImage15Channels();

  /**
   * \brief Calculate the 15 channels grasp image.
   * \return the 15 channels image
   */
  cv::Mat calculateImage();


private:

  /**
   * \brief Calculate the images for a projection of points.
   * \param points the points which are projected
   * \param normals the surface normals of the points
   * \param shadow the shadow of the points
   * \return the three images associated with the projection
   */
  Projection calculateProjection(const Eigen::Matrix3Xd& points, const Eigen::Matrix3Xd& normals,
    const Eigen::Matrix3Xd& shadow);

  /**
   * \brief Concatenate three projections into a single 15 channels image.
   * \param projection1 the first projection
   * \param projection2 the second projection
   * \param projection3 the third projection
   * \return the 15 channels image
   */
  cv::Mat concatenateProjections(const Projection& projection1, const Projection& projection2,
    const Projection& projection3) const;

  /**
   * \brief Visualize the 15 channels image.
   * \param projections list of projections
   */
  void showImage(const std::vector<Projection>& projections) const;

  Eigen::Matrix3Xd* points_; ///< pointer to points matrix
  Eigen::Matrix3Xd* normals_; ///< pointer to surface normals matrix
  Eigen::Matrix3Xd* shadow_; ///< pointer to shadow matrix

  static const int NUM_CHANNELS; ///< the number of channels used for this image
};

#endif /* GRASP_IMAGE_15_CHANNELS_H_ */
