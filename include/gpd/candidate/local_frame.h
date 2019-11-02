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

#ifndef LOCAL_FRAME_H
#define LOCAL_FRAME_H

#include <Eigen/Dense>
#include <iostream>
#include <vector>

namespace gpd {
namespace candidate {

/**
 *
 * \brief Local reference frame.
 *
 * This class estimates the local reference frame (LRF) for a point
 * neighborhood. The coordinate axes of the local frame are the normal,
 * binormal, and curvature axis.
 */
class LocalFrame {
 public:
  /**
   * \brief Constructor.
   * \param T_cams the camera poses
   * \param sample the sample for which the point neighborhood was found
   */
  LocalFrame(const Eigen::Vector3d &sample) : sample_(sample) {}

  /**
   * \brief Estimate the average normal axis for the point neighborhood.
   * \param normals the 3xn matrix of normals found for points in the point
   * neighborhood
   */
  void findAverageNormalAxis(const Eigen::MatrixXd &normals);

  /**
   * \brief Print a description of the local reference frame.
   */
  void print();

  /**
   * \brief Return the sample for which the point neighborhood was found.
   * \return the 3x1 sample for which the point neighborhood was found
   */
  const Eigen::Vector3d &getSample() const { return sample_; }

  /**
   * \brief Return the binormal.
   * \return the 3x1 binormal vector
   */
  const Eigen::Vector3d &getBinormal() const { return binormal_; }

  /**
   * \brief Return the curvature axis.
   * \return the 3x1 curvature axis vector
   */
  const Eigen::Vector3d &getCurvatureAxis() const { return curvature_axis_; }

  /**
   * \brief Return the normal.
   * \return the 3x1 normal vector
   */
  const Eigen::Vector3d &getNormal() const { return normal_; }

  /**
   * \brief Set the sample for the point neighborhood.
   * \param sample the sample to be used
   */
  void setSample(const Eigen::Vector3d &sample) { sample_ = sample; }

 private:
  Eigen::Vector3d sample_;
  Eigen::Vector3d curvature_axis_;
  Eigen::Vector3d normal_;
  Eigen::Vector3d binormal_;
};

}  // namespace candidate
}  // namespace gpd

#endif  // LOCAL_FRAME_H_
