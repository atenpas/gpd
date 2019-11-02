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

#ifndef EIGEN_UTILS_H_
#define EIGEN_UTILS_H_

#include <Eigen/Dense>

#include <vector>

namespace gpd {
namespace util {

/**
 *
 * \brief Utility functions for the Eigen matrix library
 *
 * This namespace contains utility functions for the Eigen matrix library such
 * as slicing matrices and rounding vectors.
 *
 */
namespace EigenUtils {
/**
 * \brief Slice a given matrix given a set of column indices.
 * \param mat the matrix to be sliced
 * \param indices set of column indices
 * \return the columns of the given matrix contained in the indices set
 */
Eigen::Matrix3Xd sliceMatrix(const Eigen::Matrix3Xd &mat,
                             const std::vector<int> &indices);

/**
 * \brief Slice a given matrix given a set of column indices.
 * \param mat the matrix to be sliced
 * \param indices set of column indices
 * \return the columns of the given matrix contained in the indices set
 */
Eigen::MatrixXi sliceMatrix(const Eigen::MatrixXi &mat,
                            const std::vector<int> &indices);

/**
 * \brief Round the elements of a floating point vector to the nearest, smaller
 * integers.
 * \param a the vector to be rounded down
 * \return the vector containing the rounded down elements
 */
Eigen::Vector3i floorVector(const Eigen::Vector3f &a);
}  // namespace EigenUtils

}  // namespace util
}  // namespace gpd

#endif /* EIGEN_UTILS_H_ */
