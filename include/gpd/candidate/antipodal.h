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

#ifndef ANTIPODAL_H_
#define ANTIPODAL_H_

#include <math.h>
#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <vector>

#include <Eigen/Dense>

#include <gpd/util/point_list.h>

namespace gpd {
namespace candidate {

/**
 *
 * \brief Check if a grasp is antipodal.
 *
 * This class checks if a grasp candidate satisfies the antipodal condition.
 *
 */
class Antipodal {
 public:
  Antipodal(double friction_coeff, int min_viable)
      : friction_coeff_(friction_coeff), min_viable_(min_viable) {}

  /**
   * \brief Check if a grasp is antipodal.
   * \param point_list the list of points associated with the grasp
   * \param extremal_threshold
   * \param lateral_axis the closing direction of the robot hand
   * \param forward_axis the forward direction of the robot hand
   * \param vertical_axis the vertical direction of the robot hand
   * \return 0 if it's not antipodal, 1 if one finger is antipodal, 2 if the
   * grasp is antipodal
   */
  int evaluateGrasp(const util::PointList &point_list, double extremal_thresh,
                    int lateral_axis = 0, int forward_axis = 1,
                    int vertical_axis = 2) const;

  /**
   * \brief Check if a grasp is antipodal.
   * \note Deprecated method.
   * \param normals the set of surface normals associated with the grasp
   * \param thresh_half the threshold to consider the grasp half-antipodal
   * \param thresh_full the threshold to conisder the grasp full-antipodal
   */
  int evaluateGrasp(const Eigen::Matrix3Xd &normals, double thresh_half,
                    double thresh_full) const;

  double friction_coeff_;  ///< angle of friction cone in degrees
  int min_viable_;  ///< minimum number of points on each side to be antipodal

  static const int NO_GRASP;    // normals point not toward any finger
  static const int HALF_GRASP;  // normals point towards one finger
  static const int FULL_GRASP;  // normals point towards both fingers
};

}  // namespace candidate
}  // namespace gpd

#endif /* ANTIPODAL_H_ */
