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

#ifndef FINGER_HAND_H
#define FINGER_HAND_H

#include <Eigen/Dense>
#include <iostream>
#include <vector>

namespace gpd {
namespace candidate {

/**
 *
 * \brief Calculate collision-free finger placements.
 *
 * This class calculates collision-free finger placements. The parameters are
 * the outer diameter, the width of the fingers, and the length of the fingers
 * of the robot hand. Also considers the "bite" that the grasp must have, i.e.,
 * by how much the robot hand can be moved onto the object.
 *
 */
class FingerHand {
 public:
  /**
   * \brief Default constructor.
   */
  FingerHand() {}

  /**
   * \brief Constructor.
   * \param finger_width the width of the fingers
   * \param hand_outer_diameter the maximum aperture of the robot hand
   * \param hand_depth the length of the fingers
   */
  FingerHand(double finger_width, double hand_outer_diameter, double hand_depth,
             int num_placements);

  /**
   * \brief Find possible finger placements.
   *
   * Finger placements need to be collision-free and contain at least one point
   * in between the fingers.
   *
   * \param points the points to checked for possible finger placements
   * \param bite how far the robot can be moved into the object
   * \param idx if this is larger than -1, only check the <idx>-th finger
   * placement
   */
  void evaluateFingers(const Eigen::Matrix3Xd &points, double bite,
                       int idx = -1);

  /**
   * \brief Chhose the middle among all valid finger placements.
   * \return the index of the middle finger placement
   */
  int chooseMiddleHand();

  /**
   * \brief Try to move the robot hand as far as possible onto the object.
   * \param points the points that the finger placement is evaluated on, assumed
   * to be rotated into the hand frame and
   * cropped by hand height
   * \param min_depth the minimum depth that the hand can be moved onto the
   * object
   * \param max_depth the maximum depth that the hand can be moved onto the
   * object
   * \return the index of the middle finger placement
   */
  int deepenHand(const Eigen::Matrix3Xd &points, double min_depth,
                 double max_depth);

  /**
   * \brief Compute which of the given points are located in the closing region
   * of the robot hand.
   * \param points the points
   * \param idx if this is larger than -1, only check the <idx>-th finger
   * placement
   * \return the points that are located in the closing region
   */
  std::vector<int> computePointsInClosingRegion(const Eigen::Matrix3Xd &points,
                                                int idx = -1);

  /**
   * \brief Check which 2-finger placements are feasible.
   */
  void evaluateHand();

  /**
   * \brief Check the 2-finger placement at a given index.
   * \param idx the index of the finger placement
   */
  void evaluateHand(int idx);

  /**
   * \brief Return the depth of the hand.
   * \return the hand depth
   */
  double getHandDepth() const { return hand_depth_; }

  /**
   * \brief Return the finger placement evaluations.
   * \return the hand configuration evaluations
   */
  const Eigen::Array<bool, 1, Eigen::Dynamic> &getHand() const { return hand_; }

  /**
   * \brief Return the finger placement evaluations.
   * \return the hand configuration evaluations
   */
  const Eigen::Array<bool, 1, Eigen::Dynamic> &getFingers() const {
    return fingers_;
  }

  /**
   * \brief Return where the bottom of the hand is (where the hand base is).
   * \return the bottom of the hand
   */
  double getBottom() const { return bottom_; }

  /**
   * \brief Return where the left finger is.
   * \return the left finger's location
   */
  double getLeft() const { return left_; }

  /**
   * \brief Return where the right finger is.
   * \return the right finger's location
   */
  double getRight() const { return right_; }

  /**
   * \brief Return where the top of the hand is (where the fingertips are).
   * \return the top of the hand
   */
  double getTop() const { return top_; }

  /**
   * \brief Return where the center of the hand is.
   * \return the center of the hand
   */
  double getCenter() const { return center_; }

  /**
   * \brief Return where the bottom of the hand is in the point cloud.
   * \return the bottom of the hand in the point cloud
   */
  double getSurface() const { return surface_; }

  /**
   * \brief Return the index of the forward axis.
   * \return the index of the forward axis
   */
  int getForwardAxis() const { return forward_axis_; }

  /**
   * \brief Set the index of the forward axis.
   * \param forward_axis the index of the forward axis
   */
  void setForwardAxis(int forward_axis) { forward_axis_ = forward_axis; }

  /**
   * \brief Get the index of the lateral axis (hand closing direction).
   * \param the index of the lateral axis
   */
  int getLateralAxis() const { return lateral_axis_; }

  /**
   * \brief Set the index of the lateral axis (hand closing direction).
   * \param lateral_axis the index of the lateral axis
   */
  void setLateralAxis(int lateral_axis) { lateral_axis_ = lateral_axis; }

  /**
   * \brief Set the bottom of the hand (robot hand base).
   * \param bottom the hand bottom
   */
  void setBottom(double bottom) { bottom_ = bottom; }

  /**
   * \brief Set the center of the hand.
   * \param bottom the hand center
   */
  void setCenter(double center) { center_ = center; }

  /**
   * \brief Set the lateral position of the left finger.
   * \param left the lateral position of the left finger
   */
  void setLeft(double left) { left_ = left; }

  /**
   * \brief Set the lateral position of the right finger.
   * \param right the lateral position of the right finger
   */
  void setRight(double right) { right_ = right; }

  /**
   * \brief Set where the bottom of the hand is in the point cloud.
   * \param left where the bottom of the hand is in the point cloud
   */
  void setSurface(double surface) { surface_ = surface; }

  /**
   * \brief Set where the top of the hand is (where the fingertips are).
   * \param top where the top of the hand is
   */
  void setTop(double top) { top_ = top; }

 private:
  /**
   * \brief Check that a given finger does not collide with the point cloud.
   * \param points the points to be checked for collision (size: 3 x n)
   * \param indices the indices of the points to be checked for collision
   * \param idx the index of the finger to be checked
   * \return true if it does not collide, false if it collides
   */
  bool isGapFree(const Eigen::Matrix3Xd &points,
                 const std::vector<int> &indices, int idx);

  int forward_axis_;  ///< the index of the horizontal axis in the hand frame
                      ///(grasp approach direction)
  int lateral_axis_;  ///< the index of the vertical axis in the hand frame
                      ///(closing direction of the robot hand)

  double finger_width_;  ///< the width of the robot hand fingers
  double hand_depth_;    ///< the hand depth (finger length)

  Eigen::VectorXd finger_spacing_;  ///< the possible finger placements
  Eigen::Array<bool, 1, Eigen::Dynamic>
      fingers_;  ///< indicates the feasible fingers
  Eigen::Array<bool, 1, Eigen::Dynamic>
      hand_;        ///< indicates the feasible 2-finger placements
  double bottom_;   ///< the base of the hand
  double top_;      ///< the top of the hand, where the fingertips are
  double left_;     ///< the left side of the gripper bounding box
  double right_;    ///< the right side of the gripper bounding box
  double center_;   ///< the horizontal center of the gripper bounding box
  double surface_;  ///< the corresponding vertical base point of the hand in
                    /// the point cloud
};

}  // namespace candidate
}  // namespace gpd

#endif
