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

#ifndef HAND_H_
#define HAND_H_

// Eigen
#include <Eigen/Dense>

// System
#include <fstream>
#include <iostream>
#include <vector>

// custom
#include <gpd/candidate/finger_hand.h>

namespace gpd {
namespace candidate {

/**
 *\brief Label information
 */
struct Label {
  double score_{0.0};           ///< the score given by the classifier
  bool full_antipodal_{false};  ///< whether the grasp is antipodal
  bool half_antipodal_{false};  ///< whether the grasp is indeterminate

  Label(double score, bool full_antipodal, bool half_antipodal)
      : score_(score),
        full_antipodal_(full_antipodal),
        half_antipodal_(half_antipodal) {}
};

/**
 *\brief 2-D bounding box of hand closing region with respect to hand frame
 */
struct BoundingBox {
  double center_;
  double top_;
  double bottom_;
};

/**
 *
 * \brief Grasp represented as a robot hand pose
 *
 * This class represents a grasp candidate by the position and orientation of
 * the robot hand at the grasp before the fingers are closed.
 *
 */
class Hand {
 public:
  /**
   * \brief Default constructor.
   */
  Hand();

  /**
   * \brief Constructor.
   * \param sample the center of the point neighborhood associated with the
   * grasp
   * \param frame the orientation of the grasp as a rotation matrix
   * \param finger_hand the FingerHand object describing a feasible finger
   * placement
   * \param width the opening width of the robot hand
   */
  Hand(const Eigen::Vector3d &sample, const Eigen::Matrix3d &frame,
       const FingerHand &finger_hand, double width);

  /**
   * \brief Constructor.
   * \param sample the center of the point neighborhood associated with the
   * grasp
   * \param frame the orientation of the grasp as a rotation matrix
   * \param finger_hand the FingerHand object describing a feasible finger
   * placement
   */
  Hand(const Eigen::Vector3d &sample, const Eigen::Matrix3d &frame,
       const FingerHand &finger_hand);

  /**
   * \brief Set properties of the grasp.
   * \param finger_hand the FingerHand object describing a feasible finger
   * placement
   */
  void construct(const FingerHand &finger_hand);

  /**
   * \brief Write a list of grasps to a file.
   * \param filename location of the file
   * \param hands the list of grasps
   */
  void writeHandsToFile(const std::string &filename,
                        const std::vector<Hand> &hands) const;

  /**
   * \brief Print a description of the grasp candidate to the systen's standard
   * output.
   */
  void print() const;

  /**
   * \brief Return the approach vector of the grasp.
   * \return 3x1 grasp approach vector
   */
  const Eigen::Vector3d getApproach() const { return orientation_.col(0); }

  /**
   * \brief Return the binormal of the grasp.
   * \return 3x1 binormal
   */
  const Eigen::Vector3d getBinormal() const { return orientation_.col(1); }

  /**
   * \brief Return the hand axis of the grasp.
   * \return 3x1 hand axis
   */
  const Eigen::Vector3d getAxis() const { return orientation_.col(2); }

  /**
   * \brief Return whether the grasp is antipodal.
   * \return true if the grasp is antipodal, false otherwise
   */
  bool isFullAntipodal() const { return label_.full_antipodal_; }

  /**
   * \brief Return the position of the grasp.
   * \return the grasp position
   */
  const Eigen::Vector3d &getPosition() const { return position_; }

  /**
   * \brief Return the orientation of the grasp.
   * \return the grasp orientation (rotation matrix)
   */
  const Eigen::Matrix3d &getOrientation() const { return orientation_; }

  /**
   * \brief Return the width of the object contained in the grasp.
   * \return the width of the object contained in the grasp
   */
  double getGraspWidth() const { return grasp_width_; }

  /**
   * \brief Return whether the grasp is indeterminate.
   * \return true if the grasp is indeterminate, false otherwise
   */
  bool isHalfAntipodal() const { return label_.half_antipodal_; }

  /**
   * \brief Set whether the grasp is antipodal.
   * \param b whether the grasp is antipodal
   */
  void setFullAntipodal(bool b) { label_.full_antipodal_ = b; }

  /**
   * \brief Set whether the grasp is indeterminate.
   * \param b whether the grasp is indeterminate
   */
  void setHalfAntipodal(bool b) { label_.half_antipodal_ = b; }

  /**
   * \brief Set the width of the object contained in the grasp.
   * \param w the width of the object contained in the grasp
   */
  void setGraspWidth(double w) { grasp_width_ = w; }

  /**
   * \brief Set the position of the grasp.
   * \param position the grasp position
   */
  void setPosition(const Eigen::Vector3d &position) { position_ = position; }

  /**
   * \brief Get the score of the grasp.
   * \return the score
   */
  double getScore() const { return label_.score_; }

  /**
   * \brief Set the score of the grasp.
   * \param score the score
   */
  void setScore(double score) { label_.score_ = score; }

  /**
   * \brief Return the center of the point neighborhood associated with the
   * grasp.
   * \return the center
   */
  const Eigen::Vector3d &getSample() const { return sample_; }

  /**
   * \brief Return the grasp orientation.
   * \return the orientation of the grasp as a rotation matrix
   */
  const Eigen::Matrix3d &getFrame() const { return orientation_; }

  /**
   * \brief Return the center coordinate of the hand closing region along the
   * closing direction/axis of the robot hand.
   */
  double getCenter() const { return closing_box_.center_; }

  /**
   * \brief Return the bottom coordinate of the hand closing region along the
   * closing direction/axis of the robot hand..
   */
  double getBottom() const { return closing_box_.bottom_; }

  /**
   * \brief Return the top coordinate of the hand closing region along the
   * closing direction/axis of the robot hand.
   */
  double getTop() const { return closing_box_.top_; }

  /**
   * \brief Return the index of the finger placement.
   * \return the index of the finger placement
   */
  int getFingerPlacementIndex() const { return finger_placement_index_; }

 private:
  /**
   * \brief Calculate grasp positions (bottom, top, surface).
   * \param finger_hand the FingerHand object describing a feasible finger
   * placement
   */
  void calculateGraspPositions(const FingerHand &finger_hand);

  /**
   * \brief Convert an Eigen vector to a string.
   * \param v the vector
   * \return the string
   */
  std::string vectorToString(const Eigen::VectorXd &v) const;

  Eigen::Vector3d position_;  ///< grasp position (bottom center of robot hand)
  Eigen::Matrix3d orientation_;  ///< grasp orientation (rotation of robot hand)

  Eigen::Vector3d sample_;  ///< the sample at which the grasp was found
  double grasp_width_;      ///< the width of object enclosed by the fingers

  Label label_;                 ///< labeling information
  int finger_placement_index_;  ///< index of the finger placement that resulted
                                /// in this grasp
  BoundingBox closing_box_;     ///< defines region surrounded by fingers
};

}  // namespace candidate
}  // namespace gpd

#endif /* HAND_H_ */
