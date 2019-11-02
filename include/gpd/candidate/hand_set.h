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

#ifndef HAND_SET_H_
#define HAND_SET_H_

// System
#include <memory>
#include <vector>

// Boost
#include <boost/functional/hash.hpp>
#include <boost/pool/pool.hpp>
#include <boost/random/lagged_fibonacci.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/taus88.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/unordered_set.hpp>

// Eigen
#include <Eigen/Dense>
#include <Eigen/Geometry>

// Custom
#include <gpd/candidate/antipodal.h>
#include <gpd/candidate/finger_hand.h>
#include <gpd/candidate/hand.h>
#include <gpd/candidate/hand_geometry.h>
#include <gpd/candidate/local_frame.h>
#include <gpd/util/config_file.h>
#include <gpd/util/point_list.h>

// The hash and equality functions below are necessary for boost's unordered
// set.
namespace boost {
template <>
struct hash<Eigen::Vector3i> {
  inline size_t operator()(Eigen::Vector3i const &v) const {
    std::size_t seed = 0;

    for (int i = 0; i < v.size(); i++) {
      boost::hash_combine(seed, v(i));
    }

    return seed;
  }
};
}  // namespace boost

namespace gpd {
namespace candidate {

struct Vector3iEqual {
  inline bool operator()(const Eigen::Vector3i &a,
                         const Eigen::Vector3i &b) const {
    return a(0) == b(0) && a(1) == b(1) && a(2) == b(2);
  }
};

typedef boost::unordered_set<Eigen::Vector3i, boost::hash<Eigen::Vector3i>,
                             Vector3iEqual, std::allocator<Eigen::Vector3i>>
    Vector3iSet;

/**
 *
 * \brief Calculate a set of grasp candidates.
 *
 * This class calculate sets of grasp candidates. The grasp candidates in the
 * same set share the same point neighborhood and the same local reference
 * frame (LRF). The grasps are calculated by a local search over discretized
 * orientations and positions.
 *
 * \note For details, check out: Andreas ten Pas, Marcus Gualtieri, Kate
 * Saenko, and Robert Platt. Grasp Pose Detection in Point Clouds. The
 * International Journal of Robotics Research, Vol 36, Issue 13-14,
 * pp. 1455 - 1473. October 2017. https://arxiv.org/abs/1706.09911
 *
 */
class HandSet {
 public:
  /**
   * Constructor.
   */
  //  HandSet();

  /**
   * \brief Constructor.
   * \param hand_geometry the hand geometry parameters
   * \param angles the angles to be evaluated
   * \param hand_axes the hand axes about which to rotate
   * \param num_finger_placements the number of finger placements
   * \param deepen_hand if the hand is pushed forward onto the object
   */
  HandSet(const HandGeometry &hand_geometry, const Eigen::VectorXd &angles,
          const std::vector<int> &hand_axes, int num_finger_placements,
          bool deepen_hand, Antipodal &antipodal);

  /**
   * \brief Calculate a set of grasp candidates given a local reference frame.
   * \param point_list the point neighborhood
   * \param local_frame the local reference frame
   */
  void evalHandSet(const util::PointList &point_list,
                   const LocalFrame &local_frame);

  /**
   * \brief Calculate grasp candidates for a given rotation axis.
   * \param point_list the point neighborhood
   * \param local_frame the local reference frame
   * \param axis the index of the rotation axis
   * \param start the index of the first free element in `hands_`
   */
  void evalHands(const util::PointList &point_list,
                 const LocalFrame &local_frame, int axis, int start);

  /**
   * \brief Calculate the "shadow" of the point neighborhood.
   * \param point_list the point neighborhood
   * \param shadow_length the length of the shadow
   */
  Eigen::Matrix3Xd calculateShadow(const util::PointList &point_list,
                                   double shadow_length) const;

  /**
   * \brief Return the grasps contained in this grasp set.
   * \return the grasps contained in this grasp set
   */
  const std::vector<std::unique_ptr<Hand>> &getHands() const { return hands_; }

  /**
   * \brief Return the grasps contained in this grasp set (mutable).
   * \return the grasps contained in this grasp set
   */
  std::vector<std::unique_ptr<Hand>> &getHands() { return hands_; }

  /**
   * \brief Return the center of the point neighborhood.
   * \return the center of the point neighborhood
   */
  const Eigen::Vector3d &getSample() const { return sample_; }

  /**
   * \brief Return the local reference frame.
   * \return the local reference frame (3 x 3 rotation matrix)
   */
  const Eigen::Matrix3d &getFrame() const { return frame_; }

  /**
   * \brief Set the center of the point neighborhood.
   * \param sample the center of the point neighborhood
   */
  void setSample(const Eigen::Vector3d &sample) { sample_ = sample; }

  /**
   * \brief Return a list of booleans that indicate for each grasp if it is
   * valid or not.
   * \return the list of booleans
   */
  const Eigen::Array<bool, 1, Eigen::Dynamic> &getIsValid() const {
    return is_valid_;
  }

  /**
   * \brief Set, for each grasp, if it is valid or not.
   * \param isValid the list of booleans which indicate if each grasp is valid
   * or not
   */
  void setIsValid(const Eigen::Array<bool, 1, Eigen::Dynamic> &isValid) {
    is_valid_ = isValid;
  }

  /**
   * \brief Set a single grasp to be valid or not.
   * \param idx the index of the grasp
   * \param val true if the grasp is valid, false if it is not
   */
  void setIsValidWithIndex(int idx, bool val) { is_valid_[idx] = val; }

 private:
  /**
   * \brief Calculate shadow for one camera.
   * \param[in] points the list of points for which the shadow is calculated
   * \param[in] shadow_vec the vector that describes the direction of the shadow
   * relative to the camera origin
   * \param[in] num_shadow_points the number of shadow points to be calculated
   * \param[in] voxel_grid_size the size of the voxel grid
   * \param[out] shadow_set the set of shadow points
   */
  void calculateShadowForCamera(const Eigen::Matrix3Xd &points,
                                const Eigen::Vector3d &shadow_vec,
                                int num_shadow_points, double voxel_grid_size,
                                Vector3iSet &shadow_set) const;

  /**
   * \brief Modify a grasp candidate.
   * \param hand the grasp candidate to be modified
   * \param point_list the point neighborhood
   * \param indices the indices of the points in the hand closing region
   * \param finger_hand the FingerHand object that describes valid finger
   * placements
   * \return the modified grasp candidate
   */
  void modifyCandidate(Hand &hand, const util::PointList &point_list,
                       const std::vector<int> &indices,
                       const FingerHand &finger_hand) const;

  /**
   * \brief Label a grasp candidate as a viable grasp or not.
   * \param point_list the point neighborhood associated with the grasp
   * \param finger_hand the FingerHand object that describes valid finger
   * placements
   * \param hand the grasp
   */
  void labelHypothesis(const util::PointList &point_list,
                       const FingerHand &finger_hand, Hand &hand) const;

  /**
   * \brief Convert shadow voxels to shadow points.
   * \param voxels the shadow voxels
   * \param voxel_grid_size the size of the voxel grid
   * \return the shadow points
   */
  Eigen::Matrix3Xd shadowVoxelsToPoints(
      const std::vector<Eigen::Vector3i> &voxels, double voxel_grid_size) const;

  /**
   * \brief Calculate the intersection of two shadows.
   * \param set1 the first shadow
   * \param set2 the second shadow
   * \return the intersection
   */
  Vector3iSet intersection(const Vector3iSet &set1,
                           const Vector3iSet &set2) const;

  /**
   * \brief Generate a random integer.
   * \note Source:
   * http://software.intel.com/en-us/articles/fast-random-number-generator-on-
   * the-intel-pentiumr-4-processor/
   */
  inline int fastrand() const;

  Eigen::Vector3d sample_;  ///< the center of the point neighborhood
  Eigen::Matrix3d frame_;   ///< the local reference frame
  std::vector<std::unique_ptr<Hand>>
      hands_;  ///< the grasp candidates contained in this set
  Eigen::Array<bool, 1, Eigen::Dynamic>
      is_valid_;  ///< indicates for each grasp candidate if it is valid or not
  Eigen::VectorXd
      angles_;        ///< the hand orientations to consider in the local search
  bool deepen_hand_;  ///< if the hand is pushed forward onto the object
  int num_finger_placements_;  ///< the number of finger placements to evaluate

  HandGeometry hand_geometry_;  ///< the robot hand geometry
  std::vector<int> hand_axes_;  ///< the axes about which the hand frame is
                                /// rotated to evaluate different orientations

  Antipodal &antipodal_;

  static int seed_;  ///< seed for the random generator in fastrand()

  static const Eigen::Vector3d AXES[3];  ///< standard rotation axes

  static const bool MEASURE_TIME;  ///< if runtime is measured
};

}  // namespace candidate
}  // namespace gpd

#endif /* HAND_SET_H_ */
