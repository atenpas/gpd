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

#ifndef HAND_SEARCH_H
#define HAND_SEARCH_H

#include <Eigen/Dense>

#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/random_sample.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/point_cloud.h>

#include <omp.h>

#include <memory>

#include <gpd/candidate/antipodal.h>
#include <gpd/candidate/finger_hand.h>
#include <gpd/candidate/frame_estimator.h>
#include <gpd/candidate/hand.h>
#include <gpd/candidate/hand_geometry.h>
#include <gpd/candidate/hand_set.h>
#include <gpd/candidate/local_frame.h>
#include <gpd/util/plot.h>
#include <gpd/util/point_list.h>

namespace gpd {
namespace candidate {

typedef pcl::PointCloud<pcl::PointXYZRGBA> PointCloudRGB;

/**
 *
 * \brief Search for grasp candidates.
 *
 * This class searches for grasp candidates in a point cloud by first
 * calculating a local reference frame (LRF) for a point neighborhood, and then
 * finding geometrically feasible robot hand placements. Each feasible grasp
 * candidate is checked for mechanical stability (antipodal grasp or not).
 *
 */
class HandSearch {
 public:
  /**
   * \brief Parameters for the hand search.
   */
  struct Parameters {
    /** LRF estimation parameters */
    double nn_radius_frames_;  ///< radius for point neighborhood search for LRF

    /** grasp candidate generation */
    int num_threads_;             ///< the number of CPU threads to be used
    int num_samples_;             ///< the number of samples to be used
    int num_orientations_;        ///< number of hand orientations to evaluate
    int num_finger_placements_;   ///< number of finger placements to evaluate
    std::vector<int> hand_axes_;  ///< the axes about which different hand
                                  /// orientations are generated
    bool deepen_hand_;  ///< if the hand is pushed forward onto the object

    /** antipodal grasp check */
    double friction_coeff_;  ///< angle of friction cone in degrees
    int min_viable_;  ///< minimum number of points required to be antipodal

    HandGeometry hand_geometry_;  ///< robot hand geometry
  };

  /**
   * \brief Constructor.
   * \param params Parameters for the hand search
   */
  HandSearch(Parameters params);

  /**
   * \brief Search robot hand configurations.
   * \param cloud the point cloud
   * \return list of grasp candidate sets
   */
  std::vector<std::unique_ptr<candidate::HandSet>> searchHands(
      const util::Cloud &cloud) const;

  /**
   * \brief Reevaluate a list of grasp candidates.
   * \note Used to calculate ground truth.
   * \param cloud_cam the point cloud
   * \param grasps the list of grasp candidates
   * \return the list of reevaluated grasp candidates
   */
  std::vector<int> reevaluateHypotheses(
      const util::Cloud &cloud_cam,
      std::vector<std::unique_ptr<candidate::Hand>> &grasps,
      bool plot_samples = false) const;

  /**
   * \brief Return the parameters for the hand search.
   * \return params the hand search parameters
   */
  const Parameters &getParams() const { return params_; }

  /**
   * \brief Set the parameters for the hand search.
   * \param params the parameters
   */
  void setParameters(const Parameters &params) { params_ = params; }

 private:
  /**
   * \brief Search robot hand configurations given a list of local reference
   * frames.
   * \param cloud_cam the point cloud
   * \param frames the list of local reference frames
   * \param kdtree the KDTree object used for fast neighborhood search
   * \return the list of robot hand configurations
   */
  std::vector<std::unique_ptr<candidate::HandSet>> evalHands(
      const util::Cloud &cloud_cam,
      const std::vector<candidate::LocalFrame> &frames,
      const pcl::KdTreeFLANN<pcl::PointXYZRGBA> &kdtree) const;

  /**
   * \brief Reevaluate a grasp candidate.
   * \param point_list the point neighborhood associated with the grasp
   * \param hand the grasp
   * \param finger_hand the FingerHand object that describes a valid finger
   * placement
   * \param point_list_cropped the point neigborhood transformed into the hand
   * frame
   */
  bool reevaluateHypothesis(const util::PointList &point_list,
                            const candidate::Hand &hand,
                            FingerHand &finger_hand,
                            util::PointList &point_list_cropped) const;

  /**
   * \brief Calculate the label for a grasp candidate.
   * \param point_list the point neighborhood associated with the grasp
   * \param finger_hand the FingerHand object that describes a valid finger
   * placement
   * \return the label
   */
  int labelHypothesis(const util::PointList &point_list,
                      FingerHand &finger_hand) const;

  /**
   * \brief Convert an Eigen::Vector3d object to a pcl::PointXYZRGBA.
   * \param v the Eigen vector
   * \reutrn the pcl point
   */
  pcl::PointXYZRGBA eigenVectorToPcl(const Eigen::Vector3d &v) const;

  Parameters params_;  ///< parameters for the hand search

  double nn_radius_;  ///< radius for nearest neighbors search

  std::unique_ptr<Antipodal> antipodal_;
  std::unique_ptr<util::Plot> plot_;

  /** plotting parameters (optional, not read in from config file) **/
  bool plots_local_axes_;  ///< if the LRFs are plotted

  /** constants for rotation axis */
  static const int ROTATION_AXIS_NORMAL;          ///< normal axis of LRF
  static const int ROTATION_AXIS_BINORMAL;        ///< binormal axis of LRF
  static const int ROTATION_AXIS_CURVATURE_AXIS;  ///< curvature axis of LRF
};

}  // namespace candidate
}  // namespace gpd

#endif /* HAND_SEARCH_H */
