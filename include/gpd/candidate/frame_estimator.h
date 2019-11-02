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

#ifndef FRAME_ESTIMATOR_H
#define FRAME_ESTIMATOR_H

#include <vector>

#include <Eigen/Dense>

#include <pcl/kdtree/kdtree.h>

#include <omp.h>

#include <gpd/candidate/local_frame.h>
#include <gpd/util/cloud.h>

namespace gpd {
namespace candidate {

typedef pcl::PointCloud<pcl::PointXYZRGBA> PointCloudRGBA;

/**
 *
 * \brief Estimate local reference frames.
 *
 * This class estimates local reference frames (LRFs) for point neighborhoods.
 *
 */
class FrameEstimator {
 public:
  /**
   * \brief Constructor.
   * \param num_threads the number of CPU threads to be used
   */
  FrameEstimator(int num_threads) : num_threads_(num_threads) {}

  /**
   * \brief Calculate local reference frames given a list of point cloud
   * indices.
   * \param cloud_cam the point cloud
   * \param indices the list of indices into the point cloud
   * \param radius the radius for the point neighborhood search
   * \param kdtree the kdtree used for faster neighborhood search
   * \return the list of local reference frames
   */
  std::vector<LocalFrame> calculateLocalFrames(
      const util::Cloud &cloud_cam, const std::vector<int> &indices,
      double radius, const pcl::KdTreeFLANN<pcl::PointXYZRGBA> &kdtree) const;

  /**
   * \brief Calculate local reference frames given a list of (x,y,z) samples.
   * \param cloud_cam the point cloud
   * \param samples the list of (x,y,z) samples
   * \param radius the radius for the point neighborhood search
   * \param kdtree the kdtree used for faster neighborhood search
   * \return the list of local reference frames
   */
  std::vector<LocalFrame> calculateLocalFrames(
      const util::Cloud &cloud_cam, const Eigen::Matrix3Xd &samples,
      double radius, const pcl::KdTreeFLANN<pcl::PointXYZRGBA> &kdtree) const;

  /**
   * \brief Calculate a local reference frame given a list of surface normals.
   * \param normals the list of surface normals
   * \param sample the center of the point neighborhood
   * \param radius the radius of the point neighborhood
   * \param kdtree the kdtree used for faster neighborhood search
   * \return the local reference frame
   */
  std::unique_ptr<LocalFrame> calculateFrame(
      const Eigen::Matrix3Xd &normals, const Eigen::Vector3d &sample,
      double radius, const pcl::KdTreeFLANN<pcl::PointXYZRGBA> &kdtree) const;

 private:
  /**
   * \brief Convert an Eigen::Vector3d object to a pcl::PointXYZRGBA.
   * \param v the Eigen vector
   * \reutrn the pcl point
   */
  pcl::PointXYZRGBA eigenVectorToPcl(const Eigen::Vector3d &v) const;

  int num_threads_;  ///< number of CPU threads to be used for calculating local
                     /// reference frames
};

}  // namespace candidate
}  // namespace gpd

#endif /* FRAME_ESTIMATOR_H */
