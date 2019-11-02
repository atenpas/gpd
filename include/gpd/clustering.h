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

#ifndef CLUSTERING_H_
#define CLUSTERING_H_

#include <math.h>
#include <memory>
#include <set>
#include <vector>

#include <gpd/candidate/hand.h>

namespace gpd {

/**
 *
 * \brief Group grasp candidates in clusters
 *
 * This class searches for clusters of grasps. Grasps in the same cluster are
 * geometrically similar.
 *
 */
class Clustering {
 public:
  /**
   * \brief Constructor.
   * \param min_inliers the minimum number of grasps a cluster is required to
   * contain
   */
  Clustering(int min_inliers) : min_inliers_(min_inliers){};

  /**
   * \brief Search for clusters given a list of grasps.
   * \param hand_list the list of grasps
   * \param remove_inliers if grasps already assigned to a cluster are ignored
   * for the next cluster
   */
  std::vector<std::unique_ptr<candidate::Hand>> findClusters(
      const std::vector<std::unique_ptr<candidate::Hand>> &hand_list,
      bool remove_inliers = false);

  /**
   * \brief Return the minimum number of cluster inliers.
   * \return the minimum number of cluster inliers
   */
  int getMinInliers() const { return min_inliers_; }

  /**
   * \brief Set the minimum number of cluster inliers.
   * \param min_inliers the minimum number of cluster inliers
   */
  void setMinInliers(int min_inliers) { min_inliers_ = min_inliers; }

 private:
  int min_inliers_;  ///< minimum number of geometrically aligned candidates
                     /// required to form a cluster
};

}  // namespace gpd

#endif /* CLUSTERING_H_ */
