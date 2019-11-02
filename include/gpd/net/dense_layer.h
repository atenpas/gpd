/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2017, Andreas ten Pas
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

#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include <Eigen/Dense>

#include <iostream>
#include <vector>

#include <gpd/net/layer.h>

namespace gpd {
namespace net {

/**
 *
 * \brief Dense (fully connected) layer.
 *
 * A dense (fully connected) layer for a neural network for the
 * `EigenClassifier`.
 *
 */
class DenseLayer : public Layer {
 public:
  /**
   * \brief Contructor.
   * \param num_units number of units/neurons in this layer
   */
  DenseLayer(int num_units) : num_units_(num_units) {}

  /**
   * \brief Forward pass.
   * \return output of forward pass
   */
  Eigen::MatrixXf forward(const std::vector<float> &x) const;

 private:
  int num_units_;  ///< the number of units
};

}  // namespace net
}  // namespace gpd

#endif /* DENSE_LAYER_H */
