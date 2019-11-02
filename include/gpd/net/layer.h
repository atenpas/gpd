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

#ifndef LAYER_H
#define LAYER_H

#include <Eigen/Dense>

#include <vector>

namespace gpd {
namespace net {

/**
 *
 * \brief Abstract base class for neural network layers in the custom framework.
 *
 */
class Layer {
 public:
  /**
   * \brief Constructor.
   */
  virtual ~Layer() {}

  /**
   * \brief Forward pass for the layer.
   * \return output of forward pass
   */
  virtual Eigen::MatrixXf forward(const std::vector<float> &x) const = 0;

  /**
   * \brief Set the parameters of the layer.
   * \param weights the weights
   * \param biases the biases
   */
  void setWeightsAndBiases(const std::vector<float> &weights,
                           const std::vector<float> &biases) {
    weights_ = weights;
    biases_ = biases;
  }

 protected:
  std::vector<float> weights_;
  std::vector<float> biases_;
};

}  // namespace net
}  // namespace gpd

#endif /* LAYER_H */
