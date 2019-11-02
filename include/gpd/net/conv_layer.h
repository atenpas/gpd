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

#ifndef CONV_LAYER_H
#define CONV_LAYER_H

#include <Eigen/Dense>

#include <iostream>
#include <vector>

#include <gpd/net/layer.h>

namespace gpd {
namespace net {

/**
 *
 * \brief Convolutional layer.
 *
 * A convolutional layer for a neural network for the `EigenClassifier`.
 *
 */
class ConvLayer : public Layer {
 public:
  /**
   * \brief Constructor.
   * \param width input width
   * \param height input height
   * \param depth input depth
   * \param num_filters number of convolutional filters
   * \param spatial_extent size of each filter
   * \param padding how much zero-padding is used
   */
  ConvLayer(int width, int height, int depth, int num_filters,
            int spatial_extent, int stride, int padding);

  /**
   * \brief Forward pass.
   * \param x input
   * \return output of forward pass
   */
  Eigen::MatrixXf forward(const std::vector<float> &x) const;

 private:
  typedef Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                                         Eigen::RowMajor>>
      RowMajorMatrixMap;

  /**
   * \brief Calculate the forward pass.
   * \param W weight matrix
   * \param b bias vector
   * \param X input matrix
   * \return output of forward pass
   */
  Eigen::MatrixXf forward(const Eigen::MatrixXf &W, const Eigen::VectorXf &b,
                          const Eigen::MatrixXf &X) const;

  bool is_a_ge_zero_and_a_lt_b(int a, int b) const;

  /**
   * \brief Convert image to array. Arranges image slices into columns so that
   * the forward pass can be expressed as a
   * as a matrix multiplication.
   * \param data_im the image to be converted
   * \param channels number of image channels
   * \param height image height
   * \param width image width
   * \param num_kernels number of filters to be used in the convolution
   * \param kernel_h filter height
   * \param kernel_w filter width
   * \param stride_h stride height
   * \param stride_w stride width
   * \param[out] data_col the array
   */
  void imageToColumns(const float *data_im, const int channels,
                      const int height, const int width, const int num_kernels,
                      const int kernel_h, const int kernel_w,
                      const int stride_h, const int stride_w,
                      float *data_col) const;

  int w1, h1, d1;  // size of input volume: w1 x h1 x d1
  int w2, h2, d2;  // size of output volume: w2 x h2 x d2
  int k, f, s, p;  // number of filters, their spatial extent, stride, amount of
                   // zero padding
  int W_row_r, W_row_c;  // number of rows and columns in matrix W_row
  int X_col_r, X_col_c;  // number of rows and columns in matrix X_col
};

}  // namespace net
}  // namespace gpd

#endif /* CONV_LAYER_H */
