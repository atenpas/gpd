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

#include <gpd/layer.h>


class ConvLayer : public Layer
{
  public:

    ConvLayer(int width, int height, int depth, int num_filters, int spatial_extent, int stride, int padding);

    Eigen::MatrixXf forward(const std::vector<float>& x) const;


  private:

    typedef Eigen::Map<const Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> > RowMajorMatrixMap;

    Eigen::MatrixXf forward(const Eigen::MatrixXf& W, const Eigen::VectorXf& b, const Eigen::MatrixXf& X) const;

    bool is_a_ge_zero_and_a_lt_b(int a, int b) const;

    void imageToColumns(const float* data_im, const int channels, const int height, const int width,
      const int num_kernels, const int kernel_h, const int kernel_w, const int stride_h, const int stride_w,
      float* data_col) const;

    int w1, h1, d1; // size of input volume: w1 x h1 x d1
    int w2, h2, d2; // size of output volume: w2 x h2 x d2
    int k, f, s, p; // number of filters, their spatial extent, stride, amount of zero padding
    int W_row_r, W_row_c; // number of rows and columns in matrix W_row
    int X_col_r, X_col_c; // number of rows and columns in matrix X_col
};


#endif /* CONV_LAYER_H */
