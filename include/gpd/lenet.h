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

#ifndef LENET_H
#define LENET_H


#include <Eigen/Dense>

#include <opencv2/core/core.hpp>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <gpd/conv_layer.h>
#include <gpd/dense_layer.h>


class Lenet
{
  public:

    Lenet(int num_threads, const std::string& params_dir);

    ~Lenet()
    {
      delete conv1;
      delete conv2;
      delete dense1;
      delete dense2;
    }

    std::vector<float> classifyImages(const std::vector<cv::Mat>& image_list);

    std::vector<float> forward(const std::vector<float>& x);

    std::vector<float> readFileLineByLineIntoVector(const std::string& location);

    std::vector<float> readBinaryFileIntoVector(const std::string& location);


  private:

    Eigen::MatrixXf poolForward(const Eigen::MatrixXf& X, int filter_size, int stride) const;

    ConvLayer* conv1;
    ConvLayer* conv2;
    DenseLayer* dense1;
    DenseLayer* dense2;

    std::vector<float> x_dense1, x_dense2, x_conv2;

    int num_threads_;
};


#endif /* LENET_H */
