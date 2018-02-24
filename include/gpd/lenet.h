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


/** Lenet class
 *
 * \brief Deep neural net to predict valid grasps.
 *
 * This class predicts for each grasp image, if the grasp is valid or not. It is a deep neural network, similar to
 * LeNet. Only forward pass is supported and for now, the size of all layers is fixed.
 *
 */
class Lenet
{
  public:

    /**
     * \brief Constructor.
     * \param num_threads number of CPU threads to use for forward pass
     */
    Lenet(int num_threads, const std::string& params_dir);

    /**
     * \brief Destructor.
     */
    ~Lenet()
    {
      delete conv1;
      delete conv2;
      delete dense1;
      delete dense2;
    }

    /**
     * \brief Classify grasp images.
     * \param image_list list of grasp images
     * \return list of scores
     */
    std::vector<float> classifyImages(const std::vector<cv::Mat>& image_list);

    /**
     * \brief Forward pass of the network.
     * \param x input to the network
     * \return output of the network
     */
    std::vector<float> forward(const std::vector<float>& x);

    /**
     * \brief Read a text file line by line into a vector (each line is a number).
     * \param location path to the text file
     */
    std::vector<float> readFileLineByLineIntoVector(const std::string& location);

    /**
     * \brief Read a binary file into a vector.
     * \param location path to the binary file
     */
    std::vector<float> readBinaryFileIntoVector(const std::string& location);


  private:

    /**
     * \brief Forward pass for a max pooling layer.
     * \param X input
     * \param filter_size the size of the filter
     * \param stride the stride at which to apply the filter
     */
    Eigen::MatrixXf poolForward(const Eigen::MatrixXf& X, int filter_size, int stride) const;

    /**
     * \brief Convert an image to an array (std::vector) so that it can be used as input for a layer.
     * \param img the image to be converted
     * \return the array
     */
    std::vector<float> imageToArray(const cv::Mat& img) const;

    ConvLayer* conv1; ///< 1st conv layer
    ConvLayer* conv2; ///< 2nd conv layer
    DenseLayer* dense1; ///< 1st dense layer
    DenseLayer* dense2; ///< 2nd dense layer

    std::vector<float> x_dense1, x_dense2, x_conv2; ///< inputs for layers

    int num_threads_; ///< number of CPU threads to use in forward pass
};


#endif /* LENET_H */
