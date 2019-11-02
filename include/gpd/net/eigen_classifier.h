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

#ifndef EIGEN_CLASSIFIER_H_
#define EIGEN_CLASSIFIER_H_

#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <gpd/net/classifier.h>
#include <gpd/net/conv_layer.h>
#include <gpd/net/dense_layer.h>

namespace gpd {
namespace net {

/**
 *
 * \brief Classify grasp candidates as viable grasps or not with Eigen
 *
 * Classifies grasps as viable or not using a custom neural network framework
 * based on the Eigen library.
 *
 */
class EigenClassifier : public Classifier {
 public:
  /**
   * \brief Constructor.
   * \param model_file the location of the file that describes the network model
   * \param weights_file the location of the file that contains the network
   * weights
   */
  EigenClassifier(const std::string &model_file,
                  const std::string &weights_file, Classifier::Device device,
                  int batch_size);

  /**
   * \brief Classify grasp candidates as viable grasps or not.
   * \param image_list the list of grasp images
   * \return the classified grasp candidates
   */
  std::vector<float> classifyImages(
      const std::vector<std::unique_ptr<cv::Mat>> &image_list);

  int getBatchSize() const { return 0; }

 private:
  /**
   * \brief Forward pass of the network.
   * \param x input to the network
   * \return output of the network
   */
  std::vector<float> forward(const std::vector<float> &x);

  /**
   * \brief Convert an image to an array (std::vector) so that it can be used as
   * input for a layer.
   * \param img the image to be converted
   * \return the array
   */
  std::vector<float> imageToArray(const cv::Mat &img) const;

  /**
   * \brief Forward pass for a max pooling layer.
   * \param X input
   * \param filter_size the size of the filter
   * \param stride the stride at which to apply the filter
   */
  Eigen::MatrixXf poolForward(const Eigen::MatrixXf &X, int filter_size,
                              int stride) const;

  /**
   * \brief Read a binary file into a vector.
   * \param location path to the binary file
   */
  std::vector<float> readBinaryFileIntoVector(const std::string &location);

  std::unique_ptr<ConvLayer> conv1_;                  ///< 1st conv layer
  std::unique_ptr<ConvLayer> conv2_;                  ///< 2nd conv layer
  std::unique_ptr<DenseLayer> dense1_;                ///< 1st dense layer
  std::unique_ptr<DenseLayer> dense2_;                ///< 2nd dense layer
  std::vector<float> x_dense1_, x_dense2_, x_conv2_;  ///< inputs for layers
  int num_threads_{1};
};

}  // namespace net
}  // namespace gpd

#endif /* EIGEN_CLASSIFIER_H_ */
