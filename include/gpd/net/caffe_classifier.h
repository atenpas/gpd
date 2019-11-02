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

#ifndef CAFFE_CLASSIFIER_H_
#define CAFFE_CLASSIFIER_H_

// System
#include <string>
#include <vector>

// Caffe
#include "caffe/caffe.hpp"
#include "caffe/layers/memory_data_layer.hpp"
#include "caffe/util/io.hpp"

// OpenCV
#include <opencv2/core/core.hpp>

#include <gpd/net/classifier.h>

namespace gpd {
namespace net {

/**
 *
 * \brief Classify grasp candidates as viable grasps or not with Caffe
 *
 * Classifies grasps as viable or not using a convolutional neural network (CNN)
 *  with the Caffe framework.
 *
 */
class CaffeClassifier : public Classifier {
 public:
  /**
   * \brief Constructor.
   * \param model_file the location of the file that describes the network model
   * \param weights_file the location of the file that contains the network
   * weights
   */
  CaffeClassifier(const std::string &model_file,
                  const std::string &weights_file, Classifier::Device device,
                  int batch_size);

  /**
   * \brief Classify grasp candidates as viable grasps or not.
   * \param image_list the list of grasp images
   * \return the classified grasp candidates
   */
  std::vector<float> classifyImages(
      const std::vector<std::unique_ptr<cv::Mat>> &image_list);

  /**
   * \brief Return the batch size.
   * \return the batch size
   */
  int getBatchSize() const { return input_layer_->batch_size(); }

 private:
  boost::shared_ptr<caffe::Net<float>> net_;
  boost::shared_ptr<caffe::MemoryDataLayer<float>> input_layer_;
};

}  // namespace net
}  // namespace gpd

#endif /* CAFFE_CLASSIFIER_H_ */
