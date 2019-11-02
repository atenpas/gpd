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

#ifndef CLASSIFIER_H_
#define CLASSIFIER_H_

// System
#include <memory>
#include <string>
#include <vector>

// OpenCV
#include <opencv2/core/core.hpp>

namespace gpd {
namespace net {

/**
 *
 * \brief Abstract base class for classifier that classifies grasp candidates as
 * viable grasps or not.
 *
 */
class Classifier {
 public:
  enum class Device : uint8_t { eCPU = 0, eGPU = 1, eVPU = 2, eFPGA = 3 };

  /**
   * \brief Create a classifier dependent on build options.
   * \param model_file filepath to the network model
   * \param weights_file filepath to the network parameters
   * \param device target device on which the network is run
   * \return the classifier
   */
  static std::shared_ptr<Classifier> create(const std::string &model_file,
                                            const std::string &weights_file,
                                            Device device = Device::eCPU,
                                            int batch_size = 1);

  /**
   * \brief Classify grasp candidates as viable grasps or not.
   * \param image_list the list of grasp images
   * \return the classified grasp candidates
   */
  virtual std::vector<float> classifyImages(
      const std::vector<std::unique_ptr<cv::Mat>> &image_list) = 0;

  /**
   * \brief Return the batch size.
   * \return the batch size
   */
  virtual int getBatchSize() const = 0;
};

}  // namespace net
}  // namespace gpd

#endif /* CLASSIFIER_H_ */
