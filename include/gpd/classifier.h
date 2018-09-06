/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2018 Intel Corporation
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

/** Classifier class
 *
 * \brief Abstract base class for classifier that Classify grasp candidates as viable grasps or not
 *
 */
class Classifier
{
  public:

    enum class Device : uint8_t {
      eCPU = 0,
      eGPU = 1,
      eVPU = 2,
      eFPGA = 3
    };

    /**
     * \brief Create Classifier per build option.
     * - If "USE_OPENVINO", an OpenVINO classifier will be created
     * - Otherwise a Caffe classifier will be created
     * \param model_file The location of the file that describes the network model
     * \param weights_file The location of the file that contains the network weights
     * \param device The target device where the network computation executes
     */
    static std::shared_ptr<Classifier> create(const std::string& model_file, const std::string& weights_file, Device device = Device::eCPU);

    /**
     * \brief Classify grasp candidates as viable grasps or not.
     * \param image_list the list of grasp images
     * \return the classified grasp candidates
     */
    virtual std::vector<float> classifyImages(const std::vector<cv::Mat>& image_list) = 0;

    virtual int getBatchSize() const = 0;
};


#endif /* CLASSIFIER_H_ */
