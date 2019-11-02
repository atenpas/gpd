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

#include <gpd/net/classifier.h>
#if defined(USE_OPENVINO)
#include <gpd/net/openvino_classifier.h>
#elif defined(USE_CAFFE)
#include <gpd/net/caffe_classifier.h>
#elif defined(USE_OPENCV)
#include <gpd/net/opencv_classifier.h>
#else
#include <gpd/net/eigen_classifier.h>
#endif

namespace gpd {
namespace net {

std::shared_ptr<Classifier> Classifier::create(const std::string &model_file,
                                               const std::string &weights_file,
                                               Classifier::Device device,
                                               int batch_size) {
#if defined(USE_OPENVINO)
  return std::make_shared<OpenVinoClassifier>(model_file, weights_file, device,
                                              batch_size);
#elif defined(USE_CAFFE)
  return std::make_shared<CaffeClassifier>(model_file, weights_file, device,
                                           batch_size);
#elif defined(USE_OPENCV)
  return std::make_shared<OpenCvClassifier>(model_file, weights_file, device);
#else
  return std::make_shared<EigenClassifier>(model_file, weights_file, device,
                                           batch_size);
#endif
}

}  // namespace net
}  // namespace gpd
