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

    Lenet(int num_threads);

    ~Lenet()
    {
      delete conv1;
      delete conv2;
      delete dense1;
      delete dense2;
    }

    std::vector<float> classifyImages(const std::vector<cv::Mat>& image_list);

    std::vector<float> forward(const std::vector<float>& x) const;

    std::vector<float> readFileLineByLineIntoVector(const std::string& location);

    std::vector<float> readBinaryFileIntoVector(const std::string& location);


  private:

    Eigen::MatrixXf poolForward(const Eigen::MatrixXf& X, int filter_size, int stride) const;

    ConvLayer* conv1;
    ConvLayer* conv2;
    DenseLayer* dense1;
    DenseLayer* dense2;

    int num_threads_;
};


#endif /* LENET_H */
